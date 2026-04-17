#!/usr/bin/env python3
"""
BGE-M3 粗排 + CrossEncoder 精排：5 个测试城市快速评估。

Stage 1: BGE-M3 Dense+Sparse → top-100 (from cached encodings)
Stage 2: bge-reranker-v2-m3 CrossEncoder → rerank top-100 → top-1

功率拉满：512GB RAM, 大 batch CrossEncoder inference.
"""
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder

BENCHMARK_CITIES = ['杭州市', '深圳市', '台州市', '贵阳市', '淄博市']


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', default='match_results/m3_cache')
    parser.add_argument('--query', default='完整 qcc_extracted_v3.csv')
    parser.add_argument('--candidates', default='副本天眼查数据 - 1.xlsx')
    parser.add_argument('--q-id', default='Door Social Credit Code \n门店社会信用证代码')
    parser.add_argument('--q-name', default='Door Name \n门店名称')
    parser.add_argument('--q-addr', default='Door Address\n门店营业地址')
    parser.add_argument('--q-city', default='市')
    parser.add_argument('--c-id', default='统一社会信用代码')
    parser.add_argument('--c-name', default='公司名称')
    parser.add_argument('--c-addr', default='注册地址')
    parser.add_argument('--c-city', default='所属城市')
    parser.add_argument('--reranker', default='BAAI/bge-reranker-v2-m3')
    parser.add_argument('--ce-batch', type=int, default=256)
    parser.add_argument('--topk', type=int, default=100)
    args = parser.parse_args()

    t0 = time.time()

    # ── Load data ──
    print("[1/4] Loading data...", file=sys.stderr)
    query_df = pd.read_csv(args.query) if args.query.endswith('.csv') else pd.read_excel(args.query)
    cand_df = pd.read_csv(args.candidates) if args.candidates.endswith('.csv') else pd.read_excel(args.candidates)
    n_q, n_c = len(query_df), len(cand_df)
    print(f"  Queries: {n_q}, Candidates: {n_c}", file=sys.stderr)

    # ── Build texts ──
    print("[2/4] Building texts...", file=sys.stderr)
    query_texts = []
    for _, row in query_df.iterrows():
        name = str(row.get(args.q_name, '')).strip() if pd.notna(row.get(args.q_name)) else ''
        addr = str(row.get(args.q_addr, '')).strip() if pd.notna(row.get(args.q_addr)) else ''
        query_texts.append(f"{name} {addr}".strip())

    cand_texts = []
    for _, row in cand_df.iterrows():
        name = str(row.get(args.c_name, '')).strip() if pd.notna(row.get(args.c_name)) else ''
        addr = str(row.get(args.c_addr, '')).strip() if pd.notna(row.get(args.c_addr)) else ''
        cand_texts.append(f"{name} {addr}".strip())

    # ── Load cached encodings ──
    enc_path = os.path.join(args.cache_dir, 'encodings.pkl')
    print(f"[3/4] Loading cached BGE-M3 encodings...", file=sys.stderr)
    with open(enc_path, 'rb') as f:
        enc = pickle.load(f)
    q_dense = enc['q_dense']
    c_dense = enc['c_dense']
    q_sparse = enc['q_sparse']
    c_sparse = enc['c_sparse']
    print(f"  Dense: q={q_dense.shape}, c={c_dense.shape}", file=sys.stderr)

    # ── City groups ──
    city_groups: dict[str, list[int]] = {}
    if args.c_city in cand_df.columns:
        for i, city in enumerate(cand_df[args.c_city].astype(str).str.strip()):
            if city and city != 'nan':
                city_groups.setdefault(city, []).append(i)

    city_set = set(city_groups.keys())

    def align_city(c: str) -> str:
        c = c.strip()
        if c in city_set:
            return c
        if c + '市' in city_set:
            return c + '市'
        if c.endswith('市') and c[:-1] in city_set:
            return c[:-1]
        return c

    if args.q_city in query_df.columns:
        query_cities = query_df[args.q_city].astype(str).str.strip().apply(align_city).tolist()
    else:
        query_cities = [''] * n_q

    q_ids = query_df[args.q_id].astype(str).str.strip().str.upper().values
    c_ids = cand_df[args.c_id].astype(str).str.strip().str.upper().values

    # ── Load CrossEncoder ──
    print(f"[4/4] Loading CrossEncoder: {args.reranker}...", file=sys.stderr)
    import torch
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    ce_model = CrossEncoder(args.reranker, max_length=512, device=device)
    print(f"  Device: {device}, batch_size: {args.ce_batch}", file=sys.stderr)

    # ── Run per city ──
    topk = args.topk
    all_results = {}

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  BGE-M3 粗排 → CrossEncoder 精排 (top-{topk})", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    for city in BENCHMARK_CITIES:
        tc = time.time()
        if city not in city_groups:
            print(f"  {city}: no candidates found, skipping", file=sys.stderr)
            continue

        cand_idx = city_groups[city]
        q_indices = [i for i, c in enumerate(query_cities) if c == city]
        nq, nc = len(q_indices), len(cand_idx)
        if nq == 0:
            continue

        # Stage 1: BGE-M3 Dense + Sparse → top-K
        d_sim = q_dense[q_indices] @ c_dense[cand_idx].T

        s_sim = np.zeros((nq, nc), dtype=np.float32)
        for qi_loc, qi_glob in enumerate(q_indices):
            q_sp = q_sparse[qi_glob]
            if not q_sp:
                continue
            for token, qw in q_sp.items():
                cw_arr = np.array([c_sparse[cand_idx[ci]].get(token, 0.0)
                                   for ci in range(nc)], dtype=np.float32)
                s_sim[qi_loc] += qw * cw_arr

        fused_ds = d_sim + s_sim
        actual_k = min(topk, nc)

        # Get top-K indices per query
        top_k_indices = np.zeros((nq, actual_k), dtype=np.int64)
        for qi_loc in range(nq):
            scores = fused_ds[qi_loc]
            if actual_k >= nc:
                top_local = np.argsort(-scores)
            else:
                top_local = np.argpartition(-scores, actual_k)[:actual_k]
                top_local = top_local[np.argsort(-scores[top_local])]
            top_k_indices[qi_loc] = top_local

        # Stage 1 accuracy (baseline)
        s1_correct = 0
        for qi_loc, qi_glob in enumerate(q_indices):
            ci_glob = cand_idx[top_k_indices[qi_loc, 0]]
            if c_ids[ci_glob] == q_ids[qi_glob]:
                s1_correct += 1
        s1_acc = s1_correct / nq * 100

        # Stage 2: CrossEncoder rerank
        # Build all pairs: (query_text, candidate_text) for top-K
        t_ce = time.time()
        all_pairs = []
        pair_mapping = []  # (qi_loc, list of ci_local indices)
        for qi_loc in range(nq):
            qi_glob = q_indices[qi_loc]
            qt = query_texts[qi_glob]
            local_indices = top_k_indices[qi_loc]
            pairs = [[qt, cand_texts[cand_idx[ci_loc]]] for ci_loc in local_indices]
            pair_mapping.append(len(pairs))
            all_pairs.extend(pairs)

        # Batch predict
        ce_scores = ce_model.predict(all_pairs, batch_size=args.ce_batch, show_progress_bar=False)

        # Reshape and find best per query
        s2_correct = 0
        offset = 0
        for qi_loc in range(nq):
            qi_glob = q_indices[qi_loc]
            k = pair_mapping[qi_loc]
            scores = ce_scores[offset:offset + k]
            offset += k
            best_ci_local = np.argmax(scores)
            ci_glob = cand_idx[top_k_indices[qi_loc, best_ci_local]]
            if c_ids[ci_glob] == q_ids[qi_glob]:
                s2_correct += 1

        s2_acc = s2_correct / nq * 100
        ce_time = time.time() - t_ce
        total_time = time.time() - tc
        delta = s2_acc - s1_acc

        all_results[city] = {
            's1_acc': s1_acc, 's2_acc': s2_acc, 'delta': delta,
            'nq': nq, 'nc': nc, 'ce_time': ce_time, 'total_time': total_time,
        }

        sign = '+' if delta >= 0 else ''
        print(f"  {city:<10} {nq:>4}q x {nc:>5}c | "
              f"M3: {s1_acc:.1f}% → CE: {s2_acc:.1f}% ({sign}{delta:.1f}pp) | "
              f"CE: {ce_time:.0f}s, Total: {total_time:.0f}s", file=sys.stderr)

    # ── Summary ──
    print(f"\n{'='*60}", file=sys.stderr)
    print("  SUMMARY", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"  {'City':<10} {'M3 Top-1':>10} {'M3+CE Top-1':>12} {'Delta':>8}", file=sys.stderr)
    print(f"  {'-'*42}", file=sys.stderr)

    total_q = 0
    total_s1 = 0
    total_s2 = 0
    for city in BENCHMARK_CITIES:
        if city not in all_results:
            continue
        r = all_results[city]
        sign = '+' if r['delta'] >= 0 else ''
        print(f"  {city:<10} {r['s1_acc']:>9.1f}% {r['s2_acc']:>11.1f}% {sign}{r['delta']:>7.1f}pp",
              file=sys.stderr)
        total_q += r['nq']
        total_s1 += r['s1_acc'] * r['nq'] / 100
        total_s2 += r['s2_acc'] * r['nq'] / 100

    if total_q > 0:
        w_s1 = total_s1 / total_q * 100
        w_s2 = total_s2 / total_q * 100
        w_delta = w_s2 - w_s1
        sign = '+' if w_delta >= 0 else ''
        print(f"  {'-'*42}", file=sys.stderr)
        print(f"  {'Weighted':<10} {w_s1:>9.1f}% {w_s2:>11.1f}% {sign}{w_delta:>7.1f}pp",
              file=sys.stderr)

    elapsed = time.time() - t0
    print(f"\n  Total: {elapsed:.0f}s ({elapsed/60:.1f}min)", file=sys.stderr)


if __name__ == '__main__':
    main()
