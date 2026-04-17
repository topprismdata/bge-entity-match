#!/usr/bin/env python3
"""
Phase 3: Evaluate LoRA-fine-tuned BGE-M3 Dense encoder.

Loads LoRA-adapted SentenceTransformer, re-encodes dense vectors,
combines with cached sparse + ColBERT, runs city-partitioned matching.
Compares with baseline on 5-city benchmark + full dataset.
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA fine-tuned BGE-M3")
    parser.add_argument('--lora-dir', default='match_results/m3_cache/lora_weights')
    parser.add_argument('--base-model', default='BAAI/bge-m3')
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
    parser.add_argument('--cache-dir', default='match_results/m3_cache')
    parser.add_argument('--baseline', default='match_results/m3_full_match.csv')
    parser.add_argument('--output', default='match_results/m3_lora_match.csv')
    parser.add_argument('--topk', type=int, default=100)
    parser.add_argument('--save-top', type=int, default=10)
    parser.add_argument('--fusion-weights', default='1,1,1',
                        help='Comma-separated weights: dense,sparse,colbert')
    args = parser.parse_args()

    t0 = time.time()
    w_dense, w_sparse, w_colbert = [float(w) for w in args.fusion_weights.split(',')]

    # ── Load data ──
    print("[1/6] Loading data...", file=sys.stderr)
    query_df = pd.read_csv(args.query) if args.query.endswith('.csv') else pd.read_excel(args.query)
    cand_df = pd.read_csv(args.candidates) if args.candidates.endswith('.csv') else pd.read_excel(args.candidates)
    n_q, n_c = len(query_df), len(cand_df)
    print(f"  Queries: {n_q}, Candidates: {n_c}", file=sys.stderr)

    # ── Build texts ──
    print("[2/6] Building texts...", file=sys.stderr)
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

    # ── Load cached sparse + ColBERT ──
    enc_path = os.path.join(args.cache_dir, 'encodings.pkl')
    print(f"[3/6] Loading cached encodings from {enc_path}...", file=sys.stderr)
    with open(enc_path, 'rb') as f:
        enc = pickle.load(f)
    q_sparse = enc['q_sparse']
    c_sparse = enc['c_sparse']
    q_colbert = enc['q_colbert']
    c_colbert = enc['c_colbert']
    print(f"  Sparse: q={len(q_sparse)}, c={len(c_sparse)}", file=sys.stderr)
    print(f"  ColBERT: q={len(q_colbert)}, c={len(c_colbert)}", file=sys.stderr)

    # ── Re-encode Dense with LoRA model ──
    print(f"[4/6] Re-encoding dense with LoRA model from {args.lora_dir}...", file=sys.stderr)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = SentenceTransformer(args.lora_dir, device=device)

    tq = time.time()
    q_dense = model.encode(query_texts, batch_size=128, show_progress_bar=True,
                           normalize_embeddings=True)
    print(f"  Queries encoded in {time.time()-tq:.0f}s, shape={q_dense.shape}", file=sys.stderr)

    if device == 'mps':
        torch.mps.empty_cache()

    tc = time.time()
    c_dense = model.encode(cand_texts, batch_size=256, show_progress_bar=True,
                           normalize_embeddings=True)
    print(f"  Candidates encoded in {time.time()-tc:.0f}s, shape={c_dense.shape}", file=sys.stderr)

    # ── City groups ──
    print("[5/6] Building city groups and matching...", file=sys.stderr)
    city_groups = {}
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

    # ── Matching ──
    topk = args.topk
    save_top = args.save_top
    query_city_set = set(query_cities)
    cities_to_process = sorted(query_city_set & city_set)
    no_city_count = sum(1 for c in query_cities if not c or c not in city_groups)

    print(f"  Cities: {len(cities_to_process)}, No-match queries: {no_city_count}", file=sys.stderr)

    all_results = []
    cities_done = 0

    for city in cities_to_process:
        cand_idx = city_groups[city]
        q_indices = [i for i, c in enumerate(query_cities) if c == city]
        nq, nc = len(q_indices), len(cand_idx)
        if nq == 0:
            continue

        cities_done += 1
        if cities_done <= 10 or cities_done % 20 == 0:
            print(f"  [{cities_done}/{len(cities_to_process)}] {city}: {nq}x{nc}", file=sys.stderr)

        # Dense similarity (with fusion weight)
        d_sim = w_dense * (q_dense[q_indices] @ c_dense[cand_idx].T)

        # Sparse similarity
        s_sim = np.zeros((nq, nc), dtype=np.float32)
        for qi_loc, qi_glob in enumerate(q_indices):
            q_sp = q_sparse[qi_glob]
            if not q_sp:
                continue
            for token, qw in q_sp.items():
                cw_arr = np.array([c_sparse[cand_idx[ci]].get(token, 0.0)
                                   for ci in range(nc)], dtype=np.float32)
                s_sim[qi_loc] += qw * cw_arr
        s_sim *= w_sparse

        # Pass 1: Dense + Sparse → Top-K
        fused_ds = d_sim + s_sim
        actual_k = min(topk, nc)

        for qi_loc, qi_glob in enumerate(q_indices):
            scores = fused_ds[qi_loc]
            if actual_k >= nc:
                top_local = np.argsort(-scores)
            else:
                top_local = np.argpartition(-scores, actual_k)[:actual_k]
                top_local = top_local[np.argsort(-scores[top_local])]

            # Pass 2: ColBERT on top-K
            top_k_cands = top_local[:topk]
            colbert_scores = np.zeros(len(top_k_cands), dtype=np.float32)
            q_cb = np.array(q_colbert[qi_glob])

            for ck, ci_loc in enumerate(top_k_cands):
                ci_glob = cand_idx[ci_loc]
                c_cb = np.array(c_colbert[ci_glob])
                sim_mat = q_cb @ c_cb.T
                colbert_scores[ck] = sim_mat.max(axis=1).sum()

            # Final fused
            final_scores = fused_ds[qi_loc, top_k_cands] + w_colbert * colbert_scores
            final_order = np.argsort(-final_scores)

            gt_code = q_ids[qi_glob]
            for rank, pos in enumerate(final_order[:save_top]):
                ci_loc = top_k_cands[pos]
                ci_glob = cand_idx[ci_loc]
                matched_code = c_ids[ci_glob]
                verified = matched_code == gt_code

                all_results.append({
                    'query_id': query_df.iloc[qi_glob][args.q_id],
                    'query_name': query_df.iloc[qi_glob].get(args.q_name, ''),
                    'query_city': city,
                    'matched_id': cand_df.iloc[ci_glob][args.c_id],
                    'matched_name': cand_df.iloc[ci_glob].get(args.c_name, ''),
                    'rank': rank + 1,
                    'dense_score': float(d_sim[qi_loc, ci_loc]),
                    'sparse_score': float(s_sim[qi_loc, ci_loc]),
                    'colbert_score': float(colbert_scores[
                        np.where(top_k_cands == ci_loc)[0][0]]),
                    'fused_score': float(final_scores[pos]),
                    'verified': verified,
                })

    # Queries without city match
    if no_city_count > 0:
        for qi_glob, city in enumerate(query_cities):
            if not city or city not in city_groups:
                all_results.append({
                    'query_id': query_df.iloc[qi_glob][args.q_id],
                    'query_name': query_df.iloc[qi_glob].get(args.q_name, ''),
                    'query_city': city or 'unknown',
                    'matched_id': '', 'matched_name': '',
                    'rank': 0, 'dense_score': 0, 'sparse_score': 0,
                    'colbert_score': 0, 'fused_score': 0, 'verified': False,
                })

    # ── Save ──
    print("[6/6] Saving results...", file=sys.stderr)
    df = pd.DataFrame(all_results)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    df.to_csv(args.output, index=False, encoding='utf-8-sig')

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}min)", file=sys.stderr)
    print(f"Output: {args.output} ({len(df)} rows)", file=sys.stderr)

    # ── Compare with baseline ──
    print(f"\n{'='*60}", file=sys.stderr)
    print("COMPARISON: LoRA vs Baseline", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    if os.path.exists(args.baseline):
        base_df = pd.read_csv(args.baseline)
        base_top1 = base_df[base_df['rank'] == 1]
        lora_top1 = df[df['rank'] == 1]

        # Per-city comparison for benchmark cities
        benchmark_cities = ['杭州市', '深圳市', '台州市', '贵阳市', '淄博市']
        print(f"\n  {'City':<12} {'Baseline':>10} {'LoRA':>10} {'Delta':>8}", file=sys.stderr)
        print(f"  {'-'*42}", file=sys.stderr)

        for city in benchmark_cities:
            base_city = base_top1[base_top1['query_city'] == city]
            lora_city = lora_top1[lora_top1['query_city'] == city]
            if len(base_city) == 0:
                continue
            b_acc = base_city['verified'].sum() / len(base_city) * 100
            l_acc = lora_city['verified'].sum() / len(lora_city) * 100 if len(lora_city) > 0 else 0
            delta = l_acc - b_acc
            sign = '+' if delta >= 0 else ''
            print(f"  {city:<12} {b_acc:>9.1f}% {l_acc:>9.1f}% {sign}{delta:>7.1f}pp", file=sys.stderr)

        # Overall
        b_overall = base_top1['verified'].sum() / len(base_top1) * 100
        l_overall = lora_top1['verified'].sum() / len(lora_top1) * 100 if len(lora_top1) > 0 else 0
        delta = l_overall - b_overall
        sign = '+' if delta >= 0 else ''
        print(f"  {'-'*42}", file=sys.stderr)
        print(f"  {'Overall':<12} {b_overall:>9.1f}% {l_overall:>9.1f}% {sign}{delta:>7.1f}pp", file=sys.stderr)

        # Answerable queries (those with a match in top-10)
        base_answerable = base_top1[base_top1['rank'] == 1]
        base_has_match = base_df[base_df['verified'] == True]['query_id'].nunique()
        print(f"\n  Answerable queries: {base_has_match}", file=sys.stderr)
        print(f"  Baseline Top-1: {base_top1['verified'].sum()}/{len(base_top1)} "
              f"({base_top1['verified'].sum()/len(base_top1)*100:.2f}%)", file=sys.stderr)
        print(f"  LoRA Top-1: {lora_top1['verified'].sum()}/{len(lora_top1)} "
              f"({lora_top1['verified'].sum()/len(lora_top1)*100:.2f}%)" if len(lora_top1) > 0
              else "  LoRA: no results", file=sys.stderr)
    else:
        print(f"  Baseline file not found: {args.baseline}", file=sys.stderr)

    # Top-20 cities
    if len(df[df['rank'] == 1]) > 0:
        top1 = df[df['rank'] == 1]
        city_stats = []
        for city, grp in top1.groupby('query_city'):
            t = len(grp)
            c = int(grp['verified'].sum())
            city_stats.append((city, t, c, c / t * 100))
        city_stats.sort(key=lambda x: -x[1])
        print(f"\n  Top-20 cities by volume:", file=sys.stderr)
        print(f"  {'City':<12} {'Total':>6} {'Correct':>8} {'Acc':>7}", file=sys.stderr)
        print(f"  {'-'*35}", file=sys.stderr)
        for city, t, c, acc in city_stats[:20]:
            print(f"  {city:<12} {t:>6} {c:>8} {acc:>6.1f}%", file=sys.stderr)


if __name__ == '__main__':
    main()
