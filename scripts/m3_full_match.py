#!/usr/bin/env python3
"""
Full-dataset matching: BGE-M3 (Dense + Sparse + ColBERT), no CrossEncoder.
Two-pass approach for efficiency:
  Pass 1: Dense + Sparse → Top-100 per query per city
  Pass 2: ColBERT refinement on Top-100
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
from FlagEmbedding import BGEM3FlagModel


def main():
    parser = argparse.ArgumentParser(description="BGE-M3 full-dataset matching")
    parser.add_argument('--query', required=True)
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--q-id', default='Door Social Credit Code \n门店社会信用证代码')
    parser.add_argument('--q-name', default='Door Name \n门店名称')
    parser.add_argument('--q-addr', default='Door Address\n门店营业地址')
    parser.add_argument('--q-city', default='市')
    parser.add_argument('--c-id', default='统一社会信用代码')
    parser.add_argument('--c-name', default='公司名称')
    parser.add_argument('--c-addr', default='注册地址')
    parser.add_argument('--c-city', default='所属城市')
    parser.add_argument('--output', default='match_results/m3_full_match.csv')
    parser.add_argument('--topk', type=int, default=100, help='Pass 1 top-K')
    parser.add_argument('--save-top', type=int, default=10, help='Results to save per query')
    parser.add_argument('--cache-dir', default='match_results/m3_cache')
    args = parser.parse_args()

    t0 = time.time()
    os.makedirs(args.cache_dir, exist_ok=True)

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

    # ── Encode with BGE-M3 ──
    enc_path = os.path.join(args.cache_dir, 'encodings.pkl')
    if os.path.exists(enc_path):
        print(f"[3/6] Loading cached encodings from {enc_path}...", file=sys.stderr)
        with open(enc_path, 'rb') as f:
            enc = pickle.load(f)
        q_dense = enc['q_dense']
        c_dense = enc['c_dense']
        q_sparse = enc['q_sparse']
        c_sparse = enc['c_sparse']
        q_colbert = enc['q_colbert']
        c_colbert = enc['c_colbert']
    else:
        print("[3/6] Encoding with BGE-M3...", file=sys.stderr)
        bge_m3 = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, device='mps')

        print(f"  Encoding {n_q} queries...", file=sys.stderr)
        tq = time.time()
        q_result = bge_m3.encode(
            query_texts, return_dense=True, return_sparse=True,
            return_colbert_vecs=True, batch_size=64, max_length=256,
        )
        print(f"  Done in {time.time()-tq:.0f}s", file=sys.stderr)

        torch.mps.empty_cache()

        print(f"  Encoding {n_c} candidates...", file=sys.stderr)
        tc = time.time()
        c_result = bge_m3.encode(
            cand_texts, return_dense=True, return_sparse=True,
            return_colbert_vecs=True, batch_size=256, max_length=256,
        )
        print(f"  Done in {time.time()-tc:.0f}s", file=sys.stderr)

        q_dense = np.array(q_result['dense_vecs'])
        c_dense = np.array(c_result['dense_vecs'])
        q_sparse = q_result['lexical_weights']
        c_sparse = c_result['lexical_weights']
        q_colbert = q_result['colbert_vecs']
        c_colbert = c_result['colbert_vecs']

        # Cache
        print(f"  Saving encodings to {enc_path}...", file=sys.stderr)
        with open(enc_path, 'wb') as f:
            pickle.dump({
                'q_dense': q_dense, 'c_dense': c_dense,
                'q_sparse': q_sparse, 'c_sparse': c_sparse,
                'q_colbert': q_colbert, 'c_colbert': c_colbert,
            }, f)

        del bge_m3
        torch.mps.empty_cache()

    print(f"  Dense: q={q_dense.shape}, c={c_dense.shape}", file=sys.stderr)

    # ── City groups ──
    print("[4/6] Building city groups...", file=sys.stderr)
    city_groups = {}
    if args.c_city in cand_df.columns:
        for i, city in enumerate(cand_df[args.c_city].astype(str).str.strip()):
            if city and city != 'nan':
                city_groups.setdefault(city, []).append(i)

    city_set = set(city_groups.keys())

    def align_city(c):
        c = str(c).strip()
        if c in city_set: return c
        if c + '市' in city_set: return c + '市'
        if c.endswith('市') and c[:-1] in city_set: return c[:-1]
        return c

    if args.q_city in query_df.columns:
        query_cities = query_df[args.q_city].astype(str).str.strip().apply(align_city).tolist()
    else:
        query_cities = [''] * n_q

    q_ids = query_df[args.q_id].astype(str).str.strip().str.upper().values
    c_ids = cand_df[args.c_id].astype(str).str.strip().str.upper().values

    # ── Pass 1: Dense + Sparse → Top-K per city ──
    print(f"[5/6] Pass 1: Dense + Sparse → Top-{args.topk}...", file=sys.stderr)
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
        if nq == 0: continue

        cities_done += 1
        if cities_done <= 10 or cities_done % 20 == 0:
            print(f"  [{cities_done}/{len(cities_to_process)}] {city}: "
                  f"{nq}×{nc}", file=sys.stderr)

        # Dense similarity
        d_sim = q_dense[q_indices] @ c_dense[cand_idx].T  # (nq, nc)

        # Sparse similarity (vectorized via term matching)
        s_sim = np.zeros((nq, nc), dtype=np.float32)
        for qi_loc, qi_glob in enumerate(q_indices):
            q_sp = q_sparse[qi_glob]
            if not q_sp:
                continue
            # Build array of candidate weights for each query term
            for token, qw in q_sp.items():
                cw_arr = np.array([c_sparse[cand_idx[ci]].get(token, 0.0)
                                   for ci in range(nc)], dtype=np.float32)
                s_sim[qi_loc] += qw * cw_arr

        # Fused (Pass 1)
        fused_ds = d_sim + s_sim

        # Top-K per query
        actual_k = min(topk, nc)
        for qi_loc, qi_glob in enumerate(q_indices):
            scores = fused_ds[qi_loc]
            if actual_k >= nc:
                top_local = np.argsort(-scores)
            else:
                top_local = np.argpartition(-scores, actual_k)[:actual_k]
                top_local = top_local[np.argsort(-scores[top_local])]

            # ── Pass 2: ColBERT on top-K ──
            top_k_cands = top_local[:topk]
            colbert_scores = np.zeros(len(top_k_cands), dtype=np.float32)
            q_cb = np.array(q_colbert[qi_glob])  # (Lq, dim)

            for ck, ci_loc in enumerate(top_k_cands):
                ci_glob = cand_idx[ci_loc]
                c_cb = np.array(c_colbert[ci_glob])  # (Lc, dim)
                sim_mat = q_cb @ c_cb.T  # (Lq, Lc)
                colbert_scores[ck] = sim_mat.max(axis=1).sum()

            # Final fused = Dense + Sparse + ColBERT
            final_scores = fused_ds[qi_loc, top_k_cands] + colbert_scores
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

    # ── Summary ──
    top1 = df[df['rank'] == 1]
    if len(top1) > 0:
        v = top1['verified'].sum()
        total = len(top1)
        print(f"\n  Overall Top-1: {v}/{total} ({v/total*100:.2f}%)", file=sys.stderr)

        # Per-city top-20
        city_stats = []
        for city, grp in top1.groupby('query_city'):
            t = len(grp)
            c = int(grp['verified'].sum())
            city_stats.append((city, t, c, c/t*100))
        city_stats.sort(key=lambda x: -x[1])
        print(f"\n  {'City':<12} {'Total':>6} {'Correct':>8} {'Acc':>7}", file=sys.stderr)
        print(f"  {'-'*35}", file=sys.stderr)
        for city, t, c, acc in city_stats[:25]:
            print(f"  {city:<12} {t:>6} {c:>8} {acc:>6.1f}%", file=sys.stderr)
        if len(city_stats) > 25:
            print(f"  ... and {len(city_stats)-25} more cities", file=sys.stderr)


if __name__ == '__main__':
    main()
