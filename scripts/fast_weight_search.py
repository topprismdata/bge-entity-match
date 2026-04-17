#!/usr/bin/env python3
"""
Fast fusion weight search using precomputed scores.

Key insight: with 512GB RAM, precompute ALL three scores for top-K candidates
(using a fixed base ranking), then sweep weights in pure numpy.

Each weight config takes seconds, not minutes.
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Fast fusion weight search")
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
    parser.add_argument('--baseline', default='match_results/m3_full_match.csv')
    parser.add_argument('--topk', type=int, default=100)
    parser.add_argument('--use-lora', action='store_true')
    parser.add_argument('--lora-dense-cache', default='match_results/m3_cache/lora_dense.npz')
    args = parser.parse_args()

    t0 = time.time()
    benchmark_cities = ['杭州市', '深圳市', '台州市', '贵阳市', '淄博市']

    # ── Load data ──
    print("[1/4] Loading data...", file=sys.stderr)
    query_df = pd.read_csv(args.query) if args.query.endswith('.csv') else pd.read_excel(args.query)
    cand_df = pd.read_csv(args.candidates) if args.candidates.endswith('.csv') else pd.read_excel(args.candidates)
    n_q, n_c = len(query_df), len(cand_df)

    # ── Load encodings ──
    enc_path = os.path.join(args.cache_dir, 'encodings.pkl')
    print(f"[2/4] Loading encodings from {enc_path}...", file=sys.stderr)
    with open(enc_path, 'rb') as f:
        enc = pickle.load(f)
    q_sparse = enc['q_sparse']
    c_sparse = enc['c_sparse']
    q_colbert = enc['q_colbert']
    c_colbert = enc['c_colbert']

    if args.use_lora:
        if os.path.exists(args.lora_dense_cache):
            data = np.load(args.lora_dense_cache)
            q_dense = data['q_dense']
            c_dense = data['c_dense']
        else:
            print("LoRA dense cache not found. Run with --save-dense first.", file=sys.stderr)
            sys.exit(1)
    else:
        q_dense = enc['q_dense']
        c_dense = enc['c_dense']

    # ── City groups ──
    print("[3/4] Building city groups and precomputing all scores...", file=sys.stderr)
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

    query_city_set = set(query_cities)
    cities_to_process = sorted(query_city_set & city_set)
    topk = args.topk

    # ── Precompute scores for top-K per query ──
    # Use default (1,1,1) fusion to get the initial top-K, then
    # precompute all three scores for those K candidates
    print(f"  Precomputing scores for top-{topk} candidates per query...", file=sys.stderr)

    # For each query: store arrays of (dense_score, sparse_score, colbert_score, correct)
    # for top-K candidates from base ranking
    all_dense_scores = []    # shape: (n_queries, topk)
    all_sparse_scores = []
    all_colbert_scores = []
    all_correct = []         # bool: is this the right match?
    all_city_labels = []    # which city this query belongs to
    queries_done = 0

    for city in cities_to_process:
        cand_idx = city_groups[city]
        q_indices = [i for i, c in enumerate(query_cities) if c == city]
        if not q_indices:
            continue

        nq, nc = len(q_indices), len(cand_idx)

        # Dense similarity
        d_sim = q_dense[q_indices] @ c_dense[cand_idx].T

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

        # Base fusion to get top-K
        fused_ds = d_sim + s_sim
        actual_k = min(topk, nc)

        for qi_loc in range(nq):
            qi_glob = q_indices[qi_loc]
            scores = fused_ds[qi_loc]

            if actual_k >= nc:
                top_local = np.argsort(-scores)
            else:
                top_local = np.argpartition(-scores, actual_k)[:actual_k]
                top_local = top_local[np.argsort(-scores[top_local])]
            top_local = top_local[:topk]

            # Precompute ColBERT for top-K
            q_cb = np.array(q_colbert[qi_glob])
            d_scores = np.zeros(topk, dtype=np.float32)
            s_scores = np.zeros(topk, dtype=np.float32)
            cb_scores = np.zeros(topk, dtype=np.float32)
            correct = np.zeros(topk, dtype=bool)

            for k, ci_loc in enumerate(top_local):
                ci_glob = cand_idx[ci_loc]
                d_scores[k] = d_sim[qi_loc, ci_loc]
                s_scores[k] = s_sim[qi_loc, ci_loc]
                c_cb = np.array(c_colbert[ci_glob])
                cb_scores[k] = (q_cb @ c_cb.T).max(axis=1).sum()
                correct[k] = (c_ids[ci_glob] == q_ids[qi_glob])

            all_dense_scores.append(d_scores)
            all_sparse_scores.append(s_scores)
            all_colbert_scores.append(cb_scores)
            all_correct.append(correct)
            all_city_labels.append(city)
            queries_done += 1

        if queries_done % 5000 == 0 or queries_done < 100:
            print(f"  {queries_done}/{n_q} queries processed...", file=sys.stderr)

    # Stack into arrays
    D = np.stack(all_dense_scores)      # (n_queries, topk)
    S = np.stack(all_sparse_scores)
    C = np.stack(all_colbert_scores)
    CORRECT = np.stack(all_correct)
    CITIES = np.array(all_city_labels)

    elapsed = time.time() - t0
    mem_gb = (D.nbytes + S.nbytes + C.nbytes + CORRECT.nbytes) / 1e9
    print(f"\n  Precomputed {D.shape[0]} queries x {D.shape[1]} candidates in {elapsed:.0f}s", file=sys.stderr)
    print(f"  Memory: {mem_gb:.2f}GB", file=sys.stderr)

    # ── Score magnitude analysis ──
    print(f"\n{'='*60}", file=sys.stderr)
    print("SCORE MAGNITUDES (top-1 candidates from base ranking)", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    # Take rank-0 candidates (best from base ranking)
    print(f"  Dense:   mean={D[:,0].mean():.4f}, std={D[:,0].std():.4f}, "
          f"range=[{D[:,0].min():.4f}, {D[:,0].max():.4f}]", file=sys.stderr)
    print(f"  Sparse:  mean={S[:,0].mean():.4f}, std={S[:,0].std():.4f}, "
          f"range=[{S[:,0].min():.4f}, {S[:,0].max():.4f}]", file=sys.stderr)
    print(f"  ColBERT: mean={C[:,0].mean():.2f}, std={C[:,0].std():.2f}, "
          f"range=[{C[:,0].min():.2f}, {C[:,0].max():.2f}]", file=sys.stderr)
    ratio_dc = C[:,0].mean() / max(D[:,0].mean(), 1e-9)
    print(f"  ColBERT/Dense ratio: {ratio_dc:.1f}x", file=sys.stderr)

    # ── Weight sweep (now pure numpy, very fast) ──
    print(f"\n{'='*60}", file=sys.stderr)
    print("FUSION WEIGHT SWEEP (precomputed scores)", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    weight_configs = [
        # Single components
        (1, 0, 0, "Dense only"),
        (0, 1, 0, "Sparse only"),
        (0, 0, 1, "ColBERT only"),
        # Default
        (1, 1, 1, "Default (1,1,1)"),
        # Scale-balanced
        (20, 5, 1, "Scale-balanced"),
        (30, 10, 1, "Heavy dense"),
        (15, 3, 1, "Moderate dense"),
        (10, 3, 1, "Light dense boost"),
        # Dense + ColBERT (no sparse)
        (20, 0, 1, "Dense+ColBERT"),
        (30, 0, 1, "Heavy Dense+ColBERT"),
        # Dense + Sparse (no ColBERT)
        (20, 5, 0, "No ColBERT"),
        (30, 10, 0, "No ColBERT heavy"),
        # Higher ColBERT
        (1, 1, 2, "2x ColBERT"),
        (1, 1, 5, "5x ColBERT"),
        (1, 1, 10, "10x ColBERT"),
        # Very high dense
        (50, 10, 1, "Very heavy dense"),
        (100, 20, 1, "Extreme dense"),
    ]

    print(f"  {'Config':<30} {'Overall':>8} {'杭州':>7} {'深圳':>7} "
          f"{'台州':>7} {'贵阳':>7} {'淄博':>7}", file=sys.stderr)
    print(f"  {'-'*80}", file=sys.stderr)

    best_overall = 0
    best_config = None

    for wd, ws, wc, label in weight_configs:
        # Fused score for all candidates: shape (n_queries, topk)
        fused = wd * D + ws * S + wc * C

        # Best candidate per query
        best_idx = np.argmax(fused, axis=1)
        is_correct = CORRECT[np.arange(len(CORRECT)), best_idx]

        # Overall accuracy
        overall = is_correct.mean() * 100

        # Per-city accuracy
        city_accs = {}
        for bc in benchmark_cities:
            mask = CITIES == bc
            if mask.any():
                city_accs[bc] = is_correct[mask].mean() * 100
            else:
                city_accs[bc] = 0

        print(f"  {label:<30} {overall:>7.2f}% {city_accs.get('杭州市', 0):>6.1f}% "
              f"{city_accs.get('深圳市', 0):>6.1f}% {city_accs.get('台州市', 0):>6.1f}% "
              f"{city_accs.get('贵阳市', 0):>6.1f}% {city_accs.get('淄博市', 0):>6.1f}%",
              file=sys.stderr)

        if overall > best_overall:
            best_overall = overall
            best_config = (wd, ws, wc, label)

    print(f"  {'-'*80}", file=sys.stderr)
    print(f"  BEST: {best_config[3]} ({best_config[0]},{best_config[1]},{best_config[2]}) = {best_overall:.2f}%",
          file=sys.stderr)

    # ── Fine grid around best ──
    if best_config[:3] != (1, 1, 1):
        print(f"\n  Fine grid around {best_config[:3]}...", file=sys.stderr)
        wd_b, ws_b, wc_b = best_config[:3]

        fine_results = []
        for wd in range(max(0, wd_b - 10), wd_b + 11, max(1, wd_b // 5)):
            for ws in range(max(0, ws_b - 5), ws_b + 6, max(1, ws_b // 3)):
                for wc in range(max(0, wc_b - 2), wc_b + 3):
                    fused = wd * D + ws * S + wc * C
                    best_idx = np.argmax(fused, axis=1)
                    is_correct = CORRECT[np.arange(len(CORRECT)), best_idx]
                    overall = is_correct.mean() * 100
                    fine_results.append((wd, ws, wc, overall))

        fine_results.sort(key=lambda x: -x[3])
        print(f"  Top-10 fine-grained results:", file=sys.stderr)
        for wd, ws, wc, acc in fine_results[:10]:
            print(f"    ({wd},{ws},{wc}) = {acc:.2f}%", file=sys.stderr)

    total = time.time() - t0
    print(f"\n  Total time: {total:.0f}s ({total/60:.1f}min)", file=sys.stderr)

    # ── Baseline comparison ──
    if os.path.exists(args.baseline):
        base_df = pd.read_csv(args.baseline)
        base_top1 = base_df[base_df['rank'] == 1]
        b_overall = base_top1['verified'].sum() / len(base_top1) * 100
        base_has_match = base_df[base_df['verified'] == True]['query_id'].nunique()
        print(f"\n  Baseline (from file): {b_overall:.2f}%", file=sys.stderr)
        print(f"  Answerable queries: {base_has_match}", file=sys.stderr)


if __name__ == '__main__':
    main()
