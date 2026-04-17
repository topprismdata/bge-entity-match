#!/usr/bin/env python3
"""
In-memory fusion weight search for LoRA BGE-M3.

Leverages 512GB RAM to load all vectors once, then sweep fusion weights
in seconds instead of re-encoding each time.

Usage:
    python3 bge-entity-match/scripts/fusion_weight_search.py
    python3 bge-entity-match/scripts/fusion_weight_search.py --use-lora
    python3 bge-entity-match/scripts/fusion_weight_search.py --use-lora --save-dense
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch


def main():
    parser = argparse.ArgumentParser(description="In-memory fusion weight search")
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
    parser.add_argument('--topk', type=int, default=100)
    parser.add_argument('--save-top', type=int, default=10)
    parser.add_argument('--use-lora', action='store_true',
                        help='Use LoRA model instead of baseline dense vectors')
    parser.add_argument('--save-dense', action='store_true',
                        help='Save LoRA dense vectors to cache for reuse')
    parser.add_argument('--dense-cache', default='match_results/m3_cache/lora_dense.npz')
    parser.add_argument('--topk-pre', type=int, default=100,
                        help='How many candidates to keep after dense+sparse pass')
    args = parser.parse_args()

    t0 = time.time()

    # ── Load data ──
    print("[1/5] Loading data...", file=sys.stderr)
    query_df = pd.read_csv(args.query) if args.query.endswith('.csv') else pd.read_excel(args.query)
    cand_df = pd.read_csv(args.candidates) if args.candidates.endswith('.csv') else pd.read_excel(args.candidates)
    n_q, n_c = len(query_df), len(cand_df)
    print(f"  Queries: {n_q}, Candidates: {n_c}", file=sys.stderr)

    # ── Build texts ──
    print("[2/5] Building texts...", file=sys.stderr)
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
    print(f"[3/5] Loading cached encodings from {enc_path}...", file=sys.stderr)
    with open(enc_path, 'rb') as f:
        enc = pickle.load(f)
    q_sparse = enc['q_sparse']
    c_sparse = enc['c_sparse']
    q_colbert = enc['q_colbert']
    c_colbert = enc['c_colbert']

    # ── Load or encode dense ──
    if args.use_lora:
        # Check for cached LoRA dense vectors
        if os.path.exists(args.dense_cache):
            print(f"  Loading cached LoRA dense vectors from {args.dense_cache}...", file=sys.stderr)
            data = np.load(args.dense_cache)
            q_dense = data['q_dense']
            c_dense = data['c_dense']
            print(f"  Dense: q={q_dense.shape}, c={c_dense.shape}", file=sys.stderr)
        else:
            from sentence_transformers import SentenceTransformer
            print(f"  Re-encoding dense with LoRA model from {args.lora_dir}...", file=sys.stderr)
            device = 'mps' if torch.backends.mps.is_available() else 'cpu'
            model = SentenceTransformer(args.lora_dir, device=device)

            tq = time.time()
            q_dense = model.encode(query_texts, batch_size=256, show_progress_bar=True,
                                   normalize_embeddings=True)
            print(f"  Queries encoded in {time.time()-tq:.0f}s", file=sys.stderr)

            if device == 'mps':
                torch.mps.empty_cache()

            tc = time.time()
            c_dense = model.encode(cand_texts, batch_size=512, show_progress_bar=True,
                                   normalize_embeddings=True)
            print(f"  Candidates encoded in {time.time()-tc:.0f}s", file=sys.stderr)

            if args.save_dense:
                np.savez(args.dense_cache, q_dense=q_dense, c_dense=c_dense)
                print(f"  Saved LoRA dense vectors to {args.dense_cache}", file=sys.stderr)
    else:
        q_dense = enc['q_dense']
        c_dense = enc['c_dense']
        print(f"  Using baseline dense vectors: q={q_dense.shape}, c={c_dense.shape}", file=sys.stderr)

    print(f"  Memory estimate: {(q_dense.nbytes + c_dense.nbytes) / 1e9:.1f}GB dense", file=sys.stderr)

    # ── City groups + IDs ──
    print("[4/5] Building city groups...", file=sys.stderr)
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

    print(f"  Cities: {len(cities_to_process)}", file=sys.stderr)

    # ── Precompute sparse scores per city ──
    print("[5/5] Precomputing sparse scores per city...", file=sys.stderr)
    benchmark_cities = ['杭州市', '深圳市', '台州市', '贵阳市', '淄博市']

    # Precompute city-level dense similarity matrices and sparse scores
    city_data: dict[str, dict] = {}
    for city in cities_to_process:
        cand_idx = city_groups[city]
        q_indices = [i for i, c in enumerate(query_cities) if c == city]
        if not q_indices:
            continue

        nq, nc = len(q_indices), len(cand_idx)

        # Dense similarity (raw, without weight)
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

        # Prepare colbert data for top-K reranking
        city_data[city] = {
            'q_indices': q_indices,
            'cand_idx': cand_idx,
            'd_sim': d_sim,
            's_sim': s_sim,
        }

    elapsed_load = time.time() - t0
    print(f"\n  Data loaded in {elapsed_load:.0f}s", file=sys.stderr)
    print(f"  Ready for weight sweep", file=sys.stderr)

    # ── Analyze score magnitudes ──
    print(f"\n{'='*60}", file=sys.stderr)
    print("SCORE MAGNITUDE ANALYSIS", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    for city in benchmark_cities:
        if city not in city_data:
            continue
        cd = city_data[city]
        d = cd['d_sim'].flatten()
        s = cd['s_sim'].flatten()
        # ColBERT: sample a few to estimate
        nq = len(cd['q_indices'])
        cb_samples = []
        for qi in range(min(5, nq)):
            q_cb = np.array(q_colbert[cd['q_indices'][qi]])
            for ci in range(min(10, len(cd['cand_idx']))):
                c_cb = np.array(c_colbert[cd['cand_idx'][ci]])
                cb_samples.append((q_cb @ c_cb.T).max(axis=1).sum())
        cb = np.array(cb_samples) if cb_samples else np.array([0])
        print(f"  {city}: dense [{d.min():.3f}, {d.max():.3f}] mean={d.mean():.3f} | "
              f"sparse [{s.min():.3f}, {s.max():.3f}] mean={s.mean():.3f} | "
              f"colbert [{cb.min():.1f}, {cb.max():.1f}] mean={cb.mean():.1f}",
              file=sys.stderr)

    # ── Weight sweep ──
    # ColBERT scores are ~20-40x larger than dense, so weights must compensate
    weight_configs = [
        # Current default
        (1, 1, 1),
        # Scale-aware: dense/sparse need ~20-30x more weight
        (20, 5, 1),
        (30, 10, 1),
        (15, 3, 1),
        (10, 3, 1),
        (25, 5, 1),
        (20, 10, 1),
        (30, 15, 1),
        (40, 10, 1),
        (50, 10, 1),
        # Dense-heavy
        (20, 1, 1),
        (30, 1, 1),
        (50, 1, 1),
        # ColBERT-heavy
        (1, 1, 1),
        (5, 1, 1),
        (10, 1, 1),
        (1, 1, 2),
        (1, 1, 5),
        # No colbert
        (20, 5, 0),
        (30, 10, 0),
        # No dense
        (0, 5, 1),
        (0, 10, 1),
        # No sparse
        (20, 0, 1),
        (30, 0, 1),
        # Dense-only
        (1, 0, 0),
        # Sparse-only
        (0, 1, 0),
        # ColBERT-only
        (0, 0, 1),
    ]

    topk = args.topk
    save_top = args.save_top

    print(f"\n{'='*60}", file=sys.stderr)
    print("FUSION WEIGHT SWEEP", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"  {'Weights':<12} {'Overall':>8} {'杭州':>8} {'深圳':>8} "
          f"{'台州':>8} {'贵阳':>8} {'淄博':>8}", file=sys.stderr)
    print(f"  {'-'*64}", file=sys.stderr)

    best_overall = 0
    best_weights = (1, 1, 1)
    results_table = []

    for w_dense, w_sparse, w_colbert in weight_configs:
        t_start = time.time()

        # Run matching with these weights
        top1_correct = 0
        top1_total = 0
        city_correct: dict[str, int] = {}
        city_total: dict[str, int] = {}

        for city in cities_to_process:
            if city not in city_data:
                continue
            cd = city_data[city]
            q_indices = cd['q_indices']
            cand_idx = cd['cand_idx']
            d_sim = cd['d_sim']
            s_sim = cd['s_sim']
            nq = len(q_indices)
            nc = len(cand_idx)

            fused_ds = w_dense * d_sim + w_sparse * s_sim
            actual_k = min(topk, nc)

            for qi_loc in range(nq):
                qi_glob = q_indices[qi_loc]
                scores = fused_ds[qi_loc]
                if actual_k >= nc:
                    top_local = np.argsort(-scores)
                else:
                    top_local = np.argpartition(-scores, actual_k)[:actual_k]
                    top_local = top_local[np.argsort(-scores[top_local])]

                # ColBERT reranking on top-K
                top_k_cands = top_local[:topk]
                colbert_scores = np.zeros(len(top_k_cands), dtype=np.float32)
                q_cb = np.array(q_colbert[qi_glob])

                for ck, ci_loc in enumerate(top_k_cands):
                    ci_glob = cand_idx[ci_loc]
                    c_cb = np.array(c_colbert[ci_glob])
                    sim_mat = q_cb @ c_cb.T
                    colbert_scores[ck] = sim_mat.max(axis=1).sum()

                final_scores = fused_ds[qi_loc, top_k_cands] + w_colbert * colbert_scores
                best_pos = np.argmax(final_scores)
                ci_loc = top_k_cands[best_pos]
                ci_glob = cand_idx[ci_loc]

                verified = c_ids[ci_glob] == q_ids[qi_glob]
                top1_total += 1
                if verified:
                    top1_correct += 1

                city_total[city] = city_total.get(city, 0) + 1
                if verified:
                    city_correct[city] = city_correct.get(city, 0) + 1

        overall = top1_correct / top1_total * 100 if top1_total else 0
        elapsed_w = time.time() - t_start

        city_accs = {}
        for bc in benchmark_cities:
            if bc in city_total:
                city_accs[bc] = city_correct.get(bc, 0) / city_total[bc] * 100
            else:
                city_accs[bc] = 0

        label = f"({w_dense},{w_sparse},{w_colbert})"
        print(f"  {label:<12} {overall:>7.2f}% {city_accs.get('杭州市', 0):>7.1f}% "
              f"{city_accs.get('深圳市', 0):>7.1f}% {city_accs.get('台州市', 0):>7.1f}% "
              f"{city_accs.get('贵阳市', 0):>7.1f}% {city_accs.get('淄博市', 0):>7.1f}% "
              f"  [{elapsed_w:.1f}s]", file=sys.stderr)

        results_table.append({
            'weights': label,
            'overall': overall,
            **{bc: city_accs.get(bc, 0) for bc in benchmark_cities},
        })

        if overall > best_overall:
            best_overall = overall
            best_weights = (w_dense, w_sparse, w_colbert)

    print(f"  {'-'*64}", file=sys.stderr)
    print(f"  BEST: {best_weights} = {best_overall:.2f}%", file=sys.stderr)

    # ── Fine grid around best ──
    print(f"\n  Fine grid around best {best_weights}...", file=sys.stderr)
    wd, ws, wc = best_weights
    fine_configs = []
    for d in [max(0, wd-2), wd-1, wd, wd+1, wd+2]:
        for s in [max(0, ws-1), ws, ws+1]:
            for c in [max(0, wc-1), wc, wc+1]:
                if (d, s, c) not in weight_configs and d >= 0 and s >= 0 and c >= 0:
                    fine_configs.append((d, s, c))

    if fine_configs:
        for w_dense, w_sparse, w_colbert in fine_configs:
            if w_dense == 0 and w_sparse == 0 and w_colbert == 0:
                continue
            t_start = time.time()

            top1_correct = 0
            top1_total = 0
            city_correct = {}
            city_total = {}

            for city in cities_to_process:
                if city not in city_data:
                    continue
                cd = city_data[city]
                q_indices = cd['q_indices']
                cand_idx = cd['cand_idx']
                d_sim = cd['d_sim']
                s_sim = cd['s_sim']
                nq = len(q_indices)
                nc = len(cand_idx)

                fused_ds = w_dense * d_sim + w_sparse * s_sim
                actual_k = min(topk, nc)

                for qi_loc in range(nq):
                    qi_glob = q_indices[qi_loc]
                    scores = fused_ds[qi_loc]
                    if actual_k >= nc:
                        top_local = np.argsort(-scores)
                    else:
                        top_local = np.argpartition(-scores, actual_k)[:actual_k]
                        top_local = top_local[np.argsort(-scores[top_local])]

                    top_k_cands = top_local[:topk]
                    colbert_scores = np.zeros(len(top_k_cands), dtype=np.float32)
                    q_cb = np.array(q_colbert[qi_glob])

                    for ck, ci_loc in enumerate(top_k_cands):
                        ci_glob = cand_idx[ci_loc]
                        c_cb = np.array(c_colbert[ci_glob])
                        sim_mat = q_cb @ c_cb.T
                        colbert_scores[ck] = sim_mat.max(axis=1).sum()

                    final_scores = fused_ds[qi_loc, top_k_cands] + w_colbert * colbert_scores
                    best_pos = np.argmax(final_scores)
                    ci_loc = top_k_cands[best_pos]
                    ci_glob = cand_idx[ci_loc]

                    verified = c_ids[ci_glob] == q_ids[qi_glob]
                    top1_total += 1
                    if verified:
                        top1_correct += 1

                    city_total[city] = city_total.get(city, 0) + 1
                    if verified:
                        city_correct[city] = city_correct.get(city, 0) + 1

            overall = top1_correct / top1_total * 100 if top1_total else 0
            city_accs = {}
            for bc in benchmark_cities:
                if bc in city_total:
                    city_accs[bc] = city_correct.get(bc, 0) / city_total[bc] * 100
                else:
                    city_accs[bc] = 0

            label = f"({w_dense},{w_sparse},{w_colbert})"
            print(f"  {label:<12} {overall:>7.2f}% {city_accs.get('杭州市', 0):>7.1f}% "
                  f"{city_accs.get('深圳市', 0):>7.1f}% {city_accs.get('台州市', 0):>7.1f}% "
                  f"{city_accs.get('贵阳市', 0):>7.1f}% {city_accs.get('淄博市', 0):>7.1f}% "
                  f"  [{time.time()-t_start:.1f}s]", file=sys.stderr)

            if overall > best_overall:
                best_overall = overall
                best_weights = (w_dense, w_sparse, w_colbert)

        print(f"  {'-'*64}", file=sys.stderr)

    print(f"\n  FINAL BEST: {best_weights} = {best_overall:.2f}%", file=sys.stderr)
    total_elapsed = time.time() - t0
    print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)", file=sys.stderr)


if __name__ == '__main__':
    main()
