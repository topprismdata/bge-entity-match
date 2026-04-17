"""
BGE-M3 Fused + bge-reranker-v2-gemma Benchmark (5 cities)
Uses correct LLM prompt format for Gemma-based reranker.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_models():
    print("Loading BGE-M3...")
    bge_m3 = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, device='mps')
    print("BGE-M3 loaded.")

    print("Loading bge-reranker-v2-gemma...")
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-gemma')
    model = AutoModelForCausalLM.from_pretrained(
        'BAAI/bge-reranker-v2-gemma', torch_dtype=torch.float16,
    )
    model = model.to('mps')
    model.eval()
    print("Gemma reranker loaded.")

    return bge_m3, tokenizer, model


def gemma_rerank(tokenizer, model, pairs: list[tuple[str, str]], batch_size: int = 8) -> np.ndarray:
    """Score query-passage pairs using Gemma LLM reranker with correct prompt."""
    prompt = ("Given a query A and a passage B, determine whether the passage "
              "contains an answer to the query by providing a prediction of "
              "either 'Yes' or 'No'.")
    sep = "\n"
    yes_loc = tokenizer('Yes', add_special_tokens=False)['input_ids'][0]

    all_scores = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        batch_ids = []
        for query, passage in batch:
            q_ids = tokenizer.encode(
                f'A: {query}', add_special_tokens=False, max_length=256, truncation=True,
            )
            p_ids = tokenizer.encode(
                f'B: {passage}', add_special_tokens=False, max_length=256, truncation=True,
            )
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            sep_ids = tokenizer.encode(sep, add_special_tokens=False)
            input_ids = [tokenizer.bos_token_id] + q_ids + sep_ids + p_ids + sep_ids + prompt_ids
            if len(input_ids) > 512:
                input_ids = input_ids[:512]
            batch_ids.append(input_ids)

        max_l = max(len(x) for x in batch_ids)
        padded = [x + [tokenizer.pad_token_id] * (max_l - len(x)) for x in batch_ids]
        attn = [[1] * len(x) + [0] * (max_l - len(x)) for x in batch_ids]

        input_ids_t = torch.tensor(padded, dtype=torch.long).to('mps')
        attn_t = torch.tensor(attn, dtype=torch.long).to('mps')

        with torch.no_grad():
            logits = model(input_ids=input_ids_t, attention_mask=attn_t, return_dict=True).logits
            scores = logits[:, -1, yes_loc].view(-1).float().cpu().numpy()
        all_scores.append(scores)

    return np.concatenate(all_scores)


def run_city_benchmark(
    bge_m3, tokenizer, ce_model,
    query_file: str, candidate_file: str, city_name: str,
):
    print(f"\n{'='*60}")
    print(f"City: {city_name}")
    print(f"{'='*60}")

    q_id_col = 'Door Social Credit Code \n门店社会信用证代码'
    q_name_col = 'Door Name \n门店名称'
    q_addr_col = 'Door Address\n门店营业地址'
    c_id_col = '统一社会信用代码'
    c_name_col = '公司名称'
    c_addr_col = '注册地址'

    queries = pd.read_csv(query_file)
    candidates = pd.read_csv(candidate_file)
    print(f"  Queries: {len(queries)}, Candidates: {len(candidates)}")

    query_texts = []
    for _, row in queries.iterrows():
        name = str(row[q_name_col]).strip() if pd.notna(row[q_name_col]) else ''
        addr = str(row[q_addr_col]).strip() if pd.notna(row[q_addr_col]) else ''
        query_texts.append(f"{name} {addr}".strip())

    cand_texts = []
    for _, row in candidates.iterrows():
        name = str(row[c_name_col]).strip() if pd.notna(row[c_name_col]) else ''
        addr = str(row[c_addr_col]).strip() if pd.notna(row[c_addr_col]) else ''
        cand_texts.append(f"{name} {addr}".strip())

    # BGE-M3 triple retrieval
    print("  Encoding queries with BGE-M3...")
    t0 = time.time()
    q_result = bge_m3.encode(
        query_texts, return_dense=True, return_sparse=True,
        return_colbert_vecs=True, batch_size=64, max_length=256,
    )
    print(f"  Queries encoded in {time.time()-t0:.1f}s")

    print("  Encoding candidates with BGE-M3...")
    t0 = time.time()
    c_result = bge_m3.encode(
        cand_texts, return_dense=True, return_sparse=True,
        return_colbert_vecs=True, batch_size=256, max_length=256,
    )
    print(f"  Candidates encoded in {time.time()-t0:.1f}s")

    # Fused scores
    print("  Computing fused scores...")
    q_dense = np.array(q_result['dense_vecs'])
    c_dense = np.array(c_result['dense_vecs'])
    dense_sim = q_dense @ c_dense.T

    sparse_sim = np.zeros((len(query_texts), len(cand_texts)))
    for i in range(len(query_texts)):
        q_sparse = q_result['lexical_weights'][i]
        for token, weight in q_sparse.items():
            for j in range(len(cand_texts)):
                c_sparse = c_result['lexical_weights'][j]
                if token in c_sparse:
                    sparse_sim[i, j] += weight * c_sparse[token]

    colbert_sim = np.zeros((len(query_texts), len(cand_texts)))
    for i in range(len(query_texts)):
        q_colbert = np.array(q_result['colbert_vecs'][i])
        for j in range(len(cand_texts)):
            c_colbert = np.array(c_result['colbert_vecs'][j])
            sim_matrix = q_colbert @ c_colbert.T
            colbert_sim[i, j] = sim_matrix.max(axis=1).sum()

    fused_scores = dense_sim + sparse_sim + colbert_sim

    q_ids = queries[q_id_col].values
    c_ids = candidates[c_id_col].values

    # M3 Fused only
    fused_top_indices = np.argsort(-fused_scores, axis=1)
    n = len(queries)
    fused_hits = {'top1': 0, 'top5': 0, 'top10': 0, 'top25': 0, 'top50': 0, 'top100': 0}
    thresholds = [1, 5, 10, 25, 50, 100]

    for i in range(n):
        true_id = str(q_ids[i]).strip()
        for k, idx in enumerate(fused_top_indices[i][:100]):
            if str(c_ids[idx]).strip() == true_id:
                for t in thresholds:
                    if k < t:
                        fused_hits[f'top{t}'] += 1
                break

    print(f"\n  M3 Fused (no CE):")
    for t in thresholds:
        h = fused_hits[f'top{t}']
        print(f"    Top-{t}: {h}/{n} = {h/n*100:.1f}%")

    # Gemma CE reranking
    ce_results = {}
    for topk in [10, 25, 50]:
        print(f"\n  Reranking top-{topk} with Gemma CE (LLM prompt)...")
        t0 = time.time()

        ce_hits = {'top1': 0, 'top5': 0, 'top10': 0}

        for qi in range(n):
            top_k_indices = fused_top_indices[qi][:topk]
            pairs = [(query_texts[qi], cand_texts[ci]) for ci in top_k_indices]

            scores = gemma_rerank(tokenizer, ce_model, pairs, batch_size=8)
            ce_order = np.argsort(-scores)

            true_id = str(q_ids[qi]).strip()
            for new_rank, orig_pos in enumerate(ce_order):
                ci = top_k_indices[orig_pos]
                if str(c_ids[ci]).strip() == true_id:
                    if new_rank == 0: ce_hits['top1'] += 1
                    if new_rank < 5: ce_hits['top5'] += 1
                    if new_rank < 10: ce_hits['top10'] += 1
                    break

        elapsed = time.time() - t0
        ce_results[topk] = ce_hits
        print(f"    Gemma CE top-{topk} in {elapsed:.1f}s:")
        for t in [1, 5, 10]:
            h = ce_hits[f'top{t}']
            print(f"      Top-{t}: {h}/{n} = {h/n*100:.1f}%")

    return {
        'city': city_name,
        'n_queries': n,
        'n_candidates': len(candidates),
        'fused': fused_hits,
        'ce_gemma': ce_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='match_results/gemma_ce_benchmark.json')
    args = parser.parse_args()

    bge_m3, tokenizer, ce_model = load_models()

    cities = {
        '淄博市': ('淄博_城市一致_有答案.csv', '淄博_候选库.csv'),
        '深圳市': ('match_results/city_benchmarks/深圳市_城市一致_有答案.csv',
                    'match_results/city_benchmarks/深圳市_候选库.csv'),
        '杭州市': ('match_results/city_benchmarks/杭州市_城市一致_有答案.csv',
                    'match_results/city_benchmarks/杭州市_候选库.csv'),
        '台州市': ('match_results/city_benchmarks/台州市_城市一致_有答案.csv',
                    'match_results/city_benchmarks/台州市_候选库.csv'),
        '贵阳市': ('match_results/city_benchmarks/贵阳市_城市一致_有答案.csv',
                    'match_results/city_benchmarks/贵阳市_候选库.csv'),
    }

    all_results = []
    for city, (qf, cf) in cities.items():
        if not os.path.exists(qf):
            print(f"  SKIP {city}: {qf} not found")
            continue
        result = run_city_benchmark(bge_m3, tokenizer, ce_model, qf, cf, city)
        all_results.append(result)

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY: M3 Fused + Gemma CE (correct LLM prompt)")
    print(f"{'='*70}")
    print(f"{'City':<8} {'N':>5} {'Fused':>8} {'CE-10':>8} {'CE-25':>8} {'CE-50':>8}")
    print('-'*45)
    for r in all_results:
        fused_t1 = r['fused']['top1'] / r['n_queries'] * 100
        ce10 = r['ce_gemma'][10]['top1'] / r['n_queries'] * 100
        ce25 = r['ce_gemma'][25]['top1'] / r['n_queries'] * 100
        ce50 = r['ce_gemma'][50]['top1'] / r['n_queries'] * 100
        print(f"{r['city']:<8} {r['n_queries']:>5} {fused_t1:>7.1f}% {ce10:>7.1f}% {ce25:>7.1f}% {ce50:>7.1f}%")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
