#!/usr/bin/env python3
"""
BGE Cached Embedding Match — 缓存嵌入 + 城市分治 + 信用代码验证
用法：
  # Step 1: 编码并缓存候选嵌入（只跑一次）
  python cached_match.py encode --candidates enterprises.xlsx \
      --c-id credit_code --c-name company_name --c-addr reg_addr \
      --cache-dir ./emb_cache

  # Step 2: 编码查询 + 匹配 + 验证
  python cached_match.py match --query stores.csv \
      --q-id store_id --q-name store_name --q-addr store_addr \
      --q-gt credit_code \
      --candidates enterprises.xlsx \
      --c-id credit_code --c-name company_name --c-addr reg_addr \
      --c-city city \
      --cache-dir ./emb_cache \
      --topk 10 \
      --output results.csv

核心优化：
  1. 候选嵌入只编码一次，缓存到 disk（numpy .npy）
  2. 城市分治：从查询地址提取城市，按城市索引候选
  3. 自动过滤省名脏数据（候选数 <20 的省级条目）
  4. 信用代码验证：匹配后自动核对 query_id == matched_id
"""
import os, sys, argparse, re, time
sys.path.insert(0, os.path.dirname(__file__))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path

# ── 复用 match.py 的函数 ──
from match import clean_text, build_query_text, build_cand_text, encode_texts

def build_text_simple(name, addr):
    parts = []
    n = clean_text(name)
    a = clean_text(addr)
    if n: parts.append(n)
    if a: parts.append(a)
    return ' '.join(parts)

def extract_city_from_addr(addr, valid_cities_sorted):
    """从地址中提取城市名，优先匹配最长（最具体）的城市名"""
    if not isinstance(addr, str): return ''
    for city in valid_cities_sorted:
        if city in addr:
            return city
    return ''

def get_valid_cities(cand_df, city_col):
    """获取有效城市列表，过滤掉省名等脏数据"""
    if not city_col or city_col not in cand_df.columns:
        return []
    city_counts = cand_df[city_col].astype(str).str.strip().value_counts()
    municipalities = {'北京市', '上海市', '天津市', '重庆市'}
    valid = set()
    for city, count in city_counts.items():
        if city in municipalities or (count >= 20 and city.endswith('市')):
            valid.add(city)
        elif city.endswith('市'):
            valid.add(city)
    return sorted(valid, key=len, reverse=True)

# ── 子命令: encode ──
def cmd_encode(args):
    print(f"[encode] Loading candidates...", file=sys.stderr)
    c_path = Path(args.candidates)
    cand_df = pd.read_csv(c_path) if c_path.suffix == '.csv' else pd.read_excel(c_path)
    print(f"  {len(cand_df)} candidates", file=sys.stderr)

    c_texts = [build_text_simple(row.get(args.c_name, ''), row.get(args.c_addr, ''))
               for _, row in cand_df.iterrows()]

    print(f"[encode] Encoding {len(c_texts)} texts...", file=sys.stderr)
    c_embs = encode_texts(c_texts, batch_size=64)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / 'c_embs.npy', c_embs)
    print(f"[encode] Saved {c_embs.shape} to {cache_dir / 'c_embs.npy'}", file=sys.stderr)

# ── 子命令: match ──
def cmd_match(args):
    t0 = time.time()
    print(f"[1/5] Loading data...", file=sys.stderr)

    # Load candidates
    c_path = Path(args.candidates)
    cand_df = pd.read_csv(c_path) if c_path.suffix == '.csv' else pd.read_excel(c_path)

    # Load queries
    q_path = Path(args.query)
    query_df = pd.read_csv(q_path) if q_path.suffix == '.csv' else pd.read_excel(q_path)
    print(f"  Queries: {len(query_df)}, Candidates: {len(cand_df)}", file=sys.stderr)

    # ── City extraction ──
    valid_cities = get_valid_cities(cand_df, args.c_city)
    print(f"  Valid cities: {len(valid_cities)}", file=sys.stderr)

    if valid_cities and args.q_addr:
        query_df['_city'] = query_df[args.q_addr].apply(
            lambda a: extract_city_from_addr(a, valid_cities))
        city_found = (query_df['_city'] != '').sum()
        print(f"  City extracted: {city_found}/{len(query_df)} ({city_found/len(query_df)*100:.1f}%)", file=sys.stderr)
    else:
        query_df['_city'] = ''

    # ── Encode queries ──
    print(f"[2/5] Encoding {len(query_df)} query texts...", file=sys.stderr)
    q_texts = [build_text_simple(row.get(args.q_name, ''), row.get(args.q_addr, ''))
               for _, row in query_df.iterrows()]
    q_embs = encode_texts(q_texts, batch_size=32)

    # ── Load cached candidate embeddings ──
    print(f"[3/5] Loading cached embeddings...", file=sys.stderr)
    cache_dir = Path(args.cache_dir)
    c_embs = np.load(cache_dir / 'c_embs.npy')
    print(f"  Shape: {c_embs.shape}", file=sys.stderr)

    # Free model memory
    from match import _model, _tokenizer
    if _model is not None:
        del _model, _tokenizer
        import match
        match._model = None
        match._tokenizer = None
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # ── Match per city ──
    print(f"[4/5] Matching...", file=sys.stderr)
    topk = args.topk

    # Build city -> candidate index
    city_groups = {}
    if args.c_city and args.c_city in cand_df.columns:
        for i, city in enumerate(cand_df[args.c_city].astype(str).str.strip()):
            city_groups.setdefault(city, []).append(i)

    cand_codes = cand_df[args.c_id].astype(str).str.strip().str.upper().values
    query_codes = query_df[args.q_id].astype(str).str.strip().str.upper().values

    results = []
    for qi in range(len(query_df)):
        q_city = query_df.iloc[qi]['_city']
        gt_code = query_codes[qi]

        if q_city and q_city in city_groups:
            cand_idx = city_groups[q_city]
        else:
            cand_idx = None

        if cand_idx is not None:
            city_c_embs = c_embs[cand_idx]
            city_cand_codes = cand_codes[cand_idx]
        else:
            city_c_embs = c_embs
            cand_idx = list(range(len(cand_df)))
            city_cand_codes = cand_codes

        actual_k = min(topk, len(cand_idx))
        if actual_k == 0: continue

        sims = np.dot(q_embs[qi:qi+1], city_c_embs.T)[0]

        if len(sims) <= actual_k:
            top_local = np.argsort(-sims)
        else:
            top_local = np.argpartition(-sims, actual_k)[:actual_k]
            top_local = top_local[np.argsort(-sims[top_local])]

        for rank, lci in enumerate(top_local[:actual_k]):
            ci = cand_idx[lci]
            sim = float(sims[lci])
            matched_code = city_cand_codes[lci]
            verified = (matched_code == gt_code) if args.q_gt else None
            results.append({
                args.q_id: query_df.iloc[qi][args.q_id],
                'query_name': query_df.iloc[qi].get(args.q_name, ''),
                'city': q_city or 'unknown',
                'matched_' + args.c_id: cand_df.iloc[ci][args.c_id],
                'matched_name': cand_df.iloc[ci].get(args.c_name, ''),
                'rank': rank + 1,
                'similarity': round(sim, 4),
                'verified': verified,
            })

        if (qi + 1) % 2000 == 0:
            print(f"  {qi+1}/{len(query_df)}", file=sys.stderr)

    # ── Save ──
    print(f"[5/5] Saving...", file=sys.stderr)
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False, encoding='utf-8-sig')

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}min)", file=sys.stderr)
    print(f"Output: {args.output} ({len(df)} rows)", file=sys.stderr)

    # Verification summary
    if args.q_gt and len(df) > 0:
        top1 = df[df['rank'] == 1]
        v = top1['verified'].sum()
        print(f"\n  Top-1 verified: {v}/{len(top1)} ({v/len(top1)*100:.2f}%)", file=sys.stderr)

# ── CLI ──
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command')

    # encode
    p_enc = sub.add_parser('encode', help='Encode and cache candidate embeddings')
    p_enc.add_argument('--candidates', required=True)
    p_enc.add_argument('--c-id', required=True)
    p_enc.add_argument('--c-name', default='')
    p_enc.add_argument('--c-addr', default='')
    p_enc.add_argument('--cache-dir', default='./emb_cache')

    # match
    p_match = sub.add_parser('match', help='Match queries using cached embeddings')
    p_match.add_argument('--query', required=True)
    p_match.add_argument('--candidates', required=True)
    p_match.add_argument('--q-id', required=True)
    p_match.add_argument('--q-name', default='')
    p_match.add_argument('--q-addr', default='')
    p_match.add_argument('--q-gt', default='', help='Query column with ground-truth ID for verification')
    p_match.add_argument('--c-id', required=True)
    p_match.add_argument('--c-name', default='')
    p_match.add_argument('--c-addr', default='')
    p_match.add_argument('--c-city', default='', help='Candidate city column for pre-filter')
    p_match.add_argument('--cache-dir', default='./emb_cache')
    p_match.add_argument('--topk', type=int, default=10)
    p_match.add_argument('--output', default='match_results.csv')

    args = ap.parse_args()
    if args.command == 'encode':
        cmd_encode(args)
    elif args.command == 'match':
        cmd_match(args)
    else:
        ap.print_help()

if __name__ == '__main__':
    main()
