#!/usr/bin/env python3
"""
Phase 1: Extract training data from m3_full_match.csv for LoRA fine-tuning.

Positive pairs: ALL verified=True (including rank > 1)
Hard negatives: verified=False from top-10, filtered for near-duplicate names
Excludes 5 benchmark cities (Hangzhou, Shenzhen, Taizhou, Guiyang, Zibo)
Output: JSONL for sentence-transformers training
"""
import argparse
import json
import sys
from collections import defaultdict

import pandas as pd


def normalize_text(text: str) -> str:
    """Simple normalization for dedup checking."""
    return text.strip().replace(" ", "").replace("　", "")


def edit_distance(s1: str, s2: str) -> int:
    """Simple Levenshtein distance for short strings."""
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='match_results/m3_full_match.csv')
    parser.add_argument('--query-file', default='完整 qcc_extracted_v3.csv',
                        help='Original query file with name + address columns')
    parser.add_argument('--candidates-file', default='副本天眼查数据 - 1.xlsx',
                        help='Original candidate file with name + address')
    parser.add_argument('--output', default='match_results/m3_cache/training_data.jsonl')
    parser.add_argument('--negatives-per-positive', type=int, default=5)
    parser.add_argument('--exclude-cities', default='杭州市,深圳市,台州市,贵阳市,淄博市')
    args = parser.parse_args()

    exclude_cities = set(args.exclude_cities.split(','))

    # Load original files for full name+address text
    print(f"Loading query file: {args.query_file}...", file=sys.stderr)
    if args.query_file.endswith('.xlsx'):
        query_df = pd.read_excel(args.query_file)
    else:
        query_df = pd.read_csv(args.query_file)
    q_name_col = [c for c in query_df.columns if 'Door Name' in c][0]
    q_addr_col = [c for c in query_df.columns if 'Door Address' in c][0]
    q_id_col = [c for c in query_df.columns if 'Door Social Credit' in c][0]

    # Build query text lookup: id -> "name address"
    query_texts = {}
    for _, row in query_df.iterrows():
        qid = str(row[q_id_col]).strip()
        name = str(row[q_name_col]).strip() if pd.notna(row[q_name_col]) else ''
        addr = str(row[q_addr_col]).strip() if pd.notna(row[q_addr_col]) else ''
        query_texts[qid] = f"{name} {addr}".strip()

    print(f"Loading candidates file: {args.candidates_file}...", file=sys.stderr)
    if args.candidates_file.endswith('.xlsx'):
        cand_df = pd.read_excel(args.candidates_file)
    else:
        cand_df = pd.read_csv(args.candidates_file)

    # Build candidate text lookup: id -> "name address"
    cand_texts = {}
    for _, row in cand_df.iterrows():
        cid = str(row.get('统一社会信用代码', '')).strip()
        name = str(row.get('公司名称', '')).strip() if pd.notna(row.get('公司名称')) else ''
        addr = str(row.get('注册地址', '')).strip() if pd.notna(row.get('注册地址')) else ''
        if cid:
            cand_texts[cid] = f"{name} {addr}".strip()

    print(f"  Query texts: {len(query_texts)}, Candidate texts: {len(cand_texts)}", file=sys.stderr)

    print(f"Loading {args.input}...", file=sys.stderr)
    df = pd.read_csv(args.input)
    print(f"  Total rows: {len(df)}", file=sys.stderr)

    # Group by query
    query_groups = defaultdict(list)
    for _, row in df.iterrows():
        qid = row['query_id']
        query_groups[qid].append(row)

    print(f"  Unique queries: {len(query_groups)}", file=sys.stderr)

    # Build training samples
    samples = []
    skipped_cities = 0
    no_positive = 0
    no_negative = 0

    for qid, rows in query_groups.items():
        # Get city from first row
        city = str(rows[0].get('query_city', ''))
        if city in exclude_cities:
            skipped_cities += 1
            continue

        # Find positives and negatives
        positives = [r for r in rows if r.get('verified', False)]
        negatives = [r for r in rows if not r.get('verified', False) and r.get('rank', 0) > 0]

        if not positives:
            no_positive += 1
            continue
        if not negatives:
            no_negative += 1
            continue

        # Use highest-ranked positive as canonical for near-dup filtering
        positives.sort(key=lambda r: r['rank'])
        best_positive = positives[0]
        best_pos_id = str(best_positive.get('matched_id', '')).strip()
        pos_name = normalize_text(cand_texts.get(best_pos_id, str(best_positive.get('matched_name', ''))))

        # Filter hard negatives: exclude near-duplicate names
        filtered_negs = []
        for neg in negatives:
            neg_name = normalize_text(str(neg.get('matched_name', '')))
            # Skip if too similar (likely same entity, different ID)
            if pos_name and neg_name and len(pos_name) > 2 and len(neg_name) > 2:
                dist = edit_distance(pos_name[:20], neg_name[:20])
                if dist < 3:
                    continue
            filtered_negs.append(neg)

        if not filtered_negs:
            filtered_negs = negatives[:args.negatives_per_positive]

        # Use full name+address from original files
        qid_str = str(qid).strip()
        query_text = query_texts.get(qid_str, str(rows[0].get('query_name', '')))

        for pos in positives:
            pos_id_str = str(pos.get('matched_id', '')).strip()
            pos_text = cand_texts.get(pos_id_str, str(pos.get('matched_name', '')))

            # Check near-dup against negatives
            sample_negs = []
            pos_n = normalize_text(pos_text)
            for neg in filtered_negs[:args.negatives_per_positive * 2]:
                neg_id_str = str(neg.get('matched_id', '')).strip()
                neg_text = cand_texts.get(neg_id_str, str(neg.get('matched_name', '')))
                neg_n = normalize_text(neg_text)
                if pos_n and neg_n and len(pos_n) > 2 and len(neg_n) > 2:
                    if edit_distance(pos_n[:20], neg_n[:20]) < 3:
                        continue
                sample_negs.append(neg_text)
                if len(sample_negs) >= args.negatives_per_positive:
                    break

            if not sample_negs:
                continue

            samples.append({
                'query': query_text,
                'positive': pos_text,
                'negatives': sample_negs,
            })

    # Write JSONL
    import os
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

    print(f"\n  Training samples: {len(samples)}", file=sys.stderr)
    print(f"  Skipped (benchmark cities): {skipped_cities}", file=sys.stderr)
    print(f"  Skipped (no positive): {no_positive}", file=sys.stderr)
    print(f"  Skipped (no negative): {no_negative}", file=sys.stderr)
    print(f"  Output: {args.output}", file=sys.stderr)


if __name__ == '__main__':
    main()
