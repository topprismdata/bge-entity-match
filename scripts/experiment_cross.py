#!/usr/bin/env python3
"""
城市分治 + CrossEncoder 精排对比实验
对比 BGE粗排 vs BGE+CrossEncoder 在城市分治下的效果

用法：
  python experiment_cross.py \
      --query stores.csv \
      --candidates enterprises.xlsx \
      --q-id code --q-name store_name --q-addr store_addr \
      --c-id credit_code --c-name company_name --c-addr reg_addr \
      --c-city city \
      --topk 50 \
      --max-cities 10 \
      --output-dir ./exp_results

输出：
  - JSON 结果文件（含每城市详情）
  - Markdown 报告（含对比表格）
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import sys
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import time, re, json, argparse
from pathlib import Path

from match import clean_text, encode_texts

# ── 文本构建 ──
def cl_name(n):
    if not n: return ""
    n = str(n)
    if '-' in n: n = n.split('-',1)[1]
    n = re.sub(r'店$','',n)
    n = re.sub(r'[（(][^）)]*[）)]','',n)
    n = re.sub(r'[·\-\s]+',' ',n).strip()
    return n

def cl_addr(a):
    if not a: return ""
    a = str(a)
    a = re.sub(r'^(上海市?|北京|天津|重庆)','',a)
    a = re.sub(r'(眼镜店|眼镜城|眼镜)','',a)
    a = a.replace('０','0').replace('１','1')
    a = re.sub(r'\s+',' ',a).strip()
    return a

def cl_company(n):
    if not n: return ""
    n = str(n)
    for s in ['有限公司','有限责任公司','股份有限公司','个人独资企业','个体工商户','合伙企业','工作室']:
        n = n.replace(s,'')
    n = re.sub(r'[（(][^）)]*[）)]','',n)
    n = re.sub(r'\s+',' ',n).strip()
    return n

def build_text(row, name_col, addr_col):
    parts = []
    n = cl_name(row.get(name_col, ''))
    a = cl_addr(row.get(addr_col, ''))
    if n: parts.append(n)
    if a: parts.append(a)
    return ' '.join(parts)

# ── BGE ──
def topk_batch(q_embs, cand_embs, topk, batch=500):
    n_q, n_c = q_embs.shape[0], cand_embs.shape[0]
    top_idx = np.zeros((n_q, topk), dtype=np.intp)
    for i in range(0, n_q, batch):
        j = min(i+batch, n_q)
        sims = np.dot(q_embs[i:j], cand_embs.T)
        batch_topk_idx = np.argpartition(-sims, topk, axis=1)[:, :topk]
        batch_sims = np.take_along_axis(sims, batch_topk_idx, axis=1)
        sorted_order = np.argsort(-batch_sims, axis=1)
        top_idx[i:j] = np.take_along_axis(batch_topk_idx, sorted_order, axis=1)
    return top_idx

# ── CrossEncoder ──
_ce_model = None

def load_cross(model_name='BAAI/bge-reranker-v2-m3'):
    global _ce_model
    if _ce_model is None:
        from sentence_transformers import CrossEncoder
        print(f"  [CrossEncoder] Loading {model_name}...", flush=True)
        _ce_model = CrossEncoder(model_name, max_length=512)
    return _ce_model

def cross_rerank(q_texts, cand_texts, top_idx, topk_rerank=50):
    ce = load_cross()
    reranked = []
    for qt, ti in zip(q_texts, top_idx):
        pairs = [[qt, cand_texts[j]] for j in ti[:topk_rerank]]
        scores = ce.predict(pairs)
        sorted_idx = np.argsort(-scores)
        reranked.append(ti[:topk_rerank][sorted_idx])
    return np.array(reranked)

# ── 单城市评估 ──
def eval_city(df_city_stores, cand_texts, cand_embs, cand_ids,
              q_name_col, q_addr_col, gt_col,
              city_name, topk=50, use_cross=False, topk_rerank=50):
    t0 = time.time()

    q_texts_list = [build_text(row, q_name_col, q_addr_col) for _, row in df_city_stores.iterrows()]
    valid_mask = [bool(t.strip()) for t in q_texts_list]
    q_texts_valid = [t for t in q_texts_list if t.strip()]
    df_valid = df_city_stores[valid_mask].reset_index(drop=True)

    if len(q_texts_valid) == 0:
        return None

    q_embs = encode_texts(q_texts_valid)
    actual_k = min(topk, len(cand_embs))
    top_idx = topk_batch(q_embs, cand_embs, actual_k)

    # CrossEncoder 精排
    t_cross = 0
    if use_cross:
        t1 = time.time()
        top_idx = cross_rerank(q_texts_valid, cand_texts, top_idx, topk_rerank=topk_rerank)
        t_cross = time.time() - t1

    gt_values = df_valid[gt_col].values if gt_col in df_valid.columns else df_valid.index.values
    id2idx = {str(v).strip().upper(): i for i, v in enumerate(cand_ids)}
    t1_acc, t3_acc, t10_acc = 0, 0, 0
    n = len(df_valid)
    for i, gt_val in enumerate(gt_values):
        gi = id2idx.get(str(gt_val).strip().upper(), -1)
        if gi == -1: continue
        k = top_idx[i]
        if k[0] == gi: t1_acc += 1
        if gi in k[:min(3, len(k))]: t3_acc += 1
        if gi in k[:min(10, len(k))]: t10_acc += 1

    elapsed = time.time() - t0
    return {
        'city': city_name,
        'K': topk,
        'Top1': t1_acc / n,
        'Top3': t3_acc / n,
        'Top10': t10_acc / n,
        'n': n,
        'n_cand': len(cand_embs),
        'time': elapsed,
        'cross_time': t_cross,
        'coverage': n / len(df_city_stores) if len(df_city_stores) > 0 else 0,
    }

# ── 主程序 ──
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--query', required=True)
    ap.add_argument('--candidates', required=True)
    ap.add_argument('--q-id', required=True, help='Query unique ID column')
    ap.add_argument('--q-name', required=True, help='Query name column')
    ap.add_argument('--q-addr', required=True, help='Query address column')
    ap.add_argument('--q-gt', required=True, help='Query ground-truth ID column (matches c-id)')
    ap.add_argument('--c-id', required=True, help='Candidate unique ID column')
    ap.add_argument('--c-name', required=True, help='Candidate name column')
    ap.add_argument('--c-addr', required=True, help='Candidate address column')
    ap.add_argument('--c-city', required=True, help='Candidate city column')
    ap.add_argument('--topk', type=int, default=50)
    ap.add_argument('--topk-rerank', type=int, default=50)
    ap.add_argument('--max-cities', type=int, default=10, help='Max number of top cities to evaluate')
    ap.add_argument('--reranker', default='BAAI/bge-reranker-v2-m3', help='CrossEncoder model name')
    ap.add_argument('--output-dir', default='./exp_results')
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("  城市分治对比实验: BGE粗排 vs BGE+CrossEncoder精排", flush=True)
    print(f"  (Top {args.max_cities} 城市 by 门店数)", flush=True)
    print("=" * 70, flush=True)

    # 加载数据
    print("\n加载数据...", flush=True)
    q_path = Path(args.query)
    query_df = pd.read_csv(q_path) if q_path.suffix == '.csv' else pd.read_excel(q_path)
    c_path = Path(args.candidates)
    cand_df = pd.read_csv(c_path) if c_path.suffix == '.csv' else pd.read_excel(c_path)
    cand_df = cand_df[cand_df[args.c_id].notna()].copy()
    cand_df = cand_df.drop_duplicates(subset=args.c_id)
    print(f"  候选库:{len(cand_df):,} | 门店:{len(query_df):,}", flush=True)

    # 合并获取城市信息
    gt_col = args.q_gt
    df_merged = query_df.merge(
        cand_df[[args.c_id, args.c_name, args.c_addr, args.c_city]].rename(
            columns={args.c_id: gt_col}),
        on=gt_col, how='inner'
    )
    df_merged[args.c_city] = df_merged[args.c_city].astype(str).str.strip()
    cand_df[args.c_city] = cand_df[args.c_city].astype(str).str.strip()

    # 按门店数降序
    city_store_counts = df_merged.groupby(args.c_city).size().sort_values(ascending=False)
    city_cand_counts = cand_df.groupby(args.c_city).size()

    top_cities = city_store_counts.head(args.max_cities)
    print(f"\nTop {args.max_cities} 城市 (门店数):", flush=True)
    for city, n_stores in top_cities.items():
        n_cand = city_cand_counts.get(city, 0)
        print(f"  {city}: {n_stores} 门店 / {n_cand:,} 候选", flush=True)
    print(f"  ... 共覆盖 {top_cities.sum()} 门店 = {top_cities.sum()/len(df_merged)*100:.0f}%\n", flush=True)

    all_results_bge = []
    all_results_cross = []
    t_start = time.time()

    for city, n_stores in top_cities.items():
        n_cand = city_cand_counts.get(city, 0)

        # 编码该城市的候选
        df_city_cand = cand_df[cand_df[args.c_city] == city].reset_index(drop=True)
        ct = [cl_company(r[args.c_name]) + ' ' + cl_addr(r[args.c_addr]) for _, r in df_city_cand.iterrows()]
        ce = encode_texts(ct)
        cc = df_city_cand[args.c_id].values

        df_city = df_merged[df_merged[args.c_city] == city].reset_index(drop=True)

        # Exp 1: BGE only
        r_bge = eval_city(df_city, ct, ce, cc, args.q_name, args.q_addr, gt_col,
                          city, topk=args.topk, use_cross=False)
        if r_bge:
            all_results_bge.append(r_bge)

        # Exp 2: BGE + CrossEncoder
        r_cross = eval_city(df_city, ct, ce, cc, args.q_name, args.q_addr, gt_col,
                            city, topk=args.topk, use_cross=True, topk_rerank=args.topk_rerank)
        if r_cross:
            all_results_cross.append(r_cross)

        delta_t1 = (r_cross['Top1'] - r_bge['Top1']) if r_bge else 0
        print(f"[{len(all_results_bge):3d}] {city:<14} 候选:{n_cand:>6,} 门店:{n_stores:>5} "
              f"BGE Top1:{r_bge['Top1']:.1%}  Cross Top1:{r_cross['Top1']:.1%}  Δ:{delta_t1:+.1%} "
              f"({r_cross['time']:.0f}s)", flush=True)

    # ── 汇总 ──
    def weighted_avg(results, field):
        valid = [r for r in results if r['n'] >= 5]
        if not valid: return 0, 0
        return sum(r[field] * r['n'] for r in valid) / sum(r['n'] for r in valid), sum(r['n'] for r in valid)

    w1_bge, n_bge = weighted_avg(all_results_bge, 'Top1')
    w3_bge, _ = weighted_avg(all_results_bge, 'Top3')
    w10_bge, _ = weighted_avg(all_results_bge, 'Top10')
    w1_cross, n_cross = weighted_avg(all_results_cross, 'Top1')
    w3_cross, _ = weighted_avg(all_results_cross, 'Top3')
    w10_cross, _ = weighted_avg(all_results_cross, 'Top10')
    total_cross_time = sum(r.get('cross_time', 0) for r in all_results_cross)

    print(f"\n{'='*70}", flush=True)
    print(f"  实验汇总 ({len(all_results_bge)} 个城市, n={n_bge})", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  BGE粗排:          Top1={w1_bge:.1%}  Top3={w3_bge:.1%}  Top10={w10_bge:.1%}", flush=True)
    print(f"  BGE+CrossEncoder: Top1={w1_cross:.1%}  Top3={w3_cross:.1%}  Top10={w10_cross:.1%}", flush=True)
    print(f"  提升:             Top1={w1_cross-w1_bge:+.1%}  Top3={w3_cross-w3_bge:+.1%}  Top10={w10_cross-w10_bge:+.1%}", flush=True)
    print(f"  CrossEncoder耗时: {total_cross_time:.0f}s", flush=True)

    # 保存结果
    output = {
        'bge_only': {
            'top1': w1_bge, 'top3': w3_bge, 'top10': w10_bge,
            'n_cities': len(all_results_bge), 'n_total': n_bge,
            'details': all_results_bge
        },
        'bge_cross': {
            'top1': w1_cross, 'top3': w3_cross, 'top10': w10_cross,
            'n_cities': len(all_results_cross), 'n_total': n_cross,
            'cross_time_sec': total_cross_time,
            'details': all_results_cross
        },
        'delta': {
            'top1': w1_cross - w1_bge,
            'top3': w3_cross - w3_bge,
            'top10': w10_cross - w10_bge,
        },
        'experiment': '城市分治+BGE vs 城市分治+BGE+CrossEncoder',
        'args': vars(args),
    }

    ts = time.strftime('%Y%m%d_%H%M%S')
    json_path = out_dir / f'cross_experiment_{ts}.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {json_path}", flush=True)

    # Markdown 报告
    md = [f"# 城市分治 CrossEncoder 对比实验", f"时间: {ts}", "",
          f"门店: {n_bge} | 城市: {len(all_results_bge)} | CrossEncoder: {args.reranker}", "",
          "## 汇总", "",
          "| 方法 | Top-1 | Top-3 | Top-10 |",
          "|------|-------|-------|--------|",
          f"| BGE粗排 | {w1_bge:.1%} | {w3_bge:.1%} | {w10_bge:.1%} |",
          f"| BGE+CrossEncoder | {w1_cross:.1%} | {w3_cross:.1%} | {w10_cross:.1%} |",
          f"| **提升** | **{w1_cross-w1_bge:+.1%}** | **{w3_cross-w3_bge:+.1%}** | **{w10_cross-w10_bge:+.1%}** |",
          "",
          "## 每城市详情", "",
          "| 城市 | 候选 | 门店 | BGE Top-1 | CE Top-1 | Δ Top-1 | CE耗时 |",
          "|------|------|------|-----------|----------|---------|--------|"]
    for b, c in zip(all_results_bge, all_results_cross):
        delta = c['Top1'] - b['Top1']
        md.append(f"| {b['city']} | {b['n_cand']:,} | {b['n']} | {b['Top1']:.1%} | {c['Top1']:.1%} | {delta:+.1%} | {c['cross_time']:.0f}s |")

    md_path = out_dir / f'cross_experiment_{ts}_report.md'
    md_path.write_text('\n'.join(md), encoding='utf-8')
    print(f"报告已保存: {md_path}", flush=True)

if __name__ == "__main__":
    main()
