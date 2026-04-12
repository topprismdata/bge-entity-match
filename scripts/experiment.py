#!/usr/bin/env python3
"""
BGE Entity Matching 实验追踪器（支持CrossEncoder精排）
用法：
  python experiment.py --query <csv> --candidates <csv> \
      --id-col <id> --name-col <name> --addr-col <addr> \
      [--city-col <city>] [--gt-id <gt_col>] \
      --name "实验描述" \
      --output-dir ./exp_results

功能：
  1. 运行多组配置实验（不同 TopK、是否城市分治、是否CrossEncoder精排）
  2. 如果提供了 gt-id，计算 Top-1/3/10 准确率
  3. 记录所有实验结果到 JSON + Markdown 报告
  4. 输出最优配置建议
"""
import os, sys, argparse, json, time, re
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# 注入 match.py 的函数
sys.path.insert(0, os.path.dirname(__file__))
from match import encode_texts, topk_batch, clean_text, build_query_text, build_cand_text
import torch

# ── CrossEncoder ──
_ce_model = None

def load_cross(model_name='BAAI/bge-reranker-v2-m3'):
    global _ce_model
    if _ce_model is None:
        from sentence_transformers import CrossEncoder
        print(f"  [CrossEncoder] Loading {model_name}...", flush=True)
        _ce_model = CrossEncoder(model_name, max_length=512)
    return _ce_model

def cross_rerank(q_texts, c_texts, top_idx, topk_rerank=50):
    ce = load_cross()
    reranked = []
    for qt, ti in zip(q_texts, top_idx):
        pairs = [[qt, c_texts[j]] for j in ti[:topk_rerank]]
        scores = ce.predict(pairs)
        sorted_idx = np.argsort(-scores)
        reranked.append(ti[:topk_rerank][sorted_idx])
    return np.array(reranked)

def run_exp(query_df, cand_df, config, args):
    """运行单次实验配置"""
    name_col = config.get('name_col', args.name_col)
    addr_col = config.get('addr_col', args.addr_col)
    use_city = config.get('use_city', bool(args.city_col))
    topk = config.get('topk', 50)
    use_cross = config.get('use_cross', False)
    topk_rerank = config.get('topk_rerank', topk)

    q_texts = [build_query_text(row, name_col, addr_col) for _, row in query_df.iterrows()]
    c_texts = [build_cand_text(row, name_col, addr_col) for _, row in cand_df.iterrows()]

    q_valid = [(i, t) for i, t in enumerate(q_texts) if t.strip()]
    q_valid_idx = [x[0] for x in q_valid]
    q_embs = encode_texts([x[1] for x in q_valid])
    c_embs = encode_texts(c_texts)
    cand_ids = cand_df[args.id_col].values.tolist()

    t_cross = 0

    if use_city and args.city_col:
        city_map = {}
        for i, row in cand_df.iterrows():
            c = str(row.get(args.city_col, '')).strip()
            city_map.setdefault(c, []).append(i)

        rows = []
        for qi, (qi_global, qt) in enumerate(q_valid):
            q_row = query_df.iloc[qi_global]
            q_city = str(q_row.get(args.city_col, '')).strip()
            cand_idx = city_map.get(q_city, list(range(len(cand_df))))
            c_embs_sub = c_embs[cand_idx]
            actual_k = min(topk, len(cand_idx))
            local = topk_batch(q_embs[qi:qi+1], c_embs_sub, actual_k)[0]

            # CrossEncoder精排
            if use_cross:
                t1 = time.time()
                city_c_texts = [c_texts[cand_idx[j]] for j in range(len(cand_idx))]
                pairs = [[qt, city_c_texts[j]] for j in local[:topk_rerank]]
                ce = load_cross()
                scores = ce.predict(pairs)
                sorted_idx = np.argsort(-scores)
                local = local[:topk_rerank][sorted_idx]
                t_cross += time.time() - t1

            for rank, lci in enumerate(local):
                rows.append({'q_idx': qi_global, 'c_idx': cand_idx[lci], 'rank': rank+1})
    else:
        actual_k = min(topk, len(c_embs))
        top_idx = topk_batch(q_embs, c_embs, actual_k)

        # CrossEncoder精排
        if use_cross:
            t1 = time.time()
            top_idx = cross_rerank([q_texts[i] for i in q_valid_idx],
                                   c_texts, top_idx, topk_rerank)
            t_cross += time.time() - t1

        rows = []
        for qi, qi_global in enumerate(q_valid_idx):
            for rank, ci in enumerate(top_idx[qi]):
                rows.append({'q_idx': qi_global, 'c_idx': ci, 'rank': rank+1})

    # 计算准确率
    if args.gt_id:
        code2idx = {c: i for i, c in enumerate(cand_ids)}
        t1 = t3 = t10 = 0
        n = len(q_valid)

        for qi, qi_global in enumerate(q_valid_idx):
            gt = query_df.iloc[qi_global][args.gt_id]
            gi = code2idx.get(gt, -1)
            if gi == -1:
                continue
            rd = [r for r in rows if r['q_idx'] == qi_global]
            top_cidx = [r['c_idx'] for r in sorted(rd, key=lambda x: x['rank'])]
            if top_cidx and top_cidx[0] == gi:
                t1 += 1
            if gi in top_cidx[:3]:
                t3 += 1
            if gi in top_cidx[:10]:
                t10 += 1

        top1 = t1 / n if n else 0
        top3 = t3 / n if n else 0
        top10 = t10 / n if n else 0
    else:
        top1 = top3 = top10 = None
        n = len(q_valid)

    return {
        'config': config,
        'top1': top1,
        'top3': top3,
        'top10': top10,
        'n_queries': n,
        'n_cands': len(cand_df),
        'cross_time': t_cross,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--query', required=True)
    ap.add_argument('--candidates', required=True)
    ap.add_argument('--id-col', required=True)
    ap.add_argument('--name-col', required=True)
    ap.add_argument('--addr-col', required=True)
    ap.add_argument('--city-col', default='')
    ap.add_argument('--gt-id', default='', help='Ground-truth ID列（如候选ID=查询ID则设为此列名）')
    ap.add_argument('--name', required=True, help='实验名称')
    ap.add_argument('--reranker', default='', help='CrossEncoder model name (e.g. BAAI/bge-reranker-v2-m3)')
    ap.add_argument('--topk-rerank', type=int, default=50, help='Number of candidates to rerank')
    ap.add_argument('--output-dir', default='./exp_results')
    args = ap.parse_args()
    args.city_col = args.city_col or ''
    args.gt_id = args.gt_id or ''

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_id = f"{re.sub(r'[^a-zA-Z0-9]', '_', args.name)}_{ts}"

    # 加载数据
    q_df = pd.read_csv(args.query) if Path(args.query).suffix == '.csv' else pd.read_excel(args.query)
    c_df = pd.read_csv(args.candidates) if Path(args.candidates).suffix == '.csv' else pd.read_excel(args.candidates)

    # 实验配置矩阵
    configs = []
    use_cross = bool(args.reranker)
    for topk in [10, 50, 100]:
        for city in [True, False] if args.city_col else [False]:
            configs.append({'topk': topk, 'use_city': city, 'name_col': args.name_col, 'addr_col': args.addr_col, 'use_cross': False})
            if use_cross:
                configs.append({'topk': topk, 'use_city': city, 'name_col': args.name_col, 'addr_col': args.addr_col,
                                'use_cross': True, 'topk_rerank': args.topk_rerank})

    print(f"[实验] {len(configs)} 组配置 × {len(q_df)} 查询 × {len(c_df)} 候选", flush=True)
    if use_cross:
        print(f"  CrossEncoder: {args.reranker}", flush=True)

    results = []
    for i, cfg in enumerate(configs):
        t0 = time.time()
        r = run_exp(q_df, c_df, cfg, args)
        r['elapsed_s'] = round(time.time() - t0, 1)
        results.append(r)
        acc = f"Top1={r['top1']:.1%}" if r['top1'] is not None else "N/A"
        city_str = "城市分治" if cfg['use_city'] else "全局"
        cross_str = "+CE" if cfg.get('use_cross') else ""
        print(f"  [{i+1}/{len(configs)}] TopK={cfg['topk']:>3} {city_str}{cross_str} → {acc} ({r['elapsed_s']}s)", flush=True)

    # 保存
    report = {
        'exp_id': exp_id,
        'name': args.name,
        'timestamp': ts,
        'args': vars(args),
        'results': results,
    }
    with open(out_dir / f'{exp_id}.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Markdown 报告
    best = max(results, key=lambda r: (r.get('top1') or 0))
    md = [f"# 实验报告: {args.name}", f"时间: {ts}", "",
          f"查询: {len(q_df)} 条 | 候选: {len(c_df)} 条 | GT: {'有' if args.gt_id else '无'}",
          f"CrossEncoder: {args.reranker or '未启用'}", "",
          "## 结果汇总", "",
          "| TopK | 分治 | 精排 | Top-1 | Top-3 | Top-10 | 耗时 |",
          "|------|------|------|-------|-------|--------|------|"]
    for r in sorted(results, key=lambda x: -(x.get('top1') or 0)):
        t1 = f"{r['top1']:.1%}" if r['top1'] is not None else "N/A"
        t3 = f"{r['top3']:.1%}" if r['top3'] is not None else "N/A"
        t10 = f"{r['top10']:.1%}" if r['top10'] is not None else "N/A"
        city_str = "是" if r['config']['use_city'] else "否"
        cross_str = "CE" if r['config'].get('use_cross') else "-"
        md.append(f"| {r['config']['topk']:>3} | {city_str} | {cross_str} | {t1} | {t3} | {t10} | {r['elapsed_s']}s |")

    md += ["", f"**最优配置**: TopK={best['config']['topk']}, 城市分治={best['config']['use_city']}, CrossEncoder={best['config'].get('use_cross', False)}"]
    if best.get('top1') is not None:
        md.append(f"**最优 Top-1**: {best['top1']:.1%}")

    md_path = out_dir / f'{exp_id}_report.md'
    md_path.write_text('\n'.join(md), encoding='utf-8')
    print(f"\n[完成] {md_path}", flush=True)

if __name__ == '__main__':
    main()
