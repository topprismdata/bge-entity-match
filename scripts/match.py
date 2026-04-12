#!/opt/homebrew/bin/python3.11
"""
BGE Bi-encoder Entity Matching — 抽象版
用法：
  python match.py --query <query_csv> --candidates <cand_csv> \
      --q-id <列名> --q-name <列名> --q-addr <列名> \
      --c-id <列名> --c-name <列名> --c-addr <列名> \
      [--city <城市列名>] [--topk 50] [--output <结果文件>]

核心逻辑：
  1. 文本构建：query = name + addr，cand = name + addr
  2. BGE 编码（mean pooling + L2 normalize）
  3. 候选库按城市分治（如指定城市列）
  4. Top-K cosine similarity 检索
  5. 输出：query_id, matched_id, similarity
"""
import os, sys, argparse, re, time, json
sys.path.insert(0, os.path.dirname(__file__))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path

# ── 清洗函数 ─────────────────────────────────────────────────
def clean_text(text):
    """通用文本清洗（子类可覆盖）"""
    if not text: return ""
    t = str(text)
    # 去除全角数字转半角
    t = t.replace('０','0').replace('１','1').replace('２','2')
    t = t.replace('３','3').replace('４','4').replace('５','5')
    t = t.replace('６','6').replace('７','7').replace('８','8').replace('９','9')
    # 去除多余空格
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def build_query_text(row, name_col, addr_col):
    """构建查询文本"""
    parts = []
    if name_col and name_col in row:
        t = clean_text(row.get(name_col, ''))
        if t: parts.append(t)
    if addr_col and addr_col in row:
        t = clean_text(row.get(addr_col, ''))
        if t: parts.append(t)
    return ' '.join(parts)

def build_cand_text(row, name_col, addr_col):
    """构建候选文本（可被子类覆盖做领域清洗）"""
    return build_query_text(row, name_col, addr_col)

# ── BGE 模型 ─────────────────────────────────────────────────
_model = None
_tokenizer = None
_device = None

def load_bge(model_name='BAAI/bge-large-zh-v1.5'):
    global _model, _tokenizer, _device
    if _model is None:
        from transformers import AutoTokenizer, AutoModel
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModel.from_pretrained(model_name)
        _device = 'cuda' if torch.cuda.is_available() else \
                  ('mps' if torch.backends.mps.is_available() else 'cpu')
        _model.to(_device); _model.eval()
        print(f"[BGE] model={model_name} device={_device}", file=sys.stderr)
    return _tokenizer, _model, _device

def encode_texts(texts, batch_size=32):
    tok, model, dev = load_bge()
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(batch, max_length=256, padding=True, truncation=True, return_tensors='pt')
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            h = model(**enc).last_hidden_state
        mask = enc['attention_mask'].unsqueeze(-1).float()
        pooled = F.normalize((h * mask).sum(1) / mask.sum(1).clamp(min=1), p=2, dim=-1)
        all_embs.append(pooled.cpu().numpy())
        if dev == 'mps':
            torch.mps.synchronize()
    if dev == 'mps':
        torch.mps.empty_cache()
    return np.vstack(all_embs)

def topk_batch(q_embs, c_embs, topk, batch_size=500):
    n_q = q_embs.shape[0]
    top_idx = np.zeros((n_q, topk), dtype=np.intp)
    for i in range(0, n_q, batch_size):
        j = min(i + batch_size, n_q)
        sims = np.dot(q_embs[i:j], c_embs.T)
        btki = np.argpartition(-sims, topk, axis=1)[:, :topk]
        bsims = np.take_along_axis(sims, btki, axis=1)
        so = np.argsort(-bsims, axis=1)
        top_idx[i:j] = np.take_along_axis(btki, so, axis=1)
    return top_idx

# ── 匹配主流程 ──────────────────────────────────────────────
def match(query_df, cand_df, args):
    t0 = time.time()

    # 1. 构建文本
    q_texts = [build_query_text(row, args.q_name, args.q_addr)
               for _, row in query_df.iterrows()]
    c_texts = [build_cand_text(row, args.c_name, args.c_addr)
               for _, row in cand_df.iterrows()]

    q_valid_mask = [bool(t.strip()) for t in q_texts]
    q_valid_idx = [i for i, m in enumerate(q_valid_mask) if m]
    q_embs = encode_texts([q_texts[i] for i in q_valid_idx])
    c_embs = encode_texts(c_texts)

    # 2. 分治策略
    if args.city and args.city in query_df.columns and args.city in cand_df.columns:
        print(f"[策略] 城市分治模式", file=sys.stderr)
        results = _match_by_city(query_df, q_valid_idx, q_embs, cand_df, c_texts, c_embs, args)
    else:
        results = _match_flat(query_df, q_valid_idx, q_embs, cand_df, c_texts, c_embs, args)

    print(f"[完成] {len(results)} 条结果，耗时 {time.time()-t0:.1f}s", file=sys.stderr)
    return results

def _match_flat(query_df, q_valid_idx, q_embs, cand_df, c_texts, c_embs, args):
    cand_ids = cand_df[args.c_id].values.tolist()
    actual_k = min(args.topk, len(c_embs))
    top_idx = topk_batch(q_embs, c_embs, actual_k)

    results = []
    for qi, qi_global in enumerate(q_valid_idx):
        q_row = query_df.iloc[qi_global]
        for rank in range(actual_k):
            ci = top_idx[qi, rank]
            sim = float(np.dot(q_embs[qi], c_embs[ci]))
            results.append({
                args.q_id: q_row[args.q_id],
                'matched_' + args.c_id: cand_ids[ci],
                'rank': rank + 1,
                'similarity': round(sim, 4),
                'cand_text': c_texts[ci],
                'query_text': q_texts[qi] if 'q_texts' in dir() else '',
            })
    return results

def _match_by_city(query_df, q_valid_idx, q_embs, cand_df, c_texts, c_embs, args):
    # 构建城市→候选索引映射
    city_groups = {}
    for i, row in cand_df.iterrows():
        city = str(row.get(args.city, '')).strip()
        if city not in city_groups:
            city_groups[city] = []
        city_groups[city].append(i)

    q_texts_local = [q_texts[i] if i < len(q_texts) else '' for i in q_valid_idx]
    cand_ids = cand_df[args.c_id].values.tolist()

    results = []
    for qi, qi_global in enumerate(q_valid_idx):
        q_row = query_df.iloc[qi_global]
        q_city = str(q_row.get(args.city, '')).strip()

        # 优先本城候选
        if q_city in city_groups and len(city_groups[q_city]) > 0:
            city_cand_idx = city_groups[q_city]
        else:
            city_cand_idx = list(range(len(cand_df)))

        city_c_embs = c_embs[city_cand_idx]
        actual_k = min(args.topk, len(city_cand_idx))
        local_top = topk_batch(q_embs[qi:qi+1], city_c_embs, actual_k)[0]

        for rank, local_ci in enumerate(local_top):
            ci = city_cand_idx[local_ci]
            sim = float(np.dot(q_embs[qi], c_embs[ci]))
            results.append({
                args.q_id: q_row[args.q_id],
                args.city: q_city,
                'matched_' + args.c_id: cand_ids[ci],
                'rank': rank + 1,
                'similarity': round(sim, 4),
                'cand_text': c_texts[ci],
            })
    return results

# ── CLI ────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--query', required=True, help='查询数据文件（CSV或Excel）')
    ap.add_argument('--candidates', required=True, help='候选数据文件')
    ap.add_argument('--q-id', required=True, help='查询数据中的ID列名')
    ap.add_argument('--q-name', help='查询数据中的名称列名')
    ap.add_argument('--q-addr', help='查询数据中的地址列名')
    ap.add_argument('--c-id', required=True, help='候选数据中的ID列名')
    ap.add_argument('--c-name', help='候选数据中的名称列名')
    ap.add_argument('--c-addr', help='候选数据中的地址列名')
    ap.add_argument('--city', help='城市列名（启用分治策略）')
    ap.add_argument('--topk', type=int, default=50)
    ap.add_argument('--output', default='match_results.csv', help='输出文件路径')
    ap.add_argument('--model', default='BAAI/bge-large-zh-v1.5')
    args = ap.parse_args()

    # 加载数据
    q_path, c_path = Path(args.query), Path(args.candidates)
    q_ext = q_path.suffix.lower()
    c_ext = c_path.suffix.lower()
    query_df = pd.read_csv(q_path) if q_ext == '.csv' else pd.read_excel(q_path)
    cand_df = pd.read_csv(c_path) if c_ext == '.csv' else pd.read_excel(c_path)

    # 确保必需列存在
    for col in [args.q_id, args.c_id]:
        pass  # 已在 arg 层面保证

    results = match(query_df, cand_df, args)
    df_out = pd.DataFrame(results)
    df_out.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"[输出] {args.output}  ({len(df_out)} 行)", file=sys.stderr)

if __name__ == '__main__':
    main()
