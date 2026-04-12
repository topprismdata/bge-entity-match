# BGE 实体匹配

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![sentence-transformers](https://img.shields.io/badge/sentence--transformers-2.2%2B-FF6F00.svg)](https://www.sbert.net/)
[![HuggingFace BAAI/bge-large-zh-v1.5](https://img.shields.io/badge/%F0%9F%A4%97-bge--large--zh--v1.5-yellow.svg)](https://huggingface.co/BAAI/bge-large-zh-v1.5)
[![HuggingFace BAAI/bge-reranker-v2-m3](https://img.shields.io/badge/%F0%9F%A4%97-bge--reranker--v2--m3-yellow.svg)](https://huggingface.co/BAAI/bge-reranker-v2-m3)

基于 BGE bi-encoder + CrossEncoder 精排的语义实体匹配 — 可配置、实验驱动、无硬编码。

## 概述

使用三阶段流水线匹配数据集间的实体：城市分治 → BGE粗排 → CrossEncoder精排。给定查询数据集（如门店列表）和候选数据集（如企业注册库），为每条查询找到最佳匹配候选。

```
查询文本  ──┐                         ┌── CrossEncoder ──▶ 重排 Top-K
            ├──▶ BGE 编码 ──▶ Top-K ──┘
候选文本 ───┘
            ↑
       城市分治（可选）
```

## 快速开始

```bash
# 简单一次性匹配
python scripts/match.py \
  --query stores.csv \
  --candidates enterprises.xlsx \
  --q-id code --q-name store_name --q-addr store_addr \
  --c-id credit_code --c-name company_name --c-addr reg_addr \
  --city city_name \
  --topk 50 \
  --output results.csv

# 缓存匹配 + CrossEncoder 精排
python scripts/cached_match.py encode \
  --candidates enterprises.xlsx \
  --c-id credit_code --c-name company_name --c-addr reg_addr \
  --cache-dir ./emb_cache

python scripts/cached_match.py match \
  --query stores.csv \
  --q-id store_id --q-name store_name --q-addr store_addr \
  --q-gt credit_code \
  --candidates enterprises.xlsx \
  --c-id credit_code --c-name company_name --c-addr reg_addr \
  --c-city city \
  --cache-dir ./emb_cache \
  --topk 50 \
  --reranker BAAI/bge-reranker-v2-m3 \
  --output results.csv

# 实验模式（寻找最优配置）
python scripts/experiment.py \
  --query stores.csv \
  --candidates enterprises.xlsx \
  --id-col code --name-col store_name --addr-col store_addr \
  --city-col city_name \
  --gt-id code \
  --name "匹配实验" \
  --reranker BAAI/bge-reranker-v2-m3 \
  --output-dir ./exp_results

# 城市分治 CrossEncoder 对比实验
python scripts/experiment_cross.py \
  --query stores.csv \
  --candidates enterprises.xlsx \
  --q-id code --q-name store_name --q-addr store_addr \
  --q-gt credit_code \
  --c-id credit_code --c-name company_name --c-addr reg_addr \
  --c-city city \
  --topk 50 \
  --max-cities 10 \
  --output-dir ./exp_results
```

## 三种模式

### 1. 简单匹配 (`scripts/match.py`)
一次性匹配，固定配置。

### 2. 缓存匹配 (`scripts/cached_match.py`)
编码候选一次，多次查询复用缓存嵌入。通过 `--reranker` 支持可选的 CrossEncoder 精排。

相比 `match.py` 的优势：
- **嵌入缓存**：编码候选一次，后续查询复用
- **地址提取城市**：自动从查询地址提取城市进行预过滤
- **信用代码验证**：匹配后自动核对比对结果
- **CrossEncoder 精排**：可选对 BGE 粗排 Top-K 候选重排序

### 3. 实验模式 (`scripts/experiment.py`)
系统探索配置空间并衡量准确率。支持 `--reranker` 进行 CrossEncoder 对比实验。

### 4. CrossEncoder 对比 (`scripts/experiment_cross.py`)
城市分治 A/B 对比：每城市 BGE-only vs BGE+CrossEncoder。输出每城市明细和汇总结果。

## 核心特性

- 可配置列名（无硬编码）
- 城市分治策略（候选池缩小 10-50 倍）
- 三阶段流水线：城市分治 → BGE 粗排 → CrossEncoder 精排
- BGE bi-encoder：`BAAI/bge-large-zh-v1.5`，mean pooling + L2 归一化
- CrossEncoder 精排：`BAAI/bge-reranker-v2-m3`（可选）
- MPS/CUDA 自动检测
- 实验追踪：JSON + Markdown 报告

## 性能基准

在真实实体匹配任务上测试：3,681 家门店（10 城市）匹配 28 万企业记录。

| 方法 | Top-1 | Top-3 | Top-10 |
|------|-------|-------|--------|
| BGE only（无城市过滤） | 3.9% | — | — |
| 城市分治 + BGE（269城市） | 68.9% | 81.4% | 89.3% |
| 城市分治 + BGE（Top-10城市） | 60.1% | 71.1% | 80.4% |
| **城市分治 + BGE + CrossEncoder**（Top-10城市） | **73.9%** | **81.9%** | **86.6%** |

**CrossEncoder 提升**：Top-10 城市 Top-1 提升 +13.9pp。在 BGE 单独表现差的城市效果最显著（如成都 +31pp，上海 +21pp，苏州 +18pp）。

### 每城市明细（Top-10 城市）

| 城市 | 候选数 | 门店数 | BGE Top-1 | +CE Top-1 | Δ |
|------|--------|--------|-----------|-----------|---|
| 成都 | 4,402 | 462 | 36.8% | 67.7% | +30.9pp |
| 上海 | 5,165 | 427 | 45.9% | 67.0% | +21.1pp |
| 苏州 | 3,559 | 268 | 44.0% | 61.6% | +17.5pp |
| 广州 | 5,046 | 448 | 76.6% | 80.8% | +4.2pp |
| 杭州 | 2,630 | 374 | 63.9% | 80.2% | +16.3pp |
| 深圳 | 9,492 | 374 | 73.0% | 80.2% | +7.2pp |
| 西安 | 3,442 | 419 | 61.1% | 74.2% | +13.1pp |
| 东莞 | 3,205 | 283 | 69.3% | 82.3% | +13.1pp |
| 北京 | 2,924 | 353 | 71.7% | 75.1% | +3.4pp |
| 武汉 | 2,847 | 273 | 61.2% | 68.1% | +7.0pp |

**关键发现**：CrossEncoder 在 BGE Top-1 准确率低的城市效果最好。大城市的同类型企业名称混淆度高，精排收益最大。

## 环境要求

- Python 3.11+
- `pip install torch transformers pandas numpy openpyxl sentence-transformers`
- `HF_HUB_OFFLINE=1`（防止 HuggingFace 网络超时）

## 设备与部署说明

**当前设计针对 MacBook MPS（Apple Silicon）。** BGE 设备自动检测 `cuda → mps → cpu`，但部分 MPS 专用优化是硬编码的：

| 项目 | MPS（当前） | CUDA（需手动修改） |
|------|-----------|------------------|
| BGE 编码 batch_size | 32（`match.py`、`cached_match.py`） | 64+，以更好利用 GPU |
| CrossEncoder batch_size | 1（逐条顺序处理） | 可批量处理多条查询 |
| 内存清理 | 编码后调用 `torch.mps.empty_cache()` | 改为 `torch.cuda.empty_cache()` |
| CrossEncoder 速度 | ~67 pairs/s（MPS） | ~500+ pairs/s（A100） |
| 10城市 CrossEncoder 总耗时 | ~22 分钟 | ~3-5 分钟（预估） |

**部署到 CUDA 服务器：**
1. 将 `encode_texts()` 调用中的 `batch_size=32` 改为 `batch_size=64`（或更高），涉及 `match.py` 和 `cached_match.py`
2. 将 `cached_match.py` 中的 `torch.mps.empty_cache()` 替换为 `torch.cuda.empty_cache()`
3. 同步 HuggingFace 模型缓存：`rsync -av ~/.cache/huggingface/hub/models--BAAI--bge-large-zh-v1.5/ server:~/.cache/huggingface/hub/models--BAAI--bge-large-zh-v1.5/` 和 `rsync -av ~/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/ server:~/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/`
4. `sentence_transformers.CrossEncoder` 自动检测 CUDA，无需修改

## 参考文献

- BGE bi-encoder：[HuggingFace: BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5)
- CrossEncoder 精排：[HuggingFace: BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- 城市分治策略：相比暴力搜索，Top-1 准确率提升 10-22 倍
- CrossEncoder 仅在城市分治后有效：全国数据（28 万候选）上 CrossEncoder 几乎无提升，因为正确答案已在 BGE Top-50 中
