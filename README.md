# BGE Entity Match

Semantic entity matching using BGE bi-encoder + CrossEncoder reranker — configurable, experiment-driven, no hardcoding.

## Overview

Match entities between datasets using a three-stage pipeline: city pre-filter → BGE coarse ranking → CrossEncoder fine reranking. Given a query dataset (e.g., store list) and a candidate dataset (e.g., enterprise registry), find the best matching candidate for each query entry.

```
Query text  ──┐                         ┌── CrossEncoder ──▶ Reranked Top-K
              ├──▶ BGE encode ──▶ Top-K ─┘
Candidate text ─┘
              ↑
       City pre-filter (optional)
```

## Quick Start

```bash
# Simple one-shot matching
python scripts/match.py \
  --query stores.csv \
  --candidates enterprises.xlsx \
  --q-id code --q-name store_name --q-addr store_addr \
  --c-id credit_code --c-name company_name --c-addr reg_addr \
  --city city_name \
  --topk 50 \
  --output results.csv

# Cached matching + CrossEncoder reranking
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

# Experiment mode (find best config)
python scripts/experiment.py \
  --query stores.csv \
  --candidates enterprises.xlsx \
  --id-col code --name-col store_name --addr-col store_addr \
  --city-col city_name \
  --gt-id code \
  --name "My Matching Experiment" \
  --reranker BAAI/bge-reranker-v2-m3 \
  --output-dir ./exp_results

# City-partitioned CrossEncoder comparison experiment
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

## Three Modes

### 1. Simple Match (`scripts/match.py`)
One-shot matching with fixed config.

### 2. Cached Match (`scripts/cached_match.py`)
Encode candidates once, match multiple queries against cached embeddings. Supports optional CrossEncoder reranking via `--reranker`.

Advantages over `match.py`:
- **Embedding caching**: encode candidates once, reuse for all future queries
- **City extraction from address**: auto-extracts city from query address for pre-filtering
- **Credit code verification**: auto-compares matched ID with query's ground-truth ID
- **CrossEncoder reranking**: optional fine reranking of BGE's Top-K candidates

### 3. Experiment Mode (`scripts/experiment.py`)
Systematically explore config space and measure accuracy. Supports `--reranker` for CrossEncoder comparison experiments.

### 4. CrossEncoder Comparison (`scripts/experiment_cross.py`)
City-partitioned A/B comparison: BGE-only vs BGE+CrossEncoder per city. Outputs per-city breakdown and aggregated results.

## Key Features

- Configurable column names (no hardcoding)
- City pre-filter strategy (reduces candidate pool 10-50x)
- Three-stage pipeline: city pre-filter → BGE coarse ranking → CrossEncoder fine reranking
- BGE bi-encoder: `BAAI/bge-large-zh-v1.5` with mean pooling + L2 normalization
- CrossEncoder reranker: `BAAI/bge-reranker-v2-m3` (optional)
- MPS/CUDA auto-detection
- Experiment tracking with JSON + Markdown reports

## Benchmarks

Tested on a real-world entity matching task: 3,681 stores across 10 cities matched against 281K enterprise records.

| Method | Top-1 | Top-3 | Top-10 |
|--------|-------|-------|--------|
| BGE only (no city filter) | 3.9% | — | — |
| City pre-filter + BGE (269 cities) | 68.9% | 81.4% | 89.3% |
| City pre-filter + BGE (Top-10 cities) | 60.1% | 71.1% | 80.4% |
| **City pre-filter + BGE + CrossEncoder** (Top-10 cities) | **73.9%** | **81.9%** | **86.6%** |

**CrossEncoder improvement**: +13.9pp Top-1 on Top-10 cities. Most effective on cities where BGE alone struggles (e.g., Chengdu: +31pp, Shanghai: +21pp, Suzhou: +18pp).

### Per-City Breakdown (Top-10 cities)

| City | Candidates | Stores | BGE Top-1 | +CE Top-1 | Δ |
|------|-----------|--------|-----------|-----------|---|
| Chengdu | 4,402 | 462 | 36.8% | 67.7% | +30.9pp |
| Shanghai | 5,165 | 427 | 45.9% | 67.0% | +21.1pp |
| Suzhou | 3,559 | 268 | 44.0% | 61.6% | +17.5pp |
| Guangzhou | 5,046 | 448 | 76.6% | 80.8% | +4.2pp |
| Hangzhou | 2,630 | 374 | 63.9% | 80.2% | +16.3pp |
| Shenzhen | 9,492 | 374 | 73.0% | 80.2% | +7.2pp |
| Xi'an | 3,442 | 419 | 61.1% | 74.2% | +13.1pp |
| Dongguan | 3,205 | 283 | 69.3% | 82.3% | +13.1pp |
| Beijing | 2,924 | 353 | 71.7% | 75.1% | +3.4pp |
| Wuhan | 2,847 | 273 | 61.2% | 68.1% | +7.0pp |

**Key insight**: CrossEncoder is most beneficial when BGE's Top-1 accuracy is low. Large cities with many similar business names benefit the most from fine reranking.

## Requirements

- Python 3.11+
- `pip install torch transformers pandas numpy openpyxl sentence-transformers`
- `HF_HUB_OFFLINE=1` (prevents HuggingFace network timeout)

## Device & Deployment Notes

**Current design targets MacBook MPS (Apple Silicon).** The BGE device auto-detects `cuda → mps → cpu`, but several MPS-specific optimizations are hardcoded:

| Item | MPS (current) | CUDA (needs manual change) |
|------|--------------|---------------------------|
| BGE encode batch_size | 32 (`match.py`, `cached_match.py`) | 64+ for better GPU utilization |
| CrossEncoder batch_size | 1 (sequential per query) | Can batch multiple queries |
| Memory cleanup | `torch.mps.empty_cache()` after encoding | `torch.cuda.empty_cache()` |
| CrossEncoder speed | ~67 pairs/s (MPS) | ~500+ pairs/s (A100) |
| 10-city CrossEncoder total | ~22 min | ~3-5 min (estimated) |

**To deploy on CUDA server:**
1. Change `batch_size=32` to `batch_size=64` (or higher) in `encode_texts()` calls in `match.py` and `cached_match.py`
2. Replace `torch.mps.empty_cache()` with `torch.cuda.empty_cache()` in `cached_match.py` (line 198)
3. Sync HuggingFace model cache: `rsync -av ~/.cache/huggingface/hub/models--BAAI--bge-large-zh-v1.5/ server:~/.cache/huggingface/hub/models--BAAI--bge-large-zh-v1.5/` and `rsync -av ~/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/ server:~/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/`
4. `sentence_transformers.CrossEncoder` auto-detects CUDA, no change needed

## References

- BGE bi-encoder: [HuggingFace: BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5)
- CrossEncoder reranker: [HuggingFace: BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- City pre-filter strategy: consistently improves Top-1 accuracy by 10-22x over brute force
- CrossEncoder is only effective after city pre-filtering: on national data (281K candidates), CrossEncoder shows near-zero improvement because the correct answer is already in BGE's Top-50
