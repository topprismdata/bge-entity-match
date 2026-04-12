# BGE Entity Match

Semantic entity matching using BGE bi-encoder — configurable, experiment-driven, no hardcoding.

## Overview

Match entities between datasets using BGE embeddings + cosine similarity. Given a query dataset and a candidate dataset, find the best matching candidate for each query entry.

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

# Experiment mode (find best config)
python scripts/experiment.py \
  --query stores.csv \
  --candidates enterprises.xlsx \
  --id-col code --name-col store_name --addr-col store_addr \
  --city-col city_name \
  --gt-id code \
  --name "My Matching Experiment" \
  --output-dir ./exp_results
```

## Two Modes

- **Simple Match** (`scripts/match.py`): One-shot matching with fixed config
- **Experiment Mode** (`scripts/experiment.py`): Systematically explore config space (TopK, city pre-filter) and measure Top-1/3/10 accuracy

## Key Features

- Configurable column names (no hardcoding)
- City pre-filter strategy (reduces candidate pool 10-50x, dramatically improves speed AND accuracy)
- BGE bi-encoder: `BAAI/bge-large-zh-v1.5` with mean pooling + L2 normalization
- MPS/CUDA auto-detection
- Experiment tracking with JSON + Markdown reports

## Requirements

- Python 3.11+
- `pip install torch transformers pandas numpy openpyxl`
- `HF_HUB_OFFLINE=1` (prevents HuggingFace network timeout)

## References

- BGE bi-encoder: [HuggingFace: BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5)
- City pre-filter strategy: consistently improves Top-1 accuracy by 10-22x over brute force
