---
name: bge-entity-match
description: |
  Use BGE bi-encoder semantic matching to match entities between datasets.
  Examples: match stores to enterprise records by name+address, deduplicate records,
  resolve entity references, find corresponding entries across databases.
  Triggers when user mentions: BGE matching, entity resolution, semantic search,
  fuzzy matching between datasets, name+address matching, enterprise deduplication,
  semantic similarity search, matching records by text similarity.
  Also triggers for: running matching experiments, optimizing matching configs,
  comparing matching strategies (TopK, city pre-filter, text cleaning).
allowed-tools:
  - Bash
  - Read
  - Write
  - Glob
---

# BGE Entity Match

Semantic entity matching using BGE bi-encoder — configurable, experiment-driven, no hardcoding.

## Core Idea

Given a **query dataset** (e.g., store list) and a **candidate dataset** (e.g., enterprise registry),
find the best matching candidate for each query using BGE embeddings + cosine similarity.

```
Query text  ──┐
              ├──▶ BGE encode ──▶ cosine similarity ──▶ Top-K candidates
Candidate text ─┘
```

## Two Modes

### 1. Simple Match (`scripts/match.py`)
One-shot matching with fixed config.

```bash
python scripts/match.py \
  --query stores.csv \
  --candidates enterprises.xlsx \
  --q-id code --q-name store_name --q-addr store_addr \
  --c-id credit_code --c-name company_name --c-addr reg_addr \
  --city city_name \
  --topk 50 \
  --output results.csv
```

### 2. Experiment Mode (`scripts/experiment.py`)
Systematically explore config space and measure accuracy.

```bash
python scripts/experiment.py \
  --query stores.csv \
  --candidates enterprises.xlsx \
  --id-col code --name-col store_name --addr-col store_addr \
  --city-col city_name \
  --gt-id code \
  --name "XX品牌匹配实验" \
  --output-dir ./exp_results
```

Outputs: `exp_id.json` (raw) + `exp_id_report.md` (readable summary).

## Column Configuration

**Never hardcode column names.** The skill reads them from user input or file inspection.

| Parameter | Meaning | Example |
|-----------|---------|---------|
| `--q-id` | Unique ID in query dataset | `store_id` |
| `--q-name` | Name field in query | `store_name` |
| `--q-addr` | Address field in query | `store_addr` |
| `--c-id` | Unique ID in candidate dataset | `credit_code` |
| `--c-name` | Name field in candidates | `company_name` |
| `--c-addr` | Address field in candidates | `reg_addr` |
| `--city-col` | City/region column (enables pre-filter) | `city` |
| `--gt-id` | Ground-truth ID for accuracy eval | `true_code` |

If `--gt-id` is provided and equals `--c-id` (i.e., both datasets share the ID column),
the script computes **Top-1/3/10 accuracy** automatically.

## Pre-filtering Strategy

**Always prefer pre-filtering over brute force.**

- **City pre-filter**: Split candidate pool by city. Only search within the query's city.
  → Reduces candidate pool by 10-50x, dramatically improves speed AND accuracy.
- **Custom pre-filter**: If you know another field (district, category, brand), use it.
- Fall back to brute force if no pre-filter field is available.

Rule: pre-filter cardinality should be 500–5,000 candidates per sub-pool for best results.

## Text Building

The matching signal comes from how text is constructed:

```
query_text    = name_clean + " " + addr_clean
candidate_text = name_clean + " " + addr_clean
```

**Text cleaning** (applied to both query and candidate):
1. Strip brand/entity prefixes specific to the domain (e.g., "眼镜店", "眼镜城")
2. Normalize full-width digits: `０１２３` → `0123`
3. Collapse whitespace
4. Strip city prefix if present in address (reduces overfitting to city names)

If domain-specific terms appear frequently in your data, add domain rules to
`clean_text()` in `scripts/match.py` — follow the abstraction pattern, don't hardcode paths.

## Experiment Loop

Run experiments BEFORE tuning. Systematic exploration beats intuition:

```
Config matrix:
  TopK ∈ {10, 50, 100} × {city pre-filter: on, off}
  → 6 experiments per dataset

For each experiment:
  1. Run matching with that config
  2. Compute Top-1/3/10 if GT available
  3. Record to JSON

Read the report markdown → identify the winning config → apply it.
```

Key signal: **city pre-filter almost always helps** (reduces noise from same-name businesses in other cities).

## Optimization Levers

When accuracy is low, improve in this order:

1. **Text cleaning** — add domain stopwords, strip brand suffixes
2. **Pre-filter granularity** — city → district → landmark
3. **TopK** — increase if candidate pool is very large (>5,000)
4. **Candidate pool size** — reduce if pool has many irrelevant entries
5. **Name+addr weighting** — if names are distinctive, addr noise hurts; try name-only query

## Model Selection

Default: `BAAI/bge-large-zh-v1.5` (1024-dim, MPS/CUDA)

For speed-critical scenarios: `BAAI/bge-base-zh-v1.5` (768-dim, ~2x faster)

Override with `--model <model_name>` in both scripts.

## References

This skill encapsulates patterns discovered through systematic experimentation:

- **BGE bi-encoder**: [HuggingFace: BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5)
  Mean pooling + L2 normalization is the standard approach for sentence-level similarity.

- **City pre-filter strategy**: Discovered empirically in QCC matching experiments (2026-04).
  Partitioning by city reduces 10-50x candidate pool and consistently improves Top-1 accuracy
  because businesses with identical names in different cities are common noise sources.

- **Text cleaning for entity matching**: Brand names in franchise data (e.g., "XX眼镜")
  often don't appear in official registration names — stripping them from query text
  before encoding dramatically improves recall. See `QCC_FuzzySearch.md` §BGE-QCC.

- **Candidate pool coverage ceiling**: When candidate data comes from a secondary source
  (e.g., Tianyancha), not all ground-truth entities are present. Matching accuracy
  is bounded by `n_candidates_in_pool / n_total_entities`. Measure this before optimizing.

## Environment Setup

Both scripts require:
- `HF_HUB_OFFLINE=1` (prevents HuggingFace network timeout)
- `TOKENIZERS_PARALLELISM=false` (prevents fork warning on MPS)
- Python 3.11+ with: `pip install torch transformers pandas numpy openpyxl`

On Mac M-series: MPS backend is auto-selected, batch_size=32 recommended.
On GPU server: CUDA backend is auto-selected, batch_size=64+ recommended.

## Output Format

`match.py` outputs:
```
query_id, matched_id, rank, similarity, cand_text
```

`experiment.py` additionally outputs a Markdown report with a comparison table:
```
| TopK | 分治 | Top-1 | Top-3 | Top-10 |
|------|------|-------|-------|--------|
|  50  |  是  | 76.6% | 82.1% |  86.8% |
```

## Key Principle

**Configuration over code.** When you encounter a new dataset:
1. Inspect columns → pick the right `--q-id`, `--q-name`, `--q-addr` etc.
2. Run experiment.py → find the best TopK + pre-filter strategy
3. Run match.py with those settings → get results

Do not write new Python code unless the data format is genuinely unsupported.
