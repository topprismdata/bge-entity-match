# BGE Entity Match

**A business entity resolution engine for building canonical enterprise
world models from noisy records.**

`WORLD MODEL` · `REAL-DATA VALIDATED` · `OPERATIONAL DATA` ·
`APACHE-2.0`

> **World-model question:** When two enterprise systems describe a store
> or company differently, are they referring to the same real-world
> entity?

Part of **TopPrism Business World Modeling**. This repository turns
noisy names, addresses, and city context into ranked entity matches that
can be used to construct a canonical business-entity layer.

------------------------------------------------------------------------

## Why this exists

Before an enterprise can optimize territories, opportunities, routes, or
customer coverage, it needs to know what the underlying entities
actually are.

The same real-world outlet may appear as:

``` text
System A:  华润万家 XX 店
System B:  深圳市华润万家商业有限公司 XX 分店
System C:  华润万家（XX路）
```

Names differ. Addresses are incomplete. Legal entities and storefront
names do not always match.

If entity resolution is wrong, the downstream world model is wrong.

> **Entity resolution is therefore not a data-cleaning detail. It is
> part of the Business World Model.**

------------------------------------------------------------------------

## What this engine does

``` text
Query entities
name · address · city
        ↓
City / geographic candidate filter
        ↓
BGE bi-encoder retrieval
        ↓
Top-K candidate pool
        ↓
CrossEncoder reranking
        ↓
Ranked entity matches
        ↓
Canonical business entity layer
```

The current implementation supports:

-   configurable schemas rather than fixed column names;
-   city-based candidate reduction;
-   BGE semantic retrieval;
-   optional CrossEncoder reranking;
-   cached candidate embeddings;
-   experiment mode with accuracy reporting;
-   CPU / CUDA / Apple Silicon execution paths.

------------------------------------------------------------------------

## Evidence

The repository reports a real-world entity-matching evaluation with:

-   **3,681 query stores**
-   **281K enterprise candidate records**
-   **10-city CrossEncoder comparison**

  ------------------------------------------------------------------------
  Method                       Top-1              Top-3             Top-10
  --------------- ------------------ ------------------ ------------------
  BGE only, no                  3.9%                ---                ---
  city filter                                           

  City filter +                68.9%              81.4%              89.3%
  BGE, 269 cities                                       

  City filter +                60.1%              71.1%              80.4%
  BGE,                                                  
  top-10-city                                           
  subset                                                

  City filter +            **73.9%**          **81.9%**          **86.6%**
  BGE +                                                 
  CrossEncoder,                                         
  top-10-city                                           
  subset                                                
  ------------------------------------------------------------------------

On the top-10-city comparison, CrossEncoder reranking improved Top-1
accuracy by **13.9 percentage points** over the corresponding BGE-only
setup.

### Why this result matters

The largest lesson is not simply "CrossEncoder is better."

It is:

> **Semantic similarity works much better when the candidate universe is
> constrained by business context.**

City context reduces the search space; BGE provides scalable semantic
recall; the CrossEncoder improves ranking among difficult candidates.

### What the evidence does not support

-   73.9% is not universal entity-resolution accuracy;
-   the top-10-city subset and the 269-city experiment are different
    evaluation scopes and should not be compared as if they were
    identical;
-   this public benchmark does not prove production precision for every
    industry, naming convention, or address quality level.

------------------------------------------------------------------------

## Architecture

``` text
┌──────────────────────────────────────────────────────┐
│ RAW ENTERPRISE RECORDS                               │
│ names · addresses · city · identifiers               │
└──────────────────────────┬───────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────┐
│ 1. CONTEXT FILTER                                    │
│ city / geographic candidate reduction                │
└──────────────────────────┬───────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────┐
│ 2. SEMANTIC RETRIEVAL                                │
│ BGE bi-encoder → Top-K candidates                    │
└──────────────────────────┬───────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────┐
│ 3. FINE RERANKING                                    │
│ CrossEncoder → ranked candidates                     │
└──────────────────────────┬───────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────┐
│ CANONICAL ENTITY CANDIDATE                           │
│ matched ID + score + evaluation trace                │
└──────────────────────────────────────────────────────┘
```

------------------------------------------------------------------------

## Where it fits at TopPrism

``` text
CRM / SFA / POI / registry / partner data
                  ↓
          BGE Entity Match
                  ↓
       Canonical entity identity
                  ↓
          Business World Model
                  ↓
market · territory · opportunity · visit · network
```

The repository is not just a "Claude Code matching skill." It is a
reusable **entity-resolution capability** that can also be exposed to
agents as a tool or skill.

------------------------------------------------------------------------

## Quick start

### One-shot match

``` bash
python scripts/match.py \
  --query stores.csv \
  --candidates enterprises.xlsx \
  --q-id code --q-name store_name --q-addr store_addr \
  --c-id credit_code --c-name company_name --c-addr reg_addr \
  --city city_name \
  --topk 50 \
  --output results.csv
```

### Cached matching + reranking

``` bash
python scripts/cached_match.py encode \
  --candidates enterprises.xlsx \
  --c-id credit_code --c-name company_name --c-addr reg_addr \
  --cache-dir ./emb_cache

python scripts/cached_match.py match \
  --query stores.csv \
  --q-id store_id --q-name store_name --q-addr store_addr \
  --candidates enterprises.xlsx \
  --c-id credit_code --c-name company_name --c-addr reg_addr \
  --c-city city \
  --cache-dir ./emb_cache \
  --topk 50 \
  --reranker BAAI/bge-reranker-v2-m3 \
  --output results.csv
```

------------------------------------------------------------------------

## Operating modes

  -----------------------------------------------------------------------
  Mode                                Purpose
  ----------------------------------- -----------------------------------
  Simple Match                        one-shot matching

  Cached Match                        repeated matching against reusable
                                      candidate embeddings

  Experiment Mode                     systematic configuration evaluation

  CrossEncoder Comparison             city-partitioned BGE vs BGE +
                                      reranker comparison
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## Model strategy

The current public implementation uses BGE-family models, but the
architectural pattern is more general:

``` text
business context filter
        +
high-recall semantic retrieval
        +
high-precision reranking
```

README language should therefore emphasize the **entity-resolution
architecture**, while exact model names and hardware tuning live in
implementation documentation.

------------------------------------------------------------------------

## Hardware & performance notes

Move device-specific detail to `docs/deployment.md`.

The README should only state that the implementation supports:

-   CUDA;
-   Apple Silicon / MPS;
-   CPU fallback.

Avoid putting estimated A100 throughput in the public evidence section
unless it has been directly benchmarked on that hardware.

------------------------------------------------------------------------

## Data, privacy & reproducibility

Recommended additions:

### `DATA_PROVENANCE.md`

Document:

-   evaluation dataset provenance;
-   whether records are customer, partner, public, or synthetic;
-   anonymization / redistribution constraints;
-   which raw files are intentionally excluded.

### `docs/evaluation.md`

Document:

-   ground-truth construction;
-   city-filter rules;
-   evaluation population;
-   failure taxonomy;
-   precision / recall or abstention strategy if added later.

For enterprise entity resolution, **false positives can be more damaging
than unmatched records**. A production path should therefore support
confidence thresholds, abstention, and human review.

------------------------------------------------------------------------

## Boundaries & limitations

Current public results expose an important boundary:

-   difficult cities remain materially below perfect Top-1 accuracy;
-   entity matching quality depends on candidate-pool quality and
    context;
-   legal-entity identity and storefront identity may be different
    business concepts;
-   semantic similarity alone is insufficient for many ambiguous cases;
-   production systems should support confidence calibration and human
    review.

These are world-modeling constraints, not merely model limitations.

------------------------------------------------------------------------

## Recommended next experiments

1.  confidence calibration and abstention;
2.  address-specific structured features;
3.  legal entity vs physical outlet ontology;
4.  hard-negative mining;
5.  multilingual / alias handling;
6.  human-in-the-loop review;
7.  cross-industry validation.

------------------------------------------------------------------------

## TopPrism metadata

``` yaml
topprism:
  purpose: world-model
  capability: entity-resolution
  platform_layer: business-world-model
  maturity: real-data-validated
  evidence:
    type: operational-data
    scope: "3,681 query stores; 281K enterprise candidates"
  product_context:
    - data-standardization
    - customer-master
    - outlet-resolution
    - business-world-model
```

## License

Apache-2.0.
