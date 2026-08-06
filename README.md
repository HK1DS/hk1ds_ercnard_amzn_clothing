# IKGR -> DynLLM -> CORONA Vanilla Pipeline

This repository is a dataset-agnostic vanilla pipeline for running the full
recommendation experiment stack:

1. **IKGR**: extract LLM intents, expand them with RAG, and build an intent KG.
2. **DynLLM-style dynamic profile**: add a train-only recency profile when
   interaction timestamps are available.
3. **CORONA-style retrieval/reranking**: use graph-based candidate generation or
   soft graph priors for diversity / long-tail experiments.

The old Goodreads experiment snapshot may exist locally under `legacy/`, but it
is intentionally ignored by Git. The root files are the reusable vanilla version.

## Input Files

### `profiles.csv`

Required columns:

| column | description |
|---|---|
| `user_id` | user identifier |
| `user_profile` | text used for user intent extraction |
| `item_id` | item identifier |
| `item_profile` | text used for item intent extraction |

Rows can contain repeated users/items. LLM calls are cached by profile text.

### `interactions.csv`

Required columns:

| column | description |
|---|---|
| `user_id` | user identifier |
| `item_id` | item identifier |
| `rating` | interaction strength used by RecBole |
| `timestamp` | optional Unix timestamp for temporal/DynLLM experiments |

### `item_metadata.csv` Optional

Required column: `item_id`.

Optional columns are configurable in `config.yaml`. Defaults:

| group | accepted columns |
|---|---|
| author/brand | `authors`, `author`, `brand` |
| publisher/manufacturer | `publisher`, `publishers`, `manufacturer` |
| category/tag/attribute | `shelves`, `shelf`, `categories`, `category`, `attributes`, `attribute` |

Cell values can be plain strings, delimiter-separated strings, or list literals
such as `["dress", "summer"]`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Fill `.env` with your provider key. Never commit `.env`.

The default `config.yaml` uses the Luxia OpenAI-compatible bridge:

```yaml
llm:
  base_url: https://bridge.luxiacloud.com/llm/openai/chat/completions/gpt-4o-mini
  api_key: "${LUXIA_API_KEY}"
  provider: luxia
```

## Configure

Edit `config.yaml`:

```yaml
paths:
  input_csv: data/profiles.csv
  inter_file: data/interactions.csv
  meta_file: data/item_metadata.csv
  workdir: run/

recbole:
  dataset: ikgr-custom

pipeline:
  split: RS
  seeds: "2020"
  specs: "IKGR_kgoff,IKGR_kgon_L1_frozen,BPR,LightGCN"
```

Use `split: TO` only when `interactions.csv` includes `timestamp`.
### Prepare Amazon Clothing 2018
After downloading the two official gzip JSONL files under
`data/amazon_clothing/raw/`, build a deterministic, affordable experiment
sample without loading either raw file fully into memory:
```bash
python prepare_amazon_clothing.py --users 5000 --user-k 5 --item-k 3 --min-rating 4
```

This writes `data/profiles.csv`, `data/interactions.csv`, and
`data/item_metadata.csv`. The candidate-user sample is selected by stable hash,
then positive interactions are iteratively filtered to a user/item k-core.

## Amazon Clothing Smoke Experiment (2026-08-04)

The following is a reproducibility snapshot, **not a claim about the full
Amazon Clothing dataset**. It used the deterministic sample created with:

```bash
python prepare_amazon_clothing.py --users 5000 --user-k 5 --item-k 3 --min-rating 4
```

This produced 2,998 positive interactions from 292 users and 275 items. We
used a per-user time-ordered split (`TO`), 30 epochs, and seeds
`2020,2021,2022`. Scores are mean ± standard deviation across seeds.

| model | NDCG@10 | Recall@10 | tail Recall@10 | coverage@10 |
|---|---:|---:|---:|---:|
| BPR | 0.0301 ± 0.0028 | 0.0503 | 0.0422 | 0.9976 |
| LightGCN | 0.0343 ± 0.0054 | 0.0648 | 0.0625 | 0.9988 |
| IKGR kgoff | 0.0348 ± 0.0055 | 0.0659 | 0.0637 | 0.9988 |
| IKGR intent KG | 0.0213 ± 0.0034 | 0.0381 | 0.0193 | 0.4582 |
| IKGR + metadata KG | 0.1121 ± 0.0168 | 0.1682 | 0.1560 | 0.4243 |
| IKGR DynLLM | 0.3117 ± 0.0307 | 0.4401 | 0.4423 | 0.7054 |
| IKGR + CORONA soft rerank (λ=0.5) | **0.3753 ± 0.0251** | **0.5079** | **0.5114** | 0.7345 |

The baseline grid searched embedding sizes 32/64/128 and learning rates
0.0005/0.001. LightGCN selected 128 dimensions and learning rate 0.001; the
final configuration retains 128 dimensions. In this small sample, intent-only
KG propagation reduced accuracy and coverage, while metadata and train-only
recency provided the meaningful gains.

`IKGR_cand_db` is intentionally not listed as a distinct result: its candidate
count was 500 while this sample contains only 275 items, so it did not restrict
the ranking catalog and matched DynLLM. Re-evaluate candidate generation with a
smaller M (for example 50–200) before making a CORONA candidate-retrieval
claim.

## Adaptive Recency-Intent CORONA Experiment (2026-08-05)

This follow-up implements the offline/online design used for the current
recommended pipeline. Before the k-core, each user's events older than 365 days
from that user's latest retained event are removed. The resulting deterministic
smoke dataset contains 2,016 positive interactions, 196 users, and 184 items.

During each evaluation seed, adaptive weights are computed from the **train
split only** to prevent temporal leakage:

\[
W_{u,m,i}=\gamma^{\Delta t_{u,i}}\frac{1}{1+\beta H_m},
\]

where \(\Delta t\) is measured in days from the user's latest training event,
\(\gamma=0.99\), and \(H_m\) is the entropy of metadata categories associated
with intent \(m\). The weighted user-to-intent aggregation is then consumed by
the IKGR/DynLLM model. Category entropy is built offline in `build_kg.py`; the
time component is built after RecBole creates each train split.

For online-style CORONA coarse retrieval, candidate size is catalog
proportional:

\[
M=\min(M_{max},\lfloor\alpha|I|\rfloor),\quad \alpha=0.2,\ M_{max}=500.
\]

On this 184-item catalog, this yields \(M=36\), followed by model ranking in
the retrieved set. The `IKGR_adaptive_corona` retriever had candidate recall
0.9794 at M=36 for seed 2020. Results use the same `TO`, 30-epoch, three-seed
protocol as the preceding table.

| model | NDCG@10 | Recall@10 | tail Recall@10 | coverage@10 |
|---|---:|---:|---:|---:|
| LightGCN | 0.0241 ± 0.0061 | 0.0565 | 0.0580 | 1.0000 |
| IKGR DynLLM | 0.2625 ± 0.0258 | 0.3663 | 0.3586 | 0.7971 |
| IKGR adaptive gating + DynLLM | 0.3417 ± 0.0321 | 0.4778 | 0.4692 | 0.8352 |
| IKGR adaptive gating + CORONA M=36 | **0.4910 ± 0.0287** | **0.6364** | **0.6485** | 0.8044 |

The candidate-restricted model improves ranking and tail recall on this smoke
sample while intentionally trading some catalog coverage for retrieval speed.
It is not an end-to-end serving benchmark: this repository evaluates the
offline ranking path and does not yet provide a <10 ms vector-search service.

The data used here contains 2,998 interactions, 292 users, and 275 items. The
validation grid searched \(\alpha \in \{0.2, 0.3\}\) and
\(\lambda \in \{0.5, 0.75, 0.9\}\), with \(\gamma=0.99\) and
\(\beta=1.0\). Validation selected \(\alpha=0.2\) and
\(\lambda=0.75\), which gives \(M=55\) candidates on this catalog. The
fixed setting was then evaluated across seeds 2020, 2021, and 2022 on test.

| fixed test model | NDCG@10 | tail Recall@10 | coverage@10 |
|---|---:|---:|---:|
| IKGR adaptive gating + train-only 365-day window + CORONA + validation-selected rerank | **0.8756 ± 0.0103** | **0.9523** | **0.8109** |

This is the current headline result for the smoke dataset. It remains a small,
per-user temporal evaluation rather than evidence of production latency or
generalization to the full Amazon Clothing catalog.

## Strict Train-only KG Leakage Audit (2026-08-05)

Strict mode removes test-only item-intent KG edges and recomputes each intent's
category entropy using only items present in the training split. User intent
exposure, recency weights, the 365-day interaction window, and history masking
are also train-only. As a direct diagnosis, the soft-prior weight is fixed at
\(\lambda=0\), so ranking uses the learned GNN score only.

| strict test diagnostic | NDCG@10 | tail Recall@10 | coverage@10 |
|---|---:|---:|---:|
| IKGR adaptive gating + strict train-only KG + \(\lambda=0\) | **0.3842 ± 0.0198** | **0.4973** | **0.8412** |

This is the three-seed test diagnostic (0.3570, 0.3912, 0.4043) on 2,998
interactions / 292 users / 275 items. It confirms that the anomalous 0.87
score came from the prior/transductive path, not pure GNN ranking. The next
valid experiment is a strict-mode validation-only \(\lambda\) sweep, followed
by one locked test evaluation.

## Final Strict Rerun (2026-08-05)

The previous high-score section above is obsolete and should not be used. The
final strict rerun uses train-only KG and CORONA matrices: test-item intent and
metadata rows are excluded from every graph-prior computation.

Validation selected \(\lambda=0.75\) from \(\{0, 0.1, 0.25, 0.5, 0.75\}\).
That setting was locked and evaluated once on test over seeds 2020--2022.

| final strict test model | NDCG@10 | tail Recall@10 | coverage@10 |
|---|---:|---:|---:|
| IKGR adaptive gating + strict train-only KG + CORONA (\(\lambda=0.75\)) | **0.3712 ± 0.0077** | **0.3795** | **0.7236** |

Seed NDCG@10 values: 0.3702, 0.3641, 0.3794. This is the only current
reportable result for this smoke dataset.

## Run

Minimal reusable run:

```bash
python run_pipeline.py --steps BCDE
```

Run with metadata KG:

```bash
python run_pipeline.py --steps D --metadata data/item_metadata.csv
python run_pipeline.py --steps E --specs IKGR_kgoff,IKGR_full_hetero,BPR,LightGCN
```

Run temporal DynLLM/CORONA ablations:

```bash
python run_pipeline.py --steps E --split TO --epochs 12 --seeds 2020,2021,2022 --specs IKGR_dyn,IKGR_cand_db,BPR,LightGCN
```

Optional k-core preprocessing:

```bash
python apply_k_core.py --profiles_in data/profiles.csv --interactions_in data/interactions.csv --k 20 --out_dir data/k_core
```

Then update `config.yaml` paths to the generated files.

## Stages

| stage | script | output |
|---|---|---|
| A | `apply_k_core.py` | filtered profile/interaction CSVs |
| B | `step1.py` | `run/step1_intents.csv` |
| C | `step2.py` | `run/step2_related_intents.csv` |
| D | `build_intent_banks.py`, `build_kg.py`, `build_meta_kg.py`, `convert_to_recbole_atomic.py` | intent banks, KG packs, RecBole `.inter/.kg` |
| E | `eval_slices.py` | overall, long-tail, coverage, novelty, and candidate stats |

## Eval Specs

Common specs supported by `eval_slices.py`:

| spec | meaning |
|---|---|
| `IKGR_kgoff` | MF-style no-KG baseline using the IKGR model shell |
| `IKGR_kgon_L1_frozen` | intent-KG propagation with frozen intent projection |
| `IKGR_full_hetero` | intent KG + optional metadata KG |
| `IKGR_dyn` | IKGR + recency dynamic profile, requires timestamp |
| `IKGR_full` | IKGR + DynLLM + CORONA late-fusion |
| `IKGR_cand` | CORONA graph candidate restriction |
| `IKGR_cand_db` | de-biased CORONA candidate generation |
| `IKGR_rerank_db_rel` | soft graph-prior reranking grid |
| `BPR`, `LightGCN` | RecBole reference baselines |

## Useful Environment Variables

| variable | meaning |
|---|---|
| `IKGR_CONFIG` | config path, default `config.yaml` |
| `IKGR_STEP1_WORKERS` | step1 LLM worker count |
| `IKGR_STEP2_WORKERS` | step2 LLM worker count |
| `IKGR_SPLIT` | `RS`, `TO`, or `TO_GLOBAL` |
| `IKGR_EPOCHS` | override training epochs |
| `IKGR_SEEDS` | comma-separated seeds |
| `IKGR_SPECS` | comma-separated eval specs |
| `IKGR_FORCE_ANNOY` | force Annoy for RAG; Windows usually uses sklearn fallback |

## Repository Policy

Tracked files are source code, prompts, config templates, docs, and `.env.example`.
Ignored files include `.env`, `data/`, `run/`, checkpoints, virtual environments,
and local `legacy/` experiment snapshots.
