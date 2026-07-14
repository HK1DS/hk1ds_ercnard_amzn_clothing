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
