#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a generic IKGR -> DynLLM -> CORONA pipeline.

Stages:
  A  optional k-core filtering
  B  step1 LLM exact intent extraction
  C  step2 RAG + LLM related intent expansion
  D  intent banks + intent KG + optional metadata KG + RecBole files
  E  multi-seed sliced evaluation / ablation

Examples:
  python run_pipeline.py --steps BCDE
  python run_pipeline.py --steps D --metadata data/item_metadata.csv
  python run_pipeline.py --steps E --split TO --specs IKGR_dyn,IKGR_cand_db,BPR,LightGCN
"""
import argparse
import os
import subprocess
import sys
import time

import yaml


PY = sys.executable


def sh(cmd, env=None):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    started = time.time()
    completed = subprocess.run(cmd, env=env)
    elapsed = round(time.time() - started, 1)
    if completed.returncode != 0:
        raise SystemExit(f"[FAIL] ({elapsed}s) exit={completed.returncode}: {' '.join(cmd)}")
    print(f"[ok] ({elapsed}s)", flush=True)


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _csv_exists(path):
    return bool(path) and os.path.exists(path) and os.path.getsize(path) > 0


def require_inputs(steps, paths):
    """Fail early with the exact missing dataset files for selected stages."""
    required = []
    if steps & {"B", "C", "D"}:
        required.append(("profiles CSV", paths.get("input_csv")))
    if steps & {"D", "E"}:
        required.append(("interactions CSV", paths.get("inter_file")))
    if "C" in steps and "B" not in steps:
        required.append(("step1 output", paths.get("step1_output")))
    if "D" in steps and "C" not in steps:
        required.append(("step2 output", paths.get("step2_output")))
    missing = [(label, path) for label, path in required if not _csv_exists(path)]
    if missing:
        details = "\n".join(f"  - {label}: {path or '<not configured>'}" for label, path in missing)
        raise SystemExit(
            "[preflight] Required input files are missing or empty:\n"
            f"{details}\nPrepare the dataset or update paths in the config before running."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=os.environ.get("IKGR_CONFIG", "config.yaml"))
    parser.add_argument("--steps", default="BCDE",
                        help="stage subset to run, e.g. A / BCD / E / BCDE")
    parser.add_argument("--k", type=int, default=0,
                        help="k-core value for stage A. 0 disables filtering.")
    parser.add_argument("--profiles-in", default=None,
                        help="raw profiles CSV for stage A")
    parser.add_argument("--interactions-in", default=None,
                        help="raw interactions CSV for stage A")
    parser.add_argument("--out-dir", default="data/k_core",
                        help="stage A output directory")
    parser.add_argument("--encoder", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--metadata", default=None,
                        help="optional item metadata CSV for meta-KG")
    parser.add_argument("--split", default=None,
                        help="RS, TO, or TO_GLOBAL for eval_slices.py")
    parser.add_argument("--epochs", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--specs", default=None,
                        help="comma-separated eval specs, e.g. IKGR_dyn,IKGR_cand_db,BPR")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    rag = cfg["rag"]
    rb = cfg["recbole"]
    pipeline = cfg.get("pipeline", {})
    meta_cfg = cfg.get("metadata", {})
    steps = set(args.steps.upper())

    require_inputs(steps, paths)

    env = dict(os.environ)
    env["IKGR_CONFIG"] = args.config
    if args.split or pipeline.get("split"):
        env["IKGR_SPLIT"] = args.split or str(pipeline.get("split"))
    if args.epochs or rb.get("epochs"):
        env["IKGR_EPOCHS"] = str(args.epochs or rb.get("epochs"))
    if args.seeds or pipeline.get("seeds"):
        env["IKGR_SEEDS"] = args.seeds or str(pipeline.get("seeds"))
    if args.specs or pipeline.get("specs"):
        env["IKGR_SPECS"] = args.specs or str(pipeline.get("specs"))

    os.makedirs(paths["workdir"], exist_ok=True)
    dataset = args.dataset or rb["dataset"]
    encoder = args.encoder or rag["encoder"]
    metadata_csv = args.metadata or paths.get("meta_file", "")

    if "A" in steps:
        if args.k <= 0:
            raise SystemExit("Stage A requires --k > 0.")
        profiles_in = args.profiles_in or paths["input_csv"]
        interactions_in = args.interactions_in or paths["inter_file"]
        sh([PY, "apply_k_core.py",
            "--profiles_in", profiles_in,
            "--interactions_in", interactions_in,
            "--k", str(args.k),
            "--out_dir", args.out_dir], env=env)

    if "B" in steps:
        print("[cost] Step B calls the configured LLM provider.", flush=True)
        sh([PY, "-u", "step1.py"], env=env)

    if "C" in steps:
        print("[cost] Step C calls the configured LLM provider.", flush=True)
        sh([PY, "-u", "step2.py"], env=env)

    if "D" in steps:
        sh([PY, "build_intent_banks.py",
            "--step2_csv", paths["step2_output"],
            "--encoder", encoder,
            "--user_out", paths["user_bank_pt"],
            "--item_out", paths["item_bank_pt"]], env=env)
        sh([PY, "build_kg.py",
            "--step2_csv", paths["step2_output"],
            "--vocab", rag["vocab_json"],
            "--emb", rag["encoding_npy"],
            "--out", paths["kg_pack"],
            "--metadata", metadata_csv if _csv_exists(metadata_csv) else ""], env=env)

        if _csv_exists(metadata_csv):
            meta_cmd = [
                PY, "build_meta_kg.py",
                "--metadata", metadata_csv,
                "--out", paths.get("meta_kg_pack", "run/meta_kg_pack.pt"),
            ]
            if meta_cfg.get("author_cols"):
                meta_cmd += ["--author-cols", ",".join(meta_cfg["author_cols"])]
            if meta_cfg.get("publisher_cols"):
                meta_cmd += ["--publisher-cols", ",".join(meta_cfg["publisher_cols"])]
            if meta_cfg.get("tag_cols"):
                meta_cmd += ["--tag-cols", ",".join(meta_cfg["tag_cols"])]
            sh(meta_cmd, env=env)
        else:
            print("[skip] no metadata CSV found; meta-KG specs will be unavailable.", flush=True)

        sh([PY, "convert_to_recbole_atomic.py",
            "--interactions", paths["inter_file"],
            "--intents", paths["step2_output"],
            "--out_dir", os.path.dirname(paths["inter_file"]) or ".",
            "--dataset", dataset], env=env)

    if "E" in steps:
        sh([PY, "-u", "eval_slices.py"], env=env)

    print("\n[done] generic IKGR -> DynLLM -> CORONA pipeline complete.", flush=True)


if __name__ == "__main__":
    main()
