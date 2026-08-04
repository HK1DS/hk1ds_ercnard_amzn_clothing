#!/usr/bin/env python3
"""Small deterministic BPR/LightGCN grid using eval_slices.py."""
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    trials = []
    result_path = Path("run/slice_eval_TO_result.json")
    for model in ("BPR", "LightGCN"):
        for dim in (32, 64, 128):
            for lr in (5e-4, 1e-3):
                env = dict(os.environ)
                env.update({
                    "IKGR_SPLIT": "TO", "IKGR_EPOCHS": "30", "IKGR_SEEDS": "2020",
                    "IKGR_SPECS": model, "IKGR_EMBEDDING_SIZE": str(dim),
                    "IKGR_LEARNING_RATE": str(lr),
                })
                print(f"\n=== {model} dim={dim} lr={lr:g} ===", flush=True)
                subprocess.run([sys.executable, "-u", "eval_slices.py"], env=env, check=True)
                result = json.loads(result_path.read_text(encoding="utf-8"))[model]
                seed = result["seeds"]["2020"]
                trials.append({
                    "model": model, "embedding_size": dim, "learning_rate": lr,
                    "ndcg@10": seed["overall"]["ndcg@10"],
                    "recall@10": seed["overall"]["recall@10"],
                    "tail_recall@10": seed["long_tail"]["recall@10"],
                    "coverage@10": seed["coverage@10"], "train_sec": seed["train_sec"],
                })
    trials.sort(key=lambda row: row["ndcg@10"], reverse=True)
    out = Path("run/baseline_tuning.json")
    out.write_text(json.dumps({"objective": "ndcg@10", "trials": trials}, indent=2), encoding="utf-8")
    print(f"\nBEST: {trials[0]}\nSaved {out}")


if __name__ == "__main__":
    main()
