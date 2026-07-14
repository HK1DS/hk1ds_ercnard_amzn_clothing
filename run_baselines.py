#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run RecBole built-in reference baselines on the same dataset/eval protocol as
IKGR, so model numbers are comparable.

Only the MODEL differs; split, seed, metrics, topk, epochs, eval_step, embedding
size, and negative sampling all match step3.py for a clean comparison.

General recommenders (use .inter only): Pop, BPR, LightGCN
Results -> run/baselines_result.json
"""
import os, json, time, yaml
# --- compat shim: RecBole 1.2.0 calls scipy dok_matrix._update(), removed in
#     newer scipy (>=1.8). dok_matrix still subclasses dict, so dict.update works.
import scipy.sparse as _sp
if not hasattr(_sp.dok_matrix, "_update"):
    _sp.dok_matrix._update = dict.update
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model, get_trainer

def run_one(model_name, dataset_name, base_config):
    config = Config(model=model_name, dataset=dataset_name, config_dict=dict(base_config))
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = get_model(config["model"])(config, train_data.dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=False
    )
    # Traditional models (e.g. Pop) may not persist a best checkpoint; fall back
    # to evaluating the in-memory model instead of reloading from disk.
    try:
        test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=False)
    except FileNotFoundError:
        test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=False)
    return best_valid_result, test_result

def main():
    cfg = yaml.safe_load(open(os.environ.get("IKGR_CONFIG", "config.yaml"), encoding="utf-8"))
    paths, rb = cfg["paths"], cfg["recbole"]

    workdir = paths.get("workdir", "run/")
    ckpt_dir = os.path.abspath(os.path.join(workdir, "recbole_baselines"))
    os.makedirs(ckpt_dir, exist_ok=True)

    base_config = {
        "data_path": os.path.dirname(paths["inter_file"]) or ".",
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "LABEL_FIELD": "rating",
        "load_col": {"inter": ["user_id", "item_id", "rating"]},
        "epochs": rb["epochs"],
        "metrics": rb["metrics"],
        "topk": rb["topk"],                     # [10, 30]
        "embedding_size": rb["embedding_size"], # 512 (same capacity as IKGR run)
        "learning_rate": 1e-3,
        "train_neg_sample_args": {"distribution": "uniform"},
        "eval_step": 5,
        "seed": 2020,
        "reproducibility": True,
        # --- isolate from step3 artifacts ---
        "checkpoint_dir": ckpt_dir,
        "save_dataset": False,
        "save_dataloaders": False,
        "show_progress": False,
    }

    models = ["Pop", "BPR", "LightGCN"]
    out_path = os.path.join(workdir, "baselines_result.json")
    results = {}
    if os.path.exists(out_path):
        try:
            results = json.load(open(out_path, "r", encoding="utf-8"))
        except Exception:
            results = {}

    for m in models:
        if m in results:
            print(f"[skip] {m} already done", flush=True)
            continue
        print(f"\n===== running {m} =====", flush=True)
        t = time.time()
        bv, tr = run_one(m, rb["dataset"], base_config)
        dt = round(time.time() - t, 1)
        results[m] = {
            "time_sec": dt,
            "embedding_size": base_config["embedding_size"],
            "best_valid_result": {k: float(v) for k, v in bv.items()} if bv else {},
            "test_result": {k: float(v) for k, v in tr.items()} if tr else {},
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[{m}] done in {dt}s  test={results[m]['test_result']}", flush=True)

    print("\nALL DONE")

if __name__ == "__main__":
    main()
