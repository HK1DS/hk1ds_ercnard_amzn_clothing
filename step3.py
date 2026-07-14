'''
Train/eval IKGR (intent-KG recommender) in RecBole.

The model (ikgr_core/model_ikgr.py) loads the intent KG from paths.kg_pack
(build it first with `python build_kg.py`) and enriches user/item embeddings via
a vectorized propagation over shared intent nodes, scored by inner product (BPR).

Toggle KG on/off for ablation:
    IKGR_USE_KG=1 python step3.py    # KG on  (default)  -> run/ikgr_kgon_result.json
    IKGR_USE_KG=0 python step3.py    # KG off (~MF/BPR)  -> run/ikgr_kgoff_result.json
'''

import os, json, time, yaml, torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer

from ikgr_core.model_ikgr import IKGR as IKGRModel


def main():
    cfg = yaml.safe_load(open(os.environ.get("IKGR_CONFIG", "config.yaml"), encoding="utf-8"))
    paths, rb = cfg["paths"], cfg["recbole"]

    use_kg = os.environ.get("IKGR_USE_KG", "1").lower() in ("1", "true", "yes")

    config_dict = {
        "epochs": rb["epochs"],
        "metrics": rb["metrics"],
        "topk": rb["topk"],
        "embedding_size": rb["embedding_size"],
        "learning_rate": 1e-3,
        "reg_weight": 1e-6,
        "dropout_prob": rb.get("dropout", 0.1),

        "data_path": os.path.dirname(paths["inter_file"]),  # data/k_core
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "LABEL_FIELD": "rating",
        "load_col": {"inter": ["user_id", "item_id", "rating"]},
        "train_neg_sample_args": {"distribution": "uniform"},
        "save_dataset": False,
        "save_dataloaders": False,
        "checkpoint_dir": paths["recbole_dump"],
        "show_progress": False,
        # Reproducible record
        "seed": 2020,
        "reproducibility": True,
        "eval_step": 5,
        # ---- IKGR KG settings ----
        "use_kg": use_kg,
        "kg_pack_path": os.path.abspath(paths.get("kg_pack", "run/kg_pack.pt")),
    }

    config = Config(model=IKGRModel, dataset=rb["dataset"], config_dict=config_dict)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = IKGRModel(config, train_data.dataset).to(config["device"])
    print(f"[IKGR] use_kg={use_kg} | n_users={model.n_users} n_items={model.n_items}"
          + (f" n_intents={model.n_intents}" if use_kg else ""), flush=True)

    trainer = Trainer(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=False
    )
    try:
        test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=False)
    except FileNotFoundError:
        test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=False)

    print("[valid]", best_valid_result)
    print("[test ]", test_result)

    variant = "intent-KG (propagation on)" if use_kg else "no-KG (MF/BPR baseline)"
    record = {
        "model": "IKGR",
        "variant": variant,
        "use_kg": use_kg,
        "dataset": rb["dataset"],
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
        "seed": config_dict["seed"],
        "config": {
            "epochs": rb["epochs"],
            "embedding_size": rb["embedding_size"],
            "dropout": rb.get("dropout", 0.1),
            "learning_rate": 1e-3,
            "reg_weight": 1e-6,
            "topk": rb["topk"],
            "metrics": rb["metrics"],
        },
        "n_users": int(dataset.user_num),
        "n_items": int(dataset.item_num),
        "best_valid_score": float(best_valid_score),
        "best_valid_result": {k: float(v) for k, v in best_valid_result.items()},
        "test_result": {k: float(v) for k, v in test_result.items()},
        "checkpoint": getattr(trainer, "saved_model_file", None),
    }
    tag = "kgon" if use_kg else "kgoff"
    out_path = os.path.join(paths["workdir"], f"ikgr_{tag}_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(f"[step3] result saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
