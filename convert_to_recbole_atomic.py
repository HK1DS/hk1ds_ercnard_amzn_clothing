#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert generic CSV inputs into RecBole atomic files.

Inputs:
  interactions.csv          -> <dataset>.inter
  step2_related_intents.csv -> <dataset>.kg

``interactions.csv`` must contain ``user_id,item_id,rating`` and may contain a
``timestamp`` column. Timestamps are preserved for temporal split / DynLLM
recency experiments.
"""
import argparse
import ast
import csv
import os

import pandas as pd


def _safe_eval_list(value):
    if isinstance(value, str) and value.strip():
        try:
            parsed = ast.literal_eval(value)
            return list(parsed) if isinstance(parsed, (list, tuple, set)) else []
        except Exception:
            return []
    return []


def to_recbole_inter(interactions_csv, out_dir, dataset_name):
    dataset_dir = os.path.join(out_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    out_path = os.path.join(dataset_dir, f"{dataset_name}.inter")

    df = pd.read_csv(interactions_csv)
    required = {"user_id", "item_id", "rating"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"interactions CSV missing columns: {sorted(missing)}")

    has_ts = "timestamp" in df.columns
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        header = ["user_id:token", "item_id:token", "rating:float"]
        if has_ts:
            header.append("timestamp:float")
        writer.writerow(header)
        for _, row in df.iterrows():
            record = [str(row["user_id"]), str(row["item_id"]), float(row["rating"])]
            if has_ts:
                record.append(float(row["timestamp"]))
            writer.writerow(record)
    print(f"[OK] .inter file saved: {out_path}")


def to_recbole_kg(intent_csv, out_dir, dataset_name):
    dataset_dir = os.path.join(out_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    out_path = os.path.join(dataset_dir, f"{dataset_name}.kg")
    df = pd.read_csv(intent_csv).fillna("")

    user_cols = ["user_intents_exact", "user_intents_related"]
    item_cols = ["item_intents_exact", "item_intents_related"]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["head_id:token", "relation_id:token", "tail_id:token"])
        for _, row in df.iterrows():
            if "user_id" in row:
                intents = []
                for col in user_cols:
                    intents.extend(_safe_eval_list(row.get(col, "[]")))
                for intent in sorted(set(intents)):
                    writer.writerow([f"u_{row['user_id']}", "user_has_intent", f"intent::{intent}"])

            if "item_id" in row:
                intents = []
                for col in item_cols:
                    intents.extend(_safe_eval_list(row.get(col, "[]")))
                for intent in sorted(set(intents)):
                    writer.writerow([f"i_{row['item_id']}", "item_has_intent", f"intent::{intent}"])
    print(f"[OK] .kg file saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactions", required=True)
    parser.add_argument("--intents", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    to_recbole_inter(args.interactions, args.out_dir, args.dataset)
    to_recbole_kg(args.intents, args.out_dir, args.dataset)


if __name__ == "__main__":
    main()
