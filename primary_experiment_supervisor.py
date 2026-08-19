#!/usr/bin/env python3
"""Persistently supervise the Amazon Clothing primary experiment sequence."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


STATE = Path("run/watchdog/primary_supervisor.status.json")
LOG = Path("run/watchdog/primary_supervisor.log")


def now():
    return datetime.now(timezone.utc).isoformat()


def status(state, stage, **extra):
    payload = {"state": state, "stage": stage, "updated_utc": now(),
               "pid": os.getpid(), "log": str(LOG), **extra}
    temp = STATE.with_suffix(STATE.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, STATE)


def run_until_ok(stage, args):
    command = [sys.executable, "-u", "run_pipeline.py", *args]
    attempt = 0
    while True:
        attempt += 1
        status("running", stage, attempt=attempt, command=command)
        with LOG.open("a", encoding="utf-8", buffering=1) as log:
            log.write(f"\n[{now()}] {stage} attempt={attempt}: {' '.join(command)}\n")
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                    cwd=Path.cwd(), env=os.environ.copy())
            log.write(f"[{now()}] {stage} exit={result.returncode}\n")
        if result.returncode == 0:
            status("stage_complete", stage, attempt=attempt)
            return
        status("retry_wait", stage, attempt=attempt, exit_code=result.returncode,
               retry_seconds=30)
        time.sleep(30)


def merge_static_items(source, target, profiles):
    with open(profiles, encoding="utf-8-sig", newline="") as handle:
        allowed = {row["item_profile"].strip() for row in csv.DictReader(handle)
                   if row["item_profile"].strip()}
    src = json.loads(Path(source).read_text(encoding="utf-8"))
    dst_path = Path(target)
    dst = json.loads(dst_path.read_text(encoding="utf-8")) if dst_path.exists() else {"user": {}, "item": {}}
    before = len(dst.get("item", {}))
    dst.setdefault("user", {})
    dst.setdefault("item", {}).update({key: value for key, value in src.get("item", {}).items()
                                       if key in allowed})
    temp = dst_path.with_suffix(dst_path.suffix + ".tmp")
    temp.parent.mkdir(parents=True, exist_ok=True)
    temp.write_text(json.dumps(dst, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, dst_path)
    with LOG.open("a", encoding="utf-8") as log:
        log.write(f"[{now()}] merged static items {source} -> {target}: {before} -> {len(dst['item'])}\n")


def wait_for(path, stage):
    path = Path(path)
    while not path.exists() or path.stat().st_size == 0:
        status("waiting", stage, waiting_for=str(path), retry_seconds=30)
        time.sleep(30)


def main():
    STATE.parent.mkdir(parents=True, exist_ok=True)
    wait_for("run/amazon_clothing_to/step2_related_intents.csv", "wait_to_step2")
    merge_static_items("run/amazon_clothing_to/step1_cache.json",
                       "run/amazon_clothing_global/step1_cache.json",
                       "data/amazon_clothing/reporting/global/profiles.csv")
    merge_static_items("run/amazon_clothing_to/step2_cache.json",
                       "run/amazon_clothing_global/step2_cache.json",
                       "data/amazon_clothing/reporting/global/profiles.csv")

    stages = [
        ("global_B", ["--config", "config.amazon_clothing.global.yaml", "--steps", "B"]),
        ("global_C", ["--config", "config.amazon_clothing.global.yaml", "--steps", "C"]),
        ("to_D", ["--config", "config.amazon_clothing.to.yaml", "--steps", "D"]),
        ("global_D", ["--config", "config.amazon_clothing.global.yaml", "--steps", "D"]),
        ("to_P1_smoke", ["--config", "config.amazon_clothing.to.yaml", "--steps", "E",
                         "--epochs", "1", "--seeds", "2020",
                         "--specs", "IKGR_kgoff,BPR,LightGCN"]),
        ("global_P1_smoke", ["--config", "config.amazon_clothing.global.yaml", "--steps", "E",
                             "--epochs", "1", "--seeds", "2020",
                             "--specs", "IKGR_kgoff,BPR,LightGCN"]),
        ("to_primary", ["--config", "config.amazon_clothing.to.yaml", "--steps", "E",
                        "--epochs", "12", "--seeds", "2020,2021,2022"]),
        ("global_primary", ["--config", "config.amazon_clothing.global.yaml", "--steps", "E",
                            "--epochs", "12", "--seeds", "2020,2021,2022"]),
    ]
    for stage, args in stages:
        run_until_ok(stage, args)
    status("complete", "all_primary")


if __name__ == "__main__":
    main()
