#!/usr/bin/env python3
"""Run one cache-resumable pipeline stage until it exits successfully.

Intended for long B/C stages on Windows where a foreground terminal or Codex
turn can end independently of the experiment. Status and logs are append-only.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def stamp():
    return datetime.now(timezone.utc).isoformat()


def atomic_status(path, payload):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", required=True)
    parser.add_argument("--retry-seconds", type=int, default=30)
    parser.add_argument("--max-restarts", type=int, default=100)
    args = parser.parse_args()

    config_slug = Path(args.config).stem.replace("config.", "")
    state_dir = Path("run/watchdog")
    state_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{config_slug}_{args.steps.lower()}"
    log_path = state_dir / f"{stem}.log"
    status_path = state_dir / f"{stem}.status.json"
    command = [sys.executable, "-u", "run_pipeline.py", "--config", args.config,
               "--steps", args.steps]

    for attempt in range(1, args.max_restarts + 1):
        atomic_status(status_path, {"state": "running", "attempt": attempt,
                                    "started_utc": stamp(), "command": command,
                                    "pid": os.getpid(), "log": str(log_path)})
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            log.write(f"\n[{stamp()}] watchdog attempt={attempt}: {' '.join(command)}\n")
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                    cwd=Path.cwd(), env=os.environ.copy())
            log.write(f"[{stamp()}] child exit={result.returncode}\n")
        if result.returncode == 0:
            atomic_status(status_path, {"state": "complete", "attempt": attempt,
                                        "completed_utc": stamp(), "exit_code": 0,
                                        "log": str(log_path)})
            return
        atomic_status(status_path, {"state": "retry_wait", "attempt": attempt,
                                    "updated_utc": stamp(), "exit_code": result.returncode,
                                    "retry_seconds": args.retry_seconds, "log": str(log_path)})
        time.sleep(max(1, args.retry_seconds))

    atomic_status(status_path, {"state": "failed", "attempt": args.max_restarts,
                                "updated_utc": stamp(), "log": str(log_path)})
    raise SystemExit(1)


if __name__ == "__main__":
    main()
