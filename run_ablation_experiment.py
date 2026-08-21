"""Resumable end-to-end efficiency and edge-weight ablation supervisor."""
import json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
STATE_DIR = os.path.join(ROOT, "run", "amazon_clothing_ablation")
os.makedirs(STATE_DIR, exist_ok=True)
LOG = os.path.join(STATE_DIR, "supervisor.log")
TIMES = os.path.join(STATE_DIR, "timings.json")
VENV_PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
PY = VENV_PY if os.path.exists(VENV_PY) else sys.executable


def run(label, cmd, env=None):
    completed = json.load(open(TIMES, encoding="utf-8")) if os.path.exists(TIMES) else {}
    if label in completed:
        with open(LOG, "a", encoding="utf-8") as log:
            log.write(f"\n[{time.strftime('%F %T')}] SKIP completed {label} ({completed[label]}s)\n")
        return
    started = time.time()
    with open(LOG, "a", encoding="utf-8") as log:
        log.write(f"\n[{time.strftime('%F %T')}] START {label}: {' '.join(cmd)}\n"); log.flush()
        result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        elapsed = time.time() - started
        log.write(f"[{time.strftime('%F %T')}] END {label} rc={result.returncode} seconds={elapsed:.2f}\n")
    if result.returncode: raise SystemExit(result.returncode)
    data = json.load(open(TIMES, encoding="utf-8")) if os.path.exists(TIMES) else {}
    data[label] = round(elapsed, 2)
    with open(TIMES, "w", encoding="utf-8") as f: json.dump(data, f, indent=2)


env = dict(os.environ)
env.update(IKGR_SPLIT="TO", IKGR_SEEDS="2020,2021,2022", IKGR_EPOCHS="12")

# Re-run the old uniform-edge model and run the weighted variant under the same code/data.
legacy_env = dict(env, IKGR_CONFIG="config.amazon_clothing.to.yaml",
                  IKGR_SPECS="IKGR_full_hetero,IKGR_full_hetero_weighted")
run("legacy_graph", [PY, "-u", "run_pipeline.py", "--config", "config.amazon_clothing.to.yaml",
                     "--steps", "D"], legacy_env)
run("legacy_edge_eval", [PY, "-u", "eval_slices.py"], legacy_env)

# Unified one-call pipeline (cache makes restarts inexpensive).
unified_cfg = "config.amazon_clothing.ablation.unified.yaml"
unified_env = dict(env, IKGR_CONFIG=unified_cfg, IKGR_SPECS="IKGR_full_hetero")
run("unified_llm", [PY, "-u", "step_unified.py"], unified_env)
run("unified_graph", [PY, "-u", "run_pipeline.py", "--config", unified_cfg, "--steps", "D"], unified_env)
run("unified_eval", [PY, "-u", "eval_slices.py"], unified_env)
run("summary", [PY, "-u", "summarize_ablation.py"], unified_env)
