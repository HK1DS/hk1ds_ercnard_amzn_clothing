"""One-call exact + related intent extraction for the efficiency ablation."""
import ast, hashlib, json, os, random, re, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import yaml
from tqdm import tqdm

from ikgr_core.llm_client import LocalLLM

WORKERS = int(os.environ.get("IKGR_UNIFIED_WORKERS", "16"))


def parse_answer(text):
    text = re.sub(r"```(?:json|python)?|```", "", str(text)).strip()
    match = re.search(r"\{.*\}", text, re.S)
    value = None
    for parser in (json.loads, ast.literal_eval):
        try:
            value = parser(match.group(0) if match else text); break
        except Exception:
            pass
    if not isinstance(value, dict):
        raise ValueError("response is not an object")
    clean = lambda xs: [str(x).strip() for x in xs if str(x).strip()] if isinstance(xs, list) else []
    return {"exact": clean(value.get("exact")), "related": clean(value.get("related"))}


def main():
    cfg = yaml.safe_load(open(os.environ.get("IKGR_CONFIG", "config.yaml"), encoding="utf-8"))
    paths = cfg["paths"]; work = paths["workdir"]; os.makedirs(work, exist_ok=True)
    df = pd.read_csv(paths["input_csv"]).fillna("")
    llm = LocalLLM(**cfg["llm"])
    cache_path = os.path.join(work, "unified_cache.json")
    usage_path = os.path.join(work, "llm_usage_unified.jsonl")
    cache = json.load(open(cache_path, encoding="utf-8")) if os.path.exists(cache_path) else {"user": {}, "item": {}}
    lock = threading.Lock()
    prompt = ("Given this profile, return concise domain-specific intents. Exact intents must be directly supported; "
              "related intents may be plausible close concepts. Return ONLY JSON: "
              "{{\"exact\":[\"...\"],\"related\":[\"...\"]}}. PROFILE:\n{profile}")

    def save():
        tmp = cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f: json.dump(cache, f, ensure_ascii=False)
        os.replace(tmp, cache_path)

    def call(kind, profile):
        key = str(profile).strip()
        delay = 5
        for attempt in range(10):
            try:
                answer = llm.chat("You output only valid JSON.", prompt.format(profile=key))
                parsed = parse_answer(answer); usage = llm.usage(); break
            except Exception:
                if attempt == 9: return
                time.sleep(delay + random.random()); delay = min(60, delay * 2)
        record = {"timestamp": time.time(), "phase": "unified", "kind": kind,
                  "profile_sha256": hashlib.sha256(key.encode()).hexdigest(), "usage": usage,
                  "provider": llm.provider, "model": llm.model}
        with lock:
            cache[kind][key] = parsed
            with open(usage_path, "a", encoding="utf-8") as f: f.write(json.dumps(record) + "\n")
            if (len(cache["user"]) + len(cache["item"])) % 50 == 0: save()

    jobs = []
    for kind, col in (("user", "user_profile"), ("item", "item_profile")):
        for profile in df.loc[df[col] != "", col].unique():
            if str(profile).strip() not in cache[kind]: jobs.append((kind, profile))
    print(f"[unified] pending calls={len(jobs)} workers={WORKERS}", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for _ in tqdm(as_completed([pool.submit(call, *job) for job in jobs]), total=len(jobs)): pass
    save()
    for kind, col in (("user", "user_profile"), ("item", "item_profile")):
        vals = df[col].map(lambda x: cache[kind].get(str(x).strip(), {"exact": [], "related": []}))
        df[f"{kind}_intents_exact"] = vals.map(lambda x: repr(x["exact"]))
        df[f"{kind}_intents_related"] = vals.map(lambda x: repr(x["related"]))
    df.to_csv(paths["step2_output"], index=False, encoding="utf-8-sig")
    print(f"[unified] saved {paths['step2_output']}")


if __name__ == "__main__": main()
