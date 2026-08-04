#!/usr/bin/env python3
"""Build pipeline-ready CSVs from Amazon Clothing 2018 gzip JSONL files.

The raw files are read in streaming mode.  A deterministic bottom-hash user
sample keeps the LLM-facing experiment affordable, then an iterative bipartite
k-core removes isolated users/items before metadata and text profiles are built.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path


def rows(path, max_bad_fraction=0.0001):
    bad = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    if bad <= 10:
                        print(f"  warning: skipped malformed JSON at {path}:{number}", flush=True)
                    if bad / number > max_bad_fraction and number >= 100_000:
                        raise RuntimeError(
                            f"Too many malformed records in {path}: {bad}/{number}"
                        )
    if bad:
        print(f"  skipped {bad:,} malformed records in {path}", flush=True)


def stable_hash(value, seed):
    raw = f"{seed}:{value}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")


def select_users(review_path, count, seed):
    heap = []
    seen = set()
    for index, row in enumerate(rows(review_path), 1):
        uid = str(row.get("reviewerID") or "")
        if not uid or uid in seen:
            continue
        seen.add(uid)
        score = stable_hash(uid, seed)
        entry = (-score, uid)
        if len(heap) < count:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)
        if index % 1_000_000 == 0:
            print(f"  pass1: {index:,} reviews, {len(seen):,} users", flush=True)
    chosen = {uid for _, uid in heap}
    print(f"  selected {len(chosen):,} of {len(seen):,} users", flush=True)
    return chosen


def collect_events(review_path, users, min_rating):
    events = []
    snippets = defaultdict(list)
    for index, row in enumerate(rows(review_path), 1):
        uid = str(row.get("reviewerID") or "")
        iid = str(row.get("asin") or "")
        rating = float(row.get("overall") or 0)
        timestamp = row.get("unixReviewTime")
        if uid in users and iid and timestamp is not None and rating >= min_rating:
            events.append((uid, iid, rating, int(timestamp)))
            text = " ".join(str(row.get(key) or "").strip() for key in ("summary", "reviewText")).strip()
            if text:
                snippets[uid].append((int(timestamp), text[:600]))
        if index % 1_000_000 == 0:
            print(f"  pass2: {index:,} reviews, {len(events):,} retained", flush=True)
    return events, snippets


def iterative_core(events, user_k, item_k):
    current = events
    while True:
        uc = Counter(row[0] for row in current)
        ic = Counter(row[1] for row in current)
        filtered = [row for row in current if uc[row[0]] >= user_k and ic[row[1]] >= item_k]
        print(f"  k-core: {len(filtered):,} events, {len(set(r[0] for r in filtered)):,} users, "
              f"{len(set(r[1] for r in filtered)):,} items", flush=True)
        if len(filtered) == len(current):
            return filtered
        current = filtered
        if not current:
            raise RuntimeError("k-core removed every event; lower --user-k/--item-k or increase --users")


def text_values(value):
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(text_values(item))
        return out
    if isinstance(value, dict):
        return text_values(list(value.values()))
    text = str(value).strip()
    return [text] if text else []


def collect_metadata(meta_path, item_ids):
    metadata = {}
    for index, row in enumerate(rows(meta_path), 1):
        iid = str(row.get("asin") or "")
        if iid in item_ids and iid not in metadata:
            categories = text_values(row.get("category"))
            features = text_values(row.get("feature"))
            descriptions = text_values(row.get("description"))
            metadata[iid] = {
                "item_id": iid,
                "brand": str(row.get("brand") or "").strip(),
                "category": " | ".join(categories[:8]),
                "attributes": " | ".join(features[:12]),
                "title": str(row.get("title") or "").strip(),
                "description": " ".join(descriptions)[:1200],
            }
        if index % 500_000 == 0:
            print(f"  metadata: {index:,} rows, {len(metadata):,}/{len(item_ids):,} matched", flush=True)
        if len(metadata) == len(item_ids):
            break
    return metadata


def write_csv(path, fieldnames, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", default="data/amazon_clothing/raw/Clothing_Shoes_and_Jewelry_5.json.gz")
    parser.add_argument("--metadata", default="data/amazon_clothing/raw/meta_Clothing_Shoes_and_Jewelry.json.gz")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--users", type=int, default=5000, help="candidate users before k-core")
    parser.add_argument("--user-k", type=int, default=5)
    parser.add_argument("--item-k", type=int, default=3)
    parser.add_argument("--min-rating", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=2020)
    args = parser.parse_args()

    review_path, meta_path, out = Path(args.reviews), Path(args.metadata), Path(args.out_dir)
    for path in (review_path, meta_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"Missing raw file: {path}")

    chosen = select_users(review_path, args.users, args.seed)
    events, snippets = collect_events(review_path, chosen, args.min_rating)
    events = iterative_core(events, args.user_k, args.item_k)
    valid_users = {row[0] for row in events}
    valid_items = {row[1] for row in events}
    metadata = collect_metadata(meta_path, valid_items)
    missing = valid_items - set(metadata)
    if missing:
        print(f"  warning: {len(missing):,} items lack metadata; keeping them with ID-only text", flush=True)

    events.sort(key=lambda row: (row[0], row[3], row[1]))
    write_csv(out / "interactions.csv", ["user_id", "item_id", "rating", "timestamp"], (
        {"user_id": u, "item_id": i, "rating": r, "timestamp": t} for u, i, r, t in events
    ))
    meta_records = []
    for iid in sorted(valid_items):
        meta_records.append(metadata.get(iid, {"item_id": iid, "brand": "", "category": "",
                                              "attributes": "", "title": "", "description": ""}))
    write_csv(out / "item_metadata.csv",
              ["item_id", "brand", "category", "attributes", "title", "description"], meta_records)

    user_profiles = {}
    for uid in valid_users:
        recent = sorted(snippets.get(uid, []), reverse=True)[:5]
        user_profiles[uid] = " ".join(text for _, text in recent)[:2400] or f"Amazon shopper {uid}"
    item_profiles = {}
    for row in meta_records:
        item_profiles[row["item_id"]] = " | ".join(filter(None, [
            row["title"], row["brand"], row["category"], row["attributes"], row["description"]
        ]))[:2400] or f"Amazon item {row['item_id']}"
    users, items = sorted(valid_users), sorted(valid_items)
    n = max(len(users), len(items))
    write_csv(out / "profiles.csv", ["user_id", "user_profile", "item_id", "item_profile"], (
        {"user_id": users[i % len(users)], "user_profile": user_profiles[users[i % len(users)]],
         "item_id": items[i % len(items)], "item_profile": item_profiles[items[i % len(items)]]}
        for i in range(n)
    ))
    print(f"[done] wrote {len(events):,} interactions, {len(users):,} users, {len(items):,} items to {out}")


if __name__ == "__main__":
    main()
