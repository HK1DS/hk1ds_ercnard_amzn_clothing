#!/usr/bin/env python3
"""Create the immutable Amazon Clothing reporting data manifest."""
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

ROOT = Path("data/amazon_clothing")
BASE = ROOT / "reporting/base"
FILES = {
    "raw_reviews": ROOT / "raw/Clothing_Shoes_and_Jewelry_5.json.gz",
    "raw_metadata": ROOT / "raw/meta_Clothing_Shoes_and_Jewelry.json.gz",
    "processed_interactions": BASE / "interactions.csv",
    "processed_profiles_placeholder": BASE / "profiles.csv",
    "processed_metadata": BASE / "item_metadata.csv",
}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    users, items, ratings, timestamps = Counter(), set(), Counter(), []
    with FILES["processed_interactions"].open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            users[row["user_id"]] += 1
            items.add(row["item_id"])
            ratings[row["rating"]] += 1
            timestamps.append(int(float(row["timestamp"])))
    metadata_items = set()
    with FILES["processed_metadata"].open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            metadata_items.add(row["item_id"])
    counts = sorted(users.values())
    quantiles = statistics.quantiles(counts, n=4, method="inclusive")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "Amazon Reviews 2018 / Clothing, Shoes and Jewelry 5-core",
        "source_page": "https://jmcauley.ucsd.edu/data/amazon_v2/index.html",
        "source_files": {
            "reviews": "Clothing_Shoes_and_Jewelry_5.json.gz",
            "metadata": "meta_Clothing_Shoes_and_Jewelry.json.gz",
            "local_mtime_utc": {key: datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
                                for key, path in FILES.items() if key.startswith("raw_")},
            "note": "local mtime is recorded because the original download timestamp was not preserved separately",
        },
        "preprocessing": {
            "command": "python prepare_amazon_clothing.py --out-dir data/amazon_clothing/reporting/base --users 50000 --user-k 5 --item-k 3 --min-rating 4 --max-age-days 0 --seed 2020",
            "candidate_user_sampling": "deterministic bottom blake2b hash",
            "positive_event": "rating >= 4",
            "deduplication": "exact (user,item,rating,timestamp,review_text) rows",
            "k_core": {"user_k": 5, "item_k": 3},
            "pre_split_recency_filter_days": 0,
            "timestamp": "unixReviewTime seconds since epoch",
        },
        "counts": {
            "events": sum(users.values()), "users": len(users), "items": len(items),
            "rating": dict(sorted(ratings.items())),
            "timestamp_min": min(timestamps), "timestamp_max": max(timestamps),
            "interactions_per_user": {"min": counts[0], "q25": quantiles[0],
                                      "median": statistics.median(counts), "q75": quantiles[2],
                                      "max": counts[-1]},
            "metadata_matched_items": len(items & metadata_items),
            "metadata_unmatched_items": len(items - metadata_items),
        },
        "metadata_policy": {
            "columns": ["brand", "category", "attributes", "title", "description"],
            "missing": "retain item with empty fields",
            "assumption": "static catalog side information available at recommendation time",
        },
        "sha256": {key: digest(path) for key, path in FILES.items()},
    }
    output = ROOT / "reporting/data_manifest.json"
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
