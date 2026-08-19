#!/usr/bin/env python3
"""Build leakage-safe Amazon Clothing profiles for RecBole TO/TO_GLOBAL.

The split math intentionally mirrors RecBole 1.2.0 Dataset.split_by_ratio:
non-train parts are floored and the remainder is assigned to train.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def split_counts(total, ratios):
    ratios = [value / sum(ratios) for value in ratios]
    counts = [int(value * total) for value in ratios]
    counts[0] = total - sum(counts[1:])
    for index in range(1, len(ratios)):
        if counts[0] <= 1:
            break
        if 0 < ratios[-index] * total < 1:
            counts[-index] += 1
            counts[0] -= 1
    return counts


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, columns, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def item_text(row):
    return " | ".join(filter(None, (
        row.get("title", ""), row.get("brand", ""), row.get("category", ""),
        row.get("attributes", ""), row.get("description", ""),
    )))[:2400] or f"Amazon item {row['item_id']}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactions", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--protocol", choices=("TO", "TO_GLOBAL"), required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-profile-events", type=int, default=5)
    args = parser.parse_args()

    events = read_csv(args.interactions)
    required = {"user_id", "item_id", "rating", "timestamp", "review_text"}
    missing = required - set(events[0]) if events else required
    if missing:
        raise SystemExit(f"interactions missing columns: {sorted(missing)}")
    for order, row in enumerate(events):
        row["_order"] = order
        row["_ts"] = int(float(row["timestamp"]))
    events.sort(key=lambda row: (row["_ts"], row["_order"]))

    all_users = sorted({row["user_id"] for row in events})
    if args.protocol == "TO_GLOBAL":
        train_count, valid_count, test_count = split_counts(len(events), [0.7, 0.1, 0.2])
        train = events[:train_count]
        global_cutoff = train[-1]["_ts"] if train else None
        cutoffs = {user: global_cutoff for user in all_users}
    else:
        grouped = defaultdict(list)
        for row in events:
            grouped[row["user_id"]].append(row)
        train, valid_count, test_count = [], 0, 0
        cutoffs = {}
        for user in all_users:
            rows = grouped[user]
            train_count, n_valid, n_test = split_counts(len(rows), [0.8, 0.1, 0.1])
            selected = rows[:train_count]
            train.extend(selected)
            valid_count += n_valid
            test_count += n_test
            cutoffs[user] = selected[-1]["_ts"] if selected else None
        train_count = len(train)

    train_by_user = defaultdict(list)
    for row in train:
        train_by_user[row["user_id"]].append(row)

    metadata = {row["item_id"]: row for row in read_csv(args.metadata)}
    all_items = sorted({row["item_id"] for row in events})
    profiles = {}
    audits = []
    for user in all_users:
        rows = sorted(train_by_user[user], key=lambda row: (row["_ts"], row["_order"]))
        snippets = [row["review_text"].strip() for row in rows if row["review_text"].strip()]
        snippets = snippets[-args.max_profile_events:]
        profile = " ".join(snippets)[:2400]
        profiles[user] = profile
        max_ts = max((row["_ts"] for row in rows if row["review_text"].strip()), default=None)
        cutoff = cutoffs[user]
        within = max_ts is None or (cutoff is not None and max_ts <= cutoff)
        audits.append({
            "user_id": user, "profile_event_count": len(snippets),
            "max_profile_timestamp": "" if max_ts is None else max_ts,
            "train_cutoff_timestamp": "" if cutoff is None else cutoff,
            "within_train": str(within).lower(), "train_interaction_count": len(rows),
            "profile_text_sha256": hashlib.sha256(profile.encode("utf-8")).hexdigest(),
        })
    if not all(row["within_train"] == "true" for row in audits):
        raise RuntimeError("profile time audit failed")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "interactions.csv", ["user_id", "item_id", "rating", "timestamp"], (
        {key: row[key] for key in ("user_id", "item_id", "rating", "timestamp")} for row in events
    ))
    write_csv(out / "item_metadata.csv", list(next(iter(metadata.values())).keys()), metadata.values())
    count = max(len(all_users), len(all_items))
    profile_rows = []
    for index in range(count):
        user, item = all_users[index % len(all_users)], all_items[index % len(all_items)]
        profile_rows.append({"user_id": user, "user_profile": profiles[user], "item_id": item,
                             "item_profile": item_text(metadata.get(item, {"item_id": item}))})
    write_csv(out / "profiles.csv", ["user_id", "user_profile", "item_id", "item_profile"], profile_rows)
    write_csv(out / "profile_time_audit.csv", list(audits[0]), audits)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "protocol": args.protocol,
        "profile_scope": "global_train" if args.protocol == "TO_GLOBAL" else "per_user_train",
        "split": [0.7, 0.1, 0.2] if args.protocol == "TO_GLOBAL" else [0.8, 0.1, 0.1],
        "split_math": "RecBole 1.2.0 split_by_ratio compatible; temporal ascending",
        "counts": {"events": len(events), "users": len(all_users), "items": len(all_items),
                   "train_events": train_count, "valid_events": valid_count, "test_events": test_count,
                   "nonempty_profiles": sum(bool(value) for value in profiles.values()),
                   "empty_profiles": sum(not value for value in profiles.values())},
        "max_train_timestamp": max((row["_ts"] for row in train), default=None),
        "profile_audit_passed": True,
        "inputs": {"interactions_sha256": sha256(args.interactions),
                   "metadata_sha256": sha256(args.metadata)},
        "assumptions": {"metadata": "static catalog side information available at recommendation time"},
    }
    (out / "profile_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
