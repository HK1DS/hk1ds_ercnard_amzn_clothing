#!/usr/bin/env python3
"""Seed only unchanged static-item cache entries into a protocol workdir."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with open(args.profiles, encoding="utf-8-sig", newline="") as handle:
        allowed = {row["item_profile"].strip() for row in csv.DictReader(handle)
                   if row["item_profile"].strip()}
    source = json.loads(Path(args.source).read_text(encoding="utf-8"))
    seeded = {key: value for key, value in source.get("item", {}).items() if key in allowed}
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing cache: {output}")
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text(json.dumps({"user": {}, "item": seeded}, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(output)
    print(f"seeded {len(seeded)} unchanged static item profiles; user entries=0")


if __name__ == "__main__":
    main()
