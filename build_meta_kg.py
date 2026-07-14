#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build an item-side metadata KG pack from a generic CSV file.

The output is consumed by ``ikgr_core.model_ikgr.IKGR`` when ``use_meta_kg`` is
enabled. It is intentionally dataset-agnostic: any item metadata can be mapped
into three broad relation groups:

  author_cols     -> item_authors
  publisher_cols  -> item_publishers
  tag_cols        -> item_shelves

Column values may be plain strings, delimiter-separated strings, or Python/JSON
list literals such as ``["fiction", "children"]``.
"""
import argparse
import ast
import os
import re
from collections import defaultdict

import pandas as pd
import torch


DEFAULT_AUTHOR_COLS = ["authors", "author", "brand"]
DEFAULT_PUBLISHER_COLS = ["publisher", "publishers", "manufacturer"]
DEFAULT_TAG_COLS = ["shelves", "shelf", "categories", "category", "attributes", "attribute"]


def _split_cols(raw, default):
    if raw is None:
        return list(default)
    return [c.strip() for c in raw.split(",") if c.strip()]


def _as_values(value):
    """Normalize a metadata cell into a short list of clean string values."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        seq = value
    else:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                seq = parsed
            else:
                seq = re.split(r"[|;/,]", text)
        except Exception:
            seq = re.split(r"[|;/,]", text)
    out = []
    for item in seq:
        text = str(item).strip().strip("'\"")
        text = re.sub(r"\s+", " ", text)
        if text and text.lower() not in {"nan", "none", "null"}:
            out.append(text)
    return out


def _build_group(df, item_col, cols):
    vocab = {}
    item_nodes = defaultdict(list)
    present = [c for c in cols if c in df.columns]
    for _, row in df.iterrows():
        item = str(row[item_col])
        seen = set()
        for col in present:
            for value in _as_values(row.get(col)):
                key = value.lower()
                if key not in vocab:
                    vocab[key] = len(vocab)
                node_id = vocab[key]
                if node_id not in seen:
                    item_nodes[item].append(node_id)
                    seen.add(node_id)
    return dict(item_nodes), vocab, present


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True,
                        help="CSV with item_id and optional metadata columns")
    parser.add_argument("--out", default="run/meta_kg_pack.pt")
    parser.add_argument("--item-col", default="item_id")
    parser.add_argument("--author-cols", default=None,
                        help="comma-separated columns mapped to author/brand nodes")
    parser.add_argument("--publisher-cols", default=None,
                        help="comma-separated columns mapped to publisher/manufacturer nodes")
    parser.add_argument("--tag-cols", default=None,
                        help="comma-separated columns mapped to category/tag/attribute nodes")
    args = parser.parse_args()

    df = pd.read_csv(args.metadata).fillna("")
    if args.item_col not in df.columns:
        raise SystemExit(f"metadata CSV must contain '{args.item_col}'")

    author_cols = _split_cols(args.author_cols, DEFAULT_AUTHOR_COLS)
    publisher_cols = _split_cols(args.publisher_cols, DEFAULT_PUBLISHER_COLS)
    tag_cols = _split_cols(args.tag_cols, DEFAULT_TAG_COLS)

    item_authors, author_vocab, used_authors = _build_group(df, args.item_col, author_cols)
    item_publishers, publisher_vocab, used_publishers = _build_group(df, args.item_col, publisher_cols)
    item_shelves, shelf_vocab, used_tags = _build_group(df, args.item_col, tag_cols)

    pack = {
        "item_authors": item_authors,
        "item_publishers": item_publishers,
        "item_shelves": item_shelves,
        "n_authors": len(author_vocab),
        "n_publishers": len(publisher_vocab),
        "n_shelves": len(shelf_vocab),
        "author_vocab": author_vocab,
        "publisher_vocab": publisher_vocab,
        "shelf_vocab": shelf_vocab,
        "source_csv": os.path.abspath(args.metadata),
        "used_columns": {
            "author_cols": used_authors,
            "publisher_cols": used_publishers,
            "tag_cols": used_tags,
        },
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(pack, args.out)
    print(
        "[OK] meta KG saved to "
        f"{args.out} | authors={len(author_vocab)} publishers={len(publisher_vocab)} "
        f"tags={len(shelf_vocab)} items={len(set(df[args.item_col].astype(str)))}"
    )


if __name__ == "__main__":
    main()
