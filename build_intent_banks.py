#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build intent banks from step2 CSV for IKGR.

Inputs:
  --step2_csv   Path to run/step2_related_intents.csv
  --encoder     SentenceTransformers model name (default: all-mpnet-base-v2)
  --user_out    Path to save user bank .pt (default: run/user_bank.pt)
  --item_out    Path to save item bank .pt (default: run/item_bank.pt)

Output format:
  torch.save({
      "encoder": <model name>,
      "dim": <embedding_dim>,
      "bank": { <raw_token: str> : torch.Tensor[k, d] }
  }, path)
"""

import argparse, ast, json, os
import numpy as np
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer

def _as_list(s):
    if isinstance(s, str) and s.strip():
        try:
            return list(ast.literal_eval(s))
        except Exception:
            try:
                return list(eval(s))
            except Exception:
                return []
    return []

def _unique_keep_order(lst):
    seen, out = set(), []
    for x in lst:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

def build_banks(step2_csv, encoder_name, user_out, item_out, vocab_out=None, emb_out=None):
    df = pd.read_csv(step2_csv).fillna("")
    enc = SentenceTransformer(encoder_name)

    # --- Pass 1: collect per-token intent lists (exact + related, dedup, order-preserving)
    user_intents = {}  # uid_raw -> List[str]
    item_intents = {}  # iid_raw -> List[str]
    for _, r in df.iterrows():
        uid_raw = str(r.get("user_id", "")).strip()
        iid_raw = str(r.get("item_id", "")).strip()
        u_ints = _unique_keep_order(_as_list(r.get("user_intents_exact", "[]")) + _as_list(r.get("user_intents_related", "[]")))
        i_ints = _unique_keep_order(_as_list(r.get("item_intents_exact", "[]")) + _as_list(r.get("item_intents_related", "[]")))
        if uid_raw and u_ints and uid_raw not in user_intents:
            user_intents[uid_raw] = u_ints
        if iid_raw and i_ints and iid_raw not in item_intents:
            item_intents[iid_raw] = i_ints

    # --- Encode every UNIQUE intent string exactly once (huge speedup on CPU)
    vocab = set()
    for lst in user_intents.values():
        vocab.update(lst)
    for lst in item_intents.values():
        vocab.update(lst)
    vocab = sorted(vocab)
    print(f"[encode] unique intents: {len(vocab)} (was {sum(len(v) for v in user_intents.values()) + sum(len(v) for v in item_intents.values())} non-unique)")

    emb_all = enc.encode(
        vocab,
        convert_to_tensor=True,
        batch_size=256,
        show_progress_bar=True,
    ).cpu()  # [V, d]
    emb_map = {tok: emb_all[idx] for idx, tok in enumerate(vocab)}
    if vocab_out:
        os.makedirs(os.path.dirname(vocab_out), exist_ok=True)
        with open(vocab_out, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False)
    if emb_out:
        os.makedirs(os.path.dirname(emb_out), exist_ok=True)
        np.save(emb_out, emb_all.numpy())

    # --- Assemble per-token banks by stacking the relevant intent embeddings (order preserved)
    user_bank = {uid: torch.stack([emb_map[t] for t in ints]) for uid, ints in user_intents.items()}
    item_bank = {iid: torch.stack([emb_map[t] for t in ints]) for iid, ints in item_intents.items()}

    dim = int(emb_all.shape[1]) if emb_all.numel() else 0

    os.makedirs(os.path.dirname(user_out), exist_ok=True)
    os.makedirs(os.path.dirname(item_out), exist_ok=True)

    torch.save({"encoder": encoder_name, "dim": dim, "bank": user_bank}, user_out)
    torch.save({"encoder": encoder_name, "dim": dim, "bank": item_bank}, item_out)

    print(f"[OK] user_bank → {user_out}  ({len(user_bank)} users, dim={dim})")
    print(f"[OK] item_bank → {item_out}  ({len(item_bank)} items, dim={dim})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step2_csv", required=True, help="Path to step2_related_intents.csv")
    ap.add_argument("--encoder", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--user_out", default="run/user_bank.pt")
    ap.add_argument("--item_out", default="run/item_bank.pt")
    ap.add_argument("--vocab-out", default=None)
    ap.add_argument("--emb-out", default=None)
    args = ap.parse_args()

    build_banks(args.step2_csv, args.encoder, args.user_out, args.item_out,
                args.vocab_out, args.emb_out)

if __name__ == "__main__":
    main()
