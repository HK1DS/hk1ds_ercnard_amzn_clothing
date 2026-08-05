#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CORONA stage-3 (3-1): dynamic-KG candidate generation / coarse retrieval.

Reimplements the diagram's "dynamic KG exploration -> candidate generation" as a
distance-aware, graph-neighbor retriever on top of our IKGR/meta-KG. Instead of
ranking all items (full-sort), each user gets a compact candidate set drawn from
graph neighbors; the downstream model then ranks ONLY within that set.

Channels (all TRAIN-only, LLM-free, deterministic):
  - intent : user -> intent -> item        (shared LLM-intent neighbors)
  - shelf  : user-history -> shelf  -> item (shared popular-shelf neighbors)
  - author : user-history -> author -> item
  - pub    : user-history -> publisher -> item
  - cf     : item-item co-occurrence from the user's history (collaborative)

Candidate score for (u, item) = weighted sum of co-occurrence counts across the
above channels (a simple "distance prior": more shared neighbors / shorter paths
-> higher). Top-M items per user form the candidate set.

This is purely a retrieval/eval-side mechanism: it does not change training. It
is used by eval_slices.py to mask non-candidate items before ranking, and to
report candidate recall@M (the retrieval ceiling) and average candidate size.
"""
import numpy as np
import torch
import scipy.sparse as sp


def _binary_csr(rows, cols, shape):
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    data = np.ones(len(rows), dtype=np.float32)
    m = sp.csr_matrix((data, (rows, cols)), shape=shape)
    m.data[:] = 1.0  # collapse duplicates to binary
    return m


class CoronaRetriever:
    def __init__(self, train_data, config, kg_pack_path=None, meta_kg_path=None,
                 weights=None, use_cf=True, idf=False, pop_norm=0.0,
                 train_only_kg=False):
        ds = train_data.dataset
        self.n_users = int(ds.user_num)
        self.n_items = int(ds.item_num)
        uf, itf = config["USER_ID_FIELD"], config["ITEM_ID_FIELD"]
        u_tok2id = ds.field2token_id[uf]
        i_tok2id = ds.field2token_id[itf]
        inter = ds.inter_feat
        u = inter[uf].numpy()
        it = inter[itf].numpy()
        allowed_items = set(map(int, it.tolist())) if train_only_kg else None

        # train user-item binary (history) ------------------------------------
        self.UI = _binary_csr(u, it, (self.n_users, self.n_items))
        # item popularity (train degree) for popularity de-biasing -------------
        self.item_pop = np.asarray(self.UI.sum(axis=0)).ravel().astype(np.float64)
        self.idf = bool(idf)
        self.pop_norm = float(pop_norm)

        # default channel weights (set membership is robust to exact values) ---
        w = {"intent": 1.0, "shelf": 1.0, "author": 1.0, "pub": 1.0, "cf": 1.0}
        if weights:
            w.update(weights)
        self.w = w
        self.use_cf = use_cf

        # intent channel: user<->intent, item<->intent ------------------------
        # Build user intent exposure from TRAIN user-item history, rather than
        # LLM user-profile text. The latter can contain a review written after
        # the temporal cutoff and would leak test-period preference into CORONA.
        self.UInt = None
        self.IInt = None
        if kg_pack_path:
            kp = torch.load(kg_pack_path, map_location="cpu")
            n_int = int(kp["n_intents"])
            IInt = self._tok_node_mat(kp["item_intents"], i_tok2id, self.n_items, n_int,
                                      allowed_rows=allowed_items)
            self.UInt = (self.UI @ IInt).tocsr()
            if self.UInt.nnz:
                self.UInt.data[:] = 1.0
            self.IInt = self._maybe_idf(IInt)

        # metadata channels: item<->{shelf,author,publisher} -----------------
        self.meta = {}
        if meta_kg_path:
            mp = torch.load(meta_kg_path, map_location="cpu")
            for key, nkey, name in [("item_shelves", "n_shelves", "shelf"),
                                    ("item_authors", "n_authors", "author"),
                                    ("item_publishers", "n_publishers", "pub")]:
                if key in mp:
                    M = self._tok_node_mat(mp[key], i_tok2id, self.n_items, int(mp[nkey]),
                                           allowed_rows=allowed_items)
                    if M.nnz > 0:
                        self.meta[name] = self._maybe_idf(M)

        # CF item-item co-occurrence (sparse): Co = UI^T @ UI ------------------
        self.Co = None
        if self.use_cf:
            self.Co = (self.UI.T @ self.UI).tocsr()

    def _maybe_idf(self, M):
        """Column-scale an item<->node matrix by IDF to down-weight ubiquitous
        nodes (e.g. the 'to-read'/'children' shelves that connect to most items
        and cause popularity bias). No-op if idf disabled."""
        if not self.idf:
            return M
        df = np.asarray((M > 0).sum(axis=0)).ravel().astype(np.float64)  # items per node
        idf = np.log((self.n_items + 1.0) / (df + 1.0)) + 1.0
        return (M @ sp.diags(idf)).tocsr()

    @staticmethod
    def _tok_node_mat(d, tok2id, n_rows, n_cols, allowed_rows=None):
        rows, cols = [], []
        for tok, ids in d.items():
            rid = tok2id.get(str(tok))
            if rid is None:
                continue
            if allowed_rows is not None and int(rid) not in allowed_rows:
                continue
            ids = ids.tolist() if torch.is_tensor(ids) else list(ids)
            for j in ids:
                rows.append(rid)
                cols.append(int(j))
        return _binary_csr(rows, cols, (n_rows, max(n_cols, 1)))

    def _scores(self, user_ids):
        """Dense candidate scores [B, n_items] for a (small) batch of user ids."""
        B = len(user_ids)
        score = np.zeros((B, self.n_items), dtype=np.float32)
        UIb = self.UI[user_ids]                       # [B, n_items] history
        if self.UInt is not None and self.IInt is not None and self.w["intent"]:
            score += self.w["intent"] * (self.UInt[user_ids] @ self.IInt.T).toarray()
        for name, M in self.meta.items():
            if not self.w.get(name):
                continue
            prof = UIb @ M                            # [B, n_nodes] user meta profile
            score += self.w[name] * (prof @ M.T).toarray()
        if self.Co is not None and self.w["cf"]:
            score += self.w["cf"] * (UIb @ self.Co).toarray()
        # popularity de-bias: divide candidate score by item_pop^beta so that
        # popular items stop dominating the candidate set (promotes long-tail).
        if self.pop_norm > 0:
            score = score / np.power(self.item_pop + 1.0, self.pop_norm)[None, :]
        return score

    def normalized_prior(self, user_ids):
        """Return a bounded graph-retrieval prior in [0, 1] for reranking.

        Candidate restriction uses only the top-M set, which can discard a
        relevant item before the neural ranker sees it.  The soft reranker keeps
        the full catalog and uses this normalized graph score as a small prior.
        Log compression prevents a few high-degree metadata nodes from making
        the prior dominate a user's ranking.
        """
        score = self._scores(user_ids)
        score = np.log1p(np.maximum(score, 0.0))
        denom = score.max(axis=1, keepdims=True)
        prior = np.divide(score, denom, out=np.zeros_like(score), where=denom > 0)
        # A sparse product can occasionally surface a non-finite value on a
        # degenerate user row. A NaN would poison even ``0 * prior`` in the
        # lambda=0 control, so make the neutral prior explicit.
        return np.nan_to_num(prior, nan=0.0, posinf=0.0, neginf=0.0)

    def candidates(self, user_ids, M, chunk=256):
        """Return top-M candidate item ids per user as int64 array [len(users), M].
        Processed in chunks to bound memory."""
        user_ids = np.asarray(user_ids, dtype=np.int64)
        if M <= 0:
            raise ValueError("M must be positive; use normalized_prior for soft reranking.")
        M = min(M, self.n_items)
        out = np.zeros((len(user_ids), M), dtype=np.int64)
        for s in range(0, len(user_ids), chunk):
            uchunk = user_ids[s:s + chunk]
            sc = self._scores(uchunk)                 # [b, n_items]
            # top-M by score (unsorted partition is enough for set membership)
            idx = np.argpartition(-sc, M - 1, axis=1)[:, :M]
            out[s:s + len(uchunk)] = idx
        return out
