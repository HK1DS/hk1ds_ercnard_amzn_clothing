#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-seed sliced evaluation for the IKGR "squeeze": overall vs long-tail vs
coverage/novelty, with robustness across seeds.

Models: IKGR/MF, intent-KG, metadata-KG, DynLLM recency, CORONA retrieval,
BPR, and LightGCN.
For each model x seed: train on the same protocol, compute per-user metrics,
slice by user activity (cold-start) and item popularity (long-tail), plus
catalog coverage@10 and recommended-item novelty. Aggregate mean +/- std over
seeds.  Output: run/slice_eval_result.json

Env overrides (for smoke tests):
  IKGR_SEEDS=2020            comma-separated seeds (default 2020,2021,2022)
  IKGR_EPOCHS=2              epochs override
  IKGR_SPECS=IKGR_kgon_L2,IKGR_kgoff   subset of specs
"""
import os, json, time, math, yaml, hashlib, random
import scipy.sparse as _sp
if not hasattr(_sp.dok_matrix, "_update"):
    _sp.dok_matrix._update = dict.update
import numpy as np
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model, get_trainer, init_seed
from ikgr_core.model_ikgr import IKGR as IKGRModel

KS = [10, 30]
MAXK = max(KS)
COV_K = 10
EVAL_SCHEMA_VERSION = 2


def _set_determinism(seed):
    """Reset every RNG used by this custom eval path before each training run."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)
    init_seed(seed, reproducibility=True)


def _experiment_context(name, model_arg, extra, rb, paths, split):
    """Fingerprint evaluation inputs so stale seed results cannot be reused."""
    source_paths = [__file__, "ikgr_core/model_ikgr.py", "ikgr_core/corona_retriever.py"]
    code = {}
    for path in source_paths:
        try:
            with open(path, "rb") as f:
                code[path] = hashlib.sha256(f.read()).hexdigest()
        except OSError:
            code[path] = None
    context = {
        "schema": EVAL_SCHEMA_VERSION,
        "spec": name,
        "model": model_arg if isinstance(model_arg, str) else model_arg.__name__,
        "split": split,
        "epochs": int(os.environ.get("IKGR_EPOCHS", rb["epochs"])),
        "embedding_size": int(os.environ.get("IKGR_EMBEDDING_SIZE", rb["embedding_size"])),
        "learning_rate": float(os.environ.get("IKGR_LEARNING_RATE", "1e-3")),
        "reg_weight": float(os.environ.get("IKGR_REG_WEIGHT", "1e-6")),
        "extra": extra,
        "paths": paths,
        "recbole": rb,
        "code": code,
    }
    encoded = json.dumps(context, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16], context


def _float_grid_from_env(name, default):
    raw = os.environ.get(name)
    if not raw:
        return default
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if not values or any(value < 0 for value in values):
        raise ValueError(f"{name} must contain one or more non-negative floats")
    return values


def _config(rb, paths, extra, seed):
    split = os.environ.get("IKGR_SPLIT", "RS").upper()
    is_temporal = split in ("TO", "TO_GLOBAL")
    load_inter = ["user_id", "item_id", "rating"] + (["timestamp"] if is_temporal else [])
    workdir = paths.get("workdir", "run/")
    cd = {
        "epochs": int(os.environ.get("IKGR_EPOCHS", rb["epochs"])),
        "metrics": rb["metrics"], "topk": rb["topk"],
        "embedding_size": int(os.environ.get("IKGR_EMBEDDING_SIZE", rb["embedding_size"])),
        "learning_rate": float(os.environ.get("IKGR_LEARNING_RATE", "1e-3")),
        "reg_weight": float(os.environ.get("IKGR_REG_WEIGHT", "1e-6")),
        "dropout_prob": rb.get("dropout", 0.1),
        "data_path": os.path.dirname(paths["inter_file"]),
        "USER_ID_FIELD": "user_id", "ITEM_ID_FIELD": "item_id", "LABEL_FIELD": "rating",
        "load_col": {"inter": load_inter},
        "train_neg_sample_args": {"distribution": "uniform"},
        "save_dataset": False, "save_dataloaders": False, "show_progress": False,
        "checkpoint_dir": os.path.abspath(os.path.join(workdir, "recbole_slice")),
        "eval_step": 5,
        "seed": seed, "reproducibility": True,
    }
    if split == "TO":
        # temporal (per-user time-ordered) split: train=earliest, test=latest
        cd["TIME_FIELD"] = "timestamp"
        cd["eval_args"] = {"split": {"RS": [0.8, 0.1, 0.1]}, "order": "TO",
                           "group_by": "user", "mode": "full"}
    elif split == "TO_GLOBAL":
        # GLOBAL temporal split (not grouped by user): a single time cutoff over
        # ALL interactions, so users active only in the late window become genuine
        # cold-start users (few/no train interactions). 70/10/20 -> bigger, colder
        # test window. group_by=None disables per-user grouping.
        cd["TIME_FIELD"] = "timestamp"
        cd["eval_args"] = {"split": {"RS": [0.7, 0.1, 0.2]}, "order": "TO",
                           "group_by": None, "mode": "full"}
    cd.update(extra)
    return cd


def _build_recency(model, train_data, config):
    """Per-user recency profile from TRAIN interactions only (no leakage):
    top-N most recent items + exp-decay weights -> model.set_recency()."""
    from collections import defaultdict
    inter = train_data.dataset.inter_feat
    uf, itf = config["USER_ID_FIELD"], config["ITEM_ID_FIELD"]
    if "timestamp" not in inter.columns:
        raise RuntimeError("use_dynamic requires timestamp (run with IKGR_SPLIT=TO).")
    u = inter[uf].numpy(); it = inter[itf].numpy(); ts = inter["timestamp"].numpy()
    N = model.recency_topn
    tau = max(model.recency_tau_days * 86400.0, 1.0)
    byu = defaultdict(list)
    for a, b, c in zip(u, it, ts):
        byu[int(a)].append((float(c), int(b)))
    ids = np.zeros((model.n_users, N), dtype=np.int64)
    wts = np.zeros((model.n_users, N), dtype=np.float32)
    for uid, lst in byu.items():
        lst.sort(reverse=True)            # most recent first
        lst = lst[:N]
        tref = lst[0][0]
        w = np.exp(-np.array([tref - c for c, _ in lst], dtype=np.float64) / tau)
        s = w.sum()
        w = (w / s) if s > 0 else w
        for j, (c, b) in enumerate(lst):
            ids[uid, j] = b; wts[uid, j] = w[j]
    dev = next(model.parameters()).device
    model.set_recency(torch.from_numpy(ids).to(dev), torch.from_numpy(wts).to(dev))
    print(f"  [recency] built for {len(byu)} users from {len(u)} train inters (N={N})", flush=True)


def _blend_rerank_scores(scores, graph_prior, score_scale, lam):
    """Apply a relative graph prior without altering the zero-lambda control."""
    if lam is None or lam == 0.0:
        return scores
    return scores + lam * score_scale * graph_prior


def _train_and_collect(model_arg, extra, rb, paths, seed):
    os.makedirs(os.path.join(paths.get("workdir", "run/"), "recbole_slice"), exist_ok=True)
    _set_determinism(seed)
    extra = dict(extra)
    # CORONA stage-3 (3-1): candidate-set restriction and soft reranking are
    # eval-only knobs, not model parameters, so pop them before building the
    # RecBole config. A rerank grid shares one trained model across lambdas.
    cand_m = int(extra.pop("corona_cand", 0))
    cand_cf = bool(extra.pop("corona_cf", True))
    cand_idf = bool(extra.pop("corona_idf", False))
    cand_popnorm = float(extra.pop("corona_popnorm", 0.0))
    cand_weights = extra.pop("corona_weights", None)
    rerank_lambdas = [float(x) for x in extra.pop("corona_rerank_grid", [])]
    config = Config(model=model_arg, dataset=rb["dataset"], config_dict=_config(rb, paths, extra, seed))
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    klass = model_arg if not isinstance(model_arg, str) else get_model(config["model"])
    model = klass(config, train_data.dataset).to(config["device"])
    if getattr(model, "use_dynamic", False):
        _build_recency(model, train_data, config)
    retriever = None
    if cand_m > 0 or rerank_lambdas:
        from ikgr_core.corona_retriever import CoronaRetriever
        retriever = CoronaRetriever(train_data, config,
                                    kg_pack_path=extra.get("kg_pack_path"),
                                    meta_kg_path=extra.get("meta_kg_path"),
                                    weights=cand_weights, use_cf=cand_cf,
                                    idf=cand_idf, pop_norm=cand_popnorm)
        mode = f"M={cand_m}" if cand_m > 0 else f"soft_rerank={rerank_lambdas}"
        print(f"  [corona] retriever ready ({mode}, cf={cand_cf}, "
              f"idf={cand_idf}, pop_norm={cand_popnorm})", flush=True)
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    t0 = time.time()
    trainer.fit(train_data, valid_data, saved=True, show_progress=False)
    train_sec = round(time.time() - t0, 1)
    smf = getattr(trainer, "saved_model_file", None)
    if smf and os.path.exists(smf):
        # RecBole checkpoints contain optimizer/config objects in addition to
        # tensors. PyTorch >=2.6 defaults weights_only=True, which rejects this
        # trusted checkpoint created moments earlier by this local trainer.
        ck = torch.load(smf, map_location=config["device"], weights_only=False)
        model.load_state_dict(ck["state_dict"])
        if ck.get("other_parameter"):
            model.load_other_parameter(ck["other_parameter"])
    model.eval()

    dev = config["device"]
    n_items = int(dataset.item_num)
    uid_field = config["USER_ID_FIELD"]
    # Cache all-item representations once (huge full-sort speedup for KG/meta
    # models, which otherwise rebuild every item's embedding per test batch).
    if hasattr(model, "eval_cache_items"):
        model.eval_cache_items()
    item_pop = np.zeros(n_items, dtype=np.int64)
    # None is the ordinary evaluation path. Each lambda receives its own
    # per-user rankings while sharing this exact trained model and test pass.
    variants = [None] if not rerank_lambdas else rerank_lambdas
    per_user = {lam: [] for lam in variants}
    cand_rec_sum, cand_size_sum, cand_n = 0.0, 0, 0
    with torch.no_grad():
        for interaction, history_index, positive_u, positive_i in test_data:
            interaction = interaction.to(dev)
            users = interaction[uid_field]
            B = users.shape[0]
            scores = model.full_sort_predict(interaction).view(B, -1)
            scores[:, 0] = -np.inf
            users_np = users.cpu().numpy()
            cand_sets = None
            # Soft reranking keeps full-sort candidates. Only explicit
            # candidate retrieval is allowed to mask scores.
            if cand_m > 0:
                cand = retriever.candidates(users_np, cand_m)      # [B, M]
                cand_mask = torch.zeros((B, n_items), dtype=torch.bool, device=dev)
                ct = torch.from_numpy(cand).to(dev)
                cand_mask.scatter_(1, ct, True)
                cand_mask[:, 0] = False
                scores = scores.masked_fill(~cand_mask, -np.inf)
                cand_sets = [set(row.tolist()) for row in cand]
            if history_index is not None:
                hr = history_index[0].cpu().numpy(); hc = history_index[1].cpu().numpy()
                np.add.at(item_pop, hc, 1)
                ucnt = np.bincount(hr, minlength=B)
                scores[history_index] = -np.inf
            else:
                ucnt = np.zeros(B, dtype=np.int64)
            graph_prior = None
            score_scale = None
            if rerank_lambdas:
                graph_prior = torch.from_numpy(retriever.normalized_prior(users_np)).to(dev)
                # Neural BPR scores have a model/seed-dependent scale.  An
                # absolute [0, 1] graph bonus can therefore swamp the base
                # ranker (or be inert).  Express lambda in each user's finite
                # full-sort score standard deviations instead.
                finite = torch.isfinite(scores)
                n_finite = finite.sum(dim=1, keepdim=True).clamp_min(1)
                safe_scores = scores.masked_fill(~finite, 0.0)
                mean = safe_scores.sum(dim=1, keepdim=True) / n_finite
                var = ((safe_scores - mean).square() * finite).sum(dim=1, keepdim=True) / n_finite
                score_scale = var.sqrt().clamp_min(1e-6)
            pu = positive_u.cpu().numpy(); pi = positive_i.cpu().numpy()
            rel_by_row = {}
            for r, it in zip(pu, pi):
                rel_by_row.setdefault(int(r), []).append(int(it))
            for lam in variants:
                rank_scores = _blend_rerank_scores(scores, graph_prior, score_scale, lam)
                topk = torch.topk(rank_scores, MAXK, dim=-1)[1].cpu().numpy()
                for row in range(B):
                    rel = rel_by_row.get(row)
                    if not rel:
                        continue
                    if cand_sets is not None:
                        cs = cand_sets[row]
                        cand_rec_sum += sum(1 for it in rel if it in cs) / len(rel)
                        cand_size_sum += len(cs); cand_n += 1
                    per_user[lam].append({"uid": int(users_np[row]), "activity": int(ucnt[row]),
                                          "topk": topk[row].tolist(), "rel": rel})
    cand_stats = {}
    if cand_m > 0 and cand_n:
        cand_stats = {"cand_recall": round(cand_rec_sum / cand_n, 4),
                      "cand_size": round(cand_size_sum / cand_n, 1), "cand_M": cand_m}
    return per_user, item_pop, train_sec, n_items, cand_stats


def _user_metrics(topk, rel_set, k):
    hits = [r for r, it in enumerate(topk[:k]) if it in rel_set]
    n_rel = len(rel_set)
    recall = len(hits) / n_rel if n_rel else 0.0
    dcg = sum(1.0 / math.log2(r + 2) for r in hits)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(n_rel, k)))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return recall, ndcg, (1.0 if hits else 0.0)


def _avg(rows, is_tail=None):
    agg = {}
    for k in KS:
        agg[f"recall@{k}"] = 0.0; agg[f"ndcg@{k}"] = 0.0; agg[f"hit@{k}"] = 0.0
    n = 0
    for topk, rel in rows:
        if is_tail is not None:
            rel = [it for it in rel if is_tail[it]]
            if not rel:
                continue
        rel_set = set(rel); n += 1
        for k in KS:
            rc, nd, ht = _user_metrics(topk, rel_set, k)
            agg[f"recall@{k}"] += rc; agg[f"ndcg@{k}"] += nd; agg[f"hit@{k}"] += ht
    if n:
        for kk in agg:
            agg[kk] = round(agg[kk] / n, 4)
    return agg, n


def slice_report(per_user, item_pop, n_items):
    pos = item_pop[item_pop > 0]
    head_cut = float(np.percentile(pos, 80)) if pos.size else 0.0
    is_tail = item_pop <= head_cut

    rows_all = [(r["topk"], r["rel"]) for r in per_user]
    overall, n_all = _avg(rows_all)
    tail, n_tail = _avg(rows_all, is_tail=is_tail)

    # cold-start buckets by activity quantiles
    acts = np.array([r["activity"] for r in per_user])
    q = np.percentile(acts, [20, 40, 60, 80]) if acts.size else [0, 0, 0, 0]
    def b(a):
        return ("Q1_cold" if a <= q[0] else "Q2" if a <= q[1] else "Q3" if a <= q[2]
                else "Q4" if a <= q[3] else "Q5_warm")
    buckets = {}
    for r in per_user:
        buckets.setdefault(b(r["activity"]), []).append((r["topk"], r["rel"]))
    bucket_metrics = {}
    for name, rows in buckets.items():
        m, n = _avg(rows)
        bucket_metrics[name] = {"n_users": n, "ndcg@10": m["ndcg@10"], "recall@10": m["recall@10"]}

    # ABSOLUTE cold-start buckets by # train interactions (not quantiles). These
    # isolate genuinely cold users -- meaningful under the global temporal split
    # (TO_GLOBAL), where users appearing only in the test window have 0 / few
    # train interactions. (Under per-user TO every user is warm, so these are ~empty.)
    abs_buckets = {}
    for name, lo, hi in [("cold0_train", 0, 0), ("cold_1_5", 1, 5),
                         ("cold_6_20", 6, 20), ("warm_gt20", 21, 10**9)]:
        rows = [(r["topk"], r["rel"]) for r in per_user if lo <= r["activity"] <= hi]
        m, n = _avg(rows)
        abs_buckets[name] = {"n_users": n, "ndcg@10": m["ndcg@10"],
                             "recall@10": m["recall@10"], "recall@30": m["recall@30"]}

    # coverage@10 and novelty (mean self-information of recommended items)
    rec_items = set()
    pop_sum, pop_cnt = 0.0, 0
    total = max(1, int(item_pop.sum()))
    novelty = 0.0
    for r in per_user:
        for it in r["topk"][:COV_K]:
            rec_items.add(it)
            p = item_pop[it] / total
            novelty += -math.log2(p) if p > 0 else 0.0
            pop_sum += item_pop[it]; pop_cnt += 1
    coverage = round(len(rec_items) / max(1, n_items - 1), 4)  # exclude pad item
    novelty = round(novelty / max(1, pop_cnt), 4)
    avg_rec_pop = round(pop_sum / max(1, pop_cnt), 1)

    return {
        "overall": overall, "n_test_users": n_all,
        "long_tail": {"head_cut_pop": head_cut, "n_tail_items": int(is_tail.sum()),
                      "n_users_with_tail_rel": n_tail, **tail},
        "cold_start_buckets": dict(sorted(bucket_metrics.items())),
        "cold_abs_buckets": abs_buckets,
        "activity_q_20_40_60_80": [float(x) for x in q],
        "coverage@10": coverage, "novelty_bits": novelty, "avg_rec_popularity": avg_rec_pop,
    }


def _aggregate(per_seed):
    """mean/std over seeds for headline metrics."""
    keys = [("overall", "ndcg@10"), ("overall", "recall@10"),
            ("long_tail", "recall@10"), ("long_tail", "recall@30"),
            ("coverage@10", None), ("novelty_bits", None)]
    out = {}
    for grp, sub in keys:
        vals = []
        for rep in per_seed.values():
            v = rep[grp] if sub is None else rep[grp][sub]
            vals.append(float(v))
        label = grp if sub is None else f"{grp}.{sub}"
        out[label] = {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4),
                      "seeds": vals}
    return out


def main():
    cfg = yaml.safe_load(open(os.environ.get("IKGR_CONFIG", "config.yaml"), encoding="utf-8")); paths, rb = cfg["paths"], cfg["recbole"]
    kg = os.path.abspath(paths.get("kg_pack", "run/kg_pack.pt"))
    meta = os.path.abspath(paths.get("meta_kg_pack", "run/meta_kg_pack.pt"))
    meta_extra = {"use_meta_kg": True, "meta_kg_path": meta} if os.path.exists(meta) else {}
    meta_path = meta if os.path.exists(meta) else None
    all_specs = {
        "IKGR_kgoff":   (IKGRModel, {"use_kg": False}),
        "IKGR_kgon_L1": (IKGRModel, {"use_kg": True, "kg_pack_path": kg, "kg_layers": 1, "kg_cap": 32}),
        "IKGR_kgon_L1_frozen": (IKGRModel, {"use_kg": True, "kg_pack_path": kg, "kg_layers": 1,
                                            "kg_cap": 32, "intent_learnable": False}),
        "IKGR_kgon_L2": (IKGRModel, {"use_kg": True, "kg_pack_path": kg, "kg_layers": 2, "kg_cap": 32}),
        # heterogeneous metadata KG (brand/category/attribute), LLM-free
        "IKGR_meta_only": (IKGRModel, {"use_kg": False, "use_meta_kg": True, "meta_kg_path": meta, "kg_cap": 32}),
        "IKGR_full_hetero": (IKGRModel, {"use_kg": True, "kg_pack_path": kg, "kg_layers": 1, "kg_cap": 32,
                                         "intent_learnable": False, **meta_extra}),
        "IKGR_dyn": (IKGRModel, {"use_kg": True, "kg_pack_path": kg, "kg_layers": 1, "kg_cap": 32,
                                 "intent_learnable": False, "use_dynamic": True, **meta_extra}),
        "IKGR_dyn_attn": (IKGRModel, {"use_kg": True, "kg_pack_path": kg, "kg_layers": 1, "kg_cap": 32,
                                      "intent_learnable": False, "use_dynamic": True,
                                      "profile_attn": True, **meta_extra}),
        # CORONA stage 3 = Full(IKGR+DynLLM+CORONA): per-channel weighted-sum
        # late-fusion (CF / intent+meta-KG / recency) with learnable alpha,beta,gamma.
        "IKGR_full": (IKGRModel, {"use_kg": True, "kg_pack_path": kg, "kg_layers": 1, "kg_cap": 32,
                                  "intent_learnable": False, "use_dynamic": True,
                                  "use_corona": True, **meta_extra}),
        # CORONA 3-1: graph-neighbor candidate generation (eval-only top-M restriction).
        # Same trained model as IKGR_dyn, but ranks within a retrieved candidate set.
        "IKGR_cand": (IKGRModel, {"use_kg": True, "kg_pack_path": kg, "kg_layers": 1, "kg_cap": 32,
                                  "intent_learnable": False, "use_dynamic": True,
                                  "corona_cand": 500, "meta_kg_path": meta_path, **meta_extra}),
        # CORONA 3-1 de-biased: drop popularity-biased CF channel, IDF-weight
        # ubiquitous KG/meta nodes, and divide candidate scores by item_pop^0.5
        # to push long-tail/niche items into the candidate set (diversity-first).
        "IKGR_cand_db": (IKGRModel, {"use_kg": True, "kg_pack_path": kg, "kg_layers": 1, "kg_cap": 32,
                                     "intent_learnable": False, "use_dynamic": True,
                                     "corona_cand": 500, "meta_kg_path": meta_path,
                                     "corona_cf": False, "corona_idf": True,
                                     "corona_popnorm": 0.5, **meta_extra}),
        # Soft CORONA reranking: preserve full-sort recall, then add a bounded
        # de-biased graph prior. All lambdas below are evaluated after ONE
        # training run per seed, so the sweep has no extra GPU training cost.
        "IKGR_rerank_db_rel": (IKGRModel, {"use_kg": True, "kg_pack_path": kg, "kg_layers": 1, "kg_cap": 32,
                                           "intent_learnable": False, "use_dynamic": True,
                                           "meta_kg_path": meta_path,
                                           "corona_rerank_grid": _float_grid_from_env(
                                               "IKGR_CORONA_RERANK_GRID", [0.0, 0.1, 0.25, 0.5]),
                                           "corona_cf": False, "corona_idf": True,
                                           "corona_popnorm": 0.5, **meta_extra}),
        "BPR":          ("BPR", {}),
        "LightGCN":     ("LightGCN", {}),
    }
    pipe = cfg.get("pipeline", {})
    seeds = [int(s) for s in os.environ.get("IKGR_SEEDS", str(pipe.get("seeds", "2020"))).split(",")]
    default_specs = str(pipe.get("specs", "IKGR_kgoff,IKGR_kgon_L1_frozen,BPR,LightGCN"))
    spec_names = os.environ.get("IKGR_SPECS", default_specs).split(",")

    split = os.environ.get("IKGR_SPLIT", str(pipe.get("split", "RS"))).upper()
    os.environ.setdefault("IKGR_SPLIT", split)
    wd = paths.get("workdir", "run/")
    os.makedirs(wd, exist_ok=True)
    fname = "slice_eval_result.json" if split == "RS" else f"slice_eval_{split}_result.json"
    out_path = os.path.join(wd, fname)
    results = json.load(open(out_path, encoding="utf-8")) if os.path.exists(out_path) else {}

    for name in spec_names:
        model_arg, extra = all_specs[name]
        signature, context = _experiment_context(name, model_arg, extra, rb, paths, split)
        rerank_lambdas = [float(x) for x in extra.get("corona_rerank_grid", [])]
        result_names = ([f"{name}_l{str(lam).replace('.', 'p')}" for lam in rerank_lambdas]
                        if rerank_lambdas else [name])
        for result_name in result_names:
            existing = results.get(result_name)
            if not existing or existing.get("run_signature") != signature:
                if existing:
                    print(f"[invalidate] {result_name}: evaluation inputs changed", flush=True)
                results[result_name] = {"run_signature": signature, "run_context": context, "seeds": {}}
        for seed in seeds:
            if all(str(seed) in results[result_name]["seeds"] for result_name in result_names):
                print(f"[skip] {name} seed={seed}", flush=True); continue
            print(f"\n===== {name} seed={seed} =====", flush=True)
            pu_by_variant, pop, tsec, ni, cstats = _train_and_collect(model_arg, extra, rb, paths, seed)
            for lam, result_name in zip(([None] if not rerank_lambdas else rerank_lambdas), result_names):
                rep = slice_report(pu_by_variant[lam], pop, ni); rep["train_sec"] = tsec
                if cstats:
                    rep.update(cstats)
                results[result_name]["seeds"][str(seed)] = rep
                results[result_name]["agg"] = _aggregate(results[result_name]["seeds"])
                print(f"[{result_name} s{seed}] overall_ndcg@10={rep['overall']['ndcg@10']} "
                      f"tail_recall@10={rep['long_tail']['recall@10']} cov@10={rep['coverage@10']} "
                      f"({tsec}s)", flush=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nALL DONE")


if __name__ == "__main__":
    main()
