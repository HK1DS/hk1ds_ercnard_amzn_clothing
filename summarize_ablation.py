"""Generate the standalone Markdown table and edge-weight chart."""
import json, os

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(ROOT, "run", "amazon_clothing_ablation")


def usage(path, phases=None):
    raw = {"prompt": 0, "completion": 0, "total": 0, "calls": 0}; unique = {}
    if not os.path.exists(path): return raw, raw.copy()
    for line in open(path, encoding="utf-8"):
        r = json.loads(line); phase = r.get("phase")
        if phases and phase not in phases: continue
        u = r.get("usage") or {}; raw["calls"] += 1
        for k, src in (("prompt", "prompt_tokens"), ("completion", "completion_tokens"), ("total", "total_tokens")):
            raw[k] += int(u.get(src) or 0)
        unique[(phase, r.get("kind"), r.get("profile_sha256"))] = u
    dedup = {"calls": len(unique), "prompt": 0, "completion": 0, "total": 0}
    for u in unique.values():
        for k, src in (("prompt", "prompt_tokens"), ("completion", "completion_tokens"), ("total", "total_tokens")):
            dedup[k] += int(u.get(src) or 0)
    return raw, dedup


def result(path, name):
    x = json.load(open(path, encoding="utf-8"))[name]["agg"]
    return x["overall.ndcg@10"], x["overall.recall@10"]


legacy_path = os.path.join(ROOT, "run", "amazon_clothing_to", "slice_eval_TO_result.json")
unified_path = os.path.join(OUTDIR, "unified", "slice_eval_TO_result.json")
old_ndcg, old_recall = result(legacy_path, "IKGR_full_hetero")
weighted_ndcg, weighted_recall = result(legacy_path, "IKGR_full_hetero_weighted")
uni_ndcg, uni_recall = result(unified_path, "IKGR_full_hetero")
times = json.load(open(os.path.join(OUTDIR, "timings.json"), encoding="utf-8"))
legacy_raw, legacy_unique = usage(os.path.join(ROOT, "run", "amazon_clothing_to", "llm_usage.jsonl"), {"step1", "step2"})
unified_raw, unified_unique = usage(os.path.join(OUTDIR, "unified", "llm_usage_unified.jsonl"), {"unified"})

labels = ["Uniform", "IDF weighted"]
means = [old_ndcg["mean"], weighted_ndcg["mean"]]; stds = [old_ndcg["std"], weighted_ndcg["std"]]
chart = os.path.join(OUTDIR, "edge_weight_ndcg.svg")
scale, baseline = 7000, 330
bars = []
for i, (label, mean, sd, color) in enumerate(zip(labels, means, stds, ["#8aa4c8", "#e59a62"])):
    x = 150 + i * 250; h = mean * scale; y = baseline - h; err = sd * scale
    bars.append(f'<rect x="{x}" y="{y:.1f}" width="130" height="{h:.1f}" fill="{color}"/>')
    bars.append(f'<line x1="{x+65}" y1="{y-err:.1f}" x2="{x+65}" y2="{y+err:.1f}" stroke="#222" stroke-width="3"/>')
    bars.append(f'<line x1="{x+50}" y1="{y-err:.1f}" x2="{x+80}" y2="{y-err:.1f}" stroke="#222" stroke-width="3"/>')
    bars.append(f'<text x="{x+65}" y="{y-err-10:.1f}" text-anchor="middle" font-size="18">{mean:.4f} ± {sd:.4f}</text>')
    bars.append(f'<text x="{x+65}" y="360" text-anchor="middle" font-size="18">{label}</text>')
svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="650" height="420" viewBox="0 0 650 420">
<rect width="100%" height="100%" fill="white"/><text x="325" y="32" text-anchor="middle" font-size="22" font-weight="bold">Intent Edge Weight Ablation (TO, 3 seeds)</text>
<line x1="90" y1="330" x2="590" y2="330" stroke="#222"/><line x1="90" y1="55" x2="90" y2="330" stroke="#222"/>
<text x="25" y="205" transform="rotate(-90 25 205)" text-anchor="middle" font-size="17">NDCG@10</text>{''.join(bars)}</svg>'''
with open(chart, "w", encoding="utf-8") as f: f.write(svg)

def metric(m): return f"{m['mean']:.4f} ± {m['std']:.4f}"
def pct(a, b): return "n/a" if not a else f"{(b-a)/a*100:+.2f}%"
md = f"""# Amazon Clothing Ablation Results

기준: TO split, epochs=12, seeds=2020/2021/2022. 기존 방식은 실제 구현에 맞춰 **LLM exact 추출 → LLM related 확장 → 그래프 생성**의 3단계 파이프라인으로 정의했다. 통합 방식은 exact/related를 한 API 호출에서 함께 생성한다.

## 효율성 및 정확도 비교

| 방식 | 실제 API 호출 | API 토큰(raw billable) | 중복 제거 토큰 | 그래프 생성 시간 | NDCG@10 | Recall@10 |
|---|---:|---:|---:|---:|---:|---:|
| 기존 3단계 | {legacy_raw['calls']:,} | {legacy_raw['total']:,} | {legacy_unique['total']:,} | {times['legacy_graph']:.2f}s | {metric(old_ndcg)} | {metric(old_recall)} |
| 통합 1단계 | {unified_raw['calls']:,} | {unified_raw['total']:,} | {unified_unique['total']:,} | {times['unified_graph']:.2f}s | {metric(uni_ndcg)} | {metric(uni_recall)} |

Raw billable은 재시도/중단 후 재호출까지 포함한 ledger 합계이고, 중복 제거 값은 `(phase, kind, profile hash)`별 마지막 성공 호출을 합산했다.

## Intent Edge Weight 검증

수식: `raw_w(e,i) = log((N_e + 1) / (df(i) + 1)) + 1`, `w(e,i) = raw_w(e,i) / Σ_j raw_w(e,j)`.

| 모델 | NDCG@10 | Recall@10 | NDCG Std | Recall Std | NDCG 변화 | NDCG Std 변화 |
|---|---:|---:|---:|---:|---:|---:|
| IKGR_full_hetero (uniform) | {old_ndcg['mean']:.4f} | {old_recall['mean']:.4f} | {old_ndcg['std']:.4f} | {old_recall['std']:.4f} | 기준 | 기준 |
| IKGR_full_hetero_weighted (IDF) | {weighted_ndcg['mean']:.4f} | {weighted_recall['mean']:.4f} | {weighted_ndcg['std']:.4f} | {weighted_recall['std']:.4f} | {pct(old_ndcg['mean'], weighted_ndcg['mean'])} | {pct(old_ndcg['std'], weighted_ndcg['std'])} |

![Intent edge-weight NDCG comparison](run/amazon_clothing_ablation/edge_weight_ndcg.svg)

‘분산 감소’와 ‘NDCG 방어’는 가정하지 않고 위 실측 변화의 부호로 판정한다. Std는 현재 평가 코드와 동일한 population standard deviation (`numpy.std`, ddof=0)이다.
"""
with open(os.path.join(ROOT, "AMAZON_CLOTHING_ABLATION_RESULTS.md"), "w", encoding="utf-8") as f: f.write(md)
print("wrote AMAZON_CLOTHING_ABLATION_RESULTS.md and", chart)
