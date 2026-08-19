# Amazon Clothing Primary Parity Experiment

## 1. 요약

Amazon Reviews 2018의 Clothing, Shoes and Jewelry 데이터를 사용해 vanilla
`IKGR -> DynLLM -> CORONA` 파이프라인을 `TO`와 `TO_GLOBAL` 두 protocol에서
평가했다. 모든 primary 결과는 epochs 12, embedding size 512, dropout 0.1,
seeds 2020/2021/2022로 실행했다. 표의 값은 3개 seed의 `mean ± population std
(ddof=0)`이다.

핵심 결과는 다음과 같다.

- `TO`에서는 `IKGR_full`이 overall NDCG@10 0.1781, Recall@10 0.2029,
  coverage@10 0.6383으로 가장 균형이 좋았다.
- `IKGR_kgoff`와 BPR는 `TO`에서 거의 같은 성능을 보여 learnable backbone
  sanity check를 통과했다.
- `IKGR_cand_db`는 `TO` tail Recall@10 0.0501, tail Recall@30 0.0734,
  novelty 13.1237로 long-tail/diversity에 가장 강했지만 overall NDCG는
  `IKGR_full`과 BPR보다 낮았다.
- `TO_GLOBAL`에서는 전체 정확도가 매우 낮아졌고, 4,765 test user 중
  `cold0_train` 721명이 평가에 포함됐다.
- `TO_GLOBAL` cold0에서는 LightGCN과 `IKGR_kgoff`가 강했고, hard CORONA
  candidate 모델은 cold0 성능을 크게 낮췄다. 따라서 CORONA가 일반적인
  cold-start 개선 모듈이라는 주장은 이 결과로 지지되지 않는다.
- intent+metadata를 단순 결합한 `IKGR_full_hetero`는 두 protocol에서 모두
  크게 부진했다. heterogeneous KG 결합 방식의 추가 검토가 필요하다.

## 2. 데이터 계약

| 항목 | 설정 |
|---|---|
| 원본 | Amazon Reviews 2018, Clothing, Shoes and Jewelry 5-core |
| 후보 user sampling | seed 2020 deterministic bottom Blake2b hash, 50,000명 |
| positive event | `rating >= 4` |
| deduplication | `(user, item, rating, timestamp, review_text)` exact duplicate 제거 |
| iterative k-core | user-k=5, item-k=3 |
| 사전 recency filter | 없음 (`max_age_days=0`) |
| events / users / items | 85,172 / 10,515 / 11,020 |
| rating 4 / 5 | 18,252 / 66,920 |
| metadata | 11,020 items 모두 매칭 |
| metadata 가정 | static catalog side information |

원본과 processed 파일의 SHA-256은
[`data_manifest.json`](data/amazon_clothing/reporting/data_manifest.json)에 기록했다.
실험 당시 기준 코드의 기존 HEAD는 `11bf614`였으며, 이 보고서의 실행에는 아직
commit되지 않은 leakage guard, protocol config, cache/watchdog 개선 코드도 포함된다.

## 3. Protocol과 profile audit

| 항목 | TO | TO_GLOBAL |
|---|---:|---:|
| split | per-user temporal 80/10/10 | global temporal 70/10/20 |
| profile scope | per-user train | exact global train |
| train / valid / test events | 62,796 / 11,188 / 11,188 | 59,621 / 8,517 / 17,034 |
| nonempty / empty profiles | 10,515 / 0 | 9,263 / 1,252 |
| evaluated test users | 10,515 | 4,765 |
| evaluated `cold0_train` users | - | 721 |
| profile time audit | PASS | PASS |

빈 global profile은 LLM에 보내지 않았고 zero intent로 유지했다. TO와
TO_GLOBAL의 user cache는 분리했으며, profile text가 동일한 static item cache만
TO에서 TO_GLOBAL로 재사용했다.

## 4. Primary 설정

| 영역 | 값 |
|---|---|
| LLM | Luxia bridge, `gpt-4o-mini`, temperature 0.2, top-p 0.95, max tokens 2048 |
| RAG | `sentence-transformers/all-mpnet-base-v2`, kNN=100 |
| RecBole | epochs=12, embedding size=512, dropout=0.1 |
| seeds | 2020, 2021, 2022 |
| metrics | Recall/NDCG@10, tail Recall@10/@30, coverage@10, novelty |
| CORONA candidate | fixed M=500 |
| rerank lambda | 0.00, 0.10, 0.25, 0.50 |
| hard train age filter | 0 days, 즉 비활성화 |

## 5. TO 결과

| Spec | NDCG@10 | Recall@10 | Tail R@10 | Tail R@30 | Coverage@10 | Novelty |
|---|---:|---:|---:|---:|---:|---:|
| IKGR_kgoff | 0.1744 ± 0.0012 | 0.1976 ± 0.0009 | 0.0440 ± 0.0009 | 0.0529 ± 0.0006 | 0.5757 ± 0.0053 | 11.4234 ± 0.0405 |
| BPR | 0.1744 ± 0.0011 | 0.1972 ± 0.0007 | 0.0442 ± 0.0002 | 0.0535 ± 0.0006 | 0.5722 ± 0.0023 | 11.4306 ± 0.0172 |
| LightGCN | 0.1575 ± 0.0010 | 0.1802 ± 0.0011 | 0.0292 ± 0.0003 | 0.0381 ± 0.0012 | 0.2159 ± 0.0014 | 10.1243 ± 0.0547 |
| IKGR_meta_only | 0.1359 ± 0.0028 | 0.1690 ± 0.0009 | 0.0437 ± 0.0008 | 0.0612 ± 0.0003 | 0.4255 ± 0.0036 | 11.6374 ± 0.0365 |
| IKGR_full_hetero | 0.0381 ± 0.0016 | 0.0739 ± 0.0031 | 0.0126 ± 0.0005 | 0.0200 ± 0.0008 | 0.2079 ± 0.0024 | 10.2742 ± 0.0259 |
| IKGR_dyn | 0.1287 ± 0.0007 | 0.1674 ± 0.0009 | 0.0351 ± 0.0019 | 0.0504 ± 0.0006 | 0.3681 ± 0.0020 | 11.7259 ± 0.0196 |
| **IKGR_full** | **0.1781 ± 0.0008** | **0.2029 ± 0.0005** | 0.0484 ± 0.0002 | 0.0586 ± 0.0003 | **0.6383 ± 0.0005** | 11.9374 ± 0.0103 |
| IKGR_cand | 0.1336 ± 0.0007 | 0.1750 ± 0.0014 | 0.0394 ± 0.0016 | 0.0587 ± 0.0005 | 0.4629 ± 0.0012 | 11.9383 ± 0.0131 |
| **IKGR_cand_db** | 0.1530 ± 0.0022 | 0.1933 ± 0.0025 | **0.0501 ± 0.0012** | **0.0734 ± 0.0006** | 0.6198 ± 0.0031 | **13.1237 ± 0.0005** |
| Rerank λ=0.00 | 0.1287 ± 0.0007 | 0.1674 ± 0.0009 | 0.0351 ± 0.0019 | 0.0504 ± 0.0006 | 0.3681 ± 0.0020 | 11.7259 ± 0.0196 |
| Rerank λ=0.10 | 0.1305 ± 0.0007 | 0.1695 ± 0.0011 | 0.0357 ± 0.0019 | 0.0515 ± 0.0007 | 0.3721 ± 0.0020 | 11.7523 ± 0.0185 |
| Rerank λ=0.25 | 0.1328 ± 0.0009 | 0.1720 ± 0.0010 | 0.0367 ± 0.0018 | 0.0529 ± 0.0005 | 0.3796 ± 0.0022 | 11.7917 ± 0.0179 |
| Rerank λ=0.50 | 0.1372 ± 0.0011 | 0.1774 ± 0.0008 | 0.0402 ± 0.0015 | 0.0549 ± 0.0005 | 0.3935 ± 0.0018 | 11.8584 ± 0.0169 |

### TO 해석

- `IKGR_kgoff`와 BPR의 차이는 NDCG@10에서 사실상 0이고 Recall@10도
  0.0004에 불과하다. ID mapping과 learnable backbone이 정상이라는 강한 sanity
  signal이다.
- `IKGR_full`은 BPR 대비 NDCG@10 +0.0037, Recall@10 +0.0057,
  coverage@10 +0.0661을 보였다. 이 데이터의 warm temporal protocol에서는
  overall과 coverage가 함께 개선됐다.
- `IKGR_cand_db`는 BPR보다 overall NDCG가 낮지만 tail Recall@30,
  coverage, novelty가 높다. de-biased hard candidate는 accuracy와 diversity의
  trade-off로 해석해야 한다.
- rerank lambda가 커질수록 모든 보고 지표가 점진적으로 개선됐지만 λ=0.50도
  `IKGR_full`의 overall에는 미치지 못했다.

## 6. TO_GLOBAL 결과

| Spec | NDCG@10 | Recall@10 | Tail R@10 | Tail R@30 | Coverage@10 | Novelty |
|---|---:|---:|---:|---:|---:|---:|
| IKGR_kgoff | 0.0103 ± 0.0004 | 0.0100 ± 0.0004 | 0.0033 ± 0.0007 | 0.0054 ± 0.0011 | 0.3088 ± 0.1283 | 11.0830 ± 0.0876 |
| BPR | 0.0090 ± 0.0004 | 0.0099 ± 0.0006 | 0.0042 ± 0.0005 | 0.0068 ± 0.0009 | 0.4830 ± 0.0033 | 11.1759 ± 0.0289 |
| LightGCN | 0.0101 ± 0.0008 | 0.0090 ± 0.0006 | 0.0025 ± 0.0004 | 0.0045 ± 0.0011 | 0.1598 ± 0.0113 | 10.8962 ± 0.0215 |
| IKGR_meta_only | 0.0085 ± 0.0003 | 0.0098 ± 0.0004 | 0.0062 ± 0.0005 | 0.0123 ± 0.0003 | 0.3768 ± 0.0239 | 11.1973 ± 0.0524 |
| IKGR_full_hetero | 0.0063 ± 0.0004 | 0.0080 ± 0.0004 | 0.0017 ± 0.0001 | 0.0048 ± 0.0003 | 0.3096 ± 0.0024 | 11.0934 ± 0.0243 |
| IKGR_dyn | 0.0098 ± 0.0004 | 0.0123 ± 0.0002 | 0.0050 ± 0.0004 | 0.0104 ± 0.0002 | 0.4140 ± 0.0021 | 11.4710 ± 0.0273 |
| IKGR_full | 0.0091 ± 0.0002 | 0.0106 ± 0.0005 | 0.0047 ± 0.0004 | 0.0087 ± 0.0007 | **0.5581 ± 0.0044** | 11.3730 ± 0.0168 |
| IKGR_cand | 0.0100 ± 0.0002 | **0.0128 ± 0.0004** | 0.0053 ± 0.0002 | 0.0116 ± 0.0005 | 0.4160 ± 0.0020 | 11.5890 ± 0.0296 |
| **IKGR_cand_db** | 0.0080 ± 0.0001 | 0.0106 ± 0.0003 | **0.0070 ± 0.0003** | **0.0162 ± 0.0004** | 0.5504 ± 0.0004 | **11.7427 ± 0.0332** |
| Rerank λ=0.00 | 0.0098 ± 0.0004 | 0.0123 ± 0.0002 | 0.0050 ± 0.0004 | 0.0104 ± 0.0002 | 0.4140 ± 0.0021 | 11.4710 ± 0.0273 |
| Rerank λ=0.10 | 0.0098 ± 0.0004 | 0.0123 ± 0.0001 | 0.0050 ± 0.0004 | 0.0104 ± 0.0002 | 0.4165 ± 0.0021 | 11.4789 ± 0.0262 |
| Rerank λ=0.25 | 0.0098 ± 0.0004 | 0.0123 ± 0.0003 | 0.0051 ± 0.0003 | 0.0107 ± 0.0001 | 0.4207 ± 0.0026 | 11.4922 ± 0.0299 |
| Rerank λ=0.50 | 0.0100 ± 0.0004 | **0.0128 ± 0.0006** | 0.0056 ± 0.0005 | 0.0114 ± 0.0002 | 0.4292 ± 0.0028 | 11.5097 ± 0.0250 |

## 7. TO_GLOBAL cold0_train 결과

`cold0_train`은 global train interaction이 0개이고 test에 등장한 721명이다.

| Spec | Cold NDCG@10 | Cold Recall@10 | Cold Recall@30 |
|---|---:|---:|---:|
| IKGR_kgoff | 0.0254 ± 0.0039 | 0.0220 ± 0.0024 | 0.0436 ± 0.0050 |
| BPR | 0.0129 ± 0.0026 | 0.0130 ± 0.0040 | 0.0297 ± 0.0037 |
| **LightGCN** | **0.0301 ± 0.0083** | **0.0245 ± 0.0062** | **0.0528 ± 0.0069** |
| IKGR_meta_only | 0.0100 ± 0.0046 | 0.0093 ± 0.0039 | 0.0193 ± 0.0066 |
| IKGR_full_hetero | 0.0048 ± 0.0013 | 0.0047 ± 0.0014 | 0.0095 ± 0.0018 |
| IKGR_dyn | 0.0041 ± 0.0012 | 0.0035 ± 0.0010 | 0.0067 ± 0.0010 |
| IKGR_full | 0.0074 ± 0.0013 | 0.0073 ± 0.0009 | 0.0164 ± 0.0019 |
| IKGR_cand | 0.0023 ± 0.0004 | 0.0020 ± 0.0004 | 0.0025 ± 0.0006 |
| IKGR_cand_db | 0.0023 ± 0.0004 | 0.0020 ± 0.0004 | 0.0025 ± 0.0006 |
| Rerank λ=0.00 | 0.0041 ± 0.0012 | 0.0035 ± 0.0010 | 0.0067 ± 0.0010 |
| Rerank λ=0.10 | 0.0041 ± 0.0012 | 0.0035 ± 0.0010 | 0.0067 ± 0.0010 |
| Rerank λ=0.25 | 0.0041 ± 0.0012 | 0.0035 ± 0.0010 | 0.0067 ± 0.0010 |
| Rerank λ=0.50 | 0.0041 ± 0.0012 | 0.0035 ± 0.0010 | 0.0067 ± 0.0010 |

### Cold-start 해석

- 이 실험에서 intent/KG/DynLLM 계열이 genuine cold0 성능을 개선했다는 증거는
  없다. 오히려 `IKGR_kgoff`와 LightGCN이 강했다.
- cold0 user는 train history가 없으므로 recency profile이 비어 있다. `IKGR_dyn`과
  rerank의 cold0 결과가 동일한 것도 이 조건과 일치한다.
- hard candidate restriction은 cold0 Recall@30을 0.0025까지 낮췄다. graph evidence가
  부족한 user에게 candidate restriction을 적용할 때 fallback이 필요하다.

## 8. CORONA candidate ceiling

| Protocol | Spec | Candidate M | Candidate Recall@M |
|---|---|---:|---:|
| TO | IKGR_cand | 500 | 0.3686 ± 0.0000 |
| TO | IKGR_cand_db | 500 | 0.3555 ± 0.0000 |
| TO_GLOBAL | IKGR_cand | 500 | 0.1289 ± 0.0000 |
| TO_GLOBAL | IKGR_cand_db | 500 | 0.1318 ± 0.0000 |

Catalog가 11,020개이므로 M=500은 no-op이 아니다. 특히 TO_GLOBAL candidate
recall이 약 0.13에 불과해 hard restriction의 성능 상한이 매우 낮다. de-biasing은
tail/coverage/novelty를 올리지만 candidate recall 자체를 의미 있게 회복하지 못했다.

## 9. 결론

이 결과에서 방어 가능한 주장은 다음과 같다.

1. warm-user temporal 환경에서는 `IKGR_full`이 BPR 대비 작은 overall 개선과 큰
   coverage 개선을 동시에 보였다.
2. de-biased CORONA candidate는 overall accuracy 우위가 아니라 long-tail,
   coverage, novelty 개선용 trade-off 모듈이다.
3. global temporal cold-start는 훨씬 어려웠으며, intent/KG 계열의 보편적 cold-start
   우위는 확인되지 않았다.
4. `IKGR_full_hetero`의 큰 성능 저하는 단순 heterogeneous KG 결합이 안전하지 않다는
   negative result다.
5. soft rerank는 lambda 증가에 따라 점진적으로 좋아졌지만, 현재 grid에서는
   `IKGR_full`을 대체하지 못했다.

따라서 Amazon Clothing 결과는 “IKGR 계열이 모든 지표에서 baseline을 이긴다”가
아니라, **warm temporal에서는 full fusion이 정확도와 coverage에 유효하고,
de-biased candidate는 tail/diversity와 overall 사이의 trade-off를 만들며,
genuine cold0에서는 별도 fallback 설계가 필요하다**는 결론으로 해석해야 한다.

## 10. 산출물

- TO config: [`config.amazon_clothing.to.yaml`](config.amazon_clothing.to.yaml)
- TO_GLOBAL config: [`config.amazon_clothing.global.yaml`](config.amazon_clothing.global.yaml)
- TO profile manifest: [`profile_manifest.json`](data/amazon_clothing/reporting/to/profile_manifest.json)
- TO_GLOBAL profile manifest: [`profile_manifest.json`](data/amazon_clothing/reporting/global/profile_manifest.json)
- TO result JSON: [`slice_eval_TO_result.json`](run/amazon_clothing_to/slice_eval_TO_result.json)
- TO_GLOBAL result JSON: [`slice_eval_TO_GLOBAL_result.json`](run/amazon_clothing_global/slice_eval_TO_GLOBAL_result.json)
- Supervisor completion: [`primary_supervisor.status.json`](run/watchdog/primary_supervisor.status.json)

