# Amazon Clothing Ablation Results

기준: TO split, epochs=12, seeds=2020/2021/2022. 기존 방식은 실제 구현에 맞춰 **LLM exact 추출 → LLM related 확장 → 그래프 생성**의 3단계 파이프라인으로 정의했다. 통합 방식은 exact/related를 한 API 호출에서 함께 생성한다.

## 효율성 및 정확도 비교

| 방식 | 실제 API 호출 | API 토큰(raw billable) | 중복 제거 토큰 | 그래프 생성 시간 | NDCG@10 | Recall@10 |
|---|---:|---:|---:|---:|---:|---:|
| 기존 3단계 | 41,656 | 29,129,669 | 28,844,821 | 721.27s | 0.0337 ± 0.0013 | 0.0666 ± 0.0013 |
| 통합 1단계 | 20,486 | 7,118,217 | 7,116,086 | 1323.44s | 0.0444 ± 0.0010 | 0.0782 ± 0.0015 |

Raw billable은 재시도/중단 후 재호출까지 포함한 ledger 합계이고, 중복 제거 값은 `(phase, kind, profile hash)`별 마지막 성공 호출을 합산했다.

## Intent Edge Weight 검증

수식: `raw_w(e,i) = log((N_e + 1) / (df(i) + 1)) + 1`, `w(e,i) = raw_w(e,i) / Σ_j raw_w(e,j)`.

| 모델 | NDCG@10 | Recall@10 | NDCG Std | Recall Std | NDCG 변화 | NDCG Std 변화 |
|---|---:|---:|---:|---:|---:|---:|
| IKGR_full_hetero (uniform) | 0.0337 | 0.0666 | 0.0013 | 0.0013 | 기준 | 기준 |
| IKGR_full_hetero_weighted (IDF) | 0.0361 | 0.0701 | 0.0018 | 0.0008 | +7.12% | +38.46% |

![Intent edge-weight NDCG comparison](run/amazon_clothing_ablation/edge_weight_ndcg.svg)

‘분산 감소’와 ‘NDCG 방어’는 가정하지 않고 위 실측 변화의 부호로 판정한다. Std는 현재 평가 코드와 동일한 population standard deviation (`numpy.std`, ddof=0)이다.
