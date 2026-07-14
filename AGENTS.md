# Agent Handoff Notes

이 문서는 새 대화창/새 에이전트가 이 레포에서 바로 이어 작업할 수 있도록 만든 작업 인수인계 문서입니다. 핵심은 **Goodreads에서 이미 겪은 실험 경험을 기억하되, 현재 루트 레포는 특정 데이터셋 실험 결과가 아니라 재사용 가능한 vanilla pipeline**이라는 점입니다.

## 1. Current Repository State

- 현재 루트는 `IKGR -> DynLLM -> CORONA` 전체 실험용 **dataset-agnostic vanilla pipeline**입니다.
- Goodreads 실험 산출물과 `legacy/`는 삭제되었습니다. 이 레포에는 대용량 데이터, run artifact, API key가 들어가면 안 됩니다.
- Git remote는 마지막 작업 기준으로 `https://github.com/HK1DS/hk1ds_jwkim_movielens.git`에 연결되어 있었습니다.
- `data/`, `run/`, `.env`, `.venv`, `legacy/`는 ignore 대상입니다.
- 새 에이전트는 사용자가 명시하지 않는 한 임의로 remote를 바꾸거나 push하지 마세요.

## 2. Pipeline Shape

입력 파일은 데이터셋별로 준비합니다.

- `profiles.csv`: `user_id,user_profile,item_id,item_profile`
- `interactions.csv`: `user_id,item_id,rating` 필수, `timestamp` 선택
- `item_metadata.csv`: 선택. `item_id` 필수, `brand/category/attribute/author/publisher` 계열 열은 `config.yaml`의 `metadata` 섹션에서 매핑

주요 단계:

1. `step1.py`: LLM으로 exact intent 추출
2. `step2.py`: RAG + LLM으로 related intent 확장
3. `build_intent_banks.py`, `build_kg.py`: intent embedding bank와 intent KG 생성
4. `build_meta_kg.py`: 선택 metadata KG 생성
5. `convert_to_recbole_atomic.py`: RecBole `.inter/.kg` 생성
6. `eval_slices.py`: IKGR/DynLLM/CORONA/baseline ablation 평가

`run_pipeline.py`가 위 단계를 묶는 generic orchestrator입니다.

## 3. Important Runtime Rules

- LLM 호출이 들어가는 단계는 B/C입니다. 반드시 `.env`와 `config.yaml`의 provider/key 설정을 확인하세요.
- 캐시는 `run/step1_cache.json`, `run/step2_cache.json`에 저장됩니다. 중단/재개할 때 매우 중요합니다.
- `timestamp`가 없으면 `IKGR_dyn`, `IKGR_full`, `IKGR_cand*`, `IKGR_rerank*`처럼 recency를 쓰는 실험은 실패하거나 의미가 약합니다.
- metadata CSV가 없으면 metadata KG는 건너뜁니다. 현재 `eval_slices.py`는 meta pack이 없을 때 intent-only 쪽으로 내려가도록 완화되어 있습니다.
- Windows에서는 Annoy 네이티브 크래시가 있었기 때문에 `ikgr_core/rag.py`는 기본적으로 sklearn fallback 경로가 더 안전합니다. `IKGR_FORCE_ANNOY=1`은 신중히 사용하세요.
- RecBole 1.2.0과 최신 scipy 조합에서 `dok_matrix._update` 문제가 있어 관련 스크립트에 monkeypatch가 들어가 있습니다.

## 4. Goodreads Experiment Lessons

Goodreads Children k-core 실험에서 얻은 결론입니다. 새 데이터셋에서 그대로 성능을 보장한다는 뜻이 아니라, 같은 함정을 피하기 위한 경험치입니다.

### 4.1 Dense k-core에서는 CF baseline이 매우 강함

Goodreads k=100은 유저/아이템이 매우 dense했습니다. 이런 세팅에서는 MF/BPR/LightGCN 같은 협업필터링 baseline이 overall NDCG/Recall에서 강하고, intent/KG 계열이 overall 정확도 SOTA를 주장하기 어렵습니다.

따라서 새 데이터셋에서도 먼저 다음을 확인하세요.

- split이 dense한지 sparse한지
- train/test에 cold-start 유저/아이템이 실제로 존재하는지
- overall뿐 아니라 long-tail, coverage, novelty, cold-start slice를 같이 볼 수 있는지

### 4.2 Old IKGR scoring bug

Goodreads 초반 IKGR는 `intent max-cos heuristic + small embedding score` 구조라 고정 휴리스틱이 랭킹을 지배했습니다. top-k가 사실상 동점/랜덤처럼 되어 Pop보다 낮은 결과가 나왔습니다.

현재 루트의 `ikgr_core/model_ikgr.py`는 이 문제를 피하기 위해 다음 구조를 사용합니다.

- learnable user/item embedding
- intent embedding projection
- intent/meta KG propagation
- inner-product BPR ranking
- `use_kg=False`로 MF-style sanity check 가능

새 데이터셋에서도 `IKGR_kgoff`가 BPR 근처로 나오는지 먼저 sanity check 하세요.

### 4.3 Cold-start는 k-core에서 사라질 수 있음

Goodreads k=100에서는 cold-start를 보려고 했지만, k-core가 약한 유저를 제거해버려 실제 cold-start 검증이 어려웠습니다. cold-start 주장을 하려면 다음 중 하나가 필요합니다.

- 더 낮은 k-core, 예: k=20/30
- global temporal split, 예: `IKGR_SPLIT=TO_GLOBAL`
- 별도 cold-start holdout 구성

단순 per-user temporal split(`TO`)은 모든 유저가 train에 남는 경우가 많아서 genuine cold-start가 아닐 수 있습니다.

### 4.4 Metadata KG가 intent-only KG보다 안정적일 수 있음

Goodreads에서는 LLM intent-only KG가 long-tail 정확도를 robust하게 올리지는 못했습니다. 반면 author/publisher/shelf 같은 무료 metadata KG는 long-tail 쪽에서 더 안정적인 효과를 보였습니다.

새 데이터셋에서는 가능한 한 `item_metadata.csv`를 준비하세요. Amazon Clothing이라면 `brand`, `category`, `style`, `attribute`, `description-derived tags` 같은 값이 유용할 수 있습니다.

### 4.5 DynLLM은 recency가 가장 실용적인 1차 구현

Goodreads에서 DynLLM 전체 논문 구조를 그대로 구현하지는 않았습니다. 비용/시간/공개코드 한계 때문에 우선 recency-weighted dynamic profile로 검증했습니다.

결과적으로 recency는 overall 일부 회복 + long-tail 유지라는 modest한 긍정 효과가 있었습니다. 반면 multi-facet attention fusion은 tail/coverage를 깎아서 기각했습니다.

새 데이터셋에서도 1차 개선은 복잡한 attention보다 recency부터 보세요.

### 4.6 CORONA는 naive candidate generation이 인기 편향을 키울 수 있음

Goodreads에서 naive CORONA candidate restriction은 overall은 조금 올렸지만 tail/coverage를 크게 망가뜨렸습니다. 이유는 CF co-occurrence와 ubiquitous metadata node가 인기 아이템 후보를 과도하게 밀었기 때문입니다.

개선된 방향:

- CF candidate channel 끄기
- node IDF 적용
- item popularity normalization 적용
- full-sort를 완전히 자르기보다 soft graph-prior reranking 고려

즉 CORONA는 "정확도 항상 상승" 모듈이 아니라, 설정에 따라 diversity/long-tail trade-off가 강하게 나타나는 모듈입니다.

## 5. Recommended Experiment Order For New Dataset

처음부터 Full 모델을 돌리지 말고 아래 순서로 확인하세요.

1. 데이터 스키마 확인 및 sample run
2. `step1 -> step2` LLM 캐시 생성
3. `D` 단계로 banks/KG/RecBole 변환
4. `IKGR_kgoff`, `BPR`, `LightGCN` sanity baseline
5. `IKGR_kgon_L1_frozen` 또는 `IKGR_full_hetero`
6. timestamp가 있으면 `IKGR_dyn`
7. long-tail/coverage가 목적이면 `IKGR_cand_db` 또는 `IKGR_rerank_db_rel`

권장 예시:

```bash
python run_pipeline.py --steps BCDE
python run_pipeline.py --steps E --specs IKGR_kgoff,IKGR_kgon_L1_frozen,BPR,LightGCN
python run_pipeline.py --steps E --split TO --epochs 12 --seeds 2020,2021,2022 --specs IKGR_dyn,IKGR_cand_db,BPR,LightGCN
```

## 6. How To Interpret Results

Goodreads 경험상 이 파이프라인의 정직한 주장 방향은 다음에 가깝습니다.

- SOTA overall accuracy를 주장하기는 어렵습니다.
- 강점은 cold-start, sparse, long-tail, coverage, novelty slice에서 찾아야 합니다.
- dense dataset에서 overall이 MF/BPR/LightGCN보다 낮더라도 바로 실패는 아닙니다.
- 하지만 `IKGR_kgoff`가 BPR 근처에도 못 가면 구현/변환/split 문제를 먼저 의심해야 합니다.

리포트에는 전체 평균만 쓰지 말고 다음을 같이 남기세요.

- overall Recall/NDCG
- tail Recall@10/@30
- coverage@10
- novelty
- candidate recall@M, CORONA 사용 시
- seed mean/std
- split 방식과 timestamp 유무

## 7. Safety And Git Hygiene

- `.env`, `data/`, `run/`, model checkpoints, caches는 commit하지 마세요.
- 새 데이터셋용 repo에 push하기 전 `git status -sb`와 `.gitignore`를 확인하세요.
- 사용자가 "github 연결 끊어줘"라고 하면 `git remote remove origin`만 수행하고 파일은 건드리지 마세요.
- 사용자가 특정 repo를 주면 `origin`을 그 repo로 연결한 뒤 push하세요. 다른 remote가 남아 있으면 먼저 확인/제거하세요.
- 실험 결과를 README에 과장해서 쓰지 마세요. Goodreads에서 negative 결과가 많았고, 새 데이터셋에서는 반드시 재검증이 필요합니다.
