# Amazon Clothing 데이터를 추천 파이프라인에 넣는 방법

이 문서는 Amazon Review Data 2018의 **Clothing, Shoes and Jewelry (5-core)** 데이터를 실제 순차 추천 파이프라인에 넣기 위한 최소 계약(contract)을 정리한다. 이 프로젝트의 EDA와 같은 원천 데이터를 사용하되, 대용량 JSON을 한 번에 메모리에 올리지 않고 스트리밍으로 Parquet으로 변환한다.

## 1. 원천 파일

두 파일이 모두 필요하다.

|용도|URL|핵심 필드|
|---|---|---|
|사용자 상호작용|`https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Clothing_Shoes_and_Jewelry_5.json.gz`|`reviewerID`, `asin`, `overall`, `unixReviewTime`|
|상품 메타데이터|`https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Clothing_Shoes_and_Jewelry.json.gz`|`asin`, `brand`, `category`, `title`|

`5-core`는 이미 상호작용이 너무 적은 사용자/상품을 제거한 공개 데이터셋 버전이다. 리뷰 행을 명시적 평점으로 쓰거나, 추천 학습에서는 보통 **암묵적 positive interaction**으로 취급한다. 이 파일에는 실제 주문/구매 여부가 아니라 *리뷰 이벤트*가 들어 있다는 점을 명시해야 한다.

권장 디렉터리 구조:

```text
data/amazon_clothing/
├── raw/
│   ├── Clothing_Shoes_and_Jewelry_5.json.gz
│   └── meta_Clothing_Shoes_and_Jewelry.json.gz
└── processed/
    ├── interactions.parquet
    ├── items.parquet
    └── sequences.parquet
```

## 2. 로드 및 정규화 코드

아래 코드는 `pandas`, `pyarrow`만 있으면 실행된다. 원본 JSON은 한 줄에 한 레코드(JSON Lines)이므로 `gzip.open()`으로 한 행씩 읽는다. 대형 리스트에 전부 쌓지 않는 것이 중요하다.

```python
from __future__ import annotations

import gzip
import json
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

ROOT = Path("data/amazon_clothing")
RAW = ROOT / "raw"
OUT = ROOT / "processed"
RAW.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

URLS = {
    "reviews": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Clothing_Shoes_and_Jewelry_5.json.gz",
    "metadata": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Clothing_Shoes_and_Jewelry.json.gz",
}
PATHS = {
    "reviews": RAW / "Clothing_Shoes_and_Jewelry_5.json.gz",
    "metadata": RAW / "meta_Clothing_Shoes_and_Jewelry.json.gz",
}

for name, url in URLS.items():
    if not PATHS[name].exists():
        urlretrieve(url, PATHS[name])


def jsonl_gz(path: Path):
    """gzip JSONL 파일을 한 레코드씩 반환한다."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def category_level_2(categories: object) -> str | None:
    """예: [root, Women, Clothing, ...] -> 'Clothing'.

    EDA와 동일하게 level 2를 쓴다. leaf 카테고리는 소재/기능 같은
    상품 속성이 섞여 카테고리 label로 부적합할 수 있다.
    """
    if isinstance(categories, list) and len(categories) > 2:
        value = str(categories[2]).strip()
        return value or None
    return None


# item_id, brand, category를 만들고 ASIN 중복을 제거한다.
items = []
for row in jsonl_gz(PATHS["metadata"]):
    item_id = row.get("asin")
    brand = str(row.get("brand") or "").strip()
    category = category_level_2(row.get("category"))
    if item_id and brand and category:
        items.append({
            "item_id": item_id,
            "brand": brand,
            "category": category,
            "title": str(row.get("title") or "").strip() or None,
        })

items = pd.DataFrame(items).drop_duplicates("item_id", keep="first")

# 학습에서 필요한 상호작용 열만 유지한다.
interactions = []
for row in jsonl_gz(PATHS["reviews"]):
    user_id = row.get("reviewerID")
    item_id = row.get("asin")
    timestamp = row.get("unixReviewTime")
    if user_id and item_id and timestamp is not None:
        interactions.append({
            "user_id": user_id,
            "item_id": item_id,
            "timestamp": int(timestamp),       # UTC epoch seconds
            "rating": float(row.get("overall", 0.0)),
            "verified": bool(row.get("verified", False)),
        })

interactions = pd.DataFrame(interactions)

# inner join: 브랜드/카테고리 메타데이터가 있는 상품만 남긴다.
# KG/attribute-aware 파이프라인에 필요한 정책이다.
events = interactions.merge(items, on="item_id", how="inner")
events = events.sort_values(["user_id", "timestamp", "item_id"], kind="stable").reset_index(drop=True)

items.to_parquet(OUT / "items.parquet", index=False)
events.to_parquet(OUT / "interactions.parquet", index=False)
print(events.shape, events[["user_id", "item_id", "timestamp", "brand", "category"]].head())
```

> 메모리가 제한된 서버에서는 위의 `items`/`interactions` 리스트도 청크 단위로 Parquet에 기록하도록 바꾼다. 다만 이 데이터는 정제 후 약 613만 행 규모이므로, 16GB 이상 RAM이면 위 방식도 보통 다룰 수 있다.

## 3. 추천 모델용 입력 만들기

순차 추천의 한 이벤트는 다음처럼 고정한다.

```text
(user_id, item_id, timestamp, rating, brand, category)
```

기본 next-item 학습 샘플은 사용자별 시간순 시퀀스에서 만든다.

```python
events = pd.read_parquet(OUT / "interactions.parquet")

# 최소 3개 이벤트 사용자만 남기면 train/validation/test를 만들 수 있다.
events = events[events.groupby("user_id")["item_id"].transform("size") >= 3].copy()

# 마지막 1개 = test, 끝에서 두 번째 = validation, 나머지 = train.
events["position"] = events.groupby("user_id").cumcount()
events["n_events"] = events.groupby("user_id")["item_id"].transform("size")
events["split"] = "train"
events.loc[events["position"] == events["n_events"] - 2, "split"] = "valid"
events.loc[events["position"] == events["n_events"] - 1, "split"] = "test"

sequences = (
    events.groupby("user_id", sort=False)
    .agg(
        item_ids=("item_id", list),
        timestamps=("timestamp", list),
        brands=("brand", list),
        categories=("category", list),
        splits=("split", list),
    )
    .reset_index()
)
sequences.to_parquet(OUT / "sequences.parquet", index=False)
```

모델 입력 ID는 반드시 **train split으로만** vocabulary를 만들고, validation/test에서 처음 등장한 ID는 `UNK`로 매핑한다. 이는 미래 데이터 누수를 막는다. 학습용 pair는 한 사용자 시퀀스 `i1, i2, ..., in`에서 `(i1 -> i2), (i1,i2 -> i3), ...` 식으로 생성한다.

## 4. 이 프로젝트의 attribute-aware/KG 사용법

일반 next-item 모델에는 `user_id`, `item_id`, `timestamp`만 사용해도 된다. 이 프로젝트의 가설을 반영하려면 상품 메타데이터를 함께 사용한다.

```text
user --reviewed_at--> item
item --has_brand-----> brand
item --in_category---> category
```

- **카테고리**: 직전 이벤트와 다음 이벤트가 다른지를 판단하거나, 후보군 다양성 제어에 사용한다.
- **브랜드**: 카테고리가 바뀌어도 유지될 수 있는 attribute bridge로 사용한다. 즉, 직전 브랜드와 같은 후보에는 별도의 feature/edge를 제공하되, 이를 정답 규칙으로 고정하지는 않는다.
- **평점**: 암묵적 데이터로 쓸 경우 모든 리뷰를 positive로 두거나, `rating >= 4`만 positive로 두는 두 정책을 분리 실험한다. 낮은 평점을 제거하면 표본과 행동 의미가 달라진다.

## 5. 필수 검증 항목

변환 직후 아래를 로그로 남긴다.

```python
assert events["user_id"].notna().all()
assert events["item_id"].notna().all()
assert events["timestamp"].notna().all()
assert events["brand"].notna().all()
assert events["category"].notna().all()
assert events.equals(events.sort_values(["user_id", "timestamp", "item_id"], kind="stable").reset_index(drop=True))

print("events:", len(events))
print("users:", events["user_id"].nunique())
print("items:", events["item_id"].nunique())
print("brands:", events["brand"].nunique())
print("categories:", events["category"].nunique())
print("time:", pd.to_datetime(events["timestamp"], unit="s", utc=True).agg(["min", "max"]))
```

EDA 기준으로 메타데이터 병합 뒤 브랜드와 Level-2 카테고리가 모두 있는 행만 남기면 약 **613만 이벤트, 115만 사용자, 24만 상품**이 남는다. 원본 리뷰 약 1,129만 건보다 많이 줄어드는 주된 원인은 브랜드가 비어 있는 상품이 많기 때문이다. 메타데이터 없는 상품까지 포함한 순수 collaborative-filtering 베이스라인을 만들고 싶다면, `items`와의 `inner join` 대신 `left join`을 사용하고 `brand/category = UNK`로 채우는 별도 버전을 유지한다.
