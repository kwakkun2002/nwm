AI 프로젝트에서 더 중요한 축

AI 학습 코드베이스는 보통 이 4축으로 보는 게 맞다.

1. 데이터 축

데이터를 어떻게 만들고 변환하냐

raw input
parsing
cleaning
filtering
feature extraction
dataset
batching
2. 모델 축

모델을 어떻게 정의하고 조립하냐

backbone
head
loss
tokenizer / encoder
adapter
3. 실행 축

실험을 어떻게 돌리냐

config
trainer
optimizer
scheduler
checkpoint
logger
4. 평가 축

결과를 어떻게 해석하냐

metrics
validator
benchmark
report
visualization

즉 AI 프로젝트는
“데이터-모델-실행-평가” 구조가 더 본질적이야.


그래서 내가 추천하는 구조
추천안 A. 연구/실험형 프로젝트

논문 구현, 모델 실험, 빠른 iteration이 중요할 때

project/
├── configs/
│   ├── data/
│   ├── model/
│   ├── train/
│   └── experiment/
├── scripts/
│   ├── preprocess.py
│   ├── train.py
│   ├── eval.py
│   └── infer.py
├── src/
│   ├── core/
│   │   ├── types.py
│   │   ├── constants.py
│   │   ├── paths.py
│   │   └── seed.py
│   ├── data/
│   │   ├── sources/
│   │   ├── preprocessing/
│   │   ├── datasets/
│   │   ├── collators/
│   │   └── samplers/
│   ├── features/
│   │   ├── text/
│   │   ├── image/
│   │   └── multimodal/
│   ├── models/
│   │   ├── backbones/
│   │   ├── heads/
│   │   ├── losses/
│   │   └── wrappers/
│   ├── training/
│   │   ├── engine.py
│   │   ├── trainer.py
│   │   ├── optimizers.py
│   │   └── schedulers.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── validators.py
│   │   └── benchmark.py
│   └── utils/
├── experiments/
│   ├── exp001_baseline/
│   ├── exp002_aug/
│   └── exp003_new_loss/
├── tests/
└── notebooks/

이 구조의 장점은 명확해:

전처리와 학습이 pipeline 관점으로 분리됨
실험 config와 코드가 덜 섞임
baseline과 variant 비교가 쉬움
“어디를 바꾸면 결과가 달라지는가”를 추적하기 쉬움


각 디렉토리의 역할을 딱 잘라 말하면
configs/

이건 엄청 중요하다.

여기에 들어갈 것:

dataset 설정
model 하이퍼파라미터
train 설정
optimizer/scheduler 설정
experiment override

중요한 건
설정이 코드 내부에 숨어 있으면 안 된다는 거야.

AI 프로젝트가 망하는 제일 흔한 이유 중 하나가
“진짜로 어떤 설정으로 돌았는지 모른다” 이거다.

앤드레이 카르파시가 자주 강조하는 방향도 결국 비슷해. 한국어로 말하면, **“실험은 코드만이 아니라 설정까지 포함해서 관리해야 한다”**는 거다.

scripts/

진입점이다.

여기엔 얇은 파일만 둬야 한다:

argument parse
config load
pipeline call

scripts/train.py 안에 학습 로직 500줄 넣으면 망한다.

src/core/

공용 기초층이다.

예:

공통 타입
seed 설정
path 규칙
공통 에러
config schema
registry

여기는 어디서나 써도 되는 최소 공통 기반만 둬야 한다.

src/data/

여기가 제일 중요하다.

추천 세분화:

data/
├── sources/         # raw file loading, db loading, external source
├── preprocessing/   # cleaning, filtering, normalization
├── datasets/        # Dataset class
├── collators/       # batch assembly
└── samplers/        # sampling strategy

핵심은:

전처리
dataset
batch 조립

이 셋을 분리하는 거다.

대부분 AI 프로젝트가 더러워지는 이유는
전처리 로직이 Dataset 안에 다 박혀 있기 때문이야.

src/features/

이건 선택사항인데, 나는 꽤 추천한다.

특히 멀티모달이면 더 좋다.

예:

text tokenization
image transform
audio feature extraction
metadata encoding

이걸 data/preprocessing에 다 넣어버리면 의미가 섞인다.

preprocessing: 데이터 정리/정상화
features: 모델 입력 표현 생성

이렇게 분리하면 훨씬 낫다.

src/models/

여긴 당연히 모델 정의.

하지만 이것도 이렇게 쪼개는 게 좋다:

models/
├── backbones/
├── heads/
├── losses/
└── wrappers/

왜냐면 연구 코드에서 모델이란 게 사실
“백본 + 헤드 + 로스 + 래퍼” 조합인 경우가 많기 때문이야.

src/training/

여긴 학습 실행 책임.

예:

train/eval step
optimizer creation
scheduler creation
checkpoint save/load
mixed precision
distributed training hooks

즉 모델 구조와 학습 실행을 분리해야 한다.

모델 파일 안에 optimizer, scaler, logging 다 섞이면 금방 썩는다.

src/evaluation/

보통 이걸 너무 대충 만든다. 그러면 논문 결과 재현이 안 된다.

여기엔:

metric 계산
validation loop
benchmark runner
result formatter

이런 게 들어가야 한다.

experiments/

여기가 진짜 중요하다.

AI 프로젝트는 공식 코드와 실험 찌꺼기가 섞이면 끝장이다.
그러니까 아예 분리해.

예:

baseline 실험
ablation 실험
새 loss 실험
새 augmentation 실험

즉,

src/는 재사용 가능한 본체
experiments/는 변형과 시도

이렇게 나눠야 한다.

이건 리처드 해밍이 말한 태도랑도 닿아 있어. 한국어로 옮기면, **“중요한 실험과 잡다한 시도를 머릿속에서 분리하지 않으면 생각도 흐려진다”**는 식의 문제다.


내가 추천하는 의존 방향

이건 꽤 중요하다.

configs
  ↓
scripts
  ↓
pipelines
  ↓
data / features / models / training / evaluation
  ↓
core

그리고 이상적으로는:

models는 training을 몰라야 함
data는 models를 몰라야 함
evaluation은 training 내부 구현에 과도하게 묶이지 말아야 함
scripts는 orchestration만 해야 함

즉, 실행 조율과 내부 구현을 분리해야 한다.


AI 프로젝트에서 진짜 국룰

내가 보기에 진짜 국룰은 domain/data/presentation이 아니라 이거다:

1. 실행 진입점은 얇게

train.py는 짧아야 한다.

2. 설정을 외부화

실험 파라미터가 코드에 박혀 있으면 안 된다.

3. 데이터 변환 경계를 명확히

raw → cleaned → featured → batched 구간이 보여야 한다.

4. 모델 정의와 학습 실행 분리

nn.Module과 trainer를 섞지 마.

5. 실험 코드와 재사용 코드 분리

experiments/는 따로.

6. 재현성 경로를 확보

seed, config, artifact, metric 저장 위치를 표준화.

이게 더 현실적인 국룰이다.


좋아. 여기서 빠진 핵심이 딱 **산출물(artifact) 관리**다.
사실 AI 프로젝트는 코드 구조만큼이나 **“뭘 어디에 남기느냐”**가 중요해.
체크포인트, 로그, 결과 이미지, 분석 리포트가 흩어지면 프로젝트는 금방 썩는다.

내 의견은 명확하다:

> **코드(`src/`)와 실행 산출물(`artifacts/`, `outputs/`, `runs/`)은 무조건 분리해라.**

즉, 네 구조에 **artifact 계층**을 추가하는 게 맞다.

---

# 내가 추천하는 상위 구조

```text
project/
├── configs/
├── scripts/
├── src/
├── experiments/
├── tests/
├── notebooks/
├── artifacts/
│   ├── checkpoints/
│   ├── logs/
│   ├── predictions/
│   ├── figures/
│   ├── reports/
│   └── tables/
└── data/
```

근데 이것도 조금 더 현실적으로 가면,
나는 보통 **실행 단위(run 단위)** 로 묶는 걸 더 추천한다.

---

# 제일 추천하는 방식: run 기반 저장 구조

예를 들면:

```text
artifacts/
├── exp001_baseline/
│   ├── run_2026-03-27_140501/
│   │   ├── config.yaml
│   │   ├── checkpoints/
│   │   │   ├── epoch_001.ckpt
│   │   │   ├── epoch_010.ckpt
│   │   │   ├── best.ckpt
│   │   │   └── last.ckpt
│   │   ├── logs/
│   │   │   ├── train.log
│   │   │   ├── metrics.jsonl
│   │   │   └── tensorboard/
│   │   ├── figures/
│   │   │   ├── loss_curve.png
│   │   │   ├── val_accuracy_curve.png
│   │   │   └── confusion_matrix.png
│   │   ├── predictions/
│   │   │   ├── val_predictions.csv
│   │   │   └── sample_outputs.json
│   │   ├── reports/
│   │   │   └── summary.md
│   │   └── metadata.json
│   └── run_2026-03-27_190812/
│       └── ...
└── exp002_aug/
    └── ...
```

이게 좋은 이유는 간단하다:

* **실험 1회 실행의 모든 결과가 한 폴더에 모임**
* 나중에 “이 best.ckpt가 어떤 설정으로 나온 거지?” 같은 지옥을 피할 수 있음
* config, 로그, 곡선, 예측 결과가 연결됨
* 재현성이 훨씬 좋아짐

---

# 질문 1. 중간 체크포인트는 어디에 저장?

정답:

> **각 run 내부의 `checkpoints/` 에 저장**

예:

```text
artifacts/exp001_baseline/run_2026-03-27_140501/checkpoints/
```

보통 여기에 이런 식으로 둔다:

* `best.ckpt`
* `last.ckpt`
* `epoch_001.ckpt`
* `step_010000.ckpt`

---

## 체크포인트 저장 전략도 같이 정해야 한다

체크포인트를 무한정 저장하면 디스크 터진다.
그래서 보통 이렇게 한다:

### 추천

* `last.ckpt`: 항상 최신
* `best.ckpt`: validation metric 기준 최고
* `epoch_xxx.ckpt` 또는 `step_xxx.ckpt`: 드물게 저장
* top-k best만 유지

예:

* 매 epoch 저장 안 함
* 5 epoch마다 저장
* best 1개 + last 1개 + 주기 스냅샷 몇 개만 유지

이게 현실적이다.

---

# 질문 2. 결과 분석 이미지 어디에 저장?

정답:

> **run 내부 `figures/`**

예:

```text
artifacts/exp001_baseline/run_2026-03-27_140501/figures/
```

여기에는:

* loss curve
* accuracy curve
* confusion matrix
* sample prediction visualization
* embedding projection
* attention map
* reconstruction examples

이런 걸 둔다.

즉, **모델이 남긴 시각적 증거물**이 들어가는 곳이다.

---

# 질문 3. 결과 분석 “코드”는 어디에 둬?

이건 좀 나눠야 한다.

## 1) 재사용 가능한 분석 로직

이건 `src/` 아래로 들어가야 한다.

예:

```text
src/
└── analysis/
    ├── plotting.py
    ├── error_analysis.py
    └── visualization.py
```

혹은 `evaluation/visualization.py` 로 넣어도 된다.

즉,

* confusion matrix 그리는 함수
* 샘플 오분류 분석 함수
* latent embedding 시각화 함수

이런 건 **코드 자산**이니까 `src/`에 둬야 한다.

---

## 2) 일회성 탐색 분석 코드

이건 `notebooks/` 나 `experiments/` 쪽이 맞다.

예:

```text
notebooks/
├── 01_dataset_distribution.ipynb
├── 02_failure_case_analysis.ipynb
└── 03_embedding_visualization.ipynb
```

혹은

```text
experiments/exp001_baseline/analysis.ipynb
```

이렇게 둘 수 있다.

---

# 내 강한 의견

## notebook은 “최종 진실의 원천”이 되면 안 된다

많은 AI 프로젝트가 망하는 이유가 이거다.

* 중요한 분석 로직이 notebook에만 있음
* 누가 어떤 순서로 셀 돌렸는지 모름
* 재현 안 됨
* 코드 재사용 안 됨

그래서 원칙은 이거다:

### 원칙

* notebook: 탐색, 임시 분석
* `src/analysis` 또는 `src/evaluation`: 재사용 가능한 분석 로직
* `artifacts/figures`: 분석 결과물
* `artifacts/reports`: 실행 요약 문서

이렇게 분리하는 게 맞다.

---

# 내가 추천하는 더 완성된 구조

```text
project/
├── configs/
├── scripts/
├── src/
│   ├── core/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── analysis/
│   └── utils/
├── experiments/
├── tests/
├── notebooks/
├── artifacts/
│   ├── exp001_baseline/
│   │   ├── run_2026-03-27_140501/
│   │   │   ├── config.yaml
│   │   │   ├── metadata.json
│   │   │   ├── checkpoints/
│   │   │   ├── logs/
│   │   │   ├── figures/
│   │   │   ├── predictions/
│   │   │   ├── reports/
│   │   │   └── tables/
└── data/
    ├── raw/
    ├── interim/
    ├── processed/
    └── external/
```

---

# `data/`도 같이 분리하는 게 좋다

네가 질문한 건 artifact였지만 사실 이것도 같이 가야 완성된다.

## `data/`

* `raw/`: 원본 데이터
* `interim/`: 중간 전처리 산출물
* `processed/`: 학습 입력으로 바로 쓰는 데이터
* `external/`: 외부 공개 데이터, 다운로드 데이터

## `artifacts/`

* 체크포인트
* 로그
* 예측 결과
* 시각화
* 리포트

즉:

> **data는 입력 계열, artifacts는 출력 계열**

이렇게 구분해야 머리가 안 꼬인다.

---

# 실험 중간 산출물도 저장해야 하나?

응. 근데 다 저장하면 안 된다.

저장 가치가 있는 건 보통 이런 것들이다:

* best / last checkpoint
* validation predictions
* 주요 metric history
* representative sample outputs
* confusion matrix / loss curve
* 실제 실행 config
* git commit hash
* seed
* dataset version

즉, 단순히 모델 파일만 저장하면 안 된다.
**그 모델이 어떤 세계에서 태어났는지**도 같이 남겨야 한다.

이건 공학적으로는 provenance 문제다.
어떤 결과 ( y ) 가 나왔을 때, 그 원인 집합 ( x, \theta, c, s ) 를 복원 가능해야 한다는 거다.

대충 쓰면:

[
result = f(data, config, code, seed, environment)
]

AI 실험에서 중요한 건 `result`만이 아니라
이 함수의 입력들을 같이 저장하는 거다.

---

# 그래서 metadata.json에 뭘 넣냐

이런 거 넣는 게 좋다:

```json
{
  "run_name": "exp001_baseline_run_2026-03-27_140501",
  "git_commit": "a1b2c3d",
  "seed": 42,
  "dataset_version": "v3",
  "start_time": "2026-03-27T14:05:01",
  "model_name": "baseline_resnet",
  "best_metric": {
    "name": "val_accuracy",
    "value": 0.9132
  }
}
```

이거 없으면 나중에 전부 기억에 의존하게 된다.
그럼 끝이다.

---

# 실무적으로 제일 많이 하는 실수

## 1. checkpoint만 남기고 config 안 남김

이건 반쪽짜리다.

## 2. figures를 공용 폴더에 때려넣음

나중에 어느 실험 결과인지 모른다.

## 3. notebook에서만 분석

재현 안 된다.

## 4. `outputs/final_final_v2_reallyfinal.png`

이딴 파일명
이건 멸망의 시작이다 😂

---

# 내가 추천하는 명명 규칙

실험 이름 + 런 ID를 강제해라.

예:

* `exp001_baseline`
* `exp002_aug_flipmix`
* `exp003_textcond_v2`

런:

* `run_2026-03-27_140501`
* 혹은 UUID/hash

파일:

* `best.ckpt`
* `last.ckpt`
* `loss_curve.png`
* `val_predictions.csv`
* `summary.md`

이렇게 예측 가능한 이름이 좋아.

---

# 최종 정리

## 체크포인트

* `artifacts/<experiment>/<run>/checkpoints/`

## 결과 이미지

* `artifacts/<experiment>/<run>/figures/`

## 예측 결과 / 테이블

* `artifacts/<experiment>/<run>/predictions/`
* `artifacts/<experiment>/<run>/tables/`

## 분석 리포트

* `artifacts/<experiment>/<run>/reports/`

## 재사용 가능한 분석 코드

* `src/analysis/` 또는 `src/evaluation/`

## 일회성 탐색 분석

* `notebooks/` 또는 `experiments/<exp>/`

---

# 한 줄로 말하면

> **코드는 `src/`, 입력 데이터는 `data/`, 실행 산출물은 `artifacts/`, 임시 탐색은 `notebooks/`로 분리하는 게 제일 안 망한다.**

그리고 더 세게 말하면:

> **체크포인트 파일만 남기는 건 실험을 저장한 게 아니라 착각을 저장한 거다. config, metadata, metric, figure까지 같이 남겨야 진짜 저장이다.**
