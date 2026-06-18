# 프로젝트 구조도

이 문서는 NWM 프로젝트를 다시 볼 때 각 폴더의 역할을 빠르게 파악하기 위한 구조 요약입니다.

핵심은 `scripts/`가 실행 진입점, `src/`가 실제 구현, `configs/`가 실험 설정, `datasets/`와 `data/splits/`가 데이터와 split, `weights/`, `logs/`, `artifacts/`가 실행 결과입니다.

```text
nwm/
├── scripts/                 # 학습/추론/평가/유틸 실행 스크립트
├── src/                     # 실제 라이브러리 코드
├── configs/                 # YAML 설정 파일
├── data/splits/             # train/test/eval split 메타데이터
├── datasets/                # 실제 데이터셋 및 파생 데이터
├── weights/                 # 학습 체크포인트, pretrained weight, cache
├── logs/                    # 학습 로그
├── artifacts/               # 평가 결과, 샘플 이미지, profiling 결과
├── tests/                   # smoke test
├── notebooks/               # 실험/시각화 노트북
├── docs/                    # 프로젝트 문서, 제안서, 데이터 설명
├── experiments/             # 실험 노트
├── third_party/             # 외부 코드 vendoring
└── Dockerfile, env.yaml 등  # 환경 구성 파일
```

## 가장 먼저 볼 곳

- `README.md`: 원본 NWM 사용법, 데이터 준비, 학습/평가 명령.
- `DEV_CONTAINER_WORKFLOW.md`: Docker 기반 개발/실행 방식.
- `AGENTS.md`: 이 repo에서 작업할 때의 규칙.
- `configs/experiment/*.yaml`: 어떤 실험을 돌리는지 결정하는 핵심 설정.
- `scripts/train.py`, `scripts/infer.py`, `scripts/evaluate.py`, `scripts/plan_eval.py`: 주요 실행 진입점.

## `scripts/`

실행용 스크립트 폴더입니다.

```text
scripts/
├── train.py                 # CDiT world model 학습
├── infer.py                 # checkpoint로 미래 프레임/rollout 생성
├── evaluate.py              # LPIPS, DreamSim, FID 평가
├── plan_eval.py             # planning 평가, 내부적으로 CEM planner 호출
├── submitit_train.py        # Slurm/submitit 학습 실행
├── docker/                  # nwm-start.sh, nwm-run.sh
├── analysis/                # metric table/plot 생성
├── eval/                    # checkpoint sweep 등 평가 보조
├── maintenance/             # artifacts 정리
├── profiling/               # GPU profiling
└── user_friendly/           # Jupyter start/stop
```

많이 쓰는 흐름은 다음과 같습니다.

```text
train.py
  -> configs/experiment/*.yaml 읽음
  -> src.data.datasets.TrainingDataset
  -> src.models.backbones.cdit.CDiT
  -> src.diffusion
  -> weights/checkpoints/<run_name>/ 에 저장

infer.py
  -> checkpoint 로드
  -> eval dataset 로드
  -> 예측 이미지 생성
  -> artifacts/bulk/eval 또는 지정 output_dir 에 저장

evaluate.py
  -> GT 이미지와 예측 이미지 비교
  -> LPIPS/DreamSim/FID JSON 저장

plan_eval.py
  -> src.evaluation.planning.cem_planner 실행
```

## `src/`

프로젝트의 실제 Python 패키지입니다.

```text
src/
├── config.py                  # YAML config 로딩 헬퍼
├── core/
│   ├── env/distributed.py       # DDP, rank, device 초기화
│   └── paths.py                 # logs/artifacts/weights 경로 규칙
│
├── data/
│   ├── io.py                    # trajectory/image 로드 유틸
│   ├── datasets/
│   │   ├── train_dataset.py     # 학습 데이터셋
│   │   ├── eval_dataset.py      # time/rollout 평가 데이터셋
│   │   ├── trajectory_eval_dataset.py
│   │   ├── base_dataset.py
│   │   └── factory.py           # eval/planning dataset factory
│   └── transforms/
│       ├── image.py             # 이미지 transform
│       └── action.py            # action/delta 처리
│
├── models/
│   ├── backbones/cdit.py        # 핵심 CDiT 모델
│   └── checkpoints/
│       ├── loader.py            # checkpoint 로딩
│       └── vae.py               # StabilityAI VAE 로딩
│
├── diffusion/
│   ├── gaussian_diffusion.py    # DDPM diffusion 구현
│   ├── respace.py
│   └── diffusion_utils.py
│
├── evaluation/
│   ├── inference/rollout.py     # infer.py에서 쓰는 rollout/time generation
│   ├── metrics/perceptual.py    # LPIPS, DreamSim, FID
│   └── planning/                # CEM planner, action helpers, metrics, ranker, outputs
│
└── features/
    └── text/                    # text-conditioning 파이프라인/유틸
```

## `configs/`

실험 설정 YAML입니다.

```text
configs/
├── experiment/             # 실제 학습/평가 실험 설정
├── model/                  # CDiT S/B/L/XL 모델 축
├── data/                   # 데이터셋/image-size 축
├── feature/text/           # text-conditioning 축
└── evaluation/             # 평가 기본 설정
```

중요한 예:

```text
configs/experiment/nwm_cdit_s_recon_128.yaml
configs/experiment/nwm_cdit_s_recon_128_text_dense.yaml
configs/experiment/nwm_cdit_xl.yaml
configs/model/cdit_s.yaml
configs/data/recon_raw_128.yaml
configs/feature/text/dense_recon_raw.yaml
configs/evaluation/eval_config.yaml
```

현재 로컬 실험은 `nwm_cdit_s_recon_128*`, `*_text_dense*` 쪽이 중요해 보입니다.

## `data/` vs `datasets/`

둘은 역할이 다릅니다.

`data/splits/`는 실제 이미지/센서 데이터가 아니라 split 메타데이터입니다.

```text
data/splits/recon/train/traj_names.txt
data/splits/recon/test/time.pkl
data/splits/recon/test/rollout.pkl
```

반면 `datasets/`는 실제 데이터셋 또는 파생 데이터입니다.

```text
datasets/
├── recon_raw/              # RECON 원본 HDF5 기반 데이터
├── recon_1fps_train/       # 1fps로 가공된 RECON train
├── recon_1fps_test/        # 1fps로 가공된 RECON test
├── recon_datavis/          # RECON 시각화 도구
├── scand_320/
├── sacson_320/
├── tartan_320/
├── go_stanford/
└── derived/                # Qwen caption, text embedding 등 파생 산출물
```

특히 `datasets/derived/phase1_text_embeds_dense/...`는 text-conditioned 실험에서 쓰이는 embedding 저장소입니다.

## `weights/`, `logs/`, `artifacts/`

실행 산출물입니다.

```text
weights/
├── checkpoints/            # 학습 checkpoint
├── pretrained/             # VAE, Qwen 등 pretrained 모델
├── adapters/               # LoRA/adapter류
├── cache/                  # 모델/cache 파일
└── pretrained/archives/    # legacy pretrained 압축 보관본
```

```text
logs/
```

학습 로그가 저장됩니다.

```text
logs/nwm_cdit_s_recon_128/log.txt
logs/nwm_cdit_s_recon_128_text_dense/log.txt
```

```text
artifacts/
├── bulk/train/             # 학습 샘플 dump
├── bulk/eval/              # raw inference/GT frames
├── bulk/planning/          # planning 예측 tensor/plot
├── bulk/logs/              # 재생성 가능한 runtime log
├── summaries/              # metric JSON, compact summary, deck/report
├── profiling/              # profiling CSV/PNG
├── smoke/                  # smoke test 출력
├── dvc/                    # DVC smoke/repro 출력
├── tmp/                    # 임시 수동 확인
└── _trash/                 # 정리 대기 산출물
```

## `tests/`

정식 pytest suite는 아직 없고 smoke test 중심입니다.

```text
tests/smoke/recon_smoke_test.py
```

권장 검증:

```bash
./scripts/docker/nwm-run.sh "python tests/smoke/recon_smoke_test.py --skip-forward"
./scripts/docker/nwm-run.sh "python tests/smoke/recon_smoke_test.py --horizon-steps 8"
```

## 문서/실험 보조 폴더

```text
docs/                     # RECON 구조 설명, artifacts 정책, 발표/제안서
experiments/notes/        # 날짜별 실험 노트
notebooks/                # interactive_model, diffusion/VAE visualization
```

## 헷갈릴 수 있는 폴더

- `src/diffusion/`가 현재 코드에서 import되는 diffusion 구현입니다.
- 루트의 `diffusion/` 잔재는 제거했고, 새 작업은 `src/diffusion/` 기준으로 보면 됩니다.
- 루트의 `models/` 잔재는 제거했습니다. 실제 CDiT 구현은 `src/models/backbones/cdit.py`이고, pretrained/metric weight는 `weights/pretrained/` 아래에 둡니다.
- DINO 외부 코드는 추적되는 `third_party/facebookresearch_dino_main/`을 기준으로 사용합니다.

## 한 줄 실행 흐름

```text
configs/*.yaml
   ↓
scripts/train.py
   ↓
src/data + src/models + src/diffusion
   ↓
weights/checkpoints + logs + artifacts

weights/checkpoints
   ↓
scripts/infer.py
   ↓
artifacts/eval frames
   ↓
scripts/evaluate.py
   ↓
metric JSON / summaries
```

오랜만에 다시 시작한다면 우선 다음 순서로 보면 전체 감이 가장 빨리 돌아옵니다.

1. `configs/experiment/nwm_cdit_s_recon_128_text_dense.yaml`
2. `scripts/train.py`
3. `src/data/datasets/train_dataset.py`
4. `src/models/backbones/cdit.py`
5. `src/evaluation/inference/rollout.py`
