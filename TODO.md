좋다. 이거 그냥 “연구계획”이 아니라 **실제 돌아가는 시스템 만드는 로드맵**으로 쪼개야 한다.
말 그대로 **에이전트가 바로 실행할 수 있는 TODO**로 떨어뜨린다.

---

# 전체 전략 (한 줄 요약)

> “텍스트가 진짜 쓸모 있는지”가 아니라
> **“텍스트 덕분에 더 싸고 빠르게 쓸 수 있냐”**를 증명하는 프로젝트다.

---

# Phase 0 — 환경 & 베이스라인 고정 (이거 안 하면 다 무너짐)

## TODO

* [X] NWM smallest variant 코드 확보 및 실행
  - smallest = `config/nwm_cdit_s.yaml` (`CDiT-S/2`)
  - local ckpt 확인: `logs/nwm_cdit_s/checkpoints/cdit_s_100000.pth.tar`
  - inference/eval 시 checkpoint arg는 기본값 `0100000` 대신 `--ckp cdit_s_100000` 사용
  - 남은 blocker: RECON 데이터 다운로드/전처리, `eval_datasets.recon.data_folder` 로컬 절대경로 수정, 필요 시 VAE를 `logs/sd-vae-ft-ema`로 로컬 로드
* [X] RECON 데이터셋 다운로드 
  - direct dataset archive 다운로드 후 압축 해제 완료
  - 압축 해제 프로세스 종료: `exit code 0`
  - 결과 폴더: `/home/kun/kun_ssd/nwm/datasets/recon_raw/recon_release`
  - 현재 크기: 약 `72G`
  - 파일 수: `11836`개 `.hdf5`
* [X] RECON 데이터셋 로딩 + inference pipeline 확인
  - 확인 결과: 현재 워크스페이스의 RECON은 processed JPG 폴더가 아니라 raw `.hdf5`만 존재
  - 코드 조치: `datasets.py`가 raw RECON `.hdf5`를 직접 읽도록 확장
  - 코드 조치: `config/eval_config.yaml`의 `eval_datasets.recon.data_folder`를 `datasets/recon_raw/recon_release`로 변경
  - 코드 조치: inference / training / planning에서 VAE를 로컬 `logs/sd-vae-ft-ema` 우선 로드하도록 수정
  - 코드 조치: `isolated_nwm_infer.py`에서 `dist.init_distributed()` 호출로 entrypoint 오류 수정
  - 코드 조치: `scripts/nwm-run.sh`가 비대화형 환경에서도 실행되도록 TTY 감지 추가
  - 컨테이너 상태: `nwm_dev` detached 실행 중, `./scripts/nwm-run.sh`로 컨테이너 내부 명령 실행 가능
  - 컨테이너 검증: `EvalDataset(recon)[0]` 로드 성공
  - 컨테이너 검증 결과: `loaded_shapes [(1,), (4, 3, 224, 224), (64, 3, 224, 224), (64, 3)]`
  - 컨테이너 검증: `config/nwm_cdit_s.yaml` + `logs/nwm_cdit_s/checkpoints/cdit_s_100000.pth.tar`로 1-sample forward 성공
  - 컨테이너 검증 결과: `pred_shape (1, 3, 224, 224)`
  - 추가 검증: raw RECON `.hdf5` 기준 `EvalDataset(recon)` 길이 `500`, 첫 배치 shape `(1, 4, 3, 224, 224) / (1, 64, 3, 224, 224) / (1, 64, 3)` 확인
  - 추가 검증: `logs/nwm_cdit_xl/checkpoints/cdit_xl_100000.pth.tar` + 로컬 VAE로 1-sample forward 성공, 샘플 출력 `/tmp/nwm_recon_smoke/recon_pred_t8.png`
  - 환경 조치: running container 안 `torchaudio 2.11.0.dev...` -> `torchaudio 2.10.0` 재설치로 `torch 2.10.0`과 ABI 정합성 복구
  - 환경 조치: `Dockerfile` / `README.md`의 PyTorch stack을 `torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0`으로 고정
  - 운영 조치: `logs/nwm_cdit_{s,b,l,xl}/checkpoints/0100000.pth.tar` symlink 추가, 코드 fallback 없이 기존 `--ckp 0100000` 규약 유지
  - 현재 상태: `isolated_nwm_infer.py` import 및 `0100000.pth.tar` 로드 성공
  - 환경 조치: `Dockerfile`에 `CONDA_PREFIX` / `LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}` 추가
  - 운영 조치: `scripts/nwm-run.sh`가 컨테이너 내부 실행 전 `CONDA_PREFIX`, `PATH`, `LD_LIBRARY_PATH`를 명시적으로 export하도록 수정
  - 해결 결과: `torch` import 이후에도 `sqlite3` import 성공, `planning_eval.py` import 성공
  - 운영 조치: `scripts/recon_smoke_test.py` 추가, RECON 로딩/1-sample forward를 프로젝트 내부 `artifacts/recon_smoke`에 저장하도록 정리
  - 운영 조치: `scripts/nwm-start.sh` 추가, `NWM_GPU_REQUEST=all` 또는 `NWM_GPU_REQUEST='device=0,1'`로 컨테이너 GPU 가시성을 설정 가능하게 정리
  - 문서 조치: `DEV_CONTAINER_WORKFLOW.md`에 멀티 GPU 컨테이너 실행 예시 추가
  - 재현 메모: 현재 떠 있는 `nwm:cu126` 이미지에는 `h5py`가 없어서 컨테이너 내부에서 1회 설치함. 새 이미지에서는 `env.yaml` 반영 후 재빌드 필요
  - 운영 메모: 새 Python 라이브러리는 먼저 running container 안에 임시 설치하고, 반복 사용이 확정되면 `env.yaml`에 반영한 뒤 필요 시만 이미지 재빌드
* [X] baseline metric 재현 (LPIPS / DreamSim / FID)
  - 범위: `RECON + config/nwm_cdit_s.yaml + 0100000 + time/rollout eval`
  - GT PNG `2500`장 생성 완료: `artifacts/lpips_time_recon_s/gt/recon/time`
  - 예측 PNG `2500`장 생성 완료: `artifacts/lpips_time_recon_s/nwm_cdit_s/recon/time`
  - 결과 JSON: `artifacts/lpips_time_recon_s/nwm_cdit_s/recon_time.json`
  - LPIPS: `1s 0.3106`, `2s 0.3416`, `4s 0.3699`, `8s 0.4031`, `16s 0.4632`
  - DreamSim: `1s 0.1390`, `2s 0.1499`, `4s 0.1615`, `8s 0.1784`, `16s 0.2286`
  - FID: `1s 35.0850`, `2s 34.4481`, `4s 33.9934`, `8s 33.4646`, `16s 40.3887`
  - rollout GT PNG 생성 완료: `artifacts/lpips_time_recon_s/gt/recon/rollout_1fps` `2400`장, `artifacts/lpips_time_recon_s/gt/recon/rollout_4fps` `9600`장
  - rollout 예측 PNG 생성 완료: `artifacts/lpips_time_recon_s/nwm_cdit_s/recon/rollout_1fps` `2400`장, `artifacts/lpips_time_recon_s/nwm_cdit_s/recon/rollout_4fps` `9600`장
  - rollout 결과 JSON: `artifacts/lpips_time_recon_s/nwm_cdit_s/recon_rollout_1fps.json`, `artifacts/lpips_time_recon_s/nwm_cdit_s/recon_rollout_4fps.json`
  - rollout 1fps LPIPS: `1s 0.3172`, `2s 0.3501`, `4s 0.4146`, `8s 0.4964`, `16s 0.5930`
  - rollout 1fps DreamSim: `1s 0.1414`, `2s 0.1526`, `4s 0.2034`, `8s 0.3106`, `16s 0.4629`
  - rollout 1fps FID: `1s 60.69`, `2s 65.64`, `4s 70.72`, `8s 102.75`, `16s 141.49`
  - rollout 4fps LPIPS: `1s 0.3401`, `2s 0.3825`, `4s 0.4447`, `8s 0.5115`, `16s 0.5715`
  - rollout 4fps DreamSim: `1s 0.1458`, `2s 0.1707`, `4s 0.2128`, `8s 0.2880`, `16s 0.3619`
  - rollout 4fps FID: `1s 63.40`, `2s 65.96`, `4s 72.98`, `8s 88.54`, `16s 92.41`
  - 평가 중 `cv2` 의존성으로 LPIPS가 죽는 문제를 `isolated_nwm_eval.py`에서 `PIL + numpy` 로더로 우회
  - rollout 생성 중 `batch_size=1`에서 `idxs.squeeze()`가 0-d tensor가 되는 버그를 `isolated_nwm_infer.py`에서 `idxs.reshape(-1)`로 수정
  - 캐시 조치: `scripts/nwm-run.sh`가 레포 내부 `.cache`를 쓰도록 수정, 기존 AlexNet cache도 `.cache/torch/hub/checkpoints`로 복사
  - 운영 메모: 오래 살아 있던 `nwm_dev` 컨테이너에서 CUDA 인식이 깨져 rollout eval이 실패했고, `GPU 1`로 컨테이너 재생성 후 정상 완료
  - 꽤 빨리 끝남! 다만 rollout 예측 생성은 autoregressive + diffusion sampling 때문에 오래 걸림
* [X] rollout (1s / 2s / 4s) evaluation 코드 확보
  - 현재 코드 경로로 `rollout_1fps`, `rollout_4fps` 평가 완료
  - 결과 JSON: `artifacts/lpips_time_recon_s/nwm_cdit_s/recon_rollout_1fps.json`, `artifacts/lpips_time_recon_s/nwm_cdit_s/recon_rollout_4fps.json`
* [X] GPU profiling (latency / VRAM / FLOPs baseline 기록)
  - `2026-03-19 KST`, `CDiT-S`, `recon`, `batch_size=1`, `diffusion_steps=250`, `torch.compile on`
  - single-step baseline: `1315.3 +- 10.0 ms`, `p50 1310.5 ms`, `p90 1327.6 ms`, `peak alloc 919.3 MB`, `peak reserved 1016.0 MB`
  - rollout baseline (`rollout_fps=1`, `16` autoregressive frames): `21867.5 +- 158.0 ms`, `p50 21809.8 ms`, `p90 21998.9 ms`, `peak alloc 926.1 MB`, `peak reserved 1022.0 MB`
  - FLOPs baseline (`torch.profiler`, `single-step`, `torch.compile off`): `6435.0 GFLOPs` (`6.44 TFLOPs`)
  - rollout total FLOPs (`rollout_fps=1`, `16` frames, derived): `102960.2 GFLOPs` (`102.96 TFLOPs`) = `16 x` single-step FLOPs
  - 결과 파일: `artifacts/gpu_profile_baseline/nwm_cdit_s_recon_sample0_bs1_diff250_single.json`, `artifacts/gpu_profile_baseline/nwm_cdit_s_recon_sample0_bs1_diff250_single_step.csv`, `artifacts/gpu_profile_baseline/nwm_cdit_s_recon_sample0_bs1_diff250_rollout.json`, `artifacts/gpu_profile_baseline/nwm_cdit_s_recon_sample0_bs1_diff250_rollout.csv`

## 산출물

* baseline 성능 표 (무조건 필요)
* baseline inference latency log

이거 없으면 나중에 “좋아졌는지” 판단 자체가 불가능

---

# Phase 1 — 데이터 파이프라인 구축 (여기서 승부 50% 결정됨)

진행 메모:

* [x] 오프라인 파이프라인 스크립트 추가 완료
* [x] 학습/추론 코드에 cached text embedding 경로 연결 완료
* [x] raw RECON subset 기준 end-to-end smoke 실행 완료
* [x] full dataset Qwen caption run 완료
* [ ] full dataset clean + embedding + train smoke는 아직 미완료
* [ ] 최종 aligned dataset 산출물 생성은 아직 미완료

## 1.1 프레임 추출

* [x] video → frame (1 FPS) 스크립트 구현
* [x] keyframe sampling 옵션 구현 (추후 실험용)
* [x] raw RECON `.hdf5`에서 직접 caption용 manifest 생성 스크립트 추가
* [x] raw RECON `.hdf5` -> downsampled `jpg + traj_data.pkl` export 스크립트 추가
* [x] raw RECON 1 trajectory 기준 downsampled export smoke 완료
* [x] raw RECON test split 전체 `1fps` export 완료 (`2367 traj`, `13193 frames`)
* [x] raw RECON train split 전체 `1fps` export 완료 (`9468 traj`, `53395 frames`)

## 1.2 오프라인 캡션 생성

* [x] Qwen2-VL / LLaVA(사용X) inference pipeline 구축
* [x] prompt 템플릿 설계 (scene-only / goal 포함 분리)
* [x] batch caption generation (병렬화 필수)
* [x] 도커에서 외부 모델 디렉터리 mount 가능하도록 `scripts/nwm-start.sh` 확장
* [x] `openai/clip-vit-base-patch32` HF cache 다운로드 완료
* [x] raw RECON subset (`2 traj x 4 frames`) Qwen caption smoke 완료
* [x] exported RECON test split full manifest 생성 완료 (`13193` records)
* [x] exported RECON train split full manifest 생성 완료 (`53395` records)
* [x] Qwen2-VL 실제 full-dataset caption run 완료
  - test split: `16 / 16` shards 완료, merged output 생성
  - train split: `64 / 64` shards 완료, merged output 생성
  - merged caption output:
    - `/.cache/phase1_qwen/recon_test_1fps/all.jsonl`
    - `/.cache/phase1_qwen/recon_train_1fps/all.jsonl`

## 1.3 텍스트 정제

* [x] caption → structured text (tag or short sentence)
* [x] 불필요한 문장 제거 (압축 중요)
* [x] `the scene shows/depicts/features` boilerplate 제거 규칙 추가

## 1.4 embedding precompute

* [x] CLIP text encoder 로 embedding 생성 스크립트 구현
* [x] (frame_id → text_embedding) 캐싱
* [x] dataloader에서 바로 불러오도록 설계
* [x] subset caption 결과로 embedding precompute 실행
* [x] cached embedding 붙인 dataloader/model smoke test
* [ ] full dataset cleaned caption 결과로 embedding precompute 실행

## 산출물

* [ ] frame + action + text_embedding aligned dataset
* [x] raw RECON test split에 대해 1fps processed dataset export
* [x] raw RECON train split에 대해 1fps processed dataset export
* [x] raw RECON test/train full caption merged JSONL 생성

핵심: **학습 때 텍스트 인코더 절대 돌리지 마라 (속도 병목 터짐)**

---

# Phase 2 — 모델 결합 (Text Conditioning)

## 2.1 최소 변경 버전 (무조건 먼저)

* [ ] text → projection layer 추가
* [ ] 기존 conditioning vector에 단순 concat or sum

## 2.2 안정성 장치

* [ ] projection layer zero init
* [ ] text dropout (p=0.3~0.5)
* [ ] 일부 샘플 text 제거

## 2.3 gated fusion (2차)

* [ ] gating scalar or MLP 추가
* [ ] text influence 조절

## 구조 목표

```
cond = f(image, action, timestep, text)
```

핵심: **NWM을 깨지 말고 “옆에 붙인다”**

---

# Phase 3 — PoC (진짜 중요한 첫 검증)

## 실험

* [ ] Base NWM
* [ ] + Scene text
* [ ] + Goal text
* [ ] + Scene+Goal

## 평가

* [ ] LPIPS / PSNR / DreamSim
* [ ] rollout stability (1s / 2s / 4s)

## 판단 기준

* 텍스트가 **조금이라도 의미 있게 도움 되냐**

여기서 효과 없으면 방향 바꿔야 한다 (냉정하게)

---

# Phase 4 — Resolution Scaling (이게 핵심 논문 포인트)

## TODO

* [ ] 해상도 단계별 실험

  * high → 128 → 112 → 64
* [ ] 각 단계에서

  * no-text vs text 비교

## 분석

* [ ] degradation slope 계산
* [ ] text가 손실 얼마나 줄이는지 정량화

## 목표

* low-res + text ≈ mid/high-res without text

이거 나오면 논문 먹힌다

---

# Phase 5 — OOD 일반화

## TODO

* [ ] GO Stanford 데이터셋 evaluation
* [ ] unseen scene rollout 테스트

## 분석

* [ ] hallucination 발생 시점
* [ ] trajectory 붕괴 시점
* [ ] text vs no-text 비교

## 포인트

* text가 context 유지해주냐

이건 “진짜 의미 이해했냐” 테스트다
(데이비드 흄: “인간은 반복이 아니라 의미로 세계를 이해한다”)

---

# Phase 6 — Online Lightweight Text Pipeline

## 6.1 YOLO 기반 (우선)

* [ ] YOLOv10 Nano inference
* [ ] detection → tag 변환
* [ ] tag → text template

## 6.2 optional (비교용)

* [ ] Moondream2 경량 caption

## 6.3 integration

* [ ] real-time frame → text → embedding → NWM

## 산출물

* 실제 inference pipeline

---

# Phase 7 — End-to-End 시스템 평가

## TODO

* [ ] 전체 pipeline latency 측정

  * image → text → NWM → output
* [ ] FPS 측정
* [ ] VRAM usage 기록

## 비교군

* [ ] high-res NWM
* [ ] low-res no-text
* [ ] low-res + YOLO text
* [ ] low-res + Moondream

## 핵심 질문

* “텍스트 넣었더니 느려졌다” → 바로 실패

반드시 **total system 기준으로 이득 보여야 함**

---

# Phase 8 — Ablation (논문 완성 단계)

## TODO

* [ ] text source 비교 (VLM vs YOLO vs template)
* [ ] fusion 방식 비교
* [ ] text noise robustness
* [ ] wrong tag injection 실험
* [ ] text dropout 효과

---

# Phase 9 — 결과 정리 & 메시지 만들기

## 필수 그래프

* [ ] resolution vs performance curve
* [ ] latency vs performance tradeoff
* [ ] rollout degradation graph
* [ ] OOD failure case visualization

## 핵심 주장 정리

* 텍스트 = 성능 향상이 아니라
  → **정보 압축 보조 채널**

---

# 최종 Deliverable (논문/발표 기준)

* [ ] pipeline diagram (필수)
* [ ] architecture diagram (baseline vs ours)
* [ ] quantitative table (in-domain / OOD)
* [ ] efficiency table (latency / VRAM)
* [ ] qualitative visualization (rollout 비교)

---

# 진짜 중요한 핵심 포인트 (팩트로 말한다)

1. 이 연구의 본질은 멀티모달이 아니다
   → **compression + prior injection 문제다**

2. 실패하는 가장 흔한 이유
   → 텍스트가 “정보 추가”가 아니라
   → 그냥 **노이즈 + 계산량 증가**가 되는 경우

3. 성공 조건

   * 텍스트가 **이미지보다 싸고**
   * **의미는 더 잘 보존**해야 한다

이거 한 줄로 요약하면

> “semantic bitrate를 높인다”

(클로드 섀넌: “정보는 압축될수록 가치가 드러난다” — 정보이론 창시자)

---

# 너한테 진짜 중요한 선택

솔직하게 말하면 지금 갈림길은 이거다:

* YOLO tag 기반 (빠름, 정보 제한)
* lightweight caption (느림, 정보 풍부)

나는 강하게 말한다:

> **1차는 무조건 YOLO로 가라.**

이유:

* 논문 메시지가 “경량화”인데
* heavy caption 붙이면 바로 논리 붕괴됨

---

# 한 줄 액션 플랜

> **“Baseline 재현 → 텍스트 데이터 구축 → 최소 결합 → PoC 검증 → 해상도 축소 → OOD → 시스템 평가”**

이 순서 절대 바꾸지 마라.

---

# 다음 단계 추천 (너 기준으로 현실적인 루트)

1. 이번 주
   → Phase 0 + Phase 1 일부

2. 다음 주
   → PoC 결과 확보

3. 그 다음
   → resolution scaling


원하면
바로 **코드 구조 (PyTorch + dataloader + conditioning injection)**까지 설계해줄게
이건 진짜 중요하다.
