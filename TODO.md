좋다. 이거 그냥 “연구계획”이 아니라 **실제 돌아가는 시스템 만드는 로드맵**으로 쪼개야 한다.
말 그대로 **에이전트가 바로 실행할 수 있는 TODO**로 떨어뜨린다.

---

# 전체 전략 (한 줄 요약)

> “텍스트가 진짜 쓸모 있는지”가 아니라
> **“텍스트 덕분에 더 싸고 빠르게 쓸 수 있냐”**를 증명하는 프로젝트다.

---

# 현재 상태 스냅샷

기준일: `2026-05-26`

* 작업 시작 시 상태: `main...origin/main`, local change 없음
* 환경/데이터: raw RECON, Qwen caption, cleaned caption, CLIP embedding, dense text cache까지 구축 완료
* 모델 결합: cached text embedding을 `CDiT` conditioning에 더하는 최소 text-conditioned 경로 구현 완료
* 224 실험:
  - no-text baseline `nwm_cdit_s` time/rollout metric 확보
  - text-conditioned `nwm_cdit_s_recon_raw_text_dense` `0030000` time/rollout metric 확보
  - planning JSON도 확보했으나 text가 planning에서 명확히 우세하다고 말하기는 어려움
* 128 실험:
  - no-text `nwm_cdit_s_recon_128` checkpoint는 `0055000`까지 확보
  - text `nwm_cdit_s_recon_128_text_dense` checkpoint는 `0030000`까지 확보
  - `0030000` 기준 text는 image metric에서 대체로 우세하지만, no-text가 `0055000`까지 더 학습되어 공정 비교는 아직 재정렬 필요
* 현재 가장 중요한 다음 작업:
  - `128 no-text`와 `128 text`의 checkpoint/step/eval 조건을 맞춘 공정 비교 세트 완성
  - planning eval을 repeat/seed 기준으로 다시 정리
  - TODO/발표/제안서 수치를 실제 artifact 경로와 동기화

---

# Phase 0 — 환경 & 베이스라인 고정 (이거 안 하면 다 무너짐)

## TODO

* [X] NWM smallest variant 코드 확보 및 실행
  - smallest = `configs/experiment/nwm_cdit_s.yaml` (`CDiT-S/2`)
  - local ckpt 확인: `weights/checkpoints/nwm_cdit_s/cdit_s_100000.pth.tar`
  - `weights/checkpoints/nwm_cdit_{s,b,l,xl}/0100000.pth.tar` symlink 추가 완료
  - 현재는 기존 `--ckp 0100000` 규약으로 실행 가능
  - 초기 blocker였던 RECON 다운로드/전처리, 로컬 VAE 경로, eval data path는 해결 완료
* [X] RECON 데이터셋 다운로드 
  - direct dataset archive 다운로드 후 압축 해제 완료
  - 압축 해제 프로세스 종료: `exit code 0`
  - 결과 폴더: `/home/kun/kun_ssd/nwm/datasets/recon_raw/recon_release`
  - 현재 크기: 약 `72G`
  - 파일 수: `11836`개 `.hdf5`
* [X] RECON 데이터셋 로딩 + inference pipeline 확인
  - 확인 결과: 현재 워크스페이스의 RECON은 processed JPG 폴더가 아니라 raw `.hdf5`만 존재
  - 코드 조치: `datasets.py`가 raw RECON `.hdf5`를 직접 읽도록 확장
  - 코드 조치: `configs/evaluation/eval_config.yaml`의 `eval_datasets.recon.data_folder`를 `datasets/recon_raw/recon_release`로 변경
  - 코드 조치: inference / training / planning에서 VAE를 로컬 `weights/pretrained/vae/sd-vae-ft-ema` 우선 로드하도록 수정
  - 코드 조치: `isolated_nwm_infer.py`에서 `dist.init_distributed()` 호출로 entrypoint 오류 수정
  - 코드 조치: `scripts/docker/nwm-run.sh`가 비대화형 환경에서도 실행되도록 TTY 감지 추가
  - 컨테이너 상태: `nwm_dev` detached 실행 중, `./scripts/docker/nwm-run.sh`로 컨테이너 내부 명령 실행 가능
  - 컨테이너 검증: `EvalDataset(recon)[0]` 로드 성공
  - 컨테이너 검증 결과: `loaded_shapes [(1,), (4, 3, 224, 224), (64, 3, 224, 224), (64, 3)]`
  - 컨테이너 검증: `configs/experiment/nwm_cdit_s.yaml` + `weights/checkpoints/nwm_cdit_s/cdit_s_100000.pth.tar`로 1-sample forward 성공
  - 컨테이너 검증 결과: `pred_shape (1, 3, 224, 224)`
  - 추가 검증: raw RECON `.hdf5` 기준 `EvalDataset(recon)` 길이 `500`, 첫 배치 shape `(1, 4, 3, 224, 224) / (1, 64, 3, 224, 224) / (1, 64, 3)` 확인
  - 추가 검증: `weights/checkpoints/nwm_cdit_xl/cdit_xl_100000.pth.tar` + 로컬 VAE로 1-sample forward 성공, 샘플 출력 `/tmp/nwm_recon_smoke/recon_pred_t8.png`
  - 환경 조치: running container 안 `torchaudio 2.11.0.dev...` -> `torchaudio 2.10.0` 재설치로 `torch 2.10.0`과 ABI 정합성 복구
  - 환경 조치: `Dockerfile` / `README.md`의 PyTorch stack을 `torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0`으로 고정
  - 운영 조치: `weights/checkpoints/nwm_cdit_{s,b,l,xl}/0100000.pth.tar` symlink 추가, 코드 fallback 없이 기존 `--ckp 0100000` 규약 유지
  - 현재 상태: `isolated_nwm_infer.py` import 및 `0100000.pth.tar` 로드 성공
  - 환경 조치: `Dockerfile`에 `CONDA_PREFIX` / `LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}` 추가
  - 운영 조치: `scripts/docker/nwm-run.sh`가 컨테이너 내부 실행 전 `CONDA_PREFIX`, `PATH`, `LD_LIBRARY_PATH`를 명시적으로 export하도록 수정
  - 해결 결과: `torch` import 이후에도 `sqlite3` import 성공, `planning_eval.py` import 성공
  - 운영 조치: `tests/smoke/recon_smoke_test.py` 추가, RECON 로딩/1-sample forward를 프로젝트 내부 `artifacts/smoke/recon`에 저장하도록 정리
  - 운영 조치: `scripts/docker/nwm-start.sh` 추가, `NWM_GPU_REQUEST=all` 또는 `NWM_GPU_REQUEST='device=0,1'`로 컨테이너 GPU 가시성을 설정 가능하게 정리
  - 문서 조치: `DEV_CONTAINER_WORKFLOW.md`에 멀티 GPU 컨테이너 실행 예시 추가
  - 재현 메모: 현재 떠 있는 `nwm:cu126` 이미지에는 `h5py`가 없어서 컨테이너 내부에서 1회 설치함. 새 이미지에서는 `env.yaml` 반영 후 재빌드 필요
  - 운영 메모: 새 Python 라이브러리는 먼저 running container 안에 임시 설치하고, 반복 사용이 확정되면 `env.yaml`에 반영한 뒤 필요 시만 이미지 재빌드
* [X] baseline metric 재현 (LPIPS / DreamSim / FID)
  - 범위: `RECON + configs/experiment/nwm_cdit_s.yaml + 0100000 + time/rollout eval`
  - GT PNG `2500`장 생성 완료: `artifacts/lpips_time_recon_s/gt/recon/time`
  - 예측 PNG `2500`장 생성 완료: `artifacts/lpips_time_recon_s/nwm_cdit_s/recon/time`
  - 결과 JSON: `artifacts/lpips_time_recon_s/nwm_cdit_s/recon_time.json`
  - 현재 보존 경로: `artifacts/summaries/eval/lpips_time_recon_s/nwm_cdit_s/recon_time.json`
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
  - 캐시 조치: `scripts/docker/nwm-run.sh`가 레포 내부 `weights/cache`를 쓰도록 수정, 기존 AlexNet cache도 `weights/cache/torch/hub/checkpoints`로 복사
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
  - 현재 보존 경로: `artifacts/profiling/raw/gpu_profile_baseline/`

## 산출물

* baseline 성능 표 (무조건 필요)
* baseline inference latency log

이거 없으면 나중에 “좋아졌는지” 판단 자체가 불가능

---

# Phase 1 — 데이터 파이프라인 구축 (여기서 승부 50% 결정됨)

설명:

* 이 단계의 목적은 "학습 때 텍스트 인코더를 돌리지 않고", 프레임마다 미리 계산된 text embedding을 바로 읽게 만드는 것이다.
* 처음에 `1fps`를 쓴 이유는 Qwen caption 비용을 줄이기 위해서였다. 즉 `1fps`는 **caption 생성용 sparse sampling**이었다.
* 그런데 논문 기준으로는 학습 time axis를 `1fps`로 바꾸면 안 된다. `len_traj_pred=64`의 의미가 완전히 달라지기 때문이다.
* 그래서 현재 최종 방향은 이렇게 정리된다:
  - caption은 `1fps sparse`로 생성
  - embedding도 `1fps sparse`로 생성
  - 학습은 raw RECON dense trajectory를 그대로 사용
  - sparse text embedding을 raw dense frame axis로 broadcast/ffill 해서 붙임
* 즉 지금 파이프라인의 핵심은 `cheap captioning + dense training dynamics 유지`다.

진행 메모:

* [x] 오프라인 파이프라인 스크립트 추가 완료
* [x] 학습/추론 코드에 cached text embedding 경로 연결 완료
* [x] raw RECON subset 기준 end-to-end smoke 실행 완료
* [x] full dataset Qwen caption run 완료
* [x] full dataset clean + embedding 완료
* [x] raw RECON dense axis로 text embedding alignment 완료
* [x] raw RECON + dense text cache + `len_traj_pred=64` smoke 완료
* [x] raw RECON + dense text cache용 학습 config 추가 완료
* [x] 실제 text-conditioned 학습 run 시작 및 `0030000` checkpoint/eval 확보
* [x] Phase 1 산출물 경로 1차 문서화 완료
* [ ] Phase 1 산출물 경로를 발표/논문 표와 최종 동기화

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
* [x] 도커에서 외부 모델 디렉터리 mount 가능하도록 `scripts/docker/nwm-start.sh` 확장
* [x] `openai/clip-vit-base-patch32` HF cache 다운로드 완료
* [x] raw RECON subset (`2 traj x 4 frames`) Qwen caption smoke 완료
* [x] exported RECON test split full manifest 생성 완료 (`13193` records)
* [x] exported RECON train split full manifest 생성 완료 (`53395` records)
* [x] Qwen2-VL 실제 full-dataset caption run 완료
  - test split: `16 / 16` shards 완료, merged output 생성
  - train split: `64 / 64` shards 완료, merged output 생성
  - merged caption output:
    - `/datasets/derived/phase1_qwen/recon_test_1fps/all.jsonl`
    - `/datasets/derived/phase1_qwen/recon_train_1fps/all.jsonl`

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
* [x] full dataset cleaned caption 결과로 embedding precompute 실행
* [x] `1fps sparse embedding -> raw dense embedding` 정렬 스크립트 구현
* [x] raw RECON test dense text cache 생성 완료 (`2367` traj)
* [x] raw RECON train dense text cache 생성 완료 (`9468` traj)
* [x] raw RECON + dense text cache 기준 `len_traj_pred=64` smoke 완료

## 현재 실제로 만들어진 것

* `1fps caption merged JSONL`
  - `datasets/derived/phase1_qwen/recon_test_1fps/all.jsonl`
  - `datasets/derived/phase1_qwen/recon_train_1fps/all.jsonl`
* `1fps cleaned caption JSONL`
  - `datasets/derived/phase1_qwen_clean/recon_test_1fps_clean.jsonl`
  - `datasets/derived/phase1_qwen_clean/recon_train_1fps_clean.jsonl`
* `1fps sparse text embedding cache`
  - `datasets/derived/phase1_text_embeds/recon_test_1fps`
  - `datasets/derived/phase1_text_embeds/recon_train_1fps`
* `raw dense text embedding cache`
  - `datasets/derived/phase1_text_embeds_dense/recon_test_raw`
  - `datasets/derived/phase1_text_embeds_dense/recon_train_raw`
  - `datasets/derived/phase1_text_embeds_dense/recon_all_raw_rel`
* `raw+dense 학습 config`
  - `configs/experiment/nwm_cdit_s_recon_raw_text_dense.yaml`
  - `configs/experiment/nwm_cdit_b_recon_raw_text_dense.yaml`

## 논문용 최종 데이터 경로

* `raw RECON`
  - `datasets/recon_raw/recon_release`
* `dense text cache`
  - `datasets/derived/phase1_text_embeds_dense/recon_all_raw_rel`
* `training config`
  - `configs/experiment/nwm_cdit_s_recon_raw_text_dense.yaml`
  - `configs/experiment/nwm_cdit_b_recon_raw_text_dense.yaml`
* 경로 표기 원칙
  - `/workspace/nwm/...` 대신 repo-relative 경로만 canonical로 사용

## 산출물

* [x] frame + action + text_embedding aligned dataset의 실제 캐시 생성 완료
* [x] raw RECON test split에 대해 1fps processed dataset export
* [x] raw RECON train split에 대해 1fps processed dataset export
* [x] raw RECON test/train full caption merged JSONL 생성
* [x] raw RECON dense trajectory용 text embedding cache 생성
* [x] 224 text-conditioned 학습/eval 결과물 확보
  - checkpoint/log는 `30k`까지 확보
  - RECON `time/rollout` eval은 확보
  - planning eval 최종 JSON도 `artifacts/bulk/planning/recon224_s_text_dense_0030000_full_n32/...`에 확보
  - 단, planning 수치는 no-text 대비 우세하지 않아 repeat/seed 재평가 필요
* [x] `224 baseline no-text` vs `224 text-conditioned` 비교 그래프 생성
  - 스크립트: `scripts/analysis/plot_baseline_text_comparison.py`
  - 산출물: `artifacts/profiling/compare_baseline_text/01_recon_metrics_224_text_vs_no_text.png`
  - 주의: 이 그림은 저장된 JSON을 정확히 시각화한 것이지만 공정한 text ablation은 아님
    - baseline은 멀티데이터셋 `nwm_cdit_s`
    - text 쪽은 `nwm_cdit_s`에서 시작해 `raw RECON`으로 추가 학습한 `nwm_cdit_s_recon_raw_text_dense`
* [ ] 공정한 `224 text vs no-text` 비교용 control 확보
  - 필요한 것: `224 no-text RECON-only` run/eval
  - 조건: `datasets/recon_raw/recon_release`, 동일 split, 동일 `len_traj_pred=64`, 동일 checkpoint warm-start, text only on/off 차이만 남기기
  - 현재 대안: 224 original baseline과 224 text fine-tune 비교는 가능하지만, 엄밀한 ablation으로 쓰기에는 약함

## 오늘 정리 완료

* [x] `configs/experiment/nwm_cdit_s_recon_raw_text_dense.yaml`로 실제 학습 1회 시작
  - 실행 일시: `2026-03-22 19:39 KST`
  - 컨테이너: `NWM_GPU_REQUEST='device=0'`, `NWM_SHM_SIZE='16g'`
  - 실행 커맨드: `./scripts/docker/nwm-run.sh "python scripts/train.py --config configs/experiment/nwm_cdit_s_recon_raw_text_dense.yaml --log-every 10 --ckpt-every 500 --eval-every 1000000"`
  - 런 로그: `logs/nwm_cdit_s_recon_raw_text_dense/log.txt`
  - stdout: `logs/nwm_cdit_s_recon_raw_text_dense/stdout_live_20260322_193908.log`
* [x] 학습 안정성 확인 후 `B/L/XL` 중 어떤 스케일로 갈지 결정
  - 현재 `S`는 step `30000` checkpoint까지 NaN/worker crash 없이 안정
  - loss: 초반 `0.1396 -> 0.1496 -> 0.1231`, `30k` 시점 `0.1212`
  - throughput: warmup 이후 약 `2.12 ~ 2.15 steps/s`
  - GPU 0 메모리: 약 `19.7 GiB`
  - 다음 스케일은 `B`
  - 이유: 현재 text-conditioned RECON-only 조건을 유지한 동형 확장이고, 기존 `configs/experiment/nwm_cdit_{b,l,xl}.yaml`는 데이터/텍스트 조건이 달라 직접 비교용이 아님
  - 준비한 다음 config: `configs/experiment/nwm_cdit_b_recon_raw_text_dense.yaml`
* [x] `30k` 도달 후 프로세스 정지
  - `0030000.pth.tar` 저장 확인 후 학습 종료
  - 최종 checkpoint: `weights/checkpoints/nwm_cdit_s_recon_raw_text_dense/0030000.pth.tar`
  - 추가 checkpoint: `0005000`, `0010000`, `0015000`, `0020000`, `0025000`, `0030000`, `latest`
  - 현재 train/watcher 프로세스는 모두 종료됨
* [x] TODO와 diary에 "논문용 최종 데이터 경로"를 한 번 더 정리
* [x] `0030000` checkpoint 기준 RECON `time/rollout` eval 완료
  - 예측 경로: `artifacts/eval_s_recon_raw_text_dense/nwm_cdit_s_recon_raw_text_dense_0030000`
  - 결과 JSON:
    - `artifacts/eval_s_recon_raw_text_dense/nwm_cdit_s_recon_raw_text_dense_0030000/recon_time.json`
    - `artifacts/eval_s_recon_raw_text_dense/nwm_cdit_s_recon_raw_text_dense_0030000/recon_rollout_1fps.json`
    - `artifacts/eval_s_recon_raw_text_dense/nwm_cdit_s_recon_raw_text_dense_0030000/recon_rollout_4fps.json`
  - `time`:
    - LPIPS `1s 0.3084`, `2s 0.3382`, `4s 0.3658`, `8s 0.4002`, `16s 0.4577`
    - DreamSim `1s 0.1341`, `2s 0.1447`, `4s 0.1590`, `8s 0.1728`, `16s 0.2244`
    - FID `1s 33.98`, `2s 33.80`, `4s 33.07`, `8s 31.99`, `16s 38.19`
  - `rollout 1fps`:
    - LPIPS `1s 0.3135`, `2s 0.3561`, `4s 0.3977`, `8s 0.4610`, `16s 0.5651`
    - DreamSim `1s 0.1340`, `2s 0.1526`, `4s 0.1769`, `8s 0.2317`, `16s 0.3444`
    - FID `1s 60.83`, `2s 65.76`, `4s 63.41`, `8s 74.66`, `16s 84.86`
  - `rollout 4fps`:
    - LPIPS `1s 0.3327`, `2s 0.3667`, `4s 0.4197`, `8s 0.4867`, `16s 0.5472`
    - DreamSim `1s 0.1426`, `2s 0.1607`, `4s 0.1947`, `8s 0.2655`, `16s 0.3303`
    - FID `1s 62.68`, `2s 63.45`, `4s 67.69`, `8s 76.93`, `16s 79.95`
  - baseline `artifacts/lpips_time_recon_s/nwm_cdit_s` 대비 `time` 전 구간 개선, `rollout`도 대부분 개선
* [x] planning eval 최종 JSON 확보
  - 224 no-text: `artifacts/bulk/planning/recon224_s_notext_0100000_full_n32/nwm_cdit_s/recon_CEM_N32_K5_RS1_rep1_OPT1.json`
    - `recon_ate 1.1689`, `recon_rpe_trans 0.3574`, `recon_pos_diff_norm 1.7585`, `recon_yaw_diff_norm 0.3274`
  - 224 text: `artifacts/bulk/planning/recon224_s_text_dense_0030000_full_n32/nwm_cdit_s_recon_raw_text_dense/recon_CEM_N32_K5_RS1_rep1_OPT1.json`
    - `recon_ate 1.2480`, `recon_rpe_trans 0.3713`, `recon_pos_diff_norm 1.8797`, `recon_yaw_diff_norm 0.3759`
  - 해석: image metric은 text가 개선되지만, planning metric은 현재 단일 run 기준 no-text가 더 좋음
  - 남은 일: 동일 setting으로 repeat/seed를 늘려 planning 결론을 확정
* [x] `128x128` low-resolution baseline warm-start 경로 추가
  - 목적: 기존 `224` baseline `nwm_cdit_s`를 `128` 입력용으로 적응시켜 해상도 축소 baseline 확보
  - 코드 조치: `misc.py`에 `build_transform(image_size)` 추가
  - 코드 조치: `train.py`, inference/planning 경로, `tests/smoke/recon_smoke_test.py` 등에서 `config["image_size"]` 기반 transform 사용
  - 코드 조치: `train.py`에 checkpoint warm-start helper 추가
    - `checkpoint_ignore_keys`
    - `checkpoint_ignore_shape_mismatch`
    - `checkpoint_interpolate_pos_embed`
  - 새 config:
    - `configs/experiment/nwm_cdit_s_recon_128.yaml`
  - resume은 별도 config 없이 같은 config에서 `latest.pth.tar`를 우선 로드
  - warm-start source checkpoint: `weights/checkpoints/nwm_cdit_s/0100000.pth.tar`
  - 초기 정책: `pos_embed`는 제외하고 나머지 weight 재사용
  - smoke 확인: `obs_shape (4, 3, 128, 128)`, `pred_shape (64, 3, 128, 128)`, `delta_shape (64, 3)`
* [x] `128` baseline `0005000` checkpoint 기준 RECON `time/rollout` eval 완료
  - checkpoint: `weights/checkpoints/nwm_cdit_s_recon_128/0005000.pth.tar`
  - GT 재사용: `artifacts/eval_s_recon_128/gt_latest`
  - 예측 경로: `artifacts/eval_s_recon_128/nwm_cdit_s_recon_128_0005000`
  - 결과 JSON:
    - `artifacts/eval_s_recon_128/nwm_cdit_s_recon_128_0005000/recon_time.json`
    - `artifacts/eval_s_recon_128/nwm_cdit_s_recon_128_0005000/recon_rollout_1fps.json`
    - `artifacts/eval_s_recon_128/nwm_cdit_s_recon_128_0005000/recon_rollout_4fps.json`
  - `time`:
    - LPIPS `1s 0.6231`, `2s 0.6230`, `4s 0.6213`, `8s 0.6230`, `16s 0.6242`
    - DreamSim `1s 0.6918`, `2s 0.6962`, `4s 0.7008`, `8s 0.7057`, `16s 0.7072`
    - FID `1s 242.05`, `2s 243.56`, `4s 244.55`, `8s 251.10`, `16s 251.20`
  - `rollout 1fps`:
    - LPIPS `1s 0.6263`, `2s 0.6263`, `4s 0.6351`, `8s 0.6434`, `16s 0.6320`
    - DreamSim `1s 0.6912`, `2s 0.6895`, `4s 0.6889`, `8s 0.6875`, `16s 0.6889`
    - FID `1s 250.55`, `2s 248.42`, `4s 249.85`, `8s 240.89`, `16s 244.43`
  - `rollout 4fps`:
    - LPIPS `1s 0.6397`, `2s 0.6496`, `4s 0.6516`, `8s 0.6489`, `16s 0.6475`
    - DreamSim `1s 0.6798`, `2s 0.6838`, `4s 0.6838`, `8s 0.6893`, `16s 0.6853`
    - FID `1s 245.04`, `2s 242.56`, `4s 240.79`, `8s 242.82`, `16s 233.28`
  - 비교:
    - `~550 step latest` 대비 `time 15/15`, `rollout 1fps 12/15`, `rollout 4fps 10/15` 지표 개선
    - 기존 `224` baseline `artifacts/lpips_time_recon_s/nwm_cdit_s` 대비는 여전히 `45/45` 지표 전부 열세
* [x] `128` baseline `0010000` checkpoint 저장 후 학습 일단 정지
  - resume: `configs/experiment/nwm_cdit_s_recon_128.yaml`가 `latest.pth.tar`를 우선 로드
  - `2026-03-26 04:51:05`에 `step=0010000` 도달
  - 저장 확인: `weights/checkpoints/nwm_cdit_s_recon_128/0010000.pth.tar`
  - 저장 후 잠깐 더 진행되어 마지막 확인 로그는 `step=0010540`
  - 사용자 요청 기준이 `10k`라 컨테이너 내부 train 프로세스는 여기서 정지
* [x] `128` baseline `0010000` checkpoint 기준 RECON `time/rollout` eval
  - checkpoint: `weights/checkpoints/nwm_cdit_s_recon_128/0010000.pth.tar`
  - GT 재사용: `artifacts/eval_s_recon_128/gt_latest`
  - 예측 출력: `artifacts/eval_s_recon_128/nwm_cdit_s_recon_128_0010000`
  - 결과 JSON:
    - `artifacts/eval_s_recon_128/nwm_cdit_s_recon_128_0010000/recon_time.json`
    - `artifacts/eval_s_recon_128/nwm_cdit_s_recon_128_0010000/recon_rollout_1fps.json`
    - `artifacts/eval_s_recon_128/nwm_cdit_s_recon_128_0010000/recon_rollout_4fps.json`
  - `time`:
    - LPIPS `1s 0.5388`, `2s 0.5450`, `4s 0.5441`, `8s 0.5415`, `16s 0.5563`
    - DreamSim `1s 0.5335`, `2s 0.5359`, `4s 0.5242`, `8s 0.5192`, `16s 0.5378`
    - FID `1s 161.81`, `2s 156.58`, `4s 147.42`, `8s 148.16`, `16s 145.05`
  - `rollout 1fps`:
    - LPIPS `1s 0.5353`, `2s 0.5600`, `4s 0.5894`, `8s 0.6019`, `16s 0.6122`
    - DreamSim `1s 0.5362`, `2s 0.5540`, `4s 0.5968`, `8s 0.6249`, `16s 0.6425`
    - FID `1s 176.37`, `2s 190.25`, `4s 201.29`, `8s 216.36`, `16s 212.15`
  - `rollout 4fps`:
    - LPIPS `1s 0.5884`, `2s 0.6045`, `4s 0.6092`, `8s 0.6080`, `16s 0.6044`
    - DreamSim `1s 0.5983`, `2s 0.6297`, `4s 0.6182`, `8s 0.6352`, `16s 0.6450`
    - FID `1s 205.14`, `2s 211.46`, `4s 209.80`, `8s 215.36`, `16s 209.55`
  - 비교:
    - `0005000` 대비 `time 15/15`, `rollout 1fps 15/15`, `rollout 4fps 15/15` 지표 전부 개선
    - 기존 `224` baseline `artifacts/lpips_time_recon_s/nwm_cdit_s` 대비는 여전히 `45/45` 지표 전부 열세
* [x] `128` baseline no-text 추가 학습 및 checkpoint sweep 확보
  - checkpoint: `weights/checkpoints/nwm_cdit_s_recon_128/0015000` ~ `0055000`, `latest`
  - 마지막 확인 로그: `logs/nwm_cdit_s_recon_128/log.txt` 기준 `step=0057540`
  - time eval summary:
    - `artifacts/summaries/eval/eval_s_recon_128/nwm_cdit_s_recon_128_0015000/recon_time.json`
    - `artifacts/summaries/eval/eval_s_recon_128/nwm_cdit_s_recon_128_0020000/recon_time.json`
    - `artifacts/summaries/eval/eval_s_recon_128/nwm_cdit_s_recon_128_0025000/recon_time.json`
    - `artifacts/summaries/eval/eval_s_recon_128/nwm_cdit_s_recon_128_0030000/recon_time.json`
    - `artifacts/summaries/eval/eval_s_recon_128/nwm_cdit_s_recon_128_0035000/recon_time.json`
    - `artifacts/summaries/eval/eval_s_recon_128/nwm_cdit_s_recon_128_0040000/recon_time.json`
    - `artifacts/summaries/eval/eval_s_recon_128/nwm_cdit_s_recon_128_0045000/recon_time.json`
    - `artifacts/summaries/eval/eval_s_recon_128/nwm_cdit_s_recon_128_0050000/recon_time.json`
    - `artifacts/summaries/eval/eval_s_recon_128/nwm_cdit_s_recon_128_0055000/recon_time.json`
  - plot: `artifacts/profiling/recon_time_over_checkpoints/01_recon_time_over_checkpoints.png`
  - 남은 일: `0055000` 기준 rollout eval을 text와 같은 protocol로 정리
* [x] `128 + text` 학습 run 시작 및 `0030000` checkpoint 확보
  - config:
    - `configs/experiment/nwm_cdit_s_recon_128_text_dense.yaml`
  - resume은 별도 config 없이 같은 config에서 `latest.pth.tar`를 우선 로드
  - warm-start source:
    - `weights/checkpoints/nwm_cdit_s_recon_128/0010000.pth.tar`
  - 로딩 시 missing key:
    - `text_proj.weight`
    - `text_proj.bias`
  - 의미:
    - `128 baseline` 본체는 재사용하고 text projection만 새로 초기화
  - 학습 진행:
    - text smoke test 통과 (`text_shape (4, 512)`, latent `16x16`)
    - `0030000.pth.tar` 저장 확인
    - 실제 중단 시점 로그는 `step=0030320`
  - checkpoint 경로:
    - `weights/checkpoints/nwm_cdit_s_recon_128_text_dense/0030000.pth.tar`
* [x] `128 + text` `0030000` checkpoint 기준 RECON `time/rollout` eval
  - checkpoint:
    - `weights/checkpoints/nwm_cdit_s_recon_128_text_dense/0030000.pth.tar`
  - GT 재사용:
    - `artifacts/eval_s_recon_128/gt_latest`
  - 예측 출력:
    - `artifacts/eval_s_recon_128_text_dense/nwm_cdit_s_recon_128_text_dense_0030000`
  - 결과 JSON:
    - `artifacts/eval_s_recon_128_text_dense/nwm_cdit_s_recon_128_text_dense_0030000/recon_time.json`
    - `artifacts/eval_s_recon_128_text_dense/nwm_cdit_s_recon_128_text_dense_0030000/recon_rollout_1fps.json`
    - `artifacts/eval_s_recon_128_text_dense/nwm_cdit_s_recon_128_text_dense_0030000/recon_rollout_4fps.json`
  - `time`:
    - LPIPS `1s 0.2466`, `2s 0.2695`, `4s 0.2924`, `8s 0.3274`, `16s 0.3873`
    - DreamSim `1s 0.1710`, `2s 0.1797`, `4s 0.1887`, `8s 0.2064`, `16s 0.2495`
    - FID `1s 32.77`, `2s 32.37`, `4s 33.42`, `8s 31.17`, `16s 36.69`
  - `rollout 1fps`:
    - LPIPS `1s 0.2477`, `2s 0.2754`, `4s 0.3272`, `8s 0.3915`, `16s 0.4711`
    - DreamSim `1s 0.1699`, `2s 0.1793`, `4s 0.2046`, `8s 0.2460`, `16s 0.3354`
    - FID `1s 59.11`, `2s 60.23`, `4s 60.75`, `8s 64.34`, `16s 75.81`
  - `rollout 4fps`:
    - LPIPS `1s 0.2655`, `2s 0.2971`, `4s 0.3453`, `8s 0.4140`, `16s 0.4717`
    - DreamSim `1s 0.1767`, `2s 0.1933`, `4s 0.2124`, `8s 0.2684`, `16s 0.3306`
    - FID `1s 59.24`, `2s 61.02`, `4s 62.64`, `8s 69.23`, `16s 72.90`
  - 비교:
    - `128 no-text 0010000` 대비 `time 15/15`, `rollout 1fps 15/15`, `rollout 4fps 15/15` 지표 전부 개선
    - 기존 `224` baseline `artifacts/lpips_time_recon_s/nwm_cdit_s` 대비 `time 10/15`, `rollout 1fps 12/15`, `rollout 4fps 13/15` 지표 개선
  - 현재 caveat:
    - `128 no-text`는 이후 `0055000`까지 더 학습되어 있음
    - 공정한 text 효과를 말하려면 `128 text`도 같은 step까지 학습하거나, 양쪽 모두 `0030000` checkpoint 기준으로 비교해야 함
* [x] `128` planning JSON 확보
  - 128 no-text `0030000`: `artifacts/bulk/planning/recon128_s_notext_0030000_full_n32/nwm_cdit_s_recon_128/recon_CEM_N32_K5_RS1_rep1_OPT1.json`
    - `recon_ate 1.2801`, `recon_rpe_trans 0.3785`, `recon_pos_diff_norm 1.9892`, `recon_yaw_diff_norm 0.3748`
  - 128 text `0030000`: `artifacts/bulk/planning/recon128_s_text_dense_0030000_full_n32/nwm_cdit_s_recon_128_text_dense/recon_CEM_N32_K5_RS1_rep1_OPT1.json`
    - `recon_ate 1.3408`, `recon_rpe_trans 0.3785`, `recon_pos_diff_norm 2.0168`, `recon_yaw_diff_norm 0.3061`
  - 128 no-text `0055000`: `artifacts/bulk/planning/recon128_s_0055000/nwm_cdit_s_recon_128/recon_CEM_N32_K5_RS1_rep1_OPT1.json`
    - `recon_ate 1.4922`, `recon_rpe_trans 0.4158`, `recon_pos_diff_norm 2.3959`, `recon_yaw_diff_norm 0.6101`
  - 해석: planning은 현재 noisy하고 설정별 변동이 커서 단일 JSON만으로 결론 내리면 위험함

핵심: **학습 때 텍스트 인코더 절대 돌리지 마라 (속도 병목 터짐)**

---

# Phase 2 — 모델 결합 (Text Conditioning)

## 2.1 최소 변경 버전 (무조건 먼저)

* [x] text → projection layer 추가
* [x] 기존 conditioning vector에 단순 concat or sum
  - 현재 구현은 `sum` 방식
  - 상태: `datasets.py` -> `models.py` -> `train.py` / `isolated_nwm_infer.py` / `planning_eval.py` 경로까지 연결 완료

## 2.2 안정성 장치

* [ ] projection layer zero init
* [ ] text dropout (p=0.3~0.5)
* [ ] 일부 샘플 text 제거
  - 메모: 현재 `text_proj`는 zero init이 아니라 normal init이고, text dropout / text masking은 아직 없음

## 2.3 gated fusion (2차)

* [ ] gating scalar or MLP 추가
* [ ] text influence 조절
  - 메모: 현재는 gating 없이 conditioning vector에 text projection을 직접 더하는 최소 구현만 있음

## 구조 목표

```
cond = f(image, action, timestep, text)
```

핵심: **NWM을 깨지 말고 “옆에 붙인다”**

---

# Phase 3 — PoC (진짜 중요한 첫 검증)

## 실험

* [x] Base NWM
  - 224 original baseline: `nwm_cdit_s`
  - 128 RECON-only baseline: `nwm_cdit_s_recon_128`
* [x] + Scene text
  - 현재 dense cache는 Qwen scene caption 기반
  - 224 text: `nwm_cdit_s_recon_raw_text_dense`
  - 128 text: `nwm_cdit_s_recon_128_text_dense`
* [ ] + Goal text
  - goal-conditioned caption/embedding source는 아직 별도 실험 없음
* [ ] + Scene+Goal
  - scene-only 효과를 공정 비교로 먼저 고정한 뒤 진행

## 평가

* [x] LPIPS / DreamSim / FID
  - 224 no-text/text, 128 no-text/text의 time/rollout metric 확보
  - PSNR은 현재 주요 metric 세트에 없음
* [x] rollout stability (1s / 2s / 4s)
  - `rollout_1fps`, `rollout_4fps` metric 확보
* [ ] planning stability
  - JSON은 확보했지만 text 개선 결론은 아직 불확실

## 판단 기준

* image metric에서는 텍스트가 도움 되는 신호가 확인됨
* planning metric에서는 아직 결론 보류
* 다음 판단 기준은 **같은 checkpoint/step/seed 조건에서 text가 일관되게 도움 되냐**

여기서 효과 없으면 방향 바꿔야 한다 (냉정하게)

---

# Phase 4 — Resolution Scaling (이게 핵심 논문 포인트)

## TODO

* [x] high/224 → 128 1차 실험
  - 224 no-text/text metric 확보
  - 128 no-text/text metric 확보
* [ ] `128` 공정 비교 재정렬
  - 옵션 A: `128 text`를 `0055000`까지 추가 학습 후 no-text `0055000`과 비교
  - 옵션 B: `128 no-text`와 `128 text` 모두 `0030000` 기준으로 time/rollout/planning 표를 고정
* [ ] 다음 해상도 실험
  - `112`
  - `64`
* [ ] 각 단계에서 no-text vs text 비교

## 분석

* [ ] degradation slope 계산
* [ ] text가 손실 얼마나 줄이는지 정량화
* [ ] `128 text`가 `224 no-text`와 어느 horizon/metric에서 비슷하거나 이기는지 표로 정리

## 목표

* low-res + text ≈ mid/high-res without text

이거 나오면 논문 먹힌다

---

# Phase 5 — OOD 일반화

## TODO

* [ ] GO Stanford 데이터셋 evaluation
* [ ] unseen scene rollout 테스트
* [ ] RECON in-domain에서 먼저 repeat/seed 안정성 확인

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

현재 상태: offline Qwen caption + cached CLIP embedding은 완료. Online/lightweight semantic extractor는 아직 시작 전.

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

현재 상태: NWM 단독 GPU profiling은 있음. Online text extractor까지 포함한 total system latency는 아직 없음.

## TODO

* [ ] 전체 pipeline latency 측정

  * image → text → NWM → output
* [ ] FPS 측정
* [ ] VRAM usage 기록
* [ ] 128 no-text vs 128 text의 NWM-only latency/VRAM도 같은 protocol로 측정

## 비교군

* [x] high-res NWM profiling 일부 확보
* [x] low-res no-text model/eval 일부 확보
* [ ] low-res + YOLO text
* [ ] low-res + Moondream
* [ ] low-res + cached text embedding의 NWM-only overhead

## 핵심 질문

* “텍스트 넣었더니 느려졌다” → 바로 실패

반드시 **total system 기준으로 이득 보여야 함**

---

# Phase 8 — Ablation (논문 완성 단계)

## TODO

* [ ] text source 비교 (VLM vs YOLO vs template)
* [ ] fusion 방식 비교
  - baseline: 현재 `sum(text_proj(text), cond)`
  - 다음 후보: zero-init text projection, gated fusion
* [ ] text noise robustness
* [ ] wrong tag injection 실험
* [ ] text dropout 효과

---

# Phase 9 — 결과 정리 & 메시지 만들기

## 필수 그래프

* [x] 224/128 text/no-text metric 비교 그래프 1차 생성
  - `artifacts/profiling/compare_recon_all_variants/01_recon_metrics_224_128_text_vs_no_text.png`
* [x] 128 checkpoint sweep graph 생성
  - `artifacts/profiling/recon_time_over_checkpoints/01_recon_time_over_checkpoints.png`
* [ ] resolution vs performance curve
  - 112/64 추가 후 완성
* [ ] latency vs performance tradeoff
  - online text extractor 포함 후 완성
* [ ] rollout degradation graph
  - 현재 JSON은 있으나 논문용 plot으로 재정리 필요
* [ ] OOD failure case visualization

## 핵심 주장 정리

* 텍스트 = 성능 향상이 아니라
  → **정보 압축 보조 채널**

---

# 최종 Deliverable (논문/발표 기준)

* [x] pipeline diagram 초안
  - `docs/proposal_assets/nwm_text_conditioning_mechanism.png`
* [x] architecture/result proposal asset 초안
  - `docs/proposal_assets/nwm_eval_results.png`
* [ ] architecture diagram (baseline vs ours) 최종본
* [ ] quantitative table (in-domain / OOD)
  - in-domain 초안 가능, OOD 없음
* [ ] efficiency table (latency / VRAM)
  - NWM-only profiling은 있음, total system 기준은 없음
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

> **“공정 비교 재정렬 → planning repeat → 112/64 확장 → lightweight text → OOD → 시스템 평가”**

이 순서 절대 바꾸지 마라.

---

# 다음 단계 추천 (너 기준으로 현실적인 루트)

1. 지금 바로
   → `128 no-text` vs `128 text`의 checkpoint/eval 조건 맞추기

2. 그 다음
   → planning repeat/seed 늘려서 결론 확정

3. 그 다음
   → `112`, `64` 해상도 확장 및 resolution scaling curve 작성

4. 이후
   → YOLO 기반 lightweight text pipeline과 total latency 평가
