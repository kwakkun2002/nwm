# NWM Text-Conditioned RECON 진행 발표

## 1. 발표 개요

- 주제: `Navigation World Models (NWM)`에 텍스트 조건을 붙여 RECON에서 성능 개선 검증
- 발표 범위: `2026-03-19`부터 `2026-03-23`까지 완료된 작업 기준
- 핵심 메시지: 텍스트는 "있으면 좋은 부가정보"가 아니라, 실제 예측 품질을 개선하는 조건 신호로 동작함

이미지 경로: `./images/slide01_title_overview.png`
권장 이미지: baseline NWM 예측 샘플과 text-conditioned 예측 샘플을 나란히 둔 대표 비교 이미지

---

## 2. 문제 정의와 목표

- 원본 NWM은 이미지와 액션 중심의 navigation world model임
- 이번 작업의 목표는 RECON 데이터 기준으로 텍스트 정보를 붙여 더 나은 future frame prediction을 만드는 것임
- 프로젝트 질문은 "텍스트가 의미가 있나?"보다 "`offline text pipeline + cached embedding`이 실제 성능 개선으로 이어지나?"에 가까움

### 이번 단계에서 증명하려는 것

1. raw RECON 기반 학습/평가 파이프라인을 실제로 끝까지 돌릴 수 있다
2. Qwen caption + CLIP embedding을 NWM conditioning에 안정적으로 결합할 수 있다
3. 적어도 `CDiT-S/2` 스케일에서는 baseline 대비 정량 개선이 난다

이미지 경로: `./images/slide02_problem_goal.png`
권장 이미지: 전체 연구 질문을 보여주는 one-slide 다이어그램 또는 baseline vs ours 구조 비교도

---

## 3. 진행 타임라인

| 날짜 | 완료 내용 | 의미 |
| --- | --- | --- |
| `2026-03-19` | raw RECON 로딩, 컨테이너/평가 안정화, smoke test 완료 | 베이스라인 실행 가능 상태 확보 |
| `2026-03-20` | 1fps export, Qwen caption, cleaning, CLIP embedding, dense cache 생성 완료 | 텍스트 파이프라인 완성 |
| `2026-03-22` | `config/nwm_cdit_s_recon_raw_text_dense.yaml` 실제 학습 시작 및 `30k` 확보 | 학습 가능성 검증 |
| `2026-03-23` | `0030000` checkpoint 기준 time/rollout eval 완료 | 정량 개선 확인 |

### 현재 결론

- 환경 구축 단계는 끝났고
- 데이터 파이프라인도 끝났으며
- 최소 모델 스케일 `S`에서 개선까지 확인한 상태임

이미지 경로: `./images/slide03_timeline.png`
권장 이미지: 날짜별 milestone 타임라인

---

## 4. Phase 0 완료: 환경과 베이스라인 고정

- `NWM smallest variant` 실행 환경 확보
- raw RECON `.hdf5`를 직접 읽도록 `datasets.py` 확장
- 로컬 VAE 우선 로드, Docker runtime env 정리, `torchaudio` ABI mismatch 해결
- `isolated_nwm_infer.py`, `isolated_nwm_eval.py`, `planning_eval.py`가 실제 컨테이너 환경에서 돌도록 정비
- `scripts/recon/recon_smoke_test.py` 추가로 데이터셋 로딩 + 1-sample forward 재현 가능하게 정리

### 확보한 베이스라인 결과

- RECON `time` eval 완료
- RECON `rollout 1fps / 4fps` eval 완료
- baseline 결과 JSON 확보:
  - `artifacts/lpips_time_recon_s/nwm_cdit_s/recon_time.json`
  - `artifacts/lpips_time_recon_s/nwm_cdit_s/recon_rollout_1fps.json`
  - `artifacts/lpips_time_recon_s/nwm_cdit_s/recon_rollout_4fps.json`

이미지 경로: `./images/slide04_phase0_env_baseline.png`
권장 이미지: raw RECON 샘플, smoke output, Docker/eval 흐름을 합친 환경 안정화 요약 그림

---

## 5. Phase 1 완료: 데이터 및 텍스트 파이프라인 구축

### 데이터 규모

- raw RECON: 약 `72G`, `.hdf5 11836`개
- test split: `2367 traj`, `13193`개 `1fps` 프레임 export 완료
- train split: `9468 traj`, `53395`개 `1fps` 프레임 export 완료

### 텍스트 파이프라인 완료 항목

1. `Qwen2-VL-7B-Instruct` 기반 캡션 생성
2. boilerplate 제거 및 short sentence 정제
3. `openai/clip-vit-base-patch32` 기반 text embedding precompute
4. sparse `1fps` embedding을 raw trajectory dense cache로 정렬

### 최종 산출물

- merged captions:
  - `datasets/derived/phase1_qwen/recon_test_1fps/all.jsonl`
  - `datasets/derived/phase1_qwen/recon_train_1fps/all.jsonl`
- cleaned captions:
  - `datasets/derived/phase1_qwen_clean/recon_test_1fps_clean.jsonl`
  - `datasets/derived/phase1_qwen_clean/recon_train_1fps_clean.jsonl`
- dense text cache:
  - `datasets/derived/phase1_text_embeds_dense/recon_all_raw_rel`

이미지 경로: `./images/slide05_data_pipeline.png`
권장 이미지: `raw RECON -> 1fps export -> Qwen caption -> cleaning -> CLIP embedding -> dense cache` 파이프라인 도식

---

## 6. Phase 2 완료: NWM에 Text Conditioning 결합

### 코드 레벨 완료 사항

- `datasets.py`: cached text embedding 로딩, `current/goal/context_mean` source 지원
- `models.py`: `text_proj` 추가, text embedding을 conditioning vector에 합산
- `train.py`, `isolated_nwm_infer.py`, `planning_eval.py`: text input optional 전달 지원
- 학습 config 추가:
  - `config/nwm_cdit_s_recon_raw_text_dense.yaml`
  - `config/nwm_cdit_b_recon_raw_text_dense.yaml`

### 의미

- 온라인 VLM 추론 없이도
- offline caption과 precomputed embedding만으로
- 기존 NWM 구조를 크게 깨지 않고 text-conditioned 학습이 가능해졌음

이미지 경로: `./images/slide06_model_integration.png`
권장 이미지: 기존 conditioning path 옆에 text branch가 추가된 구조도

---

## 7. 실제 학습 결과: `S` 모델 30k 안정성 확인

### 실행 정보

- 실행 시작: `2026-03-22 19:39 KST`
- config: `config/nwm_cdit_s_recon_raw_text_dense.yaml`
- checkpoint cutoff: `30000`
- 최종 checkpoint:
  - `weights/checkpoints/nwm_cdit_s_recon_raw_text_dense/0030000.pth.tar`

### 학습 안정성

- `step 10 loss 0.1396`
- `step 100 loss 0.1274`
- `step 30000 loss 0.1212`
- 속도: 약 `2.13 steps/s`, `136.53 samples/s`
- 결론: `S + raw RECON + dense text cache` 조합은 실제 학습이 안정적으로 돌아감

### 다음 스케일

- 다음 후보는 `L/XL`이 아니라 `B`
- text-conditioned 파라미터 수:
  - `S`: `49,424,672`
  - `B`: `195,678,752`

이미지 경로: `./images/slide07_training_curve.png`
권장 이미지: loss curve, checkpoint 저장 시점, GPU 메모리 사용량 요약 그래프

---

## 8. 정량 결과 1: Time Eval은 전 구간 개선

대상 비교:

- baseline: `artifacts/lpips_time_recon_s/nwm_cdit_s/recon_time.json`
- ours: `artifacts/eval_s_recon_raw_text_dense/nwm_cdit_s_recon_raw_text_dense_0030000/recon_time.json`

| Horizon | Metric | Baseline | Ours | 변화 |
| --- | --- | --- | --- | --- |
| `8s` | LPIPS | `0.4031` | `0.4002` | 개선 |
| `8s` | DreamSim | `0.1784` | `0.1728` | 개선 |
| `8s` | FID | `33.46` | `31.99` | 개선 |
| `16s` | LPIPS | `0.4632` | `0.4577` | 개선 |
| `16s` | DreamSim | `0.2286` | `0.2244` | 개선 |
| `16s` | FID | `40.39` | `38.19` | 개선 |

### 요약

- `1s ~ 16s` 전 구간에서 LPIPS, DreamSim, FID 모두 개선
- 평균 기준으로도
  - LPIPS 약 `0.94%` 개선
  - DreamSim 약 `2.72%` 개선
  - FID 약 `3.52%` 개선

이미지 경로: `./images/slide08_time_eval.png`
권장 이미지: baseline vs ours time-horizon metric plot 또는 동일 trajectory 예측 비교 그리드

---

## 9. 정량 결과 2: Rollout도 개선, Planning은 부분 완료

### Rollout 핵심 결과

- `rollout 1fps`
  - `16s FID: 141.49 -> 84.86`로 큰 폭 개선
  - `8s FID: 102.75 -> 74.66` 개선
  - 평균 기준 DreamSim 약 `13.84%`, FID 약 `15.46%` 개선
- `rollout 4fps`
  - LPIPS / DreamSim / FID 전 구간 개선
  - 평균 기준 LPIPS 약 `4.20%`, DreamSim 약 `6.61%`, FID 약 `7.76%` 개선

### Planning 상태

- `0030000` checkpoint로 planning eval을 시작했고 partial prediction 저장 확인
- 초반 배치 지표 예시:
  - `recon_ate 1.6417`
  - `recon_rpe_trans 0.4050`
- 다만 최종 JSON 생성 전 중단되어 planning 결과는 아직 확정본이 아님

이미지 경로: `./images/slide09_rollout_planning.png`
권장 이미지: rollout 장기 예측 비교와 planning partial visualization을 함께 배치한 그림

---

## 10. 결론과 다음 단계

### 지금까지 증명된 것

1. raw RECON 기반 text-conditioned NWM 파이프라인이 end-to-end로 동작함
2. offline caption + cached embedding 방식으로 실제 학습과 평가가 가능함
3. `CDiT-S/2` 기준으로 image reconstruction 성능이 baseline보다 개선됨

### 아직 남은 것

- planning eval 최종 JSON 확보
- `CDiT-B/2` 안정성 run 및 동일 조건 비교
- 발표/논문용 qualitative figure 정리

### 발표용 한 줄 메시지

> 텍스트는 navigation world model에 붙일 수 있는 보조 신호가 아니라, RECON 기준으로 이미 정량 개선을 만든 실질적 conditioning 신호다.

이미지 경로: `./images/slide10_conclusion_next_steps.png`
권장 이미지: 전체 요약 로드맵 또는 "완료/진행중/다음 단계" 3단 구성 도식
