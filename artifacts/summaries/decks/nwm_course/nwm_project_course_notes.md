# NWM Project Course Deck Notes

이 노트는 PPT 각 슬라이드의 강의 의도를 짧게 정리한 보조 자료입니다.

## 01. Navigation World Models 프로젝트 강의

프로젝트 전체를 한 과목처럼 학습할 수 있도록 문제, 데이터, 모델, 학습, 평가, 실험 결과를 순서대로 정리한 덱.

## 02. 수업 로드맵

전체 학습 순서와 각 파트에서 얻어야 하는 직관.

## 03. 프로젝트 한 줄 정의

NWM이 풀고 싶은 문제를 정책 학습이 아니라 world modeling 관점에서 정의.

## 04. 이 프로젝트에서 우리가 추가로 묻는 질문

원본 NWM 위에 text conditioning을 얹은 이 fork의 연구 질문.

## 05. Repository Architecture

폴더 단위로 이 repo가 어떻게 나뉘는지.

## 06. Runtime Architecture

config, entrypoint, library module, artifact가 이어지는 실행 구조.

## 07. 데이터 단위: trajectory sample

TrainingDataset이 한 샘플에서 무엇을 만드는지.

## 08. RECON + Text Pipeline

이 fork에서 text-conditioned 실험을 위해 추가된 offline pipeline.

## 09. 무엇을 학습하는가

정책 학습이 아니라 조건부 미래 latent denoising을 학습한다는 점을 명확히 함.

## 10. Latent Diffusion 구조

이미지를 직접 diffusion하지 않고 VAE latent에서 학습하는 이유와 흐름.

## 11. CDiT 전체 아키텍처

src/models/backbones/cdit.py 기준 모델 구조.

## 12. CDiTBlock 내부

한 블록에서 target token과 context token이 어떻게 상호작용하는지.

## 13. Text Conditioning은 어디에 들어가나

텍스트가 이미지 token에 직접 붙는 것이 아니라 condition vector에 더해짐.

## 14. Training Loop

scripts/train.py의 실제 학습 흐름.

## 15. Config를 읽는 법

실험 YAML에서 가장 먼저 봐야 할 key.

## 16. Inference: time prediction vs rollout

scripts/infer.py가 생성하는 두 가지 평가용 prediction.

## 17. Prediction Metrics

이미지 생성 품질을 평가하는 metric.

## 18. Planning Evaluation: CEM

world model을 사용해 action 후보를 고르는 평가 방식.

## 19. Trajectory Metrics

Planning 결과는 이미지 품질이 아니라 trajectory error로 봅니다.

## 20. 현재 Paper-Style CEM 결과

방금 완료된 N120/K5/rep3/OPT1 full 100-sample 결과.

## 21. Frame Prediction 결과를 읽는 관점

텍스트 조건의 효과는 frame prediction에서 먼저 확인된다.

## 22. Planning 결과 해석: 왜 metric이 엇갈리나

B224 Text에서 ATE와 RPE가 다르게 움직이는 현상.

## 23. End-to-End 실행 Workflow

처음부터 결과 JSON까지 가는 실전 순서.

## 24. Smoke Test와 Debugging

큰 학습/평가 전에 확인해야 하는 작은 검증.

## 25. Artifacts와 실험 기록

실험 결과가 어디에 쌓이고 무엇을 commit/공유할지.

## 26. 코드 읽기 로드맵

수업 후 혼자 repo를 공부할 때 추천 순서.

## 27. Lab 1: 데이터셋 한 샘플 해부

첫 번째 실습 과제.

## 28. Lab 2: 모델 forward 따라가기

두 번째 실습 과제.

## 29. Lab 3: 평가 결과 재현

세 번째 실습 과제.

## 30. 시험에 나올 만한 질문

프로젝트 이해도를 스스로 점검하는 질문.

## 31. 핵심 Takeaways

프로젝트를 한 장으로 요약.
