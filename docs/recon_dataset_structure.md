# RECON Dataset HDF5 구조

## 요약
  주요 내용 요약:

  - 11,836개 .hdf5 파일 — Jackal UGV 야외 주행 데이터
  - 파일 당 구조: T개 시간스텝(~10 Hz)의 멀티센서 데이터
  - 센서 종류:
    - 스테레오 RGB 카메라 (JPEG 바이너리)
    - 열화상 (T, 32, 32)
    - 2D 라이다 (T, 360) — 360도, 1도 간격
    - 외부 IMU + Jackal 내장 IMU
    - GPS (T, 2) + 오도메트리 position/yaw
    - 속도 명령 vs 실제 속도
    - 충돌 상태 6가지 플래그

## 개요

**RECON (Robot Exploration with Contextual and Offline Navigation)**은 Clearpath Jackal UGV(지상 로봇)로 수집한 야외 주행 데이터셋입니다.

| 속성 | 값 |
|---|---|
| 파일 경로 | `datasets/recon_raw/recon_release/*.hdf5` |
| 파일 수 | 11,836개 |
| 로봇 플랫폼 | Clearpath Jackal UGV |
| 수집 기간 | 2019년 8~10월 |
| 샘플링 | 약 10 Hz |

## 파일명 패턴

```
jackal_<YYYY-MM-DD-HH-MM-SS>_<session_id>_r<replay_id>.hdf5
```

- 같은 세션(`session_id`)이 여러 replay(`r00`, `r01`, ...)로 분할될 수 있음
- 예: `jackal_2019-08-02-16-23-30_0_r00.hdf5`

---

## HDF5 파일 내부 구조

각 파일은 **T개의 시간스텝**을 담은 계층적 구조. 아래는 실제 파일 기준 예시 (`T=70`).

```
root/
├── android/
│   └── illuminance           (T,)        float64   조도 [lux]
│
├── collision/
│   ├── any                   (T,)        bool      충돌 여부 (OR of below)
│   ├── physical              (T,)        bool      물리적 충돌
│   ├── close                 (T,)        bool      장애물 근접
│   ├── flipped               (T,)        bool      로봇 전복
│   ├── stuck                 (T,)        bool      로봇 고착
│   └── outside_geofence      (T,)        bool      지오펜스 이탈
│
├── commands/
│   ├── linear_velocity       (T,)        float64   속도 명령 [m/s]
│   └── angular_velocity      (T,)        float64   회전 명령 [rad/s]
│
├── gps/
│   ├── latlong               (T, 2)      float64   [위도, 경도] [deg]
│   ├── altitude              (T,)        float64   고도 [m]
│   ├── velocity              (T, 3)      float64   GPS 속도 벡터 [m/s]
│   └── is_fixed              (T,)        bool      GPS fix 여부
│
├── images/
│   ├── rgb_left              (T,)        |S~67KB   왼쪽 카메라 (JPEG 바이너리)
│   ├── rgb_right             (T,)        |S~61KB   오른쪽 카메라 (JPEG 바이너리)
│   └── thermal               (T, 32, 32) float64   열화상 이미지
│
├── imu/                                            외부 IMU 센서
│   ├── linear_acceleration   (T, 3)      float64   가속도 [m/s²] (x, y, z)
│   ├── angular_velocity      (T, 3)      float64   각속도 [rad/s] (x, y, z)
│   ├── magnetometer          (T, 3)      float64   자기장 (x, y, z)
│   └── compass_bearing       (T,)        float64   나침반 방위각 [rad]
│
├── jackal/                                         Jackal 내장 센서 / 오도메트리
│   ├── linear_velocity       (T,)        float64   실제 선속도 [m/s]
│   ├── angular_velocity      (T,)        float64   실제 각속도 [rad/s]
│   ├── position              (T, 3)      float64   오도메트리 위치 (x, y, z) [m]
│   ├── yaw                   (T,)        float64   오도메트리 요각 [rad]
│   └── imu/
│       ├── linear_acceleration (T, 3)    float64   내장 IMU 가속도 [m/s²]
│       └── angular_velocity    (T, 3)    float64   내장 IMU 각속도 [rad/s]
│
└── lidar                     (T, 360)    float64   2D 라이다 거리 [m], 1도 간격 360개
```

---

## 주요 데이터 상세

### 이미지 (`images/rgb_left`, `images/rgb_right`)

- **저장 형식:** JPEG 바이너리 (numpy `|S` dtype)
- **디코딩:** `bytes2im()` (PIL/OpenCV JPEG decode)
- **해상도:** 파일마다 다를 수 있음 (dtype size 기준 ~160×120 추정)
- **스테레오:** left/right 두 카메라

### 열화상 (`images/thermal`)

- **shape:** `(T, 32, 32)` — 저해상도 열화상
- **dtype:** float64 (온도 또는 raw DN 값)

### 라이다 (`lidar`)

- **shape:** `(T, 360)` — 360도 수평 스캔, 1도 간격
- **유효 범위:** 0~약 15 m, 유효하지 않은 측정값은 NaN/inf

### GPS (`gps/latlong`)

- **shape:** `(T, 2)` — `[latitude, longitude]`, 도(degree) 단위
- `gps/is_fixed == False`이면 신뢰 불가

### 오도메트리 (`jackal/position`, `jackal/yaw`)

- **position:** 시작점 기준 누적 오도메트리 (x, y, z) [m]
- **yaw:** 동쪽=0, 반시계 양수 [rad]

---

## 파일 읽기 예시

```python
import h5py
import numpy as np
from PIL import Image
import io

with h5py.File("jackal_2019-08-02-16-23-30_0_r00.hdf5", "r") as f:
    T = len(f["collision/any"])          # 시간스텝 수

    # RGB 이미지 디코딩
    jpeg_bytes = bytes(f["images/rgb_left"][0])
    img = np.array(Image.open(io.BytesIO(jpeg_bytes)))  # (H, W, 3) uint8

    # 오도메트리 궤적
    positions = f["jackal/position"][:]   # (T, 3)
    yaws      = f["jackal/yaw"][:]        # (T,)

    # 속도 명령
    cmd_v = f["commands/linear_velocity"][:]   # (T,)
    cmd_w = f["commands/angular_velocity"][:]  # (T,)

    # 충돌 마스크
    collision = f["collision/any"][:]          # (T,) bool

    # 라이다
    lidar = f["lidar"][:]                      # (T, 360)
```

---

## 관련 코드

| 파일 | 역할 |
|---|---|
| `datasets/recon_datavis/src/recon_datavis/hdf5_visualizer.py` | HDF5 데이터 시각화 도구 |
| `datasets/recon_datavis/src/recon_datavis/utils.py` | JPEG 바이너리 ↔ numpy 변환 |
| `datasets/recon_datavis/src/recon_datavis/gps/` | GPS 좌표 변환 및 지도 시각화 |
| `data_splits/` | train/val/test 분할 파일 목록 |
