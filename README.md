# ClassAuto_midterm

2026 성신여자대학교 자율이동체시스템 수업 중간고사 대체 과제  
KITTI Odometry Dataset을 이용한 Bayesian 기반 도로/차선 검출 및 카메라 기하학 분석

베이스 코드: [auto_class_2_BayesianClassifier.ipynb](https://github.com/yangjunahn/ClassAuto/blob/main/auto_class_2_BayesianClassifier.ipynb)

---

## 과제 개요

복잡한 딥러닝 모델 없이 픽셀 밝기 분포와 확률 모델만으로 도로를 추정하는 Bayesian 분류기를 구현하고,  
Projection Matrix를 활용하여 카메라 기하학적으로 해석한다.  
최종적으로 딥러닝 기반 차선 검출 모델(UFLD)과 결과를 비교한다.

---

## 데이터셋

[KITTI Odometry Dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) — sequence 09 사용

> 데이터셋은 용량 문제로 포함하지 않음. KITTI 공식 사이트에서 직접 다운로드 필요.

---

## 프로젝트 구조

```
ClassAuto_midterm/
├── bayes_road.py               # Bayesian 도로 검출 (사전 실행 필요)
├── problems.ipynb              # 문제 1~6 풀이 노트북
├── Ultra-Fast-Lane-Detection/  # UFLD 모델 소스코드
└── README.md
```

---

## 문제 구성

| 문제 | 내용 |
|------|------|
| 문제 1 | Projection Matrix 해석 (Intrinsic / Extrinsic / 투영 수식) |
| 문제 2 | 3D → 2D 투영 시각화 (도로 위 격자점 투영) |
| 문제 3 | 차량 궤적 시각화 (pose 파싱, 속도 분석) |
| 문제 4 | 차선 추출 및 P 행렬 해석 (Canny, Hough, 소실점) |
| 문제 5 | 실패 구간 분석 (road_ratio 기반 자동 탐지) |
| 문제 6 | UFLD vs Bayesian 비교 (딥러닝 차선 검출 모델 적용) |

---

## 사용 모델

**UFLD (Ultra Fast Lane Detection)**
- 논문: [Ultra Fast Structure-aware Deep Lane Detection (ECCV 2020)](https://arxiv.org/abs/2004.11757)
- Backbone: ResNet-18, pretrained on CULane dataset
- GitHub: [cfzd/Ultra-Fast-Lane-Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection)
