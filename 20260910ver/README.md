# Autonomous Taxi Fleet Dynamic Matching & Demand Forecasting

이 프로젝트는 SUMO 기반 디지털 트윈 환경에서 H3 공간 인덱싱과 시계열 AI 모델(XGBoost / CNN-LSTM)을 연동하여 수요를 예측하고, Surge Pricing 및 헝가리안 알고리즘 기반 최적 동적 배차를 검증하는 통합 파이프라인입니다.

## 모듈 구성
- **Module 1 (`module1_simulation`)**: SUMO Grid 환경 설정 및 Digital Twin 가동
- **Module 2 (`module2_preprocessing`)**: GPS H3 매핑, 5분 리샘플링, Lag/Calendar 피처 및 외부 환경 데이터(날씨/휴일) 결합
- **Module 3 (`models.py`, `train.py`, `evaluate.py`)**: XGBoost(GridSearch 튜닝) & CNN-LSTM 모델 학습 및 RMSE/MAE/MAPE 평가
- **Module 4 (`module4_dispatch`)**: 수급 불균형 기반 Surge Pricing 및 헝가리안 최적 배차

## 실행 방법
1. **모델 학습 및 저장 (Train/Test 분리 및 하이퍼파라미터 튜닝)**
   ```bash
   python train.py