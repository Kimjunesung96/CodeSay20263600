# -*- coding: utf-8 -*-
"""
[통합] 저장된 수요 예측 모델(XGBoost vs CNN-LSTM) 성능 비교·추천 도구.

배경
- origin/jwy의 evaluate.py에 있던 evaluate_saved_models()를 develop 구조로 옮긴 것.
- jwy 버전은 랜덤 데모 데이터 + train_test_split 기준이라 그대로 쓸 수 없어,
  develop의 Module 2 파이프라인(시뮬레이션 로그 → 피처 테이블)과 시간 기준 분할로 바꿨다.
- 지표 계산은 develop의 evaluate.calculate_metrics를 그대로 사용 (RMSE/MAE/MAPE/WAPE).

전제 (train.py를 먼저 실행해 두 모델이 저장돼 있어야 함)
- saved_models/xgboost_demand.pkl : {'model', 'feature_cols'}
- saved_models/cnn_lstm_demand.pt : {'state_dict', 'input_dim', 'feature_cols', 'scaler'}
  ('scaler'는 train.py의 CNN 입력 표준화에서 저장. 없으면(구버전 파일) 스케일 없이 평가)

사용
    python scripts/compare_saved_models.py              # data/sim_logs/demand_log_*.csv 전체
    python scripts/compare_saved_models.py <로그CSV>     # 지정 로그로 평가
"""
import os
import sys

import joblib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config_loader import CFG                                        # noqa: E402
from evaluate import calculate_metrics                               # noqa: E402
from module2_preprocessing.pipeline import build_feature_table       # noqa: E402
from module2_preprocessing.time_series_prep import time_based_split  # noqa: E402

TARGET = "y_h1"  # train.py의 단일 출력 모델과 같은 타겟


def _load_test_set(log_arg=None):
    """train.py와 같은 파이프라인·같은 분할로 test 구간을 만든다."""
    logs = [log_arg] if log_arg else [os.path.join(ROOT, "data", "sim_logs", "demand_log_*.csv")]
    feature_df = build_feature_table(logs)
    feature_cols = feature_df.attrs["feature_cols"]
    _train, _val, test_df, cuts = time_based_split(feature_df, CFG["test_size"])
    test_df = test_df.sort_values(["time_bucket", "h3_index"])
    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df[TARGET].values
    print(f"[평가 데이터] test {len(test_df)}행 (>= {cuts['test_cut']}) | 피처 {len(feature_cols)}개 | 타겟 {TARGET}")
    return X_test, y_test, feature_cols


def _fmt(m):
    mape = f"{m['MAPE (%)']:.2f}%" if m.get("MAPE (%)") is not None else "-"
    wape = f"{m['WAPE (%)']:.2f}%" if m.get("WAPE (%)") is not None else "-"
    return f"RMSE {m['RMSE']:.4f} | MAE {m['MAE']:.4f} | MAPE {mape} | WAPE {wape}"


def evaluate_saved_models(log_arg=None):
    xgb_path = os.path.join(ROOT, "saved_models", "xgboost_demand.pkl")
    dl_path = os.path.join(ROOT, "saved_models", "cnn_lstm_demand.pt")
    if not (os.path.exists(xgb_path) and os.path.exists(dl_path)):
        print("[오류] 저장된 모델 파일이 없습니다. 먼저 python train.py 를 실행하세요.")
        return None

    X_test, y_test, feature_cols = _load_test_set(log_arg)

    # 1) XGBoost
    xgb_data = joblib.load(xgb_path)
    if xgb_data.get("feature_cols") != list(feature_cols):
        raise ValueError("xgboost_demand.pkl의 feature_cols가 현재 파이프라인과 다릅니다. train.py를 다시 실행하세요.")
    xgb_pred = xgb_data["model"].predict(X_test)
    xgb_metrics = calculate_metrics(y_test, xgb_pred)

    # 2) CNN-LSTM (torch는 이 시점에만 필요 — XGB만 볼 때 torch 미설치로 죽지 않게 지연 임포트)
    import torch
    from module3_prediction.models import CNNLSTMModel
    dl_data = torch.load(dl_path, weights_only=False)
    if dl_data.get("feature_cols") != list(feature_cols):
        raise ValueError("cnn_lstm_demand.pt의 feature_cols가 현재 파이프라인과 다릅니다. train.py를 다시 실행하세요.")
    model = CNNLSTMModel(input_dim=dl_data["input_dim"])
    model.load_state_dict(dl_data["state_dict"])
    model.eval()
    scaler = dl_data.get("scaler")
    X_dl = scaler.transform(X_test) if scaler is not None else X_test
    if scaler is None:
        print("[주의] 저장 파일에 scaler가 없어 표준화 없이 평가합니다(구버전 모델 파일).")
    with torch.no_grad():
        dl_pred = model(torch.tensor(X_dl, dtype=torch.float32).unsqueeze(1)).numpy().ravel()
    dl_metrics = calculate_metrics(y_test, dl_pred)

    # 3) 비교표 + 추천
    print("\n" + "=" * 78)
    print("[저장 모델 성능 비교 — 같은 test 구간]")
    print("-" * 78)
    print(f"  XGBoost  : {_fmt(xgb_metrics)}")
    print(f"  CNN-LSTM : {_fmt(dl_metrics)}")
    print("-" * 78)
    if xgb_metrics["RMSE"] <= dl_metrics["RMSE"]:
        print("[추천] XGBoost — 소규모 시계열 정형 데이터에서 더 낮은 RMSE(안정적).")
    else:
        print("[추천] CNN-LSTM — 이 데이터에서는 비선형 패턴 학습이 더 효과적.")
    print("=" * 78 + "\n")
    return {"XGBoost": xgb_metrics, "CNN-LSTM": dl_metrics}


if __name__ == "__main__":
    evaluate_saved_models(sys.argv[1] if len(sys.argv) > 1 else None)
