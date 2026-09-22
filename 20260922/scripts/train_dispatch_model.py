# -*- coding: utf-8 -*-
"""
[통합] 재배치(forecast 전략)용 30분 수요 예측 모델 학습.

배경
- Module 2 파이프라인(module2_preprocessing.pipeline)은 t+1~t+6 타겟(y_h1..y_h6)을 만들지만,
  train.py의 기본 학습은 단일 타겟(y_h1)이다.
- 재배치(forecast_dispatcher)는 30분(6구간) 예측 합을 쓰므로 6개 타겟을 동시에 내는 모델이 필요하다.
- train.py 주석의 확장 방법("y_h1..y_h6 전체를 y로 넘기고 MultiOutputRegressor(XGB)") 그대로 구현했다.

사용
    python scripts/train_dispatch_model.py                 # data/sim_logs/demand_log_*.csv 전체
    python scripts/train_dispatch_model.py <로그CSV ...>    # 지정 로그(와일드카드 가능)
    python scripts/train_dispatch_model.py <로그CSV> --external data/generated/external.csv
        # 합성 외부 관측(time_bucket×h3_index)을 시간별 날씨 표로 변환해 결합.
        # forecast_dispatcher가 시뮬레이션 중 쓰는 변환과 동일 → 학습-서빙 날씨 일치.

산출물: saved_models/demand_v2.joblib
    {version: 2, model(MultiOutput XGB), feature_cols, target_cols, cells, data_until,
     max_lag, rolling_short, rolling_long}
forecast_dispatcher.py가 이 계약(키·검증값)을 그대로 읽는다.
"""
import os
import sys
import json

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config_loader import CFG                                    # noqa: E402
from module2_preprocessing.pipeline import build_feature_table   # noqa: E402
from module2_preprocessing.time_series_prep import time_based_split  # noqa: E402
from evaluate import calculate_metrics, fmt_pct                           # noqa: E402

MODEL_PATH = os.path.join(ROOT, "saved_models", "demand_v2.joblib")


def external_to_weather_csv(external_csv: str) -> str:
    """합성 외부 관측을 시간별 날씨 CSV로 변환해 data/external에 저장하고 경로를 반환.

    forecast_dispatcher._external_to_weather와 같은 축약(시각별 셀 평균)이라
    학습과 시뮬레이션 중 예측이 같은 날씨 값을 보게 된다.
    """
    from module4_dispatch.forecast_dispatcher import _external_to_weather
    ext = pd.read_csv(external_csv, parse_dates=["time_bucket"])
    w = _external_to_weather(ext)
    if w is None:
        raise ValueError(f"{external_csv}: time_bucket/temperature 컬럼이 필요합니다")
    out = os.path.join(ROOT, "data", "external", "weather_from_external.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    w.to_csv(out, index=False)
    print(f"[외부 관측 → 날씨] {external_csv} → {out} ({len(w)}행)")
    return out


def main(log_paths=None, external_csv=None):
    logs = log_paths or [os.path.join(ROOT, "data", "sim_logs", "demand_log_*.csv")]
    weather_path = external_to_weather_csv(external_csv) if external_csv else None
    feat = build_feature_table(logs, weather_path=weather_path)
    features = feat.attrs["feature_cols"]
    targets = feat.attrs["target_cols"]
    if len(targets) < 6:
        raise ValueError(f"타겟이 6개가 아닙니다: {targets}")

    train_df, _val, test_df, cuts = time_based_split(feat, CFG["test_size"])
    X_tr, Y_tr = train_df[features].fillna(0).values, train_df[targets].values
    X_te, Y_te = test_df[features].fillna(0).values, test_df[targets].values
    print(f"[학습] {len(X_tr):,}행 (< {cuts['test_cut']}), 테스트 {len(X_te):,}행, "
          f"피처 {len(features)}개, 타겟 {targets}")

    # module3_prediction.models.build_xgboost_model과 같은 설정.
    # (models.py는 torch를 항상 import해서, torch 미설치 환경에서도 돌도록 여기서 직접 구성)
    model = MultiOutputRegressor(XGBRegressor(
        n_estimators=CFG["xgb_n_estimators"], max_depth=CFG["xgb_max_depth"],
        learning_rate=CFG["xgb_learning_rate"], random_state=CFG.get("xgb_random_state", 42)))
    model.fit(X_tr, Y_tr)

    pred = np.maximum(0, model.predict(X_te))
    per_h = {t: calculate_metrics(Y_te[:, i], pred[:, i]) for i, t in enumerate(targets)}
    overall = calculate_metrics(Y_te.ravel(), pred.ravel())
    print(f"[테스트 전체] RMSE {overall['RMSE']:.4f} | MAE {overall['MAE']:.4f} | MAPE {fmt_pct(overall['MAPE (%)'])} | WAPE {fmt_pct(overall['WAPE (%)'])}")
    for t, m in per_h.items():
        print(f"  {t}: RMSE {m['RMSE']:.4f}  MAE {m['MAE']:.4f}")

    artifact = {
        "version": 2,
        "model": model,
        "feature_cols": list(features),
        "target_cols": list(targets),
        "cells": sorted(feat["h3_index"].unique()),
        "data_until": str(feat["time_bucket"].max()),
        "max_lag": CFG["max_lag"],
        "rolling_short": CFG["rolling_short"],
        "rolling_long": CFG["rolling_long"],
        "metrics": {"overall": overall, "per_horizon": per_h},
        "source_logs": logs if isinstance(logs, list) else [logs],
    }
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    print(f"[저장] {MODEL_PATH} (셀 {len(artifact['cells'])}개, data_until={artifact['data_until']})")
    return artifact


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*", default=None)
    ap.add_argument("--external", default=None, help="합성 외부 관측 CSV (time_bucket×h3_index)")
    a = ap.parse_args()
    main(a.logs or None, external_csv=a.external)