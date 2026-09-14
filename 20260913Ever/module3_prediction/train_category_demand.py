"""
module3_prediction/train_category_demand.py

H3 기준 train.py와 나란히 두는 "카테고리 기준" 버전.
목적: RL(module4_dispatch/rl_env.py)의 _predicted_demand() 자리에 바로 꽂을 수 있는
      카테고리별(school/residential/company/restaurant/subway_entrance/bus_stop) 수요예측 모델.

train.py와 다른 점은 spatial_col='category' 하나뿐 — TimeSeriesPreprocessor.aggregate_demands /
create_features가 원래 spatial_col을 인자로 받게 설계돼 있어서 h3_index용 로직을 그대로 재사용함.
(H3용 saved_models/xgboost_demand.pkl, cnn_lstm_demand.pt는 이 스크립트가 손대지 않음 — 완전 별도 파일)

실행: python module3_prediction/train_category_demand.py
출력: saved_models/category_demand_models.pkl
      {"models": {category: XGBRegressor}, "feature_cols": [...], "category_list": [...]}
"""
import glob
import os
import sys

import joblib
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config_loader import CFG
from module2_preprocessing.category_indexing import CategoryIndexer
from module2_preprocessing.time_series_prep import TimeSeriesPreprocessor
from module2_preprocessing.external_data_merge import merge_external_data  # train.py와 동일하게 재사용

DEMAND_LOG_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DEMAND_LOG_GLOB = "simulation_demand_log*.csv"
OUT_PATH = os.path.join(PROJECT_ROOT, "saved_models", "category_demand_models.pkl")

CATEGORY_LIST = ["school", "residential", "company", "restaurant", "subway_entrance", "bus_stop"]


def _load_all_demand_logs():
    """train.py의 _load_all_demand_logs()와 동일한 로직 (여기서도 그대로 필요해서 복제)"""
    paths = sorted(glob.glob(os.path.join(DEMAND_LOG_DIR, DEMAND_LOG_GLOB)))
    if not paths:
        return None
    frames = [pd.read_csv(p, parse_dates=["pickup_datetime"]) for p in paths]
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["pickup_datetime", "latitude", "longitude"])
    print(f"[안내] 로그 {len(paths)}개에서 총 {len(merged)}건 로드")
    return merged


def main():
    df = _load_all_demand_logs()
    if df is None:
        print("[에러] data/raw/simulation_demand_log*.csv 로그가 없습니다. 먼저 시뮬레이션을 돌려주세요.")
        return

    indexer = CategoryIndexer()  # 기본: module1_simulation/sumo_config/ 사용
    df = indexer.process_dataframe(df)
    df = merge_external_data(df)  # train.py와 동일한 순서: 집계 전에 원본 로그에 날씨/주말 피처 결합

    prep = TimeSeriesPreprocessor(max_lag=CFG["max_lag"])
    agg_df = prep.aggregate_demands(df, timestamp_col="pickup_datetime", spatial_col="category")
    feature_df = prep.create_features(agg_df, spatial_col="category")

    feature_df = feature_df.dropna()
    if feature_df.empty:
        print("[에러] lag/rolling 생성 후 데이터가 하나도 안 남았습니다. 로그를 더 쌓아주세요.")
        return

    feature_cols = [c for c in feature_df.columns if c not in ["time_bucket", "category", "demand"]]

    models = {}
    for category in CATEGORY_LIST:
        cat_df = feature_df[feature_df["category"] == category]
        if len(cat_df) < 20:
            print(f"[경고] '{category}' 샘플이 {len(cat_df)}건뿐이라 건너뜁니다 (최소 20건 권장).")
            continue

        X, y = cat_df[feature_cols], cat_df["demand"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=CFG["test_size"], random_state=42, shuffle=False
        )
        model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
        model.fit(X_train, y_train)
        rmse = ((model.predict(X_test) - y_test) ** 2).mean() ** 0.5
        print(f"[{category}] Train {len(X_train)} / Test {len(X_test)} | RMSE {rmse:.3f}")
        models[category] = model

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    joblib.dump({"models": models, "feature_cols": feature_cols, "category_list": list(models.keys())}, OUT_PATH)
    print(f"[완료] {OUT_PATH} 에 카테고리별 모델 {len(models)}개 저장")


if __name__ == "__main__":
    main()