"""
module4_dispatch/category_demand_predictor.py

module3_prediction/train_category_demand.py로 학습한 카테고리별 모델을
rl_env.py / rl_reposition_maintainer.py의 demand_predictor 슬롯에 그대로 꽂는 실시간 예측기.

인터페이스 (rl_env.py가 기대하는 것):
    predictor.predict_by_zone(sim_time) -> {category: 예측수요값, ...}

모델이 lag_1~lag_6 / rolling_mean 피처를 요구하기 때문에, 실제로 예측하려면
"최근 몇 구간 동안 각 카테고리에 실제 수요가 몇 건이었는지" 이력이 필요함.
그래서 이 클래스는 내부에 카테고리별 rolling history를 들고 있고, 매 decision
interval마다 rl_env.py가 실제 관측치를 update()로 넣어줘야 함.
(이력이 max_lag만큼 안 쌓였을 때는 0으로 대체 — 시뮬레이션 초반 워밍업 구간)
"""
import os
import sys
from collections import deque

import joblib
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config_loader import CFG

MODEL_PATH = os.path.join(PROJECT_ROOT, "saved_models", "category_demand_models.pkl")


class CategoryDemandPredictor:
    def __init__(self, model_path: str = None, run_date=None):
        model_path = model_path or MODEL_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"{model_path} 가 없습니다. 먼저 module3_prediction/train_category_demand.py 를 실행하세요."
            )
        bundle = joblib.load(model_path)
        self.models = bundle["models"]          # {category: XGBRegressor}
        self.feature_cols = bundle["feature_cols"]
        self.category_list = bundle["category_list"]

        # run_date: RL 에피소드 시작 시각의 실제 날짜(요일 계산용). 안 주면 오늘 날짜로 대체
        import datetime
        self.run_date = run_date or datetime.date.today()

        max_lag = CFG["max_lag"]
        long_w = CFG["rolling_long"]
        hist_len = max(max_lag, long_w) + 1
        # 카테고리별 "최근 관측된 수요 건수" 이력 (decision interval 단위)
        self._history = {c: deque([0.0] * hist_len, maxlen=hist_len) for c in self.category_list}

    def update(self, observed_counts: dict):
        """rl_env.py가 매 decision interval마다 실제 관측한 대기승객 수(또는 신규 수요)를 넣어줌"""
        for c in self.category_list:
            self._history[c].append(float(observed_counts.get(c, 0.0)))

    def _build_features(self, category: str, sim_time: float) -> np.ndarray:
        hist = list(self._history[category])
        max_lag = CFG["max_lag"]
        short_w = CFG["rolling_short"]
        long_w = CFG["rolling_long"]

        lags = {f"lag_{i}": hist[-i] for i in range(1, max_lag + 1)}
        rolling = {
            f"rolling_mean_{short_w}": float(np.mean(hist[-short_w:])) if short_w <= len(hist) else 0.0,
            f"rolling_mean_{long_w}": float(np.mean(hist[-long_w:])) if long_w <= len(hist) else 0.0,
        }

        import datetime
        sim_dt = datetime.datetime.combine(self.run_date, datetime.time(0, 0)) + datetime.timedelta(seconds=sim_time)
        calendar = {
            "hour": sim_dt.hour,
            "dayofweek": sim_dt.weekday(),
            "is_weekend": 1 if sim_dt.weekday() >= 5 else 0,
        }

        # 학습 때(merge_external_data)와 동일한 방식 - 실측 없으니 config 범위 내 랜덤으로 대체
        weather = {
            "temperature": float(np.random.uniform(CFG["temp_min"], CFG["temp_max"])),
            "precipitation": float(np.random.choice([0.0, 0.0, 0.0, 1.2, 5.5])),
        }

        feats = {**lags, **rolling, **calendar, **weather}
        return np.array([[feats.get(c, 0.0) for c in self.feature_cols]])

    def predict_by_zone(self, sim_time: float) -> dict:
        result = {}
        for category, model in self.models.items():
            X = self._build_features(category, sim_time)
            result[category] = float(model.predict(X)[0])
        return result