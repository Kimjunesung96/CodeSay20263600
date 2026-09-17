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

import datetime
import joblib
import numpy as np
import pandas as pd

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
        self.run_date = run_date or datetime.date.today()

        max_lag = CFG["max_lag"]
        long_w = CFG["rolling_long"]
        hist_len = max(max_lag, long_w) + 1
        # 카테고리별 "최근 관측된 수요 건수" 이력 (decision interval 단위)
        self._history = {c: deque([0.0] * hist_len, maxlen=hist_len) for c in self.category_list}

        # [성능] add_time_features()가 호출마다 korean_holidays()로 holidays.KR()을 새로 만들면
        # (내부적으로 음력 변환 포함) RL 스텝마다(카테고리 수 배) 반복돼 체감 속도를 떨어뜨림.
        # 에피소드 동안 쓰일 연도(전날 포함 최대 +1일 버퍼)만 미리 한 번 계산해 재사용.
        from module2_preprocessing.time_features import korean_holidays
        years = {self.run_date.year, (self.run_date + datetime.timedelta(days=1)).year}
        self._holiday_dates = korean_holidays(years)

    def update(self, observed_counts: dict):
        """rl_env.py가 매 decision interval마다 실제 관측한 대기승객 수(또는 신규 수요)를 넣어줌"""
        for c in self.category_list:
            self._history[c].append(float(observed_counts.get(c, 0.0)))

    def _build_features(self, category: str, sim_time: float) -> np.ndarray:
        """학습 때(Module 2 create_features)와 같은 이름의 피처를 만들어 준다.

        [병합 수정] 예전에는 lag_*, rolling_mean_short/long, hour/dayofweek/is_weekend, 날씨만
        만들었는데, 지금 피처 테이블에는 rolling_mean_1h / rolling_std_1h / diff_1 /
        same_time_last_week / has_last_week + 시간 피처 15종이 들어 있다. 빠진 이름은 전부
        0으로 채워져 학습 때와 다른 입력이 들어가므로 여기서 직접 만들어 준다.
        """
        from module2_preprocessing.time_features import add_time_features

        hist = list(self._history[category])
        max_lag = CFG["max_lag"]
        short_w = CFG["rolling_short"]
        long_w = CFG["rolling_long"]

        def _mean(n):
            window = hist[-n:] if n <= len(hist) else hist
            return float(np.mean(window)) if window else 0.0

        def _std(n):
            window = hist[-n:] if n <= len(hist) else hist
            return float(np.std(window, ddof=1)) if len(window) > 1 else 0.0

        feats = {f"lag_{i}": hist[-i] for i in range(1, max_lag + 1)}
        feats[f"rolling_mean_{short_w}"] = _mean(short_w)
        feats[f"rolling_mean_{long_w}"] = _mean(long_w)
        # 이력 deque는 decision interval 단위라 "1시간 칸 수"를 정확히 모른다 —
        # 가진 이력 전체를 1시간 창의 근사로 사용.
        feats["rolling_mean_1h"] = _mean(len(hist))
        feats["rolling_std_1h"] = _std(len(hist))
        feats["diff_1"] = (hist[-1] - hist[-2]) if len(hist) >= 2 else 0.0
        # 실시간 시뮬레이션에는 1주일 전 이력이 없다. 학습 테이블도 이 경우
        # same_time_last_week=0 / has_last_week=0으로 채우므로 동일하게 맞춘다.
        feats["same_time_last_week"] = 0.0
        feats["has_last_week"] = 0

        # 시간 피처: 학습 때와 똑같은 함수를 1행짜리 DataFrame에 적용해 이름/값 불일치 제거
        sim_dt = (datetime.datetime.combine(self.run_date, datetime.time(0, 0))
                  + datetime.timedelta(seconds=float(sim_time)))
        tf = add_time_features(pd.DataFrame({"time_bucket": [pd.Timestamp(sim_dt)]}), ts_col="time_bucket",
                              holiday_dates=self._holiday_dates)
        for col in tf.columns:
            if col != "time_bucket":
                val = tf.iloc[0][col]
                if not isinstance(val, str):
                    feats[col] = float(val)

        missing = [c for c in self.feature_cols if c not in feats]
        if missing and not getattr(self, "_warned_missing", False):
            print(f"[경고] 학습 피처 중 {len(missing)}개를 실시간으로 만들 수 없어 0으로 채웁니다: {missing[:8]}")
            self._warned_missing = True

        return np.array([[float(feats.get(c, 0.0)) for c in self.feature_cols]])

    def predict_by_zone(self, sim_time: float) -> dict:
        result = {}
        for category, model in self.models.items():
            X = self._build_features(category, sim_time)
            result[category] = float(model.predict(X)[0])
        return result