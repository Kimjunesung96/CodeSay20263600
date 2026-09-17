"""
[Module 2 / T3] 호출 로그(1행=호출 1건) → (H3 셀 × 시간칸) 수요 시계열 + 파생 변수 + 예측 타겟.

명세 대응
- R3 "지난 1시간 평균, 지난주 동일 시간대, 이동평균": rolling_mean_1h / same_time_last_week / rolling_mean_*
- R5 "t까지 데이터로 t+1~t+6(5분 단위) 예측": freq=5min, y_h1..y_h6 타겟

설계 원칙 (누수 방지)
- 모든 피처는 shift(1) 이후에 계산 → 현재 칸(t)의 demand는 어떤 피처에도 들어가지 않음.
- 셀별 groupby 안에서만 shift/rolling → 셀 경계에서 다른 셀 값이 섞이지 않음.
- 수요 0인 칸도 행으로 존재해야 "직전 칸"이 진짜 직전 시각이 됨 → 전체 격자로 reindex 후 0 채움.

[수정 이력] 2026-09-10 윤세빈
- 기존: 호출 있던 칸만 행 생성(lag가 '직전 호출 칸'을 가리킴), rolling이 셀 경계를 넘어 계산됨, 단일 타겟
- 수정: 위 설계 원칙 반영 + 다중 시점 타겟 + 지난주 동일 시간대 + 시간 피처 모듈 분리(time_features.py)
"""
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import CFG
from module2_preprocessing.time_features import add_time_features, TIME_FEATURE_COLS

ID_COLS = ["time_bucket", "h3_index"]
TARGET_PREFIX = "y_h"


def _buckets_per(freq: str, minutes: int) -> int:
    """freq(예: '5min') 기준으로 minutes 분이 몇 칸인지."""
    step = pd.Timedelta(freq)
    return max(1, int(pd.Timedelta(minutes=minutes) / step))


class TimeSeriesPreprocessor:
    def __init__(self, freq: str = None, max_lag: int = None,
                 rolling_short: int = None, rolling_long: int = None, horizons: int = None):
        self.freq = freq if freq is not None else CFG["freq"]
        self.max_lag = max_lag if max_lag is not None else CFG["max_lag"]
        self.rolling_short = rolling_short if rolling_short is not None else CFG["rolling_short"]
        self.rolling_long = rolling_long if rolling_long is not None else CFG["rolling_long"]
        self.horizons = horizons if horizons is not None else CFG.get("forecast_horizons", 6)
        self.buckets_1h = _buckets_per(self.freq, 60)
        self.buckets_1w = _buckets_per(self.freq, 7 * 24 * 60)

    # ---------- 1) 집계 ----------
    def aggregate_demands(self, df: pd.DataFrame, timestamp_col: str = 'pickup_datetime',
                          spatial_col: str = 'h3_index', start=None, end=None, all_cells=None) -> pd.DataFrame:
        """(셀 × 시간칸) 수요 집계. 호출이 없던 칸도 demand=0 행으로 포함."""
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df['time_bucket'] = df[timestamp_col].dt.floor(self.freq)
        demand = df.groupby(['time_bucket', spatial_col]).size().rename('demand')

        start = pd.Timestamp(start).floor(self.freq) if start is not None else df['time_bucket'].min()
        end = pd.Timestamp(end).floor(self.freq) if end is not None else df['time_bucket'].max()
        buckets = pd.date_range(start, end, freq=self.freq)
        cells = sorted(all_cells) if all_cells is not None else sorted(df[spatial_col].dropna().unique())
        full_idx = pd.MultiIndex.from_product([buckets, cells], names=['time_bucket', spatial_col])
        out = demand.reindex(full_idx, fill_value=0).reset_index()
        out['demand'] = out['demand'].astype(int)
        return out

    # ---------- 2) 피처 ----------
    def create_features(self, demand_df: pd.DataFrame, spatial_col: str = 'h3_index',
                        dropna: bool = True, add_targets: bool = True, verbose: bool = True) -> pd.DataFrame:
        d = demand_df.sort_values(by=[spatial_col, 'time_bucket']).reset_index(drop=True)
        g = d.groupby(spatial_col, sort=False)['demand']
        prev = g.shift(1)                      # 직전 칸 값 (누수 방지의 기준점)
        feat_cols = []

        # (a) lag_1..lag_k
        for lag in range(1, self.max_lag + 1):
            d[f'lag_{lag}'] = g.shift(lag); feat_cols.append(f'lag_{lag}')

        # (b) 이동평균: 짧은 창 / 긴 창 / 지난 1시간 (창은 '칸' 단위, 현재 칸 제외)
        windows = {f'rolling_mean_{self.rolling_short}': self.rolling_short,
                   f'rolling_mean_{self.rolling_long}': self.rolling_long,
                   'rolling_mean_1h': self.buckets_1h}
        for name, w in windows.items():
            d[name] = prev.groupby(d[spatial_col]).transform(lambda s, w=w: s.rolling(w).mean())
            feat_cols.append(name)
        d['rolling_std_1h'] = prev.groupby(d[spatial_col]).transform(lambda s: s.rolling(self.buckets_1h).std())
        feat_cols.append('rolling_std_1h')

        # (c) 지난주 동일 시간대 (7일 전 같은 칸). 데이터가 1주 미만이면 NaN → 0 + 플래그 0
        d['same_time_last_week'] = g.shift(self.buckets_1w)
        d['has_last_week'] = d['same_time_last_week'].notna().astype(int)
        d['same_time_last_week'] = d['same_time_last_week'].fillna(0)
        feat_cols += ['same_time_last_week', 'has_last_week']

        # (d) 추세: 직전 칸 - 그 전 칸
        d['diff_1'] = g.shift(1) - g.shift(2); feat_cols.append('diff_1')

        # (e) 시간 피처 (T2)
        d = add_time_features(d, ts_col='time_bucket')
        feat_cols += TIME_FEATURE_COLS

        # (f) 타겟: t+1 ~ t+H
        target_cols = []
        if add_targets:
            g2 = d.groupby(spatial_col, sort=False)['demand']
            for h in range(1, self.horizons + 1):
                d[f'{TARGET_PREFIX}{h}'] = g2.shift(-h); target_cols.append(f'{TARGET_PREFIX}{h}')

        if verbose:
            print(f"시계열 파생변수 생성: lag {self.max_lag}개, rolling {self.rolling_short}/{self.rolling_long}/1h({self.buckets_1h}칸), "
                  f"지난주 동일칸({self.buckets_1w}칸 전), 시간 피처 {len(TIME_FEATURE_COLS)}개, 타겟 t+1~t+{self.horizons}")
        if dropna:
            before = len(d)
            need = [f'lag_{self.max_lag}', 'rolling_mean_1h', 'diff_1'] + target_cols
            d = d.dropna(subset=need).reset_index(drop=True)
            if verbose:
                print(f"  앞부분(lag/rolling 계산 불가) + 뒷부분(미래 타겟 없음) {before - len(d)}행 제거 → {len(d)}행")
        d.attrs['feature_cols'] = feat_cols
        d.attrs['target_cols'] = target_cols
        return d


def feature_columns(df: pd.DataFrame) -> list:
    """모델 입력 컬럼: id/타겟/원 수요/문자열 제외."""
    if 'feature_cols' in df.attrs:
        return list(df.attrs['feature_cols'])
    return [c for c in df.columns if c not in ID_COLS + ['demand', 'time_slot', 'geohash']
            and not c.startswith(TARGET_PREFIX)]


def target_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c.startswith(TARGET_PREFIX)]


def time_based_split(feature_df: pd.DataFrame, test_size: float, val_size: float = 0.0, time_col: str = 'time_bucket'):
    """
    시간 순서 분할: 마지막 test_size 비율의 '시간칸'을 test, 그 앞 val_size 비율을 val.
    반환: (train, val, test, cutoffs) — val_size=0이면 val은 빈 DataFrame.
    (기존 train_test_split(shuffle=False)은 [셀, 시간] 정렬 상태에서 행을 잘라 '마지막 셀들'이 test가 되는 문제가 있었음)
    """
    buckets = np.sort(feature_df[time_col].unique())
    n = len(buckets)
    n_test = max(1, int(round(n * test_size)))
    n_val = int(round(n * val_size))
    test_cut = buckets[n - n_test]
    val_cut = buckets[max(0, n - n_test - n_val)] if n_val > 0 else test_cut
    t = feature_df[time_col]
    train = feature_df[t < val_cut].copy()
    val = feature_df[(t >= val_cut) & (t < test_cut)].copy()
    test = feature_df[t >= test_cut].copy()
    return train, val, test, {'val_cut': pd.Timestamp(val_cut), 'test_cut': pd.Timestamp(test_cut)}
