"""
[Module 2 / T2] 시간 피처 — 타임스탬프를 연·월·일·시·분·요일로 분해하고 공휴일·시간대를 붙인다.

생성 컬럼
  year, month, day, hour, minute, minute_of_day, dayofweek(월=0), is_weekend, is_holiday,
  season(1봄 2여름 3가을 4겨울), time_slot(문자열), time_slot_code(정수),
  hour_sin, hour_cos, dow_sin, dow_cos  ← 순환 인코딩: 23시와 0시가 "가깝다"는 걸 모델에 알려줌 (딥러닝용)

공휴일: `holidays` 패키지의 한국(KR) 달력 사용. 패키지가 없으면 경고 후 주말만 휴일로 간주.
time_slot 경계는 시뮬 규칙(config: school/company/lunch/evening 시간창)과 맞춰 해석이 가능하게 함.
"""
import sys
import os
import warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import holidays as _holidays
except ImportError:
    _holidays = None

# (시작 hour 포함, 끝 hour 미포함, 이름) — 시뮬 규칙: 등교 7~8, 출근 8~10, 점심 12~13, 퇴근/저녁 18~22, 심야 23~
TIME_SLOTS = [
    (0, 6, "dawn"), (6, 10, "morning_peak"), (10, 12, "daytime"), (12, 14, "lunch"),
    (14, 18, "afternoon"), (18, 22, "evening_peak"), (22, 24, "late_night"),
]
SLOT_CODE = {name: i for i, (_, _, name) in enumerate(TIME_SLOTS)}


def _slot_of_hour(h: int) -> str:
    for s, e, name in TIME_SLOTS:
        if s <= h < e:
            return name
    return "late_night"


def korean_holidays(years) -> set:
    """해당 연도들의 한국 공휴일 날짜 집합. holidays 패키지가 없으면 빈 집합 + 경고."""
    if _holidays is None:
        warnings.warn("holidays 패키지가 없어 공휴일을 반영하지 못합니다 (pip install holidays). is_holiday는 주말만 1.")
        return set()
    kr = _holidays.KR(years=list(years))
    return set(kr.keys())


def add_time_features(df: pd.DataFrame, ts_col: str = "time_bucket", holiday_dates: set = None) -> pd.DataFrame:
    df = df.copy()
    t = pd.to_datetime(df[ts_col])
    df["year"] = t.dt.year
    df["month"] = t.dt.month
    df["day"] = t.dt.day
    df["hour"] = t.dt.hour
    df["minute"] = t.dt.minute
    df["minute_of_day"] = df["hour"] * 60 + df["minute"]
    df["dayofweek"] = t.dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    if holiday_dates is None:
        holiday_dates = korean_holidays(sorted(t.dt.year.unique()))
    dates = t.dt.date
    df["is_holiday"] = (dates.isin(holiday_dates) | (df["is_weekend"] == 1)).astype(int)

    df["season"] = (((df["month"] - 3) % 12) // 3 + 1).astype(int)  # 3~5→1(봄) 6~8→2(여름) 9~11→3(가을) 12~2→4(겨울)
    df["time_slot"] = df["hour"].map(_slot_of_hour)
    df["time_slot_code"] = df["time_slot"].map(SLOT_CODE).astype(int)

    frac_day = df["minute_of_day"] / 1440.0
    df["hour_sin"] = np.sin(2 * np.pi * frac_day)
    df["hour_cos"] = np.cos(2 * np.pi * frac_day)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)
    return df


TIME_FEATURE_COLS = ["year", "month", "day", "hour", "minute", "minute_of_day", "dayofweek", "is_weekend",
                     "is_holiday", "season", "time_slot_code", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]

if __name__ == "__main__":
    demo = pd.DataFrame({"time_bucket": pd.to_datetime(["2026-09-07 08:05", "2026-10-03 12:00", "2026-09-12 23:30"])})
    print(add_time_features(demo).T)
