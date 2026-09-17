"""
[Module 2 / T4] 외부 데이터 결합 + 결측치 처리.

명세 Q3 "시공간 단위가 다른 외부 데이터(날씨 등)를 결합할 때 기준을 맞추는 병합 로직"
- 수요 시계열은 5분 칸, 날씨 관측은 1시간 단위(기상청·Open-Meteo 모두 시간별 제공).
- pd.merge_asof(direction='backward'): 각 5분 칸에 "그 시각 이전 가장 최근 관측"을 붙인다.
  → 09:05~09:55 칸은 09:00 관측, 10:00 칸부터 10:00 관측. 미래 관측을 쓰지 않으므로 누수 없음.
- tolerance=max_gap_hours: 관측이 그 이상 끊겨 있으면 붙이지 않고 결측으로 남긴다.

결측치 처리 정책 (명세 R4)
1) 날씨 파일 안의 짧은 공백(≤ interpolate_limit 시간): 시간 축 선형 보간 → weather_interpolated=1
2) 긴 공백 / tolerance 밖: NaN 유지 → weather_missing=1  (모델은 이 플래그로 "모름"을 학습)
3) 날씨 파일 자체가 없음: config의 현재 기온으로 상수 채움 + weather_source='fallback' (학습 시 경고)
※ 기존 코드의 np.random 날씨 생성은 삭제 — 랜덤값은 수요와 무관해 모델에 노이즈만 추가하고 EDA 결과를 왜곡함.

공휴일(외부 달력 데이터)은 time_features.add_time_features가 붙인다 (holidays 패키지 KR).
"""
import sys
import os
import glob
import warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import CFG

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTERNAL_DIR = os.path.join(ROOT, "data", "external")
WEATHER_COLS = ["temperature", "precipitation", "wind_speed"]


def find_weather_file(region: str = None) -> str:
    """data/external/weather_<지역>_*.csv 중 가장 최근 파일 (없으면 None)."""
    region = region or CFG.get("region", "")
    files = sorted(glob.glob(os.path.join(EXTERNAL_DIR, f"weather_{region}_*.csv"))) or \
            sorted(glob.glob(os.path.join(EXTERNAL_DIR, "weather_*.csv")))
    return files[-1] if files else None


def load_weather(path: str) -> pd.DataFrame:
    """시간별 날씨 CSV → DataFrame(time, temperature, precipitation, wind_speed), 시간순 정렬."""
    w = pd.read_csv(path)
    tcol = "time" if "time" in w.columns else w.columns[0]
    w = w.rename(columns={tcol: "time"})
    w["time"] = pd.to_datetime(w["time"])
    for c in WEATHER_COLS:
        if c not in w.columns:
            w[c] = np.nan
    return w[["time"] + WEATHER_COLS].sort_values("time").drop_duplicates("time").reset_index(drop=True)


def _fill_short_gaps(w: pd.DataFrame, interpolate_limit_hours: int) -> pd.DataFrame:
    """관측 자체에 빠진 시각이 있으면 시간 격자로 펼친 뒤 짧은 공백만 보간."""
    full = pd.date_range(w["time"].min(), w["time"].max(), freq="1h")
    w = w.set_index("time").reindex(full)
    w.index.name = "time"
    was_nan = w[WEATHER_COLS].isna().any(axis=1)
    w[WEATHER_COLS] = w[WEATHER_COLS].interpolate(method="linear", limit=interpolate_limit_hours, limit_area="inside")
    w["weather_interpolated"] = (was_nan & w[WEATHER_COLS].notna().all(axis=1)).astype(int)
    return w.reset_index()


def merge_external_data(df: pd.DataFrame, weather: pd.DataFrame = None, weather_path: str = None,
                        time_col: str = None, max_gap_hours: int = 3, interpolate_limit_hours: int = 3,
                        verbose: bool = True) -> pd.DataFrame:
    """
    df(5분 칸 수요 테이블 또는 호출 로그)에 날씨 컬럼을 붙인다.
    time_col: 기준 시각 컬럼 (없으면 time_bucket → pickup_datetime → datetime 순으로 탐색)
    """
    df = df.copy()
    if time_col is None:
        time_col = next((c for c in ("time_bucket", "pickup_datetime", "datetime") if c in df.columns), None)
    if time_col is None:
        raise ValueError("시각 컬럼(time_bucket/pickup_datetime/datetime)이 없습니다")
    df[time_col] = pd.to_datetime(df[time_col])

    if weather is None:
        weather_path = weather_path or find_weather_file()
        if weather_path and os.path.exists(weather_path):
            weather = load_weather(weather_path)
            if verbose:
                print(f"[외부 데이터] 날씨 파일 사용: {os.path.basename(weather_path)} ({len(weather)}시간, "
                      f"{weather['time'].min()} ~ {weather['time'].max()})")

    if weather is None or weather.empty:
        # 3) 폴백: 상수 채움. 학습에 쓰면 날씨 피처는 정보량 0 — 반드시 경고.
        warnings.warn("날씨 파일이 없어 config 현재 기온으로 상수 채움 (weather_source='fallback'). "
                      "scripts/fetch_weather_history.py 로 실데이터를 받으세요.")
        df["temperature"] = float(CFG.get("current_temperature", (CFG.get("temp_min", 15) + CFG.get("temp_max", 25)) / 2))
        df["precipitation"] = float(CFG.get("current_precipitation", 0.0))
        df["wind_speed"] = np.nan
        df["weather_interpolated"] = 0
        df["weather_missing"] = 1
        df["weather_source"] = "fallback"
    else:
        w = _fill_short_gaps(weather, interpolate_limit_hours)
        df["_row_order"] = np.arange(len(df))
        merged = pd.merge_asof(df.sort_values(time_col), w.sort_values("time"),
                               left_on=time_col, right_on="time", direction="backward",
                               tolerance=pd.Timedelta(hours=max_gap_hours))
        df = merged.sort_values("_row_order").drop(columns=["time", "_row_order"]).reset_index(drop=True)
        df["weather_missing"] = df[WEATHER_COLS].isna().any(axis=1).astype(int)
        df["weather_interpolated"] = df["weather_interpolated"].fillna(0).astype(int)
        df["weather_source"] = "observed"
        if verbose:
            miss = df["weather_missing"].mean() * 100
            print(f"[외부 데이터] 병합 완료: 결측 {miss:.1f}% (tolerance {max_gap_hours}h), 보간 {df['weather_interpolated'].sum()}행")

    df["is_rain"] = (df["precipitation"].fillna(0) >= 0.1).astype(int)
    return df


EXTERNAL_FEATURE_COLS = ["temperature", "precipitation", "wind_speed", "is_rain", "weather_missing", "weather_interpolated"]