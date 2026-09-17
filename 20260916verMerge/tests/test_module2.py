"""
[Module 2] 최소 완료 조건 테스트 — docs/MODULE2_MANUAL.md T1~T4/T6 의 DoD를 그대로 검사한다.

실행: python tests/test_module2.py
- h3/pygeohash/holidays 가 없는 환경(CI, 샌드박스)에서는 결정적(deterministic) 대체 구현으로 자동 전환.
  → 로직(캐시 동일성, 누수 없음, 병합 정렬, 결측 플래그)은 그대로 검증되고, 실제 셀 ID 값만 다름.
- 맥 venv(h3 설치됨)에서는 실제 라이브러리로 동작.
"""
import os, sys, types, warnings, math
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

# ---------- 외부 네트워크 조회 차단 (config_loader가 Nominatim/Open-Meteo를 부름) ----------
for m in ("geo_lookup", "weather_lookup"):
    sys.modules.setdefault(m, types.ModuleType(m))

# ---------- 선택 패키지 대체 ----------
MOCKED = []
try:
    import h3  # noqa
except ImportError:
    h3 = types.ModuleType("h3")
    h3.latlng_to_cell = lambda lat, lng, r: f"c{r}_{round(lat * 10 ** (r - 6)):d}_{round(lng * 10 ** (r - 6)):d}"
    sys.modules["h3"] = h3; MOCKED.append("h3")
try:
    import pygeohash  # noqa
except ImportError:
    pg = types.ModuleType("pygeohash"); pg.encode = lambda lat, lng, precision=7: f"g{round(lat, 3)}_{round(lng, 3)}"
    sys.modules["pygeohash"] = pg; MOCKED.append("pygeohash")
try:
    import holidays  # noqa
except ImportError:
    hol = types.ModuleType("holidays")
    class _KR(dict):
        def __init__(self, years=None):
            import datetime
            for y in years or []:
                for md in ((1, 1), (3, 1), (5, 5), (6, 6), (8, 15), (10, 3), (10, 9), (12, 25)):
                    self[datetime.date(y, *md)] = "holiday"
    hol.KR = _KR; sys.modules["holidays"] = hol; MOCKED.append("holidays(주요 공휴일 8개만)")

import numpy as np
import pandas as pd
from module2_preprocessing.spatial_indexing import SpatialIndexer
from module2_preprocessing.time_features import add_time_features, TIME_FEATURE_COLS
from module2_preprocessing.time_series_prep import TimeSeriesPreprocessor, time_based_split, feature_columns, target_columns
from module2_preprocessing.external_data_merge import merge_external_data, WEATHER_COLS
from module2_preprocessing.pipeline import build_feature_table

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(("PASS " if cond else "FAIL ") + name + (f"  — {detail}" if detail and not cond else ""))


# ============ T1 공간 인덱싱 ============
def test_spatial():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"latitude": rng.uniform(37.495, 37.505, 3000), "longitude": rng.uniform(127.02, 127.035, 3000)})
    df = pd.concat([df, df.head(500)], ignore_index=True)  # 중복 좌표 포함
    idx = SpatialIndexer(h3_resolution=9)
    fast = idx.process_dataframe(df, verbose=False)
    naive = idx.process_dataframe_naive(df)
    check("T1 h3_index/geohash 컬럼 생성", {"h3_index", "geohash"} <= set(fast.columns))
    check("T1 캐시 방식 == apply 방식 (h3)", (fast["h3_index"].values == naive["h3_index"].values).all())
    check("T1 캐시 방식 == apply 방식 (geohash)", (fast["geohash"].values == naive["geohash"].values).all())
    check("T1 행 순서 보존", (fast.index == df.index).all())
    part = idx.process_dataframe(df, chunk_size=700, verbose=False)
    check("T1 chunk 분할 결과 동일", (part["h3_index"].values == fast["h3_index"].values).all())


# ============ T2 시간 피처 ============
def test_time_features():
    d = pd.DataFrame({"time_bucket": pd.to_datetime(["2026-09-07 08:05", "2026-10-03 12:00", "2026-09-12 23:30", "2026-01-01 00:00"])})
    f = add_time_features(d)
    check("T2 모든 시간 피처 컬럼 존재", set(TIME_FEATURE_COLS) <= set(f.columns), str(set(TIME_FEATURE_COLS) - set(f.columns)))
    check("T2 2026-09-07(월) dayofweek=0, is_weekend=0", f.loc[0, "dayofweek"] == 0 and f.loc[0, "is_weekend"] == 0)
    check("T2 2026-10-03(개천절) is_holiday=1", f.loc[1, "is_holiday"] == 1)
    check("T2 2026-09-12(토) is_weekend=1, is_holiday=1", f.loc[2, "is_weekend"] == 1 and f.loc[2, "is_holiday"] == 1)
    check("T2 연월일시분 분해", (f.loc[0, ["year", "month", "day", "hour", "minute"]].tolist() == [2026, 9, 7, 8, 5]))
    check("T2 순환 인코딩 sin²+cos²=1", np.allclose(f["hour_sin"] ** 2 + f["hour_cos"] ** 2, 1) and np.allclose(f["dow_sin"] ** 2 + f["dow_cos"] ** 2, 1))
    check("T2 time_slot: 08시=morning_peak, 12시=lunch, 23시=late_night, 0시=dawn",
          f["time_slot"].tolist() == ["morning_peak", "lunch", "late_night", "dawn"])
    check("T2 season: 9월=3(가을), 1월=4(겨울)", f.loc[0, "season"] == 3 and f.loc[3, "season"] == 4)


# ============ T3 시계열 피처/타겟 ============
def _toy_log():
    # 셀 A, B 를 5분 칸 20개에 걸쳐 수요 배열로 정의 (0인 칸은 행이 없게 만들어 0-채움 검증)
    A = [1, 0, 2, 0, 0, 3, 1, 0, 4, 2, 0, 1, 5, 0, 0, 2, 3, 1, 0, 2] * 2   # 40칸 (rolling_1h=12칸 + 타겟 6칸 후에도 분할 가능하게)
    B = [0, 0, 0, 4, 0, 0, 0, 5, 1, 0, 2, 0, 0, 3, 0, 1, 0, 0, 6, 0] * 2
    base = pd.Timestamp("2026-09-07 09:00")
    rows = []
    for i, (a, b) in enumerate(zip(A, B)):
        rows += [dict(pickup_datetime=base + pd.Timedelta(minutes=5 * i + 1), latitude=37.4979, longitude=127.0276)] * a
        rows += [dict(pickup_datetime=base + pd.Timedelta(minutes=5 * i + 2), latitude=37.5030, longitude=127.0330)] * b
    return pd.DataFrame(rows), A, B, base


def test_time_series():
    df, A, B, base = _toy_log()
    df = SpatialIndexer(h3_resolution=9).process_dataframe(df, verbose=False)
    cellA, cellB = df["h3_index"].iloc[0], df[df.latitude > 37.5]["h3_index"].iloc[0]
    check("T3 toy 셀 2개 분리", cellA != cellB)
    prep = TimeSeriesPreprocessor(freq="5min", max_lag=3, rolling_short=2, rolling_long=4, horizons=6)
    agg = prep.aggregate_demands(df, start=base, end=base + pd.Timedelta(minutes=195))
    check("T3 (셀×칸) 전체 격자 80행(0인 칸 포함)", len(agg) == 80, str(len(agg)))
    fa = agg[agg.h3_index == cellA].sort_values("time_bucket")["demand"].tolist()
    check("T3 0-채움 후 셀A 수요 == 정의 배열", fa == A)

    feat = prep.create_features(agg, dropna=False, verbose=False)
    fa = feat[feat.h3_index == cellA].sort_values("time_bucket").reset_index(drop=True)
    fb = feat[feat.h3_index == cellB].sort_values("time_bucket").reset_index(drop=True)
    # 수작업 기대값
    sA = pd.Series(A, dtype=float)
    check("T3 lag_1 == 직전 칸", fa["lag_1"].tolist()[1:] == A[:-1])
    check("T3 rolling_mean_2 == 직전 2칸 평균", np.allclose(fa["rolling_mean_2"].values[2:], sA.shift(1).rolling(2).mean().values[2:]))
    check("T3 rolling_mean_1h(12칸) == 직전 12칸 평균", np.allclose(fa["rolling_mean_1h"].values[12:], sA.shift(1).rolling(12).mean().values[12:]))
    check("T3 y_h1..y_h6 == t+1..t+6 수요", all(fa[f"y_h{h}"].tolist()[:-h] == A[h:] for h in range(1, 7)))
    check("T3 셀 경계 누수 없음: 셀B 첫 칸 lag/rolling NaN", fb.loc[0, ["lag_1", "rolling_mean_2"]].isna().all() and pd.isna(fb.loc[3, "rolling_mean_1h"]))
    check("T3 지난주 동일칸: 1주 미만 → 값 0, 플래그 0", (fa["same_time_last_week"] == 0).all() and (fa["has_last_week"] == 0).all())
    check("T3 diff_1 == lag_1 - lag_2", np.allclose(fa["diff_1"].values[2:], (fa["lag_1"] - fa["lag_2"]).values[2:]))
    fcols = feature_columns(feat); tcols = target_columns(feat)
    check("T3 피처에 타겟/현재 수요 없음", not any(c.startswith("y_h") for c in fcols) and "demand" not in fcols)
    # 누수 검사: 어떤 피처도 현재 demand와 동일하지 않음
    leak = [c for c in fcols if feat[c].dtype != object and np.allclose(feat[c].fillna(-9).values, feat["demand"].values)]
    check("T3 현재 demand와 동일한 피처 없음(누수)", not leak, str(leak))
    check("T3 타겟 6개", tcols == [f"y_h{h}" for h in range(1, 7)])

    feat2 = prep.create_features(agg, verbose=False)
    tr, va, te, cut = time_based_split(feat2, 0.2, 0.1)
    check("T3 시간 분할: train < val < test", tr.time_bucket.max() < va.time_bucket.min() <= va.time_bucket.max() < te.time_bucket.min())
    check("T3 test에 두 셀 모두 포함", set(te.h3_index) == {cellA, cellB})


# ============ T4 외부 데이터 ============
def test_external():
    base = pd.Timestamp("2026-09-07 09:00")
    buckets = pd.date_range(base, base + pd.Timedelta(hours=3), freq="5min")
    df = pd.DataFrame({"time_bucket": list(buckets) * 2, "h3_index": ["A"] * len(buckets) + ["B"] * len(buckets)})
    w = pd.DataFrame({"time": pd.to_datetime(["2026-09-07 09:00", "2026-09-07 10:00", "2026-09-07 11:00", "2026-09-07 12:00"]),
                      "temperature": [20.0, 22.0, 24.0, 26.0], "precipitation": [0.0, 0.0, 1.5, 0.0], "wind_speed": [1, 2, 3, 4]})
    m = merge_external_data(df, weather=w, verbose=False)
    a = m[m.h3_index == "A"].set_index("time_bucket")
    check("T4 09:05~09:55 → 09:00 관측(20.0)", (a.loc["2026-09-07 09:05":"2026-09-07 09:55", "temperature"] == 20.0).all())
    check("T4 10:00 → 10:00 관측(22.0)", a.loc["2026-09-07 10:00", "temperature"] == 22.0)
    check("T4 미래 관측 미사용(09:55에 22.0 아님)", a.loc["2026-09-07 09:55", "temperature"] != 22.0)
    check("T4 is_rain: 11시대만 1", (a.loc["2026-09-07 11:00":"2026-09-07 11:55", "is_rain"] == 1).all() and a.loc["2026-09-07 10:30", "is_rain"] == 0)
    check("T4 결측 없음 플래그 0", (m["weather_missing"] == 0).all())
    check("T4 행 수·순서 보존", len(m) == len(df) and (m["h3_index"].values == df["h3_index"].values).all())
    # 관측 하나 제거 → 보간
    w2 = w.drop(index=1)
    m2 = merge_external_data(df, weather=w2, verbose=False)
    a2 = m2[m2.h3_index == "A"].set_index("time_bucket")
    check("T4 10시 관측 결측 → 선형 보간(22.0) + weather_interpolated=1",
          math.isclose(a2.loc["2026-09-07 10:30", "temperature"], 22.0) and a2.loc["2026-09-07 10:30", "weather_interpolated"] == 1)
    # 긴 공백(tolerance 밖) → 결측 플래그
    w3 = w.iloc[[0]]
    m3 = merge_external_data(df, weather=w3, max_gap_hours=1, verbose=False)
    a3 = m3[m3.h3_index == "A"].set_index("time_bucket")
    check("T4 tolerance(1h) 밖 → NaN + weather_missing=1", pd.isna(a3.loc["2026-09-07 11:00", "temperature"]) and a3.loc["2026-09-07 11:00", "weather_missing"] == 1)
    # 파일 없음 → fallback
    m4 = merge_external_data(df, weather=pd.DataFrame(), verbose=False)
    check("T4 날씨 없음 → fallback 상수 + source='fallback'", (m4["weather_source"] == "fallback").all() and m4["temperature"].notna().all())
    src = open(os.path.join(ROOT, "module2_preprocessing", "external_data_merge.py"), encoding="utf-8").read()
    code_lines = [l for l in src.splitlines() if not l.strip().startswith(("#", "※")) and '"""' not in l]
    check("T4 랜덤 날씨 생성 코드 없음", not any("np.random" in l for l in code_lines))


# ============ T6 파이프라인 ============
def test_pipeline(tmp="/tmp/m2_test"):
    os.makedirs(tmp, exist_ok=True)
    df, A, B, base = _toy_log()
    df["request_sec"] = (df["pickup_datetime"] - base).dt.total_seconds()
    p1 = os.path.join(tmp, "demand_log_day1.csv"); df.to_csv(p1, index=False)
    d2 = df.copy(); d2["pickup_datetime"] += pd.Timedelta(days=7)
    p2 = os.path.join(tmp, "demand_log_day2.csv"); d2.to_csv(p2, index=False)
    feat = build_feature_table([p1], freq="5min", max_lag=3, horizons=6, weather_path="/nonexistent.csv", verbose=False)
    check("T6 단일 로그 → 피처 테이블 생성", len(feat) > 0 and "y_h6" in feat.columns)
    check("T6 feature_cols에 외부·시간·시계열 피처 포함", {"temperature", "is_holiday", "lag_1", "rolling_mean_1h", "same_time_last_week"} <= set(feat.attrs["feature_cols"]))
    feat2 = build_feature_table([os.path.join(tmp, "demand_log_*.csv")], freq="5min", max_lag=3, horizons=6, weather_path="/nonexistent.csv", dropna=False, verbose=False)
    span = feat2["time_bucket"].max() - feat2["time_bucket"].min()
    check("T6 다일 로그(glob) 이어붙임: 기간 7일 이상", span >= pd.Timedelta(days=7))
    wk = feat2[feat2["has_last_week"] == 1]
    check("T6 2주차 행은 지난주 동일칸 값 존재", len(wk) > 0 and (wk["same_time_last_week"].values == wk["demand"].values).all()
          if len(wk) else False, "동일 패턴을 7일 뒤로 복사했으므로 same_time_last_week == demand 여야 함")


if __name__ == "__main__":
    if MOCKED:
        print(f"[안내] 대체 구현 사용: {', '.join(MOCKED)} — 맥 venv에서는 실제 라이브러리로 실행됨\n")
    for t in (test_spatial, test_time_features, test_time_series, test_external, test_pipeline):
        try:
            t()
        except Exception as e:
            import traceback; traceback.print_exc()
            check(f"{t.__name__} 실행", False, f"{type(e).__name__}: {e}")
    n_ok = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n===== {n_ok}/{len(RESULTS)} PASS =====")
    sys.exit(0 if n_ok == len(RESULTS) else 1)
