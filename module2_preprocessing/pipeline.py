"""
[Module 2 / T6] 전처리 파이프라인 단일 진입점.

    호출 로그 CSV(1개 이상) ──▶ 공간 인덱싱(H3/Geohash) ──▶ (셀×5분) 집계·0채움
        ──▶ 외부 데이터(날씨) 병합 + 결측 처리 ──▶ 시계열/시간 피처 + t+1~t+6 타겟 ──▶ 피처 테이블

사용 (CLI):
    python -m module2_preprocessing.pipeline --logs "data/sim_logs/*.csv" --out data/processed/features.csv
    python -m module2_preprocessing.pipeline --logs a.csv b.csv        # 여러 날 로그를 이어 붙임
코드에서:
    from module2_preprocessing.pipeline import build_feature_table
    feat = build_feature_table(["data/sim_logs/demand_log_x.csv"])
    X_cols, y_cols = feat.attrs["feature_cols"], feat.attrs["target_cols"]

입력 스키마: pickup_datetime, latitude, longitude (그 외 컬럼은 무시) — 시뮬 로그·공개 데이터 공통.
메모리: 로그는 파일별로 읽어 필요한 3컬럼만 유지하고, 집계 후에는 (셀×칸) 크기로 줄어들므로
        원본 행 수가 수백만이어도 피처 테이블은 작다 (명세 Q4).
"""
import os
import sys
import glob
import argparse
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config_loader import CFG
from module2_preprocessing.spatial_indexing import SpatialIndexer
from module2_preprocessing.time_series_prep import TimeSeriesPreprocessor, feature_columns, target_columns
from module2_preprocessing.external_data_merge import merge_external_data, EXTERNAL_FEATURE_COLS

REQUIRED = ["pickup_datetime", "latitude", "longitude"]


def load_logs(paths) -> pd.DataFrame:
    """로그 CSV들을 읽어 필수 3컬럼만 이어 붙임. request_sec가 있으면 시뮬 시작 시각 계산에 사용."""
    if isinstance(paths, str):
        paths = [paths]
    files = []
    for p in paths:
        files += sorted(glob.glob(p)) if any(ch in p for ch in "*?[") else [p]
    if not files:
        raise FileNotFoundError(f"로그 파일 없음: {paths}")
    parts = []
    for f in files:
        d = pd.read_csv(f, usecols=lambda c: c in REQUIRED + ["request_sec"])
        missing = [c for c in REQUIRED if c not in d.columns]
        if missing:
            raise ValueError(f"{f}: 필수 컬럼 없음 {missing}")
        d = d.dropna(subset=["latitude", "longitude"])
        d["pickup_datetime"] = pd.to_datetime(d["pickup_datetime"])
        d["_source"] = os.path.basename(f)
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    print(f"[로그] {len(files)}개 파일, {len(df):,}건, {df['pickup_datetime'].min()} ~ {df['pickup_datetime'].max()}")
    return df


def _time_range(df: pd.DataFrame):
    """집계 범위: 시뮬 로그면 시뮬 시작 시각(호출 없던 앞부분 포함), 아니면 첫 호출."""
    if "request_sec" in df.columns and df["request_sec"].notna().any():
        start = (df["pickup_datetime"] - pd.to_timedelta(df["request_sec"].fillna(0), unit="s")).min()
    else:
        start = df["pickup_datetime"].min()
    return start, df["pickup_datetime"].max()


def build_feature_table(log_paths, freq: str = None, max_lag: int = None, horizons: int = None,
                        weather_path: str = None, h3_resolution: int = None, dropna: bool = True,
                        verbose: bool = True) -> pd.DataFrame:
    df = load_logs(log_paths)
    start, end = _time_range(df)

    # 1) 공간 인덱싱
    df = SpatialIndexer(h3_resolution=h3_resolution).process_dataframe(df)

    # 2) (셀 × 칸) 집계 + 0 채움
    prep = TimeSeriesPreprocessor(freq=freq, max_lag=max_lag, horizons=horizons)
    agg = prep.aggregate_demands(df, start=start, end=end)

    # 3) 외부 데이터 (5분 칸 기준으로 병합 → 셀마다 같은 시각이면 같은 날씨)
    agg = merge_external_data(agg, weather_path=weather_path, time_col="time_bucket", verbose=verbose)

    # 4) 피처 + 타겟
    feat = prep.create_features(agg, dropna=dropna, verbose=verbose)
    base_feature_cols = list(feat.attrs.get("feature_cols", []))
    ext_cols = [c for c in EXTERNAL_FEATURE_COLS if c in feat.columns and c not in base_feature_cols]
    feat.attrs["feature_cols"] = base_feature_cols + ext_cols
    feat.attrs["target_cols"] = target_columns(feat)
    feat.attrs["freq"] = prep.freq
    if verbose:
        print(f"[피처 테이블] {len(feat):,}행 × 피처 {len(feat.attrs['feature_cols'])}개, 셀 {feat['h3_index'].nunique()}개, "
              f"시간칸 {feat['time_bucket'].nunique()}개 ({prep.freq}), 타겟 {feat.attrs['target_cols']}")
    return feat


def save_feature_table(feat: pd.DataFrame, out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    feat.to_csv(out_path, index=False, encoding="utf-8-sig")
    meta = {"feature_cols": feat.attrs.get("feature_cols"), "target_cols": feat.attrs.get("target_cols"),
            "freq": feat.attrs.get("freq"), "rows": len(feat)}
    import json
    with open(os.path.splitext(out_path)[0] + "_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[저장] {out_path} (+ _meta.json)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", default=[os.path.join(ROOT, "data", "sim_logs", "demand_log_*.csv")])
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "processed", "features.csv"))
    ap.add_argument("--weather", default=None)
    ap.add_argument("--freq", default=None)
    a = ap.parse_args()
    feat = build_feature_table(a.logs, freq=a.freq, weather_path=a.weather)
    save_feature_table(feat, a.out)