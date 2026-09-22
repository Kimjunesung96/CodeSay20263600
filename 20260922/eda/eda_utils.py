"""
[Module 2 / T5] EDA 노트북(notebooks/01_EDA_and_Spatial.ipynb)이 호출하는 분석 함수 모음.
노트북 셀은 얇게(호출+결론), 계산은 여기 — 테스트·재사용 가능하게.

포함: 데이터 개요 / 시간 패턴 / 공간 패턴 / 정상성(ADF) / 자기상관(ACF) / 피처 유효성(ablation) / 외부 변수 상관
"""
import os, sys, warnings
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

warnings.filterwarnings("ignore")
for f in ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Malgun Gothic"]:
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f; break
plt.rcParams["axes.unicode_minus"] = False
INK, MUTED, BLUE = "#2b2b2b", "#8a8a8a", "#1f77b4"


def _style(ax, title=None):
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#cccccc"); ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(colors=MUTED, labelsize=9); ax.grid(axis="y", color="#eeeeee")
    if title: ax.set_title(title, loc="left", color=INK, fontsize=11)


# ---------- 1. 개요 ----------
def overview(feat: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame:
    span = logs["pickup_datetime"].max() - logs["pickup_datetime"].min()
    rows = {
        "호출 건수": f"{len(logs):,}",
        "기간": f"{logs['pickup_datetime'].min()} ~ {logs['pickup_datetime'].max()} ({span})",
        "일수": logs["pickup_datetime"].dt.date.nunique(),
        "H3 셀 수": feat["h3_index"].nunique(),
        "시간칸 수(피처 테이블)": feat["time_bucket"].nunique(),
        "피처 테이블 행": len(feat),
        "수요 0인 칸 비율": f"{(feat['demand'] == 0).mean() * 100:.1f}%",
        "셀당 평균 수요/칸": f"{feat['demand'].mean():.2f}",
        "날씨 결측 비율": f"{feat['weather_missing'].mean() * 100:.1f}%" if "weather_missing" in feat else "-",
    }
    return pd.DataFrame(rows.items(), columns=["항목", "값"]).set_index("항목")


# ---------- 2. 시간 패턴 ----------
def plot_time_curve(agg: pd.DataFrame, freq_label="5분"):
    s = agg.groupby("time_bucket")["demand"].sum()
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(s.index, s.values, color=BLUE, lw=2)
    ax.fill_between(s.index, s.values, color=BLUE, alpha=.08)
    _style(ax, f"{freq_label} 단위 총 호출 수 (전체 셀 합)")
    ax.set_ylabel("호출 수", color=MUTED)
    peak = s.idxmax()
    ax.annotate(f"피크 {s.max()}건\n{peak:%H:%M}", (peak, s.max()), textcoords="offset points", xytext=(8, -4), fontsize=9, color=INK)
    return fig, s


def hour_dow_heatmap(agg: pd.DataFrame):
    """시간대×요일 히트맵. 데이터가 1일이면 요일 1행만 나온다 (그 경우 노트북에서 '예비'로 표기)."""
    d = agg.copy(); d["hour"] = d["time_bucket"].dt.hour; d["dow"] = d["time_bucket"].dt.dayofweek
    days = d.groupby("dow")["time_bucket"].apply(lambda t: t.dt.date.nunique())
    tab = d.groupby(["dow", "hour"])["demand"].sum().unstack("hour").fillna(0)
    tab = tab.div(days, axis=0)  # 요일별 일수로 나눠 '하루 평균'
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(tab) + 1.5))
    im = ax.imshow(tab.values, aspect="auto", cmap="Blues")
    ax.set_yticks(range(len(tab))); ax.set_yticklabels(["월", "화", "수", "목", "금", "토", "일"][i] for i in tab.index)
    ax.set_xticks(range(len(tab.columns))); ax.set_xticklabels(tab.columns, fontsize=8)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(length=0, colors=MUTED)
    ax.set_title(f"시간대 × 요일 호출 수 (하루 평균) — 데이터 {d['time_bucket'].dt.date.nunique()}일", loc="left", color=INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01); cb.outline.set_visible(False)
    return fig, tab


# ---------- 3. 공간 패턴 ----------
def top_cells(agg: pd.DataFrame, n=10) -> pd.DataFrame:
    t = agg.groupby("h3_index")["demand"].agg(총수요="sum", 평균="mean", 최대="max", 비율=lambda s: s.sum())
    t["비율"] = (t["비율"] / t["총수요"].sum() * 100).round(1).astype(str) + "%"
    return t.sort_values("총수요", ascending=False).head(n).round(2)


def cell_time_heatmap(agg: pd.DataFrame, top=25):
    mat = agg.pivot(index="h3_index", columns="time_bucket", values="demand").fillna(0)
    mat = mat.loc[mat.sum(axis=1).sort_values(ascending=False).index].head(top)
    fig, ax = plt.subplots(figsize=(11, 0.28 * len(mat) + 1.5))
    im = ax.imshow(mat.values, aspect="auto", cmap="Blues", interpolation="nearest")
    ax.set_yticks(range(len(mat))); ax.set_yticklabels(mat.index, fontsize=7, color=MUTED)
    step = max(1, len(mat.columns) // 12)
    ax.set_xticks(range(0, len(mat.columns), step)); ax.set_xticklabels([t.strftime("%m/%d %H:%M") for t in mat.columns[::step]], fontsize=8, rotation=0, color=MUTED)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(f"셀 × 시간 수요 히트맵 (상위 {len(mat)}셀, 수요 많은 순)", loc="left", color=INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01); cb.outline.set_visible(False)
    return fig, mat


# ---------- 5. 정상성 ----------
def stationarity(agg: pd.DataFrame) -> pd.DataFrame:
    """전체 수요 시계열 ADF 검정 (원계열, 1차 차분). statsmodels 없으면 안내."""
    s = agg.groupby("time_bucket")["demand"].sum().astype(float)
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        return pd.DataFrame({"안내": ["statsmodels 미설치: pip install statsmodels"]})
    rows = []
    for name, x in (("원계열", s), ("1차 차분", s.diff().dropna())):
        if len(x) < 10:
            rows.append({"계열": name, "ADF 통계량": np.nan, "p-value": np.nan, "결론": "표본 부족(<10)"}); continue
        stat, p, *_ = adfuller(x, autolag="AIC")
        rows.append({"계열": name, "ADF 통계량": round(stat, 3), "p-value": round(p, 4),
                     "결론": "정상(p<0.05)" if p < 0.05 else "비정상(단위근 의심)"})
    return pd.DataFrame(rows).set_index("계열")


# ---------- 6. 자기상관 ----------
def acf_by_cell(feat_or_agg: pd.DataFrame, max_lag=24, min_std=0.0):
    """셀별 ACF를 구해 평균. 유의 기준 ±1.96/√N (N=칸 수)."""
    acfs = []
    for _, g in feat_or_agg.sort_values("time_bucket").groupby("h3_index"):
        x = g["demand"].to_numpy(dtype=float)
        if x.std() <= min_std or len(x) <= max_lag + 2: continue
        x = x - x.mean(); den = (x * x).sum()
        acfs.append([1.0] + [(x[:-k] * x[k:]).sum() / den for k in range(1, max_lag + 1)])
    if not acfs:
        return None, None, 0
    acf = np.mean(acfs, axis=0)
    n = feat_or_agg["time_bucket"].nunique(); ci = 1.96 / np.sqrt(n)
    sig = int(np.argmax(np.abs(acf[1:]) < ci)) if (np.abs(acf[1:]) < ci).any() else max_lag
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.bar(range(max_lag + 1), acf, color=BLUE, width=.6)
    ax.axhline(ci, color="#d62728", lw=1, ls="--"); ax.axhline(-ci, color="#d62728", lw=1, ls="--")
    _style(ax, f"셀 평균 ACF (lag 단위: 5분칸) — 유의 lag ≈ {sig}개 (±{ci:.2f} 기준)")
    ax.set_xlabel("lag", color=MUTED)
    return fig, acf, sig


# ---------- 7. 피처 유효성 (ablation) ----------
FEATURE_GROUPS = {
    "lag만": lambda cols: [c for c in cols if c.startswith("lag_")],
    "+rolling/diff": lambda cols: [c for c in cols if c.startswith(("lag_", "rolling_", "diff_", "same_time", "has_last"))],
    "+시간(calendar)": lambda cols: [c for c in cols if c.startswith(("lag_", "rolling_", "diff_", "same_time", "has_last"))
                                   or c in ("hour", "minute_of_day", "dayofweek", "is_weekend", "is_holiday", "time_slot_code", "hour_sin", "hour_cos", "dow_sin", "dow_cos")],
    "+날씨(전체)": lambda cols: list(cols),
}


def feature_ablation(feat: pd.DataFrame, target="y_h1", test_size=0.2, seed=42) -> pd.DataFrame:
    """피처 그룹을 누적 추가하며 같은 모델로 학습 → test RMSE/MAE. 그룹 간 차이가 그 그룹의 기여."""
    from module2_preprocessing.time_series_prep import time_based_split
    try:
        from xgboost import XGBRegressor
        Model = lambda: XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=seed, n_jobs=1, verbosity=0)
        mname = "XGBoost"
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        Model = lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=seed); mname = "GBR(sklearn)"
    cols_all = feat.attrs.get("feature_cols") or [c for c in feat.columns if c not in ("time_bucket", "h3_index", "demand", "time_slot", "weather_source") and not c.startswith("y_h")]
    cols_all = [c for c in cols_all if feat[c].dtype != object]
    tr, _, te, _ = time_based_split(feat, test_size)
    base_rmse = float(np.sqrt(np.mean((te[target] - te["lag_1"]) ** 2)))  # 나이브: 다음 칸 = 직전 칸
    rows = [{"피처 그룹": "나이브(=lag_1)", "피처 수": 1, "RMSE": round(base_rmse, 3), "MAE": round(float(np.mean(np.abs(te[target] - te["lag_1"]))), 3)}]
    for name, pick in FEATURE_GROUPS.items():
        cols = [c for c in pick(cols_all) if c in feat.columns]
        m = Model().fit(tr[cols].fillna(0).values, tr[target].values)
        p = np.maximum(0, m.predict(te[cols].fillna(0).values))
        rows.append({"피처 그룹": name, "피처 수": len(cols), "RMSE": round(float(np.sqrt(np.mean((te[target] - p) ** 2))), 3),
                     "MAE": round(float(np.mean(np.abs(te[target] - p))), 3)})
    out = pd.DataFrame(rows).set_index("피처 그룹"); out.attrs["model"] = mname; out.attrs["n_test"] = len(te)
    return out


# ---------- 8. 외부 변수 상관 ----------
def external_correlation(feat: pd.DataFrame) -> pd.DataFrame:
    s = feat.groupby("time_bucket").agg(demand=("demand", "sum"), temperature=("temperature", "first"),
                                        precipitation=("precipitation", "first"), is_holiday=("is_holiday", "first"),
                                        dayofweek=("dayofweek", "first"), hour=("hour", "first"))
    rows = []
    for c in ("temperature", "precipitation", "is_holiday", "dayofweek", "hour"):
        x = s[c]
        if x.nunique() < 2:
            rows.append({"변수": c, "고유값 수": x.nunique(), "상관(Pearson)": np.nan, "판단": "변동 없음 → 현재 데이터로 판단 불가"}); continue
        r = float(np.corrcoef(x.fillna(x.mean()), s["demand"])[0, 1])
        rows.append({"변수": c, "고유값 수": x.nunique(), "상관(Pearson)": round(r, 3),
                     "판단": "강함" if abs(r) > .5 else "중간" if abs(r) > .3 else "약함"})
    return pd.DataFrame(rows).set_index("변수")
