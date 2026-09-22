"""
H3 resolution 비교 EDA (역할 3번 — 할 일 2).

같은 호출 로그를 res 7/8/9/10으로 각각 셀에 배정해서
  - 셀 수, 셀당 10분 평균 수요, 수요 0인 칸 비율, 최다 셀 점유율,
  - lag-1 자기상관(셀 평균) — 이전 칸으로 다음 칸을 얼마나 맞출 수 있나
를 비교하고, 셀×시간 히트맵을 나란히 그린다.

사용: python eda/h3_resolution_compare.py [로그CSV]   (생략 시 최신 로그)
출력: data/eda/h3_resolution_summary.csv, data/eda/h3_resolution_compare.png
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import warnings; warnings.filterwarnings('ignore', category=RuntimeWarning)
import numpy as np, pandas as pd, h3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from module2_preprocessing.sim_log_recorder import latest_log_path

# 한글 폰트 (맥 기본)
for f in ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic"]:
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f; break
plt.rcParams["axes.unicode_minus"] = False

FREQ = "10min"; RESOLUTIONS = [7, 8, 9, 10]; MAX_LAG = 6

log_path = sys.argv[1] if len(sys.argv) > 1 else latest_log_path()
df = pd.read_csv(log_path).dropna(subset=["latitude", "longitude"])
df["t"] = pd.to_datetime(df["pickup_datetime"])
sim_start = (df["t"] - pd.to_timedelta(df["request_sec"], unit="s")).min().floor(FREQ)
buckets = pd.date_range(sim_start, df["t"].max().floor(FREQ), freq=FREQ)
df["bucket"] = df["t"].dt.floor(FREQ)
print(f"로그: {log_path}\n호출 {len(df)}건, 시간칸 {len(buckets)}개 ({buckets[0]} ~ {buckets[-1]})\n")

rows, mats = [], {}
for res in RESOLUTIONS:
    df["cell"] = [h3.latlng_to_cell(a, b, res) for a, b in zip(df.latitude, df.longitude)]
    cells = sorted(df["cell"].unique())
    mat = (df.groupby(["cell", "bucket"]).size()
             .unstack("bucket").reindex(index=cells, columns=buckets).fillna(0).astype(int))
    mat = mat.loc[mat.sum(axis=1).sort_values(ascending=False).index]  # 수요 많은 셀부터
    mats[res] = mat
    vals = mat.values
    # 셀별 lag-1 자기상관 (분산 0인 셀 제외)
    ac = []
    for r in vals:
        if r.std() > 0 and len(r) > 2:
            ac.append(np.corrcoef(r[:-1], r[1:])[0, 1])
    rows.append({
        "resolution": res,
        "셀 변 길이(m)": round(h3.average_hexagon_edge_length(res, unit="m")),
        "셀 수": len(cells),
        "셀당 평균 수요/10분": round(vals.mean(), 2),
        "셀당 중앙값 수요/10분": float(np.median(vals)),
        "수요 0인 칸 비율(%)": round((vals == 0).mean() * 100, 1),
        "최다 셀 점유율(%)": round(vals.sum(axis=1).max() / vals.sum() * 100, 1),
        "lag-1 자기상관(셀 평균)": round(float(np.nanmean(ac)), 3) if ac else np.nan,
        f"학습 행 수(lag{MAX_LAG} 제거 후)": len(cells) * max(0, len(buckets) - MAX_LAG),
    })

summary = pd.DataFrame(rows)
os.makedirs(os.path.join(ROOT, "data", "eda"), exist_ok=True)
summary.to_csv(os.path.join(ROOT, "data", "eda", "h3_resolution_summary.csv"), index=False, encoding="utf-8-sig")
print(summary.to_string(index=False))

# ---- 그림: 셀×시간 히트맵 4장 (단일 색상 sequential) ----
# 색 스케일은 패널별 최대값 기준 (공통 스케일이면 res 7의 400건에 맞춰져 res 9/10이 전부 흰색으로 나옴)
fig, axes = plt.subplots(1, 4, figsize=(18, 7), gridspec_kw={"width_ratios": [1, 1, 1, 1]})
ink, muted = "#2b2b2b", "#8a8a8a"
for ax, res in zip(axes, RESOLUTIONS):
    m = mats[res]
    vmax = m.values.max()
    im = ax.imshow(m.values, aspect="auto", cmap="Blues", vmin=0, vmax=vmax, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.outline.set_visible(False); cbar.ax.tick_params(color=muted, labelcolor=muted, labelsize=8)
    s = summary.set_index("resolution").loc[res]
    ax.set_title(f"res {res}  ·  셀 {int(s['셀 수'])}개  ·  변 {int(s['셀 변 길이(m)'])}m\n"
                 f"0인 칸 {s['수요 0인 칸 비율(%)']}%  ·  lag-1 r={s['lag-1 자기상관(셀 평균)']}  ·  최대 {int(vmax)}건",
                 fontsize=10, color=ink, loc="left")
    ax.set_xticks(range(0, len(buckets), 3))
    ax.set_xticklabels([b.strftime("%H:%M") for b in buckets[::3]], fontsize=8, color=muted)
    ax.set_yticks([]); ax.set_ylabel("셀 (수요 많은 순)", color=muted, fontsize=9)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(length=0)
fig.suptitle(f"H3 resolution별 셀×시간 수요 분포 (색: 10분당 호출 수, 패널별 스케일) — {os.path.basename(log_path)}",
             x=0.01, ha="left", fontsize=12, color=ink)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = os.path.join(ROOT, "data", "eda", "h3_resolution_compare.png")
fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
print(f"\n[저장] {out}")
