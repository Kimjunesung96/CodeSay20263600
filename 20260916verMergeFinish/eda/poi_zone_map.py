"""
POI zone 검증 (역할 3번 — 할 일 1).

poi_extractor.py가 뽑은 카테고리별 zone(edge 목록)이 실제 지도와 맞는지 확인.
  1) OSM에서 파싱한 POI 좌표 수 vs 매핑된 edge 수 (몇 개가 어디로 갔나)
  2) POI → 가장 가까운 edge 거리 분포, 매핑 반경(80m) 바꿔가며 edge 수 변화
  3) edge 하나가 몇 개 카테고리에 겹치는지
  4) 지도 출력: (a) 브라우저용 Leaflet HTML — 카테고리 레이어 켜고 끄며 실제 지도 위에서 확인
               (b) PNG 2장 — 카테고리별 edge, 겹침 수별 edge

사용: python eda/poi_zone_map.py
출력: data/eda/poi_zone_map.html, poi_zone_map.png, poi_zone_summary.csv
"""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import sumolib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from poi_extractor import parse_osm_pois, CATEGORIES

for f in ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic"]:
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f; break
plt.rcParams["axes.unicode_minus"] = False

CFG_DIR = os.path.join(ROOT, "module1_simulation", "sumo_config")
NET = os.path.join(CFG_DIR, "grid.net.xml"); OSM = os.path.join(CFG_DIR, "region.osm.xml")
META = os.path.join(CFG_DIR, "runtime_meta.json"); OUT = os.path.join(ROOT, "data", "eda")
os.makedirs(OUT, exist_ok=True)
KO = {"school": "학교", "residential": "주택", "company": "회사", "restaurant": "음식점",
      "subway_entrance": "지하철입구", "bus_stop": "버스정류장"}
# 카테고리 색 (고정 순서, 텍스트는 잉크색 사용)
COLOR = {"school": "#d62728", "residential": "#2ca02c", "company": "#1f77b4",
         "restaurant": "#ff7f0e", "subway_entrance": "#9467bd", "bus_stop": "#8c564b"}

net = sumolib.net.readNet(NET)
zones = json.load(open(META, encoding="utf-8"))["zones"]
pois = parse_osm_pois(OSM)
all_edges = [e for e in net.getEdges() if e.getFunction() != "internal" and e.allows("passenger")]
print(f"도로 edge {len(all_edges)}개 (통행 가능)\n")

# ---------- 1) POI 수 vs edge 수, 2) 거리·반경 민감도 ----------
def nearest_edge(lat, lon):
    x, y = net.convertLonLat2XY(lon, lat)
    near = net.getNeighboringEdges(x, y, r=300)
    near = [(e, d) for e, d in near if e.getFunction() != "internal" and e.allows("passenger")]
    if not near: return None, np.inf
    e, d = min(near, key=lambda p: p[1]); return e.getID(), d

rows, dist_by_cat = [], {}
for cat in CATEGORIES:
    ds = [nearest_edge(la, lo)[1] for la, lo in pois[cat]]
    dist_by_cat[cat] = np.array(ds)
    r = {"카테고리": KO[cat], "OSM POI 수": len(ds), "현재 zone edge 수(80m)": len(zones.get(cat, []))}
    for rad in (40, 80, 120, 200):
        ids = {nearest_edge(la, lo)[0] for la, lo in pois[cat] if nearest_edge(la, lo)[1] <= rad}
        ids.discard(None); r[f"edge 수 @{rad}m"] = len(ids)
    r["80m 밖 POI(누락) 수"] = int((dist_by_cat[cat] > 80).sum())
    r["POI→edge 거리 중앙값(m)"] = round(float(np.median(ds)), 1) if ds else np.nan
    rows.append(r)
summary = pd.DataFrame(rows)
print(summary.to_string(index=False))

# ---------- 3) 겹침 ----------
edge_cats = {}
for cat, es in zones.items():
    for e in es: edge_cats.setdefault(e, []).append(cat)
n_cats = pd.Series({e: len(c) for e, c in edge_cats.items()})
print(f"\nzone에 속한 edge {len(edge_cats)}/{len(all_edges)}개, 카테고리 수별: "
      + ", ".join(f"{k}개={v}" for k, v in sorted(n_cats.value_counts().items())))
excl = {cat: sum(1 for e in zones[cat] if len(edge_cats[e]) == 1) for cat in CATEGORIES}
print("카테고리별 '그 카테고리에만' 속한 edge 수: " + ", ".join(f"{KO[c]} {excl[c]}/{len(zones[c])}" for c in CATEGORIES))
combo = pd.Series(["|".join(sorted(c, key=CATEGORIES.index)) for c in edge_cats.values()]).value_counts()
print("\n가장 흔한 조합 Top 8:"); print(combo.head(8).to_string())
summary.to_csv(os.path.join(OUT, "poi_zone_summary.csv"), index=False, encoding="utf-8-sig")

# 호출 로그가 있으면 edge별 호출 수도 붙임
from module2_preprocessing.sim_log_recorder import latest_log_path
calls = {}
lp = latest_log_path()
if lp:
    calls = pd.read_csv(lp)["edge_id"].value_counts().to_dict()

# ---------- 4a) Leaflet HTML ----------
def edge_latlon(e):
    return [list(net.convertXY2LonLat(x, y))[::-1] for x, y in e.getShape()]
layers = {}
for cat in CATEGORIES:
    feats = []
    for eid in zones.get(cat, []):
        try: e = net.getEdge(eid)
        except Exception: continue
        feats.append({"id": eid, "pts": edge_latlon(e), "cats": [KO[c] for c in edge_cats[eid]], "calls": calls.get(eid, 0)})
    layers[cat] = {"edges": feats, "pois": [[la, lo] for la, lo in pois[cat]]}
others = [{"id": e.getID(), "pts": edge_latlon(e)} for e in all_edges if e.getID() not in edge_cats]
lats = [p[0] for l in layers.values() for f in l["edges"] for p in f["pts"]]
lons = [p[1] for l in layers.values() for f in l["edges"] for p in f["pts"]]
center = [float(np.mean(lats)), float(np.mean(lons))]
html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>POI zone 검증 지도</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body,#map{{height:100%;margin:0}} .legend{{background:#fff;padding:8px 10px;font:13px -apple-system,sans-serif;line-height:1.7;border-radius:4px;box-shadow:0 1px 4px rgba(0,0,0,.3)}}
.legend i{{display:inline-block;width:18px;height:4px;margin-right:6px;vertical-align:middle}} .legend .dot{{width:8px;height:8px;border-radius:50%}}</style></head>
<body><div id="map"></div><script>
const DATA={json.dumps(layers, ensure_ascii=False)}; const OTHERS={json.dumps(others)}; const COLOR={json.dumps(COLOR)}; const KO={json.dumps(KO, ensure_ascii=False)};
const map=L.map('map').setView({center},16);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',{{maxZoom:19,attribution:'&copy; OpenStreetMap'}}).addTo(map);
const base=L.layerGroup(OTHERS.map(o=>L.polyline(o.pts,{{color:'#999',weight:2,opacity:.5}}).bindTooltip(o.id+' (zone 없음)'))).addTo(map);
const overlays={{'zone 없는 도로':base}};
for(const cat in DATA){{
  const g=L.layerGroup();
  DATA[cat].edges.forEach(f=>L.polyline(f.pts,{{color:COLOR[cat],weight:5,opacity:.8}})
    .bindTooltip(`<b>${{f.id}}</b><br>구역: ${{f.cats.join(' | ')}}<br>호출: ${{f.calls}}건`).addTo(g));
  DATA[cat].pois.forEach(p=>L.circleMarker(p,{{radius:4,color:'#fff',weight:1,fillColor:COLOR[cat],fillOpacity:1}}).bindTooltip(KO[cat]+' POI').addTo(g));
  g.addTo(map); overlays[`${{KO[cat]}} (edge ${{DATA[cat].edges.length}} / POI ${{DATA[cat].pois.length}})`]=g;
}}
L.control.layers(null,overlays,{{collapsed:false}}).addTo(map);
const lg=L.control({{position:'bottomleft'}}); lg.onAdd=()=>{{const d=L.DomUtil.create('div','legend');
d.innerHTML='<b>굵은 선</b> = zone edge, <b>점</b> = OSM POI 위치<br>선 위에 마우스: 소속 구역·호출 수<br>오른쪽 체크박스로 카테고리 켜고 끄기';return d;}}; lg.addTo(map);
</script></body></html>"""
open(os.path.join(OUT, "poi_zone_map.html"), "w", encoding="utf-8").write(html)

# ---------- 4b) PNG ----------
fig, axes = plt.subplots(1, 2, figsize=(18, 9))
ink, muted = "#2b2b2b", "#8a8a8a"
for ax in axes:
    for e in all_edges:
        xs, ys = zip(*e.getShape()); ax.plot(xs, ys, color="#d0d0d0", lw=1, zorder=1)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
ax = axes[0]
for cat in CATEGORIES:
    for eid in zones.get(cat, []):
        try: xs, ys = zip(*net.getEdge(eid).getShape())
        except Exception: continue
        ax.plot(xs, ys, color=COLOR[cat], lw=2.5, alpha=.75, zorder=2)
    if pois[cat]:
        xy = np.array([net.convertLonLat2XY(lo, la) for la, lo in pois[cat]])
        ax.scatter(xy[:, 0], xy[:, 1], s=14, color=COLOR[cat], edgecolor="white", lw=.5, zorder=3)
ax.set_title("카테고리별 zone edge(선)와 OSM POI(점)", loc="left", color=ink)
ax.legend(handles=[Line2D([], [], color=COLOR[c], lw=3, label=f"{KO[c]} edge {len(zones[c])} / POI {len(pois[c])}") for c in CATEGORIES],
          loc="lower left", fontsize=9, frameon=False, labelcolor=ink)
ax = axes[1]
seq = {1: "#c6dbef", 2: "#6baed6", 3: "#2171b5", 4: "#08306b"}
for eid, cs in edge_cats.items():
    try: xs, ys = zip(*net.getEdge(eid).getShape())
    except Exception: continue
    ax.plot(xs, ys, color=seq[min(len(cs), 4)], lw=2.5 + len(cs), zorder=2)
ax.set_title("edge당 겹치는 카테고리 수", loc="left", color=ink)
ax.legend(handles=[Line2D([], [], color=seq[k], lw=3 + k, label=f"{k}개 카테고리 — {int((n_cats == k).sum())}개 edge") for k in sorted(seq)],
          loc="lower left", fontsize=9, frameon=False, labelcolor=ink)
fig.suptitle("POI zone 검증 — 강남역 (poi_extractor 매핑 반경 80m)", x=0.01, ha="left", fontsize=13, color=ink)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(os.path.join(OUT, "poi_zone_map.png"), dpi=120, bbox_inches="tight", facecolor="white")
print(f"\n[저장] {OUT}/poi_zone_map.html (브라우저로 열기), poi_zone_map.png, poi_zone_summary.csv")
