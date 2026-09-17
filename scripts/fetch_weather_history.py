"""
[Module 2 / T4] 과거 시간별 날씨 수집 — Open-Meteo Archive API (무료, 키 불필요).

사용: python scripts/fetch_weather_history.py [시작일 YYYY-MM-DD] [종료일]   (지역은 config.json의 region)
  생략 시: config.json의 region 중심 좌표, sim_date 기준 앞뒤 7일.
출력: data/external/weather_<지역>_<시작>_<종료>.csv  (time, temperature, precipitation, wind_speed)

※ 네트워크가 되는 맥에서 실행. 실패 시 예외 대신 안내만 출력 (파이프라인은 fallback으로 계속 동작).
"""
import os, sys, json, csv, urllib.request, urllib.parse
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config_loader import CFG

URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch(lat, lon, start, end):
    q = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
        "hourly": "temperature_2m,precipitation,wind_speed_10m", "timezone": "Asia/Seoul",
    })
    req = urllib.request.Request(f"{URL}?{q}", headers={"User-Agent": "fetex-module2/1.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())


if __name__ == "__main__":
    base = datetime.strptime(CFG.get("sim_date", "2026-09-07"), "%Y-%m-%d")

    def _is_date(x):
        try: datetime.strptime(x, "%Y-%m-%d"); return True
        except ValueError: return False
    # 날짜 형식(YYYY-MM-DD)인 인자만 사용 — zsh에서 '# 주석'이 인자로 넘어오는 사고 방지
    args = [a for a in sys.argv[1:] if _is_date(a)]
    bad = [a for a in sys.argv[1:] if not _is_date(a) and not a.startswith("#")]
    if len(sys.argv) > 1 and (bad or len(args) != len(sys.argv) - 1):
        print(f"[안내] 날짜(YYYY-MM-DD)가 아닌 인자는 무시: {[a for a in sys.argv[1:] if a not in args]}")
    start = args[0] if len(args) > 0 else (base - timedelta(days=7)).strftime("%Y-%m-%d")
    end = args[1] if len(args) > 1 else (base + timedelta(days=7)).strftime("%Y-%m-%d")
    region = CFG.get("region", "region")
    lat = (CFG["lat_min"] + CFG["lat_max"]) / 2; lon = (CFG["lng_min"] + CFG["lng_max"]) / 2
    # Archive API는 보통 어제까지 제공 → 미래 날짜는 잘라냄
    today = datetime.now().strftime("%Y-%m-%d")
    if end > today:
        print(f"[안내] 종료일 {end}은 아직 관측이 없어 {today}까지로 조정")
        end = today
    try:
        data = fetch(lat, lon, start, end)
        h = data["hourly"]
        os.makedirs(os.path.join(ROOT, "data", "external"), exist_ok=True)
        out = os.path.join(ROOT, "data", "external", f"weather_{region}_{start}_{end}.csv")
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["time", "temperature", "precipitation", "wind_speed"])
            for row in zip(h["time"], h["temperature_2m"], h["precipitation"], h["wind_speed_10m"]):
                w.writerow(row)
        print(f"[저장] {out} ({len(h['time'])}시간, 좌표 {lat:.4f},{lon:.4f})")
    except Exception as e:
        print(f"[실패] 날씨 수집 실패({type(e).__name__}: {e}). 파이프라인은 fallback(상수)으로 동작합니다.")
