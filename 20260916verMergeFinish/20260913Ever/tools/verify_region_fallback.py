# -*- coding: utf-8 -*-
"""
[1번 담당 검증] 지역 좌표(Nominatim) / 날씨(Open-Meteo) API 폴백 자동 검증

역할 1번(지역·지도·외부데이터)의 검증 항목 중
  - geo_lookup.py: 좌표 조회 실패 시 REGION_PRESETS 폴백
  - weather_lookup.py: 날씨 조회 실패 시 temp_min/temp_max 폴백
두 가지를 네트워크 상태와 무관하게 재현 가능하게 확인한다.

실제 API를 부르지 않고 lookup_region / get_current_weather 를 성공·실패로
바꿔가며 load_config() 결과를 비교하므로, 인터넷이 끊겨 있어도 항상 같은 결과가 나온다.

실행:
    python tools/verify_region_fallback.py
종료 코드 0 = 전부 통과, 1 = 실패 항목 있음
"""
import contextlib
import io
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config_loader          # noqa: E402
import geo_lookup             # noqa: E402
import weather_lookup         # noqa: E402
from config_loader import REGION_PRESETS  # noqa: E402

MOCK_COORDS = {"lat_min": 37.0, "lat_max": 37.1,
               "lng_min": 127.0, "lng_max": 127.1, "display_name": "MOCK"}


def _load(region, geo_ok, weather):
    """lookup_region / get_current_weather 를 강제로 성공·실패시키고 load_config() 실행"""
    if geo_ok:
        geo_lookup.lookup_region = (
            lambda r, margin_deg=0.006, margin_m=None: dict(MOCK_COORDS))
    else:
        def _boom(*a, **k):
            raise OSError("simulated API failure")
        geo_lookup.lookup_region = _boom
    weather_lookup.get_current_weather = lambda lat, lon: weather
    sys.modules["geo_lookup"] = geo_lookup
    sys.modules["weather_lookup"] = weather_lookup

    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"region": region, "grid_x": 10, "grid_y": 10, "grid_length": 100},
                  f, ensure_ascii=False)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            cfg = config_loader.load_config(path)
    finally:
        os.remove(path)
    return cfg, buf.getvalue()


def main():
    results = []

    def check(name, ok, detail=""):
        results.append((name, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    print("\n1) 좌표 API 실패 시 REGION_PRESETS 폴백")
    for region, preset in REGION_PRESETS.items():
        cfg, log = _load(region, geo_ok=False, weather=None)
        ok = (cfg["lat_min"] == preset["lat_min"]
              and cfg["lng_min"] == preset["lng_min"]
              and cfg["temp_min"] == preset["temp_min"]
              and cfg["region"] == region)
        check(f"{region}: 프리셋 좌표·온도로 복구", ok,
              f"lat_min={cfg['lat_min']}, temp={cfg['temp_min']}~{cfg['temp_max']}")

    print("\n2) 날씨 API 실패 시 temp_min/temp_max 폴백")
    for region in ("강남역", "제주공항"):
        preset = REGION_PRESETS[region]
        cfg, _ = _load(region, geo_ok=True, weather=None)
        ok = cfg["temp_min"] == preset["temp_min"] and cfg["temp_max"] == preset["temp_max"]
        check(f"{region}: 프리셋 온도 유지", ok, f"{cfg['temp_min']}~{cfg['temp_max']}")
        cfg, _ = _load(region, geo_ok=True, weather={"temperature": 30.0, "precipitation": 0.0})
        ok = cfg["temp_min"] == 28.0 and cfg["temp_max"] == 32.0 and cfg.get("current_temperature") == 30.0
        check(f"{region}: 실시간 성공 시 실측 +-2도로 덮어씀", ok,
              f"{cfg['temp_min']}~{cfg['temp_max']}")

    print("\n3) 프리셋에 없는 지역")
    cfg, log = _load("부산 해운대", geo_ok=False, weather=None)
    swapped = cfg["region"] != "부산 해운대"
    check("좌표 실패 시 다른 지역으로 대체되는지 감지", swapped,
          f"요청=부산 해운대 → 실제={cfg['region']} (지도가 통째로 바뀜)")
    cfg, _ = _load("부산 해운대", geo_ok=True, weather=None)
    wrong_temp = cfg["temp_min"] == 15.0 and cfg["temp_max"] == 25.0
    check("좌표 성공+날씨 실패 시 온도가 하드코딩 기본값인지 감지", wrong_temp,
          f"좌표={cfg['lat_min']} (실측) / 온도={cfg['temp_min']}~{cfg['temp_max']} (서울 기본값)")

    failed = [n for n, ok, _ in results if not ok]
    print(f"\n총 {len(results)}건 중 통과 {len(results) - len(failed)}건, 실패 {len(failed)}건")
    if failed:
        for n in failed:
            print(f"  - {n}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
