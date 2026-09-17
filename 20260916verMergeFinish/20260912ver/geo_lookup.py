"""
지역 이름으로 실제 위경도 범위(bounding box)를 조회하는 헬퍼.
OpenStreetMap의 Nominatim 검색 API를 사용합니다. (무료, API 키 불필요)

사용법:
    python geo_lookup.py "강남역"
    python geo_lookup.py "부산 해운대"

찾은 좌표는 region_cache.json에 캐시되어, 같은 지역을 다시 조회할 때
API를 또 호출하지 않고 캐시에서 즉시 가져옵니다.
config_loader.py의 REGION_PRESETS에 항목을 손으로 추가하는 대신,
이 스크립트로 좌표를 조회해서 붙여넣으면 실제 지도 데이터 기반 좌표를 쓸 수 있습니다.

주의: nominatim.openstreetmap.org는 공용 무료 서버라 사용 정책상
초당 1회 이하로 요청해야 합니다. (이 스크립트는 자동으로 그렇게 동작합니다)
"""

import json
import math
import os
import sys
import time
import urllib.request
import urllib.parse

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(ROOT, "region_cache.json")

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
# Nominatim 사용 정책: User-Agent를 명시해야 함 (익명 요청 차단 방지)
USER_AGENT = "ai-mobility-project-region-lookup/1.0"

# 자동완성 검색(search_suggestions)이 너무 자주 호출되지 않도록 하는 간단한 쓰로틀
_last_suggest_call = 0.0
_MIN_SUGGEST_INTERVAL = 1.0  # 초


def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _make_bbox(lat: float, lon: float, margin_deg: float, margin_m: float) -> tuple:
    """중심점(lat, lon)과 반경으로 위경도 사각형을 계산한다."""
    if margin_m is not None:
        lat_margin_deg = margin_m / 111320.0
        lng_margin_deg = margin_m / (111320.0 * math.cos(math.radians(lat)))
    else:
        lat_margin_deg = margin_deg
        lng_margin_deg = margin_deg
    return lat_margin_deg, lng_margin_deg


def _cached_entries_for(cache: dict, place_name: str) -> list:
    """같은 지역명으로 저장된 캐시 항목을 전부 찾는다.

    키 형식이 두 가지 섞여 있다.
      - "강남역::m500.0"  (반경 포함, 현재 형식)
      - "강남역"          (반경 없음, 예전 형식)
    둘 다 같은 지역의 좌표를 담고 있으므로 전부 후보로 본다.
    """
    prefix = place_name + "::"
    keys = [k for k in cache if k == place_name or k.startswith(prefix)]
    # 반경이 붙은 현재 형식을 먼저 쓰고, 없을 때만 반경 없는 예전 형식을 쓴다.
    # 같은 지역이라도 조회 시점에 따라 중심점이 미세하게 다를 수 있으므로
    # 정렬해서 항상 같은 항목을 고르게 한다 (실행할 때마다 결과가 달라지지 않도록).
    keys.sort(key=lambda k: (0 if "::" in k else 1, k))
    return [cache[k] for k in keys]


def _bbox_from_cached_center(cache: dict, place_name: str,
                              margin_deg: float, margin_m: float):
    """캐시에 저장된 사각형의 중심점을 꺼내 새 반경으로 사각형을 다시 만든다.

    반환: lookup_region과 같은 형식의 dict / 쓸 만한 항목이 없으면 None
    """
    for entry in _cached_entries_for(cache, place_name):
        try:
            lat = (float(entry["lat_min"]) + float(entry["lat_max"])) / 2
            lon = (float(entry["lng_min"]) + float(entry["lng_max"])) / 2
        except (KeyError, TypeError, ValueError):
            continue
        lat_margin_deg, lng_margin_deg = _make_bbox(lat, lon, margin_deg, margin_m)
        return {
            "lat_min": round(lat - lat_margin_deg, 6),
            "lat_max": round(lat + lat_margin_deg, 6),
            "lng_min": round(lon - lng_margin_deg, 6),
            "lng_max": round(lon + lng_margin_deg, 6),
            "display_name": entry.get("display_name", place_name),
        }
    return None


def lookup_region(place_name: str, margin_deg: float = 0.006, margin_m: float = None) -> dict:
    """
    place_name(지역명)을 Nominatim으로 검색해서 위경도 범위를 반환.
    margin_deg: 검색된 중심점 기준으로 얼마나 넓은 사각형을 잡을지 (기본 약 600~700m 반경, 도 단위)
    margin_m: 미터 단위로 반경을 지정하고 싶을 때 사용 (지정되면 margin_deg 대신 이걸 씀).
              위도 기준 111,320m/도, 경도는 위도에 따라 보정(cos(lat))해서 정확한 사각형을 만듦.
                Nominatim이 boundingbox를 직접 주기도 하지만, 도로망 시뮬레이션용으로는
                중심점 기준 일정 반경을 잡는 게 더 안정적이라 이 방식을 기본으로 씁니다.
    반환: {"lat_min", "lat_max", "lng_min", "lng_max", "display_name"}
    """
    cache = load_cache()
    # 캐시 키에 반경 정보를 포함시켜서, grid 크기(margin_m)를 바꿨는데
    # 예전 반경으로 캐시된 bbox가 그대로 재사용되는 걸 방지함
    radius_key = f"m{round(margin_m, 1)}" if margin_m is not None else f"d{margin_deg}"
    cache_key = f"{place_name}::{radius_key}"
    if cache_key in cache:
        print(f"[캐시 사용] '{place_name}' → {cache[cache_key]['display_name']}")
        return cache[cache_key]

    # 반경이 다르면 위에서 캐시 미스가 난다. 하지만 같은 지역의 다른 반경 항목이
    # 있으면 그 사각형의 중심점(= 그 지역의 실제 좌표)을 꺼내 새 반경으로
    # 다시 계산할 수 있다. 중심점은 반경과 무관하게 같으므로 API를 다시 부를 필요가 없다.
    # (GUI에서 격자 슬라이더를 조금만 움직여도 Nominatim을 재호출하던 문제 해결)
    reused = _bbox_from_cached_center(cache, place_name, margin_deg, margin_m)
    if reused is not None:
        print(f"[캐시 재계산] '{place_name}' → {reused['display_name']} "
              f"(저장된 중심점을 새 반경으로 다시 계산)")
        cache[cache_key] = reused
        save_cache(cache)
        return reused

    params = urllib.parse.urlencode({
        "q": place_name,
        "format": "json",
        "limit": 1,
    })
    url = f"{NOMINATIM_URL}?{params}"

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    print(f"[조회 중] '{place_name}' → Nominatim API 호출...")
    with urllib.request.urlopen(req, timeout=10) as response:
        data = json.loads(response.read().decode("utf-8"))

    # 정책상 연속 요청 사이 최소 1초 대기
    time.sleep(1.0)

    if not data:
        raise ValueError(f"'{place_name}'에 대한 검색 결과가 없습니다. 지역명을 더 구체적으로 입력해보세요 (예: '강남역, 서울').")

    result = data[0]
    lat = float(result["lat"])
    lon = float(result["lon"])

    lat_margin_deg, lng_margin_deg = _make_bbox(lat, lon, margin_deg, margin_m)

    region = {
        "lat_min": round(lat - lat_margin_deg, 6),
        "lat_max": round(lat + lat_margin_deg, 6),
        "lng_min": round(lon - lng_margin_deg, 6),
        "lng_max": round(lon + lng_margin_deg, 6),
        "display_name": result.get("display_name", place_name),
    }

    cache[cache_key] = region
    save_cache(cache)

    print(f"[완료] {region['display_name']}")
    print(f"       lat: {region['lat_min']} ~ {region['lat_max']}")
    print(f"       lng: {region['lng_min']} ~ {region['lng_max']}")

    return region


def search_suggestions(query: str, limit: int = 5) -> list:
    """
    자동완성용 가벼운 검색. lookup_region()과 달리 결과를 캐시하지 않고
    (사용자가 아직 확정한 지역이 아니므로) 후보 이름 목록만 빠르게 반환합니다.
    실제 좌표 확정/캐싱은 사용자가 최종 선택한 뒤 lookup_region()이 담당합니다.

    반환: [{"display_name", "lat", "lon"}, ...] (실패 시 빈 리스트)
    """
    global _last_suggest_call

    query = (query or "").strip()
    if len(query) < 2:
        return []

    # 초당 1회 정책 준수를 위한 최소 간격 확보 (자동완성은 이미 GUI에서 디바운스되지만 이중 안전장치)
    elapsed = time.time() - _last_suggest_call
    if elapsed < _MIN_SUGGEST_INTERVAL:
        time.sleep(_MIN_SUGGEST_INTERVAL - elapsed)

    params = urllib.parse.urlencode({
        "q": query,
        "format": "json",
        "limit": limit,
        "accept-language": "ko",
    })
    url = f"{NOMINATIM_URL}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception:
        return []
    finally:
        _last_suggest_call = time.time()

    results = []
    for item in data:
        try:
            results.append({
                "display_name": item.get("display_name", query),
                "lat": float(item["lat"]),
                "lon": float(item["lon"]),
            })
        except (KeyError, ValueError):
            continue
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python geo_lookup.py \"지역명\"")
        print("예시: python geo_lookup.py \"강남역\"")
        sys.exit(1)

    place = sys.argv[1]
    result = lookup_region(place)

    print("\n--- config_loader.py의 REGION_PRESETS에 아래처럼 추가하세요 ---")
    print(f'"{place}": {{')
    print(f'    "lat_min": {result["lat_min"]}, "lat_max": {result["lat_max"]},')
    print(f'    "lng_min": {result["lng_min"]}, "lng_max": {result["lng_max"]},')
    print(f'    "temp_min": 15.0, "temp_max": 25.0,  # 온도는 직접 입력 필요 (지도 API가 제공 안 함)')
    print("},")