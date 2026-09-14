"""
Open-Meteo API로 현재 날씨(기온/강수량)를 조회하는 헬퍼. (무료, API 키 불필요)

config_loader.py가 지역 좌표(REGION_PRESETS 또는 geo_lookup.py 조회 결과)를
확정한 뒤, 그 중심점 기준으로 이 모듈을 호출해서 temp_min/temp_max를
mock 값 대신 실시간 값 기반 범위로 덮어씁니다.

geo_lookup.py와 동일한 패턴: API 호출 실패 시 예외를 던지지 않고 None을 반환해서
호출부(config_loader.py)가 항상 기존 폴백(REGION_PRESETS 온도값)으로 넘어갈 수 있게 함.
"""

import json
import urllib.request
import urllib.parse

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def get_current_weather(lat: float, lon: float) -> dict:
    """
    현재 기온(섭씨)과 강수량(mm/h)을 조회.
    반환: {"temperature": float, "precipitation": float} / 실패 시 None
    """
    params = urllib.parse.urlencode({
        "latitude": lat,
        "longitude": lon,
        "current_weather": "true",
        "hourly": "precipitation",
        "timezone": "auto",
    })
    url = f"{OPEN_METEO_URL}?{params}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ai-mobility-project-weather/1.0"})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))

        current = data.get("current_weather", {})
        temp = current.get("temperature")
        if temp is None:
            return None

        # 현재 시각과 가장 가까운 hourly precipitation 값을 찾아 함께 반환 (실패해도 0.0으로 대체)
        precip = 0.0
        try:
            times = data["hourly"]["time"]
            precs = data["hourly"]["precipitation"]
            current_time = current.get("time")
            if current_time in times:
                precip = float(precs[times.index(current_time)])
        except (KeyError, ValueError, IndexError):
            pass

        return {"temperature": float(temp), "precipitation": precip}

    except Exception:
        return None


if __name__ == "__main__":
    # 테스트: 홍대입구 근처 좌표
    result = get_current_weather(37.5563, 126.9236)
    if result:
        print(f"현재 기온: {result['temperature']}℃ / 강수량: {result['precipitation']}mm")
    else:
        print("날씨 조회 실패 (네트워크 확인 필요)")