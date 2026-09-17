"""
config.json 로드 공통 헬퍼
프로젝트 루트에 두고, 각 모듈에서 다음처럼 씁니다:

    from config_loader import CFG
    resolution = CFG["h3_resolution"]
"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT, "config.json")

DEFAULT_CONFIG = {
    # Module 1 - 맵/시뮬레이션 환경
    "grid_x": 3,
    "grid_y": 3,
    "grid_length": 200,
    "num_normal_cars": 20,
    "num_taxis": 3,
    "num_auto_cars": 1,
    "num_obstacles": 2,
    "num_passengers": 5,
    # 지역 선택 (자유 검색어. REGION_PRESETS에 없어도 geo_lookup.py로 실시간 조회 시도)
    "region": "홍대입구",
    # True면 netgenerate 합성 그리드 대신 실제 OSM 지도로 도로망 생성 (real_map_fetch.py)
    "use_real_map": False,
    # Module 2
    "h3_resolution": 8,
    "freq": "5min",
    "max_lag": 6,
    "rolling_short": 3,
    "rolling_long": 6,
    # Module 3 - XGBoost
    "xgb_n_estimators": 100,
    "xgb_max_depth": 6,
    "xgb_learning_rate": 0.1,
    "test_size": 0.2,
    # Module 3 - CNN-LSTM
    "cnn_hidden_dim": 64,
    "cnn_num_layers": 2,
    "cnn_kernel_size": 3,
    "cnn_epochs": 30,
    "cnn_batch_size": 16,
    "cnn_lr": 0.001,
    # Module 4
    "base_fare": 4800,
    "max_multiplier": 3.0,
    "surge_coefficient": 0.4,
    # 시뮬레이션 대상 시간대 (24시간 표기, sim_start_hour <= t < sim_end_hour 구간만 승객 생성)
    # 예: 5 ~ 24 로 두면 새벽 5시부터 자정까지의 하루치 수요를 재현
    "sim_start_hour": 5,
    "sim_end_hour": 24,
    # 택시 운영 전략: "patrol"(도로 전체를 골고루 순찰하다가 눈에 띄면 태움)
    #              vs "prepositioned"(예측/구역 가중치 기반 핫스팟에 미리 가서 대기)
    # A/B 비교 실험의 핵심 스위치입니다.
    "taxi_strategy": "patrol",
    # 배차를 못 받은 채 이 시간(초) 이상 길가에서 대기한 승객은 시뮬레이션에서 소멸(제거)시킴.
    # 이게 없으면 못 태운 승객이 시뮬레이션 끝까지 그대로 쌓여서 wait time 통계가 왜곡됨.
    "passenger_wait_timeout": 900,
}

# 지역 선택 시 위경도 범위 + 계절 온도대가 자동으로 세팅되는 프리셋
# lat_min/max, lng_min/max: 가상 데이터 생성 범위
# temp_min/max: external_data_merge.py 날씨 mock 온도 범위 (실시간 날씨 조회 실패 시 폴백으로 사용)
REGION_PRESETS = {
    "강남역": {
        "lat_min": 37.495, "lat_max": 37.505,
        "lng_min": 127.020, "lng_max": 127.035,
        "temp_min": 15.0, "temp_max": 25.0,
    },
    "홍대입구": {
        "lat_min": 37.550, "lat_max": 37.560,
        "lng_min": 126.920, "lng_max": 126.935,
        "temp_min": 15.0, "temp_max": 25.0,
    },
    "여의도": {
        "lat_min": 37.520, "lat_max": 37.530,
        "lng_min": 126.920, "lng_max": 126.935,
        "temp_min": 14.0, "temp_max": 24.0,
    },
    "제주공항": {
        "lat_min": 33.505, "lat_max": 33.515,
        "lng_min": 126.485, "lng_max": 126.500,
        "temp_min": 18.0, "temp_max": 28.0,
    },
    "NYC(맨해튼)": {
        "lat_min": 40.755, "lat_max": 40.765,
        "lng_min": -73.990, "lng_max": -73.975,
        "temp_min": 5.0, "temp_max": 20.0,
    },
}


def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        merged = {**DEFAULT_CONFIG, **user_cfg}
        print(f"[안내] config.json에서 설정값을 불러왔습니다.")
    else:
        print("[안내] config.json이 없어 기본값을 사용합니다.")
        merged = DEFAULT_CONFIG.copy()

    region = merged.get("region", "홍대입구")
    _apply_region_coords(merged, region)
    _apply_live_weather(merged, region)

    return merged


def _apply_region_coords(merged: dict, region: str) -> None:
    """
    지역 좌표를 결정하는 우선순위:
    1) Nominatim API로 실시간 조회 시도 (geo_lookup.py, 캐시 있으면 캐시로 즉시 응답)
    2) API 실패(인터넷 없음, 서버 다운, 결과 없음 등) → REGION_PRESETS 하드코딩 값으로 자동 대체
    3) 그마저도 없는 지역명이면 → 기본 지역(홍대입구)으로 최종 대체
    이 함수는 merged 딕셔너리를 직접 갱신합니다 (lat_min/max, lng_min/max, temp_min/max).
    temp_min/max는 여기서는 프리셋/기본값 기준으로만 세팅되고, 실제 실시간 값은
    아래 _apply_live_weather()에서 가능하면 덮어씁니다.
    """
    coords = None
    try:
        from geo_lookup import lookup_region  # 지연 임포트: geo_lookup.py가 없어도 config_loader는 죽지 않게
        result = lookup_region(region)
        coords = {
            "lat_min": result["lat_min"], "lat_max": result["lat_max"],
            "lng_min": result["lng_min"], "lng_max": result["lng_max"],
        }
        print(f"[안내] '{region}' 좌표를 실시간 API(Nominatim)로 조회했습니다.")
    except Exception as e:
        print(f"[안내] 좌표 API 조회 실패({type(e).__name__}: {e}) — 하드코딩된 프리셋 값으로 대체합니다.")

    if coords:
        merged.update(coords)
        # 온도는 이 시점엔 Nominatim이 제공하지 않으므로, 프리셋에 있으면 그 값을, 없으면 기본값(15~25도) 사용
        temp_data = REGION_PRESETS.get(region, {})
        merged["temp_min"] = temp_data.get("temp_min", 15.0)
        merged["temp_max"] = temp_data.get("temp_max", 25.0)
    elif region in REGION_PRESETS:
        merged.update(REGION_PRESETS[region])
    else:
        print(f"[안내] '{region}'은(는) 프리셋에도 없어 기본 지역(홍대입구)으로 대체합니다.")
        merged.update(REGION_PRESETS["홍대입구"])


def _apply_live_weather(merged: dict, region: str) -> None:
    """
    확정된 지역 중심좌표 기준으로 Open-Meteo 실시간 날씨를 조회해서
    temp_min/temp_max를 mock 프리셋 값 대신 "현재 실제 기온 ±2도" 범위로 덮어씁니다.
    실패 시(네트워크 없음 등) 아무것도 하지 않고 _apply_region_coords가 세팅한
    프리셋 기반 temp_min/max를 그대로 둡니다 (기존 동작과 동일하게 안전 폴백).
    """
    try:
        from weather_lookup import get_current_weather  # 지연 임포트: 없어도 config_loader는 죽지 않게
        center_lat = (merged["lat_min"] + merged["lat_max"]) / 2
        center_lng = (merged["lng_min"] + merged["lng_max"]) / 2

        weather = get_current_weather(center_lat, center_lng)
        if weather:
            temp = weather["temperature"]
            merged["temp_min"] = round(temp - 2.0, 1)
            merged["temp_max"] = round(temp + 2.0, 1)
            merged["current_temperature"] = round(temp, 1)
            merged["current_precipitation"] = round(weather["precipitation"], 2)
            print(f"[안내] '{region}' 실시간 날씨 반영 완료 (현재 기온 약 {temp:.1f}℃).")
        else:
            print("[안내] 실시간 날씨 조회 실패 — 프리셋 온도 범위를 그대로 사용합니다.")
    except Exception as e:
        print(f"[안내] 날씨 API 조회 중 오류({type(e).__name__}) — 프리셋 온도 범위를 그대로 사용합니다.")


CFG = load_config()