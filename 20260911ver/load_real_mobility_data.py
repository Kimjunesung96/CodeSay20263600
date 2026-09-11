"""
realdata/ 폴더에 넣은 생활이동(OD, 셀 단위 이동인구) 원본 CSV들을
지역(강남역/홍대입구 등)별로 필터링해서 data/raw/realdata_<지역>.csv로 뽑아내는 스크립트.

[좌표계 판단 근거]
원본 컬럼(o_cell_x, o_cell_y / d_cell_x, d_cell_y)의 값 범위(96만/197만대)를
EPSG:5179(한국측지계 UTM-K, 국가 공간정보 표준)로 변환해보면 실제 서울 지역
위경도(예: 37.72, 127.05)로 정확히 떨어짐 -> EPSG:5179로 확정.

pyproj로 EPSG:5179 -> EPSG:4326(WGS84 위경도) 변환한 뒤,
config_loader.py가 이미 쓰고 있는 REGION_PRESETS / region_cache.json의
지역 bbox(강남역, 홍대입구 등)로 필터링합니다. 즉 지역 bbox를 새로 정의할
필요 없이 프로젝트에 이미 있는 값을 그대로 재사용합니다.

사용법:
    1. 프로젝트 루트에 realdata/ 폴더를 만들고 원본 CSV(들)를 넣는다.
    2. pip install pyproj --break-system-packages   (아직 없다면)
    3. python load_real_mobility_data.py 강남역
       python load_real_mobility_data.py 홍대입구
       python load_real_mobility_data.py 강남역 destination   # 도착지 기준으로 필터
       python load_real_mobility_data.py 강남역 both          # 출발/도착 둘 다 해당 지역

    -> data/raw/realdata_<지역명>.csv 생성됨. 원본 컬럼 그대로 + o_lat/o_lon/d_lat/d_lon
       (변환된 위경도) 컬럼이 추가된 상태로 저장됩니다.

[주의] realdata/ 폴더 안의 CSV가 여러 개면 전부 훑어서 합쳐줍니다(여러 날짜/여러 원본 파일 가능).
"""
import os
import sys
import glob
import json
import pandas as pd

try:
    from pyproj import Transformer
except ImportError:
    print("[오류] pyproj가 설치되어 있지 않습니다. 먼저 설치하세요:")
    print("       pip install pyproj --break-system-packages")
    sys.exit(1)

ROOT = os.path.dirname(os.path.abspath(__file__))
REALDATA_DIR = os.path.join(ROOT, "realdata")
OUTPUT_DIR = os.path.join(ROOT, "data", "raw")
REGION_CACHE_PATH = os.path.join(ROOT, "region_cache.json")

sys.path.insert(0, ROOT)
from config_loader import REGION_PRESETS

# 원본 좌표계(EPSG:5179, 한국측지계 UTM-K) -> 위경도(EPSG:4326) 변환기
_transformer = Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True)

# [행정동코드 기준 필터용] 역 반경 bbox보다 훨씬 안정적으로 표본을 확보할 수 있음.
# 5자리 = 구 단위(o_admi_cd/d_admi_cd 앞 5자리와 startswith 비교).
# 필요하면 더 세밀한 동 단위 코드(8자리)로 바꿔써도 됨.
ADMIN_CODE_PRESETS = {
    "강남역": "11680",   # 강남구
    "홍대입구": "11440",  # 마포구
}


def _get_region_bbox(region_name: str) -> dict:
    """
    region_cache.json(geo_lookup.py가 Nominatim으로 실제 조회해 캐시해둔 bbox, 더 정확)을
    우선 쓰고, 없으면 config_loader.py의 REGION_PRESETS(고정 프리셋)로 폴백.
    """
    if os.path.exists(REGION_CACHE_PATH):
        with open(REGION_CACHE_PATH, "r", encoding="utf-8") as f:
            cache = json.load(f)
        if region_name in cache:
            c = cache[region_name]
            return {"lat_min": c["lat_min"], "lat_max": c["lat_max"],
                    "lng_min": c["lng_min"], "lng_max": c["lng_max"]}
        # "강남역::m1000.0" 처럼 반경이 붙은 캐시 키 중 첫 번째로 일치하는 것 사용
        for key, c in cache.items():
            if key.startswith(region_name + "::"):
                return {"lat_min": c["lat_min"], "lat_max": c["lat_max"],
                        "lng_min": c["lng_min"], "lng_max": c["lng_max"]}

    if region_name in REGION_PRESETS:
        p = REGION_PRESETS[region_name]
        return {"lat_min": p["lat_min"], "lat_max": p["lat_max"],
                "lng_min": p["lng_min"], "lng_max": p["lng_max"]}

    raise ValueError(
        f"'{region_name}'의 bbox를 region_cache.json / REGION_PRESETS 어디서도 찾지 못했습니다. "
        f"config_gui.py에서 한 번 그 지역으로 좌표 조회를 해두면 region_cache.json에 캐시됩니다."
    )


def _convert_cells(df: pd.DataFrame) -> pd.DataFrame:
    """o_cell_x/y, d_cell_x/y(EPSG:5179)를 위경도(EPSG:4326)로 변환해 컬럼 추가"""
    df = df.copy()
    o_lon, o_lat = _transformer.transform(df["o_cell_x"].values, df["o_cell_y"].values)
    d_lon, d_lat = _transformer.transform(df["d_cell_x"].values, df["d_cell_y"].values)
    df["o_lat"], df["o_lon"] = o_lat, o_lon
    df["d_lat"], df["d_lon"] = d_lat, d_lon
    return df


def load_region_data(region_name: str, match_on: str = "origin") -> pd.DataFrame:
    """
    realdata/ 폴더의 모든 CSV를 읽어 좌표 변환 후, 지정 지역 bbox에 해당하는 행만 반환.
    match_on: "origin"(출발지 기준, 기본값) / "destination"(도착지 기준) / "both"(둘 다 해당 지역)
    """
    bbox = _get_region_bbox(region_name)
    print(f"[안내] '{region_name}' bbox: lat {bbox['lat_min']}~{bbox['lat_max']}, "
          f"lng {bbox['lng_min']}~{bbox['lng_max']}")

    files = sorted(glob.glob(os.path.join(REALDATA_DIR, "*.csv")))
    if not files:
        raise FileNotFoundError(
            f"{REALDATA_DIR}에 CSV가 없습니다. 원본 파일을 realdata/ 폴더에 먼저 넣어주세요."
        )

    frames = []
    for path in files:
        try:
            raw = pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            raw = pd.read_csv(path, encoding="cp949")

        required_cols = {"o_cell_x", "o_cell_y", "d_cell_x", "d_cell_y"}
        if not required_cols.issubset(raw.columns):
            print(f"[경고] {os.path.basename(path)}에 필요한 좌표 컬럼이 없어 건너뜁니다.")
            continue

        raw = _convert_cells(raw)

        o_in = raw["o_lat"].between(bbox["lat_min"], bbox["lat_max"]) & \
               raw["o_lon"].between(bbox["lng_min"], bbox["lng_max"])
        d_in = raw["d_lat"].between(bbox["lat_min"], bbox["lat_max"]) & \
               raw["d_lon"].between(bbox["lng_min"], bbox["lng_max"])

        if match_on == "origin":
            mask = o_in
        elif match_on == "destination":
            mask = d_in
        elif match_on == "both":
            mask = o_in & d_in
        else:
            raise ValueError("match_on은 origin / destination / both 중 하나여야 합니다.")

        filtered = raw[mask]
        print(f"[안내] {os.path.basename(path)}: {len(raw)}행 중 {len(filtered)}행이 "
              f"'{region_name}'({match_on} 기준) 조건에 해당")
        if len(filtered) > 0:
            frames.append(filtered)

    if not frames:
        print(f"[경고] '{region_name}' bbox에 해당하는 행이 하나도 없습니다.")
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    return result


def load_region_data_by_admin_code(region_name: str, match_on: str = "origin") -> pd.DataFrame:
    """
    bbox 대신 o_admi_cd/d_admi_cd(행정동코드) 앞자리로 필터링.
    역 반경 bbox보다 훨씬 넓게(구 단위) 잡히지만, 표본이 적은 SAMPLE 데이터에서는
    이 방식이 실질적으로 훨씬 많은 행을 건져줌 (강남역 bbox=0건 vs 강남구 코드=35건 등).
    """
    if region_name not in ADMIN_CODE_PRESETS:
        raise ValueError(
            f"'{region_name}'의 행정동코드 프리셋이 없습니다. "
            f"ADMIN_CODE_PRESETS에 직접 추가하거나, region_name 자리에 코드(예: '11680')를 바로 넘기세요."
        )
    admin_prefix = ADMIN_CODE_PRESETS.get(region_name, region_name)
    print(f"[안내] '{region_name}' -> 행정동코드 접두어 '{admin_prefix}' 기준으로 필터링합니다.")

    files = sorted(glob.glob(os.path.join(REALDATA_DIR, "*.csv")))
    if not files:
        raise FileNotFoundError(
            f"{REALDATA_DIR}에 CSV가 없습니다. 원본 파일을 realdata/ 폴더에 먼저 넣어주세요."
        )

    frames = []
    for path in files:
        try:
            raw = pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            raw = pd.read_csv(path, encoding="cp949")

        if not {"o_admi_cd", "d_admi_cd"}.issubset(raw.columns):
            print(f"[경고] {os.path.basename(path)}에 o_admi_cd/d_admi_cd 컬럼이 없어 건너뜁니다.")
            continue

        raw = _convert_cells(raw)  # 시각화/후속 분석용으로 위경도도 같이 붙여둠

        o_in = raw["o_admi_cd"].astype(str).str.startswith(admin_prefix)
        d_in = raw["d_admi_cd"].astype(str).str.startswith(admin_prefix)

        if match_on == "origin":
            mask = o_in
        elif match_on == "destination":
            mask = d_in
        elif match_on == "both":
            mask = o_in & d_in
        else:
            raise ValueError("match_on은 origin / destination / both 중 하나여야 합니다.")

        filtered = raw[mask]
        print(f"[안내] {os.path.basename(path)}: {len(raw)}행 중 {len(filtered)}행이 "
              f"'{region_name}'(행정동코드, {match_on} 기준) 조건에 해당")
        if len(filtered) > 0:
            frames.append(filtered)

    if not frames:
        print(f"[경고] '{region_name}' 행정동코드에 해당하는 행이 하나도 없습니다.")
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python load_real_mobility_data.py <지역명> [origin|destination|both] [bbox|admin]")
        print("예시:   python load_real_mobility_data.py 강남역")
        print("        python load_real_mobility_data.py 강남역 origin admin   # 행정동코드 기준(권장, 표본 더 많음)")
        print("        python load_real_mobility_data.py 홍대입구 destination bbox")
        sys.exit(1)

    region = sys.argv[1]
    match_on_arg = sys.argv[2] if len(sys.argv) > 2 else "origin"
    filter_mode = sys.argv[3] if len(sys.argv) > 3 else "bbox"

    if filter_mode == "admin":
        result_df = load_region_data_by_admin_code(region, match_on=match_on_arg)
    elif filter_mode == "bbox":
        result_df = load_region_data(region, match_on=match_on_arg)
    else:
        print("필터 방식은 bbox 또는 admin 중 하나여야 합니다.")
        sys.exit(1)

    if len(result_df) == 0:
        print("[안내] 결과가 0건입니다. bbox 방식이었다면 filter_mode를 'admin'으로 바꿔서 다시 시도해보세요 "
              "(역 반경 bbox보다 행정동코드 기준이 표본이 훨씬 많이 잡힙니다).")
        sys.exit(0)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"realdata_{region}_{filter_mode}.csv")
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[완료] {len(result_df)}행을 {out_path}에 저장했습니다.")