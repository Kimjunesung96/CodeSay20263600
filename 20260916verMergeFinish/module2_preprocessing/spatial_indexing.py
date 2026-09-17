"""
[Module 2 / T1] 공간 인덱싱 — 위경도 → H3 셀 + Geohash.

왜 H3인가 (명세 Q1 "공간 인덱싱 선택 이유")
- 육각형 격자라 이웃 셀과의 거리가 모두 같다 → 인접 셀 집계·확산(배차 시 "옆 셀로 보내기")이 단순함.
- resolution별 셀 크기가 고정돼 지역이 바뀌어도 같은 기준으로 비교 가능.
- Geohash도 함께 출력한다 (명세가 "Geohash 또는 H3"라 둘 다 제공, 문자열 prefix로 상위 격자 조회가 쉬움).

resolution 9를 쓰는 근거 (data/eda/h3_resolution_summary.csv, 강남역 로그 7~10 비교)
- res 8: 셀 7개, 최다 셀에 수요 35% 집중, 학습 행 42 → 공간 정보 부족 (too coarse)
- res 10: 셀 70개, 수요 0인 칸 37%, 중앙값 1건/10분, lag-1 자기상관 0.17 → 노이즈 (too fine)
- res 9: 셀 22개(변 201m), 0인 칸 20%, 출근 핫스팟 5~6셀 식별 → 채택

대용량 효율화 (명세 Q1/Q4)
- 기존: df.apply(axis=1) — 행마다 파이썬 함수 호출 + Series 생성 → 10만 행에 수 초.
- 개선: (1) np.unique로 **중복 좌표를 한 번만 변환**하고 inverse 인덱스로 원래 행에 되돌림,
        (2) 남는 고유 좌표는 리스트 컴프리헨션으로 일괄 변환(Series 생성 오버헤드 제거),
        (3) 필요 시 chunk_size로 나눠 처리해 메모리 상한 유지.
  같은 위치에서 반복 호출되는 택시 데이터 특성상 고유 좌표 수 << 행 수라 효과가 큼.
- 검증: --bench 옵션으로 N행 랜덤 좌표 변환 시간을 재고, tests/test_module2.py가 apply 방식과 결과 동일함을 확인.
"""
import sys
import os
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import CFG

try:
    import h3
except ImportError:  # 테스트 환경 등 h3가 없을 때 명확한 에러
    h3 = None
try:
    import pygeohash as pgh
except ImportError:
    pgh = None


class SpatialIndexer:
    """GPS 위도/경도 → H3 셀 ID + Geohash 문자열."""

    def __init__(self, h3_resolution: int = None, geohash_precision: int = 7, coord_decimals: int = None):
        # 근거는 모듈 docstring 참고. config.json 값(기본 9)을 우선 사용.
        self.h3_resolution = h3_resolution if h3_resolution is not None else CFG["h3_resolution"]
        self.geohash_precision = geohash_precision   # 7자리 ≈ 153m × 153m, res 9(201m)와 비슷한 크기
        # coord_decimals: None이면 반올림 없이 '완전히 같은 좌표'만 캐시(결과가 apply 방식과 100% 동일).
        #   6을 주면 0.1m 단위로 묶어 캐시 적중률을 높이지만 셀 경계 0.1m 안의 점은 셀이 바뀔 수 있음.
        self.coord_decimals = coord_decimals

    # ---------- 단일 좌표 ----------
    def latlng_to_h3(self, lat: float, lng: float) -> str:
        if h3 is None:
            raise ImportError("h3 패키지가 필요합니다: pip install h3")
        if not (-90 <= lat <= 90 and -180 <= lng <= 180):
            raise ValueError("GPS 위도·경도 범위를 확인하세요.")
        return h3.latlng_to_cell(lat, lng, self.h3_resolution)

    def latlng_to_geohash(self, lat: float, lng: float) -> str:
        if pgh is None:
            raise ImportError("pygeohash 패키지가 필요합니다: pip install pygeohash")
        return pgh.encode(lat, lng, precision=self.geohash_precision)

    # ---------- 배열 (효율화 핵심) ----------
    def _encode_unique(self, lat: np.ndarray, lng: np.ndarray, fn) -> np.ndarray:
        """좌표 배열을 반올림 → 고유 좌표만 fn으로 변환 → 원래 순서로 되돌림."""
        if self.coord_decimals is not None:
            lat, lng = np.round(lat, self.coord_decimals), np.round(lng, self.coord_decimals)
        key = np.column_stack([lat, lng])
        uniq, inverse = np.unique(key, axis=0, return_inverse=True)
        codes = np.array([fn(a, b) for a, b in uniq], dtype=object)
        return codes[inverse.ravel()]

    def process_dataframe(self, df: pd.DataFrame, lat_col: str = 'latitude', lng_col: str = 'longitude',
                          chunk_size: int = 500_000, verbose: bool = True) -> pd.DataFrame:
        """DataFrame 전체에 h3_index, geohash 컬럼 추가. chunk_size 단위로 나눠 메모리 상한 유지."""
        if verbose:
            print(f"공간 인덱싱(H3 resolution={self.h3_resolution} / Geohash {self.geohash_precision}자리) 변환 중... ({len(df):,}행)")
        t0 = time.perf_counter()
        h3_out, gh_out = [], []
        for start in range(0, len(df), chunk_size):
            part = df.iloc[start:start + chunk_size]
            lat = part[lat_col].to_numpy(dtype=float); lng = part[lng_col].to_numpy(dtype=float)
            h3_out.append(self._encode_unique(lat, lng, self.latlng_to_h3))
            gh_out.append(self._encode_unique(lat, lng, self.latlng_to_geohash))
        df = df.copy()
        df['h3_index'] = np.concatenate(h3_out) if h3_out else []
        df['geohash'] = np.concatenate(gh_out) if gh_out else []
        if verbose:
            n_uniq = df[['h3_index']].nunique().iloc[0]
            print(f"변환 완료! {time.perf_counter() - t0:.2f}초, H3 셀 {n_uniq}개")
        return df

    def process_dataframe_naive(self, df: pd.DataFrame, lat_col='latitude', lng_col='longitude') -> pd.DataFrame:
        """개선 전 방식(행 단위 apply). 벤치마크/동일성 테스트 비교용으로만 남겨둠."""
        df = df.copy()
        df['h3_index'] = df.apply(lambda r: self.latlng_to_h3(r[lat_col], r[lng_col]), axis=1)
        df['geohash'] = df.apply(lambda r: self.latlng_to_geohash(r[lat_col], r[lng_col]), axis=1)
        return df


def benchmark(n_rows: int = 100_000, seed: int = 0) -> dict:
    """강남역 bbox 안 랜덤 좌표 n_rows개(중복 없음 = 최악 조건) 변환 시간 비교."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({'latitude': rng.uniform(CFG['lat_min'], CFG['lat_max'], n_rows),
                       'longitude': rng.uniform(CFG['lng_min'], CFG['lng_max'], n_rows)})
    idx = SpatialIndexer()
    t0 = time.perf_counter(); idx.process_dataframe(df, verbose=False); t_fast = time.perf_counter() - t0
    n_naive = min(n_rows, 20_000)  # naive는 느려서 2만 행만 재고 비례 환산
    t0 = time.perf_counter(); idx.process_dataframe_naive(df.head(n_naive)); t_naive = (time.perf_counter() - t0) * n_rows / n_naive
    res = {'rows': n_rows, 'sec_optimized': round(t_fast, 2), 'sec_naive_est': round(t_naive, 2),
           'speedup': round(t_naive / t_fast, 1) if t_fast > 0 else None}
    print(f"[벤치마크] {n_rows:,}행: 최적화 {res['sec_optimized']}초 / 기존 apply 방식 약 {res['sec_naive_est']}초 (≈{res['speedup']}배)")
    return res


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--bench":
        benchmark(int(sys.argv[2]))
    else:
        sample = pd.DataFrame({'latitude': [37.4979, 37.4985, 37.5000, 37.4979],
                               'longitude': [127.0276, 127.0280, 127.0300, 127.0276]})
        print(SpatialIndexer().process_dataframe(sample))