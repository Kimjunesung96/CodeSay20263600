"""
module2_preprocessing/category_indexing.py

SpatialIndexer(H3/Geohash)와 나란히 쓰는 "카테고리 버전" 공간 인덱서.
H3 파이프라인은 그대로 두고, RL(Module4)이 쓰는 POI 카테고리
(school/residential/company/restaurant/subway_entrance/bus_stop) 기준으로
좌표를 분류하는 기능만 추가한 것.

왜 병행이냐면:
- H3는 지역 무관 범용 격자라 다른 프로젝트/분석에도 재사용 가능 -> 남겨둠
- 카테고리는 이 프로젝트의 POI 구조 전용이라 RL 연동에는 이게 맞지만,
  H3를 대체해버리면 나중에 "격자 기반으로도 보고 싶다" 같은 요구가 왔을 때
  다시 복구해야 함 -> 그래서 안 지우고 옵션으로 추가

의존성: sumolib (SUMO_HOME 필요), scipy(선택 - 없으면 순수 파이썬 최근접 탐색으로 대체)
"""
import json
import os

import pandas as pd

try:
    import sumolib
except ImportError:
    sumolib = None


class CategoryIndexer:
    """
    GPS 위도/경도를 POI 카테고리(zone)로 변환.
    build_env.py가 만든 net 파일 + runtime_meta.json의 zones(카테고리별 edge 목록)를 이용해서,
    각 카테고리에 속한 edge들의 shape 좌표를 샘플 포인트로 모아두고, 쿼리 좌표에 가장 가까운
    샘플 포인트의 카테고리를 반환하는 최근접 탐색 방식.
    """

    def __init__(self, net_file: str = None, meta_file: str = None, config_dir: str = None):
        # 기본 경로: module1_simulation/sumo_config/ 안의 grid.net.xml + runtime_meta.json
        # (build_env.py의 기본 생성 위치와 동일. combo/A/B 등 다른 config_dir을 쓰는 실행이면
        #  config_dir을 직접 넘겨줄 것)
        if config_dir is None:
            config_dir = os.path.join("module1_simulation", "sumo_config")
        net_file = net_file or os.path.join(config_dir, "grid.net.xml")
        meta_file = meta_file or os.path.join(config_dir, "runtime_meta.json")

        if sumolib is None:
            raise RuntimeError(
                "sumolib을 찾을 수 없습니다. SUMO_HOME이 설정돼 있는지, "
                "`pip install sumolib` 혹은 SUMO 설치가 되어 있는지 확인하세요."
            )
        if not os.path.exists(net_file):
            raise FileNotFoundError(f"net 파일이 없습니다: {net_file} (build_env.py를 먼저 실행하세요)")
        if not os.path.exists(meta_file):
            raise FileNotFoundError(f"runtime_meta.json이 없습니다: {meta_file}")

        self.net = sumolib.net.readNet(net_file)
        with open(meta_file, "r", encoding="utf-8") as f:
            self.zones = json.load(f)["zones"]  # {category: [edge_id, ...]}

        # 카테고리별 (lat, lon) 샘플 포인트 목록 만들기 (edge shape의 각 점을 다 씀)
        self._samples = []  # [(lat, lon, category), ...]
        for category, edge_ids in self.zones.items():
            for eid in edge_ids:
                try:
                    edge = self.net.getEdge(eid)
                except KeyError:
                    continue
                for x, y in edge.getShape():
                    lon, lat = self.net.convertXY2LonLat(x, y)
                    self._samples.append((lat, lon, category))

        if not self._samples:
            raise RuntimeError("zones에서 유효한 edge shape 샘플을 하나도 못 만들었습니다.")

        # scipy 있으면 KD-tree로 빠르게, 없으면 순수 파이썬 브루트포스 (edge 수가 몇백 개
        # 수준이라 브루트포스도 감당 가능)
        try:
            from scipy.spatial import cKDTree
            import numpy as np
            self._coords = [(s[0], s[1]) for s in self._samples]
            self._tree = cKDTree(self._coords)
            self._categories = [s[2] for s in self._samples]
            self._use_tree = True
        except ImportError:
            self._use_tree = False

    def latlng_to_category(self, lat: float, lon: float) -> str:
        """단일 GPS 좌표를 가장 가까운 POI 카테고리로 변환"""
        if self._use_tree:
            _, idx = self._tree.query((lat, lon))
            return self._categories[idx]

        best_cat, best_dist = None, float("inf")
        for slat, slon, cat in self._samples:
            d = (slat - lat) ** 2 + (slon - lon) ** 2  # 근거리 비교용이라 sqrt 불필요
            if d < best_dist:
                best_dist, best_cat = d, cat
        return best_cat

    def process_dataframe(self, df: pd.DataFrame, lat_col: str = "latitude",
                           lng_col: str = "longitude") -> pd.DataFrame:
        """DataFrame 전체의 GPS 컬럼을 category 컬럼으로 변환 (SpatialIndexer.process_dataframe과 동일한 쓰임)"""
        print("공간 인덱싱(POI 카테고리) 변환 중...")
        df = df.copy()
        df["category"] = df.apply(
            lambda row: self.latlng_to_category(row[lat_col], row[lng_col]), axis=1
        )
        print("변환 완료!")
        return df


if __name__ == "__main__":
    sample_data = pd.DataFrame({
        "latitude": [37.4979, 37.4985, 37.5000],
        "longitude": [127.0276, 127.0280, 127.0300],
    })
    indexer = CategoryIndexer()
    result_df = indexer.process_dataframe(sample_data)
    print("\n--- 테스트 결과 ---")
    print(result_df)