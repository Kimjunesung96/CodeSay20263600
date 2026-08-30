import h3
import pygeohash as pgh
import pandas as pd

class SpatialIndexer:
    """
    GPS 위도/경도 데이터를 H3 Grid 및 Geohash 공간 인덱스로 변환하는 클래스
    """
    def __init__(self, h3_resolution: int = 8, geohash_precision: int = 7):
        # h3_resolution 8: 변 길이가 약 460m인 육각형 격자 (도심 지역 분석에 적합)
        self.h3_resolution = h3_resolution
        self.geohash_precision = geohash_precision

    def latlng_to_h3(self, lat: float, lng: float) -> str:
        """단일 GPS 좌표를 H3 인덱스로 변환"""
        return h3.latlng_to_cell(lat, lng, self.h3_resolution)

    def latlng_to_geohash(self, lat: float, lng: float) -> str:
        """단일 GPS 좌표를 Geohash로 변환"""
        return pgh.encode(lat, lng, precision=self.geohash_precision)

    def process_dataframe(self, df: pd.DataFrame, lat_col: str = 'latitude', lng_col: str = 'longitude') -> pd.DataFrame:
        """DataFrame 전체의 GPS 컬럼을 H3 및 Geohash 컬럼으로 변환"""
        print("공간 인덱싱(H3 / Geohash) 변환 중...")
        
        df['h3_index'] = df.apply(
            lambda row: self.latlng_to_h3(row[lat_col], row[lng_col]), axis=1
        )
        df['geohash'] = df.apply(
            lambda row: self.latlng_to_geohash(row[lat_col], row[lng_col]), axis=1
        )
        
        print("변환 완료!")
        return df

if __name__ == "__main__":
    # 테스트용 샘플 데이터 (강남역 좌표)
    sample_data = pd.DataFrame({
        'latitude': [37.4979, 37.4985, 37.5000],
        'longitude': [127.0276, 127.0280, 127.0300]
    })
    
    indexer = SpatialIndexer(h3_resolution=8, geohash_precision=7)
    result_df = indexer.process_dataframe(sample_data)
    print("\n--- 테스트 결과 ---")
    print(result_df)