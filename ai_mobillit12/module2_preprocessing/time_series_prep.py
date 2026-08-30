import pandas as pd
import numpy as np

class TimeSeriesPreprocessor:
    def __init__(self, freq: str = '5min', max_lag: int = 6):
        self.freq = freq
        self.max_lag = max_lag

    def aggregate_demands(self, df: pd.DataFrame, timestamp_col: str = 'pickup_datetime', spatial_col: str = 'h3_index') -> pd.DataFrame:
        """
        공간 격자 및 5분 단위 집계 + 외부 데이터(온도, 강수량) 평균 결합
        """
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df['time_bucket'] = df[timestamp_col].dt.floor(self.freq)
        
        # 외부 데이터 컬럼 존재 여부 확인 후 집계 규칙 설정
        agg_rules = {'pickup_datetime': 'count'}
        if 'temperature' in df.columns:
            agg_rules['temperature'] = 'mean'
        if 'precipitation' in df.columns:
            agg_rules['precipitation'] = 'mean'

        # aggregate 수행
        demand_df = df.groupby(['time_bucket', spatial_col]).agg(agg_rules).reset_index()
        
        # 컬럼명 정리 ('pickup_datetime' count 결과를 'demand'로 변경)
        demand_df = demand_df.rename(columns={'pickup_datetime': 'demand'})
        return demand_df

    def create_features(self, demand_df: pd.DataFrame, spatial_col: str = 'h3_index') -> pd.DataFrame:
        print("시계열 파생변수 (Lag Feature & Calendar Features) 생성 중...")
        demand_df = demand_df.sort_values(by=[spatial_col, 'time_bucket']).reset_index(drop=True)
        
        # 1. Lag Feature 생성 (t-1 ~ t-max_lag)
        for lag in range(1, self.max_lag + 1):
            demand_df[f'lag_{lag}'] = demand_df.groupby(spatial_col)['demand'].shift(lag)
            
        # 2. 이동평균 파생변수
        demand_df['rolling_mean_3'] = demand_df.groupby(spatial_col)['demand'].shift(1).rolling(3).mean()
        demand_df['rolling_mean_6'] = demand_df.groupby(spatial_col)['demand'].shift(1).rolling(6).mean()

        # 3. 시간 및 캘린더 피처
        demand_df['hour'] = demand_df['time_bucket'].dt.hour
        demand_df['dayofweek'] = demand_df['time_bucket'].dt.dayofweek
        demand_df['is_weekend'] = demand_df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

        # 결측치 제거
        demand_df = demand_df.dropna().reset_index(drop=True)
        print("생성 완료!")
        return demand_df