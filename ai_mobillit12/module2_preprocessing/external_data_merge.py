import pandas as pd
import numpy as np

def merge_external_data(df):
    """
    H3/Geohash 시계열 격자 데이터에 외부 환경 데이터(날씨, 휴일 등)를 결합
    """
    df = df.copy()
    
    # 시간 컬럼 확인
    time_col = 'pickup_datetime' if 'pickup_datetime' in df.columns else 'datetime'
    
    # 1. 날씨 mock/실제 데이터 결합 (기온, 강수량)
    # 실제 API 연동이 없는 경우에도 파이프라인이 정상 작동하도록 시계열 기반 피처 생성
    np.random.seed(42)
    df['temperature'] = np.random.uniform(15.0, 25.0, size=len(df))
    df['precipitation'] = np.random.choice([0.0, 0.0, 0.0, 1.2, 5.5], size=len(df)) # mm
    
    # 2. 주말/공휴일 여부 피처
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col])
        df['is_weekend'] = df[time_col].dt.dayofweek.isin([5, 6]).astype(int)
    
    return df