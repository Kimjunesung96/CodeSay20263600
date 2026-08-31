import sys
import os

# 현재 프로젝트 루트 경로를 최우선 탐색 경로로 자동 등록
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import joblib
import torch

from module2_preprocessing.spatial_indexing import SpatialIndexer
from module2_preprocessing.time_series_prep import TimeSeriesPreprocessor
from module2_preprocessing.external_data_merge import merge_external_data
from module4_dispatch.surge_pricing import SurgePricingEngine
from module4_dispatch.dynamic_matching import DynamicDispatcher
from module1_simulation.run_simulation import run_sumo_gui

# [수정] module3_forecasting -> module3_prediction 폴더명으로 변경
from module3_prediction.models import CNNLSTMModel
def run_pipeline():
    print("=== [12] AI Mobility 통합 파이프라인 시작 ===\n")
    
    # 1. 오픈 데이터 또는 가상 데이터 로드
    data_path = "data/raw/nyc_yellow_taxi_sample.parquet"
    if os.path.exists(data_path):
        print(f"[{data_path}] 실제 오픈 데이터를 로드합니다...")
        df = pd.read_parquet(data_path).sample(1000)
        df = df.rename(columns={'tpep_pickup_datetime': 'pickup_datetime'})
    else:
        print("[안내] 실제 데이터 파일이 없어 가상의 택시 호출 데이터를 생성합니다.")
        dates = pd.date_range("2026-08-28 18:00:00", periods=500, freq="1min")
        df = pd.DataFrame({
            'pickup_datetime': dates,
            'latitude': np.random.uniform(37.495, 37.505, 500),
            'longitude': np.random.uniform(127.020, 127.035, 500)
        })

    # 2. [Module 2] 시공간 전처리 및 외부데이터 결합
    print("\n▶ [Module 2] 시공간 전처리 및 외부데이터 결합 수행 중...")
    df = merge_external_data(df)
    
    indexer = SpatialIndexer()
    df = indexer.process_dataframe(df, lat_col='latitude', lng_col='longitude')
    
    prep = TimeSeriesPreprocessor()
    agg_df = prep.aggregate_demands(df, timestamp_col='pickup_datetime', spatial_col='h3_index')
    feature_df = prep.create_features(agg_df, spatial_col='h3_index')

    # 3. [Module 3] XGBoost 및 CNN-LSTM 추론
    print("\n▶ [Module 3] 저장된 AI 모델 로드 및 추론 수행 중...")
    xgb_path = os.path.join(PROJECT_ROOT, 'saved_models/xgboost_demand.pkl')
    dl_path = os.path.join(PROJECT_ROOT, 'saved_models/cnn_lstm_demand.pt')
    
    if os.path.exists(xgb_path) and os.path.exists(dl_path):
        # XGBoost 추론
        xgb_data = joblib.load(xgb_path)
        feature_cols = xgb_data['feature_cols']
        X_mat = feature_df[feature_cols].fillna(0).values
        xgb_preds = xgb_data['model'].predict(X_mat)
        feature_df['pred_xgboost'] = np.maximum(0, xgb_preds).astype(int)
        
        # CNN-LSTM 추론
# CNN-LSTM 추론 (11개 전체 피처 입력)
        dl_data = torch.load(dl_path)
        dl_model = CNNLSTMModel(input_dim=dl_data['input_dim'])
        dl_model.load_state_dict(dl_data['state_dict'])
        dl_model.eval()
        
        X_seq = torch.tensor(X_mat, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            dl_preds = dl_model(X_seq).numpy().flatten()
        feature_df['pred_cnn_lstm'] = np.maximum(0, dl_preds).astype(int)
        
        feature_df['predicted_demand'] = feature_df['pred_xgboost']
        print("  -> [성공] XGBoost & CNN-LSTM 추론 완료!")
        print(feature_df[['h3_index', 'demand', 'pred_xgboost', 'pred_cnn_lstm']].head())
    else:
        print("  -> [경고] 저장된 모델이 없습니다. python train.py를 먼저 실행하세요.")
        feature_df['predicted_demand'] = feature_df['demand']

    # 4. [Module 4] Surge Pricing & 최적 배차 전/후 비교 검증
    print("\n▶ [Module 4] Surge Pricing 산출 및 동적 배차 전/후 효과 비교 검증 중...")
    feature_df['available_taxis'] = np.random.randint(1, 5, size=len(feature_df))
    
    pricing_engine = SurgePricingEngine()
    priced_df = pricing_engine.apply_surge_pricing(feature_df)
    
    print("\n[Surge Pricing 산출 결과]")
    print(priced_df[['time_bucket', 'h3_index', 'demand', 'predicted_demand', 'available_taxis', 'surge_multiplier', 'final_fare']].head())

    # 배차 전/후 검증 (5명 탑승자 및 3대 택시 매칭 시뮬레이션)
    dispatcher = DynamicDispatcher()
    mock_taxis = [('Taxi_0', 0, 0), ('Taxi_1', 2, 2), ('Taxi_2', 5, 5)]
    mock_passengers = [('Pax_0', 1, 1), ('Pax_1', 4, 4), ('Pax_2', 6, 1)]
    
    # 배차 적용 후 (Hungarian Algorithm)
    opt_matches = dispatcher.match_vehicles(mock_taxis, mock_passengers)
    opt_dist = opt_matches['wait_distance'].sum()
    
    # 배차 적용 전 (Random Matching simulation)
    rnd_pax_idx = np.random.permutation(len(mock_passengers))
    rnd_dist = sum([np.sqrt((mock_taxis[i][1] - mock_passengers[rnd_pax_idx[i]][1])**2 + 
                            (mock_taxis[i][2] - mock_passengers[rnd_pax_idx[i]][2])**2) for i in range(len(mock_taxis))])
    
    print("\n============================================================")
    print("      [동적 배차 최적화(Module 4) 전/후 비교 검증 결과]      ")
    print("============================================================")
    print(f" - 무작위(Random) 임의 배차 시 총 대기거리: {rnd_dist:.2f} km")
    print(f" - 헝가리안(Hungarian) 최적 배차 시 총 대기거리: {opt_dist:.2f} km")
    print(f" - 이동거리/대기시간 절감률: {((rnd_dist - opt_dist) / rnd_dist * 100):.1f}% 개선")
    print("============================================================\n")

    # 5. [Module 1] Digital Twin 시각화
    print("▶ [Module 1] SUMO 디지털 트윈 시뮬레이션 가동")
    run_sumo_gui()

if __name__ == "__main__":
    run_pipeline()