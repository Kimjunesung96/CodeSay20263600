import os
import sys

# 1. 파이썬 모듈 탐색 경로(sys.path)에 최상위 프로젝트 폴더(ai_mobility_project) 등록
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 2. sys.path 등록 후 모듈들을 임포트
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, GridSearchCV

# 하위 모듈 불러오기
from module3_prediction.models import CNNLSTMModel, build_xgboost_model
from evaluate import calculate_metrics
from module2_preprocessing.spatial_indexing import SpatialIndexer
from module2_preprocessing.time_series_prep import TimeSeriesPreprocessor
from module2_preprocessing.external_data_merge import merge_external_data
def prepare_real_sequence_dataset(max_lag=6):
    """Module 2 파이프라인을 거쳐 시계열 시퀀스(N, Seq_Len, Features) 데이터셋 생성"""
    dates = pd.date_range("2026-08-28 18:00:00", periods=1000, freq="1min")
    df = pd.DataFrame({
        'pickup_datetime': dates,
        'latitude': np.random.uniform(37.495, 37.505, 1000),
        'longitude': np.random.uniform(127.020, 127.035, 1000)
    })
    df = merge_external_data(df)
    
    indexer = SpatialIndexer(h3_resolution=8)
    df = indexer.process_dataframe(df, lat_col='latitude', lng_col='longitude')
    
    prep = TimeSeriesPreprocessor(freq='5min', max_lag=max_lag)
    agg_df = prep.aggregate_demands(df, timestamp_col='pickup_datetime', spatial_col='h3_index')
    feature_df = prep.create_features(agg_df, spatial_col='h3_index')
    
    feature_cols = [c for c in feature_df.columns if c not in ['time_bucket', 'h3_index', 'demand']]
    
    # 2D Tabular Feature Matrix (XGBoost용)
    X_mat = feature_df[feature_cols].fillna(0).values
    y_vec = feature_df['demand'].fillna(0).values
    
    return X_mat, y_vec, feature_cols, max_lag

if __name__ == "__main__":
    os.makedirs('saved_models', exist_ok=True)
    
    # 1. 데이터셋 준비 및 Train/Test (8:2) 분리
    X_mat, y_vec, feature_cols, seq_len = prepare_real_sequence_dataset(max_lag=6)
    X_train, X_test, y_train, y_test = train_test_split(X_mat, y_vec, test_size=0.2, random_state=42, shuffle=False)
    
    print(f"[데이터 분리 완료] Train: {len(X_train)}개, Test: {len(X_test)}개 | Features: {len(feature_cols)}개")

    # ------------------------------------------------------------
    # 2. XGBoost 학습, 하이퍼파라미터 튜닝(GridSearch), 평가 및 저장
    # ------------------------------------------------------------
    print("\n--- [XGBoost] Hyperparameter Tuning (GridSearch) 시작 ---")
    param_grid = {
        'n_estimators': [30, 50],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1]
    }
    
    base_xgb = build_xgboost_model()
    grid_search = GridSearchCV(base_xgb, param_grid, cv=3, scoring='neg_root_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    best_xgb = grid_search.best_estimator_
    print(f"최적 파라미터: {grid_search.best_params_}")
    
    # 성능 평가 (calculate_metrics 활용)
    xgb_preds = best_xgb.predict(X_test)
    xgb_metrics = calculate_metrics(y_test, xgb_preds)
    print(f"[XGBoost Test 평가 지표] RMSE: {xgb_metrics['RMSE']:.4f} | MAE: {xgb_metrics['MAE']:.4f} | MAPE: {xgb_metrics['MAPE (%)']:.4f}%")
    
    joblib.dump({'model': best_xgb, 'feature_cols': feature_cols}, 'saved_models/xgboost_demand.pkl')

    # ------------------------------------------------------------
    # 3. CNN-LSTM 학습, 평가 및 저장 (진짜 시퀀스 차원 재구성)
    # ------------------------------------------------------------
    print("\n--- [CNN-LSTM] 진짜 시퀀스 구조(N, Seq_len, Feat_dim) 학습 시작 ---")
    
    # Feature 벡터를 (N, Seq_len, Feat_dim) 구조로 Reshape (설계 근거 확보)
    num_sub_features = len(feature_cols) // seq_len if len(feature_cols) >= seq_len else 1
    feat_dim = max(1, len(feature_cols) // seq_len)
    # ------------------------------------------------------------
    # 3. CNN-LSTM 학습, 평가 및 저장 (피처 버림 없이 11개 피처 전체 활용)
    # ------------------------------------------------------------
    print("\n--- [CNN-LSTM] 전체 피처 활용 시퀀스 구조 학습 시작 ---")
    
    num_features = X_train.shape[1] # 11개 피처 전체 사용
    
    # 11개 전체 피처를 (N, seq_len=1, num_features=11) 형태의 시계열 스냅샷 텐서로 변환
    X_train_seq = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train_seq = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    
    X_test_seq = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_test_seq = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    train_loader = DataLoader(TensorDataset(X_train_seq, y_train_seq), batch_size=16, shuffle=True)
    
    # input_dim에 11개 피처 차원 전달
    dl_model = CNNLSTMModel(input_dim=num_features)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(dl_model.parameters(), lr=0.001)
    
    dl_model.train()
    for epoch in range(10):
        for x_b, y_b in train_loader:
            optimizer.zero_grad()
            loss = criterion(dl_model(x_b), y_b)
            loss.backward()
            optimizer.step()
            
    # CNN-LSTM 평가
    dl_model.eval()
    with torch.no_grad():
        dl_preds = dl_model(X_test_seq).numpy().flatten()
    dl_metrics = calculate_metrics(y_test, dl_preds)
    print(f"[CNN-LSTM Test 평가 지표] RMSE: {dl_metrics['RMSE']:.4f} | MAE: {dl_metrics['MAE']:.4f} | MAPE: {dl_metrics['MAPE (%)']:.4f}%")
    
    torch.save({
        'state_dict': dl_model.state_dict(),
        'input_dim': num_features,
        'feature_cols': feature_cols
    }, 'saved_models/cnn_lstm_demand.pt')
    
    print("\n[완료] 학습, 튜닝, 평가 및 모델 저장 완료!")