import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    RMSE, MAE, MAPE 성능 평가 지표 계산 (MAPE 0 나누기 방지 처리)
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # 0으로 나누는 상황 방지를 위한 처리 (Small epsilon)
    epsilon = 1e-5
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100

    return {
        "RMSE": round(float(rmse), 4),
        "MAE": round(float(mae), 4),
        "MAPE (%)": round(float(mape), 4)
    }

if __name__ == "__main__":
    y_real = np.array([10, 15, 20, 25, 30])
    y_hat = np.array([11, 14, 22, 24, 28])
    
    metrics = calculate_metrics(y_real, y_hat)
    print("--- 평가 지표 테스트 ---")
    print(metrics)