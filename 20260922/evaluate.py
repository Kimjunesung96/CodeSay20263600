import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

def fmt_pct(v, digits: int = 2) -> str:
    """None(계산 불가)이면 '-', 아니면 'xx.xx%'."""
    return "-" if v is None else f"{v:.{digits}f}%"


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    RMSE, MAE, MAPE, WAPE 성능 평가 지표 계산.

    수요 데이터는 0인 칸이 많아서 MAPE를 그대로 계산하면 0으로 나눠 수백만 %가 나온다.
    - MAPE: 실제값이 0인 칸은 제외하고 계산 (전부 0이면 None)
    - WAPE: 오차 절대값 합 / 실제값 절대값 합 (0이 많아도 안정적, 합이 0이면 None)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    err = np.abs(y_true - y_pred)
    nonzero = np.abs(y_true) > 0
    mape = float(np.mean(err[nonzero] / np.abs(y_true[nonzero])) * 100) if nonzero.any() else None
    denom = float(np.sum(np.abs(y_true)))
    wape = float(np.sum(err) / denom * 100) if denom > 0 else None

    return {
        "RMSE": round(float(rmse), 4),
        "MAE": round(float(mae), 4),
        "MAPE (%)": round(mape, 4) if mape is not None else None,
        "WAPE (%)": round(wape, 4) if wape is not None else None,
    }

if __name__ == "__main__":
    y_real = np.array([10, 15, 20, 25, 30])
    y_hat = np.array([11, 14, 22, 24, 28])
    
    metrics = calculate_metrics(y_real, y_hat)
    print("--- 평가 지표 테스트 ---")
    print(metrics)