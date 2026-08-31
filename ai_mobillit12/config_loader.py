"""
config.json 로드 공통 헬퍼
프로젝트 루트에 두고, 각 모듈에서 다음처럼 씁니다:

    from config_loader import CFG
    resolution = CFG["h3_resolution"]
"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT, "config.json")

DEFAULT_CONFIG = {
    # Module 1 - 맵/시뮬레이션 환경
    "grid_x": 3,
    "grid_y": 3,
    "grid_length": 200,
    "num_normal_cars": 20,
    "num_taxis": 3,
    "num_auto_cars": 1,
    "num_obstacles": 2,
    "num_passengers": 5,
    # Module 2
    "h3_resolution": 8,
    "freq": "5min",
    "max_lag": 6,
    "rolling_short": 3,
    "rolling_long": 6,
    # Module 3 - XGBoost
    "xgb_n_estimators": 100,
    "xgb_max_depth": 6,
    "xgb_learning_rate": 0.1,
    "test_size": 0.2,
    # Module 3 - CNN-LSTM
    "cnn_hidden_dim": 64,
    "cnn_num_layers": 2,
    "cnn_kernel_size": 3,
    "cnn_epochs": 30,
    "cnn_batch_size": 16,
    "cnn_lr": 0.001,
    # Module 4
    "base_fare": 4800,
    "max_multiplier": 3.0,
    "surge_coefficient": 0.4,
}


def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        merged = {**DEFAULT_CONFIG, **user_cfg}
        print(f"[안내] config.json에서 설정값을 불러왔습니다.")
        return merged
    print("[안내] config.json이 없어 기본값을 사용합니다.")
    return DEFAULT_CONFIG.copy()


CFG = load_config()