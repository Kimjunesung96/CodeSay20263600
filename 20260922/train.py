import os
import sys
import glob

# 1. 파이썬 모듈 탐색 경로(sys.path)에 최상위 프로젝트 폴더(ai_mobility_project) 등록
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR))  # train.py는 프로젝트 루트에 위치
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 2. sys.path 등록 후 모듈들을 임포트
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, GridSearchCV

from config_loader import CFG

# 하위 모듈 불러오기
from module3_prediction.models import CNNLSTMModel, build_xgboost_model
from evaluate import calculate_metrics, fmt_pct
# [병합 2026-09-16] Module 2 단일 진입점(pipeline.py) 기반으로 통일.
# 예전에 여기서 직접 부르던 SpatialIndexer/TimeSeriesPreprocessor 호출은
# pipeline.build_feature_table() 내부로 이미 들어가 있어 더 이상 직접 쓰지 않음.
from module2_preprocessing.pipeline import build_feature_table
from module2_preprocessing.sim_log_recorder import latest_log_path

# measure_wait_time.py의 run_and_measure()가 매 스텝 DemandLogRecorder.step()으로 남기는
# 실제 수요 로그. data/sim_logs/demand_log_*.csv 로 저장됨 (단독 실행 / parallel_dispatch_worker.py /
# multi_factor_compare.py 병렬 워커 전부 여기에 로그를 남김).
DEMAND_LOG_DIR = os.path.join(PROJECT_ROOT, "data", "sim_logs")
DEMAND_LOG_GLOB = os.path.join(DEMAND_LOG_DIR, "demand_log_*.csv")

# [2026-09-21 중복 집계 수정] 팀원(Fetex-ldk) 버전에서 쓰던 축적 로그.
# measure_wait_time.py가 실행 1회마다 이 파일과 data/sim_logs/demand_log_*.csv 양쪽에
# "같은 승객 데이터"를 동시에 남기고 있어서(전자는 탑승 성공자만+랜덤 과거 90일 날짜,
# 후자는 등장자 전체+config의 실제 sim_date), 두 소스를 같이 읽으면 매 실행분이 두 번
# 집계되고 있었다(pickup_datetime이 서로 달라서 drop_duplicates로도 못 잡힘).
# data/sim_logs가 이제 항상 채워지므로, measure_wait_time.py 주석에 적힌 원래 의도대로
# 이 레거시 소스는 더 이상 읽지 않는다. (예전에 data/raw만 있던 시절의 기록을 쓰려면
# prepare_real_sequence_dataset(log_path=LEGACY_DEMAND_LOG_GLOB)로 명시적으로 지정)
LEGACY_DEMAND_LOG_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
LEGACY_DEMAND_LOG_GLOB = os.path.join(LEGACY_DEMAND_LOG_DIR, "simulation_demand_log*.csv")

# [병합 전 feat3 버전에서 이식] 학습 타겟 하나만 쓸 때 피처 테이블 행이 이 건수보다 적으면
# lag/rolling 피처 생성 시 dropna로 대부분 날아갈 수 있어 경고만 출력.
MIN_RECOMMENDED_ROWS = 200

# 학습 타겟: Module 2 피처 테이블의 y_h1(t+1, 5분 뒤) ~ y_h6(t+6, 30분 뒤).
# 아래 단일 출력 모델은 TARGET 한 개만 사용. 다중 시점(6개) 동시 예측은 y_h1..y_h6 전체를
# y로 넘기고 CNNLSTMModel(output_dim=6) / MultiOutputRegressor(XGB)로 바꾸면 됨 (Module 3 담당).
TARGET = "y_h1"


def prepare_real_sequence_dataset(max_lag=None, log_path=None):
    """
    Module 2 파이프라인(module2_preprocessing.pipeline)으로 학습용 피처 테이블 생성.
    - data/sim_logs/demand_log_*.csv (새 로그) + data/raw/simulation_demand_log*.csv (팀원
      쪽에서 축적된 예전 로그) 를 모두 이어 붙여 사용 (log_path를 주면 그 파일/패턴만)
    - [feat3에서 이식] 로그 파일이 몇 개 잡혔는지, 총 몇 건인지 안내 출력 + 건수가 너무 적으면 경고
    - 로그가 하나도 없으면 에러 (가짜 랜덤 데이터 폴백 없음 — 평가 시 혼동 방지)
    반환: X_mat, y_vec, feature_cols, max_lag  (기존 XGBoost/CNN-LSTM 학습 루프와 호환되는 형태)
    """
    max_lag = max_lag if max_lag is not None else CFG["max_lag"]
    # [중복 집계 수정] data/sim_logs만 사용 (위 LEGACY_DEMAND_LOG_GLOB 주석 참고).
    logs = [log_path] if log_path else [DEMAND_LOG_GLOB]

    if not log_path:
        found_paths = sorted(glob.glob(DEMAND_LOG_GLOB))
        if not found_paths:
            raise FileNotFoundError(
                f"{DEMAND_LOG_DIR}에 수요 로그가 없습니다. "
                f"먼저 `python measure_wait_time.py`(또는 A/B·멀티팩터 비교)를 한 번 이상 돌려 "
                f"학습용 수요 로그를 쌓아주세요."
            )
        file_list = ", ".join(os.path.basename(p) for p in found_paths)
        print(f"[안내] 로그 파일 {len(found_paths)}개({file_list})를 학습 데이터로 사용합니다.")

    feature_df = build_feature_table(logs, max_lag=max_lag)

    if len(feature_df) < MIN_RECOMMENDED_ROWS:
        print(f"[경고] 피처 테이블이 {len(feature_df)}행뿐입니다 (권장 {MIN_RECOMMENDED_ROWS}행 이상). "
              f"lag/rolling 피처(max_lag={max_lag}) 생성 시 dropna로 대부분 날아갈 수 있으니, "
              f"measure_wait_time.py나 A/B·멀티팩터 비교를 몇 차례 더 돌려서 로그를 쌓는 것을 권장합니다.")
    if len(feature_df) == 0:
        # 0행으로 그냥 넘어가면 train_test_split에서 알아보기 힘든 sklearn 스택트레이스로
        # 죽어버림. train_category_demand.py처럼 여기서 바로 알아볼 수 있는 에러로 끝냄.
        raise ValueError(
            "[에러] lag/rolling 생성 후 데이터가 하나도 안 남았습니다. "
            "measure_wait_time.py나 A/B·멀티팩터 비교를 더 돌려서 로그를 쌓아주세요 "
            f"(현재 max_lag={max_lag}, 시뮬레이션 구간이 짧으면 lag/rolling 계산에 필요한 "
            "시간칸 수를 못 채워서 전부 dropna로 날아갈 수 있습니다)."
        )

    feature_cols = feature_df.attrs["feature_cols"]
    target_cols = feature_df.attrs["target_cols"]
    if TARGET not in target_cols:
        raise ValueError(f"TARGET={TARGET!r}이 피처 테이블의 target_cols({target_cols})에 없습니다.")

    # 2D Tabular Feature Matrix (XGBoost용)
    X_mat = feature_df[feature_cols].fillna(0).values
    y_vec = feature_df[TARGET].fillna(0).values

    return X_mat, y_vec, feature_cols, max_lag


if __name__ == "__main__":
    os.makedirs('saved_models', exist_ok=True)

    # 1. 데이터셋 준비 및 Train/Test 분리 (비율은 config.json에서 조절)
    X_mat, y_vec, feature_cols, seq_len = prepare_real_sequence_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X_mat, y_vec, test_size=CFG["test_size"], random_state=42, shuffle=False
    )

    print(f"[데이터 분리 완료] Train: {len(X_train)}개, Test: {len(X_test)}개 | Features: {len(feature_cols)}개 | test_size={CFG['test_size']}")

    # ------------------------------------------------------------
    # 2. XGBoost 학습, 하이퍼파라미터 튜닝(GridSearch), 평가 및 저장
    # ------------------------------------------------------------
    print("\n--- [XGBoost] Hyperparameter Tuning (GridSearch) 시작 ---")

    # config.json 값(xgb_n_estimators 등)을 중심으로 좌우 넓게 탐색
    base_n = CFG["xgb_n_estimators"]
    base_depth = CFG["xgb_max_depth"]
    base_lr = CFG["xgb_learning_rate"]

    param_grid = {
        'n_estimators': sorted(set([max(10, base_n - 50), base_n, base_n + 100])),
        'max_depth': sorted(set([max(1, base_depth - 2), base_depth, base_depth + 2])),
        'learning_rate': sorted(set([round(max(0.01, base_lr * 0.3), 3), base_lr]))
    }
    print(f"탐색 범위: {param_grid}")

    # 데이터가 적을 때 GridSearchCV의 cv=3 fold가 표본 부족으로 실패하지 않도록 안전장치
    n_splits = min(3, len(X_train)) if len(X_train) > 1 else 1
    if n_splits < 3:
        print(f"[경고] 학습 표본이 적어({len(X_train)}개) cross-validation fold를 {n_splits}로 줄입니다.")

    base_xgb = build_xgboost_model()
    if n_splits >= 2:
        grid_search = GridSearchCV(base_xgb, param_grid, cv=n_splits, scoring='neg_root_mean_squared_error')
        grid_search.fit(X_train, y_train)
        best_xgb = grid_search.best_estimator_
        print(f"최적 파라미터: {grid_search.best_params_}")
    else:
        print("[경고] 표본이 너무 적어 GridSearchCV를 생략하고 기본 파라미터로 학습합니다.")
        best_xgb = base_xgb
        best_xgb.fit(X_train, y_train)

    # 성능 평가 (calculate_metrics 활용)
    xgb_preds = best_xgb.predict(X_test)
    xgb_metrics = calculate_metrics(y_test, xgb_preds)
    print(f"[XGBoost Test 평가 지표] RMSE: {xgb_metrics['RMSE']:.4f} | MAE: {xgb_metrics['MAE']:.4f} | MAPE: {fmt_pct(xgb_metrics['MAPE (%)'])} | WAPE: {fmt_pct(xgb_metrics['WAPE (%)'])}")

    joblib.dump({'model': best_xgb, 'feature_cols': feature_cols}, 'saved_models/xgboost_demand.pkl')

    # ------------------------------------------------------------
    # 3. CNN-LSTM 학습, 평가 및 저장 (전체 피처 활용 시퀀스 구조)
    # ------------------------------------------------------------
    print("\n--- [CNN-LSTM] 전체 피처 활용 시퀀스 구조 학습 시작 ---")

    num_features = X_train.shape[1]  # 피처 전체 사용

    X_train_seq = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train_seq = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    X_test_seq = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_test_seq = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(
        TensorDataset(X_train_seq, y_train_seq),
        batch_size=CFG["cnn_batch_size"],
        shuffle=True
    )

    # hidden_dim, num_layers, kernel_size는 CNNLSTMModel 내부에서 config.json 값을 자동으로 사용
    dl_model = CNNLSTMModel(input_dim=num_features)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(dl_model.parameters(), lr=CFG["cnn_lr"])

    print(f"CNN-LSTM 설정: hidden_dim={CFG['cnn_hidden_dim']}, num_layers={CFG['cnn_num_layers']}, "
          f"kernel_size={CFG['cnn_kernel_size']}, epochs={CFG['cnn_epochs']}, "
          f"batch_size={CFG['cnn_batch_size']}, lr={CFG['cnn_lr']}")

    dl_model.train()
    for epoch in range(CFG["cnn_epochs"]):
        epoch_loss = 0.0
        for x_b, y_b in train_loader:
            optimizer.zero_grad()
            loss = criterion(dl_model(x_b), y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % max(1, CFG["cnn_epochs"] // 10) == 0 or epoch == 0:
            print(f"  epoch {epoch + 1}/{CFG['cnn_epochs']} - loss: {epoch_loss / len(train_loader):.4f}")

    # CNN-LSTM 평가
    dl_model.eval()
    with torch.no_grad():
        dl_preds = dl_model(X_test_seq).numpy().flatten()
    dl_metrics = calculate_metrics(y_test, dl_preds)
    print(f"[CNN-LSTM Test 평가 지표] RMSE: {dl_metrics['RMSE']:.4f} | MAE: {dl_metrics['MAE']:.4f} | MAPE: {fmt_pct(dl_metrics['MAPE (%)'])} | WAPE: {fmt_pct(dl_metrics['WAPE (%)'])}")

    torch.save({
        'state_dict': dl_model.state_dict(),
        'input_dim': num_features,
        'feature_cols': feature_cols
    }, 'saved_models/cnn_lstm_demand.pt')

    print("\n[완료] 학습, 튜닝, 평가 및 모델 저장 완료!")