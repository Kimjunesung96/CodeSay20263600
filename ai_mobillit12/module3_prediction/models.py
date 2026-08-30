import torch
import torch.nn as nn
import xgboost as xgb
import numpy as np

# 1. PyTorch CNN-LSTM 딥러닝 모델
class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, output_dim: int = 1):
        super(CNNLSTMModel, self).__init__()
        
        # 1D Conv Layer: 지역적 패턴 extraction
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=1)
        self.relu = nn.ReLU()
        
        # LSTM Layer: 시계열 의존성 학습
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        
        # FC Layer: 최종 예측값 출력
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, features)
        # Conv1d 입력을 위해 (batch_size, features, seq_len)으로 permute
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1d(x))
        x = x.permute(0, 2, 1)
        
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # 마지막 타임스텝 결과
        return out

# 2. XGBoost 모델 생성 함수
def build_xgboost_model(n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.1):
    return xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42
    )

if __name__ == "__main__":
    # PyTorch 모델 동작 테스트
    dummy_input = torch.randn(16, 6, 8) # (batch_size=16, seq_len=6, features=8)
    model = CNNLSTMModel(input_dim=8)
    output = model(dummy_input)
    print("--- CNN-LSTM 모델 출력 테스트 ---")
    print(f"입력 텐서 크기: {dummy_input.shape}")
    print(f"출력 텐서 크기: {output.shape}")