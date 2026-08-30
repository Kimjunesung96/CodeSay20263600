# 팀 프로젝트 실행 가이드

압축 파일 받으면 venv는 없는 상태이므로, 아래 순서대로 새로 만들어서 실행하면 됩니다.

---

## [12] AI Mobility (택시 수요예측 / 동적배차)

### 1. SUMO 설치 (필수, pip으로 안 됨)

`sumolib`, `traci`는 파이썬 패키지일 뿐이고, 실제 시뮬레이터인 SUMO 프로그램은 따로 설치해야 합니다.

- Windows: https://sumo.dlr.de/docs/Downloads.php 에서 설치파일 받아서 실행
- Mac: `brew install sumo`
- Ubuntu/Linux: `sudo apt-get install sumo sumo-tools sumo-doc`

설치 후 환경변수 `SUMO_HOME`이 잡혀 있어야 합니다. (Windows는 설치 시 자동으로 잡히는 경우가 많고, Mac/Linux는 보통 `/usr/share/sumo` 또는 `/opt/homebrew/share/sumo` 정도이니 직접 확인)

```bash
# Mac/Linux 예시 (~/.zshrc 또는 ~/.bashrc에 추가)
export SUMO_HOME="/usr/share/sumo"
```

설치 확인:
```bash
sumo --version
```

### 2. 파이썬 가상환경 생성 및 패키지 설치

```bash
cd ai_mobility_project   # 압축 푼 폴더로 이동
python -m venv venv

# 가상환경 활성화
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 3. 실행 순서

```bash
# 1) SUMO 디지털트윈 환경(도로망/차량/탑승자) 생성
python module1_simulation/build_env.py

# 2) 모델 학습 (saved_models/ 폴더에 xgboost, cnn_lstm 저장됨)
python train.py

# 3) 전체 파이프라인 실행 (예측 → surge pricing → 배차 → SUMO 시각화까지)
python main.py
```

`train.py`를 먼저 돌려서 `saved_models/xgboost_demand.pkl`, `saved_models/cnn_lstm_demand.pt`가 생성된 걸 확인한 뒤에 `main.py`를 실행해야 실제 모델 추론까지 볼 수 있습니다. (모델 없으면 경고만 뜨고 원본 수요값으로 대체됨)

---

## [21] AI Energy VPP (전력 수급 최적화)

### 1. 파이썬 가상환경 생성 및 패키지 설치

```bash
cd ai_energy_project   # 압축 푼 폴더로 이동
python -m venv venv

source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 2. (선택) 공공데이터포털 API 키 설정

KPX 실제 데이터를 쓰고 싶으면 `.env` 파일을 프로젝트 루트에 만들고:

```
KPX_DATA_API_KEY=발급받은_키
```

키가 없어도 자동으로 mock 데이터로 작동하니, 일단 돌려보는 데는 필요 없습니다.

### 3. 실행 순서 (터미널 2개 필요)

**터미널 1 — 백엔드 (FastAPI) 실행**
```bash
# 반드시 프로젝트 루트에서 실행 (backend.xxx import 구조라 하위 폴더에서 실행하면 에러남)
uvicorn backend.main:app --reload
```
정상 실행되면 `http://127.0.0.1:8000/docs`에서 API 목록 확인 가능

**터미널 2 — 대시보드 (Streamlit) 실행**
```bash
streamlit run app.py
```
브라우저에서 자동으로 `http://localhost:8501` 열림

### 4. 확인 방법

대시보드 접속 후 사이드바에서 수요값 입력 → 각 탭(MILP/확률론적/RL/ESS/탄소/위기시나리오)에서 버튼 눌러 실행. 탭1 "서버 상태 확인" 버튼으로 백엔드 연결부터 확인하는 게 순서.

---

## 공통 주의사항

- venv 폴더는 압축에서 빠져 있으므로 위 안내대로 각자 새로 만들어야 함
- 두 프로젝트 모두 파이썬 3.10~3.12 권장 (SUMO/torch 호환성 이슈 방지)
- 패키지 설치 중 에러 나면 어떤 패키지에서 실패했는지 캡처해서 공유하기 (pandas/numpy 버전 충돌이 제일 흔함)
