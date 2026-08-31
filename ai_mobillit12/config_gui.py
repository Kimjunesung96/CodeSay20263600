"""
AI Mobility 프로젝트 통합 설정창 (탭 버전)
- Module 1(맵/시뮬레이션) / 2(전처리) / 3(예측모델) / 4(배차·가격) 탭으로 분리
- 프리셋 저장/불러오기 (presets/*.json)
- "플레이" 버튼 → config.json 저장 후 main.py 실행 (python main.py 와 동일)
- "학습만 실행" 버튼 → train.py만 실행 (모델 구조 값 바꿨을 때 재학습용)

사용법:
    python config_gui.py
(프로젝트 루트, main.py 와 같은 폴더에 두고 실행하세요)
"""

import json
import os
import subprocess
import sys
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

ROOT = os.path.dirname(os.path.abspath(__file__))
PRESET_DIR = os.path.join(ROOT, "presets")
os.makedirs(PRESET_DIR, exist_ok=True)
CONFIG_PATH = os.path.join(ROOT, "config.json")

# (키, 라벨, 최소, 최대, 기본값, 소수점자리, 정수여부)
TABS = {
    "Module 1 — 맵/시뮬레이션": [
        ("grid_x", "그리드 가로 블록 수 (x)", 2, 8, 3, 0, True),
        ("grid_y", "그리드 세로 블록 수 (y)", 2, 8, 3, 0, True),
        ("grid_length", "블록 한 변 길이 (m)", 50, 500, 200, 0, True),
        ("num_normal_cars", "일반 차량 대수", 0, 50, 20, 0, True),
        ("num_taxis", "택시 대수", 0, 20, 3, 0, True),
        ("num_auto_cars", "자율주행차 대수", 0, 10, 1, 0, True),
        ("num_obstacles", "장애물 개수", 0, 10, 2, 0, True),
        ("num_passengers", "탑승자 수", 0, 30, 5, 0, True),
    ],
    "Module 2 — 전처리": [
        ("h3_resolution", "H3 resolution", 5, 10, 8, 0, True),
        ("max_lag", "Lag 개수 (max_lag)", 1, 12, 6, 0, True),
        ("rolling_short", "이동평균 단기 윈도우", 1, 12, 3, 0, True),
        ("rolling_long", "이동평균 장기 윈도우", 1, 12, 6, 0, True),
    ],
    "Module 3 — 예측모델": [
        ("xgb_n_estimators", "[XGB] n_estimators", 10, 500, 100, 0, True),
        ("xgb_max_depth", "[XGB] max_depth", 1, 15, 6, 0, True),
        ("xgb_learning_rate", "[XGB] learning_rate", 0.01, 0.5, 0.1, 3, False),
        ("test_size", "Test size 비율", 0.05, 0.4, 0.2, 2, False),
        ("cnn_hidden_dim", "[CNN-LSTM] hidden_dim", 8, 256, 64, 0, True),
        ("cnn_num_layers", "[CNN-LSTM] num_layers", 1, 4, 2, 0, True),
        ("cnn_kernel_size", "[CNN-LSTM] kernel_size", 1, 7, 3, 0, True),
        ("cnn_epochs", "[CNN-LSTM] epochs", 5, 200, 30, 0, True),
        ("cnn_batch_size", "[CNN-LSTM] batch_size", 4, 128, 16, 0, True),
        ("cnn_lr", "[CNN-LSTM] learning_rate", 0.0001, 0.05, 0.001, 4, False),
    ],
    "Module 4 — 배차/가격": [
        ("base_fare", "기본요금 (원)", 3000, 8000, 4800, 0, True),
        ("max_multiplier", "최대 할증배수", 1.5, 5.0, 3.0, 1, False),
        ("surge_coefficient", "할증 증가계수 (0.4)", 0.1, 1.0, 0.4, 2, False),
    ],
}

FREQ_OPTIONS = ["1min", "5min", "10min", "15min", "30min"]


class ConfigGUI:
    def __init__(self, root):
        self.root = root
        root.title("AI Mobility 통합 하이퍼파라미터 설정창")
        root.geometry("560x680")

        self.vars = {}

        header = ttk.Frame(root, padding=(15, 12, 15, 0))
        header.pack(fill="x")
        ttk.Label(header, text="Module 1 / 2 / 3 / 4 통합 튜닝", font=("맑은 고딕", 13, "bold")).pack()

        # ---------- 탭 ----------
        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True, padx=15, pady=10)

        for tab_name, params in TABS.items():
            tab_frame = ttk.Frame(notebook, padding=15)
            notebook.add(tab_frame, text=tab_name.split("—")[0].strip())

            if tab_name == "Module 2 — 전처리":
                row = ttk.Frame(tab_frame)
                row.pack(fill="x", pady=3)
                ttk.Label(row, text="시계열 집계 단위 (freq)", width=24).pack(side="left")
                self.freq_var = tk.StringVar(value="5min")
                ttk.Combobox(row, textvariable=self.freq_var, values=FREQ_OPTIONS, state="readonly", width=10).pack(side="right")

            for pkey, label, lo, hi, default, decimals, is_int in params:
                row = ttk.Frame(tab_frame)
                row.pack(fill="x", pady=3)

                ttk.Label(row, text=label, width=22).pack(side="left")

                var = tk.DoubleVar(value=default)
                self.vars[pkey] = (var, decimals, is_int)

                entry = ttk.Entry(row, width=8, textvariable=var)
                entry.pack(side="right", padx=(6, 0))

                slider = ttk.Scale(row, from_=lo, to=hi, orient="horizontal", variable=var, length=220)
                slider.pack(side="right", fill="x", expand=True)

                # 정수 항목은 슬라이더를 움직이는 동안에도 즉시 정수로 버림 처리
                # (부동소수점 오차로 12.999999 같은 값이 그대로 남는 것을 방지)
                if is_int:
                    def make_snap(v=var):
                        def snap(*_):
                            try:
                                current = v.get()
                            except tk.TclError:
                                return
                            snapped = int(current)
                            if abs(current - snapped) > 1e-9:
                                v.set(snapped)
                        return snap
                    var.trace_add("write", make_snap())

        # ---------- 프리셋 ----------
        preset_frame = ttk.LabelFrame(root, text="프리셋", padding=10)
        preset_frame.pack(fill="x", padx=15, pady=(0, 8))

        self.preset_combo = ttk.Combobox(preset_frame, state="readonly")
        self.preset_combo.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.refresh_presets()

        ttk.Button(preset_frame, text="불러오기", command=self.load_preset).pack(side="left", padx=3)
        ttk.Button(preset_frame, text="저장", command=self.save_preset).pack(side="left", padx=3)
        ttk.Button(preset_frame, text="삭제", command=self.delete_preset).pack(side="left", padx=3)

        # ---------- 실행 버튼 ----------
        run_frame = ttk.Frame(root, padding=(15, 0, 15, 12))
        run_frame.pack(fill="x")

        play_btn = tk.Button(
            run_frame, text="▶  플레이  (python main.py)",
            command=self.run_main, bg="#2e7d32", fg="white",
            font=("맑은 고딕", 11, "bold"), height=2
        )
        play_btn.pack(fill="x", pady=(0, 6))

        sub_frame = ttk.Frame(run_frame)
        sub_frame.pack(fill="x")
        ttk.Button(sub_frame, text="학습만 실행 (train.py)", command=self.run_training).pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(sub_frame, text="값만 저장", command=self.save_config_only).pack(side="left", fill="x", expand=True, padx=(4, 0))

        self.status = ttk.Label(root, text="", foreground="gray")
        self.status.pack(padx=15, pady=(0, 8), anchor="w")

    # ---------- 값 읽기/쓰기 ----------
    def get_current_config(self):
        config = {"freq": self.freq_var.get()}
        for key, (var, decimals, is_int) in self.vars.items():
            try:
                val = float(var.get())
            except (tk.TclError, ValueError):
                val = 0.0
            # 정수 항목은 소수점 불안정성 방지를 위해 무조건 버림(int) 처리
            config[key] = int(val) if is_int else round(val, decimals)
        return config

    def set_config(self, config):
        if "freq" in config:
            self.freq_var.set(config["freq"])
        for key, (var, decimals, is_int) in self.vars.items():
            if key in config:
                val = config[key]
                # 정수 항목은 불러올 때도 버림 처리해서 슬라이더/입력창에 깔끔하게 표시
                var.set(int(val) if is_int else round(float(val), decimals))

    # ---------- 프리셋 ----------
    def refresh_presets(self):
        files = [f[:-5] for f in os.listdir(PRESET_DIR) if f.endswith(".json")]
        self.preset_combo["values"] = files
        if files:
            self.preset_combo.current(0)

    def save_preset(self):
        name = simpledialog.askstring("프리셋 저장", "프리셋 이름을 입력하세요:")
        if not name:
            return
        path = os.path.join(PRESET_DIR, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.get_current_config(), f, ensure_ascii=False, indent=2)
        self.refresh_presets()
        self.status.config(text=f"프리셋 '{name}' 저장 완료")

    def load_preset(self):
        name = self.preset_combo.get()
        if not name:
            messagebox.showinfo("알림", "불러올 프리셋을 선택하세요.")
            return
        path = os.path.join(PRESET_DIR, f"{name}.json")
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
        self.set_config(config)
        self.status.config(text=f"프리셋 '{name}' 불러오기 완료")

    def delete_preset(self):
        name = self.preset_combo.get()
        if not name:
            return
        if messagebox.askyesno("확인", f"프리셋 '{name}'을(를) 삭제할까요?"):
            os.remove(os.path.join(PRESET_DIR, f"{name}.json"))
            self.refresh_presets()
            self.status.config(text=f"프리셋 '{name}' 삭제 완료")

    # ---------- 실행 ----------
    def save_config_only(self):
        config = self.get_current_config()
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        self.status.config(text=f"config.json 저장 완료 ({CONFIG_PATH})")

    def _run_script(self, script_name, label, cwd=None):
        script_path = os.path.join(ROOT, script_name)
        try:
            if os.name == "nt":
                subprocess.Popen([sys.executable, script_path], creationflags=subprocess.CREATE_NEW_CONSOLE, cwd=cwd or ROOT)
            else:
                subprocess.Popen([sys.executable, script_path], cwd=cwd or ROOT)
            return True
        except Exception as e:
            messagebox.showerror("실행 오류", f"{script_name} 실행 실패: {e}")
            return False

    def _run_script_blocking(self, script_path, label):
        """build_env.py처럼 '끝날 때까지 기다렸다가' 다음 단계로 넘어가야 하는 스크립트용.
        같은 콘솔에서 출력이 그대로 보이도록 GUI 창은 잠깐 멈춘 것처럼 보일 수 있습니다."""
        self.status.config(text=f"{label} 실행 중... (완료까지 대기)")
        self.root.update()
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if result.stdout:
                print(result.stdout)
            if result.returncode != 0:
                if result.stderr:
                    print(result.stderr)
                messagebox.showerror("실행 오류", f"{label} 실행 중 오류 발생 (콘솔 로그 확인)")
                return False
            return True
        except Exception as e:
            messagebox.showerror("실행 오류", f"{label} 실행 실패: {e}")
            return False

    def run_main(self):
        """플레이 버튼: config 저장 → build_env.py 완료 대기 → main.py 실행 (SUMO gui 포함)"""
        self.save_config_only()

        build_env_path = os.path.join(ROOT, "module1_simulation", "build_env.py")
        self.status.config(text="① build_env.py 실행 중... (맵/차량 새로 생성)")
        self.root.update()

        ok = self._run_script_blocking(build_env_path, "build_env.py")
        if not ok:
            self.status.config(text="build_env.py 실패 — main.py를 실행하지 않습니다.")
            return

        self.status.config(text="② build_env.py 완료. main.py 실행 중...")
        self.root.update()

        if self._run_script("main.py", "메인 파이프라인"):
            self.status.config(text="main.py 실행 요청 완료 — 새 콘솔 창에서 SUMO까지 진행상황 확인하세요")

    def run_training(self):
        """학습만: python train.py 실행"""
        self.save_config_only()
        if self._run_script("train.py", "학습"):
            self.status.config(text="train.py 실행 요청 완료 — 콘솔에서 진행상황 확인하세요")


if __name__ == "__main__":
    root = tk.Tk()
    app = ConfigGUI(root)
    root.mainloop()