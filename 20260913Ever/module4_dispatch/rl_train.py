"""
module4_dispatch/rl_train.py

RL 재배치 정책 학습 스크립트. headless(sumo, GUI 없음)로 빠르게 여러 episode 반복.

사용법:
    python module4_dispatch/rl_train.py

사전 준비:
    1. build_env.py를 먼저 한 번 돌려서 module1_simulation/sumo_config/ 에
       simulation.sumocfg + runtime_meta.json 이 생성되어 있어야 함
       (기존 프로젝트 흐름과 동일 - measure_wait_time.py 실행 전 단계)
    2. pip install gymnasium stable-baselines3
"""
import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from module4_dispatch.rl_env import TaxiRepositionEnv
from module4_dispatch.category_demand_predictor import CategoryDemandPredictor

SUMO_CFG = os.path.join(PROJECT_ROOT, "module1_simulation", "sumo_config", "simulation.sumocfg")
META_PATH = os.path.join(PROJECT_ROOT, "module1_simulation", "sumo_config", "runtime_meta.json")
MODEL_OUT = os.path.join(PROJECT_ROOT, "module4_dispatch", "rl_reposition_model.zip")
CATEGORY_MODEL_PATH = os.path.join(PROJECT_ROOT, "saved_models", "category_demand_models.pkl")


class ProgressCallback(BaseCallback):
    """스텝마다 %와 예상 남은 시간을 콘솔에 찍어주는 콜백. n_steps(=256)마다 한 번씩만 출력."""

    def __init__(self, total_timesteps: int, print_every: int = 256):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.print_every = print_every
        self._start_time = None

    def _on_training_start(self):
        import time
        self._start_time = time.time()

    def _on_step(self) -> bool:
        import time
        if self.num_timesteps % self.print_every == 0:
            elapsed = time.time() - self._start_time
            pct = 100.0 * self.num_timesteps / self.total_timesteps
            per_step = elapsed / max(self.num_timesteps, 1)
            remaining = per_step * (self.total_timesteps - self.num_timesteps)
            print(f"[진행률] {self.num_timesteps}/{self.total_timesteps} "
                  f"({pct:.1f}%) | 경과 {elapsed/60:.1f}분 | "
                  f"예상 남은 시간 {remaining/60:.1f}분")
        return True


def main():
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # module3_prediction/train_category_demand.py를 먼저 돌려서 이 파일이 있으면 실제 예측 사용,
    # 없으면 경고만 찍고 0으로 대체(demand_predictor=None) — 학습 파이프라인 자체는 계속 동작함
    demand_predictor = None
    if os.path.exists(CATEGORY_MODEL_PATH):
        demand_predictor = CategoryDemandPredictor(CATEGORY_MODEL_PATH)
        print(f"[안내] 카테고리별 수요예측 모델을 불러왔습니다: {CATEGORY_MODEL_PATH}")
    else:
        print(f"[경고] {CATEGORY_MODEL_PATH}가 없어 예측수요=0으로 대체합니다. "
              f"module3_prediction/train_category_demand.py를 먼저 돌리는 것을 권장합니다.")

    env = TaxiRepositionEnv(
        sumo_cfg_path=SUMO_CFG,
        meta=meta,
        demand_predictor=demand_predictor,
        max_decisions=60,       # 처음엔 짧게 (60분 상당) 돌려서 학습 파이프라인부터 검증
        sumo_binary="sumo",     # GUI 없이 headless로 (학습은 GUI 필요 없음)
    )

    # 최초 1회는 환경 스펙이 맞는지 검증 (문제 있으면 여기서 assert로 알려줌)
    check_env(env, warn=True)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=256,     # 짧은 episode 기준으로 작게 시작
        batch_size=64,
        learning_rate=3e-4,
    )

    TOTAL_TIMESTEPS = 20_000  # 처음엔 작게 돌려보고 점진적으로 늘리기 (SUMO가 느려서 체감 시간 김)
    progress_cb = ProgressCallback(total_timesteps=TOTAL_TIMESTEPS, print_every=256)
    # 512스텝(=2번 rollout)마다 중간 저장 -> 도중에 취소/컴 꺼짐이 있어도
    # rl_reposition_model_<스텝수>_steps.zip 중 가장 최신 걸로 이어서 쓸 수 있음
    checkpoint_cb = CheckpointCallback(
        save_freq=512,
        save_path=os.path.join(PROJECT_ROOT, "module4_dispatch", "checkpoints"),
        name_prefix="rl_reposition_model",
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[progress_cb, checkpoint_cb])
    model.save(MODEL_OUT)
    print(f"[완료] 학습된 정책을 저장했습니다: {MODEL_OUT}")


if __name__ == "__main__":
    main()