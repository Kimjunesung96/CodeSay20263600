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

from config_loader import CFG
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

    # 설정 파일에 키가 없어도 바로 학습할 수 있는 실용적인 기본값.
    # 1분 단위 의사결정이므로 기본 에피소드 길이는 시뮬레이션 시간(분)과 같다.
    sim_minutes = max(
        1,
        int((meta.get("sim_end_hour", 24) - meta.get("sim_start_hour", 0)) * 60),
    )
    max_decisions = int(CFG.get("rl_max_decisions", sim_minutes))
    total_timesteps = int(CFG.get("rl_total_timesteps", 20_000))
    rollout_steps = int(CFG.get("rl_rollout_steps", 256))
    batch_size = int(CFG.get("rl_batch_size", 64))
    seed = int(CFG.get("rl_seed", 42))

    if total_timesteps <= 0 or rollout_steps <= 0 or batch_size <= 0 or max_decisions <= 0:
        raise ValueError("RL 학습 설정값은 모두 0보다 커야 합니다.")
    if rollout_steps % batch_size != 0:
        raise ValueError("rl_rollout_steps는 rl_batch_size로 나누어떨어져야 합니다.")

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
        max_decisions=max_decisions,
        sumo_binary="sumo",     # GUI 없이 headless로 (학습은 GUI 필요 없음)
        # [Surge Pricing 대체] GUI 탭4의 "인센티브 리워드 가중치" 슬라이더 값.
        # config.json에 없으면 0.0 (기존 대기감소 보상만 쓰는 것과 완전히 동일하게 동작)
        incentive_weight=CFG.get("incentive_weight", 0.0),
        target_hotspot_ratio=CFG.get("target_hotspot_ratio", 0.8),
        randomize_passenger_seed=CFG.get("rl_randomize_passenger_seed", True),
        training_seed_base=int(CFG.get("rl_training_seed_base", 1000)),
    )

    # 최초 1회는 환경 스펙이 맞는지 검증 (문제 있으면 여기서 assert로 알려줌)
    check_env(env, warn=True)

    print(
        f"[학습 설정] 에피소드 최대 {max_decisions}회 결정 | "
        f"총 {total_timesteps:,} steps | seed={seed} | "
        f"행동=구역별 유휴 택시 배분"
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=rollout_steps,
        batch_size=batch_size,
        learning_rate=float(CFG.get("rl_learning_rate", 3e-4)),
        gamma=float(CFG.get("rl_gamma", 0.99)),
        seed=seed,
        policy_kwargs={"net_arch": [128, 128]},
    )

    progress_cb = ProgressCallback(total_timesteps=total_timesteps, print_every=rollout_steps)
    # 512스텝(=2번 rollout)마다 중간 저장 -> 도중에 취소/컴 꺼짐이 있어도
    # rl_reposition_model_<스텝수>_steps.zip 중 가장 최신 걸로 이어서 쓸 수 있음
    checkpoint_cb = CheckpointCallback(
        save_freq=max(rollout_steps * 2, 1),
        save_path=os.path.join(PROJECT_ROOT, "module4_dispatch", "checkpoints"),
        name_prefix="rl_reposition_model",
    )
    model.learn(total_timesteps=total_timesteps, callback=[progress_cb, checkpoint_cb])
    model.save(MODEL_OUT)
    print(f"[완료] 학습된 정책을 저장했습니다: {MODEL_OUT}")


if __name__ == "__main__":
    main()
