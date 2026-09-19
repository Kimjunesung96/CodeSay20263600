# test_reset_only.py (프로젝트 루트에 임시로 만들어서 실행)
import json, os
from module4_dispatch.rl_env import TaxiRepositionEnv

META_PATH = os.path.join("module1_simulation", "sumo_config", "runtime_meta.json")
with open(META_PATH, encoding="utf-8") as f:
    meta = json.load(f)

env = TaxiRepositionEnv(
    sumo_cfg_path=os.path.join("module1_simulation", "sumo_config", "simulation.sumocfg"),
    meta=meta,
    demand_predictor=None,
    max_decisions=5,
    sumo_binary="sumo",
)
print("reset 시작...")
obs, _ = env.reset()
print("reset 성공:", obs.shape)
env.close()