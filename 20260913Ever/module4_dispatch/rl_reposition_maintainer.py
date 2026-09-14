"""
module4_dispatch/rl_reposition_maintainer.py

학습된 RL 정책을 measure_wait_time.py 루프 안에 Random/Hungarian과 같은 방식으로 끼우기 위한 wrapper.
HungarianDispatcher가 "maintain(now)"로 매 스텝 불리는 것과 동일한 인터페이스를 맞춤.

기존 measure_wait_time.py에서:
    hungarian_dispatcher = HungarianDispatcher() if meta.get("taxi_dispatch_algorithm") == "hungarian" else None
    ...
    if hungarian_dispatcher:
        hungarian_dispatcher.maintain(now)

이걸 다음처럼 바꾸면 RL도 같은 A/B 비교 플로우에 들어감:
    reposition_maintainer = (
        RLRepositionMaintainer(MODEL_PATH, meta) if meta.get("taxi_dispatch_algorithm") == "rl_reposition" else None
    )
    ...
    if hungarian_dispatcher:
        hungarian_dispatcher.maintain(now)      # 매칭은 그대로 Hungarian이 담당
    if reposition_maintainer:
        reposition_maintainer.maintain(now)     # RL은 재배치만 별도로 담당 (둘이 같이 켜도 됨)

즉 RL을 "배차 알고리즘 자체의 대체재"가 아니라 "기존 배차 위에 얹는 재배치 보조 정책"으로 넣는 구조.
test_matching_comparison.py / multi_factor_compare.py의 taxi_dispatch_algorithm 값에
"rl_reposition"을 새 옵션으로 추가하면 기존 비교 스크립트 그대로 재사용 가능.
"""
import traci
from stable_baselines3 import PPO

from poi_extractor import CATEGORIES as ZONE_CATEGORIES

DECISION_INTERVAL_SEC = 60.0


class RLRepositionMaintainer:
    def __init__(self, model_path: str, meta: dict):
        self.model = PPO.load(model_path)
        self.zones = meta["zones"]
        self._last_decision_time = -DECISION_INTERVAL_SEC

    def _zone_counts(self):
        edge_to_zone = {}
        for cat, edges in self.zones.items():
            for e in edges:
                edge_to_zone[e] = cat

        waiting = {c: 0 for c in ZONE_CATEGORIES}
        idle = {c: 0 for c in ZONE_CATEGORIES}

        for pid in traci.person.getIDList():
            try:
                if traci.person.getVehicle(pid):
                    continue
                edge = traci.person.getRoadID(pid)
            except traci.exceptions.TraCIException:
                continue
            cat = edge_to_zone.get(edge)
            if cat:
                waiting[cat] += 1

        for vid in traci.vehicle.getTaxiFleet(0):
            try:
                edge = traci.vehicle.getRoadID(vid)
            except traci.exceptions.TraCIException:
                continue
            cat = edge_to_zone.get(edge)
            if cat:
                idle[cat] += 1

        return waiting, idle, edge_to_zone

    def _build_obs(self, now, sim_start_hour, sim_end_hour):
        import numpy as np
        waiting, idle, _ = self._zone_counts()
        sim_start = sim_start_hour * 3600.0
        sim_end = sim_end_hour * 3600.0
        time_frac = min(max((now - sim_start) / max(sim_end - sim_start, 1.0), 0.0), 1.0)
        vec = []
        for c in ZONE_CATEGORIES:
            vec += [float(waiting[c]), float(idle[c]), 0.0]  # 예측수요 자리 (TODO: Module3 연동)
        vec.append(time_frac)
        return np.array(vec, dtype=np.float32)

    def maintain(self, now: float, sim_start_hour: float = 0, sim_end_hour: float = 24):
        if now - self._last_decision_time < DECISION_INTERVAL_SEC:
            return
        self._last_decision_time = now

        idle_taxis = list(traci.vehicle.getTaxiFleet(0))
        if not idle_taxis:
            return

        obs = self._build_obs(now, sim_start_hour, sim_end_hour)
        action, _ = self.model.predict(obs, deterministic=True)
        action = int(action)
        if action == 0:
            return

        target_category = ZONE_CATEGORIES[action - 1]
        target_edges = self.zones.get(target_category) or []
        if not target_edges:
            return

        taxi_id = idle_taxis[0]
        target_edge = target_edges[0]
        try:
            traci.vehicle.changeTarget(taxi_id, target_edge)
        except traci.exceptions.TraCIException:
            pass
