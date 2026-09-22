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

예측수요는 module4_dispatch/category_demand_predictor.py의 CategoryDemandPredictor를 그대로 씀
(rl_env.py 학습 때와 동일한 피처 구성 - 모델 없으면 자동으로 0 대체, 에러 안 남).
"""
import os

import numpy as np
import traci
from stable_baselines3 import PPO

from poi_extractor import CATEGORIES as ZONE_CATEGORIES
from module4_dispatch.category_demand_predictor import CategoryDemandPredictor

DECISION_INTERVAL_SEC = 60.0
CATEGORY_MODEL_PATH_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "saved_models", "category_demand_models.pkl"
)


class RLRepositionMaintainer:
    def __init__(self, model_path: str, meta: dict, category_model_path: str = None):
        self.model = PPO.load(model_path)
        self.zones = meta["zones"]
        self._last_decision_time = -DECISION_INTERVAL_SEC
        self._unreachable_edges = set()

        category_model_path = category_model_path or CATEGORY_MODEL_PATH_DEFAULT
        self.demand_predictor = None
        if os.path.exists(category_model_path):
            self.demand_predictor = CategoryDemandPredictor(category_model_path)
        # 없으면 그냥 None으로 두고 예측수요=0으로 대체 (rl_env.py와 동일한 정책)

        self._edge_to_zone = {}
        for cat, edges in self.zones.items():
            for e in edges:
                self._edge_to_zone[e] = cat

    def _zone_counts(self):
        waiting = {c: 0 for c in ZONE_CATEGORIES}
        idle = {c: 0 for c in ZONE_CATEGORIES}

        for pid in traci.person.getIDList():
            try:
                if traci.person.getVehicle(pid):
                    continue
                edge = traci.person.getRoadID(pid)
            except traci.exceptions.TraCIException:
                continue
            cat = self._edge_to_zone.get(edge)
            if cat:
                waiting[cat] += 1

        for vid in traci.vehicle.getTaxiFleet(0):
            try:
                edge = traci.vehicle.getRoadID(vid)
            except traci.exceptions.TraCIException:
                continue
            cat = self._edge_to_zone.get(edge)
            if cat:
                idle[cat] += 1

        return waiting, idle

    def _build_obs(self, now, sim_start_hour, sim_end_hour):
        waiting, idle = self._zone_counts()

        if self.demand_predictor is not None:
            self.demand_predictor.update(waiting)  # rl_env.py의 _get_obs()와 동일한 순서
            predicted = self.demand_predictor.predict_by_zone(now)
        else:
            predicted = {c: 0.0 for c in ZONE_CATEGORIES}

        sim_start = sim_start_hour * 3600.0
        sim_end = sim_end_hour * 3600.0
        time_frac = min(max((now - sim_start) / max(sim_end - sim_start, 1.0), 0.0), 1.0)

        vec = []
        for c in ZONE_CATEGORIES:
            vec += [float(waiting[c]), float(idle[c]), float(predicted.get(c, 0.0))]
        vec.append(time_frac)
        return np.array(vec, dtype=np.float32)

    def maintain(self, now: float, sim_start_hour: float = 0, sim_end_hour: float = 24):
        if now - self._last_decision_time < DECISION_INTERVAL_SEC:
            return
        self._last_decision_time = now

        obs = self._build_obs(now, sim_start_hour, sim_end_hour)
        action, _ = self.model.predict(obs, deterministic=True)
        weights = np.asarray(action, dtype=np.int64).reshape(-1)
        if weights.size != len(ZONE_CATEGORIES):
            raise ValueError(
                "현재 RL 모델은 구형 단일-택시 행동 공간으로 학습되었습니다. "
                "새 rl_env.py로 rl_train.py를 다시 실행해 모델을 재학습하세요."
            )
        if not np.any(weights):
            return

        eligible_taxis = self._eligible_idle_taxis()
        if not eligible_taxis:
            return

        current_idle = {c: 0 for c in ZONE_CATEGORIES}
        for _, edge, source in eligible_taxis:
            if source:
                current_idle[source] += 1

        total_idle = len(eligible_taxis)
        raw_targets = weights / weights.sum() * total_idle
        target_counts = np.floor(raw_targets).astype(int)
        for index in np.argsort(raw_targets - target_counts)[::-1][:total_idle - target_counts.sum()]:
            target_counts[index] += 1

        # 목표 대수를 넘긴 구역의 택시만 이동 후보로 삼는다. 이 때문에 이미
        # 적정 위치에 있는 택시가 매분 다시 목적지를 받는 일이 없다.
        movable = []
        np.random.shuffle(eligible_taxis)
        for taxi_id, edge, source in eligible_taxis:
            if source is not None:
                source_index = ZONE_CATEGORIES.index(source)
                if current_idle[source] <= target_counts[source_index]:
                    continue
                current_idle[source] -= 1
            movable.append((taxi_id, edge))

        for target_index, category in enumerate(ZONE_CATEGORIES):
            target_edges = self.zones.get(category) or []
            deficit = max(0, target_counts[target_index] - current_idle[category])
            while deficit > 0 and movable and target_edges:
                taxi_id, current_edge = movable.pop()
                if self._move_taxi(taxi_id, current_edge, target_edges):
                    current_idle[category] += 1
                    deficit -= 1

    @staticmethod
    def _has_no_pending_stop(taxi_id: str) -> bool:
        try:
            return len(traci.vehicle.getStops(taxi_id, 1)) == 0
        except traci.exceptions.TraCIException:
            return False

    def _eligible_idle_taxis(self):
        """승객 배차가 이미 예약된 택시는 제외한 재배치 후보를 반환한다."""
        eligible = []
        for taxi_id in traci.vehicle.getTaxiFleet(0):
            if not self._has_no_pending_stop(taxi_id):
                continue
            try:
                edge = traci.vehicle.getRoadID(taxi_id)
            except traci.exceptions.TraCIException:
                continue
            if edge and not edge.startswith(":"):
                eligible.append((taxi_id, edge, self._edge_to_zone.get(edge)))
        return eligible

    def _move_taxi(self, taxi_id: str, current_edge: str, target_edges: list) -> bool:
        """도달 가능한 도로만 선택해 SUMO 재배치 충돌을 피한다."""
        candidates = list(target_edges)
        np.random.shuffle(candidates)
        for target_edge in candidates:
            if target_edge == current_edge or (current_edge, target_edge) in self._unreachable_edges:
                continue
            try:
                route = traci.simulation.findRoute(current_edge, target_edge)
                if route is None or not route.edges:
                    continue
                traci.vehicle.changeTarget(taxi_id, target_edge)
                return True
            except traci.exceptions.TraCIException:
                self._unreachable_edges.add((current_edge, target_edge))
        return False
