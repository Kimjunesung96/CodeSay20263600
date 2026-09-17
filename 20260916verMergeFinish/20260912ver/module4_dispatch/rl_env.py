"""
module4_dispatch/rl_env.py

목적: 승객-택시 "매칭"은 기존 HungarianDispatcher가 그대로 담당하고,
      RL은 그 위에서 "유휴 택시를 어느 zone(카테고리)으로 미리 보내놓을지"만 결정한다.

왜 이렇게 나누냐면:
- Hungarian은 "그 순간"의 최적 매칭을 수학적으로 보장 -> 매칭 자체를 RL로 바꿔도 이득이 적음
- 반대로 "빈 택시을 미리 어디에 배치해둘지"는 미래 수요를 봐야 하는 문제라 RL이 붙을 자리가 있음

연결 지점:
- taxi_manager.TaxiFleetManager  : 택시 대수 유지 (그대로 둠, 건드리지 않음)
- hungarian_dispatcher.HungarianDispatcher.maintain() : 매 스텝 실제 매칭 수행 (그대로 재사용)
- poi_extractor.CATEGORIES       : zone 정의
- runtime_meta.json 의 "zones"   : {category: [edge_id, ...]}

설치: pip install gymnasium stable-baselines3
"""

import os
import sys

import numpy as np
import traci

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import gymnasium as gym
from gymnasium import spaces

from taxi_manager import TaxiFleetManager
from passenger_manager import PassengerTimeoutManager
from passenger_spawn_manager import PassengerSpawnManager
from hungarian_dispatcher import HungarianDispatcher
from poi_extractor import CATEGORIES as ZONE_CATEGORIES


N_ZONES = len(ZONE_CATEGORIES)
DECISION_INTERVAL_SEC = 60.0  # RL이 몇 초마다 판단할지 (1분 단위 권장)


class TaxiRepositionEnv(gym.Env):
    """
    Observation (shape = N_ZONES*3 + 1):
        [zone별 대기승객수, zone별 유휴택시수, zone별 예측수요]
        * N_ZONES + [정규화된 현재시각]

    Action (Discrete(N_ZONES + 1)):
        0            -> 재배치 안 함
        1..N_ZONES   -> 해당 zone(카테고리)으로 유휴 택시 1대 재배치

    Reward:
        이번 60초 구간의 대기 승객 수 변화.
        대기 승객이 줄어들면 양수, 늘어나면 음수.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        sumo_cfg_path: str,
        meta: dict,
        demand_predictor=None,
        max_decisions: int = 360,
        sumo_binary: str = "sumo",
    ):
        super().__init__()

        self.sumo_cfg_path = sumo_cfg_path
        self.meta = meta
        self.zones = meta["zones"]
        self.demand_predictor = demand_predictor
        self.max_decisions = max_decisions
        self.sumo_binary = sumo_binary

        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(N_ZONES * 3 + 1,),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(N_ZONES + 1)

        self._edge_to_zone = {}

        for cat, edges in self.zones.items():
            for edge in edges:
                self._edge_to_zone[edge] = cat

        self.dispatcher = None
        self.fleet_manager = None
        self.pax_manager = None
        self.spawn_manager = None
        self._decision_count = 0
        self._sumo_started = False
        self._depart_time = {}

    # ---------------- gymnasium API ----------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._close_sumo()

        traci.start(
            [
                self.sumo_binary,
                "-c",
                self.sumo_cfg_path,
                "--no-step-log",
                "true",
                "--no-warnings",
                "true",
            ]
        )

        self._sumo_started = True
        self._decision_count = 0
        self._depart_time = {}

        # -------------------------------------------------------------
        # [수정됨] 실제 시뮬레이션 환경과 동일하게 누락된 4대 매니저 모두 투입
        # -------------------------------------------------------------
        self.dispatcher = HungarianDispatcher()
        
        sim_start_hour = self.meta.get("sim_start_hour", 0)
        
        self.fleet_manager = TaxiFleetManager(
            target_count=self.meta["num_taxis"],
            boundary_edges=self.meta["boundary_edges"],
            all_edges=self.meta["edges"],
            strategy="patrol",
            hotspot_edges=self.meta.get("hotspot_edges"),
            zones=self.meta.get("zones"),
            sim_start_hour=sim_start_hour,
        )
        
        self.pax_manager = PassengerTimeoutManager(
            wait_timeout_sec=self.meta.get("passenger_wait_timeout", 900)
        )
        
        self.spawn_manager = PassengerSpawnManager(
            zones=self.meta.get("zones", {}),
            sim_start_hour=sim_start_hour,
            sim_end_hour=self.meta.get("sim_end_hour"),
            school_pop_base=self.meta.get("school_pop_base", 400),
            company_pop_base=self.meta.get("company_pop_base", 100),
            seed=self.meta.get("passenger_seed"),
        )

        # 초기 워밍업 1구간
        self._advance_sim(DECISION_INTERVAL_SEC)
        obs = self._get_obs()

        return obs, {}

    def step(self, action):
        prev_waiting, _ = self._zone_counts()
        prev_n_waiting = sum(prev_waiting.values())

        self._apply_action(int(action))
        self._advance_sim(DECISION_INTERVAL_SEC)

        waiting, _ = self._zone_counts()
        n_waiting = sum(waiting.values())

        # 동적 승객이 정상 스폰되므로 이제 진짜 보상이 집계됨
        reward = float(prev_n_waiting - n_waiting)
        obs = self._get_obs()

        self._decision_count += 1
        terminated = traci.simulation.getMinExpectedNumber() <= 0
        truncated = self._decision_count >= self.max_decisions

        if terminated or truncated:
            self._close_sumo()

        return obs, reward, terminated, truncated, {}

    def close(self):
        self._close_sumo()

    def _close_sumo(self):
        if self._sumo_started:
            try:
                traci.close()
            except Exception:
                pass
            self._sumo_started = False

    # ---------------- 내부 로직 ----------------

    def _advance_sim(self, duration_sec: float):
        end_time = traci.simulation.getTime() + duration_sec

        while (
            traci.simulation.getTime() < end_time
            and traci.simulation.getMinExpectedNumber() > 0
        ):
            traci.simulationStep()
            now = traci.simulation.getTime()

            # -------------------------------------------------------------
            # [수정됨] 모든 매니저를 실시간으로 가동하여 환경 정상화
            # -------------------------------------------------------------
            self.fleet_manager.maintain(now)
            self.pax_manager.maintain(now)
            self.dispatcher.maintain(now)
            self.spawn_manager.maintain(now, timeout_removed_pids=self.pax_manager.removed_pids)

            # depart 시각 기록
            for pid in traci.person.getIDList():
                if pid not in self._depart_time:
                    self._depart_time[pid] = now

    def _apply_action(self, action: int):
        """
        RL이 선택한 카테고리로 유휴 택시 1대를 재배치한다.
        """
        if action == 0:
            return

        target_category = ZONE_CATEGORIES[action - 1]
        target_edges = self.zones.get(target_category) or []
        if not target_edges:
            return

        idle_taxis = list(traci.vehicle.getTaxiFleet(0))
        if not idle_taxis:
            return
            
        # -------------------------------------------------------------
        # [수정됨] 특정 택시(taxi_0)가 에러 굴레에 빠지는 것을 막기 위해 셔플
        # -------------------------------------------------------------
        np.random.shuffle(idle_taxis)

        for taxi_id in idle_taxis:
            try:
                current_edge = traci.vehicle.getRoadID(taxi_id)
            except traci.exceptions.TraCIException:
                continue

            if not current_edge or current_edge.startswith(":"):
                continue

            candidates = list(target_edges)
            np.random.shuffle(candidates)

            success = False
            for target_edge in candidates:
                if target_edge == current_edge:
                    continue
                try:
                    route = traci.simulation.findRoute(current_edge, target_edge)
                    if route is None or not route.edges:
                        continue

                    traci.vehicle.changeTarget(taxi_id, target_edge)
                    success = True
                    break  # 목적지 배정 성공 시 중단
                except traci.exceptions.TraCIException:
                    continue

            # 이 택시로 배정에 성공했다면 다른 유휴 택시는 건드리지 않고 종료
            if success:
                return

    def _zone_counts(self):
        waiting = {category: 0 for category in ZONE_CATEGORIES}
        idle = {category: 0 for category in ZONE_CATEGORIES}

        for pid in traci.person.getIDList():
            try:
                if traci.person.getVehicle(pid):
                    continue
                edge = traci.person.getRoadID(pid)
            except traci.exceptions.TraCIException:
                continue
            category = self._edge_to_zone.get(edge)
            if category:
                waiting[category] += 1

        for vid in traci.vehicle.getTaxiFleet(0):
            try:
                edge = traci.vehicle.getRoadID(vid)
            except traci.exceptions.TraCIException:
                continue
            category = self._edge_to_zone.get(edge)
            if category:
                idle[category] += 1

        return waiting, idle

    def _predicted_demand(self):
        if self.demand_predictor is None:
            return {category: 0.0 for category in ZONE_CATEGORIES}
        return self.demand_predictor.predict_by_zone(traci.simulation.getTime())

    def _get_obs(self):
        waiting, idle = self._zone_counts()

        if (
            self.demand_predictor is not None
            and hasattr(self.demand_predictor, "update")
        ):
            self.demand_predictor.update(waiting)

        predicted = self._predicted_demand()
        now = traci.simulation.getTime()

        sim_start = self.meta.get("sim_start_hour", 0) * 3600.0
        sim_end = self.meta.get("sim_end_hour", 24) * 3600.0
        sim_len = max(sim_end - sim_start, 1.0)
        time_frac = min(max((now - sim_start) / sim_len, 0.0), 1.0)

        vec = []
        for category in ZONE_CATEGORIES:
            vec += [
                float(waiting[category]),
                float(idle[category]),
                float(predicted.get(category, 0.0)),
            ]
        vec.append(time_frac)

        return np.array(vec, dtype=np.float32)