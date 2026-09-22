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
from module4_dispatch.hungarian_dispatcher import HungarianDispatcher
from poi_extractor import CATEGORIES as ZONE_CATEGORIES


N_ZONES = len(ZONE_CATEGORIES)
DECISION_INTERVAL_SEC = 60.0  # RL이 몇 초마다 판단할지 (1분 단위 권장)


def _has_no_pending_stop(taxi_id: str) -> bool:
    """getTaxiFleet(0)만으론 안 걸러지는, Hungarian이 방금 배정한 픽업/하차
    스케줄(stop)이 있는지 확인. 있으면 RL 재배치 대상에서 제외해야 함
    (안 그러면 SUMO 내부에서 stop 재배정 충돌 -> bad allocation 크래시로 이어짐)."""
    try:
        return len(traci.vehicle.getStops(taxi_id, 1)) == 0
    except traci.exceptions.TraCIException:
        return False  # 조회 실패한 택시는 안전하게 건드리지 않음


class TaxiRepositionEnv(gym.Env):
    """
    Observation (shape = N_ZONES*3 + 1):
        [zone별 대기승객수, zone별 유휴택시수, zone별 예측수요]
        * N_ZONES + [정규화된 현재시각]

    Action (MultiDiscrete([4] * N_ZONES)):
        zone별 0~3 우선순위를 고른다. 우선순위를 비율로 바꿔 현재 유휴 택시를
        여러 zone에 배분한다. 모든 값이 0이면 재배치하지 않는다.

    Reward:
        탑승 성공에는 보상, 대기 승객·타임아웃·공차 재배치 거리에는 패널티를 준다.
        핫스팟/비핫스팟 유휴 택시 비율이 목표에 가까우면 작은 보조 보상을 준다.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        sumo_cfg_path: str,
        meta: dict,
        demand_predictor=None,
        max_decisions: int = 360,
        sumo_binary: str = "sumo",
        incentive_weight: float = 0.0,  # Surge Pricing 대체: 핫스팟 비율 일치도 보상 가중치 (탭4 슬라이더)
        target_hotspot_ratio: float = 0.5,  # 목표: 유휴택시 중 몇 %가 핫스팟에 있어야 하는지 (0~1, 탭4 슬라이더)
        randomize_passenger_seed: bool = False,
        training_seed_base: int = 1000,
        worker_index: int = 0,
    ):
        super().__init__()

        self.sumo_cfg_path = sumo_cfg_path
        self.meta = meta
        self.zones = meta["zones"]
        self.demand_predictor = demand_predictor
        self.max_decisions = max_decisions
        self.sumo_binary = sumo_binary
        self.incentive_weight = incentive_weight
        self.target_hotspot_ratio = target_hotspot_ratio
        self.randomize_passenger_seed = randomize_passenger_seed
        # 병렬(SubprocVecEnv)로 여러 개를 동시에 돌릴 때, 각 worker가 서로 다른
        # seed 구간을 쓰게 하기 위한 오프셋. worker_index=0이면 기존과 완전히 동일하게
        # 동작해서(단일 env 학습과 결과가 똑같음) 하위호환됨.
        # worker당 최대 10만 에피소드까지는 다른 worker와 안 겹치게 구간을 나눔.
        self.training_seed_base = training_seed_base + worker_index * 100_000

        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(N_ZONES * 3 + 1,),
            dtype=np.float32,
        )

        # 구역별 우선순위(0=배치하지 않음, 3=가장 높은 비중). PPO는 MultiDiscrete를 지원한다.
        self.action_space = spaces.MultiDiscrete(np.full(N_ZONES, 4, dtype=np.int64))

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
        self._unreachable_edges = set()  # (from_edge, to_edge) 도달 불가 캐시 - 에피소드 넘어가도 유지
        self._last_reposition_distance_m = 0.0
        self._episode_index = 0

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
        self._last_reposition_distance_m = 0.0
        self._episode_index += 1
        passenger_seed = self.meta.get("passenger_seed")
        if self.randomize_passenger_seed:
            passenger_seed = self.training_seed_base + self._episode_index

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
            wait_timeout_sec=self.meta.get("passenger_wait_timeout", 900),
            dispatcher=self.dispatcher,  # 예약된 승객 강제제거 충돌 방지 (passenger_manager.py 참고)
        )
        
        self.spawn_manager = PassengerSpawnManager(
            zones=self.meta.get("zones", {}),
            sim_start_hour=sim_start_hour,
            sim_end_hour=self.meta.get("sim_end_hour"),
            school_pop_base=self.meta.get("school_pop_base", 400),
            company_pop_base=self.meta.get("company_pop_base", 100),
            seed=passenger_seed,
        )

        # 초기 워밍업 1구간
        self._advance_sim(DECISION_INTERVAL_SEC)
        obs = self._get_obs()

        return obs, {}

    def step(self, action):
        prev_waiting, _ = self._zone_counts()
        prev_in_taxi = self._riding_passenger_ids()
        prev_timeout_count = len(self.pax_manager.removed_pids)

        self._apply_action(action)
        self._advance_sim(DECISION_INTERVAL_SEC)

        waiting, idle = self._zone_counts()
        n_waiting = sum(waiting.values())
        newly_picked_up = len(self._riding_passenger_ids() - prev_in_taxi)
        new_timeouts = len(self.pax_manager.removed_pids) - prev_timeout_count

        # 목적 함수: 많이 태우고, 오래 기다리거나 포기하는 승객과 공차 이동은 줄인다.
        reward = 3.0 * newly_picked_up
        reward -= 0.10 * n_waiting
        reward -= 8.0 * max(new_timeouts, 0)
        reward -= 0.0002 * self._last_reposition_distance_m

        # [Surge Pricing 대체] 핫스팟 분포 일치도 보상 - "예측수요 비율대로 유휴택시가
        # 얼마나 잘 퍼져있는지"를 매 스텝 점수로 매겨서 더함. incentive_weight=0이면
        # (기본값) 기존 대기감소 보상만 그대로 쓰는 것과 완전히 동일 - GUI 탭4 슬라이더로
        # 0보다 크게 주면 "인센티브가 기사 행동(재배치)에 영향을 준다"는 걸 학습이 반영함.
        if self.incentive_weight > 0:
            reward += self.incentive_weight * self._distribution_match_score(
                idle, self._predicted_demand()
            )

        obs = self._get_obs()

        self._decision_count += 1
        terminated = traci.simulation.getMinExpectedNumber() <= 0
        truncated = self._decision_count >= self.max_decisions

        if terminated or truncated:
            self._close_sumo()

        return obs, reward, terminated, truncated, {
            "picked_up": newly_picked_up,
            "timeouts": max(new_timeouts, 0),
            "waiting": n_waiting,
            "reposition_distance_m": self._last_reposition_distance_m,
        }

    def close(self):
        self._close_sumo()

    def _close_sumo(self):
        if self._sumo_started:
            try:
                traci.close()
            except Exception:
                pass
            self._sumo_started = False

        # traci.close()만으로는 Windows에서 sumo.exe가 좀비로 남는 경우가 있어서,
        # 혹시 남아있는 이번 프로세스가 띄운 sumo를 확실히 정리. 반복 reset()마다
        # 좀비가 쌓이다가 메모리 부족(bad allocation)으로 죽는 문제를 막기 위함.
        try:
            import psutil
            for p in psutil.process_iter(["pid", "name", "ppid"]):
                if p.info["name"] and p.info["name"].lower().startswith("sumo") \
                        and p.info["ppid"] == os.getpid():
                    try:
                        p.kill()
                    except Exception:
                        pass
        except ImportError:
            pass  # psutil 없으면 그냥 넘어감 (pip install psutil 권장)

    # ---------------- 내부 로직 ----------------

    def _advance_sim(self, duration_sec: float):
        # fleet_manager/pax_manager/spawn_manager는 매초 안 불러도 되는 매니저라
        # THROTTLE_SEC(10초)에 한 번만 실행 -> traci 왕복 호출 수를 10분의 1로 줄여서
        # (특히 fleet_manager가 택시마다 5개씩 traci를 부르던 게 제일 무거웠음)
        # 학습 속도를 끌어올림. measure_wait_time.py(실제 A/B 평가)는 이 파일과
        # 무관한 별개 코드 경로라 매초 그대로 돌아가고, 여기(학습용 env)에만 적용됨.
        #
        # 반대로 dispatcher(Hungarian 배차)는 그대로 매초 유지함 -- 이건 스로틀하면
        # 승객이 최대 THROTTLE_SEC초까지 매칭 안 되고 기다리게 되어 "대기시간"이라는
        # 측정 대상 자체가 왜곡되고, RL 학습 환경과 실제 A/B 평가 사이에 조건이
        # 달라져버리기 때문.
        THROTTLE_SEC = 10.0

        end_time = traci.simulation.getTime() + duration_sec

        while (
            traci.simulation.getTime() < end_time
            and traci.simulation.getMinExpectedNumber() > 0
        ):
            traci.simulationStep()
            now = traci.simulation.getTime()

            # -------------------------------------------------------------
            # [수정됨] fleet/pax/spawn = 10초 스로틀, dispatcher(Hungarian) = 매초 유지
            # -------------------------------------------------------------
            if int(now) % THROTTLE_SEC == 0:
                self.fleet_manager.maintain(now)
                self.pax_manager.maintain(now)
                self.spawn_manager.maintain(now, timeout_removed_pids=self.pax_manager.removed_pids)

            self.dispatcher.maintain(now)  # Hungarian은 스로틀 대상 아님 - 매초 그대로

            # depart 시각 기록
            for pid in traci.person.getIDList():
                if pid not in self._depart_time:
                    self._depart_time[pid] = now

    def _apply_action(self, action):
        """구역별 우선순위를 현재 유휴 택시의 목표 배분 수로 바꿔 재배치한다."""
        self._last_reposition_distance_m = 0.0
        weights = np.asarray(action, dtype=np.int64).reshape(-1)
        if weights.size != N_ZONES or not np.any(weights):
            return

        idle_taxis = list(traci.vehicle.getTaxiFleet(0))
        if not idle_taxis:
            return

        # getTaxiFleet(0)은 "손님을 안 태운 택시"만 걸러줄 뿐, Hungarian이 이번 틱에
        # 막 픽업/하차 스케줄(stop)을 배정한 택시까지 걸러주진 않음. 이 상태에서
        # RL이 같은 택시의 changeTarget을 걸면 SUMO 내부에서
        # "could not assign stop ... after rerouting" 충돌이 나고, 이게 누적되면
        # bad allocation 크래시로 이어지는 걸로 보임. 그래서 이미 스케줄(stop)이
        # 걸려있는 택시는 재배치 후보에서 제외 - 진짜로 아무 일도 없는 택시만 건드림.
        idle_taxis = [t for t in idle_taxis if _has_no_pending_stop(t)]
        if not idle_taxis:
            return
            
        np.random.shuffle(idle_taxis)
        _, current_idle = self._zone_counts()
        total_idle = len(idle_taxis)
        raw_targets = weights / weights.sum() * total_idle
        target_counts = np.floor(raw_targets).astype(int)
        # 반올림 때문에 남는 택시는 우선순위 비율의 소수 부분이 큰 구역부터 배분한다.
        for index in np.argsort(raw_targets - target_counts)[::-1][:total_idle - target_counts.sum()]:
            target_counts[index] += 1

        movable = []
        for taxi_id in idle_taxis:
            try:
                current_edge = traci.vehicle.getRoadID(taxi_id)
            except traci.exceptions.TraCIException:
                continue
            if not current_edge or current_edge.startswith(":"):
                continue

            source = self._edge_to_zone.get(current_edge)
            # 현재 구역의 목표 대수보다 초과한 택시만 이동시킨다. 이미 적정 배치인
            # 택시를 반복 이동시키지 않아 공차 이동과 SUMO 충돌을 줄인다.
            if source is not None:
                source_index = ZONE_CATEGORIES.index(source)
                if current_idle[source] <= target_counts[source_index]:
                    continue
                current_idle[source] -= 1
            movable.append((taxi_id, current_edge))

        for target_index, target_category in enumerate(ZONE_CATEGORIES):
            deficit = max(0, target_counts[target_index] - current_idle[target_category])
            target_edges = self.zones.get(target_category) or []
            if not target_edges:
                continue

            while deficit > 0 and movable:
                taxi_id, current_edge = movable.pop()
                if self._move_taxi_to_zone(taxi_id, current_edge, target_edges):
                    current_idle[target_category] += 1
                    deficit -= 1

    def _move_taxi_to_zone(self, taxi_id: str, current_edge: str, target_edges: list) -> bool:
        """도달 가능한 목적지 하나를 찾아 택시를 이동시키고 공차 거리를 기록한다."""
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
                self._last_reposition_distance_m += max(float(route.length), 0.0)
                return True
            except traci.exceptions.TraCIException:
                self._unreachable_edges.add((current_edge, target_edge))
        return False

    def _distribution_match_score(self, idle: dict, predicted: dict) -> float:
        """
        [핫스팟 비율 조절] "핫스팟(학교/회사/음식점/지하철입구/버스정류장) vs 비핫스팟(주거)
        구역에 유휴택시가 목표 비율(self.target_hotspot_ratio, GUI 슬라이더로 0~1 설정)대로
        얼마나 잘 배분돼있는지" 0~1 점수. 1.0 = 목표 비율과 정확히 일치, 0.0 = 완전히 어긋남.
        유휴택시가 하나도 없으면(워밍업 등) 판단 불가하므로 0.0(가산 없음) 반환.
        """
        total_idle = sum(idle.values())
        if total_idle <= 0:
            return 0.0

        # 핫스팟을 고정 장소가 아니라 해당 시점의 예측 수요 상위 절반으로 정의한다.
        ordered = sorted(ZONE_CATEGORIES, key=lambda c: predicted.get(c, 0.0), reverse=True)
        hotspot_categories = set(ordered[:max(1, len(ordered) // 2)])
        hotspot_idle = sum(idle.get(c, 0) for c in hotspot_categories)
        actual_hotspot_ratio = hotspot_idle / total_idle

        return max(0.0, 1.0 - abs(self.target_hotspot_ratio - actual_hotspot_ratio))

    @staticmethod
    def _riding_passenger_ids() -> set:
        """현재 택시에 탑승 중인 승객 ID 집합을 안전하게 가져온다."""
        riding = set()
        for pid in traci.person.getIDList():
            try:
                if traci.person.getVehicle(pid):
                    riding.add(pid)
            except traci.exceptions.TraCIException:
                continue
        return riding

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

        # [수정] 오차로 인한 음수 발생을 방지하기 위해 관측 범위를 0.0 이상으로 클리핑
        return np.clip(np.array(vec, dtype=np.float32), 0.0, np.inf)