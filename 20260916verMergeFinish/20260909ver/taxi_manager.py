import random
import traci

from config_loader import CFG

CATEGORY_PEAK_HOURS = {
    "company": [CFG.get("company_start_hour", 8.0), CFG.get("company_end_hour", 10.0), CFG.get("lunch_start_hour", 12.0), CFG.get("evening_start_hour", 18.0), CFG.get("late_evening_start_hour", 23.0)],
    "school": [CFG.get("school_start_hour", 7.0), CFG.get("school_end_hour", 8.0), CFG.get("school_afternoon_start_hour", 16.0)],
    "residential": [CFG.get("school_start_hour", 7.0), CFG.get("evening_start_hour", 18.0), CFG.get("late_evening_start_hour", 23.0)],
    "restaurant": [CFG.get("lunch_start_hour", 12.0), CFG.get("evening_start_hour", 18.0)],
    "subway_entrance": [CFG.get("school_start_hour", 7.0), CFG.get("company_start_hour", 8.0), CFG.get("evening_start_hour", 18.0)],
    "bus_stop": [CFG.get("school_start_hour", 7.0), CFG.get("company_start_hour", 8.0), CFG.get("evening_start_hour", 18.0)],
}


class TaxiFleetManager:
    def __init__(self, target_count: int, boundary_edges: list, all_edges: list,
                 vtype: str = "taxi_type", strategy: str = "patrol",
                 hotspot_edges: list = None, zones: dict = None,
                 sim_start_hour: float = 0, remaining_edges_threshold: int = None, **kwargs):
        self.target_count = target_count
        self.boundary_edges = boundary_edges or all_edges  # 재생성 시 사용할 검증된 출발점
        self.all_edges = all_edges
        self.vtype = vtype
        self.strategy = strategy
        self.hotspot_edges = hotspot_edges if hotspot_edges else all_edges
        self.zones = zones or {}
        self.sim_start_hour = sim_start_hour
        self.remaining_edges_threshold = (CFG.get("taxi_remaining_edges_threshold", 2)
                                          if remaining_edges_threshold is None else remaining_edges_threshold)
        self.primary_prob = float(CFG.get("taxi_primary_prob", 0.70))
        self.fail_threshold = int(CFG.get("taxi_fail_threshold", 5))
        self.target_pick_attempts = int(CFG.get("taxi_target_pick_attempts", 10))
        self.respawn_prefix = str(CFG.get("taxi_respawn_prefix", "taxi_respawn"))

        self.fail_counts = {}      # vid -> 연속 목적지 탐색 실패 횟수
        self._respawn_counter = 0  # 재생성된 택시에 부여할 고유 id 카운터

    def _current_hour(self, now_seconds: float) -> float:
        return self.sim_start_hour + (now_seconds / 3600.0)

    def _category_busyness_ranking(self, now_seconds: float) -> list:
        now_hour = self._current_hour(now_seconds) % 24
        scored = []
        for category, peak_hours in CATEGORY_PEAK_HOURS.items():
            min_diff = min(abs((now_hour - p + 12) % 24 - 12) for p in peak_hours)
            score = 3.0 - min_diff
            scored.append((category, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [cat for cat, _ in scored]

    def _current_hotspot_pools(self, now_seconds: float):
        if not self.zones:
            return self.hotspot_edges, self.hotspot_edges

        ranking = self._category_busyness_ranking(now_seconds)
        primary_pool, secondary_pool = [], []

        idx = 0
        while idx < len(ranking) and not primary_pool:
            primary_pool = self.zones.get(ranking[idx], [])
            idx += 1
        while idx < len(ranking) and not secondary_pool:
            secondary_pool = self.zones.get(ranking[idx], [])
            idx += 1

        if not primary_pool:
            primary_pool = self.hotspot_edges or self.all_edges
        if not secondary_pool:
            secondary_pool = primary_pool

        return primary_pool, secondary_pool

    def _pick_target_edge(self, now_seconds: float, current_edge: str) -> str:
        if self.strategy == "prepositioned":
            primary_pool, secondary_pool = self._current_hotspot_pools(now_seconds)
            pool = primary_pool if random.random() < self.primary_prob else secondary_pool
        else:
            pool = self.all_edges

        for _ in range(self.target_pick_attempts):
            candidate = random.choice(pool)
            if candidate == current_edge:
                continue
            # 후보 edge 자체가 내부(intersection) edge면 경로탐색 대상이 아니므로 스킵
            if candidate.startswith(":"):
                continue
            try:
                route = traci.simulation.findRoute(current_edge, candidate)
                if route and len(route.edges) > 0:
                    return candidate
            except Exception:
                continue
        return None  # 적절한 후보를 못 찾으면 이번 스텝엔 재타겟팅 포기 (제자리 유지)

    def _respawn_taxi(self, old_vid: str):
        """
        고립된 edge에 갇힌 택시를 소멸시키고, 검증된 출발점(boundary_edges)에서
        새 택시로 재생성한다. 갇혔던 그 자리에서 다시 살리면 또 갇힐 수 있으므로
        반드시 boundary_edges(길 끝)에서 새로 스폰한다.

        taxi device는 여기서 setParameter로 나중에 붙이지 않는다 — SUMO는 삽입
        이후 시점에 device를 붙이는 걸 지원하지 않아 에러가 난다. 대신
        build_env.py에서 taxi_type vType 정의 자체에 <param has.taxi.device=true>를
        박아둬서, 이 타입으로 삽입되는 모든 차량(정적/동적 무관)이 자동으로
        taxi device를 달고 태어나게 되어 있다.
        """
        try:
            traci.vehicle.remove(old_vid)
        except traci.exceptions.TraCIException:
            pass
        self.fail_counts.pop(old_vid, None)
        self._spawn_new_taxi()

    def _spawn_new_taxi(self):
        """boundary_edges(검증된 길 끝)에서 새 택시 한 대를 생성한다."""
        self._respawn_counter += 1
        new_vid = f"{self.respawn_prefix}_{self._respawn_counter}"
        route_id = f"route_{new_vid}"
        start_edge = random.choice(self.boundary_edges)

        try:
            traci.route.add(route_id, [start_edge])
            traci.vehicle.add(new_vid, route_id, typeID=self.vtype)
        except traci.exceptions.TraCIException:
            pass  # 이번 시도가 실패해도 다음 maintain() 사이클에서 대수 부족이 다시 감지되어 재시도됨

    def maintain(self, now_seconds: float = 0):
        try:
            current_ids = set(traci.vehicle.getIDList())
            # 더 이상 존재하지 않는 차량의 실패 카운터는 정리 (메모리 누수 방지)
            self.fail_counts = {vid: c for vid, c in self.fail_counts.items() if vid in current_ids}

            # ------------------------------------------------------------
            # 대수 보충(top-up) 안전장치 — "왜 줄었는지"를 하나하나 원인별로 다 잡는 대신,
            # 매 스텝마다 지금 살아있는 택시 대수를 세서 target_count보다 모자라면
            # 그 차이만큼 무조건 boundary_edges에서 새로 채워 넣는다.
            # 고립 edge(fail_counts 5회), 정지위치 배정 실패로 인한 자연 도착 처리,
            # teleport 등 원인이 무엇이든 결과(대수 유지)만은 항상 보장하기 위함.
            # ------------------------------------------------------------
            taxi_count = sum(
                1 for vid in current_ids if traci.vehicle.getTypeID(vid) == self.vtype
            )
            shortage = self.target_count - taxi_count
            for _ in range(max(0, shortage)):
                self._spawn_new_taxi()

            for vid in current_ids:
                if traci.vehicle.getTypeID(vid) != self.vtype:
                    continue

                if traci.vehicle.getPersonIDList(vid):
                    continue  # 손님 태우고 이동 중 — 건드리지 않음

                current_edge = traci.vehicle.getRoadID(vid)
                # 교차로 내부(internal) edge에 걸쳐있는 순간엔 경로탐색이 불안정하므로 건드리지 않고
                # 다음 스텝(정상 edge로 넘어간 뒤)까지 대기
                if current_edge.startswith(":"):
                    continue

                route = traci.vehicle.getRoute(vid)
                route_idx = traci.vehicle.getRouteIndex(vid)
                remaining = len(route) - route_idx - 1

                if remaining > self.remaining_edges_threshold:
                    continue

                new_target = self._pick_target_edge(now_seconds, current_edge)

                if not new_target or new_target == current_edge:
                    # 목적지 탐색 실패 — 연속 실패 카운트 누적
                    self.fail_counts[vid] = self.fail_counts.get(vid, 0) + 1
                    if self.fail_counts[vid] >= self.fail_threshold:
                        # 5번 연속 실패 = 고립된 edge로 판정, 소멸 후 재생성
                        self._respawn_taxi(vid)
                    continue

                # 목적지를 찾았으면 실패 카운트 리셋
                self.fail_counts[vid] = 0

                try:
                    # 손님 없이 대기 중인 택시는 SUMO taxi device가 자동으로 "idling 정지"를
                    # 걸어둔 상태일 수 있음. 그 상태에서 바로 changeTarget을 하면 기존 정지
                    # 스케줄과 충돌해서 "could not assign stop" 에러가 나고 결국 택시가 오도가도
                    # 못 하다 소멸함. resume()으로 idling 정지를 먼저 풀어준 뒤 목적지를 바꿈.
                    #
                    # 예전엔 getStops(vid, 1)의 "예정된 정지가 있는지"만 보고 무조건 resume()을
                    # 불렀는데, 이건 "정지가 예약돼 있음"과 "지금 실제로 정지해 있음"을 혼동한
                    # 것이었음. 아직 정지 지점에 도달 전인 택시에도 resume()을 걸어서
                    # "Failed to resume from stopping" 에러가 대량으로 찍히던 원인.
                    # isStopped()로 "지금 실제로 멈춰있는지"를 확인한 뒤에만 resume() 호출.
                    if traci.vehicle.isStopped(vid):
                        traci.vehicle.resume(vid)
                except traci.exceptions.TraCIException:
                    pass  # 정지 상태가 아니었으면 resume 호출이 실패할 수 있음 — 무시하고 진행

                try:
                    traci.vehicle.changeTarget(vid, new_target)
                except traci.exceptions.TraCIException:
                    continue  # 이번 시도가 실패해도 다음 스텝에 다시 시도됨
        except traci.exceptions.TraCIException:
            pass