"""
학교/회사/음식점 카테고리의 승객을 시뮬레이션 도중 실시간으로 생성/소멸시키는 매니저.

[배경] build_env.py는 이제 주택(residential)만 정적으로 미리 생성하고,
학교/회사/음식점은 이 매니저가 매 스텝 traci로 직접 만들고 없앱니다.
왜냐하면 "회사가 저녁에 몇 명을 뱉을지"가 "그날 아침 회사가 몇 명을 흡수했는지"에
달려있는데, 이건 시뮬레이션이 실제로 돌아가봐야 알 수 있는 값이기 때문입니다
(build_env.py 실행 시점=시뮬레이션 시작 전에는 알 수 없음).

[핵심 개념]
1. 카테고리별 흡수 카운터: 사람이 해당 카테고리 건물에 정상 도착(택시에서 하차)하면
   그 카테고리의 absorbed 카운터가 1 증가.
2. 흡수량 기반 재스폰: 예) 회사가 아침에 500명을 흡수했으면, 저녁 배출 시점에
   그 500이라는 숫자(의 일부, 모달 필터 적용)를 그대로 다시 뱉어냄.
3. 시간대별 택시확률 곡선: 등교/출근은 피크 시각에 가까울수록 택시 선택 확률이
   선형으로 증가하다가, 정해진 시각 이후로는 완전히 끊김(컷오프).
4. 음식점 개인별 타이머: 음식점에 도착한 사람은(점심 왕복 제외) 1시간 뒤 그 사람만
   개별적으로 재스폰되어 100% 주택으로 이동.

[도착(흡수) 판정 방법]
traci.person.getIDList()를 매 스텝 비교해서, "이전 스텝엔 있었는데 이번 스텝엔 사라진"
사람 중, passenger_manager(PassengerTimeoutManager)가 타임아웃으로 제거한 게 아니면
"정상 도착"으로 판정합니다. (타임아웃 제거는 흡수로 치지 않음 — 실제 그 건물에 못
들어간 것이므로.)

[사용법] run_simulation.py / measure_wait_time.py의 시뮬레이션 루프 안에서:
    pax_manager = PassengerTimeoutManager(...)
    spawn_manager = PassengerSpawnManager(zones=..., sim_start_hour=..., ...)
    ...
    while step < max_steps:
        traci.simulationStep()
        ...
        pax_manager.maintain(now)
        spawn_manager.maintain(now, timeout_removed_pids=pax_manager.removed_pids)
"""

import random
import traci
from config_loader import CFG


class PassengerSpawnManager:
    def __init__(self, zones: dict, sim_start_hour: float, sim_end_hour: float = None,
                 school_pop_base: int = None, company_pop_base: int = None,
                 seed: int = None):
        if seed is not None:
            random.seed(seed)

        self.sim_start_hour = sim_start_hour
        self.school_pop_base = int(CFG.get("school_pop_base", 400) if school_pop_base is None else school_pop_base)
        self.company_pop_base = int(CFG.get("company_pop_base", 100) if company_pop_base is None else company_pop_base)
        self.school_start_hour = float(CFG.get("school_start_hour", 7.0))
        self.school_end_hour = float(CFG.get("school_end_hour", 8.0))
        self.school_taxi_peak_hour = float(CFG.get("school_taxi_peak_hour", self.school_end_hour))
        self.school_afternoon_start_hour = float(CFG.get("school_afternoon_start_hour", 16.0))
        self.school_afternoon_end_hour = float(CFG.get("school_afternoon_end_hour", 17.0))
        self.school_afternoon_fraction = float(CFG.get("school_afternoon_fraction", 0.01))
        self.company_start_hour = float(CFG.get("company_start_hour", 8.0))
        self.company_end_hour = float(CFG.get("company_end_hour", 10.0))
        self.company_taxi_peak_hour = float(CFG.get("company_taxi_peak_hour", self.company_end_hour))
        self.lunch_start_hour = float(CFG.get("lunch_start_hour", 12.0))
        self.lunch_end_hour = float(CFG.get("lunch_end_hour", 13.0))
        self.lunch_company_release_fraction = float(CFG.get("lunch_company_release_fraction", 0.80))
        self.lunch_taxi_fraction = float(CFG.get("lunch_taxi_fraction", 0.10))
        self.evening_start_hour = float(CFG.get("evening_start_hour", 18.0))
        self.evening_end_hour = float(CFG.get("evening_end_hour", 22.0))
        self.evening_taxi_fraction = float(CFG.get("evening_taxi_fraction", 0.30))
        self.late_evening_start_hour = float(CFG.get("late_evening_start_hour", 23.0))
        self.late_evening_end_hour = float(CFG.get("late_evening_end_hour", 24.0))
        self.evening_residential_fraction = float(CFG.get("evening_residential_fraction", 25/30))
        self.restaurant_stay_sec = float(CFG.get("restaurant_stay_sec", 3600.0))

        # -------------------------------------------------------------
        # 기본 승객 수(num_passengers) 기반 독립 가챠 생성
        # 기존 시간대별(school/company/residential 스케줄) 로직과는 완전히 별개로,
        # sim 구간을 num_passengers로 나눈 간격마다 1틱씩 체크하며
        # residential_taxi_probability 확률로 "택시 승객"만 생성한다.
        # 확률에 당첨되지 않으면 그 틱은 그냥 소멸(생성 자체를 안 함).
        # -------------------------------------------------------------
        self.num_passengers = int(CFG.get("num_passengers", 1000))
        self.residential_taxi_probability = float(CFG.get("residential_taxi_probability", 0.10))
        self.sim_end_hour = sim_end_hour
        if self.sim_end_hour is not None and self.num_passengers > 0:
            duration_sec = max(1.0, (self.sim_end_hour - self.sim_start_hour) * 3600.0)
            self._gacha_interval_sec = max(1, round(duration_sec / self.num_passengers))
        else:
            self._gacha_interval_sec = None
        self._gacha_last_tick = -1  # 같은 초에 중복 실행 방지

        self.transit_edges = zones.get("subway_entrance", []) + zones.get("bus_stop", [])
        self.school_edges = zones.get("school", [])
        self.company_edges = zones.get("company", [])
        self.restaurant_edges = zones.get("restaurant", [])
        self.residential_edges = zones.get("residential", [])

        all_edges = [e for edges in zones.values() for e in edges if edges]
        self.all_edges = all_edges
        self.transit_edges = self.transit_edges or all_edges
        self.school_edges = self.school_edges or all_edges
        self.company_edges = self.company_edges or all_edges
        self.restaurant_edges = self.restaurant_edges or all_edges
        self.residential_edges = self.residential_edges or all_edges

        # 카테고리별 흡수(도착) 카운터
        self.absorbed = {"school": 0, "company": 0, "restaurant": 0}

        # 도착 판정을 위한 이전 스텝 활성 승객 집합
        self._prev_active_ids = set()
        # 이 매니저가 만든 승객의 목적지 카테고리 기록 (도착 시 어느 카운터를 올릴지 판단용)
        self._dest_category = {}
        # 점심 왕복으로 생성된 승객은 restaurant 개인 타이머 대상에서 제외
        self._is_lunch_cycle = set()

        self._counter = 0
        self.total_timeout = 0
        self.total_spawned = 0
        # 각 이벤트가 "한 번만" 실행되도록 하는 트리거 플래그/스냅샷
        self._lunch_12_done = False
        self._lunch_13_done = False
        self._lunch_release_taxi_count = 0
        self._evening_pool_snapshot = None  # 18시 시점 company 흡수량 스냅샷
        self._evening_spawned_count = 0
        self._evening_dump_done = False
        self._school_afternoon_pool_snapshot = None
        self._school_afternoon_spawned_count = 0

        # 음식점 개인별 재스폰 대기열: [(트리거시각, 출발edge), ...]
        self._restaurant_pending = []

    # ---------- 내부 유틸 ----------
    def _current_hour(self, now_seconds: float) -> float:
        return self.sim_start_hour + (now_seconds / 3600.0)

    def _new_pid(self) -> str:
        self._counter += 1
        return f"dyn_pax_{self._counter}"

    def _spawn_person(self, now_seconds: float, start_edge: str, end_edge: str,
                       dest_category: str = None, lunch_cycle: bool = False):
        pid = self._new_pid()
        try:
            traci.person.add(pid, start_edge, pos=0, depart=now_seconds)
            traci.person.appendDrivingStage(pid, end_edge, lines="taxi")
        except traci.exceptions.TraCIException:
            return None
        self.total_spawned += 1
        if dest_category:
            self._dest_category[pid] = dest_category
        if lunch_cycle:
            self._is_lunch_cycle.add(pid)
        return pid

    def _pick_edge(self, pool: list, exclude: str = None) -> str:
        candidate = random.choice(pool)
        if candidate == exclude and len(pool) > 1:
            candidate = random.choice(pool)
        return candidate

    # ---------- 도착(흡수) 감지 ----------
    def _process_arrivals(self, timeout_removed_pids: set, now_seconds: float):
        current_ids = set(traci.person.getIDList())
        disappeared = self._prev_active_ids - current_ids
        self._prev_active_ids = current_ids

        for pid in disappeared:
            if pid not in self._dest_category:
                continue  # 이 매니저가 만든 게 아닌 사람(주택발 정적 승객 등)은 흡수 집계 안 함
            category = self._dest_category.pop(pid)
            was_lunch = pid in self._is_lunch_cycle
            self._is_lunch_cycle.discard(pid)

            if pid in timeout_removed_pids:
                self.total_timeout += 1
                continue  # 타임아웃으로 못 태운 것 — 흡수(도착)로 치지 않음

            # 정상 도착 처리
            self.absorbed[category] = self.absorbed.get(category, 0) + 1

            # 음식점 도착이고 점심 왕복이 아니면 -> 1시간 뒤 개인 재스폰 예약
            if category == "restaurant" and not was_lunch:
                self._restaurant_pending.append((now_seconds + self.restaurant_stay_sec, None))

    # ---------- 아침 (7~10시): 학교 등교(7~8시, 9시 컷오프) + 회사 출근(8~10시) ----------
    def _maintain_morning(self, now_seconds: float, now_hour: float):
        # 학교 등교: 7~8시, 8시에 가까울수록 택시 확률 선형 증가, 9시 이후 완전 중단
        if self.school_start_hour <= now_hour < self.school_end_hour:
            rate = self.school_pop_base / 3600.0  # 초당 시도 횟수(윈도우 1시간 기준)
            if random.random() < rate:
                peak = min(max(self.school_taxi_peak_hour, self.school_start_hour), self.school_end_hour)
                taxi_prob = 0.0 if now_hour <= self.school_start_hour else min(1.0, (now_hour - self.school_start_hour) / max(peak - self.school_start_hour, 1e-9))
                if random.random() < taxi_prob:
                    start_e = self._pick_edge(self.transit_edges)
                    end_e = self._pick_edge(self.school_edges)
                    self._spawn_person(now_seconds, start_e, end_e, dest_category="school")
        # (9시 이후는 자연히 이 if문 범위 밖이라 스폰 없음 — 별도 컷오프 처리 불필요)

        # 회사 출근: 8~10시, 10시에 가까울수록 택시 확률 선형 증가
        if self.company_start_hour <= now_hour < self.company_end_hour:
            rate = self.company_pop_base / max((self.company_end_hour - self.company_start_hour) * 3600.0, 1.0)  # 윈도우 2시간
            if random.random() < rate:
                peak = min(max(self.company_taxi_peak_hour, self.company_start_hour), self.company_end_hour)
                taxi_prob = 0.0 if now_hour <= self.company_start_hour else min(1.0, (now_hour - self.company_start_hour) / max(peak - self.company_start_hour, 1e-9))
                if random.random() < taxi_prob:
                    start_e = self._pick_edge(self.transit_edges)
                    end_e = self._pick_edge(self.company_edges)
                    self._spawn_person(now_seconds, start_e, end_e, dest_category="company")

    # ---------- 점심 (12~13시): 회사 80% 흡수 -> 음식점(10%택시) -> 13시 복귀(10%택시) ----------
    def _maintain_lunch(self, now_seconds: float, now_hour: float):
        if not self._lunch_12_done and now_hour >= self.lunch_start_hour:
            self._lunch_12_done = True
            release_total = int(self.absorbed.get("company", 0) * self.lunch_company_release_fraction)
            self._lunch_release_taxi_count = int(release_total * self.lunch_taxi_fraction)
            for _ in range(self._lunch_release_taxi_count):
                start_e = self._pick_edge(self.company_edges)
                end_e = self._pick_edge(self.restaurant_edges)
                self._spawn_person(now_seconds, start_e, end_e,
                                    dest_category="restaurant", lunch_cycle=True)

        if not self._lunch_13_done and now_hour >= self.lunch_end_hour:
            self._lunch_13_done = True
            for _ in range(self._lunch_release_taxi_count):
                start_e = self._pick_edge(self.restaurant_edges)
                end_e = self._pick_edge(self.company_edges)
                # 13시 복귀는 다시 company로 도착하므로 흡수 카운터가 자연 누적됨(=재흡수)
                self._spawn_person(now_seconds, start_e, end_e,
                                    dest_category="company", lunch_cycle=True)

    # ---------- 학교 하교 (16~17시만, 컷오프) ----------
    def _maintain_school_afternoon(self, now_seconds: float, now_hour: float):
        if self._school_afternoon_pool_snapshot is None and now_hour >= self.school_afternoon_start_hour:
            self._school_afternoon_pool_snapshot = int(self.absorbed.get("school", 0) * self.school_afternoon_fraction)

        if self._school_afternoon_pool_snapshot and self.school_afternoon_start_hour <= now_hour < self.school_afternoon_end_hour:
            remaining = self._school_afternoon_pool_snapshot - self._school_afternoon_spawned_count
            if remaining > 0:
                rate = self._school_afternoon_pool_snapshot / 3600.0
                if random.random() < rate:
                    start_e = self._pick_edge(self.school_edges)
                    end_e = self._pick_edge(self.residential_edges)
                    self._spawn_person(now_seconds, start_e, end_e, dest_category="residential")
                    self._school_afternoon_spawned_count += 1
        # 17시 이후는 범위 밖이라 자연 중단 (하드 컷오프)

    # ---------- 회사 퇴근 (18~22시 30% 필터 / 23~24시 100%) ----------
    def _maintain_evening(self, now_seconds: float, now_hour: float):
        if self._evening_pool_snapshot is None and now_hour >= self.evening_start_hour:
            self._evening_pool_snapshot = self.absorbed.get("company", 0)

        if self._evening_pool_snapshot is None:
            return

        if self.evening_start_hour <= now_hour < self.evening_end_hour:
            rate = self._evening_pool_snapshot / max((self.evening_end_hour - self.evening_start_hour) * 3600.0, 1.0)
            if random.random() < rate:
                if random.random() < self.evening_taxi_fraction:  # 버스35%+지하철35% 제외 -> 30%만 택시
                    dest_pool = self.residential_edges if random.random() < self.evening_residential_fraction else self.restaurant_edges
                    start_e = self._pick_edge(self.company_edges)
                    end_e = self._pick_edge(dest_pool)
                    dest_cat = "residential" if dest_pool is self.residential_edges else "restaurant"
                    self._spawn_person(now_seconds, start_e, end_e, dest_category=dest_cat)
                self._evening_spawned_count += 1  # 필터 탈락도 "시도"로 카운트(모수 소진)

        elif self.late_evening_start_hour <= now_hour < self.late_evening_end_hour and not self._evening_dump_done:
            self._evening_dump_done = True
            remaining = max(0, self._evening_pool_snapshot - self._evening_spawned_count)
            for _ in range(remaining):
                dest_pool = self.residential_edges if random.random() < self.evening_residential_fraction else self.restaurant_edges
                start_e = self._pick_edge(self.company_edges)
                end_e = self._pick_edge(dest_pool)
                dest_cat = "residential" if dest_pool is self.residential_edges else "restaurant"
                self._spawn_person(now_seconds, start_e, end_e, dest_category=dest_cat)

    # ---------- 음식점 개인별 1시간 타이머 ----------
    def _maintain_restaurant_timers(self, now_seconds: float):
        still_pending = []
        for trigger_time, _ in self._restaurant_pending:
            if now_seconds >= trigger_time:
                start_e = self._pick_edge(self.restaurant_edges)
                end_e = self._pick_edge(self.residential_edges)
                self._spawn_person(now_seconds, start_e, end_e, dest_category="residential")
            else:
                still_pending.append((trigger_time, None))
        self._restaurant_pending = still_pending

    # ---------- 기본 승객 수(num_passengers) 독립 가챠 ----------
    def _maintain_residential_gacha(self, now_seconds: float):
        if self._gacha_interval_sec is None:
            return
        tick = int(now_seconds) // self._gacha_interval_sec
        if tick == self._gacha_last_tick:
            return  # 같은 간격 안에서 이미 처리함
        self._gacha_last_tick = tick

        if random.random() < self.residential_taxi_probability:
            start_e = self._pick_edge(self.all_edges)
            end_e = self._pick_edge(self.all_edges, exclude=start_e)
            self._spawn_person(now_seconds, start_e, end_e, dest_category=None)
        # 미당첨(90%)은 그냥 아무것도 생성하지 않고 소멸

    # ---------- 매 스텝 호출 ----------
    def maintain(self, now_seconds: float, timeout_removed_pids: set = None):
        timeout_removed_pids = timeout_removed_pids or set()
        now_hour = self._current_hour(now_seconds) % 24

        self._process_arrivals(timeout_removed_pids, now_seconds)
        self._maintain_morning(now_seconds, now_hour)
        self._maintain_lunch(now_seconds, now_hour)
        self._maintain_school_afternoon(now_seconds, now_hour)
        self._maintain_evening(now_seconds, now_hour)
        self._maintain_restaurant_timers(now_seconds)
        self._maintain_residential_gacha(now_seconds)