"""
배차를 못 받고 너무 오래 대기한 승객을 시뮬레이션에서 소멸(제거)시키는 매니저.

왜 필요한가:
- build_env.py는 스케줄대로 승객을 '생성'만 함.
- taxi_manager.TaxiFleetManager는 택시 '대수'만 유지함 (손님 없는 택시 재타겟팅).
- 즉, 어느 쪽도 "너무 오래 기다린 승객을 치우는" 로직을 갖고 있지 않았음.
- 그 결과 배차 못 받은 승객이 sim_end_hour까지 계속 길가에 쌓여서
  n_unpicked 수가 부풀려지고, 실제로는 "영원히 대기 중"인 상태가 됨.

동작:
- 매 스텝 traci.person.getIDList()를 보고, 아직 택시에 타지 않은(person.getVehicle == "")
  사람 중 depart 이후 wait_timeout(초)을 넘긴 사람을 traci.person.remove()로 제거.
- 제거된 사람은 removed_pids에 기록되어 measure_wait_time.py 등에서
  "타임아웃으로 소멸(집계에서 unpicked로 카운트)" 구분에 사용 가능.
"""

import traci


class PassengerTimeoutManager:
    def __init__(self, wait_timeout_sec: float):
        self.wait_timeout_sec = wait_timeout_sec
        self.depart_time = {}      # person_id -> depart 시각(초)
        self.removed_pids = set()  # 타임아웃으로 소멸 처리된 person_id

    def maintain(self, now_seconds: float):
        for pid in traci.person.getIDList():
            if pid not in self.depart_time:
                self.depart_time[pid] = now_seconds
                continue

            if pid in self.removed_pids:
                continue

            # 이미 택시에 탄 사람은 건드리지 않음
            try:
                vid = traci.person.getVehicle(pid)
            except traci.exceptions.TraCIException:
                continue
            if vid:
                continue

            waited = now_seconds - self.depart_time[pid]
            if waited >= self.wait_timeout_sec:
                try:
                    traci.person.remove(pid)
                except traci.exceptions.TraCIException:
                    pass
                self.removed_pids.add(pid)