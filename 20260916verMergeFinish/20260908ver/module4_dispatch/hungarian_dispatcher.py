"""
SUMO taxi device의 dispatch-algorithm="traci" 모드에서 실제 배정을 수행하는
헝가리안(최적 할당, scipy linear_sum_assignment) 기반 실시간 배차 매니저.

[왜 필요한가]
기존 dynamic_matching.py의 헝가리안 로직은 mock (id, x, y) 데이터로만 동작해서
시뮬레이션과 연결돼 있지 않았음. 이 매니저는 traci로 "실제" 대기 중인 예약(승객)과
"실제" 빈 택시 좌표를 가져와 같은 헝가리안 알고리즘으로 배정한다 — SUMO 안에서 진짜로 돌아감.

[동작 원리]
SUMO device.taxi.dispatch-algorithm이 "traci"면, 승객 예약이 생겨도 SUMO가
자동으로 택시를 배정하지 않고 traci.vehicle.dispatchTaxi()가 불려주기를 기다린다.
이 매니저가 매 스텝:
  1) traci.person.getTaxiReservations()로 아직 배정 안 된 예약 목록을 가져오고
  2) traci.vehicle.getTaxiFleet(0)으로 완전히 빈(empty) 택시 목록을 가져와
  3) 택시-예약 간 유클리드 거리 비용행렬을 만들어 linear_sum_assignment로 최적 조합을 구하고
  4) traci.vehicle.dispatchTaxi(vehID, [reservationID])로 실제 배정한다.

[사용법] measure_wait_time.py / run_simulation.py의 시뮬레이션 루프 안에서:
    dispatcher = HungarianDispatcher() if meta.get("taxi_dispatch_algorithm") == "hungarian" else None
    ...
    while step < max_steps:
        traci.simulationStep()
        now = traci.simulation.getTime()
        ...
        if dispatcher:
            dispatcher.maintain(now)
"""

import traci
import numpy as np
from scipy.optimize import linear_sum_assignment


class HungarianDispatcher:
    def __init__(self):
        # 이미 배정을 "시도"한 예약 id를 기록해서 같은 예약에 매 스텝 중복 dispatchTaxi 호출하는 것 방지.
        # (SUMO가 배정 후에도 한동안 getTaxiReservations에 남겨둘 수 있어 안전장치로 둠)
        self.dispatched_reservations = set()
        # dispatchTaxi 실패(연속) 시 그 예약을 포기하고 다음 스텝에 다시 시도할 수 있게
        # 별도로 걸러내진 않음 — SUMO가 상태를 관리하므로 실패해도 다음 스텝에 재시도됨.

    def _get_pending_reservations(self):
        """아직 배정 시도하지 않은 예약만 반환."""
        try:
            reservations = traci.person.getTaxiReservations(0)
        except traci.exceptions.TraCIException:
            return []
        return [r for r in reservations if r.id not in self.dispatched_reservations]

    def _get_idle_taxis(self):
        """현재 손님도 없고 픽업 중도 아닌 완전히 빈 택시 목록. (0 = TAXI_EMPTY)"""
        try:
            return list(traci.vehicle.getTaxiFleet(0))
        except traci.exceptions.TraCIException:
            return []

    def _edge_start_position(self, edge_id: str):
        """edge의 시작 좌표를 그 예약(승객)의 근사 위치로 사용."""
        try:
            shape = traci.lane.getShape(edge_id + "_0")
            return shape[0]
        except (traci.exceptions.TraCIException, IndexError):
            return (0.0, 0.0)

    def maintain(self, now_seconds: float = 0):
        try:
            reservations = self._get_pending_reservations()
            idle_taxis = self._get_idle_taxis()

            # 후보가 아예 없을 때 왜 안 잡히는지 확인하기 위한 임시 프린트
            # if not reservations or not idle_taxis:
            #     return

            if not reservations or not idle_taxis:
                return

            taxi_positions = []
            for vid in idle_taxis:
                try:
                    taxi_positions.append(traci.vehicle.getPosition(vid))
                except traci.exceptions.TraCIException:
                    taxi_positions.append((0.0, 0.0))

            pax_positions = [self._edge_start_position(r.fromEdge) for r in reservations]

            n_taxi, n_pax = len(idle_taxis), len(reservations)
            cost = np.zeros((n_taxi, n_pax))
            for i, tp in enumerate(taxi_positions):
                for j, pp in enumerate(pax_positions):
                    cost[i, j] = np.hypot(tp[0] - pp[0], tp[1] - pp[1])

            taxi_idx, pax_idx = linear_sum_assignment(cost)

            for t_i, p_i in zip(taxi_idx, pax_idx):
                vid = idle_taxis[t_i]
                res = reservations[p_i]
                try:
                    traci.vehicle.dispatchTaxi(vid, [res.id])
                    self.dispatched_reservations.add(res.id)
                    # 👈 배정 성공할 때마다 콘솔에 찍히는 디버그 로그 추가
                    #print(f"[Hungarian] {vid} -> 예약 {res.id} 배정 (누적 {len(self.dispatched_reservations)}건)")
                except traci.exceptions.TraCIException:
                    continue
        except traci.exceptions.TraCIException:
            pass