"""
평균 승객 대기시간 측정 + A/B 비교 스크립트.

측정 방법:
- traci로 시뮬레이션을 스텝마다 진행하면서, 각 person(승객)이
  "등장(depart) 시각"과 "택시에 탑승한 시각"을 기록.
- 탑승 시각은 traci.person.getVehicle(person_id)가 빈 문자열("")에서
  택시 id로 바뀌는 순간으로 판정 (그 전까지는 길가에서 대기 중).
- 대기시간 = 탑승 시각 - depart 시각.
- 시뮬레이션 종료(모든 차량/사람 소진) 후 평균/최대 대기시간을 출력.

사용법:
    python measure_wait_time.py patrol         # config.json의 sim 설정 그대로, 택시 전략만 patrol로 강제
    python measure_wait_time.py prepositioned  # 택시 전략만 prepositioned로 강제
    python measure_wait_time.py compare        # 두 전략 각각 build_env.py부터 다시 돌려서 순차 비교

주의: build_env.py가 만든 entities.rou.xml에 이미 반영된 전략을 그대로 재생하려면
      인자 없이 실행하면 됩니다 (config.json의 taxi_strategy 값 사용).
"""

import os
import sys
import json
import time
import traci

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taxi_manager import TaxiFleetManager
from passenger_manager import PassengerTimeoutManager
from passenger_spawn_manager import PassengerSpawnManager

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT, "config.json")
SUMO_CFG = os.path.join(ROOT, "module1_simulation", "sumo_config", "simulation.sumocfg")
META_PATH = os.path.join(ROOT, "module1_simulation", "sumo_config", "runtime_meta.json")


# measure_wait_time.py 내부
# max_steps 기본값을 24시간 전체 스텝(86400초 이상)을 충분히 커버할 수 있도록 확장
def run_and_measure(sumo_binary: str = "sumo", max_steps: int = 100000) -> dict:
    """
    시뮬레이션을 headless(sumo, GUI 없음)로 끝까지 돌리면서 승객별 대기시간을 측정.
    택시는 taxi_manager.TaxiFleetManager가 항상 target_count(=num_taxis)만큼 유지함
    (도착해서 사라진 택시는 즉시 길 끝에서 재스폰).
    반환: {"avg_wait": float, "max_wait": float, "n_measured": int, "n_unpicked": int, "waits": [...]}
    """
    depart_time = {}     # person_id -> depart 시각(초)
    pickup_time = {}     # person_id -> 탑승 확인된 시각(초)
    in_taxi = set()      # 이미 탑승 처리된 person_id (중복 계산 방지)
    seen_persons = set()

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    manager = TaxiFleetManager(
        target_count=meta["num_taxis"],
        boundary_edges=meta["boundary_edges"],
        all_edges=meta["edges"],
        strategy=meta["taxi_strategy"],
        hotspot_edges=meta.get("hotspot_edges"),
        zones=meta.get("zones"),
        sim_start_hour=meta.get("sim_start_hour", 0),
    )

    # 배차 못 받고 너무 오래 대기한 승객을 소멸시키는 매니저 (없으면 끝까지 길가에 쌓임)
    pax_manager = PassengerTimeoutManager(
        wait_timeout_sec=meta.get("passenger_wait_timeout", 900)
    )

    sim_end_hour = meta.get("sim_end_hour")
    sim_start_hour = meta.get("sim_start_hour", 0)
    if sim_end_hour is None:
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                fallback_cfg = json.load(f)
            sim_end_hour = fallback_cfg.get("sim_end_hour")
            if sim_end_hour is not None:
                print(f"[안내] runtime_meta.json에 sim_end_hour가 없어 config.json에서 읽어왔습니다 ({sim_end_hour}시).")
        except Exception:
            sim_end_hour = None

    # 학교/회사/음식점 승객을 실시간으로 생성·소멸(흡수→재스폰)시키는 매니저
    spawn_manager = PassengerSpawnManager(
        zones=meta.get("zones", {}),
        sim_start_hour=sim_start_hour,
        sim_end_hour=sim_end_hour,
        school_pop_base=meta.get("school_pop_base", 400),
        company_pop_base=meta.get("company_pop_base", 100),
        seed=meta.get("passenger_seed"),
    )

    sim_end_seconds = (sim_end_hour - sim_start_hour) * 3600 if sim_end_hour is not None else None
    if sim_end_seconds is None:
        print("[경고] sim_end_hour를 어디서도 찾지 못해 안전상 24시간 분량으로 자동 종료합니다.")
        sim_end_seconds = 24 * 3600
    max_steps = max(max_steps, int(sim_end_seconds) + 100)

    traci.start([sumo_binary, "-c", SUMO_CFG])

    step = 0
    empty_count = 0
    try:
        while step < max_steps:
            traci.simulationStep()
            now = traci.simulation.getTime()
            manager.maintain(now)  # 도착 임박한 택시에 새 목적지를 얹어 소멸을 막음 (대수 유지)
            pax_manager.maintain(now)  # 대기시간 초과 승객 소멸 처리
            spawn_manager.maintain(now, timeout_removed_pids=pax_manager.removed_pids)  # 학교/회사/음식점 실시간 생성·소멸

            # 이번 스텝에 새로 등장한 person 기록
            for pid in traci.person.getIDList():
                if pid not in seen_persons:
                    seen_persons.add(pid)
                    depart_time[pid] = now

                if pid not in in_taxi:
                    vid = traci.person.getVehicle(pid) if pid in traci.person.getIDList() else ""
                    if vid:  # 빈 문자열이 아니면 = 어떤 택시에 탑승함
                        pickup_time[pid] = now
                        in_taxi.add(pid)

            # 설정한 시간대(sim_end_hour)에 도달하면 승객이 남아있어도 강제 종료.
            # (승객이 0명 될 때까지 무한정 기다리던 예전 문제 수정 — sim_end_hour가 실제로 적용됨)
            if sim_end_seconds is not None and now >= sim_end_seconds:
                break

            active_persons = traci.person.getIDList()
            if step > 10 and len(active_persons) == 0:
                empty_count += 1
                if empty_count >= 5:
                    break
            else:
                empty_count = 0
            step += 1
    finally:
        traci.close()

    waits = []
    for pid, dtime in depart_time.items():
        if pid in pickup_time:
            waits.append(pickup_time[pid] - dtime)

    n_total = len(depart_time)
    n_measured = len(waits)
    n_timeout_removed = len(pax_manager.removed_pids)  # 대기시간 초과로 소멸 처리된 승객
    n_unpicked = n_total - n_measured  # 끝까지 못 탄 승객 (타임아웃 소멸 + 시뮬레이션 종료시 잔류 포함)

    result = {
        "n_total_passengers": n_total,
        "n_measured": n_measured,
        "n_unpicked": n_unpicked,
        "n_timeout_removed": n_timeout_removed,
        "avg_wait_sec": round(sum(waits) / len(waits), 1) if waits else None,
        "max_wait_sec": round(max(waits), 1) if waits else None,
        "min_wait_sec": round(min(waits), 1) if waits else None,
        "waits": waits,
    }
    return result


def set_strategy_in_config(strategy: str):
    """config.json의 taxi_strategy 값을 바꿔치기 (build_env.py가 재실행될 때 반영됨)"""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["taxi_strategy"] = strategy
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def rebuild_env():
    """build_env.py를 다시 실행해서 현재 config.json 기준으로 도로망/승객/택시를 재생성"""
    import subprocess
    build_env_path = os.path.join(ROOT, "module1_simulation", "build_env.py")
    subprocess.run([sys.executable, build_env_path], check=True, cwd=ROOT)


def print_result(label: str, result: dict):
    print(f"\n===== [{label}] 결과 =====")
    print(f" - 전체 승객 수: {result['n_total_passengers']}")
    print(f" - 탑승 성공: {result['n_measured']}명 / 끝까지 못 탄 승객: {result['n_unpicked']}명 "
          f"(그중 대기시간 초과로 소멸: {result.get('n_timeout_removed', 0)}명)")
    if result["avg_wait_sec"] is not None:
        print(f" - 평균 대기시간: {result['avg_wait_sec']}초 (약 {result['avg_wait_sec']/60:.1f}분)")
        print(f" - 최소/최대 대기시간: {result['min_wait_sec']}초 / {result['max_wait_sec']}초")
    else:
        print(" - 탑승한 승객이 없어 대기시간을 계산할 수 없습니다.")


def compare_strategies():
    """patrol vs prepositioned 두 전략을 순차 실행해서 비교"""
    results = {}
    for strategy in ("patrol", "prepositioned"):
        print(f"\n########## 전략 '{strategy}' 시뮬레이션 준비 중 ##########")
        set_strategy_in_config(strategy)
        rebuild_env()
        result = run_and_measure()
        results[strategy] = result
        print_result(strategy, result)

    print("\n===== [최종 비교] patrol vs prepositioned =====")
    a, b = results["patrol"], results["prepositioned"]
    if a["avg_wait_sec"] is not None and b["avg_wait_sec"] is not None:
        diff = a["avg_wait_sec"] - b["avg_wait_sec"]
        pct = (diff / a["avg_wait_sec"] * 100) if a["avg_wait_sec"] else 0
        better = "prepositioned" if diff > 0 else "patrol"
        print(f" - patrol 평균 대기: {a['avg_wait_sec']}초")
        print(f" - prepositioned 평균 대기: {b['avg_wait_sec']}초")
        print(f" - 차이: {abs(diff):.1f}초 ({abs(pct):.1f}%) — '{better}' 전략이 더 나음")
    else:
        print(" - 두 전략 중 하나 이상에서 측정 실패 (탑승자 없음). 승객/시간대 설정을 확인하세요.")

    return results


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "current"

    if mode == "compare":
        compare_strategies()
    elif mode in ("patrol", "prepositioned"):
        set_strategy_in_config(mode)
        rebuild_env()
        result = run_and_measure()
        print_result(mode, result)
    else:
        # config.json에 이미 설정된 전략 그대로, 재생성 없이 현재 rou.xml/sumocfg로 측정만
        result = run_and_measure()
        print_result("current config", result)