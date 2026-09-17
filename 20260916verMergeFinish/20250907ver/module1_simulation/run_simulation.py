import os
import sys
import json
import time
import traci

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taxi_manager import TaxiFleetManager
from passenger_manager import PassengerTimeoutManager
from passenger_spawn_manager import PassengerSpawnManager


def run_sumo_gui():
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)

    sumo_binary = "sumo-gui"
    sumo_cfg = "module1_simulation/sumo_config/simulation.sumocfg"
    meta_path = "module1_simulation/sumo_config/runtime_meta.json"

    print("SUMO Digital Twin 시뮬레이션을 시작합니다...")

    # build_env.py가 저장해둔 edges/boundary_edges/전략 정보 로드 (택시 대수 유지에 필요)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    sim_start_hour = meta.get("sim_start_hour", 0)

    manager = TaxiFleetManager(
        target_count=meta["num_taxis"],
        boundary_edges=meta["boundary_edges"],
        all_edges=meta["edges"],
        strategy=meta["taxi_strategy"],
        hotspot_edges=meta.get("hotspot_edges"),
        zones=meta.get("zones"),
        sim_start_hour=sim_start_hour,
        remaining_edges_threshold=meta.get("taxi_remaining_edges_threshold"),
    )

    # 배차 못 받고 너무 오래 대기한 승객을 소멸시키는 매니저 (없으면 끝까지 길가에 쌓임)
    pax_manager = PassengerTimeoutManager(
        wait_timeout_sec=meta.get("passenger_wait_timeout", 900)
    )

    sim_end_hour = meta.get("sim_end_hour")
    if sim_end_hour is None:
        # runtime_meta.json에 sim_end_hour가 없는 경우(구버전 build_env.py로 생성됐거나
        # 저장 누락) config.json에서 직접 읽어와 폴백. 이게 없으면 max_steps=9999까지
        # 그냥 다 돌아버려서 설정한 시간대가 무시되는 문제가 있었음.
        try:
            with open(os.path.join(ROOT_DIR := os.path.dirname(meta_path).rsplit(
                    "module1_simulation", 1)[0], "config.json"), "r", encoding="utf-8") as f:
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

    # sim_end_seconds도 못 구했으면(둘 다 없음) 무한정 도는 걸 막기 위한 최후 안전장치로
    # 하루(24시) 분량으로 상한을 둠 (기존 max_steps=9999 하드코딩이 실질적 상한 역할을 하던 문제 방지)
    if sim_end_seconds is None:
        print("[경고] sim_end_hour를 어디서도 찾지 못해 안전상 24시간 분량으로 자동 종료합니다.")
        sim_end_seconds = 24 * 3600

    try:
        traci.start([sumo_binary, "-c", sumo_cfg])

        step = 0
        max_steps = int(sim_end_seconds) + 100  # 종료시각보다 넉넉히 큰 안전 상한 (실제 종료는 아래 sim_end_seconds 체크가 담당)
        

        while step < max_steps:
            traci.simulationStep()
            now_seconds = traci.simulation.getTime()
            manager.maintain(now_seconds)  # 도착 임박한 택시에 새 목적지를 얹어 소멸을 막음 (대수 유지)
            pax_manager.maintain(now_seconds)  # 대기시간 초과 승객 소멸 처리
            spawn_manager.maintain(now_seconds, timeout_removed_pids=pax_manager.removed_pids)  # 학교/회사/음식점 실시간 생성·소멸
            if step % 100 == 0:
                sc = spawn_manager.spawn_counts
                print(
                        f"[DEBUG] time={now_seconds:.0f}s | "
                        f"vehicles={len(traci.vehicle.getIDList())} | "
                        f"persons={len(traci.person.getIDList())}"
                        f" | 누적 생성={spawn_manager.total_spawned} "
                        f"(학교아침={sc['school_morning']} 회사아침={sc['company_morning']} "
                        f"점심출={sc['lunch_release']} 점심복귀={sc['lunch_return']} "
                        f"학교하교={sc['school_afternoon']} 퇴근={sc['evening']} "
                        f"주거가챠={sc['residential_gacha']} 음식점시민={sc['restaurant_civilian']}) "
                        f"| 누적 타임아웃={spawn_manager.total_timeout}"
    )
            time.sleep(0.01)

            # SUMO 창 타이틀바에 실제 시각(시:분:초) 출력
            current_hour_float = sim_start_hour + (now_seconds / 3600.0)
            h = int(current_hour_float) % 24
            m = int((current_hour_float - int(current_hour_float)) * 60)
            s = int((((current_hour_float - int(current_hour_float)) * 60) - m) * 60)
            time_str = f"{h:02d}:{m:02d}:{s:02d}"

            try:
                view_ids = traci.gui.getViewIDList()
                if view_ids:
                    traci.gui.setWindowCaption(view_ids[0], f"DT Mobility Simulation | Time: {time_str} (Elapsed: {int(now_seconds)}s)")
            except Exception:
                pass

            # 설정한 시간대(sim_end_hour)에 도달하면 승객이 남아있어도 강제 종료.
            if sim_end_seconds is not None and now_seconds >= sim_end_seconds:
                print(f" -> [안내] 설정한 시간대({sim_end_hour}시)에 도달하여 시뮬레이션을 종료합니다. (Step: {step})")
                break



            step += 1

        traci.close()
        print("시뮬레이션이 정상적으로 종료되었습니다.")

    except Exception as e:
        print(f"시뮬레이션 실행 중 오류 발생: {e}")


if __name__ == "__main__":
    run_sumo_gui()