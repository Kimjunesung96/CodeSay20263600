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
    python measure_wait_time.py dispatch_compare [algo_a] [algo_b]  # 배차 알고리즘 A vs B 비교

주의: build_env.py가 만든 entities.rou.xml에 이미 반영된 전략을 그대로 재생하려면
      인자 없이 실행하면 됩니다 (config.json의 taxi_strategy 값 사용).

[헝가리안 배차 연동]
config.json의 taxi_dispatch_algorithm이 "hungarian"이면, build_env.py가 SUMO 쪽엔
"traci"로 알려주고 실제 선택값은 runtime_meta.json에 남겨둠. 이 스크립트는 그 값을 읽어
HungarianDispatcher를 매 스텝 maintain()해서 실제 배차를 수행함 (SUMO 내장 greedy 대신).

[학습용 수요 로그 (신규)]
run_and_measure()가 호출될 때마다(단독 실행 / parallel_dispatch_worker.py /
multi_factor_compare.py의 병렬 워커 전부 포함, run_simulation.py의 GUI 실행은 미포함),
새로 등장하는 승객의 "등장 시각 + 위경도"를 data/raw/simulation_demand_log.csv에
append합니다. train.py의 prepare_real_sequence_dataset()이 이 CSV를 읽어 실제
시뮬레이션 기반 수요로 학습하도록 연결되어 있습니다.

날짜는 실행마다 과거 90일 내에서 랜덤 배정되어, 여러 번 돌릴수록 dayofweek/is_weekend
같은 캘린더 피처에 실제 분산이 생기도록 했습니다.
"""

import os
import sys
import json
import time
import datetime
import random
import csv
import hashlib
import traci

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taxi_manager import TaxiFleetManager
from passenger_manager import PassengerTimeoutManager
from passenger_spawn_manager import PassengerSpawnManager
from module4_dispatch.hungarian_dispatcher import HungarianDispatcher
from module4_dispatch.rl_reposition_maintainer import RLRepositionMaintainer  # 추가
from module2_preprocessing.sim_log_recorder import DemandLogRecorder  # [Module 2] 호출 로그 기록 (feat3, 병합)

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT, "config.json")
SUMO_CFG = os.path.join(ROOT, "module1_simulation", "sumo_config", "simulation.sumocfg")
META_PATH = os.path.join(ROOT, "module1_simulation", "sumo_config", "runtime_meta.json")
DEMAND_LOG_PATH = os.path.join(ROOT, "data", "raw", "simulation_demand_log.csv")
RL_MODEL_PATH = os.path.join(ROOT, "module4_dispatch", "rl_reposition_model.zip")  # 추가
# RLRepositionMaintainer가 RL 정책과 "실행 시점"에 짝지어 쓸 수요예측 파일.
# 기본값은 기존 카테고리 모델이며, AI_CATEGORY_MODEL_PATH 환경변수로 다른 pkl
# (예: saved_models/category_demand_from_v2.pkl)을 골라 바로 조합해 쓸 수 있다.
# rl_reposition_model.zip 자체는 그대로, 여기서 붙이는 수요예측만 갈아끼우는 방식(재학습 불필요).
CATEGORY_MODEL_PATH = os.environ.get(
    "AI_CATEGORY_MODEL_PATH",
    os.path.join(ROOT, "saved_models", "category_demand_models.pkl"),
)



# measure_wait_time.py 내부
# max_steps 기본값을 24시간 전체 스텝(86400초 이상)을 충분히 커버할 수 있도록 확장
def run_and_measure(sumo_binary: str = "sumo", max_steps: int = 100000,
                     sumo_cfg_path: str = None, meta_path: str = None,
                     strategy: str = None, algorithm: str = None, recorder=None) -> dict:
    """
    시뮬레이션을 headless(sumo, GUI 없음)로 끝까지 돌리면서 승객별 대기시간을 측정.
    택시는 taxi_manager.TaxiFleetManager가 항상 target_count(=num_taxis)만큼 유지함
    (도착해서 사라진 택시는 즉시 길 끝에서 재스폰).
    반환: {"avg_wait": float, "max_wait": float, "n_measured": int, "n_unpicked": int, "waits": [...]}

    strategy/algorithm: 지정하면 config.json에 반영하고 build_env.py를 다시 돌린 뒤 시작함
        (export_unity_replay.py처럼 "이 조합으로 새로 지도까지 다시 만들어서 측정해줘"가 필요한
        호출부용. 안 넘기면 예전처럼 "지금 config.json/meta 그대로" 바로 측정함 - 기존 호출부
        전부 그대로 동작).
    recorder: module4_dispatch나 외부(export_unity_replay.py의 ReplayRecorder 등)에서 만든
        기록기. recorder.capture(now, requested, pickup, arrived, removed, failed, forecast,
        teleports)를 매 스텝 호출하고, 끝나면 recorder.finish(result)를 호출함. 안 넘기면
        (기본값) 기록 없이 예전과 완전히 동일하게 동작 - 성능/결과에 영향 없음.
    """
    if strategy is not None:
        set_strategy_in_config(strategy)
    if algorithm is not None:
        set_dispatch_algorithm_in_config(algorithm)
    if strategy is not None or algorithm is not None:
        rebuild_env()

    depart_time = {}     # person_id -> depart 시각(초)
    pickup_time = {}     # person_id -> 탑승 확인된 시각(초)
    arrived_pids = set()  # 목적지까지 도착 완료(정상 하차)한 person_id
    in_taxi = set()      # 이미 탑승 처리된 person_id (중복 계산 방지)
    seen_persons = set()
    teleport_count = 0

    _meta_path = meta_path or META_PATH
    _sumo_cfg = sumo_cfg_path or SUMO_CFG
    with open(_meta_path, "r", encoding="utf-8") as f:
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

    # taxi_dispatch_algorithm이 "hungarian"이거나 taxi_strategy가 "rl_prepositioned"일 때
    # 매칭은 항상 HungarianDispatcher가 담당함 (RL은 매칭 위에 재배치만 얹는 구조이므로)
    taxi_strategy = meta.get("taxi_strategy")
    dispatch_algo = meta.get("taxi_dispatch_algorithm")
    hungarian_dispatcher = (
        HungarianDispatcher() if dispatch_algo == "hungarian" or taxi_strategy == "rl_prepositioned" else None
    )

    # 배차 못 받고 너무 오래 대기한 승객을 소멸시키는 매니저 (없으면 끝까지 길가에 쌓임)
    # dispatcher를 넘겨주면, 이미 Hungarian이 예약을 배정한 승객의 강제제거를 한 스텝
    # 미뤄서 SUMO 내부 상태 충돌("Connection closed by SUMO")을 방지함 (passenger_manager.py 참고)
    pax_manager = PassengerTimeoutManager(
        wait_timeout_sec=meta.get("passenger_wait_timeout", 900),
        dispatcher=hungarian_dispatcher,
    )

    # taxi_strategy가 rl_prepositioned일 때만 RL 재배치 매니저를 추가로 켬. 학습된 모델 파일이 없으면
    # 경고만 찍고 None으로 둬서(=재배치 없이 hungarian만 도는 상태) 워커가 조용히 죽지 않게 함
    rl_maintainer = None
    if taxi_strategy == "rl_prepositioned":
        if os.path.exists(RL_MODEL_PATH):
            rl_maintainer = RLRepositionMaintainer(RL_MODEL_PATH, meta, category_model_path=CATEGORY_MODEL_PATH)
            print(f"[안내] RL 재배치 정책을 불러왔습니다: {RL_MODEL_PATH}")
            print(f"[안내] 수요예측은 이걸 씁니다: {CATEGORY_MODEL_PATH}")
        else:
            print(f"[경고] {RL_MODEL_PATH}가 없어 RL 재배치 없이 Hungarian 매칭만 수행합니다. "
                  f"먼저 module4_dispatch/rl_train.py를 실행하세요.")


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

    # ---- 학습용 수요 로그 준비 ----
    demand_log_rows = []  # (pickup_datetime_iso, latitude, longitude) 누적 버퍼
    # 실행마다 날짜를 과거 90일 내에서 랜덤 배정 -> 여러 번 돌릴수록 요일 다양성 확보
    run_date = datetime.date.today() - datetime.timedelta(days=random.randint(0, 90))

    # [Module 2] train.py(pipeline.py 경유)가 읽는 data/sim_logs/demand_log_*.csv 기록기.
    # 위 demand_log_rows(구 data/raw/simulation_demand_log.csv)와는 별개 경로 — train.py는
    # 이제 이쪽만 읽으므로 병합 후에는 이 recorder가 실질적인 학습 로그임.
    # [2026-09-22 날짜 충돌 수정] sim_date를 고정값(기본 2026-09-07)으로 두면 실행마다
    # 전부 같은 달력 날짜로 기록돼서, aggregate_demands()의 groupby(['time_bucket','h3_index'])가
    # 서로 다른 실행(run)의 수요를 같은 시간칸으로 합쳐버림 -> 실행 횟수만큼 수요가 부풀려짐.
    # 위에서 이미 계산해둔 run_date(레거시 로그용 랜덤 과거 날짜, 실행마다 다름)를 그대로 넘겨서
    # 실행마다 서로 다른 달력 날짜를 쓰게 함 -> 시간칸 충돌 방지 + dayofweek/is_weekend 분산 확보.
    log_recorder = DemandLogRecorder(
        sim_start_hour=sim_start_hour, zones=meta.get("zones", {}), sim_date=run_date.isoformat()
    )

    traci.start([sumo_binary, "-c", _sumo_cfg])

    step = 0
    empty_count = 0
    try:
        while step < max_steps:
            traci.simulationStep()
            now = traci.simulation.getTime()
            manager.maintain(now)  # 도착 임박한 택시에 새 목적지를 얹어 소멸을 막음 (대수 유지)
            pax_manager.maintain(now)  # 대기시간 초과 승객 소멸 처리
            if hungarian_dispatcher:
                hungarian_dispatcher.maintain(now)  # 헝가리안 실시간 배차 (taxi_dispatch_algorithm="hungarian"일 때만)
            if rl_maintainer:
                rl_maintainer.maintain(now, sim_start_hour=sim_start_hour, sim_end_hour=sim_end_hour)  # 추가

            spawn_manager.maintain(now, timeout_removed_pids=pax_manager.removed_pids)  # 학교/회사/음식점 실시간 생성·소멸
            log_recorder.step(now, timeout_removed_pids=pax_manager.removed_pids)  # [Module 2] 호출 로그 기록

            # 진행 표시: 시뮬레이션 10분마다 한 줄 (suppress_sumo_warnings=true면 이것만 보임)
            if int(now) % 600 == 0 and now > 0:
                _hh = int(sim_start_hour + now / 3600); _mm = int((now % 3600) // 60)
                print(f"[진행] 시뮬 {_hh:02d}:{_mm:02d} ({now/sim_end_seconds*100:.0f}%) | "
                      f"호출 누적 {len(log_recorder.records)}건 | 탑승 {len(pickup_time)}명 | "
                      f"타임아웃 {len(pax_manager.removed_pids)}명 | 택시 {len(traci.vehicle.getIDList())}대", flush=True)

            # 이번 스텝에 새로 등장한 person 기록
            for pid in traci.person.getIDList():
                if pid not in seen_persons:
                    seen_persons.add(pid)
                    depart_time[pid] = now

                    # 학습 데이터용: 등장 시점의 위치를 위경도로 변환해 기록
                    try:
                        x, y = traci.person.getPosition(pid)
                        lon, lat = traci.simulation.convertGeo(x, y)
                        current_hour = sim_start_hour + (now / 3600.0)
                        pickup_dt = datetime.datetime.combine(
                            run_date, datetime.time(0, 0)
                        ) + datetime.timedelta(hours=current_hour)
                        demand_log_rows.append((pickup_dt.isoformat(), lat, lon))
                    except Exception:
                        pass

                if pid not in in_taxi:
                    vid = traci.person.getVehicle(pid) if pid in traci.person.getIDList() else ""
                    if vid:  # 빈 문자열이 아니면 = 어떤 택시에 탑승함
                        pickup_time[pid] = now
                        in_taxi.add(pid)

            # 탑승했던 사람이 이번 스텝에 더 이상 안 보이면(=목적지 도착해서 SUMO가 자동
            # 제거함) "정상 도착"으로 기록. 타임아웃으로 소멸한 사람과 구분하기 위함
            # (recorder.capture()의 arrived 인자 및 최종 검증용, 없어도 측정 결과엔 영향 없음)
            if recorder is not None:
                active_now = set(traci.person.getIDList())
                for pid in in_taxi:
                    if pid not in active_now and pid not in arrived_pids and pid not in pax_manager.removed_pids:
                        arrived_pids.add(pid)

                teleport_count += traci.simulation.getStartingTeleportNumber()
                recorder.capture(
                    now, depart_time, pickup_time, arrived_pids,
                    pax_manager.removed_pids, set(), None, teleport_count,
                )

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
    except traci.exceptions.FatalTraCIError as e:
        # SUMO 프로세스 자체가 죽어서 연결이 끊긴 경우(예: 이미 예약된 승객을
        # 타임아웃으로 강제 제거하려다 SUMO 내부 상태가 깨지는 등). 이 시점부터는
        # 이 시뮬레이션을 더 진행할 수 없으므로, 지저분한 스택트레이스로 죽는 대신
        # 여기서 멈추고 지금까지 쌓인 결과(호출 로그, 대기시간 등)를 그대로 살려서
        # 아래 결과 계산/저장으로 넘어간다 - 7시간 중 83%까지 온 게 통째로 날아가는
        # 것보다, 그 지점까지의 부분 결과라도 남기는 게 훨씬 낫다.
        print(f"[경고] 시뮬레이션이 중간에 끊겼습니다 (SUMO 연결 끊김, step={step}, "
              f"마지막 정상 시각={now:.0f}초): {e}")
        print("[경고] 지금까지 쌓인 부분 결과로 계속 진행합니다 (완전한 결과가 아님에 주의).")
    finally:
        try:
            traci.close()
        except Exception:
            pass

    waits = []
    for pid, dtime in depart_time.items():
        if pid in pickup_time:
            waits.append(pickup_time[pid] - dtime)

    n_total = len(depart_time)
    n_measured = len(waits)
    n_timeout_removed = len(pax_manager.removed_pids)  # 대기시간 초과로 소멸 처리된 승객
    n_unpicked = n_total - n_measured  # 끝까지 못 탄 승객

    # [수정] 타임아웃된 승객에게 최대 대기시간 페널티를 부여하여 왜곡 방지
    max_penalty_sec = meta.get("passenger_wait_timeout", 500)
    adjusted_waits = list(waits)
    for _ in range(n_timeout_removed):
        adjusted_waits.append(max_penalty_sec)

    # ---- 학습용 수요 로그 저장 (for 루프 밖, 시뮬레이션 1회당 한 번만) ----
    if demand_log_rows:
        os.makedirs(os.path.dirname(DEMAND_LOG_PATH), exist_ok=True)
        file_exists = os.path.exists(DEMAND_LOG_PATH)
        with open(DEMAND_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["pickup_datetime", "latitude", "longitude"])
            writer.writerows(demand_log_rows)
        print(f"[안내] 학습용 수요 로그 {len(demand_log_rows)}건을 {DEMAND_LOG_PATH}에 남겼습니다.")

    if log_recorder.records:
        saved_path = log_recorder.save()
        print(f"[안내] Module 2 호출 로그 {len(log_recorder.records)}건을 {saved_path}에 저장했습니다 "
              f"(train.py가 읽는 실제 로그입니다).")

    result = {
        "strategy": strategy,
        "algorithm": algorithm,
        "duration_sec": now,  # 마지막으로 관측된 시뮬레이션 시각(초) = 실제 돈 구간 길이
        "completed_interval": sim_end_seconds is not None and now >= sim_end_seconds,
        # 같은 승객 수요(등장 시각+좌표)로 실행됐는지 확인하는 지문. patrol vs forecast처럼
        # 서로 다른 배차 방식을 "같은 손님들"로 공정 비교했는지 검증할 때 씀
        # (spawn_manager의 seed가 config.json의 passenger_seed로 고정돼 있어야 두 실행의
        # 지문이 일치함 - 안 고정돼 있으면 매번 달라지는 게 정상이니 비교 목적이 아니면 무시해도 됨)
        "demand_fingerprint": hashlib.sha256(
            ",".join(f"{pid}:{t:.1f}" for pid, t in sorted(depart_time.items())).encode()
        ).hexdigest()[:16],
        "n_total_passengers": n_total,
        "n_measured": n_measured,
        "n_unpicked": n_unpicked,
        "n_timeout_removed": n_timeout_removed,
        "avg_wait_sec": round(sum(waits) / len(waits), 1) if waits else None,
        "true_avg_wait_sec_with_penalty": round(sum(adjusted_waits) / len(adjusted_waits), 1) if adjusted_waits else None,
        "max_wait_sec": round(max(waits), 1) if waits else None,
        "min_wait_sec": round(min(waits), 1) if waits else None,
        "waits": waits,
    }

    if recorder is not None:
        recorder.finish(result)

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
        print(f" - 평균 대기시간(탑승자만): {result['avg_wait_sec']}초 (약 {result['avg_wait_sec']/60:.1f}분)")
        print(f" - 최소/최대 대기시간: {result['min_wait_sec']}초 / {result['max_wait_sec']}초")
    if result.get("true_avg_wait_sec_with_penalty") is not None:
        v = result["true_avg_wait_sec_with_penalty"]
        print(f" - 평균 대기시간(타임아웃 페널티 포함, 체감치): {v}초 (약 {v/60:.1f}분)")
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
    # 타임아웃 페널티 포함 지표로 비교 -- 배차 실패(타임아웃)가 많은 쪽이
    # 픽업만 기준으로는 오히려 좋게 보이는 착시를 막기 위함
    a_metric, b_metric = a.get("true_avg_wait_sec_with_penalty"), b.get("true_avg_wait_sec_with_penalty")
    if a_metric is not None and b_metric is not None:
        diff = a_metric - b_metric
        pct = (diff / a_metric * 100) if a_metric else 0
        better = "prepositioned" if diff > 0 else "patrol"
        print(f" - patrol 평균 대기(타임아웃 포함): {a_metric}초")
        print(f" - prepositioned 평균 대기(타임아웃 포함): {b_metric}초")
        print(f" - 차이: {abs(diff):.1f}초 ({abs(pct):.1f}%) — '{better}' 전략이 더 나음")
    else:
        print(" - 두 전략 중 하나 이상에서 측정 실패 (탑승자 없음). 승객/시간대 설정을 확인하세요.")

    return results


def set_dispatch_algorithm_in_config(algo: str):
    """config.json의 taxi_dispatch_algorithm 값을 바꿔치기"""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["taxi_dispatch_algorithm"] = algo
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def compare_dispatch_algorithms(algo_a: str = "greedy", algo_b: str = "hungarian"):
    """두 배차 알고리즘(A vs B)을 순차 실행해서 평균 대기시간 비교"""
    results = {}
    for algo in (algo_a, algo_b):
        print(f"\n########## 배차 알고리즘 '{algo}' 시뮬레이션 준비 중 ##########")
        set_dispatch_algorithm_in_config(algo)
        rebuild_env()
        result = run_and_measure()
        results[algo] = result
        print_result(algo, result)

    print(f"\n===== [최종 비교] {algo_a} vs {algo_b} =====")
    a, b = results[algo_a], results[algo_b]
    # 타임아웃 페널티 포함 지표로 비교 -- 아까 실제 테스트(greedy 56.3% 개선처럼 보였다가
    # 타임아웃 포함하면 14.5%로 줄어든 것)에서 확인된 착시를 여기서도 막기 위함
    a_metric, b_metric = a.get("true_avg_wait_sec_with_penalty"), b.get("true_avg_wait_sec_with_penalty")
    if a_metric is not None and b_metric is not None:
        diff = a_metric - b_metric
        pct = (diff / a_metric * 100) if a_metric else 0
        better = algo_b if diff > 0 else algo_a
        print(f" - {algo_a} 평균 대기(타임아웃 포함): {a_metric}초")
        print(f" - {algo_b} 평균 대기(타임아웃 포함): {b_metric}초")
        print(f" - 차이: {abs(diff):.1f}초 ({abs(pct):.1f}%) — '{better}' 알고리즘이 더 나음")
    else:
        print(" - 둘 중 하나 이상에서 측정 실패. 시간대/승객 설정을 확인하세요.")

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
    elif mode == "dispatch_compare":
        algo_a = sys.argv[2] if len(sys.argv) > 2 else "greedy"
        algo_b = sys.argv[3] if len(sys.argv) > 3 else "hungarian"
        compare_dispatch_algorithms(algo_a, algo_b)
    else:
        # config.json에 이미 설정된 전략 그대로, 재생성 없이 현재 rou.xml/sumocfg로 측정만
        result = run_and_measure()
        print_result("current config", result)