"""
멀티 팩터(taxi_strategy, taxi_dispatch_algorithm 등) 조합을 한 번에 병렬로 비교하는 오케스트레이터.

config_gui.py의 "🧪 A/B 비교하기" 다이얼로그에서 사용자가 고른 조합 목록을 JSON으로 넘기면:
1) 공용 맵을 1회만 생성 (도로망/구역은 taxi_strategy/dispatch_algorithm과 무관하므로 재사용 가능)
2) 조합 개수만큼 폴더를 복제하고, 각 폴더의 runtime_meta.json / simulation.sumocfg에
   그 조합의 값들을 패치
3) 조합 개수만큼 새 콘솔 창을 동시에 띄워 병렬 측정 (parallel_dispatch_worker.py 재사용)
4) 모든 결과 JSON이 도착할 때까지 대기 후 순위표로 비교 출력

사용법:
    python multi_factor_compare.py '[{"taxi_strategy":"patrol","taxi_dispatch_algorithm":"greedy"}, ...]'

주의: 실행 중 config.json이 combos[0] 값으로 덮어써집니다 (기존 parallel_dispatch_orchestrator.py와
동일한 제약). 팀 공용 config.json을 쓰고 있다면 실행 전에 팀에 공유하는 것을 권장합니다.
"""
import os
import sys
import json
import time
import shutil
import subprocess
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

MODULE1 = os.path.join(ROOT, "module1_simulation")
CONFIG_PATH = os.path.join(ROOT, "config.json")

# build_env.py와 동일한 규칙: "hungarian"은 SUMO 내장 알고리즘이 아니라
# traci 콜백으로 직접 배정하는 모드이므로 SUMO 쪽엔 "traci"로 알려줘야 함
DISPATCH_SUMO_VALUE = {"hungarian": "traci"}


def _sumo_dispatch_value(algo: str) -> str:
    return DISPATCH_SUMO_VALUE.get(algo, algo)


def _combo_label(combo: dict) -> str:
    return "_".join(f"{k}={v}" for k, v in combo.items())


def build_shared_map(base_combo: dict):
    """combos[0] 값을 config.json에 반영해서 공용 맵을 1회 생성."""
    print(f"[오케스트레이터] 공용 맵 생성 중... (기준 조합: {base_combo})")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.update(base_combo)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    build_env_path = os.path.join(MODULE1, "build_env.py")
    subprocess.run([sys.executable, build_env_path, "--config-dir", "sumo_config_combo0"],
                    check=True, cwd=ROOT)


def patch_combo(idx: int, combo: dict) -> str:
    """idx==0은 base 폴더를 그대로 패치, 그 외는 base를 복제한 뒤 패치."""
    base_dir = os.path.join(MODULE1, "sumo_config_combo0")
    dst_dir = base_dir if idx == 0 else os.path.join(MODULE1, f"sumo_config_combo{idx}")

    if idx != 0:
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        shutil.copytree(base_dir, dst_dir)

    meta_path = os.path.join(dst_dir, "runtime_meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta.update(combo)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if "taxi_dispatch_algorithm" in combo:
        sumocfg_path = os.path.join(dst_dir, "simulation.sumocfg")
        tree = ET.parse(sumocfg_path)
        root = tree.getroot()
        for elem in root.iter("device.taxi.dispatch-algorithm"):
            elem.set("value", _sumo_dispatch_value(combo["taxi_dispatch_algorithm"]))
        tree.write(sumocfg_path)

    return dst_dir


def launch_workers(combos: list) -> list:
    """조합마다 parallel_dispatch_worker.py를 새 콘솔에서 동시에 실행."""
    worker_path = os.path.join(ROOT, "parallel_dispatch_worker.py")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    result_paths = []
    print(f"[오케스트레이터] 조합 {len(combos)}개를 새 창에서 동시 측정 시작...")
    for idx, combo in enumerate(combos):
        label = _combo_label(combo)
        result_path = f"_result_combo{idx}.json"
        full_result_path = os.path.join(ROOT, result_path)
        if os.path.exists(full_result_path):
            os.remove(full_result_path)
        result_paths.append((label, result_path))

        config_dir_name = f"sumo_config_combo{idx}"
        if os.name == "nt":
            subprocess.Popen(
                ["cmd", "/k", sys.executable, worker_path, config_dir_name, label, result_path],
                cwd=ROOT, env=env, creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:
            subprocess.Popen(
                [sys.executable, worker_path, config_dir_name, label, result_path],
                cwd=ROOT, env=env
            )
    return result_paths


def wait_for_results(result_paths: list):
    print("[오케스트레이터] 모든 결과가 도착할 때까지 대기 중...")
    while True:
        done_flags = [os.path.exists(os.path.join(ROOT, p)) for _, p in result_paths]
        status = " / ".join(
            f"{label}:{'O' if d else 'X'}" for (label, _), d in zip(result_paths, done_flags)
        )
        print(f"  {status}", end="\r")
        if all(done_flags):
            print()
            break
        time.sleep(2)


def compare(result_paths: list):
    from measure_wait_time import print_result

    metrics = {}
    for label, path in result_paths:
        with open(os.path.join(ROOT, path), "r", encoding="utf-8") as f:
            result = json.load(f)
        print_result(label, result)
        metrics[label] = result.get("true_avg_wait_sec_with_penalty")

    print(f"\n===== [최종 비교 - 병렬 실행, 전체 {len(result_paths)}개 조합] =====")
    ranked = sorted(((v, k) for k, v in metrics.items() if v is not None))
    if not ranked:
        print(" - 모든 조합에서 측정 실패 (탑승자 없음). 시간대/승객 설정을 확인하세요.")
        return

    best_v = ranked[0][0]
    for rank, (v, k) in enumerate(ranked, 1):
        diff_pct = (v - best_v) / best_v * 100 if best_v else 0
        marker = " ← 최적" if rank == 1 else f" (+{diff_pct:.1f}%)"
        print(f" {rank}위: {k} — 평균 대기(타임아웃 포함) {v}초{marker}")


def main():
    if len(sys.argv) < 2:
        print("사용법: python multi_factor_compare.py '<json_combos>'")
        sys.exit(1)

    combos = json.loads(sys.argv[1])
    if not combos:
        print("비교할 조합이 없습니다.")
        sys.exit(1)

    build_shared_map(combos[0])
    for idx, combo in enumerate(combos):
        patch_combo(idx, combo)

    result_paths = launch_workers(combos)
    wait_for_results(result_paths)
    compare(result_paths)

    print("\n[오케스트레이터] 완료. 자동 종료하지 않습니다.")
    input("엔터를 누르면 종료합니다...")


if __name__ == "__main__":
    main()