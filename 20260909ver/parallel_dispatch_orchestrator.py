"""
배차 알고리즘 A/B를 실제로 병렬 실행하는 오케스트레이터.
1) algo_a로 맵 1회 생성 (sumo_config_A)
2) sumo_config_A → sumo_config_B 복사, 배차 알고리즘만 algo_b로 패치
3) 새 창 2개를 동시에 띄워 A/B 각각 측정 (진짜 병렬)
4) 두 결과 JSON이 다 나올 때까지 이 창에서 대기
5) 비교 출력, 자동 종료 안 함
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

from measure_wait_time import set_dispatch_algorithm_in_config, print_result

MODULE1 = os.path.join(ROOT, "module1_simulation")
CONFIG_A_DIR = os.path.join(MODULE1, "sumo_config_A")
CONFIG_B_DIR = os.path.join(MODULE1, "sumo_config_B")
RESULT_A = os.path.join(ROOT, "_result_A.json")
RESULT_B = os.path.join(ROOT, "_result_B.json")

DISPATCH_SUMO_VALUE = {"hungarian": "traci"}  # build_env.py와 동일 규칙


def _sumo_dispatch_value(algo: str) -> str:
    return DISPATCH_SUMO_VALUE.get(algo, algo)


def build_shared_map(algo_a: str):
    print(f"[오케스트레이터] 공용 맵 생성 중... (기준: {algo_a})")
    set_dispatch_algorithm_in_config(algo_a)
    build_env_path = os.path.join(MODULE1, "build_env.py")
    subprocess.run([sys.executable, build_env_path, "--config-dir", "sumo_config_A"],
                    check=True, cwd=ROOT)


def clone_for_algo_b(algo_b: str):
    print(f"[오케스트레이터] 맵 복사 중... A → B (배차만 '{algo_b}'로 교체)")
    if os.path.exists(CONFIG_B_DIR):
        shutil.rmtree(CONFIG_B_DIR)
    shutil.copytree(CONFIG_A_DIR, CONFIG_B_DIR)

    meta_path = os.path.join(CONFIG_B_DIR, "runtime_meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["taxi_dispatch_algorithm"] = algo_b
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    sumocfg_path = os.path.join(CONFIG_B_DIR, "simulation.sumocfg")
    tree = ET.parse(sumocfg_path)
    root = tree.getroot()
    for elem in root.iter("device.taxi.dispatch-algorithm"):
        elem.set("value", _sumo_dispatch_value(algo_b))
    tree.write(sumocfg_path)


def launch_workers(algo_a: str, algo_b: str):
    worker_path = os.path.join(ROOT, "parallel_dispatch_worker.py")
    for path in (RESULT_A, RESULT_B):
        if os.path.exists(path):
            os.remove(path)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"; env["PYTHONUTF8"] = "1"

    print(f"[오케스트레이터] 새 창 2개로 '{algo_a}'와 '{algo_b}' 동시 측정 시작...")

    if os.name == "nt":
        subprocess.Popen(["cmd", "/k", sys.executable, worker_path, "sumo_config_A", algo_a, "_result_A.json"],
                          cwd=ROOT, env=env, creationflags=subprocess.CREATE_NEW_CONSOLE)
        subprocess.Popen(["cmd", "/k", sys.executable, worker_path, "sumo_config_B", algo_b, "_result_B.json"],
                          cwd=ROOT, env=env, creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        subprocess.Popen([sys.executable, worker_path, "sumo_config_A", algo_a, "_result_A.json"], cwd=ROOT, env=env)
        subprocess.Popen([sys.executable, worker_path, "sumo_config_B", algo_b, "_result_B.json"], cwd=ROOT, env=env)


def wait_for_results():
    print("[오케스트레이터] 두 결과 모두 도착할 때까지 대기 중...")
    while True:
        a_done, b_done = os.path.exists(RESULT_A), os.path.exists(RESULT_B)
        print(f"  A 완료: {'O' if a_done else 'X'} / B 완료: {'O' if b_done else 'X'}", end="\r")
        if a_done and b_done:
            print(); break
        time.sleep(2)


def compare(algo_a: str, algo_b: str):
    with open(RESULT_A, "r", encoding="utf-8") as f: result_a = json.load(f)
    with open(RESULT_B, "r", encoding="utf-8") as f: result_b = json.load(f)

    print_result(algo_a, result_a)
    print_result(algo_b, result_b)

    print(f"\n===== [최종 비교 - 병렬 실행] {algo_a} vs {algo_b} =====")
    if result_a["avg_wait_sec"] is not None and result_b["avg_wait_sec"] is not None:
        diff = result_a["avg_wait_sec"] - result_b["avg_wait_sec"]
        pct = (diff / result_a["avg_wait_sec"] * 100) if result_a["avg_wait_sec"] else 0
        better = algo_b if diff > 0 else algo_a
        print(f" - {algo_a} 평균 대기: {result_a['avg_wait_sec']}초")
        print(f" - {algo_b} 평균 대기: {result_b['avg_wait_sec']}초")
        print(f" - 차이: {abs(diff):.1f}초 ({abs(pct):.1f}%) — '{better}' 알고리즘이 더 나음")
    else:
        print(" - 둘 중 하나 이상에서 측정 실패.")


def main():
    if len(sys.argv) < 3:
        print("사용법: python parallel_dispatch_orchestrator.py <algo_a> <algo_b>")
        sys.exit(1)
    algo_a, algo_b = sys.argv[1], sys.argv[2]

    build_shared_map(algo_a)
    clone_for_algo_b(algo_b)
    launch_workers(algo_a, algo_b)
    wait_for_results()
    compare(algo_a, algo_b)

    print("\n[오케스트레이터] 완료. 자동 종료하지 않습니다.")
    input("엔터를 누르면 종료합니다...")


if __name__ == "__main__":
    main()