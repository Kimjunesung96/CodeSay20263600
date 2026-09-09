"""
병렬 배차 비교용 워커. 지정된 sumo_config 폴더에서 headless로 측정하고
결과를 JSON으로 저장. 완전히 별도 프로세스(새 콘솔)라 traci 소켓 충돌 없음.

사용법: python parallel_dispatch_worker.py <config_dir_name> <algo_label> <output_json_path>
"""
import os
import sys
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from measure_wait_time import run_and_measure, print_result


def main():
    if len(sys.argv) < 4:
        print("사용법: python parallel_dispatch_worker.py <config_dir_name> <algo_label> <output_json_path>")
        sys.exit(1)

    config_dir_name, algo_label, output_path = sys.argv[1], sys.argv[2], sys.argv[3]

    sumo_cfg_path = os.path.join(ROOT, "module1_simulation", config_dir_name, "simulation.sumocfg")
    meta_path = os.path.join(ROOT, "module1_simulation", config_dir_name, "runtime_meta.json")

    print(f"[워커] '{algo_label}' ({config_dir_name}) 측정 시작 (headless)...")
    result = run_and_measure(sumo_cfg_path=sumo_cfg_path, meta_path=meta_path)
    print_result(algo_label, result)

    with open(os.path.join(ROOT, output_path), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n[워커] '{algo_label}' 완료 — {output_path}에 저장했습니다.")
    print("[워커] 이 창은 자동으로 닫히지 않습니다.")


if __name__ == "__main__":
    main()