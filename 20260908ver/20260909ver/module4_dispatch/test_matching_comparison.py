"""
헝가리안(최적) 배차 vs 무작위(Random) 배차 비교 스크립트.

dynamic_matching.py는 건드리지 않고, 그 안의 함수들을 그대로 가져다 써서
"개선율이 실제로 몇 %인지" + "매칭 못 받은 승객이 몇 명인지"를 함께 보여줍니다.

사용법:
    python module4_dispatch/test_matching_comparison.py
"""

import os
import sys
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dynamic_matching import DynamicDispatcher, make_mock_taxis_and_passengers


def random_match_distance(taxis: list, passengers: list) -> float:
    """
    무작위로 택시-승객을 1:1로 짝지었을 때의 총 이동거리(비용).
    택시 수와 승객 수 중 작은 쪽 개수만큼만 매칭 (헝가리안과 동일한 조건으로 비교).
    """
    n = min(len(taxis), len(passengers))
    pax_shuffled = random.sample(passengers, n)
    taxis_used = taxis[:n]

    total = 0.0
    for taxi, pax in zip(taxis_used, pax_shuffled):
        dist = np.sqrt((taxi[1] - pax[1]) ** 2 + (taxi[2] - pax[2]) ** 2)
        total += dist
    return total


def run_comparison():
    taxis, passengers = make_mock_taxis_and_passengers()
    n_taxis, n_pax = len(taxis), len(passengers)
    n_matched = min(n_taxis, n_pax)
    n_unmatched_pax = max(0, n_pax - n_taxis)

    print(f"[안내] mock 데이터: 택시 {n_taxis}대, 승객 {n_pax}명")
    if n_unmatched_pax > 0:
        print(f"[안내] 택시가 부족해 승객 {n_unmatched_pax}명은 이번 매칭에서 배차받지 못합니다 "
              f"(헝가리안/랜덤 모두 {n_matched}쌍만 매칭)")

    # 헝가리안 최적 배차
    dispatcher = DynamicDispatcher()
    opt_df = dispatcher.match_vehicles(taxis, passengers)
    opt_total = opt_df['wait_distance'].sum()

    # 무작위 배차 (여러 번 돌려서 평균/최선/최악도 같이 봄 - 1회성 랜덤은 운에 좌우되므로)
    n_trials = 20
    random_totals = [random_match_distance(taxis, passengers) for _ in range(n_trials)]
    rnd_avg = float(np.mean(random_totals))
    rnd_min = float(np.min(random_totals))
    rnd_max = float(np.max(random_totals))

    improvement_vs_avg = (rnd_avg - opt_total) / rnd_avg * 100 if rnd_avg > 0 else 0.0

    print("\n============================================================")
    print("           [헝가리안 vs 랜덤 배차 비교 결과]")
    print("============================================================")
    print(f" - 매칭된 쌍 수: {n_matched}쌍")
    print(f" - 헝가리안(최적) 총 대기거리: {opt_total:.2f}")
    print(f" - 랜덤 배차 총 대기거리 ({n_trials}회 시행): "
          f"평균 {rnd_avg:.2f} / 최선 {rnd_min:.2f} / 최악 {rnd_max:.2f}")
    print(f" - 개선율(평균 랜덤 대비): {improvement_vs_avg:.1f}%")
    print("============================================================\n")

    return {
        "n_taxis": n_taxis,
        "n_passengers": n_pax,
        "n_matched": n_matched,
        "n_unmatched_passengers": n_unmatched_pax,
        "hungarian_total": round(opt_total, 2),
        "random_avg_total": round(rnd_avg, 2),
        "random_min_total": round(rnd_min, 2),
        "random_max_total": round(rnd_max, 2),
        "improvement_pct_vs_avg": round(improvement_vs_avg, 1),
    }


if __name__ == "__main__":
    run_comparison()
