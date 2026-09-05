import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

class DynamicDispatcher:
    """
    승객과 가용 택시 간의 대기시간(거리)을 최소화하는 동적 배차 최적화 클래스
    """
    def __init__(self):
        pass

    def calculate_distance_matrix(self, taxis: list, passengers: list) -> np.ndarray:
        """
        택시와 승객 간의 거리 행렬 계산 (실제 프로젝트에서는 NetworkX 기반 최단거리 사용)
        taxis: [(taxi_id, x, y), ...], passengers: [(pax_id, x, y), ...]
        """
        dist_matrix = np.zeros((len(taxis), len(passengers)))
        for i, taxi in enumerate(taxis):
            for j, pax in enumerate(passengers):
                # 유클리디안 거리 기반 대기 비용 계산
                dist_matrix[i, j] = np.sqrt((taxi[1] - pax[1])**2 + (taxi[2] - pax[2])**2)
        return dist_matrix

    def match_vehicles(self, taxis: list, passengers: list) -> pd.DataFrame:
        print("대기시간 최소화를 위한 최적 매칭(Hungarian Algorithm) 수행 중...")
        dist_matrix = self.calculate_distance_matrix(taxis, passengers)
        
        # scipy 선형 할당 알고리즘으로 총 거리 비용이 최소가 되는 조합 탐색
        taxi_indices, pax_indices = linear_sum_assignment(dist_matrix)
        
        matches = []
        total_wait_distance = 0.0
        
        for t_idx, p_idx in zip(taxi_indices, pax_indices):
            dist = dist_matrix[t_idx, p_idx]
            matches.append({
                'taxi_id': taxis[t_idx][0],
                'passenger_id': passengers[p_idx][0],
                'wait_distance': round(dist, 2)
            })
            total_wait_distance += dist
            
        print(f"매칭 완료! 총 대기 거리(비용): {round(total_wait_distance, 2)}")
        return pd.DataFrame(matches)

if __name__ == "__main__":
    # (id, x좌표, y좌표) 가상 데이터
    mock_taxis = [('T_001', 0, 0), ('T_002', 2, 2), ('T_003', 5, 5)]
    mock_passengers = [('P_001', 1, 1), ('P_002', 4, 4), ('P_003', 6, 1)]
    
    dispatcher = DynamicDispatcher()
    result_df = dispatcher.match_vehicles(mock_taxis, mock_passengers)
    
    print("\n--- 동적 배차 매칭 결과 ---")
    print(result_df)