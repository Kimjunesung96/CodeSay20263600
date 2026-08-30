import numpy as np

class ESSOptimizer:
    """피크 부하 감소(Peak Shaving)를 위한 ESS 스케줄링 모듈"""
    def __init__(self, capacity_mwh: float = 100.0, round_trip_efficiency: float = 0.9):
        self.capacity = capacity_mwh
        self.efficiency = round_trip_efficiency

    def optimize_schedule(self, demand_forecast: list) -> dict:
        # 요구사항: 피크 부하 5% 이상 감소
        peak_demand = max(demand_forecast)
        target_peak = peak_demand * 0.95 
        
        schedule = []
        for demand in demand_forecast:
            if demand > target_peak:
                # 방전 (피크 컷) - 효율 적용
                discharge_amount = round((demand - target_peak) / self.efficiency, 2)
                schedule.append({"action": "discharge", "amount_MW": discharge_amount})
            elif demand < np.mean(demand_forecast):
                # 충전 (경부하 시간대)
                charge_amount = round((np.mean(demand_forecast) - demand), 2)
                schedule.append({"action": "charge", "amount_MW": charge_amount})
            else:
                schedule.append({"action": "idle", "amount_MW": 0.0})
                
        return {
            "target_peak_reduction": "5%",
            "original_peak_MW": round(peak_demand, 2),
            "new_peak_MW": round(target_peak, 2),
            "schedule": schedule
        }