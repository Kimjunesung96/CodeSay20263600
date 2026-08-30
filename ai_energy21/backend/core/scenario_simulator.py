class ScenarioSimulator:
    """
    3대 위기 시나리오 발생 시 전력 수급 및 예비율 방어 검증 모듈
    """
    def __init__(self, base_demand: float, total_capacity: float, solar_capacity: float):
        self.base_demand = base_demand
        self.total_capacity = total_capacity
        self.solar_capacity = solar_capacity
        self.required_margin = 5.0  # 위기 상황 시 최소 예비율 5% 기준

    def _calculate_margin(self, capacity: float, demand: float) -> float:
        """예비율(%) 계산"""
        if demand == 0:
            return float('inf')
        return ((capacity - demand) / demand) * 100

    def run_heatwave_scenario(self) -> dict:
        """시나리오 1: 폭염으로 인한 전력 수요 15% 증가"""
        new_demand = self.base_demand * 1.15
        margin = self._calculate_margin(self.total_capacity, new_demand)
        return {
            "scenario": "Heatwave (Demand +15%)",
            "new_demand_MW": round(new_demand, 2),
            "reserve_margin_percent": round(margin, 2),
            "is_safe": margin >= self.required_margin
        }

    def run_plant_failure_scenario(self) -> dict:
        """시나리오 2: 대형 발전소 1400MW 돌발 탈락"""
        new_capacity = self.total_capacity - 1400.0
        margin = self._calculate_margin(new_capacity, self.base_demand)
        return {
            "scenario": "Plant Failure (Capacity -1400MW)",
            "new_capacity_MW": round(new_capacity, 2),
            "reserve_margin_percent": round(margin, 2),
            "is_safe": margin >= self.required_margin
        }

    def run_renewable_drop_scenario(self) -> dict:
        """시나리오 3: 기상 악화로 인한 태양광 발전량 50% 램프다운"""
        new_capacity = self.total_capacity - (self.solar_capacity * 0.5)
        margin = self._calculate_margin(new_capacity, self.base_demand)
        return {
            "scenario": "Solar Drop (Solar Capacity -50%)",
            "new_capacity_MW": round(new_capacity, 2),
            "reserve_margin_percent": round(margin, 2),
            "is_safe": margin >= self.required_margin
        }

    def run_all_scenarios(self) -> list:
        """3대 위기 시나리오 일괄 실행"""
        return [
            self.run_heatwave_scenario(),
            self.run_plant_failure_scenario(),
            self.run_renewable_drop_scenario()
        ]