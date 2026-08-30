class CarbonTracker:
    """시간대별 발전기 가동에 따른 탄소 배출량 산출 모듈"""
    def __init__(self):
        # 발전원별 예상 탄소 배출 계수 (tCO2/MWh)
        self.emission_factors = {
            "Coal": 0.82,
            "LNG": 0.40,
            "Solar": 0.0,
            "Wind": 0.0
        }

    def calculate_emissions(self, dispatch_plan: dict, generator_types: dict) -> dict:
        total_emissions = 0.0
        details = {}

        for gen_id, power in dispatch_plan.items():
            gen_type = generator_types.get(gen_id, "LNG") # 기본값 LNG
            factor = self.emission_factors.get(gen_type, 0.4)
            
            emissions = power * factor
            details[gen_id] = round(emissions, 2)
            total_emissions += emissions

        return {
            "total_emissions_tCO2": round(total_emissions, 2),
            "details": details
        }