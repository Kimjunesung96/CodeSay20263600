import pulp
import numpy as np

class StochasticOptimizer:
    """예측 불확실성을 반영한 Two-Stage 확률론적 경제급전 최적화 모듈"""
    def __init__(self, base_demand: float, generators: list, num_scenarios: int = 10):
        self.base_demand = base_demand
        self.generators = generators
        self.num_scenarios = num_scenarios
        self.scenarios = self._generate_scenarios()

    def _generate_scenarios(self) -> list:
        np.random.seed(42)
        return [self.base_demand * np.random.uniform(0.9, 1.1) for _ in range(self.num_scenarios)]

    def run_two_stage_optimization(self) -> dict:
        # 1. EV(Expected Value) 최적화: 평균 수요 기준 결정론적 해 산출
        ev_prob = pulp.LpProblem("EV_Dispatch", pulp.LpMinimize)
        ev_outputs = {g['id']: pulp.LpVariable(f"EV_P_{g['id']}", 0, g['max_p']) for g in self.generators}
        ev_prob += pulp.lpSum([ev_outputs[g['id']] * g['cost'] for g in self.generators])
        ev_prob += pulp.lpSum([ev_outputs[g['id']] for g in self.generators]) == self.base_demand
        ev_prob.solve(pulp.PULP_CBC_CMD(msg=False))
        
        ev_dispatch = {g['id']: ev_outputs[g['id']].varValue for g in self.generators}
        ev_base_cost = pulp.value(ev_prob.objective)

        # EEV 계산: EV 해를 불확실한 실제 시나리오들에 적용했을 때의 기대 비용
        eev_scenario_costs = []
        for demand_s in self.scenarios:
            supplied = sum(ev_dispatch.values())
            shortage = max(0.0, demand_s - supplied)
            eev_scenario_costs.append(ev_base_cost + (shortage * 100)) # 패널티 단가 적용
        eev_cost = float(np.mean(eev_scenario_costs))

        # 2. Stochastic(Two-Stage) 최적화
        st_prob = pulp.LpProblem("Stochastic_Dispatch", pulp.LpMinimize)
        stage1_outputs = {g['id']: pulp.LpVariable(f"ST1_P_{g['id']}", 0, g['max_p']) for g in self.generators}
        
        expected_scenario_costs = []
        for i, demand_s in enumerate(self.scenarios):
            shortage = pulp.LpVariable(f"Shortage_s{i}", 0)
            st_prob += pulp.lpSum([stage1_outputs[g['id']] for g in self.generators]) + shortage == demand_s
            expected_scenario_costs.append(shortage * 100)

        st_prob += pulp.lpSum([stage1_outputs[g['id']] * g['cost'] for g in self.generators]) + \
                   (1/self.num_scenarios) * pulp.lpSum(expected_scenario_costs)
        st_prob.solve(pulp.PULP_CBC_CMD(msg=False))
        stochastic_cost = pulp.value(st_prob.objective)

        # 3. 실제 VSS (Value of Stochastic Solution) 계산
        vss_value = eev_cost - stochastic_cost
        vss_percentage = (vss_value / eev_cost) * 100 if eev_cost > 0 else 0.0

        return {
            "status": "Optimal",
            "scenario_count": self.num_scenarios,
            "stochastic_expected_cost": round(stochastic_cost, 2),
            "eev_cost": round(eev_cost, 2),
            "vss_value": round(vss_value, 2),
            "vss_percentage": round(vss_percentage, 2), # 하드코딩 제거 및 실제 값 반영
            "stage1_dispatch_plan": {g['id']: stage1_outputs[g['id']].varValue for g in self.generators}
        }