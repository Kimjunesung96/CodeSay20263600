import time
from typing import List, Dict
from fastapi import FastAPI
from pydantic import BaseModel

# 백엔드 및 코어 모듈 깔끔하게 단일 정리
from backend.engines.milp_solver import run_economic_dispatch, run_economic_dispatch_with_comparison
from backend.engines.stochastic_solver import StochasticOptimizer
from backend.engines.rl_agent import RLAgentOptimizer
from backend.engines.ess_optimizer import ESSOptimizer
from backend.engines.carbon_tracker import CarbonTracker
from backend.engines.forecaster import DemandForecaster
from backend.core.scenario_simulator import ScenarioSimulator

app = FastAPI(title="AI Energy VPP Agent API")

# ==========================================
# 1. 데이터 모델 정의
# ==========================================
class Generator(BaseModel):
    id: str
    max_p: float
    cost: float

class DispatchRequest(BaseModel):
    demand: float
    generators: List[Generator]

class ForecastRequest(BaseModel):
    target_hours: int

class ESSRequest(BaseModel):
    demand_forecast: List[float]
    capacity_mwh: float = 100.0
    efficiency: float = 0.9

class CarbonRequest(BaseModel):
    dispatch_plan: Dict[str, float]
    generator_types: Dict[str, str]

class ScenarioRequest(BaseModel):
    base_demand: float
    total_capacity: float
    solar_capacity: float


# ==========================================
# 2. API 엔드포인트 라우터
# ==========================================

@app.get("/api/v1/health")
def health_check():
    return {"status": "ok", "service": "VPP Backend Server Connected!"}

@app.post("/api/v1/forecast")
def get_forecast(req: ForecastRequest):
    """collector의 실측 데이터를 활용한 예측 및 실제 MAPE 반환"""
    forecaster = DemandForecaster()
    return forecaster.predict_demand(target_hours=req.target_hours)

@app.post("/api/v1/dispatch/milp")
def dispatch_milp(req: DispatchRequest):
    gen_list = [g.model_dump() for g in req.generators]
    return run_economic_dispatch_with_comparison(req.demand, gen_list)

@app.post("/api/v1/dispatch/stochastic")
def dispatch_stochastic(req: DispatchRequest):
    gen_list = [g.model_dump() for g in req.generators]
    optimizer = StochasticOptimizer(base_demand=req.demand, generators=gen_list)
    return optimizer.run_two_stage_optimization()

@app.post("/api/v1/dispatch/rl")
def dispatch_rl(req: DispatchRequest):
    gen_list = [g.model_dump() for g in req.generators]
    milp_result = run_economic_dispatch(req.demand, gen_list)
    
    if milp_result.get("status") != "Optimal":
        return {
            "status": "Error",
            "message": "MILP 해를 찾지 못해 RL과의 성능 비교를 수행할 수 없습니다."
        }
        
    real_milp_cost = milp_result["total_cost"]
    real_milp_time = milp_result["execution_time_sec"]
    
    agent = RLAgentOptimizer(demand=req.demand, generators=gen_list)
    return agent.run_inference(milp_cost=real_milp_cost, milp_time=real_milp_time)

@app.post("/api/v1/dispatch/ess")
def dispatch_ess(req: ESSRequest):
    optimizer = ESSOptimizer(
        capacity_mwh=req.capacity_mwh,
        round_trip_efficiency=req.efficiency
    )
    return optimizer.optimize_schedule(req.demand_forecast)

@app.post("/api/v1/carbon")
def calculate_carbon(req: CarbonRequest):
    tracker = CarbonTracker()
    return tracker.calculate_emissions(
        dispatch_plan=req.dispatch_plan,
        generator_types=req.generator_types
    )

@app.post("/api/v1/scenarios")
def run_scenarios(req: ScenarioRequest):
    simulator = ScenarioSimulator(
        base_demand=req.base_demand,
        total_capacity=req.total_capacity,
        solar_capacity=req.solar_capacity
    )
    return {"scenarios": simulator.run_all_scenarios()}