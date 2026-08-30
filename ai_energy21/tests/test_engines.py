import pytest

from backend.engines.milp_solver import run_economic_dispatch, run_economic_dispatch_with_comparison
from backend.engines.stochastic_solver import StochasticOptimizer
from backend.core.scenario_simulator import ScenarioSimulator

@pytest.fixture
def sample_generators():
    return [
        {"id": "G1", "max_p": 300.0, "cost": 50.0},
        {"id": "G2", "max_p": 200.0, "cost": 80.0},
        {"id": "G3", "max_p": 100.0, "cost": 120.0}
    ]

def test_milp_solver_success(sample_generators):
    result = run_economic_dispatch(demand=400.0, generators=sample_generators)
    assert result["status"] == "Optimal"
    assert "total_cost" in result

def test_milp_solver_infeasible(sample_generators):
    result = run_economic_dispatch(demand=580.0, generators=sample_generators)
    assert result["status"] == "Infeasible"

def test_milp_comparison(sample_generators):
    result = run_economic_dispatch_with_comparison(demand=400.0, generators=sample_generators)
    assert "rule_based_cost" in result
    assert "cost_saving_percent" in result

def test_stochastic_solver(sample_generators):
    optimizer = StochasticOptimizer(base_demand=400.0, generators=sample_generators, num_scenarios=5)
    result = optimizer.run_two_stage_optimization()
    assert result["status"] == "Optimal"
    assert "eev_cost" in result

def test_scenario_simulator():
    sim = ScenarioSimulator(base_demand=400.0, total_capacity=600.0, solar_capacity=100.0)
    scenarios = sim.run_all_scenarios()
    assert len(scenarios) == 3