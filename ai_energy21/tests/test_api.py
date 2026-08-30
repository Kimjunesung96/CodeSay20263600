import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health_check_api():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

# forecast stub 상태 명확히 검증
# test_api.py의 forecast 관련 테스트 함수 수정

def test_forecast_api():
    response = client.post("/api/v1/forecast", json={"target_hours": 24})
    assert response.status_code == 200
    res_data = response.json()
    
    assert res_data["status"] == "Success"
    assert "mape_percent" in res_data
    assert isinstance(res_data["mape_percent"], float)
    assert res_data["mape_percent"] >= 0.0  # 정량 수치 존재 확인
    assert len(res_data["forecast_data"]) == 24
    assert "actual_MW" in res_data["forecast_data"][0] # 실측값 존재 확인

    
def test_dispatch_milp_api():
    payload = {
        "demand": 300.0,
        "generators": [
            {"id": "G1", "max_p": 200.0, "cost": 50.0},
            {"id": "G2", "max_p": 200.0, "cost": 80.0}
        ]
    }
    response = client.post("/api/v1/dispatch/milp", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "Optimal"

def test_dispatch_stochastic_api():
    payload = {
        "demand": 300.0,
        "generators": [
            {"id": "G1", "max_p": 200.0, "cost": 50.0},
            {"id": "G2", "max_p": 200.0, "cost": 80.0}
        ]
    }
    response = client.post("/api/v1/dispatch/stochastic", json=payload)
    assert response.status_code == 200
    assert "vss_percentage" in response.json()

def test_dispatch_rl_api():
    payload = {
        "demand": 300.0,
        "generators": [
            {"id": "G1", "max_p": 200.0, "cost": 50.0},
            {"id": "G2", "max_p": 200.0, "cost": 80.0}
        ]
    }
    response = client.post("/api/v1/dispatch/rl", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "Success"

def test_dispatch_ess_api():
    payload = {
        "demand_forecast": [300.0, 450.0, 500.0, 200.0],
        "capacity_mwh": 100.0,
        "efficiency": 0.9
    }
    response = client.post("/api/v1/dispatch/ess", json=payload)
    assert response.status_code == 200
    assert "schedule" in response.json()

def test_carbon_api():
    payload = {
        "dispatch_plan": {"G1": 100.0, "G2": 200.0},
        "generator_types": {"G1": "Coal", "G2": "LNG"}
    }
    response = client.post("/api/v1/carbon", json=payload)
    assert response.status_code == 200
    assert "total_emissions_tCO2" in response.json()

def test_scenarios_api():
    payload = {
        "base_demand": 400.0,
        "total_capacity": 600.0,
        "solar_capacity": 100.0
    }
    response = client.post("/api/v1/scenarios", json=payload)
    assert response.status_code == 200
    assert "scenarios" in response.json()