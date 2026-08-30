import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="AI Energy VPP Dashboard", layout="wide")
st.title("⚡ 지능형 VPP 전력 수급 통합 관제 대시보드")

API_BASE = "http://127.0.0.1:8000/api/v1"

# 사이드바: 기본 설정
st.sidebar.header("⚙️ 시뮬레이션 설정")
demand = st.sidebar.number_input("목표 전력 수요 (MW)", min_value=100.0, max_value=1000.0, value=400.0, step=10.0)

st.sidebar.subheader("가용 발전기 제원")
generators = [
    {"id": "G1 (Base)", "max_p": 300, "cost": 50},
    {"id": "G2 (Mid)", "max_p": 200, "cost": 80},
    {"id": "G3 (Peak)", "max_p": 150, "cost": 120}
]
st.sidebar.table(pd.DataFrame(generators))

payload = {"demand": demand, "generators": generators}

# 7개 탭으로 대시보드 확장
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 시스템 상태", 
    "💡 MILP 경제급전", 
    "🎲 확률론적 최적화", 
    "🤖 강화학습(RL) 급전",
    "🔋 ESS 피크 컷",
    "🌱 탄소 배출량 추적",
    "🚨 3대 위기 시나리오"
])

with tab1:
    st.header("백엔드 API 연결 상태")
    if st.button("서버 상태 확인 (Health Check)"):
        try:
            res = requests.get(f"{API_BASE}/health")
            if res.status_code == 200:
                st.success("✅ 백엔드 정상 동작 중!")
                st.json(res.json())
            else:
                st.error("API 응답 오류")
        except requests.exceptions.ConnectionError:
            st.error("❌ 백엔드 서버(FastAPI)에 연결할 수 없습니다.")

with tab2:
    st.header("MILP 기반 경제급전 (결정론적 최적화)")
    if st.button("MILP 최적화 실행"):
        res = requests.post(f"{API_BASE}/dispatch/milp", json=payload)
        if res.status_code == 200:
            result = res.json()
            if result.get("status") == "Infeasible":
                st.error(result.get("message"))
            else:
                st.success("최적화 완료!")
                col1, col2 = st.columns(2)
                col1.metric("총 발전 비용", f"₩ {result['total_cost']:,.0f}")
                col2.metric("Rule-based 대비 절감률", f"{result.get('cost_saving_percent', 0)} %")
                st.write("발전기별 급전 계획(MW):")
                st.json(result['dispatch_plan'])

with tab3:
    st.header("Two-Stage 확률론적 경제급전")
    if st.button("Stochastic 최적화 실행"):
        res = requests.post(f"{API_BASE}/dispatch/stochastic", json=payload)
        if res.status_code == 200:
            result = res.json()
            st.success("최적화 완료!")
            col1, col2 = st.columns(2)
            col1.metric("확률론적 기대 비용", f"₩ {result['stochastic_expected_cost']:,.0f}")
            col2.metric("VSS (확률론적 해의 가치)", f"{result['vss_percentage']}% 절감")
            st.write(f"생성된 시나리오 수: {result['scenario_count']}개")
            st.json(result['stage1_dispatch_plan'])

with tab4:
    st.header("RL (강화학습) 기반 급전 에이전트")
    if st.button("RL 에이전트 추론 실행"):
        res = requests.post(f"{API_BASE}/dispatch/rl", json=payload)
        if res.status_code == 200:
            result = res.json()
            st.success("추론 완료!")
            col1, col2, col3 = st.columns(3)
            col1.metric("RL 총 비용", f"₩ {result['rl_total_cost']:,.0f}")
            col2.metric("연산 속도 향상", f"{result['milp_speedup_x']} 배")
            col3.metric("MILP 대비 비용 차이", f"{result['cost_difference_percent']} %")
            st.write(f"사용 모델: {result['model']}")
            st.write(f"추론 소요 시간: {result['inference_time_sec']}초")
            st.json(result['dispatch_plan'])

# [신규 탭 5] ESS 스케줄링
with tab5:
    st.header("🔋 ESS 피크 부하 감소(Peak Shaving) 스케줄링")
    st.write("24시간 예상 수요 프로필에 따른 충/방전 제어")
    sample_demand_profile = [300, 280, 260, 250, 270, 310, 400, 520, 580, 600, 550, 500, 480, 510, 530, 590, 620, 610, 540, 470, 420, 380, 350, 320]
    
    if st.button("ESS 스케줄링 최적화"):
        ess_payload = {
            "demand_forecast": sample_demand_profile,
            "capacity_mwh": 100.0,
            "efficiency": 0.9
        }
        res = requests.post(f"{API_BASE}/dispatch/ess", json=ess_payload)
        if res.status_code == 200:
            result = res.json()
            st.success(f"목표 피크 감축률: {result['target_peak_reduction']}")
            col1, col2 = st.columns(2)
            col1.metric("기존 피크 부하", f"{result['original_peak_MW']} MW")
            col2.metric("제어 후 피크 부하", f"{result['new_peak_MW']} MW")
            st.write("시간대별 ESS 동작 스케줄:")
            st.dataframe(pd.DataFrame(result['schedule']))

# [신규 탭 6] 탄소 배출량
with tab6:
    st.header("🌱 시간대별 발전원별 탄소 배출량 추적")
    if st.button("탄소 배출량 산출"):
        carbon_payload = {
            "dispatch_plan": {"G1 (Base)": 300.0, "G2 (Mid)": 100.0, "G3 (Peak)": 0.0},
            "generator_types": {"G1 (Base)": "Coal", "G2 (Mid)": "LNG", "G3 (Peak)": "Solar"}
        }
        res = requests.post(f"{API_BASE}/carbon", json=carbon_payload)
        if res.status_code == 200:
            result = res.json()
            st.metric("총 탄소 배출량", f"{result['total_emissions_tCO2']} tCO2")
            st.write("발전기별 배출 상세:")
            st.json(result['details'])

# [신규 탭 7] 위기 시나리오
with tab7:
    st.header("🚨 3대 위기 시나리오 실시간 수급 검증")
    base_dem = st.number_input("기준 수요 (MW)", value=400.0)
    total_cap = st.number_input("총 설비 용량 (MW)", value=600.0)
    solar_cap = st.number_input("태양광 설비 용량 (MW)", value=100.0)
    
    if st.button("위기 시나리오 시뮬레이션"):
        scenario_payload = {
            "base_demand": base_dem,
            "total_capacity": total_cap,
            "solar_capacity": solar_cap
        }
        res = requests.post(f"{API_BASE}/scenarios", json=scenario_payload)
        if res.status_code == 200:
            scenarios = res.json().get("scenarios", [])
            for sc in scenarios:
                status_icon = "✅ 안전" if sc["is_safe"] else "❌ 위험"
                st.subheader(f"{status_icon} - {sc['scenario']}")
                st.write(f"예비율: {sc['reserve_margin_percent']}% (최소 요구 예비율 5%)")
                st.divider()