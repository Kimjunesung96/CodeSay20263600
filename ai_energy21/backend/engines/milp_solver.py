import pulp
import time

def run_economic_dispatch(demand: float, generators: list) -> dict:
    start_time = time.time()
    prob = pulp.LpProblem("Economic_Dispatch", pulp.LpMinimize)
    
    outputs = {}
    for g in generators:
        outputs[g['id']] = pulp.LpVariable(f"P_{g['id']}", lowBound=0, upBound=g['max_p'])
        
    prob += pulp.lpSum([outputs[g['id']] * g['cost'] for g in generators]), "Total_Cost"
    prob += pulp.lpSum([outputs[g['id']] for g in generators]) == demand, "Supply_Demand_Balance"
    
    total_capacity = sum([g['max_p'] for g in generators])
    if total_capacity < demand * 1.10:
        return {"status": "Infeasible", "message": "예비율 10% 제약을 만족할 수 없습니다."}

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    exec_time = time.time() - start_time
    
    return {
        "status": pulp.LpStatus[prob.status],
        "total_cost": pulp.value(prob.objective),
        "execution_time_sec": round(exec_time, 4),
        "dispatch_plan": {g['id']: outputs[g['id']].varValue for g in generators}
    }

def run_rule_based_dispatch(demand: float, generators: list) -> dict:
    """Merit Order 방식의 단순 Rule-based 급전"""
    sorted_gens = sorted(generators, key=lambda x: x['cost'])
    remaining = demand
    total_cost = 0.0
    dispatch_plan = {}
    
    for g in sorted_gens:
        alloc = min(g['max_p'], remaining)
        dispatch_plan[g['id']] = alloc
        total_cost += alloc * g['cost']
        remaining -= alloc
        
    return {"total_cost": total_cost, "dispatch_plan": dispatch_plan}

def run_economic_dispatch_with_comparison(demand: float, generators: list) -> dict:
    milp_res = run_economic_dispatch(demand, generators)
    if milp_res.get("status") != "Optimal":
        return milp_res
        
    rule_res = run_rule_based_dispatch(demand, generators)
    rule_cost = rule_res["total_cost"]
    milp_cost = milp_res["total_cost"]
    
    saving_ratio = ((rule_cost - milp_cost) / rule_cost * 100) if rule_cost > 0 else 0.0
    milp_res["rule_based_cost"] = rule_cost
    milp_res["cost_saving_percent"] = round(saving_ratio, 2)
    return milp_res