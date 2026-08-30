import os
import sys
import time
import traci

def run_sumo_gui():
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)

    sumo_binary = "sumo-gui"
    sumo_cfg = "module1_simulation/sumo_config/simulation.sumocfg"

    print("SUMO Digital Twin 시뮬레이션을 시작합니다...")
    
    try:
        traci.start([sumo_binary, "-c", sumo_cfg])
        
        step = 0
        max_steps = 9999  # 최대 스텝 수 설정
        
        while step < max_steps:
            traci.simulationStep()
            time.sleep(0.03)  # 관찰용 프레임 지연
            
            # 현재 도로 위에서 주행 중인 차량 ID 목록 확인
            active_vehicles = traci.vehicle.getIDList()
            
            # 시뮬레이션 시작 후(step > 5) 도로 위 차량이 0대가 되면 자동 조기 종료
            if step > 5 and len(active_vehicles) == 0:
                print(f" -> [안내] 모든 차량이 목적지에 도착하여 시뮬레이션을 조기 종료합니다. (Step: {step})")
                break
                
            step += 1
            
        traci.close()
        print("시뮬레이션이 정상적으로 종료되었습니다.")
        
    except Exception as e:
        print(f"시뮬레이션 실행 중 오류 발생: {e}")

if __name__ == "__main__":
    run_sumo_gui()