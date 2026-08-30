import os
import subprocess
import xml.etree.ElementTree as ET
import random
import sumolib

def create_network_and_routes():
    config_dir = "module1_simulation/sumo_config"
    os.makedirs(config_dir, exist_ok=True)
    
    # 1. 3x3 격자 도로망 생성
    print("3x3 블럭 도로망 생성 중...")
    net_file = os.path.join(config_dir, "3x3_grid.net.xml")
    subprocess.run([
        'netgenerate', '--grid', 
        '--grid.x-number', '3', '--grid.y-number', '3', 
        '--grid.length', '200', '-o', net_file
    ], check=True)

    net = sumolib.net.readNet(net_file)
    edges = [e.getID() for e in net.getEdges() if not e.getFunction() == 'internal']

    # 2. 차량 및 객체/탑승자 배치 (routes XML 생성)
    routes = ET.Element("routes")
    
    # 차량 타입 정의 (width, length, scale 속성 추가로 크기 확대)
    ET.SubElement(routes, "vType", id="normal_type", vClass="passenger", color="1,1,1", guiShape="passenger", length="7.0", width="2.8", scale="2.0")
    ET.SubElement(routes, "vType", id="taxi_type", vClass="taxi", color="1,1,0", guiShape="passenger/sedan", length="8.0", width="3.0", scale="2.0")
    ET.SubElement(routes, "vType", id="auto_type", vClass="passenger", color="0,1,0", guiShape="passenger/hatchback", length="7.0", width="2.8", scale="2.0")
    ET.SubElement(routes, "vType", id="obstacle_type", vClass="ignoring", color="1,0,0", guiShape="truck", length="10.0", width="3.5", scale="2.0")
    
    def add_random_trip(v_id, v_type="normal_type", color=None):
        start_edge, end_edge = random.sample(edges, 2)
        attribs = {"id": v_id, "depart": "0", "from": start_edge, "to": end_edge, "type": v_type}
        if color: attribs["color"] = color
        ET.SubElement(routes, "trip", **attribs)

    # 요구사항 준수: 일반차량 20대, 택시 3대, 자율차 1대, 장애물 2대
    for i in range(20): add_random_trip(f"normal_car_{i}", "normal_type", color="1,1,1")
    for i in range(3): add_random_trip(f"taxi_{i}", "taxi_type")
    add_random_trip("auto_1", "auto_type")
    for i in range(2): add_random_trip(f"obstacle_{i}", "obstacle_type")

    # 요구사항 준수: 탑승자(Passenger) 5명 배치 추가
    for i in range(5):
        start_edge, end_edge = random.sample(edges, 2)
        person = ET.SubElement(routes, "person", id=f"passenger_{i}", depart="0")
        ET.SubElement(person, "walk", **{"from": start_edge, "to": end_edge})

    rou_file = os.path.join(config_dir, "entities.rou.xml")
    tree = ET.ElementTree(routes)
    tree.write(rou_file)

    # 3. SUMO Config 파일 (.sumocfg) 생성
    cfg = ET.Element("configuration")
    input_tag = ET.SubElement(cfg, "input")
    ET.SubElement(input_tag, "net-file", value="3x3_grid.net.xml")
    ET.SubElement(input_tag, "route-files", value="entities.rou.xml")
    
    cfg_file = os.path.join(config_dir, "simulation.sumocfg")
    tree_cfg = ET.ElementTree(cfg)
    tree_cfg.write(cfg_file)

    print("Digital Twin 환경 구성 (차량 크기 확대 및 탑승자 5명 포함) 세팅 완료!")

if __name__ == "__main__":
    create_network_and_routes()