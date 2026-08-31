import os
import sys
import subprocess
import xml.etree.ElementTree as ET
import random
import sumolib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import CFG


def create_network_and_routes():
    config_dir = "module1_simulation/sumo_config"
    os.makedirs(config_dir, exist_ok=True)

    # 1. 격자 도로망 생성 (크기는 config.json에서 조절)
    print(f"{CFG['grid_x']}x{CFG['grid_y']} 블럭 도로망 생성 중... (블록 길이 {CFG['grid_length']}m)")
    net_file = os.path.join(config_dir, "grid.net.xml")
    subprocess.run([
        'netgenerate', '--grid',
        '--grid.x-number', str(CFG['grid_x']), '--grid.y-number', str(CFG['grid_y']),
        '--grid.length', str(CFG['grid_length']), '-o', net_file
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
        if color:
            attribs["color"] = color
        ET.SubElement(routes, "trip", **attribs)

    # 차량/탑승자 개수도 config.json에서 조절
    for i in range(CFG['num_normal_cars']):
        add_random_trip(f"normal_car_{i}", "normal_type", color="1,1,1")
    for i in range(CFG['num_taxis']):
        add_random_trip(f"taxi_{i}", "taxi_type")
    for i in range(CFG['num_auto_cars']):
        add_random_trip(f"auto_{i}", "auto_type")
    for i in range(CFG['num_obstacles']):
        add_random_trip(f"obstacle_{i}", "obstacle_type")

    for i in range(CFG['num_passengers']):
        start_edge, end_edge = random.sample(edges, 2)
        person = ET.SubElement(routes, "person", id=f"passenger_{i}", depart="0")
        ET.SubElement(person, "walk", **{"from": start_edge, "to": end_edge})

    rou_file = os.path.join(config_dir, "entities.rou.xml")
    tree = ET.ElementTree(routes)
    tree.write(rou_file)

    # 3. SUMO Config 파일 (.sumocfg) 생성
    cfg = ET.Element("configuration")
    input_tag = ET.SubElement(cfg, "input")
    ET.SubElement(input_tag, "net-file", value="grid.net.xml")
    ET.SubElement(input_tag, "route-files", value="entities.rou.xml")

    cfg_file = os.path.join(config_dir, "simulation.sumocfg")
    tree_cfg = ET.ElementTree(cfg)
    tree_cfg.write(cfg_file)

    print(
        f"Digital Twin 환경 구성 완료! "
        f"(격자 {CFG['grid_x']}x{CFG['grid_y']}, 일반차량 {CFG['num_normal_cars']}대, "
        f"택시 {CFG['num_taxis']}대, 자율차 {CFG['num_auto_cars']}대, "
        f"장애물 {CFG['num_obstacles']}개, 탑승자 {CFG['num_passengers']}명)"
    )


if __name__ == "__main__":
    create_network_and_routes()