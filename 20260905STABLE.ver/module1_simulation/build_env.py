import os
import sys
import subprocess
import xml.etree.ElementTree as ET
import random
import numpy as np
import sumolib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import CFG
from poi_extractor import get_zones, get_boundary_edges, CATEGORIES

DEPART_JITTER_SEC = float(CFG.get("depart_jitter_sec", 300.0))


def _hour_to_sim_seconds(hour: float, sim_start_hour: float) -> float:
    return (hour - sim_start_hour) * 3600.0


def _pick_connected(net, pool_a: list, pool_b: list, max_tries: int = 15):
    """pool_a에서 하나, pool_b에서 하나 뽑되 실제 경로가 존재하는 조합 탐색"""
    a = random.choice(pool_a)
    b = random.choice(pool_b)
    if net is None:
        return a, b
    for _ in range(max_tries):
        a = random.choice(pool_a)
        b = random.choice(pool_b)
        if a == b:
            continue
        try:
            path, cost = net.getShortestPath(net.getEdge(a), net.getEdge(b))
            if path is not None:
                return a, b
        except Exception:
            continue
    return a, b


def build_passenger_schedule(zones: dict, sim_start_hour: float, sim_end_hour: float,
                              num_passengers: int, seed: int = None, net=None) -> list:
    """
    시간대별 모달 분할 및 건물 간 동적 연계 스케줄에 따른 승객 생성 (수요 정제 적용)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    sim_duration_sec = (sim_end_hour - sim_start_hour) * 3600.0
    trips = []

    # 각 구역별 edge 풀 정의
    transit_edges = zones.get("subway_entrance", []) + zones.get("bus_stop", [])
    school_edges = zones.get("school", [])
    company_edges = zones.get("company", [])
    restaurant_edges = zones.get("restaurant", [])
    residential_edges = zones.get("residential", [])

    # 안전장치: 빈 구역 폴백
    all_building_edges = [e for edges in zones.values() for e in edges if edges]
    if not all_building_edges:
        return trips

    transit_edges = transit_edges or all_building_edges
    school_edges = school_edges or all_building_edges
    company_edges = company_edges or all_building_edges
    restaurant_edges = restaurant_edges or all_building_edges
    residential_edges = residential_edges or all_building_edges

    # 학교/회사/음식점은 이제 passenger_spawn_manager.py가 시뮬레이션 도중 실시간으로
    # 생성/소멸(흡수→재스폰)을 담당하므로, 여기서는 주택(residential)만 기존 방식대로
    # 정적 생성합니다. num_passengers는 주택 인구수 산출에 사용.
    BASE_POP_PER_EDGE = int(CFG.get("residential_base_pop_per_edge", 200))
    residential_pop = max(1, len(residential_edges)) * BASE_POP_PER_EDGE + num_passengers

    # ------------------------------------------------------------------
    # 주택(Residential) 기본 통행 스케줄 (기존 유지 — 건드리지 않는 부분)
    # ------------------------------------------------------------------
    schedule_scale = float(CFG.get("residential_schedule_scale", 0.10))
    res_schedule = [
        (float(CFG.get("residential_out_hour", 7.0)), float(CFG.get("residential_out_probability", 0.15)), "out"),
        (float(CFG.get("residential_evening_in_hour", 19.0)), float(CFG.get("residential_evening_in_probability", 0.15)), "in"),
        (float(CFG.get("residential_late_in_hour", 22.0)), float(CFG.get("residential_late_in_probability", 0.20)), "in"),
    ]
    for h, prob, direction in res_schedule:
        if sim_start_hour <= h < sim_end_hour:
            n_pax = max(1, int(residential_pop * prob * schedule_scale))
            depart_center = _hour_to_sim_seconds(h, sim_start_hour)
            for _ in range(n_pax):
                depart = min(max(0, depart_center + np.random.normal(0, DEPART_JITTER_SEC)), sim_duration_sec - 1)
                if direction == "out":
                    start_e, end_e = _pick_connected(net, residential_edges, all_building_edges)
                else:
                    start_e, end_e = _pick_connected(net, all_building_edges, residential_edges)
                trips.append((int(depart), start_e, end_e))

    trips.sort(key=lambda x: x[0])
    return trips


def place_initial_taxis(routes, net, edges: list, boundary_edges: list, num_taxis: int):
    """초기 택시 배치 — 길 끝(경계 도로)에서 스폰"""
    for i in range(num_taxis):
        start_edge, end_edge = _pick_connected(net, boundary_edges, edges)
        taxi_trip = ET.SubElement(routes, "trip", id=f"taxi_{i}", depart="0",
                                   **{"from": start_edge, "to": end_edge, "type": "taxi_type"})
        ET.SubElement(taxi_trip, "param", key="has.taxi.device", value="true")


def create_network_and_routes():
    config_dir = "module1_simulation/sumo_config"
    os.makedirs(config_dir, exist_ok=True)

    sim_start_hour = CFG.get("sim_start_hour", 0)
    sim_end_hour = CFG.get("sim_end_hour", 24)
    taxi_strategy = CFG.get("taxi_strategy", "patrol")
    passenger_seed = CFG.get("passenger_seed")

    if sim_end_hour <= sim_start_hour:
        raise ValueError(f"sim_end_hour({sim_end_hour})는 sim_start_hour({sim_start_hour})보다 커야 합니다.")

    # 1. 도로망 생성
    net_file = os.path.join(config_dir, "grid.net.xml")
    raw_osm_path = None

    if CFG.get("use_real_map"):
        from real_map_fetch import build_real_map_network
        print(f"[안내] 실제 지도 모드 — '{CFG.get('region', '')}' 지역 OSM 데이터로 도로망 생성")
        net_file, raw_osm_path = build_real_map_network(
            CFG["lat_min"], CFG["lat_max"], CFG["lng_min"], CFG["lng_max"], config_dir
        )
    else:
        print(f"{CFG['grid_x']}x{CFG['grid_y']} 블럭 도로망 생성 중... (블록 길이 {CFG['grid_length']}m)")
        subprocess.run([
            'netgenerate', '--grid',
            '--grid.x-number', str(CFG['grid_x']), '--grid.y-number', str(CFG['grid_y']),
            '--grid.length', str(CFG['grid_length']),
            '--sidewalks.guess', 'true',
            '--crossings.guess', 'true',
            '-o', net_file
        ], check=True)

    # 2. 도로망 읽기 및 구역 배정
    net = sumolib.net.readNet(net_file)
    valid_edges = [
        e for e in net.getEdges()
        if not e.getFunction() == 'internal' and e.allows('passenger')
    ]
    edges = [e.getID() for e in valid_edges if len(e.getOutgoing()) > 0]

    zones = get_zones(CFG, net, raw_osm_path=raw_osm_path)
    for category in CATEGORIES:
        print(f"  - {category}: {len(zones.get(category, []))}개 도로 구역 배정")

    boundary_edges = get_boundary_edges(net, edges)
    print(f"  - 택시 스폰 후보(길 끝): {len(boundary_edges)}개 도로")

    # 3. routes XML 생성
    routes = ET.Element("routes")

    ET.SubElement(routes, "vType", id="normal_type", vClass="passenger", color="1,1,1", guiShape="passenger", length="7.0", width="2.8", scale="1.0")
    taxi_vtype = ET.SubElement(routes, "vType", id="taxi_type", vClass="taxi", color="1,1,0", guiShape="passenger/sedan", length="8.0", width="3.0", scale="1.0", personCapacity="4")
    # vType 자체에 파라미터를 박아두면, 이 타입으로 생성되는 모든 차량이 정적(아래 개별 trip)이든
    # 시뮬레이션 도중 taxi_manager.py가 traci.vehicle.add()로 동적 생성하는 것이든 상관없이
    # 자동으로 taxi device를 달고 태어남. (글로벌 옵션 device.taxi.deployProbability/vTypes는
    # 실존하지 않는 옵션이라 시도했다가 실패 — vType 파라미터 방식이 SUMO 공식 지원 방법)
    ET.SubElement(taxi_vtype, "param", key="has.taxi.device", value="true")
    ET.SubElement(routes, "vType", id="auto_type", vClass="passenger", color="0,1,0", guiShape="passenger/hatchback", length="7.0", width="2.8", scale="1.0")
    ET.SubElement(routes, "vType", id="obstacle_type", vClass="ignoring", color="1,0,0", guiShape="truck", length="10.0", width="3.5", scale="1.0")

    def add_random_trip(v_id, v_type="normal_type", color=None):
        start_edge, end_edge = _pick_connected(net, edges, edges)
        attribs = {"id": v_id, "depart": "0", "from": start_edge, "to": end_edge, "type": v_type}
        if color:
            attribs["color"] = color
        return ET.SubElement(routes, "trip", **attribs)

    for i in range(CFG['num_normal_cars']):
        add_random_trip(f"normal_car_{i}", "normal_type", color="1,1,1")
    for i in range(CFG['num_auto_cars']):
        add_random_trip(f"auto_{i}", "auto_type")
    for i in range(CFG['num_obstacles']):
        add_random_trip(f"obstacle_{i}", "obstacle_type")

    # 택시 초기 배치
    place_initial_taxis(routes, net, edges, boundary_edges, CFG['num_taxis'])

    # 승객 배치
    trips = build_passenger_schedule(zones, sim_start_hour, sim_end_hour, CFG['num_passengers'],
                                      seed=passenger_seed, net=net)
    for pax_id, (depart, start_edge, end_edge) in enumerate(trips):
        person = ET.SubElement(routes, "person", id=f"passenger_{pax_id}", depart=str(depart))
        ET.SubElement(person, "ride", **{"from": start_edge, "to": end_edge, "lines": "taxi"})

    rou_file = os.path.join(config_dir, "entities.rou.xml")
    ET.ElementTree(routes).write(rou_file)

    # 4. SUMO Config 파일 생성
    cfg = ET.Element("configuration")
    input_tag = ET.SubElement(cfg, "input")
    ET.SubElement(input_tag, "net-file", value="grid.net.xml")
    ET.SubElement(input_tag, "route-files", value="entities.rou.xml")

    proc_tag = ET.SubElement(cfg, "processing")
    ET.SubElement(proc_tag, "device.taxi.dispatch-algorithm", value=str(CFG.get("taxi_dispatch_algorithm", "greedy")))
    # 손님 없는 택시가 정지해서 대기(idling stop)하지 않고, 정해진 구역 안에서
    # 계속 순환(빙빙 돌기)하며 대기하게 함. 기존 기본값(정지 대기)은 정지 위치 계산이
    # 도로 형상에 따라 무효 판정("invalid position")나는 경우가 잦아서, 택시가
    # 이상한 자리에 낑겨 오도가도 못 하다 방치되는 문제가 있었음. randomCircling은
    # 애초에 "정지 위치"를 계산하지 않으니 이 문제 자체가 생기지 않음.
    ET.SubElement(proc_tag, "device.taxi.idle-algorithm", value=str(CFG.get("taxi_idle_algorithm", "randomCircling")))
    ET.SubElement(proc_tag, "ignore-route-errors", value="true")
    ET.SubElement(proc_tag, "time-to-teleport", value=str(int(CFG.get("sumo_time_to_teleport", 300))))

    sim_end_sec = int((sim_end_hour - sim_start_hour) * 3600)
    time_tag = ET.SubElement(cfg, "time")
    ET.SubElement(time_tag, "end", value=str(sim_end_sec))

    cfg_file = os.path.join(config_dir, "simulation.sumocfg")
    ET.ElementTree(cfg).write(cfg_file)

    # 사이드카 메타 파일 저장
    import json
    meta = {
        "edges": edges,
        "boundary_edges": boundary_edges,
        "taxi_strategy": taxi_strategy,
        "num_taxis": CFG['num_taxis'],
        "sim_start_hour": sim_start_hour,
        "sim_end_hour": sim_end_hour,
        "passenger_wait_timeout": CFG.get("passenger_wait_timeout", 900),
        "zones": zones,
        "hotspot_edges": [e for cat in ("company", "school", "subway_entrance", "bus_stop", "restaurant")
                           for e in zones.get(cat, [])],
        # passenger_spawn_manager.py(실시간 학교/회사/음식점 승객 생성기)가 필요로 하는 정보.
        # school_pop_base/company_pop_base: 아침 등교/출근 모수(400/100)에 GUI num_passengers를
        # 균등 반영한 값. num_passengers는 학교/회사 모수 산정에도 함께 쓰임(절반씩).
        "school_pop_base": CFG.get("school_pop_base", 400),
        "company_pop_base": CFG.get("company_pop_base", 100),
        "passenger_seed": passenger_seed,
        "taxi_remaining_edges_threshold": CFG.get("taxi_remaining_edges_threshold", 2),
        "taxi_target_pick_attempts": CFG.get("taxi_target_pick_attempts", 10),
        "taxi_primary_prob": CFG.get("taxi_primary_prob", 0.70),
        "taxi_fail_threshold": CFG.get("taxi_fail_threshold", 5),
        "taxi_respawn_prefix": CFG.get("taxi_respawn_prefix", "taxi_respawn"),
    }
    with open(os.path.join(config_dir, "runtime_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[안내] Digital Twin 시뮬레이션 환경 구성 완료 "
          f"(정적 생성된 주택발 승객: {len(trips)}명 / 학교·회사·음식점은 실시간 매니저가 담당)")


if __name__ == "__main__":
    create_network_and_routes()