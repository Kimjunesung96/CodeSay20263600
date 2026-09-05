"""
지도 위 건물/시설 POI(관심지점) 추출 모듈.

두 가지 모드를 지원합니다:
1) 실제 지도 모드 (use_real_map=True): OSM 원본 XML(필터링 전, region.osm.xml)에서
   학교/주택/회사/음식점/지하철입구/버스정류장 태그를 파싱하여 위경도 좌표를 뽑고,
   그 좌표를 SUMO 도로망(net.xml)의 가장 가까운 edge에 매핑합니다.
2) 격자 그리드 모드 (use_real_map=False): 건물 개념 자체가 없는 합성 그리드이므로,
   edge 목록을 6개 카테고리로 임의 분할해서 흉내냅니다.

결과는 build_env.py가 그대로 가져다 쓸 수 있도록
    {"school": [edge_id, ...], "residential": [...], "company": [...],
     "restaurant": [...], "subway_entrance": [...], "bus_stop": [...]}
형태의 dict(zones_by_category)를 반환합니다.

지하철입구는 "길(도로/선로) 자체"가 아니라 railway=subway_entrance 노드(출입구)만 대상으로 합니다.
"""

import random
import xml.etree.ElementTree as ET

CATEGORIES = ["school", "residential", "company", "restaurant", "subway_entrance", "bus_stop"]

# ---------- OSM 태그 → 카테고리 매핑 규칙 ----------
# (key, value) 튜플이 하나라도 태그에 매치되면 해당 카테고리로 분류
OSM_TAG_RULES = {
    "school": [
        ("amenity", "school"), ("amenity", "university"),
        ("amenity", "kindergarten"), ("amenity", "college"),
    ],
    "residential": [
        ("building", "residential"), ("building", "apartments"),
        ("building", "house"), ("building", "dormitory"),
    ],
    "company": [
        ("building", "office"), ("building", "commercial"),
        ("office", "*"),  # office=* 전체 (값 무관, "*"는 아래에서 와일드카드로 처리)
    ],
    "restaurant": [
        ("amenity", "restaurant"), ("amenity", "cafe"), ("amenity", "fast_food"),
    ],
    "subway_entrance": [
        ("railway", "subway_entrance"),  # 노드만 해당 (길/선로 taggeed way 아님)
    ],
    "bus_stop": [
        ("highway", "bus_stop"), ("public_transport", "platform"),
    ],
}


def _match_category(tags: dict) -> str:
    """태그 dict를 보고 어느 카테고리에 해당하는지 판정 (여러개 매치 시 첫번째 규칙 우선)"""
    for category, rules in OSM_TAG_RULES.items():
        for key, val in rules:
            if key not in tags:
                continue
            if val == "*" or tags[key] == val:
                return category
    return None


def parse_osm_pois(osm_path: str) -> dict:
    """
    OSM 원본(필터링 전) XML을 파싱해서 카테고리별 (lat, lon) 좌표 리스트를 반환.
    node(지하철입구/버스정류장 등 점 형태)와 way(건물 폴리곤, 중심점 계산) 둘 다 처리.
    반환: {"school": [(lat, lon), ...], ...}
    """
    tree = ET.parse(osm_path)
    root = tree.getroot()

    pois = {c: [] for c in CATEGORIES}

    # node 좌표 캐시 (way의 중심점 계산에 필요)
    node_coords = {}
    for node in root.findall("node"):
        nid = node.attrib.get("id")
        lat = node.attrib.get("lat")
        lon = node.attrib.get("lon")
        if nid and lat and lon:
            node_coords[nid] = (float(lat), float(lon))

    # 1) node 자체가 POI인 경우 (지하철입구, 버스정류장, 소규모 상점 등)
    for node in root.findall("node"):
        tags = {t.attrib["k"]: t.attrib["v"] for t in node.findall("tag")}
        if not tags:
            continue
        category = _match_category(tags)
        if category:
            nid = node.attrib.get("id")
            if nid in node_coords:
                pois[category].append(node_coords[nid])

    # 2) way가 POI인 경우 (건물 폴리곤 — 학교/주택/회사/음식점 대부분 여기 해당)
    for way in root.findall("way"):
        tags = {t.attrib["k"]: t.attrib["v"] for t in way.findall("tag")}
        if not tags:
            continue
        category = _match_category(tags)
        if not category:
            continue
        # subway_entrance/bus_stop은 way로 잘 안 나오지만 혹시 있으면 스킵(점 형태만 다룸)
        if category in ("subway_entrance", "bus_stop"):
            continue

        # way를 구성하는 nd(참조 노드)들의 중심점(centroid)을 그 건물의 대표 좌표로 사용
        nd_ids = [nd.attrib["ref"] for nd in way.findall("nd")]
        coords = [node_coords[n] for n in nd_ids if n in node_coords]
        if not coords:
            continue
        lat_c = sum(c[0] for c in coords) / len(coords)
        lon_c = sum(c[1] for c in coords) / len(coords)
        pois[category].append((lat_c, lon_c))

    return pois


def map_pois_to_edges(pois: dict, net, search_radius: float = 80.0) -> dict:
    """
    카테고리별 (lat, lon) 좌표들을 SUMO net 객체(sumolib.net.readNet 결과)의
    가장 가까운 통행가능 edge에 매핑합니다.
    net.convertLonLat2XY로 좌표 변환 후 net.getNeighboringEdges로 탐색.
    반환: {"school": [edge_id, ...], ...} (중복 제거)
    """
    zones_by_category = {c: set() for c in CATEGORIES}

    for category, coord_list in pois.items():
        for lat, lon in coord_list:
            try:
                x, y = net.convertLonLat2XY(lon, lat)
            except Exception:
                continue
            nearby = net.getNeighboringEdges(x, y, r=search_radius)
            if not nearby:
                continue
            # 가장 가까운 edge 하나만 채택 (passenger 통행 가능한 edge만)
            nearby_sorted = sorted(nearby, key=lambda pair: pair[1])
            for edge, dist in nearby_sorted:
                if edge.getFunction() != "internal" and edge.allows("passenger"):
                    zones_by_category[category].add(edge.getID())
                    break

    return {k: list(v) for k, v in zones_by_category.items()}


def classify_zones_random(edges: list, seed: int = 42) -> dict:
    """
    격자 그리드 모드용: edge 목록을 6개 카테고리로 임의 분할.
    건물 개념이 없는 합성 도로망이므로, 비율을 나눠서 흉내냅니다.
    비율: 학교 10%, 주택 30%, 회사 20%, 음식점 15%, 지하철입구 10%, 버스정류장 15%
    (지하철입구/버스정류장은 "지점" 성격이라 적은 edge 수만 할당)
    """
    rng = random.Random(seed)
    shuffled = edges[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    ratios = {
        "school": 0.10, "residential": 0.30, "company": 0.20,
        "restaurant": 0.15, "subway_entrance": 0.10, "bus_stop": 0.15,
    }

    zones = {}
    idx = 0
    for category, ratio in ratios.items():
        count = max(1, int(n * ratio))
        zones[category] = shuffled[idx: idx + count]
        idx += count

    # 혹시 못 채운 edge가 남으면 residential(주거지역)로 흡수
    if idx < n:
        zones.setdefault("residential", []).extend(shuffled[idx:])

    return zones


def get_boundary_edges(net, edges: list, margin_ratio: float = 0.08) -> list:
    """
    "길 끝(지도 경계 근처 도로)" edge 목록을 반환.
    택시 스폰 위치용 — 지도 한가운데가 아니라, 도시 외곽에서 택시가 들어오는 느낌을 주기 위함.

    net.getBoundary()로 전체 도로망의 (minX, minY, maxX, maxY)를 구한 뒤,
    각 edge의 시작/끝 좌표가 그 경계에서 margin_ratio(기본 8%) 이내에 있으면 "경계 도로"로 판정.
    해당하는 edge가 하나도 없으면(작은 격자 등) 안전하게 edges 전체를 반환.
    """
    try:
        min_x, min_y, max_x, max_y = net.getBoundary()
    except Exception:
        return edges

    width = max_x - min_x
    height = max_y - min_y
    if width <= 0 or height <= 0:
        return edges

    margin_x = width * margin_ratio
    margin_y = height * margin_ratio

    boundary_edges = []
    for edge_id in edges:
        try:
            edge = net.getEdge(edge_id)
            shape = edge.getShape()  # [(x1,y1), (x2,y2), ...]
        except Exception:
            continue
        near_boundary = any(
            x <= min_x + margin_x or x >= max_x - margin_x or
            y <= min_y + margin_y or y >= max_y - margin_y
            for x, y in shape
        )
        if near_boundary:
            boundary_edges.append(edge_id)

    if not boundary_edges:
        print("[안내] 경계 도로를 찾지 못해 전체 도로 중에서 택시를 스폰합니다.")
        return edges

    return boundary_edges


def get_zones(cfg, net, raw_osm_path: str = None) -> dict:
    """
    build_env.py에서 부르는 진입점.
    use_real_map=True면 OSM POI 파싱 + edge 매핑, False면 랜덤 구역 분할.
    """
    edges = [
        e.getID() for e in net.getEdges()
        if e.getFunction() != "internal" and e.allows("passenger") and len(e.getOutgoing()) > 0
    ]

    if cfg.get("use_real_map") and raw_osm_path:
        try:
            pois = parse_osm_pois(raw_osm_path)
            zones = map_pois_to_edges(pois, net)
            # 카테고리가 텅 비어있으면(그 지역에 해당 POI가 없거나 파싱 실패) 랜덤으로 보충
            for category in CATEGORIES:
                if not zones.get(category):
                    print(f"[안내] '{category}' POI를 지도에서 찾지 못해 임의 배정으로 대체합니다.")
                    zones[category] = classify_zones_random(edges).get(category, [])
            print("[안내] OSM 기반 POI(학교/주택/회사/음식점/지하철입구/버스정류장) 구역 매핑 완료.")
            return zones
        except Exception as e:
            print(f"[안내] OSM POI 파싱 실패({type(e).__name__}: {e}) — 임의 구역 분할로 대체합니다.")
            return classify_zones_random(edges)
    else:
        return classify_zones_random(edges)


if __name__ == "__main__":
    # 간단 테스트: 랜덤 분류만 확인 (실제 OSM 파싱은 net 객체 필요)
    fake_edges = [f"edge_{i}" for i in range(40)]
    zones = classify_zones_random(fake_edges)
    for cat, es in zones.items():
        print(f"{cat}: {len(es)}개 - {es[:3]}...")