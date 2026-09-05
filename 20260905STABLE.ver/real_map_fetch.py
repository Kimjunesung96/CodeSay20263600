import os
import sys
import urllib.request
import subprocess
import xml.etree.ElementTree as ET

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def fetch_osm_data(lat_min, lat_max, lng_min, lng_max, output_path):
    """Overpass API / OpenStreetMap API를 사용하여 위경도 영역 OSM 데이터 다운로드"""
    url = f"https://api.openstreetmap.org/api/0.6/map?bbox={lng_min},{lat_min},{lng_max},{lat_max}"
    print(f"[안내] OSM 실제 지도 데이터 다운로드 중... (bbox={lng_min},{lat_min},{lng_max},{lat_max})")
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response, open(output_path, 'wb') as out_file:
        out_file.write(response.read())
    
    file_size_kb = os.path.getsize(output_path) / 1024
    print(f"[완료] OSM 원본 저장: {output_path} ({file_size_kb:.1f} KB)")


def filter_osm_data(input_osm, output_osm):
    """불필요한 OSM 요소(철도, 수로, 버스전용차로 제한 등) 정제"""
    tree = ET.parse(input_osm)
    root = tree.getroot()

    removed_count = 0
    busway_fixed = 0

    for way in list(root.findall('way')):
        tags = {tag.attrib['k']: tag.attrib['v'] for tag in way.findall('tag')}
        
        # 1. 철도/수로/항공 관련 요소 제거
        if any(k in tags for k in ['railway', 'waterway', 'aeroway']):
            root.remove(way)
            removed_count += 1
            continue

        # 2. 버스전용차로(busway) 태그 정제 (일반 차량 통행 가능하도록 보정)
        if tags.get('access') in ['no', 'bus'] or tags.get('highway') == 'busway':
            for tag in way.findall('tag'):
                if tag.attrib['k'] == 'access':
                    tag.attrib['v'] = 'yes'
                if tag.attrib['k'] == 'highway' and tag.attrib['v'] == 'busway':
                    tag.attrib['v'] = 'tertiary'
            busway_fixed += 1

    tree.write(output_osm, encoding='utf-8', xml_declaration=True)
    print(f"[안내] 철도/수로/항공 요소 {removed_count}개 제거, 버스전용차로 {busway_fixed}개 일반도로 재분류 완료 → {output_osm}")


def convert_to_sumo_net(filtered_path, net_path):
    """netconvert를 사용하여 OSM 파일을 SUMO 네트워크(.net.xml)로 변환"""
    cmd = [
        'netconvert',
        '--osm-files', filtered_path,
        '-o', net_path,
        '--geometry.remove', 'true',
        '--ramps.guess', 'true',
        '--junctions.join', 'true',
        '--remove-edges.isolated', 'true',
        '--ramps.no-split', 'true',
        '--edges.join', 'true',
        # [핵심] 차도 전용 추출 (인도/계단 제거)
        '--keep-edges.by-vclass', 'passenger',
        '--no-internal-links', 'false',
        # [핵심] 교차로 꽉 막힘 방지를 위해 주요 교차로에 신호등 자동 배치
        '--tls.guess', 'true',
        '--tls.join', 'true'
    ]
    print("[안내] netconvert로 SUMO 네트워크 변환 중...")
    subprocess.run(cmd, check=True)


def build_real_map_network(lat_min, lat_max, lng_min, lng_max, output_dir):
    """전체 매핑 파이프라인 수행.
    반환값이 (net_path, raw_osm_path) 튜플로 바뀌었습니다.
    raw_osm_path(필터링 전 원본)는 poi_extractor.py가 학교/주택/회사/음식점/
    지하철입구/버스정류장 태그를 파싱하는 데 사용합니다 — filter_osm_data()를
    거치면 도로 외 요소가 일부 정리되므로, POI 파싱은 반드시 필터링 전 원본으로 합니다."""
    os.makedirs(output_dir, exist_ok=True)
    
    raw_osm = os.path.join(output_dir, "region.osm.xml")
    filtered_osm = os.path.join(output_dir, "region_filtered.osm.xml")
    net_path = os.path.join(output_dir, "grid.net.xml")

    fetch_osm_data(lat_min, lat_max, lng_min, lng_max, raw_osm)
    filter_osm_data(raw_osm, filtered_osm)
    convert_to_sumo_net(filtered_osm, net_path)

    return net_path, raw_osm