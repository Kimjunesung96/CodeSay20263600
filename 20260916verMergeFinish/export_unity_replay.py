"""실제 SUMO 상태를 Unity 재생 JSON으로 기록한다. 배차 로직은 공용 실행기를 사용한다."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import h3
import sumolib
import traci

from measure_wait_time import META_PATH, ROOT, run_and_measure


def export_map(meta_path):
    meta_path = Path(meta_path)
    meta = json.loads(meta_path.read_text())
    directory = meta_path.parent
    cfg = ET.parse(directory / 'simulation.sumocfg').getroot()
    net = sumolib.net.readNet(str(directory / cfg.find('input/net-file').get('value')))
    x0, y0, x1, y1 = net.getBoundary()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

    def point(x, y):
        return {'x': round(x - cx, 3), 'z': round(y - cy, 3)}

    def geo_point(lat, lng):
        config = meta['config']
        if config['use_real_map']:
            return point(*net.convertLonLat2XY(lng, lat))
        # build_env.py의 합성 bbox 매핑을 역변환한다. 실제 측량 좌표가 아니다.
        return point(x0 + (lng - config['lng_min']) / (config['lng_max'] - config['lng_min']) * (x1 - x0),
                     y0 + (lat - config['lat_min']) / (config['lat_max'] - config['lat_min']) * (y1 - y0))

    cells = sorted(set(meta['edge_cells'].values()))
    result = {'version': 1, 'source': 'SUMO / synthetic demand', 'coordinateMapping': meta['coordinate_mapping'],
              'startHour': meta['sim_start_hour'], 'width': x1 - x0, 'height': y1 - y0, 'offsetX': cx, 'offsetY': cy,
              'grid': not meta['config']['use_real_map'], 'blockSize': meta['config']['grid_length'],
              'roads': [{'id': lane.getID(), 'width': lane.getWidth(),
                         'points': [point(*p) for p in lane.getShape()]}
                        for edge in net.getEdges() for lane in edge.getLanes()],
              'junctions': [{'id': node.getID(), 'points': [point(*p) for p in node.getShape()]}
                            for node in net.getNodes() if len(node.getShape()) >= 3],
              'cells': [{'id': cell, 'center': geo_point(*h3.cell_to_latlng(cell)),
                         'points': [geo_point(*p) for p in h3.cell_to_boundary(cell)]} for cell in cells]}
    return result


class ReplayRecorder:
    def __init__(self, destination, map_data):
        self.destination, self.map = Path(destination), map_data
        self.ids, self.entities, self.frames, self.forecasts = {}, [], [], []
        self.last_forecast = -1

    def capture(self, now, requested, pickup, arrived, removed, failed, forecast, teleports):
        idle = set(traci.vehicle.getTaxiFleet(0))
        taxis = set(traci.vehicle.getTaxiFleet(-1))
        vehicles = []
        for vid in sorted(traci.vehicle.getIDList()):
            if vid not in self.ids:
                kind = 'taxi' if vid in taxis else traci.vehicle.getTypeID(vid)
                self.ids[vid] = len(self.entities)
                self.entities.append({'id': vid, 'kind': kind})
            state = 4
            if vid in taxis:
                state = 2 if traci.vehicle.getPersonNumber(vid) else 0 if vid in idle else 1
                if state == 0 and forecast and vid in forecast.committed:
                    state = 3
            elif 'auto' in traci.vehicle.getTypeID(vid).lower():
                state = 5
            elif 'obstacle' in traci.vehicle.getTypeID(vid).lower():
                state = 6
            x, y = traci.vehicle.getPosition(vid)
            vehicles.append({'i': self.ids[vid], 'x': round(x - self.map['offsetX'], 3),
                             'z': round(y - self.map['offsetY'], 3),
                             'a': round(traci.vehicle.getAngle(vid), 2), 's': state})
        pending = set(requested) - set(pickup) - set(removed) - failed
        people = []
        for pid in sorted(pending & set(traci.person.getIDList())):
            x, y = traci.person.getPosition(pid)
            people.append({'x': round(x - self.map['offsetX'], 3), 'z': round(y - self.map['offsetY'], 3)})
        if forecast and len(forecast.records) != self.last_forecast:
            self.last_forecast = len(forecast.records)
            rows = forecast.records[-len(self.map['cells']):]
            self.forecasts.append({'t': now, 'cells': [
                {'demand': row['predicted_demand'], 'supply': row['available_taxis'],
                 'target': row['target_taxis'], 'surge': row['surge_multiplier'],
                 'horizon': [row[f'forecast_{i}'] for i in range(1, 7)]} for row in rows]})
        waits = [pickup[pid] - requested[pid] for pid in pickup]
        self.frames.append({'t': now, 'vehicles': vehicles, 'people': people,
                            'requested': len(requested), 'picked': len(pickup), 'waiting': len(pending),
                            'arrived': len(arrived), 'timedOut': len(set(requested) & set(removed)),
                            'avgWait': sum(waits) / len(waits) if waits else 0,
                            'moves': len(forecast.moves) if forecast else 0,
                            'teleports': teleports, 'forecast': len(self.forecasts) - 1})

    def finish(self, metrics):
        payload = {'version': 1, 'strategy': metrics['strategy'], 'algorithm': metrics['algorithm'],
                   'fingerprint': metrics['demand_fingerprint'], 'duration': metrics['duration_sec'],
                   'startHour': self.map['startHour'], 'entities': self.entities, 'frames': self.frames,
                   'forecasts': self.forecasts, 'metrics': metrics}
        # 전체 기록의 인덱스/시간/수치 오류를 저장 전에 확인한다.
        validate_replay(payload)
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.destination.write_text(json.dumps(payload, separators=(',', ':'), allow_nan=False))


def validate_replay(data):
    assert data['version'] == 1 and len(data['frames']) > 1
    assert all(a['t'] < b['t'] for a, b in zip(data['frames'], data['frames'][1:]))
    assert data['frames'][-1]['t'] == data['duration']
    assert len({e['id'] for e in data['entities']}) == len(data['entities'])
    for frame in data['frames']:
        ids = [v['i'] for v in frame['vehicles']]
        assert len(ids) == len(set(ids)) and all(0 <= i < len(data['entities']) for i in ids)
        assert all(0 <= v['s'] <= 6 for v in frame['vehicles'])
        assert -1 <= frame['forecast'] < len(data['forecasts'])
        assert frame['picked'] + frame['waiting'] + frame['timedOut'] <= frame['requested']
    assert data['frames'][-1]['picked'] == data['metrics']['n_measured']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta', type=Path, default=META_PATH)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/unity_replay')
    args = parser.parse_args()
    map_data = export_map(args.meta)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'map.json').write_text(json.dumps(map_data, separators=(',', ':'), allow_nan=False))
    results = []
    for strategy, algorithm in (('patrol', 'hungarian'), ('patrol', 'rl_reposition')):
        recorder = ReplayRecorder(args.output / f'{strategy}.json', map_data)
        result = run_and_measure(meta_path=args.meta, sumo_cfg_path=args.meta.with_name('simulation.sumocfg'),
                                 strategy=strategy, algorithm=algorithm, recorder=recorder)
        results.append(result)
        print(strategy, result, flush=True)
    assert results[0]['demand_fingerprint'] == results[1]['demand_fingerprint']
    assert all(r['completed_interval'] for r in results)
