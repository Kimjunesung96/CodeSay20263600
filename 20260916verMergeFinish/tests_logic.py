"""python tests_logic.py: 핵심 회귀 검사. SUMO 서버 없이 실제 함수와 제한된 입출력 대역으로 실행."""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import h3
from config_loader import CFG, DEFAULT_CONFIG, REGION_PRESETS
from data.generate import generate_data
from module2_preprocessing.time_series_prep import TimeSeriesPreprocessor, make_supervised, chronological_split
from module2_preprocessing.external_data_merge import merge_external_data
from module4_dispatch.surge_pricing import SurgePricingEngine
from module4_dispatch.forecast_dispatcher import allocate_targets
from evaluate import calculate_metrics


def sample_meta():
    cells = [h3.latlng_to_cell(lat, 127.03, 8) for lat in (37.49, 37.51)]
    cfg = {**DEFAULT_CONFIG, **REGION_PRESETS['강남역'], 'synthetic_rate': 1.5, 'freq': '5min', 'passenger_mode': 'replay', 'passenger_seed': 42}
    return {'config': cfg, 'edge_cells': dict(zip(('a', 'b'), cells)),
            'edge_latlng': {'a': [37.49, 127.03], 'b': [37.51, 127.03]},
            'num_taxis': 0, 'boundary_edges': ['a'], 'edges': ['a', 'b'], 'zones': {},
            'taxi_strategy': 'patrol', 'taxi_dispatch_algorithm': 'greedy',
            'sim_start_hour': 0, 'sim_end_hour': 1, 'passenger_wait_timeout': 500}


def check_preprocessing():
    meta = sample_meta(); CFG.update(meta['config']); cells = list(meta['edge_cells'].values())
    calls = pd.DataFrame({'pickup_datetime': pd.to_datetime(['2026-08-01 00:00', '2026-08-01 00:10']), 'h3_index': [cells[0]] * 2})
    prep = TimeSeriesPreprocessor()
    panel = prep.aggregate_demands(calls, cells=cells, start='2026-08-01', end='2026-08-01 00:15')
    assert panel.loc[panel.h3_index == cells[0], 'demand'].tolist() == [1, 0, 1]
    assert panel.loc[panel.h3_index == cells[1], 'demand'].sum() == 0
    # 두 지역을 완전히 다른 상수로 채워 이동평균이 지역 경계를 섞지 않는지 검사한다.
    times = pd.date_range('2026-08-01', periods=200, freq='5min')
    panel = pd.DataFrame([(t, c, float(i * 10)) for t in times for i, c in enumerate(cells)], columns=['time_bucket', 'h3_index', 'demand'])
    external = panel[['time_bucket', 'h3_index']].copy()
    for col in ('temperature', 'precipitation', 'traffic_index', 'event_flag', 'is_holiday'): external[col] = 1.
    external.loc[external.time_bucket == times[0], 'temperature'] = np.nan
    merged = merge_external_data(panel, external)
    assert (merged.loc[merged.time_bucket == times[0], 'temperature'] == 0).all()
    assert (merged.loc[merged.time_bucket == times[0], 'temperature_missing'] == 1).all()
    features = prep.create_features(merged)
    frame, names, targets = make_supervised(features)
    assert frame[targets].shape[1] == 6
    for cell, group in frame.groupby('h3_index'):
        assert np.allclose(group['rolling_mean_6'], group.demand)
    train, val, test = chronological_split(frame)
    assert train.time_bucket.max() + pd.Timedelta(minutes=30) < val.time_bucket.min()
    assert val.time_bucket.max() + pd.Timedelta(minutes=30) < test.time_bucket.min()
    # 타깃은 현재값 복사가 아니라 정확히 5분 간격의 미래값이어야 한다.
    ramp = panel.copy(); ramp['demand'] = np.repeat(np.arange(len(times)), len(cells))
    ramp_frame, _, target_cols = make_supervised(prep.create_features(merge_external_data(ramp, external)))
    np.testing.assert_array_equal(ramp_frame[target_cols].iloc[0].to_numpy(), ramp_frame.demand.iloc[0] + np.arange(1, 7))
    from train import sequence_arrays
    x, y, rows = sequence_arrays(train, names, targets, 12)
    assert x.shape[1] == 12 and y.shape[1] == 6
    assert len(rows) == len(x)
    assert calculate_metrics(np.zeros(3), np.ones(3))['MAPE (%)'] is None
    print('PASS: zero grid, causal external join, grouped rolling, six targets, temporal split, real sequences, zero MAPE')


def check_replay():
    meta = sample_meta()
    calls_a, weather_a = generate_data(meta, '2026-08-01', 2)
    calls_b, weather_b = generate_data(meta, '2026-08-02', 1)
    pd.testing.assert_frame_equal(calls_a[calls_a.pickup_datetime >= '2026-08-02'].reset_index(drop=True), calls_b)
    pd.testing.assert_frame_equal(weather_a[weather_a.time_bucket >= '2026-08-02'].reset_index(drop=True), weather_b)
    print('PASS: identical daily requests and external observations regardless of generation window')


def check_pricing():
    engine = SurgePricingEngine()
    assert engine.get_multiplier(engine.calculate_imbalance(100, 0)) == CFG['max_multiplier']
    assert engine.get_multiplier(engine.calculate_imbalance(0, 0)) == CFG['min_multiplier']
    for demand in ([0, 0], [3, 7], [100, 1]):
        targets = allocate_targets(demand, [1, 2], 3)
        assert (targets >= 0).all() and targets.sum() == (3 if sum(demand) else 0)
    try: engine.calculate_imbalance(-1, 2)
    except ValueError: pass
    else: raise AssertionError('negative demand accepted')
    print('PASS: bounded surge, zero supply, conserved integer fleet budget')


def check_pickup_protection():
    import taxi_manager
    vehicle = SimpleNamespace(getIDList=lambda: ['pickup'], getTaxiFleet=lambda mode: [] if mode == 0 else ['pickup'],
                              getTypeID=lambda vid: 'taxi_type')
    fake = SimpleNamespace(vehicle=vehicle, simulation=SimpleNamespace(getPendingVehicles=lambda: []), exceptions=SimpleNamespace(TraCIException=RuntimeError))
    with patch.object(taxi_manager, 'traci', fake):
        manager = taxi_manager.TaxiFleetManager(1, ['a'], ['a', 'b'])
        manager._pick_target_edge = lambda *args: (_ for _ in ()).throw(AssertionError('pickup taxi was repositioned'))
        manager.maintain(10)
    vehicle.getIDList = lambda: []
    vehicle.getTaxiFleet = lambda mode: []
    fake.simulation.getPendingVehicles = lambda: ['pending_taxi']
    with patch.object(taxi_manager, 'traci', fake):
        manager = taxi_manager.TaxiFleetManager(1, ['a'], ['a', 'b'])
        manager._spawn_new_taxi = lambda: (_ for _ in ()).throw(AssertionError('pending taxi duplicated'))
        manager.maintain(10)
    print('PASS: assigned pickup route is untouched; pending taxi is not duplicated')


def check_empty_interval():
    import measure_wait_time as measure
    meta = sample_meta(); clock = [0]
    def step(): clock[0] += 1
    fake = SimpleNamespace(start=lambda args: None, close=lambda: None, simulationStep=step,
        simulation=SimpleNamespace(getTime=lambda: clock[0], getArrivedPersonIDList=lambda: [], getStartingTeleportNumber=lambda: 0),
        person=SimpleNamespace(getIDList=lambda: []), vehicle=SimpleNamespace(getTaxiFleet=lambda mode: []))
    class Manager:
        def __init__(self, *args, **kwargs): self.removed_pids = set(); self.depart_time = {}
        def maintain(self, *args, **kwargs): pass
    calls, weather = generate_data(meta, '2026-08-31', 1)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'meta.json'; path.write_text(json.dumps(meta))
        with patch.object(measure, 'traci', fake), patch.object(measure, 'TaxiFleetManager', Manager), \
             patch.object(measure, 'PassengerTimeoutManager', Manager), \
             patch.object(measure, 'generate_data', return_value=(calls.iloc[:0], weather)):
            result = measure.run_and_measure(meta_path=path, max_steps=60)
    assert result['duration_sec'] == 60 and not result['completed_interval']
    assert result['n_total_passengers'] == 0 and result['avg_wait_sec'] is None
    print('PASS: empty demand does not end simulation at 16 seconds; partial run is marked incomplete')


def check_forecast_causality():
    from module4_dispatch import forecast_dispatcher as module
    meta = sample_meta(); CFG.update(meta['config'])
    calls, external = generate_data(meta, '2026-08-30', 2)
    prep = TimeSeriesPreprocessor()
    panel = prep.aggregate_demands(calls, cells=set(meta['edge_cells'].values()), start='2026-08-30', end='2026-09-01')
    _, features, _ = make_supervised(prep.create_features(merge_external_data(panel, external)))
    class Model:
        def predict(self, frame):
            return np.repeat(frame[['observed_demand']].to_numpy(), 6, axis=1)
    artifact = {'version': 2, 'cells': sorted(meta['edge_cells'].values()), 'data_until': '2026-08-30',
                'model': Model(), 'feature_cols': features,
                **{k: CFG[k] for k in ('max_lag', 'rolling_short', 'rolling_long')}}
    start = pd.Timestamp('2026-08-31 17:00')
    with patch.object(module.joblib, 'load', return_value=artifact):
        dispatcher = module.ForecastDispatcher(meta, calls, external, start)
        before = dispatcher.predict(0)
        dispatcher.calls = pd.concat([calls, calls[calls.pickup_datetime >= start]] * 2, ignore_index=True)
        # 과거는 그대로, 미래 호출만 복제한 자료로 교체한다.
        dispatcher.calls = pd.concat([calls[calls.pickup_datetime < start], dispatcher.calls[dispatcher.calls.pickup_datetime >= start]], ignore_index=True)
        dispatcher.external = external.copy()
        dispatcher.external.loc[external.time_bucket >= start, 'temperature'] = 9999
        np.testing.assert_array_equal(before, dispatcher.predict(0))
    print('PASS: future calls and future weather cannot change the current forecast')


def check_sumo():
    import measure_wait_time as measure
    # 같은 입력의 반복 실행은 동일해야 하며, 예측 재배치의 성능 우위는 가정하지 않는다.
    a = measure.run_and_measure(strategy='patrol')
    b = measure.run_and_measure(strategy='patrol')
    assert a == b, '동일 조건 반복 실행의 결과가 달라졌습니다.'
    for algorithm in ('greedy', 'hungarian'):
        result = measure.run_and_measure(strategy='forecast', algorithm=algorithm)
        assert result['demand_fingerprint'] == a['demand_fingerprint']
        assert result['completed_interval'] and result['n_spawn_failed'] == 0
        assert result['n_total_passengers'] == result['n_measured'] + result['n_unpicked']
        assert result['n_unpicked'] == result['n_timeout_removed'] + result['n_pending_at_end'] + result['n_spawn_failed']
        assert result['max_sampled_fleet'] <= CFG['num_taxis']
    print('PASS: real SUMO deterministic replay, both dispatchers, complete duration, passenger conservation, fleet cap')


if __name__ == '__main__':
    saved = CFG.copy()
    try:
        check_preprocessing(); check_replay(); check_pricing(); check_pickup_protection(); check_empty_interval(); check_forecast_causality()
    finally:
        CFG.clear(); CFG.update(saved)

    import sys
    if '--sumo' in sys.argv:
        check_sumo()
