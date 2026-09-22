"""완료된 호출 관측 → 30분 예측 → 인센티브 가중 목표 분포 → 빈 택시 재배치.

[통합 이력] 예측 피처 생성을 Module 2 파이프라인(module2_preprocessing)과 동일한
구성으로 맞췄다. 학습(scripts/train_dispatch_model.py)과 추론이 같은
SpatialIndexer/TimeSeriesPreprocessor/merge_external_data 조합을 쓰므로
학습-서빙 피처 불일치가 구조적으로 생기지 않는다.
"""
from collections import Counter
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import traci
from config_loader import CFG, ROOT
from module2_preprocessing.time_series_prep import TimeSeriesPreprocessor
from module2_preprocessing.external_data_merge import merge_external_data, WEATHER_COLS
from module4_dispatch.surge_pricing import SurgePricingEngine


def allocate_targets(demand, multipliers, fleet_size):
    """한정된 빈 택시 수를 초과하지 않도록 최대 나머지 방식으로 정수 배분한다."""
    weights = np.asarray(demand, dtype=float) * np.asarray(multipliers, dtype=float)
    if fleet_size < 0 or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('목표 배분 입력을 확인하세요.')
    if weights.sum() == 0:
        return np.zeros(len(weights), dtype=int)
    quotas = weights / weights.sum() * fleet_size
    targets = np.floor(quotas).astype(int)
    order = np.argsort(-(quotas - targets), kind='stable')
    targets[order[:fleet_size - targets.sum()]] += 1
    return targets


def _external_to_weather(external):
    """합성 외부 관측(time_bucket×h3_index)을 시간별 날씨 표(time 기준)로 변환한다.

    Module 2의 merge_external_data는 '시각 → 그 이전 가장 최근 관측'(merge_asof backward)
    방식이라, 셀 구분 없는 시간별 표만 있으면 된다. 셀별 값은 시각 평균으로 축약한다.
    """
    if external is None or len(external) == 0:
        return None
    if 'time_bucket' not in external or 'temperature' not in external:
        return None
    cols = [c for c in ('temperature', 'precipitation', 'wind_speed') if c in external]
    w = (external.groupby('time_bucket', as_index=False)[cols]
         .mean().rename(columns={'time_bucket': 'time'}))
    w['time'] = pd.to_datetime(w['time'])
    for col in WEATHER_COLS:
        if col not in w:
            w[col] = np.nan
    return w[['time'] + WEATHER_COLS].sort_values('time').reset_index(drop=True)


class ForecastDispatcher:
    def __init__(self, meta, calls, external, start, enabled=True):
        artifact = joblib.load(Path(ROOT) / 'saved_models/demand_v2.joblib')
        if artifact.get('version') != 2:
            raise ValueError('scripts/train_dispatch_model.py로 새 예측 모델을 학습하세요.')
        self.cells = sorted(set(meta['edge_cells'].values()))
        if artifact['cells'] != self.cells:
            raise ValueError('모델과 도로망의 H3 영역이 다릅니다. 현재 맵으로 다시 학습하세요.')
        if pd.Timestamp(artifact['data_until']) > start:
            raise ValueError('시뮬레이션보다 미래 데이터로 학습·선택한 모델입니다.')
        if any(artifact[k] != CFG[k] for k in ('max_lag', 'rolling_short', 'rolling_long')):
            raise ValueError('모델과 전처리 설정이 다릅니다. 재학습하세요.')
        if len(artifact.get('target_cols', [])) != 6:
            raise ValueError('재배치에는 t+1~t+6 6개 타겟 모델이 필요합니다 (scripts/train_dispatch_model.py).')
        self.model, self.features = artifact['model'], artifact['feature_cols']
        self.meta, self.calls, self.start = meta, calls, start
        # 합성 외부 관측이 오면 시간별 날씨 표로 바꿔 학습과 같은 병합 경로를 태운다.
        self.weather = _external_to_weather(external)
        self.prep = TimeSeriesPreprocessor()
        self.engine = SurgePricingEngine()
        # 피처 계산에 필요한 과거 구간: 1시간 이동평균(12칸) > lag(max_lag) > diff(2칸)
        self.history_buckets = max(CFG['max_lag'], self.prep.buckets_1h) + 3
        self.last_tick = -1; self.enabled = enabled; self.committed = {}
        self.records, self.moves = [], []
        self.cell_edges = {cell: sorted(e for e, c in meta['edge_cells'].items() if c == cell) for cell in self.cells}

    def predict(self, now):
        origin = (self.start + pd.Timedelta(seconds=now)).floor('5min')
        last_bucket = origin - pd.Timedelta(minutes=5)
        history_start = origin - pd.Timedelta(minutes=5 * self.history_buckets)
        # 미래 호출이나 날씨는 접근하지 않고 완료된 구간만 집계한다.
        history = self.calls[(self.calls.pickup_datetime >= history_start) & (self.calls.pickup_datetime < origin)]
        panel = self.prep.aggregate_demands(history, start=history_start, end=last_bucket, all_cells=self.cells)
        panel = merge_external_data(panel, weather=self.weather, time_col='time_bucket', verbose=False)
        frame = self.prep.create_features(panel, dropna=False, add_targets=False, verbose=False)
        features = [c for c in self.features if c in frame.columns]
        if len(features) != len(self.features):
            missing = set(self.features) - set(features)
            raise ValueError(f'모델 피처가 피처 테이블에 없습니다: {sorted(missing)}')
        current = frame[frame.time_bucket == last_bucket].sort_values('h3_index')
        # 수요 이력 피처(lag/rolling/diff)가 비면 정말로 관측이 부족한 것 → 에러.
        # 날씨 피처의 NaN은 관측 공백(weather_missing=1로 표시됨)이며, 학습(train_dispatch_model.py)이
        # fillna(0)으로 학습했으므로 서빙도 같은 규칙으로 채운다.
        history_cols = [c for c in self.features
                        if c.startswith(('lag_', 'rolling_')) or c == 'diff_1']
        if len(current) != len(self.cells) or current[history_cols].isna().any().any():
            raise ValueError('예측에 필요한 과거 관측 구간이 부족합니다.')
        predictions = np.asarray(self.model.predict(current[self.features].fillna(0.0)))
        if predictions.ndim != 2 or predictions.shape != (len(self.cells), 6):
            raise ValueError(f'예측 출력 형태가 (셀, 6)이 아닙니다: {predictions.shape}')
        return np.maximum(0, predictions)

    def maintain(self, now):
        tick = int(now // 300)
        if tick == self.last_tick:
            return
        self.last_tick = tick
        predictions = self.predict(now)
        idle = sorted(traci.vehicle.getTaxiFleet(0))
        positions = {vid: self.meta['edge_cells'].get(traci.vehicle.getRoadID(vid)) for vid in idle}
        self.committed = {vid: cell for vid, cell in self.committed.items() if vid in positions and positions[vid] != cell}
        supply = Counter(cell for cell in positions.values() if cell is not None)
        priced = self.engine.apply_surge_pricing(pd.DataFrame({'h3_index': self.cells,
            'predicted_demand': predictions.sum(axis=1), 'available_taxis': [supply[c] for c in self.cells]}))
        target = allocate_targets(priced.predicted_demand, priced.surge_multiplier, sum(supply.values()))
        desired = dict(zip(self.cells, target)); projected = supply.copy()
        for vid, cell in self.committed.items():
            if positions[vid] is not None: projected[positions[vid]] -= 1
            projected[cell] += 1
        for i, row in priced.iterrows():
            self.records.append({'time_sec': now, **row.to_dict(), 'target_taxis': int(target[i]),
                                 **{f'forecast_{k + 1}': float(predictions[i, k]) for k in range(6)}})
        if not self.enabled or predictions.sum() <= 0:
            return
        budget = int(len(idle) * CFG['reposition_fraction'])
        donors = [vid for vid in idle if vid not in self.committed and positions[vid] is not None]
        priorities = sorted(self.cells, key=lambda c: desired[c] - projected[c], reverse=True)
        for cell in priorities:
            while projected[cell] < desired[cell] and budget > 0:
                best = None
                for vid in donors:
                    if projected[positions[vid]] <= desired[positions[vid]]:
                        continue
                    current = traci.vehicle.getRoadID(vid)
                    destination = self.cell_edges[cell][0]
                    route = traci.simulation.findRoute(current, destination, vType=traci.vehicle.getTypeID(vid), depart=now)
                    if route.edges and (best is None or route.travelTime < best[0]):
                        best = (route.travelTime, vid, destination)
                if best is None: break
                _, vid, destination = best
                donors.remove(vid)
                try:
                    traci.vehicle.changeTarget(vid, destination)
                except traci.exceptions.TraCIException:
                    continue
                self.committed[vid] = cell; projected[positions[vid]] -= 1; projected[cell] += 1; budget -= 1
                self.moves.append({'time_sec': now, 'taxi_id': vid, 'from_cell': positions[vid], 'to_cell': cell,
                                   'to_edge': destination, 'travel_time_sec': best[0]})