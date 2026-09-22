"""재현 가능한 합성 호출/외부 데이터. 실제 카카오 호출·날씨 자료가 아니다."""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

CALL_COLUMNS = ['request_id', 'pickup_datetime', 'latitude', 'longitude', 'h3_index', 'from_edge', 'to_edge']


def generate_data(meta, start, days, seed=42):
    if days < 1 or meta['config']['synthetic_rate'] <= 0:
        raise ValueError('days와 synthetic_rate는 양수여야 합니다.')
    cells = sorted(set(meta['edge_cells'].values()))
    pools = {cell: sorted(e for e, c in meta['edge_cells'].items() if c == cell) for cell in cells}
    edges = sorted(meta['edge_cells'])
    if len(edges) < 2:
        raise ValueError('서로 연결된 도로가 최소 2개 필요합니다.')
    calls, weather = [], []
    for date in pd.date_range(pd.Timestamp(start).normalize(), periods=days, freq='D'):
        # 날짜별 독립 시드: 생성 기간이나 A/B 실행 순서가 달라도 같은 날 호출은 같다.
        rng = np.random.default_rng(np.random.SeedSequence([seed, date.toordinal()]))
        # 풍속(km/h)은 별도 시드 스트림으로 뽑아서, 호출 생성용 rng 순서를 건드리지 않는다
        # (풍속을 추가해도 같은 시드의 호출 데이터는 이전과 완전히 동일).
        wind_rng = np.random.default_rng(np.random.SeedSequence([seed, date.toordinal(), 7]))
        for bucket in pd.date_range(date, periods=288, freq='5min'):
            hour = bucket.hour + bucket.minute / 60
            rain = float(rng.choice([0, 0, 0, 1.2, 5.5]))
            temp = 23 + 5 * np.sin((hour - 8) / 24 * 2 * np.pi) + rng.normal(0, 0.5)
            wind = max(0.0, 10 + 3 * np.sin((hour - 8) / 24 * 2 * np.pi) + wind_rng.normal(0, 2))
            traffic = 1 + np.exp(-((hour - 8.5) / 1.5)**2) + np.exp(-((hour - 18) / 2)**2)
            # 공휴일 열은 합성 시나리오의 지정일이며 공식 공휴일 달력이 아니다.
            holiday = int(bucket.month == 8 and bucket.day == 15)
            for cell in cells:
                cell_rank = int(hashlib.sha256(cell.encode()).hexdigest()[:8], 16) % 3
                peak = (8.5, 13, 19)[cell_rank]
                event = int(bucket.dayofweek == 4 and 18 <= hour < 22 and cell_rank == 2)
                rate = meta['config']['synthetic_rate'] * (0.2 + 2.8 * np.exp(-((hour - peak) / 2)**2))
                rate *= (1 + rain * 0.08 + event * 2) * (0.8 if bucket.dayofweek >= 5 or holiday else 1)
                weather.append((bucket, cell, temp, rain, wind, traffic, event, holiday))
                for _ in range(rng.poisson(rate)):
                    origin = str(rng.choice(pools[cell]))
                    destination = str(rng.choice(edges))
                    while destination == origin:
                        destination = str(rng.choice(edges))
                    lat, lng = meta['edge_latlng'][origin]
                    when = bucket + pd.Timedelta(seconds=int(rng.integers(0, 300)))
                    calls.append((f'call_{date:%Y%m%d}_{len(calls)}', when, lat, lng, cell, origin, destination))
    call_df = pd.DataFrame(calls, columns=CALL_COLUMNS).sort_values('pickup_datetime', kind='stable').reset_index(drop=True)
    # ID도 요청한 전체 생성 기간과 무관하도록 날짜 안의 순번으로 정한다.
    if not call_df.empty:
        dates = call_df.pickup_datetime.dt.strftime('%Y%m%d')
        call_df['request_id'] = 'call_' + dates + '_' + call_df.groupby(dates).cumcount().astype(str)
    external = pd.DataFrame(weather, columns=['time_bucket', 'h3_index', 'temperature', 'precipitation', 'wind_speed', 'traffic_index', 'event_flag', 'is_holiday'])
    return call_df, external


def save_data(calls, external, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    calls.to_csv(directory / 'calls.csv', index=False)
    external.to_csv(directory / 'external.csv', index=False)