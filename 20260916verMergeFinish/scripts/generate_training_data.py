# -*- coding: utf-8 -*-
"""
[통합] 현재 맵(runtime_meta.json) 기준 학습용 합성 호출/외부 데이터 재생성.

배경
- build_env.py로 지도를 다시 만들면 도로·H3 셀 구성이 바뀐다.
- 예측 모델(saved_models/demand_v2.joblib)은 학습 당시의 셀 목록을 기억하고,
  forecast_dispatcher는 셀이 다르면 실행을 거부한다(잘못된 예측 방지).
- 그래서 지도를 새로 만들면: 이 스크립트로 데이터 재생성 → train_dispatch_model.py로 재학습.

생성 범위: sim_date 전날까지 days일치 (기본 8일 — 지난주 동일 시간대 피처가 실제 값을 갖도록).
날짜별 독립 시드라 시뮬레이션(replay)과 겹치는 날짜의 호출은 완전히 같다.

사용
    python scripts/generate_training_data.py           # 기본 8일치
    python scripts/generate_training_data.py --days 14
이후
    python scripts/train_dispatch_model.py data/generated/calls.csv --external data/generated/external.csv
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.generate import generate_data, save_data  # noqa: E402

META_PATH = Path(os.environ.get('MOBILITY_SIM_DIR',
                 os.path.join(ROOT, 'module1_simulation', 'sumo_config'))) / 'runtime_meta.json'


def main(days: int):
    meta = json.loads(META_PATH.read_text(encoding='utf-8'))
    if 'config' not in meta:
        raise ValueError('build_env.py로 시뮬레이션 환경을 먼저 생성하세요.')
    cfg = meta['config']
    sim_date = pd.Timestamp(cfg['sim_date'])
    start = sim_date - pd.Timedelta(days=days)
    seed = cfg['passenger_seed'] if cfg.get('passenger_seed') is not None else 42
    calls, external = generate_data(meta, start, days, seed=seed)
    out = Path(ROOT) / 'data' / 'generated'
    save_data(calls, external, out)
    end = sim_date - pd.Timedelta(days=1)
    print(f"[저장] {out}  calls.csv {len(calls):,}행, external.csv {len(external):,}행 "
          f"({start.date()} ~ {end.date()}, 시뮬레이션일 {sim_date.date()} 이전까지)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=8)
    main(ap.parse_args().days)
