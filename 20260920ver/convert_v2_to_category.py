# -*- coding: utf-8 -*-
"""
demand_v2.joblib(H3 셀 x t+1~t+6 MultiOutput XGB)  →  category_demand_from_v2.pkl 변환.

목적
- 새 학습 라인이 만든 saved_models/demand_v2.joblib 을, CategoryDemandPredictor(module4_dispatch/
  category_demand_predictor.py)가 읽는 "카테고리 번들" 형식으로 바꿔 별도 파일로 저장한다.

보존 규칙 (절대 건드리지 않는 파일)
- saved_models/category_demand_models.pkl  : 기존 카테고리 모델 (여기서는 읽지도, 쓰지도 않음)
- saved_models/demand_v2.joblib            : 새 모델 원본 (읽기만 함)
- 출력은 saved_models/category_demand_from_v2.pkl 하나뿐이며, 출력 경로가 기존 모델 경로와 같으면 중단한다.

변환 방식
- v2 모델은 y_h1..y_h6 을 각각 예측하는 XGBRegressor 6개의 묶음이다.
- 카테고리 번들은 target="y_h1"(t+1) 이므로 v2 의 y_h1 추정기를 6개 카테고리에 복제해 넣는다.
- feature_cols 는 v2 의 34개를 그대로 쓴다 (모델이 학습된 입력 순서와 동일해야 하므로).

주의 (의미상 한계)
- v2 는 H3 셀 단위 수요로 학습됐고, 카테고리 모델은 카테고리 단위 수요로 학습된다. 이 변환은 형식 변환이지
  재학습이 아니다 → 예측값의 스케일/의미가 기존 카테고리 모델과 같다고 볼 수 없다.
- v2 피처에는 날씨 6종(temperature 등)이 있는데 CategoryDemandPredictor 는 실시간으로 만들지 못해 0으로 채운다.

사용
    python scripts/convert_v2_to_category.py
    python scripts/convert_v2_to_category.py --src saved_models/demand_v2.joblib --dst saved_models/category_demand_from_v2.pkl
"""
import argparse
import copy
import json
import os
import sys

import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SRC_DEFAULT = os.path.join(ROOT, "saved_models", "demand_v2.joblib")
DST_DEFAULT = os.path.join(ROOT, "saved_models", "category_demand_from_v2.pkl")
PROTECTED = os.path.join(ROOT, "saved_models", "category_demand_models.pkl")
META_PATH = os.path.join(ROOT, "module1_simulation", "sumo_config", "runtime_meta.json")  # build_env.py가 만드는 현재 맵

# module3_prediction/train_category_demand.py 의 CATEGORY_LIST 와 동일 (같은 순서)
CATEGORY_LIST = ["school", "residential", "company", "restaurant", "subway_entrance", "bus_stop"]
TARGET = "y_h1"


def cell_mismatch_message(art: dict):
    """v2 가 학습한 H3 셀 목록과 현재 맵(runtime_meta.json)의 셀 목록을 비교.
    다르면 경고 문구를, 같거나 맵 정보를 읽을 수 없으면 None 을 돌려준다(비교 불가는 경고하지 않음)."""
    try:
        with open(META_PATH, "r", encoding="utf-8") as f:
            map_cells = sorted(set(json.load(f)["edge_cells"].values()))
    except Exception:
        return None
    v2_cells = sorted(art.get("cells", []))
    if map_cells == v2_cells:
        return None
    return (f"demand_v2.joblib 의 H3 셀 {len(v2_cells)}개가 현재 맵의 셀 {len(map_cells)}개와 다릅니다 "
            f"(공통 {len(set(map_cells) & set(v2_cells))}개). forecast_dispatcher 는 셀이 다른 모델을 거부합니다. "
            f"맵을 바꿨다면 scripts/generate_training_data.py → scripts/train_dispatch_model.py 로 다시 만들어야 합니다.")


def convert(src: str = SRC_DEFAULT, dst: str = DST_DEFAULT) -> str:
    src, dst = os.path.abspath(src), os.path.abspath(dst)
    for guarded in (PROTECTED, SRC_DEFAULT):
        if os.path.normcase(dst) == os.path.normcase(os.path.abspath(guarded)):
            raise ValueError(f"출력 경로가 보존 대상 파일과 같습니다: {dst}")
    if not os.path.exists(src):
        raise FileNotFoundError(f"{src} 가 없습니다.")

    art = joblib.load(src)
    for key in ("model", "feature_cols", "target_cols"):
        if key not in art:
            raise ValueError(f"{src}: '{key}' 키가 없습니다 (demand_v2 형식이 아님).")
    estimators = getattr(art["model"], "estimators_", None)
    target_cols = list(art["target_cols"])
    if not estimators or TARGET not in target_cols or len(estimators) != len(target_cols):
        raise ValueError(f"{src}: {TARGET} 추정기를 찾을 수 없습니다 (target_cols={target_cols}).")

    warn = cell_mismatch_message(art)
    if warn:
        print(f"[경고] {warn}")  # 변환 자체는 형식 변환이라 계속 진행한다
    base = estimators[target_cols.index(TARGET)]
    bundle = {
        "models": {c: copy.deepcopy(base) for c in CATEGORY_LIST},
        "feature_cols": list(art["feature_cols"]),
        "target": TARGET,
        "category_list": list(CATEGORY_LIST),
        # CategoryDemandPredictor 는 위 4개 키만 읽는다. 아래는 출처 기록용.
        "converted_from": {
            "file": os.path.basename(src),
            "data_until": art.get("data_until"),
            "n_cells": len(art.get("cells", [])),
            "cells_match_current_map": None if warn is None else False,  # None: 일치 또는 비교 불가
            "note": "H3 셀 단위 y_h1 추정기를 카테고리별로 복제한 형식 변환 (재학습 아님)",
        },
    }

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".tmp"
    joblib.dump(bundle, tmp)
    os.replace(tmp, dst)  # 중간에 끊겨도 반쪽짜리 파일이 남지 않도록
    print(f"[변환 완료] {src}\n         → {dst}")
    print(f"  카테고리 {len(CATEGORY_LIST)}개 | 피처 {len(bundle['feature_cols'])}개 | target={TARGET}")
    print(f"  기존 모델 보존: {PROTECTED} (수정 안 함)")
    return dst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_DEFAULT)
    ap.add_argument("--dst", default=DST_DEFAULT)
    ap.add_argument("--check-cells", action="store_true",
                    help="변환 없이 v2 셀 vs 현재 맵 셀만 비교. 다르면 경고를 출력하고 종료코드 3")
    a = ap.parse_args()
    if a.check_cells:
        try:
            msg = cell_mismatch_message(joblib.load(a.src))
        except Exception:
            sys.exit(0)  # 비교 불가는 조용히 통과
        if msg:
            print(msg)
            sys.exit(3)
        sys.exit(0)
    try:
        convert(a.src, a.dst)
    except Exception as e:
        print(f"[변환 실패] {e}")
        sys.exit(1)