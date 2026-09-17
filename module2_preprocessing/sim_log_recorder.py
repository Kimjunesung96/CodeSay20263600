"""
SUMO 시뮬레이션 호출(수요) 로그 기록기.

왜 필요한가:
- 기존 train.py / main.py는 np.random으로 만든 가짜 호출 데이터를 학습에 사용했음.
- measure_wait_time.py가 승객 등장/탑승 시각을 잡긴 했지만 메모리에만 두고 버렸음.
- 이 모듈은 시뮬레이션 루프에서 매 스텝 호출되어, 승객 1명 = 호출 1건으로
  CSV에 남긴다. 이 CSV가 Module 2(전처리) → Module 3(학습)의 실제 입력이 된다.

CSV 컬럼 (1행 = 승객 1명):
    person_id        SUMO person id (passenger_N: build_env 정적 승객 / dyn_pax_N: 동적 스폰)
    request_sec      시뮬레이션 경과초 기준 등장(호출) 시각
    pickup_datetime  request_sec을 실제 시각으로 변환 (sim_date + sim_start_hour + request_sec)
                     ※ 컬럼명은 기존 time_series_prep.py와 호환되도록 pickup_datetime 유지
    edge_id          등장한 도로 edge
    origin_zone      edge_id가 속한 POI 구역(school/company/...; 여러 개면 '|'로 연결, 없으면 none)
    dest_category    동적 스폰 승객의 목적지 카테고리(PassengerSpawnManager 기준, 없으면 빈칸)
    x, y             SUMO 좌표
    latitude, longitude  위경도 (실제지도: SUMO geo 변환 / 격자: config bbox로 선형 매핑)
    pickup_sec       택시 탑승 시각(경과초), 못 탔으면 빈칸
    wait_sec         pickup_sec - request_sec
    taxi_id          탑승한 택시 id
    status           picked / timeout / unpicked(시뮬 종료 시까지 대기)
"""

import csv
import os
from datetime import datetime, timedelta

try:
    import traci  # 시뮬레이션 루프에서만 필요. train.py가 latest_log_path만 쓸 때는 없어도 됨
except ImportError:
    traci = None

CSV_COLUMNS = [
    "person_id", "request_sec", "pickup_datetime", "edge_id", "origin_zone", "dest_category",
    "x", "y", "latitude", "longitude", "pickup_sec", "wait_sec", "taxi_id", "status",
]


class DemandLogRecorder:
    def __init__(self, sim_start_hour: float, zones: dict = None, cfg: dict = None,
                 spawn_manager=None, sim_date: str = None, out_dir: str = None):
        cfg = cfg or {}
        self.sim_start_hour = float(sim_start_hour or 0)
        self.sim_date = sim_date or cfg.get("sim_date", "2026-09-07")
        self.base_dt = datetime.strptime(self.sim_date, "%Y-%m-%d") + timedelta(hours=self.sim_start_hour)
        self.spawn_manager = spawn_manager
        self.cfg = cfg

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.out_dir = out_dir or os.path.join(root, "data", "sim_logs")

        # edge -> 구역 카테고리 역인덱스
        self.edge_zone = {}
        for category, edges in (zones or {}).items():
            for e in edges or []:
                self.edge_zone.setdefault(e, []).append(category)

        self.records = {}   # person_id -> dict
        self._geo_ok = None  # 실제 지도(geo 투영) 여부, 첫 변환 때 판정
        self._net_bounds = None

    # ---------- 좌표 변환 ----------
    def _to_latlng(self, x: float, y: float):
        if self._geo_ok is not False:
            try:
                lon, lat = traci.simulation.convertGeo(x, y)
                # 투영 정보 없는 격자망은 (x, y)를 그대로 돌려주거나 0 근처 값이 나옴
                if -90 <= lat <= 90 and -180 <= lon <= 180 and not (lat == y and lon == x):
                    self._geo_ok = True
                    return lat, lon
            except Exception:
                pass
            self._geo_ok = False

        # 격자 모드 폴백: 도로망 경계 → config의 위경도 bbox로 선형 매핑
        if self._net_bounds is None:
            try:
                (xmin, ymin), (xmax, ymax) = traci.simulation.getNetBoundary()
            except Exception:
                xmin, ymin, xmax, ymax = 0.0, 0.0, 1.0, 1.0
            self._net_bounds = (xmin, ymin, max(xmax, xmin + 1e-9), max(ymax, ymin + 1e-9))
        xmin, ymin, xmax, ymax = self._net_bounds
        lat_min, lat_max = self.cfg.get("lat_min"), self.cfg.get("lat_max")
        lng_min, lng_max = self.cfg.get("lng_min"), self.cfg.get("lng_max")
        if None in (lat_min, lat_max, lng_min, lng_max):
            return None, None
        lat = lat_min + (y - ymin) / (ymax - ymin) * (lat_max - lat_min)
        lon = lng_min + (x - xmin) / (xmax - xmin) * (lng_max - lng_min)
        return lat, lon

    # ---------- 매 스텝 호출 ----------
    def step(self, now_seconds: float, timeout_removed_pids: set = None):
        for pid in traci.person.getIDList():
            rec = self.records.get(pid)
            if rec is None:
                rec = self._new_record(pid, now_seconds)
                self.records[pid] = rec
            if rec["pickup_sec"] is None:
                try:
                    vid = traci.person.getVehicle(pid)
                except traci.exceptions.TraCIException:
                    vid = ""
                if vid:
                    rec["pickup_sec"] = now_seconds
                    rec["taxi_id"] = vid
                    rec["status"] = "picked"

        for pid in (timeout_removed_pids or ()):
            rec = self.records.get(pid)
            if rec is not None and rec["status"] == "unpicked":
                rec["status"] = "timeout"

    def _new_record(self, pid: str, now_seconds: float) -> dict:
        try:
            edge = traci.person.getRoadID(pid)
        except traci.exceptions.TraCIException:
            edge = ""
        try:
            x, y = traci.person.getPosition(pid)
        except traci.exceptions.TraCIException:
            x, y = None, None
        lat, lon = self._to_latlng(x, y) if x is not None else (None, None)

        dest_category = ""
        if self.spawn_manager is not None:
            dest_category = getattr(self.spawn_manager, "_dest_category", {}).get(pid, "") or ""

        return {
            "person_id": pid,
            "request_sec": now_seconds,
            "pickup_datetime": (self.base_dt + timedelta(seconds=now_seconds)).strftime("%Y-%m-%d %H:%M:%S"),
            "edge_id": edge,
            "origin_zone": "|".join(self.edge_zone.get(edge, [])) or "none",
            "dest_category": dest_category,
            "x": x, "y": y, "latitude": lat, "longitude": lon,
            "pickup_sec": None, "taxi_id": "", "status": "unpicked",
        }

    # ---------- 저장 ----------
    def save(self, filename: str = None) -> str:
        os.makedirs(self.out_dir, exist_ok=True)
        if filename is None:
            filename = f"demand_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path = os.path.join(self.out_dir, filename)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            w.writeheader()
            for rec in self.records.values():
                row = dict(rec)
                row["wait_sec"] = (rec["pickup_sec"] - rec["request_sec"]) if rec["pickup_sec"] is not None else None
                w.writerow(row)
        n = len(self.records)
        n_picked = sum(1 for r in self.records.values() if r["status"] == "picked")
        print(f"[로그 저장] {path} (호출 {n}건, 탑승 {n_picked}건, geo변환={'SUMO' if self._geo_ok else 'bbox 선형매핑'})")
        return path


def latest_log_path(log_dir: str = None) -> str:
    """data/sim_logs에서 가장 최근 demand_log_*.csv 경로 (없으면 None)"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = log_dir or os.path.join(root, "data", "sim_logs")
    if not os.path.isdir(log_dir):
        return None
    files = sorted(f for f in os.listdir(log_dir) if f.startswith("demand_log_") and f.endswith(".csv"))
    return os.path.join(log_dir, files[-1]) if files else None
