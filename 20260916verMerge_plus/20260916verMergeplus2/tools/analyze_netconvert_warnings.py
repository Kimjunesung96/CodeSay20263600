# -*- coding: utf-8 -*-
"""
[1번 담당 검증] netconvert 경고 분류기

real_map_fetch.py가 OSM을 SUMO 도로망으로 변환할 때 쏟아지는 경고를
"시뮬레이션에 실제 영향 있는 것"과 "무시해도 되는 것"으로 자동 분류한다.

netconvert는 같은 유형 경고를 5회까지만 출력하고 마지막에
"N total messages of type: ..." 로 총계를 알려주므로, 두 형태를 모두 집계한다.

실행:
    python tools/analyze_netconvert_warnings.py <netconvert_stderr.txt>

stderr 로그를 만드는 법 (module1_simulation/sumo_config 기준):
    netconvert --osm-files region_filtered.osm.xml -o grid.net.xml \
      --geometry.remove true --ramps.guess true --junctions.join true \
      --remove-edges.isolated true --ramps.no-split true --edges.join true \
      --keep-edges.by-vclass passenger --no-internal-links false \
      --tls.guess true --tls.join true  2> nc_stderr.txt
"""
import re
import sys
from collections import Counter

# (판정, 사유) — 부분 문자열로 매칭
RULES = [
    ("pt stop",                          "무시", "대중교통 정류장 정보. SUMO에서 버스·지하철을 운행시키지 않으므로 무관"),
    ("pt line",                          "무시", "대중교통 노선 정보. 위와 동일"),
    ("Removed invalid stop",             "무시", "대중교통 정류장 정보. 위와 동일"),
    ("Could not assign stop",            "무시", "대중교통 정류장 정보. 위와 동일"),
    ("is not part of the route",         "무시", "대중교통 정류장 정보. 위와 동일"),
    ("No node found for reference",      "무시", "bbox 밖 relation 참조. 지도를 잘라 받았으니 정상"),
    ("No way found for reference",       "무시", "bbox 밖 relation 참조. 위와 동일"),
    ("restriction relation",             "영향", "회전 금지(turn restriction)가 무시됨 → 실제로는 못 하는 좌회전을 차량이 함"),
    ("Minor green",                      "영향", "비보호 좌회전 통과 속도가 과대 설정 → 교차로 지연이 실제보다 짧게 나옴"),
    ("does not control any links",       "경미", "링크 없는 신호등 1개 미생성. 통행에 영향 없음"),
    ("Could not build program",          "경미", "신호 프로그램 생성 실패. tls.guess가 대체 프로그램을 넣음"),
    ("Removed a road without junctions", "경미", "교차로에 안 붙은 도로 조각 제거. SCC 필터가 어차피 걸러냄"),
    ("Not joining junctions",            "경미", "교차로 병합 생략. 형상만 달라짐"),
    ("Shape for junction",               "경미", "교차로 형상 보정. 통행 로직 무관"),
    ("Found angle of",                   "경미", "급격한 도로 굴절. 통행 가능"),
    ("Speed of straight connection",     "경미", "회전 반경에 따른 속도 감소. netconvert의 정상 보정"),
    ("Intersecting left turns",          "경미", "좌회전 교차. 교차로 반경 문제, 통행 가능"),
    ("Could not build off-ramp",         "경미", "램프 생성 실패. 일반 도로로 유지됨"),
    ("Discarding unknown compound",      "경미", "OSM 태그 조합 미인식. 기본 도로 타입 적용"),
]

TOTAL_RE = re.compile(r"^Warning:\s+(\d+) total messages of type:\s*(.+)$")


def classify(text):
    low = text.lower()
    for key, verdict, reason in RULES:
        if key.lower() in low:
            return verdict, reason
    return "미분류", "규칙에 없음 — 직접 확인 필요"


def normalize(text):
    text = re.sub(r"'[^']*'", "'X'", text)
    # netconvert의 총계 라인은 값 자리에 %를 쓰고, 개별 라인은 실제 숫자를 쓴다.
    # 두 형태를 같은 유형으로 묶기 위해 둘 다 N으로 정규화한다.
    text = re.sub(r"[-+]?\d+(\.\d+)?", "N", text)
    text = text.replace("%", "N")
    return text.strip()


def main(path):
    counts = Counter()
    explicit_totals = {}
    for raw in open(path, encoding="utf-8", errors="replace"):
        line = raw.rstrip("\n")
        if not line.startswith("Warning:"):
            continue
        m = TOTAL_RE.match(line)
        if m:
            explicit_totals[normalize("Warning: " + m.group(2))] = int(m.group(1))
        else:
            counts[normalize(line)] += 1

    merged = {}
    for key, n in counts.items():
        merged[key] = n
    for key, n in explicit_totals.items():
        # 총계 라인이 있으면 그 값이 진짜 발생 횟수
        merged[key] = n

    groups = {"영향": [], "경미": [], "무시": [], "미분류": []}
    for key, n in merged.items():
        verdict, reason = classify(key)
        groups[verdict].append((n, key, reason))

    total = sum(merged.values())
    print(f"총 경고 {total}건 / 유형 {len(merged)}종\n")
    label = {"영향": "■ 시뮬레이션에 실제 영향",
             "경미": "▲ 영향 경미",
             "무시": "· 무시 가능",
             "미분류": "? 미분류"}
    for v in ("영향", "경미", "무시", "미분류"):
        rows = sorted(groups[v], reverse=True)
        if not rows:
            continue
        n_sum = sum(r[0] for r in rows)
        print(f"{label[v]}  ({n_sum}건, {100*n_sum/total:.0f}%)")
        for n, key, reason in rows:
            print(f"   {n:>4}회  {key[9:80]}")
            print(f"          → {reason}")
        print()
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
