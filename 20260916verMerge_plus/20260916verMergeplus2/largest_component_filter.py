"""
SUMO 도로망(net)에서 "서로 왕복 가능한" 가장 큰 도로 덩어리(강결합 컴포넌트, SCC)만
남기고 나머지 고립된 조각을 걸러내는 유틸.

[왜 필요한가]
netconvert의 --remove-edges.isolated는 완전히 동떨어진 단일 edge 하나만 잡아준다.
하지만 실제로 문제가 되는 건 "그 지역 안에서는 서로 연결돼 있지만, 전체 지도의
다른 큰 덩어리와는 편도로만(또는 전혀) 연결 안 된 도로 섬"이다.
이런 곳에 택시나 승객이 스폰/목적지로 배정되면 findRoute/changeTarget이
계속 실패해서 "No connection between edge ... found" 경고가 쌓이고,
결국 taxi:dispatch 재라우팅 실패 -> SUMO가 fatal error로 죽는 원인이 된다.

[해결 방법]
edge 그래프(각 edge -> getOutgoing()으로 이어지는 edge들)를 networkx DiGraph로 만들고
강결합 컴포넌트(Strongly Connected Component)를 구한다. 강결합 컴포넌트는
"그 안의 어떤 두 edge든 서로 왕복 가능한" 집합이므로, 가장 큰 SCC만 남기면
findRoute 실패가 원천적으로 사라진다 (그 컴포넌트 내부에서는 항상 경로가 존재).

[사용법] build_env.py에서 net 읽은 직후:
    from largest_component_filter import filter_to_largest_scc
    edges = filter_to_largest_scc(net, edges)
"""

import networkx as nx


def filter_to_largest_scc(net, edge_ids: list) -> list:
    """
    edge_ids(문자열 id 리스트)를 net 객체의 실제 연결 정보로 그래프를 만들어
    가장 큰 강결합 컴포넌트에 속한 edge id만 반환한다.

    net: sumolib.net.readNet()의 결과
    edge_ids: 후보 edge id 리스트 (예: build_env.py의 edges 변수)
    반환: 가장 큰 SCC에 속하는 edge id 리스트 (원본보다 같거나 작음)
    """
    if not edge_ids:
        return edge_ids

    edge_id_set = set(edge_ids)
    graph = nx.DiGraph()
    graph.add_nodes_from(edge_ids)

    for eid in edge_ids:
        try:
            edge_obj = net.getEdge(eid)
        except Exception:
            continue
        for out_edge in edge_obj.getOutgoing():
            out_id = out_edge.getID()
            # 후보 목록(edge_ids)에 없는 edge(internal 등)로 가는 연결은 그래프에 안 넣음
            if out_id in edge_id_set:
                graph.add_edge(eid, out_id)

    if graph.number_of_nodes() == 0:
        print("[안내] 그래프가 비어있어 원본 edge 목록을 그대로 사용합니다.")
        return edge_ids

    sccs = list(nx.strongly_connected_components(graph))
    if not sccs:
        print("[안내] 강결합 컴포넌트를 찾지 못해 원본 edge 목록을 그대로 사용합니다.")
        return edge_ids

    largest_scc = max(sccs, key=len)

    removed = len(edge_ids) - len(largest_scc)
    if removed > 0:
        pct = removed / len(edge_ids) * 100
        print(f"[안내] 고립 도로망 필터링: 전체 {len(edge_ids)}개 중 "
              f"{removed}개({pct:.1f}%) edge를 고립 컴포넌트로 판단해 제외 "
              f"(남은 도로: {len(largest_scc)}개, 서로 왕복 가능함이 보장됨)")

        # 필터링이 지나치게 과격하면(예: 절반 이상 삭제) 경고 — bbox가 너무 크거나
        # OSM 원본 자체가 애초에 파편화가 심할 수 있으므로 grid 크기 축소를 권장
        if pct > 40:
            print("[경고] 제외된 도로 비율이 40%를 넘습니다. bbox(grid_x/grid_y/grid_length)를 "
                  "줄여서 다시 시도하는 것을 권장합니다.")

    return [eid for eid in edge_ids if eid in largest_scc]
