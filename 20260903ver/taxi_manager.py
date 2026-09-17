import random
import traci

CATEGORY_PEAK_HOURS = {
    "company": [8, 9, 12, 18, 19, 20, 21, 22, 23],
    "school": [7, 8, 16],
    "residential": [7, 19, 22],
    "restaurant": [12, 13, 18, 19, 20],
    "subway_entrance": [7, 8, 9, 18, 19],
    "bus_stop": [7, 8, 9, 18, 19],
}

PRIMARY_PROB = 0.7


class TaxiFleetManager:
    def __init__(self, target_count: int, boundary_edges: list, all_edges: list,
                 vtype: str = "taxi_type", strategy: str = "patrol",
                 hotspot_edges: list = None, zones: dict = None,
                 sim_start_hour: float = 0, remaining_edges_threshold: int = 2, **kwargs):
        self.target_count = target_count
        self.all_edges = all_edges
        self.vtype = vtype
        self.strategy = strategy
        self.hotspot_edges = hotspot_edges if hotspot_edges else all_edges
        self.zones = zones or {}
        self.sim_start_hour = sim_start_hour
        self.remaining_edges_threshold = remaining_edges_threshold

    def _current_hour(self, now_seconds: float) -> float:
        return self.sim_start_hour + (now_seconds / 3600.0)

    def _category_busyness_ranking(self, now_seconds: float) -> list:
        now_hour = self._current_hour(now_seconds) % 24
        scored = []
        for category, peak_hours in CATEGORY_PEAK_HOURS.items():
            min_diff = min(abs((now_hour - p + 12) % 24 - 12) for p in peak_hours)
            score = 3.0 - min_diff
            scored.append((category, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [cat for cat, _ in scored]

    def _current_hotspot_pools(self, now_seconds: float):
        if not self.zones:
            return self.hotspot_edges, self.hotspot_edges

        ranking = self._category_busyness_ranking(now_seconds)
        primary_pool, secondary_pool = [], []

        idx = 0
        while idx < len(ranking) and not primary_pool:
            primary_pool = self.zones.get(ranking[idx], [])
            idx += 1
        while idx < len(ranking) and not secondary_pool:
            secondary_pool = self.zones.get(ranking[idx], [])
            idx += 1

        if not primary_pool:
            primary_pool = self.hotspot_edges or self.all_edges
        if not secondary_pool:
            secondary_pool = primary_pool

        return primary_pool, secondary_pool

    def _pick_target_edge(self, now_seconds: float, current_edge: str) -> str:
        if self.strategy == "prepositioned":
            primary_pool, secondary_pool = self._current_hotspot_pools(now_seconds)
            pool = primary_pool if random.random() < PRIMARY_PROB else secondary_pool
        else:
            pool = self.all_edges

        for _ in range(10):
            candidate = random.choice(pool)
            if candidate == current_edge:
                continue
            try:
                route = traci.simulation.findRoute(current_edge, candidate)
                if route and len(route.edges) > 0:
                    return candidate
            except Exception:
                continue
        return current_edge

    def maintain(self, now_seconds: float = 0):
        try:
            for vid in traci.vehicle.getIDList():
                if traci.vehicle.getTypeID(vid) != self.vtype:
                    continue

                if traci.vehicle.getPersonIDList(vid):
                    continue

                route = traci.vehicle.getRoute(vid)
                route_idx = traci.vehicle.getRouteIndex(vid)
                remaining = len(route) - route_idx - 1

                if remaining <= self.remaining_edges_threshold:
                    current_edge = traci.vehicle.getRoadID(vid)
                    new_target = self._pick_target_edge(now_seconds, current_edge)
                    if new_target and new_target != current_edge:
                        traci.vehicle.changeTarget(vid, new_target)
        except traci.exceptions.TraCIException:
            pass