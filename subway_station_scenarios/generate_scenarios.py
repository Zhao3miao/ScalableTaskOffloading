import argparse
import math
import os
import random
import sys
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import networkx as nx

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from campus_scenarios.generate_scenarios import OSMScenarioGenerator


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _coord_in_bounds(lat: float, lon: float, bounds: Dict[str, float], margin: float = 0.0) -> bool:
    return (
        bounds["min_lat"] - margin <= lat <= bounds["max_lat"] + margin
        and bounds["min_lon"] - margin <= lon <= bounds["max_lon"] + margin
    )


def _haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius = 6371000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return 2 * earth_radius * math.asin(math.sqrt(a))


class SubwayStationScenarioGenerator(OSMScenarioGenerator):
    """
    Dynamic subway-station scenario generator.

    Design principles:
    - Build a local walkable graph from the raw OSM file instead of using the
      full railway line graph.
    - Treat subway entrances as source/sink anchors, not as long-range
      in-scene movement paths.
    - Generate event-driven users tied to train arrival/departure, plus a
      background population that is not bound to the subway schedule.
    """

    LOCAL_BOUNDS = {
        "min_lat": 36.54721,
        "max_lat": 36.55846,
        "min_lon": 116.79055,
        "max_lon": 116.80645,
    }
    GRAPH_MARGIN = 0.0008

    STEP_SECONDS = 1.0
    TRAIN_ARRIVAL_STEP = 60
    TRAIN_DEPARTURE_STEP = 240
    TOTAL_STEPS_DEFAULT = 300
    SUBWAY_TO_ROAD_SPEED_RANGE = (3.00, 5.00)
    ROAD_TO_SUBWAY_SPEED_RANGE = (3.00, 6.00)

    WALKABLE_HIGHWAYS = {
        "footway",
        "path",
        "steps",
        "pedestrian",
        "service",
        "platform",
        "living_street",
        "residential",
        "unclassified",
        "tertiary",
        "secondary",
        "primary",
    }

    SUBWAY_CENTER = {
        "subway": (36.552180, 116.798420),
    }

    ROAD_ANCHORS = {
        "west_road": [(36.55420, 116.79255)],
        "east_road": [(36.55250, 116.80497)],
        "north_road": [(36.55790, 116.79962)],
        "south_road": [(36.54779, 116.79650)],
    }

    def __init__(self, osm_file_path: str):
        super().__init__(osm_file_path)
        self.scene_bounds_degrees = dict(self.LOCAL_BOUNDS)
        self.osm_graph = nx.Graph()
        self.node_latlon: Dict[int, Tuple[float, float]] = {}
        self.anchor_nodes_by_group: Dict[str, List[int]] = {}
        self.station_core_nodes: List[int] = []
        self.departure_zone_nodes: List[int] = []
        self.main_road_paths: List[List[int]] = []
        self.main_road_nodes: List[int] = []
        self.gate_nodes = []
        self.gate_nodes_by_name = {}

    def load_osm_data(self):
        print("Loading local subway-station walkable graph...")
        start_time = self._safe_time()

        tree = ET.parse(self.osm_file_path)
        root = tree.getroot()

        all_nodes: Dict[int, Tuple[float, float]] = {}
        for node_elem in root.findall("node"):
            node_id = int(node_elem.attrib["id"])
            lat = float(node_elem.attrib["lat"])
            lon = float(node_elem.attrib["lon"])
            all_nodes[node_id] = (lat, lon)

        graph = nx.Graph()
        kept_edge_count = 0
        margin = self.GRAPH_MARGIN

        for way_elem in root.findall("way"):
            tags = {
                tag.attrib["k"]: tag.attrib["v"]
                for tag in way_elem.findall("tag")
            }
            if not self._is_walkable_way(tags):
                continue

            refs = []
            for nd_elem in way_elem.findall("nd"):
                ref = int(nd_elem.attrib["ref"])
                if ref in all_nodes:
                    refs.append(ref)

            if len(refs) < 2:
                continue

            for u, v in zip(refs, refs[1:]):
                lat_u, lon_u = all_nodes[u]
                lat_v, lon_v = all_nodes[v]

                if not (
                    _coord_in_bounds(lat_u, lon_u, self.LOCAL_BOUNDS, margin)
                    or _coord_in_bounds(lat_v, lon_v, self.LOCAL_BOUNDS, margin)
                ):
                    continue

                length = _haversine_meters(lat_u, lon_u, lat_v, lon_v)
                if length <= 0:
                    continue

                graph.add_node(u, y=lat_u, x=lon_u)
                graph.add_node(v, y=lat_v, x=lon_v)

                edge_data = {
                    "length": length,
                    "name": tags.get("name"),
                    "highway": tags.get("highway"),
                    "way_id": way_elem.attrib.get("id"),
                }

                if graph.has_edge(u, v):
                    if length < graph[u][v]["length"]:
                        graph[u][v].update(edge_data)
                else:
                    graph.add_edge(u, v, **edge_data)
                    kept_edge_count += 1

        if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
            raise RuntimeError("No walkable local graph could be built from subway_station.osm")

        # Keep only the largest connected component to avoid stray local fragments.
        largest_component = max(nx.connected_components(graph), key=len)
        graph = graph.subgraph(largest_component).copy()

        self.osm_graph = graph
        self.node_latlon = {
            node_id: (attrs["y"], attrs["x"])
            for node_id, attrs in graph.nodes(data=True)
        }
        self.road_nodes = list(graph.nodes())
        self.road_edges = list(graph.edges())

        self._precompute_node_positions_local()
        self._precompute_adjacent_nodes()
        self._build_anchor_groups()

        load_time = self._safe_time() - start_time
        print(
            f"Walkable graph ready: {graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges, load time {load_time:.2f}s"
        )

    def _safe_time(self) -> float:
        import time

        return time.time()

    def _is_walkable_way(self, tags: Dict[str, str]) -> bool:
        if tags.get("railway") == "subway":
            return False
        if tags.get("route") == "subway":
            return False
        if tags.get("area") == "yes":
            return False

        highway = tags.get("highway")
        if highway is None:
            return False
        return highway in self.WALKABLE_HIGHWAYS

    def _precompute_node_positions_local(self):
        print("Precomputing node positions...")
        min_lon = self.scene_bounds_degrees["min_lon"]
        min_lat = self.scene_bounds_degrees["min_lat"]
        center_lat = (self.scene_bounds_degrees["min_lat"] + self.scene_bounds_degrees["max_lat"]) / 2.0

        self.node_positions_cache = {}
        for node_id, (lat, lon) in self.node_latlon.items():
            x_meters = self._lon_to_meters(lon, min_lon, center_lat)
            y_meters = self._lat_to_meters(lat, min_lat)
            self.node_positions_cache[node_id] = (x_meters, y_meters)

    def _get_scene_bounds_degrees(self) -> Tuple[float, float, float, float]:
        return (
            self.scene_bounds_degrees["min_lon"],
            self.scene_bounds_degrees["min_lat"],
            self.scene_bounds_degrees["max_lon"],
            self.scene_bounds_degrees["max_lat"],
        )

    def _build_anchor_groups(self):
        self.gate_nodes_by_name = {}
        all_subway_nodes = []

        for name, (lat, lon) in self.SUBWAY_CENTER.items():
            nodes = self._find_nodes_near_coordinate(lat, lon, radius_m=90.0)
            self.gate_nodes_by_name[name] = nodes
            all_subway_nodes.extend(nodes)

        self.gate_nodes = sorted(set(all_subway_nodes))
        self.anchor_nodes_by_group["subway"] = self.gate_nodes.copy()

        for group_name, coord_list in self.ROAD_ANCHORS.items():
            nodes = []
            for lat, lon in coord_list:
                nodes.extend(self._find_nodes_near_coordinate(lat, lon, radius_m=90.0))
            self.anchor_nodes_by_group[group_name] = sorted(set(nodes))

        # Station core: subway entrances plus nodes around BS2.
        core_nodes = set(self.gate_nodes)
        bs2_lat = SUBWAY_BASE_STATIONS[1]["position"][1]
        bs2_lon = SUBWAY_BASE_STATIONS[1]["position"][0]
        core_nodes.update(self._find_nodes_near_coordinate(bs2_lat, bs2_lon, radius_m=120.0))
        self.station_core_nodes = sorted(core_nodes)
        self.departure_zone_nodes = sorted(
            set(self._find_nodes_near_coordinate(bs2_lat, bs2_lon, radius_m=70.0))
        )

        # Fallbacks in case some anchor groups collapse to the same sparse local nodes.
        all_nodes = list(self.osm_graph.nodes())
        if not self.anchor_nodes_by_group["subway"]:
            self.anchor_nodes_by_group["subway"] = all_nodes[:]
        if not self.station_core_nodes:
            self.station_core_nodes = self.anchor_nodes_by_group["subway"][:]
        if not self.departure_zone_nodes:
            self.departure_zone_nodes = self.station_core_nodes[:]
        for group_name in self.ROAD_ANCHORS:
            if not self.anchor_nodes_by_group[group_name]:
                self.anchor_nodes_by_group[group_name] = all_nodes[:]
        self.main_road_nodes = self._build_main_road_nodes()

    def _build_main_road_nodes(self) -> List[int]:
        self.main_road_paths = []
        for start_group, target_group in (("west_road", "east_road"), ("south_road", "north_road")):
            path = self._shortest_anchor_group_path(start_group, target_group)
            if path:
                self.main_road_paths.append(path)

        nodes = sorted({node_id for path in self.main_road_paths for node_id in path})
        return nodes if nodes else list(self.osm_graph.nodes())

    def _shortest_anchor_group_path(self, start_group: str, target_group: str) -> List[int]:
        start_candidates = self.anchor_nodes_by_group.get(start_group, [])
        target_candidates = self.anchor_nodes_by_group.get(target_group, [])
        best_path = []
        best_length = float("inf")

        for start_node in start_candidates:
            for target_node in target_candidates:
                try:
                    path = nx.shortest_path(self.osm_graph, start_node, target_node, weight="length")
                    length = nx.path_weight(self.osm_graph, path, weight="length")
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                if length < best_length:
                    best_length = length
                    best_path = path

        return best_path

    def _pick_initial_main_road_node(self) -> int:
        paths = [path for path in self.main_road_paths if path]
        if not paths:
            return random.choice(self.main_road_nodes or list(self.osm_graph.nodes()))

        path = random.choice(paths)
        if len(path) <= 2:
            return random.choice(path)

        edge_lengths = []
        total_length = 0.0
        for u, v in zip(path, path[1:]):
            length = float(self.osm_graph[u][v].get("length", 0.0))
            edge_lengths.append(length)
            total_length += length

        if total_length <= 0:
            return random.choice(path[1:-1])

        target_length = random.uniform(0.0, total_length)
        walked = 0.0
        for idx, edge_length in enumerate(edge_lengths):
            walked += edge_length
            if walked >= target_length:
                return path[min(idx + 1, len(path) - 2)]

        return path[-2]

    def _find_nodes_near_coordinate(self, lat: float, lon: float, radius_m: float) -> List[int]:
        candidates = []
        nearest_node = None
        nearest_distance = float("inf")

        for node_id, (node_lat, node_lon) in self.node_latlon.items():
            distance = _haversine_meters(lat, lon, node_lat, node_lon)
            if distance <= radius_m:
                candidates.append(node_id)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_node = node_id

        if candidates:
            return sorted(set(candidates))
        return [nearest_node] if nearest_node is not None else []

    def generate_scenario(
        self,
        base_stations: List[Dict],
        user_distribution: List[int],
        total_steps: int = 600,
        scenario_name: str = "subway_station_scenario",
        user_speed: float = 1.5,
        speed_range: Tuple[float, float] = None,
        target_direction: float = None,
        direction_strength: float = 0.5,
        base_station_directions: Dict[int, float] = None,
        base_station_direction_strengths: Dict[int, float] = None,
        server_configs_raw: List[Dict] = None,
        gate_spawn_ratio: float = 0.65,
        gate_leave_ratio: float = 0.65,
        initial_active_ratio: float = 0.4,
    ) -> Dict:
        if self.osm_graph.number_of_nodes() == 0:
            self.load_osm_data()

        self._validate_inputs(base_stations, user_distribution)
        total_users = sum(user_distribution)
        base_stations_meters = self._convert_base_stations_to_meters(base_stations)

        mobile_devices = self._generate_dynamic_mobile_devices(
            total_users=total_users,
            total_steps=total_steps,
            base_stations=base_stations_meters,
            gate_spawn_ratio=gate_spawn_ratio,
            gate_leave_ratio=gate_leave_ratio,
            initial_active_ratio=initial_active_ratio,
        )

        scenario_config = self._build_scenario_config(
            base_stations_meters, mobile_devices, total_steps, scenario_name
        )
        scenario_config["event_config"] = {
            "step_seconds": self.STEP_SECONDS,
            "train_arrival_step": self.TRAIN_ARRIVAL_STEP,
            "train_departure_step": self.TRAIN_DEPARTURE_STEP,
            "departure_zone_radius_m": 70.0,
        }
        return scenario_config

    def _generate_dynamic_mobile_devices(
        self,
        total_users: int,
        total_steps: int,
        base_stations: List[Dict],
        gate_spawn_ratio: float,
        gate_leave_ratio: float,
        initial_active_ratio: float,
    ) -> List[Dict]:
        profile_counts = self._allocate_profile_counts(
            total_users, gate_spawn_ratio, gate_leave_ratio, initial_active_ratio
        )
        profile_sequence = []
        for profile_name, count in profile_counts.items():
            profile_sequence.extend([profile_name] * count)
        random.shuffle(profile_sequence)

        initial_count = min(
            sum(1 for profile_name in profile_sequence if profile_name in {"road_to_subway", "wanderer"}),
            int(round(total_users * _clamp(initial_active_ratio, 0.0, 1.0))),
        )
        initial_candidate_indices = [
            idx
            for idx, profile_name in enumerate(profile_sequence)
            if profile_name in {"road_to_subway", "wanderer"}
        ]
        random.shuffle(initial_candidate_indices)
        initial_indices = set(initial_candidate_indices[:initial_count])

        mobile_devices = []
        for user_id, profile_name in enumerate(profile_sequence):
            device = self._generate_device_for_profile(
                user_id=user_id,
                profile_name=profile_name,
                total_steps=total_steps,
                base_stations=base_stations,
                is_initial=user_id in initial_indices,
            )
            mobile_devices.append(device)

        return mobile_devices

    def _allocate_profile_counts(
        self,
        total_users: int,
        gate_spawn_ratio: float,
        gate_leave_ratio: float,
        initial_active_ratio: float,
    ) -> Dict[str, int]:
        road_to_subway_share = 0.45
        subway_to_road_share = 0.25
        wanderer_share = 0.30

        raw_shares = {
            "road_to_subway": road_to_subway_share,
            "subway_to_road": subway_to_road_share,
            "wanderer": wanderer_share,
        }

        total_share = sum(raw_shares.values())
        normalized = {name: value / total_share for name, value in raw_shares.items()}

        counts = {name: int(total_users * share) for name, share in normalized.items()}
        assigned = sum(counts.values())
        leftovers = total_users - assigned

        if leftovers > 0:
            ranked = sorted(
                normalized.items(),
                key=lambda item: (total_users * item[1] - counts[item[0]]),
                reverse=True,
            )
            for idx in range(leftovers):
                counts[ranked[idx % len(ranked)][0]] += 1

        for name in counts:
            counts[name] = max(1, counts[name])

        # Trim back to the exact user count while preserving at least one user per profile.
        while sum(counts.values()) > total_users:
            largest = max(counts, key=lambda key: counts[key])
            if counts[largest] > 1:
                counts[largest] -= 1
            else:
                break

        return counts

    def _generate_device_for_profile(
        self,
        user_id: int,
        profile_name: str,
        total_steps: int,
        base_stations: List[Dict],
        is_initial: bool,
    ) -> Dict:
        if profile_name == "subway_to_road":
            spawn_step, leave_step, trajectory, speed = self._build_subway_to_road_trajectory(total_steps)
        elif profile_name == "road_to_subway":
            spawn_step, leave_step, trajectory, speed = self._build_road_to_subway_trajectory(
                total_steps, is_initial
            )
        else:
            spawn_step, leave_step, trajectory, speed = self._build_wanderer_trajectory(
                total_steps, is_initial
            )

        if not trajectory:
            fallback_node = random.choice(list(self.osm_graph.nodes()))
            trajectory = self._hold_segment(fallback_node, 0, min(total_steps - 1, 30))
            spawn_step = trajectory[0]["step"]
            leave_step = trajectory[-1]["step"]
            speed = 0.5

        first_position = trajectory[0]["position"]
        base_station_id = self._nearest_base_station_id(first_position, base_stations)
        tasks = self._generate_tasks(float(spawn_step), float(leave_step), time_interval=1.0)

        return {
            "id": user_id,
            "profile": profile_name,
            "spawn_time": float(spawn_step),
            "leave_time": float(leave_step),
            "trajectory": trajectory,
            "base_station_id": base_station_id,
            "speed": speed,
            "tasks": tasks,
        }

    def _build_waiting_subway_trajectory(self, total_steps: int):
        start_node = self._pick_anchor_node("subway")
        speed = random.uniform(0.35, 0.80)
        leave_step = min(total_steps - 1, self.TRAIN_DEPARTURE_STEP + random.randint(-15, 20))
        trajectory = self._build_loiter_trajectory(
            start_node=start_node,
            candidate_nodes=self.station_core_nodes,
            start_step=0,
            end_step=leave_step,
            speed=speed,
            hold_range=(10, 35),
        )
        return 0, leave_step, trajectory, speed

    def _build_subway_to_road_trajectory(self, total_steps: int):
        spawn_step = min(total_steps, self.TRAIN_ARRIVAL_STEP)
        road_group = self._pick_road_group()
        start_node, target_node = self._pick_forward_start_and_target(
            start_candidates=self.anchor_nodes_by_group.get("subway", self.station_core_nodes),
            target_candidates=self.anchor_nodes_by_group.get(road_group, list(self.osm_graph.nodes())),
        )
        speed = random.uniform(*self.SUBWAY_TO_ROAD_SPEED_RANGE)

        path_segment, current_node, current_step = self._path_segment(
            start_node, target_node, spawn_step, speed, total_steps
        )
        trajectory = path_segment
        return spawn_step, trajectory[-1]["step"], trajectory, speed

    def _build_road_to_subway_trajectory(self, total_steps: int, is_initial: bool):
        spawn_step = 0 if is_initial else random.randint(1, total_steps)
        if is_initial:
            start_candidates = [self._pick_initial_main_road_node()]
        else:
            road_group = self._pick_road_group()
            start_candidates = self.anchor_nodes_by_group.get(road_group, list(self.osm_graph.nodes()))
        start_node, target_node = self._pick_forward_start_and_target(
            start_candidates=start_candidates,
            target_candidates=self.departure_zone_nodes,
        )
        speed = random.uniform(*self.ROAD_TO_SUBWAY_SPEED_RANGE)

        path_segment, current_node, current_step = self._path_segment(
            start_node, target_node, spawn_step, speed, total_steps
        )
        trajectory = path_segment

        if current_node in self.departure_zone_nodes:
            if current_step <= self.TRAIN_DEPARTURE_STEP:
                if current_step < self.TRAIN_DEPARTURE_STEP:
                    trajectory = self._merge_segments(
                        trajectory,
                        self._hold_segment(current_node, current_step, self.TRAIN_DEPARTURE_STEP),
                    )
                trajectory = [pt for pt in trajectory if pt["step"] <= self.TRAIN_DEPARTURE_STEP]
                return spawn_step, min(self.TRAIN_DEPARTURE_STEP, total_steps), trajectory, speed

            if current_step < total_steps:
                trajectory = self._merge_segments(
                    trajectory,
                    self._hold_segment(current_node, current_step, total_steps),
                )

        return spawn_step, trajectory[-1]["step"], trajectory, speed

    def _build_wanderer_trajectory(self, total_steps: int, is_initial: bool):
        spawn_step = 0 if is_initial else random.randint(1, total_steps)

        if is_initial:
            start_candidates = [self._pick_initial_main_road_node()]
            target_group = self._pick_road_group()
            target_candidates = self.anchor_nodes_by_group.get(target_group, list(self.osm_graph.nodes()))
        else:
            start_group = self._pick_road_group()
            target_group = self._pick_road_group(exclude=start_group)
            start_candidates = self.anchor_nodes_by_group.get(start_group, list(self.osm_graph.nodes()))
            target_candidates = self.anchor_nodes_by_group.get(target_group, list(self.osm_graph.nodes()))

        start_node, target_node = self._pick_forward_start_and_target(
            start_candidates=start_candidates,
            target_candidates=target_candidates,
        )
        speed = self._sample_background_speed()

        path_segment, current_node, current_step = self._path_segment(
            start_node, target_node, spawn_step, speed, total_steps
        )
        trajectory = path_segment
        return spawn_step, trajectory[-1]["step"], trajectory, speed

    def _sample_background_speed(self) -> float:
        roll = random.random()
        if roll < 0.45:
            # Pedestrians
            return random.uniform(0.9, 1.6)
        if roll < 0.80:
            # Bicycles / scooters
            return random.uniform(2.0, 4.0)
        # Vehicles on the surrounding roads
        return random.uniform(4.5, 7.0)

    def _shortest_path_length(self, start_node: int, target_node: int) -> float:
        try:
            path = nx.shortest_path(self.osm_graph, start_node, target_node, weight="length")
        except nx.NetworkXNoPath:
            x1, y1 = self.node_positions_cache[start_node]
            x2, y2 = self.node_positions_cache[target_node]
            return math.hypot(x2 - x1, y2 - y1)

        total_length = 0.0
        for u, v in zip(path, path[1:]):
            total_length += float(self.osm_graph[u][v].get("length", 0.0))
        return total_length

    def _pick_anchor_node(self, group_name: str) -> int:
        nodes = self.anchor_nodes_by_group.get(group_name) or self.station_core_nodes or list(self.osm_graph.nodes())
        return random.choice(nodes)

    def _pick_road_group(self, exclude: Optional[str] = None) -> str:
        groups = [name for name in self.ROAD_ANCHORS if name != exclude]
        return random.choice(groups)

    def _nearby_candidate_nodes(self, node_id: int, radius_m: float) -> List[int]:
        x0, y0 = self.node_positions_cache[node_id]
        candidates = []
        for other_id, (x, y) in self.node_positions_cache.items():
            if math.hypot(x - x0, y - y0) <= radius_m:
                candidates.append(other_id)
        return candidates if candidates else [node_id]

    def _pick_forward_start_and_target(
        self,
        start_candidates: List[int],
        target_candidates: List[int],
        max_attempts: int = 24,
    ) -> Tuple[int, int]:
        start_candidates = start_candidates or list(self.osm_graph.nodes())
        target_candidates = target_candidates or list(self.osm_graph.nodes())

        fallback_pair = (random.choice(start_candidates), random.choice(target_candidates))
        fallback_score = float("inf")

        for _ in range(max_attempts):
            start_node = random.choice(start_candidates)
            target_node = random.choice(target_candidates)
            if start_node == target_node:
                return start_node, target_node

            try:
                path = nx.shortest_path(self.osm_graph, start_node, target_node, weight="length")
            except nx.NetworkXNoPath:
                continue

            if len(path) < 2:
                return start_node, target_node

            start_pos = self.node_positions_cache[path[0]]
            next_pos = self.node_positions_cache[path[1]]
            target_pos = self.node_positions_cache[target_node]

            start_dist = math.hypot(start_pos[0] - target_pos[0], start_pos[1] - target_pos[1])
            next_dist = math.hypot(next_pos[0] - target_pos[0], next_pos[1] - target_pos[1])
            score = next_dist - start_dist

            if score <= 0:
                return start_node, target_node

            if score < fallback_score:
                fallback_score = score
                fallback_pair = (start_node, target_node)

        return fallback_pair

    def _build_loiter_trajectory(
        self,
        start_node: int,
        candidate_nodes: List[int],
        start_step: int,
        end_step: int,
        speed: float,
        hold_range: Tuple[int, int],
    ) -> List[Dict]:
        if end_step < start_step:
            return []

        candidate_nodes = candidate_nodes or [start_node]
        trajectory = []
        current_node = start_node
        current_step = start_step

        while current_step < end_step:
            hold_duration = random.randint(hold_range[0], hold_range[1])
            hold_end = min(end_step, current_step + hold_duration)
            trajectory = self._merge_segments(
                trajectory, self._hold_segment(current_node, current_step, hold_end)
            )
            current_step = hold_end
            if current_step >= end_step:
                break

            destination_pool = [node for node in candidate_nodes if node != current_node]
            if not destination_pool:
                destination_pool = candidate_nodes
            destination_node = random.choice(destination_pool)

            path_segment, current_node, current_step = self._path_segment(
                current_node, destination_node, current_step, speed, end_step
            )
            trajectory = self._merge_segments(trajectory, path_segment)

        return trajectory

    def _hold_segment(self, node_id: int, start_step: int, end_step: int) -> List[Dict]:
        if end_step < start_step:
            return []
        x, y = self.node_positions_cache[node_id]
        return [
            {
                "step": step,
                "time": float(step),
                "position": [x, y],
                "node": node_id,
            }
            for step in range(start_step, end_step + 1)
        ]

    def _path_segment(
        self,
        start_node: int,
        target_node: int,
        start_step: int,
        speed: float,
        max_end_step: int,
    ) -> Tuple[List[Dict], int, int]:
        if max_end_step < start_step:
            return [], start_node, start_step

        if start_node == target_node:
            segment = self._hold_segment(start_node, start_step, min(start_step, max_end_step))
            return segment, start_node, segment[-1]["step"] if segment else start_step

        try:
            path = nx.shortest_path(self.osm_graph, start_node, target_node, weight="length")
        except nx.NetworkXNoPath:
            segment = self._hold_segment(start_node, start_step, min(start_step, max_end_step))
            return segment, start_node, segment[-1]["step"] if segment else start_step

        step = start_step
        segment = [self._point(step, start_node, self.node_positions_cache[start_node])]
        current_node = start_node

        for u, v in zip(path, path[1:]):
            x1, y1 = self.node_positions_cache[u]
            x2, y2 = self.node_positions_cache[v]
            distance = math.hypot(x2 - x1, y2 - y1)
            seg_steps = max(1, int(math.ceil(distance / max(speed, 0.20))))

            for idx in range(1, seg_steps + 1):
                if step >= max_end_step:
                    return segment, current_node, step
                step += 1
                ratio = idx / seg_steps
                x = x1 + (x2 - x1) * ratio
                y = y1 + (y2 - y1) * ratio
                current_node = v if idx == seg_steps else u
                segment.append(self._point(step, current_node, (x, y)))

        return segment, path[-1], step

    def _point(self, step: int, node_id: int, position: Tuple[float, float]) -> Dict:
        return {
            "step": int(step),
            "time": float(step),
            "position": [float(position[0]), float(position[1])],
            "node": int(node_id),
        }

    def _merge_segments(self, base_segment: List[Dict], new_segment: List[Dict]) -> List[Dict]:
        if not new_segment:
            return base_segment
        if not base_segment:
            return list(new_segment)
        if base_segment[-1]["step"] == new_segment[0]["step"]:
            base_segment.extend(new_segment[1:])
        else:
            base_segment.extend(new_segment)
        return base_segment

    def _nearest_base_station_id(self, position: List[float], base_stations: List[Dict]) -> int:
        best_id = base_stations[0]["id"]
        best_distance = float("inf")
        for bs in base_stations:
            x, y = bs["position"]
            distance = math.hypot(position[0] - x, position[1] - y)
            if distance < best_distance:
                best_distance = distance
                best_id = bs["id"]
        return best_id


SUBWAY_BASE_STATIONS: List[Dict] = [
    {"id": 1, "position": [116.792800, 36.554800]},
    {"id": 2, "position": [116.798350, 36.551980]},
    {"id": 3, "position": [116.803800, 36.548800]},
]


def _build_user_distribution(num_users: int, num_bs: int) -> List[int]:
    distribution = [0] * num_bs
    for idx in range(num_users):
        distribution[idx % num_bs] += 1
    return distribution


def generate_randomized_subway_scenario_config(
    osm_file_path: str,
    base_stations: List[Dict],
    users_range: List[int],
    total_time: float,
    scenario_id: str,
    output_path: Optional[str] = None,
    index: int = 0,
    gate_spawn_ratio: float = 0.65,
    gate_leave_ratio: float = 0.65,
    initial_active_ratio: float = 0.40,
) -> Dict:
    generator = SubwayStationScenarioGenerator(osm_file_path)
    num_users = random.randint(users_range[0], users_range[1])
    user_distribution = _build_user_distribution(num_users, len(base_stations))

    scenario_config = generator.generate_scenario(
        base_stations=base_stations,
        user_distribution=user_distribution,
        total_steps=int(total_time),
        scenario_name=f"{scenario_id}_{index}",
        gate_spawn_ratio=gate_spawn_ratio,
        gate_leave_ratio=gate_leave_ratio,
        initial_active_ratio=initial_active_ratio,
    )

    scenario_config["initial_connections"] = {
        device["id"]: device["base_station_id"]
        for device in scenario_config["mobile_devices"]
    }

    if output_path:
        generator.save_scenario_to_file(scenario_config, output_path)

    return scenario_config


def main():
    parser = argparse.ArgumentParser(
        description="Generate subway-station OSM scenarios for ScalableMEC."
    )
    parser.add_argument(
        "--osm_file",
        type=str,
        default=os.path.join(CURRENT_DIR, "subway_station.osm"),
        help="Path to the subway station OSM file.",
    )
    parser.add_argument(
        "--scenario_id",
        type=str,
        default="subway_station_max_ue_50",
        help="Scenario folder prefix.",
    )
    parser.add_argument(
        "--num_scenarios",
        type=int,
        default=10,
        help="Number of scenario YAML files to generate.",
    )
    parser.add_argument(
        "--max_users",
        type=int,
        default=50,
        help="Maximum number of users in each scenario.",
    )
    parser.add_argument(
        "--min_users",
        type=int,
        default=None,
        help="Minimum number of users in each scenario. Defaults to max_users.",
    )
    parser.add_argument(
        "--total_time",
        type=float,
        default=float(SubwayStationScenarioGenerator.TOTAL_STEPS_DEFAULT),
        help="Total simulation steps. Default 300 steps equals 5 minutes.",
    )
    parser.add_argument(
        "--gate_spawn_ratio",
        type=float,
        default=0.65,
        help="Controls the share of users associated with subway arrival dynamics.",
    )
    parser.add_argument(
        "--gate_leave_ratio",
        type=float,
        default=0.65,
        help="Controls the share of users associated with entering the subway and leaving the area.",
    )
    parser.add_argument(
        "--initial_active_ratio",
        type=float,
        default=0.40,
        help="Controls the initial in-station user density near the subway core.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=CURRENT_DIR,
        help="Root output directory.",
    )

    args = parser.parse_args()
    if args.min_users is None:
        args.min_users = args.max_users

    output_dir = os.path.join(args.output_root, args.scenario_id)
    os.makedirs(output_dir, exist_ok=True)

    print(
        f"Generating {args.num_scenarios} subway-station scenarios for '{args.scenario_id}'..."
    )
    for i in range(args.num_scenarios):
        output_path = os.path.join(output_dir, f"scenario_{i}.yaml")
        random.seed(i)
        generate_randomized_subway_scenario_config(
            osm_file_path=args.osm_file,
            base_stations=SUBWAY_BASE_STATIONS,
            users_range=[args.min_users, args.max_users],
            total_time=args.total_time,
            scenario_id=args.scenario_id,
            output_path=output_path,
            index=i,
            gate_spawn_ratio=args.gate_spawn_ratio,
            gate_leave_ratio=args.gate_leave_ratio,
            initial_active_ratio=args.initial_active_ratio,
        )
        print(f"  - Generated {output_path}")

    print("\nSubway-station scenario generation completed!")


if __name__ == "__main__":
    main()
