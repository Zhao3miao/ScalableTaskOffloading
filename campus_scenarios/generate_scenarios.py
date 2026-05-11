import os
import yaml
import osmnx as ox
import networkx as nx
import uuid
from typing import List, Dict, Tuple
import random
import math
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed


class OSMScenarioGenerator:
    """
    Campus scenario generator based on OSM map data
    Ensures initial positions are legal and movement follows legal paths
    """

    def __init__(self, osm_file_path: str):
        """
        Initialize the scenario generator

        Args:
            osm_file_path: OSM map file path
        """
        self.osm_file_path = osm_file_path
        self.osm_graph = None
        self.road_nodes = None
        self.road_edges = None
        self.node_positions_cache = {}  # Node position cache
        self.adjacent_nodes_cache = {}  # Adjacent nodes cache
        
        # Define campus gate nodes by their exact OSM coordinates
        self.GATE_COORDS = {
            "West_Gate": (36.543179, 116.822931),  # lat, lon
            "North_Gate": (36.551051, 116.828600), # lat, lon
            # Can add more gates here (e.g., East Gate)
        }
        self.gate_nodes = []  # List of actual OSM node IDs corresponding to gates
        self.gate_nodes_by_name = {}  # Dictionary mapping gate name to its nodes

    def load_osm_data(self):
        """Load and parse OSM map data"""
        print("Loading OSM map data...")
        start_time = time.time()

        try:
            # Use osmnx to load OSM file
            self.osm_graph = ox.graph_from_xml(self.osm_file_path)
            # Convert to undirected graph so pedestrians/vehicles can travel both ways (ignoring one-way campus roads)
            self.osm_graph = ox.utils_graph.get_undirected(self.osm_graph)

            # Extract road nodes and edges first so we can query them for clusters
            self.road_nodes, self.road_edges = ox.graph_to_gdfs(self.osm_graph)

            # Map the exact coordinates to the nearest OSM node for gates
            print("Mapping physical gate coordinates to road nodes...")
            for gate_name, coords in self.GATE_COORDS.items():
                lat, lon = coords
                if gate_name == "North_Gate":
                    # Plan 1: North Gate Cluster (Plaza effect) - Find all nodes within 60m radius
                    cluster_radius = 60.0
                    distances = ox.distance.great_circle_vec(lat, lon, self.road_nodes['y'].values, self.road_nodes['x'].values)
                    close_nodes_mask = distances <= cluster_radius
                    close_nodes = self.road_nodes[close_nodes_mask].index.tolist()
                    
                    if close_nodes:
                        self.gate_nodes_by_name[gate_name] = close_nodes
                        print(f"Mapped {gate_name} to a cluster of {len(close_nodes)} nodes (radius {cluster_radius}m)")
                    else:
                        # Fallback just in case radius is too small
                        nearest_node = ox.distance.nearest_nodes(self.osm_graph, X=lon, Y=lat)
                        self.gate_nodes_by_name[gate_name] = [nearest_node]
                        print(f"Mapped {gate_name} to fallback node {nearest_node}")
                else:
                    nearest_node = ox.distance.nearest_nodes(self.osm_graph, X=lon, Y=lat)
                    self.gate_nodes_by_name[gate_name] = [nearest_node]
                    print(f"Mapped {gate_name} to node {nearest_node}")

            for nodes in self.gate_nodes_by_name.values():
                self.gate_nodes.extend(nodes)

            # Precompute node position cache
            self._precompute_node_positions()

            # Precompute adjacent node cache
            self._precompute_adjacent_nodes()

            load_time = time.time() - start_time
            print(f"OSM data loading completed: {len(self.road_nodes)} road nodes, {len(self.road_edges)} road edges")
            print(f"Data loading time: {load_time:.2f} seconds")

        except Exception as e:
            print(f"Error loading OSM data: {e}")
            raise

    def _precompute_node_positions(self):
        """Precompute all node positions in meters"""
        print("Precomputing node positions...")
        scene_bounds = self._get_scene_bounds_degrees()
        min_lon, min_lat, max_lon, max_lat = scene_bounds
        center_lat = (min_lat + max_lat) / 2

        for node_id, node_data in self.road_nodes.iterrows():
            lon = float(node_data.geometry.x)
            lat = float(node_data.geometry.y)

            x_meters = self._lon_to_meters(lon, min_lon, center_lat)
            y_meters = self._lat_to_meters(lat, min_lat)

            self.node_positions_cache[node_id] = (x_meters, y_meters)

    def _precompute_adjacent_nodes(self):
        """Precompute adjacent nodes for each node"""
        print("Precomputing adjacent nodes...")
        for node in self.osm_graph.nodes():
            try:
                neighbors = list(self.osm_graph.neighbors(node))
                self.adjacent_nodes_cache[node] = neighbors
            except Exception:
                self.adjacent_nodes_cache[node] = []

    def generate_scenario(
        self,
        base_stations: List[Dict],
        user_distribution: List[int],
        total_steps: int = 100,
        scenario_name: str = "campus_scenario",
        user_speed: float = 1.5,
        speed_range: Tuple[float, float] = None,
        target_direction: float = None,
        direction_strength: float = 0.5,
        base_station_directions: Dict[int, float] = None,
        base_station_direction_strengths: Dict[int, float] = None,
        server_configs_raw: List[Dict] = None,
        gate_spawn_ratio: float = 0.4,
        gate_leave_ratio: float = 0.4,
        initial_active_ratio: float = 0.3,
    ) -> Dict:
        """
        Generate complete scenario configuration (using metric coordinate system)
        Ensures initial positions are legal and movement follows legal paths

        Args:
            base_stations: Base station configuration list [{"id": int, "position": [lon, lat]}, ...]
            user_distribution: User distribution around each base station [num_users_bs1, num_users_bs2, ...]
            total_steps: Total simulation steps
            scenario_name: Scenario name
            user_speed: User movement speed (meters/step), default 1.5 m/step (used when speed_range is None)
            speed_range: User movement speed range [min, max] (meters/step), if provided each user's speed is randomly selected within this range
            target_direction: Target direction (radians) (global default, used if base station does not specify separately)
            direction_strength: Direction strength (0-1) (global default)
            base_station_directions: Base station specific target direction dict {base_station_id: direction_radians}
            base_station_direction_strengths: Base station specific direction strength dict {base_station_id: strength}
            server_configs_raw: Raw server configuration list (containing movement_pattern etc. detailed information)
            gate_spawn_ratio: Ratio of users spawning exactly at gates.
            gate_leave_ratio: Ratio of users that logically leave when reaching a gate.

        Returns:
            Scenario configuration dict (all coordinates converted to metric)
        """
        if self.osm_graph is None:
            self.load_osm_data()

        # Validate input parameters
        self._validate_inputs(base_stations, user_distribution)

        # Process speed parameters
        if speed_range is not None:
            min_speed, max_speed = speed_range
            if min_speed <= 0 or max_speed <= 0 or min_speed > max_speed:
                raise ValueError("Speed range must be positive and min_speed <= max_speed")
            total_users = sum(user_distribution)
            user_speeds = [
                random.uniform(min_speed, max_speed) for _ in range(total_users)
            ]
        else:
            if user_speed <= 0:
                raise ValueError("User movement speed must be greater than 0")
            user_speeds = None

        # Convert base station coordinates to meters
        base_stations_meters = self._convert_base_stations_to_meters(base_stations)

        # Generate legal initial positions (on road nodes), recording the base station each device belongs to
        initial_nodes_with_bs = self._generate_legal_initial_positions_with_bs(
            base_stations_meters, user_distribution, gate_spawn_ratio
        )

        # Generate legal movement trajectories (along road network)
        print("Generating movement trajectories (Parallel)...")
        start_time = time.time()

        # Use parallel processing to accelerate trajectory generation
        mobile_devices = self._generate_trajectories_parallel_with_bs_directions(
            initial_nodes_with_bs,
            total_steps,
            user_speed,
            user_speeds,
            target_direction,
            direction_strength,
            base_station_directions,
            base_station_direction_strengths,
            server_configs_raw,
            gate_leave_ratio,
            initial_active_ratio,
        )

        trajectory_time = time.time() - start_time
        print(f"Trajectory generation completed: Total {len(mobile_devices)} devices, time taken {trajectory_time:.2f} seconds")

        # Build scenario configuration
        scenario_config = self._build_scenario_config(
            base_stations_meters, mobile_devices, total_steps, scenario_name
        )

        return scenario_config

    def _validate_inputs(self, base_stations: List[Dict], user_distribution: List[int]):
        """Validate input parameters"""
        if len(base_stations) != len(user_distribution):
            raise ValueError("Number of base stations must match the length of user distribution list")

        if sum(user_distribution) == 0:
            raise ValueError("Sum of user distribution list cannot be 0")

    def _convert_base_stations_to_meters(self, base_stations: List[Dict]) -> List[Dict]:
        """Convert base station coordinates from lat/lon to metric coordinates"""
        scene_bounds = self._get_scene_bounds_degrees()
        min_lon, min_lat, max_lon, max_lat = scene_bounds
        center_lat = (min_lat + max_lat) / 2

        base_stations_meters = []
        for bs in base_stations:
            lon, lat = bs["position"]
            x_meters = self._lon_to_meters(lon, min_lon, center_lat)
            y_meters = self._lat_to_meters(lat, min_lat)
            base_stations_meters.append(
                {"id": bs["id"], "position": [x_meters, y_meters]}
            )
            print(f"Base station {bs['id']}: lat/lon ({lon:.6f}, {lat:.6f}) -> metric coordinates ({x_meters:.2f}, {y_meters:.2f})")
        return base_stations_meters

    def _lon_to_meters(self, lon: float, reference_lon: float, center_lat: float) -> float:
        """Convert longitude to metric coordinates"""
        earth_radius = 6371000.0
        lon_diff_rad = math.radians(lon - reference_lon)
        x_meters = lon_diff_rad * earth_radius * math.cos(math.radians(center_lat))
        return x_meters

    def _lat_to_meters(self, lat: float, reference_lat: float) -> float:
        """Convert latitude to metric coordinates"""
        earth_radius = 6371000.0
        lat_diff_rad = math.radians(lat - reference_lat)
        y_meters = lat_diff_rad * earth_radius
        return y_meters

    def _get_scene_bounds_degrees(self) -> Tuple[float, float, float, float]:
        """Get the lat/lon bounds of the road network"""
        if self.osm_graph is None:
            self.load_osm_data()
        min_lon, min_lat = (
            self.road_nodes.geometry.x.min(),
            self.road_nodes.geometry.y.min(),
        )
        max_lon, max_lat = (
            self.road_nodes.geometry.x.max(),
            self.road_nodes.geometry.y.max(),
        )
        return (min_lon, min_lat, max_lon, max_lat)

    def _get_scene_bounds_meters(self) -> Tuple[float, float, float, float]:
        """Get the metric bounds of the road network"""
        min_lon, min_lat, max_lon, max_lat = self._get_scene_bounds_degrees()
        center_lat = (min_lat + max_lat) / 2
        min_x = self._lon_to_meters(min_lon, min_lon, center_lat)
        min_y = self._lat_to_meters(min_lat, min_lat)
        max_x = self._lon_to_meters(max_lon, min_lon, center_lat)
        max_y = self._lat_to_meters(max_lat, min_lat)
        return (min_x, min_y, max_x, max_y)

    def _generate_legal_initial_positions_with_bs(
        self, base_stations: List[Dict], user_distribution: List[int], gate_spawn_ratio: float = 0.4
    ) -> List[Tuple]:
        """Generate legal initial positions. Users may spawn near base stations or exactly at the gate nodes."""
        print("Generating legal initial positions...")
        start_time = time.time()
        all_initial_nodes_with_bs = []

        total_users = sum(user_distribution)
        num_gate_spawns = int(total_users * gate_spawn_ratio)

        # We need a flat list of BS IDs based on distribution
        user_bs_list = []
        for bs, num_users in zip(base_stations, user_distribution):
            user_bs_list.extend([bs["id"]] * num_users)
        
        # Decide which ones will spawn from gate
        random.shuffle(user_bs_list) # randomize order
        
        assigned_count = 0
        for bs_id in user_bs_list:
            if assigned_count < num_gate_spawns and len(self.gate_nodes_by_name) > 0:
                # Spawn at gate: pick a gate uniformly, then pick a node within that gate
                gate_name = random.choice(list(self.gate_nodes_by_name.keys()))
                selected_node = random.choice(self.gate_nodes_by_name[gate_name])
            else:
                # Find BS position to spawn around
                bs_position = next(bs["position"] for bs in base_stations if bs["id"] == bs_id)
                nearby_nodes = self._find_nearby_road_nodes(bs_position, max_distance=300)
                
                if not nearby_nodes:
                    nearby_nodes = list(self.node_positions_cache.keys())
                selected_node = random.choice(nearby_nodes)
                
            all_initial_nodes_with_bs.append((selected_node, bs_id))
            assigned_count += 1

        initial_time = time.time() - start_time
        print(f"Initial position generation completed: Total {len(all_initial_nodes_with_bs)} legal positions, time taken {initial_time:.2f} seconds")
        return all_initial_nodes_with_bs

    def _find_nearby_road_nodes(self, position: Tuple[float, float], max_distance: float = 500):
        """Find all road nodes within a specified distance from the given position"""
        pos_x, pos_y = position
        nearby_nodes = []
        for node_id, node_pos in self.node_positions_cache.items():
            node_x, node_y = node_pos
            distance = math.sqrt((node_x - pos_x) ** 2 + (node_y - pos_y) ** 2)
            if distance <= max_distance:
                nearby_nodes.append(node_id)
        return nearby_nodes

    def _generate_trajectories_parallel_with_bs_directions(
        self,
        initial_nodes_with_bs: List[Tuple],
        total_steps: int,
        user_speed: float = 1.5,
        user_speeds: List[float] = None,
        target_direction: float = None,
        direction_strength: float = 0.5,
        base_station_directions: Dict[int, float] = None,
        base_station_direction_strengths: Dict[int, float] = None,
        server_configs_raw: List[Dict] = None,
        gate_leave_ratio: float = 0.4,
        initial_active_ratio: float = 0.3,
    ) -> List[Dict]:
        """Trajectory generation with Profile-based shortest-path, Relay respawning, and speed tiers."""
        mobile_devices = []
        all_nodes = list(self.osm_graph.nodes())
        
        def _get_random_speed():
            rand = random.random()
            if rand < 0.50:
                return random.uniform(1.0, 5.0)    # Pedestrians
            elif rand < 0.80:
                return random.uniform(5.0, 10.0)   # Bicycles
            else:
                return random.uniform(10.0, 15.0)  # Vehicles
                
        def _get_profile():
            rand = random.random()
            if rand < 0.30: return "entry"
            elif rand < 0.60: return "leaving"
            else: return "wanderer"

        # Generate paths using shortest path when possible
        def _generate_lifecycle(bs_id, spawn_time, max_steps):
            import math
            profile = _get_profile()
            speed = _get_random_speed()
            
            start_node = None
            target_node = None
            
            if profile == "entry":
                # Spawn at any gate uniformly
                start_gate = random.choice(list(self.gate_nodes_by_name.keys()))
                start_node = random.choice(self.gate_nodes_by_name[start_gate])
                # Target is anywhere
                target_node = random.choice(all_nodes)
                while target_node in self.gate_nodes:
                    target_node = random.choice(all_nodes)
            elif profile == "leaving":
                # Start is anywhere
                start_node = random.choice(all_nodes)
                # Target is any gate uniformly
                target_gate = random.choice(list(self.gate_nodes_by_name.keys()))
                target_node = random.choice(self.gate_nodes_by_name[target_gate])
            else: # wanderer
                start_node = random.choice(all_nodes)
                target_node = None # No fixed target
            
            nav_path = None
            if target_node is not None:
                try:
                    nav_path = nx.shortest_path(self.osm_graph, source=start_node, target=target_node, weight='length')
                except nx.NetworkXNoPath:
                    nav_path = None
                    profile = "wanderer"
            
            trajectory = []
            current_time = spawn_time
            current_node = start_node
            px, py = self.node_positions_cache[current_node]
            nav_idx = 0
            
            # The distance left to move on the current edge (meters)
            dist_left = 0.0
            next_node = None
            
            # Plan B: 增加使用寿命 (Lifespan Decay)
            # 对于内部游走者或入校者，模拟他们到达目的地后办完事关机下线。限制最大活跃步数
            import random as rnd
            lifespan = max_steps
            if profile == "wanderer" and rnd.random() < 0.7:  # 70% 内部人员只会溜达二三十秒
                lifespan = int(spawn_time) + rnd.randint(20, 50)
            elif profile == "entry":  # 入校者走到中途可能会断开
                lifespan = int(spawn_time) + rnd.randint(40, 80)
            
            actual_end = min(max_steps, lifespan)
            
            for step in range(int(spawn_time), actual_end):
                step_data = {
                    "time": float(current_time),
                    "node": current_node,
                    "position": [px, py]
                }
                trajectory.append(step_data)
                
                if profile == "leaving" and current_node in self.gate_nodes and step > int(spawn_time) + 5:
                    break
                    
                # Movement logic (1 second = speed meters)
                dist_moved = speed
                
                while dist_moved > 0:
                    if next_node is None:
                        # Need to pick next node
                        if nav_path and nav_idx < len(nav_path) - 1:
                            next_node = nav_path[nav_idx + 1]
                            nav_idx += 1
                        else:
                            # Random walk
                            adj_nodes = self.adjacent_nodes_cache.get(current_node, [])
                            if adj_nodes:
                                next_node = random.choice(adj_nodes)
                            else:
                                break # Stuck
                                
                        nx_px, nx_py = self.node_positions_cache[next_node]
                        dist_left = math.sqrt((nx_px - px)**2 + (nx_py - py)**2)
                    
                    if next_node is None:
                        break # Cannot move
                        
                    if dist_moved >= dist_left:
                        # Reached and surpassed next node
                        dist_moved -= dist_left
                        current_node = next_node
                        px, py = self.node_positions_cache[current_node]
                        next_node = None
                        
                        # Stop if reached target gate
                        if profile == "leaving" and current_node in self.gate_nodes:
                            dist_moved = 0
                            break
                    else:
                        # Interpolate position
                        nx_px, nx_py = self.node_positions_cache[next_node]
                        ratio = dist_moved / dist_left
                        px = px + (nx_px - px) * ratio
                        py = py + (nx_py - py) * ratio
                        dist_left -= dist_moved
                        dist_moved = 0

                current_time += 1.0
                
            return {
                "spawn_time": spawn_time,
                "leave_time": current_time,
                "trajectory": trajectory,
                "base_station_id": bs_id,
                "speed": speed
            }

        print("Generating movement trajectories with Slot-based Relay (Progressing)...")
        
        total = len(initial_nodes_with_bs)
        
        for slot_idx in range(total):
            original_node, bs_id = initial_nodes_with_bs[slot_idx]
            
            # Plan C: Staggered Spawning with initial ratio
            # A proportion of users start exactly at step 0, others stagger
            if slot_idx < total * initial_active_ratio:
                current_t = 0.0
            else:
                current_t = random.uniform(0.0, total_steps * 0.4)
            
            first_spawn = current_t
            last_leave = current_t
            device_trajectory = []
            device_tasks = []
            first_speed = None
            
            while current_t < total_steps:
                lifecycle = _generate_lifecycle(bs_id, current_t, total_steps)
                
                # Check bounds or correct leave_time
                tasks = self._generate_tasks(lifecycle["spawn_time"], lifecycle["leave_time"], 1.0)
                
                device_trajectory.extend(lifecycle["trajectory"])
                device_tasks.extend(tasks)
                
                last_leave = lifecycle["leave_time"]
                if first_speed is None:
                    first_speed = lifecycle["speed"]
                
                # Introduce a random delay before respawning to make the user count dynamic
                import random as rnd
                current_t = lifecycle["leave_time"] + rnd.uniform(5.0, 20.0)
                
            mobile_devices.append({
                "id": slot_idx,
                "spawn_time": float(first_spawn),
                "leave_time": float(last_leave),
                "trajectory": device_trajectory,
                "base_station_id": bs_id,
                "speed": first_speed,
                "tasks": device_tasks
            })
                
            if slot_idx % 10 == 0 or slot_idx == total - 1:
                print(f"Relay generation progress: {slot_idx+1}/{total} ({((slot_idx+1)/total)*100:.1f}%)")

        return mobile_devices

    def _generate_tasks(self, spawn_time: float, leave_time: float, time_interval: float = 1.0) -> List[Dict]:
        """Generate random tasks over a device's active lifecycle"""
        # Generate tasks using task-type model with correlated data/workload
        # Task types: (probability, data_range, workload_range, budget_range, name)
        task_types = [
            (0.30, (0.05, 0.2),  (100, 300),   (2.0, 4.0), "lightweight"), # e.g., IoT Sensors
            (0.40, (0.5, 1.5),   (500, 1000),  (3.0, 5.0), "moderate"),    # e.g., Image Classification
            (0.20, (2.0, 5.0),   (1500, 3000), (5.0, 10.0), "heavy"),       # e.g., AR / HD Video
            (0.10, (0.1, 0.5),   (200, 500),   (0.5, 1.0), "critical"),    # e.g., V2X Control
        ]
        type_probs = [t[0] for t in task_types]

        tasks = []
        task_id_counter = 1
        current_t = spawn_time
        while current_t <= leave_time:
            # Task generation logic (e.g., 30% chance per second)
            if random.random() < 0.3:
                # Select task type
                chosen = random.choices(task_types, weights=type_probs, k=1)[0]
                _, data_range, wl_range, budget_range, type_name = chosen

                # Generate data size
                data_size = random.uniform(*data_range)

                # Correlated workload: base + beta * normalized_data * range
                beta = random.uniform(0.3, 0.7)  # Correlation strength
                data_norm = (data_size - data_range[0]) / max(data_range[1] - data_range[0], 0.01)
                wl_base = random.uniform(wl_range[0], wl_range[0] + (wl_range[1] - wl_range[0]) * 0.5)
                workload = wl_base + beta * data_norm * (wl_range[1] - wl_range[0]) * 0.5
                workload = max(wl_range[0], min(wl_range[1], workload))

                # Time budget matched to task difficulty
                time_budget = random.uniform(*budget_range)
                deadline = current_t + time_budget

                tasks.append({
                    "task_id": task_id_counter,
                    "creation_time": float(current_t),
                    "data_size": round(data_size, 3),
                    "computation_workload": round(workload, 1),
                    "deadline": round(deadline, 1),
                    "task_type": type_name,
                })
                task_id_counter += 1
            current_t += time_interval
        return tasks

    def _generate_single_trajectory(
        self, start_node, spawn_time: float = 0.0, leave_time: float = 100.0, time_interval: float = 1.0,
        device_id: int = 0, user_speed: float = 1.5, target_direction: float = None, direction_strength: float = 0.5,
        movement_pattern: str = "directional", target_point: List[float] = None, loiter_radius: float = 150.0,
    ) -> List[Dict]:
        """Generate legal movement trajectory based on road network and movement patterns"""
        trajectory = []
        current_node = start_node
        current_time = spawn_time
        current_step = 0
        current_pos = self.node_positions_cache.get(current_node, [0, 0])
        trajectory.append({"step": current_step, "time": current_time, "position": list(current_pos), "node": current_node})

        current_direction = None
        target_node = None
        remaining_steps_in_segment = 0
        previous_node = None
        stuck_counter = 0
        max_stuck_count = 5

        scene_bounds = self._get_scene_bounds_meters()
        min_x, min_y, max_x, max_y = scene_bounds
        boundary_margin = 50

        total_steps = max(1, int((leave_time - spawn_time) / time_interval))
        for step in range(1, total_steps + 1):
            use_direction = target_direction
            if movement_pattern == "loiter" and target_point is not None:
                dx = target_point[0] - current_pos[0]
                dy = target_point[1] - current_pos[1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > loiter_radius:
                    use_direction = math.atan2(dy, dx)
                    effective_strength = 0.7
                else:
                    use_direction = None
                    effective_strength = 0.0
            else:
                effective_strength = direction_strength

            if use_direction is not None:
                if (current_pos[0] <= min_x + boundary_margin or current_pos[0] >= max_x - boundary_margin or
                    current_pos[1] <= min_y + boundary_margin or current_pos[1] >= max_y - boundary_margin):
                    adjusted = self._adjust_direction_at_boundary(current_pos, use_direction, scene_bounds)
                    if adjusted is not None: use_direction = adjusted

            if current_direction is None or remaining_steps_in_segment <= 0:
                adjacent_nodes = self.adjacent_nodes_cache.get(current_node, [])
                if not adjacent_nodes:
                    current_time += time_interval
                    current_step += 1
                    trajectory.append({"step": current_step, "time": current_time, "position": list(current_pos), "node": current_node})
                    stuck_counter += 1
                    if stuck_counter >= max_stuck_count: break
                    continue

                available_nodes = [node for node in adjacent_nodes if node != previous_node]
                if not available_nodes:
                    available_nodes = adjacent_nodes if len(adjacent_nodes) > 0 else []
                
                target_node = self._select_next_node_with_direction(current_node, available_nodes, previous_node, use_direction, effective_strength)
                if target_node is None:
                    if not available_nodes:
                        current_time += time_interval
                        current_step += 1
                        trajectory.append({"step": current_step, "time": current_time, "position": list(current_pos), "node": current_node})
                        stuck_counter += 1
                        if stuck_counter >= max_stuck_count: break
                        continue
                    else:
                        target_node = random.choice(available_nodes)

                target_pos = self.node_positions_cache.get(target_node, [0, 0])
                distance = math.sqrt((target_pos[0] - current_pos[0])**2 + (target_pos[1] - current_pos[1])**2)
                if distance <= 0:
                    previous_node, current_node = current_node, target_node
                    current_time += time_interval
                    current_step += 1
                    trajectory.append({"step": current_step, "time": current_time, "position": list(current_pos), "node": current_node})
                    continue

                remaining_steps_in_segment = max(1, int(distance / (user_speed * time_interval)))
                current_direction = [(target_pos[0] - current_pos[0]) / distance, (target_pos[1] - current_pos[1]) / distance]
                stuck_counter = 0

            step_size = user_speed * time_interval
            new_pos = [current_pos[0] + current_direction[0] * step_size, current_pos[1] + current_direction[1] * step_size]
            if new_pos[0] < min_x or new_pos[0] > max_x or new_pos[1] < min_y or new_pos[1] > max_y:
                new_pos[0] = max(min_x, min(new_pos[0], max_x))
                new_pos[1] = max(min_y, min(new_pos[1], max_y))
                remaining_steps_in_segment, current_direction = 0, None

            current_pos = new_pos
            target_pos = self.node_positions_cache.get(target_node, [0, 0])
            if math.sqrt((target_pos[0] - current_pos[0])**2 + (target_pos[1] - current_pos[1])**2) <= step_size:
                previous_node, current_node, current_pos = current_node, target_node, target_pos
                remaining_steps_in_segment = 0
            
            current_time += time_interval
            current_step += 1
            trajectory.append({"step": current_step, "time": current_time, "position": list(current_pos), "node": current_node})
        return trajectory

    def _adjust_direction_at_boundary(self, current_pos: List[float], target_direction: float, scene_bounds: Tuple[float, float, float, float]) -> float:
        """Adjust direction at boundaries to stay within scene limits"""
        min_x, min_y, max_x, max_y = scene_bounds
        boundary_margin = 50
        if current_pos[0] <= min_x + boundary_margin and (target_direction < -math.pi / 2 or target_direction > math.pi / 2): return 0.0
        if current_pos[0] >= max_x - boundary_margin and (-math.pi / 2 < target_direction < math.pi / 2): return math.pi
        if current_pos[1] <= min_y + boundary_margin and target_direction < 0: return math.pi / 2
        if current_pos[1] >= max_y - boundary_margin and target_direction > 0: return -math.pi / 2
        return target_direction

    def _select_next_node_with_direction(self, current_node, adjacent_nodes, previous_node, target_direction: float, direction_strength: float):
        """Intelligently select next node based on target direction"""
        if target_direction is None or len(adjacent_nodes) <= 1:
            return random.choice(adjacent_nodes) if adjacent_nodes else None
        
        curr_pos = self.node_positions_cache.get(current_node, [0, 0])
        node_scores = []
        for node in adjacent_nodes:
            node_pos = self.node_positions_cache.get(node, [0, 0])
            node_dir = math.atan2(node_pos[1] - curr_pos[1], node_pos[0] - curr_pos[0])
            normalized_dir = node_dir % (2 * math.pi)
            is_forbidden = False
            if abs(target_direction - 0) < 0.1: # Right
                if math.pi * 0.75 < normalized_dir < math.pi * 1.25: is_forbidden = True
            elif abs(target_direction - math.pi) < 0.1: # Left
                if normalized_dir >= math.pi * 1.75 or normalized_dir <= math.pi * 0.25: is_forbidden = True
            
            if is_forbidden:
                score = 0.0
            else:
                target_diff = abs(self._normalize_angle(node_dir - target_direction))
                score = 1.0 - target_diff / math.pi
            node_scores.append((node, score))
        
        valid_nodes = [(n, s) for n, s in node_scores if s > 0]
        if not valid_nodes: return random.choice(adjacent_nodes)
        
        if direction_strength >= 1.0: return max(valid_nodes, key=lambda x: x[1])[0]
        scores = [s for _, s in valid_nodes]
        total = sum(scores)
        if total <= 0: return random.choice(adjacent_nodes)
        
        r, acc = random.random() * total, 0
        for n, s in valid_nodes:
            acc += s
            if r <= acc: return n
        return valid_nodes[-1][0]

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-π, π] range"""
        while angle > math.pi: angle -= 2 * math.pi
        while angle < -math.pi: angle += 2 * math.pi
        return angle

    def _generate_default_trajectory(self, device_id: int, total_steps: int) -> List[Dict]:
        """Generate default stationary trajectory"""
        scene_bounds = self._get_scene_bounds_meters()
        cx, cy = (scene_bounds[0] + scene_bounds[2]) / 2, (scene_bounds[1] + scene_bounds[3]) / 2
        return [{"step": i, "position": [cx, cy]} for i in range(total_steps)]

    def _build_scenario_config(self, base_stations: List[Dict], mobile_devices: List[Dict], total_steps: int, scenario_name: str) -> Dict:
        """Build finalized scenario configuration dict"""
        min_x, min_y, max_x, max_y = self._get_scene_bounds_meters()
        sw, sh = max_x - min_x, max_y - min_y
        for bs in base_stations:
            bs.setdefault("tx_power", 46)
            bs.setdefault("frequency", 1800)
            bs.setdefault("bandwidth", 20)

        return {
            "scene": {"width": float(sw), "height": float(sh), "total_time": float(total_steps), "total_steps": total_steps},
            "base_stations": base_stations,
            "mobile_devices": mobile_devices,
            "hysteresis_config": {"enabled": True, "value": 3.0},
            "reward_config": {"type": "load_balance"},
            "cio_config": {"min_value": -6, "max_value": 6, "step_size": 0.5},
            "network_config": {
                "noise_floor": -104, "shadow_fading_std": 8, "penetration_loss": 10,
                "receiver_noise_figure": 7, "thermal_noise_density": -174,
                "constant_bit_rate": 1, "mobility_speed": "variable"
            },
            "users": len(mobile_devices)
        }

    def save_scenario_to_file(self, scenario_config: Dict, output_path: str):
        """Save config to YAML"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(scenario_config, f, default_flow_style=False, allow_unicode=True)
        print(f"Scenario configuration saved to: {output_path}")


def generate_controlled_osm_scenario(
    osm_file_path: str, base_stations: List[Dict], server_ids: List[int], num_users: int,
    total_steps: int = 100, output_file: str = None, user_speed: float = 30.0,
    speed_range: Tuple[float, float] = None, movement_pattern: str = "random",
    target_direction: float = None, direction_strength: float = 0.5,
    target_point: List[float] = None, attraction_strength: float = 0.3,
    scenario_name: str = "controlled_osm_scenario",
    base_station_directions: Dict[int, float] = None,
    base_station_direction_strengths: Dict[int, float] = None,
    server_configs_raw: List[Dict] = None,
):
    """High-level helper for generating controlled scenarios"""
    generator = OSMScenarioGenerator(osm_file_path)
    user_distribution = [0] * len(base_stations)
    temp_users = num_users
    for sid in server_ids:
        if sid <= len(base_stations) and temp_users > 0:
            user_distribution[sid - 1] = 1
            temp_users -= 1
    for _ in range(temp_users):
        sid = random.choice(server_ids)
        if sid <= len(base_stations): user_distribution[sid - 1] += 1

    scenario_config = generator.generate_scenario(
        base_stations, user_distribution, total_steps, scenario_name, user_speed,
        speed_range, target_direction, direction_strength,
        base_station_directions, base_station_direction_strengths, server_configs_raw,
        initial_active_ratio=args.initial_active_ratio if 'args' in globals() else 0.3
    )
    
    initial_connections = {}
    uid = 0
    for idx, count in enumerate(user_distribution):
        bs_id = base_stations[idx]["id"]
        for _ in range(count):
            initial_connections[uid] = bs_id
            uid += 1
    scenario_config["initial_connections"] = initial_connections
    
    if output_file:
        generator.save_scenario_to_file(scenario_config, output_file)
    return scenario_config


def generate_randomized_osm_scenario_config(
    osm_file_path: str, base_stations: List[Dict], users_range: List[int], total_time: float,
    scenario_id: str, server_configs: List[Dict] = None, output_path: str = None, i: int = 0,
    initial_active_ratio: float = 0.3
) -> Dict:
    """Generate a single randomized OSM scenario configuration"""
    generator = OSMScenarioGenerator(osm_file_path)
    num_users = random.randint(users_range[0], users_range[1])
    server_ids = [s["id"] for s in server_configs] if server_configs else [bs["id"] for bs in base_stations]
    user_distribution = [0] * len(base_stations)
    
    temp_num_users = num_users
    for s_id in server_ids:
        if s_id <= len(base_stations) and temp_num_users > 0:
            user_distribution[s_id - 1] = 1
            temp_num_users -= 1
    for _ in range(temp_num_users):
        s_id = random.choice(server_ids)
        if s_id <= len(base_stations): user_distribution[s_id - 1] += 1
            
    total_steps = int(total_time)
    
    initial_nodes_with_bs = generator._generate_legal_initial_positions_with_bs(
        generator._convert_base_stations_to_meters(base_stations), user_distribution, gate_spawn_ratio=0.4
    )
    
    mobile_devices = generator._generate_trajectories_parallel_with_bs_directions(
        initial_nodes_with_bs, total_steps=total_steps, server_configs_raw=server_configs, gate_leave_ratio=0.4,
        initial_active_ratio=initial_active_ratio
    )
    
    initial_connections = {md["id"]: md["base_station_id"] for md in mobile_devices}
    scenario_config = generator._build_scenario_config(
        generator._convert_base_stations_to_meters(base_stations), mobile_devices, total_steps, f"{scenario_id}_{i}"
    )
    scenario_config["initial_connections"] = initial_connections
    if output_path:
        generator.save_scenario_to_file(scenario_config, output_path)
    return scenario_config


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate campus OSM scenarios for ScalableMEC.")
    parser.add_argument("--max_users", type=int, default=50, help="Maximum number of users")
    parser.add_argument("--min_users", type=int, default=None, help="Minimum number of users")
    parser.add_argument("--scenario_id", type=str, default="campus_scenario", help="Scenario ID")
    parser.add_argument("--num_scenarios", type=int, default=1, help="Number of scenarios")
    parser.add_argument("--total_time", type=float, default=100.0, help="Total simulation time")
    parser.add_argument("--osm_file", type=str, default="campus_scenarios/sdnu.osm", help="Path to OSM file")
    parser.add_argument("--initial_active_ratio", type=float, default=0.3, help="Ratio of users active exactly at step 0")
    
    for i in range(1, 4):
        parser.add_argument(f"--bs{i}_pattern", type=str, default="random", choices=["random", "directional", "loiter"])
        parser.add_argument(f"--bs{i}_dir", type=float, default=None)
        parser.add_argument(f"--bs{i}_strength", type=float, default=0.5)
        parser.add_argument(f"--bs{i}_loiter_radius", type=float, default=150.0)

    args = parser.parse_args()
    if args.min_users is None: args.min_users = args.max_users
        
    base_stations = [
        {"id": 1, "position": [116.8239735, 36.5429144]},
        {"id": 2, "position": [116.8274521, 36.5456314]},
        {"id": 3, "position": [116.8306442, 36.5443304]},
    ]
    
    server_configs = []
    for i in range(1, 4):
        config = {
            "id": i,
            "movement_pattern": getattr(args, f"bs{i}_pattern"),
            "target_direction": getattr(args, f"bs{i}_dir"),
            "direction_strength": getattr(args, f"bs{i}_strength"),
            "loiter_radius": getattr(args, f"bs{i}_loiter_radius")
        }
        if config["movement_pattern"] == "loiter":
            config["target_point"] = base_stations[i-1]["position"]
        server_configs.append(config)
        
    output_dir = os.path.join("generated_scenarios", args.scenario_id)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {args.num_scenarios} campus scenarios for '{args.scenario_id}'...")
    for i in range(args.num_scenarios):
        output_path = os.path.join(output_dir, f"scenario_{i}.yaml")
        random.seed(i)
        generate_randomized_osm_scenario_config(
            args.osm_file, base_stations, [args.min_users, args.max_users],
            args.total_time, args.scenario_id, server_configs, output_path, i,
            initial_active_ratio=args.initial_active_ratio
        )
        print(f"  - Generated {output_path}")
    print("\nCampus scenario generation completed!")


if __name__ == "__main__":
    main()
