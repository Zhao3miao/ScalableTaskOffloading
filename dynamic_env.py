import yaml
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DynamicMECEnv(gym.Env):
    """
    Dynamic MEC Environment.

    Multi-agent interface (array format):
    - Observations: np.ndarray of shape (n_agents, obs_dim)
    - Actions: np.ndarray of shape (n_agents,) with discrete actions
    - Rewards: np.ndarray of shape (n_agents,)
    - Terminations: np.ndarray of shape (n_agents,)
    - Truncations: np.ndarray of shape (n_agents,)
    - Info: contains 'active_ue_masks' of shape (n_agents,)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, config, max_n=100, render_mode=None):
        super().__init__()
        
        if config is None:
            raise ValueError("A parsed config dictionary must be provided")

        self.max_n = max_n
        self.render_mode = render_mode

        # Initialize fixed constraints (Assuming all scenarios have same max specs for gym space consistency)
        self.server_capacity = 4000.0   # Powerful edge server capacity (Megacycles/s)
        self.local_capacity = 250.0     # Balanced local device capacity (Megacycles/s)

        # === Wireless Communication Parameters (5G NR Sub-6 + Shannon) ===
        self.ue_tx_power_dbm = 23.0        # UE uplink transmit power (dBm, 200 mW, 3GPP standard)
        self.noise_psd_dbm_hz = -174.0     # Thermal noise PSD (dBm/Hz)
        self.noise_figure_db = 5.0         # 5G BS receiver noise figure (dB)
        self.uplink_bandwidth_hz = 100e6   # Uplink bandwidth per BS (Hz, 100 MHz = 5G NR n78)
        # Precompute noise floor for full bandwidth: -174 + 10*log10(1e8) + 5 = -89 dBm
        self.noise_floor_dbm = self.noise_psd_dbm_hz + 10.0 * math.log10(self.uplink_bandwidth_hz) + self.noise_figure_db
        # Max spectral efficiency (for observation normalization, 5G NR 256-QAM cap)
        self.max_spectral_efficiency = 7.4  # bits/s/Hz, 3GPP 38.214 Table 5.1.3.1-2

        # Load environment configuration from memory
        self._load_config(config)

        self.n_agents = self.max_n  # Fix observation/action shape to max_n
        # Number of base stations for observation dimension
        self.n_servers = len(self.servers_info)

        # RL Statistics for tracking drop penalties
        self.drop_count_disconnection = 0
        self.drop_count_deadline = 0
        self.completed_tasks = 0
        self.total_completion_time = 0.0
        
        # Event Tracing list
        self.task_traces = []

        # === Define Gymnasium Spaces ===
        # Observation per agent: [n_active, has_task, data_size, workload, budget, local_queue_len,
        #                         *channel_qualities, *tx_congestions, *compute_congestions]
        # Confounder L = [tx_1..tx_K, cp_1..cp_K] is 2*n_servers dim (multi-modal bottleneck).
        # CD-MAPPO uses this as confounder for DEBA; baselines also consume it via obs.
        self.obs_dim = 1 + 5 + self.n_servers + 2 * self.n_servers  # 1 global + 5 local + K ch + K tx + K cp
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_agents, self.obs_dim),
            dtype=np.float32,
        )

        # Action per agent: 0=local, 1..K=server_id
        self.action_dim = 1 + self.n_servers
        self.action_space = spaces.MultiDiscrete([self.action_dim] * self.n_agents)

        # Store max users for confounder computation
        self.max_users = self.max_n

    def _load_config(self, config):
        self.config = config
        self.scene_config = self.config["scene"]
        self.time_interval = 1.0  # Time step of simulation (e.g., 1 sec)
        self.max_time = self.scene_config["total_time"]

        self.servers_info = {
            s["id"]: {
                "position": np.array(s["position"]),
                "tx_power": s.get("tx_power", 46),
                "frequency": s.get("frequency", 1800),
                "bandwidth": s.get("bandwidth", 20),
            }
            for s in self.config["base_stations"]
        }

        self.all_users_meta = {d["id"]: d for d in self.config["mobile_devices"]}

        # Get sorted list of user IDs for consistent indexing
        self.all_user_ids = sorted([int(uid) for uid in self.all_users_meta.keys()])
        self.real_user_count = len(self.all_user_ids)
        if self.real_user_count > self.max_n:
            raise ValueError(f"Scenario users {self.real_user_count} exceeds max_n {self.max_n}")

    def reset(self, seed=None, options=None):
        """
        Resets the environment to the beginning of the temporal scenario.
        Returns:
            obs: np.ndarray of shape (n_agents, obs_dim)
            info: dict with global state and metadata
        """
        super().reset(seed=seed)

        self.current_time = 0.0

        # RL Statistics Reset
        self.drop_count_disconnection = 0
        self.drop_count_deadline = 0
        self.completed_tasks = 0
        self.total_completion_time = 0.0
        self.task_traces = []

        # Active Environment dictionaries
        self.active_users = {}
        self.server_queues = {sid: [] for sid in self.servers_info.keys()}

        # 延迟拥塞观测: 所有智能体 (包括 Critic) 只能看到 t-1 时刻的 per-server 拥塞
        # 初始化为 (tx=0, cp=0) (t=0 时无历史), 在每一步末尾更新为当前真实拥塞
        # 每台服务器的 congestion 是 (tx_load_norm, cp_load_norm) 二元组
        self._prev_server_congestions = {sid: (0.0, 0.0) for sid in self.servers_info.keys()}

        # Parse and chronologically sort all task generation events
        self.task_events = []
        for uid, user_data in self.all_users_meta.items():
            for t in user_data.get("tasks", []):
                self.task_events.append(
                    {"time": t["creation_time"], "uid": uid, "task": t}
                )
        self.task_events.sort(key=lambda x: x["time"])
        self.event_ptr = 0

        # Move system to the initial state
        self._update_spawns_and_positions()
        self._last_pending_tasks = {}

        # Get observations for all agents (n_agents, obs_dim)
        obs = self._get_obs_array(pending_tasks={})
        info = self._get_global_state()
        info["real_n"] = self.real_user_count  # Mark the valid user count
        info["active_ue_masks"] = self._get_active_ue_masks()
        info["decision_masks"] = self._get_decision_masks(self._last_pending_tasks)

        return obs, info

    def step(self, actions):
        """
        Applies RL actions for generated tasks, ticks time forward, and resolves processing.

        Args:
            actions: np.ndarray of shape (n_agents,) with discrete actions

        Returns:
            obs: np.ndarray of shape (n_agents, obs_dim)
            rewards: np.ndarray of shape (n_agents,)
            terminations: np.ndarray of shape (n_agents,)
            truncations: np.ndarray of shape (n_agents,)
            info: dict with global state and metadata
        """
        # Initialize rewards for all agents
        rewards = np.zeros(self.n_agents, dtype=np.float32)

        # Convert actions array to dict for internal processing
        action_dict = {}
        for idx in range(self.real_user_count):
            uid = self.all_user_ids[idx]
            if isinstance(actions, np.ndarray):
                action_dict[uid] = int(actions[idx])
            else:
                action_dict[uid] = int(actions[idx])

        # --- 0. Queue Incoming Pending Tasks based on Actions ---
        acted_pending_tasks = (
            dict(self._last_pending_tasks)
            if hasattr(self, "_last_pending_tasks") and self._last_pending_tasks
            else {}
        )

        if acted_pending_tasks:
            for uid, task in acted_pending_tasks.items():
                if uid not in self.active_users:
                    continue
                parsed_task = {
                    "task_id": task["task_id"],
                    "creation_time": task["creation_time"],
                    "deadline": task["deadline"],
                    "remain_data": task["data_size"],
                    "remain_workload": task["computation_workload"],
                    "owner_uid": uid,
                }

                act = action_dict.get(uid, 0)
                
                trace = {
                    "task_id": task["task_id"],
                    "owner_uid": uid,
                    "target_server": act,
                    "creation_time": task["creation_time"],
                    "deadline": task["deadline"],
                    "data_size": task["data_size"],
                    "workload": task["computation_workload"],
                    "action": act,
                    "upload_start": self.current_time,
                    "upload_end": self.current_time if act == 0 else None,
                    "compute_start": None,
                    "compute_end": None,
                    "status": "pending",
                }
                parsed_task["trace"] = trace

                if act == 0:
                    self.active_users[uid]["local_queue"].append(parsed_task)
                else:
                    target_server = act
                    parsed_task["target_server"] = target_server
                    self.active_users[uid]["transmit_queue"].append(parsed_task)

        # --- 1. Physics Engine: Process Time-Slice (dt) ---
        dt = self.time_interval

        # Pipeline A: Local Computation
        for uid, state in self.active_users.items():
            if state["local_queue"]:
                head_task = state["local_queue"][0]
                if head_task["trace"]["compute_start"] is None:
                    head_task["trace"]["compute_start"] = self.current_time
                head_task["remain_workload"] -= self.local_capacity * dt

                if head_task["remain_workload"] <= 0:
                    state["local_queue"].pop(0)
                    self._task_completed(head_task, rewards)
                elif self.current_time + dt > head_task["deadline"]:
                    state["local_queue"].pop(0)
                    self._task_dropped(head_task, rewards, reason="deadline")

        # Pipeline B: Transmission to Edge (OFDMA Bandwidth Sharing)
        # Step B.1: Count concurrent uploaders per server
        n_tx_per_server = {sid: 0 for sid in self.servers_info}
        for uid, state in self.active_users.items():
            if state["transmit_queue"]:
                target_sid = state["transmit_queue"][0]["target_server"]
                n_tx_per_server[target_sid] += 1

        # Step B.2: Compute per-user rate with bandwidth sharing and process
        for uid, state in self.active_users.items():
            if state["transmit_queue"]:
                head_task = state["transmit_queue"][0]
                target_sid = head_task["target_server"]
                n_tx = n_tx_per_server[target_sid]

                transmission_rate = self._compute_transmission_rate(
                    state["position"], target_sid, n_tx
                )

                head_task["remain_data"] -= transmission_rate * dt

                if head_task["remain_data"] <= 0:
                    state["transmit_queue"].pop(0)
                    head_task["trace"]["upload_end"] = self.current_time
                    self.server_queues[target_sid].append(head_task)
                elif self.current_time + dt > head_task["deadline"]:
                    state["transmit_queue"].pop(0)
                    self._task_dropped(head_task, rewards, reason="deadline")

        # Pipeline C: Edge Server Processor Sharing Computation
        for sid, queue in self.server_queues.items():
            if not queue:
                continue

            num_tasks = len(queue)
            shared_capacity = self.server_capacity / num_tasks

            completed_indices = []
            for i, task in enumerate(queue):
                if task["trace"]["compute_start"] is None:
                    task["trace"]["compute_start"] = self.current_time
                task["remain_workload"] -= shared_capacity * dt

                if task["remain_workload"] <= 0:
                    completed_indices.append(i)
                    self._task_completed(task, rewards)
                elif self.current_time + dt > task["deadline"]:
                    completed_indices.append(i)
                    self._task_dropped(task, rewards, reason="deadline")

            for i in sorted(completed_indices, reverse=True):
                queue.pop(i)

        # --- 2. Temporal Step Forward ---
        self.current_time += dt

        # --- 3. Update Network Lifecycle (Spawn/Leave) ---
        old_users = set(self.active_users.keys())
        self._update_spawns_and_positions()
        new_users = set(self.active_users.keys())

        # Identify users that just left
        left_users = old_users - new_users
        # for uid in left_users:
        #     uid_idx = self._get_uid_index(uid)
        #     rewards[uid_idx] -= 50.0  # Massive penalty for dropping off grid

        # --- 4. Fetch newly created tasks ---
        pending_tasks = {}
        while (
            self.event_ptr < len(self.task_events)
            and self.task_events[self.event_ptr]["time"] <= self.current_time
        ):
            ev = self.task_events[self.event_ptr]
            if ev["uid"] in self.active_users:
                pending_tasks[ev["uid"]] = ev["task"]
            self.event_ptr += 1

        self._last_pending_tasks = pending_tasks

        # --- 5. Formulate Multi-Agent Observation & Context ---
        # obs 使用 self._prev_server_congestions (t-1 时刻快照)
        obs = self._get_obs_array(pending_tasks)
        # obs 构建完毕后, 将当前真实拥塞快照缓存供下一步使用
        self._prev_server_congestions = self._snapshot_server_congestions()
        global_state = self._get_global_state()

        # Gymnasium standard: terminations and truncations as arrays
        terminated = self.current_time >= self.max_time

        # Build arrays for terminations and truncations
        terminations = np.zeros(self.n_agents, dtype=bool)
        truncations = np.zeros(self.n_agents, dtype=bool)

        for idx, uid in enumerate(self.all_user_ids):
            # Only terminate when the episode reaches max time (no early departure terminations)
            if terminated:
                terminations[idx] = True

        info = {
            "global_state": global_state,
            "pending_tasks": pending_tasks,
            "decision_tasks": acted_pending_tasks,
            "active_users": list(self.active_users.keys()),
            "left_users": list(left_users),
            "active_ue_masks": self._get_active_ue_masks(),
            "decision_masks": self._get_decision_masks(acted_pending_tasks),
        }

        return obs, rewards, terminations, truncations, info

    def _get_uid_index(self, uid):
        """Get the array index for a user ID."""
        return self.all_user_ids.index(int(uid))

    def _update_spawns_and_positions(self):
        """Handles the lifecycle of users (spawn/leave) and interpolates physical location."""
        to_remove = []
        for uid in list(self.active_users.keys()):
            meta = self.all_users_meta[uid]
            if self.current_time > meta["leave_time"]:
                to_remove.append(uid)
                self.drop_count_disconnection += len(
                    self.active_users[uid]["local_queue"]
                )

        for uid in to_remove:
            del self.active_users[uid]

        for uid, meta in self.all_users_meta.items():
            if uid not in self.active_users:
                if meta["spawn_time"] <= self.current_time <= meta["leave_time"]:
                    self.active_users[uid] = {
                        "position": np.array([0.0, 0.0]),
                        "local_queue": [],
                        "transmit_queue": [],
                    }

        for uid, state in self.active_users.items():
            state["position"] = self._get_interpolated_position(uid, self.current_time)

    def _get_interpolated_position(self, uid, t):
        """Returns precise simulated spatial coordinates based on JSON Trajectory."""
        traj = self.all_users_meta[uid].get("trajectory", [])
        if not traj:
            return np.array([0.0, 0.0])

        for i in range(len(traj) - 1):
            if traj[i]["time"] <= t <= traj[i + 1]["time"]:
                p1 = np.array(traj[i]["position"])
                p2 = np.array(traj[i + 1]["position"])
                dt = traj[i + 1]["time"] - traj[i]["time"]
                if dt == 0:
                    return p1
                alpha = (t - traj[i]["time"]) / dt
                return p1 + alpha * (p2 - p1)
        return np.array(traj[-1]["position"])

    def _compute_path_loss(self, dist_m):
        """3GPP TR 38.901 UMa LoS path loss @ 3.5 GHz (5G NR n78).

        Formula: PL = 28.0 + 22*log10(d_m) + 20*log10(fc_GHz)
        With fc = 3.5 GHz → PL = 38.9 + 22*log10(d_m).

        Assumption: outdoor macro deployment with primarily line-of-sight users
        (BS height 25 m, UE height 1.5 m). This matches typical dense urban
        5G n78 rollouts where blockage is modest. NLoS variant is too harsh
        for the OSM street-level scenarios used here.

        Returns path loss in dB.
        """
        d = max(dist_m, 1.0)  # Clamp to 1m minimum
        return 28.0 + 22.0 * math.log10(d) + 20.0 * math.log10(3.5)

    def _compute_snr_db(self, dist_m):
        """Compute SNR in dB for uplink from UE to BS."""
        pl = self._compute_path_loss(dist_m)
        rx_power_dbm = self.ue_tx_power_dbm - pl
        return rx_power_dbm - self.noise_floor_dbm

    def _compute_spectral_efficiency(self, dist_m):
        """Compute spectral efficiency log2(1+SNR) in bits/s/Hz."""
        snr_db = self._compute_snr_db(dist_m)
        snr_linear = 10.0 ** (snr_db / 10.0)
        return math.log2(1.0 + snr_linear)

    def _compute_transmission_rate(self, ue_position, server_id, n_concurrent_tx):
        """
        Compute per-user uplink rate (MB/s) with OFDMA bandwidth sharing.
        R = (W / N_tx) * log2(1 + SNR)
        """
        dist = np.linalg.norm(ue_position - self.servers_info[server_id]["position"])
        se = self._compute_spectral_efficiency(dist)
        bw_per_user = self.uplink_bandwidth_hz / max(n_concurrent_tx, 1)
        rate_bps = bw_per_user * se
        rate_mbps = rate_bps / (8.0 * 1e6)  # bits/s -> MB/s
        return rate_mbps

    def _task_completed(self, task, rewards):
        uid = task["owner_uid"]
        cost_time = self.current_time - task["creation_time"]

        self.completed_tasks += 1
        self.total_completion_time += cost_time

        # Deadline-normalized reward: R = R_base * (1 - cost/budget)
        time_budget = task["deadline"] - task["creation_time"]
        time_ratio = cost_time / max(time_budget, 0.1)
        reward = 50.0 * max(0.0, 1.0 - time_ratio)

        uid_idx = self._get_uid_index(uid)
        rewards[uid_idx] += reward

        task["trace"]["compute_end"] = self.current_time
        task["trace"]["status"] = "completed"
        self.task_traces.append(task["trace"])

    def _task_dropped(self, task, rewards, reason="deadline"):
        uid = task["owner_uid"]
        if reason == "deadline":
            self.drop_count_deadline += 1
        else:
            self.drop_count_disconnection += 1

        uid_idx = self._get_uid_index(uid)
        # Drop penalty removed to encourage optimistic exploration
        # rewards[uid_idx] -= 20.0

        task["trace"]["status"] = f"dropped_{reason}"
        task["trace"]["compute_end"] = self.current_time # Serves as drop_time
        self.task_traces.append(task["trace"])

    def _get_active_ue_masks(self):
        """
        Returns a boolean array indicating which users are currently active.
        Shape: (n_agents,)
        """
        masks = np.zeros(self.n_agents, dtype=bool)
        for idx, uid in enumerate(self.all_user_ids):
            if uid in self.active_users:
                masks[idx] = True
        return masks

    def _get_decision_masks(self, decision_tasks):
        """
        Returns a boolean array indicating which users are currently making decisions.
        Shape: (n_agents,)
        """
        masks = np.zeros(self.n_agents, dtype=bool)
        if not decision_tasks:
            return masks

        decision_uids = set(int(uid) for uid in decision_tasks.keys())
        for idx, uid in enumerate(self.all_user_ids):
            if int(uid) in decision_uids:
                masks[idx] = True
        return masks

    def _snapshot_server_congestions(self):
        """
        Computes the current congestion snapshot for all servers, normalized to [0,1].
        Returns a dict: {server_id: (tx_congestion, compute_congestion)}
        """
        snapshot = {}
        for sid in self.servers_info.keys():
            compute_load = len(self.server_queues[sid])
            tx_load = sum(
                1 for u_state in self.active_users.values()
                if u_state["transmit_queue"]
                and u_state["transmit_queue"][0].get("target_server") == sid
            )
            snapshot[sid] = (tx_load / 10.0, compute_load / 10.0)
        return snapshot

    def _get_obs_array(self, pending_tasks):
        """
        Returns observations as a numpy array of shape (n_agents, obs_dim).
        For inactive users, returns zero observations.

        Per-server congestion uses t-1 snapshot (self._prev_server_congestions) so that
        all agents and their Critics observe the congestion one step behind reality.
        """
        obs_array = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)

        for idx, uid in enumerate(self.all_user_ids):
            if uid in self.active_users:
                state = self.active_users[uid]
                has_task = 1.0 if uid in pending_tasks else 0.0
                task_info = pending_tasks.get(
                    uid, {"data_size": 0.0, "computation_workload": 0.0, "deadline": 0.0, "creation_time": 0.0}
                )
                budget = float(task_info["deadline"] - task_info["creation_time"]) if has_task else 0.0

                channel_qualities = []
                tx_congestions = []
                cp_congestions = []
                for sid, s_info in self.servers_info.items():
                    # Normalized spectral efficiency (3GPP UMa NLoS based)
                    distance = np.linalg.norm(state["position"] - s_info["position"])
                    se = self._compute_spectral_efficiency(distance)
                    channel_qualities.append(min(se / self.max_spectral_efficiency, 1.0))

                    # Delayed congestion: read from t-1 snapshot, split tx & compute
                    tx_c, cp_c = self._prev_server_congestions.get(sid, (0.0, 0.0))
                    tx_congestions.append(tx_c)
                    cp_congestions.append(cp_c)

                obs_array[idx] = np.array(
                    [
                        len(self.active_users) / 100.0,
                        has_task,
                        task_info["data_size"],
                        task_info["computation_workload"] / 1000.0,
                        budget,
                        len(state["local_queue"]),
                        *channel_qualities,
                        *tx_congestions,
                        *cp_congestions,
                    ],
                    dtype=np.float32,
                )
            # else: already zeros

        return obs_array

    def _get_obs(self, pending_tasks):
        """Legacy method for backward compatibility - returns dict format. Uses t-1 congestion."""
        obs_dict = {}
        for uid, state in self.active_users.items():
            has_task = 1.0 if uid in pending_tasks else 0.0
            task_info = pending_tasks.get(
                uid, {"data_size": 0.0, "computation_workload": 0.0, "deadline": 0.0, "creation_time": 0.0}
            )
            budget = float(task_info["deadline"] - task_info["creation_time"]) if has_task else 0.0

            channel_qualities = []
            tx_congestions = []
            cp_congestions = []
            for sid, s_info in self.servers_info.items():
                distance = np.linalg.norm(state["position"] - s_info["position"])
                se = self._compute_spectral_efficiency(distance)
                channel_qualities.append(min(se / self.max_spectral_efficiency, 1.0))
                tx_c, cp_c = self._prev_server_congestions.get(sid, (0.0, 0.0))
                tx_congestions.append(tx_c)
                cp_congestions.append(cp_c)

            obs_dict[uid] = np.array(
                [
                    len(self.active_users) / 100.0,
                    has_task,
                    task_info["data_size"],
                    task_info["computation_workload"] / 1000.0,
                    budget,
                    len(state["local_queue"]),
                    *channel_qualities,
                    *tx_congestions,
                    *cp_congestions,
                ],
                dtype=np.float32,
            )

        return obs_dict

    def _get_all_users_obs(self, pending_tasks):
        """Legacy method for backward compatibility - returns dict format. Uses t-1 congestion."""
        all_obs = {}
        for uid in self.all_users_meta.keys():
            if uid in self.active_users:
                state = self.active_users[uid]
                has_task = 1.0 if uid in pending_tasks else 0.0
                task_info = pending_tasks.get(
                    uid, {"data_size": 0.0, "computation_workload": 0.0, "deadline": 0.0, "creation_time": 0.0}
                )
                budget = float(task_info["deadline"] - task_info["creation_time"]) if has_task else 0.0

                channel_qualities = []
                tx_congestions = []
                cp_congestions = []
                for sid, s_info in self.servers_info.items():
                    distance = np.linalg.norm(state["position"] - s_info["position"])
                    se = self._compute_spectral_efficiency(distance)
                    channel_qualities.append(min(se / self.max_spectral_efficiency, 1.0))
                    tx_c, cp_c = self._prev_server_congestions.get(sid, (0.0, 0.0))
                    tx_congestions.append(tx_c)
                    cp_congestions.append(cp_c)

                all_obs[uid] = np.array(
                    [
                        len(self.active_users) / 100.0,
                        has_task,
                        task_info["data_size"],
                        task_info["computation_workload"] / 1000.0,
                        budget,
                        len(state["local_queue"]),
                        *channel_qualities,
                        *tx_congestions,
                        *cp_congestions,
                    ],
                    dtype=np.float32,
                )
            else:
                all_obs[uid] = np.zeros(self.observation_space.shape, dtype=np.float32)
        return all_obs

    def _get_global_state(self):
        """
        The "Deconfounded Critic" Background Stratum (E(N)).
        Returns macroscopic congestion variables for backdoor adjustment.
        """
        completed = self.completed_tasks
        drops = self.drop_count_disconnection + self.drop_count_deadline
        total_tasks = completed + drops

        avg_cost = (self.total_completion_time / completed) if completed > 0 else 0.0
        drop_rate = (drops / total_tasks * 100) if total_tasks > 0 else 0.0

        return {
            "active_user_count": len(self.active_users),
            "global_load_ratio": sum(len(q) for q in self.server_queues.values())
            / max(1, len(self.servers_info)),
            "server_queues": {sid: len(q) for sid, q in self.server_queues.items()},
            "drop_count_disconnection": self.drop_count_disconnection,
            "drop_count_deadline": self.drop_count_deadline,
            "completed_tasks": self.completed_tasks,
            "total_completion_time": self.total_completion_time,
            "total_tasks": total_tasks,
            "avg_cost": avg_cost,
            "drop_rate": drop_rate,
        }

    def get_render_state(self):
        """
        Visualization Interface.
        """
        return {
            "time": self.current_time,
            "bounds": (self.scene_config["width"], self.scene_config["height"]),
            "servers": {
                sid: info["position"].tolist()
                for sid, info in self.servers_info.items()
            },
            "users": {
                uid: state["position"].tolist()
                for uid, state in self.active_users.items()
            },
        }

    def render(self):
        """
        Render the environment. Returns render state dict for external rendering.
        """
        if self.render_mode == "human":
            return self.get_render_state()
        elif self.render_mode == "rgb_array":
            # Return state for external rendering (matplotlib/pygame)
            return self.get_render_state()
        return None

    def close(self):
        """
        Clean up environment resources.
        """
        self.active_users.clear()
        self.server_queues.clear()
