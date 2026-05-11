import numpy as np


class RuleBasedAgent:
    """Base class for deterministic or seeded heuristic MEC offloading policies."""

    def __init__(self, config):
        self.config = config
        self.obs_dim = config["obs_dim"]
        self.act_dim = config["act_dim"]
        self.max_n = config.get("max_n", 100)
        self.seed = config.get("seed", 42)
        self.rng = np.random.RandomState(self.seed)
        self.n_servers = config.get("n_servers", max(1, self.act_dim - 1))

    def predict(self, obs):
        raise NotImplementedError

    def sample(self, obs, *args, **kwargs):
        action = self.predict(obs)
        value = np.zeros((*action.shape, 1), dtype=np.float32)
        logprob = np.zeros((*action.shape, 1), dtype=np.float32)
        return value, action, logprob

    def learn(self, rollout_storage):
        return 0.0, 0.0, 0.0, None

    def save(self, path):
        return None

    def restore(self, path):
        return None

    def _ensure_batch(self, obs):
        if len(obs.shape) == 2:
            return np.expand_dims(obs, axis=0), True
        return obs, False

    def _finish(self, action, squeezed):
        if squeezed:
            return action[0]
        return action

    def _decision_mask(self, obs):
        return obs[:, :, 1] > 0.5

    def _server_slices(self):
        channel_start = 6
        channel_end = channel_start + self.n_servers
        tx_start = channel_end
        tx_end = tx_start + self.n_servers
        cp_start = tx_end
        cp_end = cp_start + self.n_servers
        return channel_start, channel_end, tx_start, tx_end, cp_start, cp_end


class LocalOnlyAgent(RuleBasedAgent):
    """Always execute tasks locally."""

    def predict(self, obs):
        obs, squeezed = self._ensure_batch(obs)
        action = np.zeros(obs.shape[:2], dtype=np.int64)
        return self._finish(action, squeezed)


class RandomAgent(RuleBasedAgent):
    """Uniformly random action over local and all edge servers for task UEs."""

    def predict(self, obs):
        obs, squeezed = self._ensure_batch(obs)
        action = np.zeros(obs.shape[:2], dtype=np.int64)
        decision = self._decision_mask(obs)
        action[decision] = self.rng.randint(0, self.act_dim, size=int(decision.sum()))
        return self._finish(action, squeezed)


class RandomEdgeOnlyAgent(RuleBasedAgent):
    """Uniformly random edge-server selection for task UEs."""

    def predict(self, obs):
        obs, squeezed = self._ensure_batch(obs)
        action = np.zeros(obs.shape[:2], dtype=np.int64)
        decision = self._decision_mask(obs)
        if np.any(decision):
            action[decision] = self.rng.randint(
                1, self.act_dim, size=int(decision.sum())
            )
        return self._finish(action, squeezed)


class UniformSplitAgent(RuleBasedAgent):
    """Deterministically split task UEs across local and edge actions."""

    def __init__(self, config):
        super().__init__(config)
        self.next_action = 0

    def predict(self, obs):
        obs, squeezed = self._ensure_batch(obs)
        action = np.zeros(obs.shape[:2], dtype=np.int64)
        decision = self._decision_mask(obs)
        n_decisions = int(decision.sum())
        if n_decisions > 0:
            assigned = (np.arange(n_decisions) + self.next_action) % self.act_dim
            action[decision] = assigned.astype(np.int64)
            self.next_action = int((self.next_action + n_decisions) % self.act_dim)
        return self._finish(action, squeezed)


class BestChannelAgent(RuleBasedAgent):
    """Offload to the edge server with the best current channel quality."""

    def predict(self, obs):
        obs, squeezed = self._ensure_batch(obs)
        action = np.zeros(obs.shape[:2], dtype=np.int64)
        decision = self._decision_mask(obs)
        ch_start, ch_end, *_ = self._server_slices()
        channels = obs[:, :, ch_start:ch_end]
        action[decision] = 1 + np.argmax(channels[decision], axis=-1)
        return self._finish(action, squeezed)


class LeastCongestedAgent(RuleBasedAgent):
    """Offload to the edge server with the smallest delayed tx+compute load."""

    def predict(self, obs):
        obs, squeezed = self._ensure_batch(obs)
        action = np.zeros(obs.shape[:2], dtype=np.int64)
        decision = self._decision_mask(obs)
        _, _, tx_start, tx_end, cp_start, cp_end = self._server_slices()
        congestion = obs[:, :, tx_start:tx_end] + obs[:, :, cp_start:cp_end]
        action[decision] = 1 + np.argmin(congestion[decision], axis=-1)
        return self._finish(action, squeezed)


class GreedyDelayAgent(RuleBasedAgent):
    """Choose the smallest rough delay estimate using only current observations."""

    def __init__(self, config):
        super().__init__(config)
        self.local_capacity = config.get("local_capacity", 250.0)
        self.edge_capacity = config.get("edge_capacity", 4000.0)
        self.tx_load_scale = config.get("tx_load_scale", 10.0)
        self.cp_load_scale = config.get("cp_load_scale", 10.0)
        self.min_channel = config.get("min_channel", 1e-3)

    def predict(self, obs):
        obs, squeezed = self._ensure_batch(obs)
        action = np.zeros(obs.shape[:2], dtype=np.int64)
        decision = self._decision_mask(obs)
        if not np.any(decision):
            return self._finish(action, squeezed)

        ch_start, ch_end, tx_start, tx_end, cp_start, cp_end = self._server_slices()
        data_size = obs[:, :, 2]
        workload = obs[:, :, 3] * 1000.0
        local_queue = obs[:, :, 5]
        channels = np.maximum(obs[:, :, ch_start:ch_end], self.min_channel)
        tx = obs[:, :, tx_start:tx_end]
        cp = obs[:, :, cp_start:cp_end]

        local_cost = (
            workload / self.local_capacity
            + local_queue * workload / self.local_capacity
        )
        edge_upload = data_size[:, :, None] * (1.0 + self.tx_load_scale * tx) / channels
        edge_compute = workload[:, :, None] * (1.0 + self.cp_load_scale * cp) / self.edge_capacity
        edge_cost = edge_upload + edge_compute

        all_costs = np.concatenate([local_cost[:, :, None], edge_cost], axis=-1)
        action[decision] = np.argmin(all_costs[decision], axis=-1)
        return self._finish(action, squeezed)


RULE_BASED_AGENT_CLASSES = {
    "local_only": LocalOnlyAgent,
    "random": RandomAgent,
    "random_edge_only": RandomEdgeOnlyAgent,
    "uniform_split": UniformSplitAgent,
    "best_channel": BestChannelAgent,
    "nearest_edge": BestChannelAgent,
    "least_congested": LeastCongestedAgent,
    "greedy": LeastCongestedAgent,
    "greedy_delay": GreedyDelayAgent,
}
