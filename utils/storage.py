import numpy as np
import torch


class MAPPORolloutStorage:

    def __init__(
        self,
        step_nums,
        env_num,
        obs_space,
        act_space,
        cotinueios_action=False,
    ):
        self.obs_space = obs_space
        self.act_space = act_space
        self.n_agents = obs_space.shape[0]
        self.step_nums = step_nums
        self.env_num = env_num
        self.cur_step = 0

        self.obs = np.zeros(
            (
                step_nums,
                env_num,
            )
            + obs_space.shape,
            dtype="float32",
        )
        if cotinueios_action:
            self.actions = np.zeros(
                (
                    step_nums,
                    env_num,
                )
                + act_space.shape,
                dtype="float32",
            )
        else:
            self.actions = np.zeros(
                (step_nums, env_num, self.n_agents), dtype="int64"
            )  

        self.rewards = np.zeros((step_nums, env_num, self.n_agents, 1), dtype="float32")
        self.values = np.zeros((step_nums, env_num, self.n_agents, 1), dtype="float32")
        self.returns = np.zeros_like(self.values)
        self.advantages = np.zeros_like(self.values)
        self.logprobs = np.zeros(
            (step_nums, env_num, self.n_agents, 1), dtype="float32"
        )


        self.active_masks = np.ones(
            (step_nums, env_num, self.n_agents, 1), dtype="float32"
        )

        self.decision_masks = np.zeros(
            (step_nums, env_num, self.n_agents, 1), dtype="float32"
        )
        self.dones = np.zeros((step_nums, env_num, self.n_agents, 1), dtype="float32")

        obs_dim = obs_space.shape[-1]
        self.n_servers = (obs_dim - 6) // 3
        self.confounder_dim = 2 * self.n_servers
        self.congestions = np.zeros((step_nums, env_num, self.confounder_dim), dtype="float32")

    def append(
        self,
        obs,
        action,
        logprob,
        reward,
        value,
        done,
        active_mask,
        decision_mask=None,
        congestion=None,
    ):
        
        self.obs[self.cur_step] = obs
        self.actions[self.cur_step] = action
        self.logprobs[self.cur_step] = logprob
        self.rewards[self.cur_step] = reward
        self.values[self.cur_step] = value
        self.dones[self.cur_step] = done
        self.active_masks[self.cur_step] = active_mask
        if decision_mask is None:
            decision_mask = active_mask
        self.decision_masks[self.cur_step] = decision_mask
        if congestion is not None:
            self.congestions[self.cur_step] = congestion

        self.cur_step += 1

    def compute_gae(self, gamma=0.99, gae_lambda=0.95):
        
        lastgaelam = np.zeros((self.env_num, self.n_agents, 1), dtype="float32")

        for t in reversed(range(self.cur_step)):
            if t == self.cur_step - 1:
                nextnonterminal = 1.0 - self.dones[t]
                nextvalues = np.zeros_like(self.values[t])
            else:
                nextnonterminal = 1.0 - self.dones[t + 1]
                nextvalues = self.values[t + 1]

            delta = (
                self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
            )
            lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            self.advantages[t] = lastgaelam
            self.returns[t] = self.values[t] + self.advantages[t]

    def sample_batch(self, idx):
        
        valid_steps = self.cur_step

        b_obs = self.obs[:valid_steps].reshape((valid_steps * self.env_num,) + self.obs_space.shape)
        if len(self.actions.shape) == 4:
            b_actions = self.actions[:valid_steps].reshape((valid_steps * self.env_num,) + self.act_space.shape)
        else:
            b_actions = self.actions[:valid_steps].reshape((valid_steps * self.env_num, self.n_agents))
            
        b_logprobs = self.logprobs[:valid_steps].reshape((valid_steps * self.env_num, self.n_agents, 1))
        b_values = self.values[:valid_steps].reshape((valid_steps * self.env_num, self.n_agents, 1))
        b_advantages = self.advantages[:valid_steps].reshape((valid_steps * self.env_num, self.n_agents, 1))
        b_returns = self.returns[:valid_steps].reshape((valid_steps * self.env_num, self.n_agents, 1))
        b_active_masks = self.active_masks[:valid_steps].reshape((valid_steps * self.env_num, self.n_agents, 1))
        b_decision_masks = self.decision_masks[:valid_steps].reshape((valid_steps * self.env_num, self.n_agents, 1))
        b_congestions = self.congestions[:valid_steps].reshape((valid_steps * self.env_num, self.confounder_dim))

        return (
            b_obs[idx],
            b_actions[idx],
            b_logprobs[idx],
            b_advantages[idx],
            b_returns[idx],
            b_values[idx],
            b_active_masks[idx],
            b_decision_masks[idx],
            b_congestions[idx],
        )

    def get_size(self):
        
        return self.cur_step * self.env_num

    def reset(self):
        
        self.cur_step = 0
        self.obs.fill(0)
        self.actions.fill(0)
        self.rewards.fill(0)
        self.values.fill(0)
        self.logprobs.fill(0)
        self.dones.fill(0)
        self.active_masks.fill(1.0)
        self.decision_masks.fill(0.0)
        self.congestions.fill(0)


class RolloutStorage:
    def __init__(self, step_nums, env_num, obs_space, act_space):
        self.obs = np.zeros((step_nums, env_num) + obs_space.shape, dtype="float32")
        self.actions = np.zeros((step_nums, env_num) + act_space.shape, dtype="float32")
        self.logprobs = np.zeros((step_nums, env_num), dtype="float32")
        self.rewards = np.zeros((step_nums, env_num), dtype="float32")
        self.dones = np.zeros((step_nums, env_num), dtype="float32")
        self.values = np.zeros((step_nums, env_num), dtype="float32")

        self.step_nums = step_nums
        self.obs_space = obs_space
        self.act_space = act_space

        self.cur_step = 0

    def append(self, obs, action, logprob, reward, done, value):
        self.obs[self.cur_step] = obs
        self.actions[self.cur_step] = action
        self.logprobs[self.cur_step] = logprob
        self.rewards[self.cur_step] = reward
        self.dones[self.cur_step] = done
        self.values[self.cur_step] = value

        self.cur_step = (self.cur_step + 1) % self.step_nums

    def compute_returns(self, value, done, gamma=0.99, gae_lambda=0.95):
        # gamma: discounting factor
        # gae_lambda: Lambda parameter for calculating N-step advantage
        advantages = np.zeros_like(self.rewards)
        lastgaelam = 0
        for t in reversed(range(self.step_nums)):
            if t == self.step_nums - 1:
                nextnonterminal = 1.0 - done
                nextvalues = value.reshape(1, -1)
            else:
                nextnonterminal = 1.0 - self.dones[t + 1]
                nextvalues = self.values[t + 1]
            delta = (
                self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
            )
            advantages[t] = lastgaelam = (
                delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            )
        returns = advantages + self.values
        self.returns = returns
        self.advantages = advantages
        return advantages, returns

    def sample_batch(self, idx):
        # flatten rollout
        b_obs = self.obs.reshape((-1,) + self.obs_space.shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + self.act_space.shape)
        b_advantages = self.advantages.reshape(-1)
        b_returns = self.returns.reshape(-1)
        b_values = self.values.reshape(-1)

        return (
            b_obs[idx],
            b_actions[idx],
            b_logprobs[idx],
            b_advantages[idx],
            b_returns[idx],
            b_values[idx],
        )
