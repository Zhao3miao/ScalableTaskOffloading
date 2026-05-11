import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from utils.utils import to_tensor
from models.shared_actor import SharedBlindActor
from models.deconfound_critic import DeconfoundedCritic


class CDMAPPOAgent:

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.obs_dim = config["obs_dim"]
        self.act_dim = config["act_dim"]
        self.n_servers = config.get("n_servers", 3)

        self.confounder_dim = config.get("confounder_dim", 2 * self.n_servers)

        self.clip_param = config.get("clip_param", 0.2)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.value_coef = config.get("value_coef", 0.5)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.update_epochs = config.get("update_epochs", 4)
        self.num_minibatches = config.get("num_minibatches", 4)
        self.batch_size = config.get("batch_size", 64)

        # CD-MAPPO specific hyperparameters
        self.vdo_alpha = config.get("vdo_alpha", 0.7)
        self.lambda_cf = config.get("lambda_cf", 0.5)
        self.K_actor = config.get("K_actor", 4)
        self.use_deconf_critic = config.get("use_deconf_critic", True)

       
        self._vbar_values = None
        self._step_counter = 0

        #   Shared Actor
        self.actor = SharedBlindActor(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_size=config.get("hidden_size_actor", 128),
        ).to(self.device)

        # Critic: AttentionCritic + DEBA
        self.critic = DeconfoundedCritic(
            obs_dim=self.obs_dim,
            n_servers=self.n_servers,
            hidden_size=config.get("hidden_size_critic", 256),
            n_heads=config.get("n_heads", 4),
            memory_size=config.get("deba_memory_size", 5000),
            vdo_sample_size=config.get("deba_sample_size", 64),
            confounder_dim=self.confounder_dim,
        ).to(self.device)

        lr = config.get("initial_lr", 3e-4)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr, eps=1e-5)


    def _infer_active_mask(self, obs):
        if len(obs.shape) == 2:
            obs = np.expand_dims(obs, axis=0)
        return np.any(obs != 0, axis=-1).astype(np.float32)

    def init_episode(self, env_num, n_agents, max_steps):
        self._vbar_values = np.zeros(
            (max_steps, env_num, n_agents, 1), dtype=np.float32
        )
        self._step_counter = 0

  
    def predict(self, obs):
        if len(obs.shape) == 2:
            obs = np.expand_dims(obs, axis=0)

        N_envs, n_agents = obs.shape[:2]
        obs_flat = obs.reshape(-1, self.obs_dim)
        obs_tensor = to_tensor(obs_flat).to(self.device)

        with torch.no_grad():
            logits = self.actor(obs_tensor)
            action = torch.argmax(logits, dim=-1)

        return action.reshape(N_envs, n_agents).detach().cpu().numpy()

  
    def sample(self, obs, server_congestions=None):
        if len(obs.shape) == 2:
            obs = np.expand_dims(obs, axis=0)

        N_envs, n_agents = obs.shape[:2]
        active_mask = self._infer_active_mask(obs)

        if server_congestions is None:
            server_congestions = np.zeros((N_envs, self.confounder_dim), dtype=np.float32)

        obs_flat = obs.reshape(-1, self.obs_dim)
        obs_tensor = to_tensor(obs_flat).to(self.device)

        with torch.no_grad():
            
            logits = self.actor(obs_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action).unsqueeze(-1)

            obs_3d = to_tensor(obs).to(self.device)
            mask_t = to_tensor(active_mask).to(self.device)
            cong_t = to_tensor(server_congestions).to(self.device)
            V_factual = self.critic(obs_3d, mask_t, cong_t)  # (N_envs, n_agents, 1)

            # CD-MAPPO 增量: V_do (dependent on confounder samples)
            if self.use_deconf_critic:
                V_do = self.critic.compute_deconfounded_value(obs_3d, mask_t)

        V_factual_np = V_factual.detach().cpu().numpy()
        if self.use_deconf_critic:
            # V_mix = α*V_do + (1-α)*V_factual
            V_do_np = V_do.detach().cpu().numpy()
            V_mix = self.vdo_alpha * V_do_np + (1.0 - self.vdo_alpha) * V_factual_np
        else:
            V_mix = V_factual_np

        if self._vbar_values is not None:
            self._vbar_values[self._step_counter] = V_mix
        self._step_counter += 1

        # Update DEBA
        self.critic.update_prior(cong_t)

        return (
            V_factual_np,
            action.reshape(N_envs, n_agents).detach().cpu().numpy(),
            log_prob.reshape(N_envs, n_agents, 1).detach().cpu().numpy(),
        )

    def compute_deconf_advantages(self, storage, gamma=0.99, gae_lambda=0.95,
                                  current_episode=None, total_episodes=None):
        if not self.use_deconf_critic:
            return

        T = storage.cur_step
        lastgaelam = np.zeros_like(storage.advantages[0])

        for t in reversed(range(T)):
            if t == T - 1:
                nextnonterminal = 1.0 - storage.dones[t]
                next_vbar = np.zeros_like(self._vbar_values[t])
            else:
                nextnonterminal = 1.0 - storage.dones[t + 1]
                next_vbar = self._vbar_values[t + 1]

            delta = (
                storage.rewards[t]
                + gamma * next_vbar * nextnonterminal
                - self._vbar_values[t]
            )
            lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            storage.advantages[t] = lastgaelam

    def learn(self, rollout_storage):
        actor_loss_epoch = 0
        critic_loss_epoch = 0
        entropy_epoch = 0
        lr = None

        total_steps = rollout_storage.get_size()
        if total_steps < self.batch_size:
            return 0.0, 0.0, 0.0, lr

        effective_batch_size = min(self.batch_size, total_steps)
        num_minibatches = min(self.num_minibatches, max(1, total_steps // 4))
        minibatch_size = max(1, effective_batch_size // num_minibatches)

        indexes = np.arange(total_steps)

        for epoch in range(self.update_epochs):
            np.random.shuffle(indexes)
            for start in range(0, total_steps, minibatch_size):
                end = min(start + minibatch_size, total_steps)
                sample_idx = indexes[start:end]
                if len(sample_idx) < 2:
                    continue

                (
                    batch_obs,
                    batch_action,
                    batch_logprob,
                    batch_adv,
                    batch_return,
                    batch_value,
                    batch_active_masks,
                    batch_decision_masks,
                    batch_congestions,
                ) = rollout_storage.sample_batch(sample_idx)

                B, n_agents = batch_obs.shape[:2]

                if batch_congestions is not None:
                    cong_tensor = to_tensor(batch_congestions).to(self.device)
                else:
                    cong_tensor = torch.zeros(B, self.confounder_dim, device=self.device)

                batch_obs_flat = batch_obs.reshape(-1, self.obs_dim)
                batch_action_flat = batch_action.reshape(-1)
                batch_logprob_flat = batch_logprob.reshape(-1, 1)
                batch_adv_flat = batch_adv.reshape(-1, 1)
                batch_return_flat = batch_return.reshape(-1, 1)
                batch_active_masks_flat = batch_active_masks.reshape(-1, 1)
                batch_decision_masks_flat = batch_decision_masks.reshape(-1, 1)

                batch_obs_tensor = to_tensor(batch_obs_flat).to(self.device)
                batch_action_tensor = to_tensor(batch_action_flat, dtype=torch.long).to(self.device)
                batch_logprob_tensor = to_tensor(batch_logprob_flat).to(self.device)
                batch_adv_tensor = to_tensor(batch_adv_flat).to(self.device)
                batch_return_tensor = to_tensor(batch_return_flat).to(self.device)
                batch_active_masks_tensor = to_tensor(batch_active_masks_flat).to(self.device)
                batch_decision_masks_tensor = to_tensor(batch_decision_masks_flat).to(self.device)

                valid = batch_decision_masks_tensor.squeeze(-1) > 0
                if torch.any(valid):
                    adv_valid = batch_adv_tensor.squeeze(-1)[valid]
                    adv_mean = adv_valid.mean()
                    adv_std = adv_valid.std() + 1e-8
                    batch_adv_tensor = (batch_adv_tensor - adv_mean) / adv_std
                else:
                    batch_adv_tensor = (batch_adv_tensor - batch_adv_tensor.mean()) / (
                        batch_adv_tensor.std() + 1e-8
                    )

                logits = self.actor(batch_obs_tensor)
                dist = Categorical(logits=logits)
                new_logprob = dist.log_prob(batch_action_tensor).unsqueeze(-1)
                entropy = dist.entropy().unsqueeze(-1)

                ratio = torch.exp(new_logprob - batch_logprob_tensor)
                surr1 = -batch_adv_tensor * ratio
                surr2 = -batch_adv_tensor * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                pg_loss_raw = torch.max(surr1, surr2)

                decision_denom = torch.clamp(batch_decision_masks_tensor.sum(), min=1.0)
                pg_loss = (pg_loss_raw * batch_decision_masks_tensor).sum() / decision_denom
                ent_loss = (entropy * batch_decision_masks_tensor).sum() / decision_denom

                # CD-MAPPO : Counterfactual loss
                cf_loss = torch.tensor(0.0, device=self.device)
                if self.lambda_cf > 0 and self.K_actor > 0:
                    cf_loss = self._compute_actor_cf_loss(
                        batch_obs_tensor, batch_decision_masks_tensor,
                        B, n_agents,
                    )

                actor_loss = pg_loss - self.entropy_coef * ent_loss + self.lambda_cf * cf_loss

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                batch_obs_3d = to_tensor(batch_obs).to(self.device)
                batch_mask_2d = to_tensor(
                    batch_active_masks.squeeze(-1), dtype=torch.float32
                ).to(self.device)

                values = self.critic(batch_obs_3d, batch_mask_2d, cong_tensor)
                values_flat = values.reshape(-1, 1)

                critic_mse = (values_flat - batch_return_tensor) ** 2
                active_denom = torch.clamp(batch_active_masks_tensor.sum(), min=1.0)
                critic_loss = 0.5 * self.value_coef * (
                    critic_mse * batch_active_masks_tensor
                ).sum() / active_denom

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                actor_loss_epoch += pg_loss.item()
                critic_loss_epoch += critic_loss.item()
                entropy_epoch += ent_loss.item()

        num_updates = self.update_epochs * (
            (total_steps + minibatch_size - 1) // minibatch_size
        )
        if num_updates > 0:
            actor_loss_epoch /= num_updates
            critic_loss_epoch /= num_updates
            entropy_epoch /= num_updates

        return actor_loss_epoch, critic_loss_epoch, entropy_epoch, lr

    def _compute_actor_cf_loss(self, obs_flat_tensor, decision_tensor,
                               B, n_agents):
        """
        Counterfactual Policy Gradient: Under the counterfactual congestion L_cf, minimize the expected cost of offloading to a congested server.

        Multi-modal bottleneck version: L_cf = [tx_1..tx_K, cp_1..cp_K] (dimension 2*n_servers)
        For each group L_cf, define the counterfactual cost c(a, L_cf):
          - a = local  → c = 0
          - a = BS_i   → c = L_cf[tx_i] + L_cf[cp_i] (Sum of transmission + computational congestion)

        Minimize E_{a ~ π}[c(a, L_cf)] = Σ_i π(BS_i|obs_cf) · (L_tx_i + L_cp_i)

        Difference from the old version: The new cost considers both transmission and computational bottlenecks.
        The Actor learns that 'when any modality of a certain BS is congested, it should fall back to local computation'.
        """
        total_agents = B * n_agents
        cf_loss_total = torch.tensor(0.0, device=self.device)

        # Sample K_actor groups at once to avoid calling _get_vdo_samples_independent repeatedly
        L_cf_all = self.critic._get_vdo_samples_independent()[:self.K_actor]  # (K, confounder_dim)

        for k in range(L_cf_all.shape[0]):
            L_cf = L_cf_all[k:k+1]  # (1, confounder_dim=2*n_servers)

            # Construct counterfactual obs: replace the last 2*n_servers dimensions (tx + cp)
            obs_cf = obs_flat_tensor.clone()  # (B*n_agents, obs_dim)
            obs_cf[:, -self.confounder_dim:] = L_cf.expand(total_agents, -1)

            # Actor policy under counterfactual obs
            logits_cf = self.actor(obs_cf)
            probs_cf = torch.softmax(logits_cf, dim=-1)  # (BN, act_dim)

            # Expected counterfactual cost: Σ_i π(BS_i) * (L_tx_i + L_cp_i)
            offload_probs = probs_cf[:, 1:1 + self.n_servers]       # (BN, n_servers)
            L_tx = L_cf[:, :self.n_servers]                          # (1, n_servers)
            L_cp = L_cf[:, self.n_servers:2 * self.n_servers]        # (1, n_servers)
            cost_per_bs = (L_tx + L_cp).expand(total_agents, -1)     # (BN, n_servers)
            expected_cost = (offload_probs * cost_per_bs).sum(dim=-1, keepdim=True)

            cf_loss = (expected_cost * decision_tensor).sum() / torch.clamp(
                decision_tensor.sum(), min=1.0
            )
            cf_loss_total = cf_loss_total + cf_loss

        return cf_loss_total / max(self.K_actor, 1)

    
    def save(self, path):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "config": self.config,
            },
            path,
        )

    def restore(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        if "actor_optimizer" in ckpt:
            self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        if "critic_optimizer" in ckpt:
            self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
