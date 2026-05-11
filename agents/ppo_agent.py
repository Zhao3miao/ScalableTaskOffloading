import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from utils.utils import to_tensor
from models.shared_actor import SharedBlindActor


class PPOAgent:
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.obs_dim = config["obs_dim"]
        self.act_dim = config["act_dim"]

        self.clip_param = config.get("clip_param", 0.2)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.value_coef = config.get("value_coef", 0.5)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.update_epochs = config.get("update_epochs", 4)
        self.num_minibatches = config.get("num_minibatches", 4)
        self.batch_size = config.get("batch_size", 64)

        self.actor = SharedBlindActor(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_size=config.get("hidden_size_actor", config.get("hidden_size", 128)),
        ).to(self.device)

        hidden_size_critic = config.get("hidden_size_critic", config.get("hidden_size", 256))
        self.critic = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_size_critic),
            nn.Tanh(),
            nn.Linear(hidden_size_critic, hidden_size_critic),
            nn.Tanh(),
            nn.Linear(hidden_size_critic, 1),
        ).to(self.device)

        lr = config.get("initial_lr", 3e-4)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr, eps=1e-5)

    def predict(self, obs, active_mask=None):
        if len(obs.shape) == 2:
            obs = np.expand_dims(obs, axis=0)

        n_envs, n_agents = obs.shape[:2]
        obs_flat = obs.reshape(-1, self.obs_dim)
        obs_tensor = to_tensor(obs_flat)

        with torch.no_grad():
            logits = self.actor(obs_tensor)
            action = torch.argmax(logits, dim=-1)

        return action.reshape(n_envs, n_agents).detach().cpu().numpy()

    def sample(self, obs, active_mask=None):
        if len(obs.shape) == 2:
            obs = np.expand_dims(obs, axis=0)

        n_envs, n_agents = obs.shape[:2]
        obs_flat = obs.reshape(-1, self.obs_dim)
        obs_tensor = to_tensor(obs_flat)

        with torch.no_grad():
            logits = self.actor(obs_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = self.critic(obs_tensor).squeeze(-1)

        return (
            value.reshape(n_envs, n_agents, 1).detach().cpu().numpy(),
            action.reshape(n_envs, n_agents).detach().cpu().numpy(),
            log_prob.reshape(n_envs, n_agents, 1).detach().cpu().numpy(),
        )

    def learn(self, rollout_storage):
        actor_loss_epoch = 0.0
        critic_loss_epoch = 0.0
        entropy_epoch = 0.0
        lr = None

        total_steps = rollout_storage.get_size()
        if total_steps < self.batch_size:
            return 0.0, 0.0, 0.0, lr

        effective_batch_size = min(self.batch_size, total_steps)
        num_minibatches = min(self.num_minibatches, max(1, total_steps // 4))
        minibatch_size = max(1, effective_batch_size // num_minibatches)

        indexes = np.arange(total_steps)

        for _ in range(self.update_epochs):
            np.random.shuffle(indexes)
            for start in range(0, total_steps, minibatch_size):
                end = min(start + minibatch_size, total_steps)
                sample_idx = indexes[start:end]
                if len(sample_idx) < 2:
                    continue

                (
                    batch_obs, batch_action, batch_logprob,
                    batch_adv, batch_return, _batch_value,
                    batch_active_masks, batch_decision_masks,
                    _batch_congestions,
                ) = rollout_storage.sample_batch(sample_idx)

                # Flatten: (batch, n_agents, ...) -> (batch*n_agents, ...)
                obs_flat = batch_obs.reshape(-1, self.obs_dim)
                actions_flat = batch_action.reshape(-1)
                old_logprob_flat = batch_logprob.reshape(-1)
                adv_flat = batch_adv.reshape(-1)
                returns_flat = batch_return.reshape(-1)
                active_flat = batch_active_masks.reshape(-1)
                decision_flat = batch_decision_masks.reshape(-1)

                obs_tensor = to_tensor(obs_flat)
                actions_tensor = to_tensor(actions_flat, dtype=torch.long)
                old_logprob_tensor = to_tensor(old_logprob_flat)
                adv_tensor = to_tensor(adv_flat)
                returns_tensor = to_tensor(returns_flat)
                active_tensor = to_tensor(active_flat)
                decision_tensor = to_tensor(decision_flat)

                valid = decision_tensor > 0.5
                if valid.sum() > 1:
                    adv_valid = adv_tensor[valid]
                    adv_norm = (adv_tensor - adv_valid.mean()) / (adv_valid.std() + 1e-8)
                else:
                    adv_norm = adv_tensor

                # ===== Actor update (decision_mask) =====
                logits = self.actor(obs_tensor)
                dist = Categorical(logits=logits)
                new_logprob = dist.log_prob(actions_tensor)
                entropy = dist.entropy()

                ratio = torch.exp(new_logprob - old_logprob_tensor)
                pg_loss1 = -adv_norm * ratio
                pg_loss2 = -adv_norm * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                pg_loss_raw = torch.max(pg_loss1, pg_loss2)

                decision_denom = torch.clamp(decision_tensor.sum(), min=1.0)
                pg_loss = (pg_loss_raw * decision_tensor).sum() / decision_denom
                ent_loss = (entropy * decision_tensor).sum() / decision_denom
                actor_loss = pg_loss - self.entropy_coef * ent_loss

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # ===== Critic update (active_mask) =====
                values = self.critic(obs_tensor).squeeze(-1)
                critic_mse = (values - returns_tensor) ** 2
                active_denom = torch.clamp(active_tensor.sum(), min=1.0)
                critic_loss = 0.5 * self.value_coef * (
                    critic_mse * active_tensor).sum() / active_denom

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