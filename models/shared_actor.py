import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class SharedBlindActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, act_dim),
        )

    def forward(self, obs):
        return self.net(obs)

    def get_action(self, obs, action=None):
        logits = self.forward(obs)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy()

    def get_action_and_value(self, obs, action=None):
        logits = self.forward(obs)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy(), logits


class ConditionalActor(nn.Module):

    def __init__(self, obs_dim, act_dim, n_servers=3, embed_dim=16, hidden_size=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_servers = n_servers

        self.confounder_encoder = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(n_servers, embed_dim)),
            nn.LayerNorm(embed_dim),
            nn.Tanh(),
        )

        self.net = nn.Sequential(
            nn.Linear(obs_dim + embed_dim + n_servers, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, act_dim),
        )

    def forward(self, obs, server_loads_log):
        e = self.confounder_encoder(server_loads_log)  # [batch, embed_dim]
        x = torch.cat(
            [obs, e, server_loads_log], dim=-1
        )  # [batch, obs_dim + embed_dim + n_servers]
        return self.net(x)

    def get_action(self, obs, server_loads_log, action=None):
        logits = self.forward(obs, server_loads_log)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()
