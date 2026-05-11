import torch
import torch.nn as nn


def split_obs(all_obs, n_servers, confounder_dim):
    n_active = all_obs[:, 0:1, 0]                          # (B, 1)
    local_feat = all_obs[:, :, 1 : 1 + 5 + n_servers]      # (B, N, 5+K)
    cong_feat = all_obs[:, 0, -confounder_dim:]            # (B, confounder_dim)
    global_feat = torch.cat([n_active, cong_feat], dim=-1) # (B, 1+confounder_dim)
    return local_feat, global_feat


class AttentionCritic(nn.Module):
   
    def __init__(
        self,
        obs_dim,
        n_servers=3,
        hidden_size=256,
        n_heads=4,
        global_embed_dim=64,
        confounder_dim=None,
        **kwargs,  
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_servers = n_servers
        self.hidden_size = hidden_size
        self.confounder_dim = confounder_dim if confounder_dim is not None else n_servers
        self.local_dim = 5 + n_servers                      # 5 local task + K channel
        self.global_dim = 1 + self.confounder_dim           # n_active + tx/cp

        self.local_embed = nn.Linear(self.local_dim, hidden_size)
        self.self_attn = nn.MultiheadAttention(hidden_size, n_heads)
        self.norm = nn.LayerNorm(hidden_size)

        self.global_encoder = nn.Sequential(
            nn.Linear(self.global_dim, global_embed_dim),
            nn.ReLU(),
            nn.Linear(global_embed_dim, global_embed_dim),
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_size + global_embed_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def _encode_agents(self, local_feat, active_mask):
        h = self.local_embed(local_feat)                    # (B, N, hidden)
        h = h.transpose(0, 1)                               # (N, B, hidden) for MHA
        key_padding_mask = active_mask < 0.5
        h_att, _ = self.self_attn(h, h, h, key_padding_mask=key_padding_mask)
        h = self.norm(h + h_att)
        return h.transpose(0, 1)                            # (B, N, hidden)

    def forward(self, all_obs, active_mask, server_congestions=None):
        B, N, _ = all_obs.shape
        local_feat, global_feat = split_obs(all_obs, self.n_servers, self.confounder_dim)

        h = self._encode_agents(local_feat, active_mask)    # (B, N, hidden)
        g = self.global_encoder(global_feat)                # (B, global_embed_dim)
        g = g.unsqueeze(1).expand(-1, N, -1)

        h = torch.cat([h, g], dim=-1)
        return self.value_head(h)
