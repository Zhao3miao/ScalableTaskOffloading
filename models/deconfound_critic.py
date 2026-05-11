import torch
import numpy as np

from models.attention_critic import AttentionCritic, split_obs


class DeconfoundedCritic(AttentionCritic):

    def __init__(
        self,
        obs_dim,
        n_servers=3,
        hidden_size=256,
        n_heads=4,
        global_embed_dim=64,
        memory_size=5000,
        vdo_sample_size=64,
        confounder_dim=None,
        **kwargs,
    ):
        super().__init__(
            obs_dim=obs_dim,
            n_servers=n_servers,
            hidden_size=hidden_size,
            n_heads=n_heads,
            global_embed_dim=global_embed_dim,
            confounder_dim=confounder_dim,
        )
        self.memory_size = memory_size
        self.vdo_sample_size = vdo_sample_size

        # DEBA: Experience circular buffer (stores raw confounder vectors L)
        self.register_buffer("confounder_memory", torch.zeros(memory_size, self.confounder_dim))
        self.register_buffer("memory_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("memory_count", torch.zeros(1, dtype=torch.long))

        # Fallback: Random uniform sampling for cold start to avoid high-dimensional grid explosion (4^6=4096)
        fallback = np.random.RandomState(42).uniform(
            0.0, 2.0, size=(vdo_sample_size, self.confounder_dim)
        ).astype(np.float32)
        self.register_buffer("vdo_grid_fallback", torch.from_numpy(fallback))

    @torch.no_grad()
    def update_prior(self, server_congestions):
        if server_congestions.dim() == 1:
            server_congestions = server_congestions.unsqueeze(0)
        N = server_congestions.shape[0]
        ptr = self.memory_ptr.item()

        if N >= self.memory_size:
            self.confounder_memory.copy_(server_congestions[-self.memory_size:])
            self.memory_ptr.fill_(0)
            self.memory_count.fill_(self.memory_size)
        else:
            space = self.memory_size - ptr
            if N <= space:
                self.confounder_memory[ptr:ptr + N] = server_congestions
                self.memory_ptr.fill_((ptr + N) % self.memory_size)
            else:
                self.confounder_memory[ptr:] = server_congestions[:space]
                remainder = N - space
                self.confounder_memory[:remainder] = server_congestions[space:]
                self.memory_ptr.fill_(remainder)
            self.memory_count.fill_(
                min(self.memory_count.item() + N, self.memory_size)
            )

    def _get_vdo_samples_independent(self):
        """Independent marginal sampling per dimension to break spurious correlations in joint distributions.

        In the multi-modal bottleneck setting (confounder_dim=2*n_servers), simultaneously breaks:
          1. Geographical spurious correlations (tx or cp of different BSs are high simultaneously)
          2. Modal spurious correlations (tx and cp accumulate simultaneously)
        """
        count = self.memory_count.item()
        if count < self.vdo_sample_size:
            return self.vdo_grid_fallback

        valid_memory = self.confounder_memory[:count]
        device = self.confounder_memory.device
        M = self.vdo_sample_size
        samples = torch.zeros(M, self.confounder_dim, device=device)
        for d in range(self.confounder_dim):
            idxs = torch.randint(0, count, (M,), device=device)
            samples[:, d] = valid_memory[idxs, d]
        return samples


    def compute_deconfounded_value(self, all_obs, active_mask):
        """V_do(s) = (1/M) Σ_m V(s, L_m)

        1. Attention is computed once on local_feat to get h ∈ (B, N, hidden)
        2. The congestion part of global_feat is replaced by M independent marginal samples (n_active remains unchanged)
        3. Recompute global encoding + value_head for each L_m; block processing to reduce memory peak

        Returns: (B, N, 1)
        """
        B, N, _ = all_obs.shape
        local_feat, global_feat = split_obs(all_obs, self.n_servers, self.confounder_dim)
        h = self._encode_agents(local_feat, active_mask)              # (B, N, hidden)
        hidden = h.shape[-1]

        vdo_samples = self._get_vdo_samples_independent()             # (M, confounder_dim)
        M = vdo_samples.shape[0]

        # Construct M copies of global_feat: keep n_active (0th dimension), replace congestion dimension
        n_active = global_feat[:, :1]                                  # (B, 1)
        # cf_global[m, b, :] = [n_active_b, L_m]
        n_active_rep = n_active.unsqueeze(0).expand(M, B, 1)           # (M, B, 1)
        L_rep = vdo_samples.unsqueeze(1).expand(M, B, self.confounder_dim)  # (M, B, conf)
        cf_global = torch.cat([n_active_rep, L_rep], dim=-1)           # (M, B, 1+conf)

        g_all = self.global_encoder(cf_global)                         # (M, B, g_emb)
        g_emb_dim = g_all.shape[-1]

        # Block processing, up to chunk_m L samples per block to avoid allocating M*B*N simultaneously
        BN = B * N
        chunk_m = max(1, 4096 // max(BN, 1))
        h_flat = h.reshape(BN, hidden)                                  # (BN, hidden)
        v_sum = torch.zeros(BN, 1, device=h.device)

        for i in range(0, M, chunk_m):
            j = min(i + chunk_m, M)
            cm = j - i
            # h_rep: (cm, BN, hidden) — duplicate local encoding
            h_rep = h_flat.unsqueeze(0).expand(cm, BN, hidden).reshape(cm * BN, hidden)
            # g_rep: (cm, B, g_emb) -> (cm, B, N, g_emb) -> (cm*BN, g_emb)
            g_chunk = g_all[i:j]                                        # (cm, B, g_emb)
            g_rep = g_chunk.unsqueeze(2).expand(cm, B, N, g_emb_dim).reshape(cm * BN, g_emb_dim)
            x = torch.cat([h_rep, g_rep], dim=-1)
            v = self.value_head(x).reshape(cm, BN, 1)
            v_sum += v.sum(dim=0)

        return (v_sum / M).reshape(B, N, 1)

    def get_memory_stats(self):
        count = self.memory_count.item()
        if count == 0:
            return {"count": 0, "mean": None, "std": None, "using_fallback": True}
        valid = self.confounder_memory[:count]
        return {
            "count": count,
            "mean": valid.mean(dim=0).cpu().numpy().tolist(),
            "std": valid.std(dim=0).cpu().numpy().tolist(),
            "using_fallback": count < self.vdo_sample_size,
        }