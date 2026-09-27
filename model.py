"""Ensemble of small decoder-only transformers with stacked weights, so all models train and sample in single
batched GPU calls. Each model predicts a Gaussian (mean, log σ) for the next value at every position."""
import torch
import torch.nn as nn
import torch.nn.functional as F


# Every weight has a leading ensemble dimension E; activations are shaped (E, B, T, d).
class ELinear(nn.Module):
    def __init__(self, E, i, o):
        super().__init__()
        self.w = nn.Parameter(torch.empty(E, i, o).uniform_(-i ** -0.5, i ** -0.5))  # nn.Linear-style init
        self.b = nn.Parameter(torch.zeros(E, 1, o))

    def forward(self, x):
        s = x.shape
        return torch.baddbmm(self.b, x.reshape(s[0], -1, s[-1]), self.w).reshape(*s[:-1], -1)


class ELayerNorm(nn.Module):
    def __init__(self, E, d):
        super().__init__()
        self.g = nn.Parameter(torch.ones(E, 1, 1, d))
        self.b = nn.Parameter(torch.zeros(E, 1, 1, d))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:]) * self.g + self.b


class Block(nn.Module):
    """Pre-norm transformer block with causal self-attention."""
    def __init__(self, E, d, heads, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.drop = nn.Dropout(dropout)          # on each residual branch; inactive in eval mode
        self.ln1, self.ln2 = ELayerNorm(E, d), ELayerNorm(E, d)
        self.qkv, self.proj = ELinear(E, d, 3 * d), ELinear(E, d, d)
        self.fc1, self.fc2 = ELinear(E, d, 4 * d), ELinear(E, 4 * d, d)

    def forward(self, x):
        E, B, T, d = x.shape
        q, k, v = (t.reshape(E * B, T, self.heads, d // self.heads).transpose(1, 2)   # models folded into the batch
                   for t in self.qkv(self.ln1(x)).split(d, -1))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(E, B, T, d)))
        return x + self.drop(self.fc2(F.relu(self.fc1(self.ln2(x)))))


class Ensemble(nn.Module):
    def __init__(self, n_models, ctx, d_model=64, heads=4, layers=2, window_norm=True, scale_prefix=0, n_in=1, n_out=2,
                 dropout=0.0):
        """scale_prefix=k > 0: causal scale normalization — offset and scale come from the window's first k values
        (mean, std + 0.1), so predictions are only leak-free from position k-1 on; train on `first_valid` onward.
        n_in > 1: each step is a vector of n_in (already normalized) inputs, x: (E, B, T, n_in), and forward returns
        the raw (E, B, T, n_out) outputs — no per-window normalization."""
        super().__init__()
        self.n_models, self.ctx, self.window_norm, self.scale_prefix = n_models, ctx, window_norm, scale_prefix
        self.n_in = n_in
        self.first_valid = max(scale_prefix - 1, 0)
        self.inp = ELinear(n_models, n_in, d_model)
        self.pos = nn.Parameter(torch.randn(n_models, 1, ctx, d_model) * 0.02)
        self.blocks = nn.Sequential(*[Block(n_models, d_model, heads, dropout) for _ in range(layers)])
        self.out = ELinear(n_models, d_model, n_out)

    def forward(self, x):                            # x: (E, B, T), T <= ctx  (or (E, B, T, n_in) when n_in > 1)
        if self.n_in > 1:
            T = x.shape[-2]
            return self.out(self.blocks(self.inp(x) + self.pos[:, :, :T]))
        # Per-window normalization: values are taken relative to the window's first value, so the model only
        # sees the shape, not the absolute level. The first value is in the past for every position (causal).
        # With scale_prefix, the level and size both come from the first k values instead.
        off, scale = (x[..., :1] if self.window_norm else 0.0), 1.0
        if self.scale_prefix:
            prefix = x[..., :self.scale_prefix]
            off, scale = prefix.mean(-1, keepdim=True), prefix.std(-1, keepdim=True) + 0.1
        T = x.shape[-1]
        out = self.out(self.blocks(self.inp(((x - off) / scale)[..., None]) + self.pos[:, :, :T]))
        log_scale = torch.log(scale) if self.scale_prefix else 0.0
        return out[..., 0] * scale + off, out[..., 1].clamp(-7, 3) + log_scale   # next-value mean, log σ


def nll(mean, log_sigma, target):
    """Gaussian negative log-likelihood (constant dropped)."""
    return (log_sigma + 0.5 * ((target - mean) / log_sigma.exp()) ** 2).mean()


@torch.no_grad()
def sample_paths(model, histories, steps, n_paths):
    """Forecast several series at once, LLM-style: each step samples next ~ N(mean, σ) and feeds it back.
    histories: (S, ctx) normalized last-context values, one row per series.
    Returns (S, n_paths, steps); each series' paths are split evenly across the models, ordered model by model."""
    E, (S, ctx) = model.n_models, histories.shape
    P = n_paths // E
    window = histories[None, :, None, :].expand(E, S, P, ctx).reshape(E, S * P, ctx)
    out = []
    for _ in range(steps):
        mean, log_sigma = model(window)
        nxt = mean[..., -1] + log_sigma[..., -1].exp() * torch.randn_like(mean[..., -1])
        out.append(nxt)
        window = torch.cat([window[..., 1:], nxt[..., None]], -1)
    return torch.stack(out, -1).reshape(E, S, P, steps).transpose(0, 1).reshape(S, E * P, steps)
