"""
meow.model — tiny modern decoder-only transformer for Miso.

~3M parameters. Designed to be read top-to-bottom.

Architecture:
  - Rotary positional embeddings (RoPE)
  - RMSNorm (no bias, no mean subtraction)
  - SwiGLU feed-forward
  - torch SDPA (uses flash attention when available)
  - Tied input/output embeddings
  - Pre-norm residual blocks

Config:
  vocab=~1700, d_model=256, n_layers=4, n_heads=4,
  ffn=640, ctx=256
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MeowConfig:
    vocab_size: int = 2048
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    ffn_hidden: int = 640
    max_seq_len: int = 256
    dropout: float = 0.0
    rope_base: float = 10000.0
    pad_token_id: int = 0

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0, \
            f"d_model {self.d_model} must be divisible by n_heads {self.n_heads}"
        return self.d_model // self.n_heads


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


# ---------------------------------------------------------------------------
# Rotary positional embeddings
# ---------------------------------------------------------------------------

def build_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for RoPE. Shape: (seq_len, head_dim/2)."""
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq_len, device=device).float()
    angles = torch.outer(t, freqs)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary embeddings to x of shape (B, H, T, head_dim)."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    rotated = torch.stack(
        [x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1
    )
    return rotated.flatten(-2)


# ---------------------------------------------------------------------------
# Attention (SDPA)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: MeowConfig):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.cfg.n_heads, self.cfg.head_dim

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        q = apply_rope(q, cos[:T], sin[:T])
        k = apply_rope(k, cos[:T], sin[:T])

        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(y)


# ---------------------------------------------------------------------------
# SwiGLU feed-forward
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    def __init__(self, cfg: MeowConfig):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.w_down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# ---------------------------------------------------------------------------
# Transformer block (pre-norm)
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, cfg: MeowConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class Meow(nn.Module):
    def __init__(self, cfg: MeowConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)
        # Tied embeddings: no separate lm_head weight — we matmul against
        # tok_emb.weight.T in forward.

        # RoPE cache as buffers so PyTorch moves them with .to(device).
        cos, sin = build_rope_cache(
            cfg.max_seq_len, cfg.head_dim, cfg.rope_base,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        # Scale init std by depth: 0.02 for embeddings, attenuated for
        # deeper layers.  This is a simplified version of the GPT-NeoX /
        # Llama approach where residual projections get 1/sqrt(2*n_layers).
        std = 0.02
        if isinstance(m, nn.Linear):
            if hasattr(m, "_is_residual"):
                std = std / (2 * self.cfg.n_layers) ** 0.5
            nn.init.trunc_normal_(m.weight, mean=0.0, std=std)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, mean=0.0, std=std)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        input_ids: (B, T) long
        target_ids: (B, T) long or None. -100 positions are ignored in loss.
        returns: (logits, loss)
        """
        B, T = input_ids.shape
        assert T <= self.cfg.max_seq_len, \
            f"seq length {T} exceeds max {self.cfg.max_seq_len}"

        x = self.tok_emb(input_ids)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        x = self.norm_f(x)

        # Tied LM head: logits = x @ tok_emb.weight.T
        logits = x @ self.tok_emb.weight.T  # (B, T, vocab)

        loss = None
        if target_ids is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int | None = 40,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive sampling. Stops early on eos_token_id if provided."""
        self.eval()
        for _ in range(max_new_tokens):
            ctx = input_ids[:, -self.cfg.max_seq_len:]
            logits, _ = self(ctx)
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(next_logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_tok], dim=1)

            if eos_token_id is not None and (next_tok == eos_token_id).all():
                break
        return input_ids


if __name__ == "__main__":
    cfg = MeowConfig()
    model = Meow(cfg)
    n = model.num_parameters()
    print(f"model params: {n:,} ({n/1e6:.2f}M)")

    x = torch.randint(0, cfg.vocab_size, (2, 32))
    y = torch.randint(0, cfg.vocab_size, (2, 32))
    # Set some target positions to -100 to test masking
    y[0, :10] = -100
    logits, loss = model(x, y)
    print(f"logits: {tuple(logits.shape)}")
    print(f"loss:   {loss.item():.4f}")

    # Test generation
    start = torch.tensor([[1, 3, 691, 680, 4]])  # BOS USER "hi miso" MISO
    out = model.generate(start, max_new_tokens=10, eos_token_id=2)
    print(f"gen output shape: {tuple(out.shape)}")
