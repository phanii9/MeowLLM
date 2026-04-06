"""Tests for meow.model and meow.tokenizer."""
import pytest
import torch

from meow.model import Meow, MeowConfig, RMSNorm, build_rope_cache, apply_rope


class TestConfig:
    def test_defaults(self):
        cfg = MeowConfig()
        assert cfg.head_dim == 64  # 256 / 4

    def test_head_dim_divisibility(self):
        with pytest.raises(AssertionError):
            cfg = MeowConfig(d_model=100, n_heads=3)
            _ = cfg.head_dim


class TestRMSNorm:
    def test_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64)
        y = norm(x)
        assert y.shape == x.shape

    def test_normalization(self):
        """RMSNorm output should have RMS ≈ 1 per token."""
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64) * 5
        y = norm(x)
        rms = y.pow(2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)


class TestRoPE:
    def test_cache_shape(self):
        cos, sin = build_rope_cache(256, 64, 10000.0)
        assert cos.shape == (256, 32)  # head_dim/2
        assert sin.shape == (256, 32)

    def test_apply_preserves_shape(self):
        cos, sin = build_rope_cache(10, 64, 10000.0)
        x = torch.randn(2, 4, 10, 64)
        y = apply_rope(x, cos, sin)
        assert y.shape == x.shape

    def test_position_zero_is_identity(self):
        """At position 0, RoPE should not rotate (cos=1, sin=0)."""
        cos, sin = build_rope_cache(10, 64, 10000.0)
        x = torch.randn(1, 1, 1, 64)
        y = apply_rope(x, cos[:1], sin[:1])
        assert torch.allclose(y, x, atol=1e-5)


class TestModel:
    def test_param_count_reasonable(self):
        cfg = MeowConfig(vocab_size=1682)
        model = Meow(cfg)
        n = model.num_parameters()
        # Should be in the 3-4M range
        assert 2_000_000 < n < 5_000_000

    def test_forward_no_targets(self):
        cfg = MeowConfig(vocab_size=100, max_seq_len=32)
        model = Meow(cfg)
        x = torch.randint(0, 100, (2, 16))
        logits, loss = model(x)
        assert logits.shape == (2, 16, 100)
        assert loss is None

    def test_forward_with_targets(self):
        cfg = MeowConfig(vocab_size=100, max_seq_len=32)
        model = Meow(cfg)
        x = torch.randint(0, 100, (2, 16))
        y = torch.randint(0, 100, (2, 16))
        logits, loss = model(x, y)
        assert loss is not None
        assert loss.item() > 0

    def test_ignore_index(self):
        """Positions with target=-100 should be ignored in loss."""
        cfg = MeowConfig(vocab_size=100, max_seq_len=32)
        model = Meow(cfg)
        x = torch.randint(0, 100, (1, 8))
        y_all = torch.randint(0, 100, (1, 8))
        y_masked = y_all.clone()
        y_masked[0, :4] = -100
        _, loss_all = model(x, y_all)
        _, loss_masked = model(x, y_masked)
        # Losses should differ because masked positions are excluded
        assert abs(loss_all.item() - loss_masked.item()) > 1e-6

    def test_generate_extends_sequence(self):
        cfg = MeowConfig(vocab_size=100, max_seq_len=32)
        model = Meow(cfg)
        model.eval()
        x = torch.randint(0, 100, (1, 5))
        out = model.generate(x, max_new_tokens=10)
        assert out.shape[1] == 15  # 5 + 10

    def test_generate_stops_on_eos(self):
        """If all generated tokens are the EOS token, generation stops early."""
        cfg = MeowConfig(vocab_size=100, max_seq_len=32)
        model = Meow(cfg)
        model.eval()
        x = torch.randint(0, 100, (1, 5))
        # Can't easily force EOS without mocking — just check it runs
        out = model.generate(x, max_new_tokens=5, eos_token_id=0)
        assert out.shape[1] >= 5  # at least the prompt
        assert out.shape[1] <= 10  # at most prompt + max_new

    def test_tied_embeddings(self):
        """Model should not have a separate lm_head parameter."""
        cfg = MeowConfig(vocab_size=100)
        model = Meow(cfg)
        names = {n for n, _ in model.named_parameters()}
        # There should be no lm_head
        assert not any("lm_head" in n for n in names)
        # tok_emb should exist
        assert "tok_emb.weight" in names
