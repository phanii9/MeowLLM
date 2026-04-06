"""Tests for eval harness, dataset, and tokenizer."""
import json
from pathlib import Path

import pytest
import torch

from meow.eval_cases import EVAL_PROMPTS, evaluate_output, evaluate_batch
from meow.tokenizer import MeowTokenizer, train_tokenizer
from meow.dataset import MeowDataset, IGNORE_INDEX


class TestEvalPrompts:
    def test_has_all_categories(self):
        categories = {cat for _, cat in EVAL_PROMPTS}
        expected = {
            "greeting", "hunger", "naps", "boxes", "windows", "birds",
            "humans", "dogs", "vacuum", "rain", "affection", "territory",
            "nonsense_questions", "being_picked_up", "jealousy",
            "hard_negative",
        }
        assert categories == expected

    def test_at_least_two_per_category(self):
        from collections import Counter
        counts = Counter(cat for _, cat in EVAL_PROMPTS)
        for cat, n in counts.items():
            assert n >= 2, f"{cat} has only {n} eval prompts"


class TestEvaluateOutput:
    def test_good_sample_passes(self):
        result = evaluate_output("hello. i was in the sun spot.", "greeting")
        assert result.passed

    def test_capital_fails(self):
        result = evaluate_output("Hello there.", "greeting")
        assert not result.passed
        assert not result.checks["lowercase"]

    def test_assistant_speak_fails(self):
        result = evaluate_output(
            "of course! i can help you with that.", "greeting"
        )
        assert not result.passed


class TestEvaluateBatch:
    def test_batch_stats(self):
        outputs = [
            "hello.",
            "Bad Capital.",
            "yes food.",
        ]
        cats = ["greeting", "greeting", "hunger"]
        stats = evaluate_batch(outputs, cats)
        assert stats["n"] == 3
        assert 0 <= stats["pass_rate"] <= 1.0
        assert "per_check_pass_rate" in stats


class TestTokenizer:
    @pytest.fixture(scope="class")
    def tokenizer(self, tmp_path_factory):
        # Use the real dataset if available, else a mini one
        real_path = Path("data/tokenizer.json")
        if real_path.exists():
            return MeowTokenizer.from_file(real_path)
        # Otherwise train a tiny one
        tmp = tmp_path_factory.mktemp("tok")
        data_path = tmp / "train.jsonl"
        with data_path.open("w") as f:
            for i in range(100):
                f.write(json.dumps({
                    "input": f"hi there {i}",
                    "output": f"hello from the sun spot {i}",
                    "category": "greeting",
                }) + "\n")
        out_path = tmp / "tok.json"
        train_tokenizer(data_path, out_path, vocab_size=256)
        return MeowTokenizer.from_file(out_path)

    def test_encode_returns_ints(self, tokenizer):
        ids = tokenizer.encode("hi miso")
        assert all(isinstance(i, int) for i in ids)

    def test_round_trip(self, tokenizer):
        text = "hi miso"
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        assert decoded == text

    def test_chat_format(self, tokenizer):
        ids, start = tokenizer.encode_chat("hi", "hello.")
        # Should start with BOS, USER, then input, then MISO at start-1
        assert ids[0] == tokenizer.bos_id
        assert ids[1] == tokenizer.user_id
        assert ids[start - 1] == tokenizer.miso_id
        assert ids[-1] == tokenizer.eos_id

    def test_chat_prompt_only(self, tokenizer):
        ids, start = tokenizer.encode_chat("hi", output_text=None)
        # Should end with MISO token, start == len(ids)
        assert ids[-1] == tokenizer.miso_id
        assert start == len(ids)


class TestDataset:
    @pytest.fixture(scope="class")
    def setup(self, tmp_path_factory):
        # Create a mini dataset and tokenizer
        tmp = tmp_path_factory.mktemp("ds")
        data_path = tmp / "train.jsonl"
        with data_path.open("w") as f:
            for i in range(20):
                f.write(json.dumps({
                    "input": "hi miso",
                    "output": "hello. i was in the sun spot.",
                    "category": "greeting",
                }) + "\n")
        tok_path = tmp / "tok.json"
        train_tokenizer(data_path, tok_path, vocab_size=256)
        tokenizer = MeowTokenizer.from_file(tok_path)
        ds = MeowDataset(data_path, tokenizer, max_seq_len=64)
        return ds, tokenizer

    def test_length(self, setup):
        ds, _ = setup
        assert len(ds) == 20

    def test_item_shapes(self, setup):
        ds, _ = setup
        item = ds[0]
        assert item["input_ids"].shape == item["target_ids"].shape
        assert item["input_ids"].shape[0] == 63  # max_seq_len - 1

    def test_ignore_index_on_prompt(self, setup):
        """User-turn positions should be masked with IGNORE_INDEX."""
        ds, _ = setup
        item = ds[0]
        # At least some positions should be ignored (the prompt part)
        n_ignored = (item["target_ids"] == IGNORE_INDEX).sum().item()
        assert n_ignored > 0

    def test_supervised_positions(self, setup):
        """Some positions should be supervised (the output part)."""
        ds, _ = setup
        item = ds[0]
        n_supervised = (item["target_ids"] != IGNORE_INDEX).sum().item()
        assert n_supervised > 0
