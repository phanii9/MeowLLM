"""Tests for meow.generate_data."""
import random

import pytest

from meow.generate_data import (
    CATEGORIES,
    compose_sample,
    generate_template_samples,
    extract_json_array,
)
from meow.rules import passes_filters, CATEGORY_KEYWORDS, CAT_VOCAB_EXEMPT_CATEGORIES


def test_all_15_categories():
    assert len(CATEGORIES) == 15
    names = {c.name for c in CATEGORIES}
    assert len(names) == 15  # all unique


def test_every_category_has_content():
    for spec in CATEGORIES:
        assert spec.inputs, f"{spec.name} has no inputs"
        assert spec.cores, f"{spec.name} has no cores"


def test_categories_match_rules():
    """Every category in CATEGORIES must either have CATEGORY_KEYWORDS
    or be in CAT_VOCAB_EXEMPT_CATEGORIES. This catches drift between the
    generator and the rules module."""
    for spec in CATEGORIES:
        in_keywords = spec.name in CATEGORY_KEYWORDS
        in_exempt = spec.name in CAT_VOCAB_EXEMPT_CATEGORIES
        assert in_keywords or in_exempt, (
            f"category {spec.name!r} has neither CATEGORY_KEYWORDS nor "
            f"is in CAT_VOCAB_EXEMPT_CATEGORIES — it would fail filtering"
        )


def test_no_orphan_keywords():
    """Every CATEGORY_KEYWORDS entry must correspond to a real category
    in CATEGORIES (catch typos that silently disable keyword checks)."""
    category_names = {spec.name for spec in CATEGORIES}
    for name in CATEGORY_KEYWORDS:
        assert name in category_names, (
            f"CATEGORY_KEYWORDS has entry for {name!r} which is not a "
            f"known category"
        )


def test_compose_sample_returns_dict():
    rng = random.Random(0)
    for spec in CATEGORIES:
        sample = compose_sample(spec, rng)
        assert "input" in sample
        assert "output" in sample
        assert "category" in sample
        assert sample["category"] == spec.name


def test_compose_output_lowercase():
    rng = random.Random(123)
    for spec in CATEGORIES:
        for _ in range(10):
            sample = compose_sample(spec, rng)
            assert sample["output"] == sample["output"].lower()


def test_generate_template_samples_yield():
    """Every generated sample must pass its own category filter."""
    rng = random.Random(42)
    samples, _ = generate_template_samples(500, rng, verbose=False)
    assert len(samples) == 500
    for s in samples:
        ok, reason = passes_filters(s["output"], s["category"])
        assert ok, f"sample failed: {s} reason={reason}"


def test_no_duplicates_in_output():
    rng = random.Random(0)
    samples, _ = generate_template_samples(500, rng, verbose=False)
    seen = set()
    for s in samples:
        key = (s["input"], s["output"])
        assert key not in seen
        seen.add(key)


def test_category_balance():
    """Every category should be represented."""
    rng = random.Random(1)
    samples, _ = generate_template_samples(1000, rng, verbose=False)
    from collections import Counter
    cats = Counter(s["category"] for s in samples)
    assert len(cats) == 15, f"only {len(cats)} categories represented"
    for name, count in cats.items():
        assert count >= 30, f"{name} only has {count} samples"


def test_eval_prompt_exclusion():
    """Samples should not contain any eval prompts as inputs."""
    from meow.eval_cases import EVAL_PROMPTS
    eval_inputs = {p.lower().strip() for p, _ in EVAL_PROMPTS}
    rng = random.Random(99)
    samples, _ = generate_template_samples(
        500, rng, eval_prompts=eval_inputs, verbose=False
    )
    for s in samples:
        assert s["input"] not in eval_inputs


def test_extract_json_array_clean():
    raw = '[{"input": "hi", "output": "hello."}]'
    result = extract_json_array(raw)
    assert isinstance(result, list)
    assert result[0]["input"] == "hi"


def test_extract_json_array_with_fence():
    raw = '```json\n[{"input": "hi", "output": "hello."}]\n```'
    result = extract_json_array(raw)
    assert result[0]["input"] == "hi"


def test_extract_json_array_with_prose():
    raw = 'Here are the pairs:\n[{"input": "hi", "output": "hello."}]\nLet me know if you need more.'
    result = extract_json_array(raw)
    assert result[0]["input"] == "hi"


def test_extract_json_array_invalid():
    with pytest.raises((ValueError, Exception)):
        extract_json_array("no array here")
