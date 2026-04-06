# Contributing to MeowLLM

Thanks for wanting to contribute. MeowLLM is small on purpose — every
file should stay readable — but there are a few places where more
content genuinely helps.

## Before you start

**Read [`persona.md`](persona.md) first.** Every contribution should
stay in Miso's voice. If you can't hear Miso saying it, don't add it.

## Good contributions

### 1. Expand slot banks in existing categories

This is the highest-value contribution. More fragments mean more
output diversity without changing the architecture.

- Open [`meow/generate_data.py`](meow/generate_data.py)
- Find the category you want to expand (they're clearly marked)
- Add new entries to `cores`, `openers`, `sensories`, or `redirects`
- Every new core must pass the category's keyword check — see
  [`meow/rules.py`](meow/rules.py) for the required words per category
- Run `pytest tests/` and verify 100% pass
- Run `python -m meow.generate_data --n 1000 --seed 0` and read the
  outputs. If they still sound like Miso, open a PR.

### 2. Add a new category

Harder but valuable if Miso's world model has a real gap. Examples of
plausible new categories: `grooming`, `other_cats`, `midnight_zoomies`,
`closed_doors`, `the_bath`.

Steps:

1. Add the category name to an appropriate section of `persona.md`
2. Add a `CategoryKeywords` entry to `meow/rules.py` with `required_any`
3. Add a `CategorySpec` block to `meow/generate_data.py`
4. Add 2+ held-out eval prompts in `meow/eval_cases.py`
5. Run `pytest tests/` — the "15 categories" tests will now need to
   be updated to reflect the new count
6. Regenerate the dataset and verify yield stays above 90%

### 3. Fix bugs in the rules or filters

If you find a valid Miso line that the filters reject (false negative)
or an invalid line that slips through (false positive):

1. Add a test case to `tests/test_rules.py` that captures the bug
2. Fix the rule in `meow/rules.py`
3. Make sure all existing tests still pass

### 4. Improve documentation

The README, `persona.md`, and `docs/` are all welcome targets for
clarity improvements. Keep the tone honest — no marketing speak.

## Style rules

### Code

- Python 3.10+ syntax (use `|` for union types, `list[int]` not `List[int]`)
- Type hints where they help readability
- Docstrings for every public function
- No line over ~100 characters
- `pytest` for tests
- `ruff` for linting (`ruff check .`)

### Data

- **Lowercase only.** No capitals. Ever.
- **1–3 sentences per output.** Usually 1 or 2.
- **Simple vocabulary.** If a word sounds like an assistant or a
  marketing copywriter wrote it, remove it.
- **No emojis, no asterisks, no stage directions.**
- **No new banned phrases** unless they're clearly assistant-speak.
  We use whole-phrase matching, not substring matching — test your
  addition with things a valid cat line might say.

### Commit messages

- Short subject (<72 chars)
- Imperative mood ("add", "fix", "update", not "added")
- One sentence of context in the body if needed

## Running the test suite

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

All 66 tests should pass in under 10 seconds. If they don't, your
change has broken something.

## Reading before writing

- [`persona.md`](persona.md) — the character bible
- [`meow/rules.py`](meow/rules.py) — the validation rules (read
  before adding any category or banned phrase)
- [`docs/dataset_card.md`](docs/dataset_card.md) — what's in the
  training data and how it was made
- [`docs/model_card.md`](docs/model_card.md) — what the model can and
  can't do

## Things NOT to contribute

- System prompts or runtime personality overrides. The whole point
  is that Miso's voice is baked into the weights, not injected.
- Generic chatbot features (tool use, RAG, function calling). This
  is a character model, not an assistant platform.
- Scale increases. If you want a bigger model, fork and call it
  something else. MeowLLM is deliberately tiny.
- New architectures that add complexity without clear wins. RoPE +
  RMSNorm + SwiGLU + SDPA is the sweet spot for this scale.

## Questions

Open an issue on GitHub. Keep them focused.
