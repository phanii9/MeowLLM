---
license: mit
task_categories:
  - text-generation
language:
  - en
tags:
  - tiny
  - character
  - synthetic
  - cat
  - educational
pretty_name: MeowLLM Miso Dataset
size_categories:
  - 10K<n<100K
---

# MeowLLM / Miso Dataset

Training data for MeowLLM — a ~3.5M parameter character language model
that speaks as a house cat named Miso.

## Dataset Summary

20,000 single-turn (input, output) pairs across 15 topical categories
plus a hard-negative "assistant trap" deflection category. All outputs
are in a single consistent character voice (lowercase, short, cat-themed,
no assistant phrases).

The dataset is produced by a **slot-based compositional template
generator** with strict per-category filtering and de-duplication.
No human annotation — everything is deterministic from hand-written
slot banks plus a random seed.

## Supported Tasks

- **Text generation**: training a tiny character model that produces
  responses in a consistent voice.
- **Character fidelity evaluation**: the held-out prompt suite in
  `meow/eval_cases.py` is explicitly excluded from the training split.

## Languages

English (lowercase only).

## Dataset Structure

### Data Fields

Each sample is a JSON object with:

| field    | type | description                                      |
|----------|------|--------------------------------------------------|
| input    | str  | Human prompt (things a person might say to a cat)|
| output   | str  | Miso's in-character response                     |
| category | str  | One of 15 topical category names                 |
| source   | str  | "template" or "llm" (augmented)                  |

### Data Splits

| split | samples |
|-------|---------|
| train | 19,000  |
| val   |  1,000  |

### Categories

`greeting`, `hunger`, `naps`, `boxes`, `windows`, `birds`, `humans`,
`dogs`, `vacuum`, `rain`, `affection`, `territory`, `nonsense_questions`,
`being_picked_up`, `jealousy`

Category balance in the train split is between 5.4% and 7.7% per
category (fairly uniform via round-robin generation).

## Data Creation

### Source Data

All data is synthetically generated from hand-written slot banks in
[`meow/generate_data.py`](https://github.com/phanii9/MeowLLM/blob/main/meow/generate_data.py).
Each category has:

- `inputs`: a bank of prompt phrasings
- `cores`: the main response clause bank
- `openers`: optional leading phrases
- `sensories`: optional sensory detail extensions
- `redirects`: optional secondary clauses

During generation, each sample is assembled by picking one core and
optionally attaching other slots according to per-category
probabilities. The result is then validated against
[`meow/rules.py`](https://github.com/phanii9/MeowLLM/blob/main/meow/rules.py)
before being written.

### Filtering

Every generated sample must pass:

1. **Strict lowercase** — no capital letters anywhere
2. **Length**: 1–3 sentences, ≤35 words, ≥1 word
3. **No banned phrases** — whole-phrase matching against a list of
   ~40 assistant-speak patterns ("as an ai", "i can help you",
   "certainly", "of course", etc.)
4. **Per-category keyword requirement** — e.g., `hunger` outputs must
   mention food-related words; `vacuum` outputs must mention hiding or
   the vacuum itself
5. **Cat-framing fallback** — long outputs without a category-specific
   rule must contain general cat vocabulary
6. **Deduplication** — exact (input, output) duplicates are rejected
7. **Eval leakage check** — any input matching a held-out eval prompt
   is rejected

### Known Limitations

- The dataset is deliberately narrow. Miso knows about 15 topics and
  nothing else.
- Vocabulary is small (~1700 BPE tokens) and lowercase-only.
- Responses are short by design (1–3 sentences).
- Outputs are in English and reflect a specific cultural framing of
  "house cat" (American/European domestic cat).

## Personal and Sensitive Information

None. All samples are synthetic and describe a fictional cat's views
on food, naps, and boxes.

## Considerations for Using the Data

### Social Impact

This is a tiny educational dataset for a character model. It has no
realistic misuse surface — the model trained on it is too small for
open-ended generation and too in-character to be useful as a generic
assistant.

### Bias

Miso is a slightly smug indoor cat. That bias is intentional and
documented in [`persona.md`](https://github.com/phanii9/MeowLLM/blob/main/persona.md).

## Additional Information

### Dataset Curators

phanii9

### Licensing

MIT License.

### Citation

```bibtex
@software{meowllm2026,
  author = {phanii9},
  title  = {MeowLLM: a tiny character language model that talks like a house cat},
  year   = {2026},
  url    = {https://github.com/phanii9/MeowLLM}
}
```
