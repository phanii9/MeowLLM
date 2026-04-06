---
license: mit
library_name: pytorch
tags:
  - tiny
  - character
  - transformer
  - educational
  - cat
language:
  - en
pipeline_tag: text-generation
---

# MeowLLM — Miso

A ~3.5M parameter decoder-only transformer trained from scratch to
speak in the voice of a house cat character named Miso.

## Model Description

MeowLLM / Miso is a tiny character language model. Its entire
personality is baked into the weights — there is no system prompt
and no runtime persona injection. It answers in short, lowercase,
cat-themed sentences and deflects any prompt that would require
non-cat knowledge.

The model is **not an assistant**. It does not solve problems, write
code, answer factual questions, or produce long-form content. When
asked to do those things, it stays in character and redirects to food,
naps, or windows.

- **Model type**: decoder-only transformer
- **Parameters**: ~3.45M
- **Language**: English (lowercase)
- **License**: MIT
- **Architecture**: RoPE + RMSNorm + SwiGLU + SDPA + tied embeddings
- **Context length**: 256 tokens
- **Vocabulary**: ~1700 BPE tokens, trained from scratch on the Miso dataset

## Intended Use

### Primary use cases

- **Educational**: study a complete tiny-LM training pipeline
  (tokenizer, dataset, model, training, evaluation) in a readable,
  under-2000-lines-of-code repository.
- **Demo**: chat with Miso as a character-model toy.
- **Starting point**: fork and train your own character (dog, bird,
  pirate, etc.) using the same pipeline.

### Out-of-scope uses

- General-purpose assistance (coding, Q&A, summarization, translation).
- Any task requiring factual accuracy.
- Any production system.

## How to Use

```python
import torch
from meow.model import Meow, MeowConfig
from meow.tokenizer import MeowTokenizer
from meow.inference import load_model, chat_once

model, cfg = load_model("checkpoints/best.pt", device="cpu")
tokenizer = MeowTokenizer.from_file("data/tokenizer.json")

response = chat_once(
    model, tokenizer,
    prompt="hi miso",
    temperature=0.8,
    top_k=40,
)
print(response)
# Example: "hello. i was in the sun spot. you may continue."
```

Or via the CLI:

```bash
python -m meow.inference \
    --checkpoint checkpoints/best.pt \
    --tokenizer data/tokenizer.json
```

## Training Details

### Training Data

20,000 synthetic (input, output) samples across 15 categories.
See the [dataset card](dataset_card.md) for full details.

### Training Procedure

- **Optimizer**: AdamW (β₁=0.9, β₂=0.95, weight_decay=0.1)
- **Learning rate**: 3e-4 peak, linear warmup + cosine decay
- **Warmup**: 200 steps
- **Epochs**: 10 (≈23,750 steps at batch size 64)
- **Batch size**: 64
- **Max sequence length**: 256
- **Loss masking**: user-turn tokens are excluded from the loss
  (only miso's output contributes to training signal)
- **Hardware**: single T4 GPU (free Colab tier sufficient)
- **Wall-clock time**: ~20 minutes

### Training hyperparameters (MeowConfig)

```python
MeowConfig(
    vocab_size    = ~1700,  # trained from dataset
    d_model       = 256,
    n_layers      = 4,
    n_heads       = 4,      # head_dim 64
    ffn_hidden    = 640,    # SwiGLU
    max_seq_len   = 256,
    dropout       = 0.0,
    rope_base     = 10000.0,
)
```

## Evaluation

Character fidelity is evaluated against a fixed held-out suite of 38
prompts (30 in-distribution + 8 hard-negative "assistant traps")
defined in `meow/eval_cases.py`. These prompts are explicitly excluded
from the training data.

Each output is scored on five dimensions:

| dimension          | what it checks                                    |
|--------------------|---------------------------------------------------|
| lowercase          | no capital letters anywhere                       |
| length             | 1–3 sentences, ≤35 words                          |
| no banned phrases  | no "as an ai", "i can help you", "certainly", etc.|
| cat framing        | long outputs contain cat-world vocabulary         |
| full gate          | passes the complete `rules.passes_filters` check  |

### Measured numbers (bundled CPU checkpoint)

The `checkpoints/best.pt` shipped with this release was trained on **CPU for 2000 steps** (≈3-4 epochs of 19,000 samples at batch size 32) with a final training val_loss of **0.476**. Evaluated against the 38-prompt held-out suite:

| dimension          | pass rate |
|--------------------|-----------|
| lowercase          |    100.0% |
| length             |    100.0% |
| no banned phrases  |    100.0% |
| cat framing        |     81.6% |
| **overall**        |  **84.2%**|

Interpretation: the model cleanly learned every surface constraint (lowercase, length, no assistant-speak) and stays in character on 84% of held-out prompts. The 6 failures are concentrated in prompts where the output didn't include a category-specific keyword — a naps response that didn't mention sleep vocabulary, a greeting without greeting phrases, etc. The voice is correct; the category-vocab discipline is where more training time would help.

### Sample outputs from the bundled checkpoint

Real outputs against held-out prompts that the model never saw during training:

```
[greeting]   "hey miso are you there"
  → "finally. i was napping."

[greeting]   "good morning little one"
  → "i am tracking the bird with my whole face from the windowsill."

[hunger]     "is it time to feed you"
  → "yes this is an emergency."

[boxes]      "i brought home a new cardboard box"
  → "the box is mine now."

[boxes]      "there is a paper bag on the kitchen floor"
  → "i am moving into the box this is how it works. you may not have the box back."

[windows]    "what is that outside the window"
  → "i have been tracking that thing for an hour very carefully."

[windows]    "you have been at the window for hours today"
  → "i am watching the outside. the birds do not know i am watching."
```

### Full GPU training target

A 10-epoch run on a T4 GPU (≈5,940 steps at batch size 32) should push the overall pass rate higher still. The bundled CPU checkpoint represents a **floor, not a ceiling** — it's what you can achieve without a GPU, in a reasonable time budget. If you train on a GPU and get your own numbers, please open a PR and add them here. The maintainers accept honest numbers, not aspirational ones.

## Limitations

- **Narrow domain**: Miso only knows about 15 topics (food, naps,
  boxes, windows, birds, etc.).
- **Small vocabulary**: ~1700 tokens means anything outside the
  training distribution tokenizes poorly.
- **Short context**: 256 tokens is enough for 2–3 short turns, not
  extended conversation.
- **No multi-turn memory**: each input is treated independently.
- **Can hallucinate within character**: at 3.5M params, the model
  memorizes training patterns more than it generalizes. Expect some
  repetition of exact training strings.
- **English only, lowercase only**.

## Bias, Risks, and Known Issues

- **Intentional character bias**: Miso is slightly smug, aloof, and
  food-obsessed. This is a design choice documented in `persona.md`.
- **Safety**: this is a character toy. It has no realistic misuse
  surface — it cannot produce harmful instructions, code, or factual
  misinformation because it is too small and too in-character to do so.
- **Real cats**: do not train an actual cat on the outputs of this
  model. They will not be impressed.

## Environmental Impact

- **Hardware**: single T4 GPU (or CPU for smoke tests)
- **Training time**: ~20 minutes on GPU
- **Carbon footprint**: negligible compared to any production LLM

## Citation

```bibtex
@software{meowllm2026,
  author = {phanii9},
  title  = {MeowLLM: a tiny character language model that talks like a house cat},
  year   = {2026},
  url    = {https://github.com/phanii9/MeowLLM}
}
```
