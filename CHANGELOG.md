# Changelog

All notable changes to MeowLLM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] — Initial release

First public release of MeowLLM / Miso.

### Added

#### Model

- ~3.5M parameter decoder-only transformer (`meow/model.py`, 278 lines)
- Modern architecture: RoPE, RMSNorm, SwiGLU, torch SDPA, tied embeddings
- Config: 4 layers, d_model 256, 4 heads, ffn_hidden 640, context 256
- Causal self-attention with flash-attention kernels via SDPA
- Autoregressive generation with temperature, top-k, and EOS handling

#### Data pipeline

- Slot-based compositional template generator (`meow/generate_data.py`)
- 15 categories fully populated with inputs, cores, openers, sensories,
  and redirects
- Per-category probability tuning for optional slots
- Optional LLM augmentation path via Anthropic API (disabled by default)
- Robust JSON extraction from LLM responses (handles fences, prose)
- Deduplication on `(input, output)` tuples
- Eval-prompt leakage prevention during generation
- Rejection stats reporting by reason

#### Rules module

- Single source of truth for character validation (`meow/rules.py`)
- Whole-phrase matching for banned assistant phrases (no substring false
  positives)
- Asymmetric per-category keyword requirements
- Exempt-category handling for nonsense questions
- Short-response bypass (≤6 words skip keyword checks)
- 40+ banned phrases across AI disclosure, helpfulness, chirpiness, and
  refusal-as-assistant categories

#### Tokenizer

- Byte-level BPE tokenizer trained on generated dataset
  (`meow/tokenizer.py`)
- Target vocab 2048, typical trained vocab ~1682
- Special tokens: `<pad>`, `<bos>`, `<eos>`, `<user>`, `<miso>`
- Chat format encoding with `output_start` index for loss masking

#### Training

- AdamW optimizer with cosine decay + linear warmup (`meow/train.py`)
- Gradient clipping at 1.0
- Smoke mode for fast pipeline validation (`--smoke`)
- Best and final checkpoint saving
- Training metadata JSON export

#### Dataset class

- PyTorch `MeowDataset` with proper loss masking (`meow/dataset.py`)
- User-turn positions marked with `IGNORE_INDEX = -100`
- Padding to fixed max sequence length
- Collate function for batching

#### Evaluation

- Held-out prompt suite: 38 prompts across 15 categories + hard negatives
  (`meow/eval_cases.py`)
- 5-dimension evaluation: lowercase, length, banned phrases, cat framing,
  full gate
- Batch evaluation with per-check pass rates
- Top-failure-reasons reporting

#### Inference

- Checkpoint loading with config reconstruction (`meow/inference.py`)
- Single-prompt and interactive modes
- Proper device handling across CPU/GPU

#### Tests

- 68 pytest tests covering:
  - Rules module (34 test cases)
  - Generator behavior and yield
  - Cross-consistency between `CATEGORIES` and `CATEGORY_KEYWORDS`
  - Model architecture (shapes, RoPE identity, RMSNorm, ignore_index,
    tied embeddings)
  - Tokenizer round-trip and chat format
  - Dataset loss masking
  - Evaluation harness
- Runs in ~7 seconds on CPU

#### Notebooks

- `notebooks/train_meow.ipynb` — one-click Colab training
- `notebooks/chat_with_meow.ipynb` — HF-download-first chat with
  training fallback

#### Documentation

- `README.md` — project overview, quick start, architecture, evaluation
- `persona.md` — character bible with hard voice rules
- `CONTRIBUTING.md` — contribution guide
- `docs/getting_started.md` — friendly tutorial walkthrough
- `docs/troubleshooting.md` — common issues and fixes
- `docs/faq.md` — frequently asked questions
- `docs/release.md` — step-by-step release process
- `docs/dataset_card.md` — Hugging Face dataset card (valid YAML front matter)
- `docs/model_card.md` — Hugging Face model card (valid YAML front matter)
- `CITATION.cff` — valid CFF 1.2.0
- Inline docstrings on every public function and class

#### Packaging

- `pyproject.toml` with setuptools backend
- Console scripts: `meow-generate`, `meow-tokenizer`, `meow-train`,
  `meow-chat`
- Optional dependency groups: `llm` (Anthropic), `hub` (HuggingFace),
  `dev` (pytest, ruff)
- Python 3.10+ support
- `LICENSE` — MIT

#### Infrastructure

- GitHub Actions CI (`.github/workflows/test.yml`) testing Python
  3.10, 3.11, 3.12
- CI runs pytest, rules smoke test, generator, tokenizer, and smoke
  training to verify end-to-end pipeline
- `scripts/test_rules_smoke.py` — portable rules smoke test (no
  hardcoded paths)
- `scripts/upload_to_hf.sh` — one-shot Hugging Face upload for model
  and dataset

### Baseline numbers

Pass rates on held-out eval (bundled CPU checkpoint, 2000 steps, val_loss 0.476):

- lowercase: 100.0%
- length: 100.0%
- no banned phrases: 100.0%
- cat framing: 81.6%
- **overall: 84.2%**

A full 10-epoch GPU training run is expected to produce higher numbers still.
Contributors who complete a GPU run are invited to open a PR with their numbers.

### Known limitations

- Only the smoke-trained checkpoint has been validated. Full-training
  numbers pending.
- Narrow vocabulary (~1700 BPE tokens).
- Single-turn only.
- English/lowercase only.
- HF upload script untested on real credentials.
- Colab notebook untested on a real Colab instance (cells are
  syntactically correct and commands have been validated locally).

---

## [Unreleased]

Things planned for future releases:

- Pretrained weights published to Hugging Face
- Verified full-training eval numbers in the model card
- ONNX export path for browser inference
- Optional multi-turn format (experiment)
- `docs/architecture.md` deep-dive on design decisions
