# Troubleshooting

Common issues when training, running, or contributing to MeowLLM, and
how to fix them.

---

## Installation

### `pip install -e .` fails with a torch error

On some systems, pip picks the wrong torch wheel. Install torch
explicitly first:

```bash
# CPU-only (works everywhere, smaller download)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# CUDA 12.1 (for NVIDIA GPUs)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Then the rest
pip install -e .
```

### `ImportError: No module named 'tokenizers'`

The `tokenizers` package comes from Hugging Face and is a separate
install from `transformers`. If `pip install -e .` didn't pull it in,
install it directly:

```bash
pip install tokenizers>=0.15.0
```

### `ImportError: No module named 'meow'` when running scripts

You forgot `pip install -e .`, or you're running from outside the
repo root. The editable install makes the `meow` package importable
from any directory. If you can't install for some reason, run scripts
from the repo root with `PYTHONPATH`:

```bash
PYTHONPATH=. python -m meow.generate_data --out-dir data --n 20000
```

---

## Data generation

### Yield dropped below 50%

If `meow.generate_data` reports a yield much lower than the expected
~60%, check the rejection reasons in the output. Common causes:

- **You added a new banned phrase** that's matching valid outputs.
  `meow/rules.py` uses *whole-phrase* matching, but if you added a
  very short phrase like `"i am"` as banned, it will reject most lines.
  Make banned phrases specific (`"i am an ai"`, not `"i am"`).

- **You added a new category core that doesn't contain required vocab.**
  Every long core (>6 words) must contain at least one word from its
  category's `required_any` set in `rules.py`. Either rewrite the core
  or add the missing word to `required_any`.

- **You increased `--n` beyond what the slot banks can produce.** At
  20,000 samples, duplicate rejections are already ~11,700. At 50,000
  you'd hit a hard ceiling. Fix: add more fragments to existing slot
  banks in `meow/generate_data.py`.

### "missing_category_vocab:X" rejections

A generated output didn't include any of the required keywords for
its category. Look at the specific rejection and either:

1. Add the missing word to the category's `required_any` in
   `meow/rules.py`, or
2. Rewrite the core in `meow/generate_data.py` to include an
   already-required word.

### Generator is slow

It shouldn't be — 20,000 samples take ~30 seconds. If it's much
slower, you probably have a huge `n` or slow I/O. Profile with:

```bash
python -c "
import cProfile, random
from meow.generate_data import generate_template_samples
cProfile.run('generate_template_samples(20000, random.Random(0), verbose=False)', sort='cumulative')
" 2>&1 | head -30
```

---

## Tokenizer

### Vocab is less than 2048

Expected. The BPE trainer stops early when it runs out of merges that
would save at least 2 characters. With a narrow lowercase-only dataset,
vocab usually trains to 1600–1800. This is fine — smaller vocab means
smaller embedding matrix means fewer parameters.

### Round-trip fails on non-ASCII characters

MeowLLM's tokenizer is byte-level BPE, so it *does* handle non-ASCII
bytes correctly. But the training data is all lowercase ASCII, so if
you feed the model non-ASCII input it won't know how to respond. The
tokenizer will encode it fine; the model just hasn't seen it.

---

## Training

### Loss starts around 7.5 and doesn't move

Check:
- **Is the dataset actually loaded?** Look for `[train] train=19000 val=1000`
- **Is the tokenizer correct?** Vocab should be ~1700
- **Is the learning rate reasonable?** Default is 3e-4. If you changed it
  to something tiny (1e-7) it won't learn.
- **Is the data masked correctly?** Check `meow/dataset.py` —
  `IGNORE_INDEX = -100` positions are excluded from loss. If you
  changed the masking logic and set too many positions to ignored,
  there's nothing to learn from.

### CUDA out of memory

The model is only 3.5M parameters, so it should fit on any GPU.
OOM usually means batch size is too large or you're accidentally
loading the entire dataset onto GPU. Default batch_size=64 needs
<1 GB. Try:

```bash
python -m meow.train --batch-size 32 --epochs 10
```

### `RuntimeError: Expected all tensors to be on the same device`

Something wasn't moved to the correct device. The most common cause:
loading a checkpoint on CPU then calling `.to('cuda')` but not re-moving
the RoPE buffers. Use the provided `load_model()` in `meow.inference`
which handles this correctly. If you're writing your own loading code,
make sure to call `model.to(device)` *after* `load_state_dict`.

### Training is much slower than expected on GPU

Check that the model is actually on the GPU:

```python
for name, p in model.named_parameters():
    print(name, p.device)
    break
```

If it says `cpu`, you forgot `model.to('cuda')`. If it says `cuda:0`
and it's still slow, check that your batch isn't being moved one
sample at a time — ensure `.to(device)` is called on the whole batch
tensor, not inside a loop.

### Training works but val loss doesn't improve

Usually means overfitting (val goes up while train goes down). At
~3.5M params on 19K samples, overfitting is real. Options:
- Reduce epochs to 5
- Add dropout: `--dropout 0.1`
- Generate more data

### My model outputs gibberish after training

If training finished (loss went down) but outputs still look random,
check:
- Are you loading `best.pt` or `final.pt`?
- Did you pass `--temperature` too high? Try `0.7` or `0.5`.
- Was training actually completed, or did it stop at step 20 (smoke
  mode)? The smoke mode checkpoint is near-random.

---

## Inference

### `weights_only=True` error when loading checkpoint

You're on torch 2.4+ where `weights_only` defaults to `True`.
`meow.inference.load_model` explicitly passes `weights_only=False`
because the checkpoint contains a config dict. If you're writing your
own loading code, do the same:

```python
ckpt = torch.load("best.pt", weights_only=False)
```

### Generated text has `<pad>` or `<bos>` tokens in it

The `decode()` call should strip special tokens automatically. Check
that you're passing `skip_special_tokens=True`:

```python
tokenizer.decode(ids, skip_special_tokens=True)
```

The provided `chat_once()` already does this.

### Model generates the same response every time

Temperature is too low. Try `--temperature 0.8` or `--temperature 0.9`.
If still deterministic, check you're not accidentally using
`top_k=1` — that's greedy decoding.

---

## Tests

### `pytest tests/` fails on a fresh clone

First, verify the environment:

```bash
python -c "import torch, tokenizers, pytest; print(torch.__version__, tokenizers.__version__, pytest.__version__)"
```

You need `torch>=2.1.0`, `tokenizers>=0.15.0`, `pytest>=8.0.0`.

Then run tests with verbose output:

```bash
pytest tests/ -v
```

If a specific test fails, read the error. Most test failures on a
fresh clone are either missing dependencies or a mismatched torch
version (especially on Apple Silicon with MPS).

### Tests pass locally but fail in CI

Python version mismatch. The CI config in `.github/workflows/test.yml`
tests against 3.10, 3.11, and 3.12. If you're using 3.9 or 3.13 locally,
some type hints or f-string syntax might behave differently. Upgrade
to 3.11 or 3.12 for development.

---

## Hugging Face upload

### `huggingface-cli` not found

Install it:

```bash
pip install -e ".[hub]"
# or
pip install huggingface_hub
```

### Upload fails with auth error

The script expects `HF_TOKEN` to be set:

```bash
export HF_TOKEN=hf_your_token_here
export HF_USERNAME=your_hf_username
bash scripts/upload_to_hf.sh
```

Get a token from https://huggingface.co/settings/tokens. It needs
"write" permissions to create and upload to repos.

### Repo already exists error

The script uses `|| true` after repo creation so this shouldn't be
fatal. If it is, check the huggingface_hub version — very old versions
don't support `repo create`. Upgrade:

```bash
pip install --upgrade huggingface_hub
```

---

## Colab

### Colab disconnects during training

Free Colab has idle timeouts. Options:
1. Keep the browser tab active
2. Use a paid tier (Colab Pro) with background execution
3. Run training locally if you have a GPU

### "No GPU available" in Colab

Runtime → Change runtime type → T4 GPU. If T4 isn't available, try
later or use CPU fallback (slow but works).

### `!pip install -e .` fails in Colab

Colab's pip sometimes gets confused by editable installs. Try:

```
!pip install -q torch tokenizers pytest
!pip install -q -e .
```

Or install without editable mode:

```
!pip install -q .
```

---

## Contributing

### My PR fails the "cross-consistency" test

You added a category to `CATEGORIES` without adding it to
`CATEGORY_KEYWORDS` (or vice versa). The test `test_categories_match_rules`
catches this. Fix: make both files agree.

### `test_generate_template_samples_yield` fails

Your edits to the slot banks or rules are producing samples that
can't pass filters. Run the generator manually and look at the
rejection reasons:

```bash
python -m meow.generate_data --out-dir /tmp/test --n 500
```

---

## Still stuck?

- Read the source. Every file in `meow/` is under 500 lines (except
  `generate_data.py`) and heavily commented.
- Read `persona.md` for the character rules.
- Read `docs/architecture.md` (coming soon) for design decisions.
- Open a GitHub issue with: the command you ran, the exact error
  output, your Python/torch versions, and your OS.
