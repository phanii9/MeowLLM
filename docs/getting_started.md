# Getting Started with MeowLLM

This is a longer, friendlier walkthrough than the README quick-start.
If you've never worked with a small LM before, read this first.

---

## Prerequisites

You need:

- **Python 3.10 or newer** (3.11 and 3.12 are also fine)
- **pip** (comes with Python)
- **Git** (to clone the repo)
- **Optional but recommended: a GPU**. Any NVIDIA GPU from the last 5 years
  will do. Free Google Colab with a T4 runtime works perfectly.

CPU-only works too, but full training takes a few hours instead of ~20
minutes. If you only want to *chat* with Miso (not retrain), CPU is fine.

## The 30-second version

```bash
git clone https://github.com/phanii9/MeowLLM.git
cd MeowLLM
pip install -e .
python -m meow.generate_data --out-dir data --n 20000
python -m meow.tokenizer train data/train.jsonl data/tokenizer.json
python -m meow.train --epochs 10          # ~20 min on GPU
python -m meow.inference \
    --checkpoint checkpoints/best.pt \
    --tokenizer data/tokenizer.json
```

That's it. You now have a cat.

The rest of this document explains what each step does, what you
should see, and what to do if something looks wrong.

---

## Step 1: Clone and install

```bash
git clone https://github.com/phanii9/MeowLLM.git
cd MeowLLM
pip install -e .
```

`pip install -e .` installs the `meow` package in editable mode, which
means the `meow-generate`, `meow-tokenizer`, `meow-train`, and `meow-chat`
commands become available system-wide, and any edits you make to files
in `meow/` take effect immediately without reinstalling.

**What you should see**: `Successfully installed meowllm-0.1.0`

**What you'll have on disk afterward**:

```
MeowLLM/
├── meow/           # source
├── tests/          # 68 pytest tests
├── notebooks/      # 2 Colab notebooks
├── data/           # pre-generated 20K sample dataset
├── docs/           # documentation
└── ...
```

---

## Step 2: Generate the dataset

```bash
python -m meow.generate_data --out-dir data --n 20000 --seed 42
```

This runs the slot-based compositional template generator in
`meow/generate_data.py`. It produces 20,000 synthetic (input, output)
pairs across 15 categories (greeting, hunger, naps, boxes, windows,
birds, etc.), deduplicates them, runs every sample through the strict
filter module in `meow/rules.py`, and writes a train/val split.

**Duration**: ~30 seconds on a laptop.

**Expected output**:

```
[main] loaded 38 eval prompts to exclude
[main] generating 20000 template + 0 llm = 20000 total
[template] 20000 accepted / 31988 attempts (yield 62.5%)
[template] rejection reasons:
  duplicate: 11744
  missing_category_vocab:greeting: 96
  ...

[main] wrote 19000 train + 1000 val samples
[main] train category distribution:
  affection               1255  (  6.6%)
  being_picked_up         1149  (  6.0%)
  ...
```

**Important numbers to look for**:

- **~19,000 train + 1,000 val samples**: correct total split
- **Each of the 15 categories has 1,000–1,500 samples**: balanced
- **Yield ~60%**: the rest is almost entirely deduplication (11,744 of
  ~12,000 rejections). This is expected at 20K scale — the slot banks
  start running out of unique combinations.

**What if the yield is much lower (say 30%)?** Check for recent edits
to `meow/rules.py` — you may have added a new banned phrase that's
rejecting too many valid outputs. See `docs/troubleshooting.md`.

**What you'll have after this step**:

```
data/
├── train.jsonl       # 19000 samples, ~2.8 MB
└── val.jsonl         # 1000 samples, ~150 KB
```

Each line is a JSON object:

```json
{"input": "hi miso", "output": "hello. i was in the sun spot.", "category": "greeting", "source": "template"}
```

You can inspect the data directly:

```bash
head -5 data/train.jsonl
```

---

## Step 3: Train the tokenizer

```bash
python -m meow.tokenizer train data/train.jsonl data/tokenizer.json
```

This trains a byte-level BPE tokenizer on the generated dataset. The
vocabulary size is capped at 2048 but usually trains out to ~1700
tokens because the dataset is small and the vocabulary is narrow
(lowercase only, limited topics).

**Duration**: ~5 seconds.

**Expected output**:

```
[tokenizer] trained vocab=1682, saved to data/tokenizer.json
```

**Don't worry if vocab is < 2048.** That's the BPE trainer saying "I
ran out of merges worth making because your corpus is small." This
is a feature: smaller vocab means smaller embedding matrix means
fewer model parameters.

**What you'll have after this step**:

```
data/tokenizer.json   # ~100 KB — the trained tokenizer
```

---

## Step 4: Train the model

```bash
python -m meow.train --epochs 10
```

This trains the transformer. Default hyperparameters match what's in
the model card: 4 layers, d_model 256, 4 heads, FFN 640, context 256,
batch size 64, lr 3e-4, AdamW with cosine decay.

**Duration**:

| Hardware | Time |
|---|---|
| Colab T4 (free tier) | ~15–20 minutes |
| Consumer GPU (3060, 4070) | ~10 minutes |
| Apple Silicon (M1/M2) | ~30–60 minutes |
| CPU only | ~2–4 hours |

**Expected output** (first few steps):

```
[train] device: cuda
[train] tokenizer vocab: 1682
[train] train=19000 val=1000
[train] model: 3,447,552 params (3.45M)
[train] 10 epochs × 296 steps = 2960 total
[train] warmup: 200 steps
[train] step     0/2960 loss=7.4339 lr=1.50e-06 (0.8s)
[train] step    59/2960 loss=5.2118 lr=8.85e-05 (12.3s)
...
```

**How to know it's working**:

- Loss should start around **7.4** (≈ ln(vocab_size)) and drop
- By step 500: loss should be **< 3.0**
- By end of epoch 1: val loss should be **< 1.5**
- Final val loss after 10 epochs: typically **< 0.8**

**If loss plateaus above 3.0**, something is wrong. Check
`docs/troubleshooting.md`.

**What you'll have after this step**:

```
checkpoints/
├── best.pt              # lowest-val-loss checkpoint
├── final.pt             # last checkpoint
└── training_meta.json   # training run metadata
```

---

## Step 5: Chat with Miso

```bash
python -m meow.inference \
    --checkpoint checkpoints/best.pt \
    --tokenizer data/tokenizer.json
```

Or with a single prompt:

```bash
python -m meow.inference \
    --checkpoint checkpoints/best.pt \
    --tokenizer data/tokenizer.json \
    --prompt "hi miso"
```

**Expected output** after full training:

```
you:  hi miso
miso: hello. i was in the sun spot. you may continue.

you:  are you hungry
miso: yes. this is an emergency. the bowl is almost empty.

you:  what is the capital of france
miso: i do not know this word. is it food.
```

**If outputs look like gibberish** ("ut were. playing"), the model is
undertrained. Run more epochs or check that training actually finished.

---

## Step 6: Evaluate

Run the held-out eval suite to see how well your trained model stays
in character:

```python
import torch
from meow.inference import load_model, chat_once
from meow.tokenizer import MeowTokenizer
from meow.eval_cases import EVAL_PROMPTS, evaluate_batch, print_report

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, cfg = load_model('checkpoints/best.pt', device=device)
tokenizer = MeowTokenizer.from_file('data/tokenizer.json')

outputs, categories = [], []
for prompt, cat in EVAL_PROMPTS:
    response = chat_once(model, tokenizer, prompt, device=device)
    outputs.append(response)
    categories.append(cat)

stats = evaluate_batch(outputs, categories)
print_report(stats)
```

**Target numbers after a full 10-epoch run**:

- lowercase: ~100%
- length: ~95%+
- no banned phrases: ~95%+
- cat framing: 70–85%
- overall: 60–85%

If you get much lower, train for more epochs or check the samples
by eye to see what the model is actually doing.

---

## What to do next

- **Chat with Miso** interactively and see what it sounds like
- **Read `meow/rules.py`** to see how character constraints are enforced
- **Read `meow/model.py`** to see the transformer in ~280 lines
- **Read `persona.md`** to understand the character bible
- **Run `pytest tests/`** to verify everything is working
- **Publish to Hugging Face** with `scripts/upload_to_hf.sh` (see `docs/release.md`)
- **Add a new category** (see `CONTRIBUTING.md`)

Happy loafing.
