# Release Process

Step-by-step guide for taking MeowLLM from "local working directory"
to "published open-source project with pretrained weights on Hugging
Face that anyone can use."

This is a one-time process. You only need to do it once for v0.1.0.

---

## Prerequisites

Before you start, make sure you have:

- [ ] A GitHub account
- [ ] A Hugging Face account
- [ ] Git installed and configured with your GitHub credentials
- [ ] Python 3.10+ installed
- [ ] A machine with a GPU (or access to Google Colab with T4 runtime)
- [ ] About 1 hour of total time (most of it is waiting for training)

---

## Phase 1: Prepare the repo locally (~5 minutes)

### 1.1 Unzip and enter the repo

```bash
unzip MeowLLM.zip
cd meowllm
```

### 1.2 Verify everything works

```bash
# Install in editable mode
pip install -e ".[dev]"

# Run the rules smoke test (should show 34/34)
python scripts/test_rules_smoke.py

# Run the full pytest suite (should show 68/68 passing)
pytest tests/ -q

# Run a generator smoke test
python -m meow.generate_data --out-dir /tmp/test --n 200 --seed 0
```

All four must succeed. If any fails, check `docs/troubleshooting.md`
before proceeding.

### 1.3 Review and personalize

Open these files and change `phanii9` / `YOUR_USERNAME` placeholders
if you want a different GitHub identity:

- `README.md` — badges and clone URL
- `pyproject.toml` — `[project.urls]` section
- `CITATION.cff` — author field
- `scripts/upload_to_hf.sh` — repo names
- Both `.ipynb` files — HF repo ID

A quick search-and-replace:

```bash
grep -rln "phanii9" . --include="*.md" --include="*.toml" \
    --include="*.sh" --include="*.ipynb" --include="*.cff"
```

---

## Phase 2: Publish to GitHub (~2 minutes)

### 2.1 Create the GitHub repo

Go to https://github.com/new and create a new empty repository named
`MeowLLM` under your account. **Do not** initialize with a README or
LICENSE — the repo already has those.

### 2.2 Initialize and push

```bash
git init
git add .
git commit -m "initial release: MeowLLM v0.1.0"
git branch -M main
git remote add origin git@github.com:phanii9/MeowLLM.git
git push -u origin main
```

Or over HTTPS:

```bash
git remote add origin https://github.com/phanii9/MeowLLM.git
```

### 2.3 Verify CI runs

After pushing, go to `github.com/phanii9/MeowLLM/actions`. The
"tests" workflow should start automatically and complete in ~3
minutes on Python 3.10, 3.11, and 3.12. All three should pass.

**If CI fails:** read the error in the Actions log. Common causes:
Python version mismatch (CI uses fresh 3.10–3.12), missing dependency
(check `pyproject.toml`), or a test that passes locally but fails
in a clean environment (re-run locally in a fresh venv to reproduce).

---

## Phase 3: Train the actual model (~20 minutes)

You need real pretrained weights before anyone can chat with Miso
without retraining from scratch.

### Option A: Train on Google Colab (recommended)

1. Go to https://colab.research.google.com/
2. File → Open notebook → GitHub tab
3. Enter `phanii9/MeowLLM`, select `notebooks/train_meow.ipynb`
4. Runtime → Change runtime type → **T4 GPU**
5. Runtime → Run all
6. Wait ~20 minutes

The notebook will:
- Clone the repo fresh
- Install the package
- Generate the dataset
- Train the tokenizer
- Train the model for 10 epochs
- Evaluate on held-out prompts
- Save checkpoints/best.pt

**What to look for during training:**

- Loss starts around 7.4 and drops steadily
- By step 500, loss should be under 3.0
- Final val_loss should be well under 1.0

**When training finishes**, the notebook prints a summary and shows
sample chat outputs. Verify the cat actually sounds like a cat.

### 3.1 Download the artifacts

At the end of the training notebook, download these files via the
Colab file browser (left sidebar):

- `checkpoints/best.pt` — the trained model (~14 MB)
- `data/tokenizer.json` — the trained tokenizer (~100 KB)

Or use the Colab code:

```python
from google.colab import files
files.download('checkpoints/best.pt')
files.download('data/tokenizer.json')
```

Save both files to your local `meowllm/` directory, replacing any
smoke-trained versions that might be there.

### Option B: Train locally (if you have a GPU)

```bash
python -m meow.generate_data --out-dir data --n 20000 --seed 42
python -m meow.tokenizer train data/train.jsonl data/tokenizer.json
python -m meow.train --epochs 10
```

About 20 minutes on a consumer GPU (3060/4070/4090), longer on
M1/M2, hours on CPU.

### 3.2 Verify the model actually sounds like a cat

```bash
python -m meow.inference \
    --checkpoint checkpoints/best.pt \
    --tokenizer data/tokenizer.json \
    --prompt "hi miso"
```

Expected (approximately):

```
hello. i was in the sun spot. you may continue.
```

If outputs are still gibberish, something went wrong. Check that
training actually completed (look at `checkpoints/training_meta.json`
for the final val_loss) and retry.

### 3.3 Run the held-out eval and record the numbers

```bash
python -c "
import torch
from meow.inference import load_model, chat_once
from meow.tokenizer import MeowTokenizer
from meow.eval_cases import EVAL_PROMPTS, evaluate_batch, print_report

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, cfg = load_model('checkpoints/best.pt', device=device)
tokenizer = MeowTokenizer.from_file('data/tokenizer.json')

outputs, cats = [], []
for prompt, cat in EVAL_PROMPTS:
    response = chat_once(model, tokenizer, prompt, device=device)
    outputs.append(response)
    cats.append(cat)

stats = evaluate_batch(outputs, cats)
print_report(stats)
" > eval_results.txt
cat eval_results.txt
```

Save these numbers. You'll put them in the model card in Phase 5.

---

## Phase 4: Publish to Hugging Face (~5 minutes)

### 4.1 Get a Hugging Face token

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Name: `meowllm-release`
4. Type: **Write** (needed for repo creation and uploads)
5. Copy the token

### 4.2 Install huggingface_hub

```bash
pip install -e ".[hub]"
# or: pip install huggingface_hub
```

### 4.3 Run the upload script

```bash
export HF_TOKEN=hf_your_token_here
export HF_USERNAME=hunt3rx99
bash scripts/upload_to_hf.sh
```

The script creates two Hugging Face repos:

- **Model repo**: `hf.co/hunt3rx99/meowllm` — contains `best.pt`,
  `tokenizer.json`, and the model card as `README.md`
- **Dataset repo**: `hf.co/datasets/hunt3rx99/meowllm-miso` — contains
  `train.jsonl`, `val.jsonl`, and the dataset card as `README.md`

### 4.4 Verify the upload

Visit both URLs in a browser:

- https://huggingface.co/hunt3rx99/meowllm
- https://huggingface.co/datasets/hunt3rx99/meowllm-miso

Both pages should render the cards correctly (with the YAML front
matter converted into the sidebar tags) and show the uploaded files.

### 4.5 Test the chat notebook against the live upload

Open `notebooks/chat_with_meow.ipynb` on Colab (fresh, not the one
you used for training). Run all cells. Cell 4 will try to download
from `hunt3rx99/meowllm` and should succeed this time — no fallback
training needed. You should be chatting with Miso within ~60 seconds
of opening the notebook.

This is the end-user experience. If it works here, it works for
everyone.

---

## Phase 5: Update documentation with real numbers (~5 minutes)

Now that you have real eval numbers, update the documentation to
reflect them (no more placeholder "smoke baseline" text).

### 5.1 Update `docs/model_card.md`

Find the "Smoke test numbers (reference point)" section and replace
it with your real numbers from Phase 3.3. Something like:

```markdown
### Evaluation numbers (full 10-epoch run)

| dimension          | pass rate |
|--------------------|-----------|
| lowercase          |    100.0% |
| length             |     98.5% |
| no banned phrases  |     97.0% |
| cat framing        |     85.3% |
| **overall**        |  **82.1%**|

Training ran for 10 epochs on a T4 GPU (~19 minutes). Final val_loss: 0.73.
```

### 5.2 Update `README.md`

Similar update in the "Baseline numbers" section.

### 5.3 Commit and push the updates

```bash
git add docs/model_card.md README.md
git commit -m "update eval numbers from real training run"
git push
```

### 5.4 Re-upload the updated model card

```bash
bash scripts/upload_to_hf.sh
```

The script is idempotent — running it again just updates the
uploaded files.

---

## Phase 6: Announce (~5 minutes)

You now have a fully working open-source character LM. Time to tell
people.

### 6.1 Create a GitHub release

1. Go to `github.com/phanii9/MeowLLM/releases/new`
2. Tag: `v0.1.0`
3. Title: `MeowLLM v0.1.0 — initial release`
4. Description: copy from `CHANGELOG.md` (the `[0.1.0]` section)
5. Publish

### 6.2 Share

Pick one or two channels:

- **Twitter / X**: a screenshot of a Miso chat + links to GitHub and HF
- **Hacker News**: "Show HN: MeowLLM — a 3.5M parameter LM that talks
  like a cat" with a link to the GitHub repo
- **r/LocalLLaMA**: same post, more technical framing
- **Lobste.rs**: same post, even more technical framing
- **Your blog**: write a short post about what you built and what you
  learned

Keep it honest. The README is already written to match this tone.

---

## Post-release

### What to do if someone opens an issue

- **Bug report**: reproduce locally, add a failing test if possible,
  fix, push. The 68-test suite catches most regressions before they ship.
- **Feature request**: decide if it fits the scope (see "What this
  project is NOT" in the README). If yes, label and consider. If no,
  politely decline.
- **Question**: answer in the issue, and if it's a common question,
  add it to `docs/faq.md` in the next release.

### What to do if someone opens a PR

- Verify CI passes (it will run automatically)
- Read the change for voice consistency (does it match `persona.md`?)
- Check the tests didn't get weakened to make the PR pass
- Merge if good, request changes if not

### When to release v0.2.0

When you have something worth releasing. Possibilities:

- Expanded slot banks (more diversity)
- New categories
- Better eval numbers from a longer training run
- ONNX export for browser inference
- Multi-turn format experiment

Bump the version in `pyproject.toml` and `CITATION.cff`, update
`CHANGELOG.md` with a new `[0.2.0]` section, tag a new GitHub release.

---

## Troubleshooting the release process

### CI fails after push

Read the Actions log. Most common cause: a Python version difference
between your local machine and CI (3.10/3.11/3.12). Reproduce locally
in a fresh venv with the same Python version.

### `huggingface-cli login` fails

Make sure your token has **write** permissions, not just read. Check
at https://huggingface.co/settings/tokens.

### `huggingface-cli upload` fails with "repo not found"

The script creates the repos with `huggingface-cli repo create` first,
but this can fail silently on some versions. Try manually:

```bash
huggingface-cli repo create meowllm --type model
huggingface-cli repo create meowllm-miso --type dataset
```

Then re-run `scripts/upload_to_hf.sh`.

### The Colab chat notebook can't download from HF after upload

Check that the repo is public. Private repos require the token to
download. On the Hugging Face model page, click "Settings" and make
sure visibility is "Public".

### Training finished but outputs are still gibberish

Did you actually train for 10 epochs? Check `checkpoints/training_meta.json`.
If `total_steps` is small (< 1000), training was interrupted. Re-run.

If `total_steps` is correct and outputs are still bad, regenerate
the dataset (delete `data/` and rerun `meow.generate_data`), retrain
the tokenizer, and retrain the model. Something got corrupted.

---

## That's it

Once Phase 6 is done, you have a published open-source project.
Anyone can:

- Clone the repo
- Chat with Miso in 60 seconds via the chat notebook
- Retrain their own version in 20 minutes
- Fork it into a different character

The repo is designed to not need ongoing maintenance. It's a
fire-and-forget educational artifact. You can walk away and it will
still work in a year — assuming `torch`, `tokenizers`, and
`huggingface_hub` don't make breaking changes.
