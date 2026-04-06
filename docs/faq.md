# Frequently Asked Questions

## About the project

### What is MeowLLM?

A ~3.5M parameter decoder-only transformer trained from scratch to
speak in the voice of a single house cat character called Miso. It's
not an assistant. It has no system prompt. The personality is baked
into the weights by training on 20,000 carefully crafted synthetic
examples.

### Why would anyone want this?

Three reasons:

1. **Education**. MeowLLM is small enough to read end-to-end in an
   afternoon. The entire transformer is 278 lines. The whole package
   is ~2,800 lines of Python. You can understand how a tiny modern
   character LM actually works from scratch.

2. **Forkability**. The architecture and dataset pipeline is the
   blueprint for *any* character model: dog, pirate, robot, wizard.
   You can swap out `persona.md`, rewrite the slot banks in
   `generate_data.py`, and train a new character in 20 minutes.

3. **It's genuinely fun**. A small cat model that tells you to stop
   touching its belly is charming in a way that polished assistants
   aren't.

### Is this a serious project?

Serious in its engineering (68 tests, CI, strict filtering, clean
architecture), not serious in its ambition (it's a cat toy). Both of
those are intentional.

---

## About the model

### Why only 3.5M parameters?

Because bigger doesn't help at this scale. A character model's quality
is determined by **data consistency and filter discipline**, not
parameter count. At 3.5M params, the model memorizes voice patterns
well enough to be charming, fits on any hardware, and trains in 20
minutes. Making it 30M wouldn't meaningfully improve the character;
it would just take longer to train.

### Why not fine-tune a pretrained model like Llama 3 or Phi?

That's a different project and a different philosophy. Fine-tuning a
pretrained model gives you a "cat character layer" on top of a general
assistant. MeowLLM bakes the character into a model that has never
known how to be anything else. The result is more consistent and more
interesting as an educational artifact — you can see *exactly* where
the voice comes from.

Also: fine-tuning a 3B parameter model requires a beefier GPU. Training
MeowLLM from scratch works on a free Colab T4.

### Why lowercase only?

Two reasons:

1. **It's a character choice**. Cats are not SHOUTING, and they are
   not formally capitalizing the start of sentences. Lowercase fits.
2. **It's a practical choice**. Lowercase halves the effective
   vocabulary the model has to learn (no "Hello" vs "hello" distinction)
   and makes the strict filter much easier to enforce.

### Why no multi-turn conversation?

MeowLLM is trained single-turn because:

- At 256 tokens of context, there's barely room for a multi-turn
  conversation
- Training single-turn is simpler and more reliable at this scale
- Miso's voice is so narrow that each turn is essentially independent
  anyway

That said, MeowLLM uses RoPE (which extrapolates well) and has a 256-token
context window, so extending to short multi-turn conversations later is
possible.

### Why SwiGLU instead of GELU?

Modern LLMs use SwiGLU because it produces slightly better quality
at the same parameter budget. At 3.5M params the absolute improvement
is small, but it's free (same FLOPS, same memory), so there's no reason
not to use it. This also makes the repo a more honest demonstration
of modern tiny-LM practice.

### Why RoPE instead of learned positional embeddings?

Four reasons:

1. RoPE extrapolates beyond training length better
2. It doesn't add learned parameters
3. It's the industry standard in every modern LLM (Llama, Mistral,
   Qwen, DeepSeek, Gemma)
4. Teaching readers RoPE is more useful than teaching them 2017-era
   positional embeddings

### Why use SDPA?

`torch.nn.functional.scaled_dot_product_attention` gives us flash
attention kernels for free on modern GPUs, with zero extra code and
a graceful fallback on CPU. It's a one-line change from manual matmul
attention that makes training meaningfully faster on GPU with no
downside.

### Can I make it bigger?

Yes. Change the config in `meow/model.py`:

```python
cfg = MeowConfig(
    d_model=512,    # was 256
    n_layers=6,     # was 4
    n_heads=8,      # was 4
    ffn_hidden=1280 # was 640
)
```

This gives you ~15M parameters. You'll also need more data to match
— regenerate with `--n 50000` and expand the slot banks in
`generate_data.py` first (or you'll hit duplicate-rejection walls).

### Can I make it smaller?

Yes, but I wouldn't. Below ~2M parameters, character consistency
starts to break down — the model doesn't have enough capacity to
memorize 20K samples worth of voice patterns. If you want to go
smaller for research purposes, try 2 layers with d_model 192 and
see how it looks.

---

## About the data

### Why synthetic data?

Because there's no real dataset of 20,000 in-character house cat
responses. Nobody has that. Synthetic generation is the only way to
get the scale you need while keeping character consistency tight.

### Why slot-based templates instead of just LLM generation?

Pure LLM generation is expensive, slow, and drifts from the persona
over time. Pure static templates produce low-diversity data that the
model memorizes verbatim. The slot-based middle path gives you the
quality control of templates with the surface diversity of generation.

### Why 20,000 samples and not 100,000?

At 20,000 the generator starts running into duplicate-rejection walls
(you saw ~11,700 duplicates in the 20K run). Past that point you'd
need to expand the slot banks to get more unique outputs, which is
hand work. 20,000 is the comfortable ceiling of the current slot banks.

### Can I use a different character?

Absolutely. That's one of the main use cases. Steps:

1. Rewrite `persona.md` with your new character's rules
2. Rewrite `meow/generate_data.py` slot banks to match
3. Update `meow/rules.py` category keywords
4. Update `meow/eval_cases.py` held-out prompts
5. Retrain

The model architecture and training loop don't change at all.

### Why not use real cat behavior data?

There is no such dataset. Even if there were, it would be noisy, hard
to clean, and not license-safe. Synthetic character data sidesteps
all of that.

### Why are there so many banned phrases?

Every banned phrase in `meow/rules.py` was added in response to a
real failure mode — either an LLM-generated sample that slipped through
into "assistant mode" ("certainly!", "here's how"), or a template that
accidentally used corporate-speak. The list grows as new failure modes
are found. Being strict about this is the only way to keep the
character consistent at this scale.

---

## About training

### How much does training cost?

Free on Colab's free T4 tier. ~20 minutes of wall-clock time. On AWS
or GCP you'd pay maybe $0.10 for a T4 instance for 20 minutes.

### How do I know training is working?

Watch the loss. It should start around 7.4 (= ln(vocab_size)) and
decrease steadily. By step 500 it should be under 3.0. By the end of
epoch 1 it should be under 1.5. See `docs/getting_started.md` for
details.

### Why is my val loss much higher than my train loss?

Overfitting. The model is memorizing the train set. At 3.5M params on
19K samples, some overfitting is expected — voice memorization is
actually what we want at this scale. If the gap is large (train 0.3,
val 2.0) consider reducing epochs to 5 or adding dropout.

### Can I train on a CPU?

Yes, but it takes hours instead of minutes. For CPU:

```bash
python -m meow.train --batch-size 16 --epochs 5
```

Smaller batch, fewer epochs, still gets you a working cat — just
lower quality.

---

## About inference

### Why does the model generate the same response twice?

Low temperature or top_k. Try:

```bash
python -m meow.inference \
    --checkpoint checkpoints/best.pt \
    --tokenizer data/tokenizer.json \
    --temperature 0.9 \
    --top-k 40
```

### Can I run it in a browser?

Yes, with ONNX or transformers.js, but neither is set up in the
current repo. Convert with:

```python
import torch
model = ...  # load MeowLLM
dummy_input = torch.randint(0, 1700, (1, 32))
torch.onnx.export(model, dummy_input, "meow.onnx", ...)
```

This is on the "nice to have" list for v0.2.

### Can I deploy this as an API?

Technically yes, but it's a toy. If you really want to: wrap
`chat_once` in a FastAPI endpoint, containerize with Docker, and
deploy to any HTTP host. The model is so small that a single-CPU
instance can handle many requests per second.

---

## About the repo

### Why 15 categories and not more?

Because 15 is enough to cover a cat's conceptual world: food, sleep,
play, territory, affection, enemies (vacuum, dogs), humans, and
deflections (nonsense questions, being picked up, jealousy). Adding
more categories is easy (see `CONTRIBUTING.md`) but each new category
requires hand-written slot fragments.

### Why are there so many tests?

68 tests isn't "so many" for a published Python package. The tests
are fast (7 seconds total) and catch real bugs:

- Cross-consistency between `CATEGORIES` and `CATEGORY_KEYWORDS`
- The whole-phrase matching regression (`"i can"` as a substring)
- Loss masking correctness
- Tied embedding configuration
- RoPE at position 0 is identity

Without tests, any refactor could silently break character.

### Can I remove the Anthropic API path if I don't have an API key?

You never need to touch it. The default generator is template-only
(`--use-llm` is off by default). The `AnthropicClient` class is only
imported lazily inside `generate_llm_samples()`, so if you never pass
`--use-llm`, the `anthropic` package isn't even imported.

### Why MIT license and not Apache 2.0?

MIT is shorter, more permissive, and more common for educational
projects. If you want to use MeowLLM as the base for commercial work,
MIT is simpler. Apache 2.0 is also fine and has a more explicit
patent grant, but for a 3.5M cat model that's overkill.

---

## Still have a question?

Open a GitHub issue or discussion. Keep it focused.
