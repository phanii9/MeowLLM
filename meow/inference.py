"""
meow.inference — load a trained Miso checkpoint and chat.

Usage:
    python -m meow.inference --checkpoint checkpoints/best.pt \
                             --tokenizer data/tokenizer.json

    # Single-prompt mode
    python -m meow.inference --checkpoint checkpoints/best.pt \
                             --tokenizer data/tokenizer.json \
                             --prompt "hi miso"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from meow.model import Meow, MeowConfig
from meow.tokenizer import MeowTokenizer


def load_model(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> tuple[Meow, MeowConfig]:
    """Load a checkpoint and reconstruct the model."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_dict = ckpt["config"]
    # Filter to actual MeowConfig fields (in case extras were saved)
    cfg_fields = {f.name for f in MeowConfig.__dataclass_fields__.values()}
    cfg = MeowConfig(**{k: v for k, v in cfg_dict.items() if k in cfg_fields})
    model = Meow(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, cfg


def chat_once(
    model: Meow,
    tokenizer: MeowTokenizer,
    prompt: str,
    device: str = "cpu",
    temperature: float = 0.8,
    top_k: int = 40,
    max_new_tokens: int = 60,
) -> str:
    """Generate a single response to `prompt`."""
    prompt = prompt.strip().lower()
    ids, _ = tokenizer.encode_chat(prompt, output_text=None)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    out = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_token_id=tokenizer.eos_id,
    )

    # Decode only the new tokens (after the prompt)
    new_tokens = out[0, len(ids):].tolist()
    # Strip trailing EOS if present
    if new_tokens and new_tokens[-1] == tokenizer.eos_id:
        new_tokens = new_tokens[:-1]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def interactive_loop(
    model: Meow,
    tokenizer: MeowTokenizer,
    device: str,
    temperature: float,
    top_k: int,
) -> None:
    print("miso is loafing. type 'quit' to exit.\n")
    while True:
        try:
            prompt = input("you:  ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        if prompt.lower() in {"quit", "exit", "q"}:
            break
        response = chat_once(
            model, tokenizer, prompt,
            device=device, temperature=temperature, top_k=top_k,
        )
        print(f"miso: {response}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=60)
    ap.add_argument("--prompt", default=None,
                    help="Single-prompt mode: emit one response and exit")
    args = ap.parse_args()

    print(f"[inference] loading model from {args.checkpoint}...")
    model, cfg = load_model(args.checkpoint, device=args.device)
    tokenizer = MeowTokenizer.from_file(args.tokenizer)
    print(f"[inference] model: {model.num_parameters()/1e6:.2f}M params, "
          f"vocab: {tokenizer.vocab_size}")

    if args.prompt:
        response = chat_once(
            model, tokenizer, args.prompt,
            device=args.device,
            temperature=args.temperature,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
        )
        print(response)
    else:
        interactive_loop(
            model, tokenizer,
            device=args.device,
            temperature=args.temperature,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()
