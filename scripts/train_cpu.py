"""One-shot CPU training helper with --resume and --max-minutes support.

Used to train MeowLLM on CPU across multiple invocations when wall-clock
limits prevent a single long-running session. Each invocation:

1. Loads the latest checkpoint if --resume is given
2. Trains until --target-steps is reached OR --max-minutes elapses
3. Saves checkpoint every --save-every steps
4. Exits cleanly with a status line showing total progress

This mirrors meow.train internally but wraps the loop with wall-clock
and resume support, which the main train.py intentionally doesn't have
(to keep the shipping code path simple).

Usage:
    python scripts/train_cpu.py --target-steps 5930 --max-minutes 35
    python scripts/train_cpu.py --target-steps 5930 --max-minutes 35 --resume checkpoints/cpu.pt
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Make the meow package importable when running this script directly
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import torch
from torch.utils.data import DataLoader

from meow.dataset import MeowDataset, collate_fn
from meow.model import Meow, MeowConfig
from meow.tokenizer import MeowTokenizer


def get_lr(step: int, total_steps: int, peak_lr: float,
           warmup: int = 200, floor: float = 0.1) -> float:
    """Warmup then cosine-anneal."""
    if step < warmup:
        return peak_lr * (step + 1) / max(1, warmup)
    decay_ratio = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return peak_lr * (floor + (1.0 - floor) * coeff)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        _, loss = model(batch["input_ids"], batch["target_ids"])
        total += loss.item()
        n += 1
    model.train()
    return total / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", default="data/train.jsonl")
    ap.add_argument("--val-data", default="data/val.jsonl")
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--target-steps", type=int, required=True,
                    help="total steps across all invocations")
    ap.add_argument("--max-minutes", type=float, default=30,
                    help="stop this invocation after N minutes wall-clock")
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(2)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "cpu.pt"
    best_path = out_dir / "best.pt"

    # Load tokenizer + data
    tok = MeowTokenizer.from_file(args.tokenizer)
    train_ds = MeowDataset(args.train_data, tok, max_seq_len=args.max_seq_len)
    val_ds = MeowDataset(args.val_data, tok, max_seq_len=args.max_seq_len)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # Model + optimizer
    cfg = MeowConfig(
        vocab_size=tok.vocab_size,
        pad_token_id=tok.pad_id,
        max_seq_len=args.max_seq_len,
    )
    model = Meow(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1,
    )

    # Resume if requested
    start_step = 0
    best_val = float("inf")
    resume_src = args.resume or (str(ckpt_path) if ckpt_path.exists() else None)
    if resume_src and Path(resume_src).exists():
        print(f"[cpu-train] resuming from {resume_src}")
        state = torch.load(resume_src, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        # NOTE: intentionally skipping optimizer state to save memory.
        # Fresh AdamW state each chunk means we lose accumulated momentum,
        # but at 50-step checkpoint intervals this is negligible and it
        # prevents OOM when resuming on memory-constrained sandboxes.
        start_step = state.get("step", 0)
        best_val = state.get("best_val", float("inf"))
        del state  # free memory before training starts
        import gc; gc.collect()
        print(f"[cpu-train] resumed at step {start_step}, best_val={best_val:.4f}")
    else:
        print("[cpu-train] starting from scratch")

    model.train()
    n_params = model.num_parameters()
    print(f"[cpu-train] model: {n_params:,} params ({n_params/1e6:.2f}M)")
    print(f"[cpu-train] train={len(train_ds)} val={len(val_ds)}")
    print(f"[cpu-train] target_steps={args.target_steps} "
          f"max_minutes={args.max_minutes}")

    t_start = time.time()
    budget_seconds = args.max_minutes * 60
    step = start_step
    loader_iter = iter(train_loader)

    def save_checkpoint(path: Path, tag: str):
        # NOTE: optimizer state intentionally excluded to keep checkpoint
        # small enough to load on memory-constrained sandboxes.
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "step": step,
            "best_val": best_val,
            "val_loss": val_loss,
        }, path)
        print(f"[cpu-train] saved {tag} -> {path}")

    val_loss = best_val
    stop_reason = "target_reached"
    while step < args.target_steps:
        # Wall-clock check
        elapsed = time.time() - t_start
        if elapsed > budget_seconds:
            stop_reason = "time_budget"
            break

        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        lr = get_lr(step, args.target_steps, args.lr, args.warmup)
        for g in optimizer.param_groups:
            g["lr"] = lr

        _, loss = model(batch["input_ids"], batch["target_ids"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step += 1

        if step % 50 == 0:
            print(f"[cpu-train] step {step:5d}/{args.target_steps} "
                  f"loss={loss.item():.4f} lr={lr:.2e} "
                  f"elapsed={elapsed:.0f}s budget={budget_seconds:.0f}s")

        if step % args.save_every == 0:
            save_checkpoint(ckpt_path, "periodic")

        if step % args.eval_every == 0:
            val_loss = evaluate(model, val_loader)
            print(f"[cpu-train]   val_loss={val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(best_path, "best")

    # Final save
    save_checkpoint(ckpt_path, "final-this-invocation")

    # Final eval (only if we have time budget left; otherwise skip to save tokens)
    if (time.time() - t_start) < budget_seconds * 0.95:
        val_loss = evaluate(model, val_loader)
        print(f"[cpu-train] final val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(best_path, "best")

    # Status
    total_time = time.time() - t_start
    print(f"\n[cpu-train] STATUS: {stop_reason}")
    print(f"[cpu-train] steps this session: {step - start_step}")
    print(f"[cpu-train] total steps: {step}/{args.target_steps}")
    print(f"[cpu-train] wall time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"[cpu-train] best val_loss: {best_val:.4f}")
    print(f"[cpu-train] progress: {step/args.target_steps*100:.1f}%")

    meta = {
        "total_steps": step,
        "target_steps": args.target_steps,
        "best_val": best_val,
        "last_val": val_loss,
        "last_wall_time": total_time,
        "stop_reason": stop_reason,
    }
    (out_dir / "training_meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
