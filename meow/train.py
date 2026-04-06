"""
meow.train — training loop for Miso.

Usage:
    # Smoke test (CPU, small)
    python -m meow.train --smoke

    # Full training (single GPU recommended)
    python -m meow.train \
        --train-data data/train.jsonl \
        --val-data data/val.jsonl \
        --tokenizer data/tokenizer.json \
        --out-dir checkpoints \
        --batch-size 64 \
        --epochs 10 \
        --lr 3e-4
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from meow.dataset import MeowDataset, collate_fn
from meow.model import Meow, MeowConfig
from meow.tokenizer import MeowTokenizer


def get_lr(step: int, total_steps: int, peak_lr: float,
           warmup: int = 200, floor: float = 0.1) -> float:
    """Warmup to peak_lr over `warmup` steps, then cosine-anneal to floor * peak_lr."""
    if step < warmup:
        return peak_lr * (step + 1) / max(1, warmup)
    decay_ratio = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return peak_lr * (floor + (1.0 - floor) * coeff)


def evaluate(model: Meow, loader: DataLoader, device: str) -> float:
    """Compute mean loss over validation set."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            _, loss = model(input_ids, target_ids)
            total_loss += loss.item()
            n_batches += 1
    model.train()
    return total_loss / max(1, n_batches)


def save_checkpoint(
    path: Path,
    model: Meow,
    cfg: MeowConfig,
    step: int,
    val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": asdict(cfg),
        "step": step,
        "val_loss": val_loss,
    }, path)
    print(f"[train] checkpoint saved: {path} (val_loss={val_loss:.4f})")


def train(args: argparse.Namespace) -> None:
    device = args.device
    print(f"[train] device: {device}")

    # Tokenizer
    tokenizer = MeowTokenizer.from_file(args.tokenizer)
    print(f"[train] tokenizer vocab: {tokenizer.vocab_size}")

    # Datasets
    train_ds = MeowDataset(args.train_data, tokenizer, max_seq_len=args.max_seq_len)
    val_ds = MeowDataset(args.val_data, tokenizer, max_seq_len=args.max_seq_len)
    print(f"[train] train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Model
    cfg = MeowConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ffn_hidden=args.ffn_hidden,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        pad_token_id=tokenizer.pad_id,
    )
    model = Meow(cfg).to(device)
    n_params = model.num_parameters()
    print(f"[train] model: {n_params:,} params ({n_params/1e6:.2f}M)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # Training schedule
    steps_per_epoch = len(train_loader)
    max_steps = steps_per_epoch * args.epochs
    warmup_steps = min(args.warmup_steps, max_steps // 10)
    print(f"[train] {args.epochs} epochs × {steps_per_epoch} steps = {max_steps} total")
    print(f"[train] warmup: {warmup_steps} steps")

    # Training loop
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    global_step = 0
    start_time = time.time()
    log_interval = max(1, max_steps // 50) if not args.smoke else 10
    eval_interval = max(1, max_steps // 10) if not args.smoke else 50

    model.train()
    for epoch in range(args.epochs):
        for batch in train_loader:
            lr = get_lr(global_step, max_steps, args.lr, warmup_steps, floor=0.1)
            for g in optimizer.param_groups:
                g["lr"] = lr

            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)

            _, loss = model(input_ids, target_ids)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if global_step % log_interval == 0:
                elapsed = time.time() - start_time
                print(f"[train] step {global_step:5d}/{max_steps} "
                      f"loss={loss.item():.4f} lr={lr:.2e} "
                      f"({elapsed:.1f}s)")

            if global_step > 0 and global_step % eval_interval == 0:
                val_loss = evaluate(model, val_loader, device)
                print(f"[train]   val_loss={val_loss:.4f}")
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(out_dir / "best.pt", model, cfg,
                                    global_step, val_loss)

            global_step += 1

            if args.smoke and global_step >= args.max_smoke_steps:
                break
        if args.smoke and global_step >= args.max_smoke_steps:
            break

    # Final eval + checkpoint
    final_val = evaluate(model, val_loader, device)
    print(f"[train] final val_loss={final_val:.4f}")
    save_checkpoint(out_dir / "final.pt", model, cfg, global_step, final_val)
    if final_val < best_val:
        save_checkpoint(out_dir / "best.pt", model, cfg, global_step, final_val)

    # Save final metadata
    with (out_dir / "training_meta.json").open("w") as f:
        json.dump({
            "final_val_loss": final_val,
            "best_val_loss": min(best_val, final_val),
            "total_steps": global_step,
            "total_time_sec": time.time() - start_time,
            "config": asdict(cfg),
        }, f, indent=2)

    print(f"[train] done. total time: {time.time() - start_time:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", default="data/train.jsonl")
    ap.add_argument("--val-data", default="data/val.jsonl")
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--max-seq-len", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--ffn-hidden", type=int, default=640)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true",
                    help="Run a tiny smoke-test training to verify pipeline")
    ap.add_argument("--max-smoke-steps", type=int, default=50)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    if args.smoke:
        args.batch_size = 8
        args.epochs = 1
        args.max_seq_len = 128
        print("[train] SMOKE MODE: tiny training to verify pipeline")

    train(args)


if __name__ == "__main__":
    main()
