"""
meow.dataset — torch Dataset for Miso chat training.

Loads a JSONL file of (input, output) pairs, encodes each as a chat
sequence using MeowTokenizer, and returns (input_ids, target_ids) where
target_ids has -100 in the positions corresponding to the user turn
(so CrossEntropyLoss ignores them).

The loss mask ensures the model is only supervised on the output
portion of each sequence. This is important — otherwise the model
wastes capacity learning to predict user inputs, which it will never
need to generate at inference time.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from meow.tokenizer import MeowTokenizer

IGNORE_INDEX = -100  # matches torch.nn.CrossEntropyLoss default


class MeowDataset(Dataset):
    """Chat dataset for Miso.

    Each item returns a dict with:
      - input_ids:  (seq_len,) long tensor, the token sequence
      - target_ids: (seq_len,) long tensor, shifted by 1, with user-turn
                    positions set to IGNORE_INDEX

    Sequences are padded/truncated to `max_seq_len`.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: MeowTokenizer,
        max_seq_len: int = 256,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples: list[dict] = []
        with Path(jsonl_path).open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        row = self.samples[idx]
        ids, output_start = self.tokenizer.encode_chat(
            row["input"], row["output"]
        )

        # Truncate if too long (shouldn't happen in practice for this dataset)
        if len(ids) > self.max_seq_len:
            ids = ids[: self.max_seq_len]
            output_start = min(output_start, self.max_seq_len)

        # Build input_ids and target_ids for next-token prediction.
        # input_ids = ids[:-1], target_ids = ids[1:]
        # Mask positions in target_ids that correspond to the user turn.
        input_ids = ids[:-1]
        target_ids = ids[1:]

        # Positions in target_ids:
        #   target_ids[i] is the token that input_ids[i] should predict.
        #   target_ids[i] corresponds to ids[i+1].
        # We want to supervise predictions for positions where the TARGET
        # is part of the output (i.e., ids[i+1] is in the output span).
        # The output span is [output_start, len(ids)).
        # So target_ids[i] should be supervised iff i + 1 >= output_start,
        # i.e., i >= output_start - 1.
        supervise_from = max(0, output_start - 1)
        masked_targets = [
            tid if i >= supervise_from else IGNORE_INDEX
            for i, tid in enumerate(target_ids)
        ]

        # Pad to max_seq_len - 1 (because input_ids/target_ids are one
        # shorter than the original ids).
        target_len = self.max_seq_len - 1
        pad_amount = target_len - len(input_ids)
        if pad_amount > 0:
            input_ids = input_ids + [self.tokenizer.pad_id] * pad_amount
            masked_targets = masked_targets + [IGNORE_INDEX] * pad_amount

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(masked_targets, dtype=torch.long),
        }


def collate_fn(batch: list[dict]) -> dict:
    """Stack pre-padded items into a batch."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "target_ids": torch.stack([b["target_ids"] for b in batch]),
    }


if __name__ == "__main__":
    # Smoke test
    tok = MeowTokenizer.from_file("data/tokenizer.json")
    ds = MeowDataset("data/train.jsonl", tok, max_seq_len=128)
    print(f"dataset size: {len(ds)}")
    item = ds[0]
    print(f"input_ids shape: {item['input_ids'].shape}")
    print(f"target_ids shape: {item['target_ids'].shape}")
    print(f"first 20 input_ids:  {item['input_ids'][:20].tolist()}")
    print(f"first 20 target_ids: {item['target_ids'][:20].tolist()}")
    # Verify some positions are ignored
    ignored = (item["target_ids"] == IGNORE_INDEX).sum().item()
    supervised = (item["target_ids"] != IGNORE_INDEX).sum().item()
    print(f"ignored positions: {ignored}, supervised: {supervised}")
