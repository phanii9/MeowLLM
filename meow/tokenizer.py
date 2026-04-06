"""
meow.tokenizer — byte-level BPE tokenizer for Miso.

Uses Hugging Face tokenizers library to train a small BPE on the
generated dataset. The tokenizer is tiny (2048 vocab) and purpose-built
for lowercase cat speech.

Special tokens:
  <pad>   - padding, id 0
  <bos>   - beginning of sequence, id 1
  <eos>   - end of sequence, id 2
  <user>  - marks start of user turn, id 3
  <miso>  - marks start of miso turn, id 4

The chat format is:
  <bos><user>{input}<miso>{output}<eos>

During training, loss is computed on the <miso>{output}<eos> portion only
(user turn tokens are masked with -100 in the dataset).

Usage:
    # Train:
    python -m meow.tokenizer train data/train.jsonl tokenizer.json

    # Load:
    from meow.tokenizer import MeowTokenizer
    tok = MeowTokenizer.from_file("tokenizer.json")
    ids = tok.encode_chat(input_text="hi miso", output_text="hello.")
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing


VOCAB_SIZE = 2048

# Special tokens and their fixed IDs. Order matters — the trainer
# assigns these IDs in order.
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<user>", "<miso>"]
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
USER_ID = 3
MISO_ID = 4


def train_tokenizer(
    dataset_path: Path,
    out_path: Path,
    vocab_size: int = VOCAB_SIZE,
) -> Tokenizer:
    """Train a byte-level BPE tokenizer on the Miso dataset.

    We train on the concatenation of inputs and outputs, since the model
    needs to handle both.
    """
    # Collect training corpus
    lines: list[str] = []
    with dataset_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            lines.append(row["input"])
            lines.append(row["output"])

    # Build the tokenizer
    tok = Tokenizer(BPE(unk_token=None))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
        min_frequency=2,
        show_progress=False,
    )
    tok.train_from_iterator(lines, trainer=trainer)

    # Sanity check: special tokens must have the expected IDs
    for i, t in enumerate(SPECIAL_TOKENS):
        actual = tok.token_to_id(t)
        assert actual == i, f"{t} got id {actual}, expected {i}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))
    print(f"[tokenizer] trained vocab={tok.get_vocab_size()}, saved to {out_path}")
    return tok


class MeowTokenizer:
    """Thin wrapper around a trained HF tokenizer with chat-format helpers."""

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.pad_id = PAD_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID
        self.user_id = USER_ID
        self.miso_id = MISO_ID

    @classmethod
    def from_file(cls, path: str | Path) -> "MeowTokenizer":
        return cls(Tokenizer.from_file(str(path)))

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        """Encode raw text to token IDs (no special tokens added)."""
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def encode_chat(
        self,
        input_text: str,
        output_text: str | None = None,
    ) -> tuple[list[int], int]:
        """Encode a chat turn.

        Returns (ids, output_start) where `output_start` is the index in
        `ids` where the output portion begins (after <miso>). This lets
        the dataset class mask user-turn tokens from the loss.

        Format:
            <bos><user>{input}<miso>{output}<eos>

        If `output_text` is None, returns the prompt only (no output):
            <bos><user>{input}<miso>
        In that case, `output_start` equals len(ids) (nothing to supervise).
        """
        ids: list[int] = [self.bos_id, self.user_id]
        ids.extend(self.encode(input_text))
        ids.append(self.miso_id)
        output_start = len(ids)
        if output_text is not None:
            ids.extend(self.encode(output_text))
            ids.append(self.eos_id)
        return ids, output_start


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train tokenizer from dataset")
    p_train.add_argument("dataset", type=Path, help="Path to train.jsonl")
    p_train.add_argument("out", type=Path, help="Output tokenizer.json path")
    p_train.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)

    p_test = sub.add_parser("test", help="Round-trip a sample through the tokenizer")
    p_test.add_argument("tokenizer", type=Path)
    p_test.add_argument("--text", default="hi miso")

    args = ap.parse_args()

    if args.cmd == "train":
        train_tokenizer(args.dataset, args.out, args.vocab_size)
    elif args.cmd == "test":
        tok = MeowTokenizer.from_file(args.tokenizer)
        ids = tok.encode(args.text)
        print(f"input: {args.text!r}")
        print(f"ids:   {ids}")
        print(f"decode: {tok.decode(ids)!r}")
        chat_ids, start = tok.encode_chat("hi miso", "hello. i was in the sun spot.")
        print(f"chat:   {chat_ids}")
        print(f"start:  {start}")
        print(f"vocab:  {tok.vocab_size}")


if __name__ == "__main__":
    main()
