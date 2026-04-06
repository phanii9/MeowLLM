#!/usr/bin/env bash
# upload_to_hf.sh — one-shot upload of MeowLLM artifacts to Hugging Face.
#
# Usage:
#   export HF_TOKEN=your_token_here
#   export HF_USERNAME=hunt3rx99
#   bash scripts/upload_to_hf.sh
#
# This script uploads:
#   - Model checkpoint to hf.co/$HF_USERNAME/meowllm (model repo)
#   - Training dataset to hf.co/$HF_USERNAME/meowllm-miso (dataset repo)
#
# Prerequisites:
#   pip install huggingface_hub
#   checkpoints/best.pt must exist
#   data/train.jsonl, data/val.jsonl, data/tokenizer.json must exist

set -euo pipefail

: "${HF_USERNAME:?Set HF_USERNAME environment variable}"
: "${HF_TOKEN:?Set HF_TOKEN environment variable}"

MODEL_REPO="$HF_USERNAME/meowllm"
DATASET_REPO="$HF_USERNAME/meowllm-miso"

echo "=========================================="
echo "MeowLLM Hugging Face Upload"
echo "=========================================="
echo "model repo:   $MODEL_REPO"
echo "dataset repo: $DATASET_REPO"
echo

# Check required files exist
for f in checkpoints/best.pt data/tokenizer.json data/train.jsonl data/val.jsonl docs/model_card.md docs/dataset_card.md; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: missing required file: $f"
        exit 1
    fi
done

echo "[1/4] Logging in to HF..."
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

echo
echo "[2/4] Creating model repo (if it doesn't exist)..."
huggingface-cli repo create meowllm --type model || true

echo
echo "[3/4] Uploading model artifacts..."
# Copy model_card.md to README.md in the model repo
cp docs/model_card.md /tmp/model_README.md
huggingface-cli upload "$MODEL_REPO" checkpoints/best.pt best.pt
huggingface-cli upload "$MODEL_REPO" data/tokenizer.json tokenizer.json
huggingface-cli upload "$MODEL_REPO" /tmp/model_README.md README.md

echo
echo "[4/4] Creating and uploading dataset..."
huggingface-cli repo create meowllm-miso --type dataset || true
cp docs/dataset_card.md /tmp/dataset_README.md
huggingface-cli upload "$DATASET_REPO" data/train.jsonl train.jsonl --repo-type dataset
huggingface-cli upload "$DATASET_REPO" data/val.jsonl val.jsonl --repo-type dataset
huggingface-cli upload "$DATASET_REPO" /tmp/dataset_README.md README.md --repo-type dataset

echo
echo "=========================================="
echo "done. check:"
echo "  https://huggingface.co/$MODEL_REPO"
echo "  https://huggingface.co/datasets/$DATASET_REPO"
echo "=========================================="
