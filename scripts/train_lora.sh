#!/usr/bin/env bash
#
# W5 — LoRA fine-tune the local Gemma-4 E2B student on the envelope->script task.
#
# Activates the project venv, then runs mlx_vlm.lora in SFT mode (text-only LoRA;
# the model is multimodal but we train the language path only). Trains on the
# assistant completions of data/processed/train.jsonl and writes a LoRA adapter to
# --output-path. Long-lived — run it in a terminal and let it finish.
#
# Usage:
#   bash scripts/train_lora.sh                      # defaults below
#   EPOCHS=2 LORA_RANK=16 bash scripts/train_lora.sh   # override any knob via env vars
#
set -uo pipefail
cd "$(dirname "$0")/.."

# --- activate the project venv ---
if [ ! -f ".venv/bin/activate" ]; then
  echo "No .venv found at $(pwd)/.venv — create it first with 'uv sync'." >&2
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate
echo "venv active: $(command -v python)"

# --- config (override any of these via environment variables) ---
MODEL_PATH="${MODEL_PATH:-models/gemma-4-e2b-it-bf16}"
DATASET="${DATASET:-data/processed}"
OUTPUT_PATH="${OUTPUT_PATH:-models/lora_gemma4e2b_scriptgen}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
EPOCHS="${EPOCHS:-3}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-3072}"

echo "LoRA fine-tune: $MODEL_PATH -> $OUTPUT_PATH"
echo "  rank=$LORA_RANK alpha=$LORA_ALPHA lr=$LEARNING_RATE epochs=$EPOCHS max_seq=$MAX_SEQ_LENGTH"

python -m mlx_vlm.lora \
    --model-path "$MODEL_PATH" \
    --dataset "$DATASET" \
    --split train \
    --train-mode sft \
    --train-on-completions \
    --lora-rank "$LORA_RANK" \
    --lora-alpha "$LORA_ALPHA" \
    --lora-dropout "$LORA_DROPOUT" \
    --learning-rate "$LEARNING_RATE" \
    --batch-size "$BATCH_SIZE" \
    --gradient-accumulation-steps "$GRAD_ACCUM" \
    --epochs "$EPOCHS" \
    --max-seq-length "$MAX_SEQ_LENGTH" \
    --steps-per-report 20 \
    --steps-per-eval 200 \
    --steps-per-save 200 \
    --output-path "$OUTPUT_PATH"

echo ""
echo "Done. Adapter -> $OUTPUT_PATH/adapters.safetensors"
echo "Next: eval the fine-tuned student (re-run the new-pipeline baseline with the adapter)."
