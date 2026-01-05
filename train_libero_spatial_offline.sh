#!/bin/bash
# LightVLA Training Script for LIBERO-Spatial (Offline Mode)
# This script uses locally cached models to avoid network issues
#
# Quick Test (1000 steps):
#   Edit --max_steps to 1000 and --save_freq to 500 before running
#
# Full Training (40K steps):
#   Keep default values (--max_steps 40005, --save_freq 10000)
#
# Monitoring Output:
#   runs/{run_id}/pruning_logs/{run_id}_pruning.jsonl  (event log)
#   runs/{run_id}/pruning_logs/{run_id}_meta.json      (metadata)
#   runs/{run_id}/pruning_logs/{run_id}_summary.json   (final summary)

# Ensure this repo checkout's `prismatic/` is imported (avoid site-packages shadowing).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Also include the outer vla-opt repo root so we can import `utils.monitoring`.
VLA_OPT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${VLA_OPT_ROOT}:${PYTHONPATH}"

# Force offline mode to use local cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Optional: Set cache directory explicitly
export HF_HOME="${HOME}/.cache/huggingface"

echo "Running in offline mode with local cache at: $HF_HOME"

if [[ "${LIGHTVLA_VALIDATE_IMPORT_ONLY:-0}" == "1" ]]; then
  python - <<'PY'
import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
import inspect
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from utils.monitoring import Monitor

print("OpenVLAForActionPrediction file:", inspect.getsourcefile(OpenVLAForActionPrediction))
print("has set_num_images_in_input:", hasattr(OpenVLAForActionPrediction, "set_num_images_in_input"))
print("Monitor import OK:", Monitor)
PY
  exit 0
fi

echo "Starting training..."

# Create log directory and set log file path
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
echo "Logs will be saved to: ${LOG_FILE}"

CUDA_VISIBLE_DEVICES=7 \
torchrun \
  --standalone \
  --nnodes 1 \
  --nproc-per-node 1 \
  vla-scripts/finetune.py \
  --vla_path moojink/openvla-7b-oft-finetuned-libero-spatial \
  --data_root_dir libero_rlds \
  --dataset_name libero_spatial_no_noops \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 30000 \
  --max_steps 40005 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --enable_pruning_monitor True \
  --pruning_monitor_interval 100 \
  2>&1 | tee "${LOG_FILE}"
