#!/bin/bash
# 高效评估脚本 - 测量基本性能指标（时延、显存）

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 设置 LIBERO 数据集路径
export LIBERO_DATASET_PATH="/workspace/laiminxin/datasets/libero_rlds"
# tracer 已移动到上层仓库，开启 trace/离线画图需要它在 PYTHONPATH 中
export PYTHONPATH="/workspace/laiminxin/vla-opt/third_party/LightVLA:/workspace/laiminxin/vla-opt:${PYTHONPATH}"

echo "=== LightVLA Evaluation ==="
echo "Start time: $(date)"
echo "Dataset path: ${LIBERO_DATASET_PATH}"
echo ""

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TRACE_OUT_DIR="runs/libero_spatial_eval_${TIMESTAMP}"
echo "Trace output directory: ${TRACE_OUT_DIR}"
echo ""

# 记录初始显存
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i 5

# 使用 time 命令测量总时延
CUDA_VISIBLE_DEVICES=5 \
time python "${SCRIPT_DIR}/experiments/robot/libero/run_libero_eval.py" \
  --pretrained_checkpoint "/workspace/laiminxin/models/LightVLA_logs/openvla-7b-oft-finetuned-libero-spatial+libero_spatial_no_noops+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug2026-01-05 16:32:09.137127--40000_chkpt" \
  --task_suite_name libero_spatial \
  --center_crop true \
  --num_trials_per_task 1 \
  --seed 7 \
  --trace_out_dir "${TRACE_OUT_DIR}" \
  --trace_dump_attn true \
  --trace_attn_layers "1,31"

echo ""
echo "=== Evaluation Complete ==="
echo "End time: $(date)"

python -m tracer.plot_routing_overlays \
    --exp_dir "${TRACE_OUT_DIR}"
