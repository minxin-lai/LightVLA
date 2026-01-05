# LightVLA Training & Evaluation (LIBERO-Spatial)

## Environment
```bash
conda activate openvla
export PYTHONPATH="${PWD}:${PWD}/../..:${PYTHONPATH}"
```

## 1. Training (Offline)
```bash
./train_libero_spatial_offline.sh
```

## 2. Evaluation

### LightVLA (Pruned)
```bash
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint TTJiang/LightVLA-libero-spatial \
  --task_suite_name libero_spatial \
  --num_trials_per_task 50
```

### OpenVLA-OFT (Baseline)
```bash
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 50
```
