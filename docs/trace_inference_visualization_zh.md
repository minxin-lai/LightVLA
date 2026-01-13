# 推理阶段 Trace：Dump / 可视化 / Token 日志（LightVLA）

本文档说明如何在 `experiments/robot/libero/run_libero_eval.py` 评测时开启 trace，将 **剪枝(routing/pruning) 与 attention 证据**落盘，并离线生成 **叠图**与**逐次推理 token 日志**。

本项目的 tracer 已抽成可复用模块，位于父仓库：`/workspace/laiminxin/vla-opt/tracer/`。

## 1. Tracer 接入（跨项目复用）

### 1.1 环境与 PYTHONPATH

`experiments/robot/libero/run_libero_eval.py` 会 `import tracer.*`，因此需要同时把 **LightVLA 工程根目录** 与 **父仓库根目录** 加入 `PYTHONPATH`：

```bash
export PYTHONPATH=/workspace/laiminxin/vla-opt/third_party/LightVLA:/workspace/laiminxin/vla-opt:$PYTHONPATH
```

### 1.2 产出目录结构（run root）

启用 `--trace_out_dir runs/{exp_name}` 后，会生成如下结构（run root）：

```
runs/{exp_name}/
├── dumps/                       # 每次 query 一条 .pt
│   └── taskXXX/epYYY/*.pt
├── images/                      # policy 输入图（用于叠图）
│   └── taskXXX/epYYY/*__img{idx}.png
├── plots/                       # 离线叠图输出
│   ├── score_heatmap/
│   ├── mask_heatmap/
│   └── attn_task_to_vis/
└── report/                      # run-level 报告/日志
    ├── capabilities.json
    ├── tokens.jsonl
    └── tokens_summary.json
```

### 1.3 离线工具入口（统一用 `python -m tracer.*`）

- 叠图：`python -m tracer.plot_routing_overlays --exp_dir runs/{exp_name}`
- Token 日志：`python -m tracer.tokens_log --run_root runs/{exp_name}`
- run-level 报告：`python -m tracer.run_report --run_root runs/{exp_name}`（通常由 eval 自动触发）

## 2. LightVLA Adapter（最小侵入的模型绑定）

LightVLA 的“模型特有逻辑”集中在 adapter 里，避免把 trace 逻辑侵入到 eval 脚本：

- 位置：`/workspace/laiminxin/vla-opt/tracer/adapters/lightvla.py`
- 入口类：`LightVLATracer` + `LightVLATraceConfig`

### 2.1 Adapter 负责什么

- **Routing / pruning 证据**：通过运行时包装 `language_model.model.pruner`（`TokenPruner`）捕获 `keep_mask/indices/keep_counts` 与轻量统计（例如 `kept_ratio`、`seq_lens`）。
- **Attention 证据（可选）**：对指定层的 `self_attn` 注册 forward hook，并把注意力降维成 `task → vision` 的向量（每层一条），写入 `attn.task_to_vis`。
- **Schema 归一化**：把 LightVLA 的原始字段归一成 tracer 的 canonical dump schema（下游画图/统计不需要了解模型细节）。

### 2.2 Eval 如何调用 Adapter（你只需要知道这个 wiring）

在 `experiments/robot/libero/run_libero_eval.py` 中：

- `trace_out_dir != ""` 时创建 `TraceWriter`（负责落盘到 `runs/{exp_name}/`）
- 每次 query 进入 `with LightVLATracer(...):`，执行一次 `predict_action(...)`
- 由 `LightVLATracer.build_dump(...)` 构建 dump dict，再交给 `TraceWriter.write_dump(...)`

这样 eval 入口只做 wiring（少侵入），核心采集逻辑全部在 adapter + tracer 工具链里复用。

## 3. 快速开始（推荐）

仓库提供了一个便捷脚本（含 trace 参数）：

```bash
bash run_libero_eval.sh
```

脚本默认会：
- 运行一次 `libero_spatial`（每个 task 1 个 trial）
- 生成 `runs/libero_spatial_eval_{timestamp}/...`
- 评测结束后离线绘制 overlay：`python -m tracer.plot_routing_overlays`

## 4. 在 `run_libero_eval.py` 里开启 trace（手动）

核心参数在 `GenerateConfig` 中：

- `--trace_out_dir runs/{exp_name}`：开启 trace 并指定输出目录
- `--trace_max_dumps_per_run N`：最多写 N 个 dump（防止爆磁盘；默认 `1`，“一般 dump 一次就够了”）
- `--trace_save_policy_images true|false`：是否保存 policy 输入图（用于叠图）
- `--trace_dump_routing true|false`：是否写 routing/pruning 证据（建议始终打开）
- `--trace_dump_attn true|false`：是否写 attention 证据（可选，体积更大）
- `--trace_attn_layers "1,31"`：attention 抓取层号列表（逗号分隔）

示例（手动运行）：

```bash
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint /path/to/ckpt \
  --task_suite_name libero_spatial \
  --num_trials_per_task 1 \
  --trace_out_dir runs/my_trace_run \
  --trace_dump_attn true \
  --trace_attn_layers "1,31"
```

## 5. dump 文件内容（每次推理一份 .pt）

每个 `dumps/**/*.pt` 是一个 `torch.save(dict)`，核心字段：

- `schema_version`：当前 schema 版本（例如 `v1`）
- `meta`：样本标识（task/episode/step/sample_id）、instruction、checkpoint 等
- `align`：对齐信息（patch 网格、num_images_in_input、image_paths 等）
- `routing`：剪枝证据（`keep_mask/indices/keep_counts/router_scores/kept_ratio`）
- `attn`：attention 证据（`task_to_vis`，按 layer 存）
- `perf`：基础耗时（`forward_latency_s`）
- `capabilities`：本条 dump 包含哪些能力字段（用于下游判断）

### 5.1 剪枝前/后 token 数量（每次推理）

写在 `routing.seq_lens`：
- `pre_seq_len`：进入 `TokenPruner.forward` 前的序列长度
- `post_seq_len`：剪枝后输出的序列长度
- `pre_task_len`：剪枝前 task token 长度（估算：`seq_len - num_patches - 1`）
- `post_task_len`：剪枝后 task token 长度（估算：`post_seq_len - 1 - num_kept`）

用途：
- 逐次推理确认“只剪视觉 token，不影响 task token”
- 结合 `num_kept/kept_ratio` 做压缩率统计

### 5.2 打分过程统计（轻量版）

写在 `routing.score_stats`（默认只存轻量统计，避免写 `score[B,P,P]` 大矩阵）：

- `max_per_query`：`score.max(-1)`，形状 `[B,P]`
- `entropy_per_query`：`softmax(score)` 的 entropy，形状 `[B,P]`

用途：
- 判断 router 打分是否“尖锐/确定”（entropy 越低越尖锐）
- 与 `kept_ratio`、成功率做关联分析

如需存原始 `score[B,P,P]`，目前只在 `python -m tracer.tools.dump_routing_from_predict_action` 支持 `--store_raw_score`（体积很大，不建议在 LIBERO eval 默认开启）。

## 6. 离线可视化（叠图）

从 dumps + images 生成 overlay：

```bash
python -m tracer.plot_routing_overlays --exp_dir runs/{exp_name}
```

输出：
- `plots/score_heatmap/...png`：`router_scores` 叠图
- `plots/mask_heatmap/...png`：`keep_mask` 叠图（剪掉区域会变暗）
- `plots/attn_task_to_vis/layerXX/...png`：`attn.task_to_vis` 叠图（如果 dump 含 attention）

## 7. 逐次推理 Token 日志（run-level）

生成逐推理日志（每个 dump 一行）：

```bash
python -m tracer.tokens_log --run_root runs/{exp_name}
```

输出：
- `report/tokens.jsonl`：逐推理一行，含 `pre/post_seq_len`、`delta_seq_len`、`num_kept/kept_ratio`、耗时等
- `report/tokens_summary.json`：对上述字段做全 run 统计 + coverage

另外：在 `eval_libero()` 结束时，如果 `--trace_out_dir` 非空，会自动写：
- `report/capabilities.json`（来自 `python -m tracer.run_report`）
- `report/tokens.jsonl` / `report/tokens_summary.json`（来自 `python -m tracer.tokens_log`）

## 8. 不依赖 LIBERO 的最小 trace（单次 predict_action）

如果你只想验证 tracing/dump/overlay 管道，不想启动环境：

```bash
python -m tracer.tools.dump_routing_from_predict_action \
  --checkpoint /path/to/ckpt \
  --exp_name my_min_trace \
  --device cuda \
  --dtype bfloat16
```

产物位于 `runs/my_min_trace/`。
