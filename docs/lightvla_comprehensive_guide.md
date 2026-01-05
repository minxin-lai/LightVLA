# LightVLA 综合技术指南：设计、实现与集成

本文档综合了 LightVLA 的设计原理、代码实现细节，以及如何将其集成到 OpenVLA-OFT 的完整指南。文档面向需要**理解设计思想**和**动手修改代码**的开发者。

---

## 目录

1. [一句话总结](#1-一句话总结)
2. [LightVLA 设计原理](#2-lightvla-设计原理)
3. [核心模块：TokenPruner 详解](#3-核心模块tokenpruner-详解)
4. [完整代码路径索引](#4-完整代码路径索引)
5. [集成到 OpenVLA-OFT：需要修改的地方](#5-集成到-openvla-oft需要修改的地方)
6. [训练与推理流程](#6-训练与推理流程)
7. [数值示例：完整数据流](#7-数值示例完整数据流)
8. [常见问题与注意事项](#8-常见问题与注意事项)

---

## 1. 一句话总结

**LightVLA = OpenVLA-OFT + TokenPruner**

> LightVLA 的核心创新是在 LLM 的 Transformer 层之前插入一个 **TokenPruner** 模块，对视觉 patch tokens 进行**动态裁剪**：
> - **训练时**：使用 Straight-Through Estimator (STE) 做"可微聚合"，序列长度不变，学会把信息聚合到少数关键 patch
> - **推理时**：真正物理删除冗余 patch tokens，缩短序列长度，从而加速后续 Transformer 层的计算

---

## 2. LightVLA 设计原理

### 2.1 问题动机

VLA (Vision-Language-Action) 模型的瓶颈在于：
- 视觉 patch 数量大（如 256-512 个）
- Transformer self-attention 复杂度 O(n²)
- 大量视觉 patch 是冗余的（如背景/天空）

### 2.2 核心思想：自适应剪枝

LightVLA 不使用固定比例的 top-k 剪枝，而是让模型**自己学习**哪些 patch 重要：

```
核心问题：如何判断一个 patch 是否可以被删掉？
LightVLA 的答案：让每个 patch "投票"，选出能代表自己的 patch
→ 如果很多 patch 都投给同一个 patch，说明那个 patch 是"信息中心"
→ 投给别人的 patch 就是冗余的，可以删掉
```

### 2.3 投票机制流程

```python
# 输入
patches: [B, P, D]      # P 个视觉 patch (如 256 个)
task:    [B, T, D]      # T 个任务/语言 token

# Step 1: 归一化
patches_n = rms_norm(patches)
task_n = rms_norm(task)

# Step 2: 每个 patch 用 task 作为 context，生成"任务感知的 query"
queries = scaled_dot_product_attention(
    query=patches_n,    # 每个 patch 作为 query
    key=task_n,         # task tokens 作为 key
    value=task_n        # task tokens 作为 value
)  # [B, P, D]

# Step 3: 每个 query 与所有 patches 计算相似度
score = (queries @ patches_n.T) / sqrt(D)  # [B, P, P]

# Step 4: 每行选最大值 = 投票
indices = score.argmax(dim=-1)  # [B, P]

# Step 5: 取并集得到保留的 patch
# 保留数量 = len(unique(indices))
```

**直观理解**：

| Patch 内容 | 投票给谁 | 结果 |
|-----------|---------|------|
| 背景/天空 | 投给其他背景 patch | 大多被删除 |
| 物体中心 | 投给自己 | 保留 |
| 机器人末端 | 投给自己 | 保留 |

### 2.4 训练 vs 推理的区别

| 阶段 | 操作 | 序列长度 | 目的 |
|------|------|----------|------|
| **训练** | `score @ patches` 软聚合 | **不变** (P) | 学会"把信息聚合到少数 patch" |
| **推理** | `patches[mask]` 硬剪枝 | **变短** (M ≤ P) | 真正删除冗余 patch，加速 |

---

## 3. 核心模块：TokenPruner 详解

### 3.1 类定义与位置

**文件**：[prismatic/extern/hf/modeling_prismatic.py](../prismatic/extern/hf/modeling_prismatic.py)  
**行号**：第 49-136 行

```python
class TokenPruner(nn.Module):
    def __init__(self, config, num_patches):
        super().__init__()
        self.num_patches = num_patches           # 视觉 patch 数量
        self.noise_scale = None                  # 训练时的探索噪声
        self.scale_factor = 1 / math.sqrt(config.hidden_size)
```

### 3.2 关键方法

#### 3.2.1 `forward()` - 主入口（第 100-136 行）

```python
def forward(self, tokens, position_ids, attention_mask):
    # 1️⃣ 三段式切分：[BOS] + [patches] + [task]
    cls_token, patches, task = torch.split(tokens, 
        [1, self.num_patches, seq_len - self.num_patches - 1], dim=1)
    
    # 2️⃣ 计算 patch-to-patch 相似度
    score = self.get_score(patches, task)  # (B, P, P)
    
    # 3️⃣ 根据模式处理
    if not self.training:  # 推理：真正删除
        mask = self.score_to_mask(score)
        patches = patches[mask].view(bsz, -1, dim)
    else:  # 训练：软聚合
        indices, patches = self.score_to_indices(score, patches)
    
    # 4️⃣ 重新拼接
    return torch.cat([cls_token, patches, task], dim=1), position_ids, attention_mask
```

#### 3.2.2 `get_score()` - 计算相似度（第 70-78 行）

```python
def get_score(self, patches, prompts):
    patches = self.rms_norm(patches)
    prompts = self.rms_norm(prompts)
    
    # 用 task tokens 指导每个 patch 生成 query
    queries = F.scaled_dot_product_attention(patches, prompts, prompts)
    queries = self.rms_norm(queries)
    
    # patch-to-patch 相似度
    score = queries @ patches.transpose(-2, -1) * self.scale_factor
    return score  # (B, P, P)
```

#### 3.2.3 `score_to_mask()` - 推理时硬剪枝（第 80-90 行）

```python
def score_to_mask(self, score):
    mask = torch.zeros(bsz, self.num_patches, dtype=torch.bool)
    indices = score.argmax(-1)  # 每个 patch 指向谁
    mask[batch_indices, indices] = True  # 被指向的才保留
    return mask
```

#### 3.2.4 `score_to_indices()` - 训练时 STE 软聚合（第 92-98 行）

```python
def score_to_indices(self, score, patches):
    # 添加探索噪声
    if self.noise_scale is not None:
        score = score + torch.rand_like(score) * self.noise_scale
    
    # Straight-Through Estimator (STE)
    hard_score = F.one_hot(score.argmax(dim=-1), num_classes=self.num_patches)
    soft_score = torch.softmax(score, dim=-1)
    score = hard_score + soft_score - soft_score.detach()  # 前向用 hard，反向用 soft
    
    return score.argmax(dim=-1), score @ patches  # 软聚合
```

**STE 技巧解释**：
- 数值上：`hard - soft + soft = hard`（one-hot）
- 梯度上：`∂(hard + soft - soft.detach())/∂input = ∂soft/∂input`
- 效果：前向模拟离散选择，反向保持可导

---

## 4. 完整代码路径索引

### 4.1 核心模块位置

| 模块 | 文件路径 | 行号 | 说明 |
|------|----------|------|------|
| **TokenPruner** | `prismatic/extern/hf/modeling_prismatic.py` | L49-136 | 剪枝核心算法 |
| **PrunedLlamaModel** | `prismatic/extern/hf/modeling_prismatic.py` | L139-260 | 带 pruner 的 Llama |
| **PrunedLlamaForCausalLM** | `prismatic/extern/hf/modeling_prismatic.py` | L263-363 | 带 pruner 的 CausalLM |
| **PrismaticForConditionalGeneration** | `prismatic/extern/hf/modeling_prismatic.py` | L640-997 | VLM 主模型 |
| **OpenVLAForActionPrediction** | `prismatic/extern/hf/modeling_prismatic.py` | L1041-1402 | 动作预测封装 |

### 4.2 训练脚本关键位置

| 功能 | 文件路径 | 行号 | 说明 |
|------|----------|------|------|
| **噪声调度** | `vla-scripts/finetune.py` | L1032 | `pruner.set_noise_scale(...)` |
| **前向传播** | `vla-scripts/finetune.py` | L270+ | `run_forward_pass()` |
| **LoRA 解冻 pruner** | `vla-scripts/finetune.py` | L843-847 | 确保 pruner 参数可训练 |
| **Action Head 初始化** | `vla-scripts/finetune.py` | L841-868 | L1 回归 / Diffusion |

### 4.3 推理脚本关键位置

| 功能 | 文件路径 | 行号 | 说明 |
|------|----------|------|------|
| **predict_action** | `prismatic/extern/hf/modeling_prismatic.py` | L1271-1376 | 推理主入口 |
| **set_num_images_in_input** | `prismatic/extern/hf/modeling_prismatic.py` | L1056-1058 | 同步更新 pruner.num_patches |
| **LIBERO 评测** | `experiments/robot/libero/run_libero_eval.py` | L223, L438 | `vla.eval()` 启用推理态剪枝 |

---

## 5. 集成到 OpenVLA-OFT：需要修改的地方

将 LightVLA 的 TokenPruner 集成到 OpenVLA-OFT 需要修改以下文件：

### 5.1 修改语言模型替换（核心改动）

**文件**：`prismatic/extern/hf/modeling_prismatic.py`

**OpenVLA-OFT 原代码**（约 L352）：
```python
self.language_model = AutoModelForCausalLM.from_config(config.text_config, ...)
```

**LightVLA 改动**（约 L675）：
```python
num_patches = self.vision_backbone.get_num_patches() * self.vision_backbone.get_num_images_in_input()
self.language_model = PrunedLlamaForCausalLM(config.text_config, num_patches)
```

### 5.2 添加 TokenPruner 相关类

需要在 `modeling_prismatic.py` 中添加以下类：

1. **TokenPruner**（约 70 行代码）
   - 位置：文件开头
   - 功能：剪枝核心逻辑

2. **PrunedLlamaModel**（约 120 行代码）
   - 继承自 `LlamaModel`
   - 在 `forward()` 中调用 pruner

3. **PrunedLlamaForCausalLM**（约 100 行代码）
   - 继承自 `LlamaForCausalLM`
   - 使用 `PrunedLlamaModel` 作为内部模型

### 5.3 修改 OpenVLAForActionPrediction

**添加 `set_num_images_in_input()` 方法**：

```python
def set_num_images_in_input(self, num_images_in_input):
    self.vision_backbone.set_num_images_in_input(num_images_in_input)
    # ⚠️ 关键：同步更新 pruner 的 num_patches
    self.language_model.model.pruner.num_patches = (
        self.vision_backbone.get_num_patches() * 
        self.vision_backbone.get_num_images_in_input()
    )
```

### 5.4 修改训练脚本

**文件**：`vla-scripts/finetune.py` 或 `vla-scripts/train.py`

添加噪声调度（在训练循环中）：
```python
# 每个 step 线性衰减噪声
vla.module.language_model.model.pruner.set_noise_scale(1 - step / max_steps)
```

### 5.5 添加必要的 import

```python
from transformers.models.llama.modeling_llama import LlamaModel, LlamaForCausalLM
```

### 5.6 改动清单总结

| 改动点 | 文件 | 改动内容 |
|--------|------|----------|
| 1 | `modeling_prismatic.py` | 添加 `TokenPruner` 类 |
| 2 | `modeling_prismatic.py` | 添加 `PrunedLlamaModel` 类 |
| 3 | `modeling_prismatic.py` | 添加 `PrunedLlamaForCausalLM` 类 |
| 4 | `modeling_prismatic.py` | 修改 `PrismaticForConditionalGeneration.__init__()` 使用新 LLM |
| 5 | `modeling_prismatic.py` | 添加 `set_num_images_in_input()` 方法 |
| 6 | `finetune.py` / `train.py` | 添加噪声调度 `set_noise_scale()` |
| 7 | `finetune.py` / `train.py` | LoRA 时解冻 pruner 参数 |

---

## 6. 训练与推理流程

### 6.1 训练流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                      数据集构造 (RLDSBatchTransform)                │
├─────────────────────────────────────────────────────────────────────┤
│  input_ids: [BOS, prompt, GT_action_tokens, STOP]                   │
│  labels:    [-100, ..., -100, GT_action_tokens, STOP]               │
│  actions:   连续值动作 shape (NUM_ACTIONS_CHUNK, ACTION_DIM)        │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   Forward: 多模态处理                               │
├─────────────────────────────────────────────────────────────────────┤
│  1. input_embeddings = embed(input_ids)                             │
│  2. action_mask = _process_action_masks(labels)                     │
│  3. input_embeddings[action_mask] = 0  ← 清零 action 位置          │
│  4. projected_patch_embeddings = vision_backbone + projector        │
│  5. multimodal_embeddings = [BOS, patches, text]                    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   TokenPruner (训练态)                               │
├─────────────────────────────────────────────────────────────────────┤
│  1. 三段式切分：[BOS] + [patches] + [task]                          │
│  2. score = get_score(patches, task)  # (B, P, P)                   │
│  3. 添加噪声 + STE 软聚合                                            │
│  4. patches = score @ patches  # 序列长度不变                       │
│  5. 重新拼接并送入 Transformer layers                               │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   Loss 计算                                          │
├─────────────────────────────────────────────────────────────────────┤
│  1. actions_hidden = hidden_states[action_mask]                     │
│  2. predicted_actions = action_head(actions_hidden)                 │
│  3. loss = L1Loss(ground_truth_actions, predicted_actions)          │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 推理流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                    输入（来自机器人）                                  │
│  - 图像: pixel_values (1, C, H, W)                                   │
│  - 指令: "pick up the cup"                                           │
│  - [可选] proprio: 当前关节状态                                       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  准备输入：添加 56 个 action 占位符 + STOP token                     │
│  input_ids: [BOS, prompt, placeholder×56, STOP]                      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   TokenPruner (推理态) ⚡ 加速点                     │
├─────────────────────────────────────────────────────────────────────┤
│  1. score = get_score(patches, task)                                 │
│  2. mask = score_to_mask(score)  # 真正删除                          │
│  3. patches = patches[mask]  # 序列变短！                            │
│                                                                      │
│  原始: [BOS] + [512 patches] + [80 text] = 593 tokens               │
│  剪枝: [BOS] + [~150 patches] + [80 text] = 231 tokens              │
│  加速: (593/231)² ≈ 6.6x (self-attention 复杂度)                     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  动作预测 + 反归一化 → 输出给机器人                                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 7. 数值示例：完整数据流

### 7.1 假设条件

```
batch_size = 1
num_patches = 4 (简化)
seq_len = 1 + 4 + 3 = 8 (BOS + 4 patches + 3 task tokens)
dim = 4 (简化)
```

### 7.2 输入数据

```
tokens = [
    [BOS],           # 位置 0
    [P0 - 天空],      # 位置 1
    [P1 - 天空],      # 位置 2 (和 P0 相似)
    [P2 - 机器人],    # 位置 3
    [P3 - 杯子],      # 位置 4
    [T0 - "pick"],   # 位置 5
    [T1 - "up"],     # 位置 6
    [T2 - action],   # 位置 7 (清零的占位符)
]
```

### 7.3 Score 矩阵

```
           被投票者（目标 patch）
           P0    P1    P2    P3
         ┌─────────────────────┐
投票者 P0│ 0.4  0.35  0.2  0.15│  ← P0 投给 P0
       P1│ 0.4  0.35  0.2  0.15│  ← P1 投给 P0 (相似)
       P2│ 0.15 0.2   0.5  0.15│  ← P2 投给 P2 (机器人独特)
       P3│ 0.15 0.2   0.15 0.5 │  ← P3 投给 P3 (杯子独特)
         └─────────────────────┘
```

### 7.4 推理时结果

```
indices = [0, 0, 2, 3]  # P0,P1→P0, P2→P2, P3→P3
unique(indices) = {0, 2, 3}
mask = [True, False, True, True]

原始: [BOS, P0, P1, P2, P3, T0, T1, T2]  长度=8
剪枝: [BOS, P0, P2, P3, T0, T1, T2]      长度=7  ← P1 被删除
```

### 7.5 训练时结果

```
hard_score = [[1,0,0,0],  # P0 指向 P0
              [1,0,0,0],  # P1 指向 P0
              [0,0,1,0],  # P2 指向 P2
              [0,0,0,1]]  # P3 指向 P3

patches = hard_score @ patches
        = [P0,    # 原 P0 不变
           P0,    # 原 P1 变成 P0 (软聚合)
           P2,    # 原 P2 不变
           P3]    # 原 P3 不变

原始: [BOS, P0, P1, P2, P3, T0, T1, T2]  长度=8
聚合: [BOS, P0, P0, P2, P3, T0, T1, T2]  长度=8  ← 长度不变，但 P1 内容变成 P0
```

---

## 8. 常见问题与注意事项

### 8.1 序列结构依赖

TokenPruner **强依赖**固定的序列结构：`[BOS] + [patches] + [task]`

```python
# pruner 切分逻辑
cls_token, patches, task = torch.split(tokens, 
    [1, self.num_patches, seq_len - self.num_patches - 1], dim=1)
```

⚠️ 如果多模态拼接方式改变，必须同步修改 pruner 的切分逻辑。

### 8.2 多图输入时同步 num_patches

使用多图输入时，**必须**通过 `set_num_images_in_input()` 更新 pruner：

```python
# ✅ 正确
vla.set_num_images_in_input(2)

# ❌ 错误（只改了 vision_backbone，没改 pruner）
vla.vision_backbone.set_num_images_in_input(2)
```

### 8.3 训练不加速

训练时 pruner 会做软聚合但**不会缩短序列**，因此：
- 训练 FLOPs 不会因为 pruner 降低
- pruner 的作用是学习"聚合策略"，加速体现在推理时

### 8.4 保留比例是动态的

- **不是**固定的 top-k 或百分比
- 保留数量 = `|unique(argmax(score))|`
- 典型保留比例：20%-40%（取决于图像复杂度）

### 8.5 硬绑定 Llama 架构

LightVLA 的 pruner 集成**硬绑定**了 Llama：

```python
from transformers.models.llama.modeling_llama import LlamaModel, LlamaForCausalLM
```

如需支持其他 LLM（如 Mistral、Gemma），需要创建对应的 `PrunedXxxModel` 类。

### 8.6 noise_scale 调度

训练时需要设置噪声调度：

```python
# 线性衰减：从 1.0 到 0
noise_scale = 1 - current_step / max_steps
vla.module.language_model.model.pruner.set_noise_scale(noise_scale)
```

- 前期：高噪声 → 探索更多选择
- 后期：低噪声 → 收敛到稳定策略

### 8.7 proprio token 不会被剪枝

如果使用 proprio（本体感觉）：
- proprio token 会追加到 patch 序列**之后**
- 它会落在 pruner 的 `task` 段里，不会被剪枝

---

## 附录：代码对照表

| OpenVLA-OFT | LightVLA | 说明 |
|-------------|----------|------|
| `AutoModelForCausalLM.from_config(...)` | `PrunedLlamaForCausalLM(config, num_patches)` | LLM 替换 |
| 无 | `TokenPruner` | 新增剪枝模块 |
| 无 | `PrunedLlamaModel` | 新增带 pruner 的 Llama |
| 无 | `set_num_images_in_input()` 同步 pruner | 新增同步逻辑 |
| 无 | `set_noise_scale()` | 新增噪声调度 |

---

## 参考文件

- **核心实现**：[prismatic/extern/hf/modeling_prismatic.py](../prismatic/extern/hf/modeling_prismatic.py)
- **训练脚本**：[vla-scripts/finetune.py](../vla-scripts/finetune.py)、[vla-scripts/train.py](../vla-scripts/train.py)
- **评测脚本**：[experiments/robot/libero/run_libero_eval.py](../experiments/robot/libero/run_libero_eval.py)
- **Action Heads**：[prismatic/models/action_heads.py](../prismatic/models/action_heads.py)
