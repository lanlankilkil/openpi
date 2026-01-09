# 价值函数模型

## 概述

本文档描述了结合 SigLIP (400M) 和 Gemma (270M) 的价值函数 VLM（视觉语言模型）实现，用于强化学习中的价值估计。采用 C51 分布式强化学习的方法进行价值预测。

## 架构

```
                    +------------------+
                    |     输入图像      |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    |  SigLIP So400m   |
                    |  (4亿参数)        |
                    |  width=1152      |
                    +--------+---------+
                             |
                             | 投影到 640 维
                             v
+------------------+   +------------------+
|   文本 Token     |   |   图像 Token     |
+--------+---------+   +--------+---------+
         |                      |
         +----------+-----------+
                    | 序列维度拼接
                    v
           +------------------+
           |   Gemma 270M     |
           |   width=640      |
           |   depth=18       |
           +--------+---------+
                    |
                    | 最后一个 token
                    v
           +------------------+
           |   Value Head     |
           | Linear(640, 201) |
           +--------+---------+
                    |
                    v
           +------------------+
           |  价值分布 (201)   |
           |  Softmax → 期望值 |
           +------------------+
```

## C51 分布式价值函数

### 核心思想

预测价值的**分布**，然后通过期望得到价值估计。

### 支架设置

| 参数 | 值 |
|------|-----|
| NUM_ATOMS | 201 |
| V_MIN | -1.0 |
| V_MAX | 0.0 |
| DELTA_Z | 0.005 |
| SUPPORTS | [-1.0, -0.995, -0.99, ..., 0.0] |

### Two-hot 编码

由于真实目标值往往不会精准落在某个支架中心，采用线性插值投影：

```
目标值 y = -0.503

找到左右支架:
  b_left = 99  (对应 z = -0.505)
  b_right = 100 (对应 z = -0.500)

计算权重:
  weight_left = 0.4
  weight_right = 0.6

生成 201 维目标分布:
  P_target[99] = 0.4
  P_target[100] = 0.6
  其余位置 = 0
```

### 损失函数

交叉熵损失：

$$Loss = -\sum_{i=0}^{200} P_{target}^{(i)} \cdot \log(\text{Softmax}(L_{pred})^{(i)})$$

### 价值计算

期望值：

$$Value = \sum_{i=0}^{200} \text{Softmax}(L_{pred})^{(i)} \cdot z_i$$

## 添加/修改的文件

### 新增文件

1. **`src/openpi/models/value_model.py`**
   - `NUM_ATOMS, V_MIN, V_MAX, DELTA_Z, SUPPORTS`：C51 常量
   - `target_to_twohot()`：目标值转 two-hot 分布
   - `dist_to_value()`：分布 logits 转期望值
   - `ValueModel` 类：模型主体实现
   - `embed_tokens()`：将图像和文本编码为 token 序列
   - `compute_value()`：计算期望价值
   - `compute_loss()`：计算交叉熵损失

2. **`src/openpi/models/value_model_config.py`**
   - `ValueModelConfig`：配置类

### 修改的文件

1. **`src/openpi/models/gemma.py`**
   - 添加 `gemma_270m` 配置：
     ```python
     Config(
         width=640,
         depth=18,
         mlp_dim=2048,
         num_heads=4,
         num_kv_heads=1,
         head_dim=256,
     )
     ```

## 模型参数

### Gemma 270M
| 参数 | 值 |
|------|-----|
| Width | 640 |
| Depth | 18 |
| MLP Dim | 2048 |
| Num Heads | 4 |
| Head Dim | 256 |
| 总参数量 | ~2.7亿 |

### SigLIP So400m/14
| 参数 | 值 |
|------|-----|
| Width | 1152 |
| Depth | 27 |
| Patch Size | 14x14 |
| 总参数量 | ~4亿 |

### Value Head
| 参数 | 值 |
|------|-----|
| 输入维度 | 640 |
| 输出维度 | 201 (NUM_ATOMS) |

## 使用方法

```python
from openpi.models.value_model import ValueModel
from openpi.models.value_model_config import ValueModelConfig
import jax

# 创建配置
config = ValueModelConfig(
    dtype="bfloat16",
    gemma_variant="gemma_270m",
    siglip_variant="So400m/14",
)

# 初始化模型
model = config.create(jax.random.key(0))

# 计算价值（返回期望值）
value = model.compute_value(rng, observation, train=False)
# value shape: [batch]，范围 [-1, 0]

# 获取完整分布 logits
logits = model(observation, train=False)
# logits shape: [batch, 201]

# 计算损失（target_values 需在 [-1, 0] 范围内）
loss = model.compute_loss(rng, observation, target_values, train=True)
```
训练
  步骤 1：添加价值标签
  python scripts/add_value_labels.py --data_dir /home/wang/data/train_dataset/lerobot_datasets/piper_plug_libero

  步骤 2：训练模型
  python scripts/train_value.py \
      --data_dir /home/wang/data/train_dataset/lerobot_datasets/piper_plug_libero \
      --checkpoint_dir ./checkpoints/value_model \
      --batch_size 32 \
      --num_train_steps 10000 \
      --load_pretrained
