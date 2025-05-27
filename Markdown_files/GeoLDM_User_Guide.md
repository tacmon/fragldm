# GeoLDM项目使用指南

## 一、项目概述

GeoLDM是一个基于等变潜在扩散模型的3D分子生成框架，旨在生成具有物理合理性的分子结构。本项目具有以下特点：

- **等变性保持**：生成的分子在旋转、平移变换下保持不变
- **两阶段架构**：等变变分自编码器(EVA) + 潜在扩散模型
- **条件生成**：支持基于分子属性和部分结构的条件生成
- **高质量采样**：生成的分子结构具有物理化学合理性

## 二、环境配置

### 基本依赖

```bash
pip install torch rdkit numpy wandb
```

### 目录结构

```
GeoLDM-main/
├── qm9/                   # QM9数据集相关代码
├── equivariant_diffusion/ # 等变扩散模型核心代码
├── egnn/                  # 等变图神经网络实现
├── configs/               # 配置文件
├── main_qm9.py            # 主程序入口
├── train_test.py          # 训练测试逻辑
├── utils.py               # 工具函数
└── mask_utils.py          # 掩码生成工具
```

## 三、基本使用

### 3.1 分子生成

#### 无条件生成

生成分子的基本命令：

```bash
python main_qm9.py --exp_name uncond_gen \
                  --n_epochs 3000 \
                  --batch_size 256 \
                  --nf 256 \
                  --n_layers 9 \
                  --diffusion_steps 1000 \
                  --train_diffusion \
                  --trainable_ae
```

#### 属性条件生成

基于分子属性(如偶极矩、HOMO-LUMO能隙等)的条件生成：

```bash
python main_qm9.py --exp_name cond_alpha \
                  --conditioning alpha \  # 可选: homo, lumo, alpha, gap, mu, Cv
                  --n_epochs 3000 \
                  --batch_size 256 \
                  --nf 256 \
                  --n_layers 9 \
                  --train_diffusion
```

支持的条件属性：
- `homo`: 最高占有分子轨道能级
- `lumo`: 最低未占分子轨道能级
- `alpha`: 极化率
- `gap`: HOMO-LUMO能隙
- `mu`: 偶极矩
- `Cv`: 热容

可以同时指定多个条件：

```bash
python main_qm9.py --conditioning homo lumo gap
```

#### 部分结构条件生成

基于部分分子结构进行条件生成：

```bash
python main_qm9.py --exp_name struct_cond \
                  --partial_conditioning \
                  --noise_ratio 0.5 \
                  --mask_strategy connected \
                  --condition_encoder mlp \
                  --condition_encoder_dim 128
```

掩码策略选项：
- `random`: 随机选择原子
- `connected`: 生成连接的子图
- `central`: 选择分子中心区域
- `peripheral`: 选择分子外围区域

条件编码器选项：
- `mlp`: 多层感知机编码器
- `cnn`: 卷积神经网络编码器
- `gnn`: 图神经网络编码器

## 四、训练选项

### 4.1 训练模式

#### 单阶段训练（仅自编码器）

```bash
python main_qm9.py --exp_name vae_only \
                  --n_epochs 1000 \
                  --batch_size 256 \
                  --nf 256 \
                  --n_layers 9 \
                  --latent_nf 4 \
                  --kl_weight 0.01
```

#### 两阶段训练（自编码器+扩散）

```bash
python main_qm9.py --exp_name two_stage \
                  --n_epochs 3000 \
                  --batch_size 256 \
                  --nf 256 \
                  --n_layers 9 \
                  --train_diffusion \
                  --trainable_ae \
                  --latent_nf 1
```

#### 从预训练模型继续训练

```bash
python main_qm9.py --resume outputs/model_name \
                  --start_epoch 240 \
                  --n_epochs 3000
```

### 4.2 优化选项

```bash
python main_qm9.py --lr 1e-4 \
                  --clip_grad True \
                  --ema_decay 0.9999 \
                  --norm_values [1,4,10]
```

### 4.3 扩散参数

```bash
python main_qm9.py --diffusion_steps 1000 \
                  --diffusion_noise_schedule polynomial_2 \
                  --diffusion_noise_precision 1e-5 \
                  --diffusion_loss_type l2
```

噪声调度选项：
- `polynomial_2`: 二次多项式调度
- `cosine`: 余弦调度
- `learned`: 学习噪声调度

## 五、高级功能

### 5.1 实验追踪

集成Weights & Biases进行实验追踪：

```bash
python main_qm9.py --exp_name tracked_exp \
                  --wandb_usr your_username
```

禁用wandb：

```bash
python main_qm9.py --no_wandb
```

### 5.2 模型检查点

模型会按照以下规则自动保存：
- 每当验证集损失改善时
- 可通过`--test_epochs`参数控制验证频率

### 5.3 EMA模型更新

```bash
python main_qm9.py --ema_decay 0.9999  # 设置为0以禁用EMA
```

### 5.4 数据并行训练

在多GPU环境下启用数据并行：

```bash
python main_qm9.py --dp True
```

### 5.5 基于分子骨架的条件生成

GeoLDM支持从已有分子片段生成完整分子，这是药物设计和分子优化的重要功能。

#### 5.5.1 训练骨架条件生成模型

使用`scaffold_based_training.sh`脚本训练专门用于骨架条件生成的模型：

```bash
./scaffold_based_training.sh
```

这个脚本配置了适当的参数，使用`central`掩码策略和较高的噪声比例(0.7)，意味着只保留核心骨架作为条件，同时使用图神经网络(GNN)编码器来更好地理解骨架的结构特征。

#### 5.5.2 从分子骨架生成完整分子

训练完成后，使用`generate_from_scaffold.sh`脚本从分子骨架生成完整分子：

```bash
./generate_from_scaffold.sh
```

该脚本支持多种输入格式，包括:
- XYZ文件: 包含原子三维坐标的标准格式
- MOL/SDF文件: 化学结构文件格式，包含连接信息
- SMILES字符串: 可转换为3D结构

使用方法：

1. 准备分子骨架文件（例如`input_scaffolds.xyz`）
2. 修改脚本中的参数：
   ```bash
   MODEL_PATH="outputs/scaffold_based"  # 预训练模型路径
   INPUT_SCAFFOLD="input_scaffolds.xyz" # 输入的分子片段文件
   OUTPUT_DIR="generated_molecules"     # 输出目录
   NUM_SAMPLES=10                       # 每个输入生成的分子数量
   TEMPERATURE=1.0                      # 采样温度，控制多样性
   ```
3. 运行脚本生成分子：
   ```bash
   ./generate_from_scaffold.sh
   ```

输出文件包括：
- SDF文件：包含3D结构的分子文件
- MOL文件：另一种分子格式
- SMI文件：SMILES字符串表示
- PNG文件：分子2D结构图像
- summary.csv：包含生成分子的属性汇总

#### 5.5.3 高级参数调整

可通过修改`generate_from_scaffold.sh`中的参数来控制生成过程：

```bash
--temperature 0.8        # 降低温度产生更确定性的结果
--noise_ratio 0.5        # 降低噪声比例使生成的分子更接近输入骨架
--mask_strategy central  # 选择骨架策略(central/connected/random)
--num_samples 20         # 增加生成样本数
--diffusion_steps 2000   # 增加扩散步数提高生成质量
```

## 六、常用场景示例

### 场景1：药物分子设计

生成具有特定偶极矩和极化率的药物候选分子：

```bash
python main_qm9.py --exp_name drug_design \
                  --conditioning mu alpha \
                  --n_epochs 3000 \
                  --batch_size 256 \
                  --nf 256 \
                  --train_diffusion
```

### 场景2：分子骨架扩展

基于给定的分子骨架生成完整分子：

```bash
python main_qm9.py --exp_name scaffold_based \
                  --partial_conditioning \
                  --noise_ratio 0.7 \
                  --mask_strategy central \
                  --condition_encoder gnn
```

### 场景3：高通量虚拟筛选

生成大量候选分子并筛选特定性质：

```bash
python main_qm9.py --exp_name high_throughput \
                  --n_stability_samples 1000 \
                  --train_diffusion \
                  --diffusion_steps 1000
```

## 七、参数完整列表

### 模型结构参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--model` | 模型类型 | `egnn_dynamics` |
| `--n_layers` | 网络层数 | `6` |
| `--nf` | 特征维度 | `128` |
| `--latent_nf` | 潜在空间维度 | `4` |
| `--attention` | 使用注意力机制 | `True` |
| `--tanh` | 在坐标MLP中使用tanh激活 | `True` |

### 扩散模型参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--train_diffusion` | 训练扩散模型 | `False` |
| `--diffusion_steps` | 扩散步数 | `500` |
| `--diffusion_noise_schedule` | 噪声调度 | `polynomial_2` |
| `--diffusion_noise_precision` | 噪声精度 | `1e-5` |
| `--diffusion_loss_type` | 损失类型 | `l2` |

### 条件生成参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--conditioning` | 条件属性列表 | `[]` |
| `--partial_conditioning` | 启用部分结构条件 | `False` |
| `--noise_ratio` | 噪声比例 | `0.5` |
| `--mask_strategy` | 掩码策略 | `random` |
| `--condition_encoder` | 条件编码器类型 | `None` |
| `--condition_encoder_dim` | 条件编码器维度 | `64` |

### 训练参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--n_epochs` | 训练轮数 | `200` |
| `--batch_size` | 批次大小 | `128` |
| `--lr` | 学习率 | `2e-4` |
| `--clip_grad` | 梯度裁剪 | `True` |
| `--ema_decay` | EMA衰减率 | `0.999` |
| `--kl_weight` | KL散度权重 | `0.01` |
| `--trainable_ae` | 是否训练自编码器 | `False` |

### 其他参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--save_model` | 是否保存模型 | `True` |
| `--test_epochs` | 测试间隔轮数 | `10` |
| `--resume` | 恢复训练路径 | `None` |
| `--start_epoch` | 起始轮数 | `0` |
| `--no_wandb` | 禁用wandb | `False` |
| `--dp` | 数据并行 | `True` |

## 八、常见问题

**Q: 如何可视化生成的分子?**  
A: 生成的分子会保存为.xyz或.mol格式，可以使用PyMol、VMD或RDKit等工具可视化。

**Q: 训练过程中内存不足怎么办?**  
A: 尝试减小批次大小(`--batch_size`)或模型大小(`--nf`, `--n_layers`)。

**Q: 如何提高生成的分子质量?**  
A: 增加扩散步数(`--diffusion_steps`)、使用EMA(`--ema_decay 0.9999`)和增加训练轮数。

**Q: 生成的分子不符合化学规则怎么办?**  
A: 尝试使用更大的模型、更长的训练时间，或者调整归一化因子(`--normalize_factors`)。

**Q: 使用骨架生成时分子与骨架不兼容怎么办?**  
A: 可以尝试降低噪声比例(`--noise_ratio`)，使模型更尊重原始骨架结构；或者增加结构一致性检查步骤。

**Q: 生成分子时出现"无效分子"错误怎么办？**  
A: 这通常是由于生成的分子违反了化学规则。可以尝试：
   1. 增加扩散步数(`--diffusion_steps`)
   2. 降低采样温度(`--temperature`)
   3. 确保训练模型时使用了`--normalize_factors [1,4,10]`参数

## 九、高级应用

### 9.1 分子属性优化

通过迭代条件生成优化特定属性：

```bash
python main_qm9.py --conditioning gap \
                  --n_epochs 3000 \
                  --batch_size 256
```

### 9.2 骨架约束生成

固定特定骨架生成新分子：

```bash
python main_qm9.py --partial_conditioning \
                  --noise_ratio 0.3 \
                  --mask_strategy connected
```

### 9.3 多条件联合优化

同时考虑多种属性约束：

```bash
python main_qm9.py --conditioning homo lumo alpha \
                  --partial_conditioning \
                  --noise_ratio 0.4
```

希望本指南能帮助您充分利用GeoLDM模型进行各种分子生成任务！ 