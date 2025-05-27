#!/bin/bash

# 设置可调整的参数
EXP_NAME="diffusion_scaffold_alpha_bs64_nf192_nlayers9_lr1e-4"
N_EPOCHS=3000
BATCH_SIZE=64
TEST_EPOCHS=20
NOISE_RATIO=0.8
MASK_STRATEGY="connected"  # 可选: "random", "connected", "central", "peripheral"
NUM_WORKERS=8  # 数据加载并行进程数

# 设置预训练VAE路径（需要根据实际路径修改）
VAE_PATH="./outputs/vae_alpha_bs64_nf192_nlayers9_lr1e-4"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}启动GeoLDM模型训练 - 扩散模型阶段${NC}"
echo -e "${YELLOW}实验名称:${NC} $EXP_NAME"
echo -e "${YELLOW}条件类型:${NC} alpha + 骨架约束"
echo -e "${YELLOW}掩码策略:${NC} $MASK_STRATEGY"
echo -e "${YELLOW}噪声比例:${NC} $NOISE_RATIO"
echo -e "${YELLOW}训练轮数:${NC} $N_EPOCHS"
echo -e "${YELLOW}数据加载进程数:${NC} $NUM_WORKERS"
echo -e "${YELLOW}使用预训练VAE:${NC} $VAE_PATH"

# 设置PYTHONPATH
export PYTHONPATH=$(pwd)

# 运行训练命令
python main_qm9.py \
  --exp_name "$EXP_NAME" \
  --n_epochs $N_EPOCHS \
  --batch_size $BATCH_SIZE \
  --nf 192 \
  --n_layers 9 \
  --diffusion_steps 1000 \
  --diffusion_noise_schedule polynomial_2 \
  --diffusion_noise_precision 1e-5 \
  --diffusion_loss_type l2 \
  --train_diffusion \
  --ae_path "$VAE_PATH" \
  --latent_nf 1 \
  --ema_decay 0.9999 \
  --normalize_factors [1,4,10] \
  --lr 1e-4 \
  --clip_grad True \
  --test_epochs $TEST_EPOCHS \
  --conditioning alpha \
  --partial_conditioning \
  --noise_ratio $NOISE_RATIO \
  --mask_strategy $MASK_STRATEGY \
  --num_workers $NUM_WORKERS \
  --break_train_epoch False

echo -e "${GREEN}扩散模型训练完成!${NC}" 