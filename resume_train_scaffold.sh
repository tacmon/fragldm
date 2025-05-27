#!/bin/bash

# 设置可调整的参数（与原脚本相同）
EXP_NAME="scaffold_based_bs128_nf192_nlayers9_lr1e-4_noise1e-5_steps1000_poly2_l2_latent1_ema0.9999_normalize1410_clipgradTrue"
N_EPOCHS=1000
BATCH_SIZE=128
TEST_EPOCHS=20
NOISE_RATIO=0.7
MASK_STRATEGY="connected"  # 可选: "random", "connected", "central", "peripheral"
START_EPOCH=210  # 从210个epoch开始续训

# 设置断点续训所需的模型路径
RESUME_PATH="./outputs/${EXP_NAME}"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}恢复GeoLDM模型训练 - 骨架约束分子生成${NC}"
echo -e "${YELLOW}实验名称:${NC} $EXP_NAME"
echo -e "${YELLOW}掩码策略:${NC} $MASK_STRATEGY"
echo -e "${YELLOW}噪声比例:${NC} $NOISE_RATIO"
echo -e "${YELLOW}从Epoch:${NC} $START_EPOCH 开始续训"
echo -e "${YELLOW}恢复路径:${NC} $RESUME_PATH"

# 检查恢复路径是否存在
if [ ! -d "$RESUME_PATH" ]; then
    echo -e "${YELLOW}警告: 恢复路径 $RESUME_PATH 不存在!${NC}"
    read -p "是否继续训练? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}已取消训练.${NC}"
        exit 1
    fi
fi

# 设置PYTHONPATH
export PYTHONPATH=$(pwd)

# 运行训练命令（增加断点续训参数）
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
  --trainable_ae \
  --latent_nf 1 \
  --ema_decay 0.9999 \
  --normalize_factors [1,4,10] \
  --lr 1e-4 \
  --clip_grad True \
  --test_epochs $TEST_EPOCHS \
  --partial_conditioning \
  --noise_ratio $NOISE_RATIO \
  --mask_strategy $MASK_STRATEGY \
  --break_train_epoch False \
  --resume "$RESUME_PATH" \
  --start_epoch $START_EPOCH

echo -e "${GREEN}训练完成!${NC}" 