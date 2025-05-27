#!/bin/bash

# 设置可调整的参数（可根据需要修改）
EXP_NAME="unconditional_bs128_nf192_nlayers9_lr1e-4_noise1e-5_steps1000_poly2_l2_latent1_ema0.9999_normalize1410"
N_EPOCHS=1000
BATCH_SIZE=128
TEST_EPOCHS=20

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}启动GeoLDM模型训练 - 无条件分子生成${NC}"
echo -e "${YELLOW}实验名称:${NC} $EXP_NAME"
echo -e "${YELLOW}训练轮数:${NC} $N_EPOCHS"

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
  --trainable_ae \
  --latent_nf 1 \
  --ema_decay 0.9999 \
  --normalize_factors [1,4,10] \
  --lr 1e-4 \
  --clip_grad True \
  --test_epochs $TEST_EPOCHS \
  --break_train_epoch False

echo -e "${GREEN}训练完成!${NC}" 