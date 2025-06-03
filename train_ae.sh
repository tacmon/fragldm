#!/bin/bash

# 设置可调整的参数（可根据需要修改）
EXP_NAME="train_vae_test"
N_EPOCHS=500
BATCH_SIZE=16
TEST_EPOCHS=10
KL_WEIGHT=0.01  # VAE训练中KL散度的权重

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}启动VAE模型训练 - 编码器与解码器${NC}"
echo -e "${YELLOW}实验名称:${NC} $EXP_NAME"
echo -e "${YELLOW}潜在特征维度:${NC} 1"
echo -e "${YELLOW}KL权重:${NC} $KL_WEIGHT"
echo -e "${YELLOW}训练轮数:${NC} $N_EPOCHS"

# 设置PYTHONPATH
export PYTHONPATH=$(pwd)

# 运行训练命令 - 注意这里没有--train_diffusion选项
python main_qm9.py \
  --exp_name "$EXP_NAME" \
  --n_epochs $N_EPOCHS \
  --batch_size $BATCH_SIZE \
  --nf 192 \
  --n_layers 9 \
  --latent_nf 1 \
  --kl_weight $KL_WEIGHT \
  --ema_decay 0.9999 \
  --normalize_factors [1,4,10] \
  --lr 1e-4 \
  --clip_grad True \
  --test_epochs $TEST_EPOCHS \
  --break_train_epoch False \
  --no_wandb

echo -e "${GREEN}VAE训练完成!${NC}" 