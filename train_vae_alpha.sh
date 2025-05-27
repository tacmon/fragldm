#!/bin/bash

# 设置VAE训练的可调整参数
EXP_NAME="vae_alpha_bs64_nf192_nlayers9_lr1e-4"
N_EPOCHS=500
BATCH_SIZE=16
TEST_EPOCHS=10
KL_WEIGHT=0.01  # VAE训练中KL散度的权重
NUM_WORKERS=8  # 数据加载并行进程数

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}启动GeoLDM - VAE预训练阶段${NC}"
echo -e "${YELLOW}实验名称:${NC} $EXP_NAME"
echo -e "${YELLOW}条件类型:${NC} alpha"
echo -e "${YELLOW}训练轮数:${NC} $N_EPOCHS"
echo -e "${YELLOW}数据加载进程数:${NC} $NUM_WORKERS"
echo -e "${YELLOW}批次大小:${NC} $BATCH_SIZE"
echo -e "${YELLOW}KL权重:${NC} $KL_WEIGHT"

# 设置PYTHONPATH
export PYTHONPATH=$(pwd)

# 运行VAE预训练命令
# 注意: 不使用train_diffusion标志，只保留trainable_ae
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
  --conditioning alpha \
  --num_workers $NUM_WORKERS \
  --break_train_epoch False \
  --no_wandb

echo -e "${GREEN}VAE预训练完成!${NC}"
echo -e "${YELLOW}下一步:${NC} 使用训练好的VAE作为输入，运行扩散模型训练。"
echo -e "运行方法: ./train_scaffold_diffusion.sh" 