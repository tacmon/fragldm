#!/bin/bash

# 设置可调整的参数（可根据需要修改）
EXP_NAME="repaint"
RESUME_PATH="./outputs/qm9_latent2"
START_EPOCH=0
NOISE_RATIO=0.75
MASK_STRATEGY="connected"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}启动GeoLDM模型训练 - 片段约束分子生成${NC}"
echo -e "${YELLOW}实验名称:${NC} $EXP_NAME"

# 设置PYTHONPATH
export PYTHONPATH=$(pwd)

# 运行训练命令
python main_qm9_for_gen.py \
  --exp_name "$EXP_NAME" \
  --resume "$RESUME_PATH" \
  --no_wandb \
  --start_epoch $START_EPOCH \
  --noise_ratio $NOISE_RATIO \
  --mask_strategy $MASK_STRATEGY \
  --partial_conditioning

echo -e "${GREEN}生成完成!${NC}" 
