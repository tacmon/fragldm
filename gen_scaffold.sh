#!/bin/bash

# 设置可调整的参数（可根据需要修改）
EXP_NAME="scaf_alpha_less_edges"
RESUME_PATH="./outputs/scaffold_alpha_bs64_nf192_nlayers9_lr1e-4_noise1e-5_steps1000_poly2_l2_latent1_ema0.9999_normalize181_clipgradTrue_high"

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
  --resume "$RESUME_PATH"

echo -e "${GREEN}生成完成!${NC}" 
