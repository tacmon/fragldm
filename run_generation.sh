#!/bin/bash

# 设置颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}启动GeoLDM分子生成模式${NC}"
echo -e "${YELLOW}使用模型:${NC} outputs/important_par/"

# 设置PYTHONPATH
export PYTHONPATH=$(pwd)

# 运行生成命令
python main_qm9.py \
  --gen_mode 1 \
  --scaf_model_path "outputs/important_par/" \
  --no_wandb

echo -e "${GREEN}生成完成!${NC}" 