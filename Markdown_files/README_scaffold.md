# 分子骨架条件生成工具

这个工具包使用预训练的GeoLDM（几何潜在扩散模型）从分子骨架生成完整的分子结构。通过提供一个分子骨架作为条件输入，模型可以生成完整的分子，同时保留骨架的关键结构特征。

## 功能特点

- 支持多种输入格式（SMILES, XYZ, MOL, SDF）
- 可以控制生成分子的多样性（温度参数）
- 可以调整骨架保留的程度（掩码策略）
- 生成的分子以多种格式保存（SDF, XYZ）
- 自动计算生成分子的各种性质（分子量、LogP、QED等）
- 生成可视化的分子网格图像

## 安装依赖

本工具需要以下Python包：

```bash
pip install rdkit torch numpy matplotlib
```

## 使用方法

### 基本用法

使用提供的Bash脚本可以方便地运行分子生成：

```bash
./generate_from_scaffold.sh --model-path <模型路径> --input <骨架文件或SMILES>
```

### 示例

从SMILES字符串生成10个分子：

```bash
./generate_from_scaffold.sh --model-path ./models/geoldm_qm9 --input "c1ccccc1" --num-samples 10
```

从XYZ文件生成具有较高多样性的20个分子：

```bash
./generate_from_scaffold.sh --model-path ./models/geoldm_qm9 --input scaffold.xyz --num-samples 20 --temperature 1.2
```

### 高级参数

脚本支持多种参数来精细控制生成过程：

- `--output-dir`：指定输出目录（默认：results）
- `--num-samples`：要生成的分子数量（默认：10）
- `--temperature`：采样温度，控制多样性（默认：1.0）
- `--diffusion-steps`：扩散采样步骤数（默认：1000）
- `--mask-strategy`：骨架原子的掩码策略（默认：central）
  - `all`：保留所有骨架原子
  - `none`：不保留任何骨架原子
  - `central`：保留中心原子
  - `random`：随机保留原子
- `--noise-ratio`：添加到骨架的噪声比例（默认：0.7）
- `--condition-encoder`：用于条件编码的网络类型（默认：gnn）
- `--seed`：随机数种子，用于可重复的结果

## 输出文件

运行脚本后，将在指定目录（默认为"results"）中生成以下文件：

- `generated_molecules.sdf`：包含所有生成分子的SDF文件
- `xyz_files/`：包含每个分子XYZ格式文件的目录
- `summary.csv`：所有生成分子的属性摘要
- `molecules_grid.png`：生成分子的可视化网格图像
- `scaffold.sdf`：原始输入骨架
- `sampling.log`：生成过程的详细日志

## 直接使用Python脚本

您也可以直接使用Python脚本，这在需要进一步定制时很有用：

```bash
python sample_from_scaffold.py --model_path <模型路径> --input_scaffold <骨架> [其他参数]
```

## 注意事项

- 确保模型路径包含`config.json`和`best_model.pt`文件
- 对于复杂骨架，建议使用掩码策略`central`或`all`
- 增加温度会增加多样性，但可能降低生成分子的质量
- 增加扩散步骤会提高精度，但会增加计算时间

## 故障排除

如果遇到问题，请检查日志文件`sampling.log`以获取详细信息。常见问题包括：

- 模型路径不正确或模型文件缺失
- 输入骨架文件格式不受支持或解析错误
- 硬件资源不足（如GPU内存）
- Python环境中缺少必要的依赖项

## 引用

如果您在研究中使用了这个工具，请引用原始GeoLDM论文：

```
@article{xu2023geometric,
  title={Geometric Latent Diffusion Models for 3D Molecule Generation},
  author={Xu, Minkai and Powers, Alexander S and Chung, Nathaniel and Sattigeri, Prasanna and Hernandez-Lobato, Jose Miguel},
  journal={arXiv preprint arXiv:2305.01140},
  year={2023}
}
``` 