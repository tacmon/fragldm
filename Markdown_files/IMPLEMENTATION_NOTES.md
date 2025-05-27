# GeoLDM分子骨架生成 - 实现说明

本文档提供了GeoLDM分子骨架生成工具的技术实现细节，主要面向开发者或需要进行定制化修改的用户。

## 文件结构

```
├── sample_from_scaffold.py   # 主Python脚本，实现分子生成逻辑
├── generate_from_scaffold.sh # Bash脚本封装，提供易用的命令行接口
├── README_scaffold.md        # 用户使用说明
├── examples/                 # 示例文件目录
│   ├── README.txt            # 示例使用说明
│   └── benzene.smiles        # 苯的SMILES表示
└── IMPLEMENTATION_NOTES.md   # 本文档
```

## 技术架构

该实现采用了模块化设计，主要包括以下组件：

1. **模型加载模块** - 负责加载预训练的GeoLDM模型
2. **骨架准备模块** - 处理不同格式的分子骨架输入
3. **掩码生成模块** - 根据不同策略生成骨架原子的掩码
4. **分子生成模块** - 使用扩散模型从骨架生成完整分子
5. **结果保存模块** - 将生成的分子保存为多种格式并计算性质

## 关键实现细节

### 模型接口适配

`sample_from_scaffold.py`脚本的设计兼容多种可能的GeoLDM模型接口。它提供了多种尝试调用模型采样函数的方式：

```python
if hasattr(model, 'sample_from_scaffold'):
    results = model.sample_from_scaffold(**sample_args)
elif hasattr(model, 'sample'):
    # 尝试通用采样接口
    results = model.sample(
        condition=sample_args['scaffold_pos'],
        condition_types=sample_args['scaffold_atom_types'],
        condition_mask=sample_args['mask'],
        num_samples=num_samples,
        temperature=temperature,
        num_steps=diffusion_steps,
        noise_scale=noise_ratio
    )
```

### 骨架处理

脚本支持多种分子输入格式，并进行适当的预处理：

1. **XYZ文件** - 直接读取原子坐标和类型
2. **MOL/SDF文件** - 使用RDKit读取并提取化学信息
3. **SMILES字符串** - 转换为3D结构，添加氢原子并进行能量优化

### 掩码策略

提供了四种不同的掩码策略，控制哪些骨架原子保持固定：

1. **all** - 所有骨架原子固定，生成保持骨架完整
2. **none** - 不固定任何原子，仅使用骨架作为初始状态
3. **central** - 固定中心区域的原子（约30%），保持核心结构
4. **random** - 随机选择原子固定（约50%）

### 错误处理与恢复

脚本实现了健壮的错误处理机制，特别是在转换生成的原子坐标到RDKit分子时：

```python
try:
    rdmol = Chem.RemoveAllHs(rdmol)
    Chem.SanitizeMol(rdmol)
    mol = Chem.AddHs(rdmol, addCoords=True)
except:
    # 尝试使用距离矩阵添加键
    mol = mol.GetMol()
    mol = Chem.AssignBondOrdersFromTemplate(Chem.MolFromSmiles(''), mol)
    # 最后一次尝试
    try:
        Chem.SanitizeMol(mol)
    except:
        pass
```

## 可能需要的调整

### 模型目录结构适配

如果您的模型目录结构与默认预期不同，需要修改`load_model`函数中的以下部分：

```python
config_path = os.path.join(model_path, 'config.json')
checkpoint_path = os.path.join(model_path, 'best_model.pt')
```

### 模型类导入适配

如果您的模型类在不同的模块中，需要修改导入部分：

```python
if 'model' in config and config['model'] == 'geoldm':
    logging.info("检测到GeoLDM模型")
    from models.geoldm import GeoLDM
    model_class = GeoLDM
else:
    # 尝试默认导入
    logging.info("尝试默认导入GeoLDM模型")
    from models.geoldm import GeoLDM
    model_class = GeoLDM
```

### 采样函数参数调整

如果您的模型采样API与预期不同，需要修改调用部分：

```python
results = model.sample(
    condition=sample_args['scaffold_pos'],
    condition_types=sample_args['scaffold_atom_types'],
    condition_mask=sample_args['mask'],
    num_samples=num_samples,
    temperature=temperature,
    num_steps=diffusion_steps,
    noise_scale=noise_ratio
)
```

## 扩展功能的指导

### 添加新的掩码策略

在`generate_mask`函数中添加新的策略分支：

```python
elif mask_strategy == 'your_new_strategy':
    # 实现您的新策略
    # ...
    logging.info(f"掩码策略: 您的新策略")
```

### 添加新的分子性质计算

在`calculate_properties`函数中添加新的性质计算：

```python
props['your_new_property'] = calculate_your_property(mol)
```

### 添加新的输出格式

在`save_results`函数中添加新的保存逻辑：

```python
# 保存为您的新格式
your_format_dir = os.path.join(output_dir, "your_format_files")
os.makedirs(your_format_dir, exist_ok=True)
for i, mol_data in enumerate(generated_mols):
    mol = mol_data.get('mol')
    if mol:
        path = os.path.join(your_format_dir, f"mol_{i+1}.your_ext")
        save_mol_to_your_format(mol, path)
```

## 性能优化建议

1. **批处理** - 对于大量分子生成，考虑实现批处理以减少模型加载开销
2. **缓存机制** - 对相同骨架的请求实现结果缓存
3. **并行处理** - 在后处理阶段（如分子性质计算）使用多线程
4. **混合精度** - 对于大型模型，考虑使用PyTorch的混合精度训练

## 调试与诊断

脚本提供了详细的日志记录，日志文件保存在输出目录的`sampling.log`中。日志包含以下关键信息：

1. 参数设置
2. 模型加载过程
3. 骨架处理细节
4. 采样过程
5. 生成结果统计
6. 详细的错误信息（如有）

如果遇到生成质量问题，建议检查：

1. 温度参数 - 过高可能导致不稳定结果
2. 扩散步骤数 - 过少可能导致低质量结果
3. 掩码策略 - 不同骨架可能需要不同策略
4. 骨架质量 - 确保输入骨架具有合理的化学结构

## 已知限制

1. 当前实现假设模型输出格式为特定结构（positions、atom_types等）
2. XYZ文件解析可能对某些非标准格式不兼容
3. 分子可视化依赖于RDKit和matplotlib，可能在某些环境中有兼容性问题
4. 对非常大的分子（>100原子）可能需要更大的内存资源

## 未来工作方向

1. 添加Web界面
2. 支持更多分子属性和评分函数
3. 实现多模型集成生成
4. 添加批量骨架处理功能
5. 提供Docker容器化部署选项 