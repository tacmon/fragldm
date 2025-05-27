#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
QM9数据集分析与可视化
此脚本用于分析QM9分子数据集的结构特征，并生成多种可视化图表
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import Counter, defaultdict
from rdkit import Chem
from rdkit.Chem import Draw
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import warnings
from matplotlib.font_manager import FontProperties
warnings.filterwarnings('ignore')

# 导入项目中的数据集加载函数
from qm9.data.args import init_argparse
from qm9.data.utils import initialize_datasets
from configs.datasets_config import qm9_with_h, qm9_without_h
import qm9.dataset as dataset

# 创建字体属性对象
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
font = FontProperties(fname=font_path)
# 设置全局字体为Times New Roman
# plt.rcParams['font.family'] = font.get_name()
# plt.rcParams['font.sans-serif'] = [font.get_name()]
# plt.rcParams['font.serif'] = [font.get_name()]
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
plt.style.use('ggplot')

# ========== 相关性分析与可视化 ========== #
if __name__ == '__main__':
    # 1. 初始化参数和加载数据集
    args = init_argparse('qm9')
    datadir = args.datadir if hasattr(args, 'datadir') else 'qm9/temp'
    args, datasets, num_species, max_charge = initialize_datasets(
        args, datadir, 'qm9', subtract_thermo=True, force_download=False, remove_h=False)

    # 2. 需要提取的属性
    prop_names = ['mu', 'alpha', 'homo', 'lumo', 'U', 'U0', 'H', 'G']
    # 兼容大小写
    all_keys = list(datasets['train'].data.keys())
    key_map = {k.lower(): k for k in all_keys}
    prop_keys = [key_map.get(p.lower(), p) for p in prop_names]

    # 3. 合并train、valid、test数据
    all_data = {p: [] for p in prop_names}
    for split in ['train', 'valid', 'test']:
        ds = datasets[split]
        for p, k in zip(prop_names, prop_keys):
            # ds.data[k] 是一维tensor
            all_data[p].append(ds.data[k].cpu().numpy())
    # 拼接
    for p in prop_names:
        all_data[p] = np.concatenate(all_data[p], axis=0)

    # 4. 转为DataFrame
    df = pd.DataFrame(all_data)

    # 5. 计算相关性矩阵
    corr = df.corr()

    # 6. 可视化
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        corr, annot=True, cmap='coolwarm', fmt='.10f', square=True,
        linewidths=0.5, cbar_kws={"shrink": .8},
        annot_kws={'fontproperties': font}
    )

    # 设置坐标轴标签字体
    ax.set_xlabel(ax.get_xlabel(), fontproperties=font)
    ax.set_ylabel(ax.get_ylabel(), fontproperties=font)
    ax.set_xticklabels(ax.get_xticklabels(), fontproperties=font)
    ax.set_yticklabels(ax.get_yticklabels(), fontproperties=font)

    # 设置色条字体
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    for label in cbar.ax.get_yticklabels():
        label.set_fontproperties(font)

    plt.tight_layout()
    plt.savefig('outputs/qm9_correlation_matrix.pdf', dpi=200, format='pdf')
    plt.show()
    print('相关性矩阵已保存为 qm9_correlation_matrix.pdf')

    # 输出U/U0/H/G四个属性的取值上下界
    print("U/U0/H/G 四个属性的取值范围：")
    for p in ['U', 'U0', 'H', 'G']:
        vmin = df[p].min()
        vmax = df[p].max()
        print(f"{p}: min={vmin:.8f}, max={vmax:.8f}")

    # ========== U/U0/H/G 四属性两两线性关系分析 ========== #
    from sklearn.linear_model import LinearRegression
    props = ['U', 'U0', 'H', 'G']
    print("\nU/U0/H/G 两两线性关系分析：")
    for i in range(len(props)):
        for j in range(i+1, len(props)):
            x = df[props[i]].values.reshape(-1, 1)
            y = df[props[j]].values
            reg = LinearRegression().fit(x, y)
            a = reg.coef_[0]
            b = reg.intercept_
            y_pred = reg.predict(x)
            max_abs_err = np.max(np.abs(y - y_pred))
            mse = np.mean((y - y_pred) ** 2)
            print(f"{props[j]} = {a:.8f} * {props[i]} + {b:.8f} | 最大绝对误差: {max_abs_err:.8e} | MSE: {mse:.8e}")

            # 反向也做一次
            x2 = df[props[j]].values.reshape(-1, 1)
            y2 = df[props[i]].values
            reg2 = LinearRegression().fit(x2, y2)
            a2 = reg2.coef_[0]
            b2 = reg2.intercept_
            y2_pred = reg2.predict(x2)
            max_abs_err2 = np.max(np.abs(y2 - y2_pred))
            mse2 = np.mean((y2 - y2_pred) ** 2)
            print(f"{props[i]} = {a2:.8f} * {props[j]} + {b2:.8f} | 最大绝对误差: {max_abs_err2:.8e} | MSE: {mse2:.8e}")
