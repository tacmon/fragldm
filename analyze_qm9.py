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
warnings.filterwarnings('ignore')

# 导入项目中的数据集加载函数
from qm9.data.args import init_argparse
from qm9.data.utils import initialize_datasets
from configs.datasets_config import qm9_with_h, qm9_without_h
import qm9.dataset as dataset

font_path = '/usr/share/fonts/truetype/msttcorefonts/times.ttf'
# 创建字体属性对象
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=font_path)
# 设置全局字体
# plt.rcParams['font.family'] = font.get_name()

# 设置中文字体支持
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 原来的中文字体设置，可能不兼容
# plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']  # 使用更通用的字体
# plt.rcParams['font.family'] = 'Times New Roman'  # 使用Times New Roman字体
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
plt.style.use('ggplot')

class QM9Analyzer:
    def __init__(self, output_dir="qm9_analysis"):
        """初始化QM9数据集分析器
        
        Args:
            output_dir: 输出目录，用于保存分析结果和图表
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载带有氢原子的QM9数据集
        self.with_h_data = self._load_dataset(remove_h=False)
        # 加载不带氢原子的QM9数据集
        self.without_h_data = self._load_dataset(remove_h=True)
        
        # 获取数据集信息
        self.with_h_info = qm9_with_h
        self.without_h_info = qm9_without_h
        
        print(f"Dataset loading completed")
        print(f"Dataset with hydrogen training set size: {len(self.with_h_data['train'])}")
        print(f"Dataset without hydrogen training set size: {len(self.without_h_data['train'])}")
    
    def _load_dataset(self, remove_h=False):
        """加载QM9数据集
        
        Args:
            remove_h: 是否移除氢原子
            
        Returns:
            dict: 数据集字典，包含train, valid, test三个分割
        """
        class Args:
            def __init__(self):
                self.batch_size = 64
                self.num_workers = 4
                self.filter_n_atoms = None
                self.datadir = 'qm9/temp'
                self.dataset = 'qm9'
                self.remove_h = remove_h  # 使用外部传入的参数
                self.include_charges = True
                self.shuffle = False
            
        cfg = Args()
        dataloaders, _ = dataset.retrieve_dataloaders(cfg)
        
        # 转换dataloader为易于分析的格式
        data_dict = {}
        for split in dataloaders.keys():
            data_list = []
            for data_batch in dataloaders[split]:
                data_list.append(data_batch)
            data_dict[split] = data_list
            
        return data_dict
    
    def analyze_dataset_stats(self):
        """分析QM9数据集的基本统计信息"""
        print("Starting QM9 dataset statistics analysis...")
        
        # 收集带氢数据集的信息
        train_data = self.with_h_data['train']
        
        # 1. 分子总数
        total_molecules = sum(len(batch['positions']) for batch in train_data)
        print(f"QM9 dataset contains {total_molecules} molecules")
        
        # 2. 分子大小分布（原子数）
        atom_counts = []
        for batch in train_data:
            atom_counts.extend(batch['num_atoms'].tolist())
        
        # 3. 原子类型分布
        atom_types_counter = Counter()
        for batch in train_data:
            for i in range(len(batch['one_hot'])):
                one_hot = batch['one_hot'][i].type(torch.float32)  # 转换为float类型
                atom_idxs = torch.argmax(one_hot, dim=1).tolist()
                atom_types_counter.update(atom_idxs)
        
        # 4. 分子属性
        properties = defaultdict(list)
        for batch in train_data:
            for key in batch.keys():
                if key not in ['positions', 'one_hot', 'charges', 'num_atoms', 'edge_mask', 'node_mask']:
                    if isinstance(batch[key], torch.Tensor) and batch[key].numel() > 0:
                        properties[key].extend(batch[key].tolist())
        
        # 保存结果
        self.stats = {
            'total_molecules': total_molecules,
            'atom_counts': atom_counts,
            'atom_types_counter': atom_types_counter,
            'properties': properties
        }
        
        print("Dataset statistics analysis completed")
        return self.stats
    
    def visualize_atom_distribution(self):
        """可视化原子分布情况"""
        print("Generating atom distribution visualization...")
        
        # 设置图表
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
        
        # 1. 分子大小分布（原子数）
        ax1 = plt.subplot(gs[0, 0])
        atom_counts = self.stats['atom_counts']
        
        # 计算数据范围，确保bins是整数边界
        min_atoms = int(min(atom_counts))
        max_atoms = int(max(atom_counts))
        bins = np.arange(min_atoms, max_atoms + 2) - 0.5  # +2是为了包含最大值，-0.5是为了让bin中心在整数上
        
        # 使用整数bins并调整KDE带宽使曲线更平滑
        sns.histplot(atom_counts, bins=bins, discrete=True, kde=True, kde_kws={'bw_adjust': 1.5}, color='cornflowerblue', ax=ax1)
        
        # 设置x轴刻度为整数
        x_ticks = np.arange(min_atoms, max_atoms + 1, 2)  # 每隔2个显示一个刻度，防止拥挤
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels(x_ticks)
        
        # 添加均值和中位数线及标签
        mean_val = np.mean(atom_counts)
        median_val = np.median(atom_counts)
        ax1.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.1f}')
        ax1.axvline(median_val, color='green', linestyle='-.', label=f'Median: {median_val:.1f}')
        ax1.legend()
        
        ax1.set_title('Molecule Size Distribution (Number of Atoms)', fontsize=14, fontproperties=font)
        ax1.set_xlabel('Number of Atoms', fontsize=12, fontproperties=font)
        ax1.set_ylabel('Number of Molecules', fontsize=12, fontproperties=font)
        
        # 2. 原子类型分布
        ax2 = plt.subplot(gs[0, 1])
        atom_types = self.stats['atom_types_counter']
        atom_labels = [self.with_h_info['atom_decoder'][t] for t in atom_types.keys()]
        atom_counts = list(atom_types.values())
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        ax2.bar(atom_labels, atom_counts, color=colors)
        ax2.set_title('Atom Type Distribution', fontsize=14, fontproperties=font)
        ax2.set_xlabel('Atom Type', fontsize=12, fontproperties=font)
        ax2.set_ylabel('Number of Atoms', fontsize=12, fontproperties=font)
        
        # 添加百分比标签
        total = sum(atom_counts)
        for i, count in enumerate(atom_counts):
            percentage = count / total * 100
            ax2.annotate(f'{percentage:.1f}%', 
                         xy=(i, count), 
                         xytext=(0, 5),
                         textcoords='offset points',
                         ha='center')
        
        # 3. 原子类型分布饼图
        ax3 = plt.subplot(gs[1, 0])
        ax3.pie(atom_counts, labels=atom_labels, autopct='%1.1f%%', 
                startangle=90, shadow=True, colors=colors)
        ax3.set_title('Atom Type Proportion', fontsize=14, fontproperties=font)
        
        # 4. 分子中各原子数量的关系
        ax4 = plt.subplot(gs[1, 1])
        
        # 收集每个分子中不同原子的数量
        molecules_atom_counts = []
        for batch in self.with_h_data['train']:
            for i in range(len(batch['one_hot'])):
                one_hot = batch['one_hot'][i].type(torch.float32)  # 转换为float类型
                counts = torch.sum(one_hot, dim=0).tolist()
                molecules_atom_counts.append(counts)
        
        df = pd.DataFrame(molecules_atom_counts, columns=self.with_h_info['atom_decoder'])
        
        # 计算相关系数矩阵
        corr = df.corr()
        
        # 创建自定义颜色映射
        colors = [(0.8, 0.2, 0.2), (1, 1, 1), (0.2, 0.8, 0.2)]
        cmap = LinearSegmentedColormap.from_list("custom_diverging", colors, N=256)
        
        # 绘制热图
        sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, vmin=-1, vmax=1, 
                   linewidths=.5, cbar_kws={"shrink": .8}, ax=ax4)
        ax4.set_title('Atom Type Count Correlation', fontsize=14, fontproperties=font)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'atom_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Atom distribution visualization completed")
    
    def visualize_molecular_properties(self):
        """可视化分子属性分布"""
        print("Generating molecular properties visualization...")
        
        # 选择一些重要的属性进行可视化
        props_to_plot = ['U0', 'U', 'G', 'H', 'zpve', 'gap', 'homo', 'lumo']
        prop_names = {
            'U0': 'Ground State Energy (eV)',
            'U': 'Internal Energy (eV)',
            'G': 'Gibbs Free Energy (eV)',
            'H': 'Enthalpy (eV)',
            'zpve': 'Zero-Point Vibrational Energy (eV)',
            'gap': 'HOMO-LUMO Gap (eV)',
            'homo': 'HOMO Energy Level (eV)',
            'lumo': 'LUMO Energy Level (eV)'
        }
        
        # 设置图表
        fig = plt.figure(figsize=(16, 16))
        gs = gridspec.GridSpec(4, 2)
        
        for i, prop in enumerate(props_to_plot):
            if prop in self.stats['properties']:
                ax = plt.subplot(gs[i // 2, i % 2])
                
                # 获取属性数据
                data = self.stats['properties'][prop]
                
                # 绘制分布图
                sns.histplot(data, kde=True, color='cornflowerblue', ax=ax)
                ax.set_title(f'{prop_names.get(prop, prop)} Distribution', fontsize=14, fontproperties=font)
                ax.set_xlabel(prop_names.get(prop, prop), fontsize=12, fontproperties=font)
                ax.set_ylabel('Number of Molecules', fontsize=12, fontproperties=font)
                
                # 添加均值和中位数线
                mean_val = np.mean(data)
                median_val = np.median(data)
                ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
                ax.axvline(median_val, color='green', linestyle='-.', label=f'Median: {median_val:.2f}')
                ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'molecular_properties.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Molecular properties visualization completed")
    
    def visualize_molecule_samples(self, num_samples=10):
        """可视化分子样本"""
        print("Generating molecule samples visualization...")
        
        # 从数据集中选择一些分子样本
        batch = self.with_h_data['train'][0]
        n_types = len(self.with_h_info['atom_decoder'])
        mols = []
        
        # 转换为RDKit分子对象
        for i in range(min(num_samples, len(batch['positions']))):
            positions = batch['positions'][i].view(-1, 3).numpy()
            one_hot = batch['one_hot'][i].view(-1, n_types).type(torch.float32)  # 确保转换为float类型
            atom_types = torch.argmax(one_hot, dim=1).numpy()
            
            # 创建RDKit分子
            mol = Chem.RWMol()
            
            # 添加原子
            for j, atom_type in enumerate(atom_types):
                atom_symbol = self.with_h_info['atom_decoder'][atom_type]
                atom = Chem.Atom(atom_symbol)
                mol.AddAtom(atom)
                
            # 根据原子间距添加键
            for j in range(len(atom_types)):
                for k in range(j):
                    # 计算原子间距离
                    dist = np.linalg.norm(positions[j] - positions[k])
                    atom1_type = self.with_h_info['atom_decoder'][atom_types[j]]
                    atom2_type = self.with_h_info['atom_decoder'][atom_types[k]]
                    
                    # 简单的键判断逻辑
                    if atom1_type == 'H' or atom2_type == 'H':
                        if dist < 1.2:  # 氢键阈值
                            mol.AddBond(j, k, Chem.BondType.SINGLE)
                    else:
                        if dist < 1.8:  # 碳、氮、氧、氟之间的键阈值
                            mol.AddBond(j, k, Chem.BondType.SINGLE)
            
            # 尝试添加到结果列表
            try:
                mol_final = mol.GetMol()
                Chem.SanitizeMol(mol_final)
                mols.append(mol_final)
            except:
                continue
        
        # 如果成功构建了分子，绘制它们
        if mols:
            img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(300, 300), legends=[f'Molecule {i+1}' for i in range(len(mols))])
            img.save(os.path.join(self.output_dir, 'molecule_samples.png'))
        else:
            print("Cannot build valid molecule structures with RDKit")
        
        print("Molecule samples visualization completed")
    
    def generate_radar_chart(self):
        """生成雷达图比较不同模型在QM9数据集上的表现"""
        print("Generating model performance radar chart...")
        
        # 定义模型和它们在不同指标上的性能数据（这些数据是示例，根据实际情况替换）
        models = ['VAE', 'EGNN', 'DM', 'GeoLDM']
        metrics = ['Validity', 'Novelty', 'Diversity', 'Stability', 'Drug-likeness']
        
        # 模拟的性能数据（范围0-1，1为最佳）
        # 这些数据应该来自实际的实验结果，这里只是示例
        performance = np.array([
            [0.75, 0.68, 0.72, 0.80, 0.65],  # VAE
            [0.82, 0.75, 0.68, 0.85, 0.70],  # EGNN
            [0.88, 0.85, 0.80, 0.82, 0.78],  # DM
            [0.92, 0.90, 0.85, 0.88, 0.82],  # GeoLDM
        ])
        
        # 创建雷达图
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合雷达图
        
        fig, ax = plt.figure(figsize=(10, 10)), plt.subplot(111, polar=True)
        
        # 添加每个指标的标签
        plt.xticks(angles[:-1], metrics, fontsize=12)
        
        # 设置雷达图的刻度标签
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        plt.ylim(0, 1)
        
        # 为每个模型绘制雷达图
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for i, model in enumerate(models):
            values = performance[i].tolist()
            values += values[:1]  # 闭合雷达图
            ax.plot(angles, values, linewidth=2, linestyle='-', label=model, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        # 添加图例
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title('Performance Comparison of Different Models on QM9 Dataset', fontsize=15, fontproperties=font)
        plt.savefig(os.path.join(self.output_dir, 'model_performance_radar.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Model performance radar chart completed")
    
    def generate_comparison_table(self):
        """生成不同模型在QM9数据集上的表现比较表格"""
        print("Generating model performance comparison table...")
        
        # 定义模型和评估指标
        models = ['VAE', 'EGNN', 'DM', 'GeoLDM']
        metrics = ['Validity(%)', 'Novelty(%)', 'Diversity', 'Stability(%)', 'Drug-likeness', 'Comp. Cost(h)']
        
        # 模拟的性能数据（这些数据是示例，根据实际情况替换）
        performance = np.array([
            [75.2, 68.5, 0.72, 80.3, 0.65, 2.5],   # VAE
            [82.1, 75.3, 0.68, 85.2, 0.70, 4.8],   # EGNN
            [88.4, 85.1, 0.80, 82.5, 0.78, 8.2],   # DM
            [92.6, 90.2, 0.85, 88.7, 0.82, 10.5],  # GeoLDM
        ])
        
        # 创建DataFrame
        df = pd.DataFrame(performance, index=models, columns=metrics)
        
        # 使用样式设置
        styled_df = df.style.background_gradient(cmap='Blues', axis=0)
        
        # 保存为HTML和CSV
        styled_df.to_html(os.path.join(self.output_dir, 'model_comparison_table.html'))
        df.to_csv(os.path.join(self.output_dir, 'model_comparison_table.csv'))
        
        # 创建可视化表格
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=df.values, 
                         rowLabels=df.index,
                         colLabels=df.columns,
                         cellLoc='center',
                         loc='center')
        
        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        # 为表头单元格设置不同的颜色
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(color='white')
        
        # 行标签设置颜色
        for i in range(len(df.index)):
            table[(i+1, -1)].set_facecolor('#5B9BD5')
            table[(i+1, -1)].set_text_props(color='white')
        
        plt.title('Performance Comparison of Different Models on QM9 Dataset', fontsize=16, pad=20, fontproperties=font)
        plt.savefig(os.path.join(self.output_dir, 'model_comparison_table.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Model performance comparison table completed")
    
    def generate_all(self):
        """生成所有分析结果和可视化图表"""
        # 执行数据分析
        self.analyze_dataset_stats()
        
        # 生成所有可视化图表
        self.visualize_atom_distribution()
        self.visualize_molecular_properties()
        self.visualize_molecule_samples()
        self.generate_radar_chart()
        self.generate_comparison_table()
        
        print(f"All analysis results have been saved to {self.output_dir} directory")


if __name__ == "__main__":
    analyzer = QM9Analyzer(output_dir="outputs/qm9_analysis")
    analyzer.generate_all() 