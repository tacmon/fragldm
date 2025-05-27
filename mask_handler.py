"""
掩码处理工具

此模块提供用于生成和应用分子掩码的函数，主要用于片段条件生成的训练过程。
"""

import torch
import numpy as np
from collections import deque
import torch.nn.functional as F
from equivariant_diffusion.utils import remove_mean_with_mask


def generate_connected_mask(x, node_mask, noise_ratio=0.5, random_seed=None):
    """
    生成一个连通的掩码，确保保留的原子是连通的。
    
    参数:
        x (torch.Tensor): 原子坐标，形状为 [batch_size, num_nodes, 3]
        node_mask (torch.Tensor): 节点掩码，形状为 [batch_size, num_nodes, 1]
        noise_ratio (float): 添加噪声的原子比例，范围在[0, 1]之间
        random_seed (int): 随机数种子，用于可重复性
        
    返回:
        tuple: (noise_mask, condition_mask)，形状均为 [batch_size, num_nodes, 1]
               noise_mask 为1表示节点需要添加噪声
               condition_mask 为1表示节点是条件（保留的原子）
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    batch_size, num_nodes, _ = x.shape
    device = x.device
    
    # 初始化掩码
    noise_mask = torch.zeros_like(node_mask)
    
    for b in range(batch_size):
        # 获取有效节点（存在的原子）
        valid_nodes = torch.where(node_mask[b, :, 0] > 0)[0].cpu().numpy()
        
        if len(valid_nodes) == 0:
            continue
        
        # 计算要添加噪声的节点数量
        num_noise_nodes = int(len(valid_nodes) * noise_ratio)
        num_keep_nodes = len(valid_nodes) - num_noise_nodes
        
        if num_keep_nodes == 0 or num_noise_nodes == 0:
            # 如果全部保留或全部加噪，就不需要计算连通性
            if noise_ratio > 0.5:  # 如果噪声比例大于0.5，则大部分原子加噪
                noise_mask[b, valid_nodes, 0] = 1
            continue
        
        # 随机选择起始节点（保留的起始原子）
        start_idx = np.random.choice(valid_nodes)
        
        # 使用BFS生成连通的保留区域
        selected_keep_nodes = []
        queue = deque([start_idx])
        visited = set([start_idx])
        
        # 计算节点之间的距离矩阵（欧几里得距离）
        coords = x[b].detach().cpu().numpy()
        
        while len(selected_keep_nodes) < num_keep_nodes and queue:
            current = queue.popleft()
            selected_keep_nodes.append(current)
            
            if len(selected_keep_nodes) >= num_keep_nodes:
                break
            
            # 计算当前节点与所有其他节点的距离
            distances = []
            for node in valid_nodes:
                if node not in visited:
                    # 计算欧几里得距离
                    dist = np.linalg.norm(coords[current] - coords[node])
                    distances.append((node, dist))
            
            # 按距离排序
            distances.sort(key=lambda x: x[1])
            
            # 添加最近的节点到队列
            for node, _ in distances:
                if node not in visited:
                    visited.add(node)
                    queue.append(node)
        
        # 选择剩余节点作为加噪节点
        noise_nodes = [node for node in valid_nodes if node not in selected_keep_nodes]
        
        # 更新噪声掩码
        noise_mask[b, noise_nodes, 0] = 1
    
    # 条件掩码是节点掩码减去噪声掩码
    condition_mask = node_mask * (1 - noise_mask)
    
    return noise_mask, condition_mask


def apply_mask_to_batch(x, h, node_mask, noise_mask, condition_mask):
    """
    根据掩码将批次数据分为噪声部分和条件部分。
    
    参数:
        x (torch.Tensor): 原子坐标，形状为 [batch_size, num_nodes, 3]
        h (dict): 原子特征，包含 'categorical' 和 'integer' 键
        node_mask (torch.Tensor): 节点掩码，形状为 [batch_size, num_nodes, 1]
        noise_mask (torch.Tensor): 噪声掩码，形状为 [batch_size, num_nodes, 1]
        condition_mask (torch.Tensor): 条件掩码，形状为 [batch_size, num_nodes, 1]
        
    返回:
        tuple: (noise_x, noise_h, cond_x, cond_h)
    """
    # 应用噪声掩码
    noise_x = x * noise_mask
    noise_h = {
        'categorical': h['categorical'] * noise_mask,
        'integer': h['integer'] * noise_mask
    }
    
    # 应用条件掩码
    cond_x = x * condition_mask
    cond_h = {
        'categorical': h['categorical'] * condition_mask,
        'integer': h['integer'] * condition_mask
    }
    
    return noise_x, noise_h, cond_x, cond_h


def prepare_condition_encoding(cond_x, cond_h):
    """
    将条件部分的原子坐标和特征编码为上下文向量。
    
    参数:
        cond_x (torch.Tensor): 条件原子坐标，形状为 [batch_size, num_nodes, 3]
        cond_h (dict): 条件原子特征
        
    返回:
        torch.Tensor: 条件编码，形状为 [batch_size, encoding_dim]
    """
    batch_size = cond_x.size(0)
    
    # 将坐标和特征展平并连接
    context_x = cond_x.reshape(batch_size, -1)  # [B, N*3]
    context_h_cat = cond_h['categorical'].reshape(batch_size, -1)
    context_h_int = cond_h['integer'].reshape(batch_size, -1)
    
    # 创建条件编码
    condition_encoding = torch.cat([context_x, context_h_cat, context_h_int], dim=1)
    
    return condition_encoding


def enhance_context(original_context, condition_encoding):
    """
    将原始上下文和条件编码合并。
    
    参数:
        original_context: 原始上下文，可能为None
        condition_encoding: 条件编码
        
    返回:
        torch.Tensor: 增强的上下文
    """
    if original_context is None:
        return condition_encoding
    
    if isinstance(original_context, torch.Tensor) and original_context.size(0) == condition_encoding.size(0):
        return torch.cat([original_context, condition_encoding], dim=1)
    
    return condition_encoding


def visualize_masks(x, node_mask, noise_mask=None, condition_mask=None):
    """
    可视化掩码（用于调试）。打印掩码统计信息。
    
    参数:
        x (torch.Tensor): 节点坐标，形状为 [batch_size, num_nodes, 3]
        node_mask (torch.Tensor): 节点掩码，形状为 [batch_size, num_nodes, 1]
        noise_mask (torch.Tensor, optional): 噪声掩码，形状为 [batch_size, num_nodes, 1]
        condition_mask (torch.Tensor, optional): 条件掩码，形状为 [batch_size, num_nodes, 1]
    """
    batch_size, num_nodes, _ = x.shape
    
    print(f"批次大小: {batch_size}, 每个图的节点数: {num_nodes}")
    
    # 节点掩码统计
    valid_nodes = node_mask.sum(dim=(1,2)).cpu().numpy()
    print(f"有效节点平均数: {valid_nodes.mean():.2f}, 最小: {valid_nodes.min()}, 最大: {valid_nodes.max()}")
    
    # 噪声掩码统计
    if noise_mask is not None:
        noise_nodes = noise_mask.sum(dim=(1,2)).cpu().numpy()
        print(f"噪声节点平均数: {noise_nodes.mean():.2f}, 最小: {noise_nodes.min()}, 最大: {noise_nodes.max()}")
        
        # 计算噪声比例
        noise_ratio = noise_nodes / valid_nodes
        print(f"噪声比例平均值: {noise_ratio.mean():.2f}, 最小: {noise_ratio.min():.2f}, 最大: {noise_ratio.max():.2f}")
    
    # 条件掩码统计
    if condition_mask is not None:
        condition_nodes = condition_mask.sum(dim=(1,2)).cpu().numpy()
        print(f"条件节点平均数: {condition_nodes.mean():.2f}, 最小: {condition_nodes.min()}, 最大: {condition_nodes.max()}")
        
        # 计算条件比例
        condition_ratio = condition_nodes / valid_nodes
        print(f"条件比例平均值: {condition_ratio.mean():.2f}, 最小: {condition_ratio.min():.2f}, 最大: {condition_ratio.max():.2f}") 