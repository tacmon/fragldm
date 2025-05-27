"""
掩码工具

此模块提供用于生成分子图连接掩码的工具函数。
这些掩码用于支持部分条件生成。
"""

import torch
import numpy as np
from collections import deque
import torch.nn.functional as F


def get_adj_matrix(n_nodes, batch_size, device):
    """
    生成完全连接的邻接矩阵。
    
    参数:
        n_nodes (int): 节点数量
        batch_size (int): 批次大小
        device (torch.device): 运算设备
        
    返回:
        torch.Tensor: 邻接矩阵，形状为 [batch_size, n_nodes, n_nodes]
    """
    # 生成完全连接的邻接矩阵
    adj = torch.ones((batch_size, n_nodes, n_nodes), device=device)
    
    # 移除自连接（对角线设为0）
    diag_mask = torch.eye(n_nodes, device=device).unsqueeze(0).expand(batch_size, -1, -1)
    adj = adj * (1 - diag_mask)
    
    return adj


def get_edges_batch(n_nodes, batch_size, device):
    """
    生成批次中每个图的边列表。
    
    参数:
        n_nodes (int): 每个图的节点数量
        batch_size (int): 批次大小
        device (torch.device): 运算设备
        
    返回:
        torch.Tensor: 边列表，形状为 [2, batch_size * n_edges]，其中n_edges = n_nodes * (n_nodes - 1) / 2
    """
    # 计算每个图的边数
    n_edges = int(n_nodes * (n_nodes - 1) / 2)
    
    # 为每个图创建边列表
    edges = torch.zeros((2, batch_size * n_edges), dtype=torch.long, device=device)
    
    # 边计数器
    edge_count = 0
    
    # 对于每个图
    for b in range(batch_size):
        # 节点偏移量
        offset = b * n_nodes
        
        # 对于每对节点，创建一条边
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                edges[0, edge_count] = i + offset  # 源节点
                edges[1, edge_count] = j + offset  # 目标节点
                edge_count += 1
    
    return edges


def generate_connected_mask(x, h, node_mask, noise_ratio=0.5, random_seed=None):
    """
    生成一个连接性掩码，确保噪声区域是连接的。
    
    参数:
        x (torch.Tensor): 节点坐标，形状为 [batch_size, num_nodes, 3]
        h (torch.Tensor): 节点特征，形状为 [batch_size, num_nodes, h_dim]
        node_mask (torch.Tensor): 节点掩码，形状为 [batch_size, num_nodes]
        noise_ratio (float): 添加噪声的节点比例，范围在[0, 1]之间
        random_seed (int): 随机数种子，用于可重复性
        
    返回:
        torch.Tensor: 噪声掩码，形状为 [batch_size, num_nodes]，其中1表示节点已添加噪声
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    batch_size, num_nodes, _ = x.shape
    device = x.device
    
    # 初始化噪声掩码
    noise_mask = torch.zeros_like(node_mask)
    
    # 对批次中的每个图，生成连接的噪声掩码
    for b in range(batch_size):
        # 获取有效节点
        valid_nodes = torch.where(node_mask[b] > 0)[0].cpu().numpy()
        
        if len(valid_nodes) == 0:
            continue
        
        # 计算要添加噪声的节点数量
        num_noise_nodes = int(len(valid_nodes) * noise_ratio)
        
        if num_noise_nodes == 0:
            continue
        
        # 随机选择起始节点
        start_idx = np.random.choice(valid_nodes)
        
        # 使用广度优先搜索（BFS）来选择连接的节点
        selected_nodes = []
        queue = deque([start_idx])
        visited = set([start_idx])
        
        # 计算节点之间的距离矩阵（欧几里得距离）
        coords = x[b].detach().cpu().numpy()
        
        while len(selected_nodes) < num_noise_nodes and queue:
            current = queue.popleft()
            selected_nodes.append(current)
            
            if len(selected_nodes) >= num_noise_nodes:
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
        
        # 更新噪声掩码
        noise_mask[b, selected_nodes] = 1
    
    return noise_mask


def generate_mask_by_strategy(x, h, node_mask, strategy="random", noise_ratio=0.5, random_seed=None):
    """
    根据指定策略生成掩码。
    
    参数:
        x (torch.Tensor): 节点坐标，形状为 [batch_size, num_nodes, 3]
        h (torch.Tensor): 节点特征，形状为 [batch_size, num_nodes, h_dim]
        node_mask (torch.Tensor): 节点掩码，形状为 [batch_size, num_nodes]
        strategy (str): 掩码生成策略，可以是 "random", "connected", "central", "peripheral"
        noise_ratio (float): 添加噪声的节点比例，范围在[0, 1]之间
        random_seed (int): 随机数种子，用于可重复性
        
    返回:
        torch.Tensor: 噪声掩码，形状为 [batch_size, num_nodes]，其中1表示节点将被标记为噪声
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    batch_size, num_nodes, _ = x.shape
    device = x.device
    
    # 初始化噪声掩码
    noise_mask = torch.zeros_like(node_mask)
    
    if strategy == "random":
        # 对批次中的每个图，随机选择节点
        for b in range(batch_size):
            # 获取有效节点
            valid_nodes = torch.where(node_mask[b] > 0)[0].cpu().numpy()
            
            if len(valid_nodes) == 0:
                continue
            
            # 计算要添加噪声的节点数量
            num_noise_nodes = int(len(valid_nodes) * noise_ratio)
            
            if num_noise_nodes == 0:
                continue
            
            # 随机选择节点
            selected_nodes = np.random.choice(valid_nodes, size=num_noise_nodes, replace=False)
            
            # 更新噪声掩码
            noise_mask[b, selected_nodes] = 1
    
    elif strategy == "connected":
        # 使用连接性掩码生成
        noise_mask = generate_connected_mask(x, h, node_mask, noise_ratio, random_seed)
    
    elif strategy == "central":
        # 对批次中的每个图，选择中心区域的节点
        for b in range(batch_size):
            # 获取有效节点
            valid_nodes = torch.where(node_mask[b] > 0)[0].cpu().numpy()
            
            if len(valid_nodes) == 0:
                continue
            
            # 计算要添加噪声的节点数量
            num_noise_nodes = int(len(valid_nodes) * noise_ratio)
            
            if num_noise_nodes == 0:
                continue
            
            # 计算分子的中心点
            coords = x[b].detach().cpu().numpy()
            valid_coords = coords[valid_nodes]
            center = np.mean(valid_coords, axis=0)
            
            # 计算每个节点到中心的距离
            distances = []
            for i, node in enumerate(valid_nodes):
                dist = np.linalg.norm(coords[node] - center)
                distances.append((node, dist))
            
            # 按距离排序（从近到远）
            distances.sort(key=lambda x: x[1])
            
            # 选择最靠近中心的节点
            selected_nodes = [node for node, _ in distances[:num_noise_nodes]]
            
            # 更新噪声掩码
            noise_mask[b, selected_nodes] = 1
    
    elif strategy == "peripheral":
        # 对批次中的每个图，选择外围区域的节点
        for b in range(batch_size):
            # 获取有效节点
            valid_nodes = torch.where(node_mask[b] > 0)[0].cpu().numpy()
            
            if len(valid_nodes) == 0:
                continue
            
            # 计算要添加噪声的节点数量
            num_noise_nodes = int(len(valid_nodes) * noise_ratio)
            
            if num_noise_nodes == 0:
                continue
            
            # 计算分子的中心点
            coords = x[b].detach().cpu().numpy()
            valid_coords = coords[valid_nodes]
            center = np.mean(valid_coords, axis=0)
            
            # 计算每个节点到中心的距离
            distances = []
            for i, node in enumerate(valid_nodes):
                dist = np.linalg.norm(coords[node] - center)
                distances.append((node, dist))
            
            # 按距离排序（从远到近）
            distances.sort(key=lambda x: x[1], reverse=True)
            
            # 选择最远离中心的节点
            selected_nodes = [node for node, _ in distances[:num_noise_nodes]]
            
            # 更新噪声掩码
            noise_mask[b, selected_nodes] = 1
    
    else:
        raise ValueError(f"不支持的掩码生成策略: {strategy}。可用选项: 'random', 'connected', 'central', 'peripheral'。")
    
    return noise_mask


def get_condition_mask(node_mask, noise_mask):
    """
    获取条件掩码，表示哪些节点用作条件。
    
    参数:
        node_mask (torch.Tensor): 节点掩码，形状为 [batch_size, num_nodes]
        noise_mask (torch.Tensor): 噪声掩码，形状为 [batch_size, num_nodes]
        
    返回:
        torch.Tensor: 条件掩码，形状为 [batch_size, num_nodes]，其中1表示节点将用作条件
    """
    # 条件掩码是有效节点（node_mask=1）且未标记为噪声（noise_mask=0）的节点
    condition_mask = node_mask * (1 - noise_mask)
    
    return condition_mask


def apply_mask_to_data(x, h, node_mask, condition_mask):
    """
    根据条件掩码应用数据掩码。
    
    参数:
        x (torch.Tensor): 节点坐标，形状为 [batch_size, num_nodes, 3]
        h (torch.Tensor): 节点特征，形状为 [batch_size, num_nodes, h_dim]
        node_mask (torch.Tensor): 节点掩码，形状为 [batch_size, num_nodes]
        condition_mask (torch.Tensor): 条件掩码，形状为 [batch_size, num_nodes]
        
    返回:
        tuple: (masked_x, masked_h)，掩码后的节点坐标和特征
    """
    # 扩展条件掩码以匹配坐标和特征维度
    x_mask = condition_mask.unsqueeze(-1).expand_as(x)
    h_mask = condition_mask.unsqueeze(-1).expand_as(h)
    
    # 应用掩码，将非条件节点设为0
    masked_x = x * x_mask
    masked_h = h * h_mask
    
    return masked_x, masked_h


def visualize_masks(x, node_mask, noise_mask=None, condition_mask=None):
    """
    可视化掩码（用于调试）。打印掩码统计信息。
    
    参数:
        x (torch.Tensor): 节点坐标，形状为 [batch_size, num_nodes, 3]
        node_mask (torch.Tensor): 节点掩码，形状为 [batch_size, num_nodes]
        noise_mask (torch.Tensor, optional): 噪声掩码，形状为 [batch_size, num_nodes]
        condition_mask (torch.Tensor, optional): 条件掩码，形状为 [batch_size, num_nodes]
    """
    batch_size, num_nodes, _ = x.shape
    
    print(f"批次大小: {batch_size}, 每个图的节点数: {num_nodes}")
    
    # 节点掩码统计
    valid_nodes = node_mask.sum(dim=1).cpu().numpy()
    print(f"有效节点平均数: {valid_nodes.mean():.2f}, 最小: {valid_nodes.min()}, 最大: {valid_nodes.max()}")
    
    # 噪声掩码统计（如果提供）
    if noise_mask is not None:
        noise_nodes = noise_mask.sum(dim=1).cpu().numpy()
        print(f"噪声节点平均数: {noise_nodes.mean():.2f}, 最小: {noise_nodes.min()}, 最大: {noise_nodes.max()}")
        
        # 计算噪声比例
        noise_ratio = noise_nodes / valid_nodes
        print(f"噪声比例平均值: {noise_ratio.mean():.2f}, 最小: {noise_ratio.min():.2f}, 最大: {noise_ratio.max():.2f}")
    
    # 条件掩码统计（如果提供）
    if condition_mask is not None:
        condition_nodes = condition_mask.sum(dim=1).cpu().numpy()
        print(f"条件节点平均数: {condition_nodes.mean():.2f}, 最小: {condition_nodes.min()}, 最大: {condition_nodes.max()}")
        
        # 计算条件比例
        condition_ratio = condition_nodes / valid_nodes
        print(f"条件比例平均值: {condition_ratio.mean():.2f}, 最小: {condition_ratio.min():.2f}, 最大: {condition_ratio.max():.2f}") 