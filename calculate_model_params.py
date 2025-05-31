#!/usr/bin/env python3
"""
计算GeoLDM模型的总参数量
基于train_scaffold.sh中的参数设置和模型架构
"""

import torch
import torch.nn as nn
import math

def calculate_linear_params(in_features, out_features, bias=True):
    """计算Linear层的参数量"""
    params = in_features * out_features
    if bias:
        params += out_features
    return params

def calculate_gcl_params(input_nf, output_nf, hidden_nf, edges_in_d=0, attention=False):
    """计算GCL层的参数量"""
    # Edge MLP: (input_nf * 2 + edges_in_d) -> hidden_nf -> hidden_nf
    edge_mlp_params = (
        calculate_linear_params(input_nf * 2 + edges_in_d, hidden_nf) +
        calculate_linear_params(hidden_nf, hidden_nf)
    )
    
    # Node MLP: (hidden_nf + input_nf) -> hidden_nf -> output_nf
    node_mlp_params = (
        calculate_linear_params(hidden_nf + input_nf, hidden_nf) +
        calculate_linear_params(hidden_nf, output_nf)
    )
    
    # Attention MLP (如果使用)
    att_mlp_params = 0
    if attention:
        att_mlp_params = calculate_linear_params(hidden_nf, 1)
    
    return edge_mlp_params + node_mlp_params + att_mlp_params

def calculate_equivariant_update_params(hidden_nf, edges_in_d=1):
    """计算EquivariantUpdate层的参数量"""
    # Coord MLP: (hidden_nf * 2 + edges_in_d) -> hidden_nf -> hidden_nf -> 1
    coord_mlp_params = (
        calculate_linear_params(hidden_nf * 2 + edges_in_d, hidden_nf) +
        calculate_linear_params(hidden_nf, hidden_nf) +
        calculate_linear_params(hidden_nf, 1, bias=False)  # 最后一层没有bias
    )
    return coord_mlp_params

def calculate_equivariant_block_params(hidden_nf, n_sublayers=1, attention=True, sin_embedding=False):
    """计算EquivariantBlock的参数量"""
    # 边特征维度
    edge_feat_nf = 2
    if sin_embedding:
        edge_feat_nf = 64 * 2  # SinusoidsEmbeddingNew的输出维度
    
    total_params = 0
    
    # n_sublayers个GCL层
    for i in range(n_sublayers):
        total_params += calculate_gcl_params(
            hidden_nf, hidden_nf, hidden_nf, 
            edges_in_d=edge_feat_nf, attention=attention
        )
    
    # 1个EquivariantUpdate层
    total_params += calculate_equivariant_update_params(hidden_nf, edges_in_d=edge_feat_nf)
    
    return total_params

def calculate_egnn_params(in_node_nf, out_node_nf, hidden_nf, n_layers, 
                         inv_sublayers=1, attention=True, sin_embedding=False):
    """计算EGNN的参数量"""
    total_params = 0
    
    # Embedding层: in_node_nf -> hidden_nf
    total_params += calculate_linear_params(in_node_nf, hidden_nf)
    
    # n_layers个EquivariantBlock
    for i in range(n_layers):
        total_params += calculate_equivariant_block_params(
            hidden_nf, n_sublayers=inv_sublayers, 
            attention=attention, sin_embedding=sin_embedding
        )
    
    # Embedding_out层: hidden_nf -> out_node_nf
    total_params += calculate_linear_params(hidden_nf, out_node_nf)
    
    # SinusoidsEmbeddingNew (如果使用)
    if sin_embedding:
        # SinusoidsEmbeddingNew没有可训练参数，只是固定的正弦嵌入
        pass
    
    return total_params

def calculate_encoder_params(in_node_nf, out_node_nf, hidden_nf, n_layers=1, 
                           inv_sublayers=1, attention=True, sin_embedding=False):
    """计算Encoder的参数量"""
    # EGNN部分: in_node_nf -> hidden_nf (输出到隐藏空间)
    egnn_params = calculate_egnn_params(
        in_node_nf, hidden_nf, hidden_nf, n_layers,
        inv_sublayers, attention, sin_embedding
    )
    
    # Final MLP: hidden_nf -> hidden_nf -> (out_node_nf * 2 + 1)
    final_mlp_params = (
        calculate_linear_params(hidden_nf, hidden_nf) +
        calculate_linear_params(hidden_nf, out_node_nf * 2 + 1)
    )
    
    return egnn_params + final_mlp_params

def calculate_decoder_params(in_node_nf, out_node_nf, hidden_nf, n_layers, 
                           inv_sublayers=1, attention=True, sin_embedding=False):
    """计算Decoder的参数量"""
    # EGNN部分: in_node_nf -> out_node_nf (直接输出到目标维度)
    egnn_params = calculate_egnn_params(
        in_node_nf, out_node_nf, hidden_nf, n_layers,
        inv_sublayers, attention, sin_embedding
    )
    
    # Decoder没有额外的final MLP，EGNN直接输出
    return egnn_params

def calculate_dynamics_params(in_node_nf, hidden_nf, n_layers, 
                            inv_sublayers=1, attention=True, sin_embedding=False):
    """计算Dynamics模型的参数量"""
    return calculate_egnn_params(
        in_node_nf, in_node_nf, hidden_nf, n_layers,
        inv_sublayers, attention, sin_embedding
    )

def main():
    print("=" * 60)
    print("GeoLDM模型参数量计算")
    print("=" * 60)
    
    # 从train_scaffold.sh和默认参数中获取的配置
    print("\n配置参数:")
    print("-" * 40)
    
    # 基础参数
    nf = 192  # hidden_nf
    n_layers = 9
    latent_nf = 1
    inv_sublayers = 1  # 默认值
    attention = True  # 默认值
    sin_embedding = False  # 默认值
    include_charges = True  # 默认值
    
    # QM9数据集参数
    atom_types = 5  # C, N, O, F, H
    in_node_nf = atom_types + int(include_charges)  # 5 + 1 = 6
    
    print(f"hidden_nf (nf): {nf}")
    print(f"n_layers: {n_layers}")
    print(f"latent_nf: {latent_nf}")
    print(f"inv_sublayers: {inv_sublayers}")
    print(f"attention: {attention}")
    print(f"sin_embedding: {sin_embedding}")
    print(f"in_node_nf: {in_node_nf}")
    
    print("\n模型组件参数量:")
    print("-" * 40)
    
    # 1. VAE Encoder
    encoder_params = calculate_encoder_params(
        in_node_nf=in_node_nf,
        out_node_nf=latent_nf,
        hidden_nf=nf,
        n_layers=1,  # encoder只有1层
        inv_sublayers=inv_sublayers,
        attention=attention,
        sin_embedding=sin_embedding
    )
    print(f"VAE Encoder: {encoder_params:,} 参数")
    
    # 2. VAE Decoder
    decoder_params = calculate_decoder_params(
        in_node_nf=latent_nf,
        out_node_nf=in_node_nf,
        hidden_nf=nf,
        n_layers=n_layers,
        inv_sublayers=inv_sublayers,
        attention=attention,
        sin_embedding=sin_embedding
    )
    print(f"VAE Decoder: {decoder_params:,} 参数")
    
    # 3. Diffusion Dynamics
    # condition_time=True, 所以输入维度是latent_nf + 1
    dynamics_in_node_nf = latent_nf + 1
    dynamics_params = calculate_dynamics_params(
        in_node_nf=dynamics_in_node_nf,
        hidden_nf=nf,
        n_layers=n_layers,
        inv_sublayers=inv_sublayers,
        attention=attention,
        sin_embedding=sin_embedding
    )
    print(f"Diffusion Dynamics: {dynamics_params:,} 参数")
    
    # 总参数量
    total_params = encoder_params + decoder_params + dynamics_params
    
    print("\n总参数量:")
    print("=" * 40)
    print(f"VAE (Encoder + Decoder): {encoder_params + decoder_params:,} 参数")
    print(f"Diffusion Model: {dynamics_params:,} 参数")
    print(f"总计: {total_params:,} 参数")
    print(f"总计 (百万): {total_params / 1e6:.2f}M 参数")
    
    print("\n详细分解:")
    print("-" * 40)
    
    # 详细分解一个EGNN块的参数
    print(f"\n单个EquivariantBlock参数量 (hidden_nf={nf}):")
    block_params = calculate_equivariant_block_params(nf, inv_sublayers, attention, sin_embedding)
    print(f"  - {block_params:,} 参数")
    
    # GCL层参数
    edge_feat_nf = 2 if not sin_embedding else 128
    gcl_params = calculate_gcl_params(nf, nf, nf, edge_feat_nf, attention)
    equi_update_params = calculate_equivariant_update_params(nf, edge_feat_nf)
    
    print(f"\n单个GCL层参数量:")
    print(f"  - Edge MLP: {calculate_linear_params(nf * 2 + edge_feat_nf, nf) + calculate_linear_params(nf, nf):,}")
    print(f"  - Node MLP: {calculate_linear_params(nf + nf, nf) + calculate_linear_params(nf, nf):,}")
    if attention:
        print(f"  - Attention MLP: {calculate_linear_params(nf, 1):,}")
    print(f"  - 总计: {gcl_params:,}")
    
    print(f"\nEquivariantUpdate层参数量:")
    print(f"  - Coord MLP: {equi_update_params:,}")

if __name__ == "__main__":
    main() 