import torch


def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(dim=-1)


def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask)).abs().sum().item() < 1e-8


def compute_loss_and_nll(args, generative_model, nodes_dist, x, h, node_mask, edge_mask, context, noise_mask=None, condition_mask=None):
    bs, n_nodes, n_dims = x.size()

    if args.probabilistic_model == 'diffusion':
        edge_mask = edge_mask.view(bs, n_nodes * n_nodes)

        assert_correctly_masked(x, node_mask)

        # 对于部分条件，x和h已经被条件掩码处理过，
        # 这里传入噪声掩码和条件掩码，以便在扩散模型内部使用
        
        # 注释：在训练过程中，如果启用了partial_conditioning
        # x 中仅包含条件部分的原子（作为骨架）
        # h 也只包含条件部分的特征
        # 这样模型就会学习从骨架生成完整的分子
        
        # Here x is a position tensor, and h is a dictionary with keys
        # 'categorical' and 'integer'.
        if noise_mask is not None:
            nll = generative_model(x, h, node_mask, edge_mask, context, noise_mask, condition_mask)
        else:
            nll = generative_model(x, h, node_mask, edge_mask, context)

        N = node_mask.squeeze(2).sum(1).long()

        log_pN = nodes_dist.log_prob(N)

        assert nll.size() == log_pN.size()
        nll = nll - log_pN

        # Average over batch.
        nll = nll.mean(0)

        reg_term = torch.tensor([0.]).to(nll.device)
        mean_abs_z = 0.
    else:
        raise ValueError(args.probabilistic_model)

    return nll, reg_term, mean_abs_z
