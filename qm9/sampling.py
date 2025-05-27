import numpy as np
import torch
import torch.nn.functional as F
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked, sample_center_gravity_zero_gaussian_with_mask, sample_gaussian_with_mask
from qm9.analyze import check_stability
import qm9.utils as qm9utils

def rotate_chain(z):
    assert z.size(0) == 1

    z_h = z[:, :, 3:]

    n_steps = 30
    theta = 0.6 * np.pi / n_steps
    Qz = torch.tensor(
        [[np.cos(theta), -np.sin(theta), 0.],
         [np.sin(theta), np.cos(theta), 0.],
         [0., 0., 1.]]
    ).float()
    Qx = torch.tensor(
        [[1., 0., 0.],
         [0., np.cos(theta), -np.sin(theta)],
         [0., np.sin(theta), np.cos(theta)]]
    ).float()
    Qy = torch.tensor(
        [[np.cos(theta), 0., np.sin(theta)],
         [0., 1., 0.],
         [-np.sin(theta), 0., np.cos(theta)]]
    ).float()

    Q = torch.mm(torch.mm(Qz, Qx), Qy)

    Q = Q.to(z.device)

    results = []
    results.append(z)
    for i in range(n_steps):
        z_x = results[-1][:, :, :3]
        # print(z_x.size(), Q.size())
        new_x = torch.matmul(z_x.view(-1, 3), Q.T).view(1, -1, 3)
        # print(new_x.size())
        new_z = torch.cat([new_x, z_h], dim=2)
        results.append(new_z)

    results = torch.cat(results, dim=0)
    return results


def reverse_tensor(x):
    return x[torch.arange(x.size(0) - 1, -1, -1)]


def sample_chain(args, device, flow, n_tries, dataset_info, prop_dist=None):
    n_samples = 1
    if args.dataset == 'qm9' or args.dataset == 'qm9_second_half' or args.dataset == 'qm9_first_half':
        n_nodes = 19
    elif args.dataset == 'geom':
        n_nodes = 44
    else:
        raise ValueError()

    # TODO FIX: This conditioning just zeros.
    if args.context_node_nf > 0:
        context = prop_dist.sample(n_nodes).unsqueeze(1).unsqueeze(0)
        context = context.repeat(1, n_nodes, 1).to(device)
        #context = torch.zeros(n_samples, n_nodes, args.context_node_nf).to(device)
    else:
        context = None

    node_mask = torch.ones(n_samples, n_nodes, 1).to(device)

    edge_mask = (1 - torch.eye(n_nodes)).unsqueeze(0)
    edge_mask = edge_mask.repeat(n_samples, 1, 1).view(-1, 1).to(device)

    if args.probabilistic_model == 'diffusion':
        one_hot, charges, x = None, None, None
        for i in range(n_tries):
            chain = flow.sample_chain(n_samples, n_nodes, node_mask, edge_mask, context, keep_frames=100)
            chain = reverse_tensor(chain)

            # Repeat last frame to see final sample better.
            chain = torch.cat([chain, chain[-1:].repeat(10, 1, 1)], dim=0)
            x = chain[-1:, :, 0:3]
            one_hot = chain[-1:, :, 3:-1]
            one_hot = torch.argmax(one_hot, dim=2)

            atom_type = one_hot.squeeze(0).cpu().detach().numpy()
            x_squeeze = x.squeeze(0).cpu().detach().numpy()
            mol_stable = check_stability(x_squeeze, atom_type, dataset_info)[0]

            # Prepare entire chain.
            x = chain[:, :, 0:3]
            one_hot = chain[:, :, 3:-1]
            one_hot = F.one_hot(torch.argmax(one_hot, dim=2), num_classes=len(dataset_info['atom_decoder']))
            charges = torch.round(chain[:, :, -1:]).long()

            if mol_stable:
                print('Found stable molecule to visualize :)')
                break
            elif i == n_tries - 1:
                print('Did not find stable molecule, showing last sample.')

    else:
        raise ValueError

    return one_hot, charges, x


def sample(args, device, generative_model, dataset_info,
           prop_dist=None, nodesxsample=torch.tensor([10]), context=None,
           fix_noise=False):
    max_n_nodes = dataset_info['max_n_nodes']  # this is the maximum node_size in QM9

    assert int(torch.max(nodesxsample)) <= max_n_nodes
    batch_size = len(nodesxsample)

    node_mask = torch.zeros(batch_size, max_n_nodes)
    for i in range(batch_size):
        node_mask[i, 0:nodesxsample[i]] = 1

    # Compute edge_mask

    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.view(batch_size * max_n_nodes * max_n_nodes, 1).to(device)
    node_mask = node_mask.unsqueeze(2).to(device)

    # TODO FIX: This conditioning just zeros.
    if args.context_node_nf > 0:
        if context is None:
            context = prop_dist.sample_batch(nodesxsample)
        context = context.unsqueeze(1).repeat(1, max_n_nodes, 1).to(device) * node_mask
    else:
        context = None

    if args.probabilistic_model == 'diffusion':
        x, h = generative_model.sample(batch_size, max_n_nodes, node_mask, edge_mask, context, fix_noise=fix_noise)

        assert_correctly_masked(x, node_mask)
        assert_mean_zero_with_mask(x, node_mask)

        one_hot = h['categorical']
        charges = h['integer']

        assert_correctly_masked(one_hot.float(), node_mask)
        if args.include_charges:
            assert_correctly_masked(charges.float(), node_mask)

    else:
        raise ValueError(args.probabilistic_model)

    return one_hot, charges, x, node_mask


def sample_scaf(args, device, generative_model, dataset_info,
                prop_dist=None, nodesxsample=torch.tensor([10]), context=None,
                fix_noise=False, loader=None, dtype=None, mask_func=None, property_norms=None):
    for i, data in enumerate(loader):
        x = data['positions'].to(device, dtype)
        batch_size = x.size(0)
        node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
        edge_mask = data['edge_mask'].to(device, dtype)
        edge_mask = edge_mask.view(batch_size, -1)
        one_hot = data['one_hot'].to(device, dtype)
        charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

        batch_size = 1
        n_nodes = node_mask.size(1)
        x = x[:batch_size]
        node_mask = node_mask[:batch_size]
        edge_mask = edge_mask[:batch_size]
        one_hot = one_hot[:batch_size]
        charges = charges[:batch_size]

        if args.augment_noise > 0:
            # Add noise eps ~ N(0, augment_noise) around points.
            eps = sample_center_gravity_zero_gaussian_with_mask(x.size(),
                                                                x.device,
                                                                node_mask)
            x = x + eps * args.augment_noise

        x = remove_mean_with_mask(x, node_mask)
        # check_mask_correct([x, one_hot, charges], node_mask)
        # assert_mean_zero_with_mask(x, node_mask)
        h = {'categorical': one_hot, 'integer': charges}

        # 获取原始上下文条件
        # if len(args.conditioning) > 0:
        #     if context is None:
        #         context = prop_dist.sample_batch(nodesxsample)
        #     context = context.unsqueeze(1).repeat(1, x.shape[1], 1).to(device) * node_mask
        # else:
        #     context = None
        if len(args.conditioning) > 0:
            assert property_norms is not None
            context = qm9utils.prepare_context(args.conditioning, data, property_norms).to(device, dtype)
            assert_correctly_masked(context, node_mask)
        else:
            context = None
            
        if args.partial_conditioning:
            # 使用固定的随机种子来保证测试的一致性
            test_seed = None
            
            # 根据策略生成掩码
            if args.mask_strategy == "connected":
                # 使用连通掩码生成
                noise_mask, condition_mask = mask_func(
                    x, node_mask, noise_ratio=args.noise_ratio, random_seed=test_seed)
            else:
                # 使用其他策略
                from mask_utils import generate_mask_by_strategy, get_condition_mask
                noise_mask = generate_mask_by_strategy(
                    x, h['categorical'], node_mask.squeeze(2), 
                    strategy=args.mask_strategy, 
                    noise_ratio=args.noise_ratio,
                    random_seed=test_seed)
                # 根据噪声掩码获取条件掩码
                condition_mask = get_condition_mask(node_mask.squeeze(2), noise_mask).unsqueeze(2)
                noise_mask = noise_mask.unsqueeze(2)

            edge_mask = node_mask.permute(0, 2, 1) * noise_mask
            edge_mask = edge_mask.view(batch_size * n_nodes * n_nodes, 1)
            scaf_x_0 = x[0].cpu()
            scaf_h_0 = torch.argmax(h['categorical'][0], dim=1).cpu()
            scaf_x = []
            scaf_h = []
            for __ in range(scaf_x_0.shape[0]):
                if condition_mask[0, __, 0] == 1:
                    scaf_x.append([float(___) for ___ in scaf_x_0[__]])
                    scaf_h.append(int(scaf_h_0[__]))
            scaf_x = torch.tensor(scaf_x)
            scaf_h = torch.tensor(scaf_h)
        else:
            raise ValueError("args.partial_conditioning is not True")

        z_x_mu, z_x_sigma, z_h_mu, z_h_sigma =  generative_model.vae.encode(x, h, node_mask, edge_mask, context)
        # 获取编码后的潜变量表示
        encoded_condition_x = z_x_mu  # [1, n_nodes, 3]
        encoded_condition_h = z_h_mu  # [1, n_nodes, latent_nf]

        condition_x = []
        condition_h = []
        for __ in range(condition_mask.shape[1]):
            if condition_mask[0, __, 0] == 1:
                condition_x.append([float(___) for ___ in encoded_condition_x[0, __, :]])
                condition_h.append([float(___) for ___ in encoded_condition_h[0, __, :]])
        condition_x = torch.tensor(condition_x).to(device, dtype)
        condition_x = condition_x - torch.mean(condition_x, dim=0, keepdim=True)
        condition_h = torch.tensor(condition_h).to(device, dtype)
        condition_x = condition_x.unsqueeze(0)
        condition_h = condition_h.unsqueeze(0)

        noise_mask = node_mask.clone()
        condition_mask = torch.zeros_like(node_mask).to(device, dtype)
        for i in range(batch_size):
            noise_mask[i, 0:condition_x.size(1), 0] = 0
            condition_mask[i, 0:condition_x.size(1), 0] = 1
        # Compute edge_mask
        edge_mask = node_mask.permute(0, 2, 1) * noise_mask
        edge_mask = edge_mask.view(batch_size * n_nodes * n_nodes, 1)

        if args.probabilistic_model == 'diffusion':
            x, h, draw_xh = generative_model.sample_scaf(batch_size, n_nodes, node_mask, edge_mask, context, 
                                                         condition_x=condition_x, condition_h=condition_h, 
                                                         noise_mask=noise_mask, condition_mask=condition_mask)

            assert_correctly_masked(x, node_mask)
            # assert_mean_zero_with_mask(x, node_mask)

            one_hot = h['categorical']
            charges = h['integer']
            
            draw_xh = [(xh[0].detach().cpu(), xh[1]['categorical'].detach().cpu(), xh[1]['integer'].detach().cpu()) for xh in draw_xh]

            assert_correctly_masked(one_hot.float(), node_mask)
            if args.include_charges:
                assert_correctly_masked(charges.float(), node_mask)
        else:
            raise ValueError(args.probabilistic_model)

        one_hot = one_hot.detach().cpu()
        x = x.detach().cpu()
        node_mask = node_mask.detach().cpu()
        charges = charges.detach().cpu()
        break

    return one_hot, charges, x, node_mask, scaf_x, scaf_h, draw_xh


def sample_sweep_conditional(args, device, generative_model, dataset_info, prop_dist, n_nodes=19, n_frames=100):
    nodesxsample = torch.tensor([n_nodes] * n_frames)

    context = []
    for key in prop_dist.distributions:
        min_val, max_val = prop_dist.distributions[key][n_nodes]['params']
        mean, mad = prop_dist.normalizer[key]['mean'], prop_dist.normalizer[key]['mad']
        min_val = (min_val - mean) / (mad)
        max_val = (max_val - mean) / (mad)
        context_row = torch.tensor(np.linspace(min_val, max_val, n_frames)).unsqueeze(1)
        context.append(context_row)
    context = torch.cat(context, dim=1).float().to(device)

    one_hot, charges, x, node_mask = sample(args, device, generative_model, dataset_info, prop_dist, nodesxsample=nodesxsample, context=context, fix_noise=True)
    return one_hot, charges, x, node_mask
