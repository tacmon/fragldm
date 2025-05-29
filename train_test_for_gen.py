import wandb
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked, sample_center_gravity_zero_gaussian_with_mask
import numpy as np
import qm9.visualizer as vis
from qm9.analyze import analyze_stability_for_molecules
import utils
import qm9.utils as qm9utils
from qm9 import losses
import torch
from mask_handler import generate_connected_mask
import os
import numpy as np
import torch
import torch.nn.functional as F
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked, sample_center_gravity_zero_gaussian_with_mask, sample_gaussian_with_mask
from qm9.analyze import check_stability
import qm9.utils as qm9utils

def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def analyze_and_save_scaf(model_sample, nodes_dist, args, device, dataset_info, prop_dist,
                     n_samples=1, batch_size=1, loader=None, dtype=None, property_norms=None,
                     manual_condition_x=None, manual_condition_h=None):
    batch_size = min(batch_size, n_samples)
    assert(batch_size == 1)
    assert n_samples % batch_size == 0
    molecules = {'one_hot': [], 'x': [], 'node_mask': []}
    have_found_stable_molecule = False
    for iiiiii, data in enumerate(loader):
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
        check_mask_correct([x, one_hot, charges], node_mask)
        assert_mean_zero_with_mask(x, node_mask)
        h = {'categorical': one_hot, 'integer': charges}

        if len(args.conditioning) > 0:
            assert property_norms is not None
            context = qm9utils.prepare_context(args.conditioning, data, property_norms).to(device, dtype)
            assert_correctly_masked(context, node_mask)
        else:
            context = None
            
        if args.partial_conditioning:
            # 使用固定的随机种子来保证测试的一致性
            test_seed = None
            
            noise_mask, condition_mask = generate_connected_mask(
                x, node_mask, noise_ratio=args.noise_ratio, random_seed=test_seed)

            edge_mask = node_mask.permute(0, 2, 1) * noise_mask
            for i in range(batch_size):
                for u in range(n_nodes):
                    edge_mask[i, u, u] = 0 # 自己到自己没有边

            edge_mask = edge_mask.view(batch_size * n_nodes * n_nodes, 1)
            scaf_x_0 = x[0].cpu()
            scaf_h_0 = torch.argmax(h['categorical'][0], dim=1).cpu()
            scaf_x = []
            scaf_h = []
            encode_scaf_x = []
            encode_scaf_h = {'categorical': [], 'integer': []}
            for __ in range(scaf_x_0.shape[0]):
                if condition_mask[0, __, 0] == 1:
                    scaf_x.append([float(___) for ___ in scaf_x_0[__]])
                    scaf_h.append(int(scaf_h_0[__]))
                    encode_scaf_x.append([float(___) for ___ in x[0][__]])
                    encode_scaf_h['categorical'].append([float(___) for ___ in h['categorical'][0][__]])
                    encode_scaf_h['integer'].append([float(___) for ___ in h['integer'][0][__]])
            scaf_x = torch.tensor(scaf_x)
            scaf_h = torch.tensor(scaf_h)
            encode_scaf_x = torch.tensor(encode_scaf_x)
            encode_scaf_h['categorical'] = torch.tensor(encode_scaf_h['categorical'])
            encode_scaf_h['integer'] = torch.tensor(encode_scaf_h['integer'])
        else:
            raise ValueError("args.partial_conditioning is not True")

        scaf_x = scaf_x - torch.mean(scaf_x, dim=0, keepdim=True)
        encode_scaf_x = encode_scaf_x - torch.mean(encode_scaf_x, dim=0, keepdim=True)
        # scaf_h = scaf_h - torch.mean(scaf_h, dim=0, keepdim=True)
        encode_scaf_x = encode_scaf_x.unsqueeze(0).to(device, dtype)
        encode_scaf_h['categorical'] = encode_scaf_h['categorical'].unsqueeze(0).to(device, dtype)
        encode_scaf_h['integer'] = encode_scaf_h['integer'].unsqueeze(0).to(device, dtype)
        encode_node_mask = torch.ones((1, encode_scaf_x.shape[1], 1), device=device, dtype=dtype)
        encode_edge_mask = torch.ones((1, encode_scaf_x.shape[1], encode_scaf_x.shape[1]), device=device, dtype=dtype)
        for i in range(encode_scaf_x.shape[1]):
            encode_edge_mask[0, i, i] = 0
        encode_edge_mask = encode_edge_mask.view(1, encode_scaf_x.shape[1] * encode_scaf_x.shape[1], 1)
        
        if context is not None:
            context = context[:, :1, :1].repeat(1, encode_scaf_x.shape[1], 1)
        z_x_mu, z_x_sigma, z_h_mu, z_h_sigma =  model_sample.vae.encode(encode_scaf_x, encode_scaf_h, encode_node_mask, encode_edge_mask, context)
        # 获取编码后的潜变量表示
        condition_x = z_x_mu
        condition_h = z_h_mu

        # z_x_mu, z_x_sigma, z_h_mu, z_h_sigma =  model_sample.vae.encode(x, h, node_mask, edge_mask, context)
        # # 获取编码后的潜变量表示
        # encoded_condition_x = z_x_mu  # [1, n_nodes, 3]
        # encoded_condition_h = z_h_mu  # [1, n_nodes, latent_nf]

        # condition_x = []
        # condition_h = []
        # for __ in range(condition_mask.shape[1]):
        #     if condition_mask[0, __, 0] == 1:
        #         condition_x.append([float(___) for ___ in encoded_condition_x[0, __, :]])
        #         condition_h.append([float(___) for ___ in encoded_condition_h[0, __, :]])
        # condition_x = torch.tensor(condition_x).to(device, dtype)
        # condition_x = condition_x - torch.mean(condition_x, dim=0, keepdim=True)
        # condition_h = torch.tensor(condition_h).to(device, dtype)
        # condition_x = condition_x.unsqueeze(0)
        # condition_h = condition_h.unsqueeze(0)

        frag_gen_dir = 'outputs/%s' % (args.exp_name)
        frag_gen_dir = frag_gen_dir + "/fragment"
        os.makedirs(frag_gen_dir, exist_ok=True)
        # 保存输入片段为图像
        vis.plot_data3d(scaf_x.cpu(), scaf_h.cpu(), dataset_info, spheres_3d=True, 
                        save_path=os.path.join(frag_gen_dir, 'fragment.png'))
        
        # 保存输入片段为XYZ文件
        # 注意：根据scaf_x和scaf_h的确切形状，可能需要调整维度
        if len(scaf_x.shape) == 2:  # 如果没有批次维度 [n_nodes, 3]
            scaf_x_batch = scaf_x.unsqueeze(0)  # 添加批次维度 [1, n_nodes, 3]
            scaf_h_batch = scaf_h.unsqueeze(0) if len(scaf_h.shape) == 1 else scaf_h  # 对原子类型也做相同处理
        else:
            scaf_x_batch = scaf_x
            scaf_h_batch = scaf_h
        
        # 将原子类型索引转换为one-hot编码
        num_atom_types = len(dataset_info['atom_decoder'])  # 获取原子类型数量
        # 确保scaf_h_batch是long类型，这是scatter_需要的
        scaf_h_long = scaf_h_batch.long() if torch.is_tensor(scaf_h_batch) else torch.tensor(scaf_h_batch, dtype=torch.long)
        
        # 创建one-hot张量
        if len(scaf_h_long.shape) == 2:  # [batch_size, n_nodes]
            scaf_one_hot = torch.zeros(scaf_h_long.shape[0], scaf_h_long.shape[1], num_atom_types)
            # 使用scatter_填充one-hot编码
            scaf_one_hot.scatter_(2, scaf_h_long.unsqueeze(-1), 1)
        else:  # 如果scaf_h_long已经是3D张量，直接使用
            scaf_one_hot = scaf_h_long
        
        vis.save_xyz_file(
            path=frag_gen_dir + "/", 
            one_hot=scaf_one_hot.cpu(),  # 使用转换后的one-hot表示
            charges=None,  # charges不需要提供，因为我们已经提供了one_hot
            positions=scaf_x_batch.cpu(),  # 片段原子坐标
            dataset_info=dataset_info,
            name='fragment'
        )

        for i in range(int(n_samples/batch_size)):
            # nodesxsample = nodes_dist.sample(batch_size)
            nodesxsample = node_mask.shape[1]
            node_mask = torch.ones((batch_size, nodesxsample, 1), device=device)
            noise_mask = torch.ones((batch_size, nodesxsample, 1), device=device)
            condition_mask = torch.zeros((batch_size, nodesxsample, 1), device=device)

            for j in range(batch_size):
                noise_mask[j, 0:condition_x.size(1), 0] = 0
                condition_mask[j, 0:condition_x.size(1), 0] = 1

            edge_mask = torch.zeros((batch_size, nodesxsample, nodesxsample), device=device)
            for j in range(batch_size):
                for u in range(nodesxsample):
                    for v in range(nodesxsample):
                        # u是终点，v是起点
                        # edge_mask[j, u, v] = (u != v) and (condition_mask[j, v, 0] == 1 or noise_mask[j, u, 0] == 1)
                        edge_mask[j, u, v] = noise_mask[j, u, 0]
            edge_mask = edge_mask.view(batch_size * nodesxsample * nodesxsample, 1)

            if context is not None:
                context = context[:, :1, :1].repeat(1, nodesxsample, 1)

            if args.probabilistic_model == 'diffusion':
                x, h, draw_xh = model_sample.sample_scaf(batch_size, nodesxsample, node_mask, edge_mask, context, 
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
            charges = charges.detach().cpu()

            molecules['one_hot'].append(one_hot.detach().cpu())
            molecules['x'].append(x.detach().cpu())
            molecules['node_mask'].append(node_mask.detach().cpu())

            # 检查当前分子的稳定性/有效性
            current_mol = {
                'one_hot': one_hot.detach().cpu(), 
                'x': x.detach().cpu(), 
                'node_mask': node_mask.detach().cpu()
            }
            # 调用分析函数检查单个分子的稳定性
            mol_validity_dict, rdkit_mol_tuple = analyze_stability_for_molecules(current_mol, dataset_info)
            
            # 如果RDKit分析可用，进一步确认分子有效性
            is_valid_by_rdkit = False
            if rdkit_mol_tuple is not None:
                is_valid_by_rdkit = rdkit_mol_tuple[0][0] == 1.0
            
            print(f"分子 {i+1}/{int(n_samples/batch_size)} - 原子稳定性: {mol_validity_dict['atm_stable']*100:.1f}%, "
                f"分子稳定性: {mol_validity_dict['mol_stable']*100:.1f}%, "
                f"RDKit有效: {is_valid_by_rdkit}")

            if is_valid_by_rdkit:
                have_found_stable_molecule = True
                gen_dir = 'outputs/%s' % (args.exp_name)
                gen_dir = gen_dir + "/%d" % (i)
                os.makedirs(gen_dir, exist_ok=True)
                
                # 保存SMILES字符串到文件
                if rdkit_mol_tuple is not None and len(rdkit_mol_tuple) > 1:
                    smiles_save_file = os.path.join(gen_dir, 'smiles.txt')
                    # 检查rdkit_mol_tuple[1]的类型并适当处理
                    if isinstance(rdkit_mol_tuple[1], str):
                        # 直接写入单个SMILES字符串
                        with open(smiles_save_file, 'w') as f:
                            f.write(rdkit_mol_tuple[1])
                        print(f"保存的SMILES: {rdkit_mol_tuple[1]}")
                    elif isinstance(rdkit_mol_tuple[1], list) or isinstance(rdkit_mol_tuple[1], tuple):
                        # 如果是多个SMILES的列表，逐行写入
                        with open(smiles_save_file, 'w') as f:
                            for smiles in rdkit_mol_tuple[1]:
                                if smiles:  # 确保不是空字符串
                                    f.write(f"{smiles}\n")
                        print(f"保存了 {len(rdkit_mol_tuple[1])} 个SMILES字符串")
                    else:
                        print(f"警告: 无法识别的SMILES格式: {type(rdkit_mol_tuple[1])}")
                        # 尝试直接字符串转换
                        with open(smiles_save_file, 'w') as f:
                            f.write(str(rdkit_mol_tuple[1]))
                else:
                    print("未找到可保存的SMILES字符串")
                
                # 保存最终生成的分子为图像
                vis.plot_data3d(x[0].cpu(), torch.argmax(one_hot[0], dim=1).cpu(), dataset_info, spheres_3d=True, 
                                save_path=os.path.join(gen_dir, 'final.png'))
                
                # 保存最终生成的分子为XYZ文件
                vis.save_xyz_file(
                    path=gen_dir + "/", 
                    one_hot=one_hot[0:1].cpu(),  # 使用第一个分子，添加批次维度 [1, n_nodes, num_atom_types]
                    charges=charges[0:1].cpu() if charges is not None else None,  # 如果有电荷信息
                    positions=x[0:1].cpu(),  # 原子坐标
                    dataset_info=dataset_info,
                    node_mask=node_mask[0:1].cpu(),
                    name='final'
                )
                
                # 保存中间生成步骤的分子
                for j in range(len(draw_xh)):
                    # 保存中间步骤为图像
                    vis.plot_data3d(draw_xh[j][0][0].cpu(), torch.argmax(draw_xh[j][1][0].cpu(), dim=1).cpu(), dataset_info, spheres_3d=True, 
                                    save_path=os.path.join(gen_dir, '%d.png' % j))
                    
                    # 保存中间步骤为XYZ文件
                    vis.save_xyz_file(
                        path=gen_dir + "/", 
                        one_hot=draw_xh[j][1].cpu(),  # 应该已经包含批次维度
                        charges=None,  # 如果没有电荷信息
                        positions=draw_xh[j][0].cpu(),  # 原子坐标
                        dataset_info=dataset_info,
                        name='step_%d' % j
                    )
            
        break

    molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
    validity_dict, rdkit_tuple = analyze_stability_for_molecules(molecules, dataset_info)

    wandb.log(validity_dict)
    if rdkit_tuple is not None:
        wandb.log({'Validity': rdkit_tuple[0][0], 'Uniqueness': rdkit_tuple[0][1], 'Novelty': rdkit_tuple[0][2]})
    return validity_dict
