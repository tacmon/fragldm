import wandb
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked, sample_center_gravity_zero_gaussian_with_mask
import numpy as np
import qm9.visualizer as vis
from qm9.analyze import analyze_stability_for_molecules
from qm9.sampling import sample_chain, sample, sample_sweep_conditional, sample_scaf
import utils
import qm9.utils as qm9utils
from qm9 import losses
import torch
from mask_handler import generate_connected_mask
import os


def train_epoch(args, loader, epoch, model, model_dp, model_ema, ema, device, dtype, property_norms, optim,
                nodes_dist, gradnorm_queue, dataset_info, prop_dist):
    model_dp.train()
    model.train()
    nll_epoch = []
    n_iterations = len(loader)
    
    # 输出训练配置信息
    if epoch == 0:
        print(f"\n{'=' * 50}")
        print(f"训练配置:")
        print(f"- 模型: {args.model}")
        print(f"- 数据集: {args.dataset}")
        print(f"- 条件: {args.conditioning if len(args.conditioning) > 0 else '无'}")
        if args.partial_conditioning:
            print(f"- 部分条件化: 已启用")
            print(f"  - 噪声比例: {args.noise_ratio}")
            print(f"  - 掩码策略: {args.mask_strategy}")
        else:
            print(f"- 部分条件化: 未启用")
        print(f"{'=' * 50}\n")
    
    for i, data in enumerate(loader):
        x = data['positions'].to(device, dtype)
        node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
        edge_mask = data['edge_mask'].to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)
        charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

        x = remove_mean_with_mask(x, node_mask)

        if args.augment_noise > 0:
            # Add noise eps ~ N(0, augment_noise) around points.
            eps = sample_center_gravity_zero_gaussian_with_mask(x.size(), x.device, node_mask)
            x = x + eps * args.augment_noise

        x = remove_mean_with_mask(x, node_mask)
        if args.data_augmentation:
            x = utils.random_rotation(x).detach()

        check_mask_correct([x, one_hot, charges], node_mask)
        assert_mean_zero_with_mask(x, node_mask)

        h = {'categorical': one_hot, 'integer': charges}

        # 获取原始上下文条件
        if len(args.conditioning) > 0:
            context = qm9utils.prepare_context(args.conditioning, data, property_norms).to(device, dtype)
            assert_correctly_masked(context, node_mask)
        else:
            context = None

        # 应用部分条件掩码（如果启用）
        condition_stats = {}
        if args.partial_conditioning:
            # 根据策略生成掩码
            if args.mask_strategy == "connected":
                # 使用连通掩码生成
                noise_mask, condition_mask = generate_connected_mask(
                    x, node_mask, noise_ratio=args.noise_ratio)
                # 画图
                # flag_0 = condition_mask[0]
                # scaf_x_0 = x[0].cpu()
                # scaf_h_0 = torch.argmax(h['categorical'][0], dim=1).cpu()
                # scaf_x = []
                # scaf_h = []
                # for __ in range(flag_0.shape[0]):
                #     if flag_0[__][0] == 1:
                #         scaf_x.append([float(___) for ___ in scaf_x_0[__]])
                #         scaf_h.append(int(scaf_h_0[__]))
                # print(scaf_x)
                # print(scaf_h)
                # scaf_x = torch.tensor(scaf_x)
                # scaf_h = torch.tensor(scaf_h)
                # vis.plot_data3d(scaf_x, scaf_h, dataset_info, spheres_3d=True, save_path="/app/tacmon-geoldm/outputs/fragment_gen/1.png")
                # vis.plot_data3d(x[0].cpu(), torch.argmax(h['categorical'][0], dim=1).cpu(), dataset_info, spheres_3d=True, save_path="/app/tacmon-geoldm/outputs/fragment_gen/2.png")
            else:
                # 使用其他策略
                from mask_utils import generate_mask_by_strategy, get_condition_mask
                noise_mask = generate_mask_by_strategy(
                    x, h['categorical'], node_mask.squeeze(2), 
                    strategy=args.mask_strategy, 
                    noise_ratio=args.noise_ratio)
                # 根据噪声掩码获取条件掩码
                condition_mask = get_condition_mask(node_mask.squeeze(2), noise_mask).unsqueeze(2)
                noise_mask = noise_mask.unsqueeze(2)
            
            # 改变edge_mask删除连接向condition_mask的节点
            # edge_mask = node_mask将第二维移动到第三维 * noise_mask
            # edge_mask = node_mask.permute(0, 2, 1) * noise_mask
            batch_size = x.size(0)           
            n_nodes = node_mask.size(1)
            edge_mask = torch.zeros((batch_size, n_nodes, n_nodes), device=device, dtype=dtype)
            for b in range(batch_size):
                for u in range(n_nodes):
                    for v in range(n_nodes):
                        if condition_mask[b, v, 0] == 1 or noise_mask[b, u, 0] == 1:
                            edge_mask[b, u, v] = 1
            edge_mask = edge_mask.view(batch_size * n_nodes * n_nodes, 1)

            # 计算条件统计信息
            valid_nodes = node_mask.sum(dim=(1,2)).float()
            noise_nodes = noise_mask.sum(dim=(1,2)).float()
            condition_nodes = condition_mask.sum(dim=(1,2)).float()
            
            avg_valid = valid_nodes.mean().item()
            avg_noise = noise_nodes.mean().item()
            avg_condition = condition_nodes.mean().item()
            avg_noise_ratio = (noise_nodes / valid_nodes).mean().item()
            avg_condition_ratio = (condition_nodes / valid_nodes).mean().item()
            
            condition_stats = {
                'avg_valid_nodes': avg_valid,
                'avg_noise_nodes': avg_noise,
                'avg_condition_nodes': avg_condition,
                'avg_noise_ratio': avg_noise_ratio,
                'avg_condition_ratio': avg_condition_ratio
            }

        else:
            noise_mask = None
            condition_mask = None
        
        optim.zero_grad()

        # transform batch through flow
        nll, reg_term, mean_abs_z = losses.compute_loss_and_nll(args, model_dp, nodes_dist,
                                                                x, h, node_mask, edge_mask, context, 
                                                                noise_mask=noise_mask, condition_mask=condition_mask)
        # standard nll from forward KL
        loss = nll + args.ode_regularization * reg_term
        loss.backward()

        if args.clip_grad:
            grad_norm = utils.gradient_clipping(model, gradnorm_queue)
        else:
            grad_norm = 0.

        optim.step()

        # Update EMA if enabled.
        if args.ema_decay > 0:
            ema.update_model_average(model_ema, model)

        if i % args.n_report_steps == 0:
            # 输出基本训练信息
            print(f"\rEpoch: {epoch}, iter: {i}/{n_iterations}, "
                  f"Loss {loss.item():.2f}, NLL: {nll.item():.2f}, "
                  f"RegTerm: {reg_term.item():.1f}, "
                  f"GradNorm: {grad_norm:.1f}", end="")
            
            # 如果启用了部分条件化，输出相关统计信息
            if args.partial_conditioning and condition_stats:
                print(f" | Cond: {condition_stats['avg_condition_ratio']:.2f}, "
                      f"Noise: {condition_stats['avg_noise_ratio']:.2f}", end="")
            print("")
            
        nll_epoch.append(nll.item())
        
        # 记录条件统计信息到wandb
        if args.partial_conditioning and condition_stats:
            wandb.log({
                "Condition Ratio": condition_stats['avg_condition_ratio'],
                "Noise Ratio": condition_stats['avg_noise_ratio'],
                "Valid Nodes": condition_stats['avg_valid_nodes'],
                "Condition Nodes": condition_stats['avg_condition_nodes']
            }, commit=False)
            
        # if (epoch % args.test_epochs == 0) and (i % args.visualize_every_batch == 0) and not (epoch == 0 and i == 0) and args.train_diffusion:
        #     start = time.time()
        #     if len(args.conditioning) > 0:
        #         save_and_sample_conditional(args, device, model_ema, prop_dist, dataset_info, epoch=epoch)
        #     save_and_sample_chain(model_ema, args, device, dataset_info, prop_dist, epoch=epoch,
        #                           batch_id=str(i))
        #     sample_different_sizes_and_save(model_ema, nodes_dist, args, device, dataset_info,
        #                                     prop_dist, epoch=epoch)
        #     print(f'Sampling took {time.time() - start:.2f} seconds')

        #     vis.visualize(f"outputs/{args.exp_name}/epoch_{epoch}_{i}", dataset_info=dataset_info, wandb=wandb)
        #     vis.visualize_chain(f"outputs/{args.exp_name}/epoch_{epoch}_{i}/chain/", dataset_info, wandb=wandb)
        #     if len(args.conditioning) > 0:
        #         vis.visualize_chain("outputs/%s/epoch_%d/conditional/" % (args.exp_name, epoch), dataset_info,
        #                             wandb=wandb, mode='conditional')
        wandb.log({"Batch NLL": nll.item()}, commit=True)
        if args.break_train_epoch:
            break
    wandb.log({"Train Epoch NLL": np.mean(nll_epoch)}, commit=False)


def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def test(args, loader, epoch, eval_model, device, dtype, property_norms, nodes_dist, partition='Test'):
    eval_model.eval()
    with torch.no_grad():
        nll_epoch = 0
        n_samples = 0

        n_iterations = len(loader)

        for i, data in enumerate(loader):
            x = data['positions'].to(device, dtype)
            batch_size = x.size(0)
            node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
            edge_mask = data['edge_mask'].to(device, dtype)
            one_hot = data['one_hot'].to(device, dtype)
            charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

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

            # 获取原始上下文条件
            if len(args.conditioning) > 0:
                context = qm9utils.prepare_context(args.conditioning, data, property_norms).to(device, dtype)
                assert_correctly_masked(context, node_mask)
            else:
                context = None
                
            # 应用部分条件掩码（如果启用）- 测试阶段也需要与训练一致
            if args.partial_conditioning:
                # 使用固定的随机种子来保证测试的一致性
                test_seed = 42
                
                # 根据策略生成掩码
                if args.mask_strategy == "connected":
                    # 使用连通掩码生成
                    noise_mask, condition_mask = generate_connected_mask(
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
                
                # 改变edge_mask删除连接向condition_mask的节点
                # edge_mask = node_mask将第二维移动到第三维 * noise_mask
                batch_size = x.size(0)           
                n_nodes = node_mask.size(1)
                edge_mask = torch.zeros((batch_size, n_nodes, n_nodes), device=device, dtype=dtype)
                for b in range(batch_size):
                    for u in range(n_nodes):
                        for v in range(n_nodes):
                            if condition_mask[b, v, 0] == 1 or noise_mask[b, u, 0] == 1:
                                edge_mask[b, u, v] = 1
                edge_mask = edge_mask.view(batch_size * n_nodes * n_nodes, 1)
                # transform batch through flow
                nll, _, _ = losses.compute_loss_and_nll(args, eval_model, nodes_dist, x, h,
                                                       node_mask, edge_mask, context, 
                                                       noise_mask=noise_mask, condition_mask=condition_mask)
            else:
                # 如果没有使用部分条件，不传递mask参数
                noise_mask = None
                condition_mask = None
                # transform batch through flow
                nll, _, _ = losses.compute_loss_and_nll(args, eval_model, nodes_dist, x, h,
                                                       node_mask, edge_mask, context, 
                                                       noise_mask=None, condition_mask=None)
                
            # standard nll from forward KL

            nll_epoch += nll.item() * batch_size
            n_samples += batch_size
            if i % args.n_report_steps == 0:
                print(f"\r {partition} NLL \t epoch: {epoch}, iter: {i}/{n_iterations}, "
                      f"NLL: {nll_epoch/n_samples:.2f}")
            # break # 测试阶段只测试一个batch 记得删除！！！！！！！！！！！！！
    return nll_epoch/n_samples


def save_and_sample_chain(model, args, device, dataset_info, prop_dist,
                          epoch=0, id_from=0, batch_id=''):
    """
    保存和生成分子链。如果启用部分条件，则使用骨架引导生成。
    """
    # 检查模型是否支持骨架条件生成
    if args.partial_conditioning and hasattr(model, 'sample_chain_scaf'):
        try:
            # 准备一个示例骨架，使用固定的原子数和随机种子以便重现
            n_nodes = 19  # QM9分子的最大原子数
            n_samples = 1
            node_mask = torch.ones(n_samples, n_nodes, 1).to(device)
            edge_mask = torch.zeros(n_samples, n_nodes*n_nodes, 1).to(device)
            
            # 跳过使用训练集样本，直接构造随机骨架
            # 创建随机编码作为骨架
            import random
            random.seed(epoch)  # 使用epoch作为种子以产生不同的样本
            
            # 直接定义骨架的潜在表示
            z_x_dim = 3  # 位置编码维度，通常是3
            z_h_dim = args.latent_nf  # 节点特征潜在表示维度
            
            # 创建随机的潜在表示
            z_x_mu = torch.randn(n_samples, n_nodes, z_x_dim).to(device) * 0.1
            z_h_mu = torch.randn(n_samples, n_nodes, z_h_dim).to(device) * 0.1
            
            # 使用随机潜在表示生成完整分子链
            chain = model.sample_chain_scaf(
                n_samples=n_samples, 
                n_nodes=n_nodes, 
                node_mask=node_mask, 
                edge_mask=edge_mask, 
                context=None, 
                z_x=z_x_mu, 
                z_h=z_h_mu, 
                keep_frames=100
            )
            
            # 处理生成的链，提取one_hot, charges和坐标
            x = chain[:, :, :3]
            one_hot = chain[:, :, 3:-1]
            charges = chain[:, :, -1:]
        except Exception as e:
            print(f"骨架引导生成失败: {e}，使用默认生成方法")
            # 出错时回退到默认生成方法
            one_hot, charges, x = sample_chain(args=args, device=device, flow=model,
                                              n_tries=1, dataset_info=dataset_info, prop_dist=prop_dist)
    else:
        # 使用原始采样方法
        one_hot, charges, x = sample_chain(args=args, device=device, flow=model,
                                          n_tries=1, dataset_info=dataset_info, prop_dist=prop_dist)

    # 保存文件
    vis.save_xyz_file(f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/chain/',
                      one_hot, charges, x, dataset_info, id_from, name='chain')

    return one_hot, charges, x


def sample_different_sizes_and_save(model, nodes_dist, args, device, dataset_info, prop_dist,
                                    n_samples=5, epoch=0, batch_size=100, batch_id=''):
    batch_size = min(batch_size, n_samples)
    for counter in range(int(n_samples/batch_size)):
        nodesxsample = nodes_dist.sample(batch_size)
        one_hot, charges, x, node_mask = sample(args, device, model, prop_dist=prop_dist,
                                                nodesxsample=nodesxsample,
                                                dataset_info=dataset_info)
        print(f"Generated molecule: Positions {x[:-1, :, :]}")
        vis.save_xyz_file(f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/', one_hot, charges, x, dataset_info,
                          batch_size * counter, name='molecule')


def analyze_and_save(epoch, model_sample, nodes_dist, args, device, dataset_info, prop_dist,
                     n_samples=1000, batch_size=100):
    print(f'Analyzing molecule stability at epoch {epoch}...')
    batch_size = min(batch_size, n_samples)
    assert n_samples % batch_size == 0
    molecules = {'one_hot': [], 'x': [], 'node_mask': []}
    for i in range(int(n_samples/batch_size)):
        nodesxsample = nodes_dist.sample(batch_size)
        one_hot, charges, x, node_mask = sample(args, device, model_sample, dataset_info, prop_dist,
                                                nodesxsample=nodesxsample)

        molecules['one_hot'].append(one_hot.detach().cpu())
        molecules['x'].append(x.detach().cpu())
        molecules['node_mask'].append(node_mask.detach().cpu())

    molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
    validity_dict, rdkit_tuple = analyze_stability_for_molecules(molecules, dataset_info)

    wandb.log(validity_dict)
    if rdkit_tuple is not None:
        wandb.log({'Validity': rdkit_tuple[0][0], 'Uniqueness': rdkit_tuple[0][1], 'Novelty': rdkit_tuple[0][2]})
    return validity_dict


def analyze_and_save_scaf(epoch, model_sample, nodes_dist, args, device, dataset_info, prop_dist,
                     n_samples=1, batch_size=1, loader=None, dtype=None, property_norms=None):
    print(f'Analyzing molecule stability at epoch {epoch}...')
    batch_size = min(batch_size, n_samples)
    assert(batch_size == 1)
    assert n_samples % batch_size == 0
    molecules = {'one_hot': [], 'x': [], 'node_mask': []}
    have_found_stable_molecule = False
    for i in range(int(n_samples/batch_size)):
        nodesxsample = nodes_dist.sample(batch_size)
        # one_hot, charges, x, node_mask = sample(args, device, model_sample, dataset_info, prop_dist,
        #                                         nodesxsample=nodesxsample)
        one_hot, charges, x, node_mask, scaf_x, scaf_h, draw_xh = sample_scaf(args, device, model_sample, dataset_info, prop_dist,
                                                                              nodesxsample=nodesxsample, loader=loader, dtype=dtype,
                                                                              mask_func=generate_connected_mask, property_norms=property_norms)
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

        if (not have_found_stable_molecule) and (is_valid_by_rdkit or i == int(n_samples/batch_size)-1):
            have_found_stable_molecule = True
            gen_dir = 'outputs/%s/epoch_%d' % (args.exp_name, epoch)
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
            
            # 保存输入片段为图像
            vis.plot_data3d(scaf_x.cpu(), scaf_h.cpu(), dataset_info, spheres_3d=True, 
                            save_path=os.path.join(gen_dir, 'fragment.png'))
            
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
                path=gen_dir + "/", 
                one_hot=scaf_one_hot.cpu(),  # 使用转换后的one-hot表示
                charges=None,  # charges不需要提供，因为我们已经提供了one_hot
                positions=scaf_x_batch.cpu(),  # 片段原子坐标
                dataset_info=dataset_info,
                name='fragment'
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

    molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
    validity_dict, rdkit_tuple = analyze_stability_for_molecules(molecules, dataset_info)

    wandb.log(validity_dict)
    if rdkit_tuple is not None:
        wandb.log({'Validity': rdkit_tuple[0][0], 'Uniqueness': rdkit_tuple[0][1], 'Novelty': rdkit_tuple[0][2]})
    return validity_dict


def save_and_sample_conditional(args, device, model, prop_dist, dataset_info, epoch=0, id_from=0):
    one_hot, charges, x, node_mask = sample_sweep_conditional(args, device, model, dataset_info, prop_dist)

    vis.save_xyz_file(
        'outputs/%s/epoch_%d/conditional/' % (args.exp_name, epoch), one_hot, charges, x, dataset_info,
        id_from, name='conditional', node_mask=node_mask)

    return one_hot, charges, x
