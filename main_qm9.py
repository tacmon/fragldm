# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ModuleNotFoundError:
    pass
import copy
import utils
import argparse
import wandb
from configs.datasets_config import get_dataset_info
from os.path import join
from qm9 import dataset
from qm9.models import get_optim, get_model, get_autoencoder, get_latent_diffusion
from equivariant_diffusion import en_diffusion
from equivariant_diffusion.utils import assert_correctly_masked
from equivariant_diffusion import utils as flow_utils
import torch
import time
import pickle
from qm9.utils import prepare_context, compute_mean_mad
from train_test import train_epoch, test, analyze_and_save, analyze_and_save_scaf

parser = argparse.ArgumentParser(description='E3Diffusion')
parser.add_argument('--exp_name', type=str, default='debug_10')

# Latent Diffusion args
parser.add_argument('--train_diffusion', action='store_true', 
                    help='Train second stage LatentDiffusionModel model')
parser.add_argument('--ae_path', type=str, default=None,
                    help='Specify first stage model path')
parser.add_argument('--trainable_ae', action='store_true',
                    help='Train first stage AutoEncoder model')

# VAE args
parser.add_argument('--latent_nf', type=int, default=4,
                    help='number of latent features')
parser.add_argument('--kl_weight', type=float, default=0.01,
                    help='weight of KL term in ELBO')

parser.add_argument('--model', type=str, default='egnn_dynamics',
                    help='our_dynamics | schnet | simple_dynamics | '
                    'kernel_dynamics | egnn_dynamics |gnn_dynamics')
parser.add_argument('--probabilistic_model', type=str, default='diffusion',
                    help='diffusion')

# Training complexity is O(1) (unaffected), but sampling complexity is O(steps).
parser.add_argument('--diffusion_steps', type=int, default=500)
parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
                    help='learned, cosine')
parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5,
                    )
parser.add_argument('--diffusion_loss_type', type=str, default='l2',
                    help='vlb, l2')

parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--brute_force', type=eval, default=False,
                    help='True | False')
parser.add_argument('--actnorm', type=eval, default=True,
                    help='True | False')
parser.add_argument('--break_train_epoch', type=eval, default=False,
                    help='True | False')
parser.add_argument('--dp', type=eval, default=True,
                    help='True | False')
parser.add_argument('--condition_time', type=eval, default=True,
                    help='True | False')
parser.add_argument('--clip_grad', type=eval, default=True,
                    help='True | False')
parser.add_argument('--trace', type=str, default='hutch',
                    help='hutch | exact')
# EGNN args -->
parser.add_argument('--n_layers', type=int, default=6,
                    help='number of layers')
parser.add_argument('--inv_sublayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--nf', type=int, default=128,
                    help='number of layers')
parser.add_argument('--tanh', type=eval, default=True,
                    help='use tanh in the coord_mlp')
parser.add_argument('--attention', type=eval, default=True,
                    help='use attention in the EGNN')
parser.add_argument('--norm_constant', type=float, default=1,
                    help='diff/(|diff| + norm_constant)')
parser.add_argument('--sin_embedding', type=eval, default=False,
                    help='whether using or not the sin embedding')
# <-- EGNN args
parser.add_argument('--ode_regularization', type=float, default=1e-3)
parser.add_argument('--dataset', type=str, default='qm9',
                    help='qm9 | qm9_second_half (train only on the last 50K samples of the training dataset)')
parser.add_argument('--datadir', type=str, default='qm9/temp',
                    help='qm9 directory')
parser.add_argument('--filter_n_atoms', type=int, default=None,
                    help='When set to an integer value, QM9 will only contain molecules of that amount of atoms')
parser.add_argument('--dequantization', type=str, default='argmax_variational',
                    help='uniform | variational | argmax_variational | deterministic')
parser.add_argument('--n_report_steps', type=int, default=1)
parser.add_argument('--wandb_usr', type=str)
parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')
parser.add_argument('--online', type=bool, default=True, help='True = wandb online -- False = wandb offline')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--save_model', type=eval, default=True,
                    help='save model')
parser.add_argument('--generate_epochs', type=int, default=1,
                    help='save model')
parser.add_argument('--num_workers', type=int, default=0, help='Number of worker for the dataloader')
parser.add_argument('--test_epochs', type=int, default=10)
parser.add_argument('--data_augmentation', type=eval, default=False, help='use attention in the EGNN')
parser.add_argument("--conditioning", nargs='+', default=[],
                    help='arguments : homo | lumo | alpha | gap | mu | Cv' )
parser.add_argument('--resume', type=str, default=None,
                    help='')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='')
parser.add_argument('--ema_decay', type=float, default=0.999,
                    help='Amount of EMA decay, 0 means off. A reasonable value'
                    ' is 0.999.')
parser.add_argument('--augment_noise', type=float, default=0)
parser.add_argument('--n_stability_samples', type=int, default=500,
                    help='Number of samples to compute the stability')
parser.add_argument('--normalize_factors', type=eval, default=[1, 4, 1],
                    help='normalize factors for [x, categorical, integer]')
parser.add_argument('--remove_h', action='store_true')
parser.add_argument('--include_charges', type=eval, default=True,
                    help='include atom charge or not')
parser.add_argument('--visualize_every_batch', type=int, default=1e8,
                    help="Can be used to visualize multiple times per epoch")
parser.add_argument('--normalization_factor', type=float, default=1,
                    help="Normalize the sum aggregation of EGNN")
parser.add_argument('--aggregation_method', type=str, default='sum',
                    help='"sum" or "mean"')
parser.add_argument('--partial_conditioning', action='store_true',
                    help='Enable partial conditioning for generation')
parser.add_argument('--noise_ratio', type=float, default=0.5,
                    help='Ratio of atoms to be noised (0-1)')
parser.add_argument('--mask_strategy', type=str, default='random',
                    choices=['random', 'connected', 'central', 'peripheral'],
                    help='Strategy for masking atoms during partial conditioning')
parser.add_argument('--gen_mode', type=int, default=0,
                    help='for training or generating?')
parser.add_argument('--scaf_model_path', type=str, default="outputs/important_par",
                    help='Specify scaffold model path')
args = parser.parse_args()

dataset_info = get_dataset_info(args.dataset, args.remove_h)

atom_encoder = dataset_info['atom_encoder']
atom_decoder = dataset_info['atom_decoder']

# args, unparsed_args = parser.parse_known_args()
args.wandb_usr = utils.get_wandb_username(args.wandb_usr)

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32

if args.resume is not None:
    exp_name = args.exp_name + '_resume'
    start_epoch = args.start_epoch
    resume = args.resume
    wandb_usr = args.wandb_usr
    normalization_factor = args.normalization_factor
    normalize_factors = args.normalize_factors
    aggregation_method = args.aggregation_method
    num_workers = args.num_workers  # 保存当前命令行指定的num_workers值
    no_wandb = args.no_wandb
    noise_ratio = args.noise_ratio
    arg_dataset = args.dataset


    with open(join(args.resume, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)

    args.resume = resume
    args.break_train_epoch = False

    args.exp_name = exp_name
    args.start_epoch = start_epoch
    args.wandb_usr = wandb_usr
    args.num_workers = num_workers  # 确保使用当前命令行指定的num_workers值
    args.no_wandb = no_wandb
    args.noise_ratio = noise_ratio
    args.dataset = arg_dataset
    args.normalize_factors = normalize_factors

    # Careful with this -->
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = normalization_factor
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = aggregation_method

    print(args)

utils.create_folders(args)
# print(args)


# Wandb config
if args.no_wandb:
    mode = 'disabled'
else:
    mode = 'online' if args.online else 'offline'
kwargs = {'entity': args.wandb_usr, 'name': args.exp_name, 'project': 'e3_diffusion_qm9', 'config': args,
          'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
wandb.init(**kwargs)
wandb.save('*.txt')

# Retrieve QM9 dataloaders
dataloaders, charge_scale = dataset.retrieve_dataloaders(args)

data_dummy = next(iter(dataloaders['train']))


if len(args.conditioning) > 0:
    '''
    property_norms 是一个字典，键是条件变量名称，值是包含均值和标准差的元组。
    例如： {'alpha': {'mean': tensor(75.3734), 'mad': tensor(6.2728)}}
    context_dummy 是一个包含条件变量值的张量，形状为 (batch_size, num_conditions, num_features)。
    例如： torch.Size([8, 24, 1])
    context_node_nf 是 context_dummy 的特征维度。
    例如： 1
    '''
    print(f'Conditioning on {args.conditioning}')
    property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
    context_dummy = prepare_context(args.conditioning, data_dummy, property_norms)
    context_node_nf = context_dummy.size(2)
else:
    context_node_nf = 0
    property_norms = None

args.context_node_nf = context_node_nf

# Create Latent Diffusion Model or Audoencoder
if args.train_diffusion:
    model, nodes_dist, prop_dist = get_latent_diffusion(args, device, dataset_info, dataloaders['train'])
else:
    model, nodes_dist, prop_dist = get_autoencoder(args, device, dataset_info, dataloaders['train'])

if prop_dist is not None:
    prop_dist.set_normalizer(property_norms)

model = model.to(device)
optim = get_optim(args, model)
# print(model)

gradnorm_queue = utils.Queue()
gradnorm_queue.add(3000)  # Add large value that will be flushed.

def main():
    if args.resume is not None:
        flow_state_dict = torch.load(join(args.resume, 'generative_model.npy'))
        optim_state_dict = torch.load(join(args.resume, 'optim.npy'))
        model.load_state_dict(flow_state_dict)
        optim.load_state_dict(optim_state_dict)

    # Initialize dataparallel if enabled and possible.
    if args.dp and torch.cuda.device_count() > 1:
        print(f'Training using {torch.cuda.device_count()} GPUs')
        model_dp = torch.nn.DataParallel(model.cpu())
        model_dp = model_dp.cuda()
    else:
        model_dp = model

    # Initialize model copy for exponential moving average of params.
    if args.ema_decay > 0:
        model_ema = copy.deepcopy(model)
        ema = flow_utils.EMA(args.ema_decay)

        if args.dp and torch.cuda.device_count() > 1:
            model_ema_dp = torch.nn.DataParallel(model_ema)
        else:
            model_ema_dp = model_ema
    else:
        ema = None
        model_ema = model
        model_ema_dp = model_dp

    best_nll_val = 1e8
    best_nll_test = 1e8
    for epoch in range(args.start_epoch, args.n_epochs):
        start_epoch = time.time()
        if True:
            train_epoch(args=args, loader=dataloaders['train'], epoch=epoch, model=model, model_dp=model_dp,
                        model_ema=model_ema, ema=ema, device=device, dtype=dtype, property_norms=property_norms,
                        nodes_dist=nodes_dist, dataset_info=dataset_info,
                        gradnorm_queue=gradnorm_queue, optim=optim, prop_dist=prop_dist)
        print(f"Epoch took {time.time() - start_epoch:.1f} seconds.")

        if epoch % args.test_epochs == 0:
            if isinstance(model, en_diffusion.EnVariationalDiffusion):
                wandb.log(model.log_info(), commit=True)

            if not args.break_train_epoch and args.train_diffusion:
                if args.partial_conditioning:
                    # 获取原始dataloader的所有参数
                    original_loader = dataloaders['valid']
                    eval_dataset = original_loader.dataset
                    
                    # 创建新的DataLoader，保留原始配置
                    eval_loader = torch.utils.data.DataLoader(
                        dataset=eval_dataset,
                        batch_size=1,
                        shuffle=False,  # 评估时不需要打乱顺序
                        num_workers=0,  # 避免多进程问题
                        collate_fn=original_loader.collate_fn,  # 保留原始的collate函数
                        pin_memory=original_loader.pin_memory,
                        drop_last=original_loader.drop_last,
                        # 其他可能存在的参数将自动使用默认值
                    )
                    
                    analyze_and_save_scaf(args=args, epoch=epoch, model_sample=model_ema, nodes_dist=nodes_dist,
                                    dataset_info=dataset_info, device=device,
                                    prop_dist=prop_dist, n_samples=30, loader=eval_loader, dtype=dtype, property_norms=property_norms) # n_samples=args.n_stability_samples
                else:
                    analyze_and_save(args=args, epoch=epoch, model_sample=model_ema, nodes_dist=nodes_dist,
                                    dataset_info=dataset_info, device=device,
                                    prop_dist=prop_dist, n_samples=args.n_stability_samples)
            nll_val = test(args=args, loader=dataloaders['valid'], epoch=epoch, eval_model=model_ema_dp,
                           partition='Val', device=device, dtype=dtype, nodes_dist=nodes_dist,
                           property_norms=property_norms)
            nll_test = test(args=args, loader=dataloaders['test'], epoch=epoch, eval_model=model_ema_dp,
                            partition='Test', device=device, dtype=dtype,
                            nodes_dist=nodes_dist, property_norms=property_norms)

            if nll_val < best_nll_val:
                best_nll_val = nll_val
                best_nll_test = nll_test
                if args.save_model:
                    args.current_epoch = epoch + 1
                    utils.save_model(optim, 'outputs/%s/optim.npy' % args.exp_name)
                    utils.save_model(model, 'outputs/%s/generative_model.npy' % args.exp_name)
                    if args.ema_decay > 0:
                        utils.save_model(model_ema, 'outputs/%s/generative_model_ema.npy' % args.exp_name)
                    with open('outputs/%s/args.pickle' % args.exp_name, 'wb') as f:
                        pickle.dump(args, f)

                if args.save_model:
                    utils.save_model(optim, 'outputs/%s/optim_%d.npy' % (args.exp_name, epoch))
                    utils.save_model(model, 'outputs/%s/generative_model_%d.npy' % (args.exp_name, epoch))
                    if args.ema_decay > 0:
                        utils.save_model(model_ema, 'outputs/%s/generative_model_ema_%d.npy' % (args.exp_name, epoch))
                    with open('outputs/%s/args_%d.pickle' % (args.exp_name, epoch), 'wb') as f:
                        pickle.dump(args, f)
            print('Val loss: %.4f \t Test loss:  %.4f' % (nll_val, nll_test))
            print('Best val loss: %.4f \t Best test loss:  %.4f' % (best_nll_val, best_nll_test))
            wandb.log({"Val loss ": nll_val}, commit=True)
            wandb.log({"Test loss ": nll_test}, commit=True)
            wandb.log({"Best cross-validated test loss ": best_nll_test}, commit=True)


def smiles_to_3d(smiles):
    """
    将SMILES字符串转换为3D坐标和one-hot表示
    
    参数:
        smiles: SMILES字符串
        
    返回:
        positions: 原子坐标 (N, 3)
        one_hot: one-hot原子类型表示 (N, num_atom_types)
        charges: 原子序数 (N, 1)
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    
    # 获取原子坐标
    positions = torch.tensor(mol.GetConformers()[0].GetPositions(), dtype=torch.float32, device=device)
    
    # 创建one-hot表示
    num_atom_types = len(dataset_info['atom_decoder'])
    one_hot = torch.zeros((len(mol.GetAtoms()), num_atom_types), device=device)
    
    # 获取原子序数（而非电荷）
    atomic_numbers = torch.zeros((len(mol.GetAtoms()), 1), device=device)
    
    for i, atom in enumerate(mol.GetAtoms()):
        atom_type = atom.GetSymbol()
        if atom_type in dataset_info['atom_encoder']:
            one_hot[i, dataset_info['atom_encoder'][atom_type]] = 1
        # 获取原子序数
        atomic_numbers[i, 0] = atom.GetAtomicNum()
    
    return positions, one_hot, atomic_numbers


def main_gen():
    # 加载预训练模型
    with open(join(args.scaf_model_path, 'args.pickle'), 'rb') as f:
        model_args = pickle.load(f)

    # 创建模型
    if model_args.train_diffusion:
        model, nodes_dist, prop_dist = get_latent_diffusion(model_args, device, dataset_info, None)
    else:
        model, nodes_dist, prop_dist = get_autoencoder(model_args, device, dataset_info, None)

    # 加载模型权重
    model_path = join(args.scaf_model_path, 'generative_model_ema.npy' if model_args.ema_decay > 0 else 'generative_model.npy')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # 获取分子片段的3D坐标和one-hot表示
    positions, one_hot, atomic_numbers = smiles_to_3d("C(=O)[O-]")
    condition_x = positions
    # 将中心移动到原点
    condition_x = condition_x - torch.mean(condition_x, dim=0)
    condition_h = {'categorical': one_hot, 'integer': atomic_numbers}
    
    # 创建节点掩码和边掩码
    n_nodes = condition_x.size(0)
    node_mask = torch.ones(1, n_nodes, 1).to(device)
    
    # 创建正确的边掩码（完全连接但无自环）
    edge_mask = torch.ones(1, n_nodes, n_nodes).to(device)
    # 移除自环连接
    diag_mask = ~torch.eye(n_nodes, dtype=torch.bool, device=device).unsqueeze(0)
    edge_mask = edge_mask * diag_mask.float()
    
    # 使用VAE编码器将条件输入转换为潜在变量
    with torch.no_grad():
        # 将张量扩展到批处理维度
        batch_x = condition_x.unsqueeze(0)  # [1, n_nodes, 3]
        batch_h = {
            'categorical': condition_h['categorical'].unsqueeze(0),  # [1, n_nodes, num_classes]
            'integer': condition_h['integer'].unsqueeze(0)  # [1, n_nodes, 1]
        }
        
        # 编码输入到潜空间
        z_x_mu, z_x_sigma, z_h_mu, z_h_sigma = model.vae.encode(batch_x, batch_h, node_mask, edge_mask, context=None)
        
        # 获取编码后的潜变量表示
        encoded_condition_x = z_x_mu  # [1, n_nodes, 3]
        encoded_condition_h = z_h_mu  # [1, n_nodes, latent_nf]
        
        print(f"编码后的形状: x={encoded_condition_x.shape}, h={encoded_condition_h.shape}")

    n_samples = 1
    batch_size = 1
    molecules = {'one_hot': [], 'x': [], 'node_mask': []}
    for i in range(int(n_samples/batch_size)):
        nodesxsample = nodes_dist.sample(batch_size)
        max_n_nodes = dataset_info['max_n_nodes']  # this is the maximum node_size in QM9

        assert int(torch.max(nodesxsample)) <= max_n_nodes
        batch_size = len(nodesxsample)

        node_mask = torch.zeros(batch_size, max_n_nodes)
        noise_mask = torch.zeros(batch_size, max_n_nodes)
        condition_mask = torch.zeros(batch_size, max_n_nodes)
        for i in range(batch_size):
            node_mask[i, 0:nodesxsample[i]] = 1        
            noise_mask[i, condition_x.size(0):nodesxsample[i]] = 1
            condition_mask[i, 0:condition_x.size(0)] = 1
        # Compute edge_mask
        edge_mask = node_mask.unsqueeze(1) * noise_mask.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask
        edge_mask = edge_mask.view(batch_size * max_n_nodes * max_n_nodes, 1).to(device)
        node_mask = node_mask.unsqueeze(2).to(device)
        noise_mask = noise_mask.unsqueeze(2).to(device)
        condition_mask = condition_mask.unsqueeze(2).to(device)
        # TODO FIX: This conditioning just zeros.
        if args.context_node_nf > 0:
            if context is None:
                context = prop_dist.sample_batch(nodesxsample)
            context = context.unsqueeze(1).repeat(1, max_n_nodes, 1).to(device) * node_mask
        else:
            context = None

        if args.probabilistic_model == 'diffusion':
            x, h = model.sample_scaf(batch_size, max_n_nodes, node_mask, edge_mask, context, condition_x=encoded_condition_x, condition_h=encoded_condition_h, noise_mask=noise_mask, condition_mask=condition_mask)

            assert_correctly_masked(x, node_mask)
            # assert_mean_zero_with_mask(x, node_mask)

            one_hot = h['categorical']
            charges = h['integer']

            assert_correctly_masked(one_hot.float(), node_mask)
            if args.include_charges:
                assert_correctly_masked(charges.float(), node_mask)
        else:
            raise ValueError(args.probabilistic_model)

        molecules['one_hot'].append(one_hot.detach().cpu())
        molecules['x'].append(x.detach().cpu())
        molecules['node_mask'].append(node_mask.detach().cpu())

    molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
    # 添加功能：绘制分子的图片，参照train_test.py中注释掉的可视化代码
    
    # 创建必要的目录
    print("正在生成分子可视化...")
    import os
    from qm9 import visualizer as vis
    import time

    # 为输出结果创建目录结构
    output_dir = f"outputs/{args.exp_name if hasattr(args, 'exp_name') else 'fragment_gen'}"
    gen_dir = f"{output_dir}/generation"
    chain_dir = f"{gen_dir}/chain"
    fragment_dir = f"{gen_dir}/fragment"
    
    os.makedirs(gen_dir, exist_ok=True)
    os.makedirs(chain_dir, exist_ok=True)
    os.makedirs(fragment_dir, exist_ok=True)
    
    # 保存生成的分子
    start = time.time()
    
    # 1. 保存输入片段
    # 原始输入片段的SMILES字符串（仅用于显示和记录）
    fragment_smiles = "C(=O)[O-]"
    
    # 不直接使用smiles_to_3d获取片段信息，而是通过解码encoded_condition_x和encoded_condition_h
    with torch.no_grad():
        # 将编码后的条件变量组合起来用于解码
        z_fragment = torch.cat([encoded_condition_x, encoded_condition_h], dim=2)
        n_nodes = condition_x.size(0)
        node_mask = torch.ones(1, n_nodes, 1).to(device)
    
        # 创建正确的边掩码（完全连接但无自环）
        edge_mask = torch.ones(1, n_nodes, n_nodes).to(device)
        # 移除自环连接
        diag_mask = ~torch.eye(n_nodes, dtype=torch.bool, device=device).unsqueeze(0)
        edge_mask = edge_mask * diag_mask.float()
        # 使用VAE的解码器解码
        fragment_pos, fragment_h = model.vae.decode(z_fragment, node_mask=node_mask, edge_mask=edge_mask, context=None)
        
        # 从解码结果中提取one-hot表示和电荷/原子序数
        fragment_one_hot = fragment_h['categorical']  # [1, n_nodes, num_atom_types]
        fragment_atomic_numbers = fragment_h['integer']  # [1, n_nodes, 1]
        
        print(f"解码后的片段形状: pos={fragment_pos.shape}, one_hot={fragment_one_hot.shape}, atomic_numbers={fragment_atomic_numbers.shape}")
    
    # 保存输入片段为XYZ文件
    vis.save_xyz_file(
        path=f"{fragment_dir}/", 
        one_hot=fragment_one_hot,  # 已经包含批次维度 [1, n_nodes, num_atom_types]
        charges=fragment_atomic_numbers,  # 已经包含批次维度 [1, n_nodes, 1]
        positions=fragment_pos,  # 已经包含批次维度 [1, n_nodes, 3]
        dataset_info=dataset_info,
        name='input_fragment'
    )
    
    # 2. 保存生成的分子
    n_samples_to_save = min(50, len(molecules['one_hot']))  # 最多保存50个生成的分子
    
    vis.save_xyz_file(
        path=f"{gen_dir}/", 
        one_hot=molecules['one_hot'][:n_samples_to_save],
        charges=molecules['one_hot'][:n_samples_to_save].argmax(dim=2).unsqueeze(-1),  # 将one-hot变为类别索引，模拟charges
        positions=molecules['x'][:n_samples_to_save],
        dataset_info=dataset_info,
        node_mask=molecules['node_mask'][:n_samples_to_save],
        name='molecule'
    )
    
    # 3. 模拟链式采样结果（即使我们没有真正的链）
    # 为了兼容visualize_chain函数，我们创建一个假的链式结构
    sample_idx = min(5, n_samples_to_save)  # 取前几个样本作为示例
    for i in range(sample_idx):
        chain_subdir = f"{chain_dir}/molecule_{i}/"
        os.makedirs(chain_subdir, exist_ok=True)
        
        # 对单个分子进行保存，模拟链的最终状态
        vis.save_xyz_file(
            path=chain_subdir, 
            one_hot=molecules['one_hot'][i:i+1],
            charges=molecules['one_hot'][i:i+1].argmax(dim=2).unsqueeze(-1),
            positions=molecules['x'][i:i+1],
            dataset_info=dataset_info,
            node_mask=molecules['node_mask'][i:i+1],
            name='molecule'
        )
    
    print(f'保存分子数据用时 {time.time() - start:.2f} 秒')

    # 4. 使用visualizer进行可视化
    print("生成分子的3D可视化图像...")
    vis.visualize(f"{gen_dir}/", dataset_info=dataset_info, wandb=None, spheres_3d=True)
    
    # 5. 可视化输入片段
    vis.visualize(f"{fragment_dir}/", dataset_info=dataset_info, wandb=None, spheres_3d=True)
    
    # 6. 尝试可视化含有片段的分子
    # 这里我们使用了输入片段和生成分子的可视化，但没有直接在一个图中标注
    # 对比生成的分子与输入片段的3D可视化结果，可以看出哪些分子保留了输入片段结构
    
    # 分析生成结果
    from qm9.analyze import analyze_stability_for_molecules
    validity_dict, rdkit_tuple = analyze_stability_for_molecules(molecules, dataset_info)
    
    # 打印分析结果
    print("\n生成分子分析结果:")
    print(f"分子稳定性: {validity_dict['mol_stable']*100:.2f}%")
    print(f"原子稳定性: {validity_dict['atm_stable']*100:.2f}%")
    
    if rdkit_tuple is not None:
        print(f"有效分子比例: {rdkit_tuple[0][0]*100:.2f}%")
        print(f"唯一分子比例: {rdkit_tuple[0][1]*100:.2f}%")
        print(f"新颖分子比例: {rdkit_tuple[0][2]*100:.2f}%")
    
    print(f"\n可视化结果已保存到 {gen_dir} 目录")
    print(f"输入片段可视化保存到 {fragment_dir} 目录")
    
    return molecules

if __name__ == "__main__":
    if args.gen_mode == 0:
        main()
    else:
        main_gen()


# python main_qm9.py --n_epochs 3000 --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 640 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999 --train_diffusion --trainable_ae --latent_nf 1 --exp_name geoldm_qm9
# python  -m pdb main_qm9.py --n_epochs 3000 --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 16 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999 --train_diffusion --trainable_ae --latent_nf 1 --exp_name geoldm_qm9
'''
python main_qm9.py \
--resume outputs/geoldm_qm9 \
--start_epoch 240 \
--n_epochs 3000 \
--n_stability_samples 1000 \
--diffusion_noise_schedule polynomial_2 \
--diffusion_noise_precision 1e-5 \
--diffusion_steps 1000 \
--diffusion_loss_type l2 \
--batch_size 256 \
--nf 256 \
--n_layers 9 \
--lr 1e-4 \
--normalize_factors [1,4,10] \
--test_epochs 20 \
--ema_decay 0.9999 \
--train_diffusion \
--trainable_ae \
--latent_nf 1 \
--exp_name geoldm_qm9 \
'''
'''
python -m pdb main_qm9.py \
--resume outputs/geoldm_qm9 \
--start_epoch 240 \
--n_epochs 3000 \
--n_stability_samples 1000 \
--diffusion_noise_schedule polynomial_2 \
--diffusion_noise_precision 1e-5 \
--diffusion_steps 1000 \
--diffusion_loss_type l2 \
--batch_size 256 \
--nf 256 \
--n_layers 9 \
--lr 1e-4 \
--normalize_factors [1,4,10] \
--test_epochs 20 \
--ema_decay 0.9999 \
--train_diffusion \
--trainable_ae \
--latent_nf 1 \
--exp_name geoldm_qm9 \
'''


'''
python main_qm9.py --n_epochs 3000 --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 640 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999 --train_diffusion --trainable_ae --latent_nf 1 --exp_name geoldm_qm9
'''

'''
python main_qm9.py --exp_name exp_cond_alpha \
--model egnn_dynamics \
--lr 1e-4 \
--nf 192 \
--n_layers 9 \
--save_model True \
--diffusion_steps 1000 \
--sin_embedding False \
--n_epochs 3000 \
--n_stability_samples 500 \
--diffusion_noise_schedule polynomial_2 \
--diffusion_noise_precision 1e-5 \
--dequantization deterministic \
--include_charges False \
--diffusion_loss_type l2 \
--batch_size 16 \
--normalize_factors [1,8,1] \
--conditioning alpha \
--dataset qm9_second_half \
--train_diffusion \
--trainable_ae \
--latent_nf 1 \
--wandb_usr 2609028671-nudt \
--test_epochs 20
'''
