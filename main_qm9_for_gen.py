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
from train_test_for_gen import analyze_and_save_scaf

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
    exp_name = "gen_" + args.exp_name
    start_epoch = 0
    resume = args.resume
    no_wandb = True
    normalize_factors = [1,4,10]
    num_workers = 0  # 保存当前命令行指定的num_workers值


    with open(join(args.resume, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)

    args.resume = resume
    args.exp_name = exp_name
    args.start_epoch = start_epoch
    args.num_workers = num_workers  # 确保使用当前命令行指定的num_workers值
    args.no_wandb = no_wandb
    args.normalize_factors = normalize_factors

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


def main():
    if args.resume is not None:
        flow_state_dict = torch.load(join(args.resume, 'generative_model.npy'))
        optim_state_dict = torch.load(join(args.resume, 'optim.npy'))
        model.load_state_dict(flow_state_dict)
        optim.load_state_dict(optim_state_dict)

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

    positions, one_hot, atomic_numbers = smiles_to_3d("C(=O)[O-]")
    condition_x = positions
    # 将中心移动到原点
    condition_x = condition_x - torch.mean(condition_x, dim=0)
    condition_h = {'categorical': one_hot, 'integer': atomic_numbers}

    analyze_and_save_scaf(args=args, model_sample=model, nodes_dist=nodes_dist,
                    dataset_info=dataset_info, device=device,
                    prop_dist=prop_dist, n_samples=100, loader=eval_loader, dtype=dtype, property_norms=property_norms,
                    manual_condition_x=condition_x, manual_condition_h=condition_h
                    ) # n_samples=args.n_stability_samples

if __name__ == "__main__":
    main()