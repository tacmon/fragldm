import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from qm9 import visualizer as vis
from configs.datasets_config import get_dataset_info

dataset_info = get_dataset_info("qm9", False)

def smiles_to_3d(smiles, dataset_info, device):
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

if __name__ == "__main__":
    smiles = input("请输入SMILES字符串: ")
    device = torch.device("cpu")
    positions, one_hot, atomic_numbers = smiles_to_3d(smiles, dataset_info, device)
    positions.unsqueeze_(0)
    one_hot.unsqueeze_(0)
    atomic_numbers.unsqueeze_(0)
    vis.save_xyz_file(
        path="./outputs/molecule/", 
        one_hot=one_hot.cpu(),  # 使用转换后的one-hot表示
        charges=None,  # charges不需要提供，因为我们已经提供了one_hot
        positions=positions.cpu(),  # 片段原子坐标
        dataset_info=dataset_info,
        name='molecule'
    )