import torch
import numpy as np

import logging
import os

from torch.utils.data import DataLoader
from qm9.data.dataset_class import ProcessedDataset
from qm9.data.prepare import prepare_dataset


def initialize_datasets(args, datadir, dataset, subset=None, splits=None,
                        force_download=False, subtract_thermo=False,
                        remove_h=False):
    """
    Initialize datasets.

    Parameters
    ----------
    args : dict
        Dictionary of input arguments detailing the cormorant calculation.
    datadir : str
        Path to the directory where the data and calculations and is, or will be, stored.
    dataset : str
        String specification of the dataset.  If it is not already downloaded, must currently by "qm9" or "md17".
    subset : str, optional
        Which subset of a dataset to use.  Action is dependent on the dataset given.
        Must be specified if the dataset has subsets (i.e. MD17).  Otherwise ignored (i.e. GDB9).
    splits : str, optional
        TODO: DELETE THIS ENTRY
    force_download : bool, optional
        If true, forces a fresh download of the dataset.
    subtract_thermo : bool, optional
        If True, subtracts the thermochemical energy of the atoms from each molecule in GDB9.
        Does nothing for other datasets.
    remove_h: bool, optional
        If True, remove hydrogens from the dataset
    Returns
    -------
    args : dict
        Dictionary of input arguments detailing the cormorant calculation.
    datasets : dict
        Dictionary of processed dataset objects (see ????? for more information).
        Valid keys are "train", "test", and "valid"[ate].  Each associated value
    num_species : int
        Number of unique atomic species in the dataset.
    max_charge : pytorch.Tensor
        Largest atomic number for the dataset.

    Notes
    -----
    TODO: Delete the splits argument.
    """
    # Set the number of points based upon the arguments
    num_pts = {'train': args.num_train,
               'test': args.num_test, 'valid': args.num_valid}

    # Download and process dataset. Returns datafiles.
    datafiles = prepare_dataset(
        datadir, 'qm9', subset, splits, force_download=force_download)

    # Load downloaded/processed datasets
    datasets = {}
    for split, datafile in datafiles.items():
        with np.load(datafile) as f:
            datasets[split] = {key: torch.from_numpy(
                val) for key, val in f.items()}

    if dataset != 'qm9':
        np.random.seed(42)
        fixed_perm = np.random.permutation(len(datasets['train']['num_atoms']))
        # 不是都打乱了吗？怎么还区分前半和后半？
        if dataset == 'qm9_second_half':
            sliced_perm = fixed_perm[len(datasets['train']['num_atoms'])//2:]
        elif dataset == 'qm9_first_half':
            sliced_perm = fixed_perm[0:len(datasets['train']['num_atoms']) // 2]
        else:
            raise Exception('Wrong dataset name')
        # 打乱
        for key in datasets['train']:
            datasets['train'][key] = datasets['train'][key][sliced_perm]

    # Basic error checking: Check the training/test/validation splits have the same set of keys.
    keys = [list(data.keys()) for data in datasets.values()]
    assert all([key == keys[0] for key in keys]
               ), 'Datasets must have same set of keys!'

    # TODO: remove hydrogens here if needed
    if remove_h:
        for key, dataset in datasets.items():
            pos = dataset['positions']
            charges = dataset['charges']
            num_atoms = dataset['num_atoms']

            # Check that charges corresponds to real atoms
            assert torch.sum(num_atoms != torch.sum(charges > 0, dim=1)) == 0

            mask = dataset['charges'] > 1
            new_positions = torch.zeros_like(pos)
            new_charges = torch.zeros_like(charges)
            for i in range(new_positions.shape[0]):
                m = mask[i]
                p = pos[i][m]   # positions to keep
                p = p - torch.mean(p, dim=0)    # Center the new positions
                c = charges[i][m]   # Charges to keep
                n = torch.sum(m)
                new_positions[i, :n, :] = p
                new_charges[i, :n] = c

            dataset['positions'] = new_positions
            dataset['charges'] = new_charges
            dataset['num_atoms'] = torch.sum(dataset['charges'] > 0, dim=1)

    # Get a list of all species across the entire dataset
    '''
    _get_species 函数位于 qm9/data/utils.py 文件中，
    其主要目的是从数据集中提取并返回所有唯一的原子种类（元素），
    同时确保每个数据集分割（训练集、验证集、测试集）都包含所有原子种类。
    这对于确保模型在训练时能够学习到所有可能的原子类型，
    以及在测试时不会遇到未见过的原子类型非常重要。
    这个张量 tensor([1, 6, 7, 8, 9]) 代表了您的数据集中包含的所有原子类型的原子序数
    '''
    all_species = _get_species(datasets, ignore_check=False)

    # Now initialize MolecularDataset based upon loaded data
    # 此处不仅将数据集转换为ProcessedDataset，还顺手多了一个'one_hot'，也就是26个keys变成了27个
    datasets = {split: ProcessedDataset(data, num_pts=num_pts.get(
        split, -1), included_species=all_species, subtract_thermo=subtract_thermo) for split, data in datasets.items()}

    # Now initialize MolecularDataset based upon loaded data

    # Check that all datasets have the same included species:
    # tuple(data.included_species.tolist()) for data in datasets.values())是一堆(1, 6, 7, 8, 9)
    # 去重只剩一个(1, 6, 7, 8, 9)，所以 len==1
    assert(len(set(tuple(data.included_species.tolist()) for data in datasets.values())) ==
           1), 'All datasets must have same included_species! {}'.format({key: data.included_species for key, data in datasets.items()})

    # These parameters are necessary to initialize the network
    num_species = datasets['train'].num_species
    max_charge = datasets['train'].max_charge

    # Now, update the number of training/test/validation sets in args
    args.num_train = datasets['train'].num_pts
    args.num_valid = datasets['valid'].num_pts
    args.num_test = datasets['test'].num_pts

    return args, datasets, num_species, max_charge


def _get_species(datasets, ignore_check=False):
    """
    Generate a list of all species.

    Includes a check that each split contains examples of every species in the
    entire dataset.

    Parameters
    ----------
    datasets : dict
        Dictionary of datasets.  Each dataset is a dict of arrays containing molecular properties.
    ignore_check : bool
        Ignores/overrides checks to make sure every split includes every species included in the entire dataset

    Returns
    -------
    all_species : Pytorch tensor
        List of all species present in the data.  Species labels shoguld be integers.

    """
    # Get a list of all species in the dataset across all splits
    all_species = torch.cat([dataset['charges'].unique()
                             for dataset in datasets.values()]).unique(sorted=True)

    # Find the unique list of species in each dataset.
    split_species = {split: species['charges'].unique(
        sorted=True) for split, species in datasets.items()}

    # If zero charges (padded, non-existent atoms) are included, remove them
    if all_species[0] == 0:
        all_species = all_species[1:]

    # Remove zeros if zero-padded charges exst for each split
    split_species = {split: species[1:] if species[0] ==
                     0 else species for split, species in split_species.items()}

    # Now check that each split has at least one example of every atomic spcies from the entire dataset.
    if not all([split.tolist() == all_species.tolist() for split in split_species.values()]):
        # Allows one to override this check if they really want to. Not recommended as the answers become non-sensical.
        if ignore_check:
            logging.error(
                'The number of species is not the same in all datasets!')
        else:
            raise ValueError(
                'Not all datasets have the same number of species!')

    # Finally, return a list of all species
    return all_species
