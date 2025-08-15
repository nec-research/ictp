import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from typing import *
from pathlib import Path

import argparse

import yaml
import copy

from ase.data import atomic_numbers, covalent_radii

from ictp.data.datasets import DatasetHandler

from ictp.strategies import TrainingStrategy, EvaluationStrategy

from ictp.model.forward import load_model_from_folder

from ictp.utils.config import update_config
from ictp.utils.misc import set_default_dtype, find_max_r_cutoff, save_object


def train(config: Optional[Union[str, Dict]] = None) -> None:
    
    # load config
    if isinstance(config, str):
        config = yaml.safe_load(Path(config).read_text())

    # update default config
    config = update_config(config)
    
    # define default dtype
    set_default_dtype(config['default_dtype'])
    
    # manage data
    dataset_handler = DatasetHandler()
    
    train_dataset = dataset_handler.load_dataset(
        file_path=config['train_data_path'],
        r_cutoff=find_max_r_cutoff(config),
        skin=0.0,
        atomic_types=config['atomic_types'],
        neighbors=config['neighbors']
    )
    valid_dataset = dataset_handler.load_dataset(
        file_path=config['valid_data_path'],
        r_cutoff=find_max_r_cutoff(config),
        skin=0.0,
        atomic_types=config['atomic_types'],
        neighbors=config['neighbors']
    )
    
    # run training
    training = TrainingStrategy(config)
    _ = training.run(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        folder=config['model_path'],
        with_torch_script=True,
        with_torch_compile=False,
    )
    

def eval(config: Optional[Union[str, Dict]] = None) -> Dict[str, Any]:

    # load config
    if isinstance(config, str):
        config = yaml.safe_load(Path(config).read_text())

    # update default config
    config = update_config(config)
    
    # define default dtype
    set_default_dtype(config['default_dtype'])
    
    # manage data
    dataset_handler = DatasetHandler()
    
    test_dataset = dataset_handler.load_dataset(
        file_path=config['test_data_path'],
        r_cutoff=find_max_r_cutoff(config),
        skin=0.0,
        atomic_types=config['atomic_types'],
        neighbors=config['neighbors']
    )

    # load trained models and evaluate them on the test set
    model = load_model_from_folder(config['model_path'], key='best')
    
    evaluate = EvaluationStrategy(config)
    
    errors = evaluate.run(
        model=model,
        test_dataset=test_dataset,
        folder=config['model_path'],
        with_torch_script=False,
        with_torch_compile=True,
    )
    
    return errors


def eval_configs(config: Optional[Union[str, Dict]] = None, file_name: str = 'configs.extxyz') -> None:

    # load config
    if isinstance(config, str):
        config = yaml.safe_load(Path(config).read_text())

    # update default config
    config = update_config(config)
    
    # define default dtype
    set_default_dtype(config['default_dtype'])
    
    # manage data
    dataset_handler = DatasetHandler()
    
    test_dataset = dataset_handler.load_dataset(
        file_path=config['test_data_path'],
        r_cutoff=find_max_r_cutoff(config),
        skin=0.0,
        atomic_types=config['atomic_types'],
        neighbors=config['neighbors']
    )

    # load trained models and evaluate them on the test set
    model = load_model_from_folder(config['model_path'], key='best')
    
    evaluate = EvaluationStrategy(config)
    
    evaluate.run_on_configs(
        model=model,
        test_dataset=test_dataset,
        folder=config['model_path'],
        file_name=file_name,
        with_torch_script=False,
        with_torch_compile=True,
    )


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Parameters', fromfile_prefix_chars='@')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_feats', type=int, default=64)  # 64, 128, 256
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_epoch', type=int, default=500) # 500, 300, 150
    
    args = parser.parse_args()
    
    atomic_types = ['H', 'C', 'N', 'O', 'B', 'Br', 'Ca', 'Cl', 'F', 'I', 'K', 'Li', 'Mg', 'Na', 'P', 'S', 'Si']
    atomic_numbers_list = [atomic_numbers[symbol] for symbol in atomic_types]
    covalent_radii_list = [covalent_radii[number] for number in atomic_numbers_list]
    
    config = dict(
        atomic_types=atomic_types,
        atomic_energies=[0.0 for _ in range(len(atomic_types))],
        r_cutoff=5.0,
        n_polynomial_cutoff=3,
        data_seed=args.seed,
        model_seed=args.seed,
        device='cuda:0',
        train_data_path=f'../datasets/SPICE-train_{args.seed}.extxyz',
        valid_data_path=f'../datasets/SPICE-valid_{args.seed}.extxyz',
        model_path=f'../models/n_feats_{args.n_feats}-batch_size_{args.batch_size}-seed_{args.seed}',
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        n_hidden_feats=args.n_feats,
        n_product_feats=args.n_feats,
        n_workers=0,
        n_interactions=2,
        correlation=2,
        l_max_hidden_feats=1,
        l_max_edge_attrs=2,
        ke=14.399645351950548,
        ema=True,
        ema_decay=0.99,
        max_epoch=args.max_epoch,
        save_epoch=10,
        scheduler_patience=10,
        max_grad_norm=100.0,
        repulsion={
            'method': 'zbl',
            'r_cutoff': 2.0 * max(covalent_radii_list).item(),
            'n_polynomial_cutoff': 3
        },
        electrostatics={
            'method': 'coulomb',
            'r_cutoff': None
        },
        dispersion={
            'method': 'd4',
            'r_cutoff': 9.0,
            'Bohr': 0.5291772105638411,
            'Hartree': 27.211386024367243
        },
        train_loss={
            'type': 'weighted_sum',
            'losses': [
                {'type': 'energy_by_sqrt_atoms_sse'},
                {'type': 'forces_sse'},
                ],
            'weights': [
                1.0,
                0.05,
                ]
        },
        early_stopping_loss={
            'type': 'weighted_sum',
            'losses': [
                {'type': 'energy_by_sqrt_atoms_sse'},
                {'type': 'forces_sse'},
                ],
            'weights': [
                1.0,
                0.05,
                ]
        },
        eval_losses=[
            {'type': 'energy_per_atom_rmse'},
            {'type': 'energy_per_atom_mae'},
            {'type': 'forces_rmse'},
            {'type': 'forces_mae'},
        ],
        exclusion_radius=5.0,
        n_exclusion_polynomial_cutoff=3,
        use_charge_embedding=True,
        partial_charges='corrected',
        compute_regression_shift=True
    )
    
    # train model
    train(config)
    
    # evaluate model on the test subset from the SPICE data set (including configs)
    SUBSET_NAMES = [
        'Solvated_Amino_Acids',
        'Dipeptides',
        'DES370K_Monomers',
        'DES370K_Dimers',
        'PubChem',
        'Ion_Pairs',
        'Solvated_PubChem',
        'Water_Clusters',
        'Amino_Acid_Ligand',
        'QMugs',
        'NaCl_Water_Clusters',
        'Pentapeptides',
        'Large_Ligands',
        'Small_Ligands',
        'Biaryl',
        'TNet500'
    ]
    
    for SUBSET_NAME in SUBSET_NAMES:
        eval_config = copy.deepcopy(config)
        eval_config['test_data_path'] = f'../datasets/{SUBSET_NAME}-test_{args.seed}.extxyz'
        eval_config['eval_batch_size'] = 64
        eval_config['eval_losses'] = [
            {'type': 'energy_per_atom_rmse'},
            {'type': 'energy_per_atom_mae'},
            {'type': 'forces_rmse'},
            {'type': 'forces_mae'},
        ]
        errors = eval(eval_config)
        save_object(Path(eval_config['model_path']) / f'{SUBSET_NAME}-test_{args.seed}.json', errors, use_json=True)
        
        eval_configs(eval_config, file_name=f'{SUBSET_NAME}-test_{args.seed}.extxyz')
