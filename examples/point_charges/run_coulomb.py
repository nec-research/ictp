import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import math
import numpy as np
import torch.nn as nn

from typing import *
from pathlib import Path

import argparse

import yaml
import copy

from ase.io import read
from ase.data import atomic_numbers, covalent_radii

from ictp.data.datasets import DatasetHandler
from ictp.strategies import TrainingStrategy, EvaluationStrategy
from ictp.model.pair_potentials import (CoulombElectrostaticEnergy,
                                        EwaldElectrostaticEnergy,
                                        SPMEElectrostaticEnergy)
from ictp.model.forward import load_model_from_folder
from ictp.utils.config import update_config
from ictp.utils.misc import (set_default_dtype,
                             find_max_r_cutoff,
                             save_object)


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
    
    dataset = dataset_handler.load_dataset(
        file_path=config['data_path'],
        r_cutoff=find_max_r_cutoff(config),
        skin=0.0,
        atomic_types=config['atomic_types'],
        neighbors=config['neighbors']
    )
    
    split = dataset_handler.split_dataset(
        dataset,
        sizes={'train': config['n_train'], 'valid': config['n_valid']},
        seed=config['data_seed']
    )
    
    # run training
    training = TrainingStrategy(config)
    _ = training.run(
        train_dataset=split['train'],
        valid_dataset=split['valid'],
        folder=config['model_path'],
        with_torch_script=True,
        with_torch_compile=False,
    )


def eval(
    config: Optional[Union[str, Dict]] = None,
    new_pair_potential_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    # load config
    if isinstance(config, str):
        config = yaml.safe_load(Path(config).read_text())

    # update default config
    config = update_config(config)
    
    # define default dtype
    set_default_dtype(config['default_dtype'])

    # load trained models and evaluate them on the test set
    model = load_model_from_folder(config['model_path'], key='best')
    
    if new_pair_potential_config is not None:
        for pair_potential, params in new_pair_potential_config.items():
            if pair_potential not in config:
                raise ValueError(
                    f"Unknown pair potential type: {pair_potential=}"
                )
            
            if pair_potential != "electrostatics":
                raise ValueError(
                    f"Unsupported pair potential type: {pair_potential=}. "
                    f"This method only supports exchanging 'electrostatics' modules. "
                    f"Trainable modules like 'repulsion' or 'dispersion' must be handled explicitly."
                )
            config[pair_potential].update(params)
    
        new_pair_potential: Optional[nn.Module] = None
        
        if config['electrostatics']['method'] is not None:
            if config['electrostatics']['method'] == 'coulomb':
                new_pair_potential = CoulombElectrostaticEnergy(
                    r_cutoff=config['electrostatics']['r_cutoff'],
                    ke=config['ke'],
                    exclusion_radius=config['exclusion_radius'],
                    n_exclusion_polynomial_cutoff=config['n_exclusion_polynomial_cutoff']
                )
            elif config['electrostatics']['method'] == 'ewald':
                new_pair_potential = EwaldElectrostaticEnergy(
                    r_cutoff=config['electrostatics']['r_cutoff'],
                    ke=config['ke'],
                    alpha=config['electrostatics']['alpha'],
                    k_max=config['electrostatics']['k_max'],
                    exclusion_radius=config['exclusion_radius'],
                    n_exclusion_polynomial_cutoff=config['n_exclusion_polynomial_cutoff']
                )
            elif config['electrostatics']['method'] == 'spme':
                new_pair_potential = SPMEElectrostaticEnergy(
                    r_cutoff=config['electrostatics']['r_cutoff'],
                    ke=config['ke'],
                    alpha=config['electrostatics']['alpha'],
                    k_max=config['electrostatics']['k_max'],
                    spline_order=config['electrostatics']['spline_order'],
                    exclusion_radius=config['exclusion_radius'],
                    n_exclusion_polynomial_cutoff=config['n_exclusion_polynomial_cutoff']
                )
            else:
                valid_values = ['coulomb', 'ewald', 'spme']
                raise ValueError(
                    f"Invalid value for {config['electrostatics']['method']=}. "
                    f"Expected one of {valid_values}."
                )
        
        model = model.replace_pair_potential(new_pair_potential)
    
    # manage data
    dataset_handler = DatasetHandler()
    
    test_dataset = dataset_handler.load_dataset(
        file_path=config['test_data_path'],
        r_cutoff=find_max_r_cutoff(config),
        skin=0.0,
        atomic_types=config['atomic_types'],
        neighbors=config['neighbors']
    )
    
    evaluate = EvaluationStrategy(config)
    
    errors = evaluate.run(
        model=model,
        test_dataset=test_dataset,
        folder=config['model_path'],
        with_torch_script=False,
        with_torch_compile=True,
    )
    
    return errors


def eval_configs(
    config: Optional[Union[str, Dict]] = None,
    file_name: str = 'configs.extxyz',
    new_pair_potential_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    # load config
    if isinstance(config, str):
        config = yaml.safe_load(Path(config).read_text())

    # update default config
    config = update_config(config)
    
    # define default dtype
    set_default_dtype(config['default_dtype'])

    # load trained models and evaluate them on the test set
    model = load_model_from_folder(config['model_path'], key='best')
    
    if new_pair_potential_config is not None:
        for pair_potential, params in new_pair_potential_config.items():
            if pair_potential not in config:
                raise ValueError(
                    f"Unknown pair potential type: {pair_potential=}"
                )
            
            if pair_potential != "electrostatics":
                raise ValueError(
                    f"Unsupported pair potential type: {pair_potential=}. "
                    f"This method only supports exchanging 'electrostatics' modules. "
                    f"Trainable modules like 'repulsion' or 'dispersion' must be handled explicitly."
                )
            config[pair_potential].update(params)
    
        new_pair_potential: Optional[nn.Module] = None
        
        if config['electrostatics']['method'] is not None:
            if config['electrostatics']['method'] == 'coulomb':
                new_pair_potential = CoulombElectrostaticEnergy(
                    r_cutoff=config['electrostatics']['r_cutoff'],
                    ke=config['ke'],
                    exclusion_radius=config['exclusion_radius'],
                    n_exclusion_polynomial_cutoff=config['n_exclusion_polynomial_cutoff']
                )
            elif config['electrostatics']['method'] == 'ewald':
                new_pair_potential = EwaldElectrostaticEnergy(
                    r_cutoff=config['electrostatics']['r_cutoff'],
                    ke=config['ke'],
                    alpha=config['electrostatics']['alpha'],
                    k_max=config['electrostatics']['k_max'],
                    exclusion_radius=config['exclusion_radius'],
                    n_exclusion_polynomial_cutoff=config['n_exclusion_polynomial_cutoff']
                )
            elif config['electrostatics']['method'] == 'spme':
                new_pair_potential = SPMEElectrostaticEnergy(
                    r_cutoff=config['electrostatics']['r_cutoff'],
                    ke=config['ke'],
                    alpha=config['electrostatics']['alpha'],
                    k_max=config['electrostatics']['k_max'],
                    spline_order=config['electrostatics']['spline_order'],
                    exclusion_radius=config['exclusion_radius'],
                    n_exclusion_polynomial_cutoff=config['n_exclusion_polynomial_cutoff']
                )
            else:
                valid_values = ['coulomb', 'ewald', 'spme']
                raise ValueError(
                    f"Invalid value for {config['electrostatics']['method']=}. "
                    f"Expected one of {valid_values}."
                )
        
        model = model.replace_pair_potential(new_pair_potential)
    
    # manage data
    dataset_handler = DatasetHandler()
    
    test_dataset = dataset_handler.load_dataset(
        file_path=config['test_data_path'],
        r_cutoff=find_max_r_cutoff(config),
        skin=0.0,
        atomic_types=config['atomic_types'],
        neighbors=config['neighbors']
    )
    
    evaluate = EvaluationStrategy(config)
    
    evaluate.run_on_configs(
        model=model,
        test_dataset=test_dataset,
        folder=config['model_path'],
        file_name=file_name,
        with_torch_script=False,
        with_torch_compile=True,
    )
    

class EwaldParameters:
    def __init__(
        self,
        r_cutoff: float,
        edge_length: float,
        tolerance: float = 5e-5,
    ):
        self.r_cutoff = r_cutoff
        self.edge_length = edge_length
        self.tolerance = tolerance
        self.alpha = math.sqrt(-math.log(2.0 * self.tolerance)) / self.r_cutoff
    
    def ewald_error(
        self,
        k_max: int,
        box_length: float
    ) -> float:
        temp = k_max * math.pi / (box_length * self.alpha)
        return self.tolerance - 0.05 * math.sqrt(box_length * self.alpha) * k_max * math.exp(-temp ** 2)

    def specialized_ewald_error(self, x):
        return self.ewald_error(x, self.edge_length)
    
    def find_zero(
        self,
        function,
        initial_guess: int = 10
    ) -> int:
        current_arg = initial_guess
        value = function(current_arg)
        if value > 0.0:
            while value > 0 and current_arg > 0:
                current_arg -= 1
                value = function(current_arg)
            return current_arg + 1

        while value < 0.0:
            current_arg += 1
            value = function(current_arg)
        return current_arg
    
    def compute_k_max_ewald(self):
        return int(self.find_zero(self.specialized_ewald_error))
    
    def compute_k_max_spme(self):
        return int(math.ceil(2 * self.alpha * self.edge_length / (3 * math.pow(self.tolerance, 0.2))) / 2 - 1)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Parameters', fromfile_prefix_chars='@')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--size', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)
    
    args = parser.parse_args()
    
    atomic_types = ['Na', 'Cl']
    atomic_numbers_list = [atomic_numbers[symbol] for symbol in atomic_types]
    covalent_radii_list = [covalent_radii[number] for number in atomic_numbers_list]
    
    config = dict(
        atomic_types=atomic_types,
        atomic_energies=[0.0 for _ in range(len(atomic_types))],
        r_cutoff=5.0,
        n_polynomial_cutoff=3,
        data_seed=args.seed,
        model_seed=args.seed,
        device=f'cuda:{args.device}',
        data_path=f'../../datasets/point_charges/periodic-512_atoms-train/box_edge_40.0_cluster_{args.size}.extxyz',
        model_path=f'models/cluster_{args.size}-coulomb-seed_{args.seed}',
        n_train=900,
        n_valid=100,
        train_batch_size=10,
        eval_batch_size=10,
        n_hidden_feats=32,
        n_product_feats=32,
        n_workers=10,
        n_interactions=1,
        correlation=3,
        l_max_hidden_feats=1,
        l_max_edge_attrs=2,
        ke=14.399645351950548,
        ema=True,
        ema_decay=0.99,
        max_epoch=250,
        save_epoch=50,
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
                1.0,
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
        use_charge_embedding=False,
        partial_charges='corrected',
        compute_regression_shift=False
    )
    
    # train model
    train(config)
    
    # evaluate model
    
    # periodic
    eval_config = copy.deepcopy(config)
    eval_config['test_data_path'] = f'../../datasets/point_charges/periodic-512_atoms-test/box_edge_40.0.extxyz'
    
    atoms = read(f'../../datasets/point_charges/periodic-512_atoms-test/box_edge_40.0.extxyz')
    ewald_params = EwaldParameters(r_cutoff=9.0, edge_length=np.max(atoms.get_cell()), tolerance=5e-5)
    
    new_pair_potential_config = dict(
        electrostatics={
            'method': 'spme',
            'r_cutoff': 9.0,
            'alpha': ewald_params.alpha,
            'k_max': ewald_params.compute_k_max_spme(),
            'spline_order': 5
        }
    )

    errors = eval(
        eval_config,
        new_pair_potential_config
    )
    save_object(Path(eval_config['model_path']) / f'box_edge_40.0-coulomb.json', errors, use_json=True)
    
    eval_configs(
        eval_config,
        f'box_edge_40.0-coulomb.extxyz',
        new_pair_potential_config
    )
    
    # isolated
    for n_atoms in [90, 180, 270]:
        eval_config = copy.deepcopy(config)
        eval_config['test_data_path'] = f'../../datasets/point_charges/periodic-512_atoms-test/box_edge_40.0_cluster_{n_atoms}.extxyz'

        new_pair_potential_config = dict(
            electrostatics={
                'method': 'coulomb',
                'r_cutoff': None,
            }
        )
        
        errors = eval(
            eval_config,
            new_pair_potential_config
        )
        save_object(Path(eval_config['model_path']) / f'box_edge_40.0_cluster_{n_atoms}-coulomb.json', errors, use_json=True)
        
        eval_configs(
            eval_config,
            f'box_edge_40.0_cluster_{n_atoms}-coulomb.extxyz',
            new_pair_potential_config
        )
    
    # with cutoff
    for r_cutoff in [9.0, 18.0, 27.0]:
        # periodic
        eval_config = copy.deepcopy(config)
        eval_config['test_data_path'] = f'../../datasets/point_charges/periodic-512_atoms-test/box_edge_40.0.extxyz'
        
        new_pair_potential_config = dict(
            electrostatics={
                'method': 'coulomb',
                'r_cutoff': r_cutoff
            }
        )

        errors = eval(
            eval_config,
            new_pair_potential_config
        )
        save_object(Path(eval_config['model_path']) / f'box_edge_40.0-coulomb_cutoff_{r_cutoff}.json', errors, use_json=True)
        
        eval_configs(
            eval_config,
            f'box_edge_40.0-coulomb_cutoff_{r_cutoff}.extxyz',
            new_pair_potential_config
        )
        
        # isolated
        for n_atoms in [90, 180, 270]:
            eval_config = copy.deepcopy(config)
            eval_config['test_data_path'] = f'../../datasets/point_charges/periodic-512_atoms-test/box_edge_40.0_cluster_{n_atoms}.extxyz'

            new_pair_potential_config = dict(
                electrostatics={
                    'method': 'coulomb',
                    'r_cutoff': r_cutoff,
                }
            )
            
            errors = eval(
                eval_config,
                new_pair_potential_config
            )
            save_object(Path(eval_config['model_path']) / f'box_edge_40.0_cluster_{n_atoms}-coulomb_cutoff_{r_cutoff}.json', errors, use_json=True)
            
            eval_configs(
                eval_config,
                f'box_edge_40.0_cluster_{n_atoms}-coulomb_cutoff_{r_cutoff}.extxyz',
                new_pair_potential_config
            )

    # tolerance 5e-6 (running it for isolated systems makes no sense)
    # periodic
    eval_config = copy.deepcopy(config)
    eval_config['test_data_path'] = f'../../datasets/point_charges/periodic-512_atoms-test/box_edge_40.0.extxyz'
    
    atoms = read(f'../../datasets/point_charges/periodic-512_atoms-test/box_edge_40.0.extxyz')
    ewald_params = EwaldParameters(r_cutoff=9.0, edge_length=np.max(atoms.get_cell()), tolerance=5e-6)
    
    new_pair_potential_config = dict(
        electrostatics={
            'method': 'spme',
            'r_cutoff': 9.0,
            'alpha': ewald_params.alpha,
            'k_max': ewald_params.compute_k_max_spme(),
            'spline_order': 5
        }
    )

    errors = eval(
        eval_config,
        new_pair_potential_config
    )
    save_object(Path(eval_config['model_path']) / f'box_edge_40.0-coulomb-tol_5e6.json', errors, use_json=True)
    
    eval_configs(
        eval_config,
        f'box_edge_40.0-coulomb-tol_5e6.extxyz',
        new_pair_potential_config
    )
    
    # isolated
    for n_atoms in [90, 180, 270]:
        eval_config = copy.deepcopy(config)
        eval_config['test_data_path'] = f'../../datasets/point_charges/periodic-512_atoms-test/box_edge_40.0_cluster_{n_atoms}.extxyz'

        new_pair_potential_config = dict(
            electrostatics={
                'method': 'coulomb',
                'r_cutoff': None,
            }
        )
        
        errors = eval(
            eval_config,
            new_pair_potential_config
        )
        save_object(Path(eval_config['model_path']) / f'box_edge_40.0_cluster_{n_atoms}-coulomb-tol_5e6.json', errors, use_json=True)
        
        eval_configs(
            eval_config,
            f'box_edge_40.0_cluster_{n_atoms}-coulomb-tol_5e6.extxyz',
            new_pair_potential_config
        )
        