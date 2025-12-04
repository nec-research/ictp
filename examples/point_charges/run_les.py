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

import torch
from torch.utils.data import Subset

from ase.io import read
from ase.data import atomic_numbers, covalent_radii

from ictp.data.datasets import DatasetHandler
from ictp.strategies import EvaluationStrategy
from ictp.training.callbacks import FileLoggingCallback
from ictp.training.loss_fns import config_to_loss
from ictp.training.trainer import Trainer
from ictp.model.pair_potentials import ElectrostaticEnergy, EwaldElectrostaticEnergy
from ictp.model.forward import ForwardAtomisticNetwork, build_model, find_last_ckpt
from ictp.utils.config import update_config
from ictp.utils.misc import (set_default_dtype,
                             find_max_r_cutoff,
                             save_object)
from ictp.utils.math import segment_sum


class LESCoulombElectrostaticEnergy(ElectrostaticEnergy):
    def __init__(
        self,
        alpha: float = 1.0,
        **config: Any
    ):
        super(LESCoulombElectrostaticEnergy, self).__init__(**config)
        self.alpha = alpha
        self.alpha_sq = self.alpha ** 2
        
    def get_electrostatic_energy(
        self, 
        graph: Dict[str, torch.Tensor],
        results: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        edge_index, lengths = graph['edge_index'], graph['lengths'].squeeze()
        idx_i, idx_j = edge_index[0, :], edge_index[1, :]
        
        if self.r_cutoff is not None:
            # That is a kind of strange setting, imo. A certain \alpha implies a real-space cutoff.
            raise RuntimeError()
        
        if self.exclusion_radius is not None:
            raise RuntimeError()
        
        partial_charges = results['partial_charges']
        
        # compute Coulomb factor
        fac = self.ke_half * partial_charges.index_select(0, idx_i) * partial_charges.index_select(0, idx_j)
        
        # compute Coulomb contributions
        coulomb = 1.0 / lengths
        
        # combine everything and get pairwise, atomic, and total energies
        pairwise_energies = fac * coulomb
        pairwise_energies = pairwise_energies * torch.special.erf(self.alpha * lengths)
        
        atomic_energies = segment_sum(pairwise_energies, idx_i, partial_charges.shape[0])
        
        return atomic_energies
    
    def __repr__(self):
        return (f'{self.__class__.__name__}(r_cutoff={self.r_cutoff}, ke={self.ke}, '
                f'exclusion_radius={self.exclusion_radius}, n_exclusion_polynomial_cutoff={self.n_exclusion_polynomial_cutoff})')
        
        
class LESEwaldElectrostaticEnergy(EwaldElectrostaticEnergy):
    def get_electrostatic_energy(
        self, 
        graph: Dict[str, torch.Tensor],
        results: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if self.exclusion_radius is not None:
            raise RuntimeError()
        return self.get_energy_recip(graph, results)
    
    def __repr__(self):
        return (f'{self.__class__.__name__}(r_cutoff={self.r_cutoff}, ke={self.ke}, '
                f'alpha={self.alpha}, k_max={self.k_max}, '
                f'exclusion_radius={self.exclusion_radius}, n_exclusion_polynomial_cutoff={self.n_exclusion_polynomial_cutoff})')


def train(
    config: Optional[Union[str, Dict]] = None,
    new_pair_potential_config: Optional[Dict[str, Any]] = None
) -> None:
    
    # load config
    if isinstance(config, str):
        config = yaml.safe_load(Path(config).read_text())

    # update default config
    config = update_config(config.copy())
    
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
    
    # store data set sizes in the configuration
    n_train = len(split['train'])
    n_valid = len(split['valid'])
    config['n_train'] = n_train
    config['n_valid'] = n_valid
    
    # retrieve the atomic type converter from the data set
    atomic_type_converter = (
        split['train'].dataset.atomic_type_converter
        if isinstance(split['train'], Subset) else split['train'].atomic_type_converter
    )
        
    # build the model using data set information and configuration
    model = build_model(
        dataset=split['train'], 
        n_species=atomic_type_converter.get_n_type_names(),
        Z_max=int(atomic_type_converter._to_atomic_numbers.max()), 
        **config,
    )
    
    # get the new pair potential for model
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
            if config['electrostatics']['method'] == 'les_coulomb':
                new_pair_potential = LESCoulombElectrostaticEnergy(
                    r_cutoff=config['electrostatics']['r_cutoff'],
                    ke=config['ke'],
                    alpha=config['electrostatics']['alpha'],
                    exclusion_radius=config['exclusion_radius'],
                    n_exclusion_polynomial_cutoff=config['n_exclusion_polynomial_cutoff']
                )
            elif config['electrostatics']['method'] == 'les_ewald':
                new_pair_potential = LESEwaldElectrostaticEnergy(
                    r_cutoff=config['electrostatics']['r_cutoff'],
                    ke=config['ke'],
                    alpha=config['electrostatics']['alpha'],
                    k_max=config['electrostatics']['k_max'],
                    exclusion_radius=config['exclusion_radius'],
                    n_exclusion_polynomial_cutoff=config['n_exclusion_polynomial_cutoff']
                )
            else:
                valid_values = ['les_coulomb', 'les_ewald']
                raise ValueError(
                    f"Invalid value for {config['electrostatics']['method']=}. "
                    f"Expected one of {valid_values}."
                )
        model = model.replace_pair_potential(new_pair_potential)
        
    # define training and evaluation losses from configuration
    train_loss = config_to_loss(config['train_loss'])
    eval_losses = {l['type']: config_to_loss(l) for l in config['eval_losses']}
    early_stopping_loss = config_to_loss(config['early_stopping_loss'])
    
    # set up callbacks for tracking training progress
    callbacks = [FileLoggingCallback()]
    
    # configure the trainer with model and training parameters
    trainer = Trainer(
        model=model,
        model_path=config['model_path'],
        callbacks=callbacks,
        lr=config['lr'],
        lr_factor=config['lr_factor'],
        scheduler_patience=config['scheduler_patience'],
        max_epoch=config['max_epoch'],
        save_epoch=config['save_epoch'],
        validate_epoch=config['valid_epoch'],
        train_batch_size=min(config['train_batch_size'], n_train),
        valid_batch_size=min(config['eval_batch_size'], n_valid),
        n_workers=config['n_workers'],
        train_loss=train_loss,
        eval_losses=eval_losses,
        early_stopping_loss=early_stopping_loss,
        max_grad_norm=config['max_grad_norm'],
        device=config['device'],
        amsgrad=config['amsgrad'],
        weight_decay=config['weight_decay'],
        ema=config['ema'],
        ema_decay=config['ema_decay'],
        with_sam=config['with_sam'],
        rho_sam=config['rho_sam'],
        adaptive_sam=config['adaptive_sam'],
        with_torch_script=True,
        with_torch_compile=False,
    )
    
    # Train the model
    trainer.fit(train_dataset=split['train'], valid_dataset=split['valid'])


def load_model_with_les(model_path: str, key: str = 'best') -> ForwardAtomisticNetwork:
    ckpt_dir = Path(find_last_ckpt(Path(model_path) / key))
    cfg_path = ckpt_dir / 'config.yaml'
    params_path = ckpt_dir / 'params.pkl'
    
    config = yaml.safe_load(cfg_path.read_text())
    m = (config.get('electrostatics') or {}).get('method')
    if m in ('les_coulomb', 'les_ewald'):
        config['electrostatics']['method'] = 'coulomb' if m == 'les_coulomb' else 'ewald'
    
    model = build_model(None, **config)
    model.load_params(params_path)
    return model


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
    model = load_model_with_les(config['model_path'], key='best')
    
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
            if config['electrostatics']['method'] == 'les_coulomb':
                new_pair_potential = LESCoulombElectrostaticEnergy(
                    r_cutoff=config['electrostatics']['r_cutoff'],
                    ke=config['ke'],
                    alpha=config['electrostatics']['alpha'],
                    exclusion_radius=config['exclusion_radius'],
                    n_exclusion_polynomial_cutoff=config['n_exclusion_polynomial_cutoff']
                )
            elif config['electrostatics']['method'] == 'les_ewald':
                new_pair_potential = LESEwaldElectrostaticEnergy(
                    r_cutoff=config['electrostatics']['r_cutoff'],
                    ke=config['ke'],
                    alpha=config['electrostatics']['alpha'],
                    k_max=config['electrostatics']['k_max'],
                    exclusion_radius=config['exclusion_radius'],
                    n_exclusion_polynomial_cutoff=config['n_exclusion_polynomial_cutoff']
                )
            else:
                valid_values = ['les_coulomb', 'les_ewald']
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
    model = load_model_with_les(config['model_path'], key='best')
    
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
            if config['electrostatics']['method'] == 'les_coulomb':
                new_pair_potential = LESCoulombElectrostaticEnergy(
                    r_cutoff=config['electrostatics']['r_cutoff'],
                    ke=config['ke'],
                    alpha=config['electrostatics']['alpha'],
                    exclusion_radius=config['exclusion_radius'],
                    n_exclusion_polynomial_cutoff=config['n_exclusion_polynomial_cutoff']
                )
            elif config['electrostatics']['method'] == 'les_ewald':
                new_pair_potential = LESEwaldElectrostaticEnergy(
                    r_cutoff=config['electrostatics']['r_cutoff'],
                    ke=config['ke'],
                    alpha=config['electrostatics']['alpha'],
                    k_max=config['electrostatics']['k_max'],
                    exclusion_radius=config['exclusion_radius'],
                    n_exclusion_polynomial_cutoff=config['n_exclusion_polynomial_cutoff']
                )
            else:
                valid_values = ['les_coulomb', 'les_ewald']
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
        model_path=f'models/cluster_{args.size}-les-seed_{args.seed}',
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
        exclusion_radius=None,
        n_exclusion_polynomial_cutoff=3,
        use_charge_embedding=False,
        partial_charges='corrected',
        compute_regression_shift=False
    )
    
    # train model
    atoms = read(f'../../datasets/point_charges/periodic-512_atoms-train/box_edge_40.0.extxyz')
    ewald_params = EwaldParameters(r_cutoff=9.0, edge_length=np.max(atoms.get_cell()), tolerance=5e-5)
    
    new_pair_potential_config = dict(
        electrostatics={
            'method': 'les_coulomb',
            'r_cutoff': None,
            'alpha': ewald_params.alpha,
        }
    )
    
    train(config, new_pair_potential_config)
    
    # evaluate model
    
    # periodic (tolerace 5e-5)
    eval_config = copy.deepcopy(config)
    eval_config['test_data_path'] = f'../../datasets/point_charges/periodic-512_atoms-test/box_edge_40.0.extxyz'
    
    atoms = read(f'../../datasets/point_charges/periodic-512_atoms-test/box_edge_40.0.extxyz')
    ewald_params = EwaldParameters(r_cutoff=9.0, edge_length=np.max(atoms.get_cell()), tolerance=5e-5)
    
    new_pair_potential_config = dict(
        electrostatics={
            'method': 'les_ewald',
            'r_cutoff': 9.0,
            'alpha': ewald_params.alpha,
            'k_max': ewald_params.compute_k_max_ewald()
        }
    )

    errors = eval(
        eval_config,
        new_pair_potential_config
    )
    save_object(Path(eval_config['model_path']) / f'box_edge_40.0-les.json', errors, use_json=True)
    
    eval_configs(
        eval_config,
        f'box_edge_40.0-les.extxyz',
        new_pair_potential_config
    )
    
    # isolated (tolerace 5e-5)
    for n_atoms in [90, 180, 270]:
        eval_config = copy.deepcopy(config)
        eval_config['test_data_path'] = f'../../datasets/point_charges/periodic-512_atoms-test/box_edge_40.0_cluster_{n_atoms}.extxyz'

        atoms = read(f'../../datasets/point_charges/periodic-512_atoms-train/box_edge_40.0.extxyz')
        ewald_params = EwaldParameters(r_cutoff=9.0, edge_length=np.max(atoms.get_cell()), tolerance=5e-5)
        
        new_pair_potential_config = dict(
            electrostatics={
                'method': 'les_coulomb',
                'r_cutoff': None,
                'alpha': ewald_params.alpha,
            }
        )
        
        errors = eval(
            eval_config,
            new_pair_potential_config
        )
        save_object(Path(eval_config['model_path']) / f'box_edge_40.0_cluster_{n_atoms}-les.json', errors, use_json=True)
        
        eval_configs(
            eval_config,
            f'box_edge_40.0_cluster_{n_atoms}-les.extxyz',
            new_pair_potential_config
        )
    
    # periodic (tolerace 5e-6)
    eval_config = copy.deepcopy(config)
    eval_config['test_data_path'] = f'../../datasets/point_charges/periodic-512_atoms-test/box_edge_40.0.extxyz'
    
    atoms = read(f'../../datasets/point_charges/periodic-512_atoms-test/box_edge_40.0.extxyz')
    ewald_params = EwaldParameters(r_cutoff=9.0, edge_length=np.max(atoms.get_cell()), tolerance=5e-6)
    
    new_pair_potential_config = dict(
        electrostatics={
            'method': 'les_ewald',
            'r_cutoff': 9.0,
            'alpha': ewald_params.alpha,
            'k_max': ewald_params.compute_k_max_ewald()
        }
    )

    errors = eval(
        eval_config,
        new_pair_potential_config
    )
    save_object(Path(eval_config['model_path']) / f'box_edge_40.0-les-tol_5e6.json', errors, use_json=True)
    
    eval_configs(
        eval_config,
        f'box_edge_40.0-les-tol_5e6.extxyz',
        new_pair_potential_config
    )
    
    # isolated (tolerace 5e-6)
    for n_atoms in [90, 180, 270]:
        eval_config = copy.deepcopy(config)
        eval_config['test_data_path'] = f'../../datasets/point_charges/periodic-512_atoms-test/box_edge_40.0_cluster_{n_atoms}.extxyz'

        atoms = read(f'../../datasets/point_charges/periodic-512_atoms-train/box_edge_40.0.extxyz')
        ewald_params = EwaldParameters(r_cutoff=9.0, edge_length=np.max(atoms.get_cell()), tolerance=5e-6)
        
        new_pair_potential_config = dict(
            electrostatics={
                'method': 'les_coulomb',
                'r_cutoff': None,
                'alpha': ewald_params.alpha,
            }
        )
        
        errors = eval(
            eval_config,
            new_pair_potential_config
        )
        save_object(Path(eval_config['model_path']) / f'box_edge_40.0_cluster_{n_atoms}-les-tol_5e6.json', errors, use_json=True)
        
        eval_configs(
            eval_config,
            f'box_edge_40.0_cluster_{n_atoms}-les-tol_5e6.extxyz',
            new_pair_potential_config
        )
        
    