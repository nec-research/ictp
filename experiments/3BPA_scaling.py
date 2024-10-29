import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from pathlib import Path
from typing import Optional, Dict, Union

import torch

import yaml
import argparse

from ictp.data.data import AtomicStructures, AtomicTypeConverter

from ictp.strategies import TrainingStrategy, EvaluationStrategy

from ictp.model.forward import load_model_from_folder
from ictp.model.calculators import StructurePropertyCalculator

from ictp.utils.config import update_config
from ictp.utils.misc import set_default_dtype, save_object
from ictp.utils.torch_geometric import DataLoader


def train(config: Optional[Union[str, Dict]] = None):
    
    # load config from config_file
    if isinstance(config, str):
        config = yaml.safe_load(Path(config).read_text())

    # update default config with provided parameters (is done twice as here we need to update data path)
    config = update_config(config)
    
    # define default dtype
    set_default_dtype(config['default_dtype'])
    
    # manage data
    if config['data_path']:
        atomic_structures = AtomicStructures.from_file(config['data_path'], **config)
        split = atomic_structures.random_split({'train': config['n_train'], 'valid': config['n_valid']},
                                               seed=config['data_seed'])
        if 'test' not in split:
            raise RuntimeError(f'Current split does not contain test data. In this case make sure to provide '
                               f'a separate test data path.')
    else:
        train_structures = AtomicStructures.from_file(config['train_data_path'], **config)
        valid_structures = AtomicStructures.from_file(config['valid_data_path'], **config)
        split = {'train': train_structures, 'valid': valid_structures}

    # run training
    training = TrainingStrategy(config)
    _ = training.run(train_structures=split['train'], valid_structures=split['valid'], folder=config['model_path'])


def inference_time(config: Optional[Union[str, Dict]] = None):
    # load config from config_file
    if isinstance(config, str):
        config = yaml.safe_load(Path(config).read_text())

    # update default config with provided parameters (is done twice as here we need to update data path)
    config = update_config(config)
    
    # define default dtype
    set_default_dtype(config['default_dtype'])

    # manage data
    if config['data_path']:
        atomic_structures = AtomicStructures.from_file(config['data_path'], **config)
        split = atomic_structures.random_split({'train': config['n_train'], 'valid': config['n_valid']},
                                               seed=config['data_seed'])
        if 'test' not in split:
            raise RuntimeError(f'Current split does not contain test data. In this case make sure to provide '
                               f'a separate test data path.')
        test_structures = split['test']
    else:
        test_structures = AtomicStructures.from_file(config['test_data_path'], **config)

    # load trained models and evaluate them on the test set
    model = load_model_from_folder(config['model_path'], key='best')
    
    evaluate = EvaluationStrategy(config)
    _ = evaluate.measure_inference_time(model, test_structures=test_structures, folder=config['model_path'], batch_size=10)


def memory_consumption(config: Optional[Union[str, Dict]] = None):
    # load config from config_file
    if isinstance(config, str):
        config = yaml.safe_load(Path(config).read_text())

    # update default config with provided parameters (is done twice as here we need to update data path)
    config = update_config(config)
    
    print('Before loading data (GB):', torch.cuda.max_memory_allocated() / 1024 ** 3)
    start_memory = torch.cuda.max_memory_allocated() / 1024 ** 3
    
    # define default dtype
    set_default_dtype(config['default_dtype'])

    # manage data
    if config['data_path']:
        atomic_structures = AtomicStructures.from_file(config['data_path'], **config)
        split = atomic_structures.random_split({'train': config['n_train'], 'valid': config['n_valid']},
                                               seed=config['data_seed'])
        if 'test' not in split:
            raise RuntimeError(f'Current split does not contain test data. In this case make sure to provide '
                               f'a separate test data path.')
        test_structures = split['test']
    else:
        test_structures = AtomicStructures.from_file(config['test_data_path'], **config)
    
    print('Before loading the model (GB):', torch.cuda.max_memory_allocated() / 1024 ** 3)
    
    # load trained models and evaluate them on the test set
    model = load_model_from_folder(config['model_path'], key='best')
    
    print('Before launching evaluation (GB):', torch.cuda.max_memory_allocated() / 1024 ** 3)
    
    config = update_config(config.copy())
        
    calc = StructurePropertyCalculator(model, training=False).to(config['device'])
    
    atomic_type_converter = AtomicTypeConverter.from_type_list(config['atomic_types'])
    
    test_structures = test_structures.to_type_names(atomic_type_converter, check=True)
    
    test_ds = test_structures.to_data(r_cutoff=config['r_cutoff'], n_species=atomic_type_converter.get_n_type_names())
    test_dl = DataLoader(test_ds, batch_size=10, shuffle=False, drop_last=False)
    
    batch = next(iter(test_dl)).to(config['device'])
    
    for _ in range(10):
        calc(batch, forces=True, features=False)
    
    for _ in range(100):
        calc(batch, forces=True, features=False)
        
    print('After performing evaluation (GB):', torch.cuda.max_memory_allocated() / 1024 ** 3)
    
    end_memory = torch.cuda.max_memory_allocated() / 1024 ** 3
    
    to_save = {'total_memory': end_memory - start_memory}
    save_object(Path(config['model_path']) / f'memory_results.json', to_save, use_json=True)


if __name__ == '__main__':
    
    # define relevant parameters
    parser = argparse.ArgumentParser(description='Parameters', fromfile_prefix_chars='@')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--l_max_hidden_feats', type=int)
    parser.add_argument('--l_max_edge_attrs', type=int)
    parser.add_argument('--correlation', type=int)

    # parse parameters
    args = parser.parse_args()
    
    config = dict(train_data_path=f'../datasets/3BPA/train_valid_splits/train_300K_{args.seed}.extxyz',
                  valid_data_path=f'../datasets/3BPA/train_valid_splits/valid_300K_{args.seed}.extxyz',
                  data_seed=args.seed, 
                  atomic_types=['H', 
                                'C', 
                                'N', 
                                'O'], 
                  atomic_energies=[-13.587222780835477, 
                                   -1029.4889999855063, 
                                   -1484.9814568572233, 
                                   -2041.9816003861047],
                  r_cutoff=5.0,
                  n_train=450,
                  n_valid=50,
                  train_batch_size=5,
                  eval_batch_size=10,
                  model_path=f'../results/3BPA_scaling/float64-l_max_hidden_feats_{args.l_max_hidden_feats}-l_max_edge_attrs_{args.l_max_edge_attrs}-correlation_{args.correlation}/seed_{args.seed}',
                  model_seed=args.seed,
                  default_dtype='float64',
                  n_hidden_feats=8,
                  n_product_feats=8,
                  n_interactions=2,
                  l_max_hidden_feats=args.l_max_hidden_feats,
                  l_max_edge_attrs=args.l_max_edge_attrs,
                  correlation=args.correlation,
                  coupled_product_feats=False,
                  symmetric_product=False,
                  max_epoch=1,
                  scheduler_patience=50,
                  ema=True,
                  ema_decay=0.99,
                  train_loss={
                      'type': 'weighted_sum',
                      'losses': [
                          {'type': 'energy_by_sqrt_atoms_sse'},
                          {'type': 'forces_sse'}
                          ],
                      'weights': [
                          1.0,
                          10.0
                          ]
                      },
                  early_stopping_loss={
                      'type': 'weighted_sum',
                      'losses': [
                          {'type': 'energy_by_sqrt_atoms_sse'},
                          {'type': 'forces_sse'}
                          ],
                      'weights': [
                          1.0,
                          10.0
                          ]
                      },
                  eval_losses=[
                      {'type': 'energy_per_atom_rmse'},
                      {'type': 'energy_per_atom_mae'},
                      {'type': 'forces_rmse'},
                      {'type': 'forces_mae'}
                      ]
                  )

    # train model
    train(config)
    
    # measure inference time
    # new_config = config.copy()
    # new_config['test_data_path'] = f'../datasets/3BPA/test_1200K.xyz'
    # inference_time(new_config)
    
    # measure memory consumption
    # new_config = config.copy()
    # new_config['test_data_path'] = f'../datasets/3BPA/test_1200K.xyz'
    # memory_consumption(new_config)
