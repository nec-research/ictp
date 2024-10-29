import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from pathlib import Path
from typing import Optional, Dict, Union

import torch

import yaml
import argparse

from ictp.data.data import AtomicStructures, AtomicTypeConverter

from ictp.model.calculators import StructurePropertyCalculator

from ictp.strategies import TrainingStrategy, EvaluationStrategy

from ictp.model.forward import load_model_from_folder

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
    else:
        train_structures = AtomicStructures.from_file(config['train_data_path'], **config)
        valid_structures = AtomicStructures.from_file(config['valid_data_path'], **config)
        split = {'train': train_structures, 'valid': valid_structures}

    # run training
    training = TrainingStrategy(config)
    _ = training.run(train_structures=split['train'], valid_structures=split['valid'], folder=config['model_path'])
    
    
def eval(config: Optional[Union[str, Dict]] = None):

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
    errors = evaluate.run(model, test_structures=test_structures, folder=config['model_path'])
    return errors


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
    _ = evaluate.measure_inference_time(model, test_structures=test_structures, folder=config['model_path'], batch_size=50)


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
    test_dl = DataLoader(test_ds, batch_size=50, shuffle=False, drop_last=False)
    
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
    parser.add_argument('--n_hidden_feats', type=int)
    parser.add_argument('--n_product_feats', type=int)
    parser.add_argument('--coupled_product_feats', type=int)
    parser.add_argument('--symmetric_product', type=int)
    parser.add_argument('--ema', type=int)
    parser.add_argument('--l_max_hidden_feats', type=int)
    parser.add_argument('--l_max_edge_attrs', type=int)

    # parse parameters
    args = parser.parse_args()
    
    config = dict(train_data_path=f'../datasets/HEA/train_cfgs_{args.seed}.extxyz', # Note: use train_data_path=f'../datasets/HEA/total.train_cfgs_{args.seed}.extxyz' for the deformed test
                  valid_data_path=f'../datasets/HEA/valid_cfgs_{args.seed}.extxyz', # Note: use train_data_path=f'../datasets/HEA/total.valid_cfgs_{args.seed}.extxyz' for the deformed test
                  data_seed=args.seed,
                  atomic_types=['Ta', 'V', 'Cr', 'W'],
                  atomic_energies=[0., 0., 0., 0.],
                  r_cutoff=5.0,
                  n_train=4873, # Note: use n_train=6211  for the deformed test
                  n_valid=500,
                  train_batch_size=32,
                  eval_batch_size=32,
                  model_path=f'../results/HEA/float32-l_max_hidden_feats_{args.l_max_hidden_feats}-l_max_edge_attrs_{args.l_max_edge_attrs}-hidden_{args.n_hidden_feats}-product_{args.n_product_feats}_coupled_{bool(args.coupled_product_feats)}_symmetric_{bool(args.symmetric_product)}-ema_{bool(args.ema)}-max_epoch_1000-patience_50-lr_0.01-fweight_0.01-vweight_0.001/seed_{args.seed}',
                  model_seed=args.seed,
                  default_dtype='float32',
                  n_hidden_feats=args.n_hidden_feats,
                  n_product_feats=args.n_product_feats,
                  n_interactions=2,
                  correlation=3,    # Note: use correlation=2 for the TensorNet-like architecture
                  l_max_hidden_feats=args.l_max_hidden_feats,
                  l_max_edge_attrs=args.l_max_edge_attrs,
                  coupled_product_feats=bool(args.coupled_product_feats),
                  symmetric_product=bool(args.symmetric_product),
                  max_epoch=1000,
                  scheduler_patience=50,
                  lr=0.01,
                  ema=bool(args.ema),
                  ema_decay=0.99,
                  train_loss={
                      'type': 'weighted_sum',
                      'losses': [
                          {'type': 'energy_by_sqrt_atoms_sse'},
                          {'type': 'forces_sse'},
                          {'type': 'virials_by_sqrt_atoms_sse'}
                          ],
                      'weights': [
                          1.0,
                          0.01,
                          0.001
                          ]
                      },
                  early_stopping_loss={
                      'type': 'weighted_sum',
                      'losses': [
                          {'type': 'energy_by_sqrt_atoms_sse'},
                          {'type': 'forces_sse'},
                          {'type': 'virials_by_sqrt_atoms_sse'}
                          ],
                      'weights': [
                          1.0,
                          0.01,
                          0.001
                          ]
                      },
                  eval_losses=[
                      {'type': 'energy_per_atom_rmse'},
                      {'type': 'energy_per_atom_mae'},
                      {'type': 'forces_rmse'},
                      {'type': 'forces_mae'},
                      {'type': 'virials_per_atom_rmse'},
                      {'type': 'virials_per_atom_mae'}
                      ]
                  )

    # train model
    train(config)
    
    # evaluate model
    for task in ['4comp', 'CrW', 'TaCr', 'TaV', 'TaW', 'VCr', 'VW', 'noCr', 'noTa', 'noV', 'noW', 'total_md']:
        new_config = config.copy()
        new_config['data_path'] = None
        new_config['test_data_path'] = f'../datasets/HEA/{task}.test_{args.seed}.extxyz'
        errors = eval(new_config)
        save_object(Path(new_config['model_path']) / f'results_{task}.test_{args.seed}.json', errors, use_json=True)
   
    # evaluate model on deformed structures
    # for task in ['CrW', 'TaCr', 'TaV', 'TaW', 'VCr', 'VW']:
    #     new_config = config.copy()
    #     new_config['data_path'] = None
    #     new_config['test_data_path'] = f'../datasets/HEA/deformed_{task}.extxyz'
    #     errors = eval(new_config)
    #     save_object(Path(new_config['model_path']) / f'results_deformed_{task}.json', errors, use_json=True)
        
    # measure inference time
    # new_config = config.copy()
    # new_config['test_data_path'] = f'../datasets/HEA/inference_time/dump_atom.extxyz'
    # inference_time(new_config)
    
    # measure memory consumption
    # new_config = config.copy()
    # new_config['test_data_path'] = f'../datasets/HEA/inference_time/dump_atom.extxyz'
    # memory_consumption(new_config)
