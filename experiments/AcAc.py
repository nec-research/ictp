from pathlib import Path
from typing import Optional, Dict, Union

import yaml
import argparse

from ictp.data.data import AtomicStructures

from ictp.strategies import TrainingStrategy, EvaluationStrategy

from ictp.model.forward import load_model_from_folder

from ictp.utils.config import update_config
from ictp.utils.misc import set_default_dtype

from ictp.utils.misc import save_object


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


def eval_configs(config: Optional[Union[str, Dict]] = None,
                 file_name: str = 'configs.extxyz'):

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
    errors = evaluate.run_on_configs(model, test_structures=test_structures, folder=config['model_path'], file_name=file_name)
    return errors


if __name__ == '__main__':
    
    # define relevant parameters
    parser = argparse.ArgumentParser(description='Parameters', fromfile_prefix_chars='@')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--n_hidden_feats', type=int)
    parser.add_argument('--n_product_feats', type=int)
    parser.add_argument('--coupled_product_feats', type=int)
    parser.add_argument('--symmetric_product', type=int)
    parser.add_argument('--ema', type=int)

    # parse parameters
    args = parser.parse_args()
    
    config = dict(# train_data_path=f'../datasets/AcAc/train_valid_splits/train_300K_{args.seed}.extxyz',
                  # valid_data_path=f'../datasets/AcAc/train_valid_splits/valid_300K_{args.seed}.extxyz',
                  train_data_path=f'../datasets/AcAc/train_valid_splits_50_50/train_300K_{args.seed}.extxyz',
                  valid_data_path=f'../datasets/AcAc/train_valid_splits_50_50/valid_300K_{args.seed}.extxyz',
                  data_seed=args.seed, 
                  atomic_types=['H', 
                                'C',
                                'O'],
                  atomic_energies=[-13.568422178253735,
                                   -1026.8538996116154,
                                   -2037.796869412825],
                  r_cutoff=5.0,
                  # n_train=450,
                  n_train=50,
                  n_valid=50,
                  train_batch_size=5,
                  eval_batch_size=10,
                  # model_path=f'../results/AcAc/float64-hidden_{args.n_hidden_feats}-product_{args.n_product_feats}_coupled_{bool(args.coupled_product_feats)}_symmetric_{bool(args.symmetric_product)}-ema_{bool(args.ema)}-patience_50-fweight_10.0/seed_{args.seed}',
                  model_path=f'../results/AcAc_50/float64-hidden_{args.n_hidden_feats}-product_{args.n_product_feats}_coupled_{bool(args.coupled_product_feats)}_symmetric_{bool(args.symmetric_product)}-ema_{bool(args.ema)}-patience_50-fweight_10.0/seed_{args.seed}',
                  model_seed=args.seed,
                  default_dtype='float64',
                  n_hidden_feats=args.n_hidden_feats,
                  n_product_feats=args.n_product_feats,
                  n_interactions=2,
                  l_max_hidden_feats=2,
                  coupled_product_feats=bool(args.coupled_product_feats),
                  symmetric_product=bool(args.symmetric_product),
                  max_epoch=2000,
                  ema=bool(args.ema),
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
    
    # evaluate model
    for temp in [300, 600]:
        new_config = config.copy()
        new_config['test_data_path'] = f'../datasets/AcAc/test_MD_{temp}K.xyz'
        errors = eval(new_config)
        save_object(Path(new_config['model_path']) / f'test_results_{temp}K.json', errors, use_json=True)

    # evaluate model on dihedral
    new_config = config.copy()
    new_config['test_data_path'] = f'../datasets/AcAc/test_dihedral.xyz'
    errors = eval(new_config)
    save_object(Path(new_config['model_path']) / f'test_results_dihedral.json', errors, use_json=True)
    
    # evaluate model on dihedral
    new_config = config.copy()
    new_config['test_data_path'] = f'../datasets/AcAc/test_H_transfer.xyz'
    errors = eval(new_config)
    save_object(Path(new_config['model_path']) / f'test_results_H_transfer.json', errors, use_json=True)
    
    # evaluate configs for the dihedral angle
    # new_config = config.copy()
    # new_config['test_data_path'] = f'../datasets/AcAc/test_dihedral.xyz'
    # eval_configs(new_config, file_name='pred_dihedral.extxyz')
    
    # evaluate configs for the H transfer
    # new_config = config.copy()
    # new_config['test_data_path'] = f'../datasets/AcAc/test_H_transfer.xyz'
    # eval_configs(new_config, file_name='pred_H_transfer.extxyz')
