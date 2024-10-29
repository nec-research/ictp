import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from pathlib import Path
from typing import Optional, Dict, Union

import yaml
import argparse

from ictp.data.data import AtomicStructures

from ictp.strategies import TrainingStrategy, EvaluationStrategy

from ictp.model.forward import load_model_from_folder

from ictp.utils.config import update_config
from ictp.utils.misc import set_default_dtype


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
    _ = evaluate.run(model, test_structures=test_structures, folder=config['model_path'])


if __name__ == '__main__':
    
    # define relevant parameters
    parser = argparse.ArgumentParser(description='Parameters', fromfile_prefix_chars='@')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--n_hidden_feats', type=int)
    parser.add_argument('--scheduler_patience', type=int)
    parser.add_argument('--r_cutoff', type=float)
    parser.add_argument('--task', type=int)
    parser.add_argument('--n_train', type=int)
    parser.add_argument('--n_valid', type=int)
    parser.add_argument('--energy_weight', type=float)
    parser.add_argument('--force_weight', type=float)
    
    # parse parameters
    args = parser.parse_args()
    
    task_list = ['md22_Ac-Ala3-NHMe', 'md22_AT-AT-CG-CG', 'md22_AT-AT', 'md22_buckyball-catcher', 'md22_DHA', 'md22_double-walled_nanotube', 'md22_stachyose']
    atomic_types_list = [['H', 'C', 'N', 'O'],
                         ['H', 'C', 'N', 'O'],
                         ['H', 'C', 'N', 'O'],
                         ['H', 'C'],
                         ['H', 'C', 'N', 'O'],
                         ['H', 'C'],
                         ['H', 'C', 'N', 'O']]
    task_name = task_list[args.task]
    atomic_types = atomic_types_list[args.task]
    atomic_energies = [0. for _ in range(len(atomic_types))]
    
    config = dict(data_path=f'../datasets/md22/{task_name}.xyz',
                  data_seed=args.seed,
                  atomic_types=atomic_types,
                  atomic_energies=atomic_energies,
                  r_cutoff=args.r_cutoff,
                  n_train=args.n_train,
                  n_valid=args.n_valid,
                  train_batch_size=2,
                  eval_batch_size=2,
                  model_path=f'../results/md22/{task_name}_{args.n_train}/float64-hidden_{args.n_hidden_feats}-patience_{args.scheduler_patience}-r_cutoff_{args.r_cutoff}-energy_weight_{args.energy_weight}-force_weight_{args.force_weight}/seed_{args.seed}',
                  model_seed=args.seed,
                  default_dtype='float64',
                  n_hidden_feats=args.n_hidden_feats,
                  n_product_feats=args.n_hidden_feats,
                  n_interactions=2,
                  l_max_hidden_feats=2,
                  coupled_product_feats=False,
                  symmetric_product=True,
                  max_epoch=1000,
                  save_epoch=10,
                  scheduler_patience=args.scheduler_patience,
                  ema=True,
                  ema_decay=0.99,
                  train_loss={
                      'type': 'weighted_sum',
                      'losses': [
                          {'type': 'energy_by_sqrt_atoms_sse'},
                          {'type': 'forces_sse'}
                          ],
                      'weights': [
                          args.energy_weight / 23.06 ** 2,
                          args.force_weight / 23.06 ** 2
                          ]
                      },
                  early_stopping_loss={
                      'type': 'weighted_sum',
                      'losses': [
                          {'type': 'energy_by_sqrt_atoms_sse'},
                          {'type': 'forces_sse'}
                          ],
                      'weights': [
                          args.energy_weight / 23.06 ** 2,
                          args.force_weight / 23.06 ** 2
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
    eval(config)
