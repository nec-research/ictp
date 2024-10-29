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
    parser.add_argument('--n_product_feats', type=int)
    parser.add_argument('--coupled_product_feats', type=int)
    parser.add_argument('--symmetric_product', type=int)
    parser.add_argument('--ema', type=int)
    parser.add_argument('--task', type=int)
    parser.add_argument('--n_train', type=int)

    # parse parameters
    args = parser.parse_args()
    
    task_list = ['rmd17_aspirin', 'rmd17_azobenzene', 'rmd17_benzene', 'rmd17_ethanol', 'rmd17_malonaldehyde', 'rmd17_naphthalene', 'rmd17_paracetamol', 'rmd17_salicylic', 'rmd17_toluene', 'rmd17_uracil']
    task_name = task_list[args.task]
    
    config = dict(data_path=f'../datasets/rmd17/extxyz/{task_name}.extxyz',
                  data_seed=args.seed,
                  atomic_types=['H', 'C', 'N', 'O'], 
                  atomic_energies=[0.0, 0.0, 0.0, 0.0],
                  r_cutoff=5.0,
                  n_train=args.n_train,
                  n_valid=50,
                  train_batch_size=5,
                  eval_batch_size=10,
                  model_path=f'../results/rmd17/{task_name}_{args.n_train}/float32-hidden_{args.n_hidden_feats}-product_{args.n_product_feats}_coupled_{bool(args.coupled_product_feats)}_symmetric_{bool(args.symmetric_product)}-ema_{bool(args.ema)}-patience_50-fweight_10.0/seed_{args.seed}',
                  model_seed=args.seed,
                  default_dtype='float32',
                  n_hidden_feats=args.n_hidden_feats,
                  n_product_feats=args.n_product_feats,
                  n_interactions=2,
                  l_max_hidden_feats=2,
                  coupled_product_feats=bool(args.coupled_product_feats),
                  symmetric_product=bool(args.symmetric_product),
                  max_epoch=2000,
                  scheduler_patience=50,
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
    eval(config)
