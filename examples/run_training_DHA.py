import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from pathlib import Path
from typing import Optional, Dict, Union

import yaml

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
    
    # Note: choose your seed; we use 0 as an example here
    seed = 0
    
    config = dict(data_path='../datasets/md22/md22_DHA.xyz',
                  device='cuda:0',
                  data_seed=seed,
                  atomic_types=['H', 'C', 'N', 'O'],
                  atomic_energies=[0., 0., 0., 0.],
                  r_cutoff=6.0,
                  n_train=500,
                  n_valid=50,
                  train_batch_size=32,
                  eval_batch_size=50,
                  model_path=f'../results/md22_DHA/seed_{seed}',
                  model_seed=seed,
                  default_dtype='float32', # Note: we use single instead of double precision on this example
                  n_hidden_feats=128,
                  n_product_feats=128,
                  n_interactions=2,
                  l_max_hidden_feats=1,
                  l_max_edge_attrs=3,
                  correlation=3,
                  coupled_product_feats=False,
                  symmetric_product=True,
                  max_epoch=1000,
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
                          10.0 / 23.06 ** 2,
                          1.0 / 23.06 ** 2
                          ]
                      },
                  early_stopping_loss={
                      'type': 'weighted_sum',
                      'losses': [
                          {'type': 'energy_by_sqrt_atoms_sse'},
                          {'type': 'forces_sse'}
                          ],
                      'weights': [
                          10.0 / 23.06 ** 2,
                          1.0/ 23.06 ** 2
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
    
    # evaluate model on the test data
    eval(config)
