from ictp.data.data import AtomicStructures


if __name__ == '__main__':
    
    for seed in range(10):
        atomic_structures = AtomicStructures.from_file(f'train_{seed}.extxyz')
        # we run for a single split of the training data set because we have other 10 splits between test and train
        split = atomic_structures.random_split({'train': 4873, 'valid': 500}, seed=1234)
        split['train'].save_extxyz(f'train_cfgs_{seed}.extxyz')
        split['valid'].save_extxyz(f'valid_cfgs_{seed}.extxyz')
    