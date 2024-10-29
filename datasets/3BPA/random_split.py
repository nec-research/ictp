from ictp.data.data import AtomicStructures


if __name__ == '__main__':
    atomic_structures = AtomicStructures.from_file('./train_300K.xyz')
    
    # for seed in range(5):
    #     split = atomic_structures.random_split({'train': 450, 'valid': 50}, seed=seed)
    #     split['train'].save_extxyz(f'./train_valid_splits/train_300K_{seed}.extxyz')
    #     split['valid'].save_extxyz(f'./train_valid_splits/valid_300K_{seed}.extxyz')
    
    for seed in range(5):
        split = atomic_structures.random_split({'train': 50, 'valid': 50}, seed=seed)
        split['train'].save_extxyz(f'./train_valid_splits_50_50/train_300K_{seed}.extxyz')
        split['valid'].save_extxyz(f'./train_valid_splits_50_50/valid_300K_{seed}.extxyz')
    