from ictp.data.data import AtomicStructures


if __name__ == '__main__':
    
    atomic_structures = AtomicStructures.from_file(f'train_0.extxyz')
    for task in ['4comp', 'CrW', 'TaCr', 'TaV', 'TaW', 'VCr', 'VW', 'noCr', 'noTa', 'noV', 'noW', 'total_md']:
        atomic_structures = atomic_structures + AtomicStructures.from_file(f'{task}.test_0.extxyz')
    
    print(len(atomic_structures.structures))
    
    for seed in range(10):
        assert len(atomic_structures.structures) == 6711
        split = atomic_structures.random_split({'train': 6211, 'valid': 500}, seed=seed)
        split['train'].save_extxyz(f'total.train_cfgs_{seed}.extxyz')
        split['valid'].save_extxyz(f'total.valid_cfgs_{seed}.extxyz')
    