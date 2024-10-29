import sys

# set python paths
sys.path.append('/mnt/local/vzaverkin/projects/ictp')

import numpy as np

import ase

from ictp.data.data import AtomicStructure, AtomicStructures


def main(file_path):
    key_mapping = {'coords': 'positions',
                   'nuclear_charges': 'numbers',
                   'energies': 'energy',
                   'forces': 'forces'}

    atomic_dict = {}
    with np.load(file_path) as data:
        for key, value in key_mapping.items():
            atomic_dict[value] = data.get(key, None)

    structures = []
    for i_mol in range(len(atomic_dict['energy'])):
        positions = atomic_dict['positions'][i_mol]
        numbers = atomic_dict['numbers']
        energy = float(atomic_dict['energy'][i_mol])
        forces = atomic_dict['forces'][i_mol]

        atoms = ase.Atoms(numbers=numbers, positions=positions, pbc=False)
        atoms.arrays.update({'forces': forces})
        atoms.info.update({'energy': energy})

        structures.append(AtomicStructure.from_atoms(atoms, neighbors='matscipy'))

    return AtomicStructures(structures)


if __name__ == '__main__':
    
    for f in ['rmd17_aspirin.npz', 'rmd17_azobenzene.npz', 'rmd17_benzene.npz', 'rmd17_ethanol.npz', 'rmd17_malonaldehyde.npz', 'rmd17_naphthalene.npz', 'rmd17_paracetamol.npz', 'rmd17_salicylic.npz', 'rmd17_toluene.npz', 'rmd17_uracil.npz']:
        structures = main(f)
        structures.save_extxyz('extxyz' + '/' + f.split('.')[0] + '.extxyz')
