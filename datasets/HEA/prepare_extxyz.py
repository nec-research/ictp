import re

from typing import Union, List

from ase import Atoms
from ase.io import write
import numpy as np

from ase.stress import voigt_6_to_full_3x3_stress


def convert_types(atom_types):
    dict = {
            0: 'Ta',
            1: 'V',
            2: 'Cr',
            3: 'W'
            }
    out = [dict[key] for key in atom_types]
    return np.array(out)


def parse_cfg(input_files: List[str], output_file: Union[str]):
    
    for input_file in input_files:
        with open(input_file, 'r') as f:
            file_content = f.read()
        
        cfg_blocks = file_content.split('BEGIN_CFG')[1:]
        
        for block in cfg_blocks:
            n_atoms = None
            cell = None
            atom_types = []
            positions = []
            forces = []
            energy = None
            virials = None

            lines = block.strip().split('\n')
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith("Size"):
                    n_atoms = int(lines[i + 1].strip())
                    i += 2
                elif line.startswith("SuperCell") or line.startswith("Supercell"):
                    cell = []
                    for j in range(1, 4):
                        cell.append([float(x) for x in lines[i + j].split()])
                    cell = np.array(cell)
                    i += 4
                elif line.startswith("AtomData"):
                    i += 1  # Skip the header line
                    for _ in range(n_atoms):
                        parts = re.split(r'\s+', lines[i].strip())
                        atom_types.append(int(parts[1]))
                        positions.append([float(parts[2]), float(parts[3]), float(parts[4])])
                        forces.append([float(parts[5]), float(parts[6]), float(parts[7])])
                        i += 1
                    atom_types = np.array(atom_types)
                    positions = np.array(positions)
                    forces = np.array(forces)
                elif line.startswith("Energy"):
                    energy = float(lines[i + 1].strip())
                    i += 2
                elif line.startswith("PlusStress"):
                    virials = [float(x) for x in lines[i + 1].split()]
                    virials = np.array(virials)
                    i += 2
                else:
                    i += 1
            
            symbols = convert_types(atom_types=atom_types)
            stress = -1.0 * voigt_6_to_full_3x3_stress(virials) / abs(np.linalg.det(cell))
            
            atoms = Atoms(positions=positions, symbols=symbols, cell=cell, pbc=True)
            atoms.info['energy'] = energy
            atoms.info['stress'] = stress
            atoms.set_array('forces', np.array(forces))
            
            write(output_file, atoms, format='extxyz', append=True)


if __name__ == "__main__":
    
    # prepare training data set
    for split in range(10):
        print('training data set', split)
        input_files = [f'in_distribution_splits/4comp.cfg_train_{split}', 
                       f'in_distribution_splits/CrW.cfg_train_{split}', 
                       f'in_distribution_splits/TaCr.cfg_train_{split}', 
                       f'in_distribution_splits/TaV.cfg_train_{split}', 
                       f'in_distribution_splits/TaW.cfg_train_{split}', 
                       f'in_distribution_splits/VCr.cfg_train_{split}', 
                       f'in_distribution_splits/VW.cfg_train_{split}', 
                       f'in_distribution_splits/noCr.cfg_train_{split}', 
                       f'in_distribution_splits/noTa.cfg_train_{split}', 
                       f'in_distribution_splits/noV.cfg_train_{split}', 
                       f'in_distribution_splits/noW.cfg_train_{split}', 
                       f'in_distribution_splits/total_md.cfg_train_{split}']
        output_file = f'train_{split}.extxyz'
        parse_cfg(input_files, output_file)
        
    # prepare test data set
    for split in range(10):
        input_files = [f'in_distribution_splits/4comp.cfg_valid_{split}', 
                       f'in_distribution_splits/CrW.cfg_valid_{split}', 
                       f'in_distribution_splits/TaCr.cfg_valid_{split}', 
                       f'in_distribution_splits/TaV.cfg_valid_{split}', 
                       f'in_distribution_splits/TaW.cfg_valid_{split}', 
                       f'in_distribution_splits/VCr.cfg_valid_{split}', 
                       f'in_distribution_splits/VW.cfg_valid_{split}', 
                       f'in_distribution_splits/noCr.cfg_valid_{split}', 
                       f'in_distribution_splits/noTa.cfg_valid_{split}', 
                       f'in_distribution_splits/noV.cfg_valid_{split}', 
                       f'in_distribution_splits/noW.cfg_valid_{split}', 
                       f'in_distribution_splits/total_md.cfg_valid_{split}']
        for input_file in input_files:
            print(input_file)
            output_file = f'{input_file.split(".")[0].split("/")[-1]}.test_{split}.extxyz'
            parse_cfg([input_file], output_file)
    
    # prepare deformed data set
    input_files = [f'in_distribution_splits/deformed/deformed_CrW.cfg', 
                   f'in_distribution_splits/deformed/deformed_TaCr.cfg', 
                   f'in_distribution_splits/deformed/deformed_TaV.cfg', 
                   f'in_distribution_splits/deformed/deformed_TaW.cfg', 
                   f'in_distribution_splits/deformed/deformed_VCr.cfg', 
                   f'in_distribution_splits/deformed/deformed_VW.cfg']
    for input_file in input_files:
        print(input_file)
        output_file = f'{input_file.split(".")[0].split("/")[-1]}.extxyz'
        parse_cfg([input_file], output_file)
