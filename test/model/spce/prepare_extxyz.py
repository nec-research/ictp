from typing import Union

from pathlib import Path

import numpy as np

import ase
from ase import Atoms
from ase.io import write


def convert_to_extxyz(input_file: Union[str, Path], 
                      output_file: Union[str, Path],
                      energy_real_kB: float,
                      energy_recip_kB: float,
                      energy_self_kB: float,) -> None:
    # Note: energy should be provided in K (kB units).
    
    # Define the charges
    charge_H = 0.42380  # [e]
    charge_O = - 2 * charge_H
    
    # Convert energy from kB to eV
    energy_real_eV = energy_real_kB * ase.units.kB
    energy_recip_eV = energy_recip_kB * ase.units.kB
    energy_self_eV = energy_self_kB * ase.units.kB
    
    # Open the input file and read lines
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Extract lattice and atom count from the first two lines
    lattice_line = list(map(float, lines[0].split()))
    atom_count = int(lines[1].strip())
    
    # Initialize the coordinates and symbols
    positions = []
    symbols = []
    
    for line in lines[2:]:
        parts = line.split()
        x, y, z = map(float, parts[1:4])  # coordinates
        symbol = parts[4]  # element symbol
        positions.append([x, y, z])
        symbols.append(symbol)
    
    # Create an ASE Atoms object
    atoms = Atoms(symbols=symbols, positions=positions)

    # Assign charges based on element type
    charges = []
    for symbol in symbols:
        if symbol == 'H':
            charges.append(charge_H)
        elif symbol == 'O':
            charges.append(charge_O)
        else:
            raise ValueError(f"Unexpected element type '{symbol}' in the input.")
    
    # Set charges for the atoms in the ASE object
    atoms.set_initial_charges(charges)

    # Set the cell (lattice) for the ASE object (in Angstroms)
    lattice = np.diag(lattice_line)
    atoms.set_cell(lattice)
    
    atoms.set_pbc(True)
    
    atoms.info['real energy'] = energy_real_eV
    atoms.info['recip energy'] = energy_recip_eV
    atoms.info['self energy'] = energy_self_eV
    
    atoms.center()
    
    write(output_file, atoms, format='extxyz', append=True)
    
if __name__ == '__main__':
    # the data is taken from https://www.nist.gov/mml/csd/chemical-informatics-group/spce-water-reference-calculations-10a-cutoff
    # only reciprocal, self and real energies are stored for the benchmark of our Ewald sum implementation
    input_files = [f'spce_sample_config_periodic{i}.txt' for i in range(1, 5)]
    output_file = 'spce.extxyz'
    real_energies = [-5.58889E+05, -1.19295E+06, -1.96297E+06, -3.57226E+06]
    recip_energies = [6.27009E+03, 6.03495E+03, 5.24461E+03, 7.58785E+03]
    self_energies = [-2.84469E+06, -5.68938E+06, -8.53407E+06, -1.42235E+07]


    for input_file, real_energy, recip_energy, self_energy in zip(input_files, real_energies, recip_energies, self_energies):
        convert_to_extxyz(input_file, output_file, real_energy, recip_energy, self_energy)
