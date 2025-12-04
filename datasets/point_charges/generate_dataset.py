from typing import Union
from pathlib import Path
import numpy as np

from ase import Atoms, units
from ase.io import write

import openmm
from openmm import unit


PARTICLES = {
    'Na': {
        'mass': 22.99 * unit.amu,
        'charge': 1.0 * unit.elementary_charge,
        'sigma': 2.3 * unit.angstroms,
        'epsilon': 0.45 * unit.kilojoules_per_mole,
    },
    'Cl': {
        'mass': 35.45 * unit.amu,
        'charge': -1.0 * unit.elementary_charge,
        'sigma': 4.3 * unit.angstroms,
        'epsilon': 0.42 * unit.kilojoules_per_mole,
    }
}


def ensure_folder(folder: Union[str, Path]) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def create_system(n_atoms: int) -> tuple[openmm.System, list[str]]:
    system = openmm.System()
    symbols = []

    for idx in range(n_atoms):
        element = 'Na' if idx % 2 == 0 else 'Cl'
        symbols.append(element)
        system.addParticle(PARTICLES[element]['mass'])

    return system, symbols


def add_nonbonded_force(system: openmm.System,
                        n_atoms: int,
                        method: int,
                        r_cutoff: float = None,
                        error_tol: float = 5e-5):
    force = openmm.NonbondedForce()
    force.setNonbondedMethod(method)
    force.setUseSwitchingFunction(True)
    force.setSwitchingDistance(0.75 * r_cutoff * unit.angstroms)
    force.setUseDispersionCorrection(False)

    for idx in range(n_atoms):
        element = 'Na' if idx % 2 == 0 else 'Cl'
        p = PARTICLES[element]
        force.addParticle(p['charge'], p['sigma'], p['epsilon'])

    if r_cutoff:
        force.setCutoffDistance(r_cutoff * unit.angstroms)

    if method == openmm.NonbondedForce.Ewald:
        force.setEwaldErrorTolerance(error_tol)

    system.addForce(force)


def generate_periodic(n_atoms: int,
                      box_edge: float,
                      r_cutoff: float,
                      folder: Union[str, Path]):
    folder_train = ensure_folder(folder + '-train')
    folder_test = ensure_folder(folder + '-test')
    
    system, symbols = create_system(n_atoms)

    box_vectors = np.diag([box_edge] * 3) * unit.angstroms
    system.setDefaultPeriodicBoxVectors(*box_vectors)
    
    add_nonbonded_force(system, n_atoms, openmm.NonbondedForce.Ewald, r_cutoff)

    positions = box_edge * np.random.rand(n_atoms, 3) * unit.angstroms

    temperature = 5000.0 * unit.kelvin
    integrator = openmm.LangevinIntegrator(temperature, 100.0 / unit.picosecond, 4.0 * unit.femtoseconds)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName('CUDA'))

    context.setPositions(positions)
    openmm.LocalEnergyMinimizer.minimize(context)
    context.setVelocitiesToTemperature(temperature)

    # equilibrate
    for _ in range(250):
        integrator.step(1000)
        
        state = context.getState(getEnergy=True, getForces=True, getPositions=True, enforcePeriodicBox=True)
        energy = state.getPotentialEnergy() / unit.kilojoules_per_mole * (units.kJ / units.mol)
        print(f'{state.getTime() / unit.picoseconds:8.3f} ps : potential {energy:12.6f} eV')
    
    # generate training dataset
    for _ in range(1000):
        integrator.step(1000)
        state = context.getState(getEnergy=True, getForces=True, getPositions=True, enforcePeriodicBox=True)

        atoms = Atoms(
            symbols=symbols,
            positions=state.getPositions(asNumpy=True) / unit.angstroms,
            cell=box_vectors / unit.angstroms,
            pbc=True
        )
        atoms.info['REF_energy'] = state.getPotentialEnergy() / unit.kilojoules_per_mole * (units.kJ / units.mol)
        atoms.arrays['REF_forces'] = state.getForces(asNumpy=True) * (unit.angstroms / unit.kilojoules_per_mole) * (units.kJ / units.mol)
        atoms.info['REF_total_charge'] = 0.0

        write(folder_train / f'box_edge_{box_edge:.1f}.extxyz', atoms, format='extxyz', append=True)
        print(f'{state.getTime() / unit.picoseconds:8.3f} ps : potential {atoms.info["REF_energy"]:12.6f} eV')
    
    # generate test dataset
    for _ in range(1000):
        integrator.step(1000)
        state = context.getState(getEnergy=True, getForces=True, getPositions=True, enforcePeriodicBox=True)

        atoms = Atoms(
            symbols=symbols,
            positions=state.getPositions(asNumpy=True) / unit.angstroms,
            cell=box_vectors / unit.angstroms,
            pbc=True
        )
        atoms.info['REF_energy'] = state.getPotentialEnergy() / unit.kilojoules_per_mole * (units.kJ / units.mol)
        atoms.arrays['REF_forces'] = state.getForces(asNumpy=True) * (unit.angstroms / unit.kilojoules_per_mole) * (units.kJ / units.mol)
        atoms.info['REF_total_charge'] = 0.0

        write(folder_test / f'box_edge_{box_edge:.1f}.extxyz', atoms, format='extxyz', append=True)
        print(f'{state.getTime() / unit.picoseconds:8.3f} ps : potential {atoms.info["REF_energy"]:12.6f} eV')
        
        
if __name__ == '__main__':
    np.random.seed(1234)
    generate_periodic(n_atoms=512, box_edge=40.0, r_cutoff=9.0, folder='periodic-512_atoms')
