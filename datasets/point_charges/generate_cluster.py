from typing import Union
from pathlib import Path
import numpy as np

from ase import Atoms, units
from ase.io import read, write
from ase.geometry import get_distances

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


def get_energy_forces(atoms: Atoms, r_cutoff: float = None):
    # Coulomb energy
    system = openmm.System()
    
    force = openmm.NonbondedForce()
    force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    force.setUseSwitchingFunction(False)
    force.setUseDispersionCorrection(False)

    symbols = atoms.get_chemical_symbols()
    for s in symbols:
        system.addParticle(PARTICLES[s]['mass'])
        force.addParticle(PARTICLES[s]['charge'], PARTICLES[s]['sigma'], 0.0 * unit.kilojoule_per_mole)
    
    system.addForce(force)
    
    pos = atoms.get_positions() * unit.angstroms

    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("CPU"))
    context.setPositions(pos)

    state = context.getState(getEnergy=True, getForces=True)
    
    energy = state.getPotentialEnergy() / unit.kilojoules_per_mole * (units.kJ / units.mol)
    forces = state.getForces(asNumpy=True) * (unit.angstroms / unit.kilojoules_per_mole) * (units.kJ / units.mol)

    # LJ energy
    system = openmm.System()
    
    force = openmm.NonbondedForce()
    force.setNonbondedMethod(openmm.NonbondedForce.CutoffNonPeriodic)
    force.setCutoffDistance(r_cutoff * unit.angstroms)
    force.setUseSwitchingFunction(True)
    force.setSwitchingDistance(0.75 * r_cutoff * unit.angstroms)
    force.setUseDispersionCorrection(False)
    
    symbols = atoms.get_chemical_symbols()
    for s in symbols:
        system.addParticle(PARTICLES[s]['mass'])
        force.addParticle(0.0 * unit.elementary_charge, PARTICLES[s]['sigma'], PARTICLES[s]['epsilon'])
    
    system.addForce(force)
    
    pos = atoms.get_positions() * unit.angstroms

    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("CPU"))
    context.setPositions(pos)

    state = context.getState(getEnergy=True, getForces=True)
    
    energy = energy + state.getPotentialEnergy() / unit.kilojoules_per_mole * (units.kJ / units.mol)
    forces = forces + state.getForces(asNumpy=True) * (unit.angstroms / unit.kilojoules_per_mole) * (units.kJ / units.mol)
    
    return energy, forces


def get_clusters(
    traj_file: Union[str, Path],
    out_file: Union[str, Path],
    n_atoms: int,
    seed: int = 0,
):
    traj = read(traj_file, ':')
    rng = np.random.RandomState(seed=seed)
    
    fmax = 3.0
    print(traj_file)
    
    for i_frame, atoms in enumerate(traj):
        print(i_frame)
        while True:
            idx = rng.permutation(np.arange(len(atoms)))[0]
            
            dr, dists = get_distances(
                atoms.positions[idx],
                atoms.positions,
                cell=atoms.cell,
                pbc=atoms.pbc
            )
            dr, dists = dr[0], dists[0]
            
            order = np.argsort(dists)
            selection = order[:n_atoms]
            
            r0 = atoms.positions[idx]
            pos_selection = r0 + dr[selection]
        
            cluster = atoms[selection].copy()
            cluster.set_positions(pos_selection)
            cluster.set_pbc(False)
            cluster.set_cell(None)
            
            symbols = cluster.get_chemical_symbols()
            n_Na = symbols.count('Na')
            n_Cl = symbols.count('Cl')
            total_charge = n_Na - n_Cl
            
            if total_charge == 0:
                cluster.info['REF_total_charge'] = float(total_charge)
                
                energy, forces = get_energy_forces(cluster, r_cutoff=9.0)
                cluster.info['REF_energy'] = energy
                cluster.arrays['REF_forces'] = forces
                
                write(out_file, cluster, format='extxyz', append=True)
                break
            
            else:
                if total_charge % 2 != 0:
                    continue
                symbols = np.asarray(cluster.get_chemical_symbols())
                
                flips_needed = abs(total_charge) // 2
                center_in_cluster = 0
                
                flip_from = "Na" if total_charge > 0 else "Cl"
                flip_to   = "Cl" if total_charge > 0 else "Na"
                
                idx_all = np.arange(symbols.size)
                candidates = idx_all[(symbols == flip_from) & (idx_all != center_in_cluster)]
                
                if candidates.size == 0:
                    continue
                
                to_flip = rng.choice(candidates, size=flips_needed, replace=False)

                if to_flip.size < flips_needed:
                    continue
                
                symbols[to_flip] = flip_to
                cluster.set_chemical_symbols(symbols.tolist())
                
                n_Na = np.count_nonzero(symbols == "Na")
                n_Cl = np.count_nonzero(symbols == "Cl")
                total_charge = n_Na - n_Cl
                
                if total_charge != 0:
                    continue
                
                energy, forces = get_energy_forces(cluster, r_cutoff=9.0)
                
                if float(np.abs(forces[to_flip]).max()) > fmax:
                    continue
                
                if float(np.abs(cluster.arrays['REF_forces']).max()) > 10.0 or float(np.abs(forces).max()) > 10.0:
                    print(float(np.abs(cluster.arrays['REF_forces']).max()), float(np.abs(forces).max()), float(np.abs(forces[to_flip]).max()))
                
                cluster.info['REF_total_charge'] = float(total_charge)
                cluster.info['REF_energy'] = energy
                cluster.arrays['REF_forces'] = forces
                
                write(out_file, cluster, format='extxyz', append=True)
                break                


if __name__ == '__main__':
    folder = Path('periodic-512_atoms-train')
    traj_file = folder / f'box_edge_40.0.extxyz'
    for n_atoms in [90, 180, 270]:
        out_file = folder / f'box_edge_40.0_cluster_{n_atoms}.extxyz'
        get_clusters(traj_file, out_file, n_atoms, seed=1234)
            
    folder = Path('periodic-512_atoms-test')
    traj_file = folder / f'box_edge_40.0.extxyz'
    for n_atoms in [90, 180, 270]:
        out_file = folder / f'box_edge_40.0_cluster_{n_atoms}.extxyz'
        get_clusters(traj_file, out_file, n_atoms, seed=1234)