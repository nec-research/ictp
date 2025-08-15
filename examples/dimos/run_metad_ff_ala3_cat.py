import os
from typing import *
import numpy as np
import torch
import multiprocessing
import types
import dimos
from ase.io import read
from ictp.interfaces.dimos import BiasCalculator, PLUMEDMetadynamics, ASETrajectoryWriter, MDLogger


class BiasedGromacsForceField(dimos.ff.GromacsForceField):
    def __init__(self, *args, bias_calc=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias_calc = bias_calc

    def add_bias(self, bias_calc: BiasCalculator) -> None:
        self.bias_calc = bias_calc
    
    def calc_energy(
        self,
        pos: torch.Tensor,
        neighborlist,
        return_forces: bool = False,
        print_energies=False
    ):
        if not return_forces:
            energy = super().calc_energy(pos, neighborlist, return_forces, print_energies)

            if self.bias_calc is not None:
                bias_out = self.bias_calc(pos, self.masses, energy, self.box)
                
                energy += bias_out['energy'].to(torch.float64).item()
            return energy
        else:
            energy, forces, acceleration = super().calc_energy(pos, neighborlist, return_forces, print_energies)
            if self.bias_calc is not None:
                bias_out = self.bias_calc(pos, self.masses, energy, self.box)
                
                energy += bias_out['energy'].to(torch.float64).item()
                forces += bias_out['forces'].to(torch.float64)
                acceleration += bias_out["forces"] / self.masses.unsqueeze(-1)
            return energy, forces, acceleration


def run_metad(
    cuda_id: int,
    walker_id: int,
    walkers_n: int,
    timestep: float = 0.5,
    temperature: float = 298.15,
    pressure: float = 1.0,
    friction: float = 0.01,
    frequency: int = 100,
    rescale_whole_system: bool = True,
    total_steps: int = 2400000,
    write_interval: int = 200,
    integrator_dtype = torch.float64
):
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.set_default_dtype(torch.float32)
    torch.set_default_device(f'cuda:{cuda_id}')

    model = 'amber14sb+tip3p'
    
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/' + model, exist_ok=True)
    os.makedirs('results/' + model + '/metad-default_grid-frequency_100-height_02-biasfactor_6-pace_1000', exist_ok=True)
    os.makedirs('results/' + model + '/metad-default_grid-frequency_100-height_02-biasfactor_6-pace_1000' + '/WALKERS', exist_ok=True)
    
    gromacs_system = BiasedGromacsForceField(
        parameter_file='inputs/ala3_cat.top',
        xyz_file='inputs/ala3_cat.gro',
        cutoff=9.0,
        switch_distance=7.5,
        dispersion_correction=True,
        nonbonded_type='PME',
        unit_system='amber',
        periodic=True
    )
    
    # some stuff for compatibility with my trajectory writer etc.
    dimos.constants.init_constants_in_unit_system('amber')
    gromacs_system.energy_units_to_eV = 1.0 / dimos.constants.eV_to_internal
    gromacs_system.length_units_to_A = dimos.constants.internal_to_Angstrom
    gromacs_system.time_units_to_fs = dimos.constants.FS_TO_INTERNAL
    gromacs_system.atomic_numbers = torch.tensor(
            [atom.atomic_number for atom in gromacs_system.parameter_set_parmed.atoms], dtype=torch.long
        )
    
    def measure_density(self):
        volume = torch.prod(self.box)
        total_mass = torch.sum(self.masses)
        density = total_mass / (6.022140857e23 * volume * 10**(-30)) * 10**(-6)
        return density.detach().cpu().item()
    
    gromacs_system.measure_density = types.MethodType(measure_density, gromacs_system)
    
    ps = 1000.0 / dimos.constants.FS_TO_INTERNAL
    
    plumed_input = [
        f"UNITS LENGTH=A TIME={1.0 / ps} ENERGY=kcal/mol",
        "WHOLEMOLECULES ENTITY0=1-35",
        "phi:   TORSION ATOMS=11,13,15,21",
        "psi:   TORSION ATOMS=13,15,21,23",
        " ".join(["metad: METAD",
                  f"ARG=phi,psi SIGMA=0.35,0.35 HEIGHT=0.2 PACE=1000 BIASFACTOR=6 TEMP={temperature}",
                  "FILE=HILLS GRID_MIN=-pi,-pi GRID_MAX=pi,pi",
                  f"WALKERS_N={walkers_n} WALKERS_ID={walker_id} WALKERS_DIR=results/{model}/metad-default_grid-frequency_100-height_02-biasfactor_6-pace_1000/WALKERS WALKERS_RSTRIDE=250"]),
        f"PRINT ARG=phi,psi,metad.bias STRIDE=500 FILE=results/{model}/metad-default_grid-frequency_100-height_02-biasfactor_6-pace_1000/WALKERS/COLVAR.{walker_id}",
        "FLUSH STRIDE=500",
    ]
    
    atoms = read(f'results/{model}/ala3_eq.traj', '-1')
    positions = torch.tensor(atoms.get_positions(), dtype=torch.get_default_dtype(), device=torch.get_default_device())
    box = torch.tensor(np.diag(atoms.get_cell()), dtype=torch.get_default_dtype(), device=torch.get_default_device())
    gromacs_system.update_box(box)
    
    integrator = dimos.LangevinDynamics(timestep, temperature, friction, gromacs_system, integrator_dtype)
    barostat = dimos.MCBarostatIsotropic(gromacs_system.box, target_pressure=pressure, frequency=frequency, rescale_whole_system=rescale_whole_system)
    simulation = dimos.MDSimulation(gromacs_system, integrator, initial_pos=positions, temperature=temperature, barostat=barostat, seed=walker_id)
    biased_simulation = PLUMEDMetadynamics(simulation, plumed_input=plumed_input, temperature=temperature, timestep=timestep, log=f'results/{model}/metad-default_grid-frequency_100-height_02-biasfactor_6-pace_1000/ala3_{walker_id}.log')
    
    traj_writer = ASETrajectoryWriter(f'results/{model}/metad-default_grid-frequency_100-height_02-biasfactor_6-pace_1000/ala3_{walker_id}.traj', simulation, gromacs_system, write_velocities=True)
    md_logger = MDLogger(f'results/{model}/metad-default_grid-frequency_100-height_02-biasfactor_6-pace_1000/ala3_{walker_id}.csv', simulation, gromacs_system, write_interval=write_interval, write_density=True)
    
    for _ in range(total_steps // write_interval):
        biased_simulation.step(write_interval)
        md_logger.log_step()
        traj_writer.append_frame()


if __name__ == '__main__':
    
    n_gpus = 6
    n_walkers = 6
    
    # 10 ns long simulations for each walker = 60 ns in total
    args = [
        ((i % n_gpus + 2) % n_gpus, i, n_walkers, 0.5, 298.15, 1.0, 0.01, 100, True, 20000000, 1000, torch.float64)
        for i in range(n_walkers)
    ]
    
    with multiprocessing.Pool(processes=n_walkers) as pool:
        pool.starmap(run_metad, args)
    
