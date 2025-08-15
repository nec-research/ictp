import os
import numpy as np
import torch
import multiprocessing
from pathlib import Path
import dimos
from ase.io import read
from ictp.interfaces.dimos import (ICTPSystem, PLUMEDMetadynamics, 
                                   ASETrajectoryWriter, MDLogger)


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

    folder = Path('../../models')
    model = f'n_feats_64-batch_size_256-seed_0'
    
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/' + model, exist_ok=True)
    os.makedirs('results/' + model + '/metad-default_grid-frequency_100-height_02-biasfactor_6-pace_1000', exist_ok=True)
    os.makedirs('results/' + model + '/metad-default_grid-frequency_100-height_02-biasfactor_6-pace_1000' + '/WALKERS', exist_ok=True)
    
    system = ICTPSystem.from_folder_list(
        folder=folder / model / 'best',
        topology_file='inputs/ala3.top',
        structure_file='inputs/ala3.gro',
        new_pair_potential_config={'electrostatics': dict(method='spme', r_cutoff=9.0, alpha=0.3372060287522547, k_max=36, spline_order=5)},
        device=f'cuda:{cuda_id}',
        periodic=True,
        use_neighborlist=True,
        with_torch_compile=True,
    )
    
    dimos.constants.init_constants_in_unit_system(system.unit_system)
    ps = 1000.0 / dimos.constants.FS_TO_INTERNAL
    
    plumed_input = [
        f"UNITS LENGTH=A TIME={1.0 / ps} ENERGY=kcal/mol",
        "WHOLEMOLECULES ENTITY0=1-42",
        "phi:   TORSION ATOMS=15,17,19,25",
        "psi:   TORSION ATOMS=17,19,25,27",
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
    system.update_box(box)
    
    integrator = dimos.LangevinDynamics(timestep, temperature, friction, system, integrator_dtype)
    barostat = dimos.MCBarostatIsotropic(system.box, target_pressure=pressure, frequency=frequency, rescale_whole_system=rescale_whole_system)
    simulation = dimos.MDSimulation(system, integrator, initial_pos=positions, temperature=temperature, barostat=barostat, seed=walker_id)
    biased_simulation = PLUMEDMetadynamics(simulation, plumed_input=plumed_input, temperature=temperature, timestep=timestep, log=f'results/{model}/metad-default_grid-frequency_100-height_02-biasfactor_6-pace_1000/ala3_{walker_id}.log')
    
    traj_writer = ASETrajectoryWriter(f'results/{model}/metad-default_grid-frequency_100-height_02-biasfactor_6-pace_1000/ala3_{walker_id}.traj', simulation, system, write_velocities=True)
    md_logger = MDLogger(f'results/{model}/metad-default_grid-frequency_100-height_02-biasfactor_6-pace_1000/ala3_{walker_id}.csv', simulation, system, write_interval=write_interval, write_density=True)
    
    for _ in range(total_steps // write_interval):
        biased_simulation.step(write_interval)
        md_logger.log_step()
        traj_writer.append_frame()


if __name__ == '__main__':
    
    n_gpus = 6
    n_walkers = 6
    
    # 10 ns long simulations for each walker = 60 ns in total
    args = [
        (i % n_gpus, i, n_walkers, 0.5, 298.15, 1.0, 0.01, 100, True, 20000000, 1000, torch.float64)
        for i in range(n_walkers)
    ]
    
    with multiprocessing.Pool(processes=n_walkers) as pool:
        pool.starmap(run_metad, args)
    