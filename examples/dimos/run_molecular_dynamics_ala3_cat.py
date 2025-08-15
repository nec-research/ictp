import os
import torch
from pathlib import Path
import dimos
from ictp.interfaces.dimos import ICTPSystem, ASETrajectoryWriter, MDLogger


def run_npt(
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
    torch.set_default_device('cuda:0')

    folder = Path('../../models')
    model = f'n_feats_64-batch_size_256-seed_0'

    os.makedirs('results', exist_ok=True)
    os.makedirs('results/' + model, exist_ok=True)
    
    system = ICTPSystem.from_folder_list(
        folder=folder / model / 'best',
        topology_file='inputs/ala3_cat.top',
        structure_file='inputs/ala3_cat.gro',
        new_pair_potential_config={'electrostatics': dict(method='spme', r_cutoff=9.0, alpha=0.3372060287522547, k_max=36, spline_order=5)},
        device='cuda:0',
        periodic=True,
        use_neighborlist=True,
        with_torch_compile=True,
    )
    positions = dimos.read_positions('inputs/ala3_cat.gro')
    
    integrator = dimos.LangevinDynamics(timestep, temperature, friction, system, integrator_dtype)
    barostat = dimos.MCBarostatIsotropic(system.box, target_pressure=pressure, frequency=frequency, rescale_whole_system=rescale_whole_system)
    simulation = dimos.MDSimulation(system, integrator, initial_pos=positions, temperature=temperature, barostat=barostat)
    
    traj_writer = ASETrajectoryWriter('results/' + model + '/ala3_eq.traj', simulation, system, write_velocities=True)
    md_logger = MDLogger('results/' + model + '/ala3_eq.csv', simulation, system, write_interval=write_interval, write_density=True)
    
    for _ in range(total_steps // write_interval):
        simulation.step(write_interval)
        md_logger.log_step()
        traj_writer.append_frame()


if __name__ == '__main__':
    run_npt(
        timestep=0.5,
        temperature=298.15,
        pressure=1.0,
        friction=0.01,
        frequency=100,
        rescale_whole_system=True,
        total_steps=400000, # 200 ps
        write_interval=200,
        integrator_dtype=torch.float64
    )
    
