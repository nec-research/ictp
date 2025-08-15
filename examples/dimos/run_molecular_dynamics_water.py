import os
import argparse
from pathlib import Path
import torch
import dimos
from ictp.interfaces.dimos import ICTPSystem, ASETrajectoryWriter, MDLogger


def run_npt(
    n_feats: int = 64,
    timestep: float = 0.5,
    temperature: float = 273.15,
    pressure: float = 1.0,
    friction: float = 0.01,
    frequency: int = 100,
    rescale_whole_system: bool = True,
    total_steps: int = 2400000,
    write_interval: int = 200,
    integrator_dtype = torch.float64
) -> None:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cuda:0")

    folder = Path('../../models')
    model = f'n_feats_{n_feats}-batch_size_256-seed_0'
    
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/' + model, exist_ok=True)
    os.makedirs('results/' + model + f'/{temperature}', exist_ok=True)
    
    system = ICTPSystem.from_folder_list(
        folder=folder / model / 'best',
        topology_file='inputs/water.top',
        structure_file='inputs/water.gro',
        new_pair_potential_config={'electrostatics': dict(method='spme', r_cutoff=9.0, alpha=0.3372060287522547, k_max=30, spline_order=5)},
        device='cuda:0',
        periodic=True,
        use_neighborlist=True,
        with_torch_compile=True,
    )
    positions = dimos.read_positions('inputs/water.gro')
    
    integrator = dimos.LangevinDynamics(timestep, temperature, friction, system, integrator_dtype)
    barostat = dimos.MCBarostatIsotropic(system.box, target_pressure=pressure, frequency=frequency, rescale_whole_system=rescale_whole_system)
    simulation = dimos.MDSimulation(system, integrator, initial_pos=positions, temperature=temperature, barostat=barostat)
    
    traj_writer = ASETrajectoryWriter('results/' + model + f"/{temperature}/water.traj", simulation, system, write_velocities=True)
    md_logger = MDLogger('results/' + model + f"/{temperature}/water.csv", simulation, system, write_interval=write_interval, write_density=True)
    
    for i in range(total_steps // write_interval):
        simulation.step(write_interval)
        md_logger.log_step()
        traj_writer.append_frame()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Parameters', fromfile_prefix_chars='@')
    parser.add_argument('--n_feats', type=int)              # 64, 128, 256
    parser.add_argument('--temperature', type=float)        # 273.15, ..., 373.15 K
    
    args = parser.parse_args()
    
    run_npt(
        n_feats=args.n_feats,
        timestep=0.5,
        temperature=args.temperature,
        pressure=1.0,   # bar
        friction=0.01,  # 1/fs
        frequency=100,
        rescale_whole_system=True,
        total_steps=2400000,    # 1.2 ns
        write_interval=200,
        integrator_dtype=torch.float64
    )
