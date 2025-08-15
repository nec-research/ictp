import os
import torch
import types
import dimos
from ictp.interfaces.dimos import ASETrajectoryWriter, MDLogger


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

    model = 'amber14sb+tip3p'
    
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/' + model, exist_ok=True)
    
    gromacs_system = dimos.ff.GromacsForceField(
        parameter_file='inputs/trp.top',
        xyz_file='inputs/trp.gro',
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
    
    positions = dimos.read_positions('inputs/trp.gro')
    
    integrator = dimos.LangevinDynamics(timestep, temperature, friction, gromacs_system, integrator_dtype)
    barostat = dimos.MCBarostatIsotropic(gromacs_system.box, target_pressure=pressure, frequency=frequency, rescale_whole_system=rescale_whole_system)
    simulation = dimos.MDSimulation(gromacs_system, integrator, initial_pos=positions, temperature=temperature, barostat=barostat)
    
    traj_writer = ASETrajectoryWriter('results/' + model + '/trp_eq.traj', simulation, gromacs_system)
    md_logger = MDLogger('results/' + model + '/trp_eq.csv', simulation, gromacs_system, write_interval=write_interval, write_density=True)
    
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
