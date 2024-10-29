from ase import units
from ase.io import read
from ase.md import MDLogger
from ase.md.langevin import Langevin
from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from ictp.interfaces.ase import ASEWrapper


if __name__ == '__main__':
    # read initial structure
    atoms = read('../datasets/md22/md22_DHA.xyz')

    # setup calculator
    calc = ASEWrapper.from_folder_list(folder='../results/md22_DHA/seed_0/best', 
                                       wrap=False, skin=1.0, device='cuda:1', 
                                       energy_units_to_eV=units.kcal/units.mol)
    atoms.set_calculator(calc)
    
    # define temperature in K
    T = 500

    # define initial velocities
    MaxwellBoltzmannDistribution(atoms, temperature_K=T)
    
    # nvt dynamics with 0.5 fs time step
    # Note: we use ase==3.22.1
    dyn = Langevin(atoms, 0.5 * units.fs, T * units.kB, 0.02)
    dyn.attach(MDLogger(dyn, atoms, "dha-nvt.log", header=True, peratom=False, mode="w"), interval=1)
    traj = Trajectory("dha-nvt.traj", 'w', atoms)
    dyn.attach(traj.write, interval=1)
    # run dynamics for 120 ps
    dyn.run(240000)
    