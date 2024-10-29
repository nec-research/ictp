from ase import units
from ase.io import read
from ase.md import MDLogger
from ase.md.verlet import VelocityVerlet
from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import Stationary
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from ictp.interfaces.ase import ASEWrapper


if __name__ == '__main__':
    # read the inital structure
    atoms = read('../datasets/HEA/inference_time/dump_atom.extxyz')
    
    # setup calculator
    calc = ASEWrapper.from_folder_list(folder='../results/HEA/seed_0/best', 
                                       wrap=True, skin=1.0, device='cuda:0')
    atoms.set_calculator(calc)

    # define temperature in K
    T = 2500
    
    # define initial velocities
    MaxwellBoltzmannDistribution(atoms, T * units.kB)
    Stationary(atoms, preserve_temperature=False)

    f = open("hea-nve.dat", 'a')
    f.write("#! Total energy in meV/atom \n")
    f.close()
    def print_energy(a=atoms):
        # function to print the potential, kinetic, and total energy
        epot = a.get_potential_energy() / len(a) * 10**3
        ekin = a.get_kinetic_energy() / len(a) * 10**3
        f = open("hea-nve.dat", 'a')
        f.write(f"{(epot + ekin).round(5)}\n")
        f.close()

    # nve dynamics
    dyn = VelocityVerlet(atoms, 1.0 * units.fs)
    dyn_traj = Trajectory("hea-nve.traj", 'w', atoms)
    dyn.attach(dyn_traj.write, interval=20)
    dyn.attach(MDLogger(dyn, atoms, "hea-nve.log", header=True, 
                        peratom=False, mode="w"), interval=20)
    dyn.attach(print_energy, interval=20)
    # run dynamics for 1.2 ns
    dyn.run(1200000)
