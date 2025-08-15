import os
from pathlib import Path

from typing import *

import torch
import torch.nn as nn

import math
import numpy as np

import copy

import time

from ase.io import read

import dimos
from ictp.model.forward import ForwardAtomisticNetwork, find_last_ckpt
from ictp.model.pair_potentials import (CoulombElectrostaticEnergy, 
                                        EwaldElectrostaticEnergy, 
                                        SPMEElectrostaticEnergy)
from ictp.model.calculators import StructurePropertyCalculator
from ictp.data.data import (AtomicTypeConverter, 
                            AtomicStructure, 
                            AtomicData)
from ictp.interfaces.dimos import ICTPSystem
from ictp.utils.torch_geometric import DataLoader
from ictp.utils.misc import find_max_r_cutoff


class EwaldParameters:
    def __init__(
        self,
        r_cutoff: float,
        edge_length: float,
        tolerance: float = 5e-5,
    ):
        self.r_cutoff = r_cutoff
        self.edge_length = edge_length
        self.tolerance = tolerance
        self.alpha = math.sqrt(-math.log(2.0 * self.tolerance)) / self.r_cutoff
    
    def ewald_error(
        self,
        k_max: int,
        box_length: float
    ) -> float:
        temp = k_max * math.pi / (box_length * self.alpha)
        return self.tolerance - 0.05 * math.sqrt(box_length * self.alpha) * k_max * math.exp(-temp ** 2)

    def specialized_ewald_error(self, x):
        return self.ewald_error(x, self.edge_length)
    
    def find_zero(
        self,
        function,
        initial_guess: int = 10
    ) -> int:
        current_arg = initial_guess
        value = function(current_arg)
        if value > 0.0:
            while value > 0 and current_arg > 0:
                current_arg -= 1
                value = function(current_arg)
            return current_arg + 1

        while value < 0.0:
            current_arg += 1
            value = function(current_arg)
        return current_arg
    
    def compute_k_max_ewald(self):
        return self.find_zero(self.specialized_ewald_error)
    
    def compute_k_max_spme(self):
        return math.ceil(2 * self.alpha * self.edge_length / (3 * math.pow(self.tolerance, 0.2))) / 2 - 1


def get_spme_params(input: str):
    atoms = read(input)
    ewald_params = EwaldParameters(r_cutoff=9.0, edge_length=np.max(atoms.get_cell()), tolerance=5e-5)
    return ewald_params.alpha, int(ewald_params.compute_k_max_spme())


def ictp_inference(
    n_feats: int = 64,
    use_estat: bool = True,
    n_waters: int = 256
):
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cuda:0")

    folder = Path('../../models')
    model_name = f'n_feats_{n_feats}-batch_size_256-' + ('seed_0' if use_estat else 'no_estat-no_disp-seed_0')
    
    os.makedirs('results', exist_ok=True)
    os.makedirs(f'results/{model_name}', exist_ok=True)
    
    if use_estat:
        alpha, k_max = get_spme_params(f'inputs/water_{n_waters}_npt.gro')
        print("n_atoms:", n_waters)
        print("alpha:", alpha)
        print("k_max for SPME:", k_max)
        new_pair_potential_config = {
            'electrostatics': dict(method='spme', r_cutoff=9.0, alpha=alpha, k_max=k_max, spline_order=5)
        }
    
    else:
        new_pair_potential_config = None
        
    model = ForwardAtomisticNetwork.from_folder(find_last_ckpt(folder / model_name / 'best'))
    config = copy.deepcopy(model.config)
    
    if new_pair_potential_config is not None:
        for pair_potential, params in new_pair_potential_config.items():
            if pair_potential not in config:
                raise ValueError(
                    f"Unknown pair potential type: {pair_potential=}"
                )
            
            if pair_potential != "electrostatics":
                raise ValueError(
                    f"Unsupported pair potential type: {pair_potential=}. "
                    f"This method only supports exchanging 'electrostatics' modules. "
                    f"Trainable modules like 'repulsion' or 'dispersion' must be handled explicitly."
                )
            config[pair_potential].update(params)
    
        new_pair_potential: Optional[nn.Module] = None
        
        if config['electrostatics']['method'] is not None:
            if config['electrostatics']['method'] == 'coulomb':
                new_pair_potential = CoulombElectrostaticEnergy(
                    r_cutoff=config['electrostatics']['r_cutoff'],
                    ke=config['ke'],
                    exclusion_radius=config['exclusion_radius'],
                    n_exclusion_polynomial_cutoff=config['n_exclusion_polynomial_cutoff']
                )
            elif config['electrostatics']['method'] == 'ewald':
                new_pair_potential = EwaldElectrostaticEnergy(
                    r_cutoff=config['electrostatics']['r_cutoff'],
                    ke=config['ke'],
                    alpha=config['electrostatics']['alpha'],
                    k_max=config['electrostatics']['k_max'],
                    exclusion_radius=config['exclusion_radius'],
                    n_exclusion_polynomial_cutoff=config['n_exclusion_polynomial_cutoff']
                )
            elif config['electrostatics']['method'] == 'spme':
                new_pair_potential = SPMEElectrostaticEnergy(
                    r_cutoff=config['electrostatics']['r_cutoff'],
                    ke=config['ke'],
                    alpha=config['electrostatics']['alpha'],
                    k_max=config['electrostatics']['k_max'],
                    spline_order=config['electrostatics']['spline_order'],
                    exclusion_radius=config['exclusion_radius'],
                    n_exclusion_polynomial_cutoff=config['n_exclusion_polynomial_cutoff']
                )
            else:
                valid_values = ['coulomb', 'ewald', 'spme']
                raise ValueError(
                    f"Invalid value for {config['electrostatics']['method']=}. "
                    f"Expected one of {valid_values}."
                )
        
        model = model.replace_pair_potential(new_pair_potential)
    
    # build torch calculator
    calc = StructurePropertyCalculator(
        model=model,
        with_torch_script=False,
        with_torch_compile=True
    )
    
    # build also the system to get atoms...
    system = ICTPSystem.from_folder_list(
        folder=folder / model_name / 'best',
        topology_file=f'inputs/water_{n_waters}.top',
        structure_file=f'inputs/water_{n_waters}_npt.gro',
        new_pair_potential_config=new_pair_potential_config,
        device='cuda:0',
        periodic=True,
        use_neighborlist=True,
        with_torch_compile=True,
    )
    positions = dimos.read_positions(f'inputs/water_{n_waters}_npt.gro')
    
    atomic_type_converter = AtomicTypeConverter.from_type_list(config['atomic_types'])
    
    structure = AtomicStructure(
        species=system.atomic_numbers.detach().cpu().numpy(),
        atomic_numbers=system.atomic_numbers.detach().cpu().numpy(),
        positions=positions.detach().cpu().numpy() * system.length_units_to_A,
        cell=np.diag(system.box.detach().cpu().numpy()),
        pbc=True,
        total_charge=0.0,
        neighbors='matscipy'
    )
    
    del system
    
    structure = structure.to_type_names(atomic_type_converter, check=True)
    
    dl = DataLoader(
        dataset=[
            AtomicData(
                structure=structure,
                r_cutoff=find_max_r_cutoff(config),
                skin=0.0, 
                n_species=atomic_type_converter.get_n_type_names()
            )
        ],
        batch_size=1,
        shuffle=False,
        drop_last=False
    )
    batch = next(iter(dl)).to('cuda:0')
    
    for _ in range(10):
        calc(batch, forces=True)
    
    individual_measurements = []
    
    start_time = time.time()
    
    torch.cuda.synchronize()
    for _ in range(100):
        individual_start = time.time()
        for _ in range(10):
            calc(batch, forces=True)
        torch.cuda.synchronize()
        runtime = (time.time() - individual_start) / 10
        individual_measurements.append(runtime)
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    
    measurements = np.array(individual_measurements)
    mean_time = measurements.mean()
    std_sem_time = measurements.std(ddof=1) / np.sqrt(np.size(measurements))
    
    measurements_perA = np.array(individual_measurements) / n_waters / 3
    mean_time_perA = measurements_perA.mean()
    std_sem_time_perA = measurements_perA.std(ddof=1) / np.sqrt(np.size(measurements_perA))
    
    with open(f'results/{model_name}/ictp_inference.dat', 'a') as f:
        print(f"{n_waters}\t{total_time}\t{(100*10)*(24*60*60)/(total_time)/1e6}\t{mean_time}\t{std_sem_time}\t{mean_time_perA*1e6}\t{std_sem_time_perA*1e6}", file=f, flush=True)


def ictp_dimos_inference(
    n_feats: int = 64,
    use_estat: bool = True,
    n_waters: int = 256
):
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cuda:0")

    folder = Path('../../models')
    model = f'n_feats_{n_feats}-batch_size_256-' + ('seed_0' if use_estat else 'no_estat-no_disp-seed_0')
    
    os.makedirs('results', exist_ok=True)
    os.makedirs(f'results/{model}', exist_ok=True)
    
    if use_estat:
        alpha, k_max = get_spme_params(f'inputs/water_{n_waters}_npt.gro')
        print("n_atoms:", n_waters)
        print("alpha:", alpha)
        print("k_max for SPME:", k_max)
        new_pair_potential_config = {
            'electrostatics': dict(method='spme', r_cutoff=9.0, alpha=alpha, k_max=k_max, spline_order=5)
        }
    
    else:
        new_pair_potential_config = None
    
    system = ICTPSystem.from_folder_list(
        folder=folder / model / 'best',
        topology_file=f'inputs/water_{n_waters}.top',
        structure_file=f'inputs/water_{n_waters}_npt.gro',
        new_pair_potential_config=new_pair_potential_config,
        device='cuda:0',
        periodic=True,
        use_neighborlist=True,
        with_torch_compile=True,
    )
    positions = dimos.read_positions(f'inputs/water_{n_waters}_npt.gro')
    
    integrator = dimos.LangevinDynamics(0.5, 298.15, 0.01, system, torch.float64)
    simulation = dimos.MDSimulation(system, integrator, initial_pos=positions, temperature=298.15)
    
    torch.compiler.reset()
    step = torch.compile(simulation.step, mode="max-autotune-no-cudagraphs")
    
    for _ in range(10):
        step(10)
    
    individual_measurements = []
    
    start_time = time.time()
    
    torch.cuda.synchronize()
    for _ in range(100):
        individual_start = time.time()
        step(10)
        torch.cuda.synchronize()
        runtime = (time.time() - individual_start) / 10
        individual_measurements.append(runtime)
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    
    measurements = np.array(individual_measurements)
    mean_time = measurements.mean()
    std_sem_time = measurements.std(ddof=1) / np.sqrt(np.size(measurements))
    
    measurements_perA = np.array(individual_measurements) / n_waters / 3
    mean_time_perA = measurements_perA.mean()
    std_sem_time_perA = measurements_perA.std(ddof=1) / np.sqrt(np.size(measurements_perA))
    
    with open(f'results/{model}/ictp_dimos_inference.dat', 'a') as f:
        print(f"{n_waters}\t{total_time}\t{(100*10)*(24*60*60)/(total_time)/1e6}\t{mean_time}\t{std_sem_time}\t{mean_time_perA*1e6}\t{std_sem_time_perA*1e6}", file=f, flush=True)


if __name__ == '__main__':
    
    for n_feats in [64, 128, 256]:
        for n_waters in [512, 1024, 2048, 4096, 8192, 16384, 32768]:
            try:
                ictp_dimos_inference(n_feats, True, n_waters)
            except Exception as e:
                print(f'ICTP-LR+DIMOS inference, {n_feats=}, {n_waters=}: {e}')
    
    for n_feats in [128]:
        for n_waters in [512, 1024, 2048, 4096, 8192, 16384, 32768]:
            try:
                ictp_dimos_inference(n_feats, False, n_waters)
            except Exception as e:
                print(f'ICTP-SR+DIMOS inference, {n_feats=}, {n_waters=}: {e}')
                
    for n_feats in [64, 128, 256]:
        for n_waters in [512, 1024, 2048, 4096, 8192, 16384, 32768]:
            try:
                ictp_inference(n_feats, True, n_waters)
            except Exception as e:
                print(f'ICTP-LR inference, {n_feats=}, {n_waters=}: {e}')
    
    for n_feats in [128]:
        for n_waters in [512, 1024, 2048, 4096, 8192, 16384, 32768]:
            try:
                ictp_inference(n_feats, False, n_waters)
            except Exception as e:
                print(f'ICTP-SR inference, {n_feats=}, {n_waters=}: {e}')
    