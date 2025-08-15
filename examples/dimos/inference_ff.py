import os
from typing import *
import torch
import numpy as np
import time
import dimos


def amber_dimos_inference(n_waters: int = 256):
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cuda:0")

    model = 'amber14sb+tip3p'
    
    os.makedirs('results', exist_ok=True)
    os.makedirs(f'results/{model}', exist_ok=True)
    
    gromacs_system = dimos.ff.GromacsForceField(
        parameter_file=f'inputs/water_{n_waters}.top',
        xyz_file=f'inputs/water_{n_waters}_npt.gro',
        cutoff=9.0,
        switch_distance=7.5,
        # TODO: In the constructor, the memory cost of the dispersion correction is O(N^2).
        # dispersion_correction=True,
        nonbonded_type='PME',
        unit_system='amber',
        periodic=True,
    )
    positions = dimos.read_positions(f'inputs/water_{n_waters}_npt.gro')
    
    integrator = dimos.LangevinDynamics(0.5, 298.15, 0.01, gromacs_system, torch.float64)
    simulation = dimos.MDSimulation(gromacs_system, integrator, initial_pos=positions, temperature=298.15)
    
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
    
    with open(f'results/{model}/amber_dimos_inference.dat', 'a') as f:
        print(f"{n_waters}\t{total_time}\t{(100*10)*(24*60*60)/(total_time)/1e6}\t{mean_time}\t{std_sem_time}\t{mean_time_perA*1e6}\t{std_sem_time_perA*1e6}", file=f, flush=True)


if __name__ == '__main__':

    for n_waters in [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]:
        try:
            amber_dimos_inference(n_waters)
        except Exception as e:
            print(f'AMBER+DIMOS inference, {n_waters=}: {e}')
    