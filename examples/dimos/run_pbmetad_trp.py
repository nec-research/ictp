import os
import numpy as np
import torch
import multiprocessing
from pathlib import Path
import dimos
from ase.io import read
from ictp.interfaces.dimos import (ICTPSystem, PLUMEDMetadynamics, 
                                   ASETrajectoryWriter, MDLogger)


def run_pbmetad(
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
    os.makedirs('results/' + model + '/pbmetad-default_grid-frequency_100', exist_ok=True)
    os.makedirs('results/' + model + '/pbmetad-default_grid-frequency_100' + '/WALKERS', exist_ok=True)
    
    system = ICTPSystem.from_folder_list(
        folder=folder / model / 'best',
        topology_file='inputs/trp.top',
        structure_file='inputs/trp.gro',
        new_pair_potential_config={'electrostatics': dict(method='spme', r_cutoff=9.0, alpha=0.3372060287522547, k_max=42, spline_order=5)},
        device=f'cuda:{cuda_id}',
        periodic=True,
        use_neighborlist=True,
        with_torch_compile=True,
    )
    
    dimos.constants.init_constants_in_unit_system(system.unit_system)
    ps = 1000.0 / dimos.constants.FS_TO_INTERNAL
    
    plumed_input = [
        f"UNITS LENGTH=A TIME={1.0 / ps} ENERGY=kcal/mol",
        "WHOLEMOLECULES ENTITY0=1-304",
        " ".join([
            "rg: GYRATION",
            "ATOMS=5,19,38,59,78,95,119,138,160,172,179,194,200,211,222,229,261,275,289,295"
        ]),
        " ".join([
            "hb: COORDINATION",
            "GROUPA=16,35,56,75,92,116,135,157,169,176,183,197,208,219,226,250,264,278,292",
            "GROUPB=18,37,58,77,94,118,137,159,171,178,199,210,221,228,294",
            "NN=8 MM=12 R_0=2.5 NLIST NL_CUTOFF=8.0 NL_STRIDE=5"
        ]),
        " ".join([
            "hc: COORDINATION",
            "GROUPA=24,100,124,188,255,269,283",
            "NN=8 MM=12 R_0=5.0"
        ]),
        " ".join([
            "helix: ALPHABETA",
            "ATOMS1=15,17,19,34 REFERENCE1=-1.0",  
            "ATOMS2=17,19,34,36 REFERENCE2=-0.82",
            "ATOMS3=34,36,38,55 REFERENCE3=-1.0",  
            "ATOMS4=36,38,55,57 REFERENCE4=-0.82",
            "ATOMS5=55,57,59,74 REFERENCE5=-1.0",  
            "ATOMS6=57,59,74,76 REFERENCE6=-0.82",
            "ATOMS7=74,76,78,91 REFERENCE7=-1.0",  
            "ATOMS8=76,78,91,93 REFERENCE8=-0.82",
            "ATOMS9=91,93,95,115 REFERENCE9=-1.0",  
            "ATOMS10=93,95,115,117 REFERENCE10=-0.82",
            "ATOMS11=115,117,119,134 REFERENCE11=-1.0",  
            "ATOMS12=117,119,134,136 REFERENCE12=-0.82",
            "ATOMS13=134,136,138,156 REFERENCE13=-1.0",  
            "ATOMS14=136,138,156,158 REFERENCE14=-0.82",
            "ATOMS15=156,158,160,168 REFERENCE15=-1.0",  
            "ATOMS16=158,160,168,170 REFERENCE16=-0.82",
            "ATOMS17=168,170,172,175 REFERENCE17=-1.0",  
            "ATOMS18=170,172,175,177 REFERENCE18=-0.82",
            "ATOMS19=175,177,179,182 REFERENCE19=-1.0",  
            "ATOMS20=177,179,182,184 REFERENCE20=-0.82",
            "ATOMS21=182,184,194,196 REFERENCE21=-1.0",  
            "ATOMS22=184,194,196,198 REFERENCE22=-0.82",
            "ATOMS23=196,198,200,207 REFERENCE23=-1.0",  
            "ATOMS24=198,200,207,209 REFERENCE24=-0.82",
            "ATOMS25=207,209,211,218 REFERENCE25=-1.0",  
            "ATOMS26=209,211,218,220 REFERENCE26=-0.82",
            "ATOMS27=218,220,222,225 REFERENCE27=-1.0",  
            "ATOMS28=220,222,225,227 REFERENCE28=-0.82",
            "ATOMS29=225,227,229,249 REFERENCE29=-1.0",  
            "ATOMS30=227,229,249,251 REFERENCE30=-0.82",
            "ATOMS31=249,251,261,263 REFERENCE31=-1.0",  
            "ATOMS32=251,261,263,265 REFERENCE32=-0.82",
            "ATOMS33=263,265,275,277 REFERENCE33=-1.0",  
            "ATOMS34=265,275,277,279 REFERENCE34=-0.82",
            "ATOMS35=277,279,289,291 REFERENCE35=-1.0",  
            "ATOMS36=279,289,291,293 REFERENCE36=-0.82",
        ]),
        " ".join([
            "beta: ALPHABETA",
            "ATOMS1=15,17,19,34 REFERENCE1=-1.396",  
            "ATOMS2=17,19,34,36 REFERENCE2=2.618",
            "ATOMS3=34,36,38,55 REFERENCE3=-1.396",  
            "ATOMS4=36,38,55,57 REFERENCE4=2.618",
            "ATOMS5=55,57,59,74 REFERENCE5=-1.396",  
            "ATOMS6=57,59,74,76 REFERENCE6=2.618",
            "ATOMS7=74,76,78,91 REFERENCE7=-1.396",  
            "ATOMS8=76,78,91,93 REFERENCE8=2.618",
            "ATOMS9=91,93,95,115 REFERENCE9=-1.396",  
            "ATOMS10=93,95,115,117 REFERENCE10=2.618",
            "ATOMS11=115,117,119,134 REFERENCE11=-1.396",  
            "ATOMS12=117,119,134,136 REFERENCE12=2.618",
            "ATOMS13=134,136,138,156 REFERENCE13=-1.396",  
            "ATOMS14=136,138,156,158 REFERENCE14=2.618",
            "ATOMS15=156,158,160,168 REFERENCE15=-1.396",  
            "ATOMS16=158,160,168,170 REFERENCE16=2.618",
            "ATOMS17=168,170,172,175 REFERENCE17=-1.396",  
            "ATOMS18=170,172,175,177 REFERENCE18=2.618",
            "ATOMS19=175,177,179,182 REFERENCE19=-1.396",  
            "ATOMS20=177,179,182,184 REFERENCE20=2.618",
            "ATOMS21=182,184,194,196 REFERENCE21=-1.396",  
            "ATOMS22=184,194,196,198 REFERENCE22=2.618",
            "ATOMS23=196,198,200,207 REFERENCE23=-1.396",  
            "ATOMS24=198,200,207,209 REFERENCE24=2.618",
            "ATOMS25=207,209,211,218 REFERENCE25=-1.396",  
            "ATOMS26=209,211,218,220 REFERENCE26=2.618",
            "ATOMS27=218,220,222,225 REFERENCE27=-1.396",  
            "ATOMS28=220,222,225,227 REFERENCE28=2.618",
            "ATOMS29=225,227,229,249 REFERENCE29=-1.396",  
            "ATOMS30=227,229,249,251 REFERENCE30=2.618",
            "ATOMS31=249,251,261,263 REFERENCE31=-1.396",  
            "ATOMS32=251,261,263,265 REFERENCE32=2.618",
            "ATOMS33=263,265,275,277 REFERENCE33=-1.396",  
            "ATOMS34=265,275,277,279 REFERENCE34=2.618",
            "ATOMS35=277,279,289,291 REFERENCE35=-1.396",  
            "ATOMS36=279,289,291,293 REFERENCE36=2.618",
        ]),
        " ".join([
            "dih: DIHCOR",
            "ATOMS1=15,17,19,34,17,19,34,36",
            "ATOMS2=17,19,34,36,34,36,38,55",
            "ATOMS3=34,36,38,55,36,38,55,57",
            "ATOMS4=36,38,55,57,55,57,59,74",
            "ATOMS5=55,57,59,74,57,59,74,76",
            "ATOMS6=57,59,74,76,74,76,78,91",
            "ATOMS7=74,76,78,91,76,78,91,93",
            "ATOMS8=76,78,91,93,91,93,95,115",
            "ATOMS9=91,93,95,115,93,95,115,117",
            "ATOMS10=93,95,115,117,115,117,119,134",
            "ATOMS11=115,117,119,134,117,119,134,136",
            "ATOMS12=117,119,134,136,134,136,138,156",
            "ATOMS13=134,136,138,156,136,138,156,158",
            "ATOMS14=136,138,156,158,156,158,160,168",
            "ATOMS15=156,158,160,168,158,160,168,170",
            "ATOMS16=158,160,168,170,168,170,172,175",
            "ATOMS17=168,170,172,175,170,172,175,177",
            "ATOMS18=170,172,175,177,175,177,179,182",
            "ATOMS19=175,177,179,182,177,179,182,184",
            "ATOMS20=177,179,182,184,182,184,194,196",
            "ATOMS21=182,184,194,196,184,194,196,198",
            "ATOMS22=184,194,196,198,196,198,200,207",
            "ATOMS23=196,198,200,207,198,200,207,209",
            "ATOMS24=198,200,207,209,207,209,211,218",
            "ATOMS25=207,209,211,218,209,211,218,220",
            "ATOMS26=209,211,218,220,218,220,222,225",
            "ATOMS27=218,220,222,225,220,222,225,227",
            "ATOMS28=220,222,225,227,225,227,229,249",
            "ATOMS29=225,227,229,249,227,229,249,251",
            "ATOMS30=227,229,249,251,249,251,261,263",
            "ATOMS31=249,251,261,263,251,261,263,265",
            "ATOMS32=251,261,263,265,263,265,275,277",
            "ATOMS33=263,265,275,277,265,275,277,279",
            "ATOMS34=265,275,277,279,277,279,289,291",
            "ATOMS35=277,279,289,291,279,289,291,293",
        ]),
        " ".join([
            "pb: PBMETAD",
            "ARG=rg,hb,hc,helix,beta,dih",
            f"SIGMA=0.1,0.6,0.3,0.4,0.3,0.6 HEIGHT=0.25 PACE=500 BIASFACTOR=8.0 TEMP={temperature}",
            "FILE=HILLS_rg,HILLS_hb,HILLS_hc,HILLS_helix,HILLS_beta,HILLS_dih",
            "GRID_MIN=4.0,0.0,0.0,0.0,0.0,0.0 GRID_MAX=20.0,100.0,25.0,40.0,40.0,40.0",
            f"WALKERS_N={walkers_n} WALKERS_ID={walker_id} WALKERS_DIR=results/{model}/pbmetad-default_grid-frequency_100/WALKERS WALKERS_RSTRIDE=250",
        ]),
        f"PRINT ARG=rg,hb,hc,helix,beta,dih,pb.bias STRIDE=500 FILE=results/{model}/pbmetad-default_grid-frequency_100/WALKERS/COLVAR.{walker_id}",
        "FLUSH STRIDE=500",
    ]
    
    atoms = read(f'results/{model}/trp_eq.traj', '-1')
    positions = torch.tensor(atoms.get_positions(), dtype=torch.get_default_dtype(), device=torch.get_default_device())
    box = torch.tensor(np.diag(atoms.get_cell()), dtype=torch.get_default_dtype(), device=torch.get_default_device())
    system.update_box(box)
    
    integrator = dimos.LangevinDynamics(timestep, temperature, friction, system, integrator_dtype)
    barostat = dimos.MCBarostatIsotropic(system.box, target_pressure=pressure, frequency=frequency, rescale_whole_system=rescale_whole_system)
    simulation = dimos.MDSimulation(system, integrator, initial_pos=positions, temperature=temperature, barostat=barostat, seed=walker_id)
    biased_simulation = PLUMEDMetadynamics(simulation, plumed_input=plumed_input, temperature=temperature, timestep=timestep, log=f'results/{model}/pbmetad-default_grid-frequency_100/trp_{walker_id}.log')
    
    traj_writer = ASETrajectoryWriter(f'results/{model}/pbmetad-default_grid-frequency_100/trp_{walker_id}.traj', simulation, system, write_velocities=True)
    md_logger = MDLogger(f'results/{model}/pbmetad-default_grid-frequency_100/trp_{walker_id}.csv', simulation, system, write_interval=write_interval, write_density=True)
    
    for _ in range(total_steps // write_interval):
        biased_simulation.step(write_interval)
        md_logger.log_step()
        traj_writer.append_frame()


if __name__ == '__main__':
    n_gpus = 6
    n_walkers = 6
    
    # 10 ns long simulations for each walker = 60 ns in total
    args = [
        (i % n_gpus, i, n_walkers, 1.0, 298.15, 1.0, 0.01, 100, True, 10000000, 500, torch.float64)
        for i in range(n_walkers)
    ]
    
    with multiprocessing.Pool(processes=n_walkers) as pool:
        pool.starmap(run_pbmetad, args)
