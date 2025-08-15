"""
       ICTP: Irreducible Cartesian Tensor Potentials
	  
  File:     datasets.py
  Authors:  Viktor Zaverkin (viktor.zaverkin@neclab.eu)
            Matheus Ferraz (matheus@oncoimmunity.com)
            Francesco Alesiani (francesco.alesiani@neclab.eu)
            Mathias Niepert (mathias.niepert@ki.uni-stuttgart.de)

NEC Laboratories Europe GmbH, Copyright (c) 2025, All rights reserved.  

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
 
       PROPRIETARY INFORMATION ---  

SOFTWARE LICENSE AGREEMENT

ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY

BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
DOWNLOAD THE SOFTWARE.

This is a license agreement ("Agreement") between your academic institution
or non-profit organization or self (called "Licensee" or "You" in this
Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
Agreement).  All rights not specifically granted to you in this Agreement
are reserved for Licensor. 

RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
ownership of any copy of the Software (as defined below) licensed under this
Agreement and hereby grants to Licensee a personal, non-exclusive,
non-transferable license to use the Software for noncommercial research
purposes, without the right to sublicense, pursuant to the terms and
conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
Agreement, the term "Software" means (i) the actual copy of all or any
portion of code for program routines made accessible to Licensee by Licensor
pursuant to this Agreement, inclusive of backups, updates, and/or merged
copies permitted hereunder or subsequently supplied by Licensor,  including
all or any file structures, programming instructions, user interfaces and
screen formats and sequences as well as any and all documentation and
instructions related to it, and (ii) all or any derivatives and/or
modifications created or made by You to any of the items specified in (i).

CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
proprietary to Licensor, and as such, Licensee agrees to receive all such
materials and to use the Software only in accordance with the terms of this
Agreement.  Licensee agrees to use reasonable effort to protect the Software
from unauthorized use, reproduction, distribution, or publication. All
publication materials mentioning features or use of this software must
explicitly include an acknowledgement the software was developed by NEC
Laboratories Europe GmbH.

COPYRIGHT: The Software is owned by Licensor.  

PERMITTED USES:  The Software may be used for your own noncommercial
internal research purposes. You understand and agree that Licensor is not
obligated to implement any suggestions and/or feedback you might provide
regarding the Software, but to the extent Licensor does so, you are not
entitled to any compensation related thereto.

DERIVATIVES: You may create derivatives of or make modifications to the
Software, however, You agree that all and any such derivatives and
modifications will be owned by Licensor and become a part of the Software
licensed to You under this Agreement.  You may only use such derivatives and
modifications for your own noncommercial internal research purposes, and you
may not otherwise use, distribute or copy such derivatives and modifications
in violation of this Agreement.

BACKUPS:  If Licensee is an organization, it may make that number of copies
of the Software necessary for internal noncommercial use at a single site
within its organization provided that all information appearing in or on the
original labels, including the copyright and trademark notices are copied
onto the labels of the copies.

USES NOT PERMITTED:  You may not distribute, copy or use the Software except
as explicitly permitted herein. Licensee has not been granted any trademark
license as part of this Agreement.  Neither the name of NEC Laboratories
Europe GmbH nor the names of its contributors may be used to endorse or
promote products derived from this Software without specific prior written
permission.

You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
whole or in part, or provide third parties access to prior or present
versions (or any parts thereof) of the Software.

ASSIGNMENT: You may not assign this Agreement or your rights hereunder
without the prior written consent of Licensor. Any attempted assignment
without such consent shall be null and void.

TERM: The term of the license granted by this Agreement is from Licensee's
acceptance of this Agreement by downloading the Software or by using the
Software until terminated as provided below.  

The Agreement automatically terminates without notice if you fail to comply
with any provision of this Agreement.  Licensee may terminate this Agreement
by ceasing using the Software.  Upon any termination of this Agreement,
Licensee will delete any and all copies of the Software. You agree that all
provisions which operate to protect the proprietary rights of Licensor shall
remain in force should breach occur and that the obligation of
confidentiality described in this Agreement is binding in perpetuity and, as
such, survives the term of the Agreement.

FEE: Provided Licensee abides completely by the terms and conditions of this
Agreement, there is no fee due to Licensor for Licensee's use of the
Software in accordance with this Agreement.

DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
RELATED MATERIALS.

SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
provided as part of this Agreement.  

EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
permitted under applicable law, Licensor shall not be liable for direct,
indirect, special, incidental, or consequential damages or lost profits
related to Licensee's use of and/or inability to use the Software, even if
Licensor is advised of the possibility of such damage.

EXPORT REGULATION: Licensee agrees to comply with any and all applicable
export control laws, regulations, and/or other laws related to embargoes and
sanction programs administered by law.

SEVERABILITY: If any provision(s) of this Agreement shall be held to be
invalid, illegal, or unenforceable by a court or other tribunal of competent
jurisdiction, the validity, legality and enforceability of the remaining
provisions shall not in any way be affected or impaired thereby.

NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
or remedy under this Agreement shall be construed as a waiver of any future
or other exercise of such right or remedy by Licensor.

GOVERNING LAW: This Agreement shall be construed and enforced in accordance
with the laws of Germany without reference to conflict of laws principles.
You consent to the personal jurisdiction of the courts of this country and
waive their rights to venue outside of Germany.

ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
entire agreement between Licensee and Licensor as to the matter set forth
herein and supersedes any previous agreements, understandings, and
arrangements between the parties relating hereto.

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
"""
from typing import *

import copy

from pathlib import Path

import numpy as np

import parmed

import torch
import torch.nn as nn

import csv

from ase import Atoms
from ase import units as ase_units
from ase.io import Trajectory

import dimos

from ictp.data.data import AtomicTypeConverter, to_one_hot

from ictp.model.forward import ForwardAtomisticNetwork, find_last_ckpt
from ictp.model.pair_potentials import (CoulombElectrostaticEnergy, 
                                        EwaldElectrostaticEnergy, 
                                        SPMEElectrostaticEnergy)
from ictp.model.calculators import TorchCalculator, StructurePropertyCalculator

from ictp.utils.torch_geometric import Data
from ictp.utils.misc import get_default_device, find_max_r_cutoff, to_tensor


class ICTPSystem(dimos.MinimalSystem):
    """
    Wraps a `TorchCalculator` object into an ICTP system for molecular simulations with DIMOS.

    Args:
        calc (TorchCalculator): The `TorchCalculator` to evaluate energies and forces.
        topology_file (str): Path to the topology file (e.g., .top).
        structure_file (str): Path to the structure file (e.g., .gro).
        cutoff (float): Cutoff distance for interactions.
        atomic_types (List[str], optional): List of atomic types. Defaults to None.
        device (str, optional): Device ('cpu' or 'cuda'). Defaults to None.
        periodic (bool): If True, treat the atomic system as periodic. Defaults to False.
        use_neighborlist (bool): Enables neighbor list computation. Defaults to True.
        unit_system (str): The unit system. Defaults to 'amber'.
        total_charge (float): Total charge of the system. Defaults to 0.0.
    """
    def __init__(
        self,
        calc: TorchCalculator,
        topology_file: str,
        structure_file: str,
        cutoff: float,
        atomic_types: Optional[List[str]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[str] = None,
        periodic: bool = False,
        use_neighborlist: bool = True,
        unit_system: str = 'amber',
        total_charge: float = 0.0,
        **config: Any
    ):
        super(ICTPSystem, self).__init__(unit_system=unit_system, create_graph=False, dtype=dtype)
        # initialize unit system constants
        dimos.constants.init_constants_in_unit_system(unit_system)
        
        self.unit_system = unit_system
        self.device = device or get_default_device()
        self.cutoff = cutoff
        self.periodic = periodic
        self.use_neighborlist = use_neighborlist
        self.all_exclusions = torch.tensor([])
        self.bias_calc = None
        
        # move the main calculator to device
        self.calc = calc.to(self.device)
        
        # conversion factors for units
        self.energy_units_to_eV = 1.0 / dimos.constants.eV_to_internal
        self.length_units_to_A = dimos.constants.internal_to_Angstrom
        self.time_units_to_fs = dimos.constants.FS_TO_INTERNAL
        
        # load topology & structure data
        self.parmed_data = parmed.load_file(topology_file, xyz=structure_file)
        self.num_atoms = len(self.parmed_data.atoms)
        
        # define atomic numbers and masses
        self.masses = torch.tensor(
            [atom.mass for atom in self.parmed_data.atoms], dtype=torch.get_default_dtype()
        )
        self.atomic_numbers = torch.tensor(
            [atom.atomic_number for atom in self.parmed_data.atoms], dtype=torch.long
        )
        
        # convert atomic numbers to internal ICTP species
        atomic_type_converter = AtomicTypeConverter.from_type_list(atomic_types)
        species = atomic_type_converter.to_type_names(self.atomic_numbers.detach().cpu().numpy())
        n_species = atomic_type_converter.get_n_type_names()
        
        # initialize box, molecules, and batch
        self._initialize_box()
        self._initialize_molecules()
        self.batch = self._initialize_batch(species, total_charge, n_species)
        
        # cache results to avoid redundant computations
        self.results = {}
        self.last_positions = None
        self.last_box = None
    
    def _initialize_box(self):
        if self.parmed_data.box is None:
            if self.periodic:
                raise ValueError('Periodic system requires a defined box.')
            self.box = None
        else:
            if not self.periodic:
                raise ValueError('Non-periodic system provided with a box.')
            self.box = torch.tensor(self.parmed_data.box[:3], dtype=torch.float32)
            if not self._is_orthogonal_box():
                raise ValueError(
                    f"Only orthogonal boxes are supported. Provided: {self.parmed_data.box}."
                )
    
    def _is_orthogonal_box(self) -> bool:
        angles = self.parmed_data.box[3:]
        return all(angle == 90.0 for angle in angles)
    
    def _initialize_molecules(self):
        molecules = parmed.utils.tag_molecules(self.parmed_data)
        self.num_molecules = len(molecules)
        
        self.length3_molecules = torch.tensor(
            [list(mol) for mol in molecules if len(mol) == 3]
        )
        self.other_molecules = [
            torch.tensor(list(mol)) for mol in molecules if len(mol) != 3
        ]

    def _initialize_batch(
        self,
        species: Union[np.ndarray, List[int]],
        total_charge: float,
        n_species: int
    ) -> Dict[str, Any]:
        return {
            'num_nodes': to_tensor([self.num_atoms], dtype=torch.long),
            'n_atoms': to_tensor([self.num_atoms], dtype=torch.long),
            'node_attrs': to_one_hot(species, n_species),
            'species': to_tensor(species, dtype=torch.long),
            'atomic_numbers': self.atomic_numbers,
            'cell': torch.diag(self.box).unsqueeze(0) if self.box is not None else torch.zeros(3, 3).unsqueeze(0),
            'total_charge': to_tensor([total_charge]),
            'ptr': torch.tensor([0, self.num_atoms], dtype=torch.long, requires_grad=False),
            'batch': torch.zeros(self.num_atoms, dtype=torch.long, requires_grad=False)
        }
    
    def apply_boundary_conditions(self, pos: torch.Tensor) -> torch.Tensor:
        offset = torch.floor(pos / self.box) * self.box
        return pos - offset
    
    def update_box(self, box: torch.Tensor) -> None:
        self.box = box
        self.batch['cell'] = torch.diag(self.box).unsqueeze(0) if self.box is not None else torch.zeros(3, 3).unsqueeze(0)
    
    def measure_temperature(self, vel: torch.Tensor):
        return dimos.utils.measure_temperature(vel, self, num_constraints=0).detach().cpu().item()
    
    def measure_density(self):
        volume = torch.prod(self.box)
        total_mass = torch.sum(self.masses)
        density = total_mass / (6.022140857e23 * volume * 10**(-30)) * 10**(-6)
        return density.detach().cpu().item()
    
    def calculation_required(self, positions: torch.Tensor) -> bool:
        if (self.last_positions is not None and torch.equal(positions, self.last_positions)) and \
           (self.last_box is not None and self.box is not None and torch.equal(self.box, self.last_box)):
            return False
        self.last_positions = positions.clone().detach()
        if self.box is not None:
            self.last_box = self.box.clone().detach()
        return True
    
    def get_shifts(
        self,
        positions: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
    ) -> torch.Tensor:
        vectors = positions.index_select(0, idx_i) - positions.index_select(0, idx_j)
        if self.periodic is not None and self.box is not None:
            return torch.floor(vectors / self.box + 0.5) * self.box
        return torch.zeros_like(idx_i)
    
    def add_bias(self, bias_calc: 'BiasCalculator') -> None:
        """
        Attach a bias calculator to the system.
        
        Args:
            bias_calc (BiasCalculator): Bias calculator.
        """
        self.bias_calc = bias_calc
    
    def calc_energy(
        self,
        positions: torch.Tensor,
        neighborlist: torch.Tensor,
        return_forces: bool = False,
        **kwargs: Any,
    ) -> Tuple:
        positions = positions.to(self.dtype).clone().detach()
        neighborlist = neighborlist.to(torch.long).clone().detach()
        
        if self.calculation_required(positions):
            idx_i, idx_j = neighborlist
            shifts = self.get_shifts(positions, idx_i, idx_j)

            # create bidirectional neighbor list
            idx_i_bi, idx_j_bi = torch.cat([idx_i, idx_j]), torch.cat([idx_j, idx_i])
            shifts_bi = torch.cat([shifts, -shifts])
            
            # prepare input batch
            batch = {key: value.clone().detach() for key, value in self.batch.items()}
            batch.update({
                "positions": positions * self.length_units_to_A,
                "edge_index": torch.stack([idx_i_bi, idx_j_bi]),
                "shifts": shifts_bi,
            })
            
            # prform energy (and force) calculation
            out = self.calc(Data(**batch), forces=return_forces)
            
            self.results['energy'] = out['energy'].detach() / self.energy_units_to_eV
            if return_forces:
                self.results['forces'] = out['forces'].detach() * self.length_units_to_A / self.energy_units_to_eV
                self.results['acceleration'] = self.results['forces'] / self.masses.unsqueeze(-1)
            
            if self.bias_calc is not None:
                bias_out = self.bias_calc(positions, self.masses, self.results['energy'], self.box)
                
                for key in ["energy", "forces"]:
                    if key in self.results and key in bias_out:
                        self.results[key] += bias_out[key].to(torch.float64)
                
                if return_forces and "forces" in bias_out:
                    self.results["acceleration"] += bias_out["forces"] / self.masses.unsqueeze(-1)
            
        # return results
        if not return_forces:
            return self.results['energy']
        return self.results['energy'], self.results['forces'], self.results['acceleration']
    
    @staticmethod
    def from_folder_list(
        folder: Union[Path, str],
        new_pair_potential_config: Optional[Dict[str, Any]] = None,
        with_torch_script: bool = False,
        with_torch_compile: bool = False,
        **kwargs: Any
    ) -> 'ICTPSystem':
        # load model from folder
        model = ForwardAtomisticNetwork.from_folder(find_last_ckpt(folder))
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
            with_torch_script=with_torch_script,
            with_torch_compile=with_torch_compile
        )
        
        return ICTPSystem(
            calc=calc,
            cutoff=find_max_r_cutoff(config),
            atomic_types=config['atomic_types'],
            **kwargs
        )


class BiasCalculator:
    def __call__(
        self,
        positions: torch.Tensor,
        masses: torch.Tensor,
        unbiased_energy: torch.Tensor,
        box: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class PLUMEDCalculator(BiasCalculator):
    def __init__(
        self,
        plumed_input: List[str],
        num_atoms: int,
        kT: float,
        kJ_mol: float,
        nm: float,
        ps: float,
        timestep: float,
        log: str = '',
        restart: bool = False,
    ):
        from plumed import Plumed
        
        self.plumed = Plumed()
        self.plumed_input = plumed_input
        self.istep = 0
        self.last_istep = -1
        
        self._initialize_plumed(num_atoms, kT, kJ_mol, nm, ps, timestep, log, restart)
        self._load_input()
    
    def _initialize_plumed(self, num_atoms, kT, kJ_mol, nm, ps, timestep, log, restart):
        commands = {
            "setMDEnergyUnits": kJ_mol,
            "setMDLengthUnits": 1.0 / nm,
            "setMDTimeUnits": 1.0 / ps,
            "setMDChargeUnits": 1.0,
            "setMDMassUnits": 1.0,
            "setNatoms": num_atoms,
            "setMDEngine": "DIMOS",
            "setLogFile": log,
            "setTimestep": timestep,
            "setRestart": restart,
            "setKbT": kT,
        }
        
        for cmd, value in commands.items():
            self.plumed.cmd(cmd, value)
        
        self.plumed.cmd("init")

    def _load_input(self):
        for line in self.plumed_input:
            self.plumed.cmd("readInputLine", line)
    
    @torch.compiler.disable(recursive=False)
    def __call__(
        self,
        positions: torch.Tensor,
        masses: torch.Tensor,
        unbiased_energy: torch.Tensor,
        box: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        self.plumed.cmd("setStep", self.istep)
        
        if box is not None:
            cell_np = torch.diag(box).detach().cpu().numpy().astype(np.float64)
            self.plumed.cmd("setBox", cell_np)
            
        positions_np = positions.detach().cpu().numpy().astype(np.float64)
        self.plumed.cmd("setPositions", positions_np)
        
        unbiased_energy_np = unbiased_energy.detach().cpu().numpy().astype(np.float64)
        self.plumed.cmd("setEnergy", unbiased_energy_np)
        
        masses_np = masses.detach().cpu().numpy().astype(np.float64)
        self.plumed.cmd("setMasses", masses_np)
        
        forces_bias_np = np.zeros(positions_np.shape)
        self.plumed.cmd("setForces", forces_bias_np)
        
        virial_np = np.zeros((3, 3))
        self.plumed.cmd("setVirial", virial_np)

        self.plumed.cmd("prepareCalc")
        self.plumed.cmd("performCalcNoUpdate")
        if self.istep != self.last_istep:
            self.plumed.cmd("update")
            self.last_istep = self.istep
        
        energy_bias_np = np.zeros((1,))
        self.plumed.cmd("getBias", energy_bias_np)
        
        results = {
            'energy': torch.tensor(energy_bias_np, dtype=positions.dtype, device=positions.device),
            'forces': torch.tensor(forces_bias_np, dtype=positions.dtype, device=positions.device)
        }
        
        return results


class PLUMEDMetadynamics:
    def __init__(
        self,
        simulation: dimos.MDSimulation,
        plumed_input: List[str],
        temperature: float,
        timestep: float,
        log: str = '',
        restart: bool = False,
    ):
        dimos.constants.init_constants_in_unit_system(simulation.sys.unit_system)
        
        self.simulation = simulation
        
        conversion_factors = {
            "ps": 1000.0 / dimos.constants.FS_TO_INTERNAL,
            "nm": 10.0 / dimos.constants.internal_to_Angstrom,
            "kJ_mol": 96.48533288249877 / dimos.constants.eV_to_internal,
            "kT": temperature * dimos.constants.BOLTZMANN,
            "timestep": timestep / 1000.0,
        }
        
        self.bias_calc = PLUMEDCalculator(
            plumed_input,
            simulation.sys.num_atoms,
            **conversion_factors,
            log=log,
            restart=restart
        )
        
        self.simulation.sys.add_bias(self.bias_calc)
        
    def step(self, num_steps=1, detach=True):
        for _ in range(num_steps):
            self.simulation.step(1, detach=detach)
            self.bias_calc.istep += 1


class ASETrajectoryWriter:
    """
    Writes frames into an ASE trajectory.

    Args:
        filename (str): Path to the trajectory file.
        simulation (dimos.Simulation): The simulation object containing dynamic state information.
        system (dimos.MinimalSystem): The system defining atomic properties, such as atomic numbers and energies.
        write_energy (bool, optional): Whether to store total energy per frame. Defaults to False.
        write_forces (bool, optional): Whether to store atomic forces per frame. Defaults to False.
        write_velocities (bool, optional): Whether to store velocities per frame. Defaults to False.
        tag (str, optional): Prefix tag for storing additional metadata in the trajectory. Defaults to ''.
    """
    def __init__(
        self, 
        filename: str,
        simulation: dimos.Simulation,
        system: dimos.MinimalSystem,
        write_energy: bool = False,
        write_forces: bool = False,
        write_velocities: bool = False,
        tag: str = '',
    ):
        self.filename = filename
        self.simulation = simulation
        self.system = system
        self.write_energy = write_energy
        self.write_forces = write_forces
        self.write_velocities = write_velocities
        self.tag = tag

        with Trajectory(self.filename, 'w') as _:
            pass

    def append_frame(self) -> None:
        """
        Appends a frame to the trajectory file.
        """
        atoms = Atoms(
            numbers=self.system.atomic_numbers.detach().cpu().numpy(),
            positions=self.simulation.pos.detach().cpu().numpy()
        )
        
        if self.system.box is not None:
            atoms.set_cell(self.system.box.detach().cpu().numpy())
            atoms.set_pbc(True)
        
        if self.write_forces:
            self.system.calc_energy(self.simulation.pos, self.simulation.neighborlist, return_forces=True)
            atoms.arrays[self.tag + 'forces'] = self.system.results['forces'].detach().cpu().numpy() * self.system.energy_units_to_eV / self.system.length_units_to_A
        
        if self.write_energy:
            self.system.calc_energy(self.simulation.pos, self.simulation.neighborlist)
            atoms.info[self.tag + 'energy'] = self.system.results['energy'].detach().cpu().item() * self.system.energy_units_to_eV
            
        if self.write_velocities:
            atoms.set_velocities(self.simulation.vel.detach().cpu().numpy() * self.system.time_units_to_fs / self.system.length_units_to_A / ase_units.fs)
        
        with Trajectory(self.filename, 'a') as traj:
            traj.write(atoms)


class MDLogger:
    """
    Writes history of a molecular dynamics simulation.

    Args:
        filename (str): Path to the CSV file.
        simulation (dimos.Simulation): The simulation object containing dynamic state information.
        system (dimos.MinimalSystem): The system object used for temperature and density calculations.
        write_interval (int, optional): Number of steps between logging events. Defaults to 100.
        write_density (bool, optional): Whether to log density values. Defaults to False.
    """
    def __init__(
        self,
        filename: str,
        simulation: dimos.Simulation,
        system: dimos.MinimalSystem,
        write_interval: int = 100,
        write_density: bool = False
    ):
        self.filename = filename
        self.simulation = simulation
        self.system = system
        self.write_interval = write_interval
        self.write_density = write_density
        self.total_time = 0.0

        headers = ["Total Time (ps)", "Total Energy (eV)", "Potential Energy (eV)", 
                   "Kinetic Energy (eV)", "Temperature (K)"]
        if self.write_density:
            headers.append("Density (g/cm^3)")

        with open(self.filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)

    def log_step(self) -> None:
        """
        Writes the MD step values using simulation and system.
        """
        timestep = self.simulation.integrator.get_timestep() * self.write_interval
        self.total_time += timestep / 1e3

        kinetic_energy = self.simulation.measure_kinetic_energy() * self.system.energy_units_to_eV
        potential_energy = self.simulation.measure_potential_energy() * self.system.energy_units_to_eV
        total_energy = kinetic_energy + potential_energy

        temperature = self.system.measure_temperature(self.simulation.vel)

        density = None
        if self.write_density:
            try:
                density = self.system.measure_density()
            except AttributeError:
                self.write_density = False

        def to_float(value: Union[float, torch.Tensor, None]) -> Optional[float]:
            return value.detach().cpu().item() if isinstance(value, torch.Tensor) else value
        
        def format_value(value):
            return f"{value:.3f}"

        values = [
            to_float(self.total_time),
            to_float(total_energy),
            to_float(potential_energy),
            to_float(kinetic_energy),
            to_float(temperature),
        ]
        if self.write_density:
            values.append(to_float(density))

        with open(self.filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(map(format_value, values))
