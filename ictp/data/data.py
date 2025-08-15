"""
       ICTP: Irreducible Cartesian Tensor Potentials
	  
  File:     data.py
  Authors:  Viktor Zaverkin (viktor.zaverkin@neclab.eu)
            Francesco Alesiani (francesco.alesiani@neclab.eu)
            Takashi Maruyama (takashi.maruyama@neclab.eu)
            Federico Errica (federico.errica@neclab.eu)
            Henrik Christiansen (henrik.christiansen@neclab.eu)
            Makoto Takamoto (makoto.takamoto@neclab.eu)
            Nicolas Weber (nicolas.weber@neclab.eu)
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
import os

from pathlib import Path

from typing import *

import h5py

import lmdb
import pickle

import numpy as np

import scipy

import torch

import ase
import ase.data
from ase.io import write, read

from ictp.data.neighbors import get_matscipy_neighbors, get_vesin_neighbors, primitive_pairs

from ictp.utils.torch_geometric import Data
from ictp.utils.misc import to_tensor


class AtomicTypeConverter:
    """
    Converts atomic numbers to internal types and vice versa.

    Args:
        to_atomic_numbers (np.ndarray): Array for mapping from internal types to atomic numbers.
        from_atomic_numbers (np.ndarray): Array for mapping from atomic numbers to internal types.
    """
    def __init__(
        self,
        to_atomic_numbers: np.ndarray,
        from_atomic_numbers: np.ndarray
    ):
        self._to_atomic_numbers = to_atomic_numbers
        self._from_atomic_numbers = from_atomic_numbers

    def to_type_names(
        self,
        atomic_numbers: np.ndarray,
        check: bool = True
    ) -> np.ndarray:
        """
        Converts an array with atomic numbers to an array with internal types.

        Args:
            atomic_numbers (np.ndarray): Array with atomic numbers.
            check (bool, optional): If True, check if atomic numbers are supported.

        Returns:
            np.ndarray: Array with internal types.
        """
        result = self._from_atomic_numbers[atomic_numbers]
        if check:
            assert np.all(result >= 0)
        return result

    def to_atomic_numbers(self, species: np.ndarray) -> np.ndarray:
        """
        Converts an array with internal types to an array with atomic numbers.

        Args:
            species (np.ndarray): Array with internal types.

        Returns:
            np.ndarray: Array with atomic numbers.
        """
        return self._to_atomic_numbers[species]

    def get_n_type_names(self) -> int:
        """

        Returns:
            int: The total number of species/elements.
        """
        return len(self._to_atomic_numbers)

    @staticmethod
    def from_type_list(atomic_types: Optional[List[Union[str, int]]] = None) -> 'AtomicTypeConverter':
        """
        Generates an object for converting atomic numbers to internal types and vice versa from the list of elements.

        Args:
            atomic_types (Optional[List[Union[str, int]]], optional): 
                List of supported atomic numbers/elements. Defaults to None.

        Returns:
            AtomicTypeConverter: Object for converting atomic numbers to internal types and vice versa.
        """
        if atomic_types is None:
            to_atomic_numbers = np.asarray(list(range(119)))
            from_atomic_numbers = to_atomic_numbers
        else:
            to_atomic_numbers = np.asarray(
                [ase.data.atomic_numbers[atomic_type] if isinstance(atomic_type, str) else int(atomic_type) for
                 atomic_type in atomic_types])
            max_entry = np.max(to_atomic_numbers)
            from_atomic_numbers = -np.ones(max_entry + 1, dtype=int)
            from_atomic_numbers[to_atomic_numbers] = np.arange(len(to_atomic_numbers))

        return AtomicTypeConverter(to_atomic_numbers, from_atomic_numbers)


class AtomicStructure:
    """
    Defines atomic structure using atomic numbers (species), atomic positions, and other features.

    Args:
        species (np.ndarray): Atom types, typically represented by atomic numbers or internal atom types.
        atomic_numbers (np.ndarray): Atomic numbers.
        positions (np.ndarray): Atomic positions.
        cell (np.ndarray, optional): Unit cell. Defaults to None.
        pbc (bool, optional): Periodic boundaries. Defaults to None.
        energy (float, optional): Total energy. Defaults to None.
        forces (np.ndarray, optional): Atomic forces. Defaults to None.
        stress (np.ndarray, optional): Stress tensor. Defaults to None.
        total_charge (float, optional): Total charge. Defaults to None.
        partial_charges (np.ndarray, optional): Partial atomic charges. Defaults to None.
        dipole_moment (np.ndarray, optional): Dipole moment. Defaults to None.
        quadrupole_moment (np.ndarray, optional): Quadrupole moment. Defaults to None.
        neighbors (str, optional): Method for computing the neighbor list. Defaults to 'matscipy'.
    """
    def __init__(
        self,
        species: np.ndarray,
        atomic_numbers: np.ndarray,
        positions: np.ndarray,
        cell: Optional[np.ndarray] = None,
        pbc: Optional[bool] = None,
        energy: Optional[float] = None,
        forces: Optional[np.ndarray] = None,
        stress: Optional[np.ndarray] = None,
        total_charge: Optional[float] = None,
        partial_charges: Optional[np.ndarray] = None,
        dipole_moment: Optional[np.ndarray] = None,
        quadrupole_moment: Optional[np.ndarray] = None,
        neighbors: str = 'matscipy'
    ):
        # attributes should not be changed from outside,
        # because this might invalidate the computed edge_index (neighbor list) and shifts
        self.species = species
        self.atomic_numbers = atomic_numbers
        self.positions = positions
        self.cell = cell
        self.pbc = pbc
        
        if energy is not None:
            # NOTE: it was required for the DHA molecule from MD22 (an energy was integer)
            energy = float(energy)
            
        self.energy = energy  # EnergyUnit
        self.forces = forces  # EnergyUnit / DistanceUnit
        self.stress = stress  # EnergyUnit / DistanceUnit ** 3
        
        # compute virials for training
        volume = np.abs(np.linalg.det(cell)) if cell is not None else None  # DistanceUnit ** 3
        self.virials = -1.0 * stress * volume if stress is not None and volume is not None else None  # EnergyUnit
        self.n_atoms = species.shape[0]
        
        self.total_charge = total_charge            # ChargeUnit
        self.partial_charges = partial_charges      # ChargeUnit
        self.dipole_moment = dipole_moment          # ChargeUnit x DistanceUnit
        self.quadrupole_moment = quadrupole_moment  # ChargeUnit x DistanceUnit ** 2
        if self.quadrupole_moment is not None:
            trace = np.trace(self.quadrupole_moment)
            if abs(trace) > 1e-7:
                self.quadrupole_moment = self.quadrupole_moment - trace * np.eye(3) / 3.0

        self.neighbors = neighbors
        if self.neighbors == 'matscipy':
            self.neighbors_fn = get_matscipy_neighbors
        elif self.neighbors == 'vesin':
            self.neighbors_fn = get_vesin_neighbors
        else:
            raise ValueError(
                f"{self.neighbors=} is not implemented yet! Use 'matscipy'."
            )

        self._r_cutoff = None
        self._skin = None
        self._edge_index = None
        self._shifts = None

        # check shapes
        assert tuple(species.shape) == (self.n_atoms, )
        assert tuple(atomic_numbers.shape) == (self.n_atoms, )
        assert tuple(positions.shape) == (self.n_atoms, 3)
        assert cell is None or tuple(cell.shape) == (3, 3)
        assert energy is None or isinstance(energy, float)
        assert forces is None or tuple(forces.shape) == (self.n_atoms, 3)
        assert stress is None or tuple(stress.shape) == (3, 3)
        assert total_charge is None or isinstance(total_charge, float)
        assert partial_charges is None or tuple(partial_charges.shape) == (self.n_atoms, )
        assert dipole_moment is None or tuple(dipole_moment.shape) == (3, )
        assert quadrupole_moment is None or tuple(quadrupole_moment.shape) == (3, 3)

    def _compute_neighbors(
        self,
        r_cutoff: float,
        skin: float = 0.0
    ):
        """
        Computes neighbor list for the atomic structure.

        Args:
            r_cutoff (float): Cutoff radius for computing the neighbor list.
            skin (float, optional): 
                Skin distance for updating the neighbor list. Defaults to 0.0.
        """
        if (self._r_cutoff is not None and self._r_cutoff == r_cutoff) and \
                (self._skin is not None and self._skin == skin):
            return  # neighbors have already been computed for the same cutoff and skin radius
        self._r_cutoff = r_cutoff
        self._skin = skin

        self._edge_index, self._shifts = self.neighbors_fn(r_cutoff=r_cutoff, skin=skin, **vars(self))

        assert self._edge_index.shape[0] == 2 and len(self._edge_index.shape) == 2
        assert self._shifts.shape[1] == 3 and len(self._shifts.shape) == 2
    
    def _compute_primitive_neighbors(
        self,
        r_cutoff: Optional[float] = None,
        **kwargs: Any
    ):
        """
        Computes primitive all-to-all neighbor pairs.
        
        Note: After calling this method, `self._r_cutoff` and `self._skin` remain `None`, 
              which is the expected and desired behavior.
        """
        if self._edge_index is not None and self._shifts is not None:
            return
        
        if self.cell is not None and not np.all(self.cell == 0):
            raise ValueError(f"Cutoff radius cannot be `None` for a periodic structure. "
                             f"Provided {r_cutoff=} and {self.cell=}.")
        self._edge_index, self._shifts = primitive_pairs(self.positions)

    def get_edge_index(
        self,
        r_cutoff: Optional[float] = None,
        skin: float = 0.0
    ) -> np.ndarray:
        """
        Computes edge indices.

        Args:
            r_cutoff (float, optional): 
                Cutoff radius for computing the neighbor list. Defaults to None.
            skin (float, optional): 
                Skin distance for updating the neighbor list. Defaults to 0.0.

        Returns:
            np.ndarray: Edge indices (neighbor list) containing the central (out[0, :]) 
                        and neighboring (out[1, :]) atoms.
        """
        if r_cutoff is None:
            self._compute_primitive_neighbors(r_cutoff)
        else:
            self._compute_neighbors(r_cutoff, skin)
        return self._edge_index

    def get_shifts(
        self,
        r_cutoff: Optional[float] = None,
        skin: float = 0.0
    ) -> np.ndarray:
        """
        Computes shift vectors.

        Args:
            r_cutoff (float, optional): 
                Cutoff radius for computing the neighbor list. Defaults to None.
            skin (float, optional): 
                Skin distance for updating the neighbor list. Defaults to 0.0.

        Returns:
            np.ndarray: Shift vector, i.e., the number of cell boundaries crossed 
                        by the bond between atoms.
        """
        if r_cutoff is None:
            self._compute_primitive_neighbors(r_cutoff)
        else:
            self._compute_neighbors(r_cutoff, skin)
        return self._shifts

    def to_type_names(
        self,
        converter: AtomicTypeConverter,
        check: bool = False
    ) -> 'AtomicStructure':
        """
        Convert atomic numbers to internal types in the atomic structure.

        Args:
            converter (AtomicTypeConverter): 
                Object for converting atomic numbers to internal types and vice versa.
            check (bool, optional): 
                If True, check if atomic numbers are supported by `AtomicTypeConverter`. 
                Defaults to False.

        Returns:
            AtomicStructure: Atomic structure with internal types instead of atomic numbers.
        """
        return AtomicStructure(
            species=converter.to_type_names(self.species, check=check),
            atomic_numbers=self.atomic_numbers,
            positions=self.positions,
            cell=self.cell,
            pbc=self.pbc,
            forces=self.forces,
            energy=self.energy,
            stress=self.stress,
            total_charge=self.total_charge,
            partial_charges=self.partial_charges,
            dipole_moment=self.dipole_moment,
            quadrupole_moment=self.quadrupole_moment,
            neighbors=self.neighbors
        )

    def to_atomic_numbers(self, converter: AtomicTypeConverter) -> 'AtomicStructure':
        """
        Convert internal types to atomic numbers in the atomic structure.

        Args:
            converter (AtomicTypeConverter): 
                Object for converting atomic numbers to internal types and vice versa.

        Returns:
            AtomicStructure: Atomic structure with atomic numbers instead of internal types.
        """
        return AtomicStructure(
            species=converter.to_atomic_numbers(self.species),
            atomic_numbers=self.atomic_numbers,
            positions=self.positions,
            cell=self.cell,
            pbc=self.pbc,
            forces=self.forces,
            energy=self.energy,
            stress=self.stress,
            total_charge=self.total_charge,
            partial_charges=self.partial_charges,
            dipole_moment=self.dipole_moment,
            quadrupole_moment=self.quadrupole_moment,
            neighbors=self.neighbors
        )

    def to_atoms(self) -> ase.Atoms:
        """
        Converts the atomic structure to `ase.Atoms`.

        Returns:
            ase.Atoms: The `ase.Atoms` object.
        """
        atoms = ase.Atoms(positions=self.positions, numbers=self.atomic_numbers, cell=self.cell, pbc=self.pbc)
        
        if self.energy is not None:
            atoms.info['REF_energy'] = self.energy
        if self.forces is not None:
            atoms.arrays['REF_forces'] = self.forces
        if self.stress is not None:
            atoms.info['REF_stress'] = self.stress
        if self.total_charge is not None:
            atoms.info['REF_total_charge'] = self.total_charge
        if self.partial_charges is not None:
            atoms.arrays['REF_partial_charges'] = self.partial_charges
        if self.dipole_moment is not None:
            atoms.info['REF_dipole_moment'] = self.dipole_moment
        if self.quadrupole_moment is not None:
            atoms.info['REF_quadrupole_moment'] = self.quadrupole_moment
        
        return atoms

    @staticmethod
    def from_atoms(
        atoms: ase.Atoms,
        neighbors: str = 'matscipy',
        **kwargs: Any
    ) -> 'AtomicStructure':
        """
        Converts `ase.Atoms` to `AtomicStructure`.

        Args:
            atoms (ase.Atoms): The `ase.Atoms` object.
            neighbors (str, optional): 
                Method for computing the neighbor list. Defaults to 'matscipy'.

        Returns:
            AtomicStructure: 
                The `AtomicStructure` object which allows for convenient calculation of the 
                neighbor list and transformations between atomic numbers and internal types.
        """
        return AtomicStructure(
            species=atoms.get_atomic_numbers(),
            atomic_numbers=atoms.get_atomic_numbers(),
            positions=atoms.get_positions(),
            cell=np.asarray(atoms.get_cell()),
            pbc=atoms.get_pbc(),
            energy=atoms.info.get('REF_energy', None),
            forces=atoms.arrays.get('REF_forces', None),
            stress=atoms.info.get('REF_stress', None),
            total_charge=atoms.info.get('REF_total_charge', None),
            partial_charges=atoms.arrays.get('REF_partial_charges', None),
            dipole_moment=atoms.info.get('REF_dipole_moment', None),
            quadrupole_moment=atoms.info.get('REF_quadrupole_moment', None),
            neighbors=neighbors
        )

    def restore_neighbors_from_last(
        self,
        r_cutoff: float,
        structure: Optional['AtomicStructure'] = None,
        skin: float = 0.
    ) -> bool:
        """
        Restores the neighbor list from the last atomic structure. Used together with the skin distance 
        to identify when neighbors have to be re-computed.

        Args:
            r_cutoff (float): Cutoff radius for computing the neighbor list.
            structure (Optional[AtomicStructure], optional): 
                The `AtomicStructure` object from which neighbors are re-used if possible. Defaults to None.
            skin (float, optional): Skin distance for updating the neighbor list. Defaults to 0.0.

        Returns:
            bool: True, if neighbors of the last atomic structure can be re-used.
        """
        if structure is None or skin <= 0.:
            # no reference structure has been provided or skin <= 0. has been provided
            return False

        if structure._r_cutoff is None:
            raise ValueError(f'This method cannot be used if no cutoff radius has been defined before. '
                             f'Provided {structure._r_cutoff=}.')
        
        if r_cutoff != structure._r_cutoff or skin != structure._skin or np.any(self.pbc != structure.pbc) \
                or np.any(self.cell != structure.cell):
            # cutoff radius, skin radius, periodic boundaries, or periodic cell have been changed
            return False

        max_dist_sq = ((self.positions - structure.positions) ** 2).sum(-1).max()
        if max_dist_sq > (skin / 2.0) ** 2:
            # atoms moved out of the skin (r_cutoff += skin)
            return False

        # structure has not been changed considerably such that we may restore neighbors from last structure
        self._r_cutoff = structure._r_cutoff
        self._skin = structure._skin
        self._edge_index = structure._edge_index
        self._shifts = structure._shifts

        return True
    
    def wrap(self) -> 'AtomicStructure':
        """
        Wrap atomic positions inside the unit cell based on periodic boundary conditions.
            
        Returns:
            AtomicStructure: Atomic structure with wrapped atomic positions.
        """
        pbc = np.asarray(self.pbc, dtype=bool)
        
        if not np.all(pbc):
            raise ValueError(f'Cannot wrap positions: periodic boundary conditions are {self.pbc=}.')
        positions_frac = scipy.linalg.solve(self.cell.T, self.positions.T).T
        positions_frac = positions_frac - np.floor(positions_frac)
        positions = np.dot(positions_frac, self.cell)
        return AtomicStructure(
            species=self.species,
            atomic_numbers=self.atomic_numbers,
            positions=positions,
            cell=self.cell,
            pbc=self.pbc,
            forces=self.forces,
            energy=self.energy,
            stress=self.stress,
            total_charge=self.total_charge,
            partial_charges=self.partial_charges,
            dipole_moment=self.dipole_moment,
            quadrupole_moment=self.quadrupole_moment,
            neighbors=self.neighbors
        )


def write_value(value):    
    return value if value is not None else "None"


class AtomicStructures:
    """
    Atomic structures to deal with a list of `AtomicStructure` objects (atomic structures).

    Args:
        structures (List[AtomicStructure]): List of `AtomicStructure` objects.
    """
    def __init__(self, structures: List[AtomicStructure]):
        self.structures = structures

    def __len__(self) -> int:
        """
        Provides the total number of atomic structures in the list.

        Returns:
            int: Total number of atomic structures.
        """
        return len(self.structures)

    def save_extxyz(self, file_path: Union[Path, str]):
        """
        Saves atomic structures to an `.extxyz` file.

        Args:
            file_path (Union[Path, str]): Path to the `.extxyz` file.
        """
        if not str(file_path)[-7:] == '.extxyz':
            raise ValueError(f'{file_path} has been provided, while an .extxyz file is expected.')

        for structure in self.structures:
            atoms = structure.to_atoms()
            write(file_path, atoms, format='extxyz', append=True)
            
    def save_hdf5(
        self, 
        file_path: Union[Path, str],
        include_edges: bool = False,
        r_cutoff: Optional[float] = None,
        skin: float = 0.0
    ):
        """
        Saves atomic structures to an `.hdf5` file.

        Args:
            file_path (Union[Path, str]): Path to the `.hdf5` file.
            include_edges (bool): Whether to save edge indices and shifts. Defaults to False.
            r_cutoff (Optional[float]): 
                Cutoff radius. If None, an all-to-all neighbor list is used. Defaults to None.
            skin (float, optional): Skin distance. Defaults to 0.0.
        """
        if not str(file_path)[-5:] == '.hdf5':
            raise ValueError(f'{file_path} has been provided, while an .hdf5 file is expected.')
        
        with h5py.File(file_path, 'w') as f:
            for idx, structure in enumerate(self.structures):
                grp = f.create_group(f'structure_{idx}')
                grp['atomic_numbers'] = write_value(structure.atomic_numbers)
                grp['positions'] = write_value(structure.positions)
                grp['cell'] = write_value(structure.cell)
                grp['pbc'] = write_value(structure.pbc)
                grp['energy'] = write_value(structure.energy)
                grp['forces'] = write_value(structure.forces)
                grp['stress'] = write_value(structure.stress)
                grp['total_charge'] = write_value(structure.total_charge)
                grp['partial_charges'] = write_value(structure.partial_charges)
                grp['dipole_moment'] = write_value(structure.dipole_moment)
                grp['quadrupole_moment'] = write_value(structure.quadrupole_moment)
                
                if include_edges:
                    grp['r_cutoff'] = write_value(r_cutoff)
                    grp['skin'] = write_value(skin)
                    grp['edge_index'] = write_value(structure.get_edge_index(r_cutoff, skin))
                    grp['shifts'] = write_value(structure.get_shifts(r_cutoff, skin))

    def save_lmdb(
        self,
        file_path: Union[Path, str],
        include_edges: bool = False,
        r_cutoff: Optional[float] = None,
        skin: float = 0.0,
        map_size: int = 1024*1024,  # Start with 1 MB
    ):
        """
        Saves atomic structures to an LMDB database.

        Args:
            file_path (Union[Path, str]): Path to the LMDB file.
            include_edges (bool): Whether to save edge indices and shifts. Defaults to False.
            r_cutoff (Optional[float]): 
                Cutoff radius. If None, an all-to-all neighbor list is used. Defaults to None.
            skin (float, optional): Skin distance. Defaults to 0.0.
            map_size (int, optional): Maximum size database may grow to.
        """
        if not str(file_path).endswith('.lmdb'):
            raise ValueError(f"{file_path} has been provided, while an .lmdb file is expected.")
        
        env = lmdb.open(
            str(file_path),
            map_size=map_size,
            subdir=False,
            meminit=False,
            map_async=True
        )
        
        txn = env.begin(write=True)
        
        for idx, structure in enumerate(self.structures):
            key = f'structure_{idx}'.encode('ascii')

            data = {
                'atomic_numbers': structure.atomic_numbers,
                'positions': structure.positions,
                'cell': structure.cell,
                'pbc': structure.pbc,
                'energy': structure.energy,
                'forces': structure.forces,
                'stress': structure.stress,
                'total_charge': structure.total_charge,
                'partial_charges': structure.partial_charges,
                'dipole_moment': structure.dipole_moment,
                'quadrupole_moment': structure.quadrupole_moment
            }

            if include_edges:
                data.update({
                    'r_cutoff': r_cutoff,
                    'skin': skin,
                    'edge_index': structure.get_edge_index(r_cutoff, skin),
                    'shifts': structure.get_shifts(r_cutoff, skin)
                })

            serialized_data = pickle.dumps(data)
            
            try:
                txn.put(key, serialized_data)
            except lmdb.MapFullError:
                txn.abort()
                current_size = env.info()["map_size"]
                new_size = current_size * 2
                env.set_mapsize(new_size)

                txn = env.begin(write=True)
                txn.put(key, serialized_data)

            txn.commit()
            txn = env.begin(write=True)
        
        txn.commit()
        env.sync()
        
        env.close()
    
    def save_fs(
        self,
        root: Union[Path, str],
        include_edges: bool = False,
        r_cutoff: Optional[float] = None,
        skin: float = 0.0
    ):
        """
        Saves atomic structures to a file system.

        Args:
            root (Union[Path, str]): Root directory.
            include_edges (bool): Whether to save edge indices and shifts. Defaults to False.
            r_cutoff (Optional[float]): 
                Cutoff radius. If None, an all-to-all neighbor list is used. Defaults to None.
            skin (float, optional): Skin distance. Defaults to 0.0.
        """
        
        for idx, structure in enumerate(self.structures):
            key = f'structure_{idx}'

            data = {
                'atomic_numbers': structure.atomic_numbers,
                'positions': structure.positions,
                'cell': structure.cell,
                'pbc': structure.pbc,
                'energy': structure.energy,
                'forces': structure.forces,
                'stress': structure.stress,
                'total_charge': structure.total_charge,
                'partial_charges': structure.partial_charges,
                'dipole_moment': structure.dipole_moment,
                'quadrupole_moment': structure.quadrupole_moment
            }

            if include_edges:
                data.update({
                    'r_cutoff': r_cutoff,
                    'skin': skin,
                    'edge_index': structure.get_edge_index(r_cutoff, skin),
                    'shifts': structure.get_shifts(r_cutoff, skin)
                })
            
            os.makedirs(root + '/' + key, exist_ok=True)
            with open(root + '/' + key + '/data.pkl', 'wb') as file:
                pickle.dump(data, file)
    
    @staticmethod
    def from_extxyz(
        file_path: Union[Path, str],
        range_str: str = ':',
        neighbors: str = 'matscipy',
        **kwargs: Any
    ) -> 'AtomicStructures':
        """
        Loads atomic structures from an `.xyz` or `.extxyz` file.

        Args:
            file_path (Union[Path, str]): Path to the `.xyz` or `.extxyz` file.
            range_str (str): 
                Range of the atomic structures, i.e. ':10' to chose the first ten atomic structures.
            neighbors (str, optional): 
                Method for computing the neighbor list. Defaults to 'matscipy'.

        Returns:
            AtomicStructures: The `AtomicStructures` object.
        """
        if not str(file_path)[-3:] == 'xyz':
            raise ValueError(f'{file_path} has been provided, while an .extxyz file is expected.')

        traj = read(file_path, format='extxyz', index=range_str)

        structures = []
        for atoms in traj:
            structures.append(AtomicStructure.from_atoms(atoms, neighbors=neighbors, **kwargs))

        return AtomicStructures(structures)

    @staticmethod
    def from_traj(
        traj: List[ase.Atoms],
        neighbors: str = 'matscipy',
        **kwargs: Any
    ) -> 'AtomicStructures':
        """
        Loads atomic structures from a list of `ase.Atoms`.

        Args:
            traj (List[ase.Atoms]): List of `ase.Atoms`.
            neighbors (str, optional): Method for computing the neighbor list. Defaults to 'matscipy'.

        Returns:
            AtomicStructures: The `AtomicStructures` object.
        """
        return AtomicStructures([AtomicStructure.from_atoms(a, neighbors=neighbors, **kwargs) for a in traj])

    @staticmethod
    def from_file(file_path: Union[Path, str], **config: Any) -> 'AtomicStructures':
        """
        Loads atomic structures from a file.

        Args:
            file_path (Union[Path, str]): Path to the `.xyz` or `.extxyz` file.

        Returns:
            AtomicStructures: The `AtomicStructures` object.
        """
        if str(file_path)[-3:] == 'xyz':
            return AtomicStructures.from_extxyz(file_path, **config)
        else:
            raise ValueError(f'Provided wrong data format for {file_path=}. Use ".xyz" or ".extxyz" instead!')

    def to_type_names(
        self,
        converter: AtomicTypeConverter,
        check: bool = False
    ) -> 'AtomicStructures':
        """
        Converts atomic numbers to internal types for all atomic structures in the list.

        Args:
            converter (AtomicTypeConverter): 
                Object for converting atomic numbers to internal types and vice versa.
            check (bool, optional): 
                If True, check if atomic numbers are supported. Defaults to False.

        Returns:
            AtomicStructures: 
                The `AtomicStructures` object with internal types instead of atomic numbers.
        """
        return AtomicStructures([s.to_type_names(converter, check=check) for s in self.structures])

    def to_atomic_numbers(self, converter: AtomicTypeConverter) -> 'AtomicStructures':
        """
        Converts internal types to atomic numbers for all atomic structures in the list.

        Args:
            converter (AtomicTypeConverter): 
                Object for converting atomic numbers to internal types and vice versa.

        Returns:
            AtomicStructures: 
                The `AtomicStructures` object with atomic numbers instead of internal types.
        """
        return AtomicStructures([s.to_atomic_numbers(converter) for s in self.structures])

    def to_data(
        self, 
        r_cutoff: Optional[float] = None,
        n_species: Optional[int] = None
    ) -> List['AtomicData']:
        """
        Converts `AtomicStructures` to a list of `AtomicData` used by implemented models and algorithms.
        `AtomicData` handles atomic structures as graphs.

        Args:
            r_cutoff (float, optional): 
                Cutoff radius for computing neighbor lists. Defaults to None.
            n_species (int, optional): 
                Number of species (used to compute one-hot encoding). Defaults to None.

        Returns:
            List[AtomicData]: List of `AtomicData`, handling atomic structures as graphs.
        """
        return [
            AtomicData(s, r_cutoff=r_cutoff, n_species=n_species) for s in self.structures
        ]

    def random_split(
        self,
        sizes: Dict[str, int],
        seed: int = None
    ) -> Dict[str, 'AtomicStructures']:
        """
        Splits atomic structures using a random seed.
        
        Args:
            sizes (Dict[str, int]): Dictionary containing names and sizes of data splits.
            seed (int): Random seed. Defaults to None.

        Returns:
            Dict[str, AtomicStructures]: Dictionary of `AtomicStructures` splits.
        """
        random_state = np.random.RandomState(seed=seed)
        idx = random_state.permutation(np.arange(len(self.structures)))
        sub_idxs = {}
        for key, val in sizes.items():
            sub_idxs.update({key: idx[0:val]})
            idx = idx[val:]
        if len(idx) > 0:
            sub_idxs.update({"test": idx})
        return {name: self[si] for name, si in sub_idxs.items()}

    def split_by_indices(self, idxs: List[int]) -> Tuple[Union['AtomicStructures', AtomicStructure], Union['AtomicStructures', AtomicStructure]]:
        """
        Splits atomic structures using provided indices.
        
        Args:
            idxs (List[int]): Indices with which atomic structures are split.

        Returns:
            Tuple: Atomic structures defined by `idxs`, and those which remain.
        """
        remaining_idxs = list(set(range(len(self.structures))).difference(set(idxs)))
        remaining_idxs.sort()
        return self[idxs], self[remaining_idxs]

    def __getitem__(self, idxs: int) -> 'AtomicStructures':
        """
        Provides atomic structures defined by indices or slices.

        Args:
            idxs (int): Indices or slice to extract a portion from atomic structures.

        Returns:
            AtomicStructures: The `AtomicStructures` object.
        """
        if isinstance(idxs, (int, np.integer)):
            return self.structures[idxs]
        elif isinstance(idxs, slice):
            return AtomicStructures(self.structures[idxs])
        else:
            # assume idxs is array_like
            return AtomicStructures([self.structures[i] for i in idxs])

    def __add__(self, other: 'AtomicStructures') -> 'AtomicStructures':
        """
        Combines atomic structures to a single `AtomicStructures` object.

        Args:
            other (AtomicStructures): Atomic structures to be added to `self`.

        Returns:
            AtomicStructures: The combined `AtomicStructures` object.
        """
        return AtomicStructures(self.structures + other.structures)


def to_one_hot(species: np.ndarray, n_species: int) -> torch.Tensor:
    """
    Prepares one-hot encoding for atomic species/internal types. 
    
    Adapted from MACE (https://github.com/ACEsuit/mace/blob/main/mace/tools/torch_tools.py).

    Args:
        species (np.ndarray): Array containing atomic species (atomic numbers or internal types).
        n_species (int): Total number of species (number of classes).

    Returns:
        torch.Tensor: One-hot encoded atomic species.
    """
    assert len(species.shape) == 1
    # shape: n_atoms x 1
    species = torch.tensor(species, dtype=torch.long).unsqueeze(-1)
    
    shape = species.shape[:-1] + (n_species,)
    oh = torch.zeros(shape, device=species.device).view(shape)

    # scatter_ is the in-place version of scatter
    oh.scatter_(dim=-1, index=species, value=1)

    # shape: n_atoms x n_species
    return oh.view(*shape)


class AtomicData(Data):
    """
    Converts atomic structures to graphs.

    Args:
        structure (AtomicStructure): The `AtomicStructure` object.
        r_cutoff (float, optional): 
            Cutoff radius for computing the neighbor list. Defaults to None.
        skin (float, optional): 
            Skin distance for updating neighbor list, if necessary. Defaults to 0.0.
        n_species (int, optional): 
            Number of species (required to compute one-hot encoding). Defaults to None.
    """
    def __init__(
        self,
        structure: AtomicStructure,
        r_cutoff: Optional[float] = None,
        skin: float = 0.0,
        n_species: Optional[int] = None
    ):
        # default tensors for missing attributes/properties
        default_dtype = torch.get_default_dtype()
        default_cell = torch.zeros(3, 3, dtype=default_dtype)
        default_energy = torch.full((1,), float('nan'), dtype=default_dtype) # also total charge
        default_forces = torch.full((structure.n_atoms, 3), float('nan'), dtype=default_dtype)
        default_stress = torch.full((3, 3), float('nan'), dtype=default_dtype) # also virials and quadrupole moment
        default_partial_charges = torch.full((structure.n_atoms, ), float('nan'), dtype=default_dtype)
        default_dipole = torch.full((3, ), float('nan'), dtype=default_dtype)
                
        # aggregate data
        data = {
            'num_nodes': to_tensor(structure.n_atoms, dtype=torch.long),
            # duplicate, but num_nodes is not directly provided in the batch
            'n_atoms': to_tensor(structure.n_atoms, dtype=torch.long),
            'node_attrs': to_one_hot(structure.species, n_species) if n_species is not None else None,
            'species': to_tensor(structure.species, dtype=torch.long),
            'atomic_numbers': to_tensor(structure.atomic_numbers, dtype=torch.long),
            'positions': to_tensor(structure.positions),
            'edge_index': to_tensor(structure.get_edge_index(r_cutoff, skin), dtype=torch.long),
            'shifts': to_tensor(structure.get_shifts(r_cutoff, skin)),
            'cell': to_tensor(structure.cell, default=default_cell).unsqueeze(0),
            'energy': to_tensor(structure.energy, default=default_energy),
            'forces': to_tensor(structure.forces, default=default_forces),
            'stress': to_tensor(structure.stress, default=default_stress).unsqueeze(0),
            'virials': to_tensor(structure.virials, default=default_stress).unsqueeze(0),
            'total_charge': to_tensor(structure.total_charge, default=default_energy),
            'partial_charges': to_tensor(structure.partial_charges, default=default_partial_charges),
            'dipole_moment': to_tensor(structure.dipole_moment, default=default_dipole).unsqueeze(0),
            'quadrupole_moment': to_tensor(structure.quadrupole_moment, default=default_stress).unsqueeze(0),
            # strain is required to compute stress
            'strain': to_tensor(np.zeros_like(structure.cell)).unsqueeze(0) if structure.cell is not None else None,
        }
        super().__init__(**data)
        
    def to_atoms(self):
        """
        Converts the atomic data to `ase.Atoms`.

        Returns:
            ase.Atoms: The `ase.Atoms` object.
        """
        atoms = ase.Atoms(
            positions=self.positions.detach().cpu().numpy(),
            numbers=self.atomic_numbers.detach().cpu().numpy(),
        )
        
        if not torch.all(self.cell.squeeze() == 0):
            atoms.set_cell(self.cell.squeeze().detach().cpu().numpy())
            atoms.set_pbc(True)
            
        if torch.all(~torch.isnan(self.energy)):
            atoms.info['REF_energy'] = self.energy.detach().cpu().item()
        if torch.all(~torch.isnan(self.forces)):
            atoms.arrays['REF_forces'] = self.forces.detach().cpu().numpy()
        if torch.all(~torch.isnan(self.stress)):
            atoms.info['REF_stress'] = self.stress.squeeze().detach().cpu().numpy()
        if torch.all(~torch.isnan(self.total_charge)):
            atoms.info['REF_total_charge'] = self.total_charge.detach().cpu().item()
        if torch.all(~torch.isnan(self.partial_charges)):
            atoms.arrays['REF_partial_charges'] = self.partial_charges.detach().cpu().numpy()
        if torch.all(~torch.isnan(self.dipole_moment)):
            atoms.info['REF_dipole_moment'] = self.dipole_moment.squeeze().detach().cpu().numpy()
        if torch.all(~torch.isnan(self.quadrupole_moment)):
            atoms.info['REF_quadrupole_moment'] = self.quadrupole_moment.squeeze().detach().cpu().numpy()
    
        return atoms
