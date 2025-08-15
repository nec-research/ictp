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
import os

from typing import *

from pathlib import Path

import h5py

import lmdb
import pickle

from torch.utils.data import Dataset, Subset

from ictp.data.data import *


def unpack_value(value: Any) -> Any:
    """
    Handles the unpacking of values from bytes or strings.

    Args:
        value (Any): Value to unpack.

    Returns:
        Any: Unpacked value, or None if the value equals "None".
    """
    value = value.decode("utf-8") if isinstance(value, bytes) else value
    return None if str(value) == "None" else value


class XYZDataset(Dataset):
    """
    Dataset for loading data from an .extxyz file.

    Args:
        file_path (Union[Path, str]): Path to the XYZ or EXTXYZ file.
        r_cutoff (Optional[float]): Cutoff radius for neighbor calculations. If None, all-to-all 
                                    neigbors are computed. Defaults to None.
        skin (float, optionak). Skin distance. Defaults to 0.0.
        atomic_types (Optional[List[Union[str, int]]]): List of atomic types to convert.
        neighbors (str): Neighbor calculation method. Defaults to 'matscipy'.
    """
    def __init__(
        self, 
        file_path: Union[Path, str],
        r_cutoff: Optional[float] = None,
        skin: float = 0.0,
        atomic_types: Optional[List[Union[str, int]]] = None,
        neighbors: str = 'matscipy',
        **config: Any
    ):
        self.file_path = Path(file_path)
        
        if not self.file_path.exists():
                raise FileNotFoundError(f"(EXT)XYZ file not found at: {self.file_path}")
            
        # load structures
        self.structures = AtomicStructures.from_extxyz(
            self.file_path, 
            range_str=':', 
            neighbors=neighbors,
            **config
        )
        self.length = len(self.structures)
        
        # type conversion
        self.atomic_type_converter = AtomicTypeConverter.from_type_list(atomic_types)
        self.structures = self.structures.to_type_names(self.atomic_type_converter, check=True)
        
        self.r_cutoff = r_cutoff
        self.skin = skin
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, index: int) -> AtomicData:
        try:
            structure = self.structures[index]
        except IndexError:
            raise IndexError(f'Index {index} out of range for the total number of structures {len(self.structures)}!')
        
        # transform into atomic data
        atomic_data = AtomicData(
            structure=structure,
            r_cutoff=self.r_cutoff,
            skin=self.skin,
            n_species=self.atomic_type_converter.get_n_type_names()
        )
        
        return atomic_data


class HDF5Dataset(Dataset):
    """
    Dataset for loading data from an HDF5 file, with optional neighbor computation
    and atomic type conversion.
    
    Args:
        file_path (Union[Path, str]): Path to the HDF5 file.
        r_cutoff (Optional[float]): Cutoff radius for neighbor calculations. If None, all-to-all 
                                    neigbors are computed. Defaults to None.
        skin (float, optionak). Skin distance. Defaults to 0.0.
        atomic_types (Optional[List[Union[str, int]]]): List of atomic types to convert.
        neighbors (str): Neighbor calculation method. Defaults to 'matscipy'.
    """
    def __init__(
        self, 
        file_path: Union[Path, str],
        r_cutoff: Optional[float] = None,
        skin: float = 0.0,
        atomic_types: Optional[List[Union[str, int]]] = None,
        neighbors: str = 'matscipy',
        **config: Any
    ):
        self.file_path = Path(file_path)
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"HDF5 file not found at: {self.file_path}")
        
        self.r_cutoff = r_cutoff
        self.skin = skin
        self.atomic_type_converter = AtomicTypeConverter.from_type_list(atomic_types)
        self.neighbors = neighbors
        self._file = None
        self._length = None
    
    @property
    def file(self) -> h5py.File:
        """
        Lazily opens the HDF5 file and caches the file handle.
        
        Returns:
            h5py.File: Opened HDF5 file object.
        """
        if self._file is None:
            self._file = h5py.File(self.file_path, "r")
        return self._file

    def __getstate__(self) -> dict:
        """
        Custom pickle state to exclude the HDF5 file handle, as it cannot be pickled.
        
        Returns:
            dict: Object state without the '_file' attribute.
        """
        state = self.__dict__.copy()
        state["_file"] = None  # exclude the file handle from the pickled state
        return state

    def __len__(self):
        if self._length is None:
            self._length = len(self.file.keys())
        return self._length
    
    def __del__(self):
        if self._file is not None:
            self._file.close()
    
    def __getitem__(self, index: int) -> AtomicData:
        try:
            grp = self.file[f'structure_{index}']
        except KeyError:
            raise IndexError(f'Index {index} not found in the HDF5 file.')
        
        structure = AtomicStructure(
            species=grp['atomic_numbers'][()],
            atomic_numbers=grp['atomic_numbers'][()],
            positions=grp['positions'][()],
            cell=unpack_value(grp['cell'][()]),
            pbc=unpack_value(grp['pbc'][()]),
            energy=unpack_value(grp['energy'][()]),
            forces=unpack_value(grp['forces'][()]),
            stress=unpack_value(grp['stress'][()]),
            total_charge=unpack_value(grp['total_charge'][()]),
            partial_charges=unpack_value(grp['partial_charges'][()]),
            dipole_moment=unpack_value(grp['dipole_moment'][()]),
            quadrupole_moment=unpack_value(grp['quadrupole_moment'][()]),
            neighbors=self.neighbors,   # use the specified neighbor computation method
        )
        
        structure = structure.to_type_names(self.atomic_type_converter, check=True)
        
        if 'r_cutoff' in grp and 'skin' in grp:
            r_cutoff = unpack_value(grp['r_cutoff'][()])
            skin = unpack_value(grp['skin'][()])
            if r_cutoff == self.r_cutoff and skin == self.skin:
                if 'edge_index' in grp and 'shifts' in grp:
                    structure._r_cutoff = self.r_cutoff
                    structure._skin = self.skin
                    structure._edge_index = grp['edge_index'][()]
                    structure._shifts = grp['shifts'][()]
                    
        atomic_data = AtomicData(
            structure=structure,
            r_cutoff=self.r_cutoff,
            skin=self.skin,
            n_species=self.atomic_type_converter.get_n_type_names()
        )
        
        return atomic_data


class LMDBDataset(Dataset):
    """
    Dataset for loading data from an LMDB file, with optional neighbor computation
    and atomic type conversion.
    
    Args:
        file_path (Union[Path, str]): Path to the LMDB file.
        r_cutoff (Optional[float]): Cutoff radius for neighbor calculations. Defaults to None.
        skin (float, optional): Skin distance. Defaults to 0.0.
        atomic_types (Optional[List[Union[str, int]]]): List of atomic types to convert.
        neighbors (str): Neighbor calculation method. Defaults to 'matscipy'.
    """
    def __init__(
        self,
        file_path: Union[Path, str],
        r_cutoff: Optional[float] = None,
        skin: float = 0.0,
        atomic_types: Optional[List[Union[str, int]]] = None,
        neighbors: str = 'matscipy',
        **config: Any
    ):
        self.file_path = Path(file_path)
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"LMDB file not found at: {self.file_path}")
        
        self.r_cutoff = r_cutoff
        self.skin = skin
        self.atomic_type_converter = AtomicTypeConverter.from_type_list(atomic_types)
        self.neighbors = neighbors
        self._env = None
        self._length = None

    @property
    def env(self) -> lmdb.Environment:
        """
        Lazily opens the LMDB environment and caches the file handle.
        
        Returns:
            lmdb.Environment: Opened LMDB environment object.
        """
        if self._env is None:
            self._env = lmdb.open(str(self.file_path), subdir=False, readonly=True, lock=False, map_async=True)
        return self._env

    def __getstate__(self) -> dict:
        """
        Custom pickle state to exclude the LMDB environment handle, as it cannot be pickled.
        
        Returns:
            dict: Object state without the '_env' attribute.
        """
        state = self.__dict__.copy()
        state["_env"] = None  # exclude the environment handle from the pickled state
        return state

    def __len__(self):
        if self._length is None:
            with self.env.begin() as txn:
                self._length = sum(1 for _ in txn.cursor())
        return self._length
    
    def __del__(self):
        """
        Cleanup by closing the LMDB environment when the object is destroyed.
        """
        if self._env is not None:
            self._env.close()

    def __getitem__(self, index: int) -> AtomicData:
        """
        Retrieve the atomic data for the specified index from the LMDB database.
        
        Args:
            index (int): Index of the structure to retrieve.

        Returns:
            AtomicData: The atomic data for the specified structure.
        """
        with self.env.begin() as txn:
            key = f'structure_{index}'.encode('ascii')
            try:
                data = pickle.loads(txn.get(key))
            except KeyError:
                raise IndexError(f'Index {index} not found in the LMDB file.')

        structure = AtomicStructure(
            species=data['atomic_numbers'],
            atomic_numbers=data['atomic_numbers'],
            positions=data['positions'],
            cell=data['cell'],
            pbc=data['pbc'],
            energy=data['energy'],
            forces=data['forces'],
            stress=data['stress'],
            total_charge=data['total_charge'],
            partial_charges=data['partial_charges'],
            dipole_moment=data['dipole_moment'],
            quadrupole_moment=data['quadrupole_moment'],
            neighbors=self.neighbors,
        )

        structure = structure.to_type_names(self.atomic_type_converter, check=True)

        if 'r_cutoff' in data and 'skin' in data:
            r_cutoff = data['r_cutoff']
            skin = data['skin']
            if r_cutoff == self.r_cutoff and skin == self.skin:
                if 'edge_index' in data and 'shifts' in data:
                    structure._r_cutoff = self.r_cutoff
                    structure._skin = self.skin
                    structure._edge_index = data['edge_index']
                    structure._shifts = data['shifts']
        
        atomic_data = AtomicData(
            structure=structure,
            r_cutoff=self.r_cutoff,
            skin=self.skin,
            n_species=self.atomic_type_converter.get_n_type_names()
        )
        
        return atomic_data


def load(fname):     
    with open(fname, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict


class FSDataset(Dataset):
    def __init__(
        self,
        root: Union[Path, str],
        r_cutoff: Optional[float] = None,
        skin: float = 0.0,
        atomic_types: Optional[List[Union[str, int]]] = None,
        neighbors: str = 'matscipy',
        **config: Any
    ):
        self.root = Path(root)
        
        if not self.root.exists():
            raise FileNotFoundError(f"Root not found: {self.root}")
        
        self.r_cutoff = r_cutoff
        self.skin = skin
        self.atomic_type_converter = AtomicTypeConverter.from_type_list(atomic_types)
        self.neighbors = neighbors
        self._files = None
        self._length = None
    
    @property
    def files(self) -> list:
        if self._files is None:
            self._files = os.listdir(self.root)
        return self._files
    
    def __len__(self):
        if self._length is None:
            self._length = len(self.files)
        return self._length
        
    def __getitem__(self, index):
        try:
            data = load(os.path.join(self.root, self.files[index]+"/data.pkl"))
        except KeyError:
                raise IndexError(f'Index {index} not found in the file system.')
        
        structure = AtomicStructure(
            species=data['atomic_numbers'],
            atomic_numbers=data['atomic_numbers'],
            positions=data['positions'],
            cell=data['cell'],
            pbc=data['pbc'],
            energy=data['energy'],
            forces=data['forces'],
            stress=data['stress'],
            total_charge=data['total_charge'],
            partial_charges=data['partial_charges'],
            dipole_moment=data['dipole_moment'],
            quadrupole_moment=data['quadrupole_moment'],
            neighbors=self.neighbors,
        )

        structure = structure.to_type_names(self.atomic_type_converter, check=True)

        if 'r_cutoff' in data and 'skin' in data:
            r_cutoff = data['r_cutoff']
            skin = data['skin']
            if r_cutoff == self.r_cutoff and skin == self.skin:
                if 'edge_index' in data and 'shifts' in data:
                    structure._r_cutoff = self.r_cutoff
                    structure._skin = self.skin
                    structure._edge_index = data['edge_index']
                    structure._shifts = data['shifts']
        
        atomic_data = AtomicData(
            structure=structure,
            r_cutoff=self.r_cutoff,
            skin=self.skin,
            n_species=self.atomic_type_converter.get_n_type_names()
        )
        
        return atomic_data


class DatasetHandler:
    """
    Handles data set loading and splitting.

    Methods:
        load_dataset: Loads a data set from a file.
        split_dataset: Splits a data set into subsets.
    """
    def load_dataset(
        self, 
        file_path: Union[str, Path],
        r_cutoff: Optional[float],
        skin: float = 0.0,
        atomic_types: Optional[List[Union[str, int]]] = None,
        neighbors: str = 'matscipy',
        **kwargs: Any
    ) -> Dataset:
        file_path = Path(file_path)
        if file_path.suffix == '.xyz' or file_path.suffix == '.extxyz':
            return XYZDataset(
                file_path=file_path, 
                r_cutoff=r_cutoff, 
                skin=skin, 
                atomic_types=atomic_types, 
                neighbors=neighbors, 
                **kwargs
            )
        elif file_path.suffix == '.hdf5':
            return HDF5Dataset(
                file_path=file_path,
                r_cutoff=r_cutoff,
                skin=skin,
                atomic_types=atomic_types,
                neighbors=neighbors,
                **kwargs
            )
        elif file_path.suffix == '.lmdb':
            return LMDBDataset(
                file_path=file_path,
                r_cutoff=r_cutoff,
                skin=skin,
                atomic_types=atomic_types,
                neighbors=neighbors,
                **kwargs
            )
        else:
            raise RuntimeError(
                f'Unsupported file format: {file_path.suffix}. Use .xyz, .extxyz, .hdf5, or .lmdb instead.'
            )

    def split_dataset(
        self, 
        dataset: Dataset,
        sizes: Dict[str, int],
        seed: int = None
    ) -> Dict[str, Dataset]:
        random_state = np.random.RandomState(seed=seed)
        idx = random_state.permutation(np.arange(len(dataset)))
        sub_idxs = {}
        for key, val in sizes.items():
            sub_idxs.update({key: idx[0:val]})
            idx = idx[val:]
        if len(idx) > 0:
            sub_idxs.update({"test": idx})
        return {name: Subset(dataset, si) for name, si in sub_idxs.items()}
