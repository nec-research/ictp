"""
       ICTP: Irreducible Cartesian Tensor Potentials
	  
  File:     ase.py
  Authors:  Viktor Zaverkin (viktor.zaverkin@neclab.eu)
            Francesco Alesiani (francesco.alesiani@neclab.eu)
            Takashi Maruyama (takashi.maruyama@neclab.eu)
            Federico Errica (federico.errica@neclab.eu)
            Henrik Christiansen (henrik.christiansen@neclab.eu)
            Makoto Takamoto (makoto.takamoto@neclab.eu)
            Nicolas Weber (nicolas.weber@neclab.eu)
            Mathias Niepert (mathias.niepert@ki.uni-stuttgart.de)

NEC Laboratories Europe GmbH, Copyright (c) 2024, All rights reserved.  

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
from pathlib import Path
from typing import List, Optional, Union, Any

import ase
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress


from ictp.data.data import AtomicTypeConverter, AtomicStructure, AtomicData
from ictp.model.forward import ForwardAtomisticNetwork, find_last_ckpt
from ictp.model.calculators import TorchCalculator, StructurePropertyCalculator
from ictp.utils.torch_geometric import DataLoader
from ictp.utils.misc import get_default_device


class ASEWrapper(Calculator):
    """Wraps a `TorchCalculator` object into an ASE calculator. It is used to perform atomistic simulations within ASE.

    Args:
        calc (TorchCalculator): The `TorchCalculator` object which allows for energy, forces, etc. calculations.
        r_cutoff (float): Cutoff radius for computing the neighbor list.
        atomic_types (Optional[List[str]], optional): List of supported atomic numbers. Defaults to None.
        device (Optional[str], optional): Available device (e.g., 'cuda:0' or 'cpu'). Defaults to None.
        skin (float, optional): Skin distance used to update the neighbor list. Defaults to 0.0.
        wrap (bool, optional): Wrap positions back to the periodic cell. Defaults to False.
        neighbors (str, optional): Method for computing the neighbor list. Defaults to 'matscipy'.
        energy_units_to_eV (float, optional): Energy conversion factor. Defaults to 1.0.
        length_units_to_A (float, optional): Length conversion factor. Defaults to 1.0.
    """
    
    implemented_properties = ['energy', 'forces', 'stress']
    
    def __init__(self,
                 calc: TorchCalculator,
                 r_cutoff: float,
                 atomic_types: Optional[List[str]] = None,
                 device: Optional[str] = None,
                 skin: float = 0.0,
                 wrap: bool = False,
                 neighbors: str = 'matscipy',
                 energy_units_to_eV: float = 1.0,
                 length_units_to_A: float = 1.0,
                 **kwargs: Any):
        super().__init__()
        self.results = {}
        
        self.calc = calc
        # define device and move calc to it
        self.device = device or get_default_device()
        self.calc.to(self.device)
        # cutoff radius to compute neighbors
        self.r_cutoff = r_cutoff
        # skin to restore neighbors
        self.skin = skin
        # wrap atoms to the box
        self.wrap = wrap
        # convert atomic numbers to atomic types for internal use
        self.atomic_type_converter = AtomicTypeConverter.from_type_list(atomic_types)
        # method for calculating neighbor list
        self.neighbors = neighbors
        # re-scale energy and forces
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A
        # init last structure to check whether new neighbors have to be computed
        self._last_structure = None

        # storing calc results for external use
        self._torch_calc_results = {}

    def calculate(self,
                  atoms: Optional[ase.Atoms] = None,
                  properties: List[str] = ['energy'],
                  system_changes: List[str] = all_changes):
        """Calculates the total energy, atomic force, etc. for `ase.Atoms`.

        Args:
            atoms (Optional[ase.Atoms], optional): The `ase.Atoms` object. Defaults to None.
            properties (List[str], optional): List of properties to be computed. Defaults to ['energy'].
            system_changes (List[str], optional): Defaults to all_changes.
        """
        if self.calculation_required(atoms, properties):
            super().calculate(atoms)
            # prepare data
            structure = AtomicStructure.from_atoms(atoms, wrap=self.wrap, neighbors=self.neighbors)
            # check whether provided atom types are supported
            structure = structure.to_type_names(self.atomic_type_converter, check=True)
            # check if neighbors can be restored from the last structure
            if not structure.restore_neighbors_from_last(structure=self._last_structure,
                                                         r_cutoff=self.r_cutoff, skin=self.skin):
                self._last_structure = structure
            # prepare data
            dl = DataLoader([AtomicData(structure, r_cutoff=self.r_cutoff, skin=self.skin, 
                                        n_species=self.atomic_type_converter.get_n_type_names())],
                            batch_size=1, shuffle=False, drop_last=False)
            batch = next(iter(dl)).to(self.device)
            # predict
            out = self.calc(batch, forces='forces' in properties, stress='stress' in properties, create_graph='stress' in properties)
            # store results for ase
            self.results = {'energy': out['energy'].detach().cpu().item() * self.energy_units_to_eV}
            if 'forces' in out:
                forces = out['forces'].detach().cpu().numpy() * (self.energy_units_to_eV / self.length_units_to_A)
                self.results['forces'] = forces
            if 'stress' in out:
                stress = out['stress'].detach().cpu().numpy()
                stress = stress.reshape(3, 3) * (self.energy_units_to_eV / self.length_units_to_A ** 3)
                stress_voigt = full_3x3_to_voigt_6_stress(stress)
                self.results['stress'] = stress_voigt
    
    @staticmethod
    def from_folder_list(folder: Union[Path, str],
                         **kwargs: Any) -> Calculator:
        """Provides a wrapped ASE calculator from the model.

        Args:
            folder (Union[Path, str]): Folder containing the trained model.

        Returns:
            ase.Calculator: Wrapped ASE calculator.
        """
        # load model from folder
        model = ForwardAtomisticNetwork.from_folder(find_last_ckpt(folder))
        
        # build torch calculator
        calc = StructurePropertyCalculator(model, training=False)
        
        return ASEWrapper(calc, model.config['r_cutoff'], model.config['atomic_types'], **kwargs)
