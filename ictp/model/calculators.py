"""
       ICTP: Irreducible Cartesian Tensor Potentials
	  
  File:     calculators.py
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
from typing import Dict, Union, Any, List, Tuple, Optional

import torch

from ictp.model.forward import ForwardAtomisticNetwork

from ictp.utils.math import segment_sum
from ictp.utils.torch_geometric import Data


class TorchCalculator:
    """Computes atomic properties, e.g., (total) energy, atomic forces, stress.
    """
    def __call__(self,
                 graph: Data,
                 **kwargs: Any) -> Dict[str, Union[torch.Tensor, Any]]:
        """Performs calculation on the provided (batch) graph data.

        Args:
            graph (Data): Atomic data graph.

        Returns: 
            Dict[str, Union[torch.Tensor, Any]]: Results dictionary.
        """
        raise NotImplementedError()

    def get_device(self) -> str:
        """Provides the device on which calculations are performed.
        
        Returns: 
            str: Device on which calculations are performed.
        """
        raise NotImplementedError()

    def to(self, device: str) -> 'TorchCalculator':
        """Moves the calculator to the provided device.
        
        Args:
            device: Device to which calculator has to be moved.

        Returns: 
            TorchCalculator: The `TorchCalculator` object.
        """
        raise NotImplementedError()


def prepare_gradients(graph: Data,
                      forces: bool = False,
                      stress: bool = False,
                      virials: bool = False) -> Tuple[Data, List[str]]:
    """Prepares gradient calculation by setting `requires_grad=True` for the selected atomic features. 

    Args:
        graph (Data): Atomic data graph.
        forces (bool): If True, gradients with respect to positions/coordinates are calculated. 
                       Defaults to False.
        stress (bool): If True, gradients with respect to strain deformations are calculated. 
                       Defaults to False.
        virials (bool): If True, gradients with respect to strain deformations are calculated. 
                        Defaults to False.
    
        Returns:
            Tuple[Data, List[str]]: Updated graph and list of properties which require gradients.
    """
    require_gradients = []
    if forces:
        require_gradients.append('positions')
        if not graph.positions.requires_grad:
            # request gradients wrt. positions/coordinates
            graph.positions.requires_grad = True
    if stress or virials:
        require_gradients.append('strain')
        if not graph.strain.requires_grad:
            # define displacements corresponding to:
            # Knuth et. al. Comput. Phys. Commun 190, 33-50, 2015
            # similar implementations are provided by NequIP (https://github.com/mir-group/nequip)
            # and SchNetPack (https://github.com/atomistic-machine-learning/schnetpack)
            graph.strain.requires_grad = True
            # symmetrize to account for possible numerical issues
            symmetric_strain = 0.5 * (graph.strain + graph.strain.transpose(-1, -2))
            # update cell
            graph.cell = graph.cell + torch.matmul(graph.cell, symmetric_strain)
            # update positions
            symmetric_strain_i = symmetric_strain.index_select(0, graph.batch)
            graph.positions = graph.positions + torch.matmul(graph.positions.unsqueeze(-2),
                                                             symmetric_strain_i).squeeze(-2)
            # update the shifts
            symmetric_strain_ij = symmetric_strain_i.index_select(0, graph.edge_index[0, :])
            graph.shifts = graph.shifts + torch.matmul(graph.shifts.unsqueeze(-2), symmetric_strain_ij).squeeze(-2)
    return graph, require_gradients


class StructurePropertyCalculator(TorchCalculator):
    """Calculates total energy, atomic forces, stress tensors from atomic energies.

    Args:
        model (ForwardAtomisticNetwork): Forward atomistic neural network object (provides atomic/node energies).
    """
    def __init__(self,
                 model: ForwardAtomisticNetwork,
                 training: bool = False,
                 **config: Any):
        self.model = torch.jit.script(model) if training else torch.compile(model, backend='inductor')

    def __call__(self,
                 graph: Data,
                 forces: bool = False,
                 stress: bool = False,
                 virials: bool = False,
                 create_graph: bool = False) -> Dict[str, torch.Tensor]:
        """Performs calculations for the atomic data graph.

        Args:
            graph (Data): Atomic data graph.
            forces (bool): If True, atomic forces are computed. Defaults to False.
            stress (bool): If True, stress tensor is computed. Defaults to False.
            virials (bool): If True, virials = - stress * volume are computed. Defaults to False.
            create_graph (bool): If True, computational graph is created allowing the computation of 
                                 backward pass for multiple times. Defaults to False.

        Returns:
            Dict[str, torch.Tensor]: Results dict.
        """
        results = {}
        # prepare graph and the list containing graph attributes requiring gradients
        graph, require_gradients = prepare_gradients(graph=graph, forces=forces, stress=stress, virials=virials)
        # compute atomic energy
        atomic_energies = self.model(graph.to_dict())
        results['atomic_energies'] = atomic_energies
        # sum up atomic contributions for a structure
        total_energies = segment_sum(atomic_energies, idx_i=graph.batch, dim_size=graph.n_atoms.shape[0])
        # write total energy to results
        results['energy'] = total_energies
        if require_gradients:
            # compute gradients wrt. positions, strain, etc.
            grads = torch.autograd.grad([atomic_energies], [getattr(graph, key) for key in require_gradients],
                                        torch.ones_like(atomic_energies), create_graph=create_graph)
        if forces:
            # compute forces as negative of the gradient wrt. positions
            results['forces'] = torch.neg(grads[0])
        if virials:
            # compute virials as negative of the gradient wrt. strain (note that other conventions are possible,
            # but here we use virials = -1 * stress * volume)
            if grads[-1] is not None:
                results['virials'] = torch.neg(grads[-1])
            else:
                results['virials'] = torch.zeros_like(graph.cell)
        if stress:
            # compute stress as -1 * virials / volume
            volume = torch.einsum('bi, bi -> b', graph.cell[:, 0, :],
                                  torch.cross(graph.cell[:, 1, :], graph.cell[:, 2, :], dim=1)).abs()
            if grads[-1] is not None:
                results['stress'] = grads[-1] / volume[:, None, None]
            else:
                results['stress'] = torch.zeros_like(graph.cell) / volume[:, None, None]
        return results

    def get_device(self) -> str:
        return self.model.get_device()

    def to(self, device: str) -> TorchCalculator:
        self.model.to(device)
        return self
