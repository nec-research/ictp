"""
       ICTP: Irreducible Cartesian Tensor Potentials
	  
  File:     loss_fns.py
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
import torch
import torch.nn.functional as F

from typing import Dict, List, Any, Callable

from ictp.utils.torch_geometric import Data
from ictp.utils.misc import recursive_detach
from ictp.utils.math import segment_sum


class LossFunction:
    """Computes loss functions on batches and then aggregates them to a single value.
    """
    def compute_batch_loss(self,
                           results: Dict[str, torch.Tensor],
                           batch: Data) -> torch.Tensor:
        """Computes loss value for a single batch.

        Args:
            results (Dict[str, torch.Tensor]): Results dictionary; see 'calculators.py'.
            batch (Data): Atomic data graph.

        Returns:
            torch.Tensor: Loss value on batch.
        """
        raise NotImplementedError()

    def reduce_overall(self,
                       losses: List[Any],
                       n_structures_total: int,
                       n_atoms_total: int) -> torch.Tensor:
        """Combines batch losses into a single loss value.

        Args:
            losses (List[Any]): List of losses evaluated on batches.
            n_structures_total (int): Total number of structures in the data set.
            n_atoms_total (int): Total number of atoms in the data set.

        Returns:
            torch.Tensor: Loss value on the data set.
        """
        raise NotImplementedError()

    def get_output_variables(self) -> List[str]:
        """

        Returns:
            List[str]: List of keys (e.g. ['forces']) that should be present in the 'results' 
                       parameter of compute_batch_loss(). This is used in 'calculators.py' 
                       to determine which quantities should be computed.
        """
        raise NotImplementedError()


class SingleLossFunction(LossFunction):
    """Computes loss using a single property.

    Args:
        output_variable (str): Property key (e.g. ['forces']).
        batch_loss (Callable[[torch.Tensor, torch.Tensor, Data], torch.Tensor]): Loss function which is evaluated on a batch.
        overall_reduction (Callable[[torch.Tensor, int, int], torch.Tensor], optional): Function that implements an overall 
                                                                                        reduction of losses evaluated on batches. 
                                                                                        Defaults to None.
    """
    def __init__(self,
                 output_variable: str,
                 batch_loss: Callable[[torch.Tensor, torch.Tensor, Data], torch.Tensor],
                 overall_reduction: Callable[[torch.Tensor, int, int], torch.Tensor]=None):
        self.output_variable = output_variable
        self.batch_loss = batch_loss
        self.overall_reduction = overall_reduction
        
        # counters for removed structures and atoms
        self.n_structures_removed = 0
        self.n_atoms_removed = 0

    def compute_batch_loss(self,
                           results: Dict[str, torch.Tensor],
                           batch: Data) -> torch.Tensor:
        y_pred = results[self.output_variable]
        y = getattr(batch, self.output_variable)
        n_atoms = batch.n_atoms
        
        # compute mask for non-nan values in y
        mask = ~torch.isnan(y).any(dim=tuple(range(1, y.dim())), keepdim=False)
        
        # if mask is empty (all values are nan), return a dummy loss (zero loss)
        if mask.sum().item() == 0:
            return 0.0 * y_pred.sum()
        
        # if ~mask is not empty (there are nan values), apply the mask and update counters
        if (~mask).sum().item() > 0:
            # apply mask to n_atoms and update counters
            if y.shape[0] == batch.n_atoms.shape[0]:
                n_atoms = n_atoms[mask]
                self.n_structures_removed += (~mask).sum().item()
                self.n_atoms_removed += batch.n_atoms[~mask].sum().item()
            elif y.shape[0] == batch.n_atoms.sum().item():
                n_atoms = n_atoms[segment_sum(mask, batch.batch, batch.n_atoms.shape[0]) > 0]
                self.n_structures_removed += (segment_sum(~mask, batch.batch, batch.n_atoms.shape[0]) / batch.n_atoms).sum().item()
                self.n_atoms_removed += (~mask).sum().item()
            else:
                raise RuntimeError(
                    f'Expected the first dimension of output variables to match either the number of structures '
                    f'(n_structures x ...) or the total number of atoms (n_atoms x ...). Provided {y.shape=}.'
                )
            
            # apply mask to both y and y_pred
            y_pred = y_pred[mask]
            y = y[mask]
        
        return self.batch_loss(y, y_pred, n_atoms=n_atoms)

    def reduce_overall(self,
                       losses: List[Any],
                       n_structures_total: int,
                       n_atoms_total: int) -> torch.Tensor:
        collated = torch.cat([l if len(l.shape) > 0 else l[None] for l in losses], dim=0)
        if self.overall_reduction is None:
            return collated
        
        n_structures_total -= self.n_structures_removed
        n_atoms_total -= self.n_atoms_removed
        
        self.n_structures_removed = 0
        self.n_atoms_removed = 0
        
        return self.overall_reduction(collated, n_structures_total, n_atoms_total)

    def get_output_variables(self) -> List[str]:
        return [self.output_variable]


class WeightedSumLossFunction(LossFunction):
    """Computes weighted sum of losses.

    Args:
        loss_fns (List[LossFunction]): List of loss functions.
        weights (List[float]): List of weights.
        overall_reduction (Callable[[torch.Tensor, int, int], torch.Tensor], optional): Currently is unused by this class. 
                                                                                        Defaults to None.
    """
    def __init__(self,
                 loss_fns: List[LossFunction],
                 weights: List[float],
                 overall_reduction: Callable[[torch.Tensor, int, int], torch.Tensor]=None):
        self.loss_fns = loss_fns
        self.weights = weights
        self.overall_reduction = overall_reduction

    def get_output_variables(self) -> List[str]:
        return sum([l.get_output_variables() for l in self.loss_fns], [])

    def compute_batch_loss(self,
                           results: Dict[str, torch.Tensor],
                           batch: Data):
        return [l.compute_batch_loss(results, batch) for l in self.loss_fns]

    def reduce_overall(self,
                       losses: List[Any],
                       n_structures_total: int,
                       n_atoms_total: int) -> torch.Tensor:
        weighted = [self.weights[i] * self.loss_fns[i].reduce_overall(
            [l[i] for l in losses], n_structures_total, n_atoms_total) for i in range(len(self.loss_fns))]
        return sum(weighted)


class TotalLossTracker:
    """Accumulates losses during training.

    Args:
        loss_fn (LossFunction): Loss function.
        requires_grad (bool): If False, loss values are detached from the 
                              computational graph to save RAM.
    """
    def __init__(self,
                 loss_fn: LossFunction,
                 requires_grad: bool):
        self.loss_fn = loss_fn
        self.batch_losses = []
        self.requires_grad = requires_grad

    def append_batch(self,
                     results: Dict[str, torch.Tensor],
                     batch: Data):
        """Computes and stores the loss on a single batch.

        Args:
            results (Dict[str, torch.Tensor]): Results dictionary; see 'calculators.py'.
            batch (Data): Atomic data graph.
        """
        batch_loss = self.loss_fn.compute_batch_loss(results, batch)
        if not self.requires_grad:
            # use detach() to allow freeing the memory of the computation graph attached to results
            batch_loss = recursive_detach(batch_loss)
        self.batch_losses.append(batch_loss)

    def compute_final_result(self,
                             n_structures_total: int,
                             n_atoms_total: int) -> torch.Tensor:
        """Computes the overall loss.

        Args:
            n_structures_total (int): Total number of structures in the data set.
            n_atoms_total (int): Total number of atoms in the data set.

        Returns:
            torch.Tensor: Overall loss.
        """
        return self.loss_fn.reduce_overall(self.batch_losses, n_structures_total, n_atoms_total)


def full_3x3_to_voigt_6_stress(stress_matrix: torch.Tensor) -> torch.Tensor:
    """Form a six-component stress vector in Voigt notation from a 3 x 3 matrix.
    
    Args:
        stress_matrix (torch.Tensor): 3 x 3 stress matrix.

    Returns: 
        torch.Tensor: A six-component stress vector in Voigt notation.
    """
    return torch.stack([stress_matrix[..., 0, 0],
                        stress_matrix[..., 1, 1],
                        stress_matrix[..., 2, 2],
                        (stress_matrix[..., 1, 2] + stress_matrix[..., 2, 1]) / 2,
                        (stress_matrix[..., 0, 2] + stress_matrix[..., 2, 0]) / 2,
                        (stress_matrix[..., 0, 1] + stress_matrix[..., 1, 0]) / 2], dim=-1)


def full_3x3_to_5_quadrupole(quadrupole_matrix: torch.Tensor) -> torch.Tensor:
    """Form a five-component quadrupole vector from a 3 x 3 traceless matrix.
    
    Args:
        quadrupole_matrix (torch.Tensor): 3 x 3 traceless quadrupole matrix.
        
    Returns: 
        torch.Tensor: A five-component quadrupole vector.
    """
    return torch.stack([quadrupole_matrix[..., 0, 0], 
                        quadrupole_matrix[..., 1, 1],
                        (quadrupole_matrix[..., 1, 2] + quadrupole_matrix[..., 2, 1]) / 2,
                        (quadrupole_matrix[..., 0, 2] + quadrupole_matrix[..., 2, 0]) / 2, 
                        (quadrupole_matrix[..., 0, 1] + quadrupole_matrix[..., 1, 0]) / 2], dim=-1)


def normalized_dot_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes the dot product between two tensors after normalizing them to unit vectors.

    Args:
        x (torch.Tensor): A tensor of shape n_atoms x d representing target vectors.
        y (torch.Tensor): A tensor of shape n_atoms x d representing input vectors.

    Returns:
        torch.Tensor: A tensor of shape n_atoms containing the normalized dot product.
    """
    x = F.normalize(x, p=2, dim=-1, eps=1e-7)
    y = F.normalize(y, p=2, dim=-1, eps=1e-7)
    return torch.sum(x * y, dim=-1)


METRICS = dict(
    # energy loss functions
    energy_sae=SingleLossFunction('energy',
                                  lambda y, y_pred, n_atoms: (y-y_pred).abs().sum(),
                                  lambda losses, n_structures, n_atoms: losses.sum()),
    energy_mae=SingleLossFunction('energy',
                                  lambda y, y_pred, n_atoms: (y-y_pred).abs().sum(),
                                  lambda losses, n_structures, n_atoms: losses.sum() / n_structures),
    energy_sse=SingleLossFunction('energy',
                                  lambda y, y_pred, n_atoms: (y-y_pred).square().sum(),
                                  lambda losses, n_structures, n_atoms: losses.sum()),
    energy_mse=SingleLossFunction('energy',
                                  lambda y, y_pred, n_atoms: (y-y_pred).square().sum(),
                                  lambda losses, n_structures, n_atoms: losses.sum() / n_structures),
    energy_rmse=SingleLossFunction('energy',
                                   lambda y, y_pred, n_atoms: (y - y_pred).square().sum(),
                                   lambda losses, n_structures, n_atoms: (losses.sum() / n_structures).sqrt()),
    energy_l4=SingleLossFunction('energy',
                                 lambda y, y_pred, n_atoms: ((y-y_pred) ** 4).sum(),
                                 lambda losses, n_structures, n_atoms: (losses.sum() / n_structures) ** 0.25),
    energy_maxe=SingleLossFunction('energy',
                                   lambda y, y_pred, n_atoms: (y - y_pred).abs().max(),
                                   lambda losses, n_structures, n_atoms: losses.max()),
    energy_per_atom_sae=SingleLossFunction('energy',
                                           lambda y, y_pred, n_atoms: ((y-y_pred)/n_atoms).abs().sum(),
                                           lambda losses, n_structures, n_atoms: losses.sum()),
    energy_per_atom_mae=SingleLossFunction('energy',
                                           lambda y, y_pred, n_atoms: ((y-y_pred)/n_atoms).abs().sum(),
                                           lambda losses, n_structures, n_atoms: losses.sum() / n_structures),
    energy_per_atom_sse=SingleLossFunction('energy',
                                           lambda y, y_pred, n_atoms: ((y-y_pred)/n_atoms).square().sum(),
                                           lambda losses, n_structures, n_atoms: losses.sum()),
    energy_per_atom_mse=SingleLossFunction('energy',
                                           lambda y, y_pred, n_atoms: ((y-y_pred)/n_atoms).square().sum(),
                                           lambda losses, n_structures, n_atoms: losses.sum() / n_structures),
    energy_per_atom_rmse=SingleLossFunction('energy',
                                            lambda y, y_pred, n_atoms: ((y-y_pred)/n_atoms).square().sum(),
                                            lambda losses, n_structures, n_atoms: (losses.sum() / n_structures).sqrt()),
    energy_per_atom_l4=SingleLossFunction('energy',
                                          lambda y, y_pred, n_atoms: (((y-y_pred)/n_atoms) ** 4).sum(),
                                          lambda losses, n_structures, n_atoms: (losses.sum() / n_structures) ** 0.25),
    energy_per_atom_maxe=SingleLossFunction('energy',
                                            lambda y, y_pred, n_atoms: ((y-y_pred)/n_atoms).abs().max(),
                                            lambda losses, n_structures, n_atoms: losses.max()),
    energy_by_sqrt_atoms_sse=SingleLossFunction('energy',
                                                lambda y, y_pred, n_atoms: ((y-y_pred).square() / n_atoms).sum(),
                                                lambda losses, n_structures, n_atoms: losses.sum()),
    energy_by_sqrt_atoms_mse=SingleLossFunction('energy',
                                                lambda y, y_pred, n_atoms: ((y-y_pred).square() / n_atoms).sum(),
                                                lambda losses, n_structures, n_atoms: losses.sum() / n_structures),
    # forces loss functions
    forces_sae=SingleLossFunction('forces',
                                  lambda y, y_pred, n_atoms: (y-y_pred).abs().sum(),
                                  lambda losses, n_structures, n_atoms: losses.sum()),
    forces_mae=SingleLossFunction('forces',
                                  lambda y, y_pred, n_atoms: (y-y_pred).abs().sum(),
                                  lambda losses, n_structures, n_atoms: losses.sum() / (3*n_atoms)),
    forces_sse=SingleLossFunction('forces',
                                  lambda y, y_pred, n_atoms: (y-y_pred).square().sum(),
                                  lambda losses, n_structures, n_atoms: losses.sum()),
    forces_mse=SingleLossFunction('forces',
                                  lambda y, y_pred, n_atoms: (y-y_pred).square().sum(),
                                  lambda losses, n_structures, n_atoms: losses.sum() / (3*n_atoms)),
    forces_rmse=SingleLossFunction('forces',
                                   lambda y, y_pred, n_atoms: (y-y_pred).square().sum(),
                                   lambda losses, n_structures, n_atoms: (losses.sum() / (3*n_atoms)).sqrt()),
    forces_l4=SingleLossFunction('forces',
                                 lambda y, y_pred, n_atoms: ((y-y_pred) ** 4).sum(),
                                 lambda losses, n_structures, n_atoms: (losses.sum() / (3*n_atoms)) ** 0.25),
    forces_maxe=SingleLossFunction('forces',
                                   lambda y, y_pred, n_atoms: (y - y_pred).abs().max(),
                                   lambda losses, n_structures, n_atoms: losses.max()),
    # stress loss functions
    stress_sae=SingleLossFunction('stress',
                                   lambda y, y_pred, n_atoms: (full_3x3_to_voigt_6_stress(y)-full_3x3_to_voigt_6_stress(y_pred)).abs().sum(),
                                   lambda losses, n_structures, n_atoms: losses.sum()),
    stress_mae=SingleLossFunction('stress',
                                   lambda y, y_pred, n_atoms: (full_3x3_to_voigt_6_stress(y)-full_3x3_to_voigt_6_stress(y_pred)).abs().sum(),
                                   lambda losses, n_structures, n_atoms: losses.sum() / (6*n_structures)),
    stress_sse=SingleLossFunction('stress',
                                   lambda y, y_pred, n_atoms: (full_3x3_to_voigt_6_stress(y)-full_3x3_to_voigt_6_stress(y_pred)).square().sum(),
                                   lambda losses, n_structures, n_atoms: losses.sum()),
    stress_mse=SingleLossFunction('stress',
                                  lambda y, y_pred, n_atoms: (full_3x3_to_voigt_6_stress(y)-full_3x3_to_voigt_6_stress(y_pred)).square().sum(),
                                  lambda losses, n_structures, n_atoms: losses.sum() / (6*n_structures)),
    stress_rmse=SingleLossFunction('stress',
                                   lambda y, y_pred, n_atoms: (full_3x3_to_voigt_6_stress(y)-full_3x3_to_voigt_6_stress(y_pred)).square().sum(),
                                   lambda losses, n_structures, n_atoms: (losses.sum() / (6*n_structures)).sqrt()),
    stress_l4=SingleLossFunction('stress',
                                 lambda y, y_pred, n_atoms: ((full_3x3_to_voigt_6_stress(y)-full_3x3_to_voigt_6_stress(y_pred)) ** 4).sum(),
                                 lambda losses, n_structures, n_atoms: (losses.sum() / (6*n_structures)) ** 0.25),
    stress_maxe=SingleLossFunction('stress',
                                   lambda y, y_pred, n_atoms: (full_3x3_to_voigt_6_stress(y)-full_3x3_to_voigt_6_stress(y_pred)).abs().max(),
                                   lambda losses, n_structures, n_atoms: losses.max()),
    # virials loss functions
    virials_sae=SingleLossFunction('virials',
                                   lambda y, y_pred, n_atoms: (full_3x3_to_voigt_6_stress(y)-full_3x3_to_voigt_6_stress(y_pred)).abs().sum(),
                                   lambda losses, n_structures, n_atoms: losses.sum()),
    virials_mae=SingleLossFunction('virials',
                                   lambda y, y_pred, n_atoms: (full_3x3_to_voigt_6_stress(y)-full_3x3_to_voigt_6_stress(y_pred)).abs().sum(),
                                   lambda losses, n_structures, n_atoms: losses.sum() / (6*n_structures)),
    virials_sse=SingleLossFunction('virials',
                                   lambda y, y_pred, n_atoms: (full_3x3_to_voigt_6_stress(y)-full_3x3_to_voigt_6_stress(y_pred)).square().sum(),
                                   lambda losses, n_structures, n_atoms: losses.sum()),
    virials_mse=SingleLossFunction('virials',
                                  lambda y, y_pred, n_atoms: (full_3x3_to_voigt_6_stress(y)-full_3x3_to_voigt_6_stress(y_pred)).square().sum(),
                                  lambda losses, n_structures, n_atoms: losses.sum() / (6*n_structures)),
    virials_rmse=SingleLossFunction('virials',
                                   lambda y, y_pred, n_atoms: (full_3x3_to_voigt_6_stress(y)-full_3x3_to_voigt_6_stress(y_pred)).square().sum(),
                                   lambda losses, n_structures, n_atoms: (losses.sum() / (6*n_structures)).sqrt()),
    virials_l4=SingleLossFunction('virials',
                                 lambda y, y_pred, n_atoms: ((full_3x3_to_voigt_6_stress(y)-full_3x3_to_voigt_6_stress(y_pred)) ** 4).sum(),
                                 lambda losses, n_structures, n_atoms: (losses.sum() / (6 * n_structures)) ** 0.25),
    virials_maxe=SingleLossFunction('virials',
                                   lambda y, y_pred, n_atoms: (full_3x3_to_voigt_6_stress(y)-full_3x3_to_voigt_6_stress(y_pred)).abs().max(),
                                   lambda losses, n_structures, n_atoms: losses.max()),
    virials_by_sqrt_atoms_sse=SingleLossFunction('virials',
                                                 lambda y, y_pred, n_atoms: ((full_3x3_to_voigt_6_stress(y)-full_3x3_to_voigt_6_stress(y_pred)).square() / n_atoms[:, None]).sum(),
                                                 lambda losses, n_structures, n_atoms: losses.sum()),
    virials_per_atom_mae=SingleLossFunction('virials',
                                            lambda y, y_pred, n_atoms: ((full_3x3_to_voigt_6_stress(y)-full_3x3_to_voigt_6_stress(y_pred)) / n_atoms[:, None]).abs().sum(),
                                            lambda losses, n_structures, n_atoms: losses.sum() / (6 * n_structures)),
    virials_per_atom_rmse=SingleLossFunction('virials',
                                             lambda y, y_pred, n_atoms: ((full_3x3_to_voigt_6_stress(y)-full_3x3_to_voigt_6_stress(y_pred)) / n_atoms[:, None]).square().sum(),
                                             lambda losses, n_structures, n_atoms: (losses.sum() / (6 * n_structures)).sqrt()),
    virials_per_atom_mse=SingleLossFunction('virials',
                                            lambda y, y_pred, n_atoms: ((full_3x3_to_voigt_6_stress(y)-full_3x3_to_voigt_6_stress(y_pred))/n_atoms[:, None]).square().sum(),
                                            lambda losses, n_structures, n_atoms: losses.sum() / (6 * n_structures)),
    # partial charges loss functions
    partial_charges_sae=SingleLossFunction('partial_charges',
                                           lambda y, y_pred, n_atoms: (y-y_pred).abs().sum(),
                                           lambda losses, n_structures, n_atoms: losses.sum()),
    partial_charges_mae=SingleLossFunction('partial_charges',
                                           lambda y, y_pred, n_atoms: (y-y_pred).abs().sum(),
                                           lambda losses, n_structures, n_atoms: losses.sum() / n_atoms),
    partial_charges_sse=SingleLossFunction('partial_charges',
                                           lambda y, y_pred, n_atoms: (y-y_pred).square().sum(),
                                           lambda losses, n_structures, n_atoms: losses.sum()),
    partial_charges_mse=SingleLossFunction('partial_charges',
                                           lambda y, y_pred, n_atoms: (y-y_pred).square().sum(),
                                           lambda losses, n_structures, n_atoms: losses.sum() / n_atoms),
    partial_charges_rmse=SingleLossFunction('partial_charges',
                                            lambda y, y_pred, n_atoms: (y-y_pred).square().sum(),
                                            lambda losses, n_structures, n_atoms: (losses.sum() / n_atoms).sqrt()),
    # dipole moment loss function
    dipole_moment_sae=SingleLossFunction('dipole_moment',
                                           lambda y, y_pred, n_atoms: (y-y_pred).abs().sum(),
                                           lambda losses, n_structures, n_atoms: losses.sum()),
    dipole_moment_mae=SingleLossFunction('dipole_moment',
                                           lambda y, y_pred, n_atoms: (y-y_pred).abs().sum(),
                                           lambda losses, n_structures, n_atoms: losses.sum() / (3 * n_structures)),
    dipole_moment_sse=SingleLossFunction('dipole_moment',
                                           lambda y, y_pred, n_atoms: (y-y_pred).square().sum(),
                                           lambda losses, n_structures, n_atoms: losses.sum()),
    dipole_moment_mse=SingleLossFunction('dipole_moment',
                                           lambda y, y_pred, n_atoms: (y-y_pred).square().sum(),
                                           lambda losses, n_structures, n_atoms: losses.sum() / (3 * n_structures)),
    dipole_moment_rmse=SingleLossFunction('dipole_moment',
                                            lambda y, y_pred, n_atoms: (y-y_pred).square().sum(),
                                            lambda losses, n_structures, n_atoms: (losses.sum() / (3 * n_structures)).sqrt()),
    dipole_moment_by_sqrt_atoms_sse=SingleLossFunction('dipole_moment',
                                                       lambda y, y_pred, n_atoms: ((y-y_pred).square() / n_atoms[:, None]).sum(),
                                                       lambda losses, n_structures, n_atoms: losses.sum()),
    dipole_moment_per_atom_mae=SingleLossFunction('dipole_moment',
                                                  lambda y, y_pred, n_atoms: ((y-y_pred) / n_atoms[:, None]).abs().sum(),
                                                  lambda losses, n_structures, n_atoms: losses.sum() / (3 * n_structures)),
    dipole_moment_atom_mse=SingleLossFunction('dipole_moment',
                                              lambda y, y_pred, n_atoms: ((y-y_pred) / n_atoms[:, None]).square().sum(),
                                              lambda losses, n_structures, n_atoms: losses.sum() / (3 * n_structures)),
    dipole_moment_per_atom_rmse=SingleLossFunction('dipole_moment',
                                                   lambda y, y_pred, n_atoms: ((y-y_pred) / n_atoms[:, None]).square().sum(),
                                                   lambda losses, n_structures, n_atoms: (losses.sum() / (3 * n_structures)).sqrt()),
    dipole_moment_cosine_sim_sum=SingleLossFunction('dipole_moment',
                                                    lambda y, y_pred, n_atoms: (1.0 - normalized_dot_product(y, y_pred)).sum(),
                                                    lambda losses, n_structures, n_atoms: losses.sum()),
    dipole_moment_cosine_sim_mean=SingleLossFunction('dipole_moment',
                                                     lambda y, y_pred, n_atoms: (1.0 - normalized_dot_product(y, y_pred)).sum(),
                                                     lambda losses, n_structures, n_atoms: losses.sum() / n_structures),
    # quadrupole moment loss functions
    quadrupole_moment_sae=SingleLossFunction('quadrupole_moment',
                                             lambda y, y_pred, n_atoms: (full_3x3_to_5_quadrupole(y)-full_3x3_to_5_quadrupole(y_pred)).abs().sum(),
                                             lambda losses, n_structures, n_atoms: losses.sum()),
    quadrupole_moment_mae=SingleLossFunction('quadrupole_moment',
                                             lambda y, y_pred, n_atoms: (full_3x3_to_5_quadrupole(y)-full_3x3_to_5_quadrupole(y_pred)).abs().sum(),
                                             lambda losses, n_structures, n_atoms: losses.sum() / (5 * n_structures)),
    quadrupole_moment_sse=SingleLossFunction('quadrupole_moment',
                                             lambda y, y_pred, n_atoms: (full_3x3_to_5_quadrupole(y)-full_3x3_to_5_quadrupole(y_pred)).square().sum(),
                                             lambda losses, n_structures, n_atoms: losses.sum()),
    quadrupole_moment_mse=SingleLossFunction('quadrupole_moment',
                                             lambda y, y_pred, n_atoms: (full_3x3_to_5_quadrupole(y)-full_3x3_to_5_quadrupole(y_pred)).square().sum(),
                                             lambda losses, n_structures, n_atoms: losses.sum() / (5 * n_structures)),
    quadrupole_moment_rmse=SingleLossFunction('quadrupole_moment',
                                              lambda y, y_pred, n_atoms: (full_3x3_to_5_quadrupole(y)-full_3x3_to_5_quadrupole(y_pred)).square().sum(),
                                              lambda losses, n_structures, n_atoms: (losses.sum() / (5 * n_structures)).sqrt()),
    quadrupole_moment_by_sqrt_atoms_sse=SingleLossFunction('quadrupole_moment',
                                                           lambda y, y_pred, n_atoms: ((full_3x3_to_5_quadrupole(y)-full_3x3_to_5_quadrupole(y_pred)).square() / n_atoms[:, None]).sum(),
                                                           lambda losses, n_structures, n_atoms: losses.sum()),
    quadrupole_moment_per_atom_mae=SingleLossFunction('quadrupole_moment',
                                                      lambda y, y_pred, n_atoms: ((full_3x3_to_5_quadrupole(y)-full_3x3_to_5_quadrupole(y_pred)) / n_atoms[:, None]).abs().sum(),
                                                      lambda losses, n_structures, n_atoms: losses.sum() / (5 * n_structures)),
    quadrupole_moment_atom_mse=SingleLossFunction('quadrupole_moment',
                                                  lambda y, y_pred, n_atoms: ((full_3x3_to_5_quadrupole(y)-full_3x3_to_5_quadrupole(y_pred)) / n_atoms[:, None]).square().sum(),
                                                  lambda losses, n_structures, n_atoms: losses.sum() / (5 * n_structures)),
    quadrupole_moment_per_atom_rmse=SingleLossFunction('quadrupole_moment',
                                                       lambda y, y_pred, n_atoms: ((full_3x3_to_5_quadrupole(y)-full_3x3_to_5_quadrupole(y_pred)) / n_atoms[:, None]).square().sum(),
                                                       lambda losses, n_structures, n_atoms: (losses.sum() / (5 * n_structures)).sqrt()),

)

def config_to_loss(config: dict) -> LossFunction:
    """Unwraps loss types and other details in config file to respective loss functions.

    Args:
        config (dict): Dictionary containing loss types and other details.

    Returns:
        LossFunction: Loss function defined by config file.
    """
    t = config['type']
    if t == 'weighted_sum':
        losses = [config_to_loss(c) for c in config['losses']]
        return WeightedSumLossFunction(losses, config['weights'])
    else:
        if t in METRICS:
            return METRICS[t]
        else:
            raise ValueError(f'Not implemented loss "{t}"! Available losses: {list(METRICS.keys())}.')
