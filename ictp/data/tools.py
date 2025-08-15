"""
       ICTP: Irreducible Cartesian Tensor Potentials
	  
  File:     tools.py
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
from typing import Optional

import numpy as np

import torch

from ictp.utils.torch_geometric import DataLoader
from ictp.utils.math import segment_sum


def get_energy_shift_per_atom(
    data_loader: DataLoader,
    n_species: int,
    atomic_energies: Optional[np.ndarray] = None,
    compute_regression_shift: bool = True
) -> np.ndarray:
    """
    Computes energy shift parameters for each atomic species in the data set. 
    If atomic energies are provided, they are subtracted from the total energy 
    before computing the mean and the regression solution.

    Args:
        data_loader (DataLoader): Provides batches of atomic data.
        n_species (int): Total number of atomic species (atom types).
        atomic_energies (np.ndarray, optional): Predefined atomic energies. Defaults to None.
        compute_regression_shift (bool, optional): If True, computes regression shift. Defaults to True.

    Returns:
        np.ndarray: Atomic energy shift parameters.
    """
    # initialize atomic energies
    if atomic_energies is None:
        atomic_energies = torch.zeros(n_species, dtype=torch.float64)
    else:
        if len(atomic_energies) != n_species:
            raise RuntimeError(
                f'Length of atomic_energies {len(atomic_energies)} must match n_species {n_species}.'
            )
        atomic_energies = torch.tensor(atomic_energies, dtype=torch.float64)
    
    if not compute_regression_shift:
        return atomic_energies.cpu().detach().numpy()
    
    # initialize energy and atom sums
    energy_sum = torch.tensor(0.0, dtype=torch.float64)
    atoms_sum = torch.tensor(0.0, dtype=torch.float64)
    
    # compute mean energy per atom
    for batch in data_loader:
        energy = batch.energy.to(torch.float64)
        species = batch.species.to(torch.long)
        
        # compute sum of atomic energies for the batch
        atomic_energies_sum = segment_sum(
            atomic_energies[species], batch.batch, batch.n_atoms.shape[0], 0
        )
        
        energy_sum += (energy - atomic_energies_sum).sum()
        atoms_sum += batch.n_atoms.sum()
    
    energy_per_atom_mean = energy_sum / atoms_sum
    
    # prepare for regression
    XTy = torch.zeros(n_species, dtype=torch.float64)
    XTX = torch.zeros((n_species, n_species), dtype=torch.long)
    
    for batch in data_loader:
        energy = batch.energy.to(torch.float64)
        species = batch.species.to(torch.long)
        
        # compute species count matrix Z_counts for the batch
        Z_counts = torch.zeros(batch.n_atoms.shape[0], n_species, dtype=torch.long)
        for j in range(batch.n_atoms.shape[0]):
            structure_species = species[batch.ptr[j]:batch.ptr[j + 1]]
            Z_counts[j] = torch.bincount(structure_species, minlength=n_species)
        
        atomic_energies_sum = segment_sum(
            atomic_energies[species], batch.batch, batch.n_atoms.shape[0], 0
        )
        
        residual = (
            energy
            - atomic_energies_sum
            - batch.n_atoms * energy_per_atom_mean
        )
        
        # update regression matrices
        XTy += (residual[:, None] * Z_counts).sum(dim=0)
        XTX += (Z_counts[:, :, None] * Z_counts[:, None, :]).sum(dim=0)

    # solve for regression shift
    lam = 1.0  # regularization parameter
    regression_shift = torch.linalg.solve(
        XTX + lam * torch.eye(n_species, dtype=torch.float64), XTy
    )
    
    return (regression_shift + energy_per_atom_mean + atomic_energies).cpu().detach().numpy()


def get_forces_rms(
    data_loader: DataLoader,
    n_species: int
) -> np.ndarray:
    """
    Computes the root mean square (RMS) of forces across atomic structures in the data set.

    Args:
        data_loader (DataLoader): Provides batches of atomic data.
        n_species (int): Total number of atomic species (atom types).

    Returns:
        np.ndarray: Root mean square (RMS) of forces across atomic structures in the data set.
    """
    sq_forces_sum = 0.0
    atoms_sum = 0
    
    for batch in data_loader:
        forces = batch.forces.to(torch.float64)
        
        # accumulate the squared forces and atom counts
        sq_forces_sum += forces.square().sum().item()
        atoms_sum += batch.n_atoms.sum().item()
    
    forces_rms = np.sqrt(sq_forces_sum / (atoms_sum * 3.0)) * np.ones(n_species)
    
    # ensure no zero values in the output
    forces_rms[forces_rms == 0.0] = 1.0
            
    return forces_rms


def get_avg_n_neighbors(data_loader: DataLoader) -> float:
    """
    Computes the average number of neighbors in the data set.

    Adapted from MACE: https://github.com/ACEsuit/mace/blob/main/mace/modules/utils.py
    
    Args:
        data_loader (DataLoader): Provides batches of atomic data.

    Returns:
        float: Average number of neighbors per atom in the data set.
    """
    n_neighbors = []
    
    for batch in data_loader:
        idx_i, _ = batch.edge_index
        
        # count occurrences of each index
        _, counts = torch.unique(idx_i, return_counts=True)
        n_neighbors.append(counts)
        
    avg_n_neighbors = torch.mean(
        torch.cat(n_neighbors, dim=0).type(torch.get_default_dtype())
    )
    
    return avg_n_neighbors.cpu().detach().numpy().item()
