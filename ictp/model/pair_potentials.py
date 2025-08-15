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

import math
import numpy as np

import ase

import torch
import torch.nn as nn
import torch.nn.functional as F

from ictp.nn.radial import PolynomialCutoff

from ictp.utils.math import segment_sum, softplus_inverse, switch_function
from ictp.utils.dimos.bspline import bspline


class ZBLRepulsionEnergy(nn.Module):
    """
    Basic class for the  Ziegler-Biersack-Littmark (ZBL) screened nuclear repulsion. 
    
    For more details see: https://docs.lammps.org/pair_zbl.html and 
    J.F. Ziegler, J. P. Biersack and U. Littmark, “The Stopping and Range of Ions in Matter”, Volume 1, Pergamon, 1985.
    
    Some parts are adapted from SpookyNet (https://github.com/OUnke/SpookyNet/blob/main/spookynet/modules/zbl_repulsion_energy.py).
    
    Args:
        r_cutoff (float, optional): Maximal cutoff radius for the ZBL repulsion potential. 
                                    Defaults to 2 * max(ase.data.covalent_radii).
        p (int, optional): Polynomial order for the cutoff function.
        ke (float, optional): 
            Coulomb constant in the units of [Energy] * [Distance] / [Charge] ** 2. 
            Defaults to 14.399645351950548 eV * Angstrom / e ** 2.
    """
    def __init__(
        self,
        r_cutoff: float = 2 * max(ase.data.covalent_radii).item(),
        n_polynomial_cutoff: int = 6,
        ke: float = 14.399645351950548,
        **config: Any
    ):
        super(ZBLRepulsionEnergy, self).__init__()
        self.register_buffer('r_cutoff', torch.tensor(r_cutoff, dtype=torch.get_default_dtype()))
        self.register_buffer('n_polynomial_cutoff', torch.tensor(n_polynomial_cutoff, dtype=torch.int))
        self.register_buffer('covalent_radii', torch.tensor(ase.data.covalent_radii, dtype=torch.get_default_dtype()))
        
        # half the Coulomb constant (because we run the sum over all atom pairs ij)
        self.ke_half = ke / 2.0
        
        # define parameters of the ZBL pair potential (https://docs.lammps.org/pair_zbl.html)
        # apply inverse suftplus to constrain them to positive values
        self.c_zbl = nn.Parameter(softplus_inverse([0.18175, 0.50986, 0.28022, 0.02817]))
        self.d_zbl = nn.Parameter(softplus_inverse([3.19980, 0.94229, 0.40290, 0.20162]))
        self.zbl_pow = nn.Parameter(softplus_inverse([0.23]))
        self.zbl_length = nn.Parameter(softplus_inverse([0.46850]))
        
    def forward(
        self,
        graph: Dict[str, torch.Tensor],
        results: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Computes ZBL atomic repulsion energy.
        
        Args:
            graph (Dict[str, torch.Tensor]): Atomic graph dictionary.
            results (Dict[str, torch.Tensor]): Results dictionary.
            
        Returns:
            Dict[str, torch.Tensor]: Updated results dictionary.
        """
        numbers, edge_index, lengths = graph['atomic_numbers'], graph['edge_index'], graph['lengths'].squeeze()
        idx_i, idx_j = edge_index[0, :], edge_index[1, :]
        
        cutoff_mask = lengths <= self.r_cutoff
        lengths, idx_i, idx_j = lengths[cutoff_mask], idx_i[cutoff_mask], idx_j[cutoff_mask]
        
        # prepare parameters
        c_zbl, d_zbl = F.softplus(self.c_zbl).unsqueeze(0), F.softplus(self.d_zbl).unsqueeze(0)
        zbl_pow, zbl_length = F.softplus(self.zbl_pow), F.softplus(self.zbl_length)
        
        # Normalize c_zbl coefficients to get asymptotically correct behaviour for r -> 0
        c_zbl = c_zbl / c_zbl.sum()
        
        # calculate scaled distance 'a' factor
        numbers_i, numbers_j = numbers.index_select(0, idx_i), numbers.index_select(0, idx_j)
        a = zbl_length / (numbers_i ** zbl_pow + numbers_j ** zbl_pow)
        
        # get cutoff radii
        cutoff_radii = self.covalent_radii.index_select(0, numbers_i) + self.covalent_radii.index_select(0, numbers_j)
        if not torch.all(cutoff_radii <= self.r_cutoff):
            raise ValueError(
                f'Found cutoff_radius={torch.max(cutoff_radii).item()} which is larger than the maximal r_cutoff={self.r_cutoff}.'
            )
        
        # lengths_a has shape n_atoms
        lengths_a = lengths / a
        f = torch.sum(c_zbl * torch.exp(-d_zbl * lengths_a.unsqueeze(-1)), -1)
        
        zbl_prefactor = (self.ke_half * numbers_i * numbers_j) / lengths
        
        envelope = PolynomialCutoff.calculate_envelope(lengths, cutoff_radii, self.n_polynomial_cutoff)
        
        pairwise_energies = zbl_prefactor * f * envelope
        zbl_atomic_energies = segment_sum(pairwise_energies, idx_i, numbers.shape[0])
        
        results['zbl_atomic_energies'] = zbl_atomic_energies
        results['atomic_energies'] = results['atomic_energies'] + zbl_atomic_energies
        
        return results
    
    def __repr__(self):
        return f'{self.__class__.__name__}(r_cutoff={self.r_cutoff}, n_polynomial_cutoff={self.n_polynomial_cutoff})'


class ElectrostaticEnergy(nn.Module):
    """
    Basic class for evaluating the electrostatic energy from partial charges.
    
    Args:
        r_cutoff (float, optional): Cutoff radius for the electrostatic interactions. 
                                    If None, an all-to-all neighbor list is used. Defaults to None.
        ke (float, optional): Coulomb constant in the units of [Energy] * [Distance] / [Charge] ** 2. 
                              Defaults to 14.399645351950548 eV * Ang. / e ** 2.
    """
    def __init__(
        self,
        r_cutoff: Optional[float] = None,
        ke: float = 14.399645351950548,
        exclusion_radius: Optional[float] = None,
        n_exclusion_polynomial_cutoff: int = 6,
        **config: Any
    ):
        super(ElectrostaticEnergy, self).__init__()
        self.r_cutoff = r_cutoff
        
        self.ke = ke
        # half the Coulomb constant
        self.ke_half = ke / 2.0
        
        # prepare the computation of the exclusion energy
        self.exclusion_radius = exclusion_radius
        self.n_exclusion_polynomial_cutoff = n_exclusion_polynomial_cutoff
        
        self.cutoff_fn: Optional[nn.Module] = None
        
        if self.exclusion_radius is not None:
            if self.r_cutoff is not None and self.exclusion_radius > self.r_cutoff:
                raise ValueError(
                    f'Exclusion radius {self.exclusion_radius} has to be smaller than or equal to the cutoff radius {self.r_cutoff}.'
                )
            
            # initialize the cutoff function using the exclusion radius
            self.cutoff_fn = PolynomialCutoff(r_cutoff=self.exclusion_radius, p=self.n_exclusion_polynomial_cutoff)
        
    def forward(
        self,
        graph: Dict[str, torch.Tensor],
        results: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Calculates electrostatic contributions to the energy.

        Args:
            graph (Dict[str, torch.Tensor]): Atomic graph dictionary.
            results (Dict[str, torch.Tensor]): Results dictionary.

        Returns:
            Dict[str, torch.Tensor]: Updated results dictionary.
        """
        if 'partial_charges' not in results:
            raise RuntimeError("Partial charges are required to compute the electrostatic energy!")
        
        electrostatic_atomic_energies = self.get_electrostatic_energy(graph, results)
        results['electrostatic_atomic_energies'] = electrostatic_atomic_energies
        results['atomic_energies'] = results['atomic_energies'] + electrostatic_atomic_energies
        
        return results
            
    def get_electrostatic_energy(
        self, 
        graph: Dict[str, torch.Tensor],
        results: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """

        Args:
            graph (Dict[str, torch.Tensor]): Atomic graph dictionary.
            results (Dict[str, torch.Tensor]): Results dictionary.

        Returns:
            torch.Tensor: Electrostatic energy contribution.
        """
        raise NotImplementedError()


class CoulombElectrostaticEnergy(ElectrostaticEnergy):
    """
    Basic class for electrostatic interactions between each pair of particles, with or without the 
    long-range cutoff. The latter is specified by the edge indices.
    """
    def get_electrostatic_energy(
        self, 
        graph: Dict[str, torch.Tensor],
        results: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        edge_index, lengths = graph['edge_index'], graph['lengths'].squeeze()
        idx_i, idx_j = edge_index[0, :], edge_index[1, :]
        
        # long-range atom pairs
        if self.r_cutoff is not None:
            cutoff_mask = lengths <= self.r_cutoff
            lengths, idx_i, idx_j = lengths[cutoff_mask], idx_i[cutoff_mask], idx_j[cutoff_mask]
        
        partial_charges = results['partial_charges']
        
        # compute Coulomb factor
        fac = self.ke_half * partial_charges.index_select(0, idx_i) * partial_charges.index_select(0, idx_j)
        
        # compute Coulomb contributions
        if self.r_cutoff is not None:
            # the force-shifted potential from https://doi.org/10.1063/1.2206581
            coulomb = (
                1.0 / lengths
                + lengths / self.r_cutoff ** 2
                - 2.0 / self.r_cutoff
            )
        else:
            coulomb = 1.0 / lengths
        
        # combine everything and get pairwise, atomic, and total energies
        pairwise_energies = fac * coulomb
        
        if self.exclusion_radius is not None:
            pairwise_energies = pairwise_energies * (1.0 - self.cutoff_fn(lengths))
        
        atomic_energies = segment_sum(pairwise_energies, idx_i, partial_charges.shape[0])
        
        return atomic_energies
    
    def __repr__(self):
        return (f'{self.__class__.__name__}(r_cutoff={self.r_cutoff}, ke={self.ke}, '
                f'exclusion_radius={self.exclusion_radius}, n_exclusion_polynomial_cutoff={self.n_exclusion_polynomial_cutoff})')
        

class EwaldElectrostaticEnergy(ElectrostaticEnergy):
    """
    Basic class for electrostatic interactions computed using Ewald summation.

    Args:
        ewald_alpha (float, optional): 
            The damping factor for Ewald summation, controlling the rate of decay of the real-space part of the potential. 
            Defaults to 1.0.
        ewald_k_max (int, optional): 
            The maximum number of reciprocal lattice vectors (k-space cutoff) used in each direction for Fourier-space summation. 
            Can be specified as a single integer applied to all three directions. Defaults to 10.
    """
    def __init__(
        self,
        alpha: float = 1.0,
        k_max: int = 10,
        **config: Any
    ):
        super(EwaldElectrostaticEnergy, self).__init__(**config)
        assert self.r_cutoff is not None
        
        self.alpha = alpha
        self.alpha_sq = self.alpha ** 2
        
        self.two_pi = 2 * math.pi
        self.sqrt_pi = math.sqrt(math.pi)
        
        assert isinstance(k_max, int)
        self.k_max = k_max
        
        self._setup()

    def _setup(self):
        """
        Generates a three-dimensional meshgrid of reciprocal lattice vectors and symmetry factors for Ewald summation.

        This method creates a three-dimensional grid of lattice points in the reciprocal space, spanning the range 
        defined by `k_max` for each direction. Points are filtered to exclude the origin (0, 0, 0) and those outside 
        the cutoff defined by `k_max`.
        """
        # define k indices
        k_idxs = torch.arange(-self.k_max, self.k_max + 1, 1, dtype=torch.get_default_dtype())
        
        # create a 3D meshgrid of all n_u, n_v, and n_w combinations
        k_u_grid, k_v_grid, k_w_grid = torch.meshgrid(k_idxs[self.k_max:], k_idxs, k_idxs, indexing='ij')
        k_meshgrid = torch.stack([k_u_grid, k_v_grid, k_w_grid], dim=-1)
        
        # define mask to remove k = (0, 0, 0) and |k| > k_max from the meshgrid
        k_meshgrid_sq = k_meshgrid.square().sum(-1)
        mask = (k_meshgrid_sq > 0) & (k_meshgrid_sq <= self.k_max ** 2)
        
        # meshgrid of reciprocal lattice vectors
        self.register_buffer('k_meshgrid', k_meshgrid[mask])
        
        # symmetry factor to account for k_u >= 0 and has shape k_max
        self.register_buffer('sym_factor', torch.where(self.k_meshgrid[:, 0] == 0.0, 1.0, 2.0))
    
    def get_electrostatic_energy(
        self, 
        graph: Dict[str, torch.Tensor],
        results: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        energy_real = self.get_energy_real(graph, results)
        energy_recip = self.get_energy_recip(graph, results)
        return energy_real + energy_recip
    
    def get_energy_real(
        self, 
        graph: Dict[str, torch.Tensor],
        results: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Computes the real-space electrostatic interaction energy for atom pairs.
        
        This method calculates the real-space component of the Coulomb interaction energy between atom pairs 
        based on their partial charges and distances.
        
        Args:
            graph (Dict[str, torch.Tensor]): Dictionary containing atomic data.
            results (Dict[str, torch.Tensor]): Dictionary containing intermediate results.

        Returns:
            torch.Tensor: 
                A tensor of shape `(n_atoms,)`, where each element represents 
                the total real-space electrostatic energy contribution of an atom.
        """
        edge_index, lengths = graph['edge_index'], graph['lengths'].squeeze()
        idx_i, idx_j = edge_index[0, :], edge_index[1, :]
        
        # real-space atom pairs
        cutoff_mask = lengths <= self.r_cutoff
        lengths, idx_i, idx_j = lengths[cutoff_mask], idx_i[cutoff_mask], idx_j[cutoff_mask]
        
        partial_charges = results['partial_charges']
        
        # compute Coulomb factor
        fac = self.ke_half * partial_charges.index_select(0, idx_i) * partial_charges.index_select(0, idx_j)
        
        # compute Coulomb and damped-Coulomb contributions
        coulomb = 1.0 / lengths
        
        # combine everything and get pairwise, atomic, and total energies
        pairwise_energies = fac * coulomb
        
        if self.exclusion_radius is not None:
            pairwise_energies = pairwise_energies * (torch.erfc(self.alpha * lengths) - self.cutoff_fn(lengths))
        else:
            pairwise_energies = pairwise_energies * torch.erfc(self.alpha * lengths)
        
        atomic_energies = segment_sum(pairwise_energies, idx_i, partial_charges.shape[0])
        
        return atomic_energies
    
    def get_energy_recip(
        self, 
        graph: Dict[str, torch.Tensor],
        results: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Computes the reciprocal-part of the electrostatic energy for each atom based on its partial charges. 
        This function employs a k-space sum to compute the electrostatic energy, and returns a tensor of atomic 
        energy contributions.

        Args:
            graph (Dict[str, torch.Tensor]): Dictionary containing atomic data.
            results (Dict[str, torch.Tensor]): Dictionary containing intermediate results.

        Returns:
            torch.Tensor: 
                A tensor of shape `(num_atoms,)`, where each element represents 
                the total reciprocal-space electrostatic energy contribution of an atom.
        """
        positions, cell, batch, n_atoms = graph['positions'], graph['cell'], graph['batch'], graph['n_atoms']
        partial_charges = results['partial_charges']
        
        # compute reciprocal cell vectors
        recip_cell = cell.inverse().permute(0, 2, 1)
        
        # compute product between k meshgrid and the reciprocal cell
        # k_meshgrid has shape n_batch x k_max x 3
        # k_meshgrid_sq has shape n_batch x k_max
        # k_factor has shape n_batch x k_max
        k_meshgrid = torch.einsum('ki, aij -> akj', self.k_meshgrid, self.two_pi * recip_cell)
        k_meshgrid_sq = k_meshgrid.square().sum(-1)
        k_factor = torch.exp(-0.25 * k_meshgrid_sq / self.alpha_sq) / k_meshgrid_sq
        
        # compute product between k-space meshgrid and atom positions
        # k_dot_r has shape n_atoms x k_max
        k_dot_r = torch.einsum('ai, aki -> ak', positions, k_meshgrid.index_select(0, batch))
        
        # real and imaginary parts of the structure factor have shape n_batch x k_max
        sfactor_real = segment_sum(partial_charges.unsqueeze(-1) * torch.cos(k_dot_r), batch, n_atoms.shape[0])
        sfactor_imag = segment_sum(partial_charges.unsqueeze(-1) * torch.sin(k_dot_r), batch, n_atoms.shape[0])
        
        # square of the structure factor has shape n_batch x k_max
        sfactor_sq = sfactor_real.square() + sfactor_imag.square()
        
        energy = torch.sum(k_factor * sfactor_sq * self.sym_factor, dim=-1)
        volume = torch.einsum('bi, bi -> b', cell[:, 0, :], torch.cross(cell[:, 1, :], cell[:, 2, :], dim=1)).abs()
        energy = energy * self.two_pi / volume * self.ke
        
        # re-distribute reciprocal energy over atoms to get atomic contributions
        energy_per_atom = energy / n_atoms
        atomic_energies = energy_per_atom.index_select(0, batch)
        
        # compute and subtract self-energy
        partial_charges_sq = partial_charges.square()
        atomic_energies = atomic_energies - self.ke * self.alpha * partial_charges_sq / self.sqrt_pi
        
        return atomic_energies
    
    def __repr__(self):
        return (f'{self.__class__.__name__}(r_cutoff={self.r_cutoff}, ke={self.ke}, '
                f'alpha={self.alpha}, k_max={self.k_max}, '
                f'exclusion_radius={self.exclusion_radius}, n_exclusion_polynomial_cutoff={self.n_exclusion_polynomial_cutoff})')


class SPMEElectrostaticEnergy(EwaldElectrostaticEnergy):
    """
    Computes the electrostatic energy using the smooth particle mesh Ewald method.
    
    Args:
        spline_order (int): Order of the B-spline.
    """
    def __init__(
        self, 
        spline_order: int = 5,
        **config: Any
    ):
        self.spline_order = spline_order
        super(SPMEElectrostaticEnergy, self).__init__(**config)
        
    def _setup(self):
        # define grid dimension
        self.grid_dim = 2 * self.k_max + 1
        
        # define k values
        k_idxs = torch.cat([
            torch.arange(0, self.k_max + 1, 1, dtype=torch.get_default_dtype()), 
            torch.arange(-self.k_max, 0, 1, dtype=torch.get_default_dtype())
        ])
        
        # create a 3D meshgrid of all n_u, n_v, and n_w combinations
        k_u_grid, k_v_grid, k_w_grid = torch.meshgrid(k_idxs, k_idxs, k_idxs, indexing='ij')
        k_meshgrid = torch.stack([k_u_grid, k_v_grid, k_w_grid], dim=-1)
        
        # meshgrid of reciprocal lattice vectors
        self.register_buffer('k_meshgrid', k_meshgrid)
        
        # compute spline norm
        self.register_buffer('spline_idxs', torch.arange(self.spline_order))
        self.register_buffer('grid_idxs', torch.arange(self.grid_dim))
        
        tmp = torch.zeros(self.grid_dim)
        tmp[:self.spline_order] = bspline(self.spline_idxs, self.spline_order)
        tmp = torch.fft.fft(tmp)
        spline_norm = tmp.real.square() + tmp.imag.square()
        mask = spline_norm < 1e-7
        if mask.any():
            spline_norm[mask] = (spline_norm.roll(shifts=-1)[mask] + spline_norm.roll(shifts=1)[mask]) / 2
        
        self.register_buffer('spline_norm', 1.0 / torch.einsum('u,v,w->uvw', spline_norm, spline_norm, spline_norm))
        
    def get_energy_recip(
        self,
        graph: Dict[str, torch.Tensor],
        results: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        positions, cell, batch, n_atoms = graph['positions'], graph['cell'], graph['batch'], graph['n_atoms']
        partial_charges = results['partial_charges']
        
        # compute reciprocal cell vectors
        recip_cell = cell.inverse().permute(0, 2, 1)
        
        # compute product between k meshgrid and the reciprocal cell
        k_meshgrid = torch.einsum('uvwi,aij -> auvwj', self.k_meshgrid, self.two_pi * recip_cell)
        k_meshgrid_sq = k_meshgrid.square().sum(-1)
        
        # compute the influence function and its product with the spline norm
        mask = (k_meshgrid_sq == 0.0)
        k_factor = torch.zeros_like(k_meshgrid_sq)
        k_factor[~mask] = torch.exp(-0.25 * k_meshgrid_sq[~mask] / self.alpha_sq) / k_meshgrid_sq[~mask]
        kb_factor = torch.einsum('auvw,uvw->auvw', k_factor, self.spline_norm)
        
        # compute fractional coordinates
        positions_frac = torch.einsum('aij,aj-> ai', recip_cell.index_select(0, batch), positions)
        positions_frac = positions_frac - torch.floor(positions_frac) # ensure that the fractional positions always stay within the unit cell
        
        # spread charges on the grid
        grid_start = (self.grid_dim * positions_frac).to(torch.long)
        wvals = self.grid_dim * positions_frac - grid_start
        splines = [bspline(wvals[:, i][:, None] + self.spline_idxs[None], self.spline_order).flip(-1) for i in range(3)]
        
        grid_iter = ((self.spline_idxs + self.grid_idxs[:, None]) % self.grid_dim)
        x_indices, y_indices, z_indices = [grid_iter[grid_start[:, i] % self.grid_dim] for i in range(3)]

        x_indices = x_indices.unsqueeze(2).unsqueeze(3).expand(-1, -1, self.spline_order, self.spline_order)
        y_indices = y_indices.unsqueeze(1).unsqueeze(3).expand(-1, self.spline_order, -1, self.spline_order)
        z_indices = z_indices.unsqueeze(1).unsqueeze(2).expand(-1, self.spline_order, self.spline_order, -1)
        
        Qgrid_comps = torch.einsum('a,au,av,aw->auvw', partial_charges, *splines)
        
        batch_expanded = batch.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        Qgrid = torch.zeros((n_atoms.shape[0], *[self.grid_dim] * 3), dtype=Qgrid_comps.dtype, device=Qgrid_comps.device)
        Qgrid = torch.index_put(Qgrid, (batch_expanded, x_indices, y_indices, z_indices), Qgrid_comps, accumulate=True)
        
        # forward fft to get the structure factor
        Sm = torch.fft.fftn(Qgrid, dim=(1, 2, 3))
        energy = torch.sum(kb_factor * (Sm.real.square() + Sm.imag.square()), dim=(1, 2, 3))
        
        volume = torch.einsum('bi, bi -> b', cell[:, 0, :], torch.cross(cell[:, 1, :], cell[:, 2, :], dim=1)).abs()
        energy = energy * self.two_pi  / volume * self.ke
        
        # re-distribute reciprocal energy over atoms to get atomic contributions
        energy_per_atom = energy / n_atoms
        atomic_energies = energy_per_atom.index_select(0, batch)
        
        # compute and subtract self-energy
        partial_charges_sq = partial_charges.square()
        atomic_energies = atomic_energies - self.ke * self.alpha * partial_charges_sq / self.sqrt_pi
        
        return atomic_energies
    
    def __repr__(self):
        return (f'{self.__class__.__name__}(r_cutoff={self.r_cutoff}, ke={self.ke}, alpha={self.alpha}, '
                f'k_max={self.k_max}, grid_dim={self.grid_dim}, spline_order={self.spline_order}, '
                f'exclusion_radius={self.exclusion_radius}, n_exclusion_polynomial_cutoff={self.n_exclusion_polynomial_cutoff})')


class D4DispersionEnergy(nn.Module):
    """
    Computes dispersion energy using Grime's D4 model.
    
    Some parts are adapted from SpookyNet (https://github.com/OUnke/SpookyNet/blob/main/spookynet/modules/d4_dispersion_energy.py).
    
    Args:
        Z_max (int): The maximum atomic number. Grimme's D4 dispersion is only parametrized 
                     up to Rn (Z = 86). Defaults to 87.
    """
    def __init__(
        self,
        r_cutoff: Optional[float] = None,
        Z_max: int = 86,
        Bohr: float = 0.5291772105638411,
        Hartree: float = 27.211386024367243,
        exclusion_radius: Optional[float] = None,
        n_exclusion_polynomial_cutoff: int = 6,
        **config: Any
    ):
        super(D4DispersionEnergy, self).__init__()
        # ensure Z_max is within supported range
        assert (Z_max + 1) <= 87, f"Grimme's D4 dispersion is only parametrized up to Rn (Z = 86). Provided {(Z_max + 1)=}."
        
        self.r_cutoff = r_cutoff
        self.Z_max = Z_max
        self.to_Bohr = 1.0 / Bohr
        self.to_eV = Hartree
        self.two_pi = 2 * math.pi
        
        # define switching parameters
        if self.r_cutoff is not None:
            self.r_cutoff *= self.to_Bohr
            self.switch_off = r_cutoff * self.to_Bohr
            self.switch_on = self.switch_off - 1.0 / self.to_Bohr
        else:
            self.switch_on, self.switch_off = None, None
        
        # coefficients for coordination number calculations
        self.k4, self.k5, self.k6, self.k = 4.10451, 19.08857, 254.5553148552, 7.5
        
        # non-trainable parameter for s6
        s6 = torch.tensor([1.0], dtype=torch.get_default_dtype())
        self.register_buffer('s6', softplus_inverse(s6))
        
        # trainable parameters initialized to literature values 
        # HF-values from https://github.com/dftd4/dftd4/blob/main/src/dftd4/param.f90
        self.s8 = nn.Parameter(softplus_inverse([1.61679827]))
        self.a1 = nn.Parameter(softplus_inverse([0.44959224]))
        self.a2 = nn.Parameter(softplus_inverse([3.35743605]))
        self.scale_q = nn.Parameter(softplus_inverse(1.0))
        
        # load necessary D4 data files
        # NOTE: added `torch.get_default_device()` because of a different device handling by DIMOS
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'd4data')
        self.register_buffer('refsys', torch.load(os.path.join(data_dir, 'refsys.pth'), weights_only=True).to(torch.get_default_device(), torch.long))                                                    # shape: 87 x max_nref
        self.register_buffer('zeff', torch.load(os.path.join(data_dir, 'zeff.pth'), weights_only=True).to(torch.get_default_device(), torch.get_default_dtype()))                                         # shape: 87
        self.register_buffer('refh', torch.load(os.path.join(data_dir, 'refh.pth'), weights_only=True).to(torch.get_default_device(), torch.get_default_dtype()))                                         # shape: 87 x max_nref
        self.register_buffer('sscale', torch.load(os.path.join(data_dir, 'sscale.pth'), weights_only=True).to(torch.get_default_device(), torch.get_default_dtype()))                                     # shape: 18
        self.register_buffer('secaiw', torch.load(os.path.join(data_dir, 'secaiw.pth'), weights_only=True).to(torch.get_default_device(), torch.get_default_dtype()))                                     # shape: 18 x 23
        self.register_buffer('gam', torch.load(os.path.join(data_dir, 'gam.pth'), weights_only=True).to(torch.get_default_device(), torch.get_default_dtype()))                                           # shape: 87
        self.register_buffer('ascale', torch.load(os.path.join(data_dir, 'ascale.pth'), weights_only=True).to(torch.get_default_device(), torch.get_default_dtype()))                                     # shape: 87 x max_nref
        self.register_buffer('alphaiw', torch.load(os.path.join(data_dir, 'alphaiw.pth'), weights_only=True).to(torch.get_default_device(), torch.get_default_dtype()))                                   # shape: 87 x max_nref x 23
        self.register_buffer('hcount', torch.load(os.path.join(data_dir, 'hcount.pth'), weights_only=True).to(torch.get_default_device(), torch.get_default_dtype()))                                     # shape: 87 x max_nref
        self.register_buffer('casimir_polder_weights', torch.load(os.path.join(data_dir, 'casimir_polder_weights.pth'), weights_only=True).to(torch.get_default_device(), torch.get_default_dtype()))     # shape: 23
        self.register_buffer('rcov', torch.load(os.path.join(data_dir, 'rcov.pth'), weights_only=True).to(torch.get_default_device(), torch.get_default_dtype()))                                         # shape: 87
        self.register_buffer('en', torch.load(os.path.join(data_dir, 'en.pth'), weights_only=True).to(torch.get_default_device(), torch.get_default_dtype()))                                             # shape: 87
        self.register_buffer('ncount_mask', torch.load(os.path.join(data_dir, 'ncount_mask.pth'), weights_only=True).to(torch.get_default_device(), torch.get_default_dtype()))                           # shape: 87 x max_nref x max_ncount
        self.register_buffer('ncount_weight', torch.load(os.path.join(data_dir, 'ncount_weight.pth'), weights_only=True).to(torch.get_default_device(), torch.get_default_dtype()))                       # shape: 87 x 87 x max_nref x max_ncount
        self.register_buffer('cn', torch.load(os.path.join(data_dir, 'cn.pth'), weights_only=True).to(torch.get_default_device(), torch.get_default_dtype()))                                             # shape: 87 x max_nref x max_ncount
        self.register_buffer('fixgweights', torch.load(os.path.join(data_dir, 'fixgweights.pth'), weights_only=True).to(torch.get_default_device(), torch.get_default_dtype()))                           # shape: 87 x max_nref
        self.register_buffer('refq',  torch.load(os.path.join(data_dir, 'refq.pth'), weights_only=True).to(torch.get_default_device(), torch.get_default_dtype()))                                        # shape: 87 x max_nref
        self.register_buffer('sqrt_r4r2', torch.load(os.path.join(data_dir, 'sqrt_r4r2.pth'), weights_only=True).to(torch.get_default_device(), torch.get_default_dtype()))                               # shape: 87
        
        # initial computation of refc6
        self.register_buffer('refc6', self._compute_refc6())
        
        # prepare the computation of the exclusion energy
        self.exclusion_radius = exclusion_radius
        self.n_exclusion_polynomial_cutoff = n_exclusion_polynomial_cutoff
        
        self.cutoff_fn: Optional[nn.Module] = None
        
        if self.exclusion_radius is not None:
            if self.r_cutoff is not None and self.exclusion_radius > self.r_cutoff:
                raise RuntimeError(
                    f'Exclusion radius {self.exclusion_radius} has to be smaller than or equal to the cutoff radius {self.r_cutoff}.'
                )
            
            # initialize the cutoff function using the exclusion radius
            self.cutoff_fn = PolynomialCutoff(r_cutoff=self.exclusion_radius, p=self.n_exclusion_polynomial_cutoff)
    
    def _compute_refc6(self) -> torch.Tensor:
        """Compute the reference C6 dispersion coefficients."""
        with torch.no_grad():
            species_idxs = torch.arange(self.Z_max + 1, device=self.refsys.device)
            ref_idxs = self.refsys[species_idxs, :]
            
            zeff_ref = self.zeff[ref_idxs].unsqueeze(-1)
            sscale_ref = self.sscale[ref_idxs].unsqueeze(-1)
            secaiw_ref = self.secaiw[ref_idxs]
            gam_ref = self.gam[ref_idxs].unsqueeze(-1)
            
            refh_i = self.refh[species_idxs].unsqueeze(-1) * F.softplus(self.scale_q)
            ascale_i = self.ascale[species_idxs].unsqueeze(-1)
            alphaiw_i = self.alphaiw[species_idxs]
            hcount_i = self.hcount[species_idxs].unsqueeze(-1)
            
            qmod = zeff_ref + refh_i
            qmod_safe = torch.where(qmod > 1e-8, qmod, torch.ones_like(qmod))
            
            alpha = sscale_ref * secaiw_ref * torch.where(
                qmod > 1e-8, 
                torch.exp(3.0 * (1.0 - torch.exp(2.0 * gam_ref * (1.0 - zeff_ref / qmod_safe)))), 
                math.exp(3.0) * torch.ones_like(qmod)
            )
            alpha = torch.max(ascale_i * (alphaiw_i - hcount_i * alpha), torch.zeros_like(alpha))
            
            alpha_expanded = alpha.unsqueeze(1).unsqueeze(3) * alpha.unsqueeze(0).unsqueeze(2)
            casimir_polder_weights = self.casimir_polder_weights.unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            
            return 3.0 / self.two_pi * torch.sum(alpha_expanded * casimir_polder_weights, -1)
        
    def _update_refc6(self):
        """Update the reference C6 dispersion coefficients."""
        self.refc6 = self._compute_refc6()
    
    def forward(
        self, 
        graph: Dict[str, torch.Tensor],
        results: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute dispersion energies and updates the total energy provided in `results`.
        
        Args:
            graph (Dict[str, torch.Tensor]): Atomic graph dictionary.
            results (Dict[str, torch.Tensor]): Results dictionary.

        Returns:
            Dict[str, torch.Tensor]: Updated results dictionary.
        """
        # NOTE: we use atomic numbers instead of internal species for the d4 dispersion model
        species, edge_index, lengths, batch = graph['atomic_numbers'], graph['edge_index'], graph['lengths'].squeeze(), graph['batch']
        idx_i, idx_j = edge_index[0, :], edge_index[1, :]
        
        # convert lengths to atomic units (Bohr)
        lengths = lengths * self.to_Bohr
        
        # apply cutoff if needed
        if self.r_cutoff is not None:
            cutoff_mask = (lengths <= self.r_cutoff)
            lengths, idx_i, idx_j = lengths[cutoff_mask], idx_i[cutoff_mask], idx_j[cutoff_mask]
        
        # expand partial charges
        partial_charges = results['partial_charges'].unsqueeze(-1)
        
        # retrieve species-specific properties
        species_i, species_j = species.index_select(0, idx_i), species.index_select(0, idx_j)
        en_i, en_j = self.en.index_select(0, species_i), self.en.index_select(0, species_j)
        rcov_i, rcov_j = self.rcov.index_select(0, species_i), self.rcov.index_select(0, species_j)
        sqrt_r4r2_i, sqrt_r4r2_j = self.sqrt_r4r2.index_select(0, species_i), self.sqrt_r4r2.index_select(0, species_j)

        # coordination number calculation
        rcov = 4.0 / 3.0 * (rcov_i + rcov_j)
        den = self.k4 * torch.exp(-(torch.abs(en_i - en_j) + self.k5) ** 2 / self.k6)
        countf = den * 0.5 * (1.0 + torch.erf(-self.k * (lengths - rcov) / rcov))
        if self.r_cutoff is not None:
            countf = countf * switch_function(lengths, self.switch_on, self.switch_off)
        ncoord_erf = segment_sum(countf, idx_i, batch.shape[0]).unsqueeze(-1).unsqueeze(-2)

        # Gaussian weights
        gweights = torch.exp(-6.0 * self.ncount_weight[species] * (ncoord_erf - self.cn[species]) ** 2)
        gweights = torch.sum(self.ncount_mask[species] * gweights, dim=-1)
        norm = torch.sum(gweights, dim=-1, keepdim=True).clamp(min=1e-7)
        gweights = torch.where(norm > 1e-8, gweights / norm, self.fixgweights[species])

        # charge-scaling function
        refq_species = self.refq.index_select(0, species) * F.softplus(self.scale_q)
        qref = self.zeff[species].unsqueeze(-1) + refq_species
        qmod = self.zeff[species].unsqueeze(-1) + partial_charges
        qmod_safe = torch.where(qmod > 1e-8, qmod, torch.ones_like(qmod))
        
        exp_factor = 3.0 * (1.0 - torch.exp(2.0 * self.gam[species].unsqueeze(-1) * (1.0 - qref / qmod_safe)))
        zeta = torch.where(qmod > 1e-8, torch.exp(exp_factor), math.exp(3.0) * torch.ones_like(qmod)) * gweights
        zeta_i, zeta_j = zeta.index_select(0, idx_i), zeta.index_select(0, idx_j)
        
        # compute dispersion energy
        refc6ij = self.refc6[species_i, species_j]
        zetaij = zeta_i.unsqueeze(2) * zeta_j.unsqueeze(1)
        c6ij = torch.sum(refc6ij * zetaij, dim=(1, 2))
        sqrt_r4r2ij = sqrt_r4r2_i * sqrt_r4r2_j * (3 ** 0.5)
        
        # r0 calculation using trainable parameters a1 and a2
        a1, a2 = F.softplus(self.a1), F.softplus(self.a2)
        r0 = a1 * sqrt_r4r2ij + a2
        
        # r6 and r8 with cutoff if applicable
        lengths_pow6, r0_pow6 = lengths ** 6, r0 ** 6
        lengths_pow8, r0_pow8 = lengths ** 8, r0 ** 8
        if self.r_cutoff is not None:
            r_cutoff_pow6, r_cutoff_pow8 = self.r_cutoff ** 6, self.r_cutoff ** 8
            r6 = (
                1.0 / (lengths_pow6 + r0_pow6) 
                - 1.0 / (r_cutoff_pow6 + r0_pow6) 
                + 6 * r_cutoff_pow6 / (r_cutoff_pow6 + r0_pow6) ** 2 * (lengths / self.r_cutoff - 1.0)
            )
            r8 = (
                1.0 / (lengths_pow8 + r0_pow8) 
                - 1.0 / (r_cutoff_pow8 + r0_pow8) 
                + 8 * r_cutoff_pow8 / (r_cutoff_pow8 + r0_pow8) ** 2 * (lengths / self.r_cutoff - 1.0)
            )
        else:
            r6 = 1.0 / (lengths_pow6 + r0_pow6)
            r8 = 1.0 / (lengths_pow8 + r0_pow8)
        
        # dispersion energy with s6 and s8 factors
        s6, s8 = F.softplus(self.s6), F.softplus(self.s8)
        pairwise_energies = -c6ij * (s6 * r6 + s8 * sqrt_r4r2ij ** 2 * r8) * self.to_eV
        
        if self.exclusion_radius is not None:
            pairwise_energies = pairwise_energies * (1.0 - self.cutoff_fn(lengths))
        
        disp_atomic_energies = segment_sum(pairwise_energies, idx_i, batch.shape[0])
        
        # update results
        results['disp_atomic_energies'] = disp_atomic_energies
        results['atomic_energies'] = results['atomic_energies'] + disp_atomic_energies
        
        return results
    
    def __repr__(self):
        return (f'{self.__class__.__name__}(r_cutoff={self.r_cutoff / self.to_Bohr}, Z_max={self.Z_max}, '
                f'exclusion_radius={self.exclusion_radius}, n_exclusion_polynomial_cutoff={self.n_exclusion_polynomial_cutoff})')
