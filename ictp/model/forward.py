"""
       ICTP: Irreducible Cartesian Tensor Potentials
	  
  File:     forward.py
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
import os
from pathlib import Path

from typing import *

import yaml

import numpy as np

import torch
import torch.nn as nn

from ictp.data.data import AtomicStructures
from ictp.data.tools import get_avg_n_neighbors, get_energy_shift_per_atom, get_forces_rms

from ictp.nn.layers import LinearLayer, RescaledSiLULayer, ScaleShiftLayer
from ictp.nn.representations import CartesianMACE

from ictp.utils.misc import load_object, save_object


def build_model(atomic_structures: Optional[AtomicStructures] = None,
                **config: Any):
    """Builds feed-forward atomistic neural network from the config file.

    Args:
        atomic_structures (Optional[AtomicStructures], optional): Atomic structures, typically those from the training data sets, 
                                                                  used to compute scale and shift for model predictions as well as 
                                                                  the average number of neighbors. Defaults to None.
        
    Returns:
        ForwardAtomisticNetwork: Atomistic neural network.
    """
    torch.manual_seed(config['model_seed'])
    np.random.seed(config['model_seed'])
    
    # compute scale/shift parameters for the total energy
    # also, compute the average number of neighbors, i.e., the normalization factor for messages
    if atomic_structures is None:
        shift_params = np.zeros(config['n_species'])
        scale_params = np.ones(config['n_species'])
    else:
        shift_params = get_energy_shift_per_atom(atomic_structures, n_species=config['n_species'], atomic_energies=config['atomic_energies'],
                                                 compute_regression_shift=config['compute_regression_shift'])
        scale_params = get_forces_rms(atomic_structures, n_species=config['n_species'])
        if config['compute_avg_n_neighbors']:
            config['avg_n_neighbors'] = get_avg_n_neighbors(atomic_structures, config['r_cutoff']).item()

    # prepare (semi-)local atomic representation
    representation = CartesianMACE(**config)

    # prepare readouts
    readouts = nn.ModuleList([])
    for i in range(config['n_interactions']):
        if i == config['n_interactions'] - 1:
            layers = []
            for in_size, out_size in zip([config['n_hidden_feats']] + config['readout_MLP'],
                                        config['readout_MLP'] + [1]):
                layers.append(LinearLayer(in_size, out_size))
                layers.append(RescaledSiLULayer())
            readouts.append(nn.Sequential(*layers[:-1]))
        else:
            readouts.append(LinearLayer(config['n_hidden_feats'], 1))

    scale_shift = ScaleShiftLayer(shift_params=shift_params, scale_params=scale_params)

    return ForwardAtomisticNetwork(representation=representation, readouts=readouts, scale_shift=scale_shift, config=config)


class ForwardAtomisticNetwork(nn.Module):
    """An atomistic model based on feed-forward neural networks.

    Args:
        representation (nn.Module): Local atomic representation layer.
        readouts (nn.ModuleList): List of readout layers.
        scale_shift (nn.Module): Schale/shift transformation applied to the output, i.e., energy re-scaling and shift.
    """
    def __init__(self,
                 representation: nn.Module,
                 readouts: List[nn.Module],
                 scale_shift: nn.Module,
                 config: Dict[str, Any]):
        super().__init__()
        # all necessary modules
        self.representation = representation
        self.readouts = readouts
        self.scale_shift = scale_shift
        
        # provide config file to store it
        self.config = config

    def forward(self, graph: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Computes atomic energies for the provided batch.

        Args:
            graph (Dict[str, torch.Tensor]): Atomic data dictionary.

        Returns:
            torch.Tensor: Atomic energies.
        """
        # compute representation (atom/node features)
        atom_feats_list = self.representation(graph)
        
        # apply a readout layer to each representation
        atomic_energies_list = []
        for i, readout in enumerate(self.readouts):
            atomic_energies_list.append(readout(atom_feats_list[i]).squeeze(-1))
        atomic_energies = torch.sum(torch.stack(atomic_energies_list, dim=0), dim=0)
        
        # scale and shift the output
        atomic_energies = self.scale_shift(atomic_energies, graph)
        
        return atomic_energies

    def get_device(self) -> str:
        """Provides device on which calculations are performed.
        
        Returns: 
            str: Device on which calculations are performed.
        """
        return list(self.representation.parameters())[0].device

    def load_params(self, file_path: Union[str, Path]):
        """Loads network parameters from the file.

        Args:
            file_path (Union[str, Path]): Path to the file where network parameters are stored.
        """
        self.load_state_dict(load_object(file_path))

    def save_params(self, file_path: Union[str, Path]):
        """Stores network parameters to the file.

        Args:
            file_path (Union[str, Path]): Path to the file where network parameters are stored.
        """
        save_object(file_path, self.state_dict())

    def save(self, folder_path: Union[str, Path]):
        """Stores config and network parameters to the file.

        Args:
            folder_path (Union[str, Path]): Path to the folder where network parameters are stored.
        """
        (Path(folder_path) / 'config.yaml').write_text(str(yaml.safe_dump(self.config)))
        self.save_params(Path(folder_path) / 'params.pkl')

    @staticmethod
    def from_folder(folder_path: Union[str, Path]) -> 'ForwardAtomisticNetwork':
        """Loads model from the defined folder.

        Args:
            folder_path (Union[str, Path]): Path to the folder where network parameters are stored.

        Returns:
            ForwardAtomisticNetwork: The `ForwardAtomisticNetwork` object.
        """
        config = yaml.safe_load((Path(folder_path) / 'config.yaml').read_text())
        nn = build_model(None, **config)
        nn.load_params(Path(folder_path) / 'params.pkl')
        return nn


def find_last_ckpt(folder: Union[Path, str]):
    """Finds the last/best checkpoint to load the model from.

    Args:
        folder (Union[Path, str]): Path to the folder where checkpoints are stored.

    Returns:
        Last checkpoint to load the model from.
    """
    # if no checkpoint exists raise an error
    files = list(Path(folder).iterdir())
    if len(files) == 0:
        raise RuntimeError(f'Provided {folder} which is empty.')
    if len(files) >= 2:
        folders = [f for f in files if f.name.startswith('ckpt_')]
        file_epoch_numbers = [int(f.name[5:]) for f in folders]
        newest_file_idx = np.argmax(np.asarray(file_epoch_numbers))
        return folders[newest_file_idx]
    else:
        return files[0]


def load_model_from_folder(model_path: Union[str, Path],
                           key: str = 'best') -> ForwardAtomisticNetwork:
    """Loads model from the provided folder.

    Args:
        model_path (Union[str, Path]): Path to the model.
        key (str, optional): Choose which model to select, the best or last stored one: 'best' and 'log'. 
                             Defaults to 'best'.

    Returns:
        ForwardAtomisticNetwork: Atomistic model.
    """
    path = os.path.join(model_path, key)
    if not os.path.exists(path):
        raise RuntimeError(f'Provided path to the {key} model does not exist: {path=}.')
    return ForwardAtomisticNetwork.from_folder(find_last_ckpt(path))
