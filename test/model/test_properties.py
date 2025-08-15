from ase.io import read

import numpy as np

import torch
import torch.nn as nn

from ictp.data.data import AtomicStructures

from ictp.model.forward import ForwardAtomisticNetwork
from ictp.model.calculators import StructurePropertyCalculator
from ictp.model.partial_charges import CorrectedPartialCharges, EquilibratedPartialCharges

from ictp.nn.representations import CartesianMACE
from ictp.nn.layers import LinearLayer, RescaledSiLULayer, ScaleShiftLayer

from ictp.utils.torch_geometric import DataLoader


def test_partial_charges(device: str = 'cuda:1'):
    structures = AtomicStructures.from_file('spce/spce.extxyz')
    for structure in structures:
        structure.total_charge = 0.0
    
    ds = structures.to_data(r_cutoff=5.0, n_species=10)
    dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)
    
    representation = CartesianMACE(r_cutoff=5.0, n_basis=8, n_polynomial_cutoff=5,
                                   n_species=10, n_hidden_feats=16, n_product_feats=16, 
                                   coupled_product_feats=False, symmetric_product=True,
                                   l_max_hidden_feats=1, l_max_edge_attrs=3, 
                                   avg_n_neighbors=1, correlation=3, n_interactions=2,
                                   radial_MLP=[64,64,64], use_charge_embedding=True)
    

    readouts = nn.ModuleList([])
    for i in range(2):
        if i == 1:
            layers = []
            for in_size, out_size in zip([16] + [16, 16],
                                         [16, 16] + [2]):
                layers.append(LinearLayer(in_size, out_size))
                layers.append(RescaledSiLULayer())
            readouts.append(nn.Sequential(*layers[:-1]))
        else:
            readouts.append(LinearLayer(16, 2))
    
    scale_shift = ScaleShiftLayer(shift_params=np.zeros(10), scale_params=np.ones(10))
    
    model = ForwardAtomisticNetwork(representation=representation,
                                    readouts=readouts,
                                    scale_shift=scale_shift,
                                    partial_charges=CorrectedPartialCharges(),
                                    # partial_charges=EquilibratedPartialCharges(),
                                    pair_potentials=nn.ModuleList([]),
                                    config={'none': None})
    
    model = model.to(device)
    
    for batch in dl:
        results = model(batch.to(device).to_dict())
        # TODO: check how it works for a trained model
        assert sum(results['partial_charges']).abs() < 1e-7
    

# def test_quadrupole_moment(device: str = 'cuda:0'):
#     structures = AtomicStructures.from_file('spce/spce.extxyz')
#     for structure in structures:
#         structure.total_charge = 0.0
    
#     ds = structures.to_data(r_cutoff=5.0, n_species=10)
#     dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)
    
#     representation = CartesianMACE(r_cutoff=5.0, n_basis=8, n_polynomial_cutoff=5,
#                                    n_species=10, n_hidden_feats=16, n_product_feats=16, 
#                                    coupled_product_feats=False, symmetric_product=True,
#                                    l_max_hidden_feats=1, l_max_edge_attrs=3, 
#                                    avg_n_neighbors=1, correlation=3, n_interactions=2,
#                                    radial_MLP=[64,64,64], use_charge_embedding=True)
    

#     readouts = nn.ModuleList([])
#     for i in range(2):
#         if i == 1:
#             layers = []
#             for in_size, out_size in zip([16] + [16, 16],
#                                          [16, 16] + [2]):
#                 layers.append(LinearLayer(in_size, out_size))
#                 layers.append(RescaledSiLULayer())
#             readouts.append(nn.Sequential(*layers[:-1]))
#         else:
#             readouts.append(LinearLayer(16, 2))
    
#     scale_shift = ScaleShiftLayer(shift_params=np.zeros(10), scale_params=np.ones(10))
    
#     model = ForwardAtomisticNetwork(representation=representation,
#                                     readouts=readouts,
#                                     scale_shift=scale_shift,
#                                     partial_charges=CorrectedPartialCharges(),
#                                     # partial_charges=EquilibratedPartialCharges(),
#                                     pair_potentials=nn.ModuleList([]),
#                                     config={'none': None})
    
#     calc = StructurePropertyCalculator(model)
#     calc = calc.to(device)
    
#     for batch in dl:
#         results = calc(batch.to(device), forces=True, dipole_moment=True, quadrupole_moment=True)
#         assert abs(np.trace(results['quadrupole_moment'].cpu().detach().numpy()[0])) < 1e-7


if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    
    test_partial_charges()
    # test_quadrupole_moment()