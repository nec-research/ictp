from ase.io import read

import numpy as np

import torch
import torch.nn as nn

from ictp.data.data import AtomicStructure, AtomicStructures, AtomicData

from ictp.model.pair_potentials import (ZBLRepulsionEnergy, CoulombElectrostaticEnergy,
                                        EwaldElectrostaticEnergy, SPMEElectrostaticEnergy,
                                        D4DispersionEnergy)

from ictp.utils.torch_geometric import DataLoader
from ictp.utils.math import segment_sum, softplus_inverse

import time

import matplotlib.pyplot as plt

from ase import Atoms
from ase.build import molecule

from dftd4.ase import DFTD4


def test_ZBLRepulsionEnergy_dimer(device: str = 'cuda:0'):
    n_polynomial_cutoff = 3
    ke = 14.399645351950548
    
    zbl_model = ZBLRepulsionEnergy(r_cutoff=5.2, n_polynomial_cutoff=n_polynomial_cutoff, ke=ke)
    
    # zbl_model.c_zbl = nn.Parameter(softplus_inverse([0.2167, 0.5928, 0.2359, 0.0231]))
    # zbl_model.d_zbl = nn.Parameter(softplus_inverse([3.3369, 1.0562, 0.4743, 0.2397]))
    # zbl_model.zbl_pow = nn.Parameter(softplus_inverse([0.2776]))
    # zbl_model.zbl_length = nn.Parameter(softplus_inverse([0.3988]))
    
    distances = torch.linspace(0.1, 2.5, steps=100)
    
    # species = torch.tensor([1, 8])
    species = torch.tensor([1, 1])
    edge_index = torch.tensor([[0, 1], [1, 0]])
    
    graph = {
        'species': species,
        'atomic_numbers': species,
        'edge_index': edge_index,
        'lengths': distances.unsqueeze(-1)
    }
    
    results = {'atomic_energies': torch.zeros(2)}
    
    energies = []
    for length in distances:
        graph['lengths'] = torch.tensor([[length, length]])
        results = zbl_model(graph, results)
        energies.append(results['zbl_atomic_energies'].detach())
    
    energies = torch.stack(energies)
    
    plt.figure(figsize=(8, 6))
    plt.ylim(0, 3)
    plt.plot(distances.numpy(), energies[:, 0].numpy())
    plt.xlabel("Distance in Angstrom")
    plt.ylabel("ZBL repulsion energy in eV")
    plt.savefig('zbl_repulsion_energy.pdf')


def test_CoulombElectrostaticEnergy_dimer_no_cutoff(device: str = 'cuda:0'):
    r_cutoff = None
    ke = 14.399645351950548
    
    model = CoulombElectrostaticEnergy(
        r_cutoff=r_cutoff,
        ke=ke,
        exclusion_radius=None,
        n_exclusion_polynomial_cutoff=3,
    )
    
    distances = torch.linspace(0.1, 11.0, steps=100)
    
    species = torch.tensor([1, 1])
    edge_index = torch.tensor([[0, 1], [1, 0]])
    
    graph = {
        'species': species,
        'edge_index': edge_index,
        'lengths': distances.unsqueeze(-1)
    }
    
    results = {'atomic_energies': torch.zeros(2),
               'partial_charges': torch.tensor([1., -1.])}
    
    energies = []
    for length in distances:
        graph['lengths'] = torch.tensor([[length, length]])
        results = model(graph, results)
        energies.append(results['electrostatic_atomic_energies'].detach())
    
    energies = torch.stack(energies)
    
    plt.figure(figsize=(8, 6))
    plt.plot(distances.numpy(), energies[:, 0].numpy())
    
    results = {'atomic_energies': torch.zeros(2),
               'partial_charges': torch.tensor([1., 1.])}
    
    energies = []
    for length in distances:
        graph['lengths'] = torch.tensor([[length, length]])
        results = model(graph, results)
        energies.append(results['electrostatic_atomic_energies'].detach())
    
    energies = torch.stack(energies)
    
    plt.plot(distances.numpy(), energies[:, 0].numpy())
    
    plt.xlabel("Distance in Angstrom")
    plt.ylabel("Coulomb energy in eV")
    plt.savefig('coulomb_energy_no_cutoff.pdf')


def test_CoulombElectrostaticEnergy_dimer_cutoff(device: str = 'cuda:0'):
    r_cutoff = 10.0
    ke = 14.399645351950548
    
    model = CoulombElectrostaticEnergy(
        r_cutoff=r_cutoff,
        ke=ke,
        exclusion_radius=None,
        n_exclusion_polynomial_cutoff=3
    )
    
    distances = torch.linspace(0.1, 11.0, steps=100)
    
    species = torch.tensor([1, 1])
    edge_index = torch.tensor([[0, 1], [1, 0]])
    
    graph = {
        'species': species,
        'edge_index': edge_index,
        'lengths': distances.unsqueeze(-1)
    }
    
    results = {'atomic_energies': torch.zeros(2),
               'partial_charges': torch.tensor([1., -1.])}
    
    energies = []
    for length in distances:
        graph['lengths'] = torch.tensor([[length, length]])
        results = model(graph, results)
        energies.append(results['electrostatic_atomic_energies'].detach())
    
    energies = torch.stack(energies)
    
    plt.figure(figsize=(8, 6))
    plt.plot(distances.numpy(), energies[:, 0].numpy())
    
    results = {'atomic_energies': torch.zeros(2),
               'partial_charges': torch.tensor([1., 1.])}
    
    energies = []
    for length in distances:
        graph['lengths'] = torch.tensor([[length, length]])
        results = model(graph, results)
        energies.append(results['electrostatic_atomic_energies'].detach())
    
    energies = torch.stack(energies)
    
    plt.plot(distances.numpy(), energies[:, 0].numpy())
    
    plt.xlabel("Distance in Angstrom")
    plt.ylabel("Coulomb energy in eV")
    plt.savefig('coulomb_energy_cutoff.pdf')


def test_EwaldElectrostaticEnergy(device: str = 'cuda:0'):
    
    model = EwaldElectrostaticEnergy(
        r_cutoff=10.0,
        ke=14.399645351950548,
        alpha=0.3,
        k_max=30,
        exclusion_radius=5.0,
        n_exclusion_polynomial_cutoff=3
    )
    model = model.to(device)
    
    model_coulomb = CoulombElectrostaticEnergy(
        r_cutoff=None,
        ke=14.399645351950548,
        exclusion_radius=5.0,
        n_exclusion_polynomial_cutoff=3
    )
    model_coulomb = model_coulomb.to(device)
    
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
    atoms.set_initial_charges([1.0, -1.0])
    
    atoms.set_cell([80, 80, 200])
    atoms.center()
    atoms.set_pbc(True)
    
    distances = np.linspace(0.1, 9.0, 100)
    
    energies_ewald = []
    energies_coulomb = []
    for d in distances:
        
        atoms.set_distance(0, 1, d, fix=0)
        
        structure = AtomicStructure.from_atoms(atoms)
    
        dl = DataLoader([AtomicData(structure, r_cutoff=10.0)], batch_size=1, shuffle=False, drop_last=False)
        batch = next(iter(dl)).to(device)
        
        batch.positions.requires_grad = True
        
        graph = batch.to_dict()
        
        edge_index, positions, shifts = graph['edge_index'], graph['positions'], graph['shifts']
    
        idx_i, idx_j = edge_index[0, :], edge_index[1, :]
        vectors = positions.index_select(0, idx_i) - positions.index_select(0, idx_j) - shifts
        lengths = torch.norm(vectors, dim=-1, keepdim=True)
        graph['vectors'], graph['lengths'] = vectors, lengths
        
        results = {'atomic_energies': torch.zeros(2, device=device),
                   'partial_charges': torch.tensor(atoms.arrays['initial_charges'], device=device)}
        
        ewald_results = model(graph, results)
        
        energies_ewald.append(ewald_results['atomic_energies'].sum().detach().cpu().item())
        
        results = {'atomic_energies': torch.zeros(2, device=device),
                   'partial_charges': torch.tensor(atoms.arrays['initial_charges'], device=device)}
        
        coulomb_results = model_coulomb(graph, results)
        
        energies_coulomb.append(coulomb_results['atomic_energies'].sum().detach().cpu().item())
        
    plt.figure(figsize=(8, 6))
    plt.plot(distances, energies_ewald, label='ewald')
    plt.plot(distances, energies_coulomb, '--', label='coulomb')
    
    plt.legend()
    
    plt.xlabel("Distance in Angstrom")
    plt.ylabel("Coulomb energy in eV")
    plt.savefig('ewald_energy.pdf')


def test_SPMEElectrostaticEnergy(device: str = 'cuda:0'):
    
    model = SPMEElectrostaticEnergy(
        r_cutoff=10.0,
        ke=14.399645351950548,
        alpha=0.3,
        k_max=60,
        spline_order=5,
        exclusion_radius=5.0,
        n_exclusion_polynomial_cutoff=3
    )
    model = model.to(device)
    
    model_coulomb = CoulombElectrostaticEnergy(
        r_cutoff=None,
        ke=14.399645351950548,
        exclusion_radius=5.0,
        n_exclusion_polynomial_cutoff=3
    )
    model_coulomb = model_coulomb.to(device)
    
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
    atoms.set_initial_charges([1.0, -1.0])
    
    atoms.set_cell([80, 80, 200])
    atoms.center()
    atoms.set_pbc(True)
    
    distances = np.linspace(0.1, 9.0, 100)
    
    energies_spme = []
    energies_coulomb = []
    for d in distances:
        
        atoms.set_distance(0, 1, d, fix=0)
        
        structure = AtomicStructure.from_atoms(atoms)
    
        dl = DataLoader([AtomicData(structure, r_cutoff=10.0)], batch_size=1, shuffle=False, drop_last=False)
        batch = next(iter(dl)).to(device)
        
        batch.positions.requires_grad = True
        
        graph = batch.to_dict()
        
        edge_index, positions, shifts = graph['edge_index'], graph['positions'], graph['shifts']
    
        idx_i, idx_j = edge_index[0, :], edge_index[1, :]
        vectors = positions.index_select(0, idx_i) - positions.index_select(0, idx_j) - shifts
        lengths = torch.norm(vectors, dim=-1, keepdim=True)
        graph['vectors'], graph['lengths'] = vectors, lengths
        
        results = {'atomic_energies': torch.zeros(2, device=device),
                   'partial_charges': torch.tensor(atoms.arrays['initial_charges'], device=device)}
        
        spme_results = model(graph, results)
        
        energies_spme.append(spme_results['atomic_energies'].sum().detach().cpu().item())
        
        results = {'atomic_energies': torch.zeros(2, device=device),
                   'partial_charges': torch.tensor(atoms.arrays['initial_charges'], device=device)}
        
        coulomb_results = model_coulomb(graph, results)
        
        energies_coulomb.append(coulomb_results['atomic_energies'].sum().detach().cpu().item())
        
    plt.figure(figsize=(8, 6))
    plt.plot(distances, energies_spme, label='spme')
    plt.plot(distances, energies_coulomb, '--', label='coulomb')
    
    plt.legend()
    
    plt.xlabel("Distance in Angstrom")
    plt.ylabel("Coulomb energy in eV")
    plt.savefig('spme_energy.pdf')
    

def test_SPCE_EwaldElectrostaticEnergy(device: str = 'cuda:0') -> None:
    """Evaluates reciprocal and self energies for the SPCE benchmark data and compares them to the values 
    provided in https://www.nist.gov/mml/csd/chemical-informatics-group/spce-water-reference-calculations-10a-cutoff#numrecipes.
    
    Note: There seems to be something strange with real energy values provided by NIST. We compared the total energies with 
          VaspBandUnfolding (https://github.com/QijingZheng/VaspBandUnfolding/tree/master) and they are fine.
    """
    traj = read('spce/spce.extxyz', ':')
    structures = AtomicStructures.from_traj(traj)
    data = structures.to_data(r_cutoff=10.0)
    
    dl = DataLoader(data, batch_size=1, shuffle=False, drop_last=False)
    
    for atoms, batch in zip(traj, dl):
        cell = np.diag(atoms.get_cell())
        alpha = 5.6 / min(cell)

        pair_potential = EwaldElectrostaticEnergy(
            r_cutoff=10.0,
            ke=14.399645351950548,
            alpha=alpha,
            k_max=5,
            exclusion_radius=None,
            n_exclusion_polynomial_cutoff=3
        )
        pair_potential = pair_potential.to(device)
        
        batch = batch.to(device)
        graph = batch.to_dict()
        
        # prepare distances and distance vectors
        edge_index, positions, shifts = graph['edge_index'], graph['positions'], graph['shifts']
        
        idx_i, idx_j = edge_index[0, :], edge_index[1, :]
        vectors = positions.index_select(0, idx_i) - positions.index_select(0, idx_j) - shifts
        lengths = torch.norm(vectors, dim=-1, keepdim=True)
        graph['vectors'], graph['lengths'] = vectors, lengths
        
        results = {'partial_charges': torch.as_tensor(atoms.arrays['initial_charges'], dtype=torch.get_default_dtype(), device=device)}
        
        assert abs(pair_potential.get_energy_recip(graph, results).sum().cpu().numpy() - (atoms.info['recip energy'] + atoms.info['self energy'])) < 5e-3
        

def test_VASP_EwaldElectrostaticEnergy(device: str = 'cuda:0') -> None:
    """Evaluates Ewald sum for a few crystal systems from https://github.com/QijingZheng/VaspBandUnfolding/tree/master/examples/ewald.
    
    Note: We need https://github.com/QijingZheng/VaspBandUnfolding to run this test.
    """
    from ewald import ewaldsum
    
    crystals = [
        'vasp/NaCl.vasp',
        'vasp/CsCl.vasp',
        'vasp/ZnO-Hex.vasp',
        'vasp/ZnO-Cub.vasp',
        'vasp/TiO2.vasp',
        'vasp/CaF2.vasp',
    ]
    
    ZZ = {
        'Na':  1, 'Ca':  2,
        'Cl': -1,  'F': -1,
        'Cs':  1,
        'Ti':  4,
        'Zn':  2,
         'O': -2,
    }
    
    for crys in crystals:
        atoms = read(crys)
        
        esum = ewaldsum(atoms, ZZ, Rcut=4.0)
        reference = esum.get_ewaldsum()
        
        structure = AtomicStructure.from_atoms(atoms)
        dl = DataLoader([AtomicData(structure, r_cutoff=4.0)], batch_size=1, shuffle=False, drop_last=False)
        batch = next(iter(dl)).to(device)
        graph = batch.to_dict()
        
        # prepare distances and distance vectors
        edge_index, positions, shifts = graph['edge_index'], graph['positions'], graph['shifts']
        
        idx_i, idx_j = edge_index[0, :], edge_index[1, :]
        vectors = positions.index_select(0, idx_i) - positions.index_select(0, idx_j) - shifts
        lengths = torch.norm(vectors, dim=-1, keepdim=True)
        graph['vectors'], graph['lengths'] = vectors, lengths
        
        partial_charges = np.zeros(len(atoms))

        for i, atom in enumerate(atoms):
            element = atom.symbol
            if element in ZZ:
                partial_charges[i] = ZZ[element]
            else:
                raise RuntimeError()
        
        results = {'partial_charges': torch.as_tensor(partial_charges, dtype=torch.get_default_dtype(), device=device),
                   'atomic_energies': torch.zeros(1, dtype=torch.get_default_dtype(), device=device)}
            
        pair_potential = EwaldElectrostaticEnergy(
            r_cutoff=4.0,
            ke=14.399645351950548,
            alpha=4.0 / 5.0,
            k_max=5,
            exclusion_radius=None,
            n_exclusion_polynomial_cutoff=3
        )
        pair_potential = pair_potential.to(device)
        
        results = pair_potential(graph, results)
        assert abs(results['atomic_energies'].sum().cpu().numpy() - reference) < 5e-3


def test_SPCE_SPMEElectrostaticEnergy(device: str = 'cuda:0') -> None:

    traj = read('spce/spce.extxyz', ':')
    structures = AtomicStructures.from_traj(traj)
    data = structures.to_data(r_cutoff=10.0)
    
    dl = DataLoader(data, batch_size=1, shuffle=False, drop_last=False)
    
    for atoms, batch in zip(traj, dl):
        cell = np.diag(atoms.get_cell())
        alpha = 5.6 / min(cell)

        pair_potential = SPMEElectrostaticEnergy(
            r_cutoff=10.0,
            ke=14.399645351950548,
            alpha=alpha,
            k_max=10,
            exclusion_radius=None,
            n_exclusion_polynomial_cutoff=3
        )
        pair_potential = pair_potential.to(device)
        
        batch = batch.to(device)
        graph = batch.to_dict()
        
        # prepare distances and distance vectors
        edge_index, positions, shifts = graph['edge_index'], graph['positions'], graph['shifts']
        
        idx_i, idx_j = edge_index[0, :], edge_index[1, :]
        vectors = positions.index_select(0, idx_i) - positions.index_select(0, idx_j) - shifts
        lengths = torch.norm(vectors, dim=-1, keepdim=True)
        graph['vectors'], graph['lengths'] = vectors, lengths
        
        results = {'partial_charges': torch.as_tensor(atoms.arrays['initial_charges'], dtype=torch.get_default_dtype(), device=device)}
        
        assert abs(pair_potential.get_energy_recip(graph, results).sum().cpu().numpy() - (atoms.info['recip energy'] + atoms.info['self energy'])) < 5e-3


def test_VASP_SPMEElectrostaticEnergy(device: str = 'cuda:0') -> None:

    from ewald import ewaldsum
    
    crystals = [
        'vasp/NaCl.vasp',
        'vasp/CsCl.vasp',
        'vasp/ZnO-Hex.vasp',
        'vasp/ZnO-Cub.vasp',
        'vasp/TiO2.vasp',
        'vasp/CaF2.vasp',
    ]
    
    ZZ = {
        'Na':  1, 'Ca':  2,
        'Cl': -1,  'F': -1,
        'Cs':  1,
        'Ti':  4,
        'Zn':  2,
         'O': -2,
    }
    
    for crys in crystals:
        atoms = read(crys)
        
        esum = ewaldsum(atoms, ZZ, Rcut=4.0)
        reference = esum.get_ewaldsum()
        
        structure = AtomicStructure.from_atoms(atoms)
        dl = DataLoader([AtomicData(structure, r_cutoff=4.0)], batch_size=1, shuffle=False, drop_last=False)
        batch = next(iter(dl)).to(device)
        graph = batch.to_dict()
        
        # prepare distances and distance vectors
        edge_index, positions, shifts = graph['edge_index'], graph['positions'], graph['shifts']
        
        idx_i, idx_j = edge_index[0, :], edge_index[1, :]
        vectors = positions.index_select(0, idx_i) - positions.index_select(0, idx_j) - shifts
        lengths = torch.norm(vectors, dim=-1, keepdim=True)
        graph['vectors'], graph['lengths'] = vectors, lengths
        
        partial_charges = np.zeros(len(atoms))

        for i, atom in enumerate(atoms):
            element = atom.symbol
            if element in ZZ:
                partial_charges[i] = ZZ[element]
            else:
                raise RuntimeError()
        
        results = {'partial_charges': torch.as_tensor(partial_charges, dtype=torch.get_default_dtype(), device=device),
                   'atomic_energies': torch.zeros(1, dtype=torch.get_default_dtype(), device=device)}
            
        pair_potential = SPMEElectrostaticEnergy(
            r_cutoff=4.0,
            ke=14.399645351950548,
            alpha=4.0 / 5.0,
            k_max=10,
            exclusion_radius=None,
            n_exclusion_polynomial_cutoff=3
        )
        pair_potential = pair_potential.to(device)
        
        results = pair_potential(graph, results)
        assert abs(results['atomic_energies'].sum().cpu().numpy() - reference) < 5e-3


def test_gradients_SPCE_SPMEElectrostaticEnergy(device: str = 'cuda:0') -> None:

    traj = read('spce/spce.extxyz', ':')
    structures = AtomicStructures.from_traj(traj)
    data = structures.to_data(r_cutoff=10.0)
    
    dl = DataLoader(data, batch_size=1, shuffle=False, drop_last=False)
    
    for atoms, batch in zip(traj, dl):
        cell = np.diag(atoms.get_cell())
        alpha = 5.6 / min(cell)

        ewald_pair_potential = EwaldElectrostaticEnergy(
            r_cutoff=10.0,
            ke=14.399645351950548,
            alpha=alpha,
            k_max=5,
            exclusion_radius=None,
            n_exclusion_polynomial_cutoff=3
        )
        ewald_pair_potential = ewald_pair_potential.to(device)
        
        # Note: we used higher order splines to have higher accuracy of the interpolation in this test
        spme_pair_potential = SPMEElectrostaticEnergy(
            r_cutoff=10.0,
            ke=14.399645351950548,
            alpha=alpha,
            k_max=5,
            spline_order=10,
            exclusion_radius=None,
            n_exclusion_polynomial_cutoff=3
        )
        spme_pair_potential = spme_pair_potential.to(device)
        
        batch = batch.to(device)
        graph = batch.to_dict()
        
        batch.positions.requires_grad = True
        
        # prepare distances and distance vectors
        edge_index, positions, shifts = graph['edge_index'], graph['positions'], graph['shifts']
        
        idx_i, idx_j = edge_index[0, :], edge_index[1, :]
        vectors = positions.index_select(0, idx_i) - positions.index_select(0, idx_j) - shifts
        lengths = torch.norm(vectors, dim=-1, keepdim=True)
        graph['vectors'], graph['lengths'] = vectors, lengths
        
        results = {'partial_charges': torch.as_tensor(atoms.arrays['initial_charges'], dtype=torch.get_default_dtype(), device=device)}
        
        ewald_energy_recip = ewald_pair_potential.get_energy_recip(graph, results)
        ewald_grads = torch.autograd.grad([ewald_energy_recip], [positions], torch.ones_like(ewald_energy_recip), create_graph=True)[0]
        
        spme_energy_recip = spme_pair_potential.get_energy_recip(graph, results)
        spme_grads = torch.autograd.grad([spme_energy_recip], [positions], torch.ones_like(spme_energy_recip), create_graph=True)[0]
        
        assert all((spme_grads - ewald_grads).square().sum(-1).sqrt() / ewald_grads.square().sum(-1).sqrt() < 2e-2)
        

def test_gradients_VASP_SPMEElectrostaticEnergy(device: str = 'cuda:0') -> None:
    
    crystals = [
        'vasp/NaCl.vasp',
        'vasp/CsCl.vasp',
        'vasp/ZnO-Hex.vasp',
        'vasp/ZnO-Cub.vasp',
        'vasp/TiO2.vasp',
        'vasp/CaF2.vasp',
    ]
    
    ZZ = {
        'Na':  1, 'Ca':  2,
        'Cl': -1,  'F': -1,
        'Cs':  1,
        'Ti':  4,
        'Zn':  2,
         'O': -2,
    }
    
    for crys in crystals:
        atoms = read(crys)
        
        structure = AtomicStructure.from_atoms(atoms)
        dl = DataLoader([AtomicData(structure, r_cutoff=4.0)], batch_size=1, shuffle=False, drop_last=False)
        
        batch = next(iter(dl)).to(device)
        batch.positions.requires_grad = True
        
        ewald_pair_potential = EwaldElectrostaticEnergy(
            r_cutoff=4.0,
            ke=14.399645351950548,
            alpha=4.0 / 5.0,
            k_max=10,
            exclusion_radius=None,
            n_exclusion_polynomial_cutoff=3
        )
        ewald_pair_potential = ewald_pair_potential.to(device)
        
        # Note: we used higher order splines to have higher accuracy of the interpolation in this test
        spme_pair_potential = SPMEElectrostaticEnergy(
            r_cutoff=4.0,
            ke=14.399645351950548,
            alpha=4.0 / 5.0,
            k_max=5,
            spline_order=10,
            exclusion_radius=None,
            n_exclusion_polynomial_cutoff=3
        )
        spme_pair_potential = spme_pair_potential.to(device)
        
        graph = batch.to_dict()
        
        # prepare distances and distance vectors
        edge_index, positions, shifts = graph['edge_index'], graph['positions'], graph['shifts']
        
        idx_i, idx_j = edge_index[0, :], edge_index[1, :]
        vectors = positions.index_select(0, idx_i) - positions.index_select(0, idx_j) - shifts
        lengths = torch.norm(vectors, dim=-1, keepdim=True)
        graph['vectors'], graph['lengths'] = vectors, lengths
        
        partial_charges = np.zeros(len(atoms))

        for i, atom in enumerate(atoms):
            element = atom.symbol
            if element in ZZ:
                partial_charges[i] = ZZ[element]
            else:
                raise RuntimeError()
        
        ewald_results = {'partial_charges': torch.as_tensor(partial_charges, dtype=torch.get_default_dtype(), device=device),
                         'atomic_energies': torch.zeros(1, dtype=torch.get_default_dtype(), device=device)}
        
        ewald_results = ewald_pair_potential(graph, ewald_results)
        ewald_grads = torch.autograd.grad([ewald_results['atomic_energies']], [positions], torch.ones_like(ewald_results['atomic_energies']), create_graph=True)[0]
        
        spme_results = {'partial_charges': torch.as_tensor(partial_charges, dtype=torch.get_default_dtype(), device=device),
                        'atomic_energies': torch.zeros(1, dtype=torch.get_default_dtype(), device=device)}
        
        spme_results = spme_pair_potential(graph, spme_results)
        spme_grads = torch.autograd.grad([spme_results['atomic_energies']], [positions], torch.ones_like(spme_results['atomic_energies']), create_graph=True)[0]
        
        error = (spme_grads - ewald_grads).square().sum(-1).sqrt() / ewald_grads.square().sum(-1).sqrt()
        mask =  ewald_grads.square().sum(-1).sqrt() > 1e-4
        
        assert (error[mask] < 2e-2).all()


def test_times_SPCE_EwaldElectrostaticEnergy(device: str = 'cuda:0') -> None:

    atoms = read('spce/spce.extxyz', '-1')
    
    atoms = atoms.repeat((3,3,3))
    
    print(len(atoms))
    
    structure = AtomicStructure.from_atoms(atoms)
    
    dl = DataLoader([AtomicData(structure, r_cutoff=10.0)], batch_size=1, shuffle=False, drop_last=False)
    batch = next(iter(dl)).to(device)
    
    batch.positions.requires_grad = True
    
    cell = np.diag(atoms.get_cell())
    alpha = 5.6 / min(cell)

    pair_potential = EwaldElectrostaticEnergy(
        r_cutoff=10.0,
        ke=14.399645351950548,
        alpha=alpha,
        k_max=10,
        exclusion_radius=None,
        n_exclusion_polynomial_cutoff=3
    )
    pair_potential = pair_potential.to(device)

    graph = batch.to_dict()
    
    # prepare distances and distance vectors
    edge_index, positions, shifts = graph['edge_index'], graph['positions'], graph['shifts']
    
    idx_i, idx_j = edge_index[0, :], edge_index[1, :]
    vectors = positions.index_select(0, idx_i) - positions.index_select(0, idx_j) - shifts
    lengths = torch.norm(vectors, dim=-1, keepdim=True)
    graph['vectors'], graph['lengths'] = vectors, lengths
    
    results = {'partial_charges': torch.as_tensor(atoms.arrays['initial_charges'], dtype=torch.get_default_dtype(), device=device)}
    
    # need to re-iterate before time measurement
    for _ in range(10):
        energy_recip = pair_potential.get_energy_recip(graph, results)
        grads = torch.autograd.grad([energy_recip], [positions], torch.ones_like(energy_recip), create_graph=True)
    
    times = []
    
    for _ in range(100):
        
        # start with the time measurement
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        energy_recip = pair_potential.get_energy_recip(graph, results)
        grads = torch.autograd.grad([energy_recip], [positions], torch.ones_like(energy_recip), create_graph=True)
        
        if device.startswith('cuda'):
            torch.cuda.synchronize()
            
        end_time = time.time()
        
        times.append(end_time-start_time)
    
    print(np.mean(times), np.std(times))


def test_times_SPCE_SPMEElectrostaticEnergy(device: str = 'cuda:0') -> None:

    atoms = read('spce/spce.extxyz', '-1')
    
    atoms = atoms.repeat((3,3,3))
    
    structure = AtomicStructure.from_atoms(atoms)
    
    dl = DataLoader([AtomicData(structure, r_cutoff=10.0)], batch_size=1, shuffle=False, drop_last=False)
    batch = next(iter(dl)).to(device)
    
    batch.positions.requires_grad = True
    
    cell = np.diag(atoms.get_cell())
    alpha = 5.6 / min(cell)

    pair_potential = SPMEElectrostaticEnergy(
        r_cutoff=10.0,
        ke=14.399645351950548,
        alpha=alpha,
        k_max=10,
        exclusion_radius=None,
        n_exclusion_polynomial_cutoff=3
    )
    pair_potential = pair_potential.to(device)

    graph = batch.to_dict()
    
    # prepare distances and distance vectors
    edge_index, positions, shifts = graph['edge_index'], graph['positions'], graph['shifts']
    
    idx_i, idx_j = edge_index[0, :], edge_index[1, :]
    vectors = positions.index_select(0, idx_i) - positions.index_select(0, idx_j) - shifts
    lengths = torch.norm(vectors, dim=-1, keepdim=True)
    graph['vectors'], graph['lengths'] = vectors, lengths
    
    results = {'partial_charges': torch.as_tensor(atoms.arrays['initial_charges'], dtype=torch.get_default_dtype(), device=device)}
    
    
    # need to re-iterate before time measurement
    for _ in range(10):
        energy_recip = pair_potential.get_energy_recip(graph, results)
        grads = torch.autograd.grad([energy_recip], [positions], torch.ones_like(energy_recip), create_graph=True)
    
    times = []
    
    for _ in range(100):
        
        # start with the time measurement
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        energy_recip = pair_potential.get_energy_recip(graph, results)
        grads = torch.autograd.grad([energy_recip], [positions], torch.ones_like(energy_recip), create_graph=True)
        
        if device.startswith('cuda'):
            torch.cuda.synchronize()
            
        end_time = time.time()
        
        times.append(end_time-start_time)
    
    print(np.mean(times), np.std(times))


def test_batched_SPMEElectrostaticEnergy(device: str = 'cpu'):
    
    traj = read('spce/spce.extxyz', ':')
    structures = AtomicStructures.from_traj(traj)
    data = structures.to_data(r_cutoff=10.0)
    
    dl = DataLoader(data, batch_size=3, shuffle=False, drop_last=False)
    
    batch = next(iter(dl)).to(device)
    
    cell = np.diag(traj[0].get_cell())
    alpha = 5.6 / min(cell)

    pair_potential = SPMEElectrostaticEnergy(
        r_cutoff=10.0,
        ke=14.399645351950548,
        alpha=alpha,
        k_max=10,
        exclusion_radius=None,
        n_exclusion_polynomial_cutoff=3
    )
    pair_potential = pair_potential.to(device)
    
    batch = batch.to(device)
    graph = batch.to_dict()
    
    # prepare distances and distance vectors
    edge_index, positions, shifts = graph['edge_index'], graph['positions'], graph['shifts']
    
    idx_i, idx_j = edge_index[0, :], edge_index[1, :]
    vectors = positions.index_select(0, idx_i) - positions.index_select(0, idx_j) - shifts
    lengths = torch.norm(vectors, dim=-1, keepdim=True)
    graph['vectors'], graph['lengths'] = vectors, lengths
    
    results = {'partial_charges': torch.as_tensor(np.concatenate([traj[0].arrays['initial_charges'],
                                                                  traj[1].arrays['initial_charges'],
                                                                  traj[2].arrays['initial_charges']], 0), 
                                                  dtype=torch.get_default_dtype(), device=device)}
    
    energy_recip = pair_potential.get_energy_recip(graph, results)
    reference = np.stack([traj[0].info['recip energy'] + traj[0].info['self energy'], 
                          traj[1].info['recip energy'] + traj[1].info['self energy'], 
                          traj[2].info['recip energy'] + traj[2].info['self energy']])
    
    assert (abs(segment_sum(energy_recip, idx_i=batch.batch, dim_size=batch.n_atoms.shape[0]).cpu().numpy() - reference) < 5e-3).all()


def test_D4DispersionEnergy_no_cutoff(device: str = 'cuda:0'):
    
    # test against the original implementation
    # https://dftd4.readthedocs.io/en/latest/reference/ase.html#dftd4.ase.DFTD4.calculate
    # https://github.com/dftd4/dftd4/tree/main/python/dftd4
    
    r_cutoff = None
    
    d4_model = D4DispersionEnergy(
        r_cutoff=r_cutoff,
        exclusion_radius=None,
        n_exclusion_polynomial_cutoff=3
    )
    d4_model = d4_model.to(device)
    
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
    atoms.set_initial_charges([0.3, -0.55])
    
    atoms.calc = DFTD4(method='hf')
    
    distances = np.linspace(0.1, 6.0, 100)
    
    for d in distances:
        atoms.set_distance(0, 1, d, fix=0)
        
        structure = AtomicStructure.from_atoms(atoms)
    
        dl = DataLoader([AtomicData(structure, r_cutoff=7.0)], batch_size=1, shuffle=False, drop_last=False)
        batch = next(iter(dl)).to(device)
        
        batch.positions.requires_grad = True
        
        graph = batch.to_dict()
        
        edge_index, positions, shifts = graph['edge_index'], graph['positions'], graph['shifts']
    
        idx_i, idx_j = edge_index[0, :], edge_index[1, :]
        vectors = positions.index_select(0, idx_i) - positions.index_select(0, idx_j) - shifts
        lengths = torch.norm(vectors, dim=-1, keepdim=True)
        graph['vectors'], graph['lengths'] = vectors, lengths
        
        dftd4_edisp = atoms.get_potential_energy()
        dftd4_forces = atoms.get_forces()
        
        results = {'atomic_energies': torch.zeros(2, device=device),
                   'partial_charges': torch.tensor(atoms.calc._disp.get_properties()['partial charges'], device=device)}
        
        edisp_results = d4_model(graph, results)
        edisp_grads = torch.autograd.grad([edisp_results['atomic_energies']], [positions], 
                                          torch.ones_like(edisp_results['atomic_energies']), create_graph=True)[0]
        
        assert np.allclose(dftd4_forces, -1.0 * edisp_grads.detach().cpu().numpy(), atol=1e-12)
        assert np.allclose(edisp_results['atomic_energies'].sum().detach().cpu().item(), dftd4_edisp, atol=1e-12)
        
    atoms = molecule("C60")
    atoms.calc = DFTD4(params_tweaks=dict(s6=1.0, s8=1.61679827, s9=0.0, a1=0.44959224, a2=3.35743605))
    
    dftd4_edisp = atoms.get_potential_energy()
    dftd4_forces = atoms.get_forces()
    
    structure = AtomicStructure.from_atoms(atoms)
    
    dl = DataLoader([AtomicData(structure, r_cutoff=7.0)], batch_size=1, shuffle=False, drop_last=False)
    batch = next(iter(dl)).to(device)
    
    batch.positions.requires_grad = True
    
    graph = batch.to_dict()
    
    edge_index, positions, shifts = graph['edge_index'], graph['positions'], graph['shifts']

    idx_i, idx_j = edge_index[0, :], edge_index[1, :]
    vectors = positions.index_select(0, idx_i) - positions.index_select(0, idx_j) - shifts
    lengths = torch.norm(vectors, dim=-1, keepdim=True)
    graph['vectors'], graph['lengths'] = vectors, lengths
    
    results = {'atomic_energies': torch.zeros(len(atoms), device=device),
               'partial_charges': torch.tensor(atoms.calc._disp.get_properties()['partial charges'], device=device)}
        
    edisp_results = d4_model(graph, results)
    edisp_grads = torch.autograd.grad([edisp_results['atomic_energies']], [positions], 
                                        torch.ones_like(edisp_results['atomic_energies']), create_graph=True)[0]
    
    assert np.allclose(dftd4_forces, -1.0 * edisp_grads.detach().cpu().numpy(), atol=1e-3)
    assert np.allclose(edisp_results['atomic_energies'].sum().detach().cpu().item(), dftd4_edisp, atol=1e-2)
    

def test_D4DispersionEnergy_cutoff(device: str = 'cuda:0'):
    
    d4_model_no_cutoff = D4DispersionEnergy(
        r_cutoff=None,
        exclusion_radius=5.0,
        n_exclusion_polynomial_cutoff=3
    )
    d4_model_no_cutoff = d4_model_no_cutoff.to(device)
    
    d4_model_cutoff = D4DispersionEnergy(
        r_cutoff=9.0,
        exclusion_radius=5.0,
        n_exclusion_polynomial_cutoff=3
    )
    d4_model_cutoff = d4_model_cutoff.to(device)
    
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
    atoms.set_initial_charges([0.3, -0.55])
    
    atoms.calc = DFTD4(method='hf')
    
    distances = np.linspace(0.1, 10.0, 100)
    
    energies_no_cutoff = []
    energies_cutoff = []
    for d in distances:
        atoms.set_distance(0, 1, d, fix=0)
        
        structure = AtomicStructure.from_atoms(atoms)
    
        dl = DataLoader([AtomicData(structure, r_cutoff=10.0)], batch_size=1, shuffle=False, drop_last=False)
        batch = next(iter(dl)).to(device)
        
        batch.positions.requires_grad = True
        
        graph = batch.to_dict()
        
        edge_index, positions, shifts = graph['edge_index'], graph['positions'], graph['shifts']
    
        idx_i, idx_j = edge_index[0, :], edge_index[1, :]
        vectors = positions.index_select(0, idx_i) - positions.index_select(0, idx_j) - shifts
        lengths = torch.norm(vectors, dim=-1, keepdim=True)
        graph['vectors'], graph['lengths'] = vectors, lengths
        
        atoms.get_potential_energy()
        
        results = {'atomic_energies': torch.zeros(2, device=device),
                   'partial_charges': torch.tensor(atoms.calc._disp.get_properties()['partial charges'], device=device)}
        
        edisp_results = d4_model_no_cutoff(graph, results)
        
        energies_no_cutoff.append(edisp_results['atomic_energies'].sum().detach().cpu().item())
        
        results = {'atomic_energies': torch.zeros(2, device=device),
                   'partial_charges': torch.tensor(atoms.calc._disp.get_properties()['partial charges'], device=device)}
        
        edisp_results = d4_model_cutoff(graph, results)
        
        energies_cutoff.append(edisp_results['atomic_energies'].sum().detach().cpu().item())
        
    plt.figure(figsize=(8, 6))
    plt.plot(distances, energies_no_cutoff, label='no cutoff')
    plt.plot(distances, energies_cutoff, label='cutoff')
    
    plt.xlabel("Distance in Angstrom")
    plt.ylabel("Dispersion energy in eV")
    plt.savefig('dispersion_energy_cutoff.pdf')


def test_D4DispersionEnergy_update_refc6(device: str = 'cuda:0'):
    
    model = D4DispersionEnergy(
        r_cutoff=10.0,
        exclusion_radius=None,
        n_exclusion_polynomial_cutoff=3
    )
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    graph = {
        'species': torch.randint(0, model.Z_max, (100,), device=device),
        'edge_index': torch.randint(0, 100, (2, 200), device=device),
        'lengths': torch.rand(200, device=device) * 5.0,
        'batch': torch.randint(0, 10, (100,), device=device)
    }
    graph['atomic_numbers'] = graph['species']
    
    results = {
        'partial_charges': torch.rand(100, device=device),
        'atomic_energies': torch.zeros(100, device=device)
    }
    
    initial_refc6 = model.refc6.clone()

    for _ in range(1):
        output = model(graph, results)
        loss = output['disp_atomic_energies'].square().sum()
        loss.backward()
        optimizer.step()
        
        if isinstance(model, D4DispersionEnergy):
            model._update_refc6()

    updated_refc6 = model.refc6

    assert not torch.equal(initial_refc6, updated_refc6)


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    
    test_ZBLRepulsionEnergy_dimer()
    
    test_CoulombElectrostaticEnergy_dimer_no_cutoff()
    test_CoulombElectrostaticEnergy_dimer_cutoff()
    
    test_EwaldElectrostaticEnergy()
    test_SPMEElectrostaticEnergy()
    
    test_SPCE_EwaldElectrostaticEnergy()
    test_VASP_EwaldElectrostaticEnergy()
    test_SPCE_SPMEElectrostaticEnergy()
    test_VASP_SPMEElectrostaticEnergy()
    
    test_gradients_SPCE_SPMEElectrostaticEnergy()
    test_gradients_VASP_SPMEElectrostaticEnergy()
    
    test_times_SPCE_EwaldElectrostaticEnergy()
    test_times_SPCE_SPMEElectrostaticEnergy()
    
    test_batched_SPMEElectrostaticEnergy()
    
    test_D4DispersionEnergy_no_cutoff()
    test_D4DispersionEnergy_cutoff()
    test_D4DispersionEnergy_update_refc6()
    