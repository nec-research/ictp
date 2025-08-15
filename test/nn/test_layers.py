import scipy

import torch

from ictp.nn.layers import RealAgnosticResidualInteractionLayer, ProductBasisLayer
from ictp.o3.cartesian_harmonics import CartesianHarmonics

from utils import is_symmetric, is_traceless


def test_interaction_layer(seed: int):
    # set seed
    torch.manual_seed(seed)
    
    # define a random rotational matrix
    R = torch.as_tensor(scipy.spatial.transform.Rotation.random().as_matrix(), 
                        dtype=torch.get_default_dtype())
    
    # model the first interaction layer
    interaction_layer = RealAgnosticResidualInteractionLayer(l_max_node_feats=0,
                                                             l_max_edge_attrs=3,
                                                             l_max_target_feats=3,
                                                             l_max_hidden_feats=2,
                                                             n_basis=8,
                                                             n_species=4,
                                                             in_features=16,
                                                             out_features=16,
                                                             avg_n_neighbors=16,
                                                             radial_MLP=3*[64])
    
    node_feats = torch.randn(8, 16)
    node_attrs = torch.nn.functional.one_hot(torch.arange(0, 8) % 4)
    node_attrs = torch.as_tensor(node_attrs, dtype=torch.get_default_dtype())
    
    vectors = torch.randn(128, 3)
    lengths = torch.norm(vectors, dim=-1)
    
    edge_feats = torch.einsum('a, j', lengths, torch.randn(8))
    
    idx_i = torch.sort(torch.randint(0, 8, (128, )))[0]
    idx_j = torch.randint(0, 8, (128, ))
    
    cartesian_harmonics = CartesianHarmonics(l_max=3)
    edge_attrs = cartesian_harmonics(vectors)
    
    msg, sc = interaction_layer(node_attrs, node_feats, edge_attrs, edge_feats, idx_i, idx_j)
    
    # model the first interaction layer for rotated inputs
    R = torch.as_tensor(scipy.spatial.transform.Rotation.random().as_matrix(), dtype=torch.get_default_dtype())

    R_vectors = torch.einsum('ij, aj -> ai', R, vectors)
    
    R_edge_attrs = cartesian_harmonics(R_vectors)
    
    R_msg, R_sc = interaction_layer(node_attrs, node_feats, R_edge_attrs, edge_feats, idx_i, idx_j)

    # check shapes of Cartesian harmonics, as well as, their properties
    # i.e., if they are traceless, symmetric, and normalized
    slices = []
    k = 0
    for l in range(4):
        slices.append(slice(k,  k + 3 ** l * 16, None))
        k += 3 ** l * 16
    
    shapes = []
    for l in range(4):
        if l == 0:
            shapes.append([1] + [16])
        else:
            shapes.append([3 for _ in range(l)] + [16])
    
    assert msg.shape == (8, 640)
    assert sc.shape == (8, 208)
    
    assert msg[:, slices[0]].reshape(-1, *shapes[0]).shape == (8, 1, 16)
    assert msg[:, slices[1]].reshape(-1, *shapes[1]).shape== (8, 3, 16)
    assert msg[:, slices[2]].reshape(-1, *shapes[2]).shape == (8, 3, 3, 16)
    assert msg[:, slices[3]].reshape(-1, *shapes[3]).shape == (8, 3, 3, 3, 16)
    
    assert is_traceless(msg[:, slices[0]].reshape(-1, *shapes[0]))
    assert is_traceless(msg[:, slices[1]].reshape(-1, *shapes[1]))
    assert is_traceless(msg[:, slices[2]].reshape(-1, *shapes[2]))
    assert is_traceless(msg[:, slices[3]].reshape(-1, *shapes[3]))
    
    assert is_symmetric(msg[:, slices[0]].reshape(-1, *shapes[0]))
    assert is_symmetric(msg[:, slices[1]].reshape(-1, *shapes[1]))
    assert is_symmetric(msg[:, slices[2]].reshape(-1, *shapes[2]))
    assert is_symmetric(msg[:, slices[3]].reshape(-1, *shapes[3]))
    
    assert sc[:, slices[0]].reshape(-1, *shapes[0]).shape == (8, 1, 16)
    assert sc[:, slices[1]].reshape(-1, *shapes[1]).shape == (8, 3, 16)
    assert sc[:, slices[2]].reshape(-1, *shapes[2]).shape == (8, 3, 3, 16)
    
    assert (sc[:, slices[0]].reshape(-1, *shapes[0]) != 0.0).all()
    assert (sc[:, slices[1]].reshape(-1, *shapes[1]) == 0.0).all()
    assert (sc[:, slices[2]].reshape(-1, *shapes[2]) == 0.0).all()
    
    # check equivariance
    R_msg_new = torch.cat(
            [
                msg[:, slices[0]],
                torch.einsum('ij, aju -> aiu', R, msg[:, slices[1]].reshape(-1, *shapes[1])).reshape(-1, 3 * 16),
                torch.einsum('ij, kl, ajlu -> aiku', R, R, msg[:, slices[2]].reshape(-1, *shapes[2])).reshape(-1, 3 ** 2 * 16),
                torch.einsum('ij, kl, mn, ajlnu-> aikmu', R, R, R, msg[:, slices[3]].reshape(-1, *shapes[3])).reshape(-1, 3 ** 3 * 16),
            ], -1)
        
    R_sc_new = torch.cat(
        [
            sc[:, slices[0]],
            torch.einsum('ij, aju -> aiu', R, sc[:, slices[1]].reshape(-1, *shapes[1])).reshape(-1, 3 * 16),
            torch.einsum('ij, kl, ajlu -> aiku', R, R, sc[:, slices[2]].reshape(-1, *shapes[2])).reshape(-1, 3 ** 2 * 16),
        ], -1)
    
    assert (abs(R_msg - R_msg_new) < 1e-14).all()
    assert (abs(R_sc - R_sc_new) < 1e-14).all()

    # model the second interaction layer
    interaction_layer = RealAgnosticResidualInteractionLayer(l_max_node_feats=2,
                                                             l_max_edge_attrs=3,
                                                             l_max_target_feats=3,
                                                             l_max_hidden_feats=2,
                                                             n_basis=8,
                                                             n_species=4,
                                                             in_features=16,
                                                             out_features=16,
                                                             avg_n_neighbors=16,
                                                             radial_MLP=3*[64])
    
    node_feats = sc + msg[:, :208]
    
    msg, sc = interaction_layer(node_attrs, node_feats, edge_attrs, edge_feats, idx_i, idx_j)
    
    # model the second interaction layer for rotated inputs
    R_node_feats = R_sc + R_msg[:, :208]
    
    R_msg, R_sc = interaction_layer(node_attrs, R_node_feats, R_edge_attrs, edge_feats, idx_i, idx_j)

    # check shapes of Cartesian harmonics, as well as, their properties
    # i.e., if they are traceless, symmetric, and normalized
    slices = []
    k = 0
    for l in range(4):
        slices.append(slice(k,  k + 3 ** l * 16, None))
        k += 3 ** l * 16
    
    shapes = []
    for l in range(4):
        if l == 0:
            shapes.append([1] + [16])
        else:
            shapes.append([3 for _ in range(l)] + [16])
    
    assert msg.shape == (8, 640)
    assert sc.shape == (8, 208)
    
    assert msg[:, slices[0]].reshape(-1, *shapes[0]).shape == (8, 1, 16)
    assert msg[:, slices[1]].reshape(-1, *shapes[1]).shape== (8, 3, 16)
    assert msg[:, slices[2]].reshape(-1, *shapes[2]).shape == (8, 3, 3, 16)
    assert msg[:, slices[3]].reshape(-1, *shapes[3]).shape == (8, 3, 3, 3, 16)
    
    assert is_traceless(msg[:, slices[0]].reshape(-1, *shapes[0]))
    assert is_traceless(msg[:, slices[1]].reshape(-1, *shapes[1]))
    assert is_traceless(msg[:, slices[2]].reshape(-1, *shapes[2]))
    assert is_traceless(msg[:, slices[3]].reshape(-1, *shapes[3]))
    
    assert is_symmetric(msg[:, slices[0]].reshape(-1, *shapes[0]))
    assert is_symmetric(msg[:, slices[1]].reshape(-1, *shapes[1]))
    assert is_symmetric(msg[:, slices[2]].reshape(-1, *shapes[2]))
    assert is_symmetric(msg[:, slices[3]].reshape(-1, *shapes[3]))
    
    assert sc[:, slices[0]].reshape(-1, *shapes[0]).shape == (8, 1, 16)
    assert sc[:, slices[1]].reshape(-1, *shapes[1]).shape == (8, 3, 16)
    assert sc[:, slices[2]].reshape(-1, *shapes[2]).shape == (8, 3, 3, 16)
    
    assert (sc[:, slices[0]].reshape(-1, *shapes[0]) != 0.0).all()
    assert (sc[:, slices[1]].reshape(-1, *shapes[1]) != 0.0).all()
    assert (sc[:, slices[2]].reshape(-1, *shapes[2]) != 0.0).all()
    
    # check equivariance
    R_msg_new = torch.cat(
            [
                msg[:, slices[0]],
                torch.einsum('ij, aju -> aiu', R, msg[:, slices[1]].reshape(-1, *shapes[1])).reshape(-1, 3 * 16),
                torch.einsum('ij, kl, ajlu -> aiku', R, R, msg[:, slices[2]].reshape(-1, *shapes[2])).reshape(-1, 3 ** 2 * 16),
                torch.einsum('ij, kl, mn, ajlnu-> aikmu', R, R, R, msg[:, slices[3]].reshape(-1, *shapes[3])).reshape(-1, 3 ** 3 * 16),
            ], -1)
        
    R_sc_new = torch.cat(
        [
            sc[:, slices[0]],
            torch.einsum('ij, aju -> aiu', R, sc[:, slices[1]].reshape(-1, *shapes[1])).reshape(-1, 3 * 16),
            torch.einsum('ij, kl, ajlu -> aiku', R, R, sc[:, slices[2]].reshape(-1, *shapes[2])).reshape(-1, 3 ** 2 * 16),
        ], -1)
    
    assert (abs(R_msg - R_msg_new) < 1e-14).all()
    assert (abs(R_sc - R_sc_new) < 1e-14).all()


def test_product_basis_layer(seed: int):
    # set seed
    torch.manual_seed(seed)
    
    # define a random rotational matrix
    R = torch.as_tensor(scipy.spatial.transform.Rotation.random().as_matrix(), 
                        dtype=torch.get_default_dtype())
    
    # model the first interaction layer
    interaction_layer = RealAgnosticResidualInteractionLayer(l_max_node_feats=0,
                                                             l_max_edge_attrs=3,
                                                             l_max_target_feats=3,
                                                             l_max_hidden_feats=2,
                                                             n_basis=8,
                                                             n_species=4,
                                                             in_features=16,
                                                             out_features=16,
                                                             avg_n_neighbors=16,
                                                             radial_MLP=3*[64])
    
    node_feats = torch.randn(8, 16)
    node_attrs = torch.nn.functional.one_hot(torch.arange(0, 8) % 4)
    node_attrs = torch.as_tensor(node_attrs, dtype=torch.get_default_dtype())
    
    vectors = torch.randn(128, 3)
    lengths = torch.norm(vectors, dim=-1)
    
    edge_feats = torch.einsum('a, j', lengths, torch.randn(8))
    
    idx_i = torch.sort(torch.randint(0, 8, (128, )))[0]
    idx_j = torch.randint(0, 8, (128, ))
    
    cartesian_harmonics = CartesianHarmonics(l_max=3)
    edge_attrs = cartesian_harmonics(vectors)
    
    msg, sc = interaction_layer(node_attrs, node_feats, edge_attrs, edge_feats, idx_i, idx_j)
    
    # model the first interaction layer for rotated inputs
    R = torch.as_tensor(scipy.spatial.transform.Rotation.random().as_matrix(), dtype=torch.get_default_dtype())

    R_vectors = torch.einsum('ij, aj -> ai', R, vectors)
    
    R_edge_attrs = cartesian_harmonics(R_vectors)
    
    R_msg, R_sc = interaction_layer(node_attrs, node_feats, R_edge_attrs, edge_feats, idx_i, idx_j)
    
    # model the first product layer
    product_layer = ProductBasisLayer(l_max_node_feats=3, l_max_target_feats=2, 
                                      in_features=16, out_features=16, n_species=4, 
                                      correlation=3, use_sc=True, coupled_feats=False,
                                      symmetric_product=True)
    
    out_product = product_layer(msg, sc, node_attrs)
    
    # print(torch.std(out_product, -1))
    
    # model the first product layer for rotated inputs
    R_out_product = product_layer(R_msg, R_sc, node_attrs)
    
    # check shapes of Cartesian harmonics, as well as, their properties
    # i.e., if they are traceless, symmetric, and normalized
    slices = []
    k = 0
    for l in range(3):
        slices.append(slice(k,  k + 3 ** l * 16, None))
        k += 3 ** l * 16
    
    shapes = []
    for l in range(3):
        if l == 0:
            shapes.append([1] + [16])
        else:
            shapes.append([3 for _ in range(l)] + [16])
    
    assert out_product.shape == (8, 208)
    
    assert out_product[:, slices[0]].reshape(-1, *shapes[0]).shape == (8, 1, 16)
    assert out_product[:, slices[1]].reshape(-1, *shapes[1]).shape== (8, 3, 16)
    assert out_product[:, slices[2]].reshape(-1, *shapes[2]).shape == (8, 3, 3, 16)
    
    assert is_traceless(out_product[:, slices[0]].reshape(-1, *shapes[0]))
    assert is_traceless(out_product[:, slices[1]].reshape(-1, *shapes[1]))
    assert is_traceless(out_product[:, slices[2]].reshape(-1, *shapes[2]))
    
    assert is_symmetric(out_product[:, slices[0]].reshape(-1, *shapes[0]))
    assert is_symmetric(out_product[:, slices[1]].reshape(-1, *shapes[1]))
    assert is_symmetric(out_product[:, slices[2]].reshape(-1, *shapes[2]))
    
    # check equivariance
    R_out_product_new = torch.cat(
            [
                out_product[:, slices[0]],
                torch.einsum('ij, aju -> aiu', R, out_product[:, slices[1]].reshape(-1, *shapes[1])).reshape(-1, 3 * 16),
                torch.einsum('ij, kl, ajlu -> aiku', R, R, out_product[:, slices[2]].reshape(-1, *shapes[2])).reshape(-1, 3 ** 2 * 16),
            ], -1)
    
    assert (abs(R_out_product - R_out_product_new) < 1e-14).all()
    
    # model the second interaction layer
    interaction_layer = RealAgnosticResidualInteractionLayer(l_max_node_feats=2,
                                                             l_max_edge_attrs=3,
                                                             l_max_target_feats=3,
                                                             l_max_hidden_feats=2,
                                                             n_basis=8,
                                                             n_species=4,
                                                             in_features=16,
                                                             out_features=16,
                                                             avg_n_neighbors=16,
                                                             radial_MLP=3*[64])
    
    msg, sc = interaction_layer(node_attrs, out_product, edge_attrs, edge_feats, idx_i, idx_j)
    
    # model the second interaction layer for rotated inputs
    R_msg, R_sc = interaction_layer(node_attrs, R_out_product, R_edge_attrs, edge_feats, idx_i, idx_j)
    
    # model the second product layer
    product_layer = ProductBasisLayer(l_max_node_feats=3, l_max_target_feats=2, 
                                      in_features=16, out_features=16, n_species=4, 
                                      correlation=3, use_sc=True, coupled_feats=False,
                                      symmetric_product=True)
    
    out_product = product_layer(msg, sc, node_attrs)
    
    # print(torch.std(out_product, -1))
    
    # model the first product layer for rotated inputs
    R_out_product = product_layer(R_msg, R_sc, node_attrs)
    
    # check shapes of Cartesian harmonics, as well as, their properties
    # i.e., if they are traceless, symmetric, and normalized
    slices = []
    k = 0
    for l in range(3):
        slices.append(slice(k,  k + 3 ** l * 16, None))
        k += 3 ** l * 16
    
    shapes = []
    for l in range(3):
        if l == 0:
            shapes.append([1] + [16])
        else:
            shapes.append([3 for _ in range(l)] + [16])
    
    assert out_product.shape == (8, 208)
    
    assert out_product[:, slices[0]].reshape(-1, *shapes[0]).shape == (8, 1, 16)
    assert out_product[:, slices[1]].reshape(-1, *shapes[1]).shape== (8, 3, 16)
    assert out_product[:, slices[2]].reshape(-1, *shapes[2]).shape == (8, 3, 3, 16)
    
    assert is_traceless(out_product[:, slices[0]].reshape(-1, *shapes[0]))
    assert is_traceless(out_product[:, slices[1]].reshape(-1, *shapes[1]))
    assert is_traceless(out_product[:, slices[2]].reshape(-1, *shapes[2]))
    
    assert is_symmetric(out_product[:, slices[0]].reshape(-1, *shapes[0]))
    assert is_symmetric(out_product[:, slices[1]].reshape(-1, *shapes[1]))
    assert is_symmetric(out_product[:, slices[2]].reshape(-1, *shapes[2]))


if __name__ == '__main__':
    
    torch.set_default_dtype(torch.float64)
    
    for seed in range(10):
        test_interaction_layer(seed=seed)
        test_product_basis_layer(seed=seed)
