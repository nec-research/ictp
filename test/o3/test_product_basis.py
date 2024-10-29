import scipy

import torch

from ictp.o3.cartesian_harmonics import CartesianHarmonics
from ictp.o3.tensor_product import PlainTensorProduct, WeightedTensorProduct
from ictp.o3.linear_transform import LinearTransform
from ictp.o3.product_basis import WeightedPathSummation, WeightedProductBasis

from utils import is_symmetric, is_traceless


def test_weighted_sum(seed: int):
    # set seed
    torch.manual_seed(seed)
    
    # define a random rotational matrix
    R = torch.as_tensor(scipy.spatial.transform.Rotation.random().as_matrix(), 
                        dtype=torch.get_default_dtype())
    
    # define random positions, and rotate them
    x = torch.rand(32, 3)
    R_x = torch.einsum('ij, aj -> ai', R, x)
    
    # define Cartesian harmonics
    cartesian_harmonics = CartesianHarmonics(l_max=3)
    
    # compute Cartesian harmonics for original and rotated positions
    x_chs = cartesian_harmonics(x)
    R_x_chs = cartesian_harmonics(R_x)
    
     # define first random features
    slices = []
    k = 0
    for l in range(4):
        slices.append(slice(k,  k + 3 ** l, None))
        k += 3 ** l
    
    shapes = []
    for l in range(4):
        if l == 0:
            shapes.append([1])
        else:
            shapes.append([3 for _ in range(l)] + [1])
            
    A, B, C, D = torch.randn(1, 16), torch.randn(1, 16), torch.randn(1, 16), torch.randn(1, 16)
    
    x_chs = torch.cat(
        [
            torch.einsum('au,uv->av', x_chs[:, slices[0]].reshape(-1, *shapes[0]), A).reshape(-1, 16),
            torch.einsum('aiu,uv->aiv', x_chs[:, slices[1]].reshape(-1, *shapes[1]), B).reshape(-1, 3 * 16),
            torch.einsum('aiju,uv->aijv', x_chs[:, slices[2]].reshape(-1, *shapes[2]), C).reshape(-1, (3 ** 2) * 16),
            torch.einsum('aijku,uv->aijkv', x_chs[:, slices[3]].reshape(-1, *shapes[3]), D).reshape(-1, (3 ** 3) * 16),
        ], -1)
    
    R_x_chs = torch.cat(
        [
            torch.einsum('au,uv->av', R_x_chs[:, slices[0]].reshape(-1, *shapes[0]), A).reshape(-1, 16),
            torch.einsum('aiu,uv->aiv', R_x_chs[:, slices[1]].reshape(-1, *shapes[1]), B).reshape(-1, 3 * 16),
            torch.einsum('aiju,uv->aijv', R_x_chs[:, slices[2]].reshape(-1, *shapes[2]), C).reshape(-1, (3 ** 2) * 16),
            torch.einsum('aijku,uv->aijkv', R_x_chs[:, slices[3]].reshape(-1, *shapes[3]), D).reshape(-1, (3 ** 3) * 16),
        ], -1)
    
    # define the second set of random features
    y = torch.rand(32, 3)
    y = y / torch.norm(y, dim=-1, keepdim=True) # prepare unit vectors
    R_y = torch.einsum('ij, aj -> ai', R, y)
    
    cartesian_harmonics = CartesianHarmonics(l_max=3)
    
    y_chs = cartesian_harmonics(y)
    R_y_chs = cartesian_harmonics(R_y)
    
    slices = []
    k = 0
    for l in range(4):
        slices.append(slice(k,  k + 3 ** l, None))
        k += 3 ** l
    
    shapes = []
    for l in range(4):
        if l == 0:
            shapes.append([1])
        else:
            shapes.append([3 for _ in range(l)] + [1])
            
    A, B, C, D = torch.randn(1, 16, 4), torch.randn(1, 16, 6), torch.randn(1, 16, 7), torch.randn(1, 16, 6)
    
    y_chs = torch.cat(
        [
            torch.einsum('au,uvw->avw', y_chs[:, slices[0]].reshape(-1, *shapes[0]), A).reshape(-1, 16 * 4),
            torch.einsum('aiu,uvw->aivw', y_chs[:, slices[1]].reshape(-1, *shapes[1]), B).reshape(-1, 3 * 16 * 6),
            torch.einsum('aiju,uvw->aijvw', y_chs[:, slices[2]].reshape(-1, *shapes[2]), C).reshape(-1, (3 ** 2) * 16 * 7),
            torch.einsum('aijku,uvw->aijkvw', y_chs[:, slices[3]].reshape(-1, *shapes[3]), D).reshape(-1, (3 ** 3) * 16 * 6)
        ], -1)
    
    R_y_chs = torch.cat(
        [
            torch.einsum('au,uvw->avw', R_y_chs[:, slices[0]].reshape(-1, *shapes[0]), A).reshape(-1, 16 * 4),
            torch.einsum('aiu,uvw->aivw', R_y_chs[:, slices[1]].reshape(-1, *shapes[1]), B).reshape(-1, 3 * 16 * 6),
            torch.einsum('aiju,uvw->aijvw', R_y_chs[:, slices[2]].reshape(-1, *shapes[2]), C).reshape(-1, (3 ** 2) * 16 * 7),
            torch.einsum('aijku,uvw->aijkvw', R_y_chs[:, slices[3]].reshape(-1, *shapes[3]), D).reshape(-1, (3 ** 3) * 16 * 6)
        ], -1)
    
    # define tensor product
    tp = PlainTensorProduct(in1_l_max=3, in2_l_max=3, out_l_max=3, 
                            in1_features=16, in2_features=16, out_features=16, 
                            in1_paths=[4, 6, 7, 6])
    
    # compute tensor product for initial and rotated inputs
    out_tp = tp(y_chs, x_chs)
    R_out_tp = tp(R_y_chs, R_x_chs)
    
    n_paths = tp.n_paths
    
    # check the accumulated numbers of paths
    assert n_paths == [23, 36, 42, 36]
    
    # check shapes
    assert out_tp.shape == (32, 23696)
    
    y = torch.nn.functional.one_hot(torch.arange(0, 32) % 4)
    y = torch.as_tensor(y, dtype=torch.get_default_dtype())
    
    # define weighted path summation
    ws = WeightedPathSummation(in1_l_max=3, out_l_max=3, in1_features=16,
                               in2_features=4, in1_paths=n_paths)
    
    # compute weighted path sums
    out_ws = ws(out_tp, y)
    R_out_ws = ws(R_out_tp, y)
    
    # check shapes of Cartesian harmonics, as well as, their properties
    # i.e., if they are traceless, symmetric, and normalized
    assert out_ws.shape==(32, 640)
    
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
    
    assert is_traceless(out_ws[:, slices[0]].reshape(-1, *shapes[0]))
    assert is_traceless(out_ws[:, slices[1]].reshape(-1, *shapes[1]))
    assert is_traceless(out_ws[:, slices[2]].reshape(-1, *shapes[2]))
    assert is_traceless(out_ws[:, slices[3]].reshape(-1, *shapes[3]))
    
    assert is_symmetric(out_ws[:, slices[0]].reshape(-1, *shapes[0]))
    assert is_symmetric(out_ws[:, slices[1]].reshape(-1, *shapes[1]))
    assert is_symmetric(out_ws[:, slices[2]].reshape(-1, *shapes[2]))
    assert is_symmetric(out_ws[:, slices[3]].reshape(-1, *shapes[3]))
    
    # check equivariance
    R_out_ws_new = torch.cat(
        [
            out_ws[:, slices[0]].reshape(-1, 16),
            torch.einsum(
                'ij, aju -> aiu', R, out_ws[:, slices[1]].reshape(-1, *shapes[1])
                ).reshape(-1, 3 * 16),
            torch.einsum(
                'ij, kl, ajlu -> aiku', R, R, out_ws[:, slices[2]].reshape(-1, *shapes[2])
                ).reshape(-1, (3 ** 2) * 16),
            torch.einsum(
                'ij, kl, mn, ajlnu-> aikmu', R, R, R, out_ws[:, slices[3]].reshape(-1, *shapes[3])
                ).reshape(-1, (3 ** 3)  * 16),
        ], -1)
    assert (abs(R_out_ws_new - R_out_ws) < 1e-14).all()
            
            
def test_product_basis(seed: int):
    # set seed
    torch.manual_seed(seed)
    
    # define a random rotational matrix
    R = torch.as_tensor(scipy.spatial.transform.Rotation.random().as_matrix(), 
                        dtype=torch.get_default_dtype())
    
    # define random positions, and rotate them
    x = torch.rand(32, 3)
    R_x = torch.einsum('ij, aj -> ai', R, x)
    
    # define Cartesian harmonics
    cartesian_harmonics = CartesianHarmonics(l_max=3)
    
    # compute Cartesian harmonics for original and rotated positions
    x_chs = cartesian_harmonics(x)
    R_x_chs = cartesian_harmonics(R_x)
    
     # define first random features
    slices = []
    k = 0
    for l in range(4):
        slices.append(slice(k,  k + 3 ** l, None))
        k += 3 ** l
    
    shapes = []
    for l in range(4):
        if l == 0:
            shapes.append([1])
        else:
            shapes.append([3 for _ in range(l)] + [1])
            
    A, B, C, D = torch.randn(1, 8), torch.randn(1, 8), torch.randn(1, 8), torch.randn(1, 8)
    
    x_chs = torch.cat(
        [
            torch.einsum('au,uv->av', x_chs[:, slices[0]].reshape(-1, *shapes[0]), A).reshape(-1, 8),
            torch.einsum('aiu,uv->aiv', x_chs[:, slices[1]].reshape(-1, *shapes[1]), B).reshape(-1, 3 * 8),
            torch.einsum('aiju,uv->aijv', x_chs[:, slices[2]].reshape(-1, *shapes[2]), C).reshape(-1, (3 ** 2) * 8),
            torch.einsum('aijku,uv->aijkv', x_chs[:, slices[3]].reshape(-1, *shapes[3]), D).reshape(-1, (3 ** 3) * 8),
        ], -1)
    
    R_x_chs = torch.cat(
        [
            torch.einsum('au,uv->av', R_x_chs[:, slices[0]].reshape(-1, *shapes[0]), A).reshape(-1, 8),
            torch.einsum('aiu,uv->aiv', R_x_chs[:, slices[1]].reshape(-1, *shapes[1]), B).reshape(-1, 3 * 8),
            torch.einsum('aiju,uv->aijv', R_x_chs[:, slices[2]].reshape(-1, *shapes[2]), C).reshape(-1, (3 ** 2) * 8),
            torch.einsum('aijku,uv->aijkv', R_x_chs[:, slices[3]].reshape(-1, *shapes[3]), D).reshape(-1, (3 ** 3) * 8),
        ], -1)
    
    # define the second set of random features
    y = torch.rand(32, 3)
    y = y / torch.norm(y, dim=-1, keepdim=True) # prepare unit vectors
    R_y = torch.einsum('ij, aj -> ai', R, y)
    
    cartesian_harmonics = CartesianHarmonics(l_max=3)
    
    y_chs = cartesian_harmonics(y)
    R_y_chs = cartesian_harmonics(R_y)
    
    slices = []
    k = 0
    for l in range(4):
        slices.append(slice(k,  k + 3 ** l, None))
        k += 3 ** l
    
    shapes = []
    for l in range(4):
        if l == 0:
            shapes.append([1])
        else:
            shapes.append([3 for _ in range(l)] + [1])
            
    A, B, C, D = torch.randn(1, 16), torch.randn(1, 16), torch.randn(1, 16), torch.randn(1, 16)
    
    y_chs = torch.cat(
        [
            torch.einsum('au,uv->av', y_chs[:, slices[0]].reshape(-1, *shapes[0]), A).reshape(-1, 16),
            torch.einsum('aiu,uv->aiv', y_chs[:, slices[1]].reshape(-1, *shapes[1]), B).reshape(-1, 3 * 16),
            torch.einsum('aiju,uv->aijv', y_chs[:, slices[2]].reshape(-1, *shapes[2]), C).reshape(-1, (3 ** 2) * 16),
            torch.einsum('aijku,uv->aijkv', y_chs[:, slices[3]].reshape(-1, *shapes[3]), D).reshape(-1, (3 ** 3) * 16)
        ], -1)
    
    R_y_chs = torch.cat(
        [
            torch.einsum('au,uv->av', R_y_chs[:, slices[0]].reshape(-1, *shapes[0]), A).reshape(-1, 16),
            torch.einsum('aiu,uv->aiv', R_y_chs[:, slices[1]].reshape(-1, *shapes[1]), B).reshape(-1, 3 * 16),
            torch.einsum('aiju,uv->aijv', R_y_chs[:, slices[2]].reshape(-1, *shapes[2]), C).reshape(-1, (3 ** 2) * 16),
            torch.einsum('aijku,uv->aijkv', R_y_chs[:, slices[3]].reshape(-1, *shapes[3]), D).reshape(-1, (3 ** 3) * 16)
        ], -1)
    
    # define tensor product
    tp = WeightedTensorProduct(in1_l_max=3, in2_l_max=3, out_l_max=3,
                               in1_features=16, in2_features=8, out_features=16,
                               connection_mode='uvu', internal_weights=False, 
                               shared_weights=False)
    
    # define weights
    weight = torch.randn(32, tp.n_total_paths, 16, 8)
    
    # compute tensor product for initial and rotated inputs
    out_tp = tp(y_chs, x_chs, weight)
    R_out_tp = tp(R_y_chs, R_x_chs, weight)
    
    # define linear transformation
    linear = LinearTransform(in_l_max=3, out_l_max=3, 
                             in_features=16, out_features=16,
                             in_paths=[n for n in tp.n_paths])
    
    out_linear = linear(out_tp)
    R_out_linear = linear(R_out_tp)
    
    n_paths = tp.n_paths
    
    # check the accumulated numbers of paths
    assert n_paths == [4, 6, 7, 6]
    
    # check shapes
    assert out_tp.shape == (32, 3952)
    
    y = torch.nn.functional.one_hot(torch.arange(0, 32) % 4)
    y = torch.as_tensor(y, dtype=torch.get_default_dtype())
    
    # define product basis
    product_basis = WeightedProductBasis(in1_l_max=3, out_l_max=2, in1_features=16,
                                         in2_features=4, correlation=3)
    
    assert len(product_basis.blocks) == 2
    assert product_basis.blocks[0].tp.n_paths == [4, 3, 5, 3]
    assert product_basis.blocks[1].tp.n_paths == [15, 23, 26] 
    
    out_product_basis = product_basis(out_linear, y)
    R_out_product_basis = product_basis(R_out_linear, y)
    
    # check shapes of Cartesian harmonics, as well as, their properties
    # i.e., if they are traceless, symmetric, and normalized
    assert out_product_basis.shape==(32, 208)
    
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
    
    assert is_traceless(out_product_basis[:, slices[0]].reshape(-1, *shapes[0]))
    assert is_traceless(out_product_basis[:, slices[1]].reshape(-1, *shapes[1]))
    assert is_traceless(out_product_basis[:, slices[2]].reshape(-1, *shapes[2]))
    
    assert is_symmetric(out_product_basis[:, slices[0]].reshape(-1, *shapes[0]))
    assert is_symmetric(out_product_basis[:, slices[1]].reshape(-1, *shapes[1]))
    assert is_symmetric(out_product_basis[:, slices[2]].reshape(-1, *shapes[2]))
    
    # check equivariance
    R_out_product_basis_new = torch.cat(
        [
            out_product_basis[:, slices[0]].reshape(-1, 16),
            torch.einsum(
                'ij, aju -> aiu', R, out_product_basis[:, slices[1]].reshape(-1, *shapes[1])
                ).reshape(-1, 3 * 16),
            torch.einsum(
                'ij, kl, ajlu -> aiku', R, R, out_product_basis[:, slices[2]].reshape(-1, *shapes[2])
                ).reshape(-1, (3 ** 2) * 16)
        ], -1)
    
    assert (abs(R_out_product_basis_new - R_out_product_basis) < 1e-14).all()


if __name__ == '__main__':
    
    torch.set_default_dtype(torch.float64)
    
    for seed in range(10):
        test_weighted_sum(seed=seed)
        test_product_basis(seed=seed)
