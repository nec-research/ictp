import scipy

import torch

from ictp.o3.cartesian_harmonics import CartesianHarmonics
from ictp.o3.tensor_product import PlainTensorProduct, WeightedTensorProduct
from ictp.o3.linear_transform import LinearTransform

from utils import is_traceless, is_symmetric, is_normalized


def test_plain_tensor_product_for_same_unit_vectors(seed: int):
    # this function allows for checking the normalization of tensors 
    # produced via the tensor product
    
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
    
    # define tensor products
    # we do not allow path normalization to check if the output tensors 
    # are normalized correctly
    tp = PlainTensorProduct(in1_l_max=3, in2_l_max=3, out_l_max=3,
                            in1_features=1, in2_features=1, out_features=1)
    
    # compute the output of the tensor product
    out = tp(x_chs, x_chs)
    R_out = tp(R_x_chs, R_x_chs)
    
    # check shapes of Cartesian harmonics, as well as, their properties
    # i.e., if they are traceless, symmetric, and normalized
    n_paths = tp.n_paths
    
    slices = []
    k = 0
    for l in range(4):
        slices.append(slice(k,  k + 3 ** l * 1 * n_paths[l], None))
        k += 3 ** l * 1 * n_paths[l]
    
    shapes = []
    for l in range(4):
        if l == 0:
            shapes.append([1] + [1 * n_paths[l]])
        else:
            shapes.append([3 for _ in range(l)] + [1 * n_paths[l]])
            
    assert out.shape == (32, 247)
    
    assert is_traceless(out[:, slices[0]].reshape(-1, *shapes[0]))
    assert is_traceless(out[:, slices[1]].reshape(-1, *shapes[1]))
    assert is_traceless(out[:, slices[2]].reshape(-1, *shapes[2]))
    assert is_traceless(out[:, slices[3]].reshape(-1, *shapes[3]))
    
    assert is_symmetric(out[:, slices[0]].reshape(-1, *shapes[0]))
    assert is_symmetric(out[:, slices[1]].reshape(-1, *shapes[1]))
    assert is_symmetric(out[:, slices[2]].reshape(-1, *shapes[2]))
    assert is_symmetric(out[:, slices[3]].reshape(-1, *shapes[3]))
    
    assert is_normalized(out[:, slices[1]].reshape(-1, *shapes[1]), out[:, slices[0]].reshape(-1, *shapes[0]))
    assert is_normalized(out[:, slices[1]].reshape(-1, *shapes[1]), out[:, slices[1]].reshape(-1, *shapes[1]))
    assert is_normalized(out[:, slices[1]].reshape(-1, *shapes[1]), out[:, slices[2]].reshape(-1, *shapes[2]))
    assert is_normalized(out[:, slices[1]].reshape(-1, *shapes[1]), out[:, slices[3]].reshape(-1, *shapes[3]))
    
    # check equivariance of the tensor product
    R_out_new = torch.cat(
        [
            out[:, slices[0]].reshape(-1, 1 * n_paths[0]),
            torch.einsum(
                'ij, aju -> aiu', R, out[:, slices[1]].reshape(-1, *shapes[1])
                ).reshape(-1, 3 * 1 * n_paths[1]),
            torch.einsum(
                'ij, kl, ajlu -> aiku', R, R, out[:, slices[2]].reshape(-1, *shapes[2])
                ).reshape(-1, (3 ** 2) * 1 * n_paths[2]),
            torch.einsum(
                'ij, kl, mn, ajlnu-> aikmu', R, R, R, out[:, slices[3]].reshape(-1, *shapes[3])
                ).reshape(-1, (3 ** 3) * 1 * n_paths[3]),
        ], -1)
    
    assert (abs(R_out_new - R_out) < 1e-13).all()


def test_weighted_tensor_product_for_same_unit_vectors(seed: int):
    # this function allows for checking the normalization of tensors 
    # produced via the tensor product
    
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
    
    # define tensor products
    # we do not allow path normalization to check if the output tensors 
    # are normalized correctly
    tp = WeightedTensorProduct(in1_l_max=3, in2_l_max=3, out_l_max=3,
                               in1_features=1, in2_features=1, out_features=1, 
                               connection_mode='uvu', internal_weights=False,
                               shared_weights=True)
    
    # define constant weights for the weighted tensor product
    weight = torch.ones(tp.n_total_paths, 1, 1)
    
    # compute the output of the tensor product
    out = tp(x_chs, x_chs, weight=weight)
    R_out = tp(R_x_chs, R_x_chs, weight=weight)
    
    # check shapes of Cartesian harmonics, as well as, their properties
    # i.e., if they are traceless, symmetric, and normalized
    n_paths = tp.n_paths
    
    slices = []
    k = 0
    for l in range(4):
        slices.append(slice(k,  k + 3 ** l * 1 * n_paths[l], None))
        k += 3 ** l * 1 * n_paths[l]
    
    shapes = []
    for l in range(4):
        if l == 0:
            shapes.append([1] + [1 * n_paths[l]])
        else:
            shapes.append([3 for _ in range(l)] + [1 * n_paths[l]])
            
    assert out.shape == (32, 247)
    
    assert is_traceless(out[:, slices[0]].reshape(-1, *shapes[0]))
    assert is_traceless(out[:, slices[1]].reshape(-1, *shapes[1]))
    assert is_traceless(out[:, slices[2]].reshape(-1, *shapes[2]))
    assert is_traceless(out[:, slices[3]].reshape(-1, *shapes[3]))
    
    assert is_symmetric(out[:, slices[0]].reshape(-1, *shapes[0]))
    assert is_symmetric(out[:, slices[1]].reshape(-1, *shapes[1]))
    assert is_symmetric(out[:, slices[2]].reshape(-1, *shapes[2]))
    assert is_symmetric(out[:, slices[3]].reshape(-1, *shapes[3]))
    
    # check equivariance of the weighted tensor product
    
    R_out_new = torch.cat(
        [
            out[:, slices[0]].reshape(-1, 1 * n_paths[0]),
            torch.einsum(
                'ij, aju -> aiu', R, out[:, slices[1]].reshape(-1, *shapes[1])
                ).reshape(-1, 3 * 1 * n_paths[1]),
            torch.einsum(
                'ij, kl, ajlu -> aiku', R, R, out[:, slices[2]].reshape(-1, *shapes[2])
                ).reshape(-1, (3 ** 2) * 1 * n_paths[2]),
            torch.einsum(
                'ij, kl, mn, ajlnu-> aikmu', R, R, R, out[:, slices[3]].reshape(-1, *shapes[3])
                ).reshape(-1, (3 ** 3) * 1 * n_paths[3]),
        ], -1)
    
    assert (abs(R_out_new - R_out) < 1e-13).all()


def test_weighted_tensor_product_between_scalars(seed: int):
    # this function allows for checking the shapes and entries of tensors produced 
    # when using tensor product between scalars but expecting a tensor of higher order
    
    # set seed
    torch.manual_seed(seed)
    
    # define a random rotational matrix
    R = torch.as_tensor(scipy.spatial.transform.Rotation.random().as_matrix(), 
                        dtype=torch.get_default_dtype())
    
    # define random positions, and rotate them
    x = torch.rand(32, 3)
    R_x = torch.einsum('ij, aj -> ai', R, x)
    
    # define Cartesian harmonics
    cartesian_harmonics = CartesianHarmonics(l_max=0)
    
    # compute Cartesian harmonics for original and rotated positions
    x_chs = cartesian_harmonics(x)
    R_x_chs = cartesian_harmonics(R_x)
    
    # multiply with a random matrix
    random_matrix = torch.randn(1, 16)
    x_chs = torch.einsum('au,uv->av', x_chs, random_matrix)
    R_x_chs = torch.einsum('au,uv->av', R_x_chs, random_matrix)
    
    # define the one-hot encoded attributes
    y = torch.nn.functional.one_hot(torch.arange(0, 32) % 4)
    y = torch.as_tensor(y, dtype=torch.get_default_dtype())
    
    # define tensor product
    tp = WeightedTensorProduct(in1_l_max=0, in2_l_max=0, out_l_max=2,
                               in1_features=16, in2_features=4, out_features=16, 
                               connection_mode='uvw')
    
    # define constant weights
    weight = torch.ones(tp.n_total_paths, 16, 4, 16)
    
    # compute the output of the tensor product
    out = tp(x_chs, y, weight=weight)
    R_out = tp(R_x_chs, y, weight=weight)
    
    n_paths = tp.n_paths
    
    slices = []
    k = 0
    for l in range(3):
        slices.append(slice(k,  k + 3 ** l * 16 * n_paths[l], None))
        k += 3 ** l * 16 * n_paths[l]
    
    shapes = []
    for l in range(3):
        if l == 0:
            shapes.append([16 * n_paths[l]])
        else:
            shapes.append([3 for _ in range(l)] + [16 * n_paths[l]])
    
    assert out.shape == (32, 208)
    
    assert (out[:, slices[0]].reshape(-1, *shapes[0]) != 0.0).all()
    assert (out[:, slices[1]].reshape(-1, *shapes[1]) == 0.0).all()
    assert (out[:, slices[2]].reshape(-1, *shapes[2]) == 0.0).all()
    
    assert out[:, slices[0]].reshape(-1, *shapes[0]).shape == (32, 16)
    assert out[:, slices[1]].reshape(-1, *shapes[1]).shape == (32, 3, 16)
    assert out[:, slices[2]].reshape(-1, *shapes[2]).shape == (32, 3, 3, 16)
    
    # check equivariance of the tensor product between scalars (is kind of trivial)
    R_out_new = torch.cat(
        [
            out[:, slices[0]].reshape(-1, 16 * n_paths[0]),
            torch.einsum(
                'ij, aju -> aiu', R, out[:, slices[1]].reshape(-1, *shapes[1])
                ).reshape(-1, 3 * 16 * n_paths[1]),
            torch.einsum(
                'ij, kl, ajlu -> aiku', R, R, out[:, slices[2]].reshape(-1, *shapes[2])
                ).reshape(-1, (3 ** 2) * 16 * n_paths[2]),
        ], -1)
    
    assert (abs(R_out_new - R_out) < 1e-13).all()


def test_plain_tensor_product(seed: int):
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
            
    A, B, C, D = torch.randn(1, 8, 4), torch.randn(1, 8, 6), torch.randn(1, 8, 7), torch.randn(1, 8, 6)
    
    y_chs = torch.cat(
        [
            torch.einsum('au,uvw->avw', y_chs[:, slices[0]].reshape(-1, *shapes[0]), A).reshape(-1, 8 * 4),
            torch.einsum('aiu,uvw->aivw', y_chs[:, slices[1]].reshape(-1, *shapes[1]), B).reshape(-1, 3 * 8 * 6),
            torch.einsum('aiju,uvw->aijvw', y_chs[:, slices[2]].reshape(-1, *shapes[2]), C).reshape(-1, (3 ** 2) * 8 * 7),
            torch.einsum('aijku,uvw->aijkvw', y_chs[:, slices[3]].reshape(-1, *shapes[3]), D).reshape(-1, (3 ** 3) * 8 * 6)
        ], -1)
    
    R_y_chs = torch.cat(
        [
            torch.einsum('au,uvw->avw', R_y_chs[:, slices[0]].reshape(-1, *shapes[0]), A).reshape(-1, 8 * 4),
            torch.einsum('aiu,uvw->aivw', R_y_chs[:, slices[1]].reshape(-1, *shapes[1]), B).reshape(-1, 3 * 8 * 6),
            torch.einsum('aiju,uvw->aijvw', R_y_chs[:, slices[2]].reshape(-1, *shapes[2]), C).reshape(-1, (3 ** 2) * 8 * 7),
            torch.einsum('aijku,uvw->aijkvw', R_y_chs[:, slices[3]].reshape(-1, *shapes[3]), D).reshape(-1, (3 ** 3) * 8 * 6)
        ], -1)
    
    # define tensor product
    tp = PlainTensorProduct(in1_l_max=3, in2_l_max=3, out_l_max=3, 
                            in1_features=8, in2_features=8, out_features=8, 
                            in1_paths=[4, 6, 7, 6])
    
    # compute tensor product for initial and rotated inputs
    out_tp = tp(y_chs, x_chs)
    R_out_tp = tp(R_y_chs, R_x_chs)
    
    n_paths = tp.n_paths
    
    # check the accumulated numbers of paths
    assert n_paths == [23, 36, 42, 36]
    
    # check shapes of Cartesian harmonics, as well as, their properties
    # i.e., if they are traceless, symmetric, and normalized
    slices = []
    k = 0
    for l in range(4):
        slices.append(slice(k,  k + 3 ** l * 8 * n_paths[l], None))
        k += 3 ** l * 8 * n_paths[l]
    
    shapes = []
    for l in range(4):
        if l == 0:
            shapes.append([1] + [8 * n_paths[l]])
        else:
            shapes.append([3 for _ in range(l)] + [8 * n_paths[l]])
    
    assert out_tp.shape == (32, 11848)
    
    assert is_traceless(out_tp[:, slices[0]].reshape(-1, *shapes[0]))
    assert is_traceless(out_tp[:, slices[1]].reshape(-1, *shapes[1]))
    assert is_traceless(out_tp[:, slices[2]].reshape(-1, *shapes[2]))
    assert is_traceless(out_tp[:, slices[3]].reshape(-1, *shapes[3]))
    
    assert is_symmetric(out_tp[:, slices[0]].reshape(-1, *shapes[0]))
    assert is_symmetric(out_tp[:, slices[1]].reshape(-1, *shapes[1]))
    assert is_symmetric(out_tp[:, slices[2]].reshape(-1, *shapes[2]))
    assert is_symmetric(out_tp[:, slices[3]].reshape(-1, *shapes[3]))
    
    # check equivariance
    R_out_tp_new = torch.cat(
        [
            out_tp[:, slices[0]].reshape(-1, 8 * n_paths[0]),
            torch.einsum(
                'ij, aju -> aiu', R, out_tp[:, slices[1]].reshape(-1, *shapes[1])
                ).reshape(-1, 3 * 8 * n_paths[1]),
            torch.einsum(
                'ij, kl, ajlu -> aiku', R, R, out_tp[:, slices[2]].reshape(-1, *shapes[2])
                ).reshape(-1, (3 ** 2) * 8 * n_paths[2]),
            torch.einsum(
                'ij, kl, mn, ajlnu-> aikmu', R, R, R, out_tp[:, slices[3]].reshape(-1, *shapes[3])
                ).reshape(-1, (3 ** 3)  * 8 * n_paths[3]),
        ], -1)
    
    assert (abs(R_out_tp_new - R_out_tp) < 1e-13).all()
    
    # define linear transformation
    linear = LinearTransform(in_l_max=3, out_l_max=3, 
                             in_features=8, out_features=8,
                             in_paths=[n for n in tp.n_paths])
    
    out_linear = linear(out_tp)
    R_out_linear = linear(R_out_tp)
    
    # check shapes of Cartesian harmonics, as well as, their properties
    # i.e., if they are traceless, symmetric, and normalized
    slices = []
    k = 0
    for l in range(4):
        slices.append(slice(k,  k + 3 ** l * 8, None))
        k += 3 ** l * 8
    
    shapes = []
    for l in range(4):
        if l == 0:
            shapes.append([1] + [8])
        else:
            shapes.append([3 for _ in range(l)] + [8])
            
    assert out_linear.shape == (32, 320)
    
    assert is_traceless(out_linear[:, slices[0]].reshape(-1, *shapes[0]))
    assert is_traceless(out_linear[:, slices[1]].reshape(-1, *shapes[1]))
    assert is_traceless(out_linear[:, slices[2]].reshape(-1, *shapes[2]))
    assert is_traceless(out_linear[:, slices[3]].reshape(-1, *shapes[3]))
    
    assert is_symmetric(out_linear[:, slices[0]].reshape(-1, *shapes[0]))
    assert is_symmetric(out_linear[:, slices[1]].reshape(-1, *shapes[1]))
    assert is_symmetric(out_linear[:, slices[2]].reshape(-1, *shapes[2]))
    assert is_symmetric(out_linear[:, slices[3]].reshape(-1, *shapes[3]))
    
    # check equivariance
    R_out_linear_new = torch.cat(
        [
            out_linear[:, slices[0]].reshape(-1, 8),
            torch.einsum(
                'ij, aju -> aiu', R, out_linear[:, slices[1]].reshape(-1, *shapes[1])
                ).reshape(-1, 3 * 8),
            torch.einsum(
                'ij, kl, ajlu -> aiku', R, R, out_linear[:, slices[2]].reshape(-1, *shapes[2])
                ).reshape(-1, (3 ** 2) * 8),
            torch.einsum(
                'ij, kl, mn, ajlnu-> aikmu', R, R, R, out_linear[:, slices[3]].reshape(-1, *shapes[3])
                ).reshape(-1, (3 ** 3)  * 8),
        ], -1)
    
    assert (abs(R_out_linear_new - R_out_linear) < 1e-13).all()


def test_weighted_tensor_product(seed: int):
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
    
    n_paths = tp.n_paths
    
    # check the accumulated numbers of paths
    assert n_paths == [4, 6, 7, 6]
    
    # check shapes of Cartesian harmonics, as well as, their properties
    # i.e., if they are traceless, symmetric, and normalized
    slices = []
    k = 0
    for l in range(4):
        slices.append(slice(k,  k + 3 ** l * 16 * n_paths[l], None))
        k += 3 ** l * 16 * n_paths[l]
    
    shapes = []
    for l in range(4):
        if l == 0:
            shapes.append([1] + [16 * n_paths[l]])
        else:
            shapes.append([3 for _ in range(l)] + [16 * n_paths[l]])
    
    assert out_tp.shape == (32, 3952)
    
    assert is_traceless(out_tp[:, slices[0]].reshape(-1, *shapes[0]))
    assert is_traceless(out_tp[:, slices[1]].reshape(-1, *shapes[1]))
    assert is_traceless(out_tp[:, slices[2]].reshape(-1, *shapes[2]))
    assert is_traceless(out_tp[:, slices[3]].reshape(-1, *shapes[3]))
    
    assert is_symmetric(out_tp[:, slices[0]].reshape(-1, *shapes[0]))
    assert is_symmetric(out_tp[:, slices[1]].reshape(-1, *shapes[1]))
    assert is_symmetric(out_tp[:, slices[2]].reshape(-1, *shapes[2]))
    assert is_symmetric(out_tp[:, slices[3]].reshape(-1, *shapes[3]))
    
    # check equivariance
    R_out_tp_new = torch.cat(
        [
            out_tp[:, slices[0]].reshape(-1, 16 * n_paths[0]),
            torch.einsum(
                'ij, aju -> aiu', R, out_tp[:, slices[1]].reshape(-1, *shapes[1])
                ).reshape(-1, 3 * 16 * n_paths[1]),
            torch.einsum(
                'ij, kl, ajlu -> aiku', R, R, out_tp[:, slices[2]].reshape(-1, *shapes[2])
                ).reshape(-1, (3 ** 2) * 16 * n_paths[2]),
            torch.einsum(
                'ij, kl, mn, ajlnu-> aikmu', R, R, R, out_tp[:, slices[3]].reshape(-1, *shapes[3])
                ).reshape(-1, (3 ** 3)  * 16 * n_paths[3]),
        ], -1)
    
    assert (abs(R_out_tp_new - R_out_tp) < 1e-13).all()
    
    # define linear transformation
    linear = LinearTransform(in_l_max=3, out_l_max=3, 
                             in_features=16, out_features=16,
                             in_paths=[n for n in tp.n_paths])
    
    out_linear = linear(out_tp)
    R_out_linear = linear(R_out_tp)
    
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
            
    assert out_linear.shape == (32, 640)
    
    assert is_traceless(out_linear[:, slices[0]].reshape(-1, *shapes[0]))
    assert is_traceless(out_linear[:, slices[1]].reshape(-1, *shapes[1]))
    assert is_traceless(out_linear[:, slices[2]].reshape(-1, *shapes[2]))
    assert is_traceless(out_linear[:, slices[3]].reshape(-1, *shapes[3]))
    
    assert is_symmetric(out_linear[:, slices[0]].reshape(-1, *shapes[0]))
    assert is_symmetric(out_linear[:, slices[1]].reshape(-1, *shapes[1]))
    assert is_symmetric(out_linear[:, slices[2]].reshape(-1, *shapes[2]))
    assert is_symmetric(out_linear[:, slices[3]].reshape(-1, *shapes[3]))
    
    # check equivariance
    R_out_linear_new = torch.cat(
        [
            out_linear[:, slices[0]].reshape(-1, 16),
            torch.einsum(
                'ij, aju -> aiu', R, out_linear[:, slices[1]].reshape(-1, *shapes[1])
                ).reshape(-1, 3 * 16),
            torch.einsum(
                'ij, kl, ajlu -> aiku', R, R, out_linear[:, slices[2]].reshape(-1, *shapes[2])
                ).reshape(-1, (3 ** 2) * 16),
            torch.einsum(
                'ij, kl, mn, ajlnu-> aikmu', R, R, R, out_linear[:, slices[3]].reshape(-1, *shapes[3])
                ).reshape(-1, (3 ** 3)  * 16),
        ], -1)
    
    assert (abs(R_out_linear_new - R_out_linear) < 1e-13).all()

def test_symmetry_of_plain_tensor_products(seed: int):
    # set seed
    torch.manual_seed(seed)
    
    # define random positions, and rotate them
    x = torch.rand(32, 3)
    
    # define Cartesian harmonics
    cartesian_harmonics = CartesianHarmonics(l_max=3)
    
    # compute Cartesian harmonics for original and rotated positions
    x_chs = cartesian_harmonics(x)
    
    # define random features
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
    
    # define tensor product
    # we do not allow path normalization to check if the output tensors 
    # are normalized correctly
    tp = PlainTensorProduct(in1_l_max=3, in2_l_max=3, out_l_max=3,
                            in1_features=8, in2_features=8, out_features=8)
    
    # compute the output of the tensor product
    out = tp(x_chs, x_chs)
    
    assert out.shape == (32, 1976)
    
    # check shapes of Cartesian harmonics, as well as, their properties
    # i.e., if they are traceless, symmetric, and normalized
    n_paths = tp.n_paths
    
    slices = []
    k = 0
    for l in range(4):
        slices.append(slice(k,  k + 3 ** l * 8 * n_paths[l], None))
        k += 3 ** l * 8 * n_paths[l]
    
    shapes = []
    for l in range(4):
        if l == 0:
            shapes.append([1] + [8 * n_paths[l]])
        else:
            shapes.append([3 for _ in range(l)] + [8 * n_paths[l]])
            
    l1 = out[:, slices[1]].reshape(-1, *shapes[1])
    l1 = l1.reshape(*l1.shape[:-1], 8, 6)
    
    assert abs(l1[:, :, :, 0] - l1[:, :, :, 3]).mean() < 1e-6
    assert abs(l1[:, :, :, 1] - l1[:, :, :, 4]).mean() < 1e-6
    assert abs(l1[:, :, :, 2] - l1[:, :, :, 5]).mean() < 1e-6
            
    l2 = out[:, slices[2]].reshape(-1, *shapes[2])
    l2 = l2.reshape(*l2.shape[:-1], 8, 7)
    
    assert abs(l2[:, :, :, :, 0] - l2[:, :, :, :, 5]).mean() < 1e-6
    assert abs(l2[:, :, :, :, 4] - l2[:, :, :, :, 6]).mean() < 1e-6
    
    l3 = out[:, slices[3]].reshape(-1, *shapes[3])
    l3 = l3.reshape(*l3.shape[:-1], 8, 6)
    
    assert abs(l3[:, :, :, :, :, 0] - l3[:, :, :, :, :, 3]).mean() < 1e-6
    assert abs(l3[:, :, :, :, :, 1] - l3[:, :, :, :, :, 4]).mean() < 1e-6
    assert abs(l3[:, :, :, :, :, 2] - l3[:, :, :, :, :, 5]).mean() < 1e-6
    
    # define symmetric tensor product
    # we do not allow path normalization to check if the output tensors 
    # are normalized correctly
    tp = PlainTensorProduct(in1_l_max=3, in2_l_max=3, out_l_max=3,
                            in1_features=8, in2_features=8, out_features=8, 
                            symmetric_product=True)
    
    # compute the output of the tensor product
    out = tp(x_chs, x_chs)
    
    assert out.shape == (32, 1112)


if __name__ == '__main__':
    
    torch.set_default_dtype(torch.float64)
    
    for seed in range(10):
        test_plain_tensor_product_for_same_unit_vectors(seed=seed)
        test_weighted_tensor_product_for_same_unit_vectors(seed=seed)
        test_weighted_tensor_product_between_scalars(seed=seed)
        test_plain_tensor_product(seed=seed)
        test_weighted_tensor_product(seed=seed)
        test_symmetry_of_plain_tensor_products(seed=seed)
