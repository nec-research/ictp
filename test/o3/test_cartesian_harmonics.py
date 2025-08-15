import scipy
import torch

from ictp.o3.cartesian_harmonics import CartesianHarmonics
from utils import is_traceless, is_symmetric, is_normalized


def test_cartesian_harmonics(seed: int):
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
    
    # check shapes of Cartesian harmonics, as well as, their properties
    # i.e., if they are traceless, symmetric, and normalized
    slices = []
    k = 0
    for l in range(4):
        slices.append(slice(k,  k + 3 ** l * 1, None))
        k += 3 ** l * 1
    
    shapes = []
    for l in range(4):
        if l == 0:
            shapes.append([1] + [1])
        else:
            shapes.append([3 for _ in range(l)] + [1])
    
    assert x_chs.shape == (32, 40)
    
    assert x_chs[:, slices[0]].reshape(-1, *shapes[0]).shape == (32, 1, 1)
    assert x_chs[:, slices[1]].reshape(-1, *shapes[1]).shape == (32, 3, 1)
    assert x_chs[:, slices[2]].reshape(-1, *shapes[2]).shape == (32, 3, 3, 1)
    assert x_chs[:, slices[3]].reshape(-1, *shapes[3]).shape == (32, 3, 3, 3, 1)
    
    assert is_traceless(x_chs[:, slices[0]].reshape(-1, *shapes[0]))
    assert is_traceless(x_chs[:, slices[1]].reshape(-1, *shapes[1]))
    assert is_traceless(x_chs[:, slices[2]].reshape(-1, *shapes[2]))
    assert is_traceless(x_chs[:, slices[3]].reshape(-1, *shapes[3]))
    
    assert is_symmetric(x_chs[:, slices[0]].reshape(-1, *shapes[0]))
    assert is_symmetric(x_chs[:, slices[1]].reshape(-1, *shapes[1]))
    assert is_symmetric(x_chs[:, slices[2]].reshape(-1, *shapes[2]))
    assert is_symmetric(x_chs[:, slices[3]].reshape(-1, *shapes[3]))
    
    assert is_normalized(x_chs[:, slices[1]].reshape(-1, *shapes[1]), 
                         x_chs[:, slices[0]].reshape(-1, *shapes[0]))
    assert is_normalized(x_chs[:, slices[1]].reshape(-1, *shapes[1]), 
                         x_chs[:, slices[1]].reshape(-1, *shapes[1]))
    assert is_normalized(x_chs[:, slices[1]].reshape(-1, *shapes[1]), 
                         x_chs[:, slices[2]].reshape(-1, *shapes[2]))
    assert is_normalized(x_chs[:, slices[1]].reshape(-1, *shapes[1]), 
                         x_chs[:, slices[3]].reshape(-1, *shapes[3]))
    
    # check equivariance of Cartesian harmonics
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
            shapes.append([3 for _ in range(l)])
    
    R_x_chs_new = torch.cat(
        [
            x_chs[:, slices[0]],
            torch.einsum(
                'ij, aj -> ai', R, x_chs[:, slices[1]].reshape(-1, *shapes[1])
                ),
            torch.einsum(
                'ij, kl, ajl -> aik', R, R, x_chs[:, slices[2]].reshape(-1, *shapes[2])
                ).reshape(-1, 3 ** 2),
            torch.einsum(
                'ij, kl, mn, ajln-> aikm', R, R, R, x_chs[:, slices[3]].reshape(-1, *shapes[3])
                ).reshape(-1, 3 ** 3),
        ], -1)
    
    assert (abs(R_x_chs_new - R_x_chs) < 1e-14).all()


if __name__ == '__main__':
    
    torch.set_default_dtype(torch.float64)
    
    for seed in range(10):
        test_cartesian_harmonics(seed=seed)
