import sys

# set python paths
sys.path.append('/mnt/local/vzaverkin/projects/ictp')

import torch

from ictp.nn.radial import BesselRBF, PolynomialCutoff
from ictp.nn.layers import RadialEmbeddingLayer


def test_bessel_basis():
        d = torch.linspace(start=0.5, end=5.5, steps=10)
        bessel_basis = BesselRBF(r_cutoff=6.0, n_basis=5)
        output = bessel_basis(d.unsqueeze(-1))
        assert output.shape == (10, 5)


def test_polynomial_cutoff():
    d = torch.linspace(start=0.5, end=5.5, steps=10)
    cutoff_fn = PolynomialCutoff(r_cutoff=5.0)
    output = cutoff_fn(d)
    assert output.shape == (10, )
    
    
def test_radial_embedding_layer():
    d = torch.linspace(start=0.5, end=5.5, steps=10)
    radial_embedding = RadialEmbeddingLayer(r_cutoff=5.0, n_basis=8, n_polynomial_cutoff=5)
    assert radial_embedding(d.unsqueeze(-1)).shape == (10, 8)


if __name__ == '__main__':
    test_bessel_basis()
    test_polynomial_cutoff()
    test_radial_embedding_layer()
