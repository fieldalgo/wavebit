"""A contour's traversal must not depend on a half-filled raster's corner phase."""
import numpy as np
import pytest
from wavebit.rcwa._engine.factorization import normal_vector_field
from wavebit.lattice import as_lattice

pytest.importorskip('contourpy')


@pytest.mark.parametrize('percolating', [False, True])
@pytest.mark.parametrize('shear', [0., .35])
def test_half_fill_projector_survives_translation_and_material_relabeling(percolating, shear):
    n = 64
    U, V = np.meshgrid(np.arange(n), np.arange(n))
    if percolating:
        offset = np.round(4*np.sin(2*np.pi*V/n)).astype(int)
        inside = (U-offset) % n < n//2
    else:
        inside = ((V >= 8) & (V < 56) & (U >= 8) & (U < 50)) | (
            (V >= 8) & (V < 40) & (U == 50))
    assert inside.sum()*2 == inside.size
    cell = np.where(inside, 4.+.2j, 1.+.05j)
    lattice = as_lattice(lattice=[[1., shear], [0., 1.]])
    def nvf_projector(eps):
        nx, ny = normal_vector_field(eps, lattice=lattice, method='contour')
        return np.stack([nx*nx, nx*ny, ny*ny])
    baseline = nvf_projector(cell)
    for shift in ((0, 0), (11, 23), (32, 32), (49, 7)):
        rolled = np.roll(cell, shift, axis=(0, 1))
        expected = np.roll(baseline, shift, axis=(1, 2))
        np.testing.assert_allclose(nvf_projector(rolled), expected, rtol=0, atol=2e-10)
        relabeled = np.where(rolled == 4.+.2j, 1.+.05j, 4.+.2j)
        np.testing.assert_allclose(nvf_projector(relabeled), expected, rtol=0, atol=2e-10)
