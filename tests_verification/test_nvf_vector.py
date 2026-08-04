import numpy as np
import pytest
import fieldwave
from fieldwave._core.modeler import Modeling
import array_api_compat.numpy as xnp
from fieldwave._core.analytic import normal_vector_field_analytical as _nvf_analytic
from fieldwave._core.factorization import nvf_conv_matrices as _nvf_conv


def get_normal_vector_field_analytical(grid_shape, shapes, period=(1000.0, 1000.0)):
    return _nvf_analytic(xnp, grid_shape, shapes, period, xnp.float64)


def to_conv_mat_nvf_2d(eps_matrix, fto_x, fto_y, nx=None, ny=None):
    return _nvf_conv(xnp, eps_matrix, nx, ny, fto_x, fto_y)

def test_analytical_normal_vector_generation():
    modeler = Modeling(period=[1000.0, 1000.0])
    modeler.shapes_list.append(dict(type='circle', cx=500.0, cy=500.0, lx=600.0, ly=600.0))
    
    nx, ny = get_normal_vector_field_analytical((100, 100), modeler.shapes_list, period=modeler.period)
    
    assert nx.shape == (100, 100)
    assert ny.shape == (100, 100)
    
    # Check unit vector normalization
    norm = np.hypot(nx, ny)
    np.testing.assert_allclose(norm, 1.0, atol=1e-5)

def test_vector_nvf_energy_conservation():
    N = 60
    grid_y, grid_x = np.ogrid[:N, :N]
    radius = 18
    dist = np.sqrt((grid_x - N//2)**2 + (grid_y - N//2)**2)
    eps_matrix = np.ones((N, N), dtype=complex)
    eps_matrix[dist <= radius] = 2.0**2
    ucell_raster = np.expand_dims(np.sqrt(eps_matrix), axis=0)

    shapes = [dict(type='circle', cx=500.0, cy=500.0, lx=600.0, ly=600.0)]
    nx_vec, ny_vec = get_normal_vector_field_analytical((N, N), shapes, period=(1000.0, 1000.0))

    fto_val = 2
    Exx, Eyy, Exy, Eyx, Epz_i = to_conv_mat_nvf_2d(eps_matrix, fto_val, fto_val, nx=nx_vec, ny=ny_vec)

    m = fieldwave.rcwa(
        backend=0, fto=[fto_val, fto_val],
        pol=1, n_top=1.0, n_bot=1.45,
        theta=10.0 * np.pi / 180, phi=20.0 * np.pi / 180,
        wavelength=1000.0, period=[1000.0, 1000.0], thickness=[300.0],
        ucell=ucell_raster
    )
    
    res = m.solve_for_conv(1000.0, np.array([Exx]), np.array([Eyy]), np.array([Epz_i]), np.array([Exy]), np.array([Eyx]))
    
    R = float(res.de_ri.sum())
    T = float(res.de_ti.sum())
    
    # Energy conservation: R + T = 1.0 (within 5% tolerance for low-order truncated NVF)
    assert abs((R + T) - 1.0) < 5e-2, f"Energy not conserved: R+T = {R+T}"
