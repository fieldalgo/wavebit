import numpy as np
import wavebit
from wavebit._core.lattice import as_lattice
import array_api_compat.numpy as xnp
from wavebit._core.parametric import normal_vector_field_analytical as _nvf_analytic
from wavebit._core.factorization import nvf_conv_matrices as _nvf_conv


def get_normal_vector_field_analytical(grid_shape, shapes, period=(1000.0, 1000.0)):
    return _nvf_analytic(xnp, grid_shape, shapes, xnp.float64, as_lattice(period=period))


def to_conv_mat_nvf_2d(eps_matrix, half_width_u, half_width_v, normal_x=None, normal_y=None):
    return _nvf_conv(xnp, eps_matrix, normal_x, normal_y, half_width_u, half_width_v)

def test_analytical_normal_vector_generation():
    # The shapes were reached through a `Modeling` instance, which only ever held the list --
    # the block modeller it belonged to is gone and the normal field never needed it.
    period = (1000.0, 1000.0)
    shapes = [dict(type='circle', cx=0.0, cy=0.0, lx=600.0)]

    normal_x, normal_y = get_normal_vector_field_analytical((100, 100), shapes, period=period)
    
    assert normal_x.shape == (100, 100)
    assert normal_y.shape == (100, 100)
    
    # Check unit vector normalization
    norm = np.hypot(normal_x, normal_y)
    np.testing.assert_allclose(norm, 1.0, atol=1e-5)

def test_vector_nvf_energy_conservation():
    N = 60
    grid_y, grid_x = np.ogrid[:N, :N]
    radius = 18
    dist = np.sqrt((grid_x - N//2)**2 + (grid_y - N//2)**2)
    eps_matrix = np.ones((N, N), dtype=complex)
    eps_matrix[dist <= radius] = 2.0**2
    # `structure` is permittivity now, so the raster IS `eps_matrix` -- no root on the way
    # in and no square on the way back out.
    structure_raster = np.expand_dims(eps_matrix, axis=0)

    shapes = [dict(type='circle', cx=0.0, cy=0.0, lx=600.0)]
    nx_vec, ny_vec = get_normal_vector_field_analytical((N, N), shapes, period=(1000.0, 1000.0))

    max_order_val = 2
    Exx, Eyy, Epz_i, Exy, Eyx = to_conv_mat_nvf_2d(eps_matrix, max_order_val, max_order_val, normal_x=nx_vec, normal_y=ny_vec)

    m = wavebit.rcwa(
        backend='numpy', truncation_order=[max_order_val, max_order_val],
        psi=0.0, top=1.0, bot=2.1025,
        theta=10.0 * np.pi / 180, phi=20.0 * np.pi / 180,
        wavelength=1000.0, period=[1000.0, 1000.0], thickness=[300.0],
        structure=structure_raster
    )
    
    res = m._solve_conv(np.array([Exx]), np.array([Eyy]), np.array([Epz_i]), np.array([Exy]), np.array([Eyx]))
    
    R = float(res.R.sum())
    T = float(res.T.sum())
    
    # Energy conservation: R + T = 1.0 (within 5% tolerance for low-order truncated NVF)
    assert abs((R + T) - 1.0) < 5e-2, f"Energy not conserved: R+T = {R+T}"
