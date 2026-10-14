
from wavebit.compat import legacy as _legacy_wavebit
import numpy as np

import wavebit


def run_shapes(rcwa_options, backend):

    rcwa_options['backend'] = backend
    solver = _legacy_wavebit.rcwa(**rcwa_options)
    solver.structure = structure_shapes

    res = solver.solve()

    return res.R, res.T


def run_raster(rcwa_options, backend, fourier_coefficients):

    rcwa_options['backend'] = backend
    rcwa_options['fourier_coefficients'] = fourier_coefficients
    # 'sampled': discrete Fourier series; 'integrated': the closed-form series over the
    # pixels. This structure is commensurate with its grid, so the two agree here.

    # One array for every backend. Handing torch the Python list instead would build it at
    # torch's *default* dtype -- float32 -- and no later cast recovers the digits the
    # constructor already dropped. That showed up here as torch disagreeing with the shape
    # path by 5.6e-5 while numpy and jax agreed to 1e-13, which reads as a solver defect and
    # is not one.
    structure_raster_1 = np.asarray(structure_raster)

    rcwa_options['structure'] = structure_raster_1

    solver = _legacy_wavebit.rcwa(**rcwa_options)

    res = solver.solve()
    R, T = res.R, res.T

    return R, T


if __name__ == '__main__':
    # Permittivity, not refractive index -- that is the quantity the API takes, and
    # `eps_from_refractive_index` is the conversion for the nk datasheet these came off. Written as
    # bare numbers rather than as `Material(...)` on purpose: the same value has to serve both
    # paths compared here, and only one of them takes materials. The shape path below reads a
    # bare number through the scalar shorthand, `Material(epsilon=x)`; the raster is a
    # *sampled field* rather than an assignment, so its pixels are permittivity samples and a
    # Material could not go in an ndarray at all. That asymmetry is the whole reason the
    # raster keeps arrays.
    #
    # Absorbing Si would be eps_from_refractive_index(3.638751670074983+0.007498295841854125j).
    # Im(eps) > 0 is loss here, the same convention the bundled nk tables use (n + 1j*k).
    si = wavebit.eps_from_refractive_index(3.638751670074983)
    sio2 = wavebit.eps_from_refractive_index(1.4518)
    si3n4 = wavebit.eps_from_refractive_index(2.0056)

    # The silicon underneath is the substrate, so it goes in `bot`.
    rcwa_options = dict(backend='numpy', thickness=[205], period=dict(x=300, y=300),
                        truncation_order=dict(x=3, y=3),
                        top=1, bot=si,
                        wavelength=900,
                        )

    structure_shapes = [
        # layer 1
        dict(background=si3n4, shapes=[
            # obj 1
            dict(type='rectangle', cx=-100, cy=100, lx=100, ly=100, angle=0, material=si),
            # obj 2
            dict(type='rectangle', cx=90, cy=-110, lx=40, ly=60, angle=0, material=si),
        ]),
    ]

    a = si3n4
    b = si
    c = si

    # Row 0 is y = 0 and the column index is x, on every path in the package. Typed the other
    # way up, this raster is the vertical mirror of the shape structure above and not the same
    # structure: obj 1 (cy +100, so y 200-300) lands at y 0-100 and obj 2 (cy -110, so
    # y 10-70) at y 230-290. With vacuum on both sides only the
    # specular order propagates, so the mirror image radiates exactly like the original and
    # the two paths still agree to 1e-15. Let a higher order propagate -- a permittivity other
    # than 1 on either half-space is enough -- and the disagreement appears. The `bot` above
    # is silicon, so this file sits in that regime and would catch the mirror today.
    structure_raster = [
        [
            # y 0-10: background
            [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a],

            # obj 2: x 220-260, y 10-70
            [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a] + [a,a,b,b,b,b,a,a,a,a],
            [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a] + [a,a,b,b,b,b,a,a,a,a],
            [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a] + [a,a,b,b,b,b,a,a,a,a],
            [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a] + [a,a,b,b,b,b,a,a,a,a],
            [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a] + [a,a,b,b,b,b,a,a,a,a],
            [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a] + [a,a,b,b,b,b,a,a,a,a],

            # y 70-200: background
            [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a],
            [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a],
            [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a],

            [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a],
            [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a],
            [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a],
            [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a],
            [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a],
            [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a],
            [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a],
            [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a],
            [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a],
            [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a],

            # obj 1: x 0-100, y 200-300
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a],
        ],
    ]

    de_ri_v_0, de_ti_v_0 = run_shapes(rcwa_options, 'numpy')  # NumPy
    de_ri_v_1, de_ti_v_1 = run_shapes(rcwa_options, 'jax')  # JAX
    de_ri_v_2, de_ti_v_2 = run_shapes(rcwa_options, 'torch')  # PyTorch

    de_ri_v_1, de_ti_v_1 = np.array(de_ri_v_1), np.array(de_ti_v_1)
    de_ri_v_2, de_ti_v_2 = np.array(de_ri_v_2), np.array(de_ti_v_2)

    c_x, c_y = de_ri_v_0.shape[0] // 2, de_ri_v_0.shape[1] // 2

    print('\nShapes; R and T from NumPy\n')
    print('Reflectance from NumPy: \n', de_ri_v_0[c_x-1:c_x+2, c_y-1:c_y+2])
    print('Transmittance from NumPy: \n', de_ti_v_0[c_x-1:c_x+2, c_y-1:c_y+2])

    print('\nshapes across backends\n')
    print(f'Norm of difference NumPy and JAX; R: {np.linalg.norm(de_ri_v_0-de_ri_v_1)}, T: {np.linalg.norm(de_ti_v_0-de_ti_v_1)}')
    print(f'Norm of difference JAX and Torch; R: {np.linalg.norm(de_ri_v_1-de_ri_v_2)}, T: {np.linalg.norm(de_ti_v_1-de_ti_v_2)}')
    print(f'Norm of difference Torch and NumPy; R: {np.linalg.norm(de_ri_v_1-de_ri_v_2)}, T: {np.linalg.norm(de_ti_v_1-de_ti_v_2)}')

    de_ri_r_0_dfs, de_ti_r_0_dfs = run_raster(rcwa_options, 'numpy', 'sampled')  # NumPy
    de_ri_r_0_cfs, de_ti_r_0_cfs = run_raster(rcwa_options, 'numpy', 'integrated')  # NumPy
    de_ri_r_1_dfs, de_ti_r_1_dfs = run_raster(rcwa_options, 'jax', 'sampled')  # JAX
    de_ri_r_1_cfs, de_ti_r_1_cfs = run_raster(rcwa_options, 'jax', 'integrated')  # JAX
    de_ri_r_2_dfs, de_ti_r_2_dfs = run_raster(rcwa_options, 'torch', 'sampled')  # PyTorch
    de_ri_r_2_cfs, de_ti_r_2_cfs = run_raster(rcwa_options, 'torch', 'integrated')  # PyTorch

    de_ri_r_1_dfs, de_ti_r_1_dfs = np.array(de_ri_r_1_dfs), np.array(de_ti_r_1_dfs)
    de_ri_r_1_cfs, de_ti_r_1_cfs = np.array(de_ri_r_1_cfs), np.array(de_ti_r_1_cfs)
    de_ri_r_2_dfs, de_ti_r_2_dfs = np.array(de_ri_r_2_dfs), np.array(de_ti_r_2_dfs)
    de_ri_r_2_cfs, de_ti_r_2_cfs = np.array(de_ri_r_2_cfs), np.array(de_ti_r_2_cfs)

    print('\nRaster, sampled coefficients; R and T from NumPy\n')
    print('Reflectance from NumPy: \n', de_ri_r_1_dfs[c_x-1:c_x+2, c_y-1:c_y+2])
    print('Transmittance from NumPy: \n', de_ti_r_1_dfs[c_x-1:c_x+2, c_y-1:c_y+2])

    print('\nRaster, integrated coefficients; R and T from NumPy\n')
    print('Reflectance from NumPy: \n', de_ri_r_1_dfs[c_x-1:c_x+2, c_y-1:c_y+2])
    print('Transmittance from NumPy: \n', de_ti_r_1_dfs[c_x-1:c_x+2, c_y-1:c_y+2])

    print('\nshapes vs raster dfs \n')
    print(f'Norm of difference, shapes and raster dfs, NumPy;'
          f' R: {np.linalg.norm(de_ri_v_0-de_ri_r_0_dfs)}, T: {np.linalg.norm(de_ti_v_0-de_ti_r_0_dfs)}')
    print(f'Norm of difference, shapes and raster dfs, JAX;'
          f' R: {np.linalg.norm(de_ri_v_1-de_ri_r_1_dfs)}, T: {np.linalg.norm(de_ti_v_1-de_ti_r_1_dfs)}')
    print(f'Norm of difference, shapes and raster dfs, Torch;'
          f' R: {np.linalg.norm(de_ri_v_2-de_ri_r_2_dfs)}, T: {np.linalg.norm(de_ti_v_2-de_ti_r_2_dfs)}')

    print('\nshapes vs raster cfs \n')
    print(f'Norm of difference, shapes and raster cfs, NumPy;'
          f' R: {np.linalg.norm(de_ri_v_0-de_ri_r_0_cfs)}, T: {np.linalg.norm(de_ti_v_0-de_ti_r_0_cfs)}')
    print(f'Norm of difference, shapes and raster cfs, JAX;'
          f' R: {np.linalg.norm(de_ri_v_1-de_ri_r_1_cfs)}, T: {np.linalg.norm(de_ti_v_1-de_ti_r_1_cfs)}')
    print(f'Norm of difference, shapes and raster cfs, Torch;'
          f' R: {np.linalg.norm(de_ri_v_2-de_ri_r_2_cfs)}, T: {np.linalg.norm(de_ti_v_2-de_ti_r_2_cfs)}')

    # What that difference is. The rods land on pixel boundaries, so the two structures are the
    # same permittivity function -- the `1/eps` coefficient grids agree to 1e-16 -- and what
    # differs is the factorization: Li's inverse rule along the grid on one side, NVF off an
    # analytic normal field on the other. Both converge to the same answer, so the number to
    # look at is the trend, not any one row.
    #
    # Where this stops being a fair fight is the point of the shape path: give a shape an angle
    # and the raster has to staircase it while the closed form does not, so there is no
    # commensurate raster to compare against at all. This file keeps the shapes axis-aligned so
    # that the comparison stays honest.
    print('\nshapes vs raster, against truncation_order (integrated raster, NumPy)\n')
    for order in (3, 6, 10, 14):
        opts = dict(rcwa_options, backend='numpy', truncation_order=dict(x=order, y=order))
        r_s, t_s = run_shapes(dict(opts), 'numpy')
        r_r, t_r = run_raster(dict(opts), 'numpy', 'integrated')
        print(f'  truncation_order = {order:3d}   '
              f'R: {np.linalg.norm(r_s - r_r):.3e}   T: {np.linalg.norm(t_s - t_r):.3e}   '
              f'sum(T) = {float(np.sum(t_s)):.9f}')

    print(0)
