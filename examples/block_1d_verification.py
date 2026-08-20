import numpy as np

import wavebit


def run_block(rcwa_options, backend):

    rcwa_options['backend'] = backend
    solver = wavebit.rcwa(**rcwa_options)
    solver.structure = structure_block

    res = solver.conv_solve()

    return res.R, res.T


def run_raster(rcwa_options, backend, fourier_coefficients):

    rcwa_options['backend'] = backend
    rcwa_options['fourier_coefficients'] = fourier_coefficients
    # 'sampled': discrete Fourier series; 'integrated' is the closed-form series that
    # block modeling uses.

    # One array for every backend. Handing torch the Python list instead would build it at
    # torch's *default* dtype -- float32 -- and no later cast recovers the digits the
    # constructor already dropped. That showed up here as torch disagreeing with the block
    # path by 5.6e-5 while numpy and jax agreed to 1e-13, which reads as a solver defect and
    # is not one.
    structure_raster_1 = np.asarray(structure_raster)

    rcwa_options['structure'] = structure_raster_1

    solver = wavebit.rcwa(**rcwa_options)

    res = solver.conv_solve()
    R, T = res.R, res.T

    return R, T


if __name__ == '__main__':
    # Permittivity, not refractive index -- that is the quantity the API takes, and
    # `eps_from_index` is the conversion for the nk datasheet these came off. Written as bare
    # numbers rather than as `Material(...)` on purpose: the same value has to serve both
    # paths compared here, and only one of them takes materials. The block path below reads a
    # bare number through the scalar shorthand, `Material(epsilon=x)`; the raster is a
    # *sampled field* rather than an assignment, so its pixels are permittivity samples and a
    # Material could not go in an ndarray at all. That asymmetry is the whole reason the
    # raster keeps arrays.
    #
    # Absorbing Si would be eps_from_index(3.638751670074983+0.007498295841854125j).
    # Im(eps) > 0 is loss here, the same convention the bundled nk tables use (n + 1j*k).
    si = wavebit.eps_from_index(3.638751670074983)
    sio2 = wavebit.eps_from_index(1.4518)
    si3n4 = wavebit.eps_from_index(2.0056)

    # The silicon underneath is the substrate, so it goes in `bot`.
    rcwa_options = dict(backend=0, thickness=[205], period=[300, 300],
                        fto=[3, 0],
                        top=1, bot=si,
                        wavelength=900,
                        )

    structure_block = [
        # layer 1
        [si3n4,
            [
                # obj 1
                ['rectangle', 50, 150, 100, 300, si, 0, 0, 0],
                # obj 2
                ['rectangle', 200, 150, 40, 300, si, 0, 0, 0],
            ],
        ],
    ]

    a = si3n4
    b = si
    c = si

    structure_raster = [
        [
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
        ],
    ]

    de_ri_v_0, de_ti_v_0 = run_block(rcwa_options, 0)  # NumPy
    de_ri_v_1, de_ti_v_1 = run_block(rcwa_options, 1)  # JAX
    de_ri_v_2, de_ti_v_2 = run_block(rcwa_options, 2)  # PyTorch

    de_ri_v_1, de_ti_v_1 = np.array(de_ri_v_1), np.array(de_ti_v_1)
    de_ri_v_2, de_ti_v_2 = np.array(de_ri_v_2), np.array(de_ti_v_2)

    c_x, c_y = de_ri_v_0.shape[0] // 2, de_ri_v_0.shape[1] // 2

    print('\nVector; R and T from NumPy\n')
    print('Reflectance from NumPy: \n', de_ri_v_0[c_x-1:c_x+2, c_y-1:c_y+2])
    print('Transmittance from NumPy: \n', de_ti_v_0[c_x-1:c_x+2, c_y-1:c_y+2])

    print('\nblock across backends\n')
    print(f'Norm of difference NumPy and JAX; R: {np.linalg.norm(de_ri_v_0-de_ri_v_1)}, T: {np.linalg.norm(de_ti_v_0-de_ti_v_1)}')
    print(f'Norm of difference JAX and Torch; R: {np.linalg.norm(de_ri_v_1-de_ri_v_2)}, T: {np.linalg.norm(de_ti_v_1-de_ti_v_2)}')
    print(f'Norm of difference Torch and NumPy; R: {np.linalg.norm(de_ri_v_1-de_ri_v_2)}, T: {np.linalg.norm(de_ti_v_1-de_ti_v_2)}')

    de_ri_r_0_dfs, de_ti_r_0_dfs = run_raster(rcwa_options, 0, 'sampled')  # NumPy
    de_ri_r_0_cfs, de_ti_r_0_cfs = run_raster(rcwa_options, 0, 'integrated')  # NumPy
    de_ri_r_1_dfs, de_ti_r_1_dfs = run_raster(rcwa_options, 1, 'sampled')  # JAX
    de_ri_r_1_cfs, de_ti_r_1_cfs = run_raster(rcwa_options, 1, 'integrated')  # JAX
    de_ri_r_2_dfs, de_ti_r_2_dfs = run_raster(rcwa_options, 2, 'sampled')  # PyTorch
    de_ri_r_2_cfs, de_ti_r_2_cfs = run_raster(rcwa_options, 2, 'integrated')  # PyTorch

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

    print('\nblock vs raster dfs \n')
    print(f'Norm of difference, block and raster dfs, NumPy;'
          f' R: {np.linalg.norm(de_ri_v_0-de_ri_r_0_dfs)}, T: {np.linalg.norm(de_ti_v_0-de_ti_r_0_dfs)}')
    print(f'Norm of difference, block and raster dfs, JAX;'
          f' R: {np.linalg.norm(de_ri_v_1-de_ri_r_1_dfs)}, T: {np.linalg.norm(de_ti_v_1-de_ti_r_1_dfs)}')
    print(f'Norm of difference, block and raster dfs, Torch;'
          f' R: {np.linalg.norm(de_ri_v_2-de_ri_r_2_dfs)}, T: {np.linalg.norm(de_ti_v_2-de_ti_r_2_dfs)}')

    print('\nblock vs raster cfs \n')
    print(f'Norm of difference, block and raster cfs, NumPy;'
          f' R: {np.linalg.norm(de_ri_v_0-de_ri_r_0_cfs)}, T: {np.linalg.norm(de_ti_v_0-de_ti_r_0_cfs)}')
    print(f'Norm of difference, block and raster cfs, JAX;'
          f' R: {np.linalg.norm(de_ri_v_1-de_ri_r_1_cfs)}, T: {np.linalg.norm(de_ti_v_1-de_ti_r_1_cfs)}')
    print(f'Norm of difference, block and raster cfs, Torch;'
          f' R: {np.linalg.norm(de_ri_v_2-de_ri_r_2_cfs)}, T: {np.linalg.norm(de_ti_v_2-de_ti_r_2_cfs)}')

    print(0)
