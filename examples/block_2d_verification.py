import torch
import numpy as np

import fieldwave


def run_block(rcwa_options, backend):

    rcwa_options['backend'] = backend
    solver = fieldwave.rcwa(**rcwa_options)
    solver.ucell = ucell_block

    res = solver.conv_solve()

    return res.de_ri, res.de_ti


def run_raster(rcwa_options, backend, fourier_coefficients):

    # ucell_raster = ucell_raster.numpy()

    rcwa_options['backend'] = backend
    rcwa_options['fourier_coefficients'] = fourier_coefficients
    # 'sampled': discrete Fourier series; 'integrated' is the closed-form series that
    # block modeling uses.

    # One array for every backend. Handing torch the Python list instead would build it at
    # torch's *default* dtype -- float32 -- and no later cast recovers the digits the
    # constructor already dropped. That showed up here as torch disagreeing with the block
    # path by 5.6e-5 while numpy and jax agreed to 1e-13, which reads as a solver defect and
    # is not one.
    ucell_raster_1 = np.asarray(ucell_raster)

    rcwa_options['ucell'] = ucell_raster_1

    solver = fieldwave.rcwa(**rcwa_options)

    res = solver.conv_solve()
    de_ri, de_ti = res.de_ri, res.de_ti

    return de_ri, de_ti


if __name__ == '__main__':
    rcwa_options = dict(backend=0, thickness=[205, 100000], period=[300, 300],
                        fto=[3, 3],
                        n_top=1, n_bot=1,
                        wavelength=900,
                        )

    # Absorbing Si would be 3.638751670074983+0.007498295841854125j — Im(n) > 0 is loss
    # here, the same convention the bundled nk tables use (n + 1j*k).
    si = 3.638751670074983
    sio2 = 1.4518
    si3n4 = 2.0056

    ucell_block = [
        # layer 1
        [si3n4,
            [
                # obj 1
                ['rectangle', 50, 250, 100, 100, si, 0, 0, 0],
                # obj 2
                ['rectangle', 240, 40, 40, 60, si, 0, 0, 0],
            ],
        ],
        # layer 2
        [si,
         []
        ],
    ]

    a = si3n4
    b = si
    c = si

    ucell_raster = [
        [
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
            [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a] + [a,a,b,b,b,b,a,a,a,a],
            [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a] + [a,a,b,b,b,b,a,a,a,a],
            [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a] + [a,a,b,b,b,b,a,a,a,a],
            [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a] + [a,a,b,b,b,b,a,a,a,a],
            [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a] + [a,a,b,b,b,b,a,a,a,a],
            [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a] + [a,a,b,b,b,b,a,a,a,a],
            [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a] + [a,a,a,a,a,a,a,a,a,a],

        ],
        [
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
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

    print('\nRaster with DFS; R and T from NumPy\n')
    print('Reflectance from NumPy: \n', de_ri_r_1_dfs[c_x-1:c_x+2, c_y-1:c_y+2])
    print('Transmittance from NumPy: \n', de_ti_r_1_dfs[c_x-1:c_x+2, c_y-1:c_y+2])

    print('\nRaster with CFS; R and T from NumPy\n')
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
