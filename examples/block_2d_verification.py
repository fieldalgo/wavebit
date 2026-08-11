import numpy as np

import fieldwave


def run_block(rcwa_options, backend):

    rcwa_options['backend'] = backend
    solver = fieldwave.rcwa(**rcwa_options)
    solver.ucell = ucell_block

    res = solver.conv_solve()

    return res.R, res.T


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
    R, T = res.R, res.T

    return R, T


if __name__ == '__main__':
    # Absorbing Si would be 3.638751670074983+0.007498295841854125j — Im(n) > 0 is loss
    # here, the same convention the bundled nk tables use (n + 1j*k). n_bot takes a complex
    # index just as a layer does.
    si = 3.638751670074983
    sio2 = 1.4518
    si3n4 = 2.0056

    # The silicon underneath is the substrate, so it goes in n_bot.
    rcwa_options = dict(backend=0, thickness=[205], period=[300, 300],
                        fto=[3, 3],
                        n_top=1, n_bot=si,
                        wavelength=900,
                        )

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
    ]

    a = si3n4
    b = si
    c = si

    # Row 0 is y = 0 and the column index is x, on every path in the package -- the same
    # convention the block modeler grids with. Typed the other way up, this raster is the
    # vertical mirror of the ucell above and not the same structure: obj 1 (cy 250) lands at
    # y 0-100 and obj 2 (cy 40) at y 230-290. With vacuum on both sides only the specular
    # order propagates, so the mirror image radiates exactly like the original and the two
    # paths still agree to 1e-15. Let a higher order propagate -- an index other than 1 on
    # either half-space is enough -- and the disagreement appears. The n_bot above is silicon,
    # so this file sits in that regime and would catch the mirror today.
    ucell_raster = [
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
