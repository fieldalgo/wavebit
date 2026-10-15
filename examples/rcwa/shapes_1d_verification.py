"""Compare typed physical shapes and independently sampled constitutive rasters."""
import numpy as np
import wavebit
from wavebit import rcwa, Material, Units


def _solve(options, backend, layers, coefficients=None):
    if backend == 'jax':
        import jax
        jax.config.update('jax_enable_x64', True)
    period=options['period']
    orders=options['truncation_order']
    problem=rcwa.Problem(units=Units(length='nm'),
        stack=rcwa.Stack(period=(period['x'],period['y']),layers=tuple(layers),
            top=Material(options['top']),bottom=Material(options['bot'])),
        excitation=rcwa.Excitation(wavelength=options['wavelength'],sources=(rcwa.PlaneWave(),)))
    result=rcwa.run(problem,backend=backend,numerics=rcwa.Numerics(
        orders=(orders['x'],orders['y']),algorithm='etm',fourier_coefficients=coefficients))
    return result.R[0],result.T[0]


def run_shapes(rcwa_options, backend):
    layers=[]
    for thickness,layer in zip(rcwa_options['thickness'],structure_shapes):
        regions=tuple(rcwa.Region(material=Material(shape['material']),
            geometry=rcwa.Rectangle(size=(shape['lx'],shape['ly']),
                center=(shape['cx'],shape['cy']),angle=shape['angle'])) for shape in layer['shapes'])
        layers.append(rcwa.Layer(thickness=thickness,background=Material(layer['background']),regions=regions))
    return _solve(rcwa_options,backend,layers)


def run_raster(rcwa_options, backend, fourier_coefficients):
    # A single float64 source raster gives every backend the same input precision.
    layers=[rcwa.Layer(thickness=d,epsilon=epsilon) for d,epsilon in
            zip(rcwa_options['thickness'],np.asarray(structure_raster))]
    return _solve(rcwa_options,backend,layers,fourier_coefficients)


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
                        truncation_order=dict(x=3, y=0),
                        top=1, bot=si,
                        wavelength=900,
                        )

    # The rods are commensurate with the raster below on purpose: 100 wide from x = 0 and 40
    # wide centred at x = 200 -- `cx` is measured from the middle of the cell, so those are
    # -100 and +50 -- on a 300 period sampled at 30 pixels, so each rod's edges land
    # exactly on pixel boundaries. The two structures are then the *same* permittivity function
    # and any difference in the answer is the two paths' arithmetic, not their geometry.
    structure_shapes = [
        # layer 1
        dict(background=si3n4, shapes=[
            # obj 1
            dict(type='rectangle', cx=-100, cy=0, lx=100, ly=300, angle=0, material=si),
            # obj 2
            dict(type='rectangle', cx=50, cy=0, lx=40, ly=300, angle=0, material=si),
        ]),
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

    # There is no difference left to explain, and that is the point of the sweep below: the two
    # paths agree to 4e-16 at truncation_order 3 and hold at 2e-15 and 7e-14 at 10 and 20.
    #
    # They did not, and the paragraph that used to stand here got it wrong in an instructive
    # way. It read the R 9.1e-4 / T 4.7e-3 gap as a *factorization* difference -- Li's inverse
    # rule along the grid against NVF off an analytic normal field, "converging to the same
    # answer from different sides, so the gap is a truncation error rather than a disagreement".
    # The numbers were there to check that against and were not read as a check: 4.7e-3 ->
    # 3.7e-4 -> 1.7e-4 at truncation_order 3 / 10 / 20 is a plateau, not a convergence.
    #
    # It was the normal field, and specifically these rods' `ly = 300` on a 300 period. Each rod
    # spans the cell, so its top and bottom faces are the same line as its own image's and bound
    # nothing -- the material runs straight through -- and the analytic field reported a face
    # normal across them anyway. Dropping a covered face pair makes this scene one calculation
    # rather than two that nearly agree, because with the field constant the NVF projector *is*
    # Li's 1-D rule. See `docs/validation/rcwa/normal-field-known-limits.md` section 2a.
    #
    # This is the check that replaced a bit-identity. The block path -- removed on 2026-08-26 --
    # gridded its shapes and applied Li's rule to them, which on a commensurate grating like this
    # one *is* the raster's arithmetic, so it agreed to the last bit and said nothing about
    # whether either was right.
    print('\nshapes vs raster, against truncation_order (integrated raster, NumPy)\n')
    for order in (3, 10, 20, 40):
        opts = dict(rcwa_options, backend='numpy', truncation_order=dict(x=order, y=0))
        r_s, t_s = run_shapes(dict(opts), 'numpy')
        r_r, t_r = run_raster(dict(opts), 'numpy', 'integrated')
        print(f'  truncation_order = {order:3d}   '
              f'R: {np.linalg.norm(r_s - r_r):.3e}   T: {np.linalg.norm(t_s - t_r):.3e}   '
              f'sum(T) = {float(np.sum(t_s)):.9f}')

    print(0)
