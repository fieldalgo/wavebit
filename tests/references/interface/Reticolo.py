import os
import numpy as np

import wavebit


# os.environ['OCTAVE_EXECUTABLE'] = '/opt/homebrew/bin/octave-cli'


def _index(eps):
    """A RETICOLO texture value, from wavebit's permittivity. The only conversion in this file.

    Wavebit states a scene in relative **permittivity**; RETICOLO states a texture by
    **refractive index**. This bridge is therefore one of the very few places in the tree that
    runs the 2026-08-20 flip backwards, and it is deliberately the *only* one on this side: the
    generators hand the same numbers to the solver and to Octave, so a conversion sitting in a
    generator instead would have to be repeated in each of the three that drive this class, and
    a copy that must stay identical is a copy that will not.

    :func:`wavebit.index_from_eps` is the principal root, which is the passive branch under this
    project's ``e^{-iwt}``: ``Im(eps) >= 0`` (loss) maps into the first quadrant, so
    ``Im(n) >= 0`` (loss again). That is the same spelling RETICOLO uses, and it is checked
    rather than assumed -- on the ``lossy`` field case ``res3``'s index map comes back as
    ``2+0.05j``, exactly what this returns for ``eps = (2+0.05j)**2``.

    Everything crossing into Octave goes through here -- both half-spaces and every layer -- so
    there is no slot left where a permittivity could reach RETICOLO wearing an index's name. A
    missed one would be silent: ``eps = 9`` read as ``n = 9`` is a legal, energy-conserving
    medium, and the oracle would then be validating the solver against a different structure.
    """
    return wavebit.index_from_eps(eps)


class Reticolo:

    def __init__(self, engine_type='octave', *args, **kwargs):

        if engine_type == 'octave':
            try:
                from oct2py import octave
            except Exception as e:
                raise e
            self.eng = octave

        elif engine_type == 'matlab':
            try:
                import matlab.engine
            except Exception as e:
                raise e
            self.eng = matlab.engine.start_matlab()
        else:
            raise ValueError

        # path that has file to run in octave
        m_path = os.path.dirname(__file__)
        self.eng.addpath(self.eng.genpath(m_path))

    def run_res2(self, grating_type, period, truncation_order, structure, thickness, theta, phi, pol, wavelength, top, bot,
                 *args, **kwargs):
        """``structure`` / ``top`` / ``bot`` are **permittivity**, as everywhere in wavebit.

        They were ``ucell`` / ``n_top`` / ``n_bot`` and held a refractive index. The rename is
        the smaller half: :func:`_index` is what makes the numbers mean to RETICOLO what they
        mean to the solver, and a caller that merely renamed its keywords would be comparing
        two different structures with nothing to warn it.
        """

        # theta and phi both stay in radians: res1 takes the azimuth that way, same as it
        # takes k_parallel = n*sin(theta). Converting phi to degrees here solved a different
        # problem entirely -- the 2D comparison was off by 2.7e-2 and did not converge.
        n_cell, n_top, n_bot = _index(structure), _index(top), _index(bot)

        if grating_type in (0, 1):
            period = period[0]

            truncation_order = truncation_order
            Nx = n_cell.shape[2]
            extent_1 = period
            # grid_x = np.linspace(0, period, Nx + 1)[1:]
            # grid_x -= extent_1 / 2
            grid_x = np.linspace(0, period, Nx + 1)[1:]
            grid_x = np.arange(1, Nx+1) * (period/Nx)

            # grid = np.linspace(0, period, Nx)

            layer_textures = []
            for z in range(n_cell.shape[0]):
                layer_textures.append([grid_x, n_cell[z, 0]])

            textures = [n_top, *layer_textures, n_bot]

        else:
            # 2D: every raster pixel becomes one Reticolo rectangle, so the texture is the
            # same staircase the solver sees rather than a re-idealized shape. Rectangles are
            # given by centre and size; the background never shows because they tile the cell.
            Ny, Nx = n_cell.shape[1], n_cell.shape[2]
            unit_x, unit_y = period[0] / Nx, period[1] / Ny

            cx = ((np.arange(Nx) + 0.5) * unit_x)[None, :].repeat(Ny, 0).ravel()
            cy = ((np.arange(Ny) + 0.5) * unit_y)[:, None].repeat(Nx, 1).ravel()

            layer_textures = []
            for z in range(n_cell.shape[0]):
                rects = np.stack([cx, cy, np.full(Nx * Ny, unit_x), np.full(Nx * Ny, unit_y),
                                  n_cell[z].ravel(), np.ones(Nx * Ny)], axis=1)
                background = np.array([[1.0, 0, 0, 0, 0, 0]])
                layer_textures.append(np.concat([background, rects]).astype(complex))

            textures = [n_top, *layer_textures, n_bot]

        profile = np.array([[0, *thickness, 0], range(1, len(thickness) + 3)])

        top_refl_info, top_tran_info, bottom_refl_info, bottom_tran_info = \
            self._run(pol, theta, phi, period, n_top, truncation_order, textures, profile, wavelength, grating_type,
                      cal_field=False)

        return top_refl_info.efficiency, top_tran_info.efficiency, bottom_refl_info.efficiency, bottom_tran_info.efficiency

    def run_res3(self, grating_type, period, truncation_order, structure, thickness, theta, phi, pol, wavelength, top, bot,
                 matlab_plot_field=0, res3_npts=0, *args, **kwargs):

        # theta *= (180 / np.pi)
        # phi *= (180 / np.pi)
        n_cell, n_top, n_bot = _index(structure), _index(top), _index(bot)

        if grating_type in (0, 1):
            period = period[0]

            truncation_order = truncation_order
            Nx = n_cell.shape[2]
            # A texture's x origin used to be in doubt here -- RETICOLO's documentation
            # describes a texture as running -period/2 .. +period/2, and a `grid_x -= period/2`
            # sat above, overwritten twice and so never in effect. It is settled by
            # measurement rather than by reading: res3's third output is the refractive index
            # at every sample point, and on a binary grating it reproduces this project's own
            # structure -- converted by `_index` -- evaluated on `linspace(0, period, sample_x)`
            # entry for entry. The two x origins coincide as written; no shift to apply.
            grid_x = np.arange(1, Nx+1) * (period/Nx)

            layer_textures = []
            for z in range(n_cell.shape[0]):
                layer_textures.append([grid_x, n_cell[z, 0]])

            textures = [n_top, *layer_textures, n_bot]

        else:
            raise ValueError

            # Nx = ucell.shape[2]
            # Ny = ucell.shape[1]
            # extent_1 = period[0]
            # extent_2 = period[1]
            #
            # unit_x = extent_1 / Nx
            # unit_y = extent_2 / Ny
            #
            # grid_x = np.linspace(0, period[0], Nx + 1)[1:]
            # grid_y = np.linspace(0, period[1], Ny + 1)[1:]
            #
            # grid_x -= extent_1 / 2
            # grid_y -= extent_2 / 2
            #
            # ucell_new = []
            # for z in range(ucell.shape[0]):
            #     ucell_layer = [10]
            #     for y, yval in enumerate(grid_y):
            #         for x, xval in enumerate(grid_x):
            #             obj = [xval, yval, unit_x, unit_y, ucell[z, y, x], 1]
            #             ucell_layer.append(obj)
            #     ucell_new.append(ucell_layer)
            # textures = [n_top, *ucell_new, n_bot]

        profile = np.array([[0, *thickness, 0], range(1, len(thickness) + 3)])

        (top_refl_info_te, top_tran_info_te, top_refl_info_tm, top_tran_info_tm, bottom_refl_info_te,
         bottom_tran_info_te, bottom_refl_info_tm, bottom_tran_info_tm, field_cell, z, index_map) = \
            self._run(pol, theta, phi, period, n_top, truncation_order, textures, profile, wavelength, grating_type,
                      cal_field=True, matlab_plot_field=matlab_plot_field, res3_npts=res3_npts)

        return (top_refl_info_te, top_tran_info_te, top_refl_info_tm, top_tran_info_tm,
                bottom_refl_info_te, bottom_tran_info_te, bottom_refl_info_tm, bottom_tran_info_tm,
                field_cell, z, index_map)

    def _run(self, pol, theta, phi, period, n_top, truncation_order,
             textures, profile, wavelength, grating_type, cal_field=False, matlab_plot_field=0, res3_npts=0):

        if phi is None:
            phi = 0
        else:
            phi = phi * (180 / np.pi)

        if cal_field:
            # top_refl_info, top_tran_info, bottom_refl_info, bottom_tran_info, field_cell = \
            (top_refl_info_te, top_tran_info_te, top_refl_info_tm, top_tran_info_tm, bottom_refl_info_te,
             bottom_tran_info_te, bottom_refl_info_tm, bottom_tran_info_tm, field_cell, z, index_map) = \
                self.eng.reticolo_res3(pol, theta, phi, period, n_top, truncation_order,
                                       textures, profile, wavelength, grating_type, matlab_plot_field, res3_npts,
                                       nout=11)

            res = (top_refl_info_te, top_tran_info_te, top_refl_info_tm, top_tran_info_tm, bottom_refl_info_te,
                   bottom_tran_info_te, bottom_refl_info_tm, bottom_tran_info_tm, field_cell, z, index_map)
        else:
            top_refl_info, top_tran_info, bottom_refl_info, bottom_tran_info = \
                self.eng.reticolo_res2(pol, theta, phi, period, n_top, truncation_order,
                                       textures, profile, wavelength, grating_type, nout=4)
            res = (top_refl_info, top_tran_info, bottom_refl_info, bottom_tran_info)
        return res

    # def run_acs(self, pattern, n_si='SILICON'):
    #     if type(n_si) == str and n_si.upper() == 'SILICON':
    #         n_si = find_nk_index(n_si, self.mat_table, self.wavelength)
    #
    #     abseff, effi_r, effi_t = self.eng.reticolo_res2(pattern, self.wavelength, self.deflected_angle,
    #                                                     self.truncation_order,
    #                                                     self.n_top, self.n_bot, self.thickness, self.theta, n_si, nout=3)
    #     effi_r, effi_t = np.array(effi_r).flatten(), np.array(effi_t).flatten()
    #
    #     return abseff, effi_r, effi_t


if __name__ == '__main__':

    option = {
        'grating_type': 1,
        'pol': 1,
        'top': 1,
        'bot': 1,
        'theta': 1,
        'phi': 1,
        'wavelength': 1,
        'truncation_order': 1,
        'thickness': [1000],
        'period': [1000],
        'factorization': 'nvf',
        # Permittivity, like every other scene in the tree. Written through
        # `eps_from_index` rather than as squared literals because the numbers below came
        # from an index table, and that is what the migration looks like at a call site --
        # the conversion visible, rather than four rounded constants nobody can check.
        'structure': wavebit.eps_from_index(np.array(
            [
                [[3.1, 1.1, 1.2, 1.6, 3.1]],
            ]
        ).astype(np.complex128)),
    }

    # option = {
    #     'grating_type': 1,
    #     'pol': 1,
    #     'n_top': 1,
    #     'n_bot': 1,
    #     'theta': 1,
    #     'phi': 1,
    #     'wavelength': 550,
    #     'truncation_order': 1,
    #     'thickness': [220],
    #     'period': [480],
    #     'factorization': 'nvf',
    #     'ucell': (np.array(
    #         [
    #             [
    #                 [1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1, 1, -1, 1,
    #                  1, 1, 1,
    #                  1, -1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, 1,
    #                  1, 1,
    #                  1, ],
    #             ]]
    #     ).astype(np.complex128)+1) * 1.5 + 1,
    # }



    reti = Reticolo()
    # Scratch demo, kept only as a worked example of driving the bridge by hand; the tested
    # path is `tests/references/gen_goldens_reticolo_field.py`. The unpacking was already
    # stale before `z` / `index_map` were added -- it took 5 values from a 9-tuple.
    reti_de_ri, reti_de_ti, *_, r_field_cell, _z, _index_map = reti.run_res3(**option)
    print('reti R', np.array(reti_de_ri).flatten())
    print('reti T', np.array(reti_de_ti).flatten())

    # Numpy
    backend = 0
    nmee = wavebit.rcwa(backend=backend, answers='field', **option)
    n_de_ri, n_de_ti = nmee.solve()
    n_field_cell = nmee.reconstruct_field(sample_z=200, sample_x=5)
    print('numpy R', n_de_ri)
    print('numpy T', n_de_ti)

    # # JAX
    # backend = 1
    # jmee = wavebit.rcwa(backend=backend, **option)
    # j_de_ri, j_de_ti = jmee.solve()
    # j_field_cell = jmee.reconstruct_field(sample_z=200, sample_x=5)
    # print('jax R', j_de_ri)
    # print('jax T', j_de_ti)
    #
    # # Torch
    # backend = 2
    # tmee = wavebit.rcwa(backend=backend, **option)
    # t_de_ri, t_de_ti = tmee.solve()
    # t_field_cell = tmee.reconstruct_field(sample_z=200, sample_x=5)
    # print('torch R', t_de_ri)
    # print('torch T', t_de_ti)

    import matplotlib.pyplot as plt

    plt.imshow(abs(r_field_cell[:,:,0]**2), cmap='jet', aspect='auto')
    plt.colorbar()
    plt.show()

    plt.imshow(abs(n_field_cell[:,0,:,0]**2), cmap='jet', aspect='auto')
    plt.colorbar()
    plt.show()

    # plt.imshow(abs(j_field_cell[:,0,:,0]**2), cmap='jet', aspect='auto')
    # plt.colorbar()
    # plt.show()
    #
    # plt.imshow(abs(t_field_cell[:,0,:,0]**2), cmap='jet', aspect='auto')
    # plt.colorbar()
    # plt.show()
