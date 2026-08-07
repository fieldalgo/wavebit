import jax

import numpy as np
import jax.numpy as jnp

from functools import partial

from ._base import _BaseRCWA, jax_device_set
from ..._core.engine import CoreFactorizationMixin
from ..._core.fields import field_1d, field_1d_conical, field_2d, field_plot
from ..._core.modeler import Modeling


@jax.tree_util.register_pytree_node_class
class ResultJax:
    """Solve result. Registered as a pytree so a jitted method can return one."""

    def __init__(self, res=None, res_te_inc=None, res_tm_inc=None):

        self.res = res
        self.res_te_inc = res_te_inc
        self.res_tm_inc = res_tm_inc

    def tree_flatten(self):
        return (self.res, self.res_te_inc, self.res_tm_inc), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @property
    def de_ri(self):
        if self.res is not None:
            return self.res.de_ri
        elif self.res_te_inc is not None and self.res_tm_inc is not None:
            return 0.5 * (self.res_te_inc.de_ri + self.res_tm_inc.de_ri)
        elif self.res_te_inc is not None:
            return self.res_te_inc.de_ri
        elif self.res_tm_inc is not None:
            return self.res_tm_inc.de_ri
        else:
            return None

    @property
    def de_ti(self):
        if self.res is not None:
            return self.res.de_ti
        elif self.res_te_inc is not None and self.res_tm_inc is not None:
            return 0.5 * (self.res_te_inc.de_ti + self.res_tm_inc.de_ti)
        elif self.res_te_inc is not None:
            return self.res_te_inc.de_ti
        elif self.res_tm_inc is not None:
            return self.res_tm_inc.de_ti
        else:
            return None


@jax.tree_util.register_pytree_node_class
class ResultSubJax:
    """One incidence's amplitudes and efficiencies; a pytree for the same reason as ResultJax."""

    _FIELDS = ('R_s', 'R_p', 'T_s', 'T_p',
               'de_ri', 'de_ri_s', 'de_ri_p', 'de_ti', 'de_ti_s', 'de_ti_p')

    def tree_flatten(self):
        return tuple(getattr(self, name) for name in self._FIELDS), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def __init__(self, R_s, R_p, T_s, T_p, de_ri, de_ri_s, de_ri_p, de_ti, de_ti_s, de_ti_p):
        self.R_s = R_s
        self.R_p = R_p
        self.T_s = T_s
        self.T_p = T_p
        self.de_ri = de_ri
        self.de_ri_s = de_ri_s
        self.de_ri_p = de_ri_p

        self.de_ti = de_ti
        self.de_ti_s = de_ti_s
        self.de_ti_p = de_ti_p


class RCWAJax(CoreFactorizationMixin, _BaseRCWA):
    def modeling_vector_instruction(self, *args, **kwargs):
        return self.modeler.modeling_vector_instruction(*args, **kwargs)

    def rectangle(self, *args, **kwargs):
        return self.modeler.rectangle(*args, **kwargs)

    def ellipse(self, *args, **kwargs):
        return self.modeler.ellipse(*args, **kwargs)

    def __init__(self,
                 n_top=1.,
                 n_bot=1.,
                 theta=0.,
                 phi=None,
                 psi=None,
                 period=(1., 1.),
                 wavelength=1.,
                 ucell=None,
                 thickness=(0., ),
                 backend=1,
                 pol=0.,
                 fto=(0, 0),
                 ucell_materials=None,
                 connecting_algo='TMM',
                 perturbation=1E-20,
                 device='cpu',
                 type_complex=jnp.complex128,
                 fourier_coefficients=None,  # 'sampled' (default, DFS) | 'integrated' (closed-form series)
                 factorization=None,         # 'li' (default) | 'nvf' (normal vector field)
                 fourier_type=None,          # deprecated alias for the two keywords above
                 analytic_normal_grid=1024,  # normal-field sampling; sets the floor on shape gradients
                 # (a translation-invariant structure should give d/dcx == 0 exactly; the
                 #  sampling leaves a sawtooth instead -- 5e-4 at 256, 4.2e-5 at 1024, for
                 #  ~1.2-2x the runtime. Raise further to fit positions below that.)
                 ):

        super().__init__(n_top=n_top, n_bot=n_bot, theta=theta, phi=phi, psi=psi, pol=pol,
                         fto=fto, period=period, wavelength=wavelength,
                         thickness=thickness, connecting_algo=connecting_algo, perturbation=perturbation,
                         device=device, type_complex=type_complex)

        self.modeler = Modeling(period=self.period, type_complex=type_complex)

        self._modeling_type_assigned = None
        self._grating_type_assigned = None
        self._ucell = None

        self.ucell = ucell
        self.ucell_materials = ucell_materials

        self.backend = backend
        self._set_fourier_options(fourier_coefficients, factorization, fourier_type)
        self.analytic_normal_grid = analytic_normal_grid

    @property
    def ucell(self):
        return self._ucell

    @ucell.setter
    def ucell(self, ucell):

        if isinstance(ucell, jnp.ndarray):  # Raster
            self._modeling_type_assigned = 0
            if ucell.dtype in (jnp.float64, jnp.float32, jnp.int64, jnp.int32):
                dtype = self.type_float
                self._ucell = ucell.astype(dtype)
            elif ucell.dtype in (jnp.complex128, jnp.complex64):
                dtype = self.type_complex
                self._ucell = ucell.astype(dtype)

        elif isinstance(ucell, np.ndarray):  # Raster
            self._modeling_type_assigned = 0
            if ucell.dtype in (np.int64, np.float64, np.int32, np.float32):
                dtype = self.type_float
                self._ucell = jnp.array(ucell, dtype=dtype)
            elif ucell.dtype in (np.complex128, np.complex64):
                dtype = self.type_complex
                self._ucell = jnp.array(ucell, dtype=dtype)

        elif type(ucell) is list:
            # Analytic: one dict(n_base=..., shapes=[...], combine=...) per layer.
            # Shape coordinates are absolute lengths in the units of `period`, not
            # fractions of it, and `combine` picks what overlapping shapes mean --
            # this path adds them where the vector path paints. See
            # fieldwave._core.analytic; overlaps are refused until it is set.
            if len(ucell) > 0 and isinstance(ucell[0], dict):
                self._modeling_type_assigned = 2
                self._ucell = ucell
            else:  # Vector
                self._modeling_type_assigned = 1
                self._ucell = ucell
        elif ucell is None:
            self._ucell = ucell
            self._modeling_type_assigned = None
        else:
            raise ValueError

    @property
    def modeling_type_assigned(self):
        return self._modeling_type_assigned

    # @modeling_type_assigned.setter
    # def modeling_type_assigned(self, modeling_type_assigned):
    #     self._modeling_type_assigned = modeling_type_assigned

    def _assign_grating_type(self):
        """
        Select the grating type for RCWA simulation. This decides the efficient formulation for given case.

        `_grating_type_assigned` == 0(1D TETM)    is for 1D grating, no rotation (phi or azimuth), and either TE or TM.
        `_grating_type_assigned` == 1(1D conical) is for 1D grating with generality.
        `_grating_type_assigned` == 2(2D)         is for 2D grating with generality.

        Note that no rotation means 'phi' is `None`. If phi is given as '0', then it takes 1D conical form
         even though when the case itself is 1D TETM.

        1D conical is under implementation.

        Returns:

        """
        if self.ucell is None:
            if self._grating_type_assigned is None:
                self._grating_type_assigned = 2
            return

        if self.modeling_type_assigned == 0:  # Raster
            if self.ucell.shape[1] == 1:
                if (self.pol in (0, 1)) and (self.phi is None) and (self.fto[1] == 0):
                    self._grating_type_assigned = 0
                else:
                    self._grating_type_assigned = 1
            else:
                self._grating_type_assigned = 2

        # TODO: jit
        # def false_fun(): return 0  # 1D TE and TM only
        # def true_fun(): return 1
        #
        # gear = jax.lax.cond(self.phi % (2 * np.pi) + self.fto[1], true_fun, false_fun)
        #
        # self._grating_type_assigned = gear

        elif self.modeling_type_assigned == 1:  # Vector
            self.grating_type_assigned = 2

        elif self.modeling_type_assigned == 2:  # Analytic (staircase-free shapes)
            self._grating_type_assigned = 2

    @property
    def grating_type_assigned(self):
        return self._grating_type_assigned

    @grating_type_assigned.setter
    def grating_type_assigned(self, grating_type_assigned):
        self._grating_type_assigned = grating_type_assigned

    @jax_device_set
    def solve_for_conv(self, wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all,
                       epxy_conv_all=None, epyx_conv_all=None):

        # def false_fun(wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all):
        #     de_ri, de_ti, layer_info_list, T1 = self.solve_1d(wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all)
        #     return de_ri, de_ti, layer_info_list, T1
        #
        # def true_fun(wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all):
        #     de_ri, de_ti, layer_info_list, T1 = self.solve_2d(wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all)
        #     return de_ri, de_ti, layer_info_list, T1
        #
        # de_ri, de_ti, layer_info_list, T1 = jax.lax.cond(self._grating_type_assigned, true_fun, false_fun, wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all)

        self._assign_grating_type()

        if self._grating_type_assigned == 0:
            result_dict = self.solve_1d(wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all)
        elif self._grating_type_assigned == 1:
            result_dict = self.solve_1d_conical(wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all)
        else:
            result_dict = self.solve_2d(wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all,
                                        epxy_conv_all, epyx_conv_all)

        # TODO: In JAXMeent, 1D TE TM are turned off for jit compilation.

        res_psi = ResultSubJax(**result_dict['res']) if 'res' in result_dict else None
        res_te_inc = ResultSubJax(**result_dict['res_te_inc']) if 'res_te_inc' in result_dict else None
        res_tm_inc = ResultSubJax(**result_dict['res_tm_inc']) if 'res_tm_inc' in result_dict else None

        result = ResultJax(res_psi, res_te_inc, res_tm_inc)

        return result

    # @jax_device_set
    # def solve(self, wavelength, e_conv_all, o_e_conv_all):
    #     de_ri, de_ti, layer_info_list, T1, kx_vector = jax.jit(self._solve)(wavelength, e_conv_all, o_e_conv_all)
    #
    #     self.layer_info_list = layer_info_list
    #     self.T1 = T1
    #
    #     return de_ri, de_ti

    @jax_device_set
    def calculate_field(self, res_x=20, res_y=20, res_z=20, set_field_input=(True, False, False)):
        """Reconstruct the field from the state left behind by the last solve.

        Takes no ``geometry``: it never calls ``conv_solve``, and every discrete decision it
        makes -- the grating type, and which incidences ``set_field_input`` selects -- is
        already concrete by the time it runs. It reads ``self.T1`` / ``self.layer_info_list``,
        so whichever geometry the preceding solve used is the one in force here.
        """
        kx, ky = self.get_kx_ky_vector(wavelength=self.wavelength)

        if self._grating_type_assigned == 0:
            res_y = 1
            field_cell = field_1d(self.wavelength, kx, self.T1, self.layer_info_list, self.period, self.pol,
                                       res_x=res_x, res_y=res_y, res_z=res_z)
        elif self._grating_type_assigned == 1:
            field_cell = field_1d_conical(self.wavelength, kx, ky, self.T1, self.layer_info_list, self.period,
                                               res_x=res_x, res_y=res_y, res_z=res_z,  set_field_input=set_field_input)
        else:
            field_cell = field_2d(self.wavelength, kx, ky, self.T1, self.layer_info_list, self.period,
                                       res_x=res_x, res_y=res_y, res_z=res_z, set_field_input=set_field_input)

        return field_cell

    def field_plot(self, field_cell):
        field_plot(field_cell, self.pol)

    @partial(jax.jit, static_argnums=(1, 2, 3, 4, 5),
             static_argnames=('res_x', 'res_y', 'res_z', 'set_field_input', 'geometry'))
    @jax_device_set
    def conv_solve_field(self, res_x=20, res_y=20, res_z=20, set_field_input=(True, False, False),
                         geometry=None, **kwargs):
        """Solve and reconstruct the field in one jit-compiled call.

        ``geometry`` is required here, unlike on :meth:`conv_solve`. This method jits over
        ``self``, so the structure arrives as tracers and the discrete decisions inside the
        factorization and the solve have nothing concrete to branch on; a
        :class:`~fieldwave._core.geometry.StaticGeometry` from
        :meth:`~fieldwave._core.engine.CoreFactorizationMixin.freeze_geometry` supplies them as
        static data. It is a static argument -- reuse one freeze across calls and you compile
        once.

        Because this method always traces, a stale ``geometry`` is replayed on trust here: the
        comparison that would catch it is the thing the freeze exists to avoid. ``conv_solve``
        refuses one eagerly, so check a long fit there, or re-freeze on a schedule.

        ``res_x`` / ``res_y`` / ``res_z`` and ``set_field_input`` are static too: they set the
        output shape and select which incidences to reconstruct.

        Use :meth:`conv_solve_field_no_jit` for the eager equivalent that auto-detects.
        """
        if geometry is None:
            raise ValueError(
                'conv_solve_field is jit-compiled and needs the geometry frozen up front:\n'
                '    geometry = solver.freeze_geometry()\n'
                '    result, field_cell = solver.conv_solve_field(..., geometry=geometry)\n'
                'Reuse that one geometry across calls to compile only once. '
                'For the eager, auto-detecting path use solver.conv_solve_field_no_jit(...).')

        [setattr(self, k, v) for k, v in kwargs.items()]  # needed for optimization

        res = self.conv_solve(geometry=geometry)
        field_cell = self.calculate_field(res_x, res_y, res_z, set_field_input)
        return res, field_cell

    @jax_device_set
    def conv_solve_field_no_jit(self, res_x=20, res_y=20, res_z=20, set_field_input=(True, False, False),
                                geometry=None):
        """Eager solve-and-field. ``geometry`` is optional: omit it and each discrete decision
        is auto-detected from the concrete structure."""
        res = self.conv_solve(geometry=geometry)
        field_cell = self.calculate_field(res_x, res_y, res_z, set_field_input)
        return res, field_cell

    def run_ucell_vmap(self, ucell_list):
        """
        under development
        """

        de_ri, de_ti = jax.vmap(self.conv_solve)(ucell_list)

        return de_ri, de_ti

    def run_ucell_pmap(self, ucell_list):
        """
        under development
        """

        de_ri, de_ti = jax.pmap(self.conv_solve)(ucell_list)
        # de_ri.block_until_ready()
        # de_ti.block_until_ready()

        de_ri = jnp.array(de_ri)
        de_ti = jnp.array(de_ti)
        return de_ri, de_ti
