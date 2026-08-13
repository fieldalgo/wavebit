import jax

import numpy as np
import jax.numpy as jnp

from functools import partial

from ._base import _BaseRCWA, jax_device_set
from ..._core.engine import CoreFactorizationMixin
from ..._core.enums import Formulation
from ..._core.fields import field_1d, field_1d_conical, field_2d, field_plot
from ..._core.result import Result, ResultSub
from ..._core.modeler import Modeling


@jax.tree_util.register_pytree_node_class
class ResultJax(Result):
    """Shared container, registered as a pytree so a jitted method can return one."""


@jax.tree_util.register_pytree_node_class
class ResultSubJax(ResultSub):
    """Shared container; a pytree for the same reason as :class:`ResultJax`."""


class RCWAJax(CoreFactorizationMixin, _BaseRCWA):
    def modeling_block_instruction(self, *args, **kwargs):
        return self.modeler.modeling_block_instruction(*args, **kwargs)

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
                 tetm=None,
                 pol=None,   # retired; kept only to say so
                 fto=(0, 0),
                 ucell_materials=None,
                 connecting_algo='ETM',
                 perturbation=1E-20,
                 device='cpu',
                 type_complex=jnp.complex128,
                 fourier_coefficients=None,  # 'sampled' | 'integrated'
                 # (closed-form series); the default resolves per modeling path --
                 # 'sampled' on a raster, 'integrated' on both shape paths, which is
                 # the only series either can form. Naming the other one raises.
                 factorization=None,         # 'li' | 'nvf' (normal vector field); the
                 # default resolves per modeling path -- 'li' on a raster and on the
                 # vector modeler, 'nvf' on analytic shapes, which is the only rule
                 # each of those last two implements. Naming the other one raises.
                 normal_grid=1024,  # normal-field sampling; sets the floor on shape gradients
                 # (a translation-invariant structure should give d/dcx == 0 exactly; the
                 #  sampling leaves a sawtooth instead -- 5e-4 at 256, 4.2e-5 at 1024, for
                 #  ~1.2-2x the runtime. Raise further to fit positions below that.)
                 lattice=None,               # 2x2 cell basis (columns a1, a2); None = diag(period).
                 # New parameters go at the end on purpose: SolverJax._tree_unflatten passes
                 # the children positionally, so one added earlier silently shifts them.
                 incidence_side='top',       # 'top' | 'bottom': which medium the light
                 # arrives from. Scene labels (n_top / n_bot / ucell order) never move;
                 # results are incidence-relative and theta/phi are measured in the
                 # incidence medium. See docs/conventions.md.
                 chi=None,                   # ellipticity angle; with psi it is the
                 # canonical (psi, chi) chart of the polarization state. Also at the
                 # end, for the reason above.
                 polarization=None,          # complex incident amplitude pair (a_s, a_p);
                 # supersedes psi (naming both raises). See docs/conventions.md.
                 ):

        super().__init__(n_top=n_top, n_bot=n_bot, theta=theta, phi=phi, psi=psi, chi=chi, tetm=tetm, pol=pol,
                         fto=fto, period=period, wavelength=wavelength,
                         thickness=thickness, connecting_algo=connecting_algo, perturbation=perturbation,
                         device=device, type_complex=type_complex,
                         polarization=polarization)
        self.incidence_side = incidence_side

        self.modeler = Modeling(period=self.period, type_complex=type_complex)

        self._modeling_type_assigned = None
        self._grating_type_assigned = None
        self._ucell = None

        self.ucell = ucell
        self.ucell_materials = ucell_materials

        self.backend = backend
        self._set_fourier_options(fourier_coefficients, factorization)
        self.normal_grid = normal_grid
        self.lattice = lattice      # after period: assigning a basis overwrites period

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
            # this path adds them where the block path paints. See
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
            one_dimensional = self.ucell.shape[1] == 1
            self._refuse_illegal_tetm(one_dimensional)
            if one_dimensional:
                # The declaration selects the path, and nothing else does. A named incidence
                # -- psi, or a pair, which mixes s and p the pure path cannot carry -- leaves
                # the polarization a live parameter, and the conical formulation is the 1D
                # one that carries it.
                self._grating_type_assigned = 0 if self._tetm is not None else 1
            else:
                self._grating_type_assigned = 2

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
                       epxy_conv_all=None, epyx_conv_all=None,
                       epxz_conv_all=None, epyz_conv_all=None,
                       epzx_conv_all=None, epzy_conv_all=None):

        self._assign_grating_type()
        z_row = (epxz_conv_all, epyz_conv_all, epzx_conv_all, epzy_conv_all)
        if self._formulation_for(z_row) == Formulation.FIRST:
            # See the NumPy solver: a z row is also a choice of path, since the 1D paths have
            # no slots to put one in.
            self._grating_type_assigned = 2

        if self._grating_type_assigned == 0:
            result_dict = self.solve_1d(wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all)
        elif self._grating_type_assigned == 1:
            result_dict = self.solve_1d_conical(wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all)
        else:
            result_dict = self.solve_2d(wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all,
                                        epxy_conv_all, epyx_conv_all, *z_row)

        res_psi = ResultSubJax(**result_dict['res']) if 'res' in result_dict else None
        res_te_inc = ResultSubJax(**result_dict['res_te_inc']) if 'res_te_inc' in result_dict else None
        res_tm_inc = ResultSubJax(**result_dict['res_tm_inc']) if 'res_tm_inc' in result_dict else None

        result = ResultJax(res_psi, res_te_inc, res_tm_inc, fto=self.fto)

        return result


    @jax_device_set
    def calculate_field(self, res_x=20, res_y=20, res_z=20, set_field_input=(True, False, False)):
        """Reconstruct the field from the state left behind by the last solve.

        Takes no ``geometry``: it never calls ``conv_solve``, and every discrete decision it
        makes -- the grating type, and which incidences ``set_field_input`` selects -- is
        already concrete by the time it runs. It reads ``self.T1`` / ``self.layer_info_list``,
        so whichever geometry the preceding solve used is the one in force here.
        """
        self._require_field_state()
        kx, ky = self._kxy_2d(self.wavelength)

        if self._grating_type_assigned == 0:
            res_y = 1
            field_cell = field_1d(self.wavelength, kx, self.T1, self.layer_info_list, self.period, self.tetm_index,
                                       res_x=res_x, res_y=res_y, res_z=res_z)
        elif self._grating_type_assigned == 1:
            field_cell = field_1d_conical(self.wavelength, kx, ky, self.T1, self.layer_info_list, self.period,
                                               res_x=res_x, res_y=res_y, res_z=res_z,  set_field_input=set_field_input,
                                               lattice=self.lattice)
        else:
            field_cell = field_2d(self.wavelength, kx, ky, self.T1, self.layer_info_list, self.period,
                                       res_x=res_x, res_y=res_y, res_z=res_z, set_field_input=set_field_input,
                                       lattice=self.lattice)

        if self.incidence_side == 'bottom':
            # The backside solve leaves its state in the mirrored frame (z reversed); un-flip
            # once here so the returned cell is in SCENE coordinates for either side. The
            # branch is static under jit: incidence_side is aux data, never a tracer.
            field_cell = jnp.flip(field_cell,
                                  axis=0 if self._grating_type_assigned == 0 else 1)

        return field_cell

    def field_plot(self, field_cell):
        field_plot(field_cell, self.tetm_index)

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

        R, T = jax.vmap(self.conv_solve)(ucell_list)

        return R, T

    def run_ucell_pmap(self, ucell_list):
        """
        under development
        """

        R, T = jax.pmap(self.conv_solve)(ucell_list)

        return jnp.array(R), jnp.array(T)
