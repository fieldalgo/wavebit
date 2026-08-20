import jax

import numpy as np
import jax.numpy as jnp

from functools import partial

from ._base import _BaseRCWA, jax_device_set
from ..._core.domain import retired_material_name
from ..._core.engine import CoreFactorizationMixin, _refuse_retired
from ..._core.enums import Formulation
from ..._core.fields import field_1d, field_1d_conical, field_2d, field_2d_4ff, field_plot
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
                 top=1.,                     # superstrate: a Material, or a bare relative
                 # permittivity (the scalar shorthand for Material(epsilon=x)). Was `n_top`
                 # and held a refractive index; both the name and the quantity changed.
                 bot=1.,                     # substrate, same reading as `top`.
                 theta=0.,
                 phi=None,
                 psi=None,
                 period=(1., 1.),
                 wavelength=1.,
                 structure=None,             # the layer stack. A raster is an
                 # (n_layers, Ny, Nx) array of relative PERMITTIVITY; the two shape spellings
                 # name a Material per region instead. Was `ucell`, and held index samples.
                 thickness=(0., ),
                 backend=1,
                 tetm=None,
                 pol=None,   # retired; kept only to say so
                 max_order=(0, 0),
                 connecting_algo='ETM',
                 device='cpu',
                 type_complex=jnp.complex128,
                 fourier_coefficients=None,  # 'sampled' | 'integrated'
                 # (closed-form series); the default resolves per modeling path --
                 # 'sampled' on a raster, 'integrated' on both shape paths, which is
                 # the only series either can form. Naming the other one raises.
                 factorization=None,         # 'li' | 'nvf' (normal vector field); the
                 # default resolves per modeling path -- 'li' on a raster and on the
                 # block modeler, 'nvf' on analytic shapes, which is the only rule
                 # each of those last two implements. Naming the other one raises.
                 normal_grid=1024,  # normal-field sampling; sets the floor on shape gradients
                 # (a translation-invariant structure should give d/dcx == 0 exactly; the
                 #  sampling leaves a sawtooth instead -- 5e-4 at 256, 4.2e-5 at 1024, for
                 #  ~1.2-2x the runtime. Raise further to fit positions below that.)
                 lattice=None,               # 2x2 cell basis (columns a1, a2); None = diag(period).
                 # New parameters go at the end on purpose: SolverJax._tree_unflatten passes
                 # the children positionally, so one added earlier silently shifts them.
                 incidence_side='top',       # 'top' | 'bottom': which medium the light
                 # arrives from. Scene labels (top / bot / structure order) never move;
                 # results are incidence-relative and theta/phi are measured in the
                 # incidence medium. See docs/conventions.md.
                 chi=None,                   # ellipticity angle; with psi it is the
                 # canonical (psi, chi) chart of the polarization state. Also at the
                 # end, for the reason above.
                 polarization=None,          # complex incident amplitude pair (a_s, a_p);
                 # supersedes psi (naming both raises). See docs/conventions.md.
                 normal_method=None,         # raster NVF normal-field construction:
                 # 'distance' (default) | 'contour' | 'spectral' | 'smooth'. Read only
                 # where a raster normal field is built (factorization='nvf' on a raster);
                 # explicitly set anywhere else it is refused rather than ignored.
                 # See docs/nvf-normal-methods.md for the measurements that rank them.
                 answers='amplitude',        # what this solve keeps the state to answer:
                 # 'efficiency' | 'amplitude' (default) | 'field'. A declaration, like tetm --
                 # reconstruct_field is refused below 'field' rather than served, because the
                 # per-layer state it walks is 4.25 matrices of side 2*n_orders_x*n_orders_y per
                 # layer (240 MB at max_order=(15, 15)) and is kept as the solve runs or not at
                 # all. See CoreSolveMixin.answers.
                 mu=None,                    # raster path only: the permeability field,
                 # (n_layers, Ny, Nx) beside `structure`. On the two shape paths a
                 # permeability belongs to the Material filling each region, so naming it
                 # here is refused rather than ignored. New parameters go at the end -- see
                 # the note on `lattice` above.
                 **retired,                  # ucell / n_top / n_bot / ucell_materials, so
                 # each can say what replaced it *and* that the quantity changed with the
                 # name. A TypeError alone would send a reader to rename the keyword and keep
                 # the number, which is a different, legal, energy-conserving structure.
                 ):
        _refuse_retired(retired)

        super().__init__(top=top, bot=bot, theta=theta, phi=phi, psi=psi, chi=chi, tetm=tetm, pol=pol,
                         max_order=max_order, period=period, wavelength=wavelength,
                         thickness=thickness, connecting_algo=connecting_algo,
                         device=device, type_complex=type_complex,
                         polarization=polarization)
        self.incidence_side = incidence_side
        self.answers = answers

        self.modeler = Modeling(period=self.period, type_complex=type_complex)

        self._modeling_type_assigned = None
        self._grating_type_assigned = None
        self._structure = None

        self.structure = structure
        # After `structure`: `mu` is refused on the two shape paths, and which path this is
        # only becomes known when the structure is assigned.
        self.mu = mu

        self.backend = backend
        self._set_fourier_options(fourier_coefficients, factorization)
        self.normal_method = normal_method
        self.normal_grid = normal_grid
        self.lattice = lattice      # after period: assigning a basis overwrites period

    @property
    def structure(self):
        """The layer stack, in whichever of the three encodings it was given in.

        A raster is an ``(n_layers, Ny, Nx)`` array of **relative permittivity**; the block
        and analytic spellings are per-layer lists that name a
        :class:`~wavebit._core.domain.Material` per region. Assigning it settles
        :attr:`modeling_type_assigned`, which is the only thing that selects the modeling
        path -- there is no separate mode switch.
        """
        return self._structure

    @property
    def ucell(self):
        raise AttributeError(retired_material_name('ucell'))

    @ucell.setter
    def ucell(self, value):
        raise AttributeError(retired_material_name('ucell'))

    @structure.setter
    def structure(self, structure):

        if isinstance(structure, (jnp.ndarray, np.ndarray)):  # Raster
            self._modeling_type_assigned = 0
            self._structure = self._coerce_raster('structure', structure)

        elif type(structure) is list:
            # Analytic: one dict(background=..., shapes=[...], combine=...) per layer.
            # Shape coordinates are absolute lengths in the units of `period`, not
            # fractions of it, and `combine` picks what overlapping shapes mean --
            # this path adds them where the block path paints. See
            # wavebit._core.analytic; overlaps are refused until it is set.
            if len(structure) > 0 and isinstance(structure[0], dict):
                self._modeling_type_assigned = 2
                self._structure = structure
            else:  # Block
                self._modeling_type_assigned = 1
                self._structure = structure
        elif structure is None:
            self._structure = structure
            self._modeling_type_assigned = None
        else:
            raise ValueError
        # `mu` and `structure` are one raster pair and either can be assigned second, so the
        # pair is checked from whichever side completes it -- see `_revalidate_mu`.
        self._revalidate_mu()

    def _coerce_raster(self, where, grid):
        """A raster field on this backend at the solve's precision.

        Shared by ``structure`` and ``mu`` -- they are two fields of the same shape sampled on
        the same grid, so a second copy of this would be a second chance for them to disagree
        about dtype.
        """
        if not isinstance(grid, (jnp.ndarray, np.ndarray)):
            raise ValueError(
                f'{where}: expected a JAX or NumPy array on this backend, got '
                f'{type(grid).__name__}')
        if grid.dtype in (jnp.float64, jnp.float32, jnp.int64, jnp.int32,
                          np.int64, np.float64, np.int32, np.float32):
            dtype = self.type_float
        elif grid.dtype in (jnp.complex128, jnp.complex64,
                            np.complex128, np.complex64):
            dtype = self.type_complex
        else:
            raise ValueError(f'{where}: unsupported dtype {grid.dtype}')
        return jnp.asarray(grid, dtype=dtype)

    @property
    def modeling_type_assigned(self):
        return self._modeling_type_assigned


    def _assign_grating_type(self):
        """
        Select the grating type for RCWA simulation. This decides the efficient formulation for
        given case.

        `_grating_type_assigned` == 0(1D TETM)    is for 1D grating, no rotation (phi or
        azimuth), and either TE or TM. `_grating_type_assigned` == 1(1D conical) is for 1D
        grating with generality. `_grating_type_assigned` == 2(2D)         is for 2D grating
        with generality.

        Note that no rotation means 'phi' is `None`. If phi is given as '0', then it takes 1D
        conical form
         even though when the case itself is 1D TETM.

        1D conical is under implementation.

        Returns:

        """
        if self.structure is None:
            if self._grating_type_assigned is None:
                self._grating_type_assigned = 2
            return

        if self.modeling_type_assigned == 0:  # Raster
            one_dimensional = self.structure.shape[1] == 1
            self._refuse_y_orders_on_a_y_invariant_cell(one_dimensional)
            self._refuse_illegal_tetm(one_dimensional)
            if one_dimensional:
                # The declaration selects the path, and nothing else does. A named incidence
                # -- psi, or a pair, which mixes s and p the pure path cannot carry -- leaves
                # the polarization a live parameter, and the conical formulation is the 1D
                # one that carries it.
                self._grating_type_assigned = 0 if self._tetm is not None else 1
            else:
                self._grating_type_assigned = 2

        elif self.modeling_type_assigned == 1:  # Block
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
    def solve_conv(self, epx_conv_all, epy_conv_all, epz_conv_i_all,
                   epxy_conv_all=None, epyx_conv_all=None,
                   epxz_conv_all=None, epyz_conv_all=None,
                   epzx_conv_all=None, epzy_conv_all=None, mu=None,
                   mu_top=None, mu_bot=None, eps_top=None, eps_bot=None):
        """Solve a stack whose convolution matrices are already built.

        The nine permittivity slots in :meth:`conv_matrices` order -- the in-plane
        ``(epx, epy, epz_i, epxy, epyx)``, then the z row ``(epxz, epyz, epzx, epzy)`` --
        followed by ``mu`` and the half-space terms. ``solver.solve_conv(*solver.conv_matrices())``
        is ``solver.solve()``; see :mod:`wavebit._core.factorization` for why ``epz_i`` sits
        third rather than last.

        Takes no wavelength. The solver already has one, and a second spelling of it was a
        second thing that could disagree: the k vectors, the field reconstruction and the
        auto-detune ladder all read ``self.wavelength``, so an argument that differed would
        have answered one problem while the rest of the object described another.
        """

        self._assign_grating_type()
        z_row = (epxz_conv_all, epyz_conv_all, epzx_conv_all, epzy_conv_all)
        if self._formulation_for(z_row) == Formulation.FIRST:
            # See the NumPy solver: a z row is also a choice of path, since the 1D paths have
            # no slots to put one in.
            self._grating_type_assigned = 2

        # Structural, like `Material.is_magnetic`: a permeability is present when one was
        # *named*, not when its value differs from one. This used to read `mu_top != 1.0`, and
        # under `jax.jit` a named `mu` is a tracer -- so the comparison produced a traced
        # boolean and `or` raised TracerBoolConversionError on every traced solve, which took
        # the jitted `solve_field` below out of service for magnetic and non-magnetic
        # scenes alike. `None` is an empty pytree subtree and comes through a trace as itself,
        # which is the whole reason it is how absence is spelled.
        magnetic = ((mu is not None and any(m is not None for m in mu))
                    or mu_top is not None or mu_bot is not None)
        if magnetic and self._grating_type_assigned != 2:
            raise NotImplementedError(
                'mu is implemented on the 2D path only; this structure routed to a 1D solve. '
                'The 1D and conical operators carry no permeability slot at all.')

        if self._grating_type_assigned == 0:
            result_dict = self._solve_1d(self.wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all)
        elif self._grating_type_assigned == 1:
            result_dict = self._solve_1d_conical(self.wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all)
        else:
            result_dict = self._solve_2d(self.wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all,
                                         epxy_conv_all, epyx_conv_all, *z_row, mu=mu,
                                         mu_top=mu_top, mu_bot=mu_bot,
                                         eps_top=eps_top, eps_bot=eps_bot)

        given = ResultSubJax(**result_dict['given']) if 'given' in result_dict else None
        s_inc = ResultSubJax(**result_dict['s_inc']) if 's_inc' in result_dict else None
        p_inc = ResultSubJax(**result_dict['p_inc']) if 'p_inc' in result_dict else None

        result = ResultJax(given, s_inc, p_inc, max_order=self.max_order)

        return result


    @jax_device_set
    def reconstruct_field(self, sample_x=20, sample_y=20, sample_z=20, incidences=('given',)):
        """Reconstruct the field from the state left behind by the last solve.

        ``sample_x`` / ``sample_y`` / ``sample_z`` each take either a **count** -- a uniform
        grid, the long-standing behaviour and unchanged to the bit -- or the **coordinates
        themselves**, as a 1-D array. The reconstruction is pointwise in all three, so
        explicit points cost the same per point as grid ones and involve no interpolation.
        They are named for taking either; a count is one way to say where to sample.

        ``sample_x`` / ``sample_y`` are lengths in ``period``'s units and are unconstrained.
        ``sample_z`` is absolute, measured from the top of the stack, must lie inside it, and
        assigns an interface to the layer below it -- which is visible only in ``E_z``, the one
        component genuinely two-valued there. With explicit ``sample_z`` the returned z axis is
        exactly the request, in the caller's order, rather than ``sample_z * n_layers``.

        ``incidences`` names which of the incidences the solve carries to reconstruct, one
        leading axis entry each, **in the order asked for**: ``'given'`` is the incidence the
        solve was posed with (``psi``, or the ``polarization`` pair), and ``'s'`` / ``'p'``
        are the pure variants the coupled paths produce for free -- the same three the result
        object calls ``given`` / ``s_inc`` / ``p_inc``. It was a positional triple of booleans
        (``set_field_input``) until 2026-08-20; booleans are now refused by name, since a
        reconstruction of the wrong incidence is a complete and plausible field cell.

        Takes no ``geometry``: it never calls ``solve``, and every discrete decision it
        makes -- the grating type, and which incidences ``incidences`` selects -- is
        already concrete by the time it runs. It reads ``self.T1`` / ``self.layer_info_list``,
        so whichever geometry the preceding solve used is the one in force here.

        Explicit ``sample_z`` is the one thing here that cannot be traced: bucketing a z into a
        layer needs the thicknesses as numbers. It raises rather than guessing.
        """
        self._require_field_declaration()
        kx, ky = self._kxy_2d(self.wavelength)

        if self._grating_type_assigned == 0:
            sample_y = 1
            field_cell = field_1d(self.wavelength, kx, self.T1, self.layer_info_list,
                                  self.period, self.tetm_index,
                                  sample_x=sample_x, sample_y=sample_y, sample_z=sample_z)
        elif self._grating_type_assigned == 1:
            field_cell = field_1d_conical(self.wavelength, kx, ky, self.T1,
                                          self.layer_info_list, self.period,
                                          sample_x=sample_x, sample_y=sample_y,
                                          sample_z=sample_z, incidences=incidences,
                                          lattice=self.lattice)
        elif self._formulation_assigned == Formulation.FIRST:
            # The grating type cannot answer this: the first-order path sets it to 2 as well,
            # so branching on it alone would hand a 4ff solve to the 2ff reconstruction, which
            # reads the same state slots and returns a plausible, wrong cell. Static under jit
            # for the same reason the side flip below is -- the formulation is settled eagerly.
            field_cell = field_2d_4ff(self.wavelength, kx, ky, self.T1,
                                      self.layer_info_list, self.period,
                                      sample_x=sample_x, sample_y=sample_y,
                                      sample_z=sample_z, incidences=incidences,
                                      lattice=self.lattice)
        else:
            field_cell = field_2d(self.wavelength, kx, ky, self.T1,
                                  self.layer_info_list, self.period,
                                  sample_x=sample_x, sample_y=sample_y,
                                  sample_z=sample_z, incidences=incidences,
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
             static_argnames=('sample_x', 'sample_y', 'sample_z', 'incidences', 'geometry'))
    @jax_device_set
    def solve_field(self, sample_x=20, sample_y=20, sample_z=20, incidences=('given',),
                    geometry=None, **kwargs):
        """Solve and reconstruct the field in one jit-compiled call.

        ``geometry`` is required here, unlike on :meth:`solve`. This method jits over
        ``self``, so the structure arrives as tracers and the discrete decisions inside the
        factorization and the solve have nothing concrete to branch on; a
        :class:`~wavebit._core.geometry.StaticGeometry` from
        :meth:`~wavebit._core.engine.CoreFactorizationMixin.freeze_geometry` supplies them as
        static data. It is a static argument -- reuse one freeze across calls and you compile
        once.

        Because this method always traces, a stale ``geometry`` is replayed on trust here: the
        comparison that would catch it is the thing the freeze exists to avoid. ``solve``
        refuses one eagerly, so check a long fit there, or re-freeze on a schedule.

        ``sample_x`` / ``sample_y`` / ``sample_z`` and ``incidences`` are static too: they set the
        output shape and select which incidences to reconstruct.

        Use :meth:`solve_field_eager` for the eager equivalent that auto-detects.
        """
        if geometry is None:
            raise ValueError(
                'solve_field is jit-compiled and needs the geometry frozen up front:\n'
                '    geometry = solver.freeze_geometry()\n'
                '    result, field_cell = solver.solve_field(..., geometry=geometry)\n'
                'Reuse that one geometry across calls to compile only once. '
                'For the eager, auto-detecting path use solver.solve_field_eager(...).')
        # Naming the field here does not declare one: `answers` is aux data, so it is static
        # by the time this traces, and the solve below keeps the state or does not according
        # to what was declared. Checked before the trace rather than inside it.
        self._require_field_declaration()

        [setattr(self, k, v) for k, v in kwargs.items()]  # needed for optimization

        res = self.solve(geometry=geometry)
        field_cell = self.reconstruct_field(sample_x, sample_y, sample_z, incidences)
        return res, field_cell

    @jax_device_set
    def solve_field_eager(self, sample_x=20, sample_y=20, sample_z=20,
                          incidences=('given',), geometry=None):
        """Eager solve-and-field. ``geometry`` is optional: omit it and each discrete decision
        is auto-detected from the concrete structure."""
        self._require_field_declaration()
        res = self.solve(geometry=geometry)
        field_cell = self.reconstruct_field(sample_x, sample_y, sample_z, incidences)
        return res, field_cell

    def run_structure_vmap(self, structure_list):
        """
        under development
        """

        R, T = jax.vmap(self.solve)(structure_list)

        return R, T

    def run_structure_pmap(self, structure_list):
        """
        under development
        """

        R, T = jax.pmap(self.solve)(structure_list)

        return jnp.array(R), jnp.array(T)
