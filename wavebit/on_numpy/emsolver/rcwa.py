import numpy as np

from ._base import _BaseRCWA
from ..._core.engine import CoreFactorizationMixin, _refuse_retired
from ..._core.enums import Formulation
from ..._core.fields import (field_1d, field_1d_conical, field_2d, field_2d_4ff,
                             is_count, to_mirrored_z, to_scene_frame)
from ..._core.modeler import Modeling
from ..._core.result import Result, ResultSub


class RCWANumpy(CoreFactorizationMixin, _BaseRCWA):
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
                 thickness=(0.,),
                 backend=0,
                 tetm=None,
                 truncation_order=(0, 0),
                 layer_algorithm='ETM',
                 excitation=None,   # which orders the illumination arrives on; see
                 # `CoreSolveMixin.excitation`. None is one wave on the specular order.
                 complex_dtype=np.complex128,
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
                 # layer (240 MB at truncation_order=(15, 15)) and is kept as the solve runs or not at
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

        super().__init__(top=top, bot=bot, theta=theta, phi=phi, psi=psi, chi=chi, tetm=tetm,
                         truncation_order=truncation_order, period=period, wavelength=wavelength,
                         thickness=thickness, layer_algorithm=layer_algorithm,
                         excitation=excitation,
                         complex_dtype=complex_dtype,
                         polarization=polarization)
        self.incidence_side = incidence_side
        self.answers = answers

        self.modeler = Modeling(period=self.period, complex_dtype=complex_dtype)

        self._modeling_type = None
        self._grating_type = None
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
        :attr:`modeling_type`, which is the only thing that selects the modeling
        path -- there is no separate mode switch.
        """
        return self._structure

    @structure.setter
    def structure(self, structure):

        if isinstance(structure, np.ndarray):  # Raster
            self._modeling_type = 0
            self._structure = self._coerce_raster('structure', structure)

        elif type(structure) is list:
            # Analytic: one dict(background=..., shapes=[...], combine=...) per layer.
            # Shape coordinates are absolute lengths in the units of `period`, not
            # fractions of it, and `combine` picks what overlapping shapes mean --
            # this path adds them where the block path paints. See
            # wavebit._core.analytic; overlaps are refused until it is set.
            if len(structure) > 0 and isinstance(structure[0], dict):
                self._modeling_type = 2
                self._structure = structure
            else:  # Block
                self._modeling_type = 1
                self._structure = structure
        elif structure is None:
            self._structure = structure
            self._modeling_type = None
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
        if not isinstance(grid, np.ndarray):
            raise ValueError(
                f'{where}: expected a NumPy array on this backend, got '
                f'{type(grid).__name__}')
        if grid.dtype in (np.int64, np.float64, np.int32, np.float32):
            dtype = self.float_dtype
        elif grid.dtype in (np.complex128, np.complex64):
            dtype = self.complex_dtype
        else:
            raise ValueError
        return grid.astype(dtype)

    def solve_conv(self, epx_conv_all, epy_conv_all, epz_conv_i_all,
                   epxy_conv_all=None, epyx_conv_all=None,
                   epxz_conv_all=None, epyz_conv_all=None,
                   epzx_conv_all=None, epzy_conv_all=None, mu=None,
                   mu_top=None, mu_bot=None, eps_top=None, eps_bot=None):
        """Solve a stack whose convolution matrices are already built.

        The nine permittivity slots in :meth:`eps_conv_matrices` order -- the in-plane ``(epx,
        epy, epz_i, epxy, epyx)``, then the z row ``(epxz, epyz, epzx, epzy)`` -- followed by
        ``mu`` and the half-space terms. ``solver.solve_conv(*solver.eps_conv_matrices())`` is
        ``solver.solve()``; see :mod:`wavebit._core.factorization` for why ``epz_i`` sits third
        rather than last.

        Takes no wavelength. The solver already has one, and a second spelling of it was a
        second thing that could disagree: the k vectors, the field reconstruction and the
        auto-detune ladder all read ``self.wavelength``, so an argument that differed would
        have answered one problem while the rest of the object described another.
        """
        self._assign_grating_type()
        z_row = (epxz_conv_all, epyz_conv_all, epzx_conv_all, epzy_conv_all)
        if self._formulation_for(z_row) == Formulation.FOUR_FIELD:
            # The 1D paths have no z-row slots to fill, so a z row is also a choice of path:
            # the first-order solve is written on the 2D layout (a 1D structure is n_orders_y == 1
            # there, the same physics the conical path solves at conical cost).
            self._grating_type = 2

        # "Was a permeability named?", the structural question `Material.is_magnetic` answers,
        # rather than "does it differ from one?". `None` means no half-space permeability was
        # given; any value, 1.0 included, means one was. NumPy could have kept the old value
        # comparison, but the three backends have to agree on which scenes route where -- and
        # the JAX copy of this line cannot read a value at all, since a named `mu` is a tracer
        # there. See the note in on_jax/emsolver/rcwa.py.
        magnetic = ((mu is not None and any(m is not None for m in mu))
                    or mu_top is not None or mu_bot is not None)
        if magnetic and self._grating_type != 2:
            raise NotImplementedError(
                'mu is implemented on the 2D path only; this structure routed to a 1D solve. '
                'The 1D and conical operators carry no permeability slot at all.')

        if self._grating_type == 0:
            result_dict = self._solve_1d(epx_conv_all, epy_conv_all, epz_conv_i_all)
        elif self._grating_type == 1:
            result_dict = self._solve_1d_conical(epx_conv_all, epy_conv_all, epz_conv_i_all)
        else:
            result_dict = self._solve_2d(epx_conv_all, epy_conv_all, epz_conv_i_all,
                                         epxy_conv_all, epyx_conv_all, *z_row, mu=mu,
                                         mu_top=mu_top, mu_bot=mu_bot,
                                         eps_top=eps_top, eps_bot=eps_bot)

        given = ResultSub(**result_dict['given']) if 'given' in result_dict else None
        s_inc = ResultSub(**result_dict['s_inc']) if 's_inc' in result_dict else None
        p_inc = ResultSub(**result_dict['p_inc']) if 'p_inc' in result_dict else None

        result = Result(given, s_inc, p_inc, order_window=self.order_window)

        return result

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

        On the pure 1D TE/TM path only ``('given',)`` is answerable -- ``tetm`` declared the
        other degree of freedom away, so that solve carries one incidence and the cell it
        returns has no axis to put a second on. Naming ``'s'`` or ``'p'`` there is refused
        rather than ignored, which is what it was until 2026-08-20.
        """
        self._require_field_declaration()
        # An explicit z is stated in scene coordinates on both sides, so a backside request is
        # restated in the mirror the solve walks before anything buckets it.
        backside = self.incidence_side == 'bottom'
        counted_z = is_count(sample_z)
        if backside:
            sample_z = to_mirrored_z(sample_z, self.thickness)
        kx, ky = self.order_kxy()

        if self._grating_type == 0:
            sample_y = 1
            field_cell = field_1d(self.wavelength, kx, self.T1, self.layer_info_list,
                                  self.period, self.tetm_index,
                                  sample_x=sample_x, sample_y=sample_y, sample_z=sample_z,
                                  incidences=incidences)
        elif self._grating_type == 1:
            # TODO other bds
            field_cell = field_1d_conical(self.wavelength, kx, ky, self.T1,
                                          self.layer_info_list, self.period,
                                          sample_x=sample_x, sample_y=sample_y,
                                          sample_z=sample_z, incidences=incidences,
                                          lattice=self.lattice)
        elif self._formulation_assigned == Formulation.FOUR_FIELD:
            # The grating type cannot answer this: the first-order path sets it to 2 as well,
            # so branching on it alone would hand a 4ff solve to the 2ff reconstruction, which
            # reads the same state slots and returns a plausible, wrong cell.
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

        if backside:
            # The backside solve is the top-side solve of the z-mirrored scene, so its cell is
            # in that mirror -- axis and components both. `to_scene_frame` undoes both; see it
            # for why the component half is not optional, and `to_scene_frame`'s `flip_z` for
            # why an explicit z has no axis left to reverse. Static under jit either way:
            # `incidence_side` is aux data, never a tracer.
            field_cell = to_scene_frame(field_cell, pure_1d=self._grating_type == 0,
                                        tetm_index=self.tetm_index, flip_z=counted_z)

        return field_cell

    def solve_field(self, sample_x=20, sample_y=20, sample_z=20, incidences=('given',),
                    geometry=None):
        """``geometry`` is optional here — NumPy is always eager, so auto-detection works. It is
        accepted so backend-agnostic code can pass the same frozen geometry JAX requires.

        Naming the field here does not declare one: ``answers='field'`` still has to be on the
        solver, because it is the *solve* below that either keeps the per-layer state or does
        not. Checked before that solve rather than after, so the refusal does not arrive on the
        far side of the expensive part."""
        self._require_field_declaration()
        res = self.solve(geometry=geometry)
        field_cell = self.reconstruct_field(sample_x, sample_y, sample_z, incidences)
        return res, field_cell
