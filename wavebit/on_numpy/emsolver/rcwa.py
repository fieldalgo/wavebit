import numpy as np

from ._base import _BaseRCWA
from ..._core.domain import retired_material_name
from ..._core.engine import CoreFactorizationMixin, _refuse_retired
from ..._core.enums import Formulation
from ..._core.fields import field_1d, field_1d_conical, field_2d, field_2d_4ff, field_plot
from ..._core.result import Result, ResultSub

# One shared container (wavebit._core.result); the old per-backend copies are gone.
ResultNumpy = Result
ResultSubNumpy = ResultSub


from ..._core.modeler import Modeling


class RCWANumpy(CoreFactorizationMixin, _BaseRCWA):
    def modeling_block_instruction(self, *args, **kwargs):
        return self.modeler.modeling_block_instruction(*args, **kwargs)

    def rectangle(self, *args, **kwargs):
        return self.modeler.rectangle(*args, **kwargs)

    def ellipse(self, *args, **kwargs):
        return self.modeler.ellipse(*args, **kwargs)

    def run_structure_vmap(self, structure_list):
        """
        under development
        """
        pass

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
                 pol=None,   # retired; kept only to say so
                 fto=(0, 0),
                 connecting_algo='ETM',
                 type_complex=np.complex128,
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
                 # 'efficiency' | 'amplitude' (default) | 'field'. A declaration, like
                 # tetm -- calculate_field is refused below 'field' rather than served,
                 # because the per-layer state it walks is 4.25 matrices of side
                 # 2*ff_x*ff_y per layer (240 MB at fto=(15, 15)) and is kept as the
                 # solve runs or not at all. See CoreSolveMixin.answers.
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
                         fto=fto, period=period, wavelength=wavelength,
                         thickness=thickness, connecting_algo=connecting_algo,
                         type_complex=type_complex,
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

        if isinstance(structure, np.ndarray):  # Raster
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
        if not isinstance(grid, np.ndarray):
            raise ValueError(
                f'{where}: expected a NumPy array on this backend, got '
                f'{type(grid).__name__}')
        if grid.dtype in (np.int64, np.float64, np.int32, np.float32):
            dtype = self.type_float
        elif grid.dtype in (np.complex128, np.complex64):
            dtype = self.type_complex
        else:
            raise ValueError
        return grid.astype(dtype)

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
                self._grating_type_assigned = 2  # else

        elif self.modeling_type_assigned == 1:  # Block
            self.grating_type_assigned = 2  # TODO: 1d conical case

        elif self.modeling_type_assigned == 2:  # Analytic (shape-based, staircase-free)
            self._grating_type_assigned = 2

    @property
    def grating_type_assigned(self):
        return self._grating_type_assigned

    @grating_type_assigned.setter
    def grating_type_assigned(self, grating_type_assigned):
        self._grating_type_assigned = grating_type_assigned

    def solve_for_conv(self, wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all,
                       epxy_conv_all=None, epyx_conv_all=None,
                       epxz_conv_all=None, epyz_conv_all=None,
                       epzx_conv_all=None, epzy_conv_all=None, mu=None,
                       mu_top=None, mu_bot=None, eps_top=None, eps_bot=None):
        self._assign_grating_type()
        z_row = (epxz_conv_all, epyz_conv_all, epzx_conv_all, epzy_conv_all)
        if self._formulation_for(z_row) == Formulation.FIRST:
            # The 1D paths have no z-row slots to fill, so a z row is also a choice of path:
            # the first-order solve is written on the 2D layout (a 1D structure is ff_y == 1
            # there, the same physics the conical path solves at conical cost).
            self._grating_type_assigned = 2

        # "Was a permeability named?", the structural question `Material.is_magnetic` answers,
        # rather than "does it differ from one?". `None` means no half-space permeability was
        # given; any value, 1.0 included, means one was. NumPy could have kept the old value
        # comparison, but the three backends have to agree on which scenes route where -- and
        # the JAX copy of this line cannot read a value at all, since a named `mu` is a tracer
        # there. See the note in on_jax/emsolver/rcwa.py.
        magnetic = ((mu is not None and any(m is not None for m in mu))
                    or mu_top is not None or mu_bot is not None)
        if magnetic and self._grating_type_assigned != 2:
            raise NotImplementedError(
                'mu is implemented on the 2D path only; this structure routed to a 1D solve. '
                'The 1D and conical operators carry no permeability slot at all.')

        if self.grating_type_assigned == 0:
            result_dict = self._solve_1d(wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all)
        elif self._grating_type_assigned == 1:
            result_dict = self._solve_1d_conical(wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all)
        else:
            result_dict = self._solve_2d(wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all,
                                        epxy_conv_all, epyx_conv_all, *z_row, mu=mu,
                                        mu_top=mu_top, mu_bot=mu_bot,
                                        eps_top=eps_top, eps_bot=eps_bot)

        res_psi = ResultSubNumpy(**result_dict['res']) if 'res' in result_dict else None
        res_te_inc = ResultSubNumpy(**result_dict['res_te_inc']) if 'res_te_inc' in result_dict else None
        res_tm_inc = ResultSubNumpy(**result_dict['res_tm_inc']) if 'res_tm_inc' in result_dict else None

        result = ResultNumpy(res_psi, res_te_inc, res_tm_inc, fto=self.fto)

        return result

    def calculate_field(self, res_x=20, res_y=20, res_z=20, set_field_input=(True, False, False)):
        """Reconstruct the field from the state left behind by the last solve.

        ``res_x`` / ``res_y`` / ``res_z`` each take either a **count** -- a uniform grid, the
        long-standing behaviour and unchanged to the bit -- or the **coordinates themselves**,
        as a 1-D array. The reconstruction is pointwise in all three, so explicit points cost
        the same per point as grid ones and involve no interpolation.

        ``res_x`` / ``res_y`` are lengths in ``period``'s units and are unconstrained.
        ``res_z`` is absolute, measured from the top of the stack, must lie inside it, and
        assigns an interface to the layer below it -- which is visible only in ``E_z``, the one
        component genuinely two-valued there. With explicit ``res_z`` the returned z axis is
        exactly the request, in the caller's order, rather than ``res_z * n_layers``.
        """
        self._require_field_declaration()
        kx, ky = self._kxy_2d(self.wavelength)

        if self._grating_type_assigned == 0:
            res_y = 1
            field_cell = field_1d(self.wavelength, kx, self.T1, self.layer_info_list, self.period, self.tetm_index,
                                       res_x=res_x, res_y=res_y, res_z=res_z)
        elif self._grating_type_assigned == 1:
            # TODO other bds
            field_cell = field_1d_conical(self.wavelength, kx, ky, self.T1, self.layer_info_list, self.period,
                                               res_x=res_x, res_y=res_y, res_z=res_z, set_field_input=set_field_input,
                                       lattice=self.lattice)
        elif self._formulation_assigned == Formulation.FIRST:
            # The grating type cannot answer this: the first-order path sets it to 2 as well,
            # so branching on it alone would hand a 4ff solve to the 2ff reconstruction, which
            # reads the same state slots and returns a plausible, wrong cell.
            field_cell = field_2d_4ff(self.wavelength, kx, ky, self.T1, self.layer_info_list, self.period,
                                      res_x=res_x, res_y=res_y, res_z=res_z, set_field_input=set_field_input,
                                      lattice=self.lattice)
        else:
            field_cell = field_2d(self.wavelength, kx, ky, self.T1, self.layer_info_list, self.period,
                                       res_x=res_x, res_y=res_y, res_z=res_z, set_field_input=set_field_input,
                                       lattice=self.lattice)

        if self.incidence_side == 'bottom':
            # The backside solve leaves its state in the mirrored frame (z reversed); un-flip
            # once here so the returned cell is in SCENE coordinates for either side: index 0
            # of the z axis is the top of `structure[0]` regardless of where light came from.
            field_cell = np.flip(field_cell, axis=0 if self._grating_type_assigned == 0 else 1)

        return field_cell

    def conv_solve_field(self, res_x=20, res_y=20, res_z=20, set_field_input=(True, False, False),
                         geometry=None):
        """``geometry`` is optional here — NumPy is always eager, so auto-detection works. It is
        accepted so backend-agnostic code can pass the same frozen geometry JAX requires.

        Naming the field here does not declare one: ``answers='field'`` still has to be on the
        solver, because it is the *solve* below that either keeps the per-layer state or does
        not. Checked before that solve rather than after, so the refusal does not arrive on the
        far side of the expensive part."""
        self._require_field_declaration()
        res = self.solve(geometry=geometry)
        field_cell = self.calculate_field(res_x, res_y, res_z, set_field_input)
        return res, field_cell

    def field_plot(self, field_cell):
        field_plot(field_cell, self.tetm_index)
