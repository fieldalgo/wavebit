import jax

import jax.numpy as jnp

from functools import partial

from ..._core.engine import device_scoped
from ._base import _BaseRCWA
from ..._core.engine import CoreFactorizationMixin, _refuse_retired
from ..._core.result import Result, ResultSub
from ..._core.modeler import Modeling


@jax.tree_util.register_pytree_node_class
class ResultJax(Result):
    """Shared container, registered as a pytree so a jitted method can return one."""


@jax.tree_util.register_pytree_node_class
class ResultSubJax(ResultSub):
    """Shared container; a pytree for the same reason as :class:`ResultJax`."""


class RCWAJax(CoreFactorizationMixin, _BaseRCWA):
    #: The pytree-registered pair, so a jitted `solve_field` can return one. See the
    #: declaration in `CoreFactorizationMixin`.
    _RESULT = ResultJax
    _RESULT_SUB = ResultSubJax

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
                 truncation_order=(0, 0),
                 layer_algorithm='ETM',
                 excitation=None,   # which orders the illumination arrives on; see
                 # `CoreSolveMixin.excitation`. None is one wave on the specular order.
                 device='cpu',
                 complex_dtype=jnp.complex128,
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
                 # per-layer state it walks is 4.25 matrices of side 2*n_orders_u*n_orders_v per
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
                         device=device, complex_dtype=complex_dtype,
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

    @partial(jax.jit, static_argnums=(1, 2, 3, 4, 5),
             static_argnames=('sample_x', 'sample_y', 'sample_z', 'incidences', 'geometry'))
    @device_scoped
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

    @device_scoped
    def solve_field_eager(self, sample_x=20, sample_y=20, sample_z=20,
                          incidences=('given',), geometry=None):
        """Eager solve-and-field. ``geometry`` is optional: omit it and each discrete decision
        is auto-detected from the concrete structure."""
        self._require_field_declaration()
        res = self.solve(geometry=geometry)
        field_cell = self.reconstruct_field(sample_x, sample_y, sample_z, incidences)
        return res, field_cell
