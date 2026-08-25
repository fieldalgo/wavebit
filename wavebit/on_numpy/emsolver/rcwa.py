import numpy as np

from ._base import _BaseRCWA
from ..._core.engine import CoreFactorizationMixin, _refuse_retired
from ..._core.modeler import Modeling


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
