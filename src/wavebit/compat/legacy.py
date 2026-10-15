def _enable_jax_x64_if_already_imported():
    """Lift JAX's 32-bit cap now, if the caller has JAX loaded.

    Each door onto JAX lifts the cap as it opens -- ``src/wavebit/rcwa/_backends/jax_backend.py`` when the
    RCWA backend is first *used*, ``JaxTimeLoop.__init__`` when an FDTD run first builds its
    time loop; :func:`wavebit._common.backend.lift_jax_x64` lists them all. That was chosen so
    the process-wide setting is not touched on behalf of a caller who never asks for JAX, and
    that half still holds — this does not import JAX, it only acts when JAX is already there.

    What did not hold is the other half. The setting gets flipped either way, so deferring it
    does not spare the caller the side effect; it only makes the *moment* depend on import and
    call order, and an array built in the gap is silently single precision. Same code, float32
    or float64 depending on where it sat relative to the first solve. That produced a
    ``|R| = 3.94`` against a true 0.67 with every dtype reading complex128 — see
    ``docs/theory/rcwa/branch-cuts.md``.

    Note this lifts a *cap* rather than forcing a choice: with x64 off, ``jnp.asarray(x,
    jnp.float64)`` still yields float32, so double precision cannot be opted into at all; with
    it on, ``jnp.asarray(x, jnp.float32)`` still yields float32, and ``complex_dtype`` continues
    to select the solve's precision either way. The cost is that JAX's *default* dtype becomes
    64-bit for the caller's own code too, which is why this stays limited to callers who have
    already imported JAX, and why `PrecisionLossWarning` still covers the ordering this cannot
    reach (``from wavebit.compat import legacy`` before ``import jax``).

    The lift itself is the shared body, :func:`wavebit._common.backend.lift_jax_x64`, so a door
    that forgets it is a missing call rather than a copy that drifted. This is the one caller
    that tolerates failure: JAX may be absent, or mid-import, and that is not an error here.
    """
    from wavebit._common.backend import lift_jax_x64
    lift_jax_x64()


_enable_jax_x64_if_already_imported()

from wavebit._common.backend import eager_under_torch_compile as _eager_under_torch_compile  # noqa: E402


@_eager_under_torch_compile
def enable_x64():
    """Lift JAX's 32-bit cap now, so arrays built from here on can hold double precision.

    Call this before building any JAX array wavebit will be handed, when ``import jax`` came
    after importing ``wavebit.compat.legacy``::

        from wavebit.compat import legacy as wavebit
        wavebit.enable_x64()
        import jax.numpy as jnp
        theta = jnp.asarray(0.4 + 0.3j)                 # complex128, not complex64

    Without it that array is complex64, and the setter's widening cannot recover the digits
    already gone: a solve reporting ``complex_dtype=0`` everywhere ran with ~8 significant
    digits missing from its incidence angle and returned ``|R| = 3.94`` against a true 0.67.
    See :class:`~wavebit._common.backend.PrecisionLossWarning`, which is what says so at runtime,
    and ``docs/theory/rcwa/branch-cuts.md`` for the case.

    Other things already lift it, and this exists because none of them covers every ordering.
    Importing ``wavebit.compat.legacy`` lifts it if JAX is *already* loaded; the legacy JAX RCWA backend lifts it when
    it is first used, and the JAX time loop lifts it when an FDTD run first builds one. The gap
    is JAX arriving after wavebit and being used before the first solve, which is exactly the
    ordering above.

    It lifts a **cap**, not a choice. With x64 off, ``jnp.asarray(x, jnp.float64)`` still gives
    float32, so double precision cannot be opted into at all; with it on, ``jnp.asarray(x,
    jnp.float32)`` still gives float32, and ``complex_dtype`` continues to select the solve's
    own precision either way. The cost is that JAX's default dtype becomes 64-bit for the
    caller's own code too, which is why nothing does this on plain ``import wavebit``.

    Idempotent, and it does not import JAX unless JAX is installed -- but it does raise if it
    is not, rather than quietly doing nothing. Asking for double precision and silently not
    getting it is the failure this whole path exists to stop.

    Being the call for JAX arriving after wavebit, it also registers :class:`~wavebit.compat.legacy.Port`
    as a pytree, so a scattering matrix built before ``import jax`` crosses ``jax.jit`` (see
    ``wavebit.rcwa._engine.scattering._register_pytree_if_jax_loaded``).
    """
    import importlib.util

    if importlib.util.find_spec('jax') is None:
        raise ImportError(
            'enable_x64() lifts JAX\'s 32-bit cap, and JAX is not installed. '
            'Install it (`pip install jax`) or drop the call -- NumPy and PyTorch have no '
            'such cap, and `complex_dtype` selects the solve precision on every backend.')

    importlib.import_module('jax')
    from wavebit._common.backend import lift_jax_x64
    lift_jax_x64(required=True)
    import wavebit.rcwa._engine.scattering as scattering
    scattering._register_pytree_if_jax_loaded()

from wavebit.rcwa._factory import rcwa
from wavebit.fdtd._periodic.api import fdtd
from wavebit.fdtd._periodic.legacy import UnsettledFieldWarning
from wavebit.fdtd._periodic.legacy import permittivity_at
from wavebit._common.backend import AutoDetuneWarning
from wavebit._common.backend import AxisNameWarning
from wavebit._common.backend import ComplexIncidenceWarning
from wavebit._common.backend import ExceptionalPointWarning
from wavebit._common.backend import GainCutoffWarning
from wavebit._common.backend import PrecisionLossWarning
from wavebit._common.backend import RayleighPointWarning
from wavebit._common.backend import UnresolvedBranchWarning
from wavebit._common.backend import get_j1_implementation
from wavebit._common.backend import set_j1_implementation
from wavebit.rcwa.plotting import plot_structure
from wavebit.rcwa.plotting import slice_structure
from wavebit.fdtd._periodic.plotting import animate_fdtd_field
from wavebit.fdtd._periodic.plotting import plot_fdtd_field
from wavebit.fdtd._periodic.probes import fdtd_poynting
from wavebit.fdtd._periodic.probes import fit_fdtd_modes
from wavebit.fdtd._periodic.probes import sample_fdtd_fields
from wavebit.fdtd._periodic.grid import fdtd_grid
from wavebit.rcwa._engine.enums import Answers
from wavebit._common.enums import Backend
from wavebit._common.enums import Tetm
from wavebit.rcwa._engine.enums import FourierCoefficients
from wavebit.rcwa._engine.enums import Factorization
from wavebit.rcwa._engine.enums import Formulation
from wavebit.rcwa._engine.enums import Nesting
from wavebit.rcwa._engine.enums import NormalMethod
from wavebit.rcwa._engine.enums import Projector
from wavebit._common.periodic import Layer
from wavebit._common.periodic import Stack
from wavebit._common.periodic import Incidence
from wavebit.materials.response import Material
from wavebit.materials.response import eps_from_refractive_index
from wavebit.materials.response import refractive_index_from_eps
from wavebit._common.periodic import layer_thicknesses
from wavebit.materials.tabulated import TabulatedMaterial
# The type `eps_conv_matrices()` hands back and `solve_conv()` takes: a NamedTuple, so the five
# slots still splat and unpack, but `conv.epz_i` reaches one by name.
from wavebit.rcwa._engine.factorization import ConvMatrices
from wavebit.rcwa._engine.anisotropy import TensorConvMatrices
from wavebit.rcwa._engine.conforming import ConformingNVF
from wavebit.rcwa._engine.conforming.affine_profile import AffineProfileNVF
from wavebit.rcwa._engine.conforming.cartesian_rectangle import CartesianRectangleNVF
from wavebit.rcwa._engine.conforming.cartesian_pair import CartesianRectanglePairNVF
# Explicit experimental auxiliary construction; ordinary NVF stays separate.
from wavebit.rcwa._engine.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
from wavebit.rcwa._engine.conforming.cartesian_pair_transmission_galerkin import CartesianPairTransmissionGalerkinGapCell
# `Lattice` is public because the constructors live on it -- `Lattice.hexagonal(400.)` says
# what it returns where `hexagonal(400.)` did not, and a hexagon is a plausible *shape* in a
# package whose shape types are already `rectangle` / `circle` / `ellipse`. It is also the type
# `solver.lattice` hands back and the one `repr` prints, so it was reachable and unnameable:
# no import, no annotation, and since 2026-09-01 a public `.meshgrid_xy` method on a type callers
# could not spell.
from wavebit.lattice import Lattice
from wavebit.lattice import meshgrid_xy
from wavebit.rcwa._engine.geometry import StaticGeometry
from wavebit.rcwa._engine.geometry import StaleGeometryError
# What `solver.scattering` hands back after an SMM solve: the stack's own scattering matrix,
# both sides and every order, incidence-relative, with the two half-spaces as its ports.
# `None` on the ETM, which forms no such operator. `Port` rides along because
# `ScatteringMatrix.cascade` joins two of them and a join across mismatched ports is silently
# wrong -- so the basis travels with the numbers and the join refuses rather than trusting
# the caller. `cascade` is a method rather than a function here: the receiver already names
# the type, the order is directional, and it chains.
from wavebit.rcwa._engine.scattering import Port
from wavebit.rcwa._engine.scattering import ScatteringMatrix
from wavebit.rcwa._engine.sources import Source
from wavebit.rcwa._engine.sources import PlaneWave
from wavebit.rcwa._engine.sources import GaussianBeam
from wavebit.rcwa._engine.sources import ElectricDipole
from wavebit.rcwa._engine.sources import OrderExcitation
from wavebit.rcwa._engine.sources import Beam
from wavebit.rcwa._engine.illumination import BoundaryPower
from wavebit.rcwa._engine.illumination import CoherentResult
from wavebit.rcwa._engine.illumination import CoherentSolver
from wavebit import materials
from wavebit.compat._jobs_v1 import JobSpec
from wavebit.compat._jobs_v1 import Numerics
from wavebit.compat._jobs_v1 import Sweep
from wavebit.compat._jobs_v1 import SpecError
from wavebit.compat._jobs_v1 import run

# The public surface -- and the reason this file needs no F401 exemption. Every name above is
# bound here and used nowhere here, which is indistinguishable from an unused import until
# `__all__` says otherwise; ruff reads a listing here as the use. Keeping the surface in the
# file it belongs to, rather than as a blanket ignore in `pyproject.toml`, also keeps F401
# pointed at it: an export added without a line here fails the lint instead of being waved
# through. Grouped to match the import blocks above.
#
# `main` is deliberately absent. It is bound as a side effect of the import above, not
# offered, and without `__all__` a star import collected it along with the rest.
__all__ = [
    'Lattice',
    'meshgrid_xy',
    'rcwa',
    # The time-domain solve beside it -- roadmap item 2. A function, not a solver: see
    # its docstring in `main.py` for why there is no per-instance state to hold.
    'fdtd', 'fd',
    # Defined in this file rather than imported -- see `enable_x64` above.
    'enable_x64',
    'AutoDetuneWarning',
    'AxisNameWarning', 'ComplexIncidenceWarning', 'ExceptionalPointWarning', 'GainCutoffWarning',
    'PrecisionLossWarning',
    'RayleighPointWarning',
    'UnresolvedBranchWarning',
    'UnsettledFieldWarning', 'permittivity_at',
    'get_j1_implementation', 'set_j1_implementation',
    'plot_structure', 'slice_structure',
    'plot_fdtd_field', 'animate_fdtd_field',
    'sample_fdtd_fields', 'fdtd_poynting', 'fit_fdtd_modes', 'fdtd_grid',
    'Answers', 'Backend', 'Tetm', 'FourierCoefficients', 'Factorization', 'Formulation',
    'NormalMethod', 'Projector', 'Nesting',
    'Layer', 'Stack', 'Incidence', 'Material', 'TabulatedMaterial', 'eps_from_refractive_index', 'refractive_index_from_eps',
    'ConvMatrices', 'TensorConvMatrices', 'ConformingNVF', 'AffineProfileNVF', 'CartesianRectangleNVF', 'CartesianRectanglePairNVF',
    'CartesianPairOpeningPlan', 'CartesianPairTransmissionGalerkinGapCell',
    'ScatteringMatrix', 'Port',
    'Source', 'PlaneWave', 'GaussianBeam', 'ElectricDipole', 'OrderExcitation',
    'Beam', 'BoundaryPower', 'CoherentResult', 'CoherentSolver',
    'layer_thicknesses',
    'StaticGeometry', 'StaleGeometryError',
    'materials',
    'JobSpec', 'Numerics', 'Sweep', 'SpecError', 'run',
]

# The package's number, not one of its own: this module is the old flat namespace, and its
# copy stayed at 0.1.0 through the 0.0.x releases after it moved here.
import wavebit as _package  # noqa: E402
__version__ = _package.__version__

from wavebit.fdtd import _legacy as fd
