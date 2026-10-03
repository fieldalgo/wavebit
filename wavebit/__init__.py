def _enable_jax_x64_if_already_imported():
    """Lift JAX's 32-bit cap now, if the caller has JAX loaded.

    ``wavebit/backends/jax_backend.py`` enables ``jax_enable_x64`` when the JAX backend is first
    *used*. That was chosen so the process-wide setting is not touched on behalf of a caller
    who never asks for JAX, and that half still holds — this does not import JAX, it only acts
    when JAX is already there.

    What did not hold is the other half. The setting gets flipped either way, so deferring it
    does not spare the caller the side effect; it only makes the *moment* depend on import and
    call order, and an array built in the gap is silently single precision. Same code, float32
    or float64 depending on where it sat relative to the first solve. That produced a
    ``|R| = 3.94`` against a true 0.67 with every dtype reading complex128 — see
    ``docs/branch-cuts.md``.

    Note this lifts a *cap* rather than forcing a choice: with x64 off, ``jnp.asarray(x,
    jnp.float64)`` still yields float32, so double precision cannot be opted into at all; with
    it on, ``jnp.asarray(x, jnp.float32)`` still yields float32, and ``complex_dtype`` continues
    to select the solve's precision either way. The cost is that JAX's *default* dtype becomes
    64-bit for the caller's own code too, which is why this stays limited to callers who have
    already imported JAX, and why `PrecisionLossWarning` still covers the ordering this cannot
    reach (``import wavebit`` before ``import jax``).
    """
    import sys
    jax = sys.modules.get('jax')
    try:
        jax.config.update('jax_enable_x64', True)
    except Exception:                 # absent, or mid-import and not yet usable
        pass


_enable_jax_x64_if_already_imported()


def enable_x64():
    """Lift JAX's 32-bit cap now, so arrays built from here on can hold double precision.

    Call this before building any JAX array wavebit will be handed, when ``import jax`` came
    after ``import wavebit``::

        import wavebit
        wavebit.enable_x64()
        import jax.numpy as jnp
        theta = jnp.asarray(0.4 + 0.3j)                 # complex128, not complex64

    Without it that array is complex64, and the setter's widening cannot recover the digits
    already gone: a solve reporting ``complex_dtype=0`` everywhere ran with ~8 significant
    digits missing from its incidence angle and returned ``|R| = 3.94`` against a true 0.67.
    See :class:`~wavebit._core.backend.PrecisionLossWarning`, which is what says so at runtime,
    and ``docs/branch-cuts.md`` for the case.

    Two other things already lift it, and this exists because neither covers every ordering.
    ``import wavebit`` lifts it if JAX is *already* loaded, and the JAX backend lifts it when it
    is first used. The gap is JAX arriving after wavebit and being used before the first solve,
    which is exactly the ordering above.

    It lifts a **cap**, not a choice. With x64 off, ``jnp.asarray(x, jnp.float64)`` still gives
    float32, so double precision cannot be opted into at all; with it on, ``jnp.asarray(x,
    jnp.float32)`` still gives float32, and ``complex_dtype`` continues to select the solve's
    own precision either way. The cost is that JAX's default dtype becomes 64-bit for the
    caller's own code too, which is why nothing does this on plain ``import wavebit``.

    Idempotent, and it does not import JAX unless JAX is installed -- but it does raise if it
    is not, rather than quietly doing nothing. Asking for double precision and silently not
    getting it is the failure this whole path exists to stop.

    Being the call for JAX arriving after wavebit, it also registers :class:`~wavebit.Port`
    as a pytree, so a scattering matrix built before ``import jax`` crosses ``jax.jit`` (see
    ``_core.scattering._register_pytree_if_jax_loaded``).
    """
    import importlib.util

    if importlib.util.find_spec('jax') is None:
        raise ImportError(
            'enable_x64() lifts JAX\'s 32-bit cap, and JAX is not installed. '
            'Install it (`pip install jax`) or drop the call -- NumPy and PyTorch have no '
            'such cap, and `complex_dtype` selects the solve precision on every backend.')

    import jax
    jax.config.update('jax_enable_x64', True)
    from ._core import scattering
    scattering._register_pytree_if_jax_loaded()

from .main import rcwa
from ._core.backend import (AutoDetuneWarning, AxisNameWarning, ComplexIncidenceWarning,
                            PrecisionLossWarning, RayleighPointWarning,
                            UnresolvedBranchWarning,
                            get_j1_implementation, set_j1_implementation)
from ._core.draw import plot_structure, slice_structure
from ._core.enums import (Answers, Backend, Tetm, FourierCoefficients, Factorization,
                          Formulation, Nesting, NormalMethod, Projector)
from ._core.domain import (Layer, Stack, Incidence, Material,
                           eps_from_index, index_from_eps)
from ._core.materials import TabulatedMaterial
# The type `eps_conv_matrices()` hands back and `solve_conv()` takes: a NamedTuple, so the five
# slots still splat and unpack, but `conv.epz_i` reaches one by name.
from ._core.factorization import ConvMatrices
from ._core.anisotropy import TensorConvMatrices
from ._core.conforming import ConformingNVF
from ._core.conforming.affine_profile import AffineProfileNVF
from ._core.conforming.cartesian_rectangle import CartesianRectangleNVF
from ._core.conforming.cartesian_pair import CartesianRectanglePairNVF
# Explicit experimental auxiliary construction; ordinary NVF stays separate.
from ._core.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
from ._core.conforming.cartesian_pair_transmission_galerkin import CartesianPairTransmissionGalerkinGapCell
# `Lattice` is public because the constructors live on it -- `Lattice.hexagonal(400.)` says
# what it returns where `hexagonal(400.)` did not, and a hexagon is a plausible *shape* in a
# package whose shape types are already `rectangle` / `circle` / `ellipse`. It is also the type
# `solver.lattice` hands back and the one `repr` prints, so it was reachable and unnameable:
# no import, no annotation, and since 2026-09-01 a public `.meshgrid_xy` method on a type callers
# could not spell.
from ._core.lattice import Lattice, meshgrid_xy
from ._core.geometry import StaticGeometry, StaleGeometryError
# What `solver.scattering` hands back after an SMM solve: the stack's own scattering matrix,
# both sides and every order, incidence-relative, with the two half-spaces as its ports.
# `None` on the ETM, which forms no such operator. `Port` rides along because
# `ScatteringMatrix.cascade` joins two of them and a join across mismatched ports is silently
# wrong -- so the basis travels with the numbers and the join refuses rather than trusting
# the caller. `cascade` is a method rather than a function here: the receiver already names
# the type, the order is directional, and it chains.
from ._core.scattering import Port, ScatteringMatrix
from ._core.sources import Source, PlaneWave, GaussianBeam, ElectricDipole, OrderExcitation, Beam
from ._core.illumination import BoundaryPower, CoherentResult, CoherentSolver
from . import materials
from ._core.spec import JobSpec, Numerics, Sweep, SpecError, run

# The public surface -- and the reason this file needs no F401 exemption. Every name above is
# bound here and used nowhere here, which is indistinguishable from an unused import until
# `__all__` says otherwise; ruff reads a listing here as the use. Keeping the surface in the
# file it belongs to, rather than as a blanket ignore in `pyproject.toml`, also keeps F401
# pointed at it: an export added without a line here fails the lint instead of being waved
# through. Grouped to match the import blocks above.
#
# `main` is deliberately absent. It is bound as a side effect of `from .main import rcwa`, not
# offered, and without `__all__` a star import collected it along with the rest.
__all__ = [
    'Lattice',
    'meshgrid_xy',
    'rcwa',
    # Defined in this file rather than imported -- see `enable_x64` above.
    'enable_x64',
    'AutoDetuneWarning',
    'AxisNameWarning', 'ComplexIncidenceWarning', 'PrecisionLossWarning', 'RayleighPointWarning',
    'UnresolvedBranchWarning',
    'get_j1_implementation', 'set_j1_implementation',
    'plot_structure', 'slice_structure',
    'Answers', 'Backend', 'Tetm', 'FourierCoefficients', 'Factorization', 'Formulation',
    'NormalMethod', 'Projector', 'Nesting',
    'Layer', 'Stack', 'Incidence', 'Material', 'TabulatedMaterial', 'eps_from_index', 'index_from_eps',
    'ConvMatrices', 'TensorConvMatrices', 'ConformingNVF', 'AffineProfileNVF', 'CartesianRectangleNVF', 'CartesianRectanglePairNVF',
    'CartesianPairOpeningPlan', 'CartesianPairTransmissionGalerkinGapCell',
    'ScatteringMatrix', 'Port',
    'Source', 'PlaneWave', 'GaussianBeam', 'ElectricDipole', 'OrderExcitation',
    'Beam', 'BoundaryPower', 'CoherentResult', 'CoherentSolver',
    'StaticGeometry', 'StaleGeometryError',
    'materials',
    'JobSpec', 'Numerics', 'Sweep', 'SpecError', 'run',
]

__version__ = "0.1.0"
