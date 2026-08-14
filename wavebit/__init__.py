def _enable_jax_x64_if_already_imported():
    """Lift JAX's 32-bit cap now, if the caller has JAX loaded.

    ``wavebit/on_jax/__init__.py`` enables ``jax_enable_x64`` when the JAX backend is first
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
    it on, ``jnp.asarray(x, jnp.float32)`` still yields float32, and ``type_complex`` continues
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

from .main import rcwa
from ._core.backend import (PrecisionLossWarning, UnphysicalResultWarning,
                            UnresolvedBranchWarning, get_j1_implementation,
                            set_j1_implementation)
from ._core.conventions import branch_tolerance
from ._core.enums import Backend, Pol, FourierCoefficients, Factorization, Formulation
from ._core.domain import Layer, Stack, Incidence
from ._core.geometry import StaticGeometry, StaleGeometryError
from ._core.spec import JobSpec, Numerics, Sweep, SpecError, run

__version__ = "0.1.0"
