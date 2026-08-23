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

from .main import rcwa
from ._core.backend import (AutoDetuneWarning, AxisNameWarning, PrecisionLossWarning, RayleighPointWarning,
                            UnphysicalResultWarning, UnresolvedBranchWarning,
                            get_j1_implementation, set_j1_implementation)
from ._core.conventions import branch_tolerance
from ._core.draw import plot_structure, slice_structure
from ._core.enums import (Answers, Backend, Tetm, FourierCoefficients, Factorization,
                          Formulation, NormalMethod)
from ._core.domain import (Layer, Stack, Incidence, Material,
                           eps_from_index, index_from_eps)
# The type `eps_conv_matrices()` hands back and `solve_conv()` takes: a NamedTuple, so the five
# slots still splat and unpack, but `conv.epz_i` reaches one by name.
from ._core.factorization import ConvMatrices
from ._core.lattice import grid, hexagonal, oblique, rectangular
from ._core.geometry import StaticGeometry, StaleGeometryError
# What `solver.scattering` hands back after an SMM solve: the stack's own scattering matrix,
# both sides and every order, incidence-relative, with the two half-spaces as its ports.
# `None` on the ETM, which forms no such operator. `Port` rides along because
# `ScatteringMatrix.cascade` joins two of them and a join across mismatched ports is silently
# wrong -- so the basis travels with the numbers and the join refuses rather than trusting
# the caller. `cascade` is a method rather than a function here: the receiver already names
# the type, the order is directional, and it chains.
from ._core.scattering import Port, ScatteringMatrix
# The nk front door. Exported here rather than left importable from `_core.modeler` because
# the bundled `nk_data` ships in the wheel and these two are its only readers -- data with no
# public reader is the same dead end as a keyword with no reader. They hand back an *index*,
# so `eps_from_index` above is the step between them and a Material.
from ._core.modeler import find_nk_index, read_material_table
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
    'grid',
    'hexagonal',
    'oblique',
    'rectangular',
    'rcwa',
    'AutoDetuneWarning',
    'AxisNameWarning', 'PrecisionLossWarning', 'RayleighPointWarning',
    'UnphysicalResultWarning', 'UnresolvedBranchWarning',
    'get_j1_implementation', 'set_j1_implementation',
    'branch_tolerance',
    'plot_structure', 'slice_structure',
    'Answers', 'Backend', 'Tetm', 'FourierCoefficients', 'Factorization', 'Formulation',
    'NormalMethod',
    'Layer', 'Stack', 'Incidence', 'Material', 'eps_from_index', 'index_from_eps',
    'ConvMatrices', 'ScatteringMatrix', 'Port',
    'StaticGeometry', 'StaleGeometryError',
    'find_nk_index', 'read_material_table',
    'JobSpec', 'Numerics', 'Sweep', 'SpecError', 'run',
]

__version__ = "0.1.0"
