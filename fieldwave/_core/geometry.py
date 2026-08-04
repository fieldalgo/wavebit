"""Frozen geometry — the discrete decisions, taken once, so the rest can be traced.

Three steps in the pipeline decide something *structural* rather than compute a value:

===================  ===========================================================
same-material        is this layer uniform and identical to the medium below, so
                     the interface can be skipped?
normal field (NVF)   which way do the material interfaces point?
compression (CFS)    where do the piecewise-constant blocks start and stop?
===================  ===========================================================

All three branch on values, and the compression's output shape depends on them, so none can
run inside a JAX trace. They are also all properties of the *geometry*, not of the
permittivity riding on it — so the fix is to take them once, eagerly, and hold them fixed
while gradients flow through the material values.

That is what :class:`StaticGeometry` carries. Freeze it once from a concrete structure::

    solver = fieldwave.rcwa(backend=Backend.JAX, ucell=ucell, ...)
    geometry = solver.freeze_geometry()
    grad = jax.grad(lambda u: jnp.sum(solver.conv_solve(ucell=u, geometry=geometry).res.de_ti))

Without it, ``conv_solve`` still auto-detects — the eager path that reproduces the legacy
result exactly — but only ``fourier_type=DFS`` survives ``jax.grad`` (its detection narrows a
tracer whose primal is concrete), and nothing survives ``jax.jit``.
"""
from dataclasses import dataclass
from typing import Any, Optional

from .backend import array_namespace
from .solve_1d import _same_material, uniform_eps


@dataclass(eq=False)
class StaticGeometry:
    """The discrete geometry decisions of one structure, taken once.

    Attributes:
        same_material: per-layer flags for the interface short-circuit in the solve.
        normal_field: per-layer ``(nx, ny)`` for the NVF factorization.
        compression: per-layer ``(row_idx, col_idx, x, y)`` for the CFS factorization.

    A field left ``None`` means "auto-detect", which is what the strategy in use does not need
    (a DFS build never touches ``normal_field``, for instance).

    ``eq=False`` keeps identity-based ``__eq__``/``__hash__``, which is what lets a frozen
    geometry be passed as a *static* argument to a jitted method such as
    ``SolverJax.conv_solve_field``: it holds arrays, so value equality would be both ill-defined
    and unhashable, whereas one freeze reused across calls is exactly one compilation.
    """
    same_material: Optional[Any] = None
    normal_field: Optional[Any] = None
    compression: Optional[Any] = None


def detect_same_material(epx_conv_all, n_bot, rtol=1e-3):
    """Per-layer flags for the interface short-circuit, from concrete convolution matrices.

    A uniform layer of the medium below has ``eps_conv ~= eps_below * I`` — the *whole* matrix,
    not just its mean diagonal, which is only the DC coefficient and which two differently
    patterned layers can share. Walks bottom-to-top, matching the order the solve applies them
    in, and returns them indexed by layer so the solve can read ``same_material[layer_index]``
    directly.
    """
    xp = array_namespace(epx_conv_all)
    eps_below = xp.asarray(complex(n_bot).conjugate() ** 2, dtype=epx_conv_all.dtype)

    n_layers = epx_conv_all.shape[0]
    flags = [False] * n_layers
    for i in range(n_layers)[::-1]:
        layer_eps = uniform_eps(xp, epx_conv_all[i], rtol)
        flags[i] = _same_material(xp, layer_eps, eps_below, rtol)
        eps_below = layer_eps
    return flags
