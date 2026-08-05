"""Frozen geometry — the discrete decisions, taken once, so the rest can be traced.

Two steps in the pipeline decide something *structural* rather than compute a value:

===================  ===========================================================
normal field (NVF)   which way do the material interfaces point?
compression (CFS)    where do the piecewise-constant blocks start and stop?
===================  ===========================================================

(There were three. The same-material short-circuit -- is this layer uniform and identical to
the medium below, so the interface can be skipped? -- was the third, and it is gone; see
``solve_1d``.)

Both branch on values, and the compression's output shape depends on them, so neither can
run inside a JAX trace. They are also both properties of the *geometry*, not of the
permittivity riding on it — so the fix is to take them once, eagerly, and hold them fixed
while gradients flow through the material values.

That is what :class:`StaticGeometry` carries. Freeze it once from a concrete structure::

    solver = fieldwave.rcwa(backend=Backend.JAX, ucell=ucell, ...)
    geometry = solver.freeze_geometry()
    grad = jax.grad(lambda u: jnp.sum(solver.conv_solve(ucell=u, geometry=geometry).res.de_ti))

Without it, ``conv_solve`` still auto-detects — the eager path that reproduces the legacy
result exactly — but nothing survives ``jax.jit``, and ``fourier_coefficients='sampled'``
with ``factorization='li'`` is the only pairing with nothing to detect in the first place.

**What a frozen record does not survive.** Every record here describes *one* structure. Replay
it onto a structure whose geometry has moved and it describes a layer that is no longer there,
and the answer comes back wrong without looking wrong — which is the failure the checks guard.
One rule covers all of them: *the record must still be a description of the layer in front of
it.* Each path has its own reading of that, because each record depends on something different:

======================  ==================================================================
``vector_grid``         the replayed grid lines must still be in order
``normal_field``        the interfaces it was sampled from must still be where they were
``compression``         the cut it recorded must still separate every block the layer has
======================  ==================================================================

All three raise :class:`StaleGeometryError`, and all three run **only eagerly**: under a trace
the comparison is itself the thing the record exists to avoid, so a jitted call carries no
check and a long fit should re-freeze on a schedule rather than wait to be told.

On the raster paths the sweep a frozen record serves is over the *material values* — those
never move an interface, so they are always accepted, exactly. A sweep that moves the pattern
in the raster has changed the geometry and must re-freeze.
"""
from dataclasses import dataclass
from typing import Any, Optional


class StaleGeometryError(RuntimeError):
    """A frozen geometry no longer describes the structure it is being replayed onto."""


@dataclass(eq=False)
class StaticGeometry:
    """The discrete geometry decisions of one structure, taken once.

    Attributes:
        normal_field: per-layer ``(nx, ny)`` for the NVF factorization.
        normal_field_source: per-layer ``(shape, interface mask)`` that ``normal_field`` was
            derived from — carried so a replay can tell whether it still fits, not consumed by
            the build. Left ``None``, the NVF replay is unchecked.
        compression: per-layer ``(row_idx, col_idx, x, y)`` for the CFS factorization.
        vector_grid: per-layer ``VectorGridPlan`` for the vector modeler's grid build.
        shape_choices: per-shape discrete branch outcomes taken inside ``Modeling.rectangle``.

    A field left ``None`` means "auto-detect", which is what the strategy in use does not need
    (a DFS build never touches ``normal_field``, for instance).

    ``eq=False`` keeps identity-based ``__eq__``/``__hash__``, which is what lets a frozen
    geometry be passed as a *static* argument to a jitted method such as
    ``SolverJax.conv_solve_field``: it holds arrays, so value equality would be both ill-defined
    and unhashable, whereas one freeze reused across calls is exactly one compilation.
    """
    normal_field: Optional[Any] = None
    normal_field_source: Optional[Any] = None
    compression: Optional[Any] = None
    vector_grid: Optional[Any] = None
    shape_choices: Optional[Any] = None
