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
    grad = jax.grad(lambda u: jnp.sum(solver.conv_solve(ucell=u, geometry=geometry).res.T))

Without it, ``conv_solve`` still auto-detects — the eager path that reproduces the legacy
result exactly — but nothing survives ``jax.jit``, and ``fourier_coefficients='sampled'``
with ``factorization='li'`` is the only pairing with nothing to detect in the first place.

**What a frozen record does not survive.** Every record here describes *one* structure. Replay
it onto a structure whose geometry has moved and it describes a layer that is no longer there,
and the answer comes back wrong without looking wrong — which is the failure the checks guard.
One rule covers all of them: *the record must still be a description of the layer in front of
it.* Each path has its own reading of that, because each record depends on something different:

======================  ==================================================================
``block_grid``         the replayed grid lines must still be in order
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
        block_grid: per-layer ``VectorGridPlan`` for the block modeler's grid build.
        shape_choices: per-shape discrete branch outcomes taken inside ``Modeling.rectangle``.
        boundary_plans: per-layer ``(table_records, regions)`` for the boundary path — which
            loop holds which vertex and which edges cross, once the painting is resolved. Like
            the vector records these are pure Python indices, never coordinates, so a replay
            carries a gradient to the vertices and they compare by content.

    A field left ``None`` means "auto-detect", which is what the strategy in use does not need
    (a DFS build never touches ``normal_field``, for instance).

    **Why the equality below is what it is.** A frozen geometry is passed as a *static*
    argument to a jitted call, so its ``__eq__``/``__hash__`` decide when JAX recompiles. Two
    records answer that question differently:

    * The vector records are pure Python — which block edge becomes which grid line, in what
      order. Nothing but *ordering*: the edge coordinates themselves stay traced, which is what
      carries a gradient to a length. So they compare **by content**, and a freeze taken again
      on a structure whose ordering has not changed hits the compilation cache. That is the
      difference between re-freezing every step and re-freezing on a schedule — measured on an
      ellipse swept over ``lx`` 85 to 125, two compilations instead of ten, and re-freezing
      every step becomes the *cheapest* way to run, not the most expensive one.
    * ``normal_field`` and ``compression`` hold arrays, where value equality is both
      ill-defined and unhashable. A record carrying either falls back to **identity**, exactly
      as before.

    A content-hashed record must not be mutated after it is handed to a jitted call, for the
    usual reason: its hash would move underneath the cache. ``freeze_geometry`` fills one in
    before returning it, and nothing else writes to one.
    """
    normal_field: Optional[Any] = None
    normal_field_source: Optional[Any] = None
    compression: Optional[Any] = None
    block_grid: Optional[Any] = None
    shape_choices: Optional[Any] = None
    boundary_plans: Optional[Any] = None

    def _cache_key(self):
        """What ``__eq__``/``__hash__`` compare: the content when that is well defined.

        Falls back to identity the moment an array-bearing record is present, or if the vector
        records turn out to hold something unhashable — a conservative answer costs a
        recompilation, a wrong one would serve a cached program for a different structure.
        """
        if self.normal_field is None and self.normal_field_source is None \
                and self.compression is None:
            try:
                key = ('content', _hashable(self.block_grid), _hashable(self.shape_choices),
                       _hashable(self.boundary_plans))
                hash(key)
                return key
            except TypeError:
                pass
        return ('identity', id(self))

    def __eq__(self, other):
        if not isinstance(other, StaticGeometry):
            return NotImplemented
        return self._cache_key() == other._cache_key()

    def __hash__(self):
        return hash(self._cache_key())


def _hashable(obj):
    """``obj`` rebuilt out of tuples so it can be hashed, leaving leaves alone.

    Raises ``TypeError`` through :func:`hash` at the call site if a leaf is unhashable, which
    is the signal to fall back to identity rather than guess.
    """
    if isinstance(obj, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in obj.items()))
    if isinstance(obj, (list, tuple)):
        return tuple(_hashable(v) for v in obj)
    return obj
