"""Frozen geometry — the discrete decisions, taken once, so the rest can be traced.

Two steps in the pipeline decide something *structural* rather than compute a value:

===================  ===========================================================
normal field (NVF)   which way do the material interfaces point?
compression    where do the piecewise-constant blocks start and stop?
===================  ===========================================================

(There were three. The same-material short-circuit -- is this layer uniform and identical to
the medium below, so the interface can be skipped? -- was the third, and it is gone; see
``solve_1d``.)

Both branch on values, and the compression's output shape depends on them, so neither can
run inside a JAX trace. They are also both properties of the *geometry*, not of the
permittivity riding on it — so the fix is to take them once, eagerly, and hold them fixed
while gradients flow through the material values.

That is what :class:`StaticGeometry` carries. Freeze it once from a concrete structure::

    solver = wavebit.rcwa(backend=Backend.JAX, structure=eps, ...)
    geometry = solver.freeze_geometry()
    grad = jax.grad(lambda e: jnp.sum(solver.solve(structure=e, geometry=geometry).given.T))

Without it, ``solve`` still auto-detects — the eager path that reproduces the legacy
result exactly — but nothing survives ``jax.jit``, and ``fourier_coefficients='sampled'``
with ``factorization='li'`` is the only pairing with nothing to detect in the first place.

**What a frozen record does not survive.** Every record here describes *one* structure. Replay
it onto a structure whose geometry has moved and it describes a layer that is no longer there,
and the answer comes back wrong without looking wrong — which is the failure the checks guard.
One rule covers all of them: *the record must still be a description of the layer in front of
it.* Each path has its own reading of that, because each record depends on something different:

======================  ==================================================================
``normal_field``        the interfaces it was sampled from must still be where they were
``compression``         the cut it recorded must still separate every block the layer has
``boundary_plans``      the shapes must still overlap the way the painting was resolved for
======================  ==================================================================

All three raise :class:`StaleGeometryError`, and all three run **only eagerly**: under a trace
the comparison is itself the thing the record exists to avoid, so a jitted call carries no
check and a long fit should re-freeze on a schedule rather than wait to be told.

``boundary_plans`` reads that rule differently from the other two, because it holds no
coordinates at all. A shape *moving* is what it is built to survive -- that is what makes the
boundary route differentiable -- so only the combinatorics changing makes it stale: shapes that
begin to overlap, or stop, or a shape that was covered coming out from under the one covering
it. Its ``None`` entry is a recorded answer too ("painted, and nothing overlapped"), and goes
stale the same way. Both are checked by re-asking, which costs a re-plan -- 0.2% of a frozen
solve for two overlapping shapes and 1.2% for ten, falling as ``truncation_order`` rises. The freeze
here buys traceability, not speed, so no plateau is given up for it.

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
        normal_field: per-layer ``(normal_x, normal_y)`` for the NVF factorization.
        normal_field_source: per-layer ``(shape, interface mask, lattice key,
            normal_method, phase count)`` that ``normal_field`` was derived from — carried
            so a replay can tell whether it still fits, not consumed by the build. Left
            ``None``, the NVF replay is unchecked.
        compression: per-layer ``(row_idx, col_idx, x, y)`` for the integrated path's factorization.
        mu_normal_field, mu_normal_field_source, mu_compression: the same three records, taken
            off the raster ``mu`` field instead of the permittivity. Separate rather than
            shared because a magnetic inclusion need not sit where the dielectric one does,
            and the NVF rearrangement is written around *one* interface normal -- so each
            quantity is built on the normal of its own boundary. All three are ``None``
            unless the structure carries a raster ``mu``; the two shape paths need no twin
            here at all, since their records hold the geometry both quantities share.
        boundary_plans: per-layer ``(table_records, regions)`` for the boundary route — which
            loop holds which vertex and which edges cross, once the painting is resolved. These
            are pure Python indices, never coordinates, so a replay
            carries a gradient to the vertices and they compare by content. ``None`` for a layer
            is an answer rather than a gap: painted, looked at while concrete, and nothing
            overlapped, so it is summed. Both answers are re-asked eagerly
            (:func:`boundary.check_plan_still_fits`, :func:`parametric.check_no_plan_still_fits`).

    A field left ``None`` means "auto-detect", which is what the strategy in use does not need
    (a sampled build never touches ``normal_field``, for instance).

    **Why the equality below is what it is.** A frozen geometry is passed as a *static*
    argument to a jitted call, so its ``__eq__``/``__hash__`` decide when JAX recompiles. Two
    records answer that question differently:

    * ``boundary_plans`` is pure Python — which loop holds which vertex, which edges cross.
      Nothing but *combinatorics*: the vertex coordinates themselves stay traced, which is what
      carries a gradient to a shape parameter. So it compares **by content**, and a freeze
      taken again on a structure whose combinatorics have not changed hits the compilation
      cache. That is the difference between re-freezing every step and re-freezing on a
      schedule, and it is what makes re-freezing every step the *cheapest* way to run a moving
      fit rather than the most expensive.
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
    boundary_plans: Optional[Any] = None
    mu_normal_field: Optional[Any] = None
    mu_normal_field_source: Optional[Any] = None
    mu_compression: Optional[Any] = None

    def _cache_key(self):
        """What ``__eq__``/``__hash__`` compare: the content when that is well defined.

        Falls back to identity the moment an array-bearing record is present, or if the
        boundary plans turn out to hold something unhashable — a conservative answer costs a
        recompilation, a wrong one would serve a cached program for a different structure.

        All six array-bearing slots are tested, the three ``mu_*`` records included, even
        though today they are only ever filled beside their permittivity twins -- so the twins
        alone would in fact decide it. Testing them anyway costs three ``is None`` comparisons
        and does not depend on that: "the permeability pass runs whenever the permittivity one
        did" is a property of ``freeze_geometry``, one module over, and a cache key that
        inferred its answer from another module's fill order would go wrong silently the first
        time that order changed. The failure it would produce is the expensive kind -- a
        content key hashed over ``boundary_plans`` alone while an array sits in
        ``mu_normal_field``,
        which serves a cached program compiled for a different structure.
        """
        if self.normal_field is None and self.normal_field_source is None \
                and self.compression is None and self.mu_normal_field is None \
                and self.mu_normal_field_source is None and self.mu_compression is None:
            try:
                key = ('content', _hashable(self.boundary_plans))
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
