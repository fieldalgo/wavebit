"""Observers: what the step reads after the updates, into accumulators the state carries.

An observer has an accumulator that is part of the run's state -- a ``(steps,)`` array for
a point record -- and ``observe(acc, E, H, n, params, rt)`` returns the accumulator with
step ``n`` written. Written, not appended: the JAX scan carries the accumulator as an
ordinary array and sets one element per step, so the same protocol runs unchanged inside
``lax.scan``, a Python loop, and a checkpointed block. Observers never write to a field.

Time levels: an E component is read after the E update of step ``n``, at ``(n + 1) dt``; an H
component after the H update, at ``(n + 1/2) dt``. ``times`` says which for a given observer,
and the result carries them.
"""
import dataclasses

import numpy as np

__all__ = ['PointRecord']


@dataclasses.dataclass(frozen=True)
class PointRecord:
    """``Record`` lowered: ``field`` ``'E'``/``'H'``, component ``c``, indices ``k, j, i`` (z, y, x)."""

    name: str
    field: str
    c: int
    k: int
    j: int
    i: int

    def init(self, plan, rt):
        return rt.zeros((plan.steps,), plan.dtype)

    def observe(self, acc, E, H, n, params, rt):
        F = E if self.field == 'E' else H
        return rt.record(acc, n, F[self.c, self.k, self.j, self.i])

    def times(self, plan):
        """The sample times, ``(steps,)``."""
        shift = 1.0 if self.field == 'E' else 0.5
        return (np.arange(plan.steps) + shift) * plan.dt
