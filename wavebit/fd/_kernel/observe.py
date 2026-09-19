"""Observers: what the step reads after the updates, into accumulators the state carries.

An observer has an accumulator that is part of the run's state -- a ``(steps,)`` array for
a point record -- and ``observe(acc, E, H, n, params, rt)`` returns the accumulator with
step ``n`` written. Written, not appended: the JAX scan carries the accumulator as an
ordinary array and sets one element per step, so the same protocol runs unchanged inside
``lax.scan``, a Python loop, and a checkpointed block. Observers never write to a field.

Time levels: an E component is read after the E update of step ``n``, at ``(n + 1) dt``; an H
component after the H update, at ``(n + 1/2) dt``. ``times`` says which for a given observer,
and the result carries them.

**Stage 3's one frequency-domain observer, ``DFTRegion``.** A running DFT of named components
over a region at a static list of frequencies,

    F^(f) = sum_n F(t_n) exp(+i 2 pi f t_n) dt,

the positive-frequency phasor of the ``e^{-i w t}`` convention (a real ``Re(A e^{-iwt})`` running
for a time ``T`` gives ``~ A T / 2``). **Each component is transformed at its own time level**:
E at ``t_n = (n + 1) dt`` and H at ``t_n = (n + 1/2) dt``, the times the observer reads them
at. That is what makes the E and H phasors co-timed without interpolation: the leapfrog's
``H(n+1/2) - H(n-1/2) = -dt curl E(n)`` becomes, summed against ``exp(i w t)`` over a run that
starts and ends at rest, ``-i Omega H^ = -curl E^`` with ``Omega = 2 sin(w dt/2)/dt`` exactly,
and the same for Ampere's law -- the discrete Maxwell equations in the frequency domain, which
every post-processing identity (a flux that closes, an order split into its two directions)
rests on. A run that stops before the field has decayed breaks those identities by the
amount still in the region, which is what the decay diagnostic measures.

The region is one :class:`Slab` per component (each component its own Yee points), the
frequencies are static (in the plan and its skeleton) and the accumulator is one complex
``(nf, *slab.shape)`` array per component, carried in the scan; ``check()`` prints the bytes.
Beside it, ``energy`` records ``sum |F|^2`` over the region once per step -- ``(steps,)``
reals -- which is what the always-on decay diagnostic reads. It carries no gradient (it is a
diagnostic, and it would otherwise drag the whole region into the backward pass).
"""
import dataclasses

import numpy as np

__all__ = ['PointRecord', 'DFTRegion']


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


@dataclasses.dataclass(frozen=True)
class DFTRegion:
    """A running DFT of ``parts`` at ``freqs`` -- the lowering of every stage-3 monitor.

    ``parts`` is ``((label, field, Slab), ...)``: the accumulator's key, ``'E'`` or ``'H'``, and
    that component's region. ``monitor`` is the public monitor it belongs to (a box monitor
    lowers to one region per face, named ``monitor::face``). ``dt`` is the plan's. All static;
    the angular frequencies ``2 pi f`` are a params leaf (``params['dft'][name]``) only so that
    each backend holds them as its own array.
    """

    name: str
    monitor: str
    freqs: tuple
    parts: tuple
    dt: float

    def init(self, plan, rt):
        nf = len(self.freqs)
        acc = {label: rt.zeros((nf,) + tuple(slab.shape), 'complex128')
               for label, _, slab in self.parts}
        acc['energy'] = rt.zeros((plan.steps,), 'float64')
        return acc

    def observe(self, acc, E, H, n, params, rt):
        omega, dt = params['dft'][self.name], self.dt
        # Each field at its own time level, the dt of the sum folded into the phase.
        phase = {'E': rt.xp.exp(1j * (omega * ((n + 1.0) * dt))) * dt,
                 'H': rt.xp.exp(1j * (omega * ((n + 0.5) * dt))) * dt}
        out, energy = {}, None
        for label, field, slab in self.parts:
            view = (E if field == 'E' else H)[slab.slices]
            out[label] = rt.dft_add(acc[label], view, phase[field], f'dft[{self.name}]{label}')
            e = rt.energy(view)
            energy = e if energy is None else energy + e
        out['energy'] = rt.record(acc['energy'], n, energy)
        return out

    def nbytes(self, plan):
        """The accumulator's bytes: complex128 per region point per frequency, plus ``energy``."""
        size = sum(int(np.prod(slab.shape)) for _, _, slab in self.parts)
        return 16 * len(self.freqs) * size + 8 * plan.steps
