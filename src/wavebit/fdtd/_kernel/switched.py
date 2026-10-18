"""The temporal-boundary term: a region whose permittivity jumps, written for ``D``, not ``E``.

A region holds ``eps^m`` at E level ``m``: the value before the jump for ``m < k`` and the
value after it from ``m = k`` on. Ampere's law is ``dD/dt = curl H`` with ``D = eps(t) E``, so
across the jump ``D`` (and ``H``) is continuous and ``E`` is not. The Yee update of ``D`` is

    eps^(n+1) E^(n+1) = eps^n E^n + dt * curl H^(n+1/2)

**Why it is a correction of the curl.** The coefficients over the region are built from the
background ``eps_b`` (the value before the jump), so the step's own update is
``E^(n+1) = E^n + be * cH`` with ``be = dt / (eps_b dx)`` and ``cH`` the bare difference. The
``D`` update above is that same line with ``cH`` replaced by

    cH' = (eps_b / eps^(n+1)) * (cH + (eps^n - eps^(n+1)) * (dx/dt) * E^n)

which this term writes into ``cH`` in the ``before_E`` slot, after every other term (sources
included: a current density belongs on the right of ``dD/dt``, so it is rescaled with the
curl). Away from the jump ``eps^n = eps^(n+1)`` and the correction is the rescaling by
``eps_b / eps^(n+1)``, which is the identity before the jump.

**The trap this avoids.** Writing the jump as ``eps(t) dE/dt = curl H`` keeps ``E``
continuous instead of ``D``. Away from the jump the two agree, so the frequencies, the
velocities and the energy bookkeeping all look right; at the jump the wave amplitudes come out
wrong by a factor that no resolution shrinks (exactly two for the forward and backward waves at
``eps 1 -> 4``). ``tests/fdtd/test_fd_switched.py`` holds that mutant.

**Memory.** ``E^n`` is needed in the ``before_E`` of step ``n``, where only ``H`` is handed in,
so ``advance`` keeps a copy of ``E^(n+1)`` over the region for the next step. The copy starts
at zero, which is exact because the plan refuses a jump before E level 2: at step 0 both
``eps^0`` and ``eps^1`` are the value before, and the stored field is multiplied by zero.
"""
import dataclasses

__all__ = ['SwitchedTerm']


@dataclasses.dataclass(frozen=True)
class SwitchedTerm:
    """One switched structure: its cells (``region``, one ``slab`` per component) and step ``k``.

    ``k`` is the first E level that holds the value after the jump; it is static, so it lives
    in the plan and its skeleton. The two permittivities are traced leaves in
    ``params['switched'][name]`` (``before`` and ``after`` as three per-component values, and
    ``dx_over_dt``), which is what makes a gradient with respect to either one flow.
    """

    name: str
    region: object
    slabs: tuple
    k: int

    def init_mem(self, plan, rt):
        return rt.zeros((3,) + self.region.shape, plan.dtype)

    def before_H(self, cE, E, mem, n, params, rt):
        return cE

    def _eps(self, level, before, after):
        """``eps`` at E level ``level``: ``before`` below ``k``, ``after`` from it.

        Arithmetic, not a branch: ``level`` is a traced int on JAX.
        """
        jumped = level >= self.k
        return before + (after - before) * jumped

    def before_E(self, cH, H, mem, n, params, rt):
        values = params['switched'][self.name]
        scale = values['dx_over_dt']
        for c, slab in enumerate(self.slabs):
            before, after = values['before'][c], values['after'][c]
            now = self._eps(n, before, after)
            nxt = self._eps(n + 1, before, after)
            ratio = before / nxt
            # A copy: the add below writes cH in place, and Torch's autograd keeps this
            # value for the gradient with respect to the ratio.
            local = cH[slab.slices] * 1.0
            delta = (ratio - 1.0) * local + ratio * (now - nxt) * scale * mem[c]
            cH = rt.slab_add(cH, slab, delta)
        return cH

    def advance(self, mem, E, H, n, params, rt):
        # A copy, so the memory is an array of its own (the state, the Decay groups and the
        # memory report count it as one); the next before_E reads it before E is updated.
        return E[(slice(None),) + self.region.slices] * 1.0
