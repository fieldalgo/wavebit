"""Compact, self-adjoint inverse-material coupling on the Yee lattice.

Cross coefficients live at common nodes for E and common cell centres for H.
The two averages are adjoints, as in Oskooi, Kottke & Johnson, Opt. Lett. 34,
2778 (2009), Fig. 1. No external solver source is used. A term stores D or B:
dividing a coupled field by the inverse tensor's diagonal does not recover it.
"""
from dataclasses import dataclass

from .masks import Slab


def cross_action(vector, cross, xp, field='E', bounded=False):
    """Apply the xy/xz/yz entries with adjoint half-cell interpolation.

    The factor 1/2 is the exact two-point arithmetic interpolation weight.
    The production call uses zero extension at finite ends. The two shifts
    remain adjoints there. Periodic rolls are available for the independent
    Fourier-symbol instrument, not for compact production regions.
    """
    def shift(value, direction, axis):
        if not bounded:
            return xp.roll(value, direction, axis=axis)
        head, tail = [slice(None)]*3, [slice(None)]*3
        head[axis], tail[axis] = slice(0, 1), slice(1, None)
        zero = xp.zeros_like(value[tuple(head)])
        if direction < 0:
            return xp.concat([value[tuple(tail)], zero], axis=axis)
        tail[axis] = slice(None, -1)
        return xp.concat([zero, value[tuple(tail)]], axis=axis)
    direction = 1 if field == 'E' else -1
    common = [0.5 * (vector[j] + shift(vector[j], direction, axis=2-j))
              for j in range(3)]
    out = [xp.zeros_like(vector[0]) for _ in range(3)]
    for k, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
        a, b = cross[k] * common[j], cross[k] * common[i]
        out[i] = out[i] + 0.5 * (a + shift(a, -direction, axis=2-i))
        out[j] = out[j] + 0.5 * (b + shift(b, -direction, axis=2-j))
    return xp.stack(out)


@dataclass(frozen=True)
class ConstitutiveTerm:
    """One lossless region, its flux memory, and a constitutive step slot.

    Integrate D/B with the same source-corrected curl that updates E/H. Add
    only the off-diagonal field increment; the diagonal update stays in its
    original slot. For fixed material and zero initial field this preserves
    E=K_e D and H=K_m B up to accumulation roundoff, without a matrix solve.
    """
    name: str
    field: str
    region: object
    ratio: float
    masks: tuple = ()

    @property
    def view(self):
        return (slice(None),) + self.region.slices

    @property
    def box(self):
        return (self.region.z, self.region.y, self.region.x)

    def init_mem(self, plan, rt):
        return rt.zeros((3,) + self.region.shape, plan.dtype)

    def before_H(self, curl, E, mem, n, params, rt):
        return curl

    def before_E(self, curl, H, mem, n, params, rt):
        return curl

    def advance(self, mem, E, H, n, params, rt):
        return mem

    def prepare(self, curl, mem, params, rt):
        """Read the curl before NumPy's diagonal update scales it in place."""
        gain = self.ratio if self.field == 'E' else -self.ratio
        local = curl[self.view]
        if self.masks:
            local = local * 1.0
            for slab in self.masks:
                local = rt.slab_mul(local, slab, 0.0)
        flux = mem + gain * local
        delta = gain * cross_action(local, params['constitutive'][self.name], rt.xp,
                                    self.field, bounded=True)
        return flux, delta

    def add(self, field, delta, rt):
        for c in range(3):
            slab = Slab(c=c, z=self.region.z, y=self.region.y, x=self.region.x)
            field = rt.slab_add(field, slab, delta[c])
        return field


def prepare(terms, field, curl, mem, params, rt):
    """Advance compact flux memories and return the pending field increments."""
    out, pending = dict(mem), []
    for term in terms:
        if term.field == field:
            out[term.name], delta = term.prepare(curl, mem[term.name], params, rt)
            pending.append((term, delta))
    return out, pending
