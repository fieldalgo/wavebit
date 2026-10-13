"""Matched absorption for a certified commuting electric constitutive region.

This module is used only by the bounded nonmagnetic/electric-source route.
It does not change ordinary ConstitutiveTerm arithmetic or admit CPML overlap.
For K R=R K, A=(1-hR/2)/(1+hR/2), B=1/(1+hR/2),
D'=A D+(h/dx) B curl(H) and the field increment uses K_cross B curl(H).
The companion observer adds the magnetic-loss correction needed by the
positive staggered storage, rather than reusing H_old.H_new as that storage.
"""
from dataclasses import dataclass

from wavebit.fdtd._kernel.constitutive import ConstitutiveTerm
from wavebit.fdtd._kernel.constitutive import cross_action
from wavebit.fdtd._kernel.observe import InteriorEnergy


@dataclass(frozen=True)
class AbsorberConstitutiveTerm(ConstitutiveTerm):
    """One compact D memory; the complete certified scene identity is static."""

    signature: tuple = ()

    def prepare(self, curl, mem, params, rt):
        if self.field != 'E':
            raise ValueError('ConstitutiveAbsorberScope: magnetic coupling is not admitted.')
        local = curl[self.view]
        if self.masks:
            local = local * 1.0
            for slab in self.masks:
                local = rt.slab_mul(local, slab, 0.0)
        factors = params['constitutive_absorber'][self.name]
        driven = factors['drive'] * local
        flux = factors['keep'] * mem + self.ratio * driven
        delta = self.ratio * cross_action(driven, params['constitutive'][self.name],
                                          rt.xp, 'E', bounded=True)
        return flux, delta


@dataclass(frozen=True)
class AbsorberConstitutiveEnergy(InteriorEnergy):
    """The positive staggered storage for the new source-free-H route.

    E.D + H_old.H_new differs from P storage by
    (dt/2) Re H_old* R_h (H_old+H_new). Absorber cells belong to this
    observer; they are not cropped away. All materials have mu=1 and
    before_H has no magnetic source or auxiliary source column.
    """

    def _weight(self, params, field, rt, comps=slice(None), box=None, times=None, key=''):
        # Avoid recovering B=(1+A)/2 after A rounds near -1. Only this route
        # carries the undamped electric drive and fixes physical mu=1.
        box = self.box if box is None else box
        scale = 1.0 if times is None else self._cut(params[times], comps, box)
        if field == 'H':
            return scale
        drive = self._cut(params['constitutive_absorber_drive_e'], comps, box)
        return ((self.ratio * drive) / self._cut(params['be'], comps, box)) * scale

    def _magnetic(self, H, new, cE, params, rt, comps, box, key):
        view = (comps,) + tuple(slice(lo, hi) for lo, hi in box)
        old = H[view]
        if new is None:
            # NumPy: before the destructive H update, with its unscaled curl.
            following = (self._cut(params['ah'], comps, box) * old
                         + self._cut(params['minus_bh'], comps, box) * cE[view])
        else:
            following = new[view]
        rate = self._cut(params['constitutive_absorber_rate_h'], comps, box)
        return (rt.real_inner(old, following)
                + rt.real_inner(old, old + following, (self.dt / 2.0) * rate))
