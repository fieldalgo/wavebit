"""Fixed-fixture optical readouts for the shared conforming cell studies."""
import numpy as np

from studies.nvf_harmonic_energy import powers


class GapOptics:
    """Keep the historical study propagator outside the production cell code."""

    def transmission(self, gap, epsilon=12.25, width=350., xp=np, *, polarization='p'):
        slots, metadata = self.matrices(gap, epsilon, width, xp)
        t, r = powers(self.order, slots, xp, polarization=polarization)
        return dict(T=t, R=r, energy_error=xp.abs(t+r-1), **metadata)

    def shape_transmission(self, shapes, xp=np, *, polarization='p'):
        slots, metadata = self.shape_matrices(shapes, xp)
        t, r = powers(self.order, slots, xp, polarization=polarization)
        return dict(T=t, R=r, energy_error=xp.abs(t+r-1), **metadata)
