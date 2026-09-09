"""Conforming rectangular NVF cell: material grading."""
import numpy as np

from .continuous_flux import ContinuousFluxGapModel


def _invert_cdf(targets, cumulative, density, xp):
    total = cumulative(xp.asarray(1.))
    lo, hi = xp.zeros_like(targets), xp.ones_like(targets)
    for _ in range(18):
        middle = (lo+hi)/2
        below = cumulative(middle) < targets*total
        lo, hi = xp.where(below, middle, lo), xp.where(below, hi, middle)
    roots = (lo+hi)/2
    for _ in range(4):
        roots = roots-(cumulative(roots)-targets*total)/density(roots)
    roots = xp.where(targets == 0, 0., xp.where(targets == 1, 1., roots))
    residual = xp.max(xp.abs(cumulative(roots)/total-targets))
    return roots, residual


def independent_graded_fractions(targets, seams, values, strength=8., radius=.01, xp=np):
    """Historical independent-jump CDF for reproducing separated-interface studies.

    Coincident opposite jumps do not cancel in this density. It is retained
    as a comparison instrument, not used for the cell's default grading.
    """
    targets, seams, values = xp.asarray(targets), xp.asarray(seams), xp.asarray(values)
    weights = strength/np.pi*((values[1:]-values[:-1])/(values[1:]+values[:-1]))**2
    def cumulative(x):
        result = x
        for i, seam in enumerate(seams):
            result = result+weights[i]*(xp.arctan((x-seam)/radius)-xp.arctan(-seam/radius))
        return result
    def density(x):
        result = xp.ones_like(x)
        for i, seam in enumerate(seams):
            result = result+weights[i]*radius/((x-seam)**2+radius**2)
        return result
    return _invert_cdf(targets, cumulative, density, xp)


def graded_fractions(targets, seams, values, strength=8., radius=.01, xp=np):
    """Invert a positive density formed from signed, smoothed log-material jumps.

    Log jumps add before squaring, so coincident interfaces merge and opposite
    jumps cancel. Gaussian products have exact erf primitives. Eighteen bracket
    steps and four Newton steps differentiate the same forward CDF, including
    zero-width material intervals, without merging or a custom backward rule.
    """
    if xp is np:
        from scipy.special import erf
    else:
        from jax.scipy.special import erf
    targets, seams, values = xp.asarray(targets), xp.asarray(seams), xp.asarray(values)
    jumps = (xp.log(values[1:])-xp.log(values[:-1]))/2
    terms = []
    for i, seam in enumerate(seams):
        for j in range(i, len(seams)):
            weight = (strength/2*(1 if i == j else 2)*jumps[i]*jumps[j]
                      *xp.exp(-((seam-seams[j])/(2*radius))**2))
            terms.append(((seam+seams[j])/2, weight))
    def cumulative(x):
        correction = xp.zeros_like(x)
        for middle, weight in terms:
            correction = correction+weight*(erf((x-middle)/radius)-erf(-middle/radius))
        return x+correction
    def density(x):
        field = xp.zeros_like(x)
        for seam, jump in zip(seams, jumps):
            field = field+jump*xp.exp(-.5*((x-seam)/radius)**2)
        return 1+strength/(np.sqrt(np.pi)*radius)*field**2
    return _invert_cdf(targets, cumulative, density, xp)


class MaterialGradedGapModel(ContinuousFluxGapModel):
    """Resolve real seams with material-defined node positions and fixed topology.

    Signed log-material jumps are smoothed together before squaring, so a
    collapsing interval cannot leave a fictitious grading interface behind.
    grading_radius is a fraction of the envelope width. grading_strength is
    dimensionless. Zero strength selects the original fixed mesh exactly.
    These are explicit resolution settings, not fitted optical corrections.
    """

    def __init__(self, *args, grading_strength=8., grading_radius=.01, **kwargs):
        if not np.isfinite(grading_strength) or not 0 <= grading_strength <= 32:
            raise ValueError('grading_strength must be finite and between zero and 32')
        super().__init__(*args, **kwargs)
        minimum = .0001 if self.flux_basis == 'local' else .005
        if not np.isfinite(grading_radius) or not minimum <= grading_radius <= .1:
            raise ValueError(f'grading_radius must be finite and between {minimum} and .1 for this flux basis')
        self.grading_strength, self.grading_radius = grading_strength, grading_radius

    def _profile_geometry(self, gap, epsilon, width, xp):
        geometry = super()._profile_geometry(gap, epsilon, width, xp)
        if self.grading_strength == 0:
            return geometry
        g, edges, values, nodes, lengths, valid = geometry
        seams = (edges[1:-1]-edges[0])/(edges[-1]-edges[0])
        fractions, residual = graded_fractions(self.fractions, seams, values,
                                               self.grading_strength, self.grading_radius, xp)
        interior = edges[0]+(edges[-1]-edges[0])*fractions
        positive = xp.all(interior[1:] > interior[:-1])
        converged = xp.isfinite(residual) & (residual < 2e-12) & positive
        if xp is np and not converged:
            raise FloatingPointError('material grading did not produce an ordered, converged mesh')
        n = self.per_region
        nodes = xp.concatenate([nodes[:n], interior[:-1], nodes[2*n:]])
        lengths = xp.concatenate([lengths[:n], interior[1:]-interior[:-1], lengths[2*n:]])
        return g, edges, values, nodes, lengths, valid & converged
