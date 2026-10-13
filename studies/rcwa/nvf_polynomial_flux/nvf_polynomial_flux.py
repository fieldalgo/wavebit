"""Polynomial continuous x flux for the rectangular NVF research model.

Positive interior Bernstein bubbles enrich the continuous flux, then its
reciprocal-material integral supplies the periodic potential. The harmonic
lift still uses only the nodal partition of unity. No material equality test,
moving mesh topology, fitted optical correction or production dispatch.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
import math

import numpy as np
from numpy.polynomial import Polynomial

from studies.rcwa.nvf_continuous_flux.nvf_continuous_flux import flux_integrals, polynomial_moments
from studies.rcwa.nvf_polynomial_body.nvf_polynomial_body import PolynomialBodyGapModel


def flux_polynomials(elements, degree):
    """Periodic linear hats and positive local Bernstein interior functions."""
    coefficients = np.zeros((elements, degree+1, elements*degree))
    for e in range(elements):
        coefficients[e, :2, e] = [1., -1.]
        coefficients[e, 1, (e+1) % elements] += 1.
        for k in range(1, degree):
            bubble = math.comb(degree, k)*Polynomial([0., 1.])**k*Polynomial([1., -1.])**(degree-k)
            coefficients[e, :, elements+e*(degree-1)+k-1] = bubble.coef
    return coefficients


def polynomial_flux_integrals(nodes, lengths, seams, values, inside, modes, pivot, xp=np, *, degree=2):
    """Exact piecewise Fourier integrals of the enriched potential and energy."""
    if degree == 1:
        return flux_integrals(nodes, lengths, seams, values, inside, modes, pivot, xp)
    elements, count = len(nodes), len(nodes)*degree
    flux = xp.asarray(flux_polynomials(elements, degree))
    primitive = xp.concatenate([xp.zeros((elements, 1, count)),
                                flux/xp.arange(1, degree+2)[None, :, None]], axis=1)

    def evaluate(poly, r):
        return xp.sum(poly*r[:, None, None]**xp.arange(poly.shape[1])[None, :, None], axis=1)

    full = xp.sum(primitive, axis=1)
    base = xp.where(inside, values[0], 1.)
    rho = 1/base
    element = lengths[:, None]*rho[:, None]*full
    locations, jumps = [], []
    for i, seam in enumerate(seams):
        c = xp.clip((seam-nodes)/lengths, 0., 1.)
        jump = xp.where(inside, 1/values[i+1]-1/values[i], 0.)
        element = element+lengths[:, None]*jump[:, None]*(full-evaluate(primitive, c))
        locations.append(c)
        jumps.append(jump)
    totals = xp.sum(element, axis=0)
    # Interior bubbles are positive, so every normalization is nonzero. They
    # must not enter R or the affine lift: only the nodal hats sum to unity.
    total = xp.sum(totals[:elements])
    prefix = xp.cumsum(element, axis=0)-element
    cumulative = lengths[:, None, None]*rho[:, None, None]*primitive
    cumulative = cumulative+xp.concatenate([prefix[:, None], xp.zeros((elements, degree+1, count))], axis=1)
    keep = np.arange(count) != pivot

    def quadratic(poly):
        size = poly.shape[1]
        return xp.stack([sum(poly[:, i, :, None]*poly[:, j, None, :]
                             for i in range(size) for j in range(size) if i+j == k)
                         for k in range(2*size-1)], axis=1)

    def polynomials(material, cumulative):
        affine = xp.sum(cumulative[..., :elements], axis=-1)/total
        potential = (cumulative/totals[None, None]-affine[..., None])[..., keep]
        constant = xp.broadcast_to(xp.eye(degree+2)[0][None, :, None], (elements, degree+2, 1))
        potential = xp.concatenate([constant, potential], axis=-1)
        derivative = flux/(material[:, None, None]*totals[None, None])
        lift = 1/material/total
        derivative = (derivative-lift[:, None, None]*xp.eye(degree+1)[0][None, :, None])[..., keep]
        derivative = xp.concatenate([xp.zeros((elements, degree+1, 1)), derivative], axis=-1)
        unweighted = dict(mass=quadratic(potential), stiffness=quadratic(derivative),
                          load=potential, derivative=derivative, force=derivative*lift[:, None, None],
                          lift=lift[:, None], lift_squared=lift[:, None]**2, material=xp.ones((elements, 1)))
        weighted = {key: value*material.reshape((elements,)+(1,)*(value.ndim-1))
                    for key, value in unweighted.items()}
        return weighted, unweighted

    def integrate(moments, pack):
        return {key: xp.einsum('mek,ek...->m...', moments[..., :value.shape[1]], value)
                for key, value in pack.items()}

    def moments(lo):
        return polynomial_moments(nodes, lengths, lo, xp.ones_like(nodes), modes, xp, degree=2*(degree+1))

    old_body, old_gap = polynomials(base, cumulative)
    raw = moments(xp.zeros_like(nodes))
    body, gap = integrate(raw, old_body), integrate(raw, old_gap)
    for i, (c, jump) in enumerate(zip(locations, jumps)):
        offset = evaluate(primitive, c)
        increment = primitive-xp.concatenate([offset[:, None], xp.zeros((elements, degree+1, count))], axis=1)
        cumulative = cumulative+lengths[:, None, None]*jump[:, None, None]*increment
        material = xp.where(inside, values[i+1], 1.)
        next_body, next_gap = polynomials(material, cumulative)
        suffix = moments(c)
        for result, old, new in ((body, old_body, next_body), (gap, old_gap, next_gap)):
            change = integrate(suffix, {key: new[key]-old[key] for key in old})
            for key in result:
                result[key] = result[key]+change[key]
        old_body, old_gap = next_body, next_gap
    return body, gap


class PolynomialFluxGapModel(PolynomialBodyGapModel):
    """Fixed C0 polynomial flux, integrated into a conforming potential."""

    def __init__(self, *args, flux_degree=2, **kwargs):
        if not isinstance(flux_degree, int) or not 1 <= flux_degree <= 3:
            raise ValueError('flux_degree must be an integer between one and three')
        super().__init__(*args, **kwargs)
        if self.flux_basis != 'global' and flux_degree != 1:
            raise ValueError('local flux_basis is implemented only for flux_degree one')
        self.flux_degree = flux_degree
        self.nx *= flux_degree

    def _x_integrals(self, nodes, lengths, seams, values, xp):
        if self.flux_degree == 1:
            return super()._x_integrals(nodes, lengths, seams, values, xp)
        return polynomial_flux_integrals(nodes, lengths, seams, values,
                                         np.arange(len(nodes))//self.per_region == 1, self.modes,
                                         self.per_region+self.per_region//2, xp, degree=self.flux_degree)

    def coefficients(self, *args, **kwargs):
        c, metadata = super().coefficients(*args, **kwargs)
        return c, dict(**metadata, flux_degree=self.flux_degree)


if __name__ == '__main__':
    from studies.rcwa.nvf_harmonic_coordinates.nvf_harmonic_coordinates import main
    main(model_class=PolynomialFluxGapModel, kind='polynomial_flux_gap', gap_enrichment=True,
         mesh_grading=True, body_enrichment=True, flux_enrichment=True)
