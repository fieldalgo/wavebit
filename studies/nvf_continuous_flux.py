"""Integrated continuous-flux basis for the rectangular NVF gap study.

Periodic P1 flux hats are integrated against reciprocal material to give a
continuous potential with a moving field jump. Unlike a local harmonic nodal
basis, its body displacement is continuous across artificial mesh nodes.
This remains an opt-in study, not production NVF or arbitrary geometry.
"""
import math

import numpy as np
from numpy.polynomial import Legendre, Polynomial
from scipy.linalg import solve

from studies.nvf_partitioned_gap_energy import PartitionedGapEnergyModel, interval_moments


def polynomial_moments(nodes, lengths, lo, hi, modes, xp=np, *, degree=4):
    """Exact Fourier moments through the requested degree in physical local r.

    Centered integration-by-parts recurrence is used away from zero; an entire
    series handles small frequencies and zero-width intersections under JIT.
    """
    span, middle = lengths*(hi-lo), (lo+hi)/2
    z = np.pi*xp.asarray(modes)[:, None]*span[None]
    small = xp.abs(z) < 2.
    safe, series_z = xp.where(small, 1., z), xp.where(small, z, 0.)
    centered = [xp.sinc(z/np.pi)]
    previous = xp.sinc(safe/np.pi)
    for k in range(1, degree+1):
        direct = ((.5**k*xp.exp(-1j*safe)-(-.5)**k*xp.exp(1j*safe))
                  -k*previous)/(-2j*safe)
        series = sum((-2j*series_z)**n/float(math.factorial(n)*2**(k+n)*(k+n+1))
                     for n in range(28) if (k+n) % 2 == 0)
        centered.append(xp.where(small, series, direct))
        previous = direct
    common = span[None]*xp.exp(-2j*np.pi*xp.asarray(modes)[:, None]
                              *(nodes+lengths*middle)[None])
    return xp.stack([common*sum(math.comb(k, j)*middle**(k-j)
                                *(hi-lo)**j*centered[j] for j in range(k+1))
                     for k in range(degree+1)], axis=-1)


def flux_integrals(nodes, lengths, seams, values, inside, modes, pivot, xp=np, *, basis='global'):
    """Weighted basis, affine-lift and energy integrals; no equality branch."""
    if basis not in ('global', 'local'):
        raise ValueError('flux basis must be global or local')
    count = len(nodes)
    left = xp.eye(count)
    change = xp.roll(left, 1, axis=1)-left
    base = xp.where(inside, values[0], 1.)
    rho = 1/base
    element = lengths[:, None]*rho[:, None]*(left+change/2)
    locations, jumps = [], []
    for i, seam in enumerate(seams):
        c = xp.clip((seam-nodes)/lengths, 0., 1.)
        jump = xp.where(inside, 1/values[i+1]-1/values[i], 0.)
        element = element+lengths[:, None]*jump[:, None]*((1-c[:, None])*left+(1-c[:, None]**2)*change/2)
        locations.append(c)
        jumps.append(jump)
    totals = xp.sum(element, axis=0)
    total = xp.sum(totals)
    prefix = xp.cumsum(element, axis=0)-element
    cumulative = xp.stack([prefix, lengths[:, None]*rho[:, None]*left,
                           lengths[:, None]*rho[:, None]*change/2], axis=1)
    keep = np.arange(count) != pivot

    def polynomials(material, reciprocal, cumulative):
        normalized = cumulative/totals[None, None]
        if basis == 'local':
            # Adjacent normalized primitives have equal total integral. Their
            # differences are periodic and span the same correction space.
            # Form these locally before multiplying/integrating: transforming
            # already assembled mass matrices loses the tiny support energies.
            potential = xp.roll(normalized, -1, axis=-1)-normalized
            # The two functions crossing the periodic origin carry constant
            # tails from the integration cut. Remove those constants too;
            # the separately retained constant keeps the same trial space.
            origin_fraction = element[0, 0]/totals[0]
            offsets = ((1-origin_fraction)*xp.eye(count)[0]+origin_fraction*xp.eye(count)[-1])
            potential = (potential-xp.asarray([1., 0., 0.])[None, :, None]*offsets)[..., keep]
            flux = xp.stack([reciprocal[:, None]*left/totals[None],
                             reciprocal[:, None]*change/totals[None]], axis=1)
            derivative = (xp.roll(flux, -1, axis=-1)-flux)[..., keep]
        else:
            potential = (normalized-xp.sum(cumulative, axis=-1)[..., None]/total)[..., keep]
            derivative = xp.stack([reciprocal[:, None]*(left/totals[None]-1/total),
                                    reciprocal[:, None]*change/totals[None]], axis=1)[..., keep]
        constant = xp.broadcast_to(xp.asarray([1., 0., 0.])[None, :, None], (count, 3, 1))
        potential = xp.concatenate([constant, potential], axis=-1)
        derivative = xp.concatenate([xp.zeros((count, 2, 1)), derivative], axis=-1)
        lift = reciprocal/total

        def quadratic(poly):
            degree = poly.shape[1]-1
            return xp.stack([sum(poly[:, i, :, None]*poly[:, j, None, :]
                                 for i in range(degree+1) for j in range(degree+1) if i+j == k)
                             for k in range(2*degree+1)], axis=1)

        unweighted = dict(mass=potential if basis == 'local' else quadratic(potential),
                          stiffness=derivative if basis == 'local' else quadratic(derivative),
                          load=potential, derivative=derivative,
                          force=derivative*lift[:, None, None], lift=lift[:, None],
                          lift_squared=lift[:, None]**2, material=xp.ones((count, 1)))
        if basis == 'local':
            # Carry factors into integration. Expanded quadratic packs require
            # O(elements * basis_count**2) storage although most entries vanish.
            return dict(**unweighted, weight=material), dict(**unweighted, weight=xp.ones(count))
        weighted = {key: value*material.reshape((count,)+(1,)*(value.ndim-1))
                    for key, value in unweighted.items()}
        return weighted, unweighted

    def integrate(moments, pack, previous=None, lo=None):
        if basis == 'local':
            # Three positive Gauss weights integrate degree-four products
            # exactly at zero Fourier frequency. This avoids monomial-product
            # cancellation in the mean matrices used by the cell solve.
            lo = xp.zeros(count) if lo is None else lo
            q = xp.asarray([.5-np.sqrt(3/5)/2, .5, .5+np.sqrt(3/5)/2])
            r = lo[None]+(1-lo[None])*q[:, None]
            quad_weight = xp.asarray([5/18, 4/9, 5/18])[:, None]*lengths[None]*(1-lo[None])
            change_moments = None if previous is None else moments*(pack['weight']-previous['weight'])[None, :, None]
            moments = moments*pack['weight'][None, :, None]
            result = {}
            for key, value in pack.items():
                if key == 'weight':
                    continue
                if key in ('mass', 'stiffness'):
                    entries = []
                    delta = None if previous is None else value-previous[key]
                    for i in range(value.shape[1]):
                        for j in range(i+1):
                            weight = moments[..., i+j, None]
                            if previous is None:
                                entry = value[:, i].T@(weight*value[None, :, j])
                            else:
                                old = previous[key]
                                # Polarize the weighted quadratic difference
                                # before integration. Subtracting two completed
                                # Fourier matrices loses small material changes.
                                entry = (delta[:, i].T@(weight*value[None, :, j])
                                         +old[:, i].T@(weight*delta[None, :, j])
                                         +old[:, i].T@(change_moments[..., i+j, None]*old[None, :, j]))
                            entries.append(entry if i == j else entry+xp.swapaxes(entry, -1, -2))
                    result[key] = sum(entries)
                    sampled = sum(r[..., None]**i*value[None, :, i] for i in range(value.shape[1]))
                    weighted = quad_weight*pack['weight'][None]
                    if previous is None:
                        mean = sum(sampled[k].T@(weighted[k, :, None]*sampled[k]) for k in range(3))
                    else:
                        old_sampled = sum(r[..., None]**i*previous[key][None, :, i] for i in range(value.shape[1]))
                        sampled_delta = sum(r[..., None]**i*delta[None, :, i] for i in range(value.shape[1]))
                        dw = quad_weight*(pack['weight']-previous['weight'])[None]
                        mean = sum(sampled_delta[k].T@(weighted[k, :, None]*sampled[k])
                                   +old_sampled[k].T@(weighted[k, :, None]*sampled_delta[k])
                                   +old_sampled[k].T@(dw[k, :, None]*old_sampled[k]) for k in range(3))
                    result[key] = xp.where(xp.asarray(modes)[:, None, None] == 0, mean[None], result[key])
                else:
                    if previous is None:
                        result[key] = xp.einsum('mek,ek...->m...', moments[..., :value.shape[1]], value)
                    else:
                        result[key] = (xp.einsum('mek,ek...->m...', moments[..., :value.shape[1]], value-previous[key])
                                       +xp.einsum('mek,ek...->m...', change_moments[..., :value.shape[1]], previous[key]))
            return result
        return {key: xp.einsum('mek,ek...->m...', moments[..., :value.shape[1]], value)
                for key, value in pack.items()}

    raw = polynomial_moments(nodes, lengths, xp.zeros_like(nodes), xp.ones_like(nodes), modes, xp)
    old_body, old_gap = polynomials(base, rho, cumulative)
    body, gap = integrate(raw, old_body), integrate(raw, old_gap)
    for i, (c, jump) in enumerate(zip(locations, jumps)):
        delta = lengths[:, None]*jump[:, None]
        cumulative = cumulative+xp.stack([-delta*(c[:, None]*left+c[:, None]**2*change/2),
                                           delta*left, delta*change/2], axis=1)
        material = xp.where(inside, values[i+1], 1.)
        next_body, next_gap = polynomials(material, 1/material, cumulative)
        suffix = polynomial_moments(nodes, lengths, c, xp.ones_like(nodes), modes, xp)
        for result, old, new in ((body, old_body, next_body), (gap, old_gap, next_gap)):
            if basis == 'local':
                changes = integrate(suffix, new, previous=old, lo=c)
            else:
                changes = integrate(suffix, {key: new[key]-old[key] for key in old})
            for key in result:
                result[key] = result[key]+changes[key]
        old_body, old_gap = next_body, next_gap
    return body, gap


def project_flux_strip(left, right, normal, ymoments, pack, xp=np):
    """Project the actual affine lift plus periodic integrated-flux potential."""
    y0, y1, y2 = (ymoments[..., i] for i in range(3))

    def quadratic(matrix, a, b):
        return xp.einsum('mij,eic,ejd->emcd', matrix, a, b)

    def integrate(weights, value):
        return xp.einsum('ne,emcd->nmcd', weights, value)

    energy = (integrate(y0-2*y1+y2, quadratic(pack['stiffness'], left, left))
              +integrate(y1-y2, quadratic(pack['stiffness'], left, right)+quadratic(pack['stiffness'], right, left))
              +integrate(y2, quadratic(pack['stiffness'], right, right))
              +integrate(y0, quadratic(pack['mass'], normal, normal)))

    def xform(weights):
        return (xp.einsum('ne,mi,eic->nmc', y0-y1, weights, left)
                +xp.einsum('ne,mi,eic->nmc', y1, weights, right))

    dx = xform(pack['derivative'])
    dy = xp.einsum('ne,mi,eic->nmc', y0, pack['load'], normal)
    linear = xp.stack([dx, dy], axis=-2)
    energy_linear = xp.stack([xform(pack['force']), dy], axis=-2)
    scale = xp.sum(y0, axis=1)[:, None]
    material, lift, lift_squared = [scale*pack[key][None] for key in ('material', 'lift', 'lift_squared')]
    ex, ey = xp.diag(xp.asarray([1., 0.])), xp.diag(xp.asarray([0., 1.]))
    field = linear+lift[..., None, None]*ex+material[..., None, None]*ey
    gram = (energy+energy_linear+xp.swapaxes(energy_linear, -1, -2)
            +lift_squared[..., None, None]*ex+material[..., None, None]*ey)
    return field, gram, material


def project_flux_polynomial(potential, normal, moments, pack, xp=np):
    """Fourier projection of a polynomial gap potential and its normal field."""
    def quadratic(matrix, a):
        # Contract the x basis before applying the y Fourier moment. A single
        # unfactored einsum repeats that contraction for every y mode in NumPy.
        return sum(moments[:, i+j, None, None, None]
                   *xp.einsum('mij,ic,jd->mcd', matrix, a[i], a[j])[None]
                   for i in range(len(a)) for j in range(len(a)))

    def linear(weights, a):
        return sum(moments[:, i, None, None]*xp.einsum('mi,ic->mc', weights, a[i])[None] for i in range(len(a)))

    dx, dy = linear(pack['derivative'], potential), linear(pack['load'], normal)
    field_linear = xp.stack([dx, dy], axis=-2)
    energy_linear = xp.stack([linear(pack['force'], potential), dy], axis=-2)
    energy = quadratic(pack['stiffness'], potential)+quadratic(pack['mass'], normal)
    material, lift, lift_squared = [moments[:, 0, None]*pack[key][None]
                                    for key in ('material', 'lift', 'lift_squared')]
    ex, ey = xp.diag(xp.asarray([1., 0.])), xp.diag(xp.asarray([0., 1.]))
    field = field_linear+lift[..., None, None]*ex+material[..., None, None]*ey
    gram = (energy+energy_linear+xp.swapaxes(energy_linear, -1, -2)
            +lift_squared[..., None, None]*ex+material[..., None, None]*ey)
    return field, gram, material


class ContinuousFluxGapModel(PartitionedGapEnergyModel):
    """A material-integrated potential whose body flux is continuous at nodes.

    For the periodic P1 flux hats f_j, let A_j=integral(rho*f_j), R=sum(A_j).
    The periodic potentials are integral(rho*f_j)/A_j-integral(rho)/R, plus a
    constant. One redundant flux hat is omitted at a fixed topology index.
    The x affine lift is integral(rho)/R. No mesh node follows a material seam.
    gap_degree adds integrated Legendre bubbles inside the gap, vanishing on
    both faces and scaled by gap so the contact system remains nonsingular.
    """

    def __init__(self, *args, gap_degree=4, flux_basis='global', **kwargs):
        if not isinstance(gap_degree, int) or not 1 <= gap_degree <= 6:
            raise ValueError('gap_degree must be an integer between one and six')
        if flux_basis not in ('global', 'local'):
            raise ValueError('flux_basis must be global or local')
        super().__init__(*args, **kwargs)
        self.flux_basis = flux_basis
        self.gap_degree = gap_degree
        indices = np.arange(1, gap_degree)
        beta = []
        for k in indices:
            polynomial = Legendre.basis(int(k), domain=[0., 1.]).convert(kind=Polynomial).integ()
            beta.append(np.pad(polynomial.coef, (0, gap_degree+1-len(polynomial.coef))))
        self.gap_beta = np.asarray(beta).reshape(gap_degree-1, gap_degree+1)
        self.gap_mean = np.where(indices == 1, -1/6, 0.)
        self.gap_ramp = np.where(indices == 2, -1/60, 0.)
        self.gap_gradient_mass = np.diag(1/(2*indices+1))
        self.gap_mass = np.zeros((len(indices), len(indices)))
        for i, k in enumerate(indices):
            for j, l in enumerate(indices):
                self.gap_mass[i, j] = sum(sa*sb/(2*a+1) for a, sa in ((k+1, 1), (k-1, -1))
                                         for b, sb in ((l+1, 1), (l-1, -1)) if a == b)/(4*(2*k+1)*(2*l+1))

    def _solve_flux_cell(self, g, me, ke, le, de, mb, kb, lb, dg, xp):
        if self.gap_degree == 1:
            v, a, residual = self._solve_cell(g, me, ke, le, de, mb, kb, lb, xp, gap_derivative_load=dg)
            return v, a, xp.zeros((0, self.nx, 2)), residual
        length = 1-g
        boundary = xp.asarray(self.boundary)
        avv = (length*xp.kron(self.ymass, ke)+xp.kron(self.ystiffness, me)/length
               +g*xp.kron(xp.outer(boundary, boundary), kb))
        cross = length*xp.kron(self.ycross[:, None], ke)
        aa = mb+g/length*me+g*length/12*ke+g*g/12*kb
        vb = xp.kron(boundary[:, None]*xp.asarray(self.gap_mean)[None], kb)
        ab = xp.kron(xp.asarray(self.gap_ramp)[None], kb)
        bb = xp.kron(self.gap_gradient_mass, mb)+g*g*xp.kron(self.gap_mass, kb)
        # Both enrichment test rows are divided by g analytically. Bubble
        # normal derivatives integrate to zero, so they have no y forcing.
        system = xp.block([[avv[1:, 1:], g*cross[1:], g*g*vb[1:]],
                           [cross[1:].T, aa, g*g*ab],
                           [g*vb[1:].T, g*g*ab.T, bb]])
        rhs_v = xp.stack([-length*xp.kron(self.yload, de)-g*xp.kron(boundary, dg),
                           xp.zeros(self.nx*self.rows)], axis=1)
        rhs_a = xp.stack([xp.zeros(self.nx), le-lb], axis=1)
        rhs_b = xp.stack([-g*xp.kron(self.gap_mean, dg), xp.zeros((self.gap_degree-1)*self.nx)], axis=1)
        right = xp.concatenate([rhs_v[1:], rhs_a, rhs_b])
        # Tiny contact cells give the potential and gap rows very different
        # scales. A diagonal change of coordinates balances the same equations;
        # it adds no regularization and remains live under differentiation.
        scale = 1/xp.sqrt(xp.abs(xp.diag(system)))
        balanced = scale[:, None]*system*scale[None]
        forcing = scale[:, None]*right
        solution = scale[:, None]*(solve(balanced, forcing) if xp is np else xp.linalg.solve(balanced, forcing))
        # Judge the result in the original equations, not only the balanced ones.
        residual = xp.linalg.norm(system@solution-right)/xp.maximum(xp.linalg.norm(right), 1.)
        valid = xp.isfinite(residual) & (residual < 1e-8) & (g >= 0) & (g < 1)
        solution = xp.where(valid, solution, xp.nan)
        end = self.nx*self.rows-1
        v = xp.concatenate([xp.zeros((1, 2)), solution[:end]]).reshape(self.rows, self.nx, 2)
        a = solution[end:end+self.nx]
        b = solution[end+self.nx:].reshape(self.gap_degree-1, self.nx, 2)
        return v, a, b, residual

    def coefficients(self, gap, epsilon=12.25, width=350., xp=np):
        g, edges, values, nodes, lengths, valid = self._profile_geometry(gap, epsilon, width, xp)
        body, background = self._x_integrals(nodes, lengths, edges[1:-1], values, xp)
        center = 2*self.order
        me, ke, le, de = [body[key][center].real for key in ('mass', 'stiffness', 'load', 'force')]
        mb, kb, lb, dg = [background[key][center].real for key in ('mass', 'stiffness', 'load', 'force')]
        v, a, b, residual = self._solve_flux_cell(g, me, ke, le, de, mb, kb, lb, dg, xp)
        v, a = xp.where(valid, v, xp.nan), xp.where(valid, a, xp.nan)
        field, gram, eps = self._project_body(v, a, g, body, background, xp)
        gap_field, gap_gram, gap_eps = self._project_gap(v, a, b, g, background, xp)
        field, gram, eps = field+gap_field, gram+gap_gram, eps+gap_eps
        coefficients = xp.stack([field[:, :, 0, 0], field[:, :, 0, 1], field[:, :, 1, 0], field[:, :, 1, 1],
                                 gram[:, :, 0, 0], gram[:, :, 0, 1], gram[:, :, 1, 1], eps])
        return coefficients, dict(cell_residual=residual, degrees_of_freedom=self.nx*(self.rows+self.gap_degree)-1,
                                  gap_degree=self.gap_degree)

    def _x_integrals(self, nodes, lengths, seams, values, xp):
        # Omitting a very narrow localized function makes the separately
        # retained constant nearly redundant. Use a fixed exterior pivot,
        # whose support is unaffected by material-driven interior grading.
        pivot = self.per_region//2 if self.flux_basis == 'local' else self.per_region+self.per_region//2
        return flux_integrals(nodes, lengths, seams, values,
                              np.arange(len(nodes))//self.per_region == 1, self.modes,
                              pivot, xp, basis=self.flux_basis)

    def _project_body(self, v, a, g, body, background, xp):
        t = xp.asarray(self.yedges)-.5
        dt = xp.asarray(np.diff(self.yedges))
        following = xp.roll(v, -1, axis=0)
        left, right = v-g*t[:-1, None, None]*a, following-g*t[1:, None, None]*a
        normal = (following-v)/((1-g)*dt[:, None, None])-g/(1-g)*a
        body_y = interval_moments((1-g)*t[:-1], (1-g)*dt, xp.zeros_like(dt), xp.ones_like(dt), self.modes, xp)
        field, _, _ = project_flux_strip(left, right, normal, body_y, background, xp)
        _, gram, eps = project_flux_strip(left, right, normal, body_y, body, xp)
        return field, gram, eps

    def _project_gap(self, v, a, b, g, background, xp):
        if self.gap_degree == 1:
            gap_y = interval_moments(xp.asarray([.5-g/2]), xp.asarray([g]), xp.asarray([0.]), xp.asarray([1.]), self.modes, xp)
            gap_field, gap_gram, gap_eps = project_flux_strip((v[0]-g/2*a)[None], (v[0]+g/2*a)[None],
                                                             a[None], gap_y, background, xp)
        else:
            beta = xp.asarray(self.gap_beta)
            bubble = xp.einsum('bk,bic->kic', beta, b)
            potential = g*bubble+xp.stack([v[0]-g/2*a, g*a]+[xp.zeros_like(a)]*(self.gap_degree-1))
            normal_gap = (xp.arange(1, self.gap_degree+1)[:, None, None]*bubble[1:]
                          +xp.stack([a]+[xp.zeros_like(a)]*(self.gap_degree-1)))
            moments = polynomial_moments(xp.asarray([.5-g/2]), xp.asarray([g]), xp.asarray([0.]),
                                          xp.asarray([1.]), self.modes, xp, degree=2*self.gap_degree)[:, 0]
            gap_field, gap_gram, gap_eps = project_flux_polynomial(potential, normal_gap, moments, background, xp)
        return gap_field, gap_gram, gap_eps


if __name__ == '__main__':
    from studies.nvf_harmonic_coordinates import main
    main(model_class=ContinuousFluxGapModel, kind='continuous_flux', gap_enrichment=True)
