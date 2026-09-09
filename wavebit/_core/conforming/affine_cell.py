"""Full-metric conforming cells for affine images of reference strip profiles.

This internal kernel maps the entire reference geometry through a basis matrix.
A reference rectangle becomes a physical parallelogram. It must not be used to
reinterpret physical rectangle dictionaries in the public ConformingNVF API.
"""
from numbers import Integral

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import solve

from ..backend import _concrete
from ..factorization import ConvMatrices
from .continuous_flux import flux_integrals, polynomial_moments
from .polynomial_body import ComplexMaterialGapModel, body_space
from .thin_gap import CONTACT_ROUNDOFF


def affine_lift_axes(metric, xp=np):
    """F0 = lift X + Y has identity mean and exact laminated normal flux."""
    c = metric[0, 1]/metric[0, 0]
    zero, one = xp.zeros_like(c), xp.ones_like(c)
    return (xp.stack([xp.stack([one, c]), xp.stack([zero, zero])]),
            xp.stack([xp.stack([zero, -c]), xp.stack([zero, one])]))


def project_metric_polynomial(potential, normal, moments, material, background, metric, xp=np):
    """Exact Fourier projection of F and F^H epsilon M F, including Mxy.

    Potential and normal have (element, polynomial, x basis, forcing) axes;
    moments have (y Fourier mode, element, polynomial) axes. Conjugation acts
    on trial factors, never on epsilon or the Fourier phase.
    """
    def linear(weight, coefficients):
        return sum(xp.einsum('ne,mi,eic->nmc', moments[..., i], weight, coefficients[:, i])
                   for i in range(coefficients.shape[1]))

    def quadratic(matrix, first, second):
        return sum(xp.einsum('ne,emcd->nmcd', moments[..., i+j],
                             xp.einsum('mij,eic,ejd->emcd', matrix,
                                       xp.conj(first[:, i]), second[:, j]))
                   for i in range(first.shape[1]) for j in range(second.shape[1]))

    def gradient(pack, xkey, ykey, dual=False):
        p, n = (xp.conj(potential), xp.conj(normal)) if dual else (potential, normal)
        return xp.stack([linear(pack[xkey], p), linear(pack[ykey], n)], axis=-2)

    scale = xp.sum(moments[..., 0], axis=1)[:, None]
    x, y = affine_lift_axes(metric, xp)
    field = (gradient(background, 'derivative', 'load')
             +(scale*background['lift'][None])[..., None, None]*x
             +(scale*background['material'][None])[..., None, None]*y)
    gram = (metric[0, 0]*quadratic(material['stiffness'], potential, potential)
            +metric[0, 1]*quadratic(material['mixed'], potential, normal)
            +metric[1, 0]*quadratic(material['reverse_mixed'], normal, potential)
            +metric[1, 1]*quadratic(material['mass'], normal, normal))
    primal = gradient(material, 'derivative', 'load')
    lifted = gradient(material, 'reverse_force', 'reverse_potential_force')
    dual = xp.swapaxes(gradient(material, 'dual_derivative', 'dual_load', True), -1, -2)
    dual_lifted = xp.swapaxes(gradient(material, 'force', 'potential_force', True), -1, -2)
    gram = gram+x.T@metric@lifted+y.T@metric@primal+dual_lifted@metric@x+dual@metric@y
    for key, first, second in (('lift_squared', x, x), ('dual_lift', x, y),
                               ('lift', y, x), ('material', y, y)):
        gram = gram+(scale*material[key][None])[..., None, None]*(first.T@metric@second)
    return field, gram, scale*material['material'][None]


class AffineCellGapModel(ComplexMaterialGapModel):
    """A reference rectangle profile with a live, full physical cell metric.

    ``basis`` contains physical lattice vectors in columns. Width, seams and
    gap keep their historical reference units. All reference shapes and gaps
    undergo the same affine map. Physical rectangle input remains a separate,
    unsupported geometry problem for genuinely nonorthogonal lattices.

    The trial potentials are H1-conforming; at finite resolution their flux
    need not satisfy the oblique transmission condition pointwise. The affine
    lift does satisfy that condition exactly at full perpendicular contact.

    ``order`` is a nonnegative integer or (x, y) pair. The coefficient envelope
    covers both directions, while matrix closure keeps the exact requested
    Fourier set. ``truncation_order`` records the pair for the consuming engine.
    """

    def __init__(self, order=7, *args, **kwargs):
        pair = (order, order) if isinstance(order, Integral) else order
        if (not isinstance(pair, (tuple, list)) or len(pair) != 2
                or any(isinstance(n, bool) or not isinstance(n, Integral) or n < 0 for n in pair)):
            raise ValueError('affine cell order must be a nonnegative integer or integer pair')
        self.truncation_order = tuple(int(n) for n in pair)
        super().__init__(max(self.truncation_order), *args, **kwargs)
        ox, oy = self.truncation_order
        n, m = np.meshgrid(np.arange(-oy, oy+1), np.arange(-ox, ox+1), indexing='ij')
        self.count = m.size
        self.dm = m.ravel()[:, None]-m.ravel()+2*self.order
        self.dn = n.ravel()[:, None]-n.ravel()+2*self.order
        # These small real tables integrate the polynomial y factors exactly.
        # Material and geometry differentiation happens in the x packs and g.
        self._indices, self._polynomials, _ = body_space(self.yedges, self.body_degree)
        q, w = leggauss(max(self.body_degree, self.gap_degree)+1)
        r, w = (q+1)/2, w/2
        dy = np.diff(self.yedges)
        self._body_t = (self.yedges[:-1, None]+dy[:, None]*r-.5).ravel()
        self._body_w = (dy[:, None]*w).ravel()
        h = np.zeros((len(self._body_t), self.rows))
        dh = np.zeros_like(h)
        value = sum(self._polynomials[:, k][None]*r[:, None]**k
                    for k in range(self.body_degree+1))
        derivative = sum(k*self._polynomials[:, k][None]*r[:, None]**(k-1)
                         for k in range(1, self.body_degree+1))
        for e, indices in enumerate(self._indices):
            selection = np.ix_(np.arange(e*len(r), (e+1)*len(r)), indices)
            h[selection] += value
            dh[selection] += derivative/dy[e]
        self._body_h, self._body_dh = h, dh
        self._gap_s, self._gap_w = r-.5, w
        self._gap_h = sum(self.gap_beta[:, k][None]*r[:, None]**k
                          for k in range(self.gap_degree+1))
        self._gap_dh = sum(k*self.gap_beta[:, k][None]*r[:, None]**(k-1)
                           for k in range(1, self.gap_degree+1))

    def _x_integrals(self, nodes, lengths, seams, values, xp):
        pivot = self.per_region//2 if self.flux_basis == 'local' else self.per_region+self.per_region//2
        return flux_integrals(nodes, lengths, seams, values,
                              np.arange(len(nodes))//self.per_region == 1, self.modes,
                              pivot, xp, basis=self.flux_basis, sesquilinear=True, mixed=True)

    def _solve_metric_cell(self, g, body, background, metric, xp):
        """Weak equations with enrichment test rows divided by g analytically."""
        length = 1-g
        zeros = xp.zeros((len(self._body_t), self.gap_degree-1))
        bx = xp.concatenate([xp.asarray(self._body_h), -xp.asarray(self._body_t)[:, None], zeros], axis=1)
        by = xp.concatenate([xp.asarray(self._body_dh), -xp.ones((len(self._body_t), 1)), zeros], axis=1)/length
        scales = xp.concatenate([xp.ones(self.rows), g*xp.ones(self.gap_degree)])
        gx = xp.concatenate([xp.broadcast_to(xp.asarray(self.boundary), (len(self._gap_s), self.rows)),
                             xp.asarray(self._gap_s)[:, None], xp.asarray(self._gap_h)], axis=1)
        gy = xp.concatenate([xp.zeros((len(self._gap_s), self.rows)), xp.ones((len(self._gap_s), 1)),
                             xp.asarray(self._gap_dh)], axis=1)
        x, y = affine_lift_axes(metric, xp)
        matrix, right = 0., 0.
        for test, trial, weights, pack in (
                ((bx, by), (bx*scales, by*scales), length*xp.asarray(self._body_w), body),
                ((g*gx, gy), (gx*scales, gy), xp.asarray(self._gap_w), background)):
            for i, j, key in ((0, 0, 'stiffness'), (0, 1, 'mixed'),
                              (1, 0, 'reverse_mixed'), (1, 1, 'mass')):
                matrix = matrix+metric[i, j]*xp.kron((test[i].T*weights)@trial[j], pack[key])
            for i, constant, lifted in ((0, 'dual_derivative', 'force'), (1, 'dual_load', 'potential_force')):
                forcing = pack[lifted][:, None]*(metric@x)[i]+pack[constant][:, None]*(metric@y)[i]
                right = right-((weights@test[i])[:, None, None]*forcing[None]).reshape(-1, 2)
        matrix, right = matrix[1:, 1:], right[1:]
        scale = 1/xp.sqrt(xp.abs(xp.diag(matrix)))
        balanced, forcing = scale[:, None]*matrix*scale[None], scale[:, None]*right
        solution = scale[:, None]*(solve(balanced, forcing) if xp is np else xp.linalg.solve(balanced, forcing))
        residual = xp.linalg.norm(matrix@solution-right)/xp.maximum(xp.linalg.norm(right), 1.)
        valid = xp.isfinite(residual) & (residual < 1e-8) & (g >= -CONTACT_ROUNDOFF) & (g < 1)
        solution = xp.where(valid, solution, xp.nan)
        end = self.nx*self.rows-1
        v = xp.concatenate([xp.zeros((1, 2)), solution[:end]]).reshape(self.rows, self.nx, 2)
        a = solution[end:end+self.nx]
        b = solution[end+self.nx:].reshape(self.gap_degree-1, self.nx, 2)
        return v, a, b, residual

    def coefficients(self, gap, epsilon=12.25, width=350., xp=np, *, basis):
        basis = xp.asarray(basis)
        if basis.shape != (2, 2):
            raise ValueError('affine cell basis must be a real finite right-handed 2-by-2 matrix')
        real = xp.all(xp.imag(basis) == 0)
        basis = xp.real(basis)
        finite = xp.all(xp.isfinite(basis))
        basis = xp.where(xp.isfinite(basis), basis, 0.)
        magnitude = xp.max(xp.abs(basis))
        basis = basis/xp.where(magnitude > 0, magnitude, 1.)
        valid_basis = real & finite & (xp.linalg.det(basis) > 0)
        if _concrete(valid_basis) is False:
            raise ValueError('affine cell basis must be a real finite right-handed 2-by-2 matrix')
        # The PDE is invariant under a common physical length scale. Normalize
        # before forming the metric; for diag(Lx,Ly) this is diag(1, aspect^2).
        basis = xp.where(valid_basis, basis/xp.sqrt(xp.sum(basis[:, 0]**2)), xp.eye(2))
        inverse = xp.linalg.inv(basis)
        metric = inverse@inverse.T
        g, edges, values, nodes, lengths, valid = self._profile_geometry(gap, epsilon, width, xp)
        body, background = self._x_integrals(nodes, lengths, edges[1:-1], values, xp)
        center = 2*self.order
        v, a, b, residual = self._solve_metric_cell(
            g, {k: v[center] for k, v in body.items()},
            {k: v[center] for k, v in background.items()}, metric, xp)
        v, a, b = [xp.where(valid & valid_basis, value, xp.nan) for value in (v, a, b)]
        dt, t = xp.asarray(np.diff(self.yedges)), xp.asarray(self.yedges[:-1])-.5
        potential = xp.einsum('jk,ejic->ekic', xp.asarray(self._polynomials), v[self._indices])
        ramp = xp.stack([t, dt]+[xp.zeros_like(t)]*(self.body_degree-1), axis=1)
        potential = potential-g*ramp[..., None, None]*a[None, None]
        normal = (xp.arange(1, self.body_degree+1)[None, :, None, None]*potential[:, 1:]
                  /((1-g)*dt[:, None, None, None]))
        moments = polynomial_moments((1-g)*t, (1-g)*dt, xp.zeros_like(t), xp.ones_like(t),
                                     self.modes, xp, degree=2*self.body_degree)
        field, gram, eps = project_metric_polynomial(potential, normal, moments, body, background, metric, xp)
        bubble = xp.einsum('bk,bic->kic', xp.asarray(self.gap_beta), b)
        potential = g*bubble+xp.stack([v[0]-g/2*a, g*a]+[xp.zeros_like(a)]*(self.gap_degree-1))
        normal = (xp.arange(1, self.gap_degree+1)[:, None, None]*bubble[1:]
                  +xp.stack([a]+[xp.zeros_like(a)]*(self.gap_degree-1)))
        moments = polynomial_moments(xp.asarray([.5-g/2]), xp.asarray([g]), xp.asarray([0.]),
                                     xp.asarray([1.]), self.modes, xp, degree=2*self.gap_degree)
        gf, gg, ge = project_metric_polynomial(potential[None], normal[None], moments,
                                               background, background, metric, xp)
        field = inverse.T@(field+gf)@basis.T
        gram = basis@(gram+gg)@basis.T
        coefficients = xp.stack([field[..., 0, 0], field[..., 0, 1], field[..., 1, 0], field[..., 1, 1],
                                 gram[..., 0, 0], gram[..., 0, 1], gram[..., 1, 1], eps+ge, gram[..., 1, 0]])
        return coefficients, dict(cell_residual=residual, degrees_of_freedom=self.nx*(self.rows+self.gap_degree)-1,
                                  body_degree=self.body_degree, gap_degree=self.gap_degree)

    def matrices(self, gap, epsilon=12.25, width=350., xp=np, *, basis):
        coefficients, metadata = self.coefficients(gap, epsilon, width, xp, basis=basis)
        return self._matrices_from_coefficients(coefficients, metadata, xp)

    def conv_matrices(self, gap, epsilon=12.25, width=350., xp=np, *, basis):
        """One layer for solve_conv with this basis and truncation_order.

        The profile background is one. Relative materials with a negative
        imaginary part require restoration of their physical background
        before closure; this convenience method does not perform that step.
        """
        slots, _ = self.matrices(gap, epsilon, width, xp, basis=basis)
        return ConvMatrices(*(slot[None] for slot in slots))
