"""Conforming rectangular NVF cell: thin gap."""
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import solve

from ..factorization import ConvMatrices

# Internal reference length; the public adapter scales the physical cell uniformly.
PERIOD = 700.


def _periodic_line(intervals, quadrature):
    """Reference P1 values/derivatives at element Gauss points, periodic DOFs."""
    points, weights = leggauss(quadrature)
    points, weights = (points+1)/2, weights/2
    values = np.zeros((intervals*quadrature, intervals))
    derivatives = np.zeros_like(values)
    for i in range(intervals):
        rows = slice(i*quadrature, (i+1)*quadrature)
        values[rows, i] += 1-points
        values[rows, (i+1) % intervals] += points
        derivatives[rows, i] -= 1
        derivatives[rows, (i+1) % intervals] += 1
    return points, weights, values, derivatives


def _line_mass(weights, xp=np):
    """Exact periodic P1 mass matrix for element weights epsilon * length."""
    off = xp.roll(xp.diag(weights), 1, axis=1)
    return xp.diag((weights+xp.roll(weights, 1))/3)+(off+off.T)/6


def _line_stiffness(weights, xp=np):
    """Exact periodic P1 stiffness for element weights epsilon / length."""
    off = xp.roll(xp.diag(weights), 1, axis=1)
    return xp.diag(weights+xp.roll(weights, 1))-off-off.T


class ThinGapEnergyModel:
    """Fixed conforming topology with a nonsingular one-sided contact limit.

    Geometry/material values remain live under JAX JIT; no trace-time sorting
    or zero-width interval deletion occurs. The x mesh is graded at the material
    interfaces because the limiting gap field has a jump there. Its resolution
    remains a study parameter, not a hidden acceptance tolerance.
    """

    def __init__(self, order=7, rows=16, edge_fraction=1e-5, quadrature=12,
                 *, grade_y=True, x_subdivisions=1, y_subdivisions=1):
        if not isinstance(order, int) or order < 0:
            raise ValueError('order must be a nonnegative integer')
        if not isinstance(rows, int) or rows < 2:
            raise ValueError('rows must be an integer at least two')
        if not 0 < edge_fraction < .001:
            raise ValueError('edge_fraction must lie between zero and .001')
        if not isinstance(quadrature, int) or quadrature < 2:
            raise ValueError('quadrature must be an integer at least two')
        if not isinstance(x_subdivisions, int) or x_subdivisions < 1:
            raise ValueError('x_subdivisions must be a positive integer')
        if not isinstance(y_subdivisions, int) or y_subdivisions < 1:
            raise ValueError('y_subdivisions must be a positive integer')
        self.order = order
        self.edge_fraction, self.quadrature = edge_fraction, quadrature
        fractions = np.array([0., edge_fraction, .001, .01, .1, .25,
                              .5, .75, .9, .99, .999, 1-edge_fraction, 1.])
        self.x_subdivisions = x_subdivisions
        self.fractions = np.r_[(fractions[:-1, None]+np.diff(fractions)[:, None]
                                *np.arange(x_subdivisions)/x_subdivisions).ravel(), 1.]
        self.per_region = len(self.fractions)-1
        self.nx = 3*self.per_region
        self.xrule = _periodic_line(self.nx, quadrature)
        self.yedges = np.linspace(0., 1., rows+1)
        if grade_y:
            self.yedges = np.unique(np.concatenate([self.yedges, fractions]))
        self.y_subdivisions = y_subdivisions
        if y_subdivisions > 1:
            self.yedges = np.r_[(self.yedges[:-1, None]+np.diff(self.yedges)[:, None]
                                 *np.arange(y_subdivisions)/y_subdivisions).ravel(), 1.]
        self.rows = len(self.yedges)-1
        dy = np.diff(self.yedges)
        self.yrule = _periodic_line(self.rows, quadrature)
        self.yt = (self.yedges[:-1, None]+dy[:, None]*self.yrule[0]-.5).ravel()
        self.yw = (dy[:, None]*self.yrule[1]).ravel()
        self.yvalue = self.yrule[2]
        self.yderivative = self.yrule[3]/np.repeat(dy, quadrature)[:, None]
        self.ymass = _line_mass(dy)
        self.ystiffness = _line_stiffness(1/dy)
        self.yload = (dy+np.roll(dy, 1))/2
        left, right = self.yedges[:-1]-.5, self.yedges[1:]-.5
        self.ycross = -dy*(2*left+right)/6-np.roll(dy*(left+2*right)/6, 1)
        self.boundary = np.eye(self.rows)[0]
        n, m = np.meshgrid(np.arange(-order, order+1), np.arange(-order, order+1), indexing='ij')
        self.count = m.size
        self.dm = m.ravel()[:, None]-m.ravel()+2*order
        self.dn = n.ravel()[:, None]-n.ravel()+2*order
        self.modes = np.arange(-2*order, 2*order+1)

    def _cell(self, gap, epsilon, width, xp):
        if xp is np:
            if (not np.isfinite([gap, epsilon, width]).all()
                    or not (0 <= gap < PERIOD and 0 < width < PERIOD and epsilon > 0)):
                raise ValueError('requires 0 <= gap < period, 0 < width < period, and positive epsilon')
        elif xp.asarray(0.).dtype != np.dtype('float64'):
            raise ValueError('the thin-gap energy study requires JAX x64')
        g, w = gap/PERIOD, width/PERIOD
        length = 1-g
        starts = xp.array([-.5, -w/2, w/2])
        sizes = xp.array([(1-w)/2, w, (1-w)/2])
        nodes = (starts[:, None]+sizes[:, None]*xp.asarray(self.fractions[:-1])).ravel()
        ends = (starts[:, None]+sizes[:, None]*xp.asarray(self.fractions[1:])).ravel()
        dx = ends-nodes
        points, weights, values, derivatives = self.xrule
        x = (nodes[:, None]+dx[:, None]*xp.asarray(points)).ravel()
        wx = (dx[:, None]*xp.asarray(weights)).ravel()
        f = xp.asarray(values)
        df = xp.asarray(derivatives)/xp.repeat(dx, self.quadrature)[:, None]
        eps_elements = xp.where(np.arange(self.nx)//self.per_region == 1, epsilon, 1.)
        eps = xp.repeat(eps_elements, self.quadrature)
        # The cell energy is polynomial on each material element. Assemble it
        # analytically, independent of the later Fourier projection quadrature.
        weighted_dx = dx*eps_elements
        me, ke = _line_mass(weighted_dx, xp), _line_stiffness(eps_elements/dx, xp)
        mb, kb = _line_mass(dx, xp), _line_stiffness(1/dx, xp)
        load = (weighted_dx+xp.roll(weighted_dx, 1))/2
        background_load = (dx+xp.roll(dx, 1))/2
        derivative_load = xp.roll(eps_elements, 1)-eps_elements
        v, a, residual = self._solve_cell(g, me, ke, load, derivative_load,
                                          mb, kb, background_load, xp)
        valid = (w > 0) & (w < 1) & (epsilon > 0)
        v, a = xp.where(valid, v, xp.nan), xp.where(valid, a, xp.nan)
        return g, length, x, wx, f, df, eps, v, a, residual

    def _solve_cell(self, g, me, ke, load, derivative_load, mb, kb, background_load, xp,
                    *, gap_derivative_load=None):
        """Common variational solve for any positive x material profile."""
        length = 1-g
        boundary = xp.asarray(self.boundary)
        avv = (length*xp.kron(self.ymass, ke)+xp.kron(self.ystiffness, me)/length
               +g*xp.kron(xp.outer(boundary, boundary), kb))
        # A_va = g*C. The normal-gradient cross term vanishes because every
        # periodic body basis has integral(dh/dt)=0. The gap's mean ramp is zero.
        cross = length*xp.kron(self.ycross[:, None], ke)
        scaled_aa = mb+g/length*me+g*length/12*ke+g*g/12*kb
        rhs_x = -length*xp.kron(self.yload, derivative_load)
        if gap_derivative_load is not None:
            rhs_x = rhs_x-g*xp.kron(boundary, gap_derivative_load)
        rhs_v = xp.stack([rhs_x,
                          xp.zeros(self.nx*self.rows)], axis=1)
        rhs_a = xp.stack([xp.zeros(self.nx), load-background_load], axis=1)
        # Pin one ordinary potential DOF. The enrichment test rows have already
        # been divided by g; there is no numerical 0/0, cutoff, or stopped grad.
        system = xp.block([[avv[1:, 1:], g*cross[1:]],
                           [cross[1:].T, scaled_aa]])
        right = xp.concatenate([rhs_v[1:], rhs_a])
        solution = solve(system, right) if xp is np else xp.linalg.solve(system, right)
        residual = xp.linalg.norm(system@solution-right)/xp.maximum(xp.linalg.norm(right), 1.)
        valid = xp.isfinite(residual) & (residual < 1e-8) & (g >= 0) & (g < 1)
        solution = xp.where(valid, solution, xp.nan)
        v = xp.concatenate([xp.zeros((1, 2)), solution[:self.nx*self.rows-1]])
        v = v.reshape(self.rows, self.nx, 2)
        a = solution[self.nx*self.rows-1:]
        return v, a, residual

    def coefficients(self, gap, epsilon=12.25, width=350., xp=np):
        g, length, x, wx, f, df, eps, v, a, residual = self._cell(gap, epsilon, width, xp)
        h, dh = xp.asarray(self.yvalue), xp.asarray(self.yderivative)
        t = xp.asarray(self.yt)
        fa, dfa = f@a, df@a
        bx = xp.einsum('yi,ijc,xj->yxc', h, v, df)-g*t[:, None, None]*dfa[None]
        by = (xp.einsum('yi,ijc,xj->yxc', dh, v, f)-g*fa[None])/length
        body = xp.stack([bx, by], axis=2)+xp.eye(2)
        points, weights = self.xrule[:2]
        s = xp.asarray(points)-.5
        gap_bx = (df@v[0])[None]+g*s[:, None, None]*dfa[None]
        gap_by = xp.broadcast_to(fa[None], gap_bx.shape)
        gap_basis = xp.stack([gap_bx, gap_by], axis=2)+xp.eye(2)
        expx = xp.exp(-2j*np.pi*xp.asarray(self.modes)[:, None]*x[None])

        def project(basis, material, y, wy):
            fields = xp.stack([basis[:, :, 0, 0], basis[:, :, 0, 1],
                               basis[:, :, 1, 0], basis[:, :, 1, 1],
                               material*xp.sum(basis[:, :, :, 0]**2, axis=2),
                               material*xp.sum(basis[:, :, :, 0]*basis[:, :, :, 1], axis=2),
                               material*xp.sum(basis[:, :, :, 1]**2, axis=2),
                               xp.broadcast_to(material, basis.shape[:2])])
            expy = xp.exp(-2j*np.pi*xp.asarray(self.modes)[:, None]*y[None])
            return xp.einsum('ny,ayx,mx->anm', expy*wy[None], fields, expx*wx[None])

        coefficients = (project(body, eps[None], length*t, length*xp.asarray(self.yw))
                        +project(gap_basis, xp.ones((1, len(x))), .5+g*s, g*xp.asarray(weights)))
        return coefficients, dict(cell_residual=residual, degrees_of_freedom=self.nx*(self.rows+1)-1)

    def matrices(self, gap, epsilon=12.25, width=350., xp=np):
        coefficients, metadata = self.coefficients(gap, epsilon, width, xp)
        return self._matrices_from_coefficients(coefficients, metadata, xp)

    def conv_matrices(self, gap, epsilon=12.25, width=350., xp=np):
        """One layer for Wavebit solve_conv, in the fixed local square cell.

        Use period=(700, 700) and truncation_order=(self.order, self.order)
        on the consuming solver. Its wavelength, incidence, thickness and
        exterior media belong to that solver, not the fixed study propagator.
        The cell's positive-real material and rectangular geometry restrictions
        remain in force. No geometry-selection fallback is performed.
        """
        slots, _ = self.matrices(gap, epsilon, width, xp)
        return ConvMatrices(*(slot[None] for slot in slots))

    def _matrices_from_coefficients(self, coefficients, metadata, xp):
        convolutions = coefficients[:, self.dn, self.dm]
        change = xp.block([[convolutions[0], convolutions[1]],
                           [convolutions[2], convolutions[3]]])
        gram = xp.block([[convolutions[4], convolutions[5]],
                         [convolutions[5], convolutions[6]]])
        slots = ConvMatrices.from_energy(change, gram, convolutions[7])
        hermiticity = xp.maximum(xp.maximum(
            xp.max(xp.abs(slots.epx-slots.epx.conj().T)),
            xp.max(xp.abs(slots.epy-slots.epy.conj().T))),
            xp.max(xp.abs(slots.epxy-slots.epyx.conj().T)))
        slots = ConvMatrices(*(xp.where(hermiticity < 1e-9, slot, xp.nan) for slot in slots))
        return slots, dict(**metadata, hermiticity=hermiticity)
