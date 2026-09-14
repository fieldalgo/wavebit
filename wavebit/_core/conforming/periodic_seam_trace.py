"""Real continuous seam lifts and their k_y=0 normal Galerkin response.

This NumPy component supplies the missing contact traces through positive-gap
potentials. It is not a full transverse closure or an ordinary NVF dispatcher.
The original finite-element family is unchanged.
"""
from numbers import Integral
import numpy as np
from numpy.polynomial.legendre import leggauss

from .cartesian_pair_opening import real_array, materials_array
from .cartesian_pair_galerkin import _solve, _relative
from .material_sector import phase_from_angle, minimum_rotated_storage


def _trigonometric_polynomials(x, order, period, width):
    """A stable real basis for restricted Fourier modes -order..order.

    Even Legendre polynomials of sin(pi*x/period)/sin(pi*width/(2*period)),
    and cosine times the odd polynomials, span the same trigonometric space.
    Unlike restricted monomials or raw standing waves, they remain distinct
    on a narrow interval. Return values and two spatial derivatives.
    """
    frequency = np.pi/period
    angle = frequency*x
    u = np.sin(angle)/np.sin(frequency*width/2)
    du = frequency*np.cos(angle)/np.sin(frequency*width/2)
    ddu = -frequency**2*u
    p = np.zeros((len(x), 2*order+1))
    dp, ddp = np.zeros_like(p), np.zeros_like(p)
    p[:, 0] = 1.
    if order:
        p[:, 1], dp[:, 1] = u, 1.
    for k in range(1, 2*order):
        p[:, k+1] = ((2*k+1)*u*p[:, k]-k*p[:, k-1])/(k+1)
        dp[:, k+1] = ((2*k+1)*(p[:, k]+u*dp[:, k])-k*dp[:, k-1])/(k+1)
        ddp[:, k+1] = ((2*k+1)*(2*dp[:, k]+u*ddp[:, k])-k*ddp[:, k-1])/(k+1)
    value = p.copy()
    first, second = dp*du[:, None], ddp*du[:, None]**2+dp*ddu[:, None]
    c, dc, ddc = np.cos(angle), -frequency*np.sin(angle), -frequency**2*np.cos(angle)
    value[:, 1::2] = c[:, None]*p[:, 1::2]
    first[:, 1::2] = dc[:, None]*p[:, 1::2]+c[:, None]*dp[:, 1::2]*du[:, None]
    second[:, 1::2] = (ddc[:, None]*p[:, 1::2]+2*dc[:, None]*dp[:, 1::2]*du[:, None]
                       +c[:, None]*(ddp[:, 1::2]*du[:, None]**2+dp[:, 1::2]*ddu[:, None]))
    return value, first, second


class PeriodicSeamTrace:
    """Auxiliary normal response for a rectangular bar and its periodic gap.

    The retained fields have k_y=0. The potential is T_j(x)*f(y), where T is
    a tapered real Fourier trace and f joins 0 at the periodic seam to +/-g/2
    at the two body boundaries, linearly across the body. The corner taper
    shrinks like sqrt(g), so the body energy is o(g) and the full normal L2
    trace is available in the contact derivative. No material enters the
    basis. A common material half-plane is required, as in the other cells.

    ``normal_response`` returns only the yy constitutive matrix, its gap and
    material directional derivative, and diagnostics. It cannot be passed
    to solve_conv as a full layer. NumPy quadrature is explicit and checked
    independently; integrating these lifts with the full mesh is separate.
    """

    def __init__(self, order=3, *, quadrature=64, coercivity_angle=0.):
        if isinstance(order, (bool, np.bool_)) or not isinstance(order, Integral) or order < 0:
            raise ValueError('order must be a nonnegative integer')
        if (isinstance(quadrature, (bool, np.bool_)) or not isinstance(quadrature, Integral)
                or quadrature < max(16, 2*order+4)):
            raise ValueError('quadrature must be an integer at least max(16, 2*order+4)')
        self.order, self.quadrature = int(order), int(quadrature)
        self.phase = phase_from_angle(coercivity_angle)
        nodes, weights = leggauss(self.quadrature)
        self.nodes, self.weights = (nodes+1)/2, weights/2

    def _geometry(self, gap, period, width, offset):
        period = real_array(period, (2,), 'period')
        gap = real_array(gap, (), 'gap').item()
        width = real_array(width, (), 'width').item()
        offset = real_array(offset, (), 'offset').item()
        if np.any(period <= 0) or not 0 < width < period[0] or not 0 <= gap < period[1]:
            raise ValueError('positive period, internal width and gap in [0, period_y) required')
        length = np.sqrt(np.prod(period))
        px, py = period/length
        s, w, center = gap/length, width/length, px/2+offset/length
        points = w*(self.nodes-.5)
        basis = _trigonometric_polynomials(points, self.order, px, w)[0]
        _, transform = np.linalg.qr(np.sqrt(w*self.weights)[:, None]*basis)
        frame = np.linalg.solve(transform.T, basis.T).T
        gram = frame.T@((w*self.weights)[:, None]*frame)
        if np.linalg.norm(gram-np.eye(len(gram))) > 1e-10:
            raise ValueError('requested seam trace basis is numerically unresolved')
        return dict(length=length, px=px, py=py, s=s, width=w, center=center, transform=transform)

    def _basis(self, x, geometry):
        values = _trigonometric_polynomials(x, self.order, geometry['px'], geometry['width'])
        return tuple(np.linalg.solve(geometry['transform'].T, v.T).T for v in values)

    @staticmethod
    def _taper(geometry):
        s, py, w, px = (geometry[k] for k in ('s', 'py', 'width', 'px'))
        cap = min(w, px-w)/4
        raw = np.sqrt(s*(py-s)/12)
        draw = (py-2*s)/(24*raw)
        return cap*raw/(cap+raw), cap**2*draw/(cap+raw)**2

    def sample(self, points, *, gap, period=(700., 700.), width=350., offset=0.):
        """Physical-coordinate samples of the positive-gap real potentials.

        Return potentials in caller length units and dimensionless gradients.
        The endpoint has an L2 trace limit, not a pointwise gradient sample.
        """
        return self.sample_with_jvp(points, gap=gap, period=period, width=width,
                                    offset=offset, gap_tangent=0.)[:2]

    def sample_with_jvp(self, points, *, gap, period=(700., 700.), width=350., offset=0.,
                        gap_tangent=1., points_tangent=None):
        """Positive-gap field samples and their moving-point/gap derivative.

        Period, width and x translation stay fixed. Points may move with a
        quadrature chart. A point on a material y boundary must be evaluated
        from the intended integration side, since the normal field jumps.
        """
        geometry = self._geometry(gap, period, width, offset)
        if geometry['s'] == 0:
            raise ValueError('pointwise seam samples require a positive gap')
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 2 or points.dtype.kind not in 'iuf' or not np.isfinite(points).all():
            raise ValueError('finite real points with shape (count, 2) required')
        direction = (np.zeros_like(points, dtype=float) if points_tangent is None else
                     real_array(points_tangent, points.shape, 'points_tangent'))
        length, px, py, s, w = (geometry[k] for k in ('length', 'px', 'py', 's', 'width'))
        ds = real_array(gap_tangent, (), 'gap_tangent').item()/length
        dx, dy = (direction/length).T
        x = (points[:, 0]/length-geometry['center']+px/2) % px-px/2
        # Keep points near the upper seam as small negative coordinates.
        # Adding a period and subtracting it again loses a subpixel voltage.
        y = np.fmod(points[:, 1], py*length)/length
        y = np.where(y > py/2, y-py, np.where(y < -py/2, y+py, y))
        delta, delta_slope = self._taper(geometry)
        ddelta = delta_slope*ds
        fraction = np.clip((w/2+delta-np.abs(x))/(2*delta), 0., 1.)
        hat = fraction**2*(3-2*fraction)
        active = (np.abs(x) > w/2-delta) & (np.abs(x) < w/2+delta)
        dhat = np.where(active, -np.sign(x)*3*fraction*(1-fraction)/delta, 0.)
        dfraction = np.where(active, -np.sign(x)*dx/(2*delta)
                            -(w/2-np.abs(x))*ddelta/(2*delta**2), 0.)
        hat_change = 6*fraction*(1-fraction)*dfraction
        dhat_change = np.where(active, -np.sign(x)*3*(1-2*fraction)*dfraction/delta
                              +np.sign(x)*3*fraction*(1-fraction)*ddelta/delta**2, 0.)
        b, db, ddb = self._basis(x, geometry)
        trace, derivative = hat[:, None]*b, dhat[:, None]*b+hat[:, None]*db
        dtrace = hat_change[:, None]*b+hat[:, None]*db*dx[:, None]
        dderivative = (dhat_change[:, None]*b+dhat[:, None]*db*dx[:, None]
                       +hat_change[:, None]*db+hat[:, None]*ddb*dx[:, None])
        in_gap = np.abs(y) < s/2
        body_coordinate = np.sign(y)*py/2-y
        f = np.where(in_gap, y, s*body_coordinate/(py-s))
        df = np.where(in_gap, 1., -s/(py-s))
        f_change = np.where(in_gap, dy, ds*py*body_coordinate/(py-s)**2-s*dy/(py-s))
        df_change = np.where(in_gap, 0., -ds*py/(py-s)**2)
        gradient = np.stack([f[:, None]*derivative, df[:, None]*trace], axis=1)
        change = np.stack([f_change[:, None]*derivative+f[:, None]*dderivative,
                           df_change[:, None]*trace+df[:, None]*dtrace], axis=1)
        return (length*f[:, None]*trace, gradient,
                length*(f_change[:, None]*trace+f[:, None]*dtrace), change)

    def normal_response(self, *, gap, period=(700., 700.), width=350., epsilon=12.25,
                        background=1., offset=0., gap_tangent=1., epsilon_tangent=0.,
                        background_tangent=0.):
        geometry = self._geometry(gap, period, width, offset)
        material = materials_array((epsilon, epsilon), background, self.phase)
        bg, e = material[:2]
        de, dbg = np.asarray(epsilon_tangent), np.asarray(background_tangent)
        if any(v.shape != () or v.dtype.kind not in 'iufc' or not np.isfinite(v) for v in (de, dbg)):
            raise ValueError('finite scalar material tangents required')
        ds = real_array(gap_tangent, (), 'gap_tangent').item()/geometry['length']
        s, px, py, w, center = (geometry[k] for k in ('s', 'px', 'py', 'width', 'center'))
        modes = np.arange(-self.order, self.order+1)
        count = len(modes)
        wave = 2*np.pi*modes/px
        difference = modes[:, None]-modes
        indicator = w*np.sinc(difference*w/px)*np.exp(-2j*np.pi*difference*center/px)
        a = bg*np.eye(count)+(e-bg)*(py-s)*indicator
        da = dbg*np.eye(count)+((de-dbg)*(py-s)-(e-bg)*ds)*indicator
        moment = np.zeros((count, count), complex)
        dmoment = np.zeros_like(moment)
        mass, dmass, stiffness, dstiffness = (np.zeros((2, count, count)) for _ in range(4))
        if s == 0:
            segments = [(-w/2, w/2, 0., 0., 1., 1., 1)]
        else:
            delta, ddelta = self._taper(geometry)
            knots = [-w/2-delta, -w/2, -w/2+delta, w/2-delta, w/2, w/2+delta]
            speed = [-ddelta, 0., ddelta, -ddelta, 0., ddelta]
            values = [0., .5, 1., 1., .5, 0.]
            segments = [(knots[i], knots[i+1], speed[i], speed[i+1], values[i], values[i+1],
                         int(1 <= i <= 3)) for i in range(5)]
        for left, right, dleft, dright, low, high, tag in segments:
            x = left+(right-left)*self.nodes
            dx = (dleft+(dright-dleft)*self.nodes)*ds
            weight, dweight = (right-left)*self.weights, (dright-dleft)*self.weights*ds
            fraction = low+(high-low)*self.nodes
            hat = fraction**2*(3-2*fraction)
            dhat = 6*fraction*(1-fraction)*(high-low)/(right-left)
            ddhat = -dhat*(dright-dleft)*ds/(right-left)
            b, db, ddb = self._basis(x, geometry)
            value, dvalue = hat[:, None]*b, hat[:, None]*db*dx[:, None]
            first = dhat[:, None]*b+hat[:, None]*db
            dfirst = ddhat[:, None]*b+dhat[:, None]*db*dx[:, None]+hat[:, None]*ddb*dx[:, None]
            phase = np.exp(-1j*(x+center)[:, None]*wave)
            dphase = -1j*dx[:, None]*wave*phase
            for result, change, v, dv in ((mass, dmass, value, dvalue),
                                          (stiffness, dstiffness, first, dfirst)):
                result[tag] += v.T@(weight[:, None]*v)
                change[tag] += dv.T@(weight[:, None]*v)+v.T@(dweight[:, None]*v+weight[:, None]*dv)
            if tag:
                moment += phase.T@(weight[:, None]*value)
                dmoment += dphase.T@(weight[:, None]*value)+phase.T@(
                    dweight[:, None]*value+weight[:, None]*dvalue)
        r, dr = (bg-e)*moment, (dbg-de)*moment+(bg-e)*dmoment
        # K/s is formed directly. In particular, no difference of two O(1)
        # responses is divided by a subpixel gap to obtain its contact slope.
        m, dm = bg*mass.sum(0), dbg*mass.sum(0)+bg*dmass.sum(0)
        me, dme = bg*mass[0]+e*mass[1], dbg*mass[0]+de*mass[1]+bg*dmass[0]+e*dmass[1]
        kx, dkx = bg*stiffness.sum(0), dbg*stiffness.sum(0)+bg*dstiffness.sum(0)
        kxe = bg*stiffness[0]+e*stiffness[1]
        dkxe = dbg*stiffness[0]+de*stiffness[1]+bg*dstiffness[0]+e*dstiffness[1]
        h = py-s
        k = m+s/h*me+s*s/12*kx+s*h/12*kxe
        source = r[::-1].T
        u = _solve(k, source)
        response = a-s*r@u
        weak = _relative(k@u-source, k@u, source)
        if s == 0:
            # The original energy divided by gap has a finite limit. Its
            # derivative need not: the taper contributes O(sqrt(gap)). Only
            # this finite limit enters the physical response's first derivative.
            derivative = da-ds*r@u
            dweak = None
        else:
            dk = (dm+ds*py/h**2*me+s/h*dme+ds*s/6*kx+s*s/12*dkx
                  +ds*(py-2*s)/12*kxe+s*h/12*dkxe)
            dsource = dr[::-1].T
            du = _solve(k, dsource-dk@u)
            derivative = da-ds*r@u-s*dr@u-s*r@du
            dweak = _relative(dk@u+k@du-dsource, dk@u, k@du, dsource)
        reciprocity = _relative(response-response[::-1, ::-1].T, response)
        dreciprocity = _relative(derivative-derivative[::-1, ::-1].T, derivative)
        storage = minimum_rotated_storage(response, self.phase)
        loss = np.linalg.eigvalsh((response-response.conj().T)/2j)[0]
        if (not all(np.isfinite(v).all() for v in (response, derivative))
                or max(weak, dweak or 0.) > 1e-10 or max(reciprocity, dreciprocity) > 1e-9
                or storage <= 0 or loss < -1e-9):
            raise ValueError('seam trace response failed original/physical/reciprocity guards')
        return response, derivative, dict(
            component='periodic_seam_normal_trace', order=self.order, quadrature=self.quadrature,
            gap=gap, solve_condition=np.linalg.cond(k), weak_residual=weak,
            weak_derivative_residual=dweak, relative_reciprocity_defect=reciprocity,
            derivative_reciprocity_defect=dreciprocity,
            minimum_rotated_storage=storage, minimum_dissipation=loss)
