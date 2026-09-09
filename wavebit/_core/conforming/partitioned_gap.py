"""Conforming rectangular NVF cell: partitioned gap."""
import math
import numpy as np

from .thin_gap import PERIOD, ThinGapEnergyModel, _line_mass, _line_stiffness


def interval_moments(nodes, lengths, lo, hi, modes, xp=np):
    """Integrals of r**(0,1,2) exp(-2*pi*i*k*x), r=(x-node)/length.

    lo/hi are local coordinates in [0,1]. Zero-width intersections are retained
    and differentiable. Centered moments use convergent Taylor series near zero
    to avoid cancellation in the first and second sinc derivatives.
    """
    span = lengths*(hi-lo)
    middle = (lo+hi)/2
    center = nodes+lengths*middle
    z = np.pi*xp.asarray(modes)[:, None]*span[None]
    small = xp.abs(z) < .5
    safe = xp.where(small, 1., z)
    first = .5j*(safe*xp.cos(safe)-xp.sin(safe))/safe**2
    second = (2*safe*xp.cos(safe)+(safe**2-2)*xp.sin(safe))/(4*safe**3)
    # Evaluate the unused polynomial branch at zero to prevent overflow in a
    # large-frequency JAX trace. Both branches have finite derivatives at zero.
    series_z = xp.where(small, z, 0.)
    first_series = sum((-.5j)*(-1)**j*series_z**(2*j+1)
                       / (math.factorial(2*j+1)*(2*j+3)) for j in range(9))
    second_series = sum(.25*(-1)**j*series_z**(2*j)
                        / (math.factorial(2*j)*(2*j+3)) for j in range(9))
    first, second = xp.where(small, first_series, first), xp.where(small, second_series, second)
    common = span[None]*xp.exp(-2j*np.pi*xp.asarray(modes)[:, None]*center[None])
    zero = xp.sinc(z/np.pi)
    local_span = hi-lo
    return xp.stack([common*zero,
                     common*(middle*zero+local_span*first),
                     common*(middle**2*zero+2*middle*local_span*first+local_span**2*second)], axis=-1)


def moment_matrices(moments, lengths, xp=np):
    """Weighted P1 mass, stiffness and linear forms for every Fourier mode."""
    zero, first, second = (moments[..., i] for i in range(3))
    count = len(lengths)
    identity = xp.eye(count)
    diagonal = lambda v: v[..., :, None]*identity
    off = xp.roll(diagonal(first-second), 1, axis=-1)
    mass = diagonal(zero-2*first+second+xp.roll(second, 1, axis=-1))+off+xp.swapaxes(off, -1, -2)
    stiffness_weights = zero/lengths**2
    off = xp.roll(diagonal(stiffness_weights), 1, axis=-1)
    stiffness = diagonal(stiffness_weights+xp.roll(stiffness_weights, 1, axis=-1))-off-xp.swapaxes(off, -1, -2)
    load = zero-first+xp.roll(first, 1, axis=-1)
    derivative_load = xp.roll(zero/lengths, 1, axis=-1)-zero/lengths
    return mass, stiffness, load, derivative_load


def _project_strip(left, right, normal, ymoments, xmoments, lengths, xp,
                   *, derivative_moments=None, stiffness_moments=None,
                   lift_moments=None, lift_squared_moments=None, lift_derivative_moments=None):
    """Project a bilinear strip's weighted field, energy and material.

    left/right are potential coefficients on the y endpoints. normal is their
    physical normal derivative, supplied directly for a zero-width gap.
    Use background moments for the unweighted change-of-basis field C(B).
    """
    x0, x1, x2 = (xmoments[..., i] for i in range(3))
    y0, y1, y2 = (ymoments[..., i] for i in range(3))

    def difference(a):
        return xp.roll(a, -1, axis=1)-a

    def product(weight, a, b):
        return xp.einsum('mi,eic,eid->emcd', weight, a, b)

    def stiffness(a, b):
        # Form local potential differences before squaring. Multiplying U.T K U
        # with an assembled stiffness loses digits on the tiny interface cells.
        if stiffness_moments is not None:
            return product(stiffness_moments, difference(a), difference(b))
        return product(x0, difference(a)/lengths[None, :, None],
                       difference(b)/lengths[None, :, None])

    def mass(a, b):
        da, db = difference(a), difference(b)
        return product(x0, a, b)+product(x1, a, db)+product(x1, da, b)+product(x2, da, db)

    def integrate(weights, value):
        return xp.einsum('ne,emcd->nmcd', weights, value)

    energy = (integrate(y0-2*y1+y2, stiffness(left, left))
              +integrate(y1-y2, stiffness(left, right)+stiffness(right, left))
              +integrate(y2, stiffness(right, right))
              +integrate(y0, mass(normal, normal)))
    if derivative_moments is None:
        dx = (xp.einsum('ne,mi,eic->nmc', y0-y1, x0, difference(left)/lengths[None, :, None])
              +xp.einsum('ne,mi,eic->nmc', y1, x0, difference(right)/lengths[None, :, None]))
    else:
        dx = (xp.einsum('ne,mi,eic->nmc', y0-y1, derivative_moments, difference(left))
              +xp.einsum('ne,mi,eic->nmc', y1, derivative_moments, difference(right)))
    dy = (xp.einsum('ne,mi,eic->nmc', y0, x0, normal)
          +xp.einsum('ne,mi,eic->nmc', y0, x1, difference(normal)))
    linear = xp.stack([dx, dy], axis=-2)
    material = xp.sum(y0, axis=1)[:, None]*xp.sum(xmoments[..., 0], axis=1)[None]
    if lift_moments is None:
        gram = energy+linear+xp.swapaxes(linear, -1, -2)+material[..., None, None]*xp.eye(2)
        field = linear+material[..., None, None]*xp.eye(2)
    else:
        # A material-harmonic coordinate also needs its affine lift. Only the
        # total quasi-periodic potential (lift + periodic correction) has the
        # required field jump; changing the correction basis alone is wrong.
        lx = (xp.einsum('ne,mi,eic->nmc', y0-y1, lift_derivative_moments, difference(left))
              +xp.einsum('ne,mi,eic->nmc', y1, lift_derivative_moments, difference(right)))
        energy_linear = xp.stack([lx, dy], axis=-2)
        lift = xp.sum(y0, axis=1)[:, None]*xp.sum(lift_moments, axis=1)[None]
        lift_squared = xp.sum(y0, axis=1)[:, None]*xp.sum(lift_squared_moments, axis=1)[None]
        ex, ey = xp.diag(xp.asarray([1., 0.])), xp.diag(xp.asarray([0., 1.]))
        gram = (energy+energy_linear+xp.swapaxes(energy_linear, -1, -2)
                +lift_squared[..., None, None]*ex+material[..., None, None]*ey)
        field = linear+lift[..., None, None]*ex+material[..., None, None]*ey
    return field, gram, material


class PartitionedGapEnergyModel(ThinGapEnergyModel):
    """The same conforming space, with an independently partitioned material.

    Pass epsilon as a scalar or a dictionary with `seams` (ordered x positions
    in nm) and `values` (one more positive permittivity than seams). These fill
    the centered envelope of the supplied width. Geometry, seams and each
    material value remain live under JAX. Mesh density depends on the envelope,
    not on the number, positions or equality of the interior material pieces.

    An unequal interior interface is unfitted to this mesh. Representation
    invariance does not assert accuracy for that unresolved material interface.
    """

    def shape_coefficients(self, shapes, xp=np):
        """Adapter for an ordered horizontal tiling of translated rectangles.

        Study dictionaries use lx, ly, epsilon and optional cx/cy/angle. The
        envelope must be rectangular; every piece has the same height and y
        center, with no gap or overlap in x. This is not a general CSG adapter.
        """
        if not shapes:
            raise ValueError('the shape adapter needs at least one rectangle')
        if any(shape.get('type', 'rectangle') != 'rectangle' for shape in shapes):
            raise ValueError('the shape adapter only supports rectangles')
        widths = xp.asarray([shape['lx'] for shape in shapes])
        heights = xp.asarray([shape['ly'] for shape in shapes])
        xs = xp.asarray([shape.get('cx', 0.) for shape in shapes])
        ys = xp.asarray([shape.get('cy', 0.) for shape in shapes])
        angles = xp.asarray([shape.get('angle', 0.) for shape in shapes])
        values = xp.asarray([shape['epsilon'] for shape in shapes])
        left, right = xs-widths/2, xs+widths/2
        width, cx, cy = right[-1]-left[0], (right[-1]+left[0])/2, ys[0]
        # Only floating-point roundoff in an already contiguous tiling is
        # allowed; this adapter does not remove physical internal gaps.
        tolerance = 32*np.finfo(float).eps*PERIOD
        valid = (xp.all(widths > 0) & xp.all(heights > 0) & xp.all(angles == 0)
                 & xp.all(xp.abs(heights-heights[0]) <= tolerance)
                 & xp.all(xp.abs(ys-cy) <= tolerance)
                 & xp.all(xp.abs(right[:-1]-left[1:]) <= tolerance)
                 & xp.all(xp.isfinite(xp.stack([widths, heights, xs, ys, angles]))))
        if xp is np and not valid:
            raise ValueError('rectangles must form an ordered, axis-aligned horizontal tiling')
        seams = (right[:-1]+left[1:])/2-cx
        coefficients, metadata = self.coefficients(PERIOD-heights[0],
                                                    dict(seams=seams, values=values), width, xp)
        phase = xp.exp(-2j*np.pi*(xp.asarray(self.modes)[:, None]*cy
                                  +xp.asarray(self.modes)[None]*cx)/PERIOD)
        return xp.where(valid, coefficients*phase[None], xp.nan), metadata

    def shape_matrices(self, shapes, xp=np):
        return self._matrices_from_coefficients(*self.shape_coefficients(shapes, xp), xp)


    def _profile_geometry(self, gap, epsilon, width, xp):
        if isinstance(epsilon, dict):
            seams = xp.asarray(epsilon['seams'])
            values = xp.asarray(epsilon['values'])
        else:
            seams, values = xp.asarray([]), xp.asarray([epsilon])
        if seams.ndim != 1 or values.shape != (len(seams)+1,):
            raise ValueError('profile needs an ordered seam vector and one more material value')
        if xp is not np and xp.asarray(0.).dtype != np.dtype('float64'):
            raise ValueError('the partitioned-gap study requires JAX x64')
        edges = xp.concatenate([xp.asarray([-width/2]), seams, xp.asarray([width/2])])/PERIOD
        g, w = gap/PERIOD, width/PERIOD
        valid = (xp.all(xp.isfinite(edges)) & xp.all(xp.isfinite(values)) & xp.all(values > 0)
                 & xp.all(edges[1:] > edges[:-1]) & (w > 0) & (w < 1) & (g >= 0) & (g < 1))
        if xp is np and not valid:
            raise ValueError('profile requires finite positive materials and strictly ordered interior seams')
        starts = xp.asarray([-.5, -w/2, w/2])
        sizes = xp.asarray([(1-w)/2, w, (1-w)/2])
        nodes = (starts[:, None]+sizes[:, None]*xp.asarray(self.fractions[:-1])).ravel()
        ends = (starts[:, None]+sizes[:, None]*xp.asarray(self.fractions[1:])).ravel()
        lengths = ends-nodes
        return g, edges, values, nodes, lengths, valid

    def coefficients(self, gap, epsilon=12.25, width=350., xp=np):
        g, edges, values, nodes, lengths, valid = self._profile_geometry(gap, epsilon, width, xp)
        background = interval_moments(nodes, lengths, xp.zeros_like(nodes), xp.ones_like(nodes), self.modes, xp)
        base_eps = xp.where(np.arange(self.nx)//self.per_region == 1, values[0], 1.)
        material = background*base_eps[None, :, None]
        # Preserve the original envelope's exact static assembly, independent
        # of the Fourier integral evaluation order on its smallest elements.
        mb, kb = _line_mass(lengths, xp), _line_stiffness(1/lengths, xp)
        lb = (lengths+xp.roll(lengths, 1))/2
        weighted = lengths*base_eps
        me, ke = _line_mass(weighted, xp), _line_stiffness(base_eps/lengths, xp)
        le, de = (weighted+xp.roll(weighted, 1))/2, xp.roll(base_eps, 1)-base_eps
        # Telescoping material jumps: eps0 fills the envelope, and each later
        # contrast fills the suffix to its right. An equal-material seam has an
        # exactly zero weight; derivatives of the independent values stay live.
        # No equality predicate, conditional merging, or mesh change is needed.
        for i in range(1, len(values)):
            lo = xp.clip((edges[i]-nodes)/lengths, 0., 1.)
            hi = xp.clip((edges[-1]-nodes)/lengths, 0., 1.)
            increment = interval_moments(nodes, lengths, lo, hi, self.modes, xp)
            contrast = values[i]-values[i-1]
            material = material+contrast*increment
            mi, ki, li, di = (v[2*self.order].real for v in moment_matrices(increment, lengths, xp))
            me, ke, le, de = me+contrast*mi, ke+contrast*ki, le+contrast*li, de+contrast*di
        v, a, residual = self._solve_cell(g, me, ke, le, de, mb, kb, lb, xp)
        v, a = xp.where(valid, v, xp.nan), xp.where(valid, a, xp.nan)
        return self._project_cell(g, lengths, dict(mass=background), dict(mass=material), v, a, residual, xp)

    def _project_cell(self, g, lengths, background, material, v, a, residual, xp):
        t = xp.asarray(self.yedges)-.5
        dt = xp.asarray(np.diff(self.yedges))
        following = xp.roll(v, -1, axis=0)
        body_left, body_right = v-g*t[:-1, None, None]*a, following-g*t[1:, None, None]*a
        normal = (following-v)/((1-g)*dt[:, None, None])-g/(1-g)*a
        body_y = interval_moments((1-g)*t[:-1], (1-g)*dt,
                                  xp.zeros_like(dt), xp.ones_like(dt), self.modes, xp)
        gap_left, gap_right = (v[0]-g/2*a)[None], (v[0]+g/2*a)[None]
        gap_y = interval_moments(xp.asarray([.5-g/2]), xp.asarray([g]),
                                 xp.asarray([0.]), xp.asarray([1.]), self.modes, xp)
        def project(left, right, normal, ymoments, pack):
            return _project_strip(left, right, normal, ymoments, pack['mass'], lengths, xp,
                                   derivative_moments=pack.get('derivative'),
                                   stiffness_moments=pack.get('stiffness'),
                                   lift_moments=pack.get('lift'),
                                   lift_squared_moments=pack.get('lift_squared'),
                                   lift_derivative_moments=pack.get('lift_derivative'))
        field, _, _ = project(body_left, body_right, normal, body_y, background)
        _, gram, eps = project(body_left, body_right, normal, body_y, material)
        gap_field, gap_gram, gap_eps = project(gap_left, gap_right, a[None], gap_y, background)
        field, gram, eps = field+gap_field, gram+gap_gram, eps+gap_eps
        coefficients = xp.stack([field[:, :, 0, 0], field[:, :, 0, 1], field[:, :, 1, 0], field[:, :, 1, 1],
                                 gram[:, :, 0, 0], gram[:, :, 0, 1], gram[:, :, 1, 1], eps])
        return coefficients, dict(cell_residual=residual, degrees_of_freedom=self.nx*(self.rows+1)-1)
