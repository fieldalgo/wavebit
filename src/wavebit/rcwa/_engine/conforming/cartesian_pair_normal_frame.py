"""Experimental material-referenced normal frame on fitted rectangle quadrature.

No electrostatic auxiliary solve is needed. This explicit NumPy/JAX x64
instrument keeps the material and normal products inside their physical
integration regions, including the exact right-opening strip at contact.
It does not change ordinary NVF routing or certify optical convergence.
"""
from numbers import Integral
import numpy as np

import wavebit.rcwa._engine.parametric as P
from wavebit.rcwa._engine.factorization import ConvMatrices
from wavebit.lattice import as_lattice
from wavebit.rcwa._engine.conforming.cartesian_pair import _require
from wavebit.rcwa._engine.conforming.cartesian_pair_opening import validate_geometry
from wavebit.rcwa._engine.conforming.material_sector import passive_pair_materials
from wavebit.rcwa._engine.conforming.material_sector import phase_from_angle
from wavebit.rcwa._engine.conforming.material_sector import minimum_rotated_storage


# NumPy field temporaries use a fixed row count rather than the full q-by-q
# tensor grid. JAX retains one array expression so tracing does not unroll
# the field geometry once per tile; no JAX memory improvement is claimed.
_QUADRATURE_TILE_ROWS = 64


def _reciprocity_defects(tensor, xp):
    """Lorentz transpose defect with both Cartesian Fourier sets reversed.

    A passive finite energy congruence need not be reciprocal when its
    complex trial space is not closed under conjugation. Report the defect;
    do not project it out or reinterpret an algebraic guard as certification.
    """
    count = tensor.shape[0]//2
    reverse = xp.concatenate((xp.arange(count-1, -1, -1), xp.arange(2*count-1, count-1, -1)))
    defect = tensor-tensor[reverse][:, reverse].T
    return dict(reciprocity_error=xp.max(xp.abs(defect)),
                relative_reciprocity_defect=xp.linalg.norm(defect)/xp.maximum(xp.linalg.norm(tensor), 1.))


def _order_pair(order):
    pair = (order, order) if isinstance(order, Integral) else order
    if (not isinstance(pair, (tuple, list)) or len(pair) != 2
            or any(isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral) or n < 0 for n in pair)):
        raise ValueError('order must be a nonnegative integer or pair')
    return tuple(map(int, pair))


class CartesianPairNormalFrame:
    """Fixed axis-aligned cell geometry with live gap and scalar permittivities.

    order is the two-dimensional Fourier order, not a finite-element degree.
    quadrature is an even Gauss order per fitted interval. Compare quadrature
    and Fourier orders separately; their minima are admission rules only.
    Gap derivatives at zero are right derivatives. NumPy supplies values;
    ordinary JAX x64 AD supplies gap and material derivatives of this finite
    quadrature. Geometry other than gap is fixed by construction.
    The validity flag covers the declared algebraic guards, not reciprocity
    or optical accuracy. Complex frames report their reciprocity defect.
    """

    def __init__(self, *, basis, widths, height, offset=(0., 0.), order=1,
                 quadrature=32, coercivity_angle=0.):
        geometry = validate_geometry(basis, widths, height, offset)
        a, w, h, o = (geometry[k] for k in ('basis', 'widths', 'height', 'offset'))
        if a[0, 1] != 0 or a[1, 0] != 0 or np.any(np.diag(a) <= 0):
            raise ValueError('the normal-frame instrument requires positive diagonal lattice columns')
        if w.sum() >= a[0, 0] or h > a[1, 1]:
            raise ValueError('the pair must leave horizontal background and fit vertically')
        order = _order_pair(order)
        if (isinstance(quadrature, (bool, np.bool_)) or not isinstance(quadrature, Integral)
                or quadrature % 2 or quadrature < max(8, 2*max(order)+4)):
            raise ValueError('quadrature must be even and at least max(8, 2*max(order)+4)')
        self._phase = phase_from_angle(coercivity_angle)
        self._angle = float(coercivity_angle)
        self._geometry = tuple(tuple(np.asarray(v).ravel()) for v in (a, w, h, o))
        self._order, self._quadrature = order, int(quadrature)
        nodes, weights = np.polynomial.legendre.leggauss(self.quadrature)
        self._nodes, self._weights = tuple(nodes), tuple(weights)
        lattice = as_lattice(lattice=a)
        shapes = self._shapes(0., (1., 1.))
        plan = P.plan_gamma(shapes, lattice)
        # Cross-shape coincidence substitutions freeze a contact constraint.
        # The explicit gap family must replay the unsnapped physical distances;
        # self-periodic records remain valid because each width/height is fixed.
        pairs = tuple(pair for pair in plan.pairs
                      if plan.pieces[pair.p].shape == plan.pieces[pair.q].shape)
        self._plan = plan._replace(pairs=pairs)

    @property
    def order(self):
        return self._order

    @property
    def quadrature(self):
        return self._quadrature

    @property
    def coercivity_angle(self):
        return self._angle

    def _shapes(self, gap, epsilon):
        _, widths, height, offset = self._geometry
        wa, wb = widths
        return [dict(type='rectangle', lx=width, ly=height[0], cy=offset[1],
                     cx=offset[0]+center, material=e)
                for width, center, e in ((wa, -(wb+gap)/2, epsilon[0]),
                                         (wb, (wa+gap)/2, epsilon[1]))]

    def coefficients(self, *, gap, epsilon, background=1., xp=np):
        if xp is not np and getattr(xp, '__name__', '') != 'jax.numpy':
            raise NotImplementedError('normal-frame construction supports NumPy and JAX x64')
        if xp.asarray(0.).dtype != np.dtype('float64'):
            raise ValueError('normal-frame construction requires 64-bit precision')
        gap = xp.asarray(gap)
        if gap.shape != () or gap.dtype.kind not in 'iuf':
            raise ValueError('gap must be a real numeric scalar')
        raw, material_valid = passive_pair_materials(epsilon, background, self._phase, xp)
        materials = xp.where(material_valid, raw, xp.ones_like(raw))
        a, widths, height, offset = self._geometry
        px, py = a[0], a[3]
        wa, wb = widths
        h, = height
        ox, oy = offset
        valid_gap = xp.isfinite(gap) & (gap >= 0) & (gap < px-wa-wb)
        _require(valid_gap, 'gap must be finite, nonnegative and leave horizontal background')
        s = xp.where(valid_gap, gap, 0.)
        shapes = self._shapes(s, materials[1:])
        lattice = xp.asarray(a).reshape(2, 2)
        q = self.quadrature
        nodes, weights = xp.asarray(self._nodes), xp.asarray(self._weights)
        xcuts = xp.stack([xp.asarray(0.), (px-wa-wb-s)/2, (px+wa-wb-s)/2,
                         (px+wa-wb+s)/2, (px+wa+wb+s)/2, xp.asarray(px)])+ox
        ycuts = xp.asarray([0., (py-h)/2, (py+h)/2, py])+oy
        def interval_points(cuts):
            span = cuts[1:]-cuts[:-1]
            points = (cuts[1:, None]+cuts[:-1, None])/2+span[:, None]*nodes/2
            return points.reshape(-1), (span[:, None]*weights/2).reshape(-1)
        x, wx = interval_points(xcuts)
        y, wy = interval_points(ycuts)
        column = xp.arange(5*q)[None, :]//q
        reference = (materials[1]+materials[2])/2
        hu, hv = self.order
        u, v = xp.arange(-2*hu, 2*hu+1), xp.arange(-2*hv, 2*hv+1)
        fx = xp.exp(-2j*xp.pi*x[:, None]*u[None, :]/px)
        fy = xp.exp(-2j*xp.pi*v[:, None]*y[None, :]/py)
        tile_rows = min(_QUADRATURE_TILE_ROWS, len(y)) if xp is np else len(y)
        accumulated = None
        for first in range(0, len(y), tile_rows):
            last = min(first+tile_rows, len(y))
            X, Y = xp.meshgrid(x, y[first:last])
            row = xp.arange(first, last)[:, None]//q
            strip = (row == 1) & (column == 2)
            collapsed = (s == 0) & strip
            # A zero-width strip has zero value weight but nonzero opening
            # weight. Its limiting normal is x, independently of the normal
            # tie on the closed seam. Do not differentiate that tie.
            field_x = xp.where(collapsed, ox+px/2-wb/2, X)
            nx, ny = P._piece_field(xp, field_x, Y, shapes, self._plan, materials[1:], lattice, xp.float64)
            xx = xp.where(collapsed, 1., xp.real(nx*xp.conj(nx)))
            yy = xp.where(collapsed, 0., xp.real(ny*xp.conj(ny)))
            xy = xp.where(collapsed, 0., nx*xp.conj(ny))
            yx = xp.conj(xy)
            local = xp.where((row == 1) & (column == 1), materials[1],
                             xp.where((row == 1) & (column == 3), materials[2], materials[0]))
            # B=I+(reference/epsilon-1)P, with unit Hermitian P.
            # epsilon B^H B=epsilon I+(|reference|^2/conj(epsilon)-epsilon)P.
            change = reference/local-1
            energy = xp.real(reference*xp.conj(reference))/xp.conj(local)-local
            def fourier(field):
                return (fy[:, first:last]@(wy[first:last, None]*field*wx[None, :])@fx)/(px*py)
            current = xp.stack([fourier(change*xx), fourier(change*xy),
                                fourier(change*yx), fourier(change*yy),
                                fourier(energy*xx), fourier(energy*xy),
                                fourier(energy*yx), fourier(energy*yy), fourier(xp.ones_like(X))])
            accumulated = current if accumulated is None else accumulated+current
        direct, _ = P.assemble_eps_coeffs(xp, shapes, materials[0], hu, hv, lattice, xp.complex128)
        identity = xp.asarray((np.arange(-2*hv, 2*hv+1)[:, None] == 0)
                              & (np.arange(-2*hu, 2*hu+1)[None, :] == 0), dtype=xp.complex128)
        value = xp.stack([identity+accumulated[0], accumulated[1], accumulated[2],
                          identity+accumulated[3], direct+accumulated[4], accumulated[5],
                          direct+accumulated[7], direct, accumulated[6]])
        # The constant Fourier integral is an independent quadrature resolution
        # check for the analytic base terms. It cannot certify nonpolynomial P.
        constant_error = xp.max(xp.abs(accumulated[8]-identity))
        valid = valid_gap & material_valid & xp.all(xp.isfinite(value)) & (constant_error < 1e-11)
        _require(valid, 'normal-frame coefficients failed finite/base-integral guards')
        meta = dict(valid=valid, gap=gap, order=self.order, quadrature=q,
                    reference_epsilon=reference, constant_integral_error=constant_error,
                    coercivity_angle=self.coercivity_angle, sample_count=15*q*q,
                    field_tile_rows=tile_rows, maximum_field_samples=tile_rows*5*q,
                    derivative_units='per caller length unit')
        return xp.where(valid, value, xp.nan), meta

    def evaluate(self, *, gap, epsilon, background=1., xp=np):
        value, meta = self.coefficients(gap=gap, epsilon=epsilon, background=background, xp=xp)
        hu, hv = self.order
        v, u = xp.meshgrid(xp.arange(-hv, hv+1), xp.arange(-hu, hu+1), indexing='ij')
        dn, dm = v.reshape(-1, 1)-v.reshape(1, -1)+2*hv, u.reshape(-1, 1)-u.reshape(1, -1)+2*hu
        delta = ((xp.arange(-2*hv, 2*hv+1)[:, None] == 0)
                 & (xp.arange(-2*hu, 2*hu+1)[None, :] == 0))
        fallback = xp.asarray([1, 0, 0, 1, 1, 0, 1, 1, 0])[:, None, None]*delta
        c = xp.where(meta['valid'], value, fallback)[:, dn, dm]
        change = xp.block([[c[0], c[1]], [c[2], c[3]]])
        gram = xp.block([[c[4], c[5]], [c[8], c[6]]])
        # A common material half-plane also supplies a half-plane for B.
        # This excludes the uncontrolled zero determinant of a cell corrector.
        reference = meta['reference_epsilon']
        phase = xp.conj(reference)/xp.abs(reference)*np.conj(self._phase)
        physical = xp.concatenate([xp.reshape(xp.asarray(background), (1,)), xp.asarray(epsilon)])
        safe_physical = xp.where(meta['valid'], physical, xp.ones_like(physical))
        scalar_min = xp.minimum(xp.real(phase), xp.min(xp.real(phase*reference/safe_physical)))
        # The analytic identity term differs from its sampled Fourier matrix
        # by at most mode_count*max_coefficient_error in spectral norm.
        quadrature_margin = len(c[7])*(xp.abs(xp.real(phase))+scalar_min)*meta['constant_integral_error']
        certified_min = scalar_min-quadrature_margin
        frame_min = minimum_rotated_storage(change, phase, xp)
        frame_valid = xp.isfinite(frame_min) & (frame_min > 0) & (certified_min > 0)
        _require(frame_valid, 'compressed normal frame failed coercivity')
        safe_change = xp.where(frame_valid, change, xp.eye(len(change)))
        slots = ConvMatrices.from_energy(safe_change, gram, c[7])
        tensor = xp.block([[slots.epx, -slots.epxy], [-slots.epyx, slots.epy]])
        rotated = minimum_rotated_storage(tensor, self._phase, xp)
        loss = xp.linalg.eigvalsh((tensor-tensor.conj().T)/(2j))[0]
        lossless = xp.all(xp.imag(safe_physical) == 0)
        hermitian_error = xp.max(xp.abs(tensor-tensor.conj().T))
        inverse_residual = xp.linalg.norm(change.conj().T@tensor@change-gram)/xp.maximum(xp.linalg.norm(gram), 1.)
        z_residual = xp.linalg.norm(c[7]@slots.epz_i-xp.eye(len(c[7])))
        valid = (meta['valid'] & frame_valid & xp.all(xp.isfinite(tensor))
                 & (rotated > 0) & (loss > -1e-9)
                 & (~lossless | (hermitian_error < 1e-9))
                 & (inverse_residual <= 1e-10) & (z_residual <= 1e-10))
        _require(valid, 'normal-frame closure failed physical/inverse guards')
        meta = dict(meta, valid=valid, frame_minimum_rotated_storage=frame_min,
                    frame_condition=xp.linalg.cond(change), frame_storage_lower_bound=certified_min,
                    lossless=lossless, hermitian_error=hermitian_error, minimum_rotated_storage=rotated,
                    minimum_dissipation=loss, inverse_residual=inverse_residual,
                    z_inverse_residual=z_residual, **_reciprocity_defects(tensor, xp))
        return (xp.where(valid, value, xp.nan),
                ConvMatrices(*(xp.where(valid, matrix, xp.nan)[None] for matrix in slots)), meta)

    def paired_closure(self, *, gap, epsilon, background=1., xp=np):
        """Explicit reciprocal energy of conjugate-paired normalized fields.

        For F=B C(B)^(-1), average the physical energies of F and its
        conjugate with reversed Fourier labels. This distinct finite model
        gives (H+J H.T J)/2. Both fields have identity retained Fourier
        projection; their averaged energies preserve passivity and rotated
        coercivity while satisfying reciprocity exactly.

        Returns (slots, metadata), not coefficient grids for the original
        pointwise B. Raw frame diagnostics remain under metadata['raw'],
        including its inverse residual and unmodified reciprocity defect.
        The paired closure has its own physical guards. No failed raw guard
        is repaired. This construction does not certify optical convergence
        or equivalence to a full Maxwell variational discretization.
        """
        _, original, raw = self.evaluate(gap=gap, epsilon=epsilon, background=background, xp=xp)
        tensor = xp.block([[original.epx[0], -original.epxy[0]],
                           [-original.epyx[0], original.epy[0]]])
        count = len(original.epx[0])
        reverse = xp.concatenate((xp.arange(count-1, -1, -1), xp.arange(2*count-1, count-1, -1)))
        paired = (tensor+tensor[reverse][:, reverse].T)/2
        reciprocity = _reciprocity_defects(paired, xp)
        rotated = minimum_rotated_storage(paired, self._phase, xp)
        loss = xp.linalg.eigvalsh((paired-paired.conj().T)/(2j))[0]
        hermitian_error = xp.max(xp.abs(paired-paired.conj().T))
        valid = (raw['valid'] & xp.all(xp.isfinite(paired)) & (rotated > 0) & (loss > -1e-9)
                 & (reciprocity['reciprocity_error'] < 1e-9)
                 & (~raw['lossless'] | (hermitian_error < 1e-9)))
        _require(valid, 'paired normal-frame closure failed raw/physical/reciprocity guards')
        slots = ConvMatrices(paired[:count, :count], paired[count:, count:], original.epz_i[0],
                             -paired[:count, count:], -paired[count:, :count])
        meta = dict(valid=valid, closure='conjugate_paired_energy', raw=raw,
                    minimum_rotated_storage=rotated, minimum_dissipation=loss,
                    hermitian_error=hermitian_error, maximum_pairing_change=xp.max(xp.abs(paired-tensor)),
                    relative_pairing_change=xp.linalg.norm(paired-tensor)/xp.maximum(xp.linalg.norm(tensor), 1.),
                    **reciprocity)
        return ConvMatrices(*(xp.where(valid, matrix, xp.nan)[None] for matrix in slots)), meta
