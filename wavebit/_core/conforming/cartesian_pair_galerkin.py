"""Real constrained-gradient Galerkin response on a finite contact pair.

Every retained transverse Fourier column has its own potential correction.
The physical material stays in the original bilinear weak equations; there
is no material-dependent complex trial frame or reciprocity projection.
NumPy values and analytical gap/material JVPs only. This explicit finite
constitutive family does not certify Maxwell optical convergence.
"""
import numpy as np

from .cartesian_pair import CartesianPairPlan, _physical_geometry
from .cartesian_pair_opening import (
    CartesianPairOpeningCell, validate_geometry, real_array, materials_array,
)
from .material_sector import phase_from_angle, minimum_rotated_storage
from .triangle_moments import triangle_phase_average
from ..factorization import ConvMatrices


def _solve(matrix, rhs):
    """Correct the residual of the same original equations, without damping."""
    result = np.linalg.solve(matrix, rhs)
    for _ in range(2):
        result += np.linalg.solve(matrix, rhs - matrix @ result)
    return result


def _relative(residual, *terms):
    return np.linalg.norm(residual) / max(1., *(np.linalg.norm(t) for t in terms))


def _slots(tensor, longitudinal):
    size = len(longitudinal)
    return ConvMatrices(*(v[None] for v in (
        tensor[:size, :size], tensor[size:, size:], longitudinal,
        -tensor[:size, size:], -tensor[size:, :size])))


class CartesianPairGalerkinGapCell:
    """Material-holomorphic finite transverse response across gap zero.

    Uses the fixed CartesianPairOpeningPlan (refinements 0..3). A call to
    ``closure`` returns slots, their directional derivative and diagnostics.
    The default direction opens the gap by one caller length unit. Materials,
    including the background, can vary simultaneously through their tangents.
    Geometry other than gap is held fixed. At zero, the gap direction has
    right-derivative meaning. No ordinary NVF routing is changed.
    """

    def __init__(self, plan, order=1, *, coercivity_angle=0.):
        endpoint = CartesianPairOpeningCell(plan, order, coercivity_angle=coercivity_angle)
        self._plan, self._order, self._data = endpoint.plan, endpoint.order, endpoint.data
        self._angle = endpoint.coercivity_angle
        self._phase = phase_from_angle(self._angle)

    @property
    def plan(self):
        return self._plan

    @property
    def order(self):
        return self._order

    @property
    def data(self):
        return self._data

    @property
    def coercivity_angle(self):
        return self._angle

    def _assemble(self, *, gap, basis, widths, height, epsilon, background,
                  offset, gap_tangent, epsilon_tangent, background_tangent):
        geometry = validate_geometry(basis, widths, height, offset)
        material = materials_array(epsilon, background, self._phase)
        gap = real_array(gap, (), 'gap').item()
        gap_tangent = real_array(gap_tangent, (), 'gap_tangent').item()
        if gap < 0:
            raise ValueError('gap must be nonnegative')
        tangent = np.asarray(epsilon_tangent)
        bg_tangent = np.asarray(background_tangent)
        if (tangent.shape != (2,) or bg_tangent.shape != ()
                or tangent.dtype.kind not in 'iufc' or bg_tangent.dtype.kind not in 'iufc'
                or not np.isfinite(tangent).all() or not np.isfinite(bg_tangent)):
            raise ValueError('finite numeric epsilon/background tangents with shapes (2,)/() required')
        dmaterial = np.r_[bg_tangent, tangent].astype(complex)
        length = np.sqrt(np.linalg.det(geometry['basis']))
        normalized = {k: geometry[k]/length for k in ('basis', 'widths', 'height', 'offset')}
        s, ds = gap/length, gap_tangent/length
        _physical_geometry(gap=s, **normalized)
        contact, _, _, _, _ = CartesianPairPlan(
            True, self.plan.shift, self.plan.refinement).replay(gap=0., **normalized)
        zero = contact[self.data['mapping']]
        velocity = (self.data['open_numerators']/self.data['weight_denominator']) @ np.array(
            [[0., 0.]]*4 + [[v, 0.] for v in (-.5, -.5, .5, .5, .5, .5, -.5, -.5)])
        vertices = zero+s*velocity
        nc, nz = self.data['C'].shape[1], self.data['Z'].shape[1]
        ox, oy = self.order
        n, m = np.meshgrid(np.arange(-oy, oy+1), np.arange(-ox, ox+1), indexing='ij')
        n, m = n.ravel(), m.ravel()
        size = len(n)
        nn, mm = np.meshgrid(np.arange(-2*oy, 2*oy+1), np.arange(-2*ox, 2*ox+1), indexing='ij')
        waves = 2*np.pi*np.column_stack([mm.ravel(), nn.ravel()]) @ np.linalg.inv(normalized['basis'])
        # Store C and unscaled Z moments separately. The physical Z columns
        # carry s; their weak test rows are divided by s analytically.
        qc, qz = np.zeros((2, size, nc), complex), np.zeros((2, size, nz), complex)
        rc, rz, dqc, dqz, drc, drz = (np.zeros_like(a) for a in (qc, qz, qc, qz, qc, qz))
        h00, h01, h11 = np.zeros((nc, nc), complex), np.zeros((nc, nz), complex), np.zeros((nz, nz), complex)
        dh00, dh01, dh11 = (np.zeros_like(a) for a in (h00, h01, h11))
        coeff, dcoeff = np.zeros(len(waves), complex), np.zeros(len(waves), complex)
        local = []
        # All retained modes are in the larger difference grid. Integrate that
        # grid once, in bounded triangle batches, and gather the retained means.
        # The divided-difference expressions and physical nodes are unchanged.
        retained_index = ((n+2*oy)*(4*ox+1)+m+2*ox).astype(int)
        all_average = np.empty((len(self.data['triangles']), len(waves)), complex)
        all_change = np.empty_like(all_average)
        for first in range(0, len(all_average), 16):
            triangles = self.data['triangles'][first:first+16]
            avg, davg = triangle_phase_average(vertices[triangles], velocity[triangles], waves)
            all_average[first:first+16], all_change[first:first+16] = avg, davg
        adj = lambda j: np.array([[j[1, 1], -j[0, 1]], [-j[1, 0], j[0, 0]]])
        reference = np.array([[-1., -1.], [1., 0.], [0., 1.]])
        for ie, (triangle, tag, thin) in enumerate(zip(self.data['triangles'], self.data['tags'], self.data['thin'])):
            p, dp = zero[triangle], velocity[triangle]
            j0, j1 = (np.column_stack([v[1]-v[0], v[2]-v[0]]) for v in (p, dp))
            d0 = np.linalg.det(j0) if not thin else 0.
            d1 = j1[0, 0]*j0[1, 1]-j1[0, 1]*j0[1, 0]
            if np.any(j1[1] != 0):
                raise ValueError('only horizontal opening is supported')
            if thin:
                mapped = self.data['mapping'][triangle]
                repeated = next((i, j) for i in range(3) for j in range(i+1, 3) if mapped[i] == mapped[j])
                if not np.array_equal(p[repeated[0]], p[repeated[1]]) or d1 <= 0:
                    raise ValueError('exact thin collapse and positive area slope required')
            elif min(d0, d0+s*d1) <= 0:
                raise ValueError('regular triangles must remain positive')
            n0, n1 = reference@adj(j0), reference@adj(j1)
            normal = n0+s*n1
            ids = self.data['periodic'][triangle]
            c, z = self.data['Cfull'][ids, 1:], self.data['Zfull'][ids]
            x1, y1 = n1.T@c, n1.T@z
            y = normal.T@z
            x = s*x1 if thin else normal.T@c
            e, de = material[tag], dmaterial[tag]
            # Material-independent geometric blocks retain their own scaling;
            # no square root of gap, small-gap switch, or inverse gap appears.
            if thin:
                a00, a01, a11 = s*x1.T@x1/(2*d1), x1.T@y/(2*d1), y.T@y/(2*d1)
                da00 = x1.T@x1/(2*d1)
                da01 = x1.T@y1/(2*d1)
                da11 = (y1.T@y+y.T@y1)/(2*d1)
            else:
                d = d0+s*d1
                a00, a01, a11 = x.T@x/(2*d), x.T@y/(2*d), s*y.T@y/(2*d)
                da00 = (x1.T@x+x.T@x1)/(2*d)-a00*d1/d
                da01 = (x1.T@y+x.T@y1)/(2*d)-a01*d1/d
                da11 = (y.T@y+s*(y1.T@y+y.T@y1))/(2*d)-a11*d1/d
            h00 += e*a00; h01 += e*a01; h11 += e*a11
            dh00 += de*a00+e*ds*da00
            dh01 += de*a01+e*ds*da01
            dh11 += de*a11+e*ds*da11
            avg, davg = all_average[ie], all_change[ie]
            average, change = avg[retained_index], davg[retained_index]
            xc, xz = x[:, None, :]*average[None, :, None]/2, y[:, None, :]*average[None, :, None]/2
            dxc = ds*(x1[:, None, :]*average[None, :, None]+x[:, None, :]*change[None, :, None])/2
            dxz = ds*(y1[:, None, :]*average[None, :, None]+y[:, None, :]*change[None, :, None])/2
            qc += xc; qz += xz; rc += e*xc; rz += e*xz
            dqc += dxc; dqz += dxz; drc += de*xc+e*dxc; drz += de*xz+e*dxz
            area = (d0+s*d1)/2
            ft, dft = area*avg, ds*(d1*avg/2+area*davg)
            coeff += e*ft; dcoeff += de*ft+e*dft
            local.append((ids, normal, n1, x, x1, y, y1, d0, d1, thin, e, de, average, change))
        def join(c, z):
            return np.hstack([c.reshape(2*size, nc), z.reshape(2*size, nz)])
        q, r = join(qc, s*qz), join(rc, s*rz)
        dq, dr = join(dqc, ds*qz+s*dqz), join(drc, ds*rz+s*drz)
        qtest = join(qc, qz).conj().T
        dqtest = join(dqc, dqz).conj().T
        reverse = np.r_[np.arange(size-1, -1, -1), np.arange(2*size-1, size-1, -1)]
        source = join(rc, rz)[reverse].T
        dsource = join(drc, drz)[reverse].T
        matrix = np.block([[h00, s*h01], [h01.T, h11]])
        dmatrix = np.block([[dh00, ds*h01+s*dh01], [dh01.T, dh11]])
        index = (n[:, None]-n+2*oy, m[:, None]-m+2*ox)
        e = coeff.reshape(4*oy+1, 4*ox+1)[index]
        de = dcoeff.reshape(4*oy+1, 4*ox+1)[index]
        return (matrix, dmatrix, q, dq, r, dr, qtest, dqtest, source, dsource, e, de,
                dict(length=length, s=s, ds=ds, nc=nc, nz=nz, size=size,
                     reverse=reverse, local=local, material=material))

    def closure(self, *, gap, basis, widths, height, epsilon, background=1.,
                offset=(0., 0.), gap_tangent=1., epsilon_tangent=(0., 0.), background_tangent=0.):
        (k, dk, q, dq, r, dr, qt, dqt, source, dsource, e, de, info) = self._assemble(
            gap=gap, basis=basis, widths=widths, height=height, epsilon=epsilon, background=background,
            offset=offset, gap_tangent=gap_tangent, epsilon_tangent=epsilon_tangent,
            background_tangent=background_tangent)
        count = 2*info['size']
        if not all(np.isfinite(v).all() for v in (k, dk, q, dq, r, dr, qt, dqt, source, dsource, e, de)):
            raise ValueError('nonfinite constrained Galerkin assembly')
        a, da = np.kron(np.eye(2), e), np.kron(np.eye(2), de)
        identity = np.eye(count)
        # Eliminate retained x and d exactly, leaving one weak system for all
        # potential corrections. C and sZ scaling remains in q/qt and r/source.
        b, c = r-a@q, source-qt@a
        db, dc = dr-da@q-a@dq, dsource-dqt@a-qt@da
        projected = k-source@q-qt@b
        dprojected = dk-dsource@q-source@dq-dqt@b-qt@db
        potential = _solve(projected, -c)
        dpotential = _solve(projected, -dc-dprojected@potential)
        x, dx = identity-q@potential, -dq@potential-q@dpotential
        tensor = a+b@potential
        dtensor = da+db@potential+b@dpotential
        weak = _relative(k@potential+source@x-qt@tensor, k@potential, source@x, qt@tensor)
        dweak = _relative(dk@potential+k@dpotential+dsource@x+source@dx-dqt@tensor-qt@dtensor,
                          dk@potential, k@dpotential, dsource@x, source@dx, dqt@tensor, qt@dtensor)
        retained = _relative(x+q@potential-identity, identity)
        dretained = _relative(dx+dq@potential+q@dpotential, dx, dq@potential, q@dpotential)
        flux = _relative(a@x+r@potential-tensor, a@x, r@potential, tensor)
        dflux = _relative(da@x+a@dx+dr@potential+r@dpotential-dtensor,
                          da@x, a@dx, dr@potential, r@dpotential, dtensor)
        original, doriginal = self._original_residual(x, dx, tensor, dtensor, potential, dpotential, info)
        rotated = minimum_rotated_storage(tensor, self._phase)
        loss = np.linalg.eigvalsh((tensor-tensor.conj().T)/2j)[0]
        reverse = info['reverse']
        reciprocity = _relative(tensor-tensor[reverse][:, reverse].T, tensor)
        dreciprocity = _relative(dtensor-dtensor[reverse][:, reverse].T, dtensor)
        zi = _solve(e, np.eye(info['size']))
        dzi = -zi@de@zi
        zres = _relative(e@zi-np.eye(info['size']), np.eye(info['size']))
        dzres = _relative(de@zi+e@dzi, de@zi, e@dzi)
        residuals = (weak, dweak, retained, dretained, flux, dflux, original, doriginal, zres, dzres)
        if (not all(np.isfinite(v).all() for v in (tensor, dtensor, zi, dzi))
                or max(residuals) > 1e-10 or max(reciprocity, dreciprocity) > 1e-9
                or rotated <= 0 or loss < -1e-9):
            raise ValueError('constrained Galerkin response failed original/physical/reciprocity guards')
        metadata = dict(valid=True, closure='real_constrained_gradient_galerkin',
                        order=self.order, refinement=self.plan.refinement, shift=self.plan.shift,
                        gap=gap, normalized_gap=info['s'], normalization_length=info['length'],
                        coercivity_angle=self.coercivity_angle,
                        open_free_dofs=len(k), contact_free_dofs=info['nc'], difference_dofs=info['nz'],
                        projected_condition=np.linalg.cond(projected), weak_condition=np.linalg.cond(k),
                        minimum_rotated_storage=rotated, minimum_dissipation=loss,
                        relative_reciprocity_defect=reciprocity, derivative_reciprocity_defect=dreciprocity,
                        weak_residual=weak, weak_derivative_residual=dweak,
                        retained_residual=retained, retained_derivative_residual=dretained,
                        flux_residual=flux, flux_derivative_residual=dflux,
                        original_residual=original, original_derivative_residual=doriginal,
                        z_inverse_residual=zres, z_inverse_derivative_residual=dzres)
        return _slots(tensor, zi), _slots(dtensor, dzi), metadata

    def _original_residual(self, x, dx, tensor, dtensor, potential, dpotential, info):
        """Reassemble every unprojected nodal equation, including the gauge."""
        count, size, nc = len(self.data['pmap']), info['size'], info['nc']
        reaction, change, scale, dscale = (np.zeros((count, 2*size), complex) for _ in range(4))
        uc, uz = potential[:nc], potential[nc:]
        dc, dz = dpotential[:nc], dpotential[nc:]
        s, ds = info['s'], info['ds']
        for ids, normal, n1, c, c1, z, z1, d0, d1, thin, e, de, avg, davg in info['local']:
            if thin:
                field = (c1@uc+z@uz)/d1
                dfield = (c1@dc+ds*z1@uz+z@dz)/d1
            else:
                d = d0+s*d1
                numerator = c@uc+s*z@uz
                field = numerator/d
                dfield = (ds*c1@uc+c@dc+ds*z@uz+s*ds*z1@uz+s*z@dz)/d-field*ds*d1/d
            # Test integration uses exp(+i G.r), while avg stores exp(-i G.r).
            moment = (normal[:, :, None]*avg.conj()[None, None, :]/2).reshape(3, 2*size)
            dmoment = ds*(n1[:, :, None]*avg.conj()[None, None, :]
                          +normal[:, :, None]*davg.conj()[None, None, :]).reshape(3, 2*size)/2
            term = e*normal@field/2
            dterm = (de*normal@field+e*ds*n1@field+e*normal@dfield)/2
            load = moment@(e*x-tensor)
            dload = dmoment@(e*x-tensor)+moment@(de*x+e*dx-dtensor)
            np.add.at(reaction, ids, term+load)
            np.add.at(change, ids, dterm+dload)
            np.add.at(scale, ids, np.abs(term)+np.abs(load))
            np.add.at(dscale, ids, np.abs(dterm)+np.abs(dload))
        return _relative(reaction, scale), _relative(change, dscale)
