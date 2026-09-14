"""Real seam potentials coupled to the complete periodic Galerkin tensor.

NumPy-only finite constitutive family. All auxiliary fields share the original
material energy and retained constraints; no tensor block or derivative is
spliced from another factorizer. Optical convergence remains separate.
"""
from numbers import Integral
import numpy as np
from numpy.polynomial.legendre import leggauss
from .periodic_rectangle_galerkin import PeriodicRectangleGalerkinGapCell
from .periodic_seam_trace import PeriodicSeamTrace
from .cartesian_pair_galerkin import _response, _slots, _relative, _solve
from .cartesian_pair_opening import real_array
from .material_sector import minimum_rotated_storage

def _free_fourier_energy(order, settings):
    """Independent exact rectangle energy for two retained plane waves.

    Their product reaches twice the retained frequency. Integrating it with
    the mixed field rule aliases high orders even when the weak solve and
    every original nodal reaction are resolved. This separated rectangle
    integral checks the original triangle convolution independently.
    """
    nx, ny = order
    modes = np.array([(m, n) for n in range(-ny, ny + 1) for m in range(-nx, nx + 1)])
    difference = modes[:, None, :] - modes[None, :, :]
    px, py = settings['period']
    width = settings['width'] / px
    height = (py - settings['gap']) / py
    phase = np.exp(-2j * np.pi * (difference @ (.5 + np.asarray(settings['offset']) / (px, py))))
    horizontal = width * np.sinc(difference[:, :, 0] * width)
    indicator = horizontal * height * np.sinc(difference[:, :, 1] * height) * phase
    change = (-settings['gap_tangent'] / py * horizontal
              * np.cos(np.pi * difference[:, :, 1] * height) * phase)
    epsilon, background = settings['epsilon'], settings['background']
    de, db = settings['epsilon_tangent'], settings['background_tangent']
    identity = np.eye(len(modes))
    value = background * identity + (epsilon - background) * indicator
    derivative = db * identity + (de - db) * indicator + (epsilon - background) * change
    return np.kron(np.eye(2), value), np.kron(np.eye(2), derivative)

def _quadrature(obj, trace, base, settings, qorder, split=True):
    """Moving triangle/trace intersections, physical gradients and their JVP.

The unsplit form is used only for the original mesh energy; its trace
samples are unused. Known moving endpoints retain exact taper fractions.
Subtracting rounded cell coordinates would lose a subpixel gap. Thin y
coordinates stay local to the seam before the translation enters a phase.
"""
    k, dk, q, dq, r, dr, qt, dqt, source, dsource, e, de, info = base
    length = info['length']
    s = info['s']
    ds = info['ds']
    period = np.asarray(settings['period'])
    offset = np.asarray(settings['offset'])
    geometry = dict(basis=np.diag(period), widths=np.repeat(settings['width'] / 2, 2), height=period[1], offset=offset)
    _, _, zero, velocity = obj._chart_geometry(settings['gap'], geometry)
    tg = trace._geometry(settings['gap'], period, settings['width'], offset[0])
    ratio = tg['length'] / length
    delta, d_delta = trace._taper(tg)
    delta *= ratio
    d_delta *= ratio * settings['gap_tangent'] / tg['length']
    qtop = obj.data['divisions']
    coords = obj.data['coordinates']
    left = zero[np.flatnonzero(coords[:, 0] == qtop)[0], 0]
    right = zero[np.flatnonzero(coords[:, 0] == 2 * qtop)[0], 0]
    moving = [(left - delta, -d_delta), (left, 0.0), (left + delta, d_delta), (right - delta, -d_delta), (right, 0.0), (right + delta, d_delta)]
    t, w = leggauss(qorder)
    t = (t + 1) / 2
    w = w / 2
    nx, ny = obj.order
    modes = np.array([(i, j) for j in range(-ny, ny + 1) for i in range(-nx, nx + 1)])
    wave = 2 * np.pi * modes / (period / length)
    material = np.r_[settings['background'], settings['epsilon']]
    dmaterial = np.r_[settings['background_tangent'], settings['epsilon_tangent']]
    for ie, (triangle, tag, thin) in enumerate(zip(obj.data['triangles'], obj.data['tags'], obj.data['thin'])):
        vertices = triangle[:3]
        p = zero[vertices]
        speed = velocity[vertices]
        xl, xr = (p[:, 0].min(), p[:, 0].max())
        width = xr - xl
        iy = np.argmin(coords[vertices, 1])
        y0 = p[iy, 1]
        ys = speed[iy, 1]
        if thin:
            y0 = offset[1] / length
        knots = [(xl, 0.0)] + ([(x, dx) for x, dx in moving if xl < x < xr] if split else []) + [(xr, 0.0)]
        ids, ci, zi, normal, n1, c, c1, z, z1, d0, d1, _, _, _, _, _ = info['local'][ie]
        det = d0 + s * d1
        ddet = ds * d1
        height = det / width
        dheight = ddet / width
        for (lo, dlo), (hi, dhi) in zip(knots, knots[1:]):
            xx = lo + (hi - lo) * t
            dxx = dlo + (dhi - dlo) * t
            u = (xx - xl) / width
            du = dxx / width
            if ie % 2 == 0:
                bary = np.stack([np.broadcast_to(1 - u[:, None], (qorder, qorder)), u[:, None] * (1 - t), u[:, None] * t], axis=-1)
                dbary = np.stack([np.broadcast_to(-du[:, None], (qorder, qorder)), du[:, None] * (1 - t), du[:, None] * t], axis=-1)
                v = u[:, None] * t
                dv = du[:, None] * t
                factor = u
                dfactor = du
            else:
                bary = np.stack([(1 - u[:, None]) * (1 - t), np.broadcast_to(u[:, None], (qorder, qorder)), (1 - u[:, None]) * t], axis=-1)
                dbary = np.stack([-du[:, None] * (1 - t), np.broadcast_to(du[:, None], (qorder, qorder)), -du[:, None] * t], axis=-1)
                v = u[:, None] + (1 - u[:, None]) * t
                dv = du[:, None] * (1 - t)
                factor = 1 - u
                dfactor = -du
            local_y = (0.0 if thin else y0 - offset[1] / length) + s * ys + height * v
            yy = local_y + offset[1] / length
            dyy = ds * ys + dheight * v + height * dv
            weights = det * (hi - lo) / width * factor[:, None] * w[:, None] * w
            dweights = ((ddet * (hi - lo) + det * (dhi - dlo)) / width * factor + det * (hi - lo) / width * dfactor)[:, None] * w[:, None] * w
            bb, dbb = (bary.reshape(-1, 3), dbary.reshape(-1, 3))
            xx = np.broadcast_to(xx[:, None], yy.shape).ravel()
            dxx = np.broadcast_to(dxx[:, None], yy.shape).ravel()
            point = np.column_stack([xx, yy.ravel()])
            dpoint = np.column_stack([dxx, dyy.ravel()])
            middle = (lo + hi) / 2
            if left - delta < middle < left + delta:
                side, edge = (1.0, left)
            elif right - delta < middle < right + delta:
                side, edge = (-1.0, right)
            else:
                side, edge = (0.0, 0.0)
            if side:

                def endpoint(x, dx):
                    for index, (knot, speed) in enumerate(moving):
                        if x == knot:
                            return ([0.0, 0.5, 1.0, 1.0, 0.5, 0.0][index], 0.0)
                    return (0.5 + side * (x - edge) / (2 * delta), side * (dx - (x - edge) * d_delta / delta) / (2 * delta))
                fl, dfl = endpoint(lo, dlo)
                fh, dfh = endpoint(hi, dhi)
                fraction = fl + (fh - fl) * t
                dfraction = dfl + (dfh - dfl) * t
                hat = fraction ** 2 * (3 - 2 * fraction)
                dhat = 6 * fraction * (1 - fraction) * dfraction
                hx = side * 3 * fraction * (1 - fraction) / delta
                dhx = side * 3 * ((1 - 2 * fraction) * dfraction / delta - fraction * (1 - fraction) * d_delta / delta ** 2)
            else:
                hat = np.full(qorder, float(left < middle < right))
                dhat = np.zeros(qorder)
                hx = dhat
                dhx = dhat
            hat, dhat, hx, dhx = [np.broadcast_to(a[:, None], (qorder, qorder)).ravel() for a in (hat, dhat, hx, dhx)]
            B, Bx, Bxx = trace._basis((xx - (left + right) / 2) / ratio, tg)
            Bx = Bx / ratio
            Bxx = Bxx / ratio ** 2
            T = hat[:, None] * B
            dT = dhat[:, None] * B + hat[:, None] * Bx * dxx[:, None]
            Tx = hx[:, None] * B + hat[:, None] * Bx
            dTx = dhx[:, None] * B + hx[:, None] * Bx * dxx[:, None] + dhat[:, None] * Bx + hat[:, None] * Bxx * dxx[:, None]
            py = period[1] / length
            h = py - s
            local_y = local_y.ravel()
            dy = dyy.ravel()
            f = local_y if thin else s * (py / 2 - local_y) / h
            fy = np.full(len(f), 1.0 if thin else -s / h)
            df = dy if thin else ds * py * (py / 2 - local_y) / h ** 2 - s * dy / h
            dfy = np.full(len(f), 0.0 if thin else -ds * py / h ** 2)
            le = np.stack([f[:, None] * Tx, fy[:, None] * T], axis=1)
            dle = np.stack([df[:, None] * Tx + f[:, None] * dTx, dfy[:, None] * T + fy[:, None] * dT], axis=1)
            phase = np.exp(-1j * point @ wave.T)
            dphase = -1j * (dpoint @ wave.T) * phase

            def gradient(n, dn, den, dden):
                v = _contract('qv,vai->qai', bb, n) / den
                dv = (_contract('qv,vai->qai', dbb, n) + ds * _contract('qv,vai->qai', bb, dn)) / den - v * dden / den
                return (v, dv)
            gc, dgc = gradient(c1, np.zeros_like(c1), d1, 0.0) if thin else gradient(c, c1, det, ddet)
            gz, dgz = gradient(z, z1, det, ddet)
            gn, dgn = gradient(normal.swapaxes(-1, -2), n1.swapaxes(-1, -2), det, ddet)
            yield (ids, ci, zi, weights.ravel(), dweights.ravel(), material[tag], dmaterial[tag], phase, dphase, gc, dgc, gz, dgz, le, dle, gn, dgn)

def _positive(obj, trace, *, gap, period, width, offset, epsilon, background, gap_tangent, epsilon_tangent, background_tangent, quadrature_order):
    """Mutual original weak assembly and reconstructed physical-field reactions.

The derivative check uses the stationary reciprocal bilinear energy.
Integrate only the added energy on the moving taper partition, and the
original mesh energy on unsplit triangles. This cancels the baseline's
O(1/sqrt(gap)) artificial transport analytically, without changing the
energy or discarding a weak-system derivative.
"""
    order = obj.order
    settings = dict(gap=gap, period=period, width=width, offset=offset, epsilon=epsilon, background=background, gap_tangent=gap_tangent, epsilon_tangent=epsilon_tangent, background_tangent=background_tangent)
    base = obj._assemble(gap=gap, basis=np.diag(period), widths=(width / 2, width / 2), height=period[1], epsilon=(epsilon, epsilon), background=background, offset=offset, gap_tangent=gap_tangent, epsilon_tangent=(epsilon_tangent, epsilon_tangent), background_tangent=background_tangent)
    k, dk, q, dq, r, dr, qt, dqt, source, dsource, e, de, info = base
    N = len(e)
    nc, nz = (info['nc'], info['nz'])
    NE = 2 * order[0] + 1
    s = info['s']
    ds = info['ds']
    Q = np.zeros((2 * N, NE), complex)
    R = Q.copy()
    dQ = Q.copy()
    dR = Q.copy()
    K = np.zeros((nc + nz, NE), complex)
    dK = K.copy()
    EE = np.zeros((NE, NE), complex)
    dEE = EE.copy()
    for ids, ci, zi, w, dw, mat, dmat, phase, dphase, gc, dgc, gz, dgz, le, dle, gn, dgn in _quadrature(obj, trace, base, settings, quadrature_order):
        moment = _contract('q,qm,qaj->amj', w, phase, le).reshape(2 * N, NE)
        dm = (_contract('q,qm,qaj->amj', dw, phase, le) + _contract('q,qm,qaj->amj', w, dphase, le) + _contract('q,qm,qaj->amj', w, phase, dle)).reshape(2 * N, NE)
        Q += moment
        dQ += dm
        R += mat * moment
        dR += dmat * moment + mat * dm
        for rows, g, dg in ((ci, gc, dgc), (nc + zi, gz, dgz)):
            cross = _contract('q,qai,qaj->ij', w, g, le)
            dcross = _contract('q,qai,qaj->ij', dw, g, le) + _contract('q,qai,qaj->ij', w, dg, le) + _contract('q,qai,qaj->ij', w, g, dle)
            K[rows] += mat * cross
            dK[rows] += dmat * cross + mat * dcross
        en = _contract('q,qai,qaj->ij', w, le, le) / s
        den = (_contract('q,qai,qaj->ij', dw, le, le) + _contract('q,qai,qaj->ij', w, dle, le) + _contract('q,qai,qaj->ij', w, le, dle)) / s - en * ds / s
        EE += mat * en
        dEE += dmat * en + mat * den
    scale = np.r_[np.ones(nc), np.full(nz, s)]
    dscale = np.r_[np.zeros(nc), np.full(nz, ds)]
    row = K.T * scale[None] / s
    drow = (dK.T * scale[None] + K.T * dscale[None]) / s - row * ds / s
    KK = np.block([[k, K], [row, EE]])
    dKK = np.block([[dk, dK], [drow, dEE]])
    QQ = np.hstack([q, Q])
    dQQ = np.hstack([dq, dQ])
    RR = np.hstack([r, R])
    dRR = np.hstack([dr, dR])
    QT = np.vstack([qt, Q.conj().T / s])
    dQT = np.vstack([dqt, dQ.conj().T / s - Q.conj().T * ds / s ** 2])
    SS = np.vstack([source, R[info['reverse']].T / s])
    dSS = np.vstack([dsource, dR[info['reverse']].T / s - R[info['reverse']].T * ds / s ** 2])
    A = np.kron(np.eye(2), e)
    DA = np.kron(np.eye(2), de)
    U, dU, x, dx, H, DH, cond = _response(KK, dKK, QQ, dQQ, RR, dRR, QT, dQT, SS, dSS, A, DA, 'potential')
    count = len(obj.data['pmap'])
    res = np.zeros((count, 2 * N), complex)
    dres = res.copy()
    res_scale = np.zeros_like(res, float)
    dres_scale = res_scale.copy()
    exact_free, exact_free_derivative = _free_fourier_energy(order, settings)
    dx_geometry = -dQQ @ U
    reverse = info['reverse']
    xr, dxr = x[:, reverse].T[:, reverse], dx_geometry[:, reverse].T[:, reverse]
    energy = x.conj().T @ exact_free @ x
    denergy = (dxr @ exact_free @ x + xr @ exact_free_derivative @ x
               + xr @ exact_free @ dx_geometry)
    for ids, ci, zi, w, dw, mat, dmat, phase, dphase, gc, dgc, gz, dgz, le, dle, gn, dgn in _quadrature(obj, trace, base, settings, quadrature_order):
        F = np.zeros((len(w), 2, 2 * N), complex)
        DF = F.copy()
        F[:, 0, :N] = phase.conj()
        F[:, 1, N:] = phase.conj()
        DF[:, 0, :N] = dphase.conj()
        DF[:, 1, N:] = dphase.conj()
        E = F @ x + gc @ U[ci] + s * gz @ U[nc + zi] + le @ U[nc + nz:]
        DE = DF @ x + F @ dx + dgc @ U[ci] + gc @ dU[ci] + ds * gz @ U[nc + zi] + s * dgz @ U[nc + zi] + s * gz @ dU[nc + zi] + dle @ U[nc + nz:] + le @ dU[nc + nz:]
        T = mat * E - F @ H
        DT = dmat * E + mat * DE - DF @ H - F @ DH
        react = _contract('q,qai,qaj->ij', w, gn, T)
        change = _contract('q,qai,qaj->ij', dw, gn, T) + _contract('q,qai,qaj->ij', w, dgn, T) + _contract('q,qai,qaj->ij', w, gn, DT)
        np.add.at(res, ids, react)
        np.add.at(dres, ids, change)
        np.add.at(res_scale, ids, _contract('q,qai,qaj->ij', abs(w), abs(gn), abs(mat * E) + abs(F @ H)))
        np.add.at(dres_scale, ids, _contract('q,qai,qaj->ij', abs(dw), abs(gn), abs(T)) + _contract('q,qai,qaj->ij', abs(w), abs(dgn), abs(T)) + _contract('q,qai,qaj->ij', abs(w), abs(gn), abs(DT)))
        free = F @ x
        correction = gc @ U[ci] + s * gz @ U[nc + zi] + le @ U[nc + nz:]
        for left, right in ((free, correction), (correction, free), (correction, correction)):
            energy += mat * _contract('q,qai,qaj->ij', w, left.conj(), right)
        reverse = info['reverse']
        added = le @ U[nc + nz:]
        dadded = dle @ U[nc + nz:]
        old = F @ x + gc @ U[ci] + s * gz @ U[nc + zi]
        dold = DF @ x + dgc @ U[ci] + ds * gz @ U[nc + zi] + s * dgz @ U[nc + zi] - F @ (dQQ @ U)
        for L, DL, R, DR in ((old, dold, added, dadded), (added, dadded, old, dold), (added, dadded, added, dadded)):
            denergy += dmat * _contract('q,qai,qaj->ij', w, L[:, :, reverse], R) + mat * _contract('q,qai,qaj->ij', dw, L[:, :, reverse], R) + mat * _contract('q,qai,qaj->ij', w, DL[:, :, reverse], R) + mat * _contract('q,qai,qaj->ij', w, L[:, :, reverse], DR)
    for ids, ci, zi, w, dw, mat, dmat, phase, dphase, gc, dgc, gz, dgz, le, dle, gn, dgn in _quadrature(obj, trace, base, settings, quadrature_order, split=False):
        F = np.zeros((len(w), 2, 2 * N), complex)
        DF = F.copy()
        F[:, 0, :N] = phase.conj()
        F[:, 1, N:] = phase.conj()
        DF[:, 0, :N] = dphase.conj()
        DF[:, 1, N:] = dphase.conj()
        free = F @ x
        dfree = DF @ x - F @ (dQQ @ U)
        correction = gc @ U[ci] + s * gz @ U[nc + zi]
        dcorrection = dgc @ U[ci] + ds * gz @ U[nc + zi] + s * dgz @ U[nc + zi]
        reverse = info['reverse']
        for left, dl, right, dright in ((free, dfree, correction, dcorrection),
                                       (correction, dcorrection, free, dfree),
                                       (correction, dcorrection, correction, dcorrection)):
            denergy += (dmat * _contract('q,qai,qaj->ij', w, left[:, :, reverse], right)
                        + mat * _contract('q,qai,qaj->ij', dw, left[:, :, reverse], right)
                        + mat * _contract('q,qai,qaj->ij', w, dl[:, :, reverse], right)
                        + mat * _contract('q,qai,qaj->ij', w, left[:, :, reverse], dright))
    rr = info['reverse']
    meta = dict(condition=cond, reciprocity=_relative(H - H[rr][:, rr].T, H), derivative_reciprocity=_relative(DH - DH[rr][:, rr].T, DH), original_residual=_relative(res, res_scale), original_derivative_residual=_relative(dres, dres_scale), energy_residual=_relative(energy - H, H), energy_derivative_residual=_relative(denergy - DH, DH), minimum_rotated_storage=minimum_rotated_storage(H, obj._phase), minimum_dissipation=np.linalg.eigvalsh((H - H.conj().T) / 2j)[0])
    meta['weak_residual'] = _relative(KK @ U + SS @ x - QT @ H, KK @ U, SS @ x, QT @ H)
    meta['weak_derivative_residual'] = _relative(dKK @ U + KK @ dU + dSS @ x + SS @ dx - dQT @ H - QT @ DH, dKK @ U, KK @ dU, dSS @ x, SS @ dx, dQT @ H, QT @ DH)
    meta['direct_fourier_residual'] = _relative(A - exact_free, exact_free)
    meta['direct_fourier_derivative_residual'] = _relative(DA - exact_free_derivative, exact_free_derivative)
    inv = _solve(e, np.eye(N))
    return (_slots(H, inv), _slots(DH, -inv @ de @ inv), meta)

def _contact(obj, trace, *, period, width, offset, epsilon, background, gap_tangent, epsilon_tangent, background_tangent, quadrature_order):
    """Finite physical weak limit, including both material-side edge terms.

Only the contact rows enter the differentiable physical response. For
an independent energy derivative, hold the auxiliary coefficients fixed
and account for the retained constraint with -dQ U. Physical weak
coefficients have finite limits; normalized disappearing rows need not
have derivatives. The left Fourier coefficient also needs a mode reversal
when evaluating the reciprocal bilinear energy.
"""
    n, ny = obj.order
    quadrature = quadrature_order
    period = np.asarray(period)
    offset = np.asarray(offset)
    pars = dict(gap=0.0, basis=np.diag(period), widths=(width / 2, width / 2), height=period[1], epsilon=(epsilon, epsilon), background=background, offset=offset, gap_tangent=gap_tangent, epsilon_tangent=(epsilon_tangent, epsilon_tangent), background_tangent=background_tangent)
    k, dk, q, dq, r, dr, qt, dqt, source, dsource, e, de, info = obj._assemble(**pars)
    geo = trace._geometry(0.0, period, width, offset[0])
    N = len(e)
    NE = 2 * n + 1
    nc, nz = (info['nc'], info['nz'])
    s = 0.0
    ds = info['ds']
    C = np.zeros((nc, NE), complex)
    ED = C.copy()
    Z = np.zeros((nz, NE), complex)
    length, s, zero, velocity = obj._chart_geometry(0.0, dict(basis=pars['basis'], widths=np.asarray(pars['widths']), height=period[1], offset=offset))
    edge_a = zero[np.flatnonzero(obj.data['coordinates'][:, 0] == obj.data['divisions'])[0], 0]
    edge_b = zero[np.flatnonzero(obj.data['coordinates'][:, 0] == 2 * obj.data['divisions'])[0], 0]
    px, py = period / length
    oy = offset[1] / length
    center = (edge_a + edge_b) / 2
    ratio = geo['length'] / length

    def basis(x):
        b, bx, bxx = trace._basis((x - center) / ratio, geo)
        return (b, bx / ratio, bxx / ratio ** 2)
    t, w = leggauss(quadrature)
    t = (t + 1) / 2
    w = w / 2
    for ie, (triangle, tag, thin) in enumerate(zip(obj.data['triangles'], obj.data['tags'], obj.data['thin'])):
        points = zero[triangle[:3]]
        xl, xr = (np.min(points[:, 0]), np.max(points[:, 0]))
        yb, yt = (np.min(points[:, 1]), np.max(points[:, 1]))
        dy = yt - yb
        ids, ci, zi, normal, n1, c, c1, z, z1, d0, d1, thin, mat, *_ = info['local'][ie]
        lo, hi = (max(xl, edge_a), min(xr, edge_b))
        if hi > lo:
            xx = lo + (hi - lo) * t
            u = (xx - xl) / (xr - xl)
            v = t
            if ie % 2 == 0:
                bary = np.stack([np.broadcast_to(1 - u[:, None], (quadrature, quadrature)), u[:, None] * (1 - v), u[:, None] * v], axis=-1)
                yy = yb + dy * u[:, None] * v
                ww = (hi - lo) / (xr - xl) * u[:, None] * w[:, None] * w
            else:
                bary = np.stack([(1 - u[:, None]) * (1 - v), np.broadcast_to(u[:, None], (quadrature, quadrature)), (1 - u[:, None]) * v], axis=-1)
                yy = yb + dy * (u[:, None] + (1 - u[:, None]) * v)
                ww = (hi - lo) / (xr - xl) * (1 - u[:, None]) * w[:, None] * w
            bb = bary.reshape(-1, 3)
            xx = np.broadcast_to(xx[:, None], yy.shape).ravel()
            yy = yy.ravel()
            ww = ww.ravel()
            B, Bx, _ = basis(xx)
            if thin:
                G = _contract('qv,vai->qai', bb, z)
                Z[zi] += mat * _contract('q,qi,qj->ij', ww, G[:, 1], B)
            else:
                G = _contract('qv,vai->qai', bb, c) / d0
                L = np.stack([(0.5 - (yy - oy) / py)[:, None] * Bx, -B / py], axis=1)
                C[ci] += mat * d0 * _contract('q,qai,qaj->ij', ww, G, L)
        if not thin:
            for edge, sign in ((edge_a, 1), (edge_b, -1)):
                local = np.flatnonzero(points[:, 0] == edge)
                if len(local) != 2:
                    continue
                a, b = local
                if points[a, 1] > points[b, 1]:
                    a, b = (b, a)
                yy = points[a, 1] + (points[b, 1] - points[a, 1]) * t
                bb = np.zeros((quadrature, 3))
                bb[:, a] = 1 - t
                bb[:, b] = t
                B = basis(np.array([edge]))[0][0]
                G = _contract('qv,vai->qai', bb, c) / d0
                ED[ci] += 0.5 * sign * mat * _contract('q,qi,j->ij', w * (points[b, 1] - points[a, 1]) * (0.5 - (yy - oy) / py), G[:, 0], B)
    C += ED
    modes = np.array([(m, j) for j in range(-ny, ny + 1) for m in range(-n, n + 1)])
    wave = 2 * np.pi * modes / (period / length)
    xx = edge_a + (edge_b - edge_a) * t
    B, Bx, _ = basis(xx)
    phase = np.exp(-1j * xx[:, None] * wave[:, 0])
    X = phase.T @ (((edge_b - edge_a) * w)[:, None] * B)
    Xp = phase.T @ (((edge_b - edge_a) * w)[:, None] * Bx)
    edge = basis(np.array([edge_a, edge_b]))[0]
    jump = np.exp(-1j * edge_a * wave[:, 0, None]) * edge[0] - np.exp(-1j * edge_b * wave[:, 0, None]) * edge[1]
    Y = np.divide(1.0, 1j * wave[:, 1], out=np.zeros(N, complex), where=wave[:, 1] != 0)
    YP = np.exp(-1j * wave[:, 1] * oy)
    Qbar = np.vstack([1j * wave[:, 0, None] * X * Y[:, None] * YP[:, None], X * (wave[:, 1] != 0)[:, None] * YP[:, None]])
    Rbar = np.vstack([(epsilon * Xp + (epsilon + background) / 2 * jump) * Y[:, None] * YP[:, None], X * (background - epsilon * (wave[:, 1] == 0))[:, None] * YP[:, None]])
    EE = background * B.T @ (((edge_b - edge_a) * w)[:, None] * B)
    KK = np.block([[k, np.vstack([np.zeros_like(C), Z])], [np.hstack([C.T, Z.T]), EE]])
    QQ = np.hstack([q, np.zeros_like(Qbar)])
    RR = np.hstack([r, np.zeros_like(Rbar)])
    QT = np.vstack([qt, Qbar.conj().T])
    SS = np.vstack([source, Rbar[info['reverse']].T])
    A = np.kron(np.eye(2), e)
    DA = np.kron(np.eye(2), de)
    BB = RR - A @ QQ
    CC = SS - QT @ A
    DD = KK - SS @ QQ - QT @ BB
    U = _solve(DD, -CC)
    H = A + BB @ U
    dQQ = np.hstack([dq, ds * Qbar])
    dRR = np.hstack([dr, ds * Rbar])
    dBB = dRR - DA @ QQ - A @ dQQ
    dKK = np.hstack([dk[:nc], ds * C])
    dDD = dKK - dsource[:nc] @ QQ - source[:nc] @ dQQ - dqt[:nc] @ BB - qt[:nc] @ dBB
    dCC = dsource[:nc] - dqt[:nc] @ A - qt[:nc] @ DA
    dUC = _solve(DD[:nc, :nc], -dCC - dDD @ U)
    DH = DA + dBB @ U + BB[:, :nc] @ dUC
    x = np.eye(2 * N) - QQ @ U
    rr = info['reverse']
    scale = np.r_[np.ones(nc), np.zeros(nz + NE)]
    dscale = np.r_[np.zeros(nc), np.full(nz + NE, ds)]
    physical = scale[:, None] * KK
    dphysical = dscale[:, None] * KK
    dphysical[:nc] += dKK
    source_physical = RR[rr].T
    dsource_physical = dRR[rr].T
    dx_geometry = -dQQ @ U
    xr = x[:, rr].T[:, rr]
    dxr = dx_geometry[:, rr].T[:, rr]
    ur = U[:, rr].T
    energy = xr @ A @ x + xr @ RR @ U + ur @ source_physical @ x + ur @ physical @ U
    denergy = dxr @ A @ x + xr @ DA @ x + xr @ A @ dx_geometry + dxr @ RR @ U + xr @ dRR @ U + ur @ dsource_physical @ x + ur @ source_physical @ dx_geometry + ur @ dphysical @ U
    weak = _relative(KK @ U + SS @ x - QT @ H, KK @ U, SS @ x, QT @ H)
    cweak = _relative(dDD @ U + DD[:nc, :nc] @ dUC + dCC, dDD @ U, DD[:nc, :nc] @ dUC, dCC)
    meta = dict(condition=np.linalg.cond(DD), reciprocity=_relative(H - H[rr][:, rr].T, H), derivative_reciprocity=_relative(DH - DH[rr][:, rr].T, DH), weak_residual=weak, weak_derivative_residual=None, contact_derivative_residual=cweak, energy_residual=_relative(energy - H, H), energy_derivative_residual=_relative(denergy - DH, DH), minimum_rotated_storage=minimum_rotated_storage(H, obj._phase), minimum_dissipation=np.linalg.eigvalsh((H - H.conj().T) / 2j)[0])
    inv = _solve(e, np.eye(N))
    return (_slots(H, inv), _slots(DH, -inv @ de @ inv), meta)

class PeriodicSeamGalerkinGapCell:
    """Explicit quadratic mesh plus continuous real seam-potential closure.

    A rectangle has fixed width and height period_y-gap. Gap and both complex
    materials can vary together; other geometry stays fixed. At contact, the
    finite physical-energy limit supplies the right derivative, while the
    normalized disappearing weak rows need not have a derivative. Ordinary
    NVF dispatch and other backends are unchanged.
    """

    def __init__(self, order=1, *, refinement=0, quadrature=32, coercivity_angle=0.0):
        self._base = PeriodicRectangleGalerkinGapCell(order, degree=2, refinement=refinement, coercivity_angle=coercivity_angle)
        self.order = self._base.order
        if isinstance(quadrature, (bool, np.bool_)) or not isinstance(quadrature, Integral) or quadrature < max(16, 2 * max(self.order) + 4):
            raise ValueError('quadrature must be an integer at least max(16, 2*max(order)+4)')
        self.quadrature = int(quadrature)
        self._trace = PeriodicSeamTrace(self.order[0], coercivity_angle=coercivity_angle)

    def closure(self, *, gap, period=(700.0, 700.0), width=350.0, epsilon=12.25, background=1.0, offset=(0.0, 0.0), gap_tangent=1.0, epsilon_tangent=0.0, background_tangent=0.0):
        gap = real_array(gap, (), 'gap').item()
        period = real_array(period, (2,), 'period')
        width = real_array(width, (), 'width').item()
        offset = real_array(offset, (2,), 'offset')
        self._trace._geometry(gap, period, width, offset[0])
        args = dict(period=period, width=width, offset=offset, epsilon=epsilon, background=background, gap_tangent=gap_tangent, epsilon_tangent=epsilon_tangent, background_tangent=background_tangent, quadrature_order=self.quadrature)
        if gap == 0:
            value, tangent, metadata = _contact(self._base, self._trace, **args)
        else:
            value, tangent, metadata = _positive(self._base, self._trace, gap=gap, **args)
        residuals = [v for key, v in metadata.items() if key.endswith('residual') and v is not None]
        if not np.isfinite(np.asarray(value)).all() or not np.isfinite(np.asarray(tangent)).all() or (not all((np.isfinite(v) for v in residuals))) or (max(residuals) > 1e-10) or (max(metadata['reciprocity'], metadata['derivative_reciprocity']) > 1e-09) or (metadata['minimum_rotated_storage'] <= 0) or (metadata['minimum_dissipation'] < -1e-09):
            raise ValueError('coupled seam Galerkin response failed original/energy/physical guards')
        metadata.update(closure='real_coupled_periodic_seam_galerkin', order=self.order, refinement=self._base.plan.refinement, quadrature=self.quadrature, degree=2, gap=gap)
        return (value, tangent, metadata)


def _contract(*arguments):
    """Contract local weighted fields with bounded intermediates and BLAS."""
    return np.einsum(*arguments, optimize=True)
