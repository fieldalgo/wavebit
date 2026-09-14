"""Complete real-trace Galerkin coupling and original physical verification.

NumPy-only assembly checks shared by the explicit pair and periodic families.
Geometry supplies physical moving quadrature and exact contact coefficients;
all free, nodal and added trace fields share one retained-constraint solve.
"""
import numpy as np
from .cartesian_pair_galerkin import _response, _relative, _slots, _solve
from .material_sector import minimum_rotated_storage


def _contract(*arguments):
    """Contract local weighted fields with bounded intermediates and BLAS."""
    return np.einsum(*arguments, optimize=True)


def coupled_positive(obj, base, quadrature, free_energy, trace_order):
    """Solve the original weak system and verify reconstructed physical fields.

    Pure Fourier energy is supplied independently. Added mixed/trace energies
    use the moving partition, while baseline mesh terms use unsplit triangles.
    The differentiated energy is the stationary reciprocal bilinear form.
    """
    k, dk, q, dq, r, dr, qt, dqt, source, dsource, e, de, info = base
    N = len(e)
    nc, nz = (info['nc'], info['nz'])
    NE = 2 * trace_order + 1
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
    for ids, ci, zi, w, dw, mat, dmat, phase, dphase, gc, dgc, gz, dgz, le, dle, gn, dgn in quadrature():
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
    exact_free, exact_free_derivative = free_energy
    dx_geometry = -dQQ @ U
    reverse = info['reverse']
    xr, dxr = x[:, reverse].T[:, reverse], dx_geometry[:, reverse].T[:, reverse]
    energy = x.conj().T @ exact_free @ x
    denergy = (dxr @ exact_free @ x + xr @ exact_free_derivative @ x
               + xr @ exact_free @ dx_geometry)
    for ids, ci, zi, w, dw, mat, dmat, phase, dphase, gc, dgc, gz, dgz, le, dle, gn, dgn in quadrature():
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
    for ids, ci, zi, w, dw, mat, dmat, phase, dphase, gc, dgc, gz, dgz, le, dle, gn, dgn in quadrature(split=False):
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


def coupled_contact(obj, base, C, Z, Qbar, Rbar, EE, *, elimination=None):
    """Exact physical limit and derivative, without differentiating singular rows.

    C is the regular/trace cross limit divided by gap, Z is the disappearing
    nodal/trace cross limit, and EE is the trace energy divided by gap.
    The finite physical energy verifies the contact derivative independently.
    """
    k, dk, q, dq, r, dr, qt, dqt, source, dsource, e, de, info = base
    nc, nz, N, ds = info['nc'], info['nz'], len(e), info['ds']
    NE = Qbar.shape[1]
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
    if elimination is None:
        elimination = 'retained' if nc > 4*N else 'potential'
    if elimination == 'retained':
        # The contact value lives entirely in the closed C space. Eliminate
        # its original physical K before applying retained constraints; a
        # large projected K-SQ-Q.HB unnecessarily magnifies roundoff on fine
        # meshes. Both eliminations solve exactly the same weak equations.
        kc, qc, rc = KK[:nc, :nc], QQ[:, :nc], RR[:, :nc]
        v, w = np.split(_solve(kc, np.hstack([QT[:nc], SS[:nc]])), 2, axis=1)
        identity = np.eye(2*N)
        retained = np.block([[identity-qc@w, qc@v],
                             [-A+rc@w, identity-rc@v]])
        field, flux = np.split(_solve(retained, np.vstack([identity, np.zeros_like(identity)])), 2, axis=0)
        uc = v@flux-w@field
        ud = _solve(DD[nc:, nc:], -CC[nc:]-DD[nc:, :nc]@uc)
        U = np.vstack([uc, ud])
    elif elimination == 'potential':
        U = _solve(DD, -CC)
    else:
        raise ValueError('unknown contact Galerkin elimination')
    H = A + BB @ U
    dQQ = np.hstack([dq, ds * Qbar])
    dRR = np.hstack([dr, ds * Rbar])
    dBB = dRR - DA @ QQ - A @ dQQ
    dKK = np.hstack([dk[:nc], ds * C])
    dDD = dKK - dsource[:nc] @ QQ - source[:nc] @ dQQ - dqt[:nc] @ BB - qt[:nc] @ dBB
    dCC = dsource[:nc] - dqt[:nc] @ A - qt[:nc] @ DA
    if elimination == 'retained':
        # Differentiate the same closed physical equations with the full
        # gap/contact forcing, including every disappearing potential.
        field = np.eye(2*N)-QQ@U
        weak_rhs = -dKK@U-dsource[:nc]@field+dqt[:nc]@H
        particular = _solve(kc, weak_rhs)
        field_rhs = -dQQ@U
        flux_rhs = DA@field+dRR@U
        dfield, DH = np.split(_solve(retained,
            np.vstack([field_rhs-qc@particular, flux_rhs+rc@particular])), 2, axis=0)
        dUC = v@DH-w@dfield+particular
    else:
        dUC = _solve(DD[:nc, :nc], -dCC-dDD@U)
        DH = DA+dBB@U+BB[:, :nc]@dUC
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
    meta['closed_elimination'] = elimination
    inv = _solve(e, np.eye(N))
    return (_slots(H, inv), _slots(DH, -inv @ de @ inv), meta)

