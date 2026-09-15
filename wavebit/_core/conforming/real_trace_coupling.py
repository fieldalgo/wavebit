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


def _field_product(field, coefficients):
    """Apply all quadrature/component rows in one dense multiplication."""
    count, components, columns = field.shape
    return (field.reshape(count*components, columns)@coefficients).reshape(
        count, components, coefficients.shape[1])


def coupled_positive(obj, base, quadrature, free_energy, trace_order, *, elimination=None, response_solver=None, normalized_quadrature=None, energy_verifier=None):
    """Solve the original weak system and verify reconstructed physical fields.

    Pure Fourier energy is supplied independently. Added mixed/trace energies
    use the moving partition, while baseline mesh terms use unsplit triangles.
    The differentiated energy is the stationary reciprocal bilinear form.
    An optional verifier receives x, its fixed-potential geometric variation,
    and U; it never receives the candidate H/DH or the solved variation of U.
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
    if normalized_quadrature is not None:
        # Divide the thin measure by gap before differentiating. Applying the
        # quotient rule to already integrated O(gap) blocks loses the finite
        # normalized derivative through cancellation at subpixel gaps.
        rowC=np.zeros((NE,nc),complex);drowC=np.zeros_like(rowC)
        Qbar=np.zeros_like(Q);dQbar=np.zeros_like(Q)
        Rbar=np.zeros_like(Q);dRbar=np.zeros_like(Q)
        for row_data in normalized_quadrature():
            (ids,ci,zi,w,dw,mat,dmat,phase,dphase,gc,dgc,gz,dgz,
             le,dle,gn,dgn,wb,dwb,sgz,dsgz,test,dtest)=row_data
            moment=_contract('q,qm,qaj->amj',wb,phase,test).reshape(2*N,NE)
            dm=(_contract('q,qm,qaj->amj',dwb,phase,test)
                +_contract('q,qm,qaj->amj',wb,dphase,test)
                +_contract('q,qm,qaj->amj',wb,phase,dtest)).reshape(2*N,NE)
            Qbar+=moment;dQbar+=dm
            Rbar+=mat*moment;dRbar+=dmat*moment+mat*dm
            for indices,g,dg in ((ci,gc,dgc),(nc+zi,sgz,dsgz)):
                cross=_contract('q,qai,qaj->ij',wb,g,test)
                dcross=(_contract('q,qai,qaj->ij',dwb,g,test)
                        +_contract('q,qai,qaj->ij',wb,dg,test)
                        +_contract('q,qai,qaj->ij',wb,g,dtest))
                if indices is ci:
                    rowC[:,ci]+=mat*cross.T
                    drowC[:,ci]+=dmat*cross.T+mat*dcross.T
                else:
                    K[indices]+=mat*cross
                    dK[indices]+=dmat*cross+mat*dcross
            en=_contract('q,qai,qaj->ij',wb,test,le)
            den=(_contract('q,qai,qaj->ij',dwb,test,le)
                 +_contract('q,qai,qaj->ij',wb,dtest,le)
                 +_contract('q,qai,qaj->ij',wb,test,dle))
            EE+=mat*en;dEE+=dmat*en+mat*den
        K[:nc]=s*rowC.T;dK[:nc]=ds*rowC.T+s*drowC.T
        Q=s*Qbar;dQ=ds*Qbar+s*dQbar
        R=s*Rbar;dR=ds*Rbar+s*dRbar
        row=np.hstack([rowC,K[nc:].T]);drow=np.hstack([drowC,dK[nc:].T])
        KK=np.block([[k,K],[row,EE]]);dKK=np.block([[dk,dK],[drow,dEE]])
        QQ=np.hstack([q,Q]);dQQ=np.hstack([dq,dQ])
        RR=np.hstack([r,R]);dRR=np.hstack([dr,dR])
        QT=np.vstack([qt,Qbar.conj().T]);dQT=np.vstack([dqt,dQbar.conj().T])
        SS=np.vstack([source,Rbar[info['reverse']].T])
        dSS=np.vstack([dsource,dRbar[info['reverse']].T])
    else:
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
    if elimination is None:
        elimination = obj._elimination(len(KK), 2*N)
    if response_solver is None:
        U, dU, x, dx, H, DH, cond = _response(KK, dKK, QQ, dQQ, RR, dRR, QT, dQT, SS, dSS, A, DA, elimination)
    else:
        U, dU, x, dx, H, DH, cond = response_solver(QQ, dQQ, (KK,dKK,RR,dRR,QT,dQT,SS,dSS,A,DA))
        elimination = 'physical_projected'
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
    if response_solver is not None:
        energy=np.zeros_like(energy)
    denergy = (dxr @ exact_free @ x + xr @ exact_free_derivative @ x
               + xr @ exact_free @ dx_geometry)
    direct_energy = response_solver is not None and normalized_quadrature is not None
    if direct_energy:
        denergy = np.zeros_like(denergy)
    physical_rows = normalized_quadrature() if direct_energy else quadrature()
    for physical_row in physical_rows:
        ids, ci, zi, w, dw, mat, dmat, phase, dphase, gc, dgc, gz, dgz, le, dle, gn, dgn = physical_row[:17]
        F = np.zeros((len(w), 2, 2 * N), complex)
        DF = F.copy()
        F[:, 0, :N] = phase.conj()
        F[:, 1, N:] = phase.conj()
        DF[:, 0, :N] = dphase.conj()
        DF[:, 1, N:] = dphase.conj()
        if direct_energy:
            # The physical s*Z field and its variation are already finite.
            # Recovering d(s*Z) as ds*Z+s*dZ would subtract inverse-gap terms.
            sgz, dsgz = physical_row[19:21]
            E = (_field_product(F, x) + _field_product(gc, U[ci])
                 + _field_product(sgz, U[nc+zi]) + _field_product(le, U[nc+nz:]))
            DE = (_field_product(DF, x) + _field_product(F, dx)
                  + _field_product(dgc, U[ci]) + _field_product(gc, dU[ci])
                  + _field_product(dsgz, U[nc+zi]) + _field_product(sgz, dU[nc+zi])
                  + _field_product(dle, U[nc+nz:]) + _field_product(le, dU[nc+nz:]))
        else:
            E = _field_product(F, x) + _field_product(gc, U[ci]) + s * gz @ U[nc + zi] + _field_product(le, U[nc + nz:])
            DE = _field_product(DF, x) + _field_product(F, dx) + _field_product(dgc, U[ci]) + _field_product(gc, dU[ci]) + ds * gz @ U[nc + zi] + s * dgz @ U[nc + zi] + s * gz @ dU[nc + zi] + _field_product(dle, U[nc + nz:]) + _field_product(le, dU[nc + nz:])
        T = mat * E - _field_product(F, H)
        DT = dmat * E + mat * DE - _field_product(DF, H) - _field_product(F, DH)
        react = _contract('q,qai,qaj->ij', w, gn, T)
        change = _contract('q,qai,qaj->ij', dw, gn, T) + _contract('q,qai,qaj->ij', w, dgn, T) + _contract('q,qai,qaj->ij', w, gn, DT)
        np.add.at(res, ids, react)
        np.add.at(dres, ids, change)
        np.add.at(res_scale, ids, _contract('q,qai,qaj->ij', abs(w), abs(gn), abs(mat * E) + abs(_field_product(F, H))))
        np.add.at(dres_scale, ids, _contract('q,qai,qaj->ij', abs(dw), abs(gn), abs(T)) + _contract('q,qai,qaj->ij', abs(w), abs(dgn), abs(T)) + _contract('q,qai,qaj->ij', abs(w), abs(gn), abs(DT)))
        if energy_verifier is not None:
            continue
        if response_solver is not None:
            # Nearly coincident polynomial and retained fields have large
            # cancelling coefficients. Integrate their complete physical E.
            energy += mat * _contract('q,qai,qaj->ij', w, E.conj(), E)
        else:
            free = _field_product(F, x)
            correction = _field_product(gc, U[ci]) + s * gz @ U[nc + zi] + _field_product(le, U[nc + nz:])
            for left, right in ((free, correction), (correction, free), (correction, correction)):
                energy += mat * _contract('q,qai,qaj->ij', w, left.conj(), right)
        if direct_energy:
            # Differentiate the complete physical field at fixed auxiliary
            # coefficients. This independent stationary variation consumes
            # neither the candidate tensor JVP nor its potential derivative.
            stationary = (_field_product(DF, x) + _field_product(F, dx_geometry)
                          + _field_product(dgc, U[ci]) + _field_product(dsgz, U[nc+zi])
                          + _field_product(dle, U[nc+nz:]))
            denergy += (_contract('q,qai,qaj->ij', dmat*w+mat*dw, E[:, :, reverse], E)
                        + mat*_contract('q,qai,qaj->ij', w, stationary[:, :, reverse], E)
                        + mat*_contract('q,qai,qaj->ij', w, E[:, :, reverse], stationary))
            continue
        reverse = info['reverse']
        added = _field_product(le, U[nc + nz:])
        dadded = _field_product(dle, U[nc + nz:])
        old = _field_product(F, x) + _field_product(gc, U[ci]) + s * gz @ U[nc + zi]
        dold = _field_product(DF, x) + _field_product(dgc, U[ci]) + ds * gz @ U[nc + zi] + s * dgz @ U[nc + zi] - _field_product(F, dQQ @ U)
        for L, DL, R, DR in ((old, dold, added, dadded), (added, dadded, old, dold), (added, dadded, added, dadded)):
            denergy += dmat * _contract('q,qai,qaj->ij', w, L[:, :, reverse], R) + mat * _contract('q,qai,qaj->ij', dw, L[:, :, reverse], R) + mat * _contract('q,qai,qaj->ij', w, DL[:, :, reverse], R) + mat * _contract('q,qai,qaj->ij', w, L[:, :, reverse], DR)
    unsplit_rows = () if direct_energy or energy_verifier is not None else quadrature(split=False)
    for ids, ci, zi, w, dw, mat, dmat, phase, dphase, gc, dgc, gz, dgz, le, dle, gn, dgn in unsplit_rows:
        F = np.zeros((len(w), 2, 2 * N), complex)
        DF = F.copy()
        F[:, 0, :N] = phase.conj()
        F[:, 1, N:] = phase.conj()
        DF[:, 0, :N] = dphase.conj()
        DF[:, 1, N:] = dphase.conj()
        free = _field_product(F, x)
        dfree = _field_product(DF, x) - _field_product(F, dQQ @ U)
        correction = _field_product(gc, U[ci]) + s * gz @ U[nc + zi]
        dcorrection = _field_product(dgc, U[ci]) + ds * gz @ U[nc + zi] + s * dgz @ U[nc + zi]
        reverse = info['reverse']
        for left, dl, right, dright in ((free, dfree, correction, dcorrection),
                                       (correction, dcorrection, free, dfree),
                                       (correction, dcorrection, correction, dcorrection)):
            denergy += (dmat * _contract('q,qai,qaj->ij', w, left[:, :, reverse], right)
                        + mat * _contract('q,qai,qaj->ij', dw, left[:, :, reverse], right)
                        + mat * _contract('q,qai,qaj->ij', w, dl[:, :, reverse], right)
                        + mat * _contract('q,qai,qaj->ij', w, left[:, :, reverse], dright))
    if energy_verifier is not None:
        energy, denergy = energy_verifier(x, dx_geometry, U)
    rr = info['reverse']
    meta = dict(condition=cond, reciprocity=_relative(H - H[rr][:, rr].T, H), derivative_reciprocity=_relative(DH - DH[rr][:, rr].T, DH), original_residual=_relative(res, res_scale), original_derivative_residual=_relative(dres, dres_scale), energy_residual=_relative(energy - H, H), energy_derivative_residual=_relative(denergy - DH, DH), minimum_rotated_storage=minimum_rotated_storage(H, obj._phase), minimum_dissipation=np.linalg.eigvalsh((H - H.conj().T) / 2j)[0])
    meta['weak_residual'] = _relative(KK @ U + SS @ x - QT @ H, KK @ U, SS @ x, QT @ H)
    meta['weak_derivative_residual'] = _relative(dKK @ U + KK @ dU + dSS @ x + SS @ dx - dQT @ H - QT @ DH, dKK @ U, KK @ dU, dSS @ x, SS @ dx, dQT @ H, QT @ DH)
    meta['direct_fourier_residual'] = _relative(A - exact_free, exact_free)
    meta['direct_fourier_derivative_residual'] = _relative(DA - exact_free_derivative, exact_free_derivative)
    meta['retained_residual'] = _relative(x+QQ@U-np.eye(2*N), np.eye(2*N))
    meta['flux_residual'] = _relative(A@x+RR@U-H, A@x, RR@U, H)
    meta['retained_derivative_residual'] = _relative(dx+dQQ@U+QQ@dU, dx, dQQ@U, QQ@dU)
    meta['flux_derivative_residual'] = _relative(DA@x+A@dx+dRR@U+RR@dU-DH,
                                               DA@x, A@dx, dRR@U, RR@dU, DH)
    meta['elimination'] = elimination
    inv = _solve(e, np.eye(N))
    return (_slots(H, inv), _slots(DH, -inv @ de @ inv), meta)


def coupled_contact(obj, base, C, Z, Qbar, Rbar, EE, *, elimination=None, energy_verifier=None, response_solver=None, energy_coordinates=None):
    """Exact physical limit and derivative, without differentiating singular rows.

    C is the regular/trace cross limit divided by gap, Z is the disappearing
    nodal/trace cross limit, and EE is the trace energy divided by gap.
    The finite physical energy verifies the contact derivative independently.
    Optional energy coordinates carry the same field, fixed-potential field
    variation and potential coefficients in their solver precision. They do
    not contain the candidate response or its derivative.
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
    if response_solver is None:
        BB = RR - A @ QQ
        CC = SS - QT @ A
        DD = KK - SS @ QQ - QT @ BB
    if elimination is None:
        elimination = obj._elimination(nc, 2*N)
    if response_solver is not None:
        U, dUC, field, dfield, H, DH, condition = response_solver()
        elimination = 'physical_projected'
    elif elimination == 'retained':
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
        H = flux
    elif elimination == 'potential':
        U = _solve(DD, -CC)
        H = A + BB@U
        field = np.eye(2*N)-QQ@U
    else:
        raise ValueError('unknown contact Galerkin elimination')
    dQQ = np.hstack([dq, ds * Qbar])
    dRR = np.hstack([dr, ds * Rbar])
    dKK = np.hstack([dk[:nc], ds * C])
    if response_solver is None:
        dBB = dRR - DA @ QQ - A @ dQQ
        dDD = dKK - dsource[:nc] @ QQ - source[:nc] @ dQQ - dqt[:nc] @ BB - qt[:nc] @ dBB
        dCC = dsource[:nc] - dqt[:nc] @ A - qt[:nc] @ DA
        condition = np.linalg.cond(DD)
    if elimination == 'retained':
        # Differentiate the same closed physical equations with the full
        # gap/contact forcing, including every disappearing potential.
        weak_rhs = -dKK@U-dsource[:nc]@field+dqt[:nc]@H
        particular = _solve(kc, weak_rhs)
        field_rhs = -dQQ@U
        flux_rhs = DA@field+dRR@U
        dfield, DH = np.split(_solve(retained,
            np.vstack([field_rhs-qc@particular, flux_rhs+rc@particular])), 2, axis=0)
        dUC = v@DH-w@dfield+particular
    elif elimination == 'potential':
        dUC = _solve(DD[:nc, :nc], -dCC-dDD@U)
        DH = DA+dBB@U+BB[:, :nc]@dUC
        dfield = -dQQ@U-QQ[:, :nc]@dUC
    x = field
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
    if energy_verifier is not None:
        coordinates=(x,dx_geometry,U) if energy_coordinates is None else energy_coordinates
        energy,denergy=energy_verifier(*coordinates)
    weak = _relative(KK @ U + SS @ x - QT @ H, KK @ U, SS @ x, QT @ H)
    terms = (dKK@U, KK[:nc, :nc]@dUC, dsource[:nc]@x,
             source[:nc]@dfield, -dqt[:nc]@H, -qt[:nc]@DH)
    cweak = _relative(sum(terms), *terms)
    meta = dict(condition=condition, reciprocity=_relative(H - H[rr][:, rr].T, H), derivative_reciprocity=_relative(DH - DH[rr][:, rr].T, DH), weak_residual=weak, weak_derivative_residual=None, contact_derivative_residual=cweak, energy_residual=_relative(energy - H, H), energy_derivative_residual=_relative(denergy - DH, DH), minimum_rotated_storage=minimum_rotated_storage(H, obj._phase), minimum_dissipation=np.linalg.eigvalsh((H - H.conj().T) / 2j)[0])
    meta['closed_elimination'] = elimination
    meta['retained_residual'] = _relative(x+QQ@U-np.eye(2*N), np.eye(2*N))
    meta['flux_residual'] = _relative(A@x+RR@U-H, A@x, RR@U, H)
    meta['retained_derivative_residual'] = _relative(dfield+dQQ@U+QQ[:, :nc]@dUC,
                                                   dfield, dQQ@U, QQ[:, :nc]@dUC)
    meta['flux_derivative_residual'] = _relative(DA@x+A@dfield+dRR@U+RR[:, :nc]@dUC-DH,
                                               DA@x, A@dfield, dRR@U, RR[:, :nc]@dUC, DH)
    inv = _solve(e, np.eye(N))
    return (_slots(H, inv), _slots(DH, -inv @ de @ inv), meta)
