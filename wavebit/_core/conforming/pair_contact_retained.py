"""Complete contact elimination through the original physical stiffness.

Solve for every common potential through K, impose both retained constraints
in a smaller complete block, then recover all disappearing potentials and
physical variations. The independent verifier receives full field and
potential coefficients, without candidate response information.
"""
import numpy as np
from .pair_complete_solve import factor as complete_factor, apply as complete_apply
from .pair_contact_precision import pair, mat, cp, add, neg, transpose, solve


def stack(values, axis=0):
    return tuple(np.concatenate([value[i] for value in values], axis=axis)
                 for i in range(2))


def reverse_transpose(value, reverse):
    return tuple(part[:, reverse].T for part in value)


def response(base, free, dfree, K, DK, Q, DQ, R, DR, L, SZ, MZZ, qb, assembled):
    info = base[-1]
    reverse, ds, size = info['reverse'], info['ds'], len(Q[0])
    A = tuple(np.kron(np.eye(2), value) for value in free)
    DA = tuple(np.kron(np.eye(2), value) for value in dfree)
    QT, DQT = transpose(Q, True), transpose(DQ, True)
    S = tuple(value[reverse].T for value in R)
    DS = tuple(value[reverse].T for value in DR)
    identity = pair(np.eye(size))

    # K*uc + S*x = Q.H*H, x + Q*uc = I, H = A*x + R*uc.
    # All common columns enter K. The smaller block only changes elimination.
    columns = solve(K, stack((QT, S), 1))
    V, W = (tuple(value[:, section] for value in columns)
            for section in (slice(None, size), slice(size, None)))
    aa, bb = add(identity, neg(mat(Q, W))), mat(Q, V)
    cc, dd = add(neg(A), mat(R, W)), add(identity, neg(mat(R, V)))
    retained = stack((stack((aa, bb), 1), stack((cc, dd), 1)))
    factorization = complete_factor(retained)
    answer = complete_apply(factorization,
                            stack((identity, pair(np.zeros((size, size))))))
    x, H = (tuple(value[section] for value in answer)
            for section in (slice(None, size), slice(size, None)))
    uc = add(mat(V, H), neg(mat(W, x)))
    uz = solve(MZZ, neg(add(mat(L, uc), mat(SZ, x))))

    # The complete physical variation includes every disappearing normal
    # potential. This forcing differentiates the original equations, not
    # the computational factorization used to solve them.
    MZ = add(L, neg(mat(SZ, Q)))
    SC = add(S, neg(mat(QT, A)))
    DSC = add(DS, neg(mat(DQT, A)), neg(mat(QT, DA)))
    dxgeo = neg(mat(DQ, uc))
    forcing = add(mat(DK, uc), mat(DS, x), mat(S, dxgeo), neg(mat(DQT, H)),
                  neg(mat(QT, add(mat(DR, uc), mat(DA, x), mat(A, dxgeo)))),
                  cp(pair(ds), mat(transpose(MZ), uz)))
    particular = solve(K, neg(forcing))
    answer = complete_apply(factorization,
                            stack((neg(mat(Q, particular)), mat(R, particular))))
    xi, eta = (tuple(value[section] for value in answer)
               for section in (slice(None, size), slice(size, None)))
    duc = add(particular, neg(mat(W, xi)), mat(V, eta))
    dx = neg(add(mat(DQ, uc), mat(Q, duc), cp(pair(ds), mat(qb, uz))))
    DH = add(DA, mat(reverse_transpose(DSC, reverse), uc),
             mat(reverse_transpose(SC, reverse), duc),
             cp(pair(ds), mat(reverse_transpose(SZ, reverse), uz)))
    U = stack((uc, uz))
    geometry_dx = neg(add(mat(DQ, uc), cp(pair(ds), mat(qb, uz))))
    if assembled is not None:
        assembled['energy_coordinates'] = (x, geometry_dx, U)
        assembled['elimination'] = 'physical_retained'
        assembled['contact_solve_size'] = 2*size
    condition = float(np.linalg.cond(retained[0]+retained[1]))
    return (*(high+low for high, low in (U, duc, x, dx, H, DH)), condition)
