"""Independent 2D check: finite-volume solve of div(eps grad phi) = 0 in
PHYSICAL polar coordinates on a geometrically graded annulus
r in [e^{-L}, 1], theta periodic, sector edges on grid nodes.

  outer ring r=1     : Dirichlet phi = g_m(theta)  (12 data: cos/sin m theta, m=1..6)
  inner ring r=e^{-L}: zero flux (a tiny hole; its reflection ~ e^{-2 nu (s+L)})

Fluxes are physical: radial face  eps * r_{j+1/2} * (phi_{j+1}-phi_j)/(r_{j+1}-r_j) * dtheta
                     angular face eps * int dr/r  * (phi_{i+1}-phi_i)/dtheta
with eps integrated over each face (a node on a sector edge sees half of each).

Exponents come from the solution itself: the ring data Phi(s) (Ntheta x 12),
eps-weighted mean removed, is a sum of f_k exp(nu_k s) a_k^T; a matrix pencil
between two rings returns exp(nu_k ds).  No angular eigenproblem is solved.
"""
import sys, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from angular import CASES, sectors_from_deg


def build(spec, ntheta, L):
    ang, eps = sectors_from_deg(spec)
    dth = 2 * np.pi / ntheta
    # eps on angular face i+1/2 (between node i and i+1)
    edges = np.concatenate([[0.0], np.cumsum(ang)])
    thf = (np.arange(ntheta) + 0.5) * dth
    sec = np.searchsorted(edges, thf, side='right') - 1
    eps_face = eps[sec]
    nodes_per = ang / dth
    assert np.allclose(nodes_per, np.round(nodes_per)), "sector edges must sit on nodes"
    eps_node = 0.5 * (eps_face + np.roll(eps_face, 1))   # face integral / dtheta
    hs = dth
    ns = int(round(L / hs))
    s = -ns * hs + hs * np.arange(ns + 1)        # s_0=-L ... s_ns = 0
    r = np.exp(s)
    nunk = ns * ntheta                            # rows j=0..ns-1
    idx = lambda j, i: j * ntheta + (i % ntheta)
    rows, cols, vals = [], [], []
    rhs_cols = []   # (row, i) pairs coupling to the Dirichlet ring
    diag = np.zeros(nunk)
    for j in range(ns):
        rp = np.exp(s[j] + hs / 2)
        Cup = eps_node * dth * rp / (r[j + 1] - r[j])
        if j > 0:
            rm = np.exp(s[j] - hs / 2)
            Cdn = eps_node * dth * rm / (r[j] - r[j - 1])
            lnlen = hs
        else:
            Cdn = None
            lnlen = hs / 2
        Cang = eps_face * lnlen / dth            # face i+1/2
        base = j * ntheta
        ii = np.arange(ntheta)
        # angular
        rows += [base + ii, base + ii]
        cols += [base + (ii + 1) % ntheta, base + (ii - 1) % ntheta]
        vals += [Cang, np.roll(Cang, 1)]
        d = Cang + np.roll(Cang, 1) + Cup
        if j + 1 < ns:
            rows.append(base + ii); cols.append(base + ntheta + ii); vals.append(Cup)
        else:
            rhs_cols.append(Cup)
        if Cdn is not None:
            rows.append(base + ii); cols.append(base - ntheta + ii); vals.append(Cdn)
            d = d + Cdn
        diag[base + ii] = d
    A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(nunk, nunk)).tocsc()
    A = sp.diags(diag) - A
    return A, rhs_cols[0], s, eps_node, ntheta, ns


def solve_rings(spec, ntheta, L, mmax=6):
    A, Ctop, s, w, nt, ns = build(spec, ntheta, L)
    th = 2 * np.pi * np.arange(nt) / nt
    G = np.column_stack([f(m * th) for m in range(1, mmax + 1) for f in (np.cos, np.sin)])
    B = np.zeros((A.shape[0], G.shape[1]))
    B[(ns - 1) * nt:, :] = Ctop[:, None] * G
    lu = spla.splu(A)
    X = lu.solve(B)
    Phi = X.reshape(ns, nt, -1)                    # (ring, theta, rhs)
    Phi = np.concatenate([Phi, G[None]], axis=0)  # include the Dirichlet ring
    # remove the eps-weighted mean (the nu=0 mode) ring by ring
    mean = np.einsum('i,jim->jm', w, Phi) / w.sum()
    Phi = Phi - mean[:, None, :]
    return s, Phi, w


def pencil(Phi, s, w, s0, ds, rtol=1e-10):
    j0 = int(np.argmin(abs(s - s0)))
    d = int(round(ds / (s[1] - s[0])))
    sw = np.sqrt(w)[:, None]
    Y0 = sw * Phi[j0]; Y1 = sw * Phi[j0 + d]
    U, S, Vt = np.linalg.svd(Y0, full_matrices=False)
    R = int(np.sum(S > rtol * S[0]))
    Am = U[:, :R].T @ Y1 @ Vt[:R].T / S[:R]
    mu = np.linalg.eigvals(Am)
    nu = np.sort(np.log(mu).real / (s[j0 + d] - s[j0]))
    return nu, R


if __name__ == '__main__':
    names = sys.argv[1:] or list(CASES)
    L = 24.0
    for name in names:
        spec = CASES[name]
        print(f"\ncase {name}: {spec}   annulus r in [e^-{L:g}, 1]")
        vals = []
        for nt in [64, 128, 256, 512]:
            t = time.time()
            s, Phi, w = solve_rings(spec, nt, L)
            nu, R = pencil(Phi, s, w, s0=-10.0, ds=3.0)
            nu_b, _ = pencil(Phi, s, w, s0=-14.0, ds=3.0)   # window check
            v = nu[nu > 1e-3][0]
            vals.append(v)
            line = (f"  Ntheta={nt:4d} unknowns={Phi.shape[0]*nt:8d} rank={R}"
                    f"  nu={np.array2string(nu[:3], precision=8)}"
                    f"  (window s=-14: {nu_b[nu_b>1e-3][0]:.10f})  {time.time()-t:5.1f}s")
            if len(vals) >= 3:
                a, b, c = vals[-3:]
                line += f"\n      obs.order={np.log2((a-b)/(b-c)):.3f}  richardson={c+(c-b)/3:.9f}"
            print(line, flush=True)
