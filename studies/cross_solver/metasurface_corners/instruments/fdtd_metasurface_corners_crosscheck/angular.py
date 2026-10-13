"""Angular eigenproblem for a 2D dielectric corner, solved numerically.

phi = r^nu f(theta) solves div(eps grad phi) = 0 near the vertex iff, on the
periodic circle theta in [0, 2 pi),

    -(eps f')' = nu^2 eps f,   f and eps f' continuous at sector edges.

Weak form: sum_k eps_k int (f' g' - nu^2 f g) = 0, i.e. K f = lam M f with
lam = nu^2 and both K and M eps-weighted.  The interface condition
(eps f' continuous) is natural in the weak form, so no closed form is used.

Two discretisations:
  * sem(...)  -- hierarchical (integrated-Legendre) spectral elements, one or
                 more elements per sector, degree p; exact Gauss quadrature.
  * p1(...)   -- linear finite elements on a uniform mesh whose nodes include
                 the sector edges (consistent mass); O(h^2), Richardson.
"""
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.polynomial import legendre as L

TWO_PI = 2.0 * np.pi


def sectors_from_deg(spec):
    """spec: list of (angle_deg, eps) counter-clockwise; must sum to 360."""
    ang = np.array([a for a, _ in spec], float)
    assert abs(ang.sum() - 360.0) < 1e-9, ang.sum()
    eps = np.array([e for _, e in spec], float)
    return np.deg2rad(ang), eps


def _basis(p, x):
    """Integrated-Legendre basis on [-1,1]: 2 vertex + (p-1) bubbles.
    Returns values (nb, nx) and x-derivatives (nb, nx)."""
    nb = p + 1
    V = np.zeros((nb, x.size))
    D = np.zeros((nb, x.size))
    V[0] = (1 - x) / 2
    D[0] = -0.5
    V[1] = (1 + x) / 2
    D[1] = 0.5
    for k in range(2, p + 1):
        ck = np.zeros(k + 1); ck[k] = 1.0
        ckm2 = np.zeros(k + 1); ckm2[k - 2] = 1.0
        s = 1.0 / np.sqrt(2.0 * (2 * k - 1))
        V[k] = s * (L.legval(x, ck) - L.legval(x, ckm2))
        # d/dx (P_k - P_{k-2}) = (2k-1) P_{k-1}
        ckm1 = np.zeros(k + 1); ckm1[k - 1] = 1.0
        D[k] = s * (2 * k - 1) * L.legval(x, ckm1)
    return V, D


def sem_matrices(angles, eps, p, nsub=1):
    """Assemble K, M with nsub equal elements per sector, degree p."""
    xq, wq = L.leggauss(p + 3)
    V, D = _basis(p, xq)
    Kref = (D * wq) @ D.T          # int phi_i' phi_j' dx
    Mref = (V * wq) @ V.T          # int phi_i phi_j dx
    elems = []
    for a, e in zip(angles, eps):
        for _ in range(nsub):
            elems.append((a / nsub, e))
    ne = len(elems)
    nb_bub = p - 1
    ndof = ne + ne * nb_bub        # vertex dofs (periodic) + bubbles
    K = np.zeros((ndof, ndof))
    M = np.zeros((ndof, ndof))
    for ie, (H, e) in enumerate(elems):
        dofs = [ie, (ie + 1) % ne] + [ne + ie * nb_bub + k for k in range(nb_bub)]
        dofs = np.array(dofs)
        K[np.ix_(dofs, dofs)] += e * (2.0 / H) * Kref
        M[np.ix_(dofs, dofs)] += e * (H / 2.0) * Mref
    return K, M


def sem(angles, eps, p, nsub=1, nev=8):
    K, M = sem_matrices(angles, eps, p, nsub)
    lam = sla.eigh(K, M, eigvals_only=True)
    lam = np.sort(lam)
    nu = np.sqrt(np.clip(lam, 0, None))
    return nu[:nev]


def sem_vec(angles, eps, p, nsub=1):
    K, M = sem_matrices(angles, eps, p, nsub)
    lam, X = sla.eigh(K, M)
    return lam, X, K, M


def p1_matrices(angles, eps, n_per_rad):
    """Uniform-ish P1 mesh: each sector gets round(angle*n_per_rad) elements
    (exactly uniform when angles are commensurate)."""
    hs, es = [], []
    for a, e in zip(angles, eps):
        n = int(round(a * n_per_rad))
        hs += [a / n] * n
        es += [e] * n
    hs = np.array(hs); es = np.array(es)
    n = hs.size
    i = np.arange(n); j = (i + 1) % n
    kd = es / hs
    md = es * hs / 3.0
    mo = es * hs / 6.0
    rows = np.concatenate([i, j, i, j]); cols = np.concatenate([i, j, j, i])
    K = sp.coo_matrix((np.concatenate([kd, kd, -kd, -kd]), (rows, cols)), shape=(n, n)).tocsc()
    M = sp.coo_matrix((np.concatenate([md, md, mo, mo]), (rows, cols)), shape=(n, n)).tocsc()
    return K, M, hs.max()


def p1(angles, eps, nel_total, nev=6, sigma=0.3):
    K, M, h = p1_matrices(angles, eps, nel_total / TWO_PI)
    if K.shape[0] <= 1500:
        lam = sla.eigh(K.toarray(), M.toarray(), eigvals_only=True)[:nev]
    else:
        lam = spla.eigsh(K, k=nev, M=M, sigma=sigma, which='LM', return_eigenvectors=False)
        lam = np.sort(lam)
    return np.sqrt(np.clip(np.sort(lam), 0, None)), h


CASES = {
    'A': [(90, 1.0), (270, 12.0)],
    'B': [(90, 12.0), (270, 1.0)],
    'C': [(90, 5.8), (270, 1.0)],
    'D': [(90, 4.0), (270, 1.0)],
    'E': [(90, 12.0), (90, 1.0), (180, 2.25)],
    'F': [(90, 5.8), (90, 1.0), (180, 2.13)],
    'G': [(90, 4.0), (90, 1.0), (180, 2.25)],
}


def singular(nu, lo=1e-3, hi=1.0):
    return [v for v in nu if lo < v < hi - 1e-12]
