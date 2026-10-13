"""3D vertex exponents: phi = R^lam f(sphere point) solves div(eps grad phi)=0
near a vertex iff  -div_S(eps grad_S f) = lam(lam+1) eps f  on the unit sphere
(f, eps d_n f continuous across the octant planes).  P1 FEM on an equiangular
cube-sphere whose grid lines contain the three great circles x=0, y=0, z=0, so
every triangle lies inside one octant.  Field |E| ~ R^(lam-1).

Integrated-error power from a vertex: int_{R<dx} R^(2lam-2) R^2 dR ~ dx^(2lam+1);
from an edge (exponent nu): dx^(2nu) per unit length.  Vertex dominates iff
2 lam + 1 < 2 nu_edge.

(Restored copy: the first version in ../vertex3d.py was overwritten by a
concurrent session; this is the same source, the logs in this folder came
from it.)
"""
import sys, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def cubesphere(n):
    a = np.linspace(-np.pi / 4, np.pi / 4, n + 1)
    A, B = np.meshgrid(np.tan(a), np.tan(a), indexing='ij')
    one = np.ones_like(A)
    faces = [(one, A, B), (-one, A, B), (A, one, B), (A, -one, B), (A, B, one), (A, B, -one)]
    P, T = [], []
    off = 0
    for X, Y, Z in faces:
        pts = np.stack([X, Y, Z], -1).reshape(-1, 3)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        P.append(pts)
        idx = off + np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
        q0 = idx[:-1, :-1].ravel(); q1 = idx[1:, :-1].ravel()
        q2 = idx[1:, 1:].ravel(); q3 = idx[:-1, 1:].ravel()
        T.append(np.stack([q0, q1, q2], 1)); T.append(np.stack([q0, q2, q3], 1))
        off += (n + 1) ** 2
    P = np.concatenate(P); T = np.concatenate(T)
    key = np.round(P * 1e10).astype(np.int64)
    _, uniq, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return P[uniq], inv.ravel()[T]


def assemble(P, T, epsfun):
    p0, p1, p2 = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    c = (p0 + p1 + p2) / 3
    eps = epsfun(c)
    e0, e1, e2 = p2 - p1, p0 - p2, p1 - p0            # edge opposite vertex i
    area = 0.5 * np.linalg.norm(np.cross(e2, -e1), axis=1)
    E = [e0, e1, e2]
    npnt = P.shape[0]
    rows, cols, kv, mv = [], [], [], []
    for i in range(3):
        for j in range(3):
            rows.append(T[:, i]); cols.append(T[:, j])
            kv.append(eps * np.einsum('ij,ij->i', E[i], E[j]) / (4 * area))
            mv.append(eps * area * (2.0 if i == j else 1.0) / 12.0)
    rows = np.concatenate(rows); cols = np.concatenate(cols)
    K = sp.coo_matrix((np.concatenate(kv), (rows, cols)), shape=(npnt, npnt)).tocsc()
    M = sp.coo_matrix((np.concatenate(mv), (rows, cols)), shape=(npnt, npnt)).tocsc()
    return K, M


def lams(n, epsfun, k=6):
    P, T = cubesphere(n)
    K, M = assemble(P, T, epsfun)
    mu = spla.eigsh(K, k=k, M=M, sigma=-0.05, which='LM', return_eigenvectors=False)
    mu = np.sort(mu)
    lam = 0.5 * (-1 + np.sqrt(1 + 4 * np.clip(mu, 0, None)))
    return lam, P.shape[0]


def octant_eps(table):
    """table: dict sign-tuple -> eps (default 1)."""
    def f(c):
        s = np.sign(c)
        out = np.ones(c.shape[0])
        for key, val in table.items():
            m = np.all(s == np.array(key), axis=1)
            out[m] = val
        return out
    return f


def lower(eps_s):
    return {(sx, sy, -1): eps_s for sx in (1, -1) for sy in (1, -1)}


CONFIGS = {
    'uniform (expect lam=1)': octant_eps({}),
    'cube corner, octant eps=1e6 in air (Dirichlet-exterior limit)': octant_eps({(1, 1, 1): 1e6}),
    'cube corner, eps-12 octant in air (pillar top vertex, eps 12)': octant_eps({(1, 1, 1): 12.0}),
    'cube corner, air octant in eps 12 (hole corner)': octant_eps({(1, 1, 1): 1.0, **{k: 12.0 for k in [
        (1, 1, -1), (1, -1, 1), (-1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)]}}),
    'pillar base vertex 12 | air | substrate 2.25': octant_eps({(-1, -1, 1): 12.0, **lower(2.25)}),
    'pillar base vertex 5.8 | air | substrate 2.13': octant_eps({(-1, -1, 1): 5.8, **lower(2.13)}),
    'pillar base vertex 4 | air | substrate 2.25': octant_eps({(-1, -1, 1): 4.0, **lower(2.25)}),
    'cube corner, eps-5.8 octant in air': octant_eps({(1, 1, 1): 5.8}),
    'cube corner, eps-4 octant in air': octant_eps({(1, 1, 1): 4.0}),
}

if __name__ == '__main__':
    ns = [int(x) for x in sys.argv[1:]] or [16, 32, 64, 128]
    for name, f in CONFIGS.items():
        print(f"\n{name}")
        vals = []
        for n in ns:
            t = time.time()
            lam, npnt = lams(n, f)
            v = lam[lam > 1e-4][:3]
            vals.append(v[0])
            line = f"   n/face={n:4d} nodes={npnt:7d}  lam={np.array2string(v, precision=6)}  {time.time()-t:5.1f}s"
            if len(vals) >= 3:
                a, b, c = vals[-3:]
                q = (a - b) / (b - c)
                if q > 1:
                    p = np.log2(q)
                    line += f"\n        obs.order={p:.3f}  extrap={c + (c - b)/(2**p - 1):.5f}"
            print(line, flush=True)
