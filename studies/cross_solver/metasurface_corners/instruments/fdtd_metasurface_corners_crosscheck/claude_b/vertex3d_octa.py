"""3D vertex exponents for octant-wise piecewise-constant eps (box corners).

phi = R^lam g(angles); on the unit sphere: -div_S(eps grad_S g) = mu eps g,
mu = lam (lam + 1). P1 finite elements on an octahedral subdivision of the sphere:
each octant is one face of the octahedron, so every eps interface is a mesh edge.
The angular eigenfunction is itself singular at the points where edges pierce
the sphere (strength = the 2D edge exponent), so mu converges like h^(2 nu_edge);
the rate is fitted from successive levels and used for Richardson extrapolation.
"""
import itertools
import math
import time
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import eigsh
import mpmath as mp


def octa_mesh(n):
    key2id, pts, tris, tri_oct = {}, [], [], []

    def node(key):
        if key not in key2id:
            key2id[key] = len(pts)
            v = np.array(key, float)
            pts.append(v / np.linalg.norm(v))
        return key2id[key]

    for sx, sy, sz in itertools.product((1, -1), repeat=3):
        def nid(i, j):
            k = n - i - j
            return node((sx * i, sy * j, sz * k))
        for i in range(n):
            for j in range(n - i):
                tris.append((nid(i, j), nid(i + 1, j), nid(i, j + 1))); tri_oct.append((sx, sy, sz))
                if i + j <= n - 2:
                    tris.append((nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1))); tri_oct.append((sx, sy, sz))
    return np.array(pts), np.array(tris), tri_oct


def assemble(pts, tris, eps_of_tri):
    p0, p1, p2 = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    e1, e2 = p1 - p0, p2 - p0
    nrm = np.cross(e1, e2)
    A2 = np.linalg.norm(nrm, axis=1)            # twice the area
    nh = nrm / A2[:, None]
    # gradients of barycentric functions on the flat triangle: grad L_i = (n x e_opp)/(2A)
    g0 = np.cross(nh, p2 - p1) / A2[:, None]
    g1 = np.cross(nh, p0 - p2) / A2[:, None]
    g2 = np.cross(nh, p1 - p0) / A2[:, None]
    G = np.stack([g0, g1, g2], axis=1)          # (T,3,3)
    area = A2 / 2
    Ke = np.einsum('tid,tjd->tij', G, G) * (eps_of_tri * area)[:, None, None]
    Mloc = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]]) / 12.0
    Me = Mloc[None] * (eps_of_tri * area)[:, None, None]
    rows = np.repeat(tris, 3, axis=1).ravel()
    cols = np.tile(tris, (1, 3)).ravel()
    N = len(pts)
    K = sps.csr_matrix((Ke.ravel(), (rows, cols)), shape=(N, N))
    M = sps.csr_matrix((Me.ravel(), (rows, cols)), shape=(N, N))
    return K, M


def vertex_lams(eps_fn, n, k=5):
    pts, tris, octs = octa_mesh(n)
    eps = np.array([eps_fn(o) for o in octs], float)
    K, M = assemble(pts, tris, eps)
    vals = eigsh(K, k=k, M=M, sigma=-0.01, which='LM', return_eigenvectors=False)
    mu = np.sort(vals)[1:]                        # drop the constant mode
    lam = (-1 + np.sqrt(1 + 4 * mu)) / 2
    return lam, len(pts)


def extrapolate(seq):
    """seq: values at h, h/2, h/4,...; fit rate from last three, Richardson the last two."""
    a, b, c = seq[-3:]
    r = (a - b) / (b - c)
    p = math.log2(r) if r > 0 else float('nan')
    ext = c + (c - b) / (2 ** p - 1) if p == p else float('nan')
    return ext, p


def run(label, eps_fn, levels=(16, 32, 64, 128), k=5, show=True):
    seq = []
    t0 = time.time()
    for n in levels:
        lam, N = vertex_lams(eps_fn, n, k)
        seq.append(lam[0])
        if show:
            print(f"   {label:46s} n={n:4d} nodes={N:7d} lam1={lam[0]:.7f}  next: "
                  + ' '.join(f"{x:.5f}" for x in lam[1:3]))
    ext, p = extrapolate(seq)
    print(f"   {label:46s} extrapolated lam1 = {ext:.5f}  (fitted rate h^{p:.2f}; last raw {seq[-1]:.5f}; {time.time()-t0:.0f}s)")
    return ext, p, seq


def cone_exterior_lam(beta):
    """Dirichlet on a circular cone of half-angle beta, field outside:
    smallest lam > 0 with P_lam(cos(pi - beta)) = 0."""
    x = -math.cos(beta)
    f = lambda l: mp.hyp2f1(-l, l + 1, 1, (1 - x) / 2)
    ls = np.linspace(1e-3, 1.0, 400)
    v = [float(f(l)) for l in ls]
    for i in range(len(ls) - 1):
        if v[i] * v[i + 1] < 0:
            return float(mp.findroot(f, (ls[i], ls[i + 1]), solver='bisect'))
    return float('nan')


if __name__ == '__main__':
    print("=== validation ===")
    lam, N = vertex_lams(lambda o: 1.0, 32)
    print(f"   uniform eps, n=32: lam = {lam[:3]} (exact 1,1,1)")
    b_in, b_out = math.asin(1 / math.sqrt(3)), math.acos(1 / math.sqrt(3))
    print(f"   Dirichlet bracket for the 7-octant exterior of a cube corner (cone comparison):"
          f" {cone_exterior_lam(b_in):.5f} <= lam <= {cone_exterior_lam(b_out):.5f}")
    big = 1e6
    oct_in = (1, 1, 1)
    run("Dirichlet limit: 1 octant eps=1e6, 7 air", lambda o: big if o == oct_in else 1.0)
    run("Neumann limit: 1 air octant, 7 octants eps=1e6", lambda o: 1.0 if o == oct_in else big)

    print("\n=== dielectric box corners ===")
    res = {}
    for ep in [12.0, 5.8, 4.0]:
        res[('top', ep)] = run(f"box top corner: 1 octant eps={ep}, 7 air", lambda o, ep=ep: ep if o == oct_in else 1.0)
        res[('hole', ep)] = run(f"hole corner: 1 air octant in eps={ep}", lambda o, ep=ep: 1.0 if o == oct_in else ep)
    for ep, es in [(12.0, 2.25), (5.8, 2.13), (4.0, 2.25)]:
        def f(o, ep=ep, es=es):
            if o[2] < 0:
                return es
            return ep if o == oct_in else 1.0
        res[('foot', ep, es)] = run(f"pillar foot: eps {ep} octant / air / sub {es}", f)
