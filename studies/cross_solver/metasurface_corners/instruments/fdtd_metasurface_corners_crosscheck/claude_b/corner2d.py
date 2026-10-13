"""Quasi-static corner exponents for piecewise-constant dielectric sectors (2D).

phi = r^nu f(theta);  inside sector j: f'' = -nu^2 f;
at each interface: f continuous and eps f' continuous; f periodic in theta.

Four independent routes, no transfer-matrix trace anywhere:
  P  parity (two media): even/odd about the bisector of each sector
  N  nodal NxN system: unknowns are the interface values u_k, each sector is
     the exact sin-interpolant between its two end values; flux balance at
     every interface gives K(nu) u = 0 (symmetric, cyclic)
  D  direct 2Nx2N system for the sector coefficients (A_j, B_j)
  F  P1 finite elements for -(eps f')' = nu^2 eps f on the circle, nodes on
     the interfaces, Richardson-extrapolated in h
plus closed forms (derived below and checked here) for the 90-degree cases.
"""
import math
import numpy as np
from scipy.optimize import brentq
from scipy.linalg import eigh
import mpmath as mp

PI = np.pi
D2R = PI / 180.0


# ---------------------------------------------------------------- parity (P)
def parity_eqs(alpha, e1, e2):
    """Sector 1: width alpha, eps e1, centred at 0. Sector 2: the rest, eps e2.
    a = nu*alpha/2 (half-width of sector 1), b = nu*(pi - alpha/2) (half-width of 2).
    even (f = cos about both bisectors):  e1 tan a + e2 tan b = 0
    odd  (f = sin about both bisectors):  e2 tan a + e1 tan b = 0
    multiplied through by cos a cos b (pole-free)."""
    def even(nu):
        a = nu * alpha / 2; b = nu * (PI - alpha / 2)
        return e1 * np.sin(a) * np.cos(b) + e2 * np.sin(b) * np.cos(a)

    def odd(nu):
        a = nu * alpha / 2; b = nu * (PI - alpha / 2)
        return e2 * np.sin(a) * np.cos(b) + e1 * np.sin(b) * np.cos(a)
    return even, odd


def roots_in(fun, lo=1e-6, hi=1 - 1e-12, n=20001):
    x = np.linspace(lo, hi, n)
    y = np.array([fun(t) for t in x])
    out = []
    for i in range(n - 1):
        if y[i] == 0.0:
            out.append(x[i])
        elif y[i] * y[i + 1] < 0:
            out.append(brentq(fun, x[i], x[i + 1], xtol=1e-16, rtol=1e-15, maxiter=200))
    return out


def parity_roots(alpha, e1, e2):
    ev, od = parity_eqs(alpha, e1, e2)
    return roots_in(ev), roots_in(od)


# ---------------------------------------------------------------- nodal (N)
def nodal_K(nu, widths, eps):
    N = len(widths)
    K = np.zeros((N, N))
    for j in range(N):            # sector j joins interface j and interface j+1
        s = np.sin(nu * widths[j]); c = np.cos(nu * widths[j]); e = eps[j]
        k0, k1 = j, (j + 1) % N
        K[k0, k0] += e * c / s
        K[k1, k1] += e * c / s
        K[k0, k1] -= e / s
        K[k1, k0] -= e / s
    return K


def nodal_det(nu, widths, eps):
    # det K has at most a simple pole per sector at sin(nu w)=0; remove it
    return np.linalg.det(nodal_K(nu, widths, eps)) * np.prod(np.sin(nu * np.asarray(widths)))


# ---------------------------------------------------------------- direct (D)
def direct_M(nu, widths, eps):
    N = len(widths)
    w = np.asarray(widths, float)
    M = np.zeros((2 * N, 2 * N))
    for k in range(N):            # interface at the START of sector k
        km = (k - 1) % N
        tk = -w[k] / 2            # local angle (from centre) of sector k's start
        tm = +w[km] / 2           # local angle of sector km's end
        r = 2 * k
        # continuity f_km(end) - f_k(start) = 0
        M[r, 2 * km] += np.cos(nu * tm); M[r, 2 * km + 1] += np.sin(nu * tm)
        M[r, 2 * k] -= np.cos(nu * tk); M[r, 2 * k + 1] -= np.sin(nu * tk)
        # flux (divided by nu): eps_km f'_km(end) - eps_k f'_k(start) = 0
        M[r + 1, 2 * km] += -eps[km] * np.sin(nu * tm); M[r + 1, 2 * km + 1] += eps[km] * np.cos(nu * tm)
        M[r + 1, 2 * k] -= -eps[k] * np.sin(nu * tk); M[r + 1, 2 * k + 1] -= eps[k] * np.cos(nu * tk)
    return M


def direct_det(nu, widths, eps):
    return np.linalg.det(direct_M(nu, widths, eps))


def smin_ratio(M):
    s = np.linalg.svd(M, compute_uv=False)
    return s[-1] / s[0]


def residual_check(nu, widths, eps):
    """Null vector of the nodal system -> reconstruct f in every sector and
    measure the jump in f and eps f' at every interface (relative)."""
    K = nodal_K(nu, widths, eps)
    _, s, vt = np.linalg.svd(K)
    u = vt[-1]
    N = len(widths)
    w = np.asarray(widths)
    jf, jflux, scale = [], [], np.max(np.abs(u))
    for k in range(N):
        km = (k - 1) % N
        # sector km: f(s) = [u_km sin(nu(w-s)) + u_k sin(nu s)]/sin(nu w)
        def f_and_df(j, s_loc):
            a, b = u[j], u[(j + 1) % N]
            S = np.sin(nu * w[j])
            f = (a * np.sin(nu * (w[j] - s_loc)) + b * np.sin(nu * s_loc)) / S
            df = nu * (-a * np.cos(nu * (w[j] - s_loc)) + b * np.cos(nu * s_loc)) / S
            return f, df
        f_end, df_end = f_and_df(km, w[km])
        f_st, df_st = f_and_df(k, 0.0)
        jf.append(abs(f_end - f_st) / scale)
        jflux.append(abs(eps[km] * df_end - eps[k] * df_st) / (scale * max(eps)))
    return s[-1] / s[0], max(jf), max(jflux), u / scale


def scan_roots(fun, lo=1e-4, hi=1 - 1e-9, n=40001):
    return roots_in(fun, lo, hi, n)


# ---------------------------------------------------------------- FE (F)
def fe_nu(widths, eps, n_per_rad, nev=6):
    """P1 FE, periodic, nodes on every interface. Returns sqrt of the smallest
    nonzero generalized eigenvalues."""
    xs, es = [], []
    for w, e in zip(widths, eps):
        m = max(2, int(math.ceil(w * n_per_rad)))
        h = w / m
        xs += [h] * m
        es += [e] * m
    ne = len(xs)
    K = np.zeros((ne, ne)); Mm = np.zeros((ne, ne))
    for i, (h, e) in enumerate(zip(xs, es)):
        a, b = i, (i + 1) % ne
        k = e / h; m = e * h / 6
        K[a, a] += k; K[b, b] += k; K[a, b] -= k; K[b, a] -= k
        Mm[a, a] += 2 * m; Mm[b, b] += 2 * m; Mm[a, b] += m; Mm[b, a] += m
    lam = eigh(K, Mm, eigvals_only=True, subset_by_index=[0, nev])
    lam = np.sort(lam)[1:]       # drop the constant
    return np.sqrt(np.clip(lam, 0, None)), max(xs)


def fe_extrapolated(widths, eps, target_count=2):
    levels = [40, 80, 160, 320, 640]
    vals = []
    for n in levels:
        nus, h = fe_nu(widths, eps, n)
        vals.append(nus[:target_count])
    vals = np.array(vals)
    # nu^2 has O(h^2) error; extrapolate nu^2 (Richardson, two passes)
    lam = vals ** 2
    r1 = (4 * lam[1:] - lam[:-1]) / 3
    r2 = (16 * r1[1:] - r1[:-1]) / 15
    return vals, np.sqrt(r1), np.sqrt(r2)


# ---------------------------------------------------------------- closed forms
def nu_two_media_90(e_in, e_out):
    """90 deg of e_in inside 270 deg of e_out:  cos(nu pi/2) = |e_in-e_out|/(2(e_in+e_out))."""
    return (2 / PI) * math.acos(abs(e_in - e_out) / (2 * (e_in + e_out)))


def nu_three_media(e1, e2, e3):
    """90 deg e1, 90 deg e2, 180 deg e3 (the 180 deg sector touches both).
    cos^2(nu pi/2) = e3 (e1-e2)^2 / (2 (e1+e2)(e1+e3)(e2+e3))."""
    c2 = e3 * (e1 - e2) ** 2 / (2 * (e1 + e2) * (e1 + e3) * (e2 + e3))
    return (2 / PI) * math.acos(math.sqrt(c2))


def nu_three_media_mp(e1, e2, e3, dps=30):
    mp.mp.dps = dps
    e1, e2, e3 = mp.mpf(e1), mp.mpf(e2), mp.mpf(e3)
    c2 = e3 * (e1 - e2) ** 2 / (2 * (e1 + e2) * (e1 + e3) * (e2 + e3))
    return 2 / mp.pi * mp.acos(mp.sqrt(c2))


def symbolic_three_media():
    """Re-derive the 90/90/180 determinant symbolically from the nodal 3x3."""
    import sympy as sp
    x, e1, e2, e3, tau = sp.symbols('x e1 e2 e3 tau', positive=True)
    w = [x, x, 2 * x]            # x = nu*pi/2 ; widths*nu = (pi/2, pi/2, pi)*nu
    eps = [e1, e2, e3]
    K = sp.zeros(3, 3)
    for j in range(3):
        s = sp.sin(w[j]); c = sp.cos(w[j]); e = eps[j]
        k0, k1 = j, (j + 1) % 3
        K[k0, k0] += e * c / s; K[k1, k1] += e * c / s
        K[k0, k1] -= e / s; K[k1, k0] -= e / s
    det = sp.simplify(sp.trigsimp(K.det() * sp.sin(x) ** 3 * sp.cos(x)))
    det_tau = sp.simplify(sp.expand_trig(det).rewrite(sp.cos))
    # express via tau = cos(2x) = cos(nu pi)
    q = sp.simplify(sp.expand(sp.expand_trig(det)).subs(sp.sin(x) ** 2, 1 - sp.cos(x) ** 2))
    q = sp.simplify(q.subs(sp.cos(x), sp.sqrt((1 + tau) / 2)))
    return sp.factor(sp.expand(q))


# ---------------------------------------------------------------- cases
CASES = {
    'A': ([90, 270], [1.0, 12.0]),
    'B': ([90, 270], [12.0, 1.0]),
    'C': ([90, 270], [5.8, 1.0]),
    'D': ([90, 270], [4.0, 1.0]),
    'E': ([90, 90, 180], [12.0, 1.0, 2.25]),
    'F': ([90, 90, 180], [5.8, 1.0, 2.13]),
    'G': ([90, 90, 180], [4.0, 1.0, 2.25]),
}


def all_routes(name, wdeg, eps, verbose=True):
    w = [d * D2R for d in wdeg]
    out = {}
    if len(w) == 2:
        ev, od = parity_roots(w[0], eps[0], eps[1])
        out['parity_even'] = ev; out['parity_odd'] = od
        out['closed'] = nu_two_media_90(eps[0], eps[1])
    else:
        out['closed'] = nu_three_media(*eps)
        out['closed_mp'] = nu_three_media_mp(*eps)
    out['nodal'] = scan_roots(lambda t: nodal_det(t, w, eps))
    out['direct'] = scan_roots(lambda t: direct_det(t, w, eps))
    vals, r1, r2 = fe_extrapolated(w, eps)
    out['fe_raw'] = vals[-1]; out['fe_r1'] = r1[-1]; out['fe_r2'] = r2[-1]
    out['fe_levels'] = vals[:, 0]
    if verbose:
        print(f"--- case {name}: widths {wdeg} eps {eps}")
        for k, v in out.items():
            if k == 'fe_levels':
                print(f"   {k:12s}", ' '.join(f"{t:.9f}" for t in v))
            elif isinstance(v, (list, np.ndarray)):
                print(f"   {k:12s}", ' '.join(f"{t:.12f}" for t in np.atleast_1d(v)))
            else:
                print(f"   {k:12s} {v}")
        nu0 = out['closed']
        sr, jf, jx, u = residual_check(nu0, w, eps)
        print(f"   null-vector check at closed form: smin/smax(K)={sr:.2e}  max|[f]|={jf:.2e}  max|[eps f']|={jx:.2e}")
        print(f"   smin/smax(direct M) at closed form = {smin_ratio(direct_M(nu0, w, eps)):.2e}")
    return out


if __name__ == '__main__':
    np.set_printoptions(precision=12)
    print("=== symbolic 3x3 nodal determinant for 90/90/180 (x = nu pi/2, tau = cos(nu pi)):")
    try:
        print("   ", symbolic_three_media())
    except Exception as exc:  # pragma: no cover
        print("   sympy failed:", exc)

    results = {}
    for name, (wdeg, eps) in CASES.items():
        results[name] = all_routes(name, wdeg, eps)

    print("\n=== summary (smallest nu in (0,1)) ===")
    for name, r in results.items():
        cands = [r['closed'], min(r['nodal']), min(r['direct']), r['fe_r2'][0]]
        print(f"   {name}: closed {r['closed']:.12f}  nodal {min(r['nodal']):.12f}  direct {min(r['direct']):.12f}"
              f"  FE(R2) {r['fe_r2'][0]:.10f}  spread {max(cands)-min(cands):.1e}  all roots in (0,1): "
              + ', '.join(f"{t:.6f}" for t in r['nodal']))
