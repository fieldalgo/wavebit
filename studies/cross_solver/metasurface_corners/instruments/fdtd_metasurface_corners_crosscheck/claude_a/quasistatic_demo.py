"""Does an integrated quantity converge as h^(2 nu)?  A quasi-static test (no
FDTD/RCWA): periodic unit cell, square inclusion of side 1/2, contrast 12,
cell-centred 5-point finite volumes, applied mean field e_x.

  phi = -x + u, u periodic;  div(eps (grad u - e_x)) = 0.

Quantities
  eps_eff  = <D_x>                       variational (= discrete energy)
  J_box    = <u>_box2 - <u>_box1         linear, non-variational
  E_box    = <E_x> over box1             linear, non-variational (field average)

Runs
  (1) grid-aligned square (corners on grid nodes; eps per cell is exact):
      pure discretisation error of the singular field.
  (2) square shifted by an irrational offset (eps_eff is translation invariant,
      so the reference from (1) still applies), two eps rules:
        smooth : face eps = arithmetic-over-face of the harmonic-along-normal
                 line average (exact for laminates; Kottke/Farjadpour-type)
        stair  : eps per cell from the cell centre, harmonic face mean.
Expected (nu = 0.7219 at all four corners): (1) and (2-smooth) ~ h^1.444,
(2-stair) ~ h^1.
"""
import sys, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

EIN, EOUT = 12.0, 1.0


def frac_in(a, b, lo, hi):
    """fraction of [a,b] inside the periodic interval [lo,hi) (mod 1); arrays a,b."""
    out = np.zeros_like(a)
    for k in (-1, 0, 1):
        l, r = lo + k, hi + k
        out += np.clip(np.minimum(b, r) - np.maximum(a, l), 0, None)
    return out / (b - a)


def face_eps(N, x0, y0, side, rule):
    h = 1.0 / N
    c = (np.arange(N) + 0.5) * h
    if rule == 'cell':
        inx = frac_in(c - 1e-12, c + 1e-12, x0, x0 + side) > 0.5
        iny = frac_in(c - 1e-12, c + 1e-12, y0, y0 + side) > 0.5
        epsc = np.where(inx[:, None] & iny[None, :], EIN, EOUT)    # [i(x), j(y)]
        er = np.roll(epsc, -1, 0)
        ex = 2 * epsc * er / (epsc + er)                             # face i+1/2
        eu = np.roll(epsc, -1, 1)
        ey = 2 * epsc * eu / (epsc + eu)
        return ex, ey
    # smoothed: x-face between centres c_i and c_i + h, tube y in [c_j-h/2, c_j+h/2]
    fx_line = frac_in(c, c + h, x0, x0 + side)          # along-normal fraction (x)
    fy_face = frac_in(c - h / 2, c + h / 2, y0, y0 + side)  # across-face fraction (y)
    harm = 1.0 / (fx_line / EIN + (1 - fx_line) / EOUT)
    ex = fy_face[None, :] * harm[:, None] + (1 - fy_face[None, :]) * EOUT
    fy_line = frac_in(c, c + h, y0, y0 + side)
    fx_face = frac_in(c - h / 2, c + h / 2, x0, x0 + side)
    harm = 1.0 / (fy_line / EIN + (1 - fy_line) / EOUT)
    ey = fx_face[:, None] * harm[None, :] + (1 - fx_face[:, None]) * EOUT
    return ex, ey


def solve(N, x0=0.25, y0=0.25, side=0.5, rule='cell'):
    h = 1.0 / N
    ex, ey = face_eps(N, x0, y0, side, rule)
    idx = np.arange(N * N).reshape(N, N)
    R = np.roll(idx, -1, 0); U = np.roll(idx, -1, 1)
    rows = np.concatenate([idx.ravel(), R.ravel(), idx.ravel(), U.ravel()])
    cols = np.concatenate([R.ravel(), idx.ravel(), U.ravel(), idx.ravel()])
    w = np.concatenate([ex.ravel(), ex.ravel(), ey.ravel(), ey.ravel()])
    W = sp.coo_matrix((w, (rows, cols)), shape=(N * N, N * N)).tocsr()
    A = sp.diags(np.asarray(W.sum(1)).ravel()) - W
    # sum_out = (eps_R - eps_L) h + (A u)_c = 0
    b = -(ex - np.roll(ex, 1, 0)).ravel() * h
    A = A.tocsc()[1:, 1:]                     # pin u_0 = 0
    u = np.zeros(N * N)
    u[1:] = spla.splu(A).solve(b[1:])
    u = u.reshape(N, N)
    Ex_face = 1 - (np.roll(u, -1, 0) - u) / h
    eps_eff = np.mean(ex * Ex_face)           # <D_x> over all x-faces
    out = {'eps_eff': eps_eff}
    if rule == 'cell' and abs(x0 - 0.25) < 1e-15:
        m = N // 8
        box1 = (slice(0, m), slice(0, m))                       # [0,1/8]^2
        box2 = (slice(N // 2, N // 2 + m), slice(N // 2, N // 2 + m))  # [1/2,5/8]^2
        out['J_box'] = u[box2].mean() - u[box1].mean()
        # <E_x> over box1: faces strictly inside + half weight of the bounding faces
        Ex_in = Ex_face[0:m - 1, 0:m].sum() + 0.5 * (Ex_face[m - 1, 0:m].sum() + Ex_face[-1, 0:m].sum())
        out['E_box'] = Ex_in / (m * m)
    return out


def three_grid(vals, Ns):
    for k in range(2, len(vals)):
        a, b, c = vals[k - 2:k + 1]
        q = (a - b) / (b - c)
        p = np.log2(abs(q)) if q > 0 else float('nan')
        yield Ns[k], p, c + (c - b) / (2 ** p - 1) if np.isfinite(p) else float('nan')


if __name__ == '__main__':
    Nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    Ns = [n for n in [32, 64, 128, 256, 512, 1024, 2048] if n <= Nmax]
    print(f"(1) grid-aligned square, contrast {EIN/EOUT:g}")
    res = []
    for N in Ns:
        t = time.time()
        r = solve(N)
        res.append(r)
        print(f"   N={N:5d}  eps_eff={r['eps_eff']:.12f}  J_box={r['J_box']:.12f}  E_box={r['E_box']:.12f}  {time.time()-t:5.1f}s", flush=True)
    refs = {}
    for key in ['eps_eff', 'J_box', 'E_box']:
        v = [r[key] for r in res]
        print(f"   {key}: three-grid observed orders")
        for N, p, ext in three_grid(v, Ns):
            print(f"      up to N={N:5d}: order={p:.4f}   extrapolated={ext:.12f}")
        # reference: Richardson with the theoretical order 2 nu
        p = 2 * 0.7219000057
        refs[key] = v[-1] + (v[-1] - v[-2]) / (2 ** p - 1)
    print(f"   reference eps_eff (Richardson, order 1.4438): {refs['eps_eff']:.12f}")

    print("\n(2) square shifted by (0.1234567, 0.0765432): eps_eff is unchanged in the continuum")
    x0, y0 = 0.25 + 0.1234567, 0.25 + 0.0765432
    NN = np.unique(np.round(np.geomspace(40, min(Nmax, 700), 14)).astype(int))
    for rule in ['smooth', 'cell']:
        errs = []
        for N in NN:
            e = solve(N, x0, y0, rule=rule)['eps_eff'] - refs['eps_eff']
            errs.append(e)
        errs = np.array(errs)
        sl = np.polyfit(np.log(1.0 / NN), np.log(abs(errs)), 1)[0]
        print(f"   rule={rule:6s}: least-squares slope of log|err| vs log h = {sl:.3f}")
        for N, e in zip(NN, errs):
            print(f"      N={N:4d}  err={e:+.3e}  |err|*N^1.444={abs(e)*N**1.4438:.3f}  |err|*N={abs(e)*N:.3f}")

    print("\n(3) RMS over 8 random sub-pixel offsets, odd N (so the stair rule cannot")
    print("    land on the exact side length by parity luck)")
    rng = np.random.default_rng(1)
    offs = rng.random((8, 2))
    NN = [n for n in [33, 65, 129, 257, 513] if n <= Nmax + 1]
    for rule in ['smooth', 'cell']:
        rms = []
        for N in NN:
            e = [solve(N, 0.25 + ox, 0.25 + oy, rule=rule)['eps_eff'] - refs['eps_eff'] for ox, oy in offs]
            rms.append(np.sqrt(np.mean(np.square(e))))
        rms = np.array(rms)
        sl = np.log(rms[:-1] / rms[1:]) / np.log(np.array(NN[1:]) / np.array(NN[:-1]))
        print(f"   rule={rule:6s}: RMS err {np.array2string(rms, precision=3)}")
        print(f"                 local slopes {np.array2string(sl, precision=3)}")
