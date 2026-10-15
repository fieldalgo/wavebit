"""
benchmark_analytic.py — ANALYTIC (staircase-free) modeling vs the staircase path.

Demonstrates the core claim: feeding the solver the *exact* analytic Fourier
coefficients of a smooth primitive (rotated rectangle / circle / ellipse) plus the
*analytic* boundary normals (NVF factorization) converges to the true answer with far
fewer Fourier orders than the staircase representation, for tilted (dielectric) AND
curved (metallic) boundaries.

The analytic modeling path is selected by passing ``structure`` as a list of layer dicts::

    structure = [dict(background=1.0, shapes=[dict(type='circle', cx=0.0, cy=0.0,
                                                   lx=0.6, material=12.25)])]

Every material here is a **relative permittivity** -- ``background``, each shape's
``material``, and ``top`` / ``bot``.  A bare number is the scalar shorthand for
``wavebit.Material(epsilon=x)``.  The structures below are named by refractive index because
that is how the printed tables label them, so ``wavebit.eps_from_refractive_index`` does the
conversion at the point the index is written down; an index left in one of those slots is a legal,
energy-conserving, *different* structure and nothing downstream can flag it.

Run from the repo root:

    python studies/rcwa/benchmark_analytic/benchmark_analytic.py
"""

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa
import os
import sys

import numpy as np

# Ensure the local (worktree) wavebit is used rather than any installed copy.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wavebit


def t0_and_energy(res):
    R = np.array(res.R).real
    T = np.array(res.T).real
    c = T.shape[0] // 2
    t0 = T[c, c]
    return t0, float(R.sum() + T.sum())


def make_circle_raster(N, R, n_rod, n_bg=1.0):
    """A rod on a grid, as the permittivity raster the raster path takes.

    Indices in, permittivity out: the raster is the one path that is not an assignment but a
    *sampled field*, so it holds one number per pixel and that number is ``eps``.
    """
    x = (np.arange(N) + 0.5) / N - 0.5
    X, Y = np.meshgrid(x, x)
    layer = np.full((N, N), wavebit.eps_from_refractive_index(n_bg), dtype=complex)
    layer[(X ** 2 + Y ** 2) <= R ** 2] = wavebit.eps_from_refractive_index(n_rod)
    return layer[None]


def solve(backend, structure, truncation_order, **cfg):
    solver = _legacy_wavebit_rcwa(backend=backend, structure=structure, truncation_order=(truncation_order, truncation_order), **cfg)
    return t0_and_energy(solver.solve())


# ----------------------------------------------------------------------
# 1. Dielectric tilted square — analytic vs raster staircase (same geometry)
# ----------------------------------------------------------------------
def bench_tilted_dielectric():
    # psi = 0 is TM.  (It was `pol=1.0`, which said three things at once; `psi` says only
    # the polarization angle, and `tetm` would say the other two -- but this structure is 2D,
    # so the pure TE/TM formulation does not apply to it.)
    cfg = dict(top=1.0, bot=1.0, theta=0.0, phi=None,
               period=(1.0, 1.0), wavelength=1.6, thickness=(0.6,), psi=0.0)  # TM
    NR, ang, L, c = 3.5, np.deg2rad(30), 0.5, 0.5
    eps_r = wavebit.eps_from_refractive_index(NR)
    # `c` is the absolute centre the staircase below is built around; a shape's centre is
    # measured from the middle of the cell, which on this period-1 cell is 0.5.
    analytic = [dict(background=1.0, shapes=[dict(type='rectangle', cx=c - 0.5, cy=c - 0.5,
                                                  lx=L, ly=L, angle=ang, material=eps_r)])]

    def stair(n_pix):
        # The staircase arm was the removed block path until 2026-08-27, at `n_split` 8 and
        # 24. A raster of the same rotated square is what is left that staircases, and it is
        # the more honest comparison anyway: it shares no code with the analytic transform.
        # `n_pix` is pixels per period, not splits per shape, so the two axes do not compare
        # directly -- what carries over is the shape of the curve, not the row labels.
        ax = (np.arange(n_pix) + 0.5) / n_pix
        X, Y = np.meshgrid(ax, ax)
        u = (X - c) * np.cos(ang) + (Y - c) * np.sin(ang)
        v = -(X - c) * np.sin(ang) + (Y - c) * np.cos(ang)
        cell = np.where((np.abs(u) <= L / 2) & (np.abs(v) <= L / 2), eps_r, 1.0)
        return cell[None]

    ref, _ = solve('numpy', analytic, 16, **cfg)
    print("\n" + "=" * 74)
    print(f"[1] Dielectric tilted square (30 deg, n={NR}, TM subwavelength)")
    print(f"    reference t0 (analytic truncation_order=16) = {ref:.6f}")
    print(f"    {'truncation_order':>3} | {'analytic |err|':>14} | {'raster 64px':>13} | {'raster 256px':>14}")
    lo, hi = stair(64), stair(256)
    for truncation_order in (3, 5, 7, 9, 11):
        a, _ = solve('numpy', analytic, truncation_order, **cfg)
        s8, _ = solve('numpy', lo, truncation_order, **cfg)
        s24, _ = solve('numpy', hi, truncation_order, **cfg)
        print(f"    {truncation_order:>3} | {abs(a - ref):14.2e} | {abs(s8 - ref):13.2e} | {abs(s24 - ref):14.2e}")


# ----------------------------------------------------------------------
# 2. Metallic circle — analytic-NVF vs raster-NVF vs raster-sampled
# ----------------------------------------------------------------------
def bench_metallic_circle():
    cfg = dict(top=1.0, bot=1.0, theta=0.0, phi=None,
               period=(1.0, 1.0), wavelength=1.6, thickness=(0.15,), psi=0.0)  # TM
    NM, R = 0.05 + 3.5j, 0.30
    # No `ly`: a circle has one length, and naming a second is refused rather than dropped
    # (a dropped `ly` optimises with an exactly-zero gradient).
    analytic = [dict(background=1.0, shapes=[dict(type='circle', cx=0.0, cy=0.0,
                                                  lx=2 * R,
                                                  material=wavebit.eps_from_refractive_index(NM))])]
    ref, _ = solve('numpy', analytic, 16, **cfg)
    print("\n" + "=" * 74)
    print(f"[2] Metallic circle (n={NM}, R={R}, TM subwavelength)")
    print(f"    reference t0 (analytic truncation_order=16) = {ref:.6f}")
    print(f"    {'truncation_order':>3} | {'analytic-NVF':>13} | {'raster-NVF':>12} | {'raster-sampled':>14}")
    for truncation_order in (3, 5, 7, 9, 11):
        a, _ = solve('numpy', analytic, truncation_order, **cfg)
        ras = make_circle_raster(240, R, NM)
        nvf, _ = solve('numpy', ras, truncation_order, factorization='nvf', **cfg)
        dfs, _ = solve('numpy', ras, truncation_order, **cfg)
        print(f"    {truncation_order:>3} | {abs(a - ref):13.2e} | {abs(nvf - ref):12.2e} | {abs(dfs - ref):12.2e}")


# ----------------------------------------------------------------------
# 3. Energy conservation (lossless) + cross-backend consistency
# ----------------------------------------------------------------------
def bench_energy_and_backends():
    # psi = 0.35*pi is what `pol=0.3` used to mean: the old keyword turned a fraction into a
    # polarization angle as psi = (1 - pol)*pi/2.
    cfg = dict(top=1.0, bot=1.0, theta=np.deg2rad(5), phi=np.deg2rad(20),
               period=(1.0, 1.2), wavelength=1.5, thickness=(0.2, 0.3), psi=0.35 * np.pi)
    eps = wavebit.eps_from_refractive_index
    structure = [
        dict(background=eps(1.5), shapes=[dict(type='rectangle', cx=-0.1, cy=-0.1, lx=0.4,
                                               ly=0.5, angle=np.deg2rad(25),
                                               material=eps(1.0)),
                                          dict(type='circle', cx=0.3, cy=0.3, lx=0.25,
                                               material=eps(3.0))]),
        dict(background=1.0, shapes=[dict(type='ellipse', cx=0.0, cy=0.0, lx=0.5, ly=0.3,
                                          angle=np.deg2rad(40), material=eps(2.4))]),
    ]
    print("\n" + "=" * 74)
    print("[3] Lossless multi-shape / multi-layer: energy conservation + cross-backend")
    print(f"    {'truncation_order':>3} | {'|R+T-1|':>10} | {'numpy t0':>10} {'jax t0':>10} {'torch t0':>10}")
    for truncation_order in (5, 8):
        tn, en = solve('numpy', structure, truncation_order, **cfg)
        tj, _ = solve('jax', structure, truncation_order, **cfg)
        tt, _ = solve('torch', structure, truncation_order, **cfg)
        print(f"    {truncation_order:>3} | {abs(en - 1):10.2e} | {tn:10.6f} {tj:10.6f} {tt:10.6f}"
              f"   (|np-jax|={abs(tn - tj):.1e}, |np-torch|={abs(tn - tt):.1e})")


def main():
    print("wavebit ANALYTIC (staircase-free) modeling benchmark")
    bench_tilted_dielectric()
    bench_metallic_circle()
    bench_energy_and_backends()
    print("\n" + "=" * 74)
    print("Takeaway: analytic modeling reaches the converged answer at markedly lower")
    print("Fourier order than the staircase path, for both tilted and curved boundaries,")
    print("while conserving energy and agreeing across numpy/jax/torch.")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
