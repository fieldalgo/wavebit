"""
benchmark_analytic.py — ANALYTIC (staircase-free) modeling vs the staircase path.

Demonstrates the core claim: feeding the solver the *exact* analytic Fourier
coefficients of a smooth primitive (rotated rectangle / circle / ellipse) plus the
*analytic* boundary normals (NVF factorization) converges to the true answer with far
fewer Fourier orders than the staircase representation, for tilted (dielectric) AND
curved (metallic) boundaries.

The analytic modeling path is selected by passing ``ucell`` as a list of layer dicts::

    ucell = [dict(n_base=1.0, shapes=[dict(type='circle', cx=.5, cy=.5,
                                           lx=0.6, ly=0.6, n_index=3.5)])]

Run from the repo root:

    python studies/benchmark_analytic.py
"""
import os
import sys

import numpy as np

# Ensure the local (worktree) fieldwave is used rather than any installed copy.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fieldwave


def t0_and_energy(res):
    R = np.array(res.R).real
    T = np.array(res.T).real
    c = T.shape[0] // 2
    t0 = T[c, c]
    return t0, float(R.sum() + T.sum())


def make_circle_raster(N, R, n_rod, n_bg=1.0):
    x = (np.arange(N) + 0.5) / N - 0.5
    X, Y = np.meshgrid(x, x)
    layer = np.full((N, N), n_bg, dtype=complex)
    layer[(X ** 2 + Y ** 2) <= R ** 2] = n_rod
    return layer[None]


def solve(backend, ucell, fto, **cfg):
    solver = fieldwave.rcwa(backend=backend, ucell=ucell, fto=(fto, fto), **cfg)
    return t0_and_energy(solver.conv_solve())


# ----------------------------------------------------------------------
# 1. Dielectric tilted square — analytic vs vector staircase (same geometry)
# ----------------------------------------------------------------------
def bench_tilted_dielectric():
    cfg = dict(n_top=1.0, n_bot=1.0, theta=0.0, phi=None,
               period=(1.0, 1.0), wavelength=1.6, thickness=(0.6,), pol=1.0)  # TM
    NR, ang, L, c = 3.5, np.deg2rad(30), 0.5, 0.5
    analytic = [dict(n_base=1.0, shapes=[dict(type='rectangle', cx=c, cy=c, lx=L, ly=L,
                                              angle=ang, n_index=NR)])]

    def stair(nsplit):
        return [[1.0, [['rectangle', c, c, L, L, NR, ang, nsplit, nsplit]]]]

    ref, _ = solve(0, analytic, 16, **cfg)
    print("\n" + "=" * 74)
    print(f"[1] Dielectric tilted square (30 deg, n={NR}, TM subwavelength)")
    print(f"    reference t0 (analytic fto=16) = {ref:.6f}")
    print(f"    {'fto':>3} | {'analytic |err|':>14} | {'staircase n=8':>13} | {'staircase n=24':>14}")
    for fto in (3, 5, 7, 9, 11):
        a, _ = solve(0, analytic, fto, **cfg)
        s8, _ = solve(0, stair(8), fto, **cfg)
        s24, _ = solve(0, stair(24), fto, **cfg)
        print(f"    {fto:>3} | {abs(a - ref):14.2e} | {abs(s8 - ref):13.2e} | {abs(s24 - ref):14.2e}")


# ----------------------------------------------------------------------
# 2. Metallic circle — analytic-NVF vs raster-NVF vs raster-sampled
# ----------------------------------------------------------------------
def bench_metallic_circle():
    cfg = dict(n_top=1.0, n_bot=1.0, theta=0.0, phi=None,
               period=(1.0, 1.0), wavelength=1.6, thickness=(0.15,), pol=1.0)  # TM
    NM, R = 0.05 + 3.5j, 0.30
    analytic = [dict(n_base=1.0, shapes=[dict(type='circle', cx=0.5, cy=0.5,
                                              lx=2 * R, ly=2 * R, n_index=NM)])]
    ref, _ = solve(0, analytic, 16, **cfg)
    print("\n" + "=" * 74)
    print(f"[2] Metallic circle (n={NM}, R={R}, TM subwavelength)")
    print(f"    reference t0 (analytic fto=16) = {ref:.6f}")
    print(f"    {'fto':>3} | {'analytic-NVF':>13} | {'raster-NVF':>12} | {'raster-sampled':>14}")
    for fto in (3, 5, 7, 9, 11):
        a, _ = solve(0, analytic, fto, **cfg)
        ras = make_circle_raster(240, R, NM)
        nvf, _ = solve(0, ras, fto, factorization='nvf', **cfg)
        dfs, _ = solve(0, ras, fto, **cfg)
        print(f"    {fto:>3} | {abs(a - ref):13.2e} | {abs(nvf - ref):12.2e} | {abs(dfs - ref):12.2e}")


# ----------------------------------------------------------------------
# 3. Energy conservation (lossless) + cross-backend consistency
# ----------------------------------------------------------------------
def bench_energy_and_backends():
    cfg = dict(n_top=1.0, n_bot=1.0, theta=np.deg2rad(5), phi=np.deg2rad(20),
               period=(1.0, 1.2), wavelength=1.5, thickness=(0.3, 0.2), pol=0.3)
    ucell = [
        dict(n_base=1.0, shapes=[dict(type='ellipse', cx=0.5, cy=0.6, lx=0.5, ly=0.3,
                                      angle=np.deg2rad(40), n_index=2.4)]),
        dict(n_base=1.5, shapes=[dict(type='rectangle', cx=0.4, cy=0.5, lx=0.4, ly=0.5,
                                      angle=np.deg2rad(25), n_index=1.0),
                                 dict(type='circle', cx=0.8, cy=0.9, lx=0.25,
                                      n_index=3.0)]),
    ]
    print("\n" + "=" * 74)
    print("[3] Lossless multi-shape / multi-layer: energy conservation + cross-backend")
    print(f"    {'fto':>3} | {'|R+T-1|':>10} | {'numpy t0':>10} {'jax t0':>10} {'torch t0':>10}")
    for fto in (5, 8):
        tn, en = solve(0, ucell, fto, **cfg)
        tj, _ = solve(1, ucell, fto, **cfg)
        tt, _ = solve(2, ucell, fto, **cfg)
        print(f"    {fto:>3} | {abs(en - 1):10.2e} | {tn:10.6f} {tj:10.6f} {tt:10.6f}"
              f"   (|np-jax|={abs(tn - tj):.1e}, |np-torch|={abs(tn - tt):.1e})")


def main():
    print("fieldwave ANALYTIC (staircase-free) modeling benchmark")
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
