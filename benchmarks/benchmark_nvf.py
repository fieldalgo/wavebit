"""
benchmark_nvf.py — 2D NVF Fourier-factorization convergence/accuracy benchmark

Purpose
-------
1. Quantify the accuracy of the NVF path (fourier_type=2) against the standard DFS
   path (fourier_type=0) via the "energy-conservation error |R+T-1|".
   (For a lossless structure R+T=1 must hold exactly, so as the order increases
    |R+T-1| -> 0 means convergent/accurate; if it does not shrink, the implementation
    is wrong.)
2. Provide a harness to immediately verify NVF tensor construction (product order /
   epz convention / normal-vector source) by varying it and re-running the same benchmark.
   => definition of "fixed" = error ->0 like DFS, and t0 agreeing with high-order DFS.

Status (2026-08, resolved)
--------------------------
The NVF path now converges and conserves energy. For a lossless circular grating
|R+T-1| shrinks to machine precision from low order, and at high order it is much more
accurate than DFS. Fix summary: (1) epz = inv(⟦ε⟧), (2) off-diagonal slot sign,
(3) build the anisotropic tensor with Hermitian symmetrization ½(ΔC+CΔ). See
NVF_FINDINGS.md for the derivation/verification.

Run (from repo root)
--------------------
    python benchmarks/benchmark_nvf.py
"""

import numpy as np

import fieldwave
from fieldwave.on_numpy.emsolver import convolution_matrix as CM
import fieldwave.on_numpy.emsolver.nvf as NVF
from fieldwave.on_numpy.emsolver.fourier_analysis import dfs2d
from fieldwave.on_numpy.emsolver.primitives import meeinv


# ----------------------------------------------------------------------
# Test structure: lossless 2D circular dielectric rod (n=3.5), normal incidence, TE
# ----------------------------------------------------------------------
CFG = dict(n_top=1.0, n_bot=1.0, theta=0.0, phi=None,
           period=(1.0, 1.0), wavelength=0.5, thickness=(0.3,), pol=0.0)
N_GRID = 81
N_ROD = 3.5
R_FRAC = 0.30


def make_circle(N=N_GRID, n_bg=1.0, n_rod=N_ROD, rf=R_FRAC):
    x = (np.arange(N) + 0.5) / N - 0.5
    X, Y = np.meshgrid(x, x)
    layer = np.full((N, N), n_bg, dtype=float)
    layer[(X ** 2 + Y ** 2) <= rf ** 2] = n_rod
    return layer[None, :, :]  # (1, Ny, Nx)


def analytic_circle_nv(shape):
    """Exact unit normal of the circle boundary (radial). Used to remove NVF quality as a variable."""
    Ny, Nx = shape
    x = (np.arange(Nx) + 0.5) / Nx - 0.5
    y = (np.arange(Ny) + 0.5) / Ny - 0.5
    X, Y = np.meshgrid(x, y)
    nrm = np.hypot(X, Y)
    nrm[nrm == 0] = 1.0
    return X / nrm, Y / nrm


def energy_and_t0(res):
    de_ri = np.array(res.de_ri).real
    de_ti = np.array(res.de_ti).real
    c = de_ti.shape[0] // 2
    t0 = de_ti[c, c] if de_ti.ndim == 2 else de_ti[c]
    return abs(de_ri.sum() + de_ti.sum() - 1.0), t0


def solve(ucell, fourier_type, fto):
    solver = fieldwave.rcwa(backend=0, ucell=ucell, fto=(fto, fto),
                          fourier_type=fourier_type, **CFG)
    return energy_and_t0(solver.conv_solve())


# ----------------------------------------------------------------------
# NVF tensor variation harness
#   order : 'right' = current implementation (C @ Delta),  'left' = Delta @ C
#   epz   : 'raw'    = ⟦1/eps⟧            (current implementation)
#           'invEps' = inv(⟦eps⟧)         (same as DFS path)
#           'invInv' = inv(⟦1/eps⟧)
#   nv    : 'sdf' | 'analytic'
# Monkeypatch to_conv_mat_nvf_2d with this builder to verify via the same benchmark.
# ----------------------------------------------------------------------
def make_nvf_builder(order='right', epz='raw', nv='sdf'):
    def f(eps_matrix, fto_x, fto_y, nx=None, ny=None,
          type_complex=np.complex128, use_pinv=False):
        em = eps_matrix.astype(type_complex)
        if nv == 'analytic':
            a, b = analytic_circle_nv(em.shape)
        else:
            a, b = NVF.get_normal_vector_field_raster(em)
        Cxx = dfs2d((a ** 2).astype(type_complex), 1, 1, fto_x, fto_y, type_complex=type_complex)
        Cxy = dfs2d((a * b).astype(type_complex), 1, 1, fto_x, fto_y, type_complex=type_complex)
        Cyy = dfs2d((b ** 2).astype(type_complex), 1, 1, fto_x, fto_y, type_complex=type_complex)
        ep_dir = dfs2d(em, 1, 1, fto_x, fto_y, type_complex=type_complex)
        ep_inv = dfs2d(1.0 / em, 1, 1, fto_x, fto_y, type_complex=type_complex)
        ep_inv_i = meeinv(ep_inv, use_pinv=use_pinv)
        Delta = ep_dir - ep_inv_i
        if order == 'right':
            Exx = ep_dir - Cxx @ Delta; Exy = -Cxy @ Delta
            Eyx = -Cxy @ Delta;          Eyy = ep_dir - Cyy @ Delta
        else:
            Exx = ep_dir - Delta @ Cxx;  Exy = -Delta @ Cxy
            Eyx = -Delta @ Cxy;          Eyy = ep_dir - Delta @ Cyy
        Epz = {'raw': ep_inv,
               'invEps': meeinv(ep_dir, use_pinv=use_pinv),
               'invInv': ep_inv_i}[epz]
        return Exx, Eyy, Exy, Eyx, Epz
    return f


def solve_with_patch(ucell, fto, builder):
    CM.to_conv_mat_nvf_2d = builder          # replace the name convolution_matrix references
    try:
        return solve(ucell, 2, fto)
    finally:
        CM.to_conv_mat_nvf_2d = NVF.to_conv_mat_nvf_2d  # restore


# ----------------------------------------------------------------------
def main(ftos=(3, 5, 7, 9, 11)):
    ucell = make_circle()

    print(f"# 2D circular rod  n={N_ROD}  lambda={CFG['wavelength']}  "
          f"period={CFG['period']}  grid={N_GRID}  (lossless -> |R+T-1| should ->0)\n")

    header = f"{'method':30} | " + " ".join(f"fto{f:>2}" for f in ftos)
    print("[energy error |R+T-1|]")
    print(header)

    rows = [("DFS (standard)", lambda f: solve(ucell, 0, f)),
            ("NVF current (fourier_type=2)", lambda f: solve(ucell, 2, f))]
    # To test a proposed variant, add e.g.:
    # rows.append(("NVF left/invEps/analytic",
    #              lambda f: solve_with_patch(ucell, f,
    #                          make_nvf_builder('left','invEps','analytic'))))

    for name, fn in rows:
        cells = []
        for f in ftos:
            try:
                e, _ = fn(f); cells.append(f"{e:7.1e}")
            except Exception:
                cells.append(" ERR   ")
        print(f"{name:30} | " + " ".join(cells))

    print("\n[zeroth-order transmittance t0]")
    print(header)
    for name, fn in rows:
        cells = []
        for f in ftos:
            try:
                _, t = fn(f); cells.append(f"{t:7.4f}")
            except Exception:
                cells.append(" ERR   ")
        print(f"{name:30} | " + " ".join(cells))

    # Plot (if matplotlib is available)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        eD = [solve(ucell, 0, f)[0] for f in ftos]
        eN = [solve(ucell, 2, f)[0] for f in ftos]
        plt.figure(figsize=(6, 4))
        plt.semilogy(ftos, eD, 'o-', label='DFS')
        plt.semilogy(ftos, eN, 's--', label='NVF (current)')
        plt.xlabel('Fourier order'); plt.ylabel('|R+T-1|')
        plt.title('NVF vs DFS energy-conservation convergence')
        plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
        # dev/ is gitignored — save here so the artifact is not tracked by git.
        import os
        out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dev")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, "nvf_convergence.png")
        plt.savefig(out, dpi=130)
        print(f"\nsaved: {out}")
    except Exception as ex:
        print(f"\n(plot skipped: {ex})")


if __name__ == "__main__":
    main()
