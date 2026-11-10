"""Measurements for docs/theory/rcwa/scalar-nvf-short-grid-study.md.

    python measure.py a|b|c|e|const      # writes results/<section>.json

Arms: ``current`` is today's builder (wrapped DFT bins, short grids refused); ``prototype`` is
``factorization._NVF_SHORT_GRID_INTERPOLANT = True`` (interpolant coefficients, short grids
read); ``wrapped`` is today's builder with the refusal lifted, to show what the floor guards.
"""
import contextlib
import json
import pathlib
import sys
import time
import tracemalloc

import numpy as np

import cases
from cases import CASES, arm, solve, to_np
from wavebit import Material, rcwa
from wavebit.rcwa import _lifecycle
from wavebit.rcwa._engine import boundary, factorization as fz, parametric

OUT = pathlib.Path(__file__).with_name('results')
OUT.mkdir(exist_ok=True)


@contextlib.contextmanager
def wrapped():
    """Today's wrapped bins with every refusal of a short grid lifted."""
    saved = (parametric.check_normal_grid, boundary.check_normal_grid, _lifecycle._refuse_aliasing_grid)
    parametric.check_normal_grid = boundary.check_normal_grid = lambda *a: None
    _lifecycle._refuse_aliasing_grid = lambda *a: None
    parametric._PROJECTOR_CACHE.clear()
    try:
        with arm(False):
            yield
    finally:
        parametric.check_normal_grid, boundary.check_normal_grid, _lifecycle._refuse_aliasing_grid = saved
        parametric._PROJECTOR_CACHE.clear()


def arm_ctx(name):
    return {'current': lambda: arm(False), 'prototype': lambda: arm(True), 'wrapped': wrapped}[name]()


def maxdiff(a, b):
    return float(max(np.max(np.abs(a[k] - b[k])) for k in a))


# ------------------------------------------------------------------ (a) bit-identity above the floor
def section_a():
    import ast
    part = OUT / 'a.part1.log'
    rows = [ast.literal_eval(l) for l in part.read_text().splitlines() if l.startswith('{')] if part.exists() else []
    done = {(tuple(r['orders']), r['backend'], r['case'], r['grid']) for r in rows}
    for orders in ((3, 3), (4, 2)):
        M = max(orders)
        grids = [4*M + 1, 4*M + 2, 64, 128, None]
        names = list(CASES) if orders == (3, 3) else ['circle', 'rect-tilted', 'overlap', 'oblique']
        for backend in ('numpy', 'jax', 'torch'):
            for case in names:
                for g in grids:
                    if (orders, backend, case, g) in done:
                        continue
                    if case == 'overlap' and g is None and (backend != 'numpy' or orders != (3, 3)):
                        g = 256   # the boundary route at 1024 takes over an hour per solve here
                        if (orders, backend, case, g) in done:
                            continue
                    with arm(False):
                        cur = solve(case, g, orders, backend, field=True)
                    with arm(True):
                        pro = solve(case, g, orders, backend, field=True)
                    same = all(np.array_equal(cur[k], pro[k]) for k in cur)
                    rows.append(dict(orders=orders, backend=backend, case=case, grid=g, identical=same,
                                     max_diff=maxdiff(cur, pro)))
                    print(rows[-1], flush=True)
    return rows


# ------------------------------------------------------------------ (b) below the floor
def section_b():
    rows = []
    for orders in ((3, 3), (5, 5)):
        M = max(orders)
        ladder = sorted({1, 2, 3, 4, 5, 7, 8, 9, 4*M - 1, 4*M, 4*M + 1, 16, 25, 32, 64, 128, 256})
        for case in CASES:
            lossless = CASES[case][2]
            with arm(False):
                ref = solve(case, None, orders)
            for which in ('prototype', 'wrapped'):
                for g in ladder:
                    if case == 'overlap' and g > 64:
                        continue
                    with arm_ctx(which):
                        try:
                            r = solve(case, g, orders)
                        except Exception as e:  # noqa: BLE001
                            rows.append(dict(orders=orders, case=case, arm=which, grid=g, error=repr(e)[:200]))
                            continue
                    rows.append(dict(orders=orders, case=case, arm=which, grid=g,
                                     dR=float(np.max(np.abs(r['R'] - ref['R']))),
                                     dT=float(np.max(np.abs(r['T'] - ref['T']))),
                                     energy=float(abs(np.sum(r['R']) + np.sum(r['T']) - 1)) if lossless else None))
                    print(rows[-1], flush=True)
        # The tensor builder on a comparable layer, for the convergence shape it has today.
        tensor = cases._problem((cases.R(rcwa.Circle(radius=.3), np.diag([4., 3., 2.5])),))
        ref = rcwa.run(tensor, numerics=rcwa.Numerics(orders=orders))
        for g in ladder:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                r = rcwa.run(tensor, numerics=rcwa.Numerics(orders=orders, shape_nvf_grid_size=g))
            rows.append(dict(orders=orders, case='tensor-circle', arm='tensor', grid=g,
                             dR=float(np.max(np.abs(r.R - ref.R))), dT=float(np.max(np.abs(r.T - ref.T))),
                             energy=float(abs(np.sum(r.R) + np.sum(r.T) - 1))))
            print(rows[-1], flush=True)
    return rows


# ------------------------------------------------------------------ constant field -> identity
def section_const():
    rows = []
    M = 3
    for g in (1, 2, 3, 4, 4*M - 1, 4*M, 4*M + 1, 64):
        for which in ('prototype', 'wrapped'):
            with arm_ctx(which):
                c = to_np(parametric._sampled_coeffs(np, np.ones((g, g), dtype=complex), M, M))
            mid = 2*M
            off = c.copy(); off[mid, mid] = 0
            # The projector C_xx + C_yy for a real unit field must also be exactly the identity.
            th = np.random.default_rng(g).uniform(0, 2*np.pi, (g, g))
            with arm_ctx(which):
                sx = to_np(parametric._sampled_coeffs(np, (np.cos(th)**2).astype(complex), M, M))
                sy = to_np(parametric._sampled_coeffs(np, (np.sin(th)**2).astype(complex), M, M))
            s = sx + sy; s[mid, mid] -= 1
            rows.append(dict(grid=g, arm=which, max_offdiag=float(np.max(np.abs(off))),
                             diag_err=float(abs(c[mid, mid] - 1)), Cxx_plus_Cyy_minus_I=float(np.max(np.abs(s)))))
            print(rows[-1], flush=True)
    # In a solve: a region of the background's own material must give the uniform slab.
    uniform = cases._problem(())
    ref = rcwa.run(uniform, numerics=rcwa.Numerics(orders=(3, 3)))
    same = cases._problem((cases.R(rcwa.Circle(radius=.3), 1.5),))
    for g in (1, 2, 3, 11, 12, 13, 64):
        for which in ('prototype', 'wrapped'):
            with arm_ctx(which):
                r = rcwa.run(same, numerics=rcwa.Numerics(orders=(3, 3), shape_nvf_grid_size=g))
            rows.append(dict(grid=g, arm=which, test='region-of-background-material vs uniform slab',
                             dR=float(np.max(np.abs(r.R - ref.R))), dT=float(np.max(np.abs(r.T - ref.T)))))
            print(rows[-1], flush=True)
    return rows


# ------------------------------------------------------------------ (c) gradients on JAX
def section_c():
    import jax
    import jax.numpy as jnp
    rows = []
    orders = (3, 3)
    names = ['circle', 'rect-tilted', 'ellipse', 'polygon', 'hexagonal', 'lossy']
    for case in names:
        build, x0, _ = CASES[case]
        for g in (1, 2, 3, 11, 12, 13, 64, None):
            numerics = rcwa.Numerics(orders=orders, shape_nvf_grid_size=g)
            with arm(True):
                with jax.enable_x64(True):
                    prepared = rcwa.prepare(build(x0), numerics=numerics, backend='jax')
                    ad = float(jax.grad(lambda v: jnp.sum(prepared.run(build(v)).T))(jnp.asarray(x0)))
                h = 1e-5
                concrete = rcwa.prepare(build(x0), numerics=numerics)
                f = lambda v: float(np.sum(concrete.run(build(v)).T))
                fd = (f(x0 + h) - f(x0 - h))/(2*h)
            rows.append(dict(case=case, grid=g, ad=ad, fd=fd, ad_vs_fd=abs(ad - fd)/max(abs(fd), 1e-30)))
            print(rows[-1], flush=True)
    # One PyTorch reverse-mode check below the floor.
    import torch
    with arm(True):
        build, x0, _ = CASES['circle']
        numerics = rcwa.Numerics(orders=orders, shape_nvf_grid_size=8)
        prepared = rcwa.prepare(build(torch.tensor(x0, dtype=torch.float64)), numerics=numerics, backend='torch')
        live = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
        torch.sum(prepared.run(build(live)).T).backward()
        rows.append(dict(case='circle', grid=8, backend='torch', ad=float(live.grad)))
        print(rows[-1], flush=True)
    return rows


# ------------------------------------------------------------------ (e) cost at the default grid
def section_e():
    rows = []
    orders = (3, 3)
    for case in ('rectangle', 'circle', 'overlap'):
        for which in ('current', 'prototype', 'current', 'prototype'):
            times, peaks = [], []
            for _ in range(3 if case == 'overlap' else 7):
                with arm_ctx(which):
                    t = time.perf_counter()
                    solve(case, None, orders)
                    times.append(time.perf_counter() - t)
                    parametric._PROJECTOR_CACHE.clear()
                    tracemalloc.start()
                    solve(case, None, orders)
                    peaks.append(tracemalloc.get_traced_memory()[1])
                    tracemalloc.stop()
            rows.append(dict(case=case, arm=which, median_s=float(np.median(times)), min_s=float(np.min(times)),
                             peak_MiB=float(np.median(peaks))/2**20))
            print(rows[-1], flush=True)
    field = np.exp(1j*np.random.default_rng(0).uniform(size=(1024, 1024)))
    for which in ('current', 'prototype', 'current', 'prototype'):
        with arm_ctx(which):
            ts = []
            for _ in range(20):
                t = time.perf_counter(); parametric._sampled_coeffs(np, field, 3, 3); ts.append(time.perf_counter() - t)
        rows.append(dict(case='_sampled_coeffs 1024x1024', arm=which, median_s=float(np.median(ts))))
        print(rows[-1], flush=True)
    return rows


if __name__ == '__main__':
    section = sys.argv[1]
    rows = globals()[f'section_{section}']()
    (OUT / f'{section}.json').write_text(json.dumps(rows, indent=1, default=str))
