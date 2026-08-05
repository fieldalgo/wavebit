"""Seconds per solve, by backend and by path, with and without ``jax.jit``.

What this is for: ``jit`` is worth its constraints only where Python overhead is a real share
of the call, and that share collapses as the harmonic count grows — the cost moves into dense
eigendecomposition, which is LAPACK's either way. So the question is not "does jit help" but
"at which ``fto`` does it stop helping", and that is a measurement, not a rule.

Every path here is traceable through ``freeze_geometry()`` + ``conv_solve(geometry=...)``,
including the vector one. Calling ``conv_solve()`` with no geometry is the auto-detect path and
needs concrete values by design; it is not a jit failure.

Run: ``python benchmarks/bench_jit.py``. Numbers in benchmarks/README.md are from one CPU run
and are there for the shape of the trend, not as a target.
"""
import time
import warnings

import numpy as np

import fieldwave

warnings.simplefilter('ignore')

P = 300.0
RASTER = np.where((np.arange(64)[None, :] - 32) ** 2
                  + (np.arange(64)[:, None] - 32) ** 2 < 400, 2.0, 3.5)[None]


def config(fto):
    return dict(period=[P, P], thickness=[205.0], fto=[fto, fto], n_top=1.0, n_bot=1.0,
                theta=0.15, phi=0.35, pol=0, wavelength=900.0)


def ellipse(lx):
    return [[3.5, [['ellipse', 140.0, 160.0, lx, 55.0, 2.0, 0.27, 9, 9]]]]


def rectangle(lx):
    return [[3.5, [['rectangle', 140.0, 160.0, lx, 55.0, 2.0, 0.27, 4, 4]]]]


def analytic(lx):
    return [dict(n_base=3.5, shapes=[dict(type='rectangle', cx=140.0, cy=160.0, lx=lx,
                                          ly=55.0, angle=0.27, n_index=2.0)])]


def raster(_):
    return RASTER


PATHS = [('vector ellipse', ellipse, {}),
         ('vector rectangle', rectangle, {}),
         ('analytic', analytic, {}),
         ('raster DFS', raster, dict(fourier_type=0))]


def timed(call, repeats=5):
    call()
    call()
    t0 = time.perf_counter()
    for _ in range(repeats):
        call()
    return (time.perf_counter() - t0) / repeats


def run(fto_values=(3, 6, 10)):
    import jax
    import jax.numpy as jnp
    import torch
    import fieldwave.on_jax  # noqa: F401

    print(f'{"path":>18} {"fto":>4} {"numpy":>9} {"jax":>9} {"jax+jit":>9} {"torch":>9}'
          f' {"jit speedup":>12}')
    for fto in fto_values:
        kw = config(fto)
        for name, make, extra in PATHS:
            start = 85.0

            t_np = timed(lambda: float(np.sum(np.asarray(
                fieldwave.rcwa(backend=0, ucell=make(start), **extra, **kw)
                .conv_solve().de_ti))))

            t_pt = timed(lambda: float(torch.sum(
                fieldwave.rcwa(backend=2,
                               ucell=make(torch.tensor(start, dtype=torch.float64)),
                               **extra, **kw).conv_solve().de_ti)))

            x = jnp.asarray(start)
            frozen = fieldwave.rcwa(backend=1, ucell=make(x), **extra, **kw).freeze_geometry()

            def total(v):
                return jnp.sum(fieldwave.rcwa(backend=1, ucell=make(v), **extra, **kw)
                               .conv_solve(geometry=frozen).de_ti).real

            jitted = jax.jit(total)
            t_jx = timed(lambda: float(total(x)))
            t_jit = timed(lambda: float(jitted(x)))

            print(f'{name:>18} {fto:>4} {t_np:>9.4f} {t_jx:>9.4f} {t_jit:>9.4f} {t_pt:>9.4f}'
                  f' {t_jx / t_jit:>11.2f}x', flush=True)


if __name__ == '__main__':
    run()
