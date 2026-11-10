"""Proof that ``arm('integrated')`` really runs the integrated builder, on every backend.

Run from the repository root::

    PYTHONPATH=src python studies/rcwa/isotropic_integrated/check_switch.py

Prints, per arm: the number of calls into ``tensor_shapes._layer`` (0 on the scalar arm),
``sum(R)``/``sum(T)`` of ``circle`` on NumPy, JAX (x64) and Torch, ``d sum(T) / d radius``
by JAX and Torch autodiff beside a central difference, the largest change in ``R``/``T`` when a
uniform layer is split by a region of its own material (round-off on the integrated arm), and
that a geometry prepared under one arm is refused under the other.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from wavebit import Material, rcwa                                         # noqa: E402
from wavebit.rcwa._engine import tensor_shapes                             # noqa: E402
from studies.rcwa.isotropic_integrated import cases as C                   # noqa: E402

CALLS = [0]
_layer = tensor_shapes._layer


def _counted(*args, **kwargs):
    CALLS[0] += 1
    return _layer(*args, **kwargs)


tensor_shapes._layer = _counted


def _split(regions):
    return rcwa.Problem(
        stack=rcwa.Stack(layers=(rcwa.Layer(thickness=C.THICKNESS, background=Material(4.),
                                            regions=regions),),
                         bottom=Material(C.SUBSTRATE), period=(C.PERIOD, C.PERIOD)),
        excitation=rcwa.Excitation(wavelength=C.WAVELENGTH, sources=(
            rcwa.PlaneWave(theta=C.THETA, phi=C.PHI, polarization='s'),)))


def main(case='circle', orders=3, h=1e-5):
    import jax
    import jax.numpy as jnp
    import torch
    x0 = C.CASES[case][1]
    num = C.numerics(orders)
    for name in C.ARMS:
        with C.arm(name):
            CALLS[0] = 0
            values = {b: C.solve(case, orders, backend=b) for b in ('numpy', 'jax', 'torch')}
            print(f'[{name}] _layer calls over three solves: {CALLS[0]}')
            for b, (r, t) in values.items():
                print(f'[{name}] {b:5s} sum(R)={r.sum():.12f} sum(T)={t.sum():.12f}')
            fd = (C.solve(case, orders, x=x0 + h)[1].sum()
                  - C.solve(case, orders, x=x0 - h)[1].sum()) / (2 * h)
            with jax.enable_x64(True):
                gj = jax.grad(lambda x: jnp.sum(
                    rcwa.run(C.problem(case, x), numerics=num, backend='jax').T))(jnp.asarray(x0))
            xt = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
            torch.sum(rcwa.run(C.problem(case, xt), numerics=num, backend='torch').T).backward()
            print(f'[{name}] d sum(T)/dx: central {fd:.8f} jax {float(gj):.8f} '
                  f'torch {float(xt.grad):.8f}')
            uniform = rcwa.run(_split(()), numerics=num)
            for g in (rcwa.Circle(radius=.3), rcwa.Rectangle(size=(.3, .4), angle=.35),
                      rcwa.Ellipse(radii=(.3, .2), angle=.4)):
                split = rcwa.run(_split((rcwa.Region(geometry=g, material=Material(4.)),)),
                                 numerics=num)
                print(f'[{name}] uniform split by {type(g).__name__:9s}: '
                      f'max|dR|={np.max(abs(np.asarray(split.R) - np.asarray(uniform.R))):.1e} '
                      f'max|dT|={np.max(abs(np.asarray(split.T) - np.asarray(uniform.T))):.1e}')
    problem = C.problem(case)
    for a, b in (('scalar', 'integrated'), ('integrated', 'scalar')):
        with C.arm(a):
            prepared = rcwa.prepare(problem, numerics=num)
        with C.arm(b):
            try:
                prepared.run(problem)
                print(f'prepared under {a}, run under {b}: NOT REFUSED')
            except Exception as error:                     # noqa: BLE001 - reported
                print(f'prepared under {a}, run under {b}: refused ({type(error).__name__})')


if __name__ == '__main__':
    main()
