"""Gradients of sum(T) w.r.t. the shape parameter on both arms (scalar NVF vs integrated PN).

Run from the repository root (single-threaded)::

    PYTHONPATH=src python studies/rcwa/isotropic_integrated/gradients_run.py [grad|torch|sweep ...]

grad : jax.grad (x64) through rcwa.prepare(..., backend='jax') + prepared.run(build(v)),
       against a NumPy central difference (h = 1e-5, 1e-6) from a NumPy prepared object on
       the same arm; orders (3,3) and (6,6), default shape_nvf_grid_size.
torch: one PyTorch backward per arm for circle (orders (3,3) and (6,6)).
sweep: 30-point (polygon: 16) shape-parameter sweep at orders (4,4) on NumPy, both arms, with a local
       central difference (h = 1e-5) at every point.
Results are merged into gradients.json next to this file.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..'))

import wavebit                                                   # noqa: E402
from wavebit import rcwa                                         # noqa: E402
from studies.rcwa.isotropic_integrated import cases as C         # noqa: E402

OUT = os.path.join(HERE, 'gradients.json')
GRAD_CASES = ('circle', 'rect-tilted', 'ellipse', 'polygon', 'hexagonal', 'lossy', 'metallic')
GRAD_ORDERS = (3, 6)
STEPS = (1e-5, 1e-6)
SWEEP_ORDERS = 4
SWEEPS = {  # case -> (lo, hi); 30 points each
    'circle': (.2, .4), 'rect-tilted': (.2, .4), 'ellipse': (.25, .45), 'hexagonal': (.2, .36),
    'lossy': (.2, .4), 'metallic': (.2, .4), 'polygon': (.24, .40)}
NPTS = 30
NPTS_CASE = {'polygon': 16}  # polygon solves cost ~20 s CPU each at the default grid
H_SWEEP = 1e-5


def load():
    if os.path.exists(OUT):
        with open(OUT) as f:
            return json.load(f)
    return {'meta': {}, 'grad': {}, 'torch': {}, 'sweep': {}}


def save(data):
    data['meta'] = dict(wavebit_file=os.path.relpath(wavebit.__file__, os.path.join(HERE, '..', '..', '..')),
                        steps=STEPS, grad_orders=GRAD_ORDERS, sweep_orders=SWEEP_ORDERS,
                        npts=NPTS, h_sweep=H_SWEEP, grid='default (engine)')
    tmp = OUT + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, OUT)


def numpy_value(case, orders):
    build = C.CASES[case][0]
    x0 = C.CASES[case][1]
    prepared = rcwa.prepare(build(x0), numerics=C.numerics(orders))
    return lambda v: float(np.sum(np.asarray(prepared.run(build(v)).T)))


def jax_grad(case, orders):
    import jax
    import jax.numpy as jnp
    build, x0, _ = C.CASES[case]
    with jax.enable_x64(True):
        prepared = rcwa.prepare(build(x0), numerics=C.numerics(orders), backend='jax')
        f = lambda v: jnp.sum(prepared.run(build(v)).T)
        val, g = jax.value_and_grad(f)(jnp.asarray(x0, dtype=jnp.float64))
        return float(val), float(g)


def torch_grad(case, orders):
    import torch
    build, x0, _ = C.CASES[case]
    prepared = rcwa.prepare(build(torch.tensor(x0, dtype=torch.float64)),
                            numerics=C.numerics(orders), backend='torch')
    live = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
    s = torch.sum(prepared.run(build(live)).T)
    s.backward()
    return float(s.detach()), float(live.grad)


def do_grad(data):
    for case in GRAD_CASES:
        for o in GRAD_ORDERS:
            for arm in C.ARMS:
                key = f'{case}|{o}|{arm}'
                if key in data['grad']:
                    continue
                x0 = C.CASES[case][1]
                rec = {}
                with C.arm(arm):
                    try:
                        rec['T_jax'], rec['ad_jax'] = jax_grad(case, o)
                    except Exception as e:  # noqa: BLE001 - recorded
                        rec['jax_error'] = f'{type(e).__name__}: {e}'[:400]
                    value = numpy_value(case, o)
                    rec['T_np'] = value(x0)
                    for h in STEPS:
                        rec[f'fd_{h:g}'] = (value(x0 + h) - value(x0 - h)) / (2 * h)
                data['grad'][key] = rec
                save(data)
                print(key, rec, flush=True)


def do_torch(data):
    for o in GRAD_ORDERS:
        for arm in C.ARMS:
            key = f'circle|{o}|{arm}'
            if key in data['torch']:
                continue
            with C.arm(arm):
                try:
                    t, g = torch_grad('circle', o)
                    data['torch'][key] = dict(T_torch=t, ad_torch=g)
                except Exception as e:  # noqa: BLE001 - recorded
                    data['torch'][key] = dict(error=f'{type(e).__name__}: {e}'[:400])
            save(data)
            print(key, data['torch'][key], flush=True)


def do_sweep(data, only=None):
    for case, (lo, hi) in SWEEPS.items():
        if only and case not in only:
            continue
        xs = np.linspace(lo, hi, NPTS_CASE.get(case, NPTS))
        for arm in C.ARMS:
            key = f'{case}|{arm}'
            if key in data['sweep'] and len(data['sweep'][key]['T']) == len(xs):
                continue
            with C.arm(arm):
                value = numpy_value(case, SWEEP_ORDERS)
                T, G = [], []
                for x in xs:
                    T.append(value(x))
                    G.append((value(x + H_SWEEP) - value(x - H_SWEEP)) / (2 * H_SWEEP))
            data['sweep'][key] = dict(x=xs.tolist(), T=T, fd=G)
            save(data)
            print(key, 'done', flush=True)


if __name__ == '__main__':
    print(wavebit.__file__, flush=True)
    assert wavebit.__file__.startswith(os.path.abspath(os.path.join(HERE, '..', '..', '..', 'src')))
    what = sys.argv[1:] or ['grad', 'torch', 'sweep']
    data = load()
    if 'grad' in what:
        do_grad(data)
    if 'torch' in what:
        do_torch(data)
    if 'sweep' in what:
        do_sweep(data, only=[w for w in what if w in SWEEPS] or None)
