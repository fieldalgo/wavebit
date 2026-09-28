"""Measure the CSG pair predicate without changing its geometric rule.

The legacy spelling is the kernel at cd5ea836. Both spellings are timed in
alternating order after warmup; every output must agree exactly. Run `kernel`
for random half spaces, or `nodes` for full substrate/pillar CSG nodes.
"""
import argparse
import ast
import inspect
import json
from pathlib import Path
import time

import numpy as np

from wavebit.fd._runtime import runtime
from wavebit.fd._shapes import Box, Cylinder, Union
from wavebit.fd._shapes import _bodies


def legacy(d_a, n_a, d_b, n_b, dx, xp):
    zero, one = xp.zeros_like(d_a), xp.ones_like(d_a)
    knots = [zero, one]
    for k in range(3):
        den = n_b[..., k] - n_a[..., k]
        live = den != 0
        safe = xp.where(live, den, xp.ones_like(den))
        knots.append(xp.clip(xp.where(live, n_b[..., k] / safe, zero), zero, one))
    worst = None
    for t in knots:
        g = t * d_a + (1 - t) * d_b
        for k in range(3):
            g = g - (dx / 2) * xp.abs(t * n_a[..., k] + (1 - t) * n_b[..., k])
        worst = g if worst is None else xp.maximum(worst, g)
    return worst >= 0


def candidate(d_a, n_a, d_b, n_b, dx, xp):
    zero, one = xp.zeros_like(d_a), xp.ones_like(d_a)
    end_a, end_b = d_a, d_b
    for k in range(3):
        end_a = end_a - (dx / 2) * xp.abs(n_a[..., k])
        end_b = end_b - (dx / 2) * xp.abs(n_b[..., k])
    worst = xp.maximum(end_b, end_a)
    for k in range(3):
        den = n_b[..., k] - n_a[..., k]
        live = den != 0
        safe = xp.where(live, den, one)
        t = xp.clip(xp.where(live, n_b[..., k] / safe, zero), zero, one)
        other = 1 - t
        g = t * d_a + other * d_b
        for j in range(3):
            g = g - (dx / 2) * xp.abs(t * n_a[..., j] + other * n_b[..., j])
        worst = xp.maximum(worst, g)
    return worst >= 0


def host(value):
    if isinstance(value, (tuple, list)):
        return [host(v) for v in value]
    return np.asarray(value)


def sample(fn):
    start = time.perf_counter()
    result = host(fn())  # synchronizes JAX; Torch measurements use CPU
    return time.perf_counter()-start, result


def compare(functions, repeats):
    warm = {name: sample(fn)[1] for name, fn in functions.items()}
    def exact(a, b):
        if isinstance(a, list):
            for left, right in zip(a, b):
                exact(left, right)
        else:
            np.testing.assert_array_equal(a, b)
    exact(warm['legacy'], warm['candidate'])
    times = {name: [] for name in functions}
    for repeat in range(repeats):
        names = list(functions)
        if repeat % 2:
            names.reverse()
        for name in names:
            seconds, result = sample(functions[name])
            exact(result, warm['legacy'])
            times[name].append(seconds)
    return dict(seconds=times, median={k: float(np.median(v)) for k, v in times.items()},
                exact=True)


def main():
    shipped = ast.parse(inspect.getsource(_bodies._no_common_volume)).body[0].body[1:]
    measured = ast.parse(inspect.getsource(candidate)).body[0].body
    assert [ast.dump(s) for s in shipped] == [ast.dump(s) for s in measured]
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('kernel', 'nodes'))
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--backends', nargs='+', default=['numpy', 'jax', 'torch', 'jit'])
    parser.add_argument('--children', nargs='+', type=int, default=[2, 4, 10])
    parser.add_argument('--origins', nargs='+', type=float, default=[0., .25, .5])
    args = parser.parse_args()
    records = []
    output = Path(__file__).parent/'results'/('fdtd_csg_pair_'+args.mode+'.json')
    rng = np.random.default_rng(20260928)
    count, dx = 66**3, .0152
    arrays = [rng.uniform(-dx, dx, count), rng.normal(size=(count, 3)),
              rng.uniform(-dx, dx, count), rng.normal(size=(count, 3))]
    for i in (1, 3):
        arrays[i] /= np.linalg.norm(arrays[i], axis=-1)[:, None]
    # Parallel, antiparallel, zero components and exact flush faces accompany
    # random oblique planes. No tolerance is used in the equivalence check.
    arrays[3][:count//4] = -arrays[1][:count//4]
    arrays[2][:count//4] = -arrays[0][:count//4]
    arrays[1][count//4:count//2] = [1., 0., 0.]
    arrays[3][count//4:count//2] = [0., 1., 0.]
    for backend in args.backends:
        rt = runtime('jax' if backend == 'jit' else backend)
        xp = rt.xp
        if args.mode == 'kernel':
            for dtype in ('float32', 'float64'):
                values = [xp.asarray(a, dtype=getattr(xp, dtype)) for a in arrays]
                functions = {}
                for name, fn in (('legacy', legacy), ('candidate', candidate)):
                    if backend == 'jit':
                        import jax
                        fn = jax.jit(lambda a, b, c, d, f=fn: f(a, b, c, d, dx, xp))
                        functions[name] = lambda f=fn: f(*values)
                    else:
                        functions[name] = lambda f=fn: f(*values, dx, xp)
                row = dict(backend=backend, dtype=dtype, voxels=count,
                           **compare(functions, args.repeats))
                records.append(row)
                print(json.dumps(row), flush=True)
        else:
            for children in args.children:
                positions = [(x, y) for x in (-.3, 0., .3) for y in (-.3, 0., .3)]
                body = Union(Box((1., 1., .3), center=(0., 0., -.15)),
                             *(Cylinder(.08, .5, center=(x, y, .25))
                               for x, y in positions[:children-1]))
                for origin in args.origins:
                    axis = (np.arange(66)-32.5+origin)*dx
                    points = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), axis=-1)
                    points = rt.array(points, 'float64')
                    functions = {}
                    original = _bodies._no_common_volume
                    def node(kernel, p):
                        try:
                            _bodies._no_common_volume = kernel
                            return body.parts(p, dx)
                        finally:
                            _bodies._no_common_volume = original
                    for name, fn in (('legacy', legacy), ('candidate', candidate)):
                        call = lambda p, f=fn: node(f, p)
                        if backend == 'jit':
                            import jax
                            call = jax.jit(call)
                        functions[name] = lambda f=call: f(points)
                    row = dict(backend=backend, children=children, origin=origin,
                               voxels=count, **compare(functions, args.repeats))
                    records.append(row)
                    print(json.dumps(row), flush=True)
                    # Release compiled graphs before the next child count.
                    if backend == 'jit':
                        jax.clear_caches()
        output.write_text(json.dumps(dict(base='cd5ea836', rows=records), indent=2)+'\n',
                          encoding='utf-8')


if __name__ == '__main__':
    main()
