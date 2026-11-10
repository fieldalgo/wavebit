"""Adversarial review of the isotropic-integrated switch: conventions, materials, caches, gradients.

Each section is one argument; results go to ``review_physics_<section>.json``.

    PYTHONPATH=src python studies/rcwa/isotropic_integrated/review_physics.py SECTION

Sections:

* ``limit``     -- do both arms converge to one limit as the order grows (no convention offset)?
* ``offset``    -- two patterned layers at a relative offset, square and oblique cells: the
                   shift phase and lattice convention, which a lone layer cannot see.
* ``translate`` -- a rigid translation leaves R, T unchanged on the integrated arm.
* ``magnetic``  -- a magnetic region (mu != 1): mu reaches the integrated builder.
* ``background``-- a lossy background: background read by the integrated builder.
* ``cache``     -- arm order within one process: scalar/integrated/scalar is bitwise stable.
* ``grad``      -- autodiff of sum(T) with respect to a material and a position.
* ``polyscan``  -- T(x) on the integrated polygon (hard nearest projector) near x0.
* ``other``     -- factorization='laurent' and nvf_projector='circular' under the switch.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import wavebit                                                             # noqa: E402
from wavebit import Material, rcwa                                         # noqa: E402
from wavebit.rcwa._engine import tensor_shapes                             # noqa: E402
from studies.rcwa.isotropic_integrated import cases as C                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
assert wavebit.__file__.startswith(os.path.join(HERE, '..', '..', '..', 'src')) or \
    os.path.realpath(wavebit.__file__).startswith(os.path.realpath(os.path.join(HERE, '..', '..', '..', 'src')))

CALLS = []
_layer = tensor_shapes._layer


def _counted(solver, layer, plan, quantity):
    CALLS.append((plan[0], quantity))
    return _layer(solver, layer, plan, quantity)


tensor_shapes._layer = _counted


def stack(layers, lattice=None, bottom=C.SUBSTRATE, pol='s'):
    cell = dict(lattice=lattice) if lattice is not None else dict(period=(C.PERIOD, C.PERIOD))
    return rcwa.Problem(
        stack=rcwa.Stack(layers=tuple(layers), bottom=Material(bottom), **cell),
        excitation=rcwa.Excitation(wavelength=C.WAVELENGTH, sources=(
            rcwa.PlaneWave(theta=C.THETA, phi=C.PHI, polarization=pol),)))


def layer(regions, background=1.5, thickness=C.THICKNESS):
    return rcwa.Layer(thickness=thickness, background=Material(background), regions=tuple(regions))


def reg(g, eps, mu=None):
    return rcwa.Region(geometry=g, material=Material(eps) if mu is None else Material(eps, mu))


def run(problem, orders, grid=None, backend='numpy', **kw):
    res = rcwa.run(problem, numerics=rcwa.Numerics(orders=(orders, orders), factorization='nvf',
                                                   shape_nvf_grid_size=grid, **kw), backend=backend)
    return C.to_np(res.R), C.to_np(res.T)


def both(problem, orders, grid=None):
    out = {}
    for arm in C.ARMS:
        with C.arm(arm):
            CALLS.clear()
            r, t = run(problem, orders, grid)
            out[arm] = dict(R=r.tolist(), T=t.tolist(), sumR=float(r.sum()), sumT=float(t.sum()),
                            calls=list(map(list, CALLS)))
    return out


def dump(name, data):
    with open(os.path.join(HERE, f'review_physics_{name}.json'), 'w') as f:
        json.dump(data, f, indent=1)


def limit():
    data = {}
    for case in ('circle', 'ellipse', 'rect-tilted', 'hexagonal', 'oblique', 'high-contrast',
                 'metallic', 'lossy'):
        data[case] = {}
        for m in (3, 5, 7, 9, 11):
            row = {}
            for arm in C.ARMS:
                with C.arm(arm):
                    r, t = C.solve(case, m, grid=512)
                row[arm] = [float(r.sum()), float(t.sum())]
            data[case][m] = row
            print(case, m, row, f"dT={row['integrated'][1] - row['scalar'][1]:+.2e}", flush=True)
    dump('limit', data)


def offset():
    data = {}
    for name, lattice in (('square', None), ('oblique', wavebit.Lattice.oblique(1., 1.1, 1.3)),
                          ('hexagonal', wavebit.Lattice.hexagonal(1.))):
        data[name] = {}
        for label, centre in (('+s', (.2, .1)), ('-s', (-.2, -.1)), ('mirror-y', (.2, -.1))):
            p = stack([layer([reg(rcwa.Circle(radius=.2), 4.)]),
                       layer([reg(rcwa.Ellipse(radii=(.3, .14), angle=.4, center=centre), 6.)],
                             background=1.2)], lattice=lattice)
            for m in (5, 9):
                res = both(p, m, grid=512)
                data[name][f'{label}/{m}'] = {a: [res[a]['sumR'], res[a]['sumT'],
                                                  len(res[a]['calls'])] for a in C.ARMS}
                print(name, label, m, data[name][f'{label}/{m}'], flush=True)
    dump('offset', data)


def translate():
    data = {}
    with C.arm('integrated'):
        for name, regions0, regions1, grid in (
                ('circle', [reg(rcwa.Circle(radius=.3), 4.)],
                 [reg(rcwa.Circle(radius=.3, center=(.17, -.11)), 4.)], None),
                ('overlap', [reg(rcwa.Circle(radius=.25, center=(-.12, 0.)), 4.),
                             reg(rcwa.Circle(radius=.22, center=(.15, .05)), 9.)],
                 [reg(rcwa.Circle(radius=.25, center=(-.12 + .2, .0 - .13)), 4.),
                  reg(rcwa.Circle(radius=.22, center=(.15 + .2, .05 - .13)), 9.)], 64)):
            r0, t0 = run(stack([layer(regions0)]), 3, grid)
            r1, t1 = run(stack([layer(regions1)]), 3, grid)
            data[name] = dict(dR=float(np.max(abs(r1 - r0))), dT=float(np.max(abs(t1 - t0))))
            print('translate', name, data[name], flush=True)
    dump('translate', data)


def magnetic():
    data = {}
    for label, regions in (('mu=2', [reg(rcwa.Circle(radius=.3), 4., 2.)]),
                           ('mu=2 eps=bg', [reg(rcwa.Circle(radius=.3), 1.5, 2.)])):
        for m in (3, 7):
            res = both(stack([layer(regions)]), m, grid=256)
            data[f'{label}/{m}'] = {a: [res[a]['sumR'], res[a]['sumT'], res[a]['calls']]
                                    for a in C.ARMS}
            print('magnetic', label, m, data[f'{label}/{m}'], flush=True)
    # The duality check: swapping (eps, mu) and polarization s <-> p maps R, T onto each other.
    for arm in C.ARMS:
        with C.arm(arm):
            a = run(stack([layer([reg(rcwa.Circle(radius=.3), 4., 1.)], background=1.)],
                          bottom=1.), 5, 256)
            p_dual = stack([layer([reg(rcwa.Circle(radius=.3), 1., 4.)], background=1.)],
                           bottom=1., pol='p')
            b = run(p_dual, 5, 256)
            data[f'duality/{arm}'] = dict(sumR=[float(a[0].sum()), float(b[0].sum())],
                                         sumT=[float(a[1].sum()), float(b[1].sum())])
            print('duality', arm, data[f'duality/{arm}'], flush=True)
    dump('magnetic', data)


def background():
    data = {}
    regions = [reg(rcwa.Circle(radius=.3), 4.)]
    for bg in (2. + .5j, 1.5):
        for m in (3, 7):
            res = both(stack([layer(regions, background=bg)]), m, grid=256)
            data[f'{bg}/{m}'] = {a: [res[a]['sumR'], res[a]['sumT'], len(res[a]['calls'])]
                                 for a in C.ARMS}
            print('background', bg, m, data[f'{bg}/{m}'], flush=True)
    dump('background', data)


def cache():
    data = {}
    for case, grid in (('circle', 256), ('overlap', 64), ('hexagonal', 256)):
        seq = []
        for arm in ('scalar', 'integrated', 'scalar', 'integrated'):
            with C.arm(arm):
                seq.append(C.solve(case, 3, grid=grid))
        same_s = all(np.array_equal(a, b) for a, b in zip(seq[0], seq[2]))
        same_i = all(np.array_equal(a, b) for a, b in zip(seq[1], seq[3]))
        differ = float(np.max(abs(seq[0][1] - seq[1][1])))
        data[case] = dict(scalar_bitwise=same_s, integrated_bitwise=same_i, arms_differ=differ)
        print('cache', case, data[case], flush=True)
    # A prepared object reused across arms, in a tensor stack.
    t = stack([layer([reg(rcwa.Circle(radius=.3), 4.)]),
               rcwa.Layer(thickness=.1, background=Material(np.diag([2.1, 2.5, 3.0])))])
    for a, b in (('scalar', 'integrated'), ('integrated', 'scalar')):
        with C.arm(a):
            prep = rcwa.prepare(t, numerics=rcwa.Numerics(orders=(3, 3), factorization='nvf',
                                                          shape_nvf_grid_size=256))
        with C.arm(b):
            try:
                prep.run(t)
                data[f'tensor {a}->{b}'] = 'NOT REFUSED'
            except Exception as e:                       # noqa: BLE001 - reported
                data[f'tensor {a}->{b}'] = type(e).__name__
        print('cache tensor', a, b, data[f'tensor {a}->{b}'], flush=True)
    dump('cache', data)


def grad():
    import jax
    import jax.numpy as jnp
    import torch
    data = {}
    h = 1e-5
    num = rcwa.Numerics(orders=(3, 3), factorization='nvf', shape_nvf_grid_size=256)

    def eps_problem(e):
        return stack([layer([rcwa.Region(geometry=rcwa.Circle(radius=.3), material=Material(e))])])

    def pos_problem(c):
        return stack([layer([rcwa.Region(geometry=rcwa.Circle(radius=.3, center=(c, .05)),
                                         material=Material(4.))])])

    def ell_pos_problem(c):
        return stack([layer([reg(rcwa.Circle(radius=.2, center=(-.15, 0.)), 4.),
                             rcwa.Region(geometry=rcwa.Ellipse(radii=(.15, .1), angle=.3,
                                                               center=(c, .1)),
                                         material=Material(6.))])])

    for arm in C.ARMS:
        with C.arm(arm):
            for name, build, x0 in (('eps', eps_problem, 4.), ('cx', pos_problem, .1),
                                    ('two-shape cx', ell_pos_problem, .2)):
                f = lambda x: float(np.sum(C.to_np(rcwa.run(build(x), numerics=num).T)))  # noqa: E731
                fd = (f(x0 + h) - f(x0 - h)) / (2 * h)
                with jax.enable_x64(True):
                    gj = float(jax.grad(lambda x: jnp.sum(
                        rcwa.run(build(x), numerics=num, backend='jax').T))(jnp.asarray(x0)))
                xt = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
                torch.sum(rcwa.run(build(xt), numerics=num, backend='torch').T).backward()
                data[f'{arm}/{name}'] = dict(fd=fd, jax=gj, torch=float(xt.grad))
                print('grad', arm, name, data[f'{arm}/{name}'], flush=True)
    dump('grad', data)


def polyscan():
    data = {}
    x0 = C.CASES['polygon'][1]
    for arm in C.ARMS:
        with C.arm(arm):
            vals = []
            for k in range(-5, 6):
                x = x0 + k * 1e-5
                vals.append([x, float(C.solve('polygon', 2, grid=128, x=x)[1].sum())])
            data[arm] = vals
            print('polyscan', arm, np.diff([v[1] for v in vals]).tolist(), flush=True)
    dump('polyscan', data)


def other():
    data = {}
    p = C.problem('circle')
    for arm in C.ARMS:
        with C.arm(arm):
            for label, kw in (('laurent', dict(factorization='laurent')),
                              ('nvf circular', dict(factorization='nvf', nvf_projector='circular')),
                              ('nvf linear', dict(factorization='nvf', nvf_projector='linear'))):
                CALLS.clear()
                try:
                    res = rcwa.run(p, numerics=rcwa.Numerics(orders=(3, 3), shape_nvf_grid_size=256,
                                                             **kw))
                    data[f'{arm}/{label}'] = [float(np.sum(res.T)), len(CALLS)]
                except Exception as e:                   # noqa: BLE001 - reported
                    data[f'{arm}/{label}'] = f'{type(e).__name__}: {str(e)[:200]}'
                print('other', arm, label, data[f'{arm}/{label}'], flush=True)
    dump('other', data)


if __name__ == '__main__':
    globals()[sys.argv[1]]()
