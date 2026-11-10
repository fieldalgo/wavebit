"""Physics and exactness checks of the two isotropic shape builders (``scalar`` vs ``integrated``).

Run one check at a time from the repository root (each writes ``physics_<check>.json`` beside
this file)::

    PYTHONPATH=src python studies/rcwa/isotropic_integrated/physics_run.py uniform
    ... lamellar | mirror | energy | reciprocity | metal | metal_scan | fem

Checks
------
uniform      a uniform layer split by regions of its own material, against the unpatterned slab.
lamellar     a rectangle spanning the whole cell in one direction (1D lamellar grating), against
             ``factorization='li'`` (exact rule for a lamellar grating) at the same orders and at
             a high 1D order.
mirror       a shape mirrored in x with mirrored incidence (phi -> pi - phi) must give mirrored
             order efficiencies; matched by mirrored k_parallel, so any lattice works.
energy       |1 - R - T| for lossless cases; absorption sign for lossy/metallic; the
             anti-Hermitian part of the in-plane effective permittivity matrix.
reciprocity  specular reflection R_ab(k) against R_ba(-k) on non-centrosymmetric layers.
metal        the metallic disk: whether the scalar arm's inverse-rule floor binds, and the
             conditioning of the integrated arm's [[B]].
metal_scan   R + T over a wavelength window on the metallic disk, including an unfloored
             scalar control.
fem          the two-piece affine profile of docs/validation/rcwa/nvf-independent-fem.md
             (independent NGSolve Maxwell FEM) through the public API.

``factorization='li'`` is refused while the study switch is on (the prototype's stale-record
check also runs for Li/Laurent records, which never carry ``tensor_geometry``), so Li references
are always computed with the switch off; Li itself does not read the switch.
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..'))

import wavebit                                                            # noqa: E402
from wavebit import Lattice, Material, rcwa                               # noqa: E402
from wavebit.rcwa._engine import engine as _engine                        # noqa: E402
from wavebit.rcwa._engine import factorization as _fac                    # noqa: E402
from wavebit.rcwa._engine import parametric as _P                         # noqa: E402
from wavebit.rcwa._engine import tensor_shapes                            # noqa: E402
from studies.rcwa.isotropic_integrated import cases as C                  # noqa: E402

assert os.path.realpath(wavebit.__file__).startswith(
    os.path.realpath(os.path.join(HERE, '..', '..', '..', 'src'))), wavebit.__file__

# ----------------------------------------------------------------------------- instrumentation
CALLS = {'layer': 0}
CAPTURE = {'slots': [], 'B': [], 'floor': []}
_layer0 = tensor_shapes._layer
_slots0 = _engine.CoreFactorizationMixin._slots_for
_floor0 = _fac._floor_singular_values
_bound0 = _fac.inverse_rule_bound
OPTIONS = {'capture': False, 'unfloored': False}


def _layer(solver, layer, plan, quantity):
    CALLS['layer'] += 1
    out = _layer0(solver, layer, plan, quantity)
    if OPTIONS['capture'] and quantity == 'epsilon':
        xp = solver.xp
        (_, hu), (_, hv) = solver.order_window
        A, B, zz = out
        def block(c):
            return np.asarray(xp.concat([xp.concat([_P.conv_matrix_2d(xp, v, hu, hv) for v in row],
                                                   axis=1) for row in c], axis=0))
        Bm = block(B)
        s = np.linalg.svd(Bm, compute_uv=False)
        ev = np.linalg.eigvals(Bm)
        CAPTURE['B'].append(dict(size=Bm.shape[0], smin=float(s.min()), smax=float(s.max()),
                                 cond=float(s.max() / s.min()),
                                 min_abs_eig=float(np.abs(ev).min()),
                                 finite=bool(np.isfinite(Bm).all())))
    return out


def _slots(self, structure, geometry, quantity):
    out = _slots0(self, structure, geometry, quantity)
    if OPTIONS['capture'] and quantity == 'permittivity':
        CAPTURE['slots'].append([None if m is None else np.array(m[0]) for m in out])
    return out


def _floor(xp, B, floor):
    if OPTIONS['capture']:
        Bn = np.asarray(B)
        s = np.linalg.svd(Bn.reshape(-1, *Bn.shape[-2:])[0], compute_uv=False)
        f = float(np.asarray(floor).reshape(-1)[0])
        CAPTURE['floor'].append(dict(n=int(s.size), smin=float(s.min()), floor=f,
                                     below=int((s < f).sum())))
    return _floor0(xp, B, floor)


def _bound(xp, values):
    return None if OPTIONS['unfloored'] else _bound0(xp, values)


tensor_shapes._layer = _layer
_engine.CoreFactorizationMixin._slots_for = _slots
_fac._floor_singular_values = _floor
_fac.inverse_rule_bound = _bound
_P.inverse_rule_bound = _bound            # imported by name there and in boundary
from wavebit.rcwa._engine import boundary as _boundary                    # noqa: E402
_boundary.inverse_rule_bound = _bound


def passivity(slots):
    """In-plane effective permittivity [[eps_xx, eps_xy], [eps_yx, eps_yy]] from the 5 slots."""
    epx, epy, epz_i, epxy, epyx = slots[:5]
    if epxy is None:
        epxy = np.zeros_like(epx)
        epyx = epxy
    E = np.block([[epx, -epxy], [-epyx, epy]])
    H = (E - E.conj().T) / 2j
    ev = np.linalg.eigvalsh(H)
    return dict(antiherm_rel=float(np.linalg.norm(E - E.conj().T) / np.linalg.norm(E)),
                antiherm_min_eig=float(ev.min()), antiherm_max_eig=float(ev.max()),
                finite=bool(np.isfinite(E).all()))


# ----------------------------------------------------------------------------- problems
def problem(regions, *, lattice=None, background=1.5, theta=C.THETA, phi=C.PHI, pol='s',
            wavelength=C.WAVELENGTH, thickness=C.THICKNESS, bottom=C.SUBSTRATE):
    cell = dict(lattice=lattice) if lattice is not None else dict(period=(C.PERIOD, C.PERIOD))
    return rcwa.Problem(
        stack=rcwa.Stack(layers=(rcwa.Layer(thickness=thickness, background=Material(background),
                                            regions=tuple(regions)),),
                         bottom=Material(bottom), **cell),
        excitation=rcwa.Excitation(wavelength=wavelength, sources=(
            rcwa.PlaneWave(theta=theta, phi=phi, polarization=pol),)))


def R(geometry, eps):
    return rcwa.Region(geometry=geometry, material=Material(eps))


def run(prob, orders, grid=None, factorization='nvf'):
    if isinstance(orders, int):
        orders = (orders, orders)
    num = rcwa.Numerics(orders=tuple(orders), factorization=factorization,
                        shape_nvf_grid_size=grid)
    calls = CALLS['layer']
    res = rcwa.run(prob, numerics=num)
    return res, CALLS['layer'] - calls


def RT(res):
    return np.asarray(res.R)[0], np.asarray(res.T)[0]


def li(prob, orders):
    with C.arm('scalar'):
        res, _ = run(prob, orders, factorization='li')
    return res


def save(name, data):
    path = os.path.join(HERE, f'physics_{name}.json')
    with open(path, 'w') as f:
        json.dump(data, f, indent=1, allow_nan=True)
    print('wrote', path, flush=True)


def log(*a):
    print(*a, flush=True)


PENT = C.PENTAGON
HEX = Lattice.hexagonal(C.PERIOD)
OBL = Lattice.oblique(C.PERIOD, 1.1, 1.3)


def mirror_lattice(lat):
    u, v = np.asarray(lat.u, float), np.asarray(lat.v, float)
    M = np.diag([-1., 1.])
    # (Mv, Mu) keeps a positive orientation and maps the retained order set onto its mirror.
    return Lattice.from_columns(np.stack([M @ v, M @ u], axis=1))


# ----------------------------------------------------------------------------- 1 uniform
def check_uniform(orders=5):
    out = []
    eq = 1.5
    tests = [
        ('circle', [R(rcwa.Circle(radius=.3), eq)], None, None),
        ('rect-tilted', [R(rcwa.Rectangle(size=(.3, .4), angle=.35), eq)], None, None),
        ('ellipse', [R(rcwa.Ellipse(radii=(.35, .2), angle=.4), eq)], None, None),
        ('polygon', [R(rcwa.Polygon(vertices=PENT), eq)], None, None),
        ('overlap', [R(rcwa.Circle(radius=.25, center=(-.12, 0.)), eq),
                     R(rcwa.Circle(radius=.22, center=(.15, .05)), eq)], None, 64),
        ('hexagonal', [R(rcwa.Circle(radius=.28), eq)], HEX, None),
        ('oblique', [R(rcwa.Ellipse(radii=(.3, .18), angle=.2), eq)], OBL, None),
    ]
    for bg in (1.5, 4. + 1.j, -20. + 1.5j):
        for name, regs, lat, grid in tests:
            if bg != 1.5 and name not in ('circle', 'polygon', 'overlap'):
                continue
            for rel in (0., 1e-12):
                regs2 = [R(r.geometry, bg * (1 + rel)) for r in regs]
                for arm in C.ARMS:
                    with C.arm(arm):
                        ref, _ = run(problem([], lattice=lat, background=bg), orders, grid)
                        res, calls = run(problem(regs2, lattice=lat, background=bg), orders, grid)
                    r0, t0 = RT(ref)
                    r1, t1 = RT(res)
                    row = dict(case=name, background=str(bg), rel_perturbation=rel, arm=arm,
                               orders=orders, grid=grid, layer_calls=calls,
                               dR=float(np.abs(r1 - r0).max()), dT=float(np.abs(t1 - t0).max()))
                    log(row)
                    out.append(row)
    save('uniform', out)


# ----------------------------------------------------------------------------- 2 lamellar
def check_lamellar():
    out = []
    cases = [
        ('lam-x eps4/1.5', [R(rcwa.Rectangle(size=(.4, 1.)), 4.)], 1.5, 'x'),
        ('lam-x off-centre', [R(rcwa.Rectangle(size=(.4, 1.), center=(.17, 0.)), 4.)], 1.5, 'x'),
        ('lam-x eps12.25/air', [R(rcwa.Rectangle(size=(.4, 1.)), 12.25)], 1., 'x'),
        ('lam-x metal/air', [R(rcwa.Rectangle(size=(.4, 1.)), -20. + 1.5j)], 1., 'x'),
        ('lam-y eps4/1.5', [R(rcwa.Rectangle(size=(1., .4)), 4.)], 1.5, 'y'),
    ]
    for name, regs, bg, axis in cases:
        prob = problem(regs, background=bg)
        # converged 1D Li reference
        one_d = (lambda M: (M, 0)) if axis == 'x' else (lambda M: (0, M))
        refs = {}
        for M in (100, 200, 400):
            res = li(prob, one_d(M))
            r, t = RT(res)
            refs[M] = (float(r.sum()), float(t.sum()))
            log(name, 'li 1D', M, refs[M])
        ref_R, ref_T = refs[400]
        orders_list = [(M, M) for M in (2, 3, 5, 8)] + [one_d(M) for M in (5, 10, 20, 40)]
        for o in orders_list:
            lres = li(prob, o)
            lr, lt = RT(lres)
            for arm in C.ARMS:
                with C.arm(arm):
                    try:
                        res, calls = run(prob, o)
                    except Exception as e:                        # noqa: BLE001 - reported
                        row = dict(case=name, arm=arm, orders=o, error=f'{type(e).__name__}: {e}'[:300])
                        log(row)
                        out.append(row)
                        continue
                r, t = RT(res)
                row = dict(case=name, arm=arm, orders=o, layer_calls=calls,
                           max_dR_vs_li=float(np.abs(r - lr).max()),
                           max_dT_vs_li=float(np.abs(t - lt).max()),
                           sumT=float(t.sum()), sumR=float(r.sum()),
                           li_sumT=float(lt.sum()),
                           dT_vs_li400=float(t.sum() - ref_T), li_dT_vs_li400=float(lt.sum() - ref_T),
                           energy=float(1 - r.sum() - t.sum()))
                log(row)
                out.append(row)
        out.append(dict(case=name, li_1d_reference={str(k): v for k, v in refs.items()}))
    save('lamellar', out)


def check_lamellar_floor():
    """The metal lamellar rows where the scalar arm left Li: is it the inverse-rule floor?"""
    out = []
    prob = problem([R(rcwa.Rectangle(size=(.4, 1.)), -20. + 1.5j)], background=1.)
    for o in ((3, 3), (10, 0), (20, 0), (40, 0)):
        lr, lt = RT(li(prob, o))
        for arm in ('scalar', 'scalar-unfloored'):
            OPTIONS['capture'] = True
            OPTIONS['unfloored'] = arm == 'scalar-unfloored'
            CAPTURE['floor'].clear()
            with C.arm('scalar'):
                res, calls = run(prob, o)
            OPTIONS['capture'] = OPTIONS['unfloored'] = False
            r, t = RT(res)
            row = dict(case='lam-x metal/air', arm=arm, orders=o,
                       max_d_vs_li=float(max(np.abs(r - lr).max(), np.abs(t - lt).max())),
                       floor=CAPTURE['floor'][-1] if CAPTURE['floor'] else None)
            log(row)
            out.append(row)
    save('lamellar_floor', out)


# ----------------------------------------------------------------------------- 3 mirror
def _kmatch(res_a, res_b, which):
    da = getattr(res_a, which)
    db = getattr(res_b, which)
    ka = np.asarray(da.k_parallel)[0].real
    kb = np.asarray(db.k_parallel)[0].real
    ea = np.asarray(da.efficiency)[0]
    eb = np.asarray(db.efficiency)[0]
    scale = np.abs(ka).max()
    keys = {}
    for idx in np.ndindex(eb.shape):
        keys[tuple(np.round(kb[idx] / scale * 1e8).astype(np.int64))] = idx
    worst, matched, total = 0., 0, 0
    for idx in np.ndindex(ea.shape):
        k = ka[idx] * np.array([-1., 1.])
        key = tuple(np.round(k / scale * 1e8).astype(np.int64))
        total += 1
        if key in keys:
            matched += 1
            worst = max(worst, abs(ea[idx] - eb[keys[key]]))
        elif ea[idx] > 1e-12:
            worst = math.inf
    return worst, matched, total


def mirror_regions(regs):
    out = []
    for r in regs:
        g = r.geometry
        if isinstance(g, rcwa.Circle):
            ng = rcwa.Circle(radius=g.radius, center=(-g.center[0], g.center[1]))
        elif isinstance(g, rcwa.Rectangle):
            ng = rcwa.Rectangle(size=g.size, center=(-g.center[0], g.center[1]), angle=-g.angle)
        elif isinstance(g, rcwa.Ellipse):
            ng = rcwa.Ellipse(radii=g.radii, center=(-g.center[0], g.center[1]), angle=-g.angle)
        elif isinstance(g, rcwa.Polygon):
            ng = rcwa.Polygon(vertices=[(-x, y) for x, y in g.vertices][::-1])
        else:
            raise TypeError(g)
        out.append(rcwa.Region(geometry=ng, material=r.material))
    return out


def check_mirror(orders=5):
    out = []
    tests = [
        ('rect-tilted', [R(rcwa.Rectangle(size=(.3, .4), angle=.35), 4.)], None, None),
        ('ellipse', [R(rcwa.Ellipse(radii=(.35, .2), angle=.4), 6.)], None, None),
        ('circle-offset', [R(rcwa.Circle(radius=.2, center=(.15, .05)), 4.)], None, None),
        ('polygon', [R(rcwa.Polygon(vertices=PENT), 4.)], None, None),
        ('overlap', [R(rcwa.Circle(radius=.25, center=(-.12, 0.)), 4.),
                     R(rcwa.Circle(radius=.22, center=(.15, .05)), 9.)], None, 64),
        ('hexagonal-offset', [R(rcwa.Circle(radius=.2, center=(.1, .05)), 4.)], HEX, None),
        ('oblique', [R(rcwa.Ellipse(radii=(.3, .18), angle=.2), 5.)], OBL, None),
        ('metal-ellipse', [R(rcwa.Ellipse(radii=(.3, .2), angle=.4), -20. + 1.5j)], None, None),
        ('circle (self)', [R(rcwa.Circle(radius=.3), 4.)], None, None),
        ('rectangle (self)', [R(rcwa.Rectangle(size=(.3, .4)), 4.)], None, None),
    ]
    for name, regs, lat, grid in tests:
        mregs = mirror_regions(regs)
        mlat = None if lat is None else mirror_lattice(lat)
        bg = 1. if name.startswith('metal') else 1.5
        for theta, phi in ((C.THETA, C.PHI), (0., 0.)):
            for pol in ('s', 'p'):
                for arm in C.ARMS:
                    with C.arm(arm):
                        a, ca = run(problem(regs, lattice=lat, theta=theta, phi=phi, pol=pol,
                                            background=bg), orders, grid)
                        b, cb = run(problem(mregs, lattice=mlat, theta=theta, phi=math.pi - phi,
                                            pol=pol, background=bg), orders, grid)
                    wr, m1, n1 = _kmatch(a, b, 'reflection')
                    wt, m2, n2 = _kmatch(a, b, 'transmission')
                    row = dict(case=name, arm=arm, theta=theta, phi=phi, pol=pol, orders=orders,
                               grid=grid, layer_calls=ca + cb, max_dR=wr, max_dT=wt,
                               matched=f'{m1}/{n1},{m2}/{n2}',
                               sumR=float(np.asarray(a.R).sum()), sumT=float(np.asarray(a.T).sum()))
                    log(row)
                    out.append(row)
    save('mirror', out)


def check_mirror_overlap(orders=5):
    """The painted 'overlap' mirror defect of the integrated arm against the projector grid."""
    out = []
    regs = [R(rcwa.Circle(radius=.25, center=(-.12, 0.)), 4.),
            R(rcwa.Circle(radius=.22, center=(.15, .05)), 9.)]
    mregs = mirror_regions(regs)
    for grid in (32, 48, 64, 96, 128, 256, 512):
        for arm in C.ARMS:
            if arm == 'scalar' and grid > 128:
                continue                       # scalar painted cost ~ grid**2; 64 shown exact
            with C.arm(arm):
                try:
                    a, _ = run(problem(regs), orders, grid)
                    b, _ = run(problem(mregs, phi=math.pi - C.PHI), orders, grid)
                except Exception as e:                    # noqa: BLE001 - reported
                    out.append(dict(arm=arm, grid=grid, error=f'{type(e).__name__}: {e}'[:200]))
                    log(out[-1])
                    continue
            wr, _, _ = _kmatch(a, b, 'reflection')
            wt, _, _ = _kmatch(a, b, 'transmission')
            out.append(dict(arm=arm, grid=grid, orders=orders, max_dR=wr, max_dT=wt,
                            sumT=float(np.asarray(a.T).sum())))
            log(out[-1])
    save('mirror_overlap', out)


# ----------------------------------------------------------------------------- 4 energy
def _case_problem(case, pol):
    import dataclasses
    p = C.problem(case)
    src = p.excitation.sources[0]
    src = dataclasses.replace(src, polarization=pol)
    return dataclasses.replace(p, excitation=dataclasses.replace(p.excitation, sources=(src,)))


def check_energy():
    out = []
    names = ['rectangle', 'rect-tilted', 'circle', 'ellipse', 'polygon', 'overlap', 'hexagonal',
             'oblique', 'high-contrast', 'lossy', 'metallic']
    for case in names:
        grid = 64 if case == 'overlap' else None
        lossless = C.CASES[case][2]
        for M in (3, 5, 8):
            for pol in ('s', 'p'):
                for arm in C.ARMS:
                    OPTIONS['capture'] = pol == 's'
                    CAPTURE['slots'].clear()
                    with C.arm(arm):
                        res, calls = run(_case_problem(case, pol), M, grid)
                    OPTIONS['capture'] = False
                    r, t = RT(res)
                    row = dict(case=case, arm=arm, orders=M, pol=pol, grid=grid, lossless=lossless,
                               layer_calls=calls, sumR=float(r.sum()), sumT=float(t.sum()),
                               one_minus_RT=float(1 - r.sum() - t.sum()),
                               finite=bool(np.isfinite(r).all() and np.isfinite(t).all()))
                    if CAPTURE['slots']:
                        row.update(passivity(CAPTURE['slots'][-1]))
                    log(row)
                    out.append(row)
        save('energy', out)


# ----------------------------------------------------------------------------- 4b reciprocity
def _specular_channels(res):
    d = res.reflection
    e = np.asarray(d.efficiency)[0]
    k = np.asarray(d.k_parallel)[0]
    ou, ov = list(d.orders_u), list(d.orders_v)
    i, j = ov.index(0), ou.index(0)
    a_s = np.asarray(d.amplitude_ratio_s)[0][i, j]
    a_p = np.asarray(d.amplitude_ratio_p)[0][i, j]
    w = abs(a_s) ** 2 + abs(a_p) ** 2
    return dict(s=float(e[i, j] * abs(a_s) ** 2 / w), p=float(e[i, j] * abs(a_p) ** 2 / w))


def check_reciprocity(orders=5):
    out = []
    tests = [
        ('polygon', [R(rcwa.Polygon(vertices=PENT), 4.)], 1.5, None),
        ('polygon-lossy', [R(rcwa.Polygon(vertices=PENT), 4. + 1.j)], 1.5, None),
        ('overlap', [R(rcwa.Circle(radius=.25, center=(-.12, 0.)), 4.),
                     R(rcwa.Circle(radius=.22, center=(.15, .05)), 9.)], 1.5, 64),
        ('circle-offset metal', [R(rcwa.Circle(radius=.2, center=(.15, .05)), -20. + 1.5j),
                                 R(rcwa.Rectangle(size=(.2, .1), center=(-.2, -.2), angle=.3), 4.)],
         1., None),
    ]
    for name, regs, bg, grid in tests:
        for arm in C.ARMS:
            ch = {}
            with C.arm(arm):
                for sign, phi in (('+', C.PHI), ('-', C.PHI + math.pi)):
                    for pol in ('s', 'p'):
                        res, calls = run(problem(regs, background=bg, phi=phi, pol=pol), orders, grid)
                        ch[sign + pol] = _specular_channels(res)
            # R_{a->b}(k) = R_{b->a}(-k)
            d = dict(ss=abs(ch['+s']['s'] - ch['-s']['s']), pp=abs(ch['+p']['p'] - ch['-p']['p']),
                     sp=abs(ch['+s']['p'] - ch['-p']['s']), ps=abs(ch['+p']['s'] - ch['-s']['p']))
            rel = max(d.values()) / max(max(v.values()) for v in ch.values())
            row = dict(case=name, arm=arm, orders=orders, grid=grid,
                       channels=ch, abs_diff=d, max_rel=rel)
            log(row)
            out.append(row)
    save('reciprocity', out)


def check_reciprocity_unfloored(orders=5):
    """The scalar arm's metal reciprocity residual with the inverse-rule floor removed."""
    regs = [R(rcwa.Circle(radius=.2, center=(.15, .05)), -20. + 1.5j),
            R(rcwa.Rectangle(size=(.2, .1), center=(-.2, -.2), angle=.3), 4.)]
    out = []
    for unfloored in (False, True):
        OPTIONS['unfloored'] = unfloored
        ch = {}
        with C.arm('scalar'):
            for sign, phi in (('+', C.PHI), ('-', C.PHI + math.pi)):
                for pol in ('s', 'p'):
                    res, _ = run(problem(regs, background=1., phi=phi, pol=pol), orders)
                    ch[sign + pol] = _specular_channels(res)
        OPTIONS['unfloored'] = False
        d = dict(ss=abs(ch['+s']['s'] - ch['-s']['s']), pp=abs(ch['+p']['p'] - ch['-p']['p']),
                 sp=abs(ch['+s']['p'] - ch['-p']['s']), ps=abs(ch['+p']['s'] - ch['-s']['p']))
        out.append(dict(case='circle-offset metal', arm='scalar-unfloored' if unfloored else 'scalar',
                        abs_diff=d, max_rel=max(d.values()) / max(max(v.values()) for v in ch.values())))
        log(out[-1])
    save('reciprocity_unfloored', out)


# ----------------------------------------------------------------------------- 5 metal
def check_metal():
    out = []
    for M in (3, 5, 8, 10, 12):
        for pol in ('s', 'p'):
            for arm in ('scalar', 'scalar-unfloored', 'integrated'):
                OPTIONS['capture'] = True
                OPTIONS['unfloored'] = arm == 'scalar-unfloored'
                for v in CAPTURE.values():
                    v.clear()
                with C.arm('integrated' if arm == 'integrated' else 'scalar'):
                    res, calls = run(_case_problem('metallic', pol), M)
                OPTIONS['capture'] = False
                OPTIONS['unfloored'] = False
                r, t = RT(res)
                row = dict(case='metallic', arm=arm, orders=M, pol=pol, layer_calls=calls,
                           sumR=float(r.sum()), sumT=float(t.sum()),
                           absorption=float(1 - r.sum() - t.sum()),
                           finite=bool(np.isfinite(r).all() and np.isfinite(t).all()))
                if CAPTURE['slots']:
                    row.update(passivity(CAPTURE['slots'][-1]))
                if CAPTURE['floor']:
                    row['floor'] = CAPTURE['floor'][-1]
                if CAPTURE['B']:
                    row['B'] = CAPTURE['B'][-1]
                log(row)
                out.append(row)
        save('metal', out)


def check_metal_scan(M=5):
    out = []
    import dataclasses
    for wl in np.round(np.linspace(1.0, 1.6, 25), 6):
        for arm in ('scalar', 'scalar-unfloored', 'integrated'):
            OPTIONS['unfloored'] = arm == 'scalar-unfloored'
            p = C.problem('metallic')
            p = dataclasses.replace(p, excitation=dataclasses.replace(p.excitation,
                                                                     wavelength=float(wl)))
            with C.arm('integrated' if arm == 'integrated' else 'scalar'):
                res, calls = run(p, M)
            OPTIONS['unfloored'] = False
            r, t = RT(res)
            row = dict(wavelength=float(wl), arm=arm, orders=M, layer_calls=calls,
                       sumR=float(r.sum()), sumT=float(t.sum()), RT=float(r.sum() + t.sum()))
            log(row)
            out.append(row)
    save('metal_scan', out)


def check_metal_eps_scan(M=5):
    """R + T of a disk r=0.3 in air as Re(eps) sweeps -4..-40 (Im 0.5): the inverse-rule pole
    sits where kappa_j = 1/(1 - eps), so the sweep moves it across the truncated spectrum."""
    out = []
    for x in np.round(np.linspace(4., 40., 37), 6):
        eps = complex(-x, .5)
        prob = problem([R(rcwa.Circle(radius=.3), eps)], background=1.)
        for arm in ('scalar', 'scalar-unfloored', 'integrated'):
            OPTIONS['unfloored'] = arm == 'scalar-unfloored'
            with C.arm('integrated' if arm == 'integrated' else 'scalar'):
                res, calls = run(prob, M)
            OPTIONS['unfloored'] = False
            r, t = RT(res)
            row = dict(re_eps=-float(x), arm=arm, orders=M, RT=float(r.sum() + t.sum()),
                       finite=bool(np.isfinite(r).all() and np.isfinite(t).all()))
            log(row)
            out.append(row)
    save('metal_eps_scan', out)


# ----------------------------------------------------------------------------- 6 fem
FEM_LATTICE = np.array([[.7, .315], [0., .49]])
FEM_J = np.array([[1., .45], [0., .7]])
#: docs/validation/rcwa/nvf-independent-fem.md, "Vertex and degree refinement at degrees 6 and 7"
FEM_ROWS = {'E6': (.8521478711930495, .7307770528597978), 'E7': (.8522134546586737, .7308459794359807),
            'V6': (.8521985908019867, .7308750185489832), 'V7': (.8522509294793323, .7309173918139138)}
#: same page, "Comparing the physical diffraction orders with Wavebit": conforming affine-cell
#: NVF with a fitted degree-10 auxiliary field at optical order (24, 32) (not the shape builder).
CONFORMING_24_32 = (.852380710977, .731013218636)


def fem_problem(pol):
    def quad(x0, x1, y0, y1):
        return [tuple(float(c) for c in FEM_J @ np.array(p))
                for p in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    h = (.7 - .017) / 2
    regs = (R(rcwa.Polygon(vertices=quad(-.175, .031, -h, h)), 4.),
            R(rcwa.Polygon(vertices=quad(.031, .175, -h, h)), 9.))
    return rcwa.Problem(
        stack=rcwa.Stack(layers=(rcwa.Layer(thickness=.3, background=Material(1.), regions=regs),),
                         bottom=Material(2.25), lattice=Lattice.from_columns(FEM_LATTICE)),
        excitation=rcwa.Excitation(wavelength=.9, sources=(
            rcwa.PlaneWave(theta=.2, phi=0., polarization=pol),)))


def check_fem(orders_list=((4, 5), (8, 10), (12, 16), (16, 21))):
    # (18, 24) was started and stopped: its RSS would exceed the ~3 GB budget ((16, 21) peaked ~2.2 GB).
    out = []
    path = os.path.join(HERE, 'physics_fem.json')
    if os.path.exists(path):
        out = json.load(open(path))
    done = {(r['arm'], tuple(r['orders']), r['pol']) for r in out if 'arm' in r}
    for o in orders_list:
        for arm in C.ARMS:
            for pol in ('p', 's'):
                if (arm, tuple(o), pol) in done:
                    continue
                cpu = time.process_time()
                with C.arm(arm):
                    res, calls = run(fem_problem(pol), o)
                r, t = RT(res)
                ref = FEM_ROWS['V7'][0 if pol == 'p' else 1]
                row = dict(arm=arm, orders=list(o), pol=pol, layer_calls=calls,
                           sumT=float(t.sum()), sumR=float(r.sum()),
                           energy=float(1 - r.sum() - t.sum()), dT_vs_fem_V7=float(t.sum() - ref),
                           incidental_cpu_s=round(time.process_time() - cpu, 1))
                log(row)
                out.append(row)
                save('fem', out)


if __name__ == '__main__':
    globals()['check_' + sys.argv[1]](*[int(a) for a in sys.argv[2:]])
