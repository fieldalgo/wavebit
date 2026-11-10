"""Tables for the physics/exactness checks (reads ``physics_*.json``, writes ``physics.txt``)."""
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    path = os.path.join(HERE, f'physics_{name}.json')
    return json.load(open(path)) if os.path.exists(path) else None


def e(x):
    return f'{x:.2e}' if isinstance(x, float) else str(x)


lines = []
P = lines.append

d = load('uniform')
if d:
    P('## 1 uniform layer split by regions of its own material, orders (5,5): max(|dR|,|dT|) vs unpatterned')
    P('| case | background | region eps | scalar | integrated | _layer calls s/i |')
    P('|---|---|---|---:|---:|---|')
    g = defaultdict(dict)
    for r in d:
        g[(r['case'], r['background'], r['rel_perturbation'])][r['arm']] = r
    for (c, b, rel), v in g.items():
        s, i = v['scalar'], v['integrated']
        P(f"| {c} | {b} | {'bg' if rel == 0 else 'bg*(1+1e-12)'} | {e(max(s['dR'], s['dT']))} | "
          f"{e(max(i['dR'], i['dT']))} | {s['layer_calls']}/{i['layer_calls']} |")
    P('')

d = load('lamellar')
if d:
    P('## 2 1D lamellar (rectangle spanning the cell): max per-order |d eff| vs Li at the same orders;'
      ' sum T - Li(400) 1D')
    P('| case | orders | scalar vs Li | integrated vs Li | scalar dT(Li400) | integrated dT(Li400) |'
      ' Li dT(Li400) |')
    P('|---|---|---:|---:|---:|---:|---:|')
    g = defaultdict(dict)
    for r in d:
        if 'arm' in r:
            g[(r['case'], tuple(r['orders']))][r['arm']] = r
    for (c, o), v in g.items():
        s, i = v.get('scalar', {}), v.get('integrated', {})
        def m(x):
            return e(max(x['max_dR_vs_li'], x['max_dT_vs_li'])) if 'max_dR_vs_li' in x else x.get('error', '-')
        P(f"| {c} | {o} | {m(s)} | {m(i)} | {e(s.get('dT_vs_li400', '-'))} | "
          f"{e(i.get('dT_vs_li400', '-'))} | {e(s.get('li_dT_vs_li400', '-'))} |")
    P('')

d = load('lamellar_floor')
if d:
    P('## 2b lamellar metal: scalar arm with and without the inverse-rule floor, max per-order |d eff| vs Li')
    P('| orders | scalar (floored) | scalar unfloored | floor: below/n, smin/floor |')
    P('|---|---:|---:|---|')
    g = defaultdict(dict)
    for r in d:
        g[tuple(r['orders'])][r['arm']] = r
    for o, v in g.items():
        f = v['scalar']['floor']
        P(f"| {o} | {e(v['scalar']['max_d_vs_li'])} | {e(v['scalar-unfloored']['max_d_vs_li'])} | "
          f"{f['below']}/{f['n']}, {f['smin']:.2e}/{f['floor']:.2e} |")
    P('')

d = load('mirror')
if d:
    P('## 3 mirror x -> -x with phi -> pi - phi, orders (5,5): max per-order |d eff| (R, T)')
    P('| case | theta | pol | scalar R / T | integrated R / T |')
    P('|---|---|---|---|---|')
    g = defaultdict(dict)
    for r in d:
        g[(r['case'], r['theta'], r['pol'])][r['arm']] = r
    for (c, th, pol), v in g.items():
        s, i = v['scalar'], v['integrated']
        P(f"| {c} | {th} | {pol} | {e(s['max_dR'])} / {e(s['max_dT'])} | "
          f"{e(i['max_dR'])} / {e(i['max_dT'])} |")
    P('')

d = load('energy')
if d:
    P('## 4 energy: 1 - R - T (lossless: should be 0; lossy: absorption, should be > 0);'
      ' antiHerm = ||E - E^H|| / ||E|| and min eig of (E - E^H)/2i of the in-plane tensor (s rows)')
    P('| case | M | pol | scalar 1-R-T | integrated 1-R-T | scalar antiHerm, min eig |'
      ' integrated antiHerm, min eig |')
    P('|---|---|---|---:|---:|---|---|')
    g = defaultdict(dict)
    for r in d:
        g[(r['case'], r['orders'], r['pol'])][r['arm']] = r
    for (c, M, pol), v in g.items():
        s, i = v['scalar'], v['integrated']
        def ah(x):
            return f"{e(x['antiherm_rel'])}, {x['antiherm_min_eig']:.2e}" if 'antiherm_rel' in x else ''
        P(f"| {c} | {M} | {pol} | {s['one_minus_RT']:.2e} | {i['one_minus_RT']:.2e} | {ah(s)} | {ah(i)} |")
    P('')

d = load('mirror_overlap')
if d:
    P('## 3b painted overlap mirror defect vs projector grid, orders (5,5), s: max per-order |d eff| (R / T)')
    P('| grid | scalar | integrated |')
    P('|---|---|---|')
    g = defaultdict(dict)
    for r in d:
        g[r['grid']][r['arm']] = r
    for gr, v in g.items():
        def m(x):
            return f"{e(x['max_dR'])} / {e(x['max_dT'])}" if x and 'max_dR' in x else '-'
        P(f"| {gr} | {m(v.get('scalar'))} | {m(v.get('integrated'))} |")
    P('')

d = load('reciprocity')
if d:
    P('## 4b reciprocity of specular reflection, R_ab(k) vs R_ba(-k), orders (5,5)')
    P('| case | arm | ss | pp | sp/ps | ps/sp | max rel |')
    P('|---|---|---:|---:|---:|---:|---:|')
    for r in d:
        a = r['abs_diff']
        P(f"| {r['case']} | {r['arm']} | {e(a['ss'])} | {e(a['pp'])} | {e(a['sp'])} | {e(a['ps'])} | "
          f"{e(r['max_rel'])} |")
    P('')

d = load('reciprocity_unfloored')
if d:
    for r in d:
        P(f"reciprocity, circle-offset metal, {r['arm']}: max rel {e(r['max_rel'])}")
    P('')

d = load('metal')
if d:
    P('## 5 metallic disk eps -20+1.5j in air (default grid)')
    P('| M | pol | arm | 1-R-T | antiHerm min eig | floor binds (below/n, smin/floor) | [[B]] cond, smin |')
    P('|---|---|---|---:|---:|---|---|')
    for r in d:
        f = r.get('floor')
        fl = f"{f['below']}/{f['n']}, {f['smin']:.2e}/{f['floor']:.2e}" if f else '-'
        b = r.get('B')
        bb = f"{b['cond']:.2e}, {b['smin']:.2e}" if b else '-'
        P(f"| {r['orders']} | {r['pol']} | {r['arm']} | {r['absorption']:.3e} | "
          f"{r.get('antiherm_min_eig', float('nan')):.2e} | {fl} | {bb} |")
    P('')

d = load('metal_scan')
if d:
    P('## 5b metallic disk, R+T over wavelength 1.0..1.6 (25 points), orders (5,5), s')
    g = defaultdict(list)
    for r in d:
        g[r['arm']].append(r)
    P('| arm | max R+T | points with R+T > 1 | min R+T |')
    P('|---|---:|---:|---:|')
    for a, rows in g.items():
        v = [r['RT'] for r in rows]
        P(f"| {a} | {max(v):.5f} | {sum(x > 1 for x in v)} | {min(v):.5f} |")
    P('')

d = load('metal_eps_scan')
if d:
    P('## 5c disk r=.3 in air, eps = -x + 0.5j, x = 4..40 (37 points), orders (5,5), s: R + T')
    g = defaultdict(list)
    for r in d:
        g[r['arm']].append(r)
    P('| arm | max R+T (at Re eps) | points with R+T > 1 | all finite |')
    P('|---|---|---:|---|')
    for a, rows in g.items():
        top = max(rows, key=lambda r: r['RT'])
        P(f"| {a} | {top['RT']:.4f} ({top['re_eps']}) | {sum(r['RT'] > 1 for r in rows)} | "
          f"{all(r['finite'] for r in rows)} |")
    P('')

d = load('fem')
if d:
    P('## 6 independent 3D FEM (NGSolve, two-piece affine profile, V7 row): T - T_FEM')
    P('FEM V7 p .8522509 s .7309174; FEM spread (E6..V7) ~1e-4. Conforming affine NVF (24,32): '
      'p .8523807 s .7310132')
    P('| orders | pol | scalar T | scalar dT | scalar 1-R-T | integrated T | integrated dT | integrated 1-R-T |')
    P('|---|---|---:|---:|---:|---:|---:|---:|')
    g = defaultdict(dict)
    for r in d:
        g[(tuple(r['orders']), r['pol'])][r['arm']] = r
    for (o, pol), v in g.items():
        s, i = v.get('scalar'), v.get('integrated')
        def c(x):
            return (f"{x['sumT']:.6f} | {x['dT_vs_fem_V7']:+.2e} | {x['energy']:+.1e}" if x else '- | - | -')
        P(f"| {o} | {pol} | {c(s)} | {c(i)} |")
    P('')

open(os.path.join(HERE, 'physics.txt'), 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
