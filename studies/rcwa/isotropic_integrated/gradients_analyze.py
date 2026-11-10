"""Summarize gradients.json: AD vs FD per arm/case, arm-vs-arm gradients, sweep smoothness.

    python studies/rcwa/isotropic_integrated/gradients_analyze.py   (writes gradients.txt)
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, 'gradients.json')))
out = []
p = out.append


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


p('## AD (jax.grad, x64) vs NumPy central difference, default grid')
p('| case | orders | arm | sum T | AD | FD h=1e-5 | FD h=1e-6 | rel(AD,FD5) | rel(AD,FD6) | rel(FD5,FD6) |')
p('|---|---|---|---|---|---|---|---|---|---|')
for key, r in data['grad'].items():
    case, o, arm = key.split('|')
    if 'ad_jax' not in r:
        p(f'| {case} | {o} | {arm} | {r["T_np"]:.8f} | ERROR {r.get("jax_error","")[:60]} | '
          f'{r["fd_1e-05"]:.8f} | {r["fd_1e-06"]:.8f} | - | - | {rel(r["fd_1e-06"], r["fd_1e-05"]):.1e} |')
        continue
    p(f'| {case} | {o} | {arm} | {r["T_np"]:.8f} | {r["ad_jax"]:.8f} | {r["fd_1e-05"]:.8f} | '
      f'{r["fd_1e-06"]:.8f} | {rel(r["ad_jax"], r["fd_1e-05"]):.1e} | {rel(r["ad_jax"], r["fd_1e-06"]):.1e} | '
      f'{rel(r["fd_1e-06"], r["fd_1e-05"]):.1e} |')

p('\n## Arm vs arm (same orders): gradient and value differences')
p('| case | orders | AD scalar | AD integrated | dAD | rel dAD | dT (int-scal) | |T_jax-T_np| max over arms |')
p('|---|---|---|---|---|---|---|---|')
seen = []
for key in data['grad']:
    c, o, _ = key.split('|')
    if (c, o) in seen:
        continue
    seen.append((c, o))
    s, i = data['grad'].get(f'{c}|{o}|scalar'), data['grad'].get(f'{c}|{o}|integrated')
    if not s or not i or 'ad_jax' not in s or 'ad_jax' not in i:
        continue
    d = i['ad_jax'] - s['ad_jax']
    tj = max(abs(s['T_jax'] - s['T_np']), abs(i['T_jax'] - i['T_np']))
    p(f'| {c} | {o} | {s["ad_jax"]:.6f} | {i["ad_jax"]:.6f} | {d:+.2e} | {abs(d)/abs(s["ad_jax"]):.1e} | '
      f'{i["T_np"]-s["T_np"]:+.2e} | {tj:.1e} |')

p('\n## Order convergence of the gradient (AD, (6,6) vs (3,3))')
p('| case | arm | AD(3) | AD(6) | |AD6-AD3| |')
p('|---|---|---|---|---|')
for c in dict.fromkeys(c for c, _ in seen):
    for arm in ('scalar', 'integrated'):
        a, b = data['grad'].get(f'{c}|3|{arm}'), data['grad'].get(f'{c}|6|{arm}')
        if a and b and 'ad_jax' in a and 'ad_jax' in b:
            p(f'| {c} | {arm} | {a["ad_jax"]:.6f} | {b["ad_jax"]:.6f} | {abs(b["ad_jax"]-a["ad_jax"]):.2e} |')

p('\n## PyTorch backward, circle')
p('| orders | arm | AD torch | AD jax | FD 1e-5 | rel(torch,jax) | rel(torch,FD5) |')
p('|---|---|---|---|---|---|---|')
for key, r in data['torch'].items():
    c, o, arm = key.split('|')
    g = data['grad'].get(key, {})
    if 'ad_torch' in r and 'ad_jax' in g:
        p(f'| {o} | {arm} | {r["ad_torch"]:.10f} | {g["ad_jax"]:.10f} | {g["fd_1e-05"]:.10f} | '
          f'{rel(r["ad_torch"], g["ad_jax"]):.1e} | {rel(r["ad_torch"], g["fd_1e-05"]):.1e} |')
    else:
        p(f'| {o} | {arm} | {r} | | | | |')

p('\n## Sweeps, orders (4,4), NumPy, 30 points, default grid')
p('d2 = T[i+1]-2T[i]+T[i-1] (uniform step dx). fd = local central difference h=1e-5 at each point. '
  'noise(fd) = fd - derivative of a cubic spline through the 30 T values, over interior points '
  '[2:-2]: this is where a step dT in T(x) at the 1e-5 scale shows (fd error = dT/2h). '
  'polyres(T) = max residual of a degree-9 Chebyshev least-squares fit to T (30-point-scale wiggles). '
  'arm-vs-arm columns: integrated minus scalar, all points.')
p('| case | arm | range | max|d2|/dx^2 | rms noise(fd) | max noise(fd) at x | rel rms noise | polyres(T) | '
  'max|fd_int-fd_scal| | rms | rel rms | max|T_int-T_scal| |')
p('|---|---|---|---|---|---|---|---|---|---|---|---|')
from numpy.polynomial import chebyshev as cheb
from scipy.interpolate import CubicSpline
for key, r in data['sweep'].items():
    c, arm = key.split('|')
    x, T, G = map(np.asarray, (r['x'], r['T'], r['fd']))
    dx = x[1] - x[0]
    d2 = T[2:] - 2 * T[1:-1] + T[:-2]
    nz = (G - CubicSpline(x, T)(x, 1))[2:-2]
    k = int(np.argmax(abs(nz)))
    fit = cheb.chebfit(x, T, 9)
    polyres = np.max(abs(cheb.chebval(x, fit) - T))
    other = data['sweep'].get(f'{c}|scalar')
    if arm == 'integrated' and other:
        dg = G - np.asarray(other['fd'])
        dT = T - np.asarray(other['T'])
        cmp_ = (f'{np.max(abs(dg)):.2e} | {np.sqrt(np.mean(dg**2)):.2e} | '
                f'{np.sqrt(np.mean(dg**2))/np.sqrt(np.mean(np.asarray(other["fd"])**2)):.1e} | {np.max(abs(dT)):.2e}')
    else:
        cmp_ = '- | - | - | -'
    p(f'| {c} | {arm} | {x[0]:.2f}..{x[-1]:.2f} | {np.max(abs(d2))/dx**2:.3f} | '
      f'{np.sqrt(np.mean(nz**2)):.1e} | {abs(nz[k]):.1e} at {x[k+2]:.3f} | '
      f'{np.sqrt(np.mean(nz**2))/np.sqrt(np.mean(G**2)):.1e} | {polyres:.1e} | {cmp_} |')


for name, title in (('gradients_polyscan.json', 'polygon fine scan, orders (3,3), default grid, step 5e-6'),
                    ('gradients_polygrid.json', 'polygon fine scan, integrated arm, orders (3,3), step 1e-5, by grid')):
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        continue
    p(f'\n## {title}')
    p('| arm/grid | AD (if run) | local slopes dT/dx between consecutive points | avg slope over scan |')
    p('|---|---|---|---|')
    for k, r in json.load(open(path)).items():
        x, T = np.asarray(r['x']), np.asarray(r['T'])
        sl = np.diff(T) / np.diff(x)
        p(f'| {k} | {r.get("ad_jax", "-")} | {np.array2string(np.round(sl, 4), max_line_width=400)} | '
          f'{(T[-1]-T[0])/(x[-1]-x[0]):.5f} |')
path = os.path.join(HERE, 'gradients_metalscan.json')
if os.path.exists(path):
    p('\n## metallic fine scan 0.29..0.31, 41 points (dx 5e-4), orders (4,4), default grid')
    p('| arm | T min..max | max |fd| | max|d2|/dx^2 | rms noise(fd) | max noise(fd) |')
    p('|---|---|---|---|---|---|')
    for k, r in json.load(open(path)).items():
        x, T, G = map(np.asarray, (r['x'], r['T'], r['fd']))
        dx = x[1] - x[0]
        d2 = T[2:] - 2 * T[1:-1] + T[:-2]
        nz = (G - CubicSpline(x, T)(x, 1))[2:-2]
        p(f'| {k} | {T.min():.4f}..{T.max():.4f} | {abs(G).max():.2f} | {abs(d2).max()/dx**2:.0f} | '
          f'{np.sqrt(np.mean(nz**2)):.3f} | {abs(nz).max():.3f} |')

txt = '\n'.join(out)
open(os.path.join(HERE, 'gradients.txt'), 'w').write(txt + '\n')
print(txt)
