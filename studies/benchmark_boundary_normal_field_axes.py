"""Which axis flips the verdict on the normal-field swap?

Two measurements disagreed. They differed on four things at once, so neither could say what
mattered. This varies three of them one at a time over a single fixed base -- same shapes, same
period, same thickness, same normal grid -- and reads both metrics:

    ratio = int(g) / (L(b) - L(a))                        the landed script's read-out
    err   = |int(g) - (L(b) - L(a))| / (max L - min L)    normalised by the loss's own variation

err near 0 means the autodiff gradient accounts for the loss change; >= 1 means it accounts for
none of it. Three window positions per configuration, so one placement cannot decide.

Rectangles only. Circles are the expensive case (a 256-gon puts 512 segments in the distance
scan) and the least informative -- a circle's normal is radial and smooth in the centre, which is
the one case where the sampled field is known to be right.
"""
import os, itertools, sys
os.environ.setdefault('JAX_ENABLE_X64', '1')
sys.path.insert(0, os.path.join(os.getcwd(), 'studies'))

import numpy as np, jax, jax.numpy as jnp
import wavebit
from wavebit._core import analytic as _analytic
from wavebit._core import boundary as _boundary
from wavebit._core.geometry import StaticGeometry

WL = 0.777
P = 2.0 * WL
NGRID, HALF_PX, NPTS = 512, 6.0, 25
PIX = P / NGRID
CX0 = 0.28 * P

# Materials are **relative permittivity**. The contrast axis keeps naming its media by
# refractive index, because that is what the printed row labels say and what the two earlier
# measurements were posed in; `eps_from_index` converts where the shape is built.
EPS = wavebit.eps_from_index
BASE = dict(period=[P, P], thickness=[0.25], psi=0.0, wavelength=WL, top=1.0, bot=EPS(1.5))
AXES = dict(contrast={'high': (3.0, 2.2), 'low': (1.9, 1.6)},
            incidence={'oblique': (0.35, 0.4), 'normal': (0.0, 0.0)},
            readout={'R': 'R', 'T': 'T'})


def build(cx, na, nb):
    def rect(cx_, cy, lx, ly, ang, n):
        return dict(type='rectangle', cx=cx_ * P, cy=cy * P, lx=lx * P, ly=ly * P,
                    angle=ang, material=EPS(n))
    first = rect(0.28, 0.30, 0.20, 0.15, 0.4, na)
    first['cx'] = cx
    return [dict(background=1.0,
                 shapes=[first, rect(0.68, 0.70, 0.20, 0.15, 0.9, nb)])]


def geos(na, nb):
    structure = build(CX0, na, nb)
    return (StaticGeometry(boundary_plans=[None]),
            StaticGeometry(boundary_plans=[
                _boundary.plan_layer(structure[0]['shapes'], (P, P))]))


def scalar(geometry, cx, na, nb, theta, phi, key, backend):
    s = wavebit.rcwa(backend=backend, structure=build(cx, na, nb), truncation_order=[8, 8],
                     normal_grid=NGRID, theta=theta, phi=phi, **BASE)
    r = getattr(s.solve(geometry=geometry), key)
    return jnp.real(jnp.sum(r)) if backend == 1 else float(np.real(np.sum(np.asarray(r))))


print(f'base: period {P:.4f}  wl {WL}  thickness 0.25  grid {NGRID}  '
      f'window +-{HALF_PX:.0f} px  {NPTS} pts  truncation_order 8')
print(f'{"contrast":>9} {"incidence":>10} {"read":>5} {"cx0/P":>7} | '
      f'{"sampled err":>11} {"swapped err":>11} | {"s ratio":>8} {"w ratio":>8} | {"spread":>9}')

for cname, inc, rname in itertools.product(AXES['contrast'], AXES['incidence'],
                                           AXES['readout']):
    na, nb = AXES['contrast'][cname]
    theta, phi = AXES['incidence'][inc]
    key = AXES['readout'][rname]
    g_sampled, g_swapped = geos(na, nb)
    for frac in (-0.015, 0.0, 0.015):
        c0 = CX0 + frac * P
        xs = c0 + np.linspace(-HALF_PX * PIX, HALF_PX * PIX, NPTS)
        cells = []
        for geometry in (g_sampled, g_swapped):
            g = np.array([float(jax.grad(
                lambda c: scalar(geometry, c, na, nb, theta, phi, key, 1))(jnp.asarray(x)))
                for x in xs])
            v = np.array([scalar(geometry, x, na, nb, theta, phi, key, 0) for x in xs])
            integ, d, spread = np.trapezoid(g, xs), v[-1] - v[0], v.max() - v.min()
            cells.append((abs(integ - d) / spread, integ / d, spread))
        print(f'{cname:>9} {inc:>10} {rname:>5} {c0/P:>7.3f} | '
              f'{cells[0][0]:>11.3f} {cells[1][0]:>11.3f} | '
              f'{cells[0][1]:>8.2f} {cells[1][1]:>8.2f} | {cells[0][2]:>9.2e}', flush=True)


# --------------------------------------------------------------------------------------------
# Second half: which matrix block carries the error.
#
# The normal field does not know the incidence angle, and `dC/dcx` is geometry only -- identical
# at every angle. So an error in it that shows at oblique incidence and hides at normal incidence
# has to sit in a block the oblique loss weighs and the normal one does not, which points at
# epxy / epyx: the off-diagonal coupling NVF adds over Li, and the one normal incidence barely
# exercises. Run the same windowed check on each block on its own to see it.

WL = 0.777
P = 2.0 * WL
NGRID, HALF_PX, NPTS, MAX_ORDER = 512, 6.0, 25, 8
PIX, CX0 = P / NGRID, 0.28 * P
NAMES = ('epx', 'epy', 'epz_i', 'epxy', 'epyx')


PLAN = _boundary.plan_layer(build(CX0, 3.0, 2.2)[0]['shapes'], (P, P))


def probe(na, nb, plans, block):
    def loss(cx):
        m = _analytic.to_conv_mat_smooth_layers(
            build(cx, na, nb), MAX_ORDER, MAX_ORDER, (P, P), normal_grid=NGRID,
            complex_dtype=jnp.complex128, xp=jnp, boundary_plans=plans)[block]
        return jnp.real(jnp.sum(m * jnp.conj(m)))
    f = jax.jit(jax.value_and_grad(loss))
    xs = CX0 + np.linspace(-HALF_PX * PIX, HALF_PX * PIX, NPTS)
    vg = [f(x) for x in xs]
    v = np.array([float(a) for a, _ in vg]); g = np.array([float(b) for _, b in vg])
    spread = v.max() - v.min()
    if spread == 0:
        return float('nan')
    return abs(np.trapezoid(g, xs) - (v[-1] - v[0])) / spread


for cname, (na, nb) in (('high 3.0/2.2', (3.0, 2.2)), ('low 1.9/1.6', (1.9, 1.6))):
    print(f'\n--- {cname}   (err per matrix block, 0 = gradient accounts for the change)')
    print(f'{"":>18} ' + ' '.join(f'{n:>10}' for n in NAMES))
    for label, plans in (('sampled', [None]), ('grad-of-distance', [PLAN])):
        row = [probe(na, nb, plans, b) for b in range(5)]
        print(f'{label:>18} ' + ' '.join(f'{v:>10.3f}' for v in row), flush=True)
