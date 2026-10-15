"""How does ``factorization='laurent'`` converge against Li's rules and NVF, and what does it save?

Laurent's rule (the direct rule for every product, Moharam & Gaylord, JOSA 71, 811, 1981) is an
opt-in, offered so a result can be compared like for like with solvers that use it. L. Li (JOSA
A 13, 1870, 1996) showed why it converges slowly: where a field component jumps at an interface
together with the permittivity, the truncated product does not converge to the product. This
study records how that looks on the scenes this repository measures, for
``docs/theory/rcwa/factorization-terms.md`` §9:

  1d       a binary grating, 50% duty, eps 12.25 / 1, TE and TM: |T - T_ref| at orders 5..80,
           the reference Li at order 400. TE reads only the slot where the rules agree.
  2d       a 350 x 245 nm rod of eps 12.25 in a 700 nm square cell, drawn on the pixels of a
           40 x 40 raster so every rule sees the same staircase: Li, NVF and Laurent at orders
           2..14, the reference Li at order 18, s and p.
  circle   a 400 nm disc of eps 12.25 on the shape route, NVF against Laurent at orders 4..12.
           No reference here -- the route has no Li -- so only the distance between the two.
  cost     wall time of one solve at order 12, median of three, per rule: on the rod, and on
           the disc, where NVF samples its normal field on the default 1024 grid.

Scene: wavelength 900 nm, thickness 250 nm, top 1, bottom 2.25, theta 10 deg, phi 20 deg,
NumPy float64. Run from the repository root; prints the log committed as
``studies/rcwa/laurent_convergence/records/laurent_convergence.log``.
"""

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa
import time

import numpy as np


SCENE = dict(backend='numpy', top=1.0, bot=2.25, wavelength=900.0, thickness=[250.0])
ANGLES = dict(theta=np.deg2rad(10.0), phi=np.deg2rad(20.0), psi=0.0)


def one_dimensional():
    eps = np.ones((1, 1, 40))
    eps[0, 0, :20] = 12.25
    kw = dict(SCENE, theta=np.deg2rad(10.0), phi=0.0, period=[700.0], structure=eps)

    def T(rule, order, tetm):
        result = _legacy_wavebit_rcwa(**kw, truncation_order=(order, 0), tetm=tetm,
                              factorization=rule).solve()
        return float(np.sum(result.T))

    print('1d grating, 50% duty, eps 12.25 / 1, 700 nm period: |T - T(li, 400)|')
    for tetm in ('te', 'tm'):
        reference = T('li', 400, tetm)
        print(f'  {tetm}: T_ref = {reference:.12f}')
        print('    order        li      laurent')
        for order in (5, 10, 20, 40, 80):
            print(f'    {order:5d}  {abs(T("li", order, tetm) - reference):9.2e}  '
                  f'{abs(T("laurent", order, tetm) - reference):9.2e}')


def _rod():
    eps = np.ones((1, 40, 40))
    eps[0, 13:27, 10:30] = 12.25          # 20 x 14 pixels of 17.5 nm: 350 x 245 nm
    return eps


def _totals(result):
    return np.array([float(np.sum(result.s_inc.T)), float(np.sum(result.p_inc.T))])


def two_dimensional():
    kw = dict(SCENE, **ANGLES, period=[700.0, 700.0], structure=_rod())
    energy = {}

    def T(rule, order):
        result = _legacy_wavebit_rcwa(**kw, truncation_order=(order, order), factorization=rule).solve()
        for sub in (result.s_inc, result.p_inc):
            miss = abs(float(np.sum(sub.R) + np.sum(sub.T)) - 1.0)
            energy[rule] = max(energy.get(rule, 0.0), miss)
        return _totals(result)

    reference = T('li', 18)
    print('\n2d rod, 350 x 245 nm, eps 12.25 in air, 700 nm cell: |T - T(li, 18)| per polarization')
    print(f'  T_ref (s, p) = {reference[0]:.10f}, {reference[1]:.10f}')
    print('    order        li (s, p)              nvf (s, p)            laurent (s, p)')
    for order in (2, 4, 6, 8, 10, 12, 14):
        row = [np.abs(T(rule, order) - reference) for rule in ('li', 'nvf', 'laurent')]
        print(f'    {order:5d}  ' + '  '.join(f'{e[0]:9.2e} {e[1]:9.2e}' for e in row))
    print('  max |R + T - 1| over these solves: ' +
          ', '.join(f'{rule} {miss:.1e}' for rule, miss in energy.items()))


def circle():
    disc = [dict(background=1.0, shapes=[dict(type='circle', cx=0.0, cy=0.0, lx=400.0,
                                              material=12.25)])]
    kw = dict(SCENE, **ANGLES, period=[700.0, 700.0], structure=disc)
    print('\ncircle, 400 nm, eps 12.25 in air, shape route: T per polarization')
    print('    order       nvf (s, p)              laurent (s, p)          laurent - nvf')
    for order in (4, 6, 8, 10, 12):
        nvf = _totals(_legacy_wavebit_rcwa(**kw, truncation_order=(order, order)).solve())
        laurent = _totals(_legacy_wavebit_rcwa(**kw, truncation_order=(order, order),
                                       factorization='laurent').solve())
        print(f'    {order:5d}  {nvf[0]:.6f} {nvf[1]:.6f}     {laurent[0]:.6f} {laurent[1]:.6f}'
              f'     {laurent[0] - nvf[0]:+.2e} {laurent[1] - nvf[1]:+.2e}')


def cost():
    disc = [dict(background=1.0, shapes=[dict(type='circle', cx=0.0, cy=0.0, lx=400.0,
                                              material=12.25)])]
    print('\none solve at order 12, median of three (s)')
    for name, structure, rules in (('rod', _rod(), ('li', 'nvf', 'laurent')),
                                   ('disc', disc, ('nvf', 'laurent'))):
        kw = dict(SCENE, **ANGLES, period=[700.0, 700.0], structure=structure,
                  truncation_order=(12, 12))
        for rule in rules:
            times = []
            for _ in range(3):
                start = time.perf_counter()
                _legacy_wavebit_rcwa(**kw, factorization=rule).solve()
                times.append(time.perf_counter() - start)
            print(f'    {name:5s} {rule:8s} {np.median(times):.3f}')


if __name__ == '__main__':
    one_dimensional()
    two_dimensional()
    circle()
    cost()
