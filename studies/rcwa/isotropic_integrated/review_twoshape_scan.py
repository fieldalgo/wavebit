"""Adversarial review: T(c) of a disjoint two-shape layer (circle + ellipse) under each arm.

The integrated arm builds a multi-shape layer on the interface plan with hard nearest choices
(``average=False``); this scans whether T moves smoothly as one shape slides. Writes
``review_twoshape_scan.json``.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from wavebit import Material, rcwa                                         # noqa: E402
from wavebit.rcwa._engine import tensor_shapes                             # noqa: E402
from studies.rcwa.isotropic_integrated import cases as C                   # noqa: E402

PLANS = []
_layer = tensor_shapes._layer
tensor_shapes._layer = lambda s, l, p, q: (PLANS.append((p[0], len(p))), _layer(s, l, p, q))[1]


def problem(c, second):
    g = (rcwa.Ellipse(radii=(.15, .1), angle=.3, center=(c, .1)) if second == 'ellipse'
         else rcwa.Rectangle(size=(.2, .15), angle=.3, center=(c, .1)))
    return rcwa.Problem(
        stack=rcwa.Stack(layers=(rcwa.Layer(thickness=C.THICKNESS, background=Material(1.5),
            regions=(rcwa.Region(geometry=rcwa.Circle(radius=.2, center=(-.15, 0.)),
                                 material=Material(4.)),
                     rcwa.Region(geometry=g, material=Material(6.)))),),
            bottom=Material(C.SUBSTRATE), period=(1., 1.)),
        excitation=rcwa.Excitation(wavelength=C.WAVELENGTH, sources=(
            rcwa.PlaneWave(theta=C.THETA, phi=C.PHI, polarization='s'),)))


out = {}
for second in ('ellipse', 'rectangle'):
    for arm in C.ARMS:
        with C.arm(arm):
            PLANS.clear()
            t = [float(np.sum(rcwa.run(problem(.2 + k * 1e-5, second), numerics=rcwa.Numerics(
                orders=(3, 3), factorization='nvf', shape_nvf_grid_size=256)).T))
                 for k in range(-6, 7)]
            d = np.diff(t)
            out[f'{second}/{arm}'] = dict(T=t, plans=sorted(set(map(str, PLANS))),
                                          step_median=float(np.median(d)),
                                          step_max_dev=float(np.max(abs(d - np.median(d)))))
            print(second, arm, out[f'{second}/{arm}']['plans'],
                  f'median step {np.median(d):.3e}  max deviation {np.max(abs(d - np.median(d))):.3e}',
                  flush=True)
with open(os.path.join(os.path.dirname(__file__), 'review_twoshape_scan.json'), 'w') as f:
    json.dump(out, f, indent=1)
