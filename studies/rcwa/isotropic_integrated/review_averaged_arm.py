"""Adversarial review: the integrated arm with the scalar route's averaged projector.

The integrated arm reads ``boundary.material_projector_fields(..., average=False)`` (hard nearest
choices) on polygon and multi-shape layers, the scalar arm the averaged field. This patches the
integrated projector call to ``average=True`` (study-side only, no src edit) and rescans the
cases whose T(x) was stepped. Writes ``review_averaged_arm.json``.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from wavebit.rcwa._engine import boundary, tensor_shapes                   # noqa: E402
from studies.rcwa.isotropic_integrated import cases as C                   # noqa: E402
from studies.rcwa.isotropic_integrated import review_twoshape_scan as S    # noqa: E402,F401

_mpf = boundary.material_projector_fields
FORCE = [False]


class _Proxy:
    def __getattr__(self, name):
        if name == 'material_projector_fields' and FORCE[0]:
            return lambda *a, **k: _mpf(*a, **{**k, 'average': True})
        return getattr(boundary, name)


tensor_shapes.boundary = _Proxy()
from wavebit import rcwa                                                   # noqa: E402

out = {}
for force in (False, True):
    FORCE[0] = force
    with C.arm('integrated'):
        for label, f in (
                ('polygon g128 M2', lambda k: C.solve('polygon', 2, grid=128,
                                                      x=C.CASES['polygon'][1] + k * 1e-5)[1].sum()),
                ('overlap g48 M2', lambda k: C.solve('overlap', 2, grid=48, x=.25 + k * 1e-5)[1].sum()),
                ('circle+rect g256 M3', lambda k: np.sum(rcwa.run(S.problem(.2 + k * 1e-5, 'rectangle'),
                    numerics=rcwa.Numerics(orders=(3, 3), factorization='nvf',
                                           shape_nvf_grid_size=256)).T))):
            t = [float(f(k)) for k in range(-5, 6)]
            d = np.diff(t)
            out[f'{label}/average={force}'] = dict(T=t, median=float(np.median(d)),
                                                  max_dev=float(np.max(abs(d - np.median(d)))))
            print(label, 'average', force, f'median step {np.median(d):.3e} max dev '
                  f'{np.max(abs(d - np.median(d))):.3e}', flush=True)
with open(os.path.join(os.path.dirname(__file__), 'review_averaged_arm.json'), 'w') as fh:
    json.dump(out, fh, indent=1)
