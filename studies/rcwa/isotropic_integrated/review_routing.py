"""Adversarial review of the isotropic-integrated switch: which builder runs, per case and arm.

Counts, per (case, arm), the calls into each shape builder, the projector rule each one read
(``average`` of ``boundary.material_projector_fields``), the normal-field grid, the integrated
plan kind and quantity, and ``R + T``. Writes ``review_routing.json``.

    PYTHONPATH=src python studies/rcwa/isotropic_integrated/review_routing.py
"""
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import wavebit                                                             # noqa: E402
from wavebit.rcwa._engine import boundary, parametric, tensor_shapes      # noqa: E402
from studies.rcwa.isotropic_integrated import cases as C                   # noqa: E402

assert wavebit.__file__.startswith(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                                '..', '..', '..', 'src')))
LOG = collections.Counter()
DETAIL = collections.defaultdict(set)


def wrap(module, name, tag=None, record=None):
    original = getattr(module, name)

    def wrapped(*args, **kwargs):
        LOG[tag or name] += 1
        if record is not None:
            DETAIL[tag or name].add(record(*args, **kwargs))
        return original(*args, **kwargs)
    setattr(module, name, wrapped)


wrap(tensor_shapes, '_layer', record=lambda solver, layer, plan, quantity:
     (plan[0], len(plan), quantity, solver.shape_nvf_grid_size))
wrap(tensor_shapes, '_scalar_slots')
wrap(parametric, 'to_conv_mat_parametric_layers',
     record=lambda *a, **k: ('grid', k.get('shape_nvf_grid_size')))
wrap(parametric, 'to_conv_mat_parametric_nvf')
wrap(parametric, 'normal_vector_field_analytical')
wrap(boundary, 'to_conv_mat_boundary_layer')
wrap(boundary, 'material_projector_fields',
     record=lambda *a, **k: ('average', k.get('average', True)))
wrap(boundary, 'distance_gradient_normal_field')

GRID = {'overlap': 64, 'polygon': 128}


def main():
    out = {}
    for case in C.CASES:
        for arm in C.ARMS:
            LOG.clear()
            DETAIL.clear()
            with C.arm(arm):
                r, t = C.solve(case, 3, grid=GRID.get(case))
            out[f'{case}/{arm}'] = dict(
                calls=dict(LOG), detail={k: sorted(map(str, v)) for k, v in DETAIL.items()},
                sumR=float(r.sum()), sumT=float(t.sum()), RT=float(r.sum() + t.sum()))
            print(case, arm, dict(LOG), {k: sorted(map(str, v)) for k, v in DETAIL.items()},
                  f'R+T={r.sum() + t.sum():.12f}', flush=True)
    with open(os.path.join(os.path.dirname(__file__), 'review_routing.json'), 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
