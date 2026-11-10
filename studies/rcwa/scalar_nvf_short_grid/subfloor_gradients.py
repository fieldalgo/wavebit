"""The sub-floor gradient outliers of section (c), with PyTorch backward beside eager jax.grad.

    python subfloor_gradients.py      # writes results/subfloor_gradients.json

Section (c) of docs/theory/rcwa/scalar-nvf-short-grid-study.md found AD-versus-central-difference
gaps below the floor on the hexagonal cell (grids 3 and 12) and the pentagon (grid 12). Today's
code refuses those grids, so they can only be run with the prototype on. If PyTorch backward
agrees with the central difference where jax.grad does not, the gap is the JAX one recorded by
lattice_gradients.py, not the prototype's.
"""
import json
import pathlib

from cases import arm
from lattice_gradients import grads

OUT = pathlib.Path(__file__).with_name('results')

rows = []
with arm(True):
    for case, grid in [('hexagonal', 3), ('hexagonal', 12), ('polygon', 12), ('hexagonal', 11), ('polygon', 11)]:
        rows.append(grads(case, grid))
        print(rows[-1], flush=True)
(OUT / 'subfloor_gradients.json').write_text(json.dumps(rows, indent=1, default=str))
