"""Optional conforming 2D FEM reference for the partitioned gap auxiliary field.

This is a static scalar cell solve, not an independent 3D Maxwell solution.
Material faces are fitted independently of the candidate's fixed flux mesh.
"""
import argparse
import json
import time

import ngsolve as ng
import netgen.occ as occ
import numpy as np
from netgen.meshing import IdentificationType

from studies.nvf_cell_corrector import PERIOD, project_mesh
from studies.nvf_fem_reference import periodic_counts


def make_profile_mesh(h=.06, gap=17., width=350., seams=(31.,), values=(4., 9.), refine=20.):
    edges = np.r_[-width/2, seams, width/2]/1000
    height = PERIOD-gap/1000
    if (len(values) != len(edges)-1 or not np.isfinite(np.r_[edges, values, h, gap, refine]).all()
            or np.min(values) <= 0 or not np.all(np.diff(edges) > 0)
            or not 0 < width < 1000*PERIOD or not 0 < height <= PERIOD
            or not 0 < h < PERIOD/2 or refine < 1):
        raise ValueError('invalid FEM material profile or mesh settings')
    cell = occ.WorkPlane().MoveTo(-PERIOD/2, -PERIOD/2).Rectangle(PERIOD, PERIOD).Face()
    parts, materials, expected = [], {}, {}
    remaining = cell
    for i, (left, right, value) in enumerate(zip(edges[:-1], edges[1:], values)):
        part = occ.WorkPlane().MoveTo(left, -height/2).Rectangle(right-left, height).Face()
        name = f'piece_{i}'
        part.faces.name = name
        materials[name] = value
        expected[name] = (right-left)*height
        parts.append(part)
        remaining = remaining-part
    remaining.faces.name = 'background'
    shape = occ.Glue(parts+[remaining])
    for axis, label in ((occ.X, 'periodic_x'), (occ.Y, 'periodic_y')):
        lower = shape.edges[axis < -PERIOD/2+1e-6]
        upper = shape.edges[axis > PERIOD/2-1e-6]
        if not len(lower) or len(lower) != len(upper):
            raise ValueError(f'{label}: unmatched outer edges')
        lower.Identify(upper, label, IdentificationType.PERIODIC, occ.Translation(PERIOD*axis))
    shape.vertices.maxh = h/refine
    mesh = ng.Mesh(occ.OCCGeometry(shape, dim=2).GenerateMesh(maxh=h))
    periodic_counts(mesh)
    for name, area in expected.items():
        actual = ng.Integrate(1, mesh, definedon=mesh.Materials(name))
        if not np.isclose(actual, area, rtol=0, atol=1e-11):
            raise ValueError(f'{name}: material area mismatch')
    if not np.isclose(ng.Integrate(1, mesh), PERIOD**2, rtol=0, atol=1e-11):
        raise ValueError('periodic cell area mismatch')
    return mesh, materials


def project_profile(cap, h=.06, degree=6, quadrature=16, *, gap=17., width=350., seams=(31.,), values=(4., 9.), refine=20.):
    mesh, materials = make_profile_mesh(h, gap, width, seams, values, refine)
    coefficients, metadata = project_mesh(mesh, cap, degree, quadrature, materials)
    return coefficients, dict(fixture='partitioned_gap', gap=gap, width=width, seams=list(seams), values=list(values),
                              h=h, refine=refine, **metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cap', type=int, default=32)
    parser.add_argument('--h', type=float, default=.06)
    parser.add_argument('--degree', type=int, default=6)
    parser.add_argument('--quadrature', type=int, default=16)
    parser.add_argument('--gap', type=float, default=17.)
    parser.add_argument('--width', type=float, default=350.)
    parser.add_argument('--seams', type=float, nargs='*', default=[31.])
    parser.add_argument('--values', type=float, nargs='+', default=[4., 9.])
    parser.add_argument('--refine', type=float, default=20.)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    start = time.monotonic()
    coefficients, metadata = project_profile(args.cap, args.h, args.degree, args.quadrature, gap=args.gap,
                                            width=args.width, seams=args.seams, values=args.values, refine=args.refine)
    np.savez(args.output, coefficients=coefficients, metadata=json.dumps(metadata))
    print(json.dumps(dict(kind='gap_fem_cell', **metadata, seconds=time.monotonic()-start)), flush=True)


if __name__ == '__main__':
    main()
