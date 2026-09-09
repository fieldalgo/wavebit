"""Independent fitted FEM auxiliary cell with exact y-parity boundary spaces.

The rectangular profile is solved on 0 <= y <= period/2. The x corrector
is even (natural Neumann boundaries); the y corrector is odd (zero Dirichlet
boundaries). Fourier integration extends these half-cell fields by parity.
No full-cell export is symmetrized, and this is not a 3D Maxwell reference.
"""
import argparse
import json
import time

import ngsolve as ng
import netgen.occ as occ
import numpy as np
from netgen.meshing import IdentificationType

from studies.nvf_cell_bounds import energy_bounds, serializable
from studies.nvf_cell_corrector import PERIOD, CellSolution
from studies.nvf_fem_quadrature import triangle_quadrature


def make_half_mesh(h=.06, gap=17., width=350., seams=(31.,), values=(4., 9.), refine=20.):
    edges = np.r_[-width/2, seams, width/2]/1000
    height = PERIOD-gap/1000
    if (len(values) != len(edges)-1 or np.iscomplexobj(np.r_[edges, values, h, gap, refine])
            or not np.isfinite(np.r_[edges, values, h, gap, refine]).all()
            or np.min(values) <= 0 or not np.all(np.diff(edges) > 0)
            or not 0 < width < 1000*PERIOD or not 0 < height <= PERIOD
            or not 0 < h < PERIOD/2 or refine < 1):
        raise ValueError('invalid half-cell profile or mesh settings')
    cell = occ.WorkPlane().MoveTo(-PERIOD/2, 0).Rectangle(PERIOD, PERIOD/2).Face()
    remaining = cell
    parts, materials, expected = [], {}, {}
    for i, (left, right, value) in enumerate(zip(edges[:-1], edges[1:], values)):
        part = occ.WorkPlane().MoveTo(left, 0).Rectangle(right-left, height/2).Face()
        name = f'piece_{i}'
        part.faces.name = name
        parts.append(part)
        remaining = remaining-part
        materials[name], expected[name] = value, (right-left)*height/2
    remaining.faces.name = 'background'
    shape = occ.Glue(parts+[remaining])
    for edges_on_mirror in (shape.edges[occ.Y < 1e-6], shape.edges[occ.Y > PERIOD/2-1e-6]):
        if not len(edges_on_mirror):
            raise ValueError('missing half-cell mirror boundary')
        edges_on_mirror.name = 'mirror'
    lower, upper = shape.edges[occ.X < -PERIOD/2+1e-6], shape.edges[occ.X > PERIOD/2-1e-6]
    if not len(lower) or len(lower) != len(upper):
        raise ValueError('unmatched half-cell periodic edges')
    lower.Identify(upper, 'periodic_x', IdentificationType.PERIODIC, occ.Translation(PERIOD*occ.X))
    shape.vertices.maxh = h/refine
    mesh = ng.Mesh(occ.OCCGeometry(shape, dim=2).GenerateMesh(maxh=h))
    pairs = mesh.ngmesh.GetIdentifications()
    if not pairs:
        raise ValueError('missing half-cell periodic point identifications')
    for a, b in pairs:
        delta = abs(np.asarray(mesh.ngmesh[b].p)-np.asarray(mesh.ngmesh[a].p))
        if not np.allclose(delta, [PERIOD, 0, 0], rtol=0, atol=1e-9):
            raise ValueError(f'unexpected half-cell periodic point displacement: {delta}')
    for name, area in {**expected, '.*': PERIOD**2/2}.items():
        actual = ng.Integrate(1, mesh, definedon=mesh.Materials(name))
        if not np.isclose(actual, area, rtol=0, atol=1e-11):
            raise ValueError(f'{name}: half-cell area mismatch')
    return mesh, materials


def half_correctors(mesh, degree, materials, *, reciprocal=False):
    if not isinstance(degree, int) or degree < 1:
        raise ValueError('degree must be a positive integer')
    if (set(materials) != set(mesh.GetMaterials())-{'background'}
            or not all(not np.iscomplexobj(value) and np.isfinite(value) and value > 0 for value in materials.values())):
        raise ValueError('invalid half-cell material mapping')
    eps = mesh.MaterialCF({name: 1/value if reciprocal else value for name, value in materials.items()}, default=1.)
    basis, fields, residuals, dimensions = [], [], [], []
    for j in range(2):
        scalar = ng.Compress(ng.Periodic(ng.H1(mesh, order=degree, dirichlet='mirror' if j else '')))
        space = scalar if j else ng.FESpace([scalar, ng.NumberSpace(mesh)])
        form = ng.BilinearForm(space)
        if j:
            u, v = space.TnT()
        else:
            u, lagrange = space.TrialFunction()
            v, multiplier = space.TestFunction()
            form += (u*multiplier+v*lagrange)*ng.dx
        form += eps*ng.grad(u)*ng.grad(v)*ng.dx
        form.Assemble()
        right = ng.LinearForm(space)
        right += -eps*ng.grad(v)[j]*ng.dx
        right.Assemble()
        field = ng.GridFunction(space)
        field.vec.data = form.mat.Inverse(space.FreeDofs(), inverse='umfpack')*right.vec
        residual = right.vec.CreateVector()
        residual.data = form.mat*field.vec-right.vec
        free = np.asarray(list(space.FreeDofs()), dtype=bool)
        residual.FV().NumPy()[~free] = 0  # Dirichlet reactions are not equation residuals.
        right_norm = np.linalg.norm(right.vec.FV().NumPy()[free])
        relative = float(ng.Norm(residual)/max(right_norm, 1.))
        if not np.isfinite(relative) or relative > 1e-10:
            raise FloatingPointError(f'invalid half-cell residual: {relative}')
        residuals.append(relative)
        fields.append(field)
        dimensions.append(space.ndof)
        potential = field if j else field.components[0]
        basis.append(ng.grad(potential)+ng.CF((int(j == 0), int(j == 1))))
    # Cross energy is odd in y; its full-cell integral is exactly zero.
    energy = np.diag([2*ng.Integrate(eps*(a*a), mesh, order=2*degree+2)/PERIOD**2 for a in basis])
    return CellSolution(basis, fields, eps, energy, residuals, sum(dimensions)), dimensions


def half_quadrature(mesh, quadrature, batch_size=24000):
    """Element-owned quadrature with half-cell normalization."""
    return triangle_quadrature(mesh, quadrature, period=PERIOD, area_fraction=.5, batch_size=batch_size)


def project_half_profile(cap, h=.06, degree=6, quadrature=16, *, gap=17., width=350.,
                         seams=(31.,), values=(4., 9.), refine=20.):
    if not isinstance(cap, int) or cap < 0 or not isinstance(quadrature, int) or quadrature < degree:
        raise ValueError('require nonnegative integer cap and integer quadrature at least the degree')
    mesh, materials = make_half_mesh(h, gap, width, seams, values, refine)
    primal, dimensions = half_correctors(mesh, degree, materials)
    reciprocal, _ = half_correctors(mesh, degree, materials, reciprocal=True)
    bounds = serializable(energy_bounds(primal.energy, reciprocal.energy))
    modes = np.arange(-cap, cap+1)[:, None]
    signs = [1, -1, -1, 1, 1, -1, 1, 1]
    coefficients = np.zeros((8, 2*cap+1, 2*cap+1), complex)
    for mapped, xy, weights in half_quadrature(mesh, quadrature):
        basis = np.stack([field(mapped).T for field in primal.basis], axis=1)
        eps = np.asarray(primal.epsilon(mapped)).ravel()
        fields = np.stack([basis[0, 0], basis[0, 1], basis[1, 0], basis[1, 1],
                           eps*np.sum(basis[:, 0]**2, axis=0),
                           eps*np.sum(basis[:, 0]*basis[:, 1], axis=0),
                           eps*np.sum(basis[:, 1]**2, axis=0), eps])
        ex = np.exp(-2j*np.pi/PERIOD*modes*xy[:, 0])
        ey = np.exp(-2j*np.pi/PERIOD*modes*xy[:, 1])
        for row, value in enumerate(fields):
            # Pair the two physical half-cell integrals, using the solved
            # field's parity, rather than averaging an asymmetric export.
            coefficients[row] += ((ey+signs[row]*ey.conj())*(weights*value))@ex.T
    gram = np.array([[coefficients[4, cap, cap], coefficients[5, cap, cap]],
                     [coefficients[5, cap, cap], coefficients[6, cap, cap]]])
    error = float(np.max(abs(gram-primal.energy)))
    mean_error = float(np.max(abs(coefficients[:4, cap, cap].reshape(2, 2)-np.eye(2))))
    if not np.isfinite(coefficients).all() or error > 1e-9 or mean_error > 1e-10:
        raise FloatingPointError(f'invalid half-cell projection: energy={error}, mean={mean_error}')
    metadata = dict(fixture='partitioned_gap', gap=gap, width=width, seams=list(seams), values=list(values),
                    cell_method='fitted_half_cell', h=h, refine=refine, degree=degree, quadrature=quadrature,
                    cap=cap, elements=mesh.ne, ndof=primal.ndof, matrix_dimensions=dimensions,
                    bounds=bounds, residuals=primal.residuals, reciprocal_residuals=reciprocal.residuals,
                    mean_energy_quadrature_error=error, mean_field_error=mean_error,
                    quadrature_points=mesh.ne*quadrature**2, quadrature_mapping='element_local')
    return coefficients, metadata


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
    coefficients, metadata = project_half_profile(args.cap, args.h, args.degree, args.quadrature,
                                                 gap=args.gap, width=args.width, seams=args.seams,
                                                 values=args.values, refine=args.refine)
    np.savez(args.output, coefficients=coefficients, metadata=json.dumps(metadata))
    print(json.dumps(dict(kind='mirror_fem_cell', **metadata, seconds=time.monotonic()-start)), flush=True)


if __name__ == '__main__':
    main()
