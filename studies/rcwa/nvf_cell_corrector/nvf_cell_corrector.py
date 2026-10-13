"""Optional conforming 2D FEM corrector for the harmonic-energy investigation.

This solves only the static auxiliary cell problem, not Maxwell scattering.
It imports no Wavebit solver. Geometry is fixed to uniform, bar and tilted test
fixtures, in micrometers. The NPZ export is consumed by nvf_cell_bounds.py.
No autodiff or moving-contact contract is supplied by this instrument.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
import argparse
import json
import time
from dataclasses import dataclass

import ngsolve as ng
import netgen.occ as occ
import numpy as np
from netgen.meshing import IdentificationType

from studies.rcwa.nvf_cell_bounds.nvf_cell_bounds import energy_bounds, serializable
from studies.rcwa.nvf_fem_reference.nvf_fem_reference import periodic_counts
from studies.rcwa.nvf_fem_quadrature.nvf_fem_quadrature import triangle_quadrature

PERIOD = .7


def make_mesh(h=.06, fixture='tilted', refine=20.):
    if fixture not in ('uniform', 'bar', 'tilted'):
        raise ValueError('unknown cell fixture')
    if not np.isfinite([h, refine]).all() or not 0 < h < PERIOD/2 or refine < 1:
        raise ValueError('require 0 < h < period/2 and refine >= 1')
    cell = occ.WorkPlane().MoveTo(-PERIOD/2, -PERIOD/2).Rectangle(PERIOD, PERIOD).Face()
    expected = PERIOD**2
    if fixture == 'uniform':
        cell.faces.name = 'inclusion'
        shape = cell
    else:
        lx, ly, angle = ((.64, PERIOD/np.sqrt(5), np.arctan2(1, 2))
                         if fixture == 'tilted' else (.35, PERIOD, 0.))
        original = occ.WorkPlane().MoveTo(-lx/2, -ly/2).Rectangle(lx, ly).Face()
        original = original.Rotate(occ.Axis((0, 0, 0), occ.Z), np.rad2deg(angle))
        inclusion = original
        for shift in (-PERIOD, PERIOD):
            inclusion = inclusion+original.Move(occ.Vec(shift, 0, 0))
        inclusion = inclusion*cell
        background = cell-inclusion
        # In a 2D OCC mesh, face names (not volume .mat tags) name the materials.
        inclusion.faces.name = 'inclusion'
        background.faces.name = 'background'
        shape = occ.Glue([inclusion, background])
        expected = lx*ly
    for axis, label in ((occ.X, 'periodic_x'), (occ.Y, 'periodic_y')):
        lower = shape.edges[axis < -PERIOD/2+1e-6]
        upper = shape.edges[axis > PERIOD/2-1e-6]
        if not len(lower) or len(lower) != len(upper):
            raise ValueError(f'{label}: unmatched outer edges')
        lower.Identify(upper, label, IdentificationType.PERIODIC,
                       occ.Translation(PERIOD*axis))
    shape.vertices.maxh = h/refine
    mesh = ng.Mesh(occ.OCCGeometry(shape, dim=2).GenerateMesh(maxh=h))
    periodic_counts(mesh)
    actual = ng.Integrate(1, mesh, definedon=mesh.Materials('inclusion'))
    if not np.isclose(actual, expected, rtol=0, atol=1e-11):
        raise ValueError(f'inclusion area mismatch: {actual} != {expected}')
    return mesh


@dataclass
class CellSolution:
    basis: list
    fields: list  # Retain ownership of the grid functions used in basis.
    epsilon: object
    energy: np.ndarray
    residuals: list
    ndof: int


def correctors(mesh, degree, epsilon=12.25, *, reciprocal=False, transform=None):
    """Periodic scalar correctors, optionally on an affine image of the mesh.

    ``transform`` maps reference coordinates to physical coordinates. All
    material interfaces and lattice translations undergo the same map. The
    weak form uses physical gradients and a Cartesian unit forcing directly;
    it does not use the candidate's material-harmonic lift or trial space.
    """
    if not isinstance(degree, int) or degree < 1:
        raise ValueError('degree must be a positive integer')
    materials = dict(epsilon) if isinstance(epsilon, dict) else {'inclusion': epsilon}
    if not all(np.isfinite(value) and value > 0 for value in materials.values()):
        raise ValueError('epsilon must be finite, real and positive')
    if isinstance(epsilon, dict) and set(materials) != set(mesh.GetMaterials())-{'background'}:
        raise ValueError('material mapping must cover every non-background mesh region')
    gradient = ng.grad
    if transform is not None:
        transform = np.asarray(transform)
        if (transform.shape != (2, 2) or not np.isrealobj(transform)
                or not np.isfinite(transform).all() or np.linalg.det(transform) <= 0):
            raise ValueError('transform must be finite, real and right-handed with shape (2,2)')
        inverse_transpose = ng.CF(tuple(np.linalg.inv(transform).T.ravel()), dims=(2, 2))
        gradient = lambda value: inverse_transpose*ng.grad(value)
    eps = mesh.MaterialCF({name: 1/value if reciprocal else value for name, value in materials.items()}, default=1.)
    scalar = ng.Compress(ng.Periodic(ng.H1(mesh, order=degree)))
    space = ng.FESpace([scalar, ng.NumberSpace(mesh)])
    u, lagrange = space.TrialFunction()
    v, multiplier = space.TestFunction()
    form = ng.BilinearForm(space)
    form += (eps*gradient(u)*gradient(v)+u*multiplier+v*lagrange)*ng.dx
    form.Assemble()
    inverse = form.mat.Inverse(space.FreeDofs(), inverse='umfpack')
    basis, fields, residuals = [], [], []
    for j in range(2):
        right = ng.LinearForm(space)
        right += -eps*gradient(v)[j]*ng.dx
        right.Assemble()
        field = ng.GridFunction(space)
        field.vec.data = inverse*right.vec
        residual = right.vec.CreateVector()
        residual.data = form.mat*field.vec-right.vec
        relative = float(ng.Norm(residual)/max(ng.Norm(right.vec), 1.))
        if not np.isfinite(relative) or relative > 1e-10:
            raise FloatingPointError(f'invalid cell residual: {relative}')
        residuals.append(relative)
        fields.append(field)
        basis.append(gradient(field.components[0])+ng.CF((int(j == 0), int(j == 1))))
    energy = np.array([[ng.Integrate(eps*(a*b), mesh, order=2*degree+2)/PERIOD**2
                        for b in basis] for a in basis])
    return CellSolution(basis, fields, eps, energy, residuals, space.ndof)


def project_mesh(mesh, cap, degree=6, quadrature=16, epsilon=12.25, *, transform=None):
    """Integrate Fourier coefficients on conforming triangles, without a raster.

    Duffy-mapped Gauss quadrature is polynomial-exact for the unmodulated Gram
    field at quadrature >= degree. Fourier modulation requires a separate
    quadrature refinement. Point batches bound temporary Fourier-array memory.
    With transform, field components are physical while Fourier labels remain
    on the reference lattice. The constant area Jacobian cancels in averages.
    """
    if not isinstance(cap, int) or cap < 0:
        raise ValueError('cap must be a nonnegative integer')
    if not isinstance(quadrature, int) or quadrature < degree:
        raise ValueError('quadrature must be an integer at least the degree')
    primal = correctors(mesh, degree, epsilon, transform=transform)
    reciprocal = correctors(mesh, degree, epsilon, reciprocal=True, transform=transform)
    bounds = serializable(energy_bounds(primal.energy, reciprocal.energy))
    modes = np.arange(-cap, cap+1)[:, None]
    coefficients = np.zeros((8, 2*cap+1, 2*cap+1), complex)
    for mapped, xy, weights in triangle_quadrature(mesh, quadrature, period=PERIOD, area_fraction=1.):
        basis = np.stack([field(mapped).T for field in primal.basis], axis=1)
        eps = np.asarray(primal.epsilon(mapped)).ravel()
        values = np.stack([basis[0, 0], basis[0, 1], basis[1, 0], basis[1, 1],
                           eps*np.sum(basis[:, 0]**2, axis=0),
                           eps*np.sum(basis[:, 0]*basis[:, 1], axis=0),
                           eps*np.sum(basis[:, 1]**2, axis=0), eps])
        ex = np.exp(-2j*np.pi/PERIOD*modes*xy[:, 0])
        ey = np.exp(-2j*np.pi/PERIOD*modes*xy[:, 1])
        for row, value in enumerate(values):
            coefficients[row] += (ey*(weights*value))@ex.T
    gram_mean = np.array([[coefficients[4, cap, cap], coefficients[5, cap, cap]],
                          [coefficients[5, cap, cap], coefficients[6, cap, cap]]])
    error = float(np.max(abs(gram_mean-primal.energy)))
    if not np.isfinite(coefficients).all() or error > 1e-9:
        raise FloatingPointError(f'invalid energy projection: {error}')
    metadata = dict(degree=degree, quadrature=quadrature,
                    epsilon=epsilon, cap=cap, elements=mesh.ne,
                    ndof=primal.ndof, bounds=bounds, residuals=primal.residuals,
                    reciprocal_residuals=reciprocal.residuals,
                    mean_energy_quadrature_error=error, quadrature_points=mesh.ne*quadrature**2,
                    quadrature_mapping='element_local')
    if transform is not None:
        metadata['transform'] = np.asarray(transform).tolist()
    return coefficients, metadata


def project(cap, h=.06, degree=6, quadrature=16, refine=20., fixture='tilted', epsilon=12.25):
    mesh = make_mesh(h, fixture, refine)
    coefficients, metadata = project_mesh(mesh, cap, degree, quadrature, epsilon)
    return coefficients, dict(fixture=fixture, h=h, refine=refine, **metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cap', type=int, default=28)
    parser.add_argument('--h', type=float, default=.06)
    parser.add_argument('--degree', type=int, default=6)
    parser.add_argument('--quadrature', type=int, default=20)
    parser.add_argument('--refine', type=float, default=20.)
    parser.add_argument('--fixture', choices=['uniform', 'bar', 'tilted'], default='tilted')
    parser.add_argument('--epsilon', type=float, default=12.25)
    parser.add_argument('--output', required=True, help='path of the coefficient NPZ export')
    args = parser.parse_args()
    start = time.monotonic()
    coefficients, metadata = project(args.cap, args.h, args.degree, args.quadrature,
                                     args.refine, args.fixture, args.epsilon)
    np.savez(args.output, coefficients=coefficients, metadata=json.dumps(metadata))
    print(json.dumps(dict(kind='fem_cell', **metadata, output=args.output,
                          seconds=time.monotonic()-start)), flush=True)


if __name__ == '__main__':
    main()
