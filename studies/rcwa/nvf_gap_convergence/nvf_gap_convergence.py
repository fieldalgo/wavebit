"""Separate rectangular NVF cell error from optical Fourier truncation.

A single maximal coefficient projection feeds every outer order. Independent
Cartesian material-interval factorization shares only the Maxwell scattering
instrument. Primal/complementary bounds concern static energy, never optical T.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
import argparse
import json
import time

import numpy as np

from studies.rcwa.nvf_cell_bounds.nvf_cell_bounds import energy_bounds, projected_matrices, serializable
from studies.rcwa.nvf_continuous_flux.nvf_continuous_flux import ContinuousFluxGapModel
from studies.rcwa.nvf_fourier_orders.nvf_fourier_orders import orders_xy, parse_order
from studies.rcwa.nvf_harmonic_energy.nvf_harmonic_energy import PERIOD, polarized_powers
from wavebit.rcwa._engine.factorization import fourier2d_integrated


def profile_li(order, gap, width, seams, values):
    """Independent exact Cartesian intervals for a horizontal material tiling."""
    seams, values = np.asarray(seams), np.asarray(values)
    edges = np.r_[-width/2, seams, width/2]
    if (seams.ndim != 1 or values.shape != (len(seams)+1,)
            or not np.isfinite(np.r_[edges, values, gap]).all() or np.min(values) <= 0
            or not np.all(np.diff(edges) > 0) or not 0 < width < PERIOD or not 0 <= gap < PERIOD):
        raise ValueError('invalid rectangular material profile')
    x = np.r_[.5+edges/PERIOD, 1.][:, None]
    y = np.array([gap/(2*PERIOD), 1-gap/(2*PERIOD), 1.])[:, None]
    material = np.array([np.ones(len(values)+2), np.r_[1., values, 1.], np.ones(len(values)+2)], complex)
    ox, oy = orders_xy(order)
    ex, ey, eps = [fourier2d_integrated(np, material, x, y, ox, oy, inverse_rule=rule)
                   for rule in ('u', 'v', None)]
    return ex, ey, np.linalg.inv(eps), np.zeros_like(ex), np.zeros_like(ex)


def mean_energy(coefficients):
    cap = (coefficients.shape[-1]-1)//2
    xx, xy, yy = coefficients[4:7, cap, cap]
    return np.array([[xx, xy], [xy, yy]])


def power_record(order, slots):
    t, r = polarized_powers(order, slots)
    if not np.isfinite(np.r_[t, r]).all():
        raise FloatingPointError('nonfinite power is not convergence data')
    error = float(np.max(abs(t+r-1)))
    if error > 1e-9:
        raise FloatingPointError(f'lossless power error: {error}')
    return dict(p_T=float(t[0]), s_T=float(t[1]), p_R=float(r[0]), s_R=float(r[1]), energy_error=error)


def load_projection(path, gap, width, seams, values):
    with np.load(path, allow_pickle=False) as data:
        coefficients = data['coefficients']
        metadata = json.loads(str(data['metadata']))
    if metadata.get('fixture') != 'partitioned_gap':
        raise ValueError('projection fixture must be partitioned_gap')
    for key, expected in dict(gap=gap, width=width, seams=seams, values=values).items():
        actual, expected = np.asarray(metadata[key]), np.asarray(expected)
        if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=0, atol=1e-12):
            raise ValueError(f'projection profile mismatch: {key}')
    return coefficients, metadata


def convergence(orders, meshes, gap_degrees, *, gap=17., width=350., seams=(31.,), values=(4., 9.), edge_fraction=1e-4,
                projection=None, grading=None, body_degree=1, flux_degree=1, flux_basis='global'):
    if not orders:
        raise ValueError('orders must be a nonempty sequence')
    maximum = max(max(orders_xy(order)) for order in orders)
    if not isinstance(body_degree, int) or not 1 <= body_degree <= 4:
        raise ValueError('body_degree must be an integer between one and four')
    if not isinstance(flux_degree, int) or not 1 <= flux_degree <= 3:
        raise ValueError('flux_degree must be an integer between one and three')
    profile = dict(seams=np.asarray(seams), values=np.asarray(values))
    context = dict(gap=gap, width=width, seams=list(seams), values=list(values), edge_fraction=edge_fraction)
    for order in orders:
        start = time.monotonic()
        result = power_record(order, profile_li(order, gap, width, seams, values))
        yield dict(kind='gap_cartesian', order=order, **context, **result, seconds=time.monotonic()-start)
    if projection:
        coefficients, metadata = load_projection(projection, gap, width, seams, values)
        for order in orders:
            start = time.monotonic()
            result = power_record(order, projected_matrices(coefficients, order))
            yield dict(kind='gap_projected_fem', order=order, **context, **result,
                       projection=metadata, seconds=time.monotonic()-start)
    model_class = ContinuousFluxGapModel
    if grading is not None:
        from studies.rcwa.nvf_interface_resolution.nvf_interface_resolution import MaterialGradedGapModel
        model_class = MaterialGradedGapModel
    if body_degree > 1:
        from studies.rcwa.nvf_polynomial_body.nvf_polynomial_body import PolynomialBodyGapModel
        model_class = PolynomialBodyGapModel
    if flux_degree > 1:
        from studies.rcwa.nvf_polynomial_flux.nvf_polynomial_flux import PolynomialFluxGapModel
        model_class = PolynomialFluxGapModel
    for mesh in meshes:
        subdivisions, rows = mesh[:2]
        y_subdivisions = mesh[2] if len(mesh) == 3 else 1
        for gap_degree in gap_degrees:
            start = time.monotonic()
            settings = dict(rows=rows, x_subdivisions=subdivisions, y_subdivisions=y_subdivisions,
                            edge_fraction=edge_fraction, gap_degree=gap_degree, flux_basis=flux_basis)
            if grading is not None:
                settings.update(grading_strength=grading[0], grading_radius=grading[1])
            if body_degree > 1 or flux_degree > 1:
                settings['body_degree'] = body_degree
                if grading is None:
                    settings['grading_strength'] = 0.
            if flux_degree > 1:
                settings['flux_degree'] = flux_degree
            model = model_class(maximum, **settings)
            coefficients, metadata = model.coefficients(gap, profile, width)
            reciprocal = model_class(0, **settings)
            dual, dual_metadata = reciprocal.coefficients(gap, dict(seams=profile['seams'], values=1/profile['values']), width)
            bounds = energy_bounds(mean_energy(coefficients), mean_energy(dual))
            configuration = dict(**context, rows=rows, subdivisions=subdivisions,
                                 y_subdivisions=y_subdivisions, gap_degree=gap_degree,
                                 body_degree=body_degree, flux_degree=flux_degree, flux_basis=flux_basis)
            if grading is not None:
                configuration.update(grading_strength=grading[0], grading_radius=grading[1])
            yield dict(kind='gap_cell_bounds', **configuration, **serializable(bounds),
                       cell_residual=float(metadata['cell_residual']), reciprocal_residual=float(dual_metadata['cell_residual']),
                       degrees_of_freedom=metadata['degrees_of_freedom'], seconds=time.monotonic()-start)
            for order in orders:
                start = time.monotonic()
                result = power_record(order, projected_matrices(coefficients, order))
                kind = ('gap_polynomial_flux' if flux_degree > 1 else 'gap_polynomial_body' if body_degree > 1 else
                        'gap_material_graded' if grading is not None else 'gap_continuous_flux')
                yield dict(kind=kind,
                           order=order, **configuration, **result, seconds=time.monotonic()-start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--orders', type=parse_order, nargs='+', default=[2, 4, 8, 12],
                        help='square half width N or directional pair NX:NY')
    parser.add_argument('--meshes', nargs='*', default=['1:4', '2:8', '4:8'],
                        help='x subdivisions:base body rows[:y subdivisions]; empty for reference only')
    parser.add_argument('--gap-degrees', type=int, nargs='+', default=[1, 4])
    parser.add_argument('--body-degree', type=int, default=1, help='optional polynomial body degree, 1 through 4')
    parser.add_argument('--flux-degree', type=int, default=1, help='optional continuous x flux degree, 1 through 3')
    parser.add_argument('--flux-basis', choices=('global', 'local'), default='global')
    parser.add_argument('--gap', type=float, default=17.)
    parser.add_argument('--width', type=float, default=350.)
    parser.add_argument('--seams', type=float, nargs='*', default=[31.])
    parser.add_argument('--values', type=float, nargs='+', default=[4., 9.])
    parser.add_argument('--edge-fraction', type=float, default=1e-4)
    parser.add_argument('--projection', help='optional fitted 2D FEM auxiliary-field NPZ for the identical profile')
    parser.add_argument('--grading', help='optional material grading strength:radius, e.g. 8:.01')
    args = parser.parse_args()
    meshes = [tuple(map(int, mesh.split(':'))) for mesh in args.meshes]
    if any(len(mesh) not in (2, 3) for mesh in meshes):
        parser.error('each mesh must specify x subdivisions:rows[:y subdivisions]')
    grading = tuple(map(float, args.grading.split(':'))) if args.grading else None
    if grading is not None and len(grading) != 2:
        parser.error('grading must specify strength:radius')
    for record in convergence(args.orders, meshes, args.gap_degrees, gap=args.gap, width=args.width,
                               seams=args.seams, values=args.values, edge_fraction=args.edge_fraction,
                               projection=args.projection, grading=grading, body_degree=args.body_degree,
                               flux_degree=args.flux_degree, flux_basis=args.flux_basis):
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
