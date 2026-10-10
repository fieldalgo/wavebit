"""Separate affine-cell error, FEM projection error and optical order changes.

Independent fitted scalar FEM fields and the candidate share only the energy
closure and Maxwell engine. Static enclosures are not optical error bounds.
Each auxiliary field is projected once at the maximal requested Fourier cap.
"""
import argparse
import json
import time

import numpy as np

import wavebit
from wavebit._core.conforming.affine_cell import AffineCellGapModel
from studies.nvf_cell_bounds import energy_bounds, projected_matrices, serializable
from studies.nvf_fourier_orders import orders_xy, parse_order


def real_coefficients(coefficients):
    """Keep the real-field layout only after checking its independent yx term."""
    c = np.asarray(coefficients)
    if c.ndim != 3 or c.shape[0] not in (8, 9) or c.shape[1] != c.shape[2] or c.shape[1] % 2 != 1:
        raise ValueError('expected eight or nine square odd-sized coefficient arrays')
    if not np.isfinite(c).all() or np.max(abs(c-c[:, ::-1, ::-1].conj())) > 1e-9:
        raise ValueError('the affine enclosure study requires finite real-field Fourier coefficients')
    if len(c) == 9 and np.max(abs(c[8]-c[5])) > 1e-10:
        raise ValueError('the affine enclosure study requires a symmetric real Gram field')
    center = c.shape[-1]//2
    if np.max(abs(c[:4, center, center].reshape(2, 2)-np.eye(2))) > 1e-9:
        raise ValueError('the physical auxiliary field must have identity mean')
    return c[:8]


def material_coefficients(cap, gap, width, seams, values):
    """Exact reference-interval material coefficients, independent of any field."""
    seams, values = np.asarray(seams), np.asarray(values)
    if (seams.ndim != 1 or values.shape != (len(seams)+1,) or not np.isrealobj(seams)
            or not np.isrealobj(values) or not np.isfinite(values).all() or np.min(values) <= 0):
        raise ValueError('static enclosures require ordered seams and finite positive real materials')
    edges = np.r_[-width/2, seams, width/2]/700
    if (not np.isfinite(np.r_[edges, gap]).all() or not 0 < width < 700
            or not 0 <= gap < 700 or not np.all(np.diff(edges) > 0)):
        raise ValueError('invalid affine reference profile')
    n, m = np.meshgrid(np.arange(-cap, cap+1), np.arange(-cap, cap+1), indexing='ij')
    result = ((m == 0) & (n == 0)).astype(complex)
    height = 1-gap/700
    for left, right, value in zip(edges[:-1], edges[1:], values):
        result += ((value-1)*(right-left)*height*np.sinc(m*(right-left))*np.sinc(n*height)
                   *np.exp(-1j*np.pi*m*(left+right)))
    return result


def load_projection(path, *, transform, gap, width, seams, values):
    with np.load(path, allow_pickle=False) as data:
        c = real_coefficients(data['coefficients'])
        metadata = json.loads(str(data['metadata']))
    if metadata.get('fixture') != 'affine_partitioned_gap':
        raise ValueError('projection fixture must be affine_partitioned_gap')
    for key, expected in dict(transform=transform, gap=gap, width=width, seams=seams, values=values).items():
        try:
            actual, expected = np.asarray(metadata.get(key), dtype=float), np.asarray(expected)
        except (TypeError, ValueError) as error:
            raise ValueError(f'affine projection profile mismatch: {key}') from error
        if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.allclose(actual, expected, rtol=0, atol=1e-12):
            raise ValueError(f'affine projection profile mismatch: {key}')
    if metadata.get('cap') != c.shape[-1]//2:
        raise ValueError('affine projection cap disagrees with its coefficient arrays')
    expected = material_coefficients(c.shape[-1]//2, gap, width, seams, values)
    if np.max(abs(c[7]-expected)) > 1e-9:
        raise ValueError('affine projection material differs from the analytic profile')
    return c, metadata


def mean_energy(coefficients):
    center = coefficients.shape[-1]//2
    xx, xy, yy = coefficients[4:7, center, center]
    return np.array([[xx, xy], [xy, yy]])


def power_record(coefficients, order, transform, *, wavelength=900., thickness=300., theta=.2, phi=0.):
    slots = projected_matrices(real_coefficients(coefficients), order)
    solver = wavebit.rcwa(backend='numpy', lattice=700*np.asarray(transform), wavelength=wavelength,
                          thickness=[thickness], theta=theta, phi=phi,
                          truncation_order=orders_xy(order), top=1., bot=2.25)
    result = solver._solve_conv(*(slot[None] for slot in slots))
    powers = np.asarray([[np.asarray(getattr(getattr(result, pol), key)).sum() for key in ('T', 'R')]
                         for pol in ('p_inc', 's_inc')])
    error = float(np.max(abs(powers.sum(axis=1)-1)))
    if not np.isfinite(powers).all() or error > 1e-9:
        raise FloatingPointError(f'invalid lossless optical readout: energy error {error}')
    return dict(p_T=float(powers[0, 0]), p_R=float(powers[0, 1]),
                s_T=float(powers[1, 0]), s_R=float(powers[1, 1]), energy_error=error)


def directional_changes(records):
    """Signed changes on complete adjacent order rectangles, with no verdict.

    Geometry, material, candidate resolution and reference metadata identify
    separate series. Both parallel edges are reported: a small diagonal
    change can hide cancellation or an unresolved mixed contribution.
    """
    measurements = {'order', 'p_T', 'p_R', 's_T', 's_R', 'energy_error', 'seconds'}
    groups = {}
    for record in records:
        if record['kind'] == 'affine_cell_bounds':
            continue
        if record['kind'] not in ('affine_fitted_cell', 'affine_candidate'):
            raise ValueError('directional changes require affine optical records')
        context = {key: value for key, value in record.items() if key not in measurements}
        key = json.dumps(context, sort_keys=True, allow_nan=False)
        order = orders_xy(record['order'])
        powers = np.array([[record[f'{pol}_{quantity}'] for quantity in ('T', 'R')] for pol in ('p', 's')])
        if not np.isfinite(powers).all() or np.max(abs(powers.sum(axis=1)-1)) > 1e-9:
            raise ValueError('directional changes require finite lossless powers')
        context, points = groups.setdefault(key, (context, {}))
        if order in points:
            raise ValueError('duplicate Fourier order in the same affine series')
        points[order] = powers[:, 0]
    result = []
    for context, points in groups.values():
        xs, ys = sorted({x for x, _ in points}), sorted({y for _, y in points})
        for x0, x1 in zip(xs[:-1], xs[1:]):
            for y0, y1 in zip(ys[:-1], ys[1:]):
                corners = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
                if not all(corner in points for corner in corners):
                    continue
                low, right, up, high = [points[corner] for corner in corners]
                changes = dict(x_at_lower_y=right-low, x_at_upper_y=high-up,
                               y_at_lower_x=up-low, y_at_upper_x=high-right,
                               mixed=high-right-up+low, diagonal=high-low)
                source = dict(context)
                source['source_kind'] = source.pop('kind')
                result.append(dict(kind='affine_directional_change', **source,
                                   x_orders=[x0, x1], y_orders=[y0, y1],
                                   **{pol: {name: float(value[i]) for name, value in changes.items()}
                                      for i, pol in enumerate(('p', 's'))}))
    return result


def convergence(orders, cells, *, projections=(), transform=((1., .45), (0., .7)), gap=17., width=350.,
                seams=(31.,), values=(4., 9.), edge_fraction=1e-5):
    if not orders:
        raise ValueError('orders must be nonempty')
    maximum = max(max(orders_xy(order)) for order in orders)
    transform = np.asarray(transform)
    if (transform.shape != (2, 2) or not np.isrealobj(transform) or not np.isfinite(transform).all()
            or np.linalg.det(transform) <= 0):
        raise ValueError('transform must be finite, real and right-handed with shape (2,2)')
    profile = dict(seams=np.asarray(seams), values=np.asarray(values))
    material_coefficients(0, gap, width, seams, values)
    context = dict(transform=transform.tolist(), gap=gap, width=width, seams=list(seams), values=list(values))
    for projection in projections:
        c, metadata = load_projection(projection, **context)
        for order in orders:
            start = time.monotonic()
            result = power_record(c, order, transform)
            yield dict(kind='affine_fitted_cell', order=order, **context, **result,
                       projection=metadata, seconds=time.monotonic()-start)
    for cell in cells:
        if len(cell) not in (4, 5):
            raise ValueError('each cell needs x subdivisions, rows, body degree, gap degree and optional y subdivisions')
        subdivisions, rows, body_degree, gap_degree = cell[:4]
        settings = dict(x_subdivisions=subdivisions, rows=rows, body_degree=body_degree,
                        gap_degree=gap_degree, y_subdivisions=cell[4] if len(cell) == 5 else 1,
                        flux_basis='local', edge_fraction=edge_fraction)
        start = time.monotonic()
        model = AffineCellGapModel(maximum, **settings)
        coefficients, metadata = model.coefficients(gap, profile, width, basis=700*transform)
        c = real_coefficients(coefficients)
        reciprocal = AffineCellGapModel(0, **settings)
        d, dual_metadata = reciprocal.coefficients(gap, dict(seams=profile['seams'], values=1/profile['values']),
                                                  width, basis=700*transform)
        bounds = serializable(energy_bounds(mean_energy(c), mean_energy(real_coefficients(d))))
        yield dict(kind='affine_cell_bounds', **context, **settings, **bounds,
                   cell_residual=float(metadata['cell_residual']), reciprocal_residual=float(dual_metadata['cell_residual']),
                   degrees_of_freedom=metadata['degrees_of_freedom'], seconds=time.monotonic()-start)
        for order in orders:
            start = time.monotonic()
            yield dict(kind='affine_candidate', order=order, **context, **settings,
                       **power_record(c, order, transform), seconds=time.monotonic()-start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--orders', type=parse_order, nargs='+', default=[4, 6, 8])
    parser.add_argument('--cells', nargs='*', default=['1:4:2:4', '1:4:4:6'],
                        help='x subdivisions:rows:body degree:gap degree[:y subdivisions]')
    parser.add_argument('--projections', nargs='*', default=[])
    parser.add_argument('--transform', type=float, nargs=4, default=[1., .45, 0., .7], help='row-major affine map')
    parser.add_argument('--gap', type=float, default=17.)
    parser.add_argument('--width', type=float, default=350.)
    parser.add_argument('--seams', type=float, nargs='*', default=[31.])
    parser.add_argument('--values', type=float, nargs='+', default=[4., 9.])
    parser.add_argument('--edge-fraction', type=float, default=1e-5)
    parser.add_argument('--directional-changes', action='store_true',
                        help='append signed changes for complete adjacent x/y order rectangles; no convergence verdict')
    args = parser.parse_args()
    cells = [tuple(map(int, cell.split(':'))) for cell in args.cells]
    records = []
    for record in convergence(args.orders, cells, projections=args.projections,
                               transform=np.asarray(args.transform).reshape(2, 2), gap=args.gap, width=args.width,
                               seams=args.seams, values=args.values, edge_fraction=args.edge_fraction):
        print(json.dumps(record), flush=True)
        if args.directional_changes:
            records.append(record)
    if args.directional_changes:
        for record in directional_changes(records):
            print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
