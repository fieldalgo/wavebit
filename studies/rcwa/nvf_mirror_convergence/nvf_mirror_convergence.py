"""Mirror-sector acceleration of the unchanged rectangular NVF energy closure.

This NumPy research instrument requires y-reflection symmetry and the existing
zero-y-Bloch incidence. It builds both parity blocks directly, including the
change-of-field solves, then matches tangential fields in each sector. It is
not a new factorization, an asymmetric-geometry fallback or a production API.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
import argparse
import json
import time

import numpy as np
from scipy.linalg import eig, solve
from scipy.sparse import bmat, csc_matrix, diags
from scipy.sparse.linalg import splu

from studies.rcwa.nvf_energy_closure.nvf_energy_closure import congruence_solve, hermitian_defect
from studies.rcwa.nvf_fourier_orders.nvf_fourier_orders import orders_xy, parse_order
from studies.rcwa.nvf_gap_convergence.nvf_gap_convergence import load_projection
from studies.rcwa.nvf_harmonic_energy.nvf_harmonic_energy import PERIOD, THETA, THICKNESS, WAVELENGTH
from wavebit._common.conventions import mode_q


def parity_basis(order, parity):
    """Orthonormal even/odd combinations of positive and negative y orders."""
    if parity not in (-1, 1):
        raise ValueError('parity must be +1 or -1')
    x, y = orders_xy(order)
    n, m = np.meshgrid(np.arange(0 if parity == 1 else 1, y+1), np.arange(-x, x+1), indexing='ij')
    n, m = n.ravel(), m.ravel()
    return dict(n=n, m=m, primary=np.where(n == 0, 1., 1/np.sqrt(2)),
                reflected=np.where(n == 0, 0., parity/np.sqrt(2)))


def parity_convolution(coefficients, row, column, xp=np):
    """Restrict a scalar Fourier convolution without constructing full bases."""
    cap = (len(coefficients)-1)//2
    if xp is np:
        # Only the returned convolution is dense. Bound the temporary Fourier
        # index and weighting grids independently of the retained row count.
        result = np.zeros((len(row['n']), len(column['n'])), dtype=np.complex128)
        for start in range(0, len(row['n']), 128):
            stop = min(start+128, len(row['n']))
            dm = row['m'][start:stop, None]-column['m'][None]
            for rs, rw in ((1, row['primary']), (-1, row['reflected'])):
                for cs, cw in ((1, column['primary']), (-1, column['reflected'])):
                    dn = rs*row['n'][start:stop, None]-cs*column['n'][None]
                    result[start:stop] += rw[start:stop, None]*cw[None]*coefficients[dn+cap, dm+cap]
        return result
    dm = row['m'][:, None]-column['m'][None]
    result = xp.zeros(dm.shape, dtype=xp.complex128)
    for rs, rw in ((1, row['primary']), (-1, row['reflected'])):
        for cs, cw in ((1, column['primary']), (-1, column['reflected'])):
            dn = rs*row['n'][:, None]-cs*column['n'][None]
            result += rw[:, None]*cw[None]*coefficients[dn+cap, dm+cap]
    return result


def validate_coefficient_shape(c, order):
    if c.ndim != 3 or c.shape[0] != 8 or c.shape[1] != c.shape[2] or c.shape[1] % 2 != 1:
        raise ValueError('expected eight square odd-sized coefficient arrays')
    if 2*max(orders_xy(order)) > (c.shape[-1]-1)//2:
        raise ValueError('coefficient cap must cover all retained order differences')


def validate_mirror(coefficients, order):
    c = np.asarray(coefficients)
    validate_coefficient_shape(c, order)
    if not np.isfinite(c).all() or np.max(abs(c-c[:, ::-1, ::-1].conj())) > 1e-9:
        raise ValueError('invalid real-field Fourier coefficients')
    signs = np.array([1, -1, -1, 1, 1, -1, 1, 1])
    error = float(np.max(abs(c-signs[:, None, None]*c[:, ::-1]))/max(np.max(abs(c)), 1.))
    if error > 1e-10:
        raise ValueError(f'coefficient y-reflection symmetry is not resolved: {error}')
    return error


def mirror_sector(coefficients, order, polarization, xp=np):
    """Form the same finite energy closure in one exact mirror sector."""
    if xp is np:
        error = validate_mirror(coefficients, order)
    else:
        if xp.asarray(0.).dtype != np.dtype('float64'):
            raise ValueError('the mirror study requires JAX x64')
        coefficients = xp.asarray(coefficients)
        validate_coefficient_shape(coefficients, order)
        signs = xp.asarray([1, -1, -1, 1, 1, -1, 1, 1])
        error = xp.max(abs(coefficients-signs[:, None, None]*coefficients[:, ::-1]))/xp.maximum(xp.max(abs(coefficients)), 1.)
        real_error = xp.max(abs(coefficients-coefficients[:, ::-1, ::-1].conj()))
        valid = xp.all(xp.isfinite(coefficients)) & (error <= 1e-10) & (real_error <= 1e-9)
        coefficients = xp.where(valid, coefficients, xp.nan)
    if polarization not in ('p', 's'):
        raise ValueError('polarization must be p or s')
    a = parity_basis(order, 1 if polarization == 'p' else -1)
    b = parity_basis(order, -1 if polarization == 'p' else 1)
    def conv(row, left, right):
        return parity_convolution(coefficients[row], left, right, xp)
    def blocks(indices):
        if xp is not np:
            return xp.block([[conv(indices[0][0], a, a), conv(indices[0][1], a, b)],
                             [conv(indices[1][0], b, a), conv(indices[1][1], b, b)]])
        na, nb = len(a['n']), len(b['n'])
        result = np.empty((na+nb, na+nb), dtype=np.complex128, order='F')
        result[:na, :na] = conv(indices[0][0], a, a)
        result[:na, na:] = conv(indices[0][1], a, b)
        result[na:, :na] = conv(indices[1][0], b, a)
        result[na:, na:] = conv(indices[1][1], b, b)
        return result
    linear_solve = solve if xp is np else xp.linalg.solve
    change, gram = blocks(((0, 1), (2, 3))), blocks(((4, 5), (5, 6)))
    if xp is np:
        # The two arrays are owned here. Fortran storage allows both solves
        # to overwrite their operands instead of retaining an explicit inverse.
        effective = congruence_solve(change, gram, overwrite=True)
        del change, gram
    else:
        inverse = linear_solve(change, xp.eye(len(change)))
        effective = inverse.conj().T@gram@inverse
    if xp is np:
        valid = np.isfinite(effective).all() and hermitian_defect(effective) <= 1e-9
    else:
        valid = xp.all(xp.isfinite(effective)) & (xp.max(abs(effective-effective.conj().T)) <= 1e-9)
    if xp is np and not valid:
        raise FloatingPointError('invalid projected mirror energy closure')
    if xp is not np:
        effective = xp.where(valid, effective, xp.nan)
    z = linear_solve(conv(7, a, a), xp.eye(len(a['n']))) if len(a['n']) else xp.empty((0, 0), dtype=xp.complex128)
    return dict(a=a, b=b, effective=effective, z_inverse=z, mirror_error=error, polarization=polarization)


def sector_geometry(sector, *, sparse=False):
    a, b = sector['a'], sector['b']
    ka, kb = [np.sin(THETA)+WAVELENGTH/PERIOD*q['m'] for q in (a, b)]
    ya, yb = [WAVELENGTH/PERIOD*q['n'] for q in (a, b)]
    # Multiplication by ky exchanges the even and odd spaces at equal |n|,m.
    if sparse:
        # ky maps each nonzero-y wave to exactly one wave of opposite parity.
        # Construct that map in O(H), without dense pairwise index grids.
        rows = {(int(n), int(m)): i for i, (n, m) in enumerate(zip(a['n'], a['m']))}
        pairs = [(rows[(int(n), int(m))], j) for j, (n, m) in enumerate(zip(b['n'], b['m']))
                 if n != 0 and (int(n), int(m)) in rows]
        ri, ci = np.asarray(pairs, dtype=int).reshape(-1, 2).T
        coupling = csc_matrix((ya[ri], (ri, ci)), shape=(len(ka), len(kb)))
    else:
        coupling = ((a['n'][:, None] == b['n'][None]) & (a['m'][:, None] == b['m'][None]))*ya[:, None]
    return ka, kb, ya, yb, coupling


def sector_operator(e, z, geometry):
    """Assemble LR directly in its LAPACK storage, with no full block copy."""
    ka, kb, ya, yb, y = geometry
    na, nb = len(ka), len(kb)
    ex, ey, exy, eyx = e[:na, :na], e[na:, na:], e[:na, na:], e[na:, :na]
    a = ka[:, None]*z*ka[None]
    d = y.T@z@y
    a[np.diag_indices(na)] -= 1
    d[np.diag_indices(nb)] -= 1
    c, f = ka[:, None]*(z@y), (y.T@z)*ka[None]
    operator = np.empty(e.shape, dtype=np.complex128, order='F')
    # matmul accepts the leading dimensions of these quadrant views. Only
    # one quadrant-sized addition is temporary; np.block and conversion to
    # Fortran order would each retain another complete sector matrix.
    np.matmul(a, ex, out=operator[:na, :na])
    operator[np.arange(na), np.arange(na)] += ya**2
    operator[:na, :na] += c@eyx
    coupling = y.tocoo()
    np.matmul(c, ey, out=operator[:na, na:])
    operator[coupling.row, na+coupling.col] -= ka[coupling.row]*coupling.data
    operator[:na, na:] += a@exy
    np.matmul(f, ex, out=operator[na:, :na])
    operator[na+coupling.col, coupling.row] -= kb[coupling.col]*coupling.data
    operator[na:, :na] += d@eyx
    np.matmul(d, ey, out=operator[na:, na:])
    operator[na+np.arange(nb), na+np.arange(nb)] += kb**2
    operator[na:, na:] += f@exy
    return operator


def sector_magnetic_modes(e, w, rates, geometry):
    """Apply R to all modes without allocating the dense R factor."""
    ka, kb, ya, yb, y = geometry
    na, nb = len(ka), len(kb)
    v = np.empty(w.shape, dtype=np.complex128, order='F')
    np.matmul(e[na:], w, out=v[:nb])
    v[:nb] *= -1
    np.matmul(e[:na], w, out=v[nb:])
    # Geometry contributes only a diagonal and the sparse ky exchange.
    # Bound its temporary modal columns independently of the outer order.
    for start in range(0, len(w), 128):
        stop = min(start+128, len(w))
        wa, wb = w[:na, start:stop], w[na:, start:stop]
        v[:nb, start:stop] += kb[:, None]*(kb[:, None]*wb-y.T@wa)
        v[nb:, start:stop] += y@(kb[:, None]*wb)-ya[:, None]**2*wa
    v /= rates[None]
    return v


def sector_modes(sector):
    """Restrict the two first-order Maxwell factors before their eigensolve."""
    geometry = sector_geometry(sector, sparse=True)
    operator = sector_operator(sector['effective'], sector['z_inverse'], geometry)
    # At high outer orders each dense sector matrix is large. Discard the
    # assembly intermediates and let LAPACK reuse its Fortran-order input.
    values, w = eig(operator, overwrite_a=True)
    del operator
    # The existing fixture is off cutoff. Do not substitute this reduced
    # instrument for the production coupled-cluster treatment near grazing.
    if np.min(abs(values)) < 1e-9*np.max(abs(values)):
        raise FloatingPointError('mirror study does not implement the near-grazing cluster treatment')
    rates = mode_q(np, values)
    v = sector_magnetic_modes(sector['effective'], w, rates, geometry)
    return w, v, rates


def exterior_admittance(sector, epsilon):
    """Sparse tangential H/E map, using the same outgoing half-space modes."""
    ka, kb, ya, yb, y = sector_geometry(sector, sparse=True)
    za, zb = np.sqrt((epsilon-ka*ka-ya*ya).astype(complex)), np.sqrt((epsilon-kb*kb-yb*yb).astype(complex))
    if min(np.min(abs(za), initial=np.inf), np.min(abs(zb), initial=np.inf)) < 1e-12:
        raise FloatingPointError('mirror study requires nongrazing exterior orders')
    return 1j*bmat([[diags(-kb/zb)@y.T, diags((kb*kb-epsilon)/zb)],
                    [diags((epsilon-ya*ya)/za), diags(1/za)@y.multiply(kb[None])]], format='csc')


def sector_power(sector, progress=None):
    polarization = sector['polarization']
    na = len(sector['a']['n'])
    basis = sector['a'] if polarization == 'p' else sector['b']
    origin = np.flatnonzero((basis['n'] == 0) & (basis['m'] == 0))[0]
    w, v, rates = sector_modes(sector)
    if progress:
        progress('modes_ready')
    yt, yb = exterior_admittance(sector, 1.), exterior_admittance(sector, 2.25)
    # A fresh sector can release the constitutive matrices after the modes and
    # ports are formed; callers holding their own sector still retain it.
    del sector
    decay = np.exp(-2*np.pi/WAVELENGTH*THICKNESS*rates)
    # With B = V^-1 Yb W and D = diag(decay), (I+B) O = D is
    # equivalently (V+Yb W) O = V D. Also (I-B) O = 2 O-D.
    # This removes one dense solve, one dense product and the identity array.
    bottom = v.copy(order='F')
    for start in range(0, len(w), 128):
        bottom[:, start:start+128] += yb@w[:, start:start+128]
    outgoing = v.copy(order='F')
    outgoing *= decay[None]
    outgoing = solve(bottom, outgoing, overwrite_a=True, overwrite_b=True)
    del bottom
    # Rround = D(2 O-D). Form V(I-Rround) by row panels, retaining
    # neither the dense round trip nor a scaled copy of all magnetic modes.
    # Every mode participates, including arbitrarily small nonzero decays.
    hf = np.empty_like(v, order='F')
    for start in range(0, len(w), 128):
        rows = slice(start, start+128)
        np.matmul(v[rows]*decay[None], outgoing, out=hf[rows])
        hf[rows] *= -2
        hf[rows] += v[rows]*(1+decay**2)[None]
    del v
    top = splu(yt)
    for start in range(0, len(w), 128):
        columns = slice(start, start+128)
        hf[:, columns] = top.solve(hf[:, columns])
    del top
    # Add W(I+Rround) directly to the entrance system. Reflected fields
    # need only its action on one solution vector, not a saved Efront matrix.
    for start in range(0, len(w), 128):
        rows = slice(start, start+128)
        panel = (w[rows]*decay[None])@outgoing
        panel *= 2
        panel += w[rows]*(1-decay**2)[None]
        hf[rows] += panel
    del panel
    if progress:
        progress('matching_ready')
    incoming = np.zeros(len(w), complex)
    incoming[origin if polarization == 'p' else na+origin] = np.cos(THETA) if polarization == 'p' else 1.
    forward = 2*solve(np.asfortranarray(hf), incoming, overwrite_a=True)
    del hf
    bottom_modes = outgoing@forward
    transmitted = 2*(w@bottom_modes)
    reflected = w@(forward+decay*(2*bottom_modes-decay*forward))-incoming
    def flux(field, admittance):
        h = admittance@field
        return (np.vdot(field[:na], h[len(w)-na:])-np.vdot(field[na:], h[:len(w)-na])).imag
    return float(flux(transmitted, yb)/np.cos(THETA)), float(flux(reflected, yt)/np.cos(THETA))


def powers_from_sectors(make_sector, progress=None, *, propagate=sector_power, polarizations=('p', 's')):
    """Run explicitly selected exact sectors; by default return both."""
    polarizations = tuple(polarizations)
    if not polarizations or len(set(polarizations)) != len(polarizations) or any(pol not in ('p', 's') for pol in polarizations):
        raise ValueError('polarizations must be a nonempty selection of distinct p/s sectors')
    results, errors = {}, []
    for pol in polarizations:
        def report(stage, **values):
            if progress:
                progress(dict(polarization=pol, stage=stage, **values))
        report('closure_start')
        def prepare():
            sector = make_sector(pol)
            report('closure_ready')
            return sector
        power = propagate(prepare(), report)
        error = float(abs(sum(power)-1))
        if not np.isfinite(power).all() or error > 1e-9:
            raise FloatingPointError(f'invalid mirror power balance: {error}')
        results.update({pol+'_T': power[0], pol+'_R': power[1]})
        errors.append(error)
        report('power_ready', T=power[0], R=power[1], energy_error=error)
    return dict(**results, energy_error=max(errors))


def propagation_method(method, doublings=6, terms=12):
    """Explicit research propagation selection; no automatic method fallback."""
    if method == 'modal':
        return sector_power
    if method == 'doubling':
        from studies.rcwa.nvf_mirror_doubling.nvf_mirror_doubling import sector_doubling_power
        return lambda sector, progress: sector_doubling_power(sector, progress, doublings=doublings, terms=terms)
    raise ValueError('propagation must be modal or doubling')


def mirror_powers(coefficients, order, progress=None, *, propagation='modal', doublings=6, terms=12,
                  polarizations=('p', 's')):
    return powers_from_sectors(lambda pol: mirror_sector(coefficients, order, pol), progress,
                               propagate=propagation_method(propagation, doublings, terms), polarizations=polarizations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--projection', help='reuse a fixed coefficient export instead of solving a new cell')
    parser.add_argument('--save-projection', help='optional output NPZ for reproducing this same cell field')
    parser.add_argument('--orders', type=parse_order, nargs='+', default=[16, 24, 32])
    parser.add_argument('--gap', type=float, default=17.)
    parser.add_argument('--width', type=float, default=350.)
    parser.add_argument('--seams', type=float, nargs='*', default=[31.])
    parser.add_argument('--values', type=float, nargs='+', default=[4., 9.])
    parser.add_argument('--rows', type=int, default=8, help='new cell only: base body rows')
    parser.add_argument('--subdivisions', type=int, default=3, help='new cell only: x subdivisions')
    parser.add_argument('--body-degree', type=int, default=4, help='new cell only: y potential degree')
    parser.add_argument('--gap-degree', type=int, default=6, help='new cell only: gap potential degree')
    parser.add_argument('--edge-fraction', type=float, default=1e-5, help='new cell only: contact edge resolution')
    parser.add_argument('--flux-basis', choices=('global', 'local'), default='global', help='new cell only: flux coordinates')
    parser.add_argument('--grading-strength', type=float, default=8., help='new cell only: material density strength')
    parser.add_argument('--grading-radius', type=float, default=.01, help='new cell only: radius divided by envelope width')
    parser.add_argument('--progress', action='store_true', help='emit stage records during high-order calculations')
    parser.add_argument('--polarizations', choices=('p', 's'), nargs='+', default=['p', 's'],
                        help='explicit sector selection for costly directional checks; default both')
    parser.add_argument('--propagation', choices=('modal', 'doubling'), default='modal')
    parser.add_argument('--doublings', type=int, default=6)
    parser.add_argument('--terms', type=int, default=12)
    args = parser.parse_args()
    if args.projection:
        c, metadata = load_projection(args.projection, args.gap, args.width, args.seams, args.values)
    else:
        from studies.rcwa.nvf_polynomial_body.nvf_polynomial_body import PolynomialBodyGapModel
        maximum = max(max(orders_xy(order)) for order in args.orders)
        settings = dict(rows=args.rows, x_subdivisions=args.subdivisions, body_degree=args.body_degree,
                        gap_degree=args.gap_degree, edge_fraction=args.edge_fraction, flux_basis=args.flux_basis,
                        grading_strength=args.grading_strength, grading_radius=args.grading_radius)
        start = time.monotonic()
        c, info = PolynomialBodyGapModel(maximum, **settings).coefficients(
            args.gap, dict(seams=args.seams, values=args.values), args.width)
        metadata = dict(fixture='partitioned_gap', gap=args.gap, width=args.width,
                        seams=args.seams, values=args.values, **settings)
        print(json.dumps(dict(kind='mirror_projection', projection=metadata, cap=2*maximum,
                              cell_residual=float(info['cell_residual']), degrees_of_freedom=info['degrees_of_freedom'],
                              mirror_error=validate_mirror(c, maximum), seconds=time.monotonic()-start)), flush=True)
    if args.save_projection:
        np.savez(args.save_projection, coefficients=c, metadata=json.dumps(metadata))
    for order in args.orders:
        start = time.monotonic()
        def report(stage):
            print(json.dumps(dict(kind='mirror_progress', order=order, **stage,
                                  seconds=time.monotonic()-start)), flush=True)
        result = mirror_powers(c, order, report if args.progress else None, propagation=args.propagation,
                                doublings=args.doublings, terms=args.terms, polarizations=args.polarizations)
        print(json.dumps(dict(kind='mirror_gap', order=order, projection=metadata,
                              polarizations=args.polarizations,
                              propagation=args.propagation, doublings=args.doublings, terms=args.terms,
                              mirror_error=validate_mirror(c, order), **result, seconds=time.monotonic()-start)), flush=True)


if __name__ == '__main__':
    main()
