"""Separate radiation-transform error from field-solver error using exact Mie data.

No FDTD step, material coefficient or incident-field normalization is used.
The production box layout and radiation transform consume exact exterior
fields at their own Yee points. A diagnostic collocated-H arm isolates the
two-plane H average. It supplies unavailable exact values, not an algorithm
that can be applied to a computed field. See docs/fd-mie-validation.md.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tests')]
import numpy as np
from references.mie import sphere_efficiencies
from references.mie_fields import exterior_scattered, far_amplitude
from wavebit.fd._kernel.grid import Grid, OFFSETS
from wavebit.fd._lower.monitors import box_faces
from wavebit.fd._runtime import runtime
from wavebit.fd._post import far_field, box_flux
from wavebit.fd.geometry import Region


# Existing C3 inputs, not acceptance thresholds. The three surface sizes
# are the previously measured phase-control surfaces.
PLACEMENTS = ((0., 0., 0.), (.5, 1/3, .2), (.25, 2/3, .4))
HALF_WIDTHS = (.65, .75, .85)
THETA = np.linspace(0., 180., 37)[:, None]
PHI = np.array((0., 90.))[None, :]


def records(cpw, eps=1.1, half=.75, offset=(0., 0., 0.), collocated=False,
            radius=.25):
    """Exact phasors in the unchanged production monitor accumulator layout."""
    dx = 1/cpw
    # Same grid origin as sphere_scene; no plan or source is constructed.
    grid = Grid.from_domain((2*(.75+.2+12*dx),)*3, (0.,)*3, dx)
    faces = box_faces('exact Mie', grid, Region(size=(2*half,)*3))
    center = np.asarray(offset)*dx
    coordinates = []
    for name, sign, layout in faces:
        for label, field, slab in layout.parts():
            axes = [grid.low[d]+(np.arange(*rr)+.5*OFFSETS[field][slab.c][d])*dx
                    for d, rr in enumerate((slab.x, slab.y, slab.z))]
            if collocated and field == 'H':
                # Diagnostic exact H at the E plane, duplicated for _mean_planes.
                axes[layout.P] = np.full(2, grid.low[layout.P]+layout.k*dx)
            z, y, x = np.meshgrid(axes[2], axes[1], axes[0], indexing='ij')
            coordinates.append((name, label, field, slab.c, np.stack((x, y, z), axis=-1)))
    # Host-only whole-surface check before evaluating any part.
    if any(np.min(np.linalg.norm(p-center, axis=-1)) <= radius
           for _, _, _, _, p in coordinates):
        raise ValueError('ExteriorMieSurfaceRefused: all samples must be outside the sphere.')
    accs = {name: {} for name, _, _ in faces}
    for name, label, field, c, points in coordinates:
        fields = exterior_scattered(points, eps, radius, center)
        accs[name][label] = fields[field == 'H'][..., c][None]
    return grid, faces, accs


def gauss_surface(eps, half, theta, phi, order, center=(0., 0., 0.), radius=.25):
    """Independent collocated continuum surface integral in Cartesian vectors.

    Tensor Gauss quadrature, no production layouts, half-plane averaging,
    surface-transform helper, or cyclic-component indexing is used.
    """
    t, p = np.broadcast_arrays(np.radians(theta), np.radians(phi))
    rhat = np.stack((np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t)), axis=-1)
    that = np.stack((np.cos(t)*np.cos(p), np.cos(t)*np.sin(p), -np.sin(t)), axis=-1)
    phat = np.stack((-np.sin(p), np.cos(p), 0*p), axis=-1)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    weights = (half**2*weights[:, None]*weights[None, :]).ravel()
    q, v = np.meshgrid(half*nodes, half*nodes, indexing='ij')
    N = np.zeros(rhat.shape, dtype=complex)
    L = np.zeros_like(N)
    for axis in range(3):
        transverse = [a for a in range(3) if a != axis]
        for sign in (-1., 1.):
            points = np.zeros((order**2, 3))
            points[:, axis] = sign*half
            points[:, transverse[0]], points[:, transverse[1]] = q.ravel(), v.ravel()
            E, H = exterior_scattered(points, eps, radius, center)
            normal = np.eye(3)[axis]*sign
            phase = np.exp(-2j*np.pi*np.einsum('...c,pc->...p', rhat, points))*weights
            N += np.einsum('...p,pc->...c', phase, np.cross(normal, H))
            L += np.einsum('...p,pc->...c', phase, -np.cross(normal, E))
    radiation = .5j*(N-np.sum(N*rhat, axis=-1)[..., None]*rhat+np.cross(L, rhat))
    return np.sum(radiation*that, axis=-1), np.sum(radiation*phat, axis=-1)


def measure(cpw, eps=1.1, half=.75, offset=(0., 0., 0.), collocated=False, radius=.25):
    rt = runtime('numpy')
    grid, faces, accs = records(cpw, eps, half, offset, collocated, radius)
    et, ep, intensity = far_field(rt, accs, faces, grid, (1.,), 1., 1., THETA, PHI)
    center = np.asarray(offset)/cpw
    refet, refep = far_amplitude(THETA, PHI, eps, radius, center)
    ref_intensity = .5*(abs(refet)**2+abs(refep)**2)
    qexact = sphere_efficiencies(np.sqrt(eps), 2*np.pi*radius)[1]
    forward, exact_forward = complex(et[0, 0, 0]), complex(refet[0, 0])
    flux = float(box_flux(rt, accs, faces, grid.dx)[0][0])
    qflux = 2*flux/(np.pi*radius**2)  # Unit E0 has incident intensity 1/2.
    optical = 2*forward.imag/(np.pi*radius**2)  # 4*pi/k = 2 at wavelength 1.
    return dict(cpw=cpw, eps=eps, radius=radius, half=half,
        actual_faces={name: grid.low[lay.P]+lay.k*grid.dx for name, _, lay in faces},
        collocated_H=collocated, offset=tuple(offset),
        forward=[forward.real, forward.imag], reference_forward=[exact_forward.real, exact_forward.imag],
        forward_error=abs(forward/exact_forward-1), phase_degrees=float(np.degrees(np.angle(forward/exact_forward))),
        qflux=qflux, qexact=float(qexact), optical=optical,
        optical_vs_exact=optical/qexact-1, power_vs_exact=qflux/qexact-1,
        optical_closure=optical/qflux-1,
        peak_error=float(np.max(abs(intensity[0]-ref_intensity))/np.max(ref_intensity)),
        complex_peak_error=float(max(np.max(abs(et[0]-refet)), np.max(abs(ep[0]-refep)))
                                 /max(np.max(abs(refet)), np.max(abs(refep)))))


def main():
    import argparse
    import hashlib
    import json
    import time
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--cells', default='20,28,40')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('DuplicateMeasurementRefused')
    watched = sorted((ROOT/'src/wavebit/fd').rglob('*.py'))+[
        Path(__file__), ROOT/'tests/references/mie.py', ROOT/'tests/references/mie_fields.py']
    def hashes():
        return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in watched}
    before = hashes()
    out = dict(status='running_analytic_diagnostic', source_sha256=before, rows=[])
    def save():
        args.output.write_text(json.dumps(out, indent=2)+'\n', encoding='utf8')
    save()
    try:
        for eps in (1.1, 2.25, 12.):
            for cpw in map(int, args.cells.split(',')):
                for index, offset in enumerate(PLACEMENTS):
                    for half in HALF_WIDTHS:
                        for collocated in (False, True):
                            begin = time.perf_counter()
                            row = measure(cpw, eps, half, offset, collocated)
                            row.update(placement_index=index, seconds=time.perf_counter()-begin)
                            out['rows'].append(row)
                    if hashes() != before:
                        raise ValueError('MeasurementSourceChangedRefused')
                    save()
                    print(f'epsilon={eps:g} cpw={cpw} placement={index}: six measurements completed', flush=True)
        out.update(status='completed_analytic_diagnostic_not_C3_pass', source_unchanged=hashes()==before)
        save()
    except Exception as error:
        out.update(status='stopped_for_review', failure=f'{type(error).__name__}: {error}', source_unchanged=hashes()==before)
        save()
        raise


if __name__ == '__main__':
    main()
