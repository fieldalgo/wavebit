"""Separate intrinsic Yee-grid scattering error from pulse/CPML contributions.

An independent sparse scalar Maxwell reference with outgoing Fourier boundaries.
This is a study instrument for the fixed nonmagnetic dielectric grating, not a
public general-purpose solver. Scene rasterization is shared; time updates,
source calibration, CPML and diffraction extraction are not used by the solve.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

import wavebit.fdtd._periodic.scene as scene
from wavebit._common.conventions import boundary_kz, yee_longitudinal_phase
from studies.fdtd.rayleigh.fdtd_rayleigh import BAND, STRUCTURE, THETA, PERIOD, reference
from wavebit.fdtd._periodic.legacy import cpml_profiles


from studies._support._fdtd_legacy import legacy_fdtd  # noqa: E402


def capture_grid(resolution, structure=STRUCTURE, *, pml_width=.6, pml_reflection=1e-40,
                 transverse_resolution=None):
    """Capture this study's static material arrays before any time-domain run.

    The reference run returns a placeholder, then the actual call captures its
    material arrays and stops. The original entry point is always restored.
    """
    original, captured, calls = scene.fdtd_grid, {}, 0
    class Captured(Exception):
        pass
    def capture(epsilon, **settings):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {}
        captured.update(epsilon=np.asarray(epsilon), **settings)
        raise Captured()
    scene.fdtd_grid = capture
    try:
        legacy_fdtd(
            structure=structure, thickness=[.2], top=1., bot=1., period=PERIOD,
            wavelength=BAND, theta=THETA, tetm='te', truncation_order=1,
            backend='numpy', resolution=resolution, n_pml=round(pml_width*resolution), n_periods=800,
            transverse_resolution=transverse_resolution,
            pulse_width=60., align_layers=True, diffraction_distance=.2,
            diffraction_method='directional', pml_reflection=pml_reflection, source_distance=1.)
    except Captured:
        pass
    finally:
        scene.fdtd_grid = original
    return captured


def _scalar_matrix(eps, omega, dx, dz, bloch, polarization, stretch_e, stretch_h, outgoing=None):
    """Scalar Maxwell stencil with optional stretched z derivatives.

    A supplied outgoing map closes the two exterior faces; otherwise z wraps
    periodically, as it does past both ends of the native CPML grid.
    """
    _, nj, nx = eps.shape
    matrix = lil_matrix((nj*nx, nj*nx), dtype=complex)
    for j in range(nj):
        for i in range(nx):
            row = j*nx+i
            if polarization == 'te':
                left = right = 1/dx**2
                up = stretch_e[j]*stretch_h[(j-1)%nj]/dz**2
                down = stretch_e[j]*stretch_h[j]/dz**2
                mass = omega**2*eps[1, j, i]
            else:
                left, right = 1/eps[2, j, i]/dx**2, 1/eps[2, j, (i+1)%nx]/dx**2
                up = stretch_h[j]*stretch_e[j]/eps[0, j, i]/dz**2
                down = stretch_h[j]*stretch_e[(j+1)%nj]/eps[0, (j+1)%nj, i]/dz**2
                mass = omega**2
            matrix[row, row] = mass-left-right-up-down
            matrix[row, j*nx+(i-1)%nx] += left/(bloch if i == 0 else 1.)
            matrix[row, j*nx+(i+1)%nx] += right*(bloch if i == nx-1 else 1.)
            if j or outgoing is None:
                matrix[row, ((j-1)%nj)*nx+i] += up
            else:
                matrix[row, :nx] += up*outgoing[i]
            if j < nj-1 or outgoing is None:
                matrix[row, ((j+1)%nj)*nx+i] += down
            else:
                matrix[row, -nx:] += down*outgoing[i]
    return matrix.tocsr()


def solve_harmonic(data, wavelength, polarization):
    """Solve scalar Yee Helmholtz equations with exact discrete exterior ghosts.

    TE uses Ey on z/x nodes; TM uses Hy on z/x half nodes. Material coefficients
    are the corresponding native tensor components, not a fitted effective index.
    The returned E/H are phasors at a common time; native H initialization needs
    its usual exp(+i*omega*dt/2) factor. Only orders -1 and 0 are returned as r/t.
    """
    if polarization not in ('te', 'tm'):
        raise ValueError('This reference supports scalar TE or TM only.')
    eps = data['epsilon'][:, :, 0, :]
    dz, _, dx = data['spacing']; dt = data['dt']; nx = eps.shape[-1]
    if not np.allclose(data['z_spacing'], dz) or not np.allclose(data['mu'], 1.):
        raise ValueError('This reference requires a uniform, nonmagnetic grid.')
    if not np.allclose(data['epsilon'], data['epsilon'][:, :, :1, :]) or data['bloch_phase'][1] != 0:
        raise ValueError('This reference requires y-invariant, nonconical fields.')
    jlo, jhi = data['monitor_z'][:2]; nj = jhi-jlo+1
    if not np.allclose(eps[:, [jlo, jlo+1, jhi, jhi+1]], 1.):
        raise ValueError('Both truncation planes must lie in uniform vacuum.')
    source_index = np.flatnonzero(np.any(abs(data['source']) > 0, axis=(0, 2, 3)))[0]
    origin = -1.-source_index*dz  # capture_grid fixes the source exactly at z=-1.
    kx0 = data['bloch_phase'][2]/(nx*dx)
    order = np.arange(nx)-nx//2
    kx = kx0+2*np.pi*order/(nx*dx)
    k0 = 2*np.pi/wavelength
    omega = 2*np.sin(k0*dt/2)/dt
    transverse = 2*np.sin(kx*dx/2)/dx
    q = boundary_kz(np, 1., ((transverse/omega)**2).astype(complex))
    phase = yee_longitudinal_phase(np, q, omega, dz)
    kz, multiplier = phase/dz, np.exp(1j*phase)
    offset = 0. if polarization == 'te' else .5
    x = (np.arange(nx)+offset)*dx
    transform = np.exp(1j*x[:, None]*kx)
    inverse = transform.conj().T/nx
    outgoing = (transform*multiplier)@inverse
    ztop, zbot = origin+(jlo+offset)*dz, origin+(jhi+offset)*dz
    incident = np.zeros(nx, complex)
    incident[nx//2] = np.exp(1j*kz[nx//2]*ztop)
    forcing = transform@((1/multiplier-multiplier)*incident)
    rhs = np.zeros(nj*nx, complex)
    bloch = np.exp(1j*kx0*nx*dx)
    matrix = _scalar_matrix(eps[:, jlo:jhi+1], omega, dx, dz, bloch, polarization,
                             np.ones(nj), np.ones(nj), outgoing)
    rhs[:nx] = -forcing/dz**2  # Both boundary rows are vacuum.
    solution = spsolve(matrix, rhs)
    if not np.isfinite(solution).all():
        raise ValueError('The harmonic reference is singular or unresolved at this frequency.')
    residual = np.linalg.norm(matrix@solution-rhs)/np.linalg.norm(rhs)
    field = solution.reshape(nj, nx)
    reflected = (inverse@field[0]-incident)*np.exp(1j*kz*ztop)
    transmitted = (inverse@field[-1])*np.exp(-1j*kz*(zbot-.2))
    selected = np.array([nx//2-1, nx//2])
    if polarization == 'tm':
        # Convert scalar Hy ratios into the scene's physical s/p electric-field
        # convention; its incident-order normalization is also discrete.
        physical = boundary_kz(np, 1., ((kx/k0)**2).astype(complex))
        factor = physical[nx//2]/physical*q/q[nx//2]
        reflected *= factor
        transmitted *= factor
    electric = np.zeros((3, nj, nx), complex)
    magnetic = np.zeros_like(electric)
    if polarization == 'te':
        electric[1] = field
        forward_x = np.roll(field, -1, axis=1); forward_x[:, -1] *= bloch
        forward_z = np.vstack((field[1:], outgoing@field[-1]))
        magnetic[0] = -(forward_z-field)/(1j*omega*dz)
        magnetic[2] = (forward_x-field)/(1j*omega*dx)
    else:
        magnetic[1] = field
        backward_x = np.roll(field, 1, axis=1); backward_x[:, 0] /= bloch
        backward_z = np.vstack((outgoing@field[0]+forcing, field[:-1]))
        electric[0] = (field-backward_z)/(1j*omega*dz*eps[0, jlo:jhi+1])
        electric[2] = -(field-backward_x)/(1j*omega*dx*eps[2, jlo:jhi+1])
    return dict(amplitudes=np.array([reflected[selected], transmitted[selected]]),
                relative_residual=float(residual), unknowns=nj*nx,
                electric=np.repeat(electric[:, :, None, :], 2, axis=2),
                magnetic=np.repeat(magnetic[:, :, None, :], 2, axis=2),
                epsilon=data['epsilon'][:, jlo:jhi+1], dt=dt, spacing=data['spacing'],
                bloch_phase=data['bloch_phase'])


def solve_cpml_harmonic(data, wavelength, polarization):
    """Eliminate CPML memory in frequency, then calibrate actual/reference sheets.

    This shares the native profile coefficients, but derives a separate sparse
    frequency equation. It separates finite-time error from error in the finite
    CPML model; it does not independently validate how profiles are generated.
    """
    eps = data['epsilon'][:, :, 0, :]
    dz, _, dx = data['spacing']; dt = data['dt']; nz, nx = eps.shape[1:]
    if (not np.allclose(data['z_spacing'], dz) or not np.allclose(data['mu'], 1.)
            or data['bloch_phase'][1] != 0 or polarization not in ('te', 'tm')):
        raise ValueError('CPML reference requires the uniform nonmagnetic scalar study grid.')
    widths = data['z_spacing']; nodes = np.r_[0., np.cumsum(widths)]
    cells = data['pml_cells']
    pe, ph = cpml_profiles(nodes, nodes[:-1]+widths/2, nodes[cells],
                           nodes[-1]-nodes[-cells-1], dt, data['pml_wavelength'], 1., 1.,
                           target_reflection=data['pml_reflection'])
    k0 = 2*np.pi/wavelength; omega = 2*np.sin(k0*dt/2)/dt
    previous_phase = np.exp(1j*k0*dt)
    stretch_e = (1/pe['kappa']+pe['a']/(1-pe['b']*previous_phase))[:-1]
    stretch_h = 1/ph['kappa']+ph['a']/(1-ph['b']*previous_phase)
    kx0 = data['bloch_phase'][2]/(nx*dx); bloch = np.exp(1j*kx0*nx*dx)
    order = np.arange(nx)-nx//2; kx = kx0+2*np.pi*order/(nx*dx)
    q = boundary_kz(np, 1., ((2*np.sin(kx*dx/2)/(dx*omega))**2).astype(complex))
    phase = yee_longitudinal_phase(np, q, omega, dz); kz = phase/dz
    offset = 0. if polarization == 'te' else .5
    x = (np.arange(nx)+offset)*dx
    transform = np.exp(1j*x[:, None]*kx); inverse = transform.conj().T/nx
    source_index = np.flatnonzero(np.any(abs(data['source']) > 0, axis=(0, 2, 3)))[0]
    origin = -1.-source_index*dz
    current = np.zeros((nz, nx), complex)
    current[source_index] = np.exp(1j*kx0*x)*np.exp(1j*k0*dt/2)
    solves, residuals = [], []
    for material in (np.ones_like(eps), eps):
        matrix = _scalar_matrix(material, omega, dx, dz, bloch, polarization, stretch_e, stretch_h)
        if polarization == 'te':
            rhs = -1j*omega*current
        else:
            scaled = current/material[0]
            rhs = -stretch_h[:, None]*(np.roll(scaled, -1, axis=0)-scaled)/dz
        solution = spsolve(matrix, rhs.ravel())
        residuals.append(np.linalg.norm(matrix@solution-rhs.ravel())/np.linalg.norm(rhs))
        field = solution.reshape(nz, nx)
        if polarization == 'te':
            electric = field
            magnetic = -stretch_h[:, None]*(np.roll(field, -1, axis=0)-field)/(1j*omega*dz)
        else:
            magnetic = field
            electric = (stretch_e[:, None]*(field-np.roll(field, 1, axis=0))/dz+current)/(1j*omega*material[0])
        planes = []
        for j in data['monitor_z'][:2]:
            e = inverse@electric[j]
            h = inverse@((magnetic[j]+magnetic[j-1])/2)
            difference = (-h/(q*np.cos(phase/2)) if polarization == 'te'
                          else q*h/np.cos(phase/2))
            planes.append(np.array([(e+difference)/2, (e-difference)/2]))
        solves.append(np.array(planes))
    reference_fields, actual = solves
    zr, zt = [origin+j*dz for j in data['monitor_z'][:2]]
    physical = boundary_kz(np, 1., ((kx/k0)**2).astype(complex))
    incident = reference_fields[0, 0, nx//2]*np.exp(-1j*kz[nx//2]*zr)
    if polarization == 'tm':
        incident *= -1/physical[nx//2]
    reflected = (actual[0, 1]-reference_fields[0, 1])*np.exp(1j*kz*zr)/incident
    transmitted = actual[1, 0]*np.exp(-1j*kz*(zt-.2))/incident
    incoming = actual[1, 1]*np.exp(1j*kz*(zt-.2))/incident
    if polarization == 'tm':
        reflected /= physical
        transmitted /= -physical
        incoming /= physical
    selected = [nx//2-1, nx//2]
    result = np.array([reflected[selected], transmitted[selected]])
    if not np.isfinite(result).all():
        raise ValueError('The CPML harmonic reference has unresolved incident calibration.')
    e = np.zeros((3, nz, nx), complex); h = np.zeros_like(e); source = np.zeros_like(e)
    if polarization == 'te':
        e[1], h[0] = electric, magnetic
        next_x = np.roll(electric, -1, axis=1); next_x[:, -1] *= bloch
        h[2] = (next_x-electric)/(1j*omega*dx)
        source[1] = current*np.exp(-1j*k0*dt/2)
    else:
        e[0], h[1] = electric, magnetic
        previous_x = np.roll(magnetic, 1, axis=1); previous_x[:, 0] /= bloch
        e[2] = -(magnetic-previous_x)/(1j*omega*dx*eps[2])
        source[0] = current*np.exp(-1j*k0*dt/2)
    return dict(amplitudes=result, incoming=incoming[selected],
                relative_residual=float(max(residuals)), unknowns=nz*nx,
                electric=np.repeat(e[:, :, None, :], 2, axis=2),
                magnetic=np.repeat(h[:, :, None, :], 2, axis=2),
                source=np.repeat(source[:, :, None, :], 2, axis=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resolution', type=int, nargs='+', default=[40, 80, 160, 320])
    parser.add_argument('--time-records', action='append', default=[],
                        help='Optional JSON-lines outputs from fdtd_source_clearance for direct amplitude comparison.')
    parser.add_argument('--cpml', action='store_true', help='Also solve the full finite CPML grid; use moderate resolutions.')
    parser.add_argument('--pml-width', type=float, default=.6, help='Physical CPML width for the optional finite-grid comparison.')
    parser.add_argument('--pml-reflection', type=float, default=1e-40, help='Nominal CPML amplitude target, not a measured error bound.')
    args = parser.parse_args()
    expected = reference(32)
    records = []
    for path in args.time_records:
        records.extend(json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())
    for resolution in args.resolution:
        data = capture_grid(resolution, pml_width=args.pml_width, pml_reflection=args.pml_reflection)
        result, amplitudes, cpml_amplitudes = {}, {}, {}
        for pol in ('te', 'tm'):
            solved = [solve_harmonic(data, wavelength, pol) for wavelength in BAND]
            values = np.stack([v['amplitudes'] for v in solved])
            amplitudes[pol] = values
            result[pol] = dict(amplitude_real=values.real.tolist(), amplitude_imag=values.imag.tolist(),
                               error_rcwa=float(np.max(abs(values-expected[pol]))),
                               relative_residual=max(v['relative_residual'] for v in solved),
                               unknowns=solved[0]['unknowns'])
            if args.cpml:
                cpml = [solve_cpml_harmonic(data, wavelength, pol) for wavelength in BAND]
                cpml_values = np.stack([v['amplitudes'] for v in cpml])
                cpml_amplitudes[pol] = cpml_values
                result[pol]['cpml'] = dict(
                    amplitude_real=cpml_values.real.tolist(), amplitude_imag=cpml_values.imag.tolist(),
                    error_rcwa=float(np.max(abs(cpml_values-expected[pol]))),
                    error_open=float(np.max(abs(cpml_values-values))),
                    incoming_max=float(np.max(abs(np.stack([v['incoming'] for v in cpml])))),
                    relative_residual=max(v['relative_residual'] for v in cpml))
        comparisons = []
        for row in records:
            if row.get('resolution') != resolution or 'amplitude_real' not in row:
                continue
            if row.get('method') != 'directional' or not np.isclose(row['dt'], data['dt'], rtol=1e-12):
                raise ValueError('Time comparison requires the same discrete step and directional extraction.')
            if not np.allclose(row.get('transverse_resolution', [resolution, resolution]),
                               [resolution, resolution], rtol=1e-12, atol=0):
                raise ValueError('Time comparison requires the same transverse resolution.')
            measured = {pol: np.array(row['amplitude_real'][pol])+1j*np.array(row['amplitude_imag'][pol]) for pol in expected}
            comparison = dict(case=row.get('case'), difference={
                pol: float(np.max(abs(measured[pol]-amplitudes[pol]))) for pol in expected})
            if args.cpml:
                same = (row.get('pad_scale') == 1. and row.get('source_distance') == 1.
                        and row.get('pml_reflection') == data['pml_reflection']
                        and np.isclose(row.get('pml_width', 0.), data['pml_cells']*data['spacing'][0]))
                if same:
                    comparison['difference_cpml'] = {pol: float(np.max(abs(measured[pol]-cpml_amplitudes[pol])))
                                                     for pol in expected}
            comparisons.append(comparison)
        print(json.dumps(dict(resolution=resolution, dt=data['dt'], pml_width=args.pml_width,
                              pml_reflection=args.pml_reflection, harmonic=result,
                              time_comparisons=comparisons), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
