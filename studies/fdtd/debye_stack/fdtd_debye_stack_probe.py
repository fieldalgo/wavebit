"""Separated passive Debye-stack instrument with independent continuum controls.

The fixed cases are prospective until their six backend rows are disclosed.
No order verdict or optical ceiling is selected here. Declared Debye parameters,
film depths and explicit vacuum gaps feed the characteristic-matrix reference.
A separate backward scattering recursion and a generic single-film Airy formula
check that optical oracle before any field solve; neither reads production
profiles, transformed poles, recurrence coefficients or prepared geometry.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
import argparse
import datetime
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import pathlib
import sys
import time
import warnings

import numpy as np
import wavebit.fdtd._legacy as fd

from studies.fdtd.layer_alignment.fdtd_layer_alignment import multilayer


ENV_NAMES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'PYTHONUTF8')
WAVELENGTHS = (.9, 1., 1.1)
FAMILIES = ('debye-pair', 'debye-triple')
# Top-to-bottom tuples (eps_inf, declared delta_eps values, taus, thickness).
FILMS = {
    'debye-pair': ((2.25, (3.,), (.08,), .10),
                  (1.8, (1.5, 2.), (.04, .16), .15)),
    'debye-triple': ((2., (.8, 1.1, .7), (.035, .11, .3), .10),
                    (2.4, (.5, 0., .9, .6), (.02, .04, .08, .25), .15),
                    (3., (0.,), (.1,), .10)),
}
GAPS = {'debye-pair': (.15,), 'debye-triple': (.125, .175)}
REFERENCE_DEPENDENCIES = ('studies/fdtd/layer_alignment/fdtd_layer_alignment.py',
                          'studies/fdtd/oblique_pulse/fdtd_oblique_pulse.py')


def specifications(family='debye-pair', films=None, gaps=None):
    if family not in FAMILIES:
        raise ValueError('Unknown fixed Debye-stack family')
    declared = FILMS[family] if films is None else films
    if any(len(row) != 4 for row in declared):
        raise ValueError('Every film has eps_inf, delta_eps tuple, tau tuple and thickness')
    films = tuple((float(a), tuple(float(d) for d in deltas),
                   tuple(float(t) for t in taus), float(h))
                  for a, deltas, taus, h in declared)
    gaps = tuple(float(v) for v in (GAPS[family] if gaps is None else gaps))
    if len(films) < 2 or len(gaps) != len(films)-1:
        raise ValueError('At least two films and exactly one gap between successive films')
    for a, deltas, taus, h in films:
        if (not math.isfinite(a) or a <= 0 or not math.isfinite(h) or h <= 0
                or not deltas or len(deltas) != len(taus)
                or any(not math.isfinite(d) or d < 0 for d in deltas)
                or any(not math.isfinite(t) or t <= 0 for t in taus)):
            raise ValueError('Finite positive eps/thickness/tau and nonnegative Debye strengths required')
    if any(not math.isfinite(v) or v <= 0 for v in gaps):
        raise ValueError('Every explicit host gap is finite and positive')
    return films, gaps


def geometry(family='debye-pair', films=None, gaps=None):
    films, gaps = specifications(family, films, gaps)
    depth, faces = 0., []
    for index, (_, _, _, thickness) in enumerate(films):
        faces.append((-depth, -depth-thickness))
        depth += thickness
        if index < len(gaps):
            depth += gaps[index]
    return tuple(faces), depth


def epsilon(wavelengths, *, family='debye-pair', films=None, gaps=None):
    """Declared continuum epsilon for exp(-i omega t); no discrete material data."""
    films, _ = specifications(family, films, gaps)
    omega = 2*math.pi/np.atleast_1d(np.asarray(wavelengths, dtype=float))
    return tuple(a+sum((d/(1-1j*omega*t) for d, t in zip(deltas, taus)),
                       np.zeros_like(omega, dtype=complex))
                 for a, deltas, taus, _ in films)


def layers(wavelengths, *, family='debye-pair', films=None, gaps=None):
    """Only the declared continuum media and physical depths, including every gap."""
    films, gaps = specifications(family, films, gaps)
    media = epsilon(wavelengths, family=family, films=films, gaps=gaps)
    materials, thicknesses = [], []
    for index, (material, (_, _, _, depth)) in enumerate(zip(media, films)):
        materials.append(material)
        thicknesses.append(depth)
        if index < len(gaps):
            materials.append(np.ones_like(material))
            thicknesses.append(gaps[index])
    return tuple(materials), tuple(thicknesses)


def reference(wavelengths, *, family='debye-pair', theta=35., films=None, gaps=None, pol='s'):
    """Continuum outer-face r/t at fixed central kx; explicit vacuum gaps."""
    materials, thicknesses = layers(wavelengths, family=family, films=films, gaps=gaps)
    if pol not in ('s', 'p'):
        raise ValueError('pol is s or p')
    return multilayer(wavelengths, materials, thicknesses,
                      theta=math.radians(theta))['te' if pol == 's' else 'tm']


def recursive(wavelengths, materials, thicknesses, *, kx, pol):
    """Independent backward Fresnel recursion, in tangential-E amplitudes first.

    Each step composes an interface with a propagated suffix. Its denominator
    is 1+r_interface*r_suffix*p**2 because the reverse-interface reflection
    is -r_interface. The final p reflection changes sign for Wavebit's basis.
    """
    if pol not in ('s', 'p') or len(materials) != len(thicknesses):
        raise ValueError('Matched media/depth lengths and s/p polarization required')
    omega = 2*math.pi/np.atleast_1d(np.asarray(wavelengths, float))
    media = [np.ones_like(omega, dtype=complex)]
    media += [np.broadcast_to(np.asarray(e, complex), omega.shape) for e in materials]
    media.append(np.ones_like(omega, dtype=complex))
    q = [np.sqrt(e-(kx/omega)**2+0j) for e in media]
    admittance = q if pol == 's' else [e/normal for e, normal in zip(media, q)]
    r, t = np.zeros_like(omega, dtype=complex), np.ones_like(omega, dtype=complex)
    for j in range(len(media)-2, -1, -1):
        ri = (admittance[j]-admittance[j+1])/(admittance[j]+admittance[j+1])
        ti = 2*admittance[j]/(admittance[j]+admittance[j+1])
        depth = 0. if j == len(materials) else thicknesses[j]
        p = np.exp(1j*omega*q[j+1]*depth)
        denominator = 1+ri*r*p*p
        r, t = (ri+r*p*p)/denominator, ti*t*p/denominator
    return np.array([r if pol == 's' else -r, t])


def single_airy(wavelengths, material, thickness, *, kx, pol):
    """Closed complex single-slab formula from only continuum epsilon and depth."""
    omega = 2*math.pi/np.atleast_1d(np.asarray(wavelengths, float))
    eps = np.broadcast_to(np.asarray(material, complex), omega.shape)
    q0, q1 = np.sqrt(1-(kx/omega)**2+0j), np.sqrt(eps-(kx/omega)**2+0j)
    y0, y1 = (q0, q1) if pol == 's' else (1/q0, eps/q1)
    ri = (y0-y1)/(y0+y1)
    p = np.exp(1j*omega*q1*thickness)
    denominator = 1-ri*ri*p*p
    r, t = ri*(1-p*p)/denominator, (1-ri*ri)*p/denominator
    return np.array([r if pol == 's' else -r, t])


def band_certificate(family, films=None, gaps=None):
    """Independent sampled monotone endpoint; a bulk scale check, not accuracy."""
    films, _ = specifications(family, films, gaps)
    width = math.sqrt(2*math.log(1e6))*.1
    low, high = 2*math.pi*(1-width), 2*math.pi*(1+width)
    if low <= 0:
        raise ValueError('The declared source band must stay positive')
    # Positive Debye sums make omega**4*|epsilon(omega)|**2 monotone.
    # This independent floating evaluation is not the production rational enclosure.
    bounds = [high*math.sqrt(abs(a+sum(d/(1-1j*high*t) for d, t in zip(ds, ts))))
              for a, ds, ts, _ in films]
    return {'angular_band': [low, high], 'member_modulus_bounds': bounds,
            'host_modulus_bound': high, 'qualification': 'ordinary floating endpoint evaluation'}


def preflight():
    """Analytic/geometry controls only, to execute before any field response."""
    wavelengths = np.asarray(WAVELENGTHS)
    omega = 2*math.pi/wavelengths
    kx = 2*math.pi*math.sin(math.radians(35.))
    kz0 = np.sqrt(omega**2-kx**2)
    residuals, bounds, supports, sampled = {}, {}, {}, {}
    def close(name, actual, expected):
        residuals[name] = float(np.max(abs(np.asarray(actual)-expected)))
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)
    close('fixed-kx-transverse', np.sin(math.radians(35.))*wavelengths, kx/omega)
    reference_theta = np.degrees(np.arcsin(kx/omega))
    for family in FAMILIES:
        films, gaps = specifications(family)
        faces, depth = geometry(family)
        bounds[family] = band_certificate(family)
        assert max(bounds[family]['member_modulus_bounds']+[bounds[family]['host_modulus_bound']])/40 < 2*math.pi/8
        sampled[family] = []
        for eps in epsilon(wavelengths, family=family):
            index = np.sqrt(eps)
            phase_cells = 40*2*math.pi/(omega*index.real)
            skin_cells = [None if v == 0 else float(40/(w*v)) for w, v in zip(omega, index.imag)]
            assert np.all(phase_cells >= 8) and all(v is None or v >= 1 for v in skin_cells)
            sampled[family].append({'phase_cells': phase_cells.tolist(), 'skin_cells': skin_cells})
        supports[family] = []
        for n in (40, 80, 160):
            dx = 1./n
            for fraction in (.3, .1):
                low = -(math.ceil((depth+.8)/dx)+24+fraction)*dx
                spans = [(math.floor((bottom-low)/dx)-1, math.ceil((top-low)/dx)+1)
                         for top, bottom in faces]
                ordered = sorted(spans)
                assert all(a[1] <= b[0] for a, b in zip(ordered, ordered[1:]))
                phases = [((face-low)/dx) % 1 for pair in faces for face in pair]
                close(f'{family}-n{n}-f{fraction}-phases', phases, fraction)
                supports[family].append({'n': n, 'fraction': fraction, 'spans': spans,
                                         'face_fractions': phases})
        materials, thicknesses = layers(wavelengths, family=family)
        for pol in ('s', 'p'):
            host = tuple((1., tuple(0. for _ in ds), ts, h) for _, ds, ts, h in films)
            r, t = reference(wavelengths, family=family, films=host, pol=pol)
            close(f'{family}-{pol}-matched-r', r, 0.)
            close(f'{family}-{pol}-matched-t', t, np.exp(1j*kz0*depth))
            lossless = tuple((a, tuple(0. for _ in ds), ts, h) for a, ds, ts, h in films)
            r, t = reference(wavelengths, family=family, films=lossless, pol=pol)
            close(f'{family}-{pol}-lossless-power', abs(r)**2+abs(t)**2, 1.)
            r, t = reference(wavelengths, family=family, pol=pol)
            assert np.all(abs(r)**2+abs(t)**2 < 1.)
            close(f'{family}-{pol}-recursive', np.array([r, t]),
                  recursive(wavelengths, materials, thicknesses, kx=kx, pol=pol))
            reverse_r, reverse_t = reference(wavelengths, family=family,
                films=films[::-1], gaps=gaps[::-1], pol=pol)
            close(f'{family}-{pol}-reciprocal-t', reverse_t, t)
            residuals[f'{family}-{pol}-reflection-asymmetry'] = float(np.max(abs(reverse_r-r)))
            assert residuals[f'{family}-{pol}-reflection-asymmetry'] > 1e-10
            reversed_media, reversed_depths = layers(wavelengths, family=family,
                films=films[::-1], gaps=gaps[::-1])
            close(f'{family}-{pol}-reverse-recursive', np.array([reverse_r, reverse_t]),
                  recursive(wavelengths, reversed_media, reversed_depths, kx=kx, pol=pol))
            for index, (material, (_, _, _, h)) in enumerate(zip(epsilon(wavelengths, family=family), films)):
                direct = multilayer(wavelengths, [material], [h], theta=math.radians(35.))[
                    'te' if pol == 's' else 'tm']
                close(f'{family}-{pol}-single-{index}', direct,
                      single_airy(wavelengths, material, h, kx=kx, pol=pol))
                close(f'{family}-{pol}-zero-depth-{index}',
                      multilayer(wavelengths, [material, np.ones_like(material)], [0., .15],
                                 theta=math.radians(35.))['te' if pol == 's' else 'tm'],
                      np.array([np.zeros_like(kz0), np.exp(1j*kz0*.15)]))
        rs, ts = reference(wavelengths, family=family, theta=0., pol='s')
        rp, tp = reference(wavelengths, family=family, theta=0., pol='p')
        close(f'{family}-normal-r-sign', rp, -rs)
        close(f'{family}-normal-t', tp, ts)
    return {'analytic_controls': 'passed', 'residuals': residuals, 'bulk_bounds': bounds,
            'support_controls': supports, 'sampled_cells': sampled,
            'reference_theta_at': reference_theta.tolist(), 'field_solves': 0}


def build(n=40, pol='p', family='debye-pair', theta=35., periods=120, pml=24,
          fraction=.3, courant=.5, films=None, gaps=None):
    films, gaps = specifications(family, films, gaps)
    faces, depth = geometry(family, films, gaps)
    dx = 1./n
    kx = 2*math.pi*math.sin(math.radians(theta))
    source = fd.PlaneWave('pw', z=.6, direction='-z', theta=theta, phi=0.,
        polarization=pol, band=fd.Band(wavelength=1., df=.1, cutoff=1e-6))
    low = -(math.ceil((depth+.8)/dx)+pml+fraction)*dx
    nz = math.ceil((1.4+pml*dx-low)/dx)
    structures = tuple(fd.Structure(fd.Box(size=(5*dx, 5*dx, h),
        center=(0., 0., (top+bottom)/2)),
        fd.Dispersive(eps_inf=a, poles=tuple(fd.Debye(delta_eps=d, tau=t)
                                           for d, t in zip(deltas, taus))))
        for (a, deltas, taus, h), (top, bottom) in zip(films, faces))
    monitors = (
        fd.Diffraction('r', z=1.1, side='reflect', wavelengths=WAVELENGTHS,
                       reference_plane=0., orders=(0, 0)),
        fd.Diffraction('t', z=-depth-.5, side='transmit', wavelengths=WAVELENGTHS,
                       reference_plane=-depth, incident_plane=0., orders=(0, 0)))
    scene = fd.Scene(domain=fd.Domain(size=(dx, dx, nz*dx), center=(0., 0., low+nz*dx/2)),
        resolution=fd.Resolution(dx=dx),
        boundaries=fd.Boundaries(x=fd.Axis1D(k=kx), y=fd.Periodic(),
            z=fd.PML(cells=pml, wavelength=1., reflection=1e-12)),
        structures=structures, sources=(source,), monitors=monitors,
        run=fd.Run(periods=periods, courant=courant))
    return scene, kx


def _numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _json(value):
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, complex):
        return [value.real, value.imag]
    if isinstance(value, np.generic):
        return _json(value.item())
    return value


def source_identity():
    helper = pathlib.Path(__file__).resolve()
    root = next(parent for parent in helper.parents
                if (parent/'pyproject.toml').is_file() and (parent/'src/wavebit').is_dir())
    package = root/'src/wavebit'
    def digest(folder):
        value = hashlib.sha256()
        for path in sorted(folder.rglob('*.py')):
            value.update(path.relative_to(folder).as_posix().encode('utf-8')+b'\0')
            value.update(path.read_bytes().replace(b'\r\n', b'\n')+b'\0')
        return value.hexdigest()
    return {'helper_sha256': hashlib.sha256(helper.read_bytes()).hexdigest(),
            'fd_source_sha256': digest(package/'fdtd'), 'wavebit_source_sha256': digest(package),
            'reference_dependencies': {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                                       for name in REFERENCE_DEPENDENCIES},
            'fd_digest_rule': 'sorted relative POSIX path + NUL + LF-normalized bytes + NUL'}


def runtime_identity():
    return {'python': sys.version, 'executable': sys.executable,
            'backend_versions': {name: importlib.metadata.version(name)
                                 for name in ('numpy', 'jax', 'jaxlib', 'torch')},
            'environment': {name: os.environ.get(name) for name in ENV_NAMES}}


def _measure(*, backend='numpy', caught, **kwargs):
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    before = source_identity()
    bound = inspect.signature(build).bind(**kwargs)
    bound.apply_defaults()
    config = dict(bound.arguments)
    config['films'], config['gaps'] = specifications(config['family'], config['films'], config['gaps'])
    scene, kx = build(**config)
    begin = time.perf_counter()
    result = fd.run(scene, backend=backend)
    duration = time.perf_counter()-begin
    end = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if before != source_identity():
        raise RuntimeError('Source/reference dependency changed during probe; preserve the attempt.')
    pol = config['pol']
    r = _numpy(getattr(result['r'].orders.order(0, 0), 'r_'+pol))
    t = _numpy(getattr(result['t'].orders.order(0, 0), 't_'+pol))
    reference_r, reference_t = reference(WAVELENGTHS, **{name: config[name] for name in
        ('family', 'films', 'gaps', 'theta', 'pol')})
    reflect = _numpy(result['r'].orders.order(0, 0).R)
    transmit = _numpy(result['t'].orders.order(0, 0).T)
    energy_t, energy = result.diagnostics['interior_energy']
    energy = _numpy(energy)
    faces, depth = geometry(config['family'], config['films'], config['gaps'])
    reference_theta = np.degrees(np.arcsin(kx/(2*math.pi/np.asarray(WAVELENGTHS))))
    np.testing.assert_allclose(_numpy(result['r'].theta_at), reference_theta, atol=2e-12, rtol=0.)
    row = {'kind': 'Debye-stack-probe', 'started_utc': start, 'ended_utc': end,
        'wall_s': duration, 'source_identity': before, 'source_unchanged_during_run': True,
        **runtime_identity(), 'config': {'backend': backend, **config}, 'grid': result.plan.shape,
        'source_band': {'wavelength': 1., 'df': .1, 'cutoff': 1e-6},
        'dt': result.plan.dt, 'steps': result.plan.steps, 'kx': kx, 'wavelengths': WAVELENGTHS,
        'theta_at': _numpy(result['r'].theta_at), 'reference_theta_at': reference_theta,
        'film_faces': faces, 'total_depth': depth,
        'outer_faces': [0., -depth], 'grid_low': result.plan.low,
        'epsilon': epsilon(WAVELENGTHS, **{name: config[name] for name in ('family', 'films', 'gaps')}),
        'r': r, 't': t, 'reference_r': reference_r, 'reference_t': reference_t,
        'error_r': abs(r-reference_r), 'error_t': abs(t-reference_t),
        'R': reflect, 'T': transmit, 'absorption': 1-reflect-transmit,
        'reference_absorption': 1-abs(reference_r)**2-abs(reference_t)**2,
        'stable': result.diagnostics['stable'], 'decay': result.diagnostics['decay'],
        'energy_shape': energy.shape, 'energy_rows': len(energy_t),
        'energy_finite': bool(np.all(np.isfinite(energy))), 'energy_max': float(np.max(energy)),
        'bytes_per_state': result.report.bytes_per_state,
        'warnings': [{'category': type(w.message).__name__, 'message': str(w.message)} for w in caught]}
    return _json(row)


class ProbeFailure(RuntimeError):
    """Preserve captured warnings even when construction, solve or postprocessing fails."""

    def __init__(self, row):
        self.row = row
        super().__init__(row['error'])


def measure(*, backend='numpy', **kwargs):
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            return _measure(backend=backend, caught=caught, **kwargs)
        except Exception as error:
            row = {'kind': 'Debye-stack-probe-failure', 'started_utc': start,
                   'ended_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'config': {'backend': backend, **kwargs}, 'error': repr(error),
                   'warnings': [{'category': type(w.message).__name__, 'message': str(w.message)}
                                for w in caught]}
            raise ProbeFailure(row) from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--n', type=int, default=40)
    parser.add_argument('--family', choices=FAMILIES, default='debye-pair')
    parser.add_argument('--pol', choices=('s', 'p'), default='p')
    for name, default in (('theta', 35.), ('fraction', .3), ('courant', .5)):
        parser.add_argument('--'+name, type=float, default=default)
    parser.add_argument('--films', type=json.loads)
    parser.add_argument('--gaps', type=json.loads)
    parser.add_argument('--periods', type=int, default=120)
    parser.add_argument('--pml', type=int, default=24)
    parser.add_argument('--backend', choices=('numpy', 'jax', 'torch'), default='numpy')
    parser.add_argument('--output', type=pathlib.Path)
    args = vars(parser.parse_args())
    preflight_only, output = args.pop('preflight'), args.pop('output')
    failed = None
    try:
        row = preflight() if preflight_only else measure(**args)
    except Exception as error:
        failed = error
        row = error.row if isinstance(error, ProbeFailure) else {
            'kind': 'Debye-stack-preflight-failure', 'error': repr(error),
            'ended_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    encoded = json.dumps(_json(row), indent=2, allow_nan=False)
    if output is not None:
        with output.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(encoded+'\n')
    print(encoded, flush=True)
    if failed is not None:
        raise failed


if __name__ == '__main__':
    main()
