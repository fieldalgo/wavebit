"""Separated conductive-stack instrument against an independent multilayer reference.

The two fixed cases are prospective until their six backend rows are disclosed.
This helper computes no order verdict and sets no optical acceptance ceiling.
Physical eps=a+i*sigma/omega and explicit host gaps feed the existing independent
characteristic matrix; production coefficient/profile helpers are never read.
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
from studies.fdtd.conductive_film.fdtd_conductive_film_probe import airy as single_airy


ENV_NAMES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'PYTHONUTF8')
WAVELENGTHS = (.9, 1., 1.1)
FAMILIES = ('lossy-pair', 'mixed-triple')
# Top-to-bottom tuples (real eps, conductivity, thickness), in vacuum.
FILMS = {'lossy-pair': ((4., 8., .10), (2.25, 4., .15)),
         'mixed-triple': ((3., 0., .10), (2., 6., .10), (5., 10., .15))}
GAPS = {'lossy-pair': (.15,), 'mixed-triple': (.125, .175)}
REFERENCE_DEPENDENCIES = ('studies/fdtd/layer_alignment/fdtd_layer_alignment.py',
                          'studies/fdtd/oblique_pulse/fdtd_oblique_pulse.py',
                          'studies/fdtd/conductive_film/fdtd_conductive_film_probe.py')


def specifications(family='lossy-pair', films=None, gaps=None):
    if family not in FAMILIES:
        raise ValueError('Unknown fixed conductive-stack family')
    films = tuple(tuple(float(v) for v in row) for row in (FILMS[family] if films is None else films))
    gaps = tuple(float(v) for v in (GAPS[family] if gaps is None else gaps))
    if len(films) < 2 or len(gaps) != len(films)-1:
        raise ValueError('At least two films and exactly one gap between successive films')
    if any(len(row) != 3 or not all(math.isfinite(v) for v in row)
           or row[0] <= 0 or row[1] < 0 or row[2] <= 0 for row in films):
        raise ValueError('Every film has finite positive eps/thickness and nonnegative sigma')
    if any(not math.isfinite(v) or v <= 0 for v in gaps):
        raise ValueError('Every explicit host gap is finite and positive')
    return films, gaps


def geometry(family='lossy-pair', films=None, gaps=None):
    films, gaps = specifications(family, films, gaps)
    depth, faces = 0., []
    for index, (_, _, thickness) in enumerate(films):
        faces.append((-depth, -depth-thickness))
        depth += thickness
        if index < len(gaps):
            depth += gaps[index]
    return tuple(faces), depth


def epsilon(wavelengths, *, family='lossy-pair', films=None, gaps=None):
    films, _ = specifications(family, films, gaps)
    omega = 2*math.pi/np.atleast_1d(np.asarray(wavelengths, dtype=float))
    return tuple(a+1j*sigma/omega for a, sigma, _ in films)


def reference(wavelengths, *, family='lossy-pair', theta=35., films=None, gaps=None, pol='s'):
    """Continuum outer-face r/t at fixed central kx; explicit vacuum gaps."""
    films, gaps = specifications(family, films, gaps)
    media = epsilon(wavelengths, family=family, films=films, gaps=gaps)
    layers, thicknesses = [], []
    for index, (material, (_, _, depth)) in enumerate(zip(media, films)):
        layers.append(material)
        thicknesses.append(depth)
        if index < len(gaps):
            layers.append(np.ones_like(material))
            thicknesses.append(gaps[index])
    if pol not in ('s', 'p'):
        raise ValueError('pol is s or p')
    return multilayer(wavelengths, layers, thicknesses,
                      theta=math.radians(theta))['te' if pol == 's' else 'tm']


def band_certificate(family, films=None, gaps=None):
    """Independent monotone bulk modulus bound; no effective-cell accuracy claim."""
    films, _ = specifications(family, films, gaps)
    width = math.sqrt(2*math.log(1e6))*.1
    low, high = 2*math.pi*(1-width), 2*math.pi*(1+width)
    if low <= 0:
        raise ValueError('The declared source band must stay positive')
    # |k|^4=a^2*w^4+sigma^2*w^2 increases for positive frequency.
    return {'angular_band': [low, high], 'member_modulus_bounds': [
        math.sqrt(math.hypot(a*high*high, sigma*high)) for a, sigma, _ in films],
        'host_modulus_bound': high}


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
        for a, sigma, _ in films:
            index = np.sqrt(a+1j*sigma/omega)
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
        for pol in ('s', 'p'):
            host = tuple((1., 0., h) for _, _, h in films)
            r, t = reference(wavelengths, family=family, films=host, pol=pol)
            close(f'{family}-{pol}-matched-r', r, 0.)
            close(f'{family}-{pol}-matched-t', t, np.exp(1j*kz0*depth))
            lossless = tuple((a, 0., h) for a, _, h in films)
            r, t = reference(wavelengths, family=family, films=lossless, pol=pol)
            close(f'{family}-{pol}-lossless-power', abs(r)**2+abs(t)**2, 1.)
            r, t = reference(wavelengths, family=family, pol=pol)
            assert np.all(abs(r)**2+abs(t)**2 < 1.)
            reverse_r, reverse_t = reference(wavelengths, family=family,
                films=films[::-1], gaps=gaps[::-1], pol=pol)
            close(f'{family}-{pol}-reciprocal-t', reverse_t, t)
            residuals[f'{family}-{pol}-reflection-asymmetry'] = float(np.max(abs(reverse_r-r)))
            assert residuals[f'{family}-{pol}-reflection-asymmetry'] > 1e-10
            for index, (a, sigma, h) in enumerate(films):
                material = a+1j*sigma/omega
                direct = multilayer(wavelengths, [material], [h], theta=math.radians(35.))[
                    'te' if pol == 's' else 'tm']
                close(f'{family}-{pol}-single-{index}', direct,
                      np.asarray(single_airy(wavelengths, eps=a, sigma=sigma,
                                            thickness=h, kx=kx, pol=pol)))
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


def build(n=40, pol='p', family='lossy-pair', theta=35., periods=120, pml=24,
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
        center=(0., 0., (top+bottom)/2)), fd.Conductor(eps=a, sigma=sigma))
        for (a, sigma, h), (top, bottom) in zip(films, faces))
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


def measure(*, backend='numpy', **kwargs):
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    before = source_identity()
    bound = inspect.signature(build).bind(**kwargs)
    bound.apply_defaults()
    config = dict(bound.arguments)
    config['films'], config['gaps'] = specifications(config['family'], config['films'], config['gaps'])
    scene, kx = build(**config)
    begin = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
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
    row = {'kind': 'conductive-stack-probe', 'started_utc': start, 'ended_utc': end,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--n', type=int, default=40)
    parser.add_argument('--family', choices=FAMILIES, default='lossy-pair')
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
    row = preflight() if preflight_only else measure(**args)
    encoded = json.dumps(_json(row), indent=2, allow_nan=False)
    if output is not None:
        with output.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(encoded+'\n')
    print(encoded)


if __name__ == '__main__':
    main()
