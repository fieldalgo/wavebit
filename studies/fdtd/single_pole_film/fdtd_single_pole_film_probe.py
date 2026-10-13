"""Bounded single-pole film instrument with an independent complex Airy reference.

This helper alone makes no convergence claim. Every row retains concrete settings,
complex coefficients, scalar energy/decay diagnostics and before/after source hashes.
Use the calibration/controller tools to preserve provenance before field responses.
"""
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


WAVELENGTHS = (.9, 1., 1.1)
FAMILIES = ('drude', 'lorentz')
ENV_NAMES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'PYTHONUTF8')


def epsilon(wavelengths, *, family='drude', eps=1., gamma=1., omega_p=9., omega0=8., strength=40.):
    """Independent rational continuum formula; no production pole/material helper."""
    omega = 2*math.pi/np.asarray(wavelengths, dtype=float)
    if family == 'drude':
        return eps-omega_p**2/(omega**2+1j*gamma*omega)
    if family == 'lorentz':
        if strength == 0:
            return np.full(omega.shape, complex(eps))
        return eps+strength/(omega0**2-omega**2-1j*gamma*omega)
    raise ValueError('family is drude or lorentz')


def airy(wavelengths, *, family='drude', eps=1., gamma=1., omega_p=9., omega0=8.,
         strength=40., thickness=.2, kx=0., pol='s'):
    omega = 2*math.pi/np.asarray(wavelengths, dtype=float)
    own = epsilon(wavelengths, family=family, eps=eps, gamma=gamma,
                  omega_p=omega_p, omega0=omega0, strength=strength)
    kz0 = np.sqrt(omega**2-kx**2+0j)
    kz = np.sqrt(own*omega**2-kx**2+0j)
    kz = np.where((kz.imag < 0) | ((kz.imag == 0) & (kz.real < 0)), -kz, kz)
    if pol == 's':
        y0, y = kz0/omega, kz/omega
    elif pol == 'p':
        y0, y = omega/kz0, own*omega/kz
    else:
        raise ValueError('pol is s or p')
    rho = (y0-y)/(y0+y)
    phase = np.exp(1j*kz*thickness)
    denominator = 1-rho**2*phase**2
    r, t = rho*(1-phase**2)/denominator, (1-rho**2)*phase/denominator
    return (-r if pol == 'p' else r), t


def preflight():
    """Analytic identities only, without a time-domain solve."""
    kx = 2*math.pi*math.sin(math.radians(35.))
    omega = 2*math.pi/np.asarray(WAVELENGTHS)
    for pol in ('s', 'p'):
        r, t = airy(WAVELENGTHS, family='lorentz', strength=0., kx=kx, pol=pol)
        np.testing.assert_allclose(r, 0., atol=2e-15)
        np.testing.assert_allclose(t, np.exp(1j*np.sqrt(omega**2-kx**2)*.2), atol=2e-15)
        for family in FAMILIES:
            r, t = airy(WAVELENGTHS, family=family, thickness=0., kx=kx, pol=pol)
            np.testing.assert_allclose(r, 0., atol=2e-15)
            np.testing.assert_allclose(t, 1., atol=2e-15)
            r, t = airy(WAVELENGTHS, family=family, gamma=0., kx=kx, pol=pol)
            np.testing.assert_allclose(abs(r)**2+abs(t)**2, 1., atol=3e-15)
            r, t = airy(WAVELENGTHS, family=family, kx=kx, pol=pol)
            assert np.all(abs(r)**2+abs(t)**2 < 1.)
    for family in FAMILIES:
        rs, ts = airy(WAVELENGTHS, family=family, pol='s')
        rp, tp = airy(WAVELENGTHS, family=family, pol='p')
        np.testing.assert_allclose(rp, -rs, atol=2e-15)
        np.testing.assert_allclose(tp, ts, atol=2e-15)
    assert np.all(epsilon(WAVELENGTHS, family='drude').real < 0)
    np.testing.assert_array_equal(epsilon((2*math.pi/8,), family='lorentz',
                                        strength=0., gamma=0.), [1.+0j])
    return {'independent_airy_controls': 'passed', 'field_solves': 0}


def build(n=40, pol='p', family='drude', theta=35., periods=60, pml=24, eps=1., gamma=1.,
          omega_p=9., omega0=8., strength=40., thickness=.2, fraction=.3, courant=.5):
    dx = 1./n
    kx = 2*math.pi*math.sin(math.radians(theta))
    band = fd.Band(wavelength=1., df=.1, cutoff=1e-6)
    source = fd.PlaneWave('pw', z=.6, direction='-z', theta=theta, phi=0.,
                          polarization=pol, band=band)
    low = -(math.ceil((thickness+.8)/dx)+pml+fraction)*dx
    nz = math.ceil((1.4+pml*dx-low)/dx)
    domain = fd.Domain(size=(dx, dx, nz*dx), center=(0., 0., low+nz*dx/2))
    box = fd.Box(size=(5*dx, 5*dx, thickness), center=(0., 0., -thickness/2))
    pole = (fd.Drude(omega_p=omega_p, gamma=gamma) if family == 'drude'
            else fd.Lorentz(omega_0=omega0, gamma=gamma, strength=strength))
    if family not in FAMILIES:
        raise ValueError('family is drude or lorentz')
    monitors = (
        fd.Diffraction('r', z=1.1, side='reflect', wavelengths=WAVELENGTHS,
                       reference_plane=0., orders=(0, 0)),
        fd.Diffraction('t', z=-thickness-.5, side='transmit', wavelengths=WAVELENGTHS,
                       reference_plane=-thickness, incident_plane=0., orders=(0, 0)))
    scene = fd.Scene(domain=domain, resolution=fd.Resolution(dx=dx),
        boundaries=fd.Boundaries(x=fd.Axis1D(k=kx), y=fd.Periodic(),
                                 z=fd.PML(cells=pml, wavelength=1., reflection=1e-12)),
        structures=(fd.Structure(box, fd.Dispersive(eps_inf=eps, poles=(pole,))),),
        sources=(source,), monitors=monitors, run=fd.Run(periods=periods, courant=courant))
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
    package = next(parent for parent in helper.parents
                if (parent/'pyproject.toml').is_file() and (parent/'src/wavebit').is_dir())/'src/wavebit'

    def digest(root):
        value = hashlib.sha256()
        for path in sorted(root.rglob('*.py')):
            value.update(path.relative_to(root).as_posix().encode('utf-8')+b'\0')
            value.update(path.read_bytes().replace(b'\r\n', b'\n')+b'\0')
        return value.hexdigest()

    return {'helper_sha256': hashlib.sha256(helper.read_bytes()).hexdigest(),
            'fd_source_sha256': digest(package/'fdtd'), 'wavebit_source_sha256': digest(package),
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
    scene, kx = build(**config)
    begin = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = fd.run(scene, backend=backend)
    duration = time.perf_counter()-begin
    end = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if before != source_identity():
        raise RuntimeError('Source changed during the probe; preserve this failed attempt.')
    pol = config['pol']
    r = _numpy(getattr(result['r'].orders.order(0, 0), 'r_'+pol))
    t = _numpy(getattr(result['t'].orders.order(0, 0), 't_'+pol))
    reference_r, reference_t = airy(WAVELENGTHS, kx=kx, pol=pol,
        **{name: config[name] for name in ('family', 'eps', 'gamma', 'omega_p', 'omega0', 'strength', 'thickness')})
    reflect = _numpy(result['r'].orders.order(0, 0).R)
    transmit = _numpy(result['t'].orders.order(0, 0).T)
    energy_t, energy = result.diagnostics['interior_energy']
    energy = _numpy(energy)
    row = {
        'kind': 'single-pole-film-probe', 'started_utc': start, 'ended_utc': end, 'wall_s': duration,
        'source_identity': before, 'source_unchanged_during_run': True, **runtime_identity(),
        'config': {'backend': backend, **config}, 'grid': result.plan.shape,
        'source_band': {'wavelength': 1., 'df': .1, 'cutoff': 1e-6},
        'dt': result.plan.dt, 'steps': result.plan.steps, 'kx': kx, 'wavelengths': WAVELENGTHS,
        'theta_at': _numpy(result['r'].theta_at), 'faces': [0., -config['thickness']],
        'epsilon': epsilon(WAVELENGTHS, **{name: config[name] for name in
                           ('family', 'eps', 'gamma', 'omega_p', 'omega0', 'strength')}),
        'r': r, 't': t, 'reference_r': reference_r, 'reference_t': reference_t,
        'error_r': abs(r-reference_r), 'error_t': abs(t-reference_t),
        'R': reflect, 'T': transmit, 'absorption': 1-reflect-transmit,
        'reference_absorption': 1-abs(reference_r)**2-abs(reference_t)**2,
        'stable': result.diagnostics['stable'], 'decay': result.diagnostics['decay'],
        'energy_shape': energy.shape, 'energy_rows': len(energy_t),
        'energy_finite': bool(np.all(np.isfinite(energy))), 'energy_max': float(np.max(energy)),
        'warnings': [{'category': type(w.message).__name__, 'message': str(w.message)} for w in caught],
    }
    return _json(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--n', type=int, default=40)
    parser.add_argument('--family', choices=FAMILIES, default='drude')
    parser.add_argument('--pol', choices=('s', 'p'), default='p')
    for name, default in (('theta', 35.), ('eps', 1.), ('gamma', 1.), ('omega-p', 9.),
                           ('omega0', 8.), ('strength', 40.), ('fraction', .3), ('courant', .5)):
        parser.add_argument('--'+name, type=float, default=default)
    parser.add_argument('--periods', type=int, default=60)
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
