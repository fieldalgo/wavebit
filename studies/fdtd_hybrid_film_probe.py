"""Bounded finite mixed Debye/oscillator film instrument with an independent complex Airy reference.

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
import wavebit.fd as fd


ENV_NAMES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'PYTHONUTF8')
WAVELENGTHS = (.9, 1., 1.1)
FAMILIES = ('mixed-metal', 'mixed-dielectric')
# (kind, tau-or-angular-resonance, oscillator damping, delta-or-strength).
# Debye uses its tau and delta; its third entry is a literal unused zero.
POLE_SPECS = {
    'mixed-metal': (('debye', .2, 0., .25), ('drude', 0., 1., 81.),
                    ('lorentz', 8., 2., 10.)),
    'mixed-dielectric': (('debye', .08, 0., 2.), ('lorentz', 5., .8, 10.),
                         ('debye', .25, 0., 1.), ('lorentz', 8., 1.5, 20.),
                         ('lorentz', 11., 2., 15.)),
}


def specifications(family, pole_specs=None):
    if family not in FAMILIES:
        raise ValueError('Unknown fixed mixed Debye/oscillator family')
    result = tuple(POLE_SPECS[family] if pole_specs is None else pole_specs)
    if (len(result) < 2 or any(p[0] not in ('debye', 'drude', 'lorentz') for p in result)
            or not any(p[0] == 'debye' for p in result) or not any(p[0] != 'debye' for p in result)):
        raise ValueError('Declared Debye and oscillator specifications are both required')
    return result


def epsilon(wavelengths, *, family='mixed-metal', eps=1., pole_specs=None):
    """Independent continuum sum; specifications retain both declared physical families."""
    omega = 2*math.pi/np.asarray(wavelengths, dtype=float)
    result = np.full(omega.shape, complex(eps))
    for kind, omega0, gamma, strength in specifications(family, pole_specs):
        if strength != 0:
            if kind == 'debye':
                result += strength/(1-1j*omega*omega0)
            else:
                d = 0. if kind == 'drude' else omega0**2
                result += strength/(d-omega**2-1j*gamma*omega)
    return result


def band_certificate(family, eps=1., pole_specs=None, *, endpoints_only=False, omit_debye=False):
    """Independent analytic sufficient bulk bound; no production spectral helper."""
    width = math.sqrt(2*math.log(1e6))*.1
    low, high = 2*math.pi*(1-width), 2*math.pi*(1+width)
    xmin, xmax = low**2, high**2
    bound_squared = eps*xmax
    for kind, omega0, gamma, strength in specifications(family, pole_specs):
        if strength == 0:
            continue
        if kind == 'debye':
            if not omit_debye:
                bound_squared += strength*xmax/math.hypot(1., omega0*high)
            continue
        d = 0. if kind == 'drude' else omega0**2
        if gamma == 0 and xmin <= d <= xmax:
            raise ValueError('Active undamped resonance lies inside the source band')
        points = [xmin, xmax]
        if 2*d > gamma**2 and not endpoints_only:
            critical = 2*d*d/(2*d-gamma**2)
            if xmin < critical < xmax:
                points.append(critical)
        bound_squared += strength*max(x/math.sqrt((d-x)**2+gamma**2*x) for x in points)
    return math.sqrt(bound_squared)


def airy(wavelengths, *, family='mixed-metal', eps=1., pole_specs=None,
         thickness=.2, kx=0., pol='s'):
    omega = 2*math.pi/np.asarray(wavelengths, dtype=float)
    own = epsilon(wavelengths, family=family, eps=eps, pole_specs=pole_specs)
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
    """Analytic identities and the fixed n40 band floor, BEFORE any field response."""
    kx = 2*math.pi*math.sin(math.radians(35.))
    omega = 2*math.pi/np.asarray(WAVELENGTHS)
    inactive = (('debye', .2, 0., 0.), ('lorentz', 2*math.pi, 0., 0.))
    bounds = {}
    for family in FAMILIES:
        material = epsilon(WAVELENGTHS, family=family)
        assert np.all(material.imag > 0)
        bounds[family] = band_certificate(family)
        assert bounds[family]/40 < 2*math.pi/8
        for pol in ('s', 'p'):
            r, t = airy(WAVELENGTHS, family=family, thickness=0., kx=kx, pol=pol)
            np.testing.assert_allclose(r, 0., atol=2e-15)
            np.testing.assert_allclose(t, 1., atol=2e-15)
            r, t = airy(WAVELENGTHS, family=family, kx=kx, pol=pol)
            assert np.all(abs(r)**2+abs(t)**2 < 1.)
            lossless = tuple((kind, w0, 0., 0. if kind == 'debye' else strength)
                             for kind, w0, _, strength in POLE_SPECS[family])
            r, t = airy(WAVELENGTHS, family=family, pole_specs=lossless, kx=kx, pol=pol)
            np.testing.assert_allclose(abs(r)**2+abs(t)**2, 1., atol=3e-15)
            r, t = airy(WAVELENGTHS, family=family, pole_specs=inactive, kx=kx, pol=pol)
            np.testing.assert_allclose(r, 0., atol=2e-15)
            np.testing.assert_allclose(t, np.exp(1j*np.sqrt(omega**2-kx**2)*.2), atol=2e-15)
        rs, ts = airy(WAVELENGTHS, family=family, pol='s')
        rp, tp = airy(WAVELENGTHS, family=family, pol='p')
        np.testing.assert_allclose(rp, -rs, atol=2e-15)
        np.testing.assert_allclose(tp, ts, atol=2e-15)
        np.testing.assert_allclose(epsilon(WAVELENGTHS, family=family, pole_specs=POLE_SPECS[family][::-1]),
                                   material, rtol=2e-15, atol=2e-15)
    assert np.all(epsilon(WAVELENGTHS, family='mixed-metal').real < 0)
    np.testing.assert_array_equal(epsilon((1.,), pole_specs=inactive), [1.+0j])
    upper_omega = 2*math.pi*(1+math.sqrt(2*math.log(1e6))*.1)
    upper_angular_dt = upper_omega*.5/(40*math.sqrt(3.))
    assert upper_angular_dt < math.pi  # A positive material-rate policy only tightens admission; the fixed planned dt must still pass.
    return {'analytic_n40_angular_dt_upper_bound': upper_angular_dt,
            'independent_airy_controls': 'passed', 'analytic_n40_band_bounds': bounds,
            'metallic_sampled_real_epsilon': True, 'field_solves': 0}


def build(n=40, pol='p', family='mixed-metal', theta=35., periods=120, pml=24, eps=1.,
          pole_specs=None, thickness=.2, fraction=.3, courant=.5):
    dx = 1./n
    kx = 2*math.pi*math.sin(math.radians(theta))
    band = fd.Band(wavelength=1., df=.1, cutoff=1e-6)
    source = fd.PlaneWave('pw', z=.6, direction='-z', theta=theta, phi=0.,
                          polarization=pol, band=band)
    low = -(math.ceil((thickness+.8)/dx)+pml+fraction)*dx
    nz = math.ceil((1.4+pml*dx-low)/dx)
    domain = fd.Domain(size=(dx, dx, nz*dx), center=(0., 0., low+nz*dx/2))
    box = fd.Box(size=(5*dx, 5*dx, thickness), center=(0., 0., -thickness/2))
    poles = tuple(fd.Debye(tau=omega0, delta_eps=strength) if kind == 'debye'
                  else fd.Drude(omega_p=math.sqrt(strength), gamma=gamma) if kind == 'drude'
                  else fd.Lorentz(omega_0=omega0, gamma=gamma, strength=strength)
                  for kind, omega0, gamma, strength in specifications(family, pole_specs))
    monitors = (
        fd.Diffraction('r', z=1.1, side='reflect', wavelengths=WAVELENGTHS,
                       reference_plane=0., orders=(0, 0)),
        fd.Diffraction('t', z=-thickness-.5, side='transmit', wavelengths=WAVELENGTHS,
                       reference_plane=-thickness, incident_plane=0., orders=(0, 0)))
    scene = fd.Scene(domain=domain, resolution=fd.Resolution(dx=dx),
        boundaries=fd.Boundaries(x=fd.Axis1D(k=kx), y=fd.Periodic(),
                                 z=fd.PML(cells=pml, wavelength=1., reflection=1e-12)),
        structures=(fd.Structure(box, fd.Dispersive(eps_inf=eps, poles=poles)),),
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
    package = helper.parent.parent/'src/wavebit'

    def digest(root):
        value = hashlib.sha256()
        for path in sorted(root.rglob('*.py')):
            value.update(path.relative_to(root).as_posix().encode('utf-8')+b'\0')
            value.update(path.read_bytes().replace(b'\r\n', b'\n')+b'\0')
        return value.hexdigest()

    return {'helper_sha256': hashlib.sha256(helper.read_bytes()).hexdigest(),
            'fd_source_sha256': digest(package/'fd'), 'wavebit_source_sha256': digest(package),
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
    config['pole_specs'] = specifications(config['family'], config['pole_specs'])
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
        **{name: config[name] for name in ('family', 'eps', 'pole_specs', 'thickness')})
    reflect = _numpy(result['r'].orders.order(0, 0).R)
    transmit = _numpy(result['t'].orders.order(0, 0).T)
    energy_t, energy = result.diagnostics['interior_energy']
    energy = _numpy(energy)
    row = {
        'kind': 'hybrid-film-probe', 'started_utc': start, 'ended_utc': end, 'wall_s': duration,
        'source_identity': before, 'source_unchanged_during_run': True, **runtime_identity(),
        'config': {'backend': backend, **config}, 'grid': result.plan.shape,
        'source_band': {'wavelength': 1., 'df': .1, 'cutoff': 1e-6},
        'dt': result.plan.dt, 'steps': result.plan.steps, 'kx': kx, 'wavelengths': WAVELENGTHS,
        'theta_at': _numpy(result['r'].theta_at), 'faces': [0., -config['thickness']],
        'epsilon': epsilon(WAVELENGTHS, **{name: config[name] for name in
                           ('family', 'eps', 'pole_specs')}),
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
    parser.add_argument('--family', choices=FAMILIES, default='mixed-metal')
    parser.add_argument('--pol', choices=('s', 'p'), default='p')
    for name, default in (('theta', 35.), ('eps', 1.), ('fraction', .3), ('courant', .5)):
        parser.add_argument('--'+name, type=float, default=default)
    parser.add_argument('--pole-specs', type=json.loads)
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
