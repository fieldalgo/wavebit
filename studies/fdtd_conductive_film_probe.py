"""Small C8 film instrument: the new FD engine against the complex Airy formula.

This is a measurement helper, not a registered convergence verdict. It preserves the
complex r/t values per wavelength, fixed transverse wavevector, requested face positions,
decay and energy diagnostics. A caller must fix its grid ladder, time/PML controls and
decision thresholds before using rows to assert an order. No legacy FDTD or RCWA solver
is used as the reference.

Example (one deliberately small row)::

    python -m studies.fdtd_conductive_film_probe --n 40 --pol p --periods 60

The x direction is Axis1D, so its derivative is exactly i kx and the transverse size
does not introduce a differencing error. Both film faces keep the same subcell offset
on a 40/80/160/... ladder when thickness=.2 and fraction=.3. Physical source/monitor
positions and film faces stay fixed; only the outer grid origin moves by less than dx.
"""
import argparse
import datetime
import hashlib
import importlib.metadata
import inspect
import json
import math
import pathlib
import time
import warnings

import numpy as np

import wavebit.fd as fd


WAVELENGTHS = (.9, 1.0, 1.1)


def airy(wavelengths, *, eps=4.0, sigma=8.0, thickness=.2, kx=0.0, pol='s'):
    """Continuum film between identical vacuum half-spaces, exp(-i omega t).

    Tangential-E admittance is kz/omega for s and eps*omega/kz for p. The
    reflected p basis reverses its tangential direction, so the returned r_p is
    minus the tangential-E reflection coefficient; t has no such reversal.
    """
    omega = 2 * math.pi / np.asarray(wavelengths, dtype=float)
    material = eps + 1j * sigma / omega
    kz0 = np.sqrt(omega ** 2 - kx ** 2 + 0j)
    kz = np.sqrt(material * omega ** 2 - kx ** 2 + 0j)
    kz = np.where(kz.imag < 0, -kz, kz)
    if pol == 's':
        y0, y = kz0 / omega, kz / omega
    elif pol == 'p':
        y0, y = omega / kz0, material * omega / kz
    else:
        raise ValueError('pol is s or p')
    rho = (y0 - y) / (y0 + y)
    phase = np.exp(1j * kz * thickness)
    den = 1 - rho ** 2 * phase ** 2
    r = rho * (1 - phase ** 2) / den
    t = (1 - rho ** 2) * phase / den
    if pol == 'p':
        r = -r
    return r, t


def preflight():
    """Analytic controls; no FD solve and no production film helper in the oracle."""
    kx = 2 * math.pi * math.sin(math.radians(35))
    omega = 2 * math.pi / np.asarray(WAVELENGTHS)
    for pol in ('s', 'p'):
        r, t = airy(WAVELENGTHS, eps=1., sigma=0., kx=kx, pol=pol)
        np.testing.assert_allclose(r, 0., atol=2e-15)
        np.testing.assert_allclose(t, np.exp(1j * np.sqrt(omega ** 2 - kx ** 2) * .2),
                                   atol=2e-15)
        r, t = airy(WAVELENGTHS, thickness=0., kx=kx, pol=pol)
        np.testing.assert_allclose(r, 0., atol=2e-15)
        np.testing.assert_allclose(t, 1., atol=2e-15)
        r, t = airy(WAVELENGTHS, sigma=0., kx=kx, pol=pol)
        np.testing.assert_allclose(abs(r) ** 2 + abs(t) ** 2, 1., atol=2e-15)
        r, t = airy(WAVELENGTHS, kx=kx, pol=pol)
        assert np.all(abs(r) ** 2 + abs(t) ** 2 < 1.)
    rs, ts = airy(WAVELENGTHS, kx=0., pol='s')
    rp, tp = airy(WAVELENGTHS, kx=0., pol='p')
    np.testing.assert_allclose(rp, -rs, atol=2e-15)
    np.testing.assert_allclose(tp, ts, atol=2e-15)
    return {'analytic_controls': 'passed', 'field_solves': 0}


def build(n=40, pol='p', theta=35., periods=60, pml=24, eps=4., sigma=8.,
          thickness=.2, fraction=.3, courant=.5):
    dx = 1. / n
    kx = 2 * math.pi * math.sin(math.radians(theta))
    band = fd.Band(wavelength=1., df=.1, cutoff=1e-6)
    source = fd.PlaneWave('pw', z=.6, direction='-z', theta=theta, phi=0.,
                          polarization=pol, band=band)
    # Front face z=0 has this prescribed fraction of a grid spacing above its node.
    low = -(math.ceil((thickness + .8) / dx) + pml + fraction) * dx
    nz = math.ceil((1.4 + pml * dx - low) / dx)
    size_z = nz * dx
    domain = fd.Domain(size=(dx, dx, size_z), center=(0., 0., low + size_z / 2))
    box = fd.Box(size=(5 * dx, 5 * dx, thickness), center=(0., 0., -thickness / 2))
    monitors = (
        fd.Diffraction('r', z=1.1, side='reflect', wavelengths=WAVELENGTHS,
                       reference_plane=0., orders=(0, 0)),
        fd.Diffraction('t', z=-thickness - .5, side='transmit', wavelengths=WAVELENGTHS,
                       reference_plane=-thickness, incident_plane=0., orders=(0, 0)),)
    scene = fd.Scene(
        domain=domain, resolution=fd.Resolution(dx=dx),
        boundaries=fd.Boundaries(x=fd.Axis1D(k=kx), y=fd.Periodic(),
                                 z=fd.PML(cells=pml, wavelength=1., reflection=1e-12)),
        structures=(fd.Structure(box, fd.Conductor(eps=eps, sigma=sigma)),),
        sources=(source,), monitors=monitors,
        run=fd.Run(periods=periods, courant=courant))
    return scene, kx


def _numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _json(value):
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, complex):
        return [value.real, value.imag]
    if isinstance(value, np.generic):
        return _json(value.item())
    return value


def source_identity():
    """Helper bytes and normalized sources; no claim about the old calibration rows."""
    helper = pathlib.Path(__file__).resolve()
    package = helper.parent.parent / 'wavebit'

    def digest(root):
        value = hashlib.sha256()
        for path in sorted(root.rglob('*.py')):
            value.update(path.relative_to(root).as_posix().encode('utf-8') + b'\0')
            value.update(path.read_bytes().replace(b'\r\n', b'\n') + b'\0')
        return value.hexdigest()

    return {'helper_sha256': hashlib.sha256(helper.read_bytes()).hexdigest(),
            'fd_source_sha256': digest(package / 'fd'),
            'wavebit_source_sha256': digest(package),
            'fd_digest_rule': 'sorted relative POSIX path + NUL + LF-normalized bytes + NUL'}


def measure(*, backend='numpy', **kwargs):
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    identity_before = source_identity()
    bound = inspect.signature(build).bind(**kwargs)
    bound.apply_defaults()
    config = dict(bound.arguments)
    scene, kx = build(**config)
    begin = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = fd.run(scene, backend=backend)
    duration = time.perf_counter() - begin
    ended = datetime.datetime.now(datetime.timezone.utc).isoformat()
    identity_after = source_identity()
    if identity_before != identity_after:
        raise RuntimeError('FD or helper source changed during this probe; preserve the failed attempt.')
    pol = config['pol']
    r = _numpy(getattr(result['r'].orders.order(0, 0), 'r_' + pol))
    t = _numpy(getattr(result['t'].orders.order(0, 0), 't_' + pol))
    reference_r, reference_t = airy(
        WAVELENGTHS, kx=kx, pol=pol, eps=config['eps'],
        sigma=config['sigma'], thickness=config['thickness'])
    reflect = _numpy(result['r'].orders.order(0, 0).R)
    transmit = _numpy(result['t'].orders.order(0, 0).T)
    energy_t, energy = result.diagnostics['interior_energy']
    energy = _numpy(energy)
    record = {
        'kind': 'conductive-film-probe', 'started_utc': started, 'ended_utc': ended,
        'wall_s': duration, 'source_identity': identity_before,
        'source_unchanged_during_run': True,
        'backend_versions': {name: importlib.metadata.version(name)
                             for name in ('numpy', backend) if name in ('numpy', 'jax', 'torch')},
        'config': {'backend': backend, **config}, 'grid': result.plan.shape,
        'source_band': {'wavelength': float(scene.sources[0].band.wavelength),
                        'df': float(scene.sources[0].band.df),
                        'cutoff': float(scene.sources[0].band.cutoff)},
        'dt': result.plan.dt, 'steps': result.plan.steps, 'kx': kx,
        'wavelengths': WAVELENGTHS, 'theta_at': _numpy(result['r'].theta_at),
        'faces': [0., -config['thickness']],
        'r': r, 't': t, 'reference_r': reference_r, 'reference_t': reference_t,
        'error_r': abs(r - reference_r), 'error_t': abs(t - reference_t),
        'R': reflect, 'T': transmit, 'absorption': 1 - reflect - transmit,
        'reference_absorption': 1 - abs(reference_r) ** 2 - abs(reference_t) ** 2,
        'stable': result.diagnostics['stable'], 'decay': result.diagnostics['decay'],
        'energy_shape': energy.shape, 'energy_rows': len(energy_t),
        'energy_finite': bool(np.all(np.isfinite(energy))),
        'energy_max': float(np.max(energy)),
        'warnings': [{'category': w.category.__name__, 'message': str(w.message)} for w in caught],
    }
    return _json(record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--n', type=int, default=40)
    parser.add_argument('--pol', choices=('s', 'p'), default='p')
    parser.add_argument('--theta', type=float, default=35.)
    parser.add_argument('--periods', type=int, default=60)
    parser.add_argument('--pml', type=int, default=24)
    parser.add_argument('--eps', type=float, default=4.)
    parser.add_argument('--sigma', type=float, default=8.)
    parser.add_argument('--fraction', type=float, default=.3)
    parser.add_argument('--courant', type=float, default=.5)
    parser.add_argument('--backend', choices=('numpy', 'jax', 'torch'), default='numpy')
    parser.add_argument('--output', type=pathlib.Path)
    args = vars(parser.parse_args())
    preflight_only, output = args.pop('preflight'), args.pop('output')
    row = preflight() if preflight_only else measure(**args)
    encoded = json.dumps(row, indent=2, allow_nan=False)
    if output is not None:
        with output.open('x', encoding='utf-8') as stream:
            stream.write(encoded + '\n')
    print(encoded)


if __name__ == '__main__':
    main()
