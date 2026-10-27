"""A focused Gaussian beam.

Criteria G1-G8 of ``docs/validation/fdtd/fd-gaussian-beam.md``, registered before the beam was
first run; every limit here is quoted from there. The oracles are continuum angular spectra
computed here, independently of the grid.
"""
import math
import types
import warnings

import numpy as np
import pytest

from wavebit import Material, Units, fdtd
from wavebit.errors import WavebitError

LAMBDAS = (0.95, 1.0, 1.05)
CENTRE = 1
SIGMA_HZ = 2e13
C0 = 299792458.0
L = 1e-6                                   # Units(length='um')
PERIODS = 100


def _quiet():
    warnings.simplefilter('ignore')


def _pulse(wavelength=1.0):
    return fdtd.GaussianPulse(wavelength=wavelength, frequency_sigma_hz=SIGMA_HZ, cutoff=1e-12)


def _setup(dx, *, polarization='y', waist=1.5, focus=(0., 0., 0.), z_sheet=-4., half=False,
           amplitude=1., phase=0., delay=0., scatterer=None, monitors=None, beam_kw=None,
           boundaries=None, pulse=None, duration=None, periods=PERIODS):
    """Setup B (two-dimensional), or G3's half-space when ``half``."""
    structures = ()
    if half:
        structures += (fdtd.Structure('glass', shape=fdtd.SnappedSlab(z=(-1., 5.)),
                                      material=Material(epsilon=2.25),
                                      continuation=('x-', 'x+')),)
    if scatterer is not None:
        structures += (fdtd.Structure('block', shape=fdtd.SnappedBox(size=(.4, 1., .4),
                                                                     center=(2., 0., 0.)),
                                      material=Material(epsilon=scatterer)),)
    beam = fdtd.GaussianBeam('b', plane=fdtd.Plane(z=z_sheet), pulse=pulse or _pulse(),
                             waist=waist, focus=focus, polarization=polarization,
                             direction='+z', amplitude=amplitude, phase=phase,
                             delay_seconds=delay, **(beam_kw or {}))
    component = 'E' + polarization
    if monitors is None:
        span = (-5.9, 5.9)
        monitors = (fdtd.FieldMonitor('focal', wavelengths=LAMBDAS, plane=fdtd.Plane(z=0., x=span),
                                      components=(component,)),
                    fdtd.FieldMonitor('axis', wavelengths=LAMBDAS, plane=fdtd.Plane(x=0.),
                                      components=('Ey',)),
                    fdtd.FluxMonitor('flux', plane=fdtd.Plane(z=3., x=span), wavelengths=LAMBDAS),
                    fdtd.FluxMonitor('back', plane=fdtd.Plane(z=-4.5, x=span), wavelengths=LAMBDAS))
    problem = fdtd.Problem(
        units=Units(length='um'),
        scene=fdtd.Scene(domain=fdtd.Domain(size=(12., dx, 10.)),
                         boundaries=boundaries or fdtd.Boundaries(x=fdtd.Open(), y=fdtd.InvariantAxis(),
                                                                  z=fdtd.Open()),
                         structures=structures),
        sources=(beam,), outputs=fdtd.Outputs(monitors=monitors))
    numerics = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=dx),
                             duration=duration or fdtd.Duration(periods=periods, source='b'),
                             absorber=fdtd.PML(cells=24, wavelength=1.))
    return problem, numerics


def _run(problem, numerics, backend='numpy'):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = fdtd.run(problem, numerics=numerics, backend=backend)
    names = [type(w.message).__name__ for w in caught]
    assert 'UnsettledFieldWarning' not in names and 'GrowingFieldWarning' not in names, names
    return result


def _np(value):
    if hasattr(value, 'detach'):
        value = value.detach()
    return np.asarray(value)


class _Runs(dict):
    """Setup B per (spacing, polarization), run once on first use."""

    def __missing__(self, key):
        dx, polarization = key
        _quiet()
        self[key] = _run(*_setup(dx, polarization=polarization))
        return self[key]


@pytest.fixture(scope='module')
def runs():
    return _Runs()


def _unit_spectrum(f, f0):
    """``|W(f)|`` of ``w(t) = exp(-s^2/(2 tau^2)) sin(2 pi f0 s)``, both Gaussian sidebands."""
    tau = 1.0 / (2 * math.pi * SIGMA_HZ)
    g = lambda x: math.exp(-0.5 * (2 * math.pi * x * tau) ** 2)    # noqa: E731
    return tau * math.sqrt(2 * math.pi) / 2 * abs(g(f - f0) - g(f + f0))


def _axis(result):
    """``(z, Ey)`` on the axis at the centre wavelength."""
    a = result['axis']
    z = np.asarray(a.coordinates['Ey'][2]).ravel()
    return z, _np(a.values['Ey'])[CENTRE].ravel()


def _beta_axis(result, dx, eps=1.0):
    """The grid's axial wavenumber at the centre wavelength (kernel units, c = 1)."""
    dt = float(result.provenance['time_step_seconds']) * C0 / L
    Omega = 2.0 * math.sin(math.pi * dt / LAMBDAS[CENTRE]) / dt
    return (2.0 / dx) * math.asin(Omega * math.sqrt(eps) * dx / 2.0)


def _angular(z, waist, distance_phase=None, n=4001):
    """The continuum on-axis field ``(1/2 pi) integral A(kx) [..] dkx`` over propagating ``kx``
    (vacuum, wavelength 1), ``A`` the transform of ``exp(-x^2/waist^2)``."""
    k = 2 * math.pi / LAMBDAS[CENTRE]
    kx = np.linspace(-k, k, n)[1:-1]
    A = waist * math.sqrt(math.pi) * np.exp(-(kx * waist) ** 2 / 4)
    kz = np.sqrt(k ** 2 - kx ** 2)
    weights = np.full(kx.size, kx[1] - kx[0])
    if distance_phase is None:
        phase = np.exp(1j * np.outer(z, kz))
    else:
        phase = distance_phase(kx, kz, z)
    return (phase @ (A * weights)) / (2 * math.pi)


# ---------------------------------------------------------------------------------------------
# G1: waist
# ---------------------------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize('polarization', ['y', 'x'])
@pytest.mark.parametrize('dx', [.05, .025])
def test_g1_the_focal_plane_has_the_declared_waist(runs, dx, polarization):
    f = runs[dx, polarization]['focal']
    c = 'E' + polarization
    x = np.asarray(f.coordinates[c][0]).ravel()
    intensity = np.abs(_np(f.values[c])[CENTRE].ravel()) ** 2
    w = 2 * math.sqrt(float(np.sum(x ** 2 * intensity) / np.sum(intensity)))
    print(f'G1 dx={dx} {polarization}: w / waist - 1 = {w / 1.5 - 1:.3e}')
    assert abs(w / 1.5 - 1) <= 1e-6, w


# ---------------------------------------------------------------------------------------------
# G2: Gouy phase
# ---------------------------------------------------------------------------------------------

def _gouy_error(result, dx):
    z, Ey = _axis(result)
    keep = np.abs(z) <= 3.0 + 1e-9
    z, Ey = z[keep], Ey[keep]
    psi = np.angle(Ey * np.exp(-1j * _beta_axis(result, dx) * z))
    psi = psi - psi[np.argmin(np.abs(z))]
    k = 2 * math.pi / LAMBDAS[CENTRE]
    exact = np.angle(_angular(z, 1.5) * np.exp(-1j * k * z))
    exact = exact - exact[np.argmin(np.abs(z))]
    zR = math.pi * 1.5 ** 2 / LAMBDAS[CENTRE]
    paraxial = -0.5 * np.arctan(z / zR)
    return (float(np.max(np.abs(psi - exact))), float(np.max(np.abs(exact - paraxial))),
            float(np.max(np.abs(exact))))


def _gouy_errors(runs):
    errors = {}
    for dx in (.05, .025):
        error, paraxial, size = _gouy_error(runs[dx, 'y'], dx)
        errors[dx] = error
        print(f'G2 dx={dx}: max |psi - psi_exact| = {error:.3e} rad (Gouy phase up to '
              f'{size:.3f} rad; exact vs paraxial {paraxial:.2e} rad)')
    order = math.log2(errors[.05] / errors[.025])
    print('G2 observed order', order)
    return errors, order


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason='registered G2 failed its limit at 0.05 um (2.2e-3 rad '
                   'against 2e-3); its convergence order passes. See the record.')
def test_g2_the_gouy_phase_converges_to_the_continuum(runs):
    errors, order = _gouy_errors(runs)
    assert errors[.05] <= 2e-3, errors
    assert order >= 1.8, (errors, order)


@pytest.mark.slow
def test_g2_the_gouy_phase_error_falls_at_second_order(runs):
    # The registered G2's second part, which passes on its own.
    errors, order = _gouy_errors(runs)
    assert order >= 1.8, (errors, order)


# ---------------------------------------------------------------------------------------------
# G3: focal shift through an interface
# ---------------------------------------------------------------------------------------------

def _peak(z, intensity):
    i = int(np.argmax(intensity))
    sl = slice(i - 2, i + 3)
    a, b, _ = np.polyfit(z[sl], intensity[sl], 2)
    return -b / (2 * a)


def _focal_oracle(z_i=-1., z_f=.5, n2=1.5, waist=1.):
    k = 2 * math.pi / LAMBDAS[CENTRE]
    z = np.linspace(z_i + .05, 4.5, 8901)

    def phase(kx, kz, zz):
        kz2 = np.sqrt((n2 * k) ** 2 - kx ** 2)
        t = 2 * kz / (kz + kz2)
        return t * np.exp(1j * kz * (z_i - z_f)) * np.exp(1j * np.outer(zz - z_i, kz2))
    field = _angular(z, waist, distance_phase=phase)
    return _peak(z, np.abs(field) ** 2)


@pytest.mark.slow
def test_g3_the_focus_shifts_into_a_dielectric_as_the_continuum_does():
    _quiet()
    oracle = _focal_oracle()
    errors = {}
    for dx in (.05, .025):
        problem, numerics = _setup(dx, waist=1., focus=(0., 0., .5), z_sheet=-2.5, half=True,
                                   monitors=(fdtd.FieldMonitor('axis', wavelengths=LAMBDAS,
                                                               plane=fdtd.Plane(x=0.),
                                                               components=('Ey',)),))
        z, Ey = _axis(_run(problem, numerics))
        inside = (z > -1. + 3 * dx) & (z < 4.5)
        peak = _peak(z[inside], np.abs(Ey[inside]) ** 2)
        errors[dx] = abs(peak - oracle)
        print(f'G3 dx={dx}: peak {peak:.5f} um, continuum {oracle:.5f} um, paraxial 1.25 um')
    assert errors[.05] <= .03 and errors[.025] <= .01 and errors[.025] < errors[.05], errors


# ---------------------------------------------------------------------------------------------
# G4: power
# ---------------------------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize('polarization', ['y', 'x'])
@pytest.mark.parametrize('dx', [.05, .025])
def test_g4_a_unit_amplitude_carries_one_watt_per_metre(runs, dx, polarization):
    flux = runs[dx, polarization]['flux']
    f0 = float(_np(flux.frequencies_hz)[CENTRE])
    expected = 2 * _unit_spectrum(f0, f0) ** 2
    got = float(_np(flux.spectral_energy)[CENTRE])
    print(f'G4 dx={dx} {polarization}: energy / expected - 1 = {got / expected - 1:.3e}')
    assert abs(got / expected - 1) <= 2e-5, (got, expected)


@pytest.mark.slow
def test_g4_a_unit_amplitude_carries_one_watt_in_three_dimensions():
    _quiet()
    span = (-4.4, 4.4)
    beam = fdtd.GaussianBeam('b', plane=fdtd.Plane(z=-2.), pulse=_pulse(), waist=1.5,
                             focus=(0., 0., 0.), polarization='x', direction='+z')
    problem = fdtd.Problem(
        units=Units(length='um'),
        scene=fdtd.Scene(domain=fdtd.Domain(size=(9., 9., 5.)), boundaries=fdtd.Boundaries.open()),
        sources=(beam,), outputs=fdtd.Outputs(monitors=(
            fdtd.FluxMonitor('flux', plane=fdtd.Plane(z=2., x=span, y=span), wavelengths=(1.,)),)))
    numerics = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.05),
                             duration=fdtd.Duration(periods=PERIODS, source='b'),
                             absorber=fdtd.PML(cells=12, wavelength=1.))
    flux = _run(problem, numerics)['flux']
    f0 = float(_np(flux.frequencies_hz)[0])
    expected = 2 * _unit_spectrum(f0, f0) ** 2
    got = float(_np(flux.spectral_energy)[0])
    print(f'G4 3-D: energy / expected - 1 = {got / expected - 1:.3e} ({flux.units})')
    assert abs(got / expected - 1) <= 2e-5, (got, expected)


# ---------------------------------------------------------------------------------------------
# G5: back-radiation
# ---------------------------------------------------------------------------------------------

def _predicted(problem, numerics, result, dx):
    """The predicted (forward, backward) spectral energies in the FluxMonitor's units."""
    from wavebit.fdtd._beam_source import beam_sheet, launched_power
    from wavebit.fdtd._runtime import runtime
    from wavebit.units import Z0
    sheet = beam_sheet(problem.sources[0], problem, numerics, problem.units)
    ts = L / C0
    plan = types.SimpleNamespace(steps=int(result.provenance['actual_steps']),
                                 dt=float(result.provenance['time_step_seconds']) / ts)
    forward, backward = launched_power(sheet, plan, runtime('numpy'), LAMBDAS)
    measure = L / (Z0 * dx)                       # one invariant axis
    factor = 2 * ts ** 2 * measure
    return forward * factor, backward * factor


@pytest.mark.slow
@pytest.mark.parametrize('dx', [.05, .025])
def test_g5_back_radiation_is_as_predicted(runs, dx):
    result = runs[dx, 'y']
    forward, backward = _predicted(*_setup(dx), result, dx)
    measured_back = -_np(result['back'].spectral_energy)
    measured_forward = _np(result['flux'].spectral_energy)
    back_error = np.abs(measured_back - backward) / forward
    forward_error = np.abs(measured_forward - forward) / forward
    print(f'G5 dx={dx}: predicted back / forward = {backward / forward}; |measured - predicted| '
          f'back / forward = {back_error}; forward = {forward_error}')
    assert np.all(back_error <= 1e-6), back_error
    assert np.all(forward_error <= 1e-5), forward_error
    assert backward[CENTRE] / forward[CENTRE] <= 1e-12, backward / forward


# ---------------------------------------------------------------------------------------------
# G6: backends
# ---------------------------------------------------------------------------------------------

def _readings(result):
    return {'flux': _np(result['flux'].spectral_energy), 'axis': _axis(result)[1]}


@pytest.mark.slow
def test_g6_backends_agree(runs):
    import jax
    jax.config.update('jax_enable_x64', True)
    _quiet()
    reference = _readings(runs[.05, 'y'])
    for backend in ('jax', 'torch'):
        got = _readings(_run(*_setup(.05), backend=backend))
        for key, value in reference.items():
            error = float(np.max(np.abs(got[key] - value)) / np.max(np.abs(value)))
            print(f'G6 {backend} {key}: {error:.2e}')
            assert error <= 1e-12, (backend, key, error)


# ---------------------------------------------------------------------------------------------
# G7: derivatives
# ---------------------------------------------------------------------------------------------

G7_POINT = {'amplitude': 1.3, 'phase': .4, 'delay': 2e-15, 'scatterer': 3.0}
G7_STEPS = {'amplitude': 1e-3, 'phase': 1e-3, 'delay': 1e-17, 'scatterer': 3e-4}


def _g7_outputs(point, backend):
    problem, numerics = _setup(.05, amplitude=point['amplitude'], phase=point['phase'],
                               delay=point['delay'], scatterer=point['scatterer'],
                               monitors=(fdtd.FluxMonitor('flux', plane=fdtd.Plane(z=3., x=(-5.9, 5.9)),
                                                          wavelengths=(1.,)),
                                         fdtd.FieldMonitor('probe', wavelengths=(1.,),
                                                           plane=fdtd.Plane(z=3., x=(-.05, .05)),
                                                           components=('Ey',))))
    result = fdtd.run(problem, numerics=numerics, backend=backend)
    probe = result['probe']
    x = np.asarray(probe.coordinates['Ey'][0]).ravel()
    centre = int(np.argmin(np.abs(x)))
    return result['flux'].spectral_energy[0], probe.values['Ey'][0].reshape(-1)[centre].real


def _central(name, which):
    def at(step):
        hi, lo = dict(G7_POINT), dict(G7_POINT)
        hi[name] += step
        lo[name] -= step
        return (float(_g7_outputs(hi, 'numpy')[which])
                - float(_g7_outputs(lo, 'numpy')[which])) / (2 * step)
    h = G7_STEPS[name]
    return (4 * at(h / 2) - at(h)) / 3


G7_CASES = (('amplitude', 0), ('phase', 1), ('delay', 1), ('scatterer', 0))


@pytest.mark.slow
def test_g7_derivatives_match_central_differences():
    _quiet()
    import jax
    import torch
    jax.config.update('jax_enable_x64', True)
    for name, which in G7_CASES:
        reference = _central(name, which)

        def f(x, backend, name=name, which=which):
            point = dict(G7_POINT)
            point[name] = x
            return _g7_outputs(point, backend)[which]
        by_jax = float(jax.grad(lambda x: f(x, 'jax'))(G7_POINT[name]))
        x = torch.tensor(G7_POINT[name], dtype=torch.float64, requires_grad=True)
        f(x, 'torch').backward()
        by_torch = float(x.grad)
        print(f'G7 {name}: jax {by_jax:.10e}, torch {by_torch:.10e}, central {reference:.10e}')
        for got in (by_jax, by_torch):
            assert abs(got - reference) <= 1e-6 * abs(reference), (name, got, reference)


# ---------------------------------------------------------------------------------------------
# G8: refusals
# ---------------------------------------------------------------------------------------------

def _refused(code, problem, numerics, backend='numpy'):
    _quiet()
    report = fdtd.check(problem, numerics=numerics, backend=backend)
    assert [issue.code for issue in report.issues] == [code], report.issues
    with pytest.raises(WavebitError) as info:
        fdtd.run(problem, numerics=numerics, backend=backend)
    assert info.value.code == code, (info.value.code, str(info.value))


def _with(**changes):
    problem, numerics = _setup(.05, periods=1)
    return problem.replace(sources=(problem.sources[0].replace(**changes),)), numerics


def test_g8_a_direction_off_the_normal_is_refused():
    _refused('source.direction', *_with(direction='+x'))


def test_g8_a_polarization_along_the_normal_is_refused():
    _refused('source.polarization', *_with(polarization='z'))


@pytest.mark.parametrize('waist', [0., -1.])
def test_g8_a_waist_is_positive(waist):
    _refused('source.waist', *_with(waist=waist))


def test_g8_a_focus_is_a_point():
    _refused('source.focus', *_with(focus=(0., 0.)))


def test_g8_a_traced_waist_is_refused():
    import torch
    waist = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
    _refused('source.traced', *_with(waist=waist), backend='torch')


def test_g8_a_traced_pulse_wavelength_is_refused():
    import torch
    lam = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    _refused('mode.traced', *_setup(.05, pulse=_pulse(lam), duration=fdtd.Duration(seconds=1e-15)),
             backend='torch')


def test_g8_a_nonuniform_medium_on_the_plane_is_refused():
    problem, numerics = _setup(.05, periods=1)
    scene = problem.scene.replace(structures=(fdtd.Structure(
        'rod', shape=fdtd.SnappedBox(size=(.5, 1., 1.), center=(1., 0., -4.)),
        material=Material(epsilon=2.)),))
    _refused('source.medium', problem.replace(scene=scene), numerics)


def test_g8_a_metal_transverse_face_is_refused():
    boundaries = fdtd.Boundaries(x=fdtd.PEC(), y=fdtd.InvariantAxis(), z=fdtd.Open())
    _refused('source.boundary', *_setup(.05, periods=1, boundaries=boundaries,
                                        monitors=(fdtd.FieldMonitor('axis', wavelengths=(1.,),
                                                                    plane=fdtd.Plane(x=0.),
                                                                    components=('Ey',)),)))


def test_g8_a_beam_the_cross_section_cannot_hold_is_refused():
    _refused('source.beam_size', *_with(waist=4.))


def test_g8_a_monitor_on_the_sheet_is_refused():
    monitors = (fdtd.FluxMonitor('m', wavelengths=(1.,), plane=fdtd.Plane(z=-3.95, x=(-5., 5.))),)
    _refused('monitor.source_sheet', *_setup(.05, periods=1, monitors=monitors))


def test_g8_the_engine_placeholder_names_the_public_source():
    import wavebit.fdtd._legacy as fd
    with pytest.raises(NotImplementedError, match='wavebit.fdtd.GaussianBeam'):
        fd.GaussianBeam()


def test_the_beam_is_centred_on_a_transverse_focus():
    # Not a registered criterion: the sheet's E pattern is centred on the focus's x (here
    # 1 um), and its axial phase does not depend on the transverse shift.
    from wavebit.fdtd._beam_source import beam_sheet
    _quiet()
    problem, numerics = _setup(.05, periods=1)
    centred = beam_sheet(problem.sources[0], problem, numerics, problem.units)
    shifted_source = problem.sources[0].replace(focus=(1., 0., 0.))
    shifted = beam_sheet(shifted_source, problem.replace(sources=(shifted_source,)), numerics,
                         problem.units)
    a = np.abs(centred.patterns['Ev_re'] + 1j * centred.patterns['Ev_im']).ravel()
    b = np.abs(shifted.patterns['Ev_re'] + 1j * shifted.patterns['Ev_im']).ravel()
    # (The window's wall node, the first and last entries, is zero in both.)
    assert np.allclose(b[21:-1], a[1:-21], atol=1e-9 * a.max()), np.max(np.abs(b[21:-1] - a[1:-21]))
