"""A PlaneWave behind ConductiveAbsorber faces: the source re-injected the absorber's echo.

A PlaneWave's incident field comes from a 1-D column of the background, hard-driven one cell
upstream of its sheet. The column used to end on the scene's own faces, so behind a
``ConductiveAbsorber`` -- PEC-backed, round-trip amplitude ~exp(-3) = 5e-2 at any cell count
-- the column's echo came back to the drive and was sent down again as a second incident wave.
In an empty periodic cell (vacuum, normal incidence, 20 nm cells, lambda 0.7..0.9 um) the
transmitted flux moved by -9.6 % .. +10.6 % against the same scene under the default PML, and
the measured incident spectrum by +-5 %. The column now ends in a PML of its own there
(``_prepare.sources.column_pads``), and what is left is the main grid's own absorber return.

Independent references:

* the default-PML run of the same scene, and the closed-form Fresnel/Airy slab (R + T = 1);
* a 1-D cavity model -- a unit one-way source, the slab, and a reflector r_a on each face --
  whose r_a (modulus and phase) is fitted on the vacuum run alone and then predicts the slab.
"""
import math

import numpy as np
import pytest

from wavebit import ConductiveMaterial, Material, Units, fdtd
from wavebit.units import C0, EPSILON0

WAVELENGTHS = tuple(float(w) for w in np.round(np.linspace(.7, .9, 9), 4))
EPS, THICKNESS, CENTER, SOURCE_Z, MONITOR_Z = 2.25, .3, .01, .8, -.6


def _problem(slab, background=None):
    structures = ()
    if slab:
        structures = (fdtd.Structure('slab', shape=fdtd.SnappedBox(size=(.1, .1, THICKNESS),
                                                                   center=(0., 0., CENTER)),
                                     material=Material(epsilon=EPS)),)
    monitors = (fdtd.FluxMonitor('flux', plane=fdtd.Plane(z=MONITOR_Z), wavelengths=WAVELENGTHS),
                fdtd.DiffractionMonitor('r', plane=fdtd.Plane(z=.6), wavelengths=WAVELENGTHS,
                                        source='pw', wave='reflected'),
                fdtd.DiffractionMonitor('t', plane=fdtd.Plane(z=MONITOR_Z),
                                        wavelengths=WAVELENGTHS, source='pw', wave='transmitted'))
    return fdtd.Problem(units=Units(length='um'), scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.1, .1, 2.)), structures=structures,
        background=Material(epsilon=1.) if background is None else background,
        boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.Open())),
        sources=(fdtd.PlaneWave('pw', plane=fdtd.Plane(z=SOURCE_Z), direction='-z',
                                pulse=fdtd.GaussianPulse(wavelength=.8, frequency_sigma_hz=4e13)),),
        outputs=fdtd.Outputs(monitors=monitors))


def _numerics(absorber):
    return fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.02), duration=fdtd.Duration(steps=6000),
                         absorber=absorber)


def _efficiency(orders):
    value = np.asarray(orders.efficiency)
    return value.sum(axis=tuple(range(1, value.ndim)))


@pytest.fixture(scope='module')
def runs():
    out = {}
    for slab in (False, True):
        for label, absorber in (('pml', fdtd.PML(cells=12)),
                                ('graded', fdtd.ConductiveAbsorber(cells=12))):
            result = fdtd.run(_problem(slab), numerics=_numerics(absorber))
            out[slab, label] = dict(
                flux=-np.asarray(result['flux'].spectral_energy),
                incident=np.asarray(result['t'].reference['incident_spectrum']),
                R=_efficiency(result['r']), T=_efficiency(result['t']))
    return out


def _airy(wavelength):
    """Closed-form normal-incidence reflectance of the lossless slab in vacuum."""
    n, k0 = math.sqrt(EPS), 2 * math.pi / wavelength
    r12 = (1 - n) / (1 + n)
    phase = np.exp(2j * n * k0 * THICKNESS)
    return abs((r12 - r12 * phase) / (1 - r12 * r12 * phase)) ** 2


def _cavity_flux(wavelength, r_a, slab):
    """Net downward flux at the monitor: unit one-way source, slab, reflector r_a at z = +-1.

    Amplitudes ``E = D exp(-ikz) + U exp(ikz)`` per region; ``r_a`` is referred to the faces
    (``u(-1) = r_a d(-1)``, ``d(1) = r_a u(1)``); the slab is the exact transfer matrix.
    """
    k = 2 * math.pi / wavelength
    n = math.sqrt(EPS) if slab else 1.

    def m(index, z):
        q = index * k
        return np.array([[np.exp(-1j * q * z), np.exp(1j * q * z)],
                         [index * np.exp(-1j * q * z), -index * np.exp(1j * q * z)]])
    z0, z1 = CENTER - THICKNESS / 2, CENTER + THICKNESS / 2
    across = np.linalg.solve(m(1, z1), m(n, z1)) @ np.linalg.solve(m(n, z0), m(1, z0))
    below = np.array([1., r_a * np.exp(2j * k)])
    above = across @ below
    scale = 1. / (above[0] - r_a * above[1] * np.exp(2j * k))
    down, up = scale * below
    return abs(down) ** 2 - abs(up) ** 2


def test_the_incident_spectrum_is_the_pml_scenes(runs):
    # The column no longer echoes: its incident spectrum is the PML scene's (it was +-5 %).
    graded, pml = runs[False, 'graded']['incident'], runs[False, 'pml']['incident']
    np.testing.assert_allclose(np.abs(graded), np.abs(pml), rtol=1e-3)


def test_empty_cell_flux_is_off_by_the_absorbers_own_return_only(runs):
    # In vacuum the only error left is the main grid's own echo: |r_a|^2 = 2.5e-3 off the net
    # flux and +-2 |r_a|^2 of a second round trip, so within 1e-2 (it was -9.6 % .. +10.6 %).
    ratio = runs[False, 'graded']['flux'] / runs[False, 'pml']['flux']
    assert np.max(np.abs(ratio - 1)) < 1e-2, ratio
    assert np.all(np.abs(runs[False, 'graded']['R'] - np.exp(-6)) < 2e-4)


def test_pml_slab_is_the_closed_form_and_conserves_energy(runs):
    pml = runs[True, 'pml']
    np.testing.assert_allclose(pml['R'], [_airy(w) for w in WAVELENGTHS], atol=5e-3)
    np.testing.assert_allclose(pml['R'] + pml['T'], 1., atol=1e-4)


def test_slab_error_is_predicted_by_the_absorbers_reflection_alone(runs):
    # Fit r_a = sign * rho * exp(2 i k d) on the empty cell only, then predict the slab.
    vacuum = runs[False, 'graded']['flux'] / runs[False, 'pml']['flux']
    rho = math.sqrt(float(np.mean(runs[False, 'graded']['R'])))
    k = 2 * math.pi / np.asarray(WAVELENGTHS)

    def ratio(r_a, slab):
        return np.array([_cavity_flux(w, r, slab) / _cavity_flux(w, 0., slab)
                         for w, r in zip(WAVELENGTHS, r_a)])
    candidates = []
    for sign in (1., -1.):
        for d in np.linspace(0., .5, 501):
            r_a = sign * rho * np.exp(2j * k * d)
            candidates.append((np.max(np.abs(ratio(r_a, False) - vacuum)), sign, d))
    # The empty cell sees r_a squared, so a sign is two equal fits; the slab tells them apart.
    best = min(candidates)[0]
    assert best < 5e-4
    slab = runs[True, 'graded']['flux'] / runs[True, 'pml']['flux']
    errors = [np.max(np.abs(ratio(sign * rho * np.exp(2j * k * d), True) - slab))
              for e, sign, d in candidates if e <= best + 1e-12]
    # The source used to add its own echo here: 9.9e-2 off the model at its best fit.
    assert min(errors) < 5e-3, errors
    # And that return is not small at first order: it is the warned-about few percent.
    assert np.max(np.abs(slab - 1)) > 2e-2


def test_check_warns_of_the_absorbers_return_with_a_plane_wave():
    report = fdtd.check(_problem(True), numerics=_numerics(fdtd.ConductiveAbsorber(cells=12)))
    assert report.ok
    (issue,) = [i for i in report.issues if i.code == 'numerics.absorber_reflection']
    assert issue.severity == 'warning' and issue.path == 'numerics.absorber'
    assert issue.actual == ('z-', 'z+') and 'exp(-3)' in issue.message
    report = fdtd.check(_problem(True), numerics=_numerics(fdtd.PML(cells=12)))
    assert not [i for i in report.issues if i.code == 'numerics.absorber_reflection']


def test_a_lossy_background_behind_the_absorber_is_refused():
    # The column's PML cannot be matched to a conductor, so the combination is refused.
    lossy = ConductiveMaterial(epsilon=1., conductivity=.05 * EPSILON0 * C0 / 1e-6)
    problem = _problem(False, background=lossy)
    problem = problem.replace(outputs=fdtd.Outputs(monitors=problem.outputs.monitors[:1]))
    assert fdtd.check(problem, numerics=_numerics(fdtd.PML(cells=12))).issues[0].code != \
        'fdtd.unsupported'                       # the PML refuses it in its own words
    report = fdtd.check(problem, numerics=_numerics(fdtd.ConductiveAbsorber(cells=12)))
    assert not report.ok
    (issue,) = [i for i in report.issues if i.severity == 'error']
    assert issue.code == 'fdtd.unsupported'
    assert 'ConductiveAbsorber' in issue.message and 'second incident wave' in issue.message
    assert 'ConductiveMaterial' in issue.message
