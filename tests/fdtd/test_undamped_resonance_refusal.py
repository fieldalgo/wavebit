"""An undamped Lorentz pole resonant at a wavelength the plan reads.

Its permittivity is infinite there, so the material has no finite index and no mesh
resolves it. The plan refuses with the same structured ``fdtd.invalid`` the detuned scene
gets (whose required spacing goes to zero with the detuning), naming the pole and the
wavelength, instead of a raw ``ZeroDivisionError``; nothing returns inf or NaN.
"""
import cmath
import math

import pytest

from wavebit import DispersiveMaterial, Lorentz, Units, fdtd
from wavebit.errors import ValidationError
from wavebit.units import C0


def problem(detune=0.0, wavelengths=(.9, 1., 1.1), gamma=0.):
    w0 = 2 * math.pi * C0 / 1e-6 * (1 + detune)
    material = DispersiveMaterial(epsilon_inf=2., poles=(
        Lorentz(omega_0=w0, gamma=gamma, strength=.1 * w0 ** 2),))
    return fdtd.Problem(units=Units(length='um'), scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.2, .2, 1.3)),
        boundaries=fdtd.Boundaries.periodic(z=fdtd.Open()),
        structures=(fdtd.Structure('ball', shape=fdtd.Sphere(radius=.08), material=material),)),
        sources=(fdtd.PlaneWave('pw', plane=fdtd.Plane(z=.45),
                                pulse=fdtd.GaussianPulse(wavelength=1., frequency_sigma_hz=.15 * C0 / 1e-6)),),
        outputs=fdtd.Outputs(monitors=(fdtd.DiffractionMonitor(
            'T', plane=fdtd.Plane(z=-.45), source='pw', wave='transmitted',
            wavelengths=wavelengths, orders=(0, 0)),)))


NUMERICS = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.05), duration=fdtd.Duration(steps=40),
                         absorber=fdtd.PML(cells=6, wavelength=1., allow_thin=True))


def issue(detune):
    report = fdtd.check(problem(detune), numerics=NUMERICS)
    assert not report.ok
    (found,) = report.issues
    return found


def test_resonant_monitor_wavelength_is_a_structured_refusal_not_zero_division():
    found = issue(0.0)
    assert found.code == 'fdtd.invalid'
    for text in ("DiffractionMonitor('T') asks for wavelength 1", 'structures[0]',
                 'poles[0] = Lorentz(omega_0=6.28319, gamma=0)', 'undamped resonance',
                 'permittivity is infinite', 'gamma > 0'):
        assert text in found.message
    for door in (fdtd.prepare, fdtd.run):
        with pytest.raises(ValidationError) as caught:
            door(problem(), numerics=NUMERICS, backend='numpy')
        assert caught.value.code == 'fdtd.invalid'
        assert 'undamped resonance at wavelength 1' in str(caught.value)


@pytest.mark.parametrize('detune', [1e-3, 1e-9, 1e-15])
def test_refusal_is_the_limit_of_the_detuned_scene(detune):
    # Detuned (eps -> +inf), the same wavelength is refused for a spacing that shrinks with
    # the detuning. From below (eps -> -inf) the material reads as a metal and the scene's
    # next refusal is its own; either way nothing is returned at the undefined wavelength.
    found = issue(detune)
    assert found.code == 'fdtd.invalid'
    assert "DiffractionMonitor('T') asks for wavelength 1, where structures[0]" in found.message


def test_index_matches_closed_form_lorentz_off_resonance_and_refuses_on_it():
    from wavebit.fdtd._prepare.checks import UndampedResonance, _index_at
    from wavebit.materials._time_domain import Dispersive, Lorentz as TimeLorentz
    w0 = 2 * math.pi
    material = Dispersive(eps_inf=2., poles=(TimeLorentz(omega_0=w0, gamma=0., strength=.1 * w0 * w0),))
    for wavelength in (.5, .9, 1.1, 3.):
        w = 2 * math.pi / wavelength
        expected = cmath.sqrt(2. + .1 * w0 * w0 / (w0 * w0 - w * w))
        n, skin = _index_at(material, wavelength)
        if skin is None:
            assert n == pytest.approx(expected.real, rel=1e-13)
    with pytest.raises(UndampedResonance, match=r'poles\[0\].*wavelength 1\b'):
        _index_at(material, 1.0, where='structures[0]')


def test_damped_pole_at_the_same_wavelength_is_unchanged():
    # Physical damping keeps eps finite: the check reads it as before (coarse, not resonant).
    found = fdtd.check(problem(gamma=1e13), numerics=NUMERICS).issues
    assert found and all('undamped' not in i.message for i in found)


def test_post_run_round_trip_declines_to_judge_at_a_resonant_source_wavelength():
    # The column diagnostic reads the source's centre wavelength after the run; there an
    # undamped pole has no finite index, so it reports "not judged" instead of raising.
    from types import SimpleNamespace
    from wavebit.fdtd._assemble import _round_trip
    from wavebit.materials._time_domain import Dispersive, Lorentz as TimeLorentz
    w0 = 2 * math.pi
    material = Dispersive(eps_inf=2., poles=(TimeLorentz(omega_0=w0, gamma=0., strength=.1 * w0 * w0),))
    scene = SimpleNamespace(background=None, domain=SimpleNamespace(size=(1., 1., 2.)),
                            structures=(SimpleNamespace(material=material),))
    source = lambda wavelength: SimpleNamespace(band=SimpleNamespace(wavelength=wavelength),
                                                theta=0., axis=2)
    assert _round_trip(scene, source(1.0), None) is None
    assert math.isfinite(_round_trip(scene, source(1.3), None))
