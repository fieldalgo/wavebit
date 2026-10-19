"""Apply the grazing diagnostic's public edits through the real preparation guard.

These tests do no time stepping: existing grazing return/stability measurements
hold the numerical rule. Here the candidates must be valid public declarations,
keep the physical region and unmentioned controls, and recheck the whole problem.
"""
import math
import warnings

import pytest

from wavebit import Material, Units, fdtd
from wavebit.errors import ValidationError
from wavebit.fdtd import _legacy as legacy
from wavebit.fdtd._lower_public import lower_problem
from wavebit.units import C0


def _column(cos=.6, name='light'):
    source = fdtd.PlaneWave(name, plane=fdtd.Plane(z=.4), theta=math.acos(cos),
                           pulse=fdtd.GaussianPulse(wavelength=1.,
                               frequency_sigma_hz=.005 * C0 / 1e-6))
    units = Units(length='um')
    return fdtd.Problem(units=units, scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.05, .05, 1.6)),
        boundaries=fdtd.Boundaries(**fdtd.Bloch.from_source(source, units=units,
                                    invariant_axes=('x', 'y')), z=fdtd.Open())),
        sources=(source,), outputs=fdtd.Outputs(final_fields=True))


def _numerics(cells=12, reflection=1e-8, stabilize=None, periods=160, name='light', **pml):
    return fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.05),
        duration=fdtd.Duration(periods=periods, source=name),
        absorber=fdtd.PML(cells=cells, reflection=reflection, stabilizing_loss=stabilize, **pml))


def _refusal(problem, numerics):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        report = fdtd.check(problem, numerics=numerics)
        assert not report.ok
        (issue,) = report.issues
        for door in (fdtd.prepare, fdtd.run):
            with pytest.raises(ValidationError) as error:
                door(problem, numerics=numerics)
            assert error.value.path == issue.path
            assert issue.message in str(error.value)
    for stale in ('z=PML(', 'Run(', 'Duration(periods>=', 'grow Scene(domain=)',
                  'background=Material(', 'give the axis one face'):
        assert stale not in issue.message
    return issue.message


def _candidates(message, numerics):
    out = []
    for line in message.splitlines():
        if line.startswith('updated = '):
            scope = {'numerics': numerics, 'fdtd': fdtd}
            exec(line, scope)
            out.append(scope['updated'])
    return out


def _apply(problem, numerics, updated):
    # Do not grow or replace the user's physical scene to make an offer pass.
    report = fdtd.check(problem, numerics=updated)
    assert report.ok, report.issues
    before, _ = lower_problem(problem, numerics)
    after, _ = lower_problem(problem, updated)
    extra = 2 * (updated.absorber.cells - numerics.absorber.cells) * numerics.grid.spacing
    assert after.domain.size[2] - before.domain.size[2] == pytest.approx(extra)
    assert after.domain.center == before.domain.center
    for name in ('wavelength', 'reference_media', 'allow_thin'):
        assert getattr(updated.absorber, name) == getattr(numerics.absorber, name)


@pytest.mark.parametrize('cells, reflection, periods', [
    (12, 1e-8, 40), (12, 1e-12, 40), (32, 1e-8, 200), (12, 1e-12, 200)])
def test_pure_grazing_offers_a_public_edit_preserving_stronger_controls(cells, reflection, periods):
    name = "li'ght"
    problem = _column(.342, name)
    numerics = _numerics(cells, reflection, periods=periods, name=name,
                         wavelength=1., allow_thin=True,
                         reference_media={'z-': Material(epsilon=1.)})
    message = _refusal(problem, numerics)
    assert 'Grazing order refused' in message
    assert 'common numerical controls to every Open face' in message
    assert 'not a guarantee that the whole problem passes' in message
    assert 'keep problem.scene.domain unchanged' in message
    assert 'Refused too' in message and "'z+'" in message and "'z-'" in message
    assert 'measured PML return at f0' in message
    (updated,) = _candidates(message, numerics)
    assert updated.absorber.cells >= cells
    assert updated.absorber.reflection <= reflection
    assert updated.duration.source == name
    assert updated.duration.periods >= max(periods, 160)
    _apply(problem, numerics, updated)


@pytest.mark.parametrize('cos, reflection, stabilize, periods', [
    (.6, 1e-8, .003, 160), (.38, 1e-12, .0005, 40), (.6, 1e-8, .003, 200)])
def test_lossy_candidates_pass_with_external_padding_and_preserved_controls(cos, reflection,
                                                                          stabilize, periods):
    problem = _column(cos)
    numerics = _numerics(reflection=reflection, stabilize=stabilize, periods=periods,
                         wavelength=1., allow_thin=True,
                         reference_media={'z-': Material(epsilon=1.)})
    message = _refusal(problem, numerics)
    assert 'Stabilized face refused' in message
    assert 'padding outside the physical domain automatically' in message
    assert 'bound mode grow again' in message
    assert 'not a guarantee that the whole problem passes' in message
    candidates = _candidates(message, numerics)
    assert candidates
    for updated in candidates:
        assert updated.duration.periods >= periods
        _apply(problem, numerics, updated)


@pytest.mark.parametrize('face', ['z-', 'z+'])
def test_different_reference_media_require_a_common_edit_and_full_recheck(face):
    problem = _column()
    numerics = _numerics(stabilize=.003, reference_media={face: Material(epsilon=1.01)})
    message = _refusal(problem, numerics)
    assert 'different resolved reference media' in message
    assert 'check all faces together' in message
    assert 'no numeric candidate is offered' in message
    assert 'Refused too' in message and "'z+'" in message and "'z-'" in message
    assert not _candidates(message, numerics)
    # Follow the qualitative advice with the same common edit on both faces.
    updated = numerics.replace(absorber=numerics.absorber.replace(cells=24))
    _apply(problem, numerics, updated)


def test_floor_refusal_keeps_the_measured_limit_and_requires_geometry_change():
    problem = _column(.15)
    numerics = _numerics(cells=24, reflection=1e-16)
    message = _refusal(problem, numerics)
    assert 'no measured setting meets the tolerance' in message
    assert 'targets past 1e-16 were not measured' in message
    assert 'change the period, the angle or the wavelengths' in message
    assert not _candidates(message, numerics)
    # More padding/time do not turn the measured floor into an accepted setting.
    _refusal(problem, numerics.replace(absorber=numerics.absorber.replace(cells=48),
                                      duration=fdtd.Duration(periods=320, source='light')))
    assert fdtd.check(_column(.3), numerics=numerics).ok


def test_public_context_does_not_change_private_engine_guidance():
    problem, numerics = _column(), _numerics(stabilize=.003)
    scene, _ = lower_problem(problem, numerics)
    with pytest.raises(ValueError) as before:
        legacy.plan(scene)
    public_message = _refusal(problem, numerics)
    with pytest.raises(ValueError) as after:
        legacy.plan(scene)
    assert str(after.value) == str(before.value)
    assert 'z=PML(' in str(after.value) and 'grow Scene(domain=)' in str(after.value)
    assert 'numerics.replace(' in public_message
