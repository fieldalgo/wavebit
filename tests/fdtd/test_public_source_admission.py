"""Reject malformed source declarations before duration, geometry or reuse reads them."""
import pytest

from wavebit import Units, fdtd
from wavebit.errors import ValidationError


def _problem(kind='PlaneWave'):
    pulse = fdtd.GaussianPulse(wavelength=.8, frequency_sigma_hz=2e13)
    if kind == 'PlaneWave':
        source = fdtd.PlaneWave('light', plane=fdtd.Plane(z=.4), pulse=pulse)
        domain = fdtd.Domain(size=(.1, .1, 1.6))
        boundaries = fdtd.Boundaries.periodic(z=fdtd.Open())
    else:
        source = (fdtd.CurrentDipole('light', center=(0., 0., 0.), pulse=pulse)
                  if kind == 'Dipole' else
                  fdtd.TFSF('light', box=fdtd.SnappedBox(size=(.8,) * 3), pulse=pulse))
        domain = fdtd.Domain(size=(1.6,) * 3)
        boundaries = fdtd.Boundaries.open()
    return fdtd.Problem(units=Units(length='um'),
        scene=fdtd.Scene(domain=domain, boundaries=boundaries), sources=(source,),
        outputs=fdtd.Outputs(final_fields=True))


def _numerics(duration='steps'):
    return fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.05),
        duration=(fdtd.Duration(steps=2) if duration == 'steps' else
                  fdtd.Duration(periods=2, source='light')))


def _refuse_at_all_doors(valid, bad, numerics, path, message):
    prepared = fdtd.prepare(valid, numerics=numerics)
    # Both direct entry and prepared reuse must diagnose the same declaration.
    for report in (fdtd.check(bad, numerics=numerics), prepared.validate(bad)):
        assert not report.ok
        (issue,) = report.issues
        assert issue.path == path
        assert message in issue.message
    for call in (lambda: fdtd.prepare(bad, numerics=numerics),
                 lambda: fdtd.run(bad, numerics=numerics),
                 lambda: prepared.run(bad)):
        with pytest.raises(ValidationError) as error:
            call()
        assert error.value.path == path
        assert message in str(error.value)
    # Refusing a bad replacement does not damage the valid preparation.
    assert fdtd.check(valid, numerics=numerics).ok
    assert prepared.validate(valid).ok


@pytest.mark.parametrize('source', [None, .5, {'name': 'light'}, 'light'])
def test_unknown_source_is_refused_before_reading_its_name(source):
    valid = _problem()
    _refuse_at_all_doors(valid, valid.replace(sources=(source,)), _numerics('periods'),
                         'sources[0]', 'expected PlaneWave, Dipole or TFSF')


@pytest.mark.parametrize('name', [None, 1, [], ''])
def test_source_name_is_checked_before_building_paths_or_resolving_duration(name):
    valid = _problem()
    bad = valid.replace(sources=(valid.sources[0].replace(name=name),))
    _refuse_at_all_doors(valid, bad, _numerics('periods'), 'sources[0].name',
                         'nonempty string')


def test_duplicate_source_names_are_refused_before_resolving_duration():
    valid = _problem()
    bad = valid.replace(sources=valid.sources * 2)
    for numerics in (_numerics(), _numerics('periods')):
        _refuse_at_all_doors(valid, bad, numerics, 'sources[1].name', 'unique')
    # Two distinct sources still enter the existing coherent-source path.
    pair = valid.replace(sources=valid.sources + (valid.sources[0].replace(name='second'),))
    assert fdtd.check(pair, numerics=_numerics('periods')).ok


@pytest.mark.parametrize('kind', ['PlaneWave', 'Dipole', 'TFSF'])
@pytest.mark.parametrize('duration', ['steps', 'periods'])
@pytest.mark.parametrize('pulse', [None, {'wavelength': .8}])
def test_pulse_type_is_checked_before_duration_or_prepared_signature(kind, duration, pulse):
    valid = _problem(kind)
    bad = valid.replace(sources=(valid.sources[0].replace(pulse=pulse),))
    _refuse_at_all_doors(valid, bad, _numerics(duration), 'sources.light.pulse',
                         'expected GaussianPulse')


@pytest.mark.parametrize('box', [None, .5, {'size': (1.,) * 3}, fdtd.Plane(z=0.),
                                fdtd.Box(size=(.8,) * 3)])
def test_tfsf_box_type_is_checked_before_physical_bounds(box):
    valid = _problem('TFSF')
    bad = valid.replace(sources=(valid.sources[0].replace(box=box),))
    _refuse_at_all_doors(valid, bad, _numerics(), 'sources.light.box', 'expected Region')
    # A real Region is still judged against the physical domain, before padding.
    outside = valid.replace(sources=(valid.sources[0].replace(
        box=fdtd.SnappedBox(size=(.8,) * 3, center=(0., 0., .7))),))
    report = fdtd.check(outside, numerics=_numerics())
    assert not report.ok
    assert report.issues[0].path == 'sources.light.box'
    assert 'outside the physical domain' in report.issues[0].message
