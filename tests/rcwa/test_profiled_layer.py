"""``rcwa.ProfiledLayer``: z-varying cross-sections sliced on the region path.

The pre-registered criteria (2026-11-18), each held by a test below:

(a) A vertical, unrounded Trapezoid is a Rectangle layer, and a Loft with equal outlines a
    Polygon layer, to the bit; a vertical wall whose angle carries a derivative stays sliced
    and agrees with them to 1e-12.
(b) The midpoint staircase converges at second order: a straight taper's observed order lies
    in [1.8, 2.2] for TE and TM, and a rounded profile under the default arc-length slicing
    keeps an observed order of at least 1.8 at its finest levels.
(c) Automatic derivatives with respect to the sidewall angle, the rounding radii, the width
    and the height agree with central differences to 1e-6, relative.
(f) NumPy, PyTorch and JAX agree to 1e-12.
(g) A sampled (raster) layer beside a ProfiledLayer is refused with ``layer.profile_raster``.

RETICOLO is the external oracle, in ``tests/references/test_reticolo_taper_golden.py``.

Lines run on the 2D path with ``orders=(M, 0)`` and Li's factorization, which is exact for the
full-height rectangles their slices are and a hundred times faster than the default NVF
normal field on them (the two agree to 1e-14; the golden test checks the default too).
"""
import math

import numpy as np
import pytest

from wavebit import Material, Units, ValidationError, rcwa
from wavebit.rcwa import diagnostics

SI = Material(epsilon=12.25)
AIR = Material(epsilon=1.0)
PERIOD = 0.5


def _problem(layers, theta=0.2, polarization='s', period=(PERIOD, PERIOD)):
    return rcwa.Problem(units=Units(length='um'),
                        stack=rcwa.Stack(period=period, bottom=Material(epsilon=2.25), layers=tuple(layers)),
                        excitation=rcwa.Excitation(wavelength=0.6, sources=(
                            rcwa.PlaneWave(theta=theta, phi=0., polarization=polarization),)))


def _line(height=0.2, **geometry):
    geometry = {'cd': 0.2, **geometry}
    return rcwa.ProfiledLayer(thickness=height, background=AIR,
                              regions=(rcwa.Region(geometry=rcwa.Trapezoid(**geometry), material=SI),))


def _numerics(count=16, rule='arclength', orders=10, **slicing):
    slicing = {'estimate': None, **slicing}
    return rcwa.Numerics(orders=(orders, 0), factorization='li',
                         slicing=rcwa.Slicing(count=count, rule=rule, **slicing))


def _efficiencies(result):
    return np.concatenate([np.ravel(np.asarray(result.R)), np.ravel(np.asarray(result.T))])


# (a) -----------------------------------------------------------------------------------------

@pytest.mark.parametrize('cd_at', ('bottom', 'middle', 'top'))
@pytest.mark.parametrize('swa', (math.pi/2, (math.pi/2, math.pi/2)))
def test_a_vertical_trapezoid_is_the_rectangle_layer_to_the_bit(cd_at, swa):
    rectangle = rcwa.Layer(thickness=0.2, background=AIR, regions=(
        rcwa.Region(geometry=rcwa.Rectangle(size=(0.2, PERIOD), center=(0.031, 0.)), material=SI),))
    reference = rcwa.run(_problem([rectangle]), numerics=_numerics())
    result = rcwa.run(_problem([_line(swa=swa, center=0.031, cd_at=cd_at)]), numerics=_numerics())
    assert np.array_equal(result.R, reference.R) and np.array_equal(result.T, reference.T)
    report = result.diagnostics['z_slicing'][0]
    assert report['merged'] and report['count'] == 1 and report['staircase_error'] == 0.


def test_a_vertical_trapezoid_is_the_rectangle_layer_under_the_default_factorization():
    rectangle = rcwa.Layer(thickness=0.2, background=AIR, regions=(
        rcwa.Region(geometry=rcwa.Rectangle(size=(0.2, PERIOD)), material=SI),))
    numerics = rcwa.Numerics(orders=(6, 0))
    reference = rcwa.run(_problem([rectangle]), numerics=numerics)
    result = rcwa.run(_problem([_line(swa=np.pi/2)]), numerics=numerics)
    assert np.array_equal(result.R, reference.R) and np.array_equal(result.T, reference.T)


def test_a_loft_with_equal_outlines_is_the_polygon_layer_to_the_bit():
    square = ((-0.1, -0.12), (0.12, -0.1), (0.1, 0.12), (-0.12, 0.1))
    numerics = rcwa.Numerics(orders=(3, 3), factorization='laurent', slicing=rcwa.Slicing(count=4))
    polygon = rcwa.Layer(thickness=0.15, background=AIR, regions=(
        rcwa.Region(geometry=rcwa.Polygon(vertices=square), material=SI),))
    loft = rcwa.ProfiledLayer(thickness=0.15, background=AIR, regions=(
        rcwa.Region(geometry=rcwa.Loft(bottom=square, top=square), material=SI),))
    reference = rcwa.run(_problem([polygon]), numerics=numerics)
    result = rcwa.run(_problem([loft]), numerics=numerics)
    assert np.array_equal(result.R, reference.R) and np.array_equal(result.T, reference.T)


def test_a_vertical_wall_that_carries_a_derivative_stays_sliced():
    torch = pytest.importorskip('torch')
    reference = rcwa.run(_problem([_line(swa=math.pi/2)]), numerics=_numerics(count=8))
    swa = torch.tensor(math.pi/2, dtype=torch.float64, requires_grad=True)
    prepared = rcwa.prepare(_problem([_line(swa=math.pi/2)]), numerics=_numerics(count=8), backend='torch')
    result = prepared.run(_problem([_line(swa=swa)]))
    assert not result.diagnostics['z_slicing'][0]['merged']
    assert result.diagnostics['z_slicing'][0]['count'] == 8
    values = np.concatenate([result.R.detach().numpy().ravel(), result.T.detach().numpy().ravel()])
    assert np.max(np.abs(values - _efficiencies(reference))) < 1e-12


# (b) -----------------------------------------------------------------------------------------

def _observed_orders(layer, counts, rule, polarization):
    values = [_efficiencies(rcwa.run(_problem([layer], polarization=polarization),
                                     numerics=_numerics(count=n, rule=rule, orders=15))) for n in counts]
    steps = [np.max(np.abs(b - a)) for a, b in zip(values[:-1], values[1:])]
    return [math.log2(a/b) for a, b in zip(steps[:-1], steps[1:])]


@pytest.mark.parametrize('polarization', ('s', 'p'))
def test_a_straight_taper_converges_at_second_order(polarization):
    orders = _observed_orders(_line(swa=math.radians(80.)), (16, 32, 64), 'uniform', polarization)
    assert 1.8 <= orders[-1] <= 2.2


@pytest.mark.slow
@pytest.mark.parametrize('polarization', ('s', 'p'))
def test_a_rounded_profile_keeps_second_order_under_arc_length_slicing(polarization):
    """Measured 2026-11-18 at levels 256/512: 1.89 to 2.07; uniform slicing falls to 1.35 to 1.48."""
    layer = _line(cd=0.22, swa=(math.radians(82.), math.radians(86.)), top_rounding=0.03, footing=0.02)
    assert _observed_orders(layer, (128, 256, 512), 'arclength', polarization)[-1] >= 1.8


def test_arc_length_slicing_crowds_the_rounded_top():
    layer = _line(swa=math.radians(84.), top_rounding=0.06)
    result = rcwa.run(_problem([layer]), numerics=_numerics(count=16))
    steps = np.diff(result.diagnostics['z_slicing'][0]['fractions'])
    assert steps[-1] < 0.5/16 < 1/16 < steps[0]


# (c) -----------------------------------------------------------------------------------------

_POINT = {'cd': 0.2, 'swa': 1.4, 'top_rounding': 0.03, 'footing': 0.02, 'height': 0.2}


def _rounded(values):
    values = {**_POINT, **values}
    height = values.pop('height')
    return _problem([_line(height=height, cd_at='middle', **values)], polarization='p')


def _central_difference(name, numerics, step=1e-6):
    """On one preparation: the slice boundaries it planned stay put, as they do for the derivative.

    A fresh ``rcwa.run`` at each point would plan arc-length boundaries for that point and
    difference two different staircases.
    """
    prepared = rcwa.prepare(_rounded({}), numerics=numerics)
    def total(value):
        return float(np.sum(prepared.run(_rounded({name: value})).R))
    return (total(_POINT[name] + step) - total(_POINT[name] - step))/(2*step)


@pytest.mark.parametrize('name', tuple(_POINT))
def test_torch_derivatives_match_central_differences(name):
    torch = pytest.importorskip('torch')
    prepared = rcwa.prepare(_rounded({}), numerics=_numerics(count=16), backend='torch')
    value = torch.tensor(_POINT[name], dtype=torch.float64, requires_grad=True)
    torch.sum(prepared.run(_rounded({name: value})).R).backward()
    reference = _central_difference(name, _numerics(count=16))
    assert abs(float(value.grad) - reference) <= 1e-6*abs(reference)


@pytest.mark.slow
@pytest.mark.parametrize('name', ('swa', 'height'))
def test_jax_derivatives_match_central_differences(name):
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    numerics = _numerics(count=4)
    prepared = rcwa.prepare(_rounded({}), numerics=numerics, backend='jax')
    gradient = jax.grad(lambda v: jax.numpy.sum(prepared.run(_rounded({name: v})).R))(_POINT[name])
    reference = _central_difference(name, numerics)
    assert abs(float(gradient) - reference) <= 1e-6*abs(reference)


# (f) -----------------------------------------------------------------------------------------

def test_torch_and_numpy_agree():
    pytest.importorskip('torch')
    reference = rcwa.run(_rounded({}), numerics=_numerics(count=16))
    result = rcwa.run(_rounded({}), numerics=_numerics(count=16), backend='torch')
    values = np.concatenate([result.R.numpy().ravel(), result.T.numpy().ravel()])
    assert np.max(np.abs(values - _efficiencies(reference))) < 1e-12


@pytest.mark.slow
def test_jax_and_numpy_agree():
    pytest.importorskip('jax')
    reference = rcwa.run(_rounded({}), numerics=_numerics(count=4))
    result = rcwa.run(_rounded({}), numerics=_numerics(count=4), backend='jax')
    assert np.max(np.abs(_efficiencies(result) - _efficiencies(reference))) < 1e-12


# (g) -----------------------------------------------------------------------------------------

def test_a_raster_layer_beside_a_profiled_layer_is_refused():
    raster = rcwa.Layer(thickness=0.1, epsilon=np.array([[12.25]*4 + [1.]*4]))
    problem = _problem([raster, _line(swa=1.45)])
    with pytest.raises(ValidationError) as caught:
        rcwa.run(problem, numerics=_numerics())
    assert caught.value.issue.code == 'layer.profile_raster'
    assert caught.value.path == 'stack.layers[0]'
    report = rcwa.check(problem, numerics=_numerics())
    assert [issue.code for issue in report.issues] == ['layer.profile_raster']


def test_a_profile_shape_in_an_ordinary_layer_is_refused():
    with pytest.raises(ValidationError) as caught:
        rcwa.Layer(thickness=0.1, regions=(rcwa.Region(geometry=rcwa.Trapezoid(cd=0.2), material=SI),))
    assert caught.value.issue.code == 'layer.profile_geometry'


# The contract ----------------------------------------------------------------------------------

def test_the_report_names_each_public_layer():
    cap = rcwa.Layer(thickness=0.05, background=Material(epsilon=2.25))
    result = rcwa.run(_problem([_line(swa=1.45), cap]), numerics=_numerics(count=8))
    first, second = result.diagnostics['z_slicing']
    assert second is None
    assert (first['count'], first['planned_count'], first['rule'], first['merged']) == (8, 8, 'arclength', False)
    assert len(first['fractions']) == 9 and first['fractions'][0] == 0. and first['fractions'][-1] == 1.
    assert 'z_slicing_converged' not in result.validity.checks       # a stated count is not assessed


def test_the_automatic_count_doubles_until_the_estimate_meets_the_tolerance():
    result = rcwa.run(_problem([_line(swa=math.radians(80.))]),
                      numerics=rcwa.Numerics(orders=(10, 0), factorization='li', slicing=rcwa.Slicing(tolerance=1e-4)))
    report = result.diagnostics['z_slicing'][0]
    assert report['levels'] == tuple(4*2**k for k in range(len(report['levels'])))
    assert report['count'] == report['levels'][-1]
    assert report['converged'] and report['staircase_error'] <= 1e-4 and report['source'] == 'prepare'
    assert 1.8 <= report['observed_order'] <= 2.2
    assert bool(result.validity.checks['z_slicing_converged'])


def test_stopping_short_of_the_tolerance_warns_and_is_recorded():
    with pytest.warns(diagnostics.SlicingWarning):
        result = rcwa.run(_problem([_line(swa=math.radians(80.))]), numerics=rcwa.Numerics(
            orders=(10, 0), factorization='li', slicing=rcwa.Slicing(tolerance=1e-9, max_count=16)))
    report = result.diagnostics['z_slicing'][0]
    assert report['count'] == 16 and not report['converged']
    assert not bool(result.validity.checks['z_slicing_converged'])


def test_an_estimate_on_every_run_solves_half_the_slices_too():
    numerics = _numerics(count=16, estimate='run')
    prepared = rcwa.prepare(_problem([_line(swa=1.4)]), numerics=numerics)
    report = prepared.run(_problem([_line(swa=1.38)])).diagnostics['z_slicing'][0]
    half = _efficiencies(rcwa.run(_problem([_line(swa=1.38)]), numerics=_numerics(count=8)))
    full = _efficiencies(rcwa.run(_problem([_line(swa=1.38)]), numerics=_numerics(count=16)))
    assert report['source'] == 'run'
    assert np.isclose(report['staircase_error'], np.max(np.abs(full - half))/3, rtol=1e-12)


def test_one_preparation_runs_other_angles_and_a_vertical_wall():
    numerics = _numerics(count=8, rule='uniform')
    prepared = rcwa.prepare(_problem([_line(swa=1.4)]), numerics=numerics)
    for swa in (1.38, math.pi/2):
        reused = prepared.run(_problem([_line(swa=swa)]))
        fresh = rcwa.run(_problem([_line(swa=swa)]), numerics=numerics)
        assert np.array_equal(reused.R, fresh.R) and np.array_equal(reused.T, fresh.T)


def test_a_profile_is_the_loft_it_describes():
    bottom = ((-0.12, -0.12), (0.12, -0.12), (0.12, 0.12), (-0.12, 0.12))
    top = tuple((0.6*x, 0.6*y) for x, y in bottom)

    def section(z):
        return tuple((b[0] + (t[0] - b[0])*z/0.15, b[1] + (t[1] - b[1])*z/0.15) for b, t in zip(bottom, top))

    numerics = rcwa.Numerics(orders=(2, 2), factorization='laurent', slicing=rcwa.Slicing(count=4, rule='uniform'))
    def layer(geometry):
        return rcwa.ProfiledLayer(thickness=0.15, background=AIR, regions=(rcwa.Region(geometry=geometry, material=SI),))
    loft = rcwa.run(_problem([layer(rcwa.Loft(bottom=bottom, top=top))]), numerics=numerics)
    profile = rcwa.run(_problem([layer(rcwa.Profile(section=section))]), numerics=numerics)
    assert np.max(np.abs(_efficiencies(loft) - _efficiencies(profile))) < 1e-14


@pytest.mark.parametrize('geometry, code, path', [
    ({'swa': 0.}, 'shape.profile', 'geometry.swa'),
    ({'swa': (1.4, 3.2)}, 'shape.profile', 'geometry.swa'),
    ({'cd': -0.1}, 'shape.profile', 'geometry.cd'),
    ({'top_rounding': -0.01}, 'shape.profile', 'geometry.top_rounding'),
    ({'cd_at': 'centre'}, 'shape.profile', 'geometry.cd_at'),
    ({'swa': (1.4, 1.4, 1.4)}, 'shape.dimension', 'geometry.swa'),
])
def test_a_trapezoid_refuses_what_is_no_profile(geometry, code, path):
    with pytest.raises(ValidationError) as caught:
        rcwa.Trapezoid(**{'cd': 0.2, **geometry})
    assert (caught.value.issue.code, caught.value.path) == (code, path)


@pytest.mark.parametrize('geometry, height, period, fragment', [
    ({'cd': 0.1, 'swa': 1.0}, 0.2, (PERIOD, PERIOD), 'closes inside the layer'),
    ({'top_rounding': 0.15}, 0.2, (PERIOD, PERIOD), 'top width'),
    ({'top_rounding': 0.09, 'footing': 0.12, 'cd': 0.45}, 0.2, (PERIOD, PERIOD), 'up a sidewall'),
    ({'cd': 0.48, 'footing': 0.03}, 0.2, (PERIOD, PERIOD), 'periodic image'),
])
def test_a_profile_that_does_not_fit_its_layer_is_refused(geometry, height, period, fragment):
    with pytest.raises(ValidationError) as caught:
        rcwa.run(_problem([_line(height=height, **geometry)], period=period), numerics=_numerics(count=4))
    assert caught.value.issue.code == 'shape.profile' and fragment in str(caught.value)


def test_a_line_needs_a_rectangular_cell():
    problem = rcwa.Problem(units=Units(length='um'), stack=rcwa.Stack(
        lattice=((PERIOD, 0.), (0., PERIOD)), layers=(_line(swa=1.45),)),
        excitation=rcwa.Excitation(wavelength=0.6, sources=(rcwa.PlaneWave(theta=0., phi=0.),)))
    with pytest.raises(ValidationError) as caught:
        rcwa.run(problem, numerics=_numerics(count=4))
    assert caught.value.issue.code == 'shape.profile_cell'


@pytest.mark.parametrize('slicing, path', [
    ({'count': 0}, 'numerics.slicing.count'),
    ({'count': 2.}, 'numerics.slicing.count'),
    ({'rule': 'curvature'}, 'numerics.slicing.rule'),
    ({'tolerance': 0.}, 'numerics.slicing.tolerance'),
    ({'estimate': 'always'}, 'numerics.slicing.estimate'),
    ({'min_count': 8, 'max_count': 4}, 'numerics.slicing.max_count'),
])
def test_slicing_refuses_a_malformed_declaration(slicing, path):
    with pytest.raises(ValidationError) as caught:
        rcwa.Slicing(**slicing)
    assert (caught.value.issue.code, caught.value.path) == ('numerics.slicing', path)


def test_a_profiled_problem_saves_as_a_job_unless_it_holds_a_callable():
    from wavebit.jobs import Job
    problem = _problem([_line(swa=(1.4, 1.45), top_rounding=0.02)])
    job = Job(solver='rcwa', problem=problem, numerics=_numerics(count=8))
    assert Job.from_json(job.to_json()) == job
    profile = rcwa.ProfiledLayer(thickness=0.1, regions=(rcwa.Region(
        geometry=rcwa.Profile(section=lambda z: ((0., 0.), (0.1, 0.), (0., 0.1))), material=SI),))
    with pytest.raises(ValidationError, match='Callables'):
        Job(solver='rcwa', problem=_problem([profile]), numerics=_numerics()).to_json()


def test_check_reports_a_sound_profile_without_solving():
    report = rcwa.check(_problem([_line(swa=1.45, footing=0.01)]), numerics=_numerics(count=8))
    assert report.issues == () or all(issue.severity != 'error' for issue in report.issues)


def test_the_tools_that_read_one_literal_layer_refuse_a_profiled_one():
    from wavebit import UnsupportedFeatureError
    from wavebit.rcwa import operators
    problem = _problem([_line(swa=1.45)])
    with pytest.raises(UnsupportedFeatureError) as caught:
        rcwa.sample_structure(problem, layer=0)
    assert caught.value.issue.code == 'structure.profiled_layer'
    with pytest.raises(UnsupportedFeatureError) as caught:
        operators.assemble(problem, numerics=_numerics())
    assert caught.value.issue.code == 'operators.profiled_layer'
