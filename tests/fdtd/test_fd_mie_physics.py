"""Public production solves against Mie; no cached measurement is the oracle."""
from functools import lru_cache
import warnings

import pytest

import wavebit.fdtd._legacy as fd
from studies.fdtd.subpixel_mie_sphere.fdtd_subpixel_mie_sphere import sphere_scene, read_result
from studies.fdtd.subpixel_mie_cylinder.fdtd_subpixel_mie_cylinder import convergence


pytestmark = pytest.mark.slow

# The design fixture samples three fixed subcell translations. These are
# experiment coordinates, not fitted parameters or accuracy thresholds.
PLACEMENTS = ((0., 0., 0.), (.5, 1/3, .2), (.25, 2/3, .4))
LEVELS = (20, 28, 40)


@lru_cache(maxsize=None)
def measured(cpw, eps=2.25, offset=PLACEMENTS[0]):
    pytest.importorskip('jax')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', fd.CoarseMeshWarning)
        result = fd.run(sphere_scene(cpw=cpw, eps=eps, offset=offset), backend='jax')
    return read_result(result, eps)


# The weak sphere's sweep shares nothing with the glass rows the tests below read, so CI
# may run it apart (tools/run_verification.py --shard); the glass sweep stays with them.
@pytest.mark.parametrize('eps', (2.25, pytest.param(1.1, marks=pytest.mark.shard_unit)))
def test_public_sphere_mie_convergence(eps):
    errors = [[measured(cpw, eps, place)['error'] for place in PLACEMENTS]
              for cpw in LEVELS]
    fits = convergence(LEVELS, errors)
    # Predeclared G6 criteria, measured before this test: glass mean/RMS
    # orders 2.02267/2.02268 and weak 2.07490/2.07494; scatters below .01.
    # Both fits are required: opposite placement errors must not cancel.
    for key in ('mean', 'rms'):
        assert fits[key]['order'] is not None and fits[key]['order'] >= 1.8, fits
        assert fits[key]['scatter'] is not None and fits[key]['scatter'] < .1, fits
    # The independent open-scene control measured a 9.7e-5 relative floor.
    # The predeclared tenfold separation prevents fitting the instrument.
    assert abs(fits['mean_errors'][-1]) > 10*9.7e-5, fits


def test_public_lossless_sphere_absorption():
    row = measured(40, 2.25, PLACEMENTS[0])
    # Origin at 40 cells: Qabs=2.55936e-7. Twice that measured residual,
    # rounded upward to one significant digit, is 6e-7. Reading the outer
    # scattered-field surface as absorption instead gives |Qabs|~.867.
    assert abs(row['qabs']) < 6e-7, row['qabs']
    # Existing settled-field warning threshold is 1e-5. The glass fixture
    # measured <=1.6e-11; require the design's extra decade below the warning.
    assert max(row['decay'].values()) < 1e-6, row['decay']


def test_public_sphere_radiated_power_closures():
    row = measured(40, 2.25, PLACEMENTS[0])
    # Independent surfaces: measured relative gap 7.60701e-8 at the
    # origin, rounded up to 1e-7. Angular integral: 1.68863e-4, rounded
    # up to 2e-4. Doubling angular orders changes it by <8e-15.
    assert row['qsca'] > 0, row['qsca']
    assert abs(row['ffq']/row['qsca']-1) < 1e-7
    assert abs(row['farfield_integral']/row['qsca']-1) < 2e-4


def test_public_sphere_optical_phase_origin():
    row = measured(40, 2.25, PLACEMENTS[0])
    # G6's predeclared 1.6% criterion; measured closure is 1.23656%.
    # Referring the incident amplitude to the entry plane, rather than
    # the radiation transform's origin, changes the complex phase.
    assert abs(row['optical_closure']) < .016, row['optical_closure']


def test_public_sphere_angular_pattern():
    row = measured(40, 2.25, PLACEMENTS[0])
    assert set(row['pattern']) == {'0.0', '90.0'}
    for cut in row['pattern'].values():
        # Largest 40-cell glass peak error across three placements is
        # .00941144, rounded upward to .01. At theta=90 the relative
        # error is at most .0154881, rounded upward to .02. The side
        # direction prevents a large forward peak hiding a wrong normal.
        assert cut['peak_error'] < .01, cut['peak_error']
        assert abs(cut['values'][18]/cut['reference'][18]-1) < .02


@pytest.mark.shard_unit
def test_public_high_contrast_material_rule():
    pytest.importorskip('jax')
    # Additional radius-.1 control, not the radius-.25 convergence fixture.
    # Measured at 40 cells: 24 -> 48 periods changes Qsca by -4.022e-6
    # relative. The longer run has Mie error .0205768, |Qabs|=2.70344e-6
    # and maximum monitor tail 1.48130e-11. See the measurement log.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', fd.CoarseMeshWarning)
        result = fd.run(sphere_scene(cpw=40, eps=12., periods=48., radius=.1), backend='jax')
    row = read_result(result, 12., radius=.1)
    # Round the measured 2.05768% error upward to 3%. This guards the
    # high-contrast material rule; the separate 40/56/80, radius-.25 sweep
    # supplies the convergence claim. Swapping arithmetic/harmonic is a mutant.
    assert abs(row['error']) < .03, row['error']
    # Twice the measured absorption residual, rounded upward to one digit.
    assert abs(row['qabs']) < 6e-6, row['qabs']
    assert max(row['decay'].values()) < 1e-6, row['decay']
