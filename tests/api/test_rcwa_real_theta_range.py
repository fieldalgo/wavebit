"""A real ``theta`` lies in ``[-pi/2, pi/2]``, and both sides of the normal end alike.

``theta`` is the angle of the incident light from the inward normal of the lit face
(``docs/reference/common/conventions.md``), so a real angle past ``pi/2`` has no direction
toward the structure. The check refuses one with ``source.direction``. Until then a negative
one was folded by ``sin`` (``theta = -2`` lit as ``-(pi - 2)``, with no notice) while a positive
one was held at grazing. The bound is ``float32(pi/2)``, which admits a single-precision
grazing angle; there, and for a traced angle the check cannot read, the engine now holds both
sides at grazing.
"""
import numpy as np
import pytest

from wavebit import Material, Units, rcwa
from wavebit.errors import ValidationError

GRAZING = float(np.float32(np.pi / 2))


def problem(theta, phi=.4, source='plane'):
    if source == 'plane':
        light = rcwa.PlaneWave(side='top', polarization='p', theta=theta, phi=phi)
    else:
        light = rcwa.GaussianBeam(side='top', polarization='p', theta=theta, phi=phi,
                                  waist=2., sampling=(5, 5))
    return rcwa.Problem(
        units=Units(length='um'),
        stack=rcwa.Stack(period=(.7, .8), top=Material(2.25), bottom=Material(3.24),
                         layers=(rcwa.Layer(thickness=.18, background=Material(4.2)),)),
        excitation=rcwa.Excitation(wavelength=.8, sources=(light,)))


@pytest.mark.parametrize('source', ['plane', 'beam'])
@pytest.mark.parametrize('theta', [-2., 2., -1.6, 1.6, -4., 7., -np.pi, -2. + 0j,
                                   np.nextafter(GRAZING, 2.), -np.nextafter(GRAZING, 2.)])
def test_a_real_theta_past_a_right_angle_is_refused_at_check(theta, source):
    report = rcwa.check(problem(theta, source=source), numerics=rcwa.Numerics())
    assert not report.ok
    issue, = report.issues
    assert issue.code == 'source.direction'
    assert issue.path == 'excitation.sources[0].theta'
    assert '[-pi/2, pi/2]' in issue.message and 'phi' in issue.message
    # run raises the same refusal.
    with pytest.raises(ValidationError) as caught:
        rcwa.run(problem(theta, source=source), numerics=rcwa.Numerics())
    assert caught.value.code == 'source.direction'


@pytest.mark.parametrize('theta', [GRAZING, -GRAZING, np.pi / 2 + .3j, -2. + .3j, 1.2, -1.2])
def test_angles_in_range_and_complex_angles_pass_the_check(theta):
    assert rcwa.check(problem(theta), numerics=rcwa.Numerics()).ok


def test_the_two_sides_of_the_normal_are_mirror_images():
    """On a uniform stack ``-t`` and ``t`` at one ``phi`` are mirror images: same R and T.

    ``sin`` is odd, so the two transverse wavevectors are exact negatives, grazing included.
    Before, ``-float32(pi/2)`` was not held as ``+float32(pi/2)`` was: R differed by 2.5e-7.
    """
    for t in (.29, 1.2, GRAZING):
        minus = rcwa.run(problem(-t), numerics=rcwa.Numerics())
        plus = rcwa.run(problem(t), numerics=rcwa.Numerics())
        np.testing.assert_allclose(minus.R, plus.R, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(minus.T, plus.T, rtol=1e-12, atol=1e-15)


def test_a_traced_theta_past_either_bound_is_held_at_grazing_not_folded():
    jax = pytest.importorskip('jax')
    from wavebit._common.backend import _enable_x64
    _enable_x64()
    jnp = jax.numpy
    prepared = rcwa.prepare(problem(.3), numerics=rcwa.Numerics(), backend='jax')
    reflect = jax.jit(lambda theta: prepared.run(problem(theta)).R[0, 0, 0])
    grazing = float(reflect(jnp.asarray(GRAZING)))
    assert grazing > .99
    for theta in (2., -2., -GRAZING):
        assert float(reflect(jnp.asarray(theta))) == grazing
    # The fold the negative side took before: -2 was lit as -(pi - 2).
    assert abs(float(reflect(jnp.asarray(-(np.pi - 2)))) - grazing) > .5
