"""A stack with no layers: ``field()`` refuses, and the fields it points to are Fresnel's.

The engine solves a bare interface through one zero-thickness layer of the top medium. Sampling
that layer returned its face value at every requested height -- a count of three gave three
identical rows, the documented count being per finite layer of which there are none -- so
``field()`` refuses and names ``boundary_fields``, which reads the two half-spaces' limits at
the interface. Those are checked here against the closed-form single-interface field.
"""
import numpy as np
import pytest

from wavebit import Material, Units, rcwa
from wavebit.errors import ValidationError
from wavebit.units import Z0

EPS_TOP, EPS_BOTTOM, THETA, PHI = 2.25, 3.24, .29, .4


def bare(side, polarization):
    return rcwa.run(rcwa.Problem(
        units=Units(length='um'),
        stack=rcwa.Stack(period=(.7, .8), top=Material(EPS_TOP), bottom=Material(EPS_BOTTOM), layers=()),
        excitation=rcwa.Excitation(wavelength=.8, sources=(rcwa.PlaneWave(
            side=side, polarization=polarization, theta=THETA, phi=PHI),)),
        outputs=rcwa.Outputs(fields=True)), numerics=rcwa.Numerics())


@pytest.mark.parametrize('z', [3, 1, [0.], [-1., 0., 1.], [3.]])
def test_field_on_a_bare_interface_refuses_and_names_boundary_fields(z):
    result = bare('top', 's')
    with pytest.raises(ValidationError) as caught:
        result.field(x=[0.], y=[0.], z=z)
    assert caught.value.code == 'field.coordinates'
    assert caught.value.path == 'z'
    assert 'no layers' in str(caught.value) and 'boundary_fields' in str(caught.value)


def test_a_stack_with_one_zero_thickness_layer_still_samples_it():
    """The refusal is about a declared empty stack, not about a layer's thickness."""
    result = rcwa.run(rcwa.Problem(
        units=Units(length='um'),
        stack=rcwa.Stack(period=(.7, .8), top=Material(EPS_TOP), bottom=Material(EPS_BOTTOM),
                         layers=(rcwa.Layer(thickness=0., background=Material(4.)),)),
        excitation=rcwa.Excitation(wavelength=.8, sources=(rcwa.PlaneWave(
            side='top', polarization='s', theta=THETA, phi=PHI),)),
        outputs=rcwa.Outputs(fields=True)), numerics=rcwa.Numerics())
    assert result.field(x=[0.], y=[0.], z=[0.]).values.shape == (1, 1, 1, 1, 6)


@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('polarization', ['s', 'p'])
def test_the_half_space_limits_at_a_bare_interface_are_fresnel(side, polarization):
    """At the cell centre the incident wave is unit E; Fresnel gives each side's total."""
    result = bare(side, polarization)
    lit, far = (EPS_TOP, EPS_BOTTOM) if side == 'top' else (EPS_BOTTOM, EPS_TOP)
    n1, n2 = np.sqrt(lit), np.sqrt(far)
    c1 = np.cos(THETA)
    c2 = np.sqrt(1 - (n1 * np.sin(THETA) / n2) ** 2)
    total = {s: result.boundary_fields(side=s, x=[0.], y=[0.], part='total').values[0, 0, 0]
             for s in ('top', 'bottom')}
    near, other = total[side], total['bottom' if side == 'top' else 'top']
    plane = np.array([np.cos(PHI), np.sin(PHI), 0.])     # in-plane transverse direction
    if polarization == 's':
        r = (n1 * c1 - n2 * c2) / (n1 * c1 + n2 * c2)
        # E is all tangential and normal to the plane of incidence; it is continuous.
        for f in (near, other):
            assert abs(f[2]) < 1e-12 and abs(f[:3] @ plane) < 1e-12
        np.testing.assert_allclose(np.linalg.norm(near[:3]), abs(1 + r), rtol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(other[:3]), abs(1 + r), rtol=1e-12)
        expected_r = abs(r) ** 2
    else:
        # H is all tangential for p; its Fresnel ratio is (eps2 kz1 - eps1 kz2)/(eps2 kz1 + eps1 kz2).
        r = (n2 * c1 - n1 * c2) / (n2 * c1 + n1 * c2)
        for f in (near, other):
            assert abs(f[5]) < 1e-12 / Z0 and abs(f[3:] @ plane) < 1e-12 / Z0
        np.testing.assert_allclose(np.linalg.norm(near[3:]), abs(1 + r) * n1 / Z0, rtol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(other[3:]), abs(1 + r) * n1 / Z0, rtol=1e-12)
        # Normal D is continuous, so Ez jumps by the permittivity ratio.
        np.testing.assert_allclose(lit * near[2], far * other[2], rtol=1e-12)
        expected_r = abs(r) ** 2
    np.testing.assert_allclose(result.R[0, 0, 0], expected_r, rtol=1e-12)
