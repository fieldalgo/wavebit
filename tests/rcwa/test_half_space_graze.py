"""Stacks with a closed-form answer: an empty interface, and layers of a half-space's medium.

* At an exact Rayleigh point of the incident (or exit) medium, a layer of that medium holds the
  ``q = 0`` cluster no exponential basis represents, and the 2D solve refused it with
  ``rcwa.singular`` -- an empty stack too, which reaches the engine as a zero-width layer of the
  top medium. The half-space beside it is fine, so the run merges into it there: the answer is
  the single interface's Fresnel coefficients, continuous with the stack one step off the point.
  A coherent multi-source solve cannot merge and refuses that point as the graze it is.
* Ordinary stacks keep their bits: the merge does not run off its band.
"""
import warnings

import numpy as np
import pytest

from wavebit import Material, rcwa
from wavebit.errors import NumericalError



def _problem(layers, *, top, bottom, period, wavelength=1.0, theta=0.0, phi=0.0, side='top',
             polarization='s'):
    return rcwa.Problem(
        stack=rcwa.Stack(period=period, top=Material(top), bottom=Material(bottom),
                         layers=layers),
        excitation=rcwa.Excitation(wavelength=wavelength, sources=(rcwa.PlaneWave(
            side=side, polarization=polarization, theta=theta, phi=phi),)))


def _fresnel(e1, e2, theta, polarization):
    """Power ``R`` and ``T`` of the bare ``e1 | e2`` interface lit from ``e1``, written here."""
    k2 = e1 * np.sin(theta) ** 2
    q1, q2 = np.sqrt(e1 - k2 + 0j), np.sqrt(e2 - k2 + 0j)
    if polarization == 's':
        r, t, ratio = (q1 - q2) / (q1 + q2), 2 * q1 / (q1 + q2), q2 / q1
    else:
        r = (e2 * q1 - e1 * q2) / (e2 * q1 + e1 * q2)
        t, ratio = 2 * np.sqrt(e1 * e2) * q1 / (e2 * q1 + e1 * q2), q2 / q1
    return abs(r) ** 2, (ratio * abs(t) ** 2).real


def _layers(kind, top, bottom):
    return {'empty': (),
            'lit medium': (rcwa.Layer(thickness=0.3, background=Material(top)),),
            'both media': (rcwa.Layer(thickness=0.2, background=Material(bottom)),
                           rcwa.Layer(thickness=0.3, background=Material(top)))}[kind]


def _solve(problem, **numerics):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return rcwa.run(problem, numerics=rcwa.Numerics(orders=(2, 2), **numerics))


@pytest.mark.parametrize('kind', ['empty', 'lit medium', 'both media'])
@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('polarization', ['s', 'p'])
@pytest.mark.parametrize('algorithm,formulation', [('smm', 'auto'), ('etm', 'auto'),
                                                   ('smm', 'first_order')])
def test_a_graze_of_the_incident_medium_is_the_bare_interface(kind, side, polarization,
                                                              algorithm, formulation):
    """``wavelength == period`` at normal incidence puts orders (+-1, 0) and (0, +-1) on the
    light line of the ``eps = 1`` medium: the layers of that medium used to refuse there."""
    top, bottom = (1.0, 2.25) if side == 'top' else (2.25, 1.0)
    lit, far = (top, bottom) if side == 'top' else (bottom, top)

    def at(wavelength):
        return _solve(_problem(_layers(kind, top, bottom), top=top, bottom=bottom,
                               period=(1.0, 1.0), wavelength=wavelength, side=side,
                               polarization=polarization),
                      algorithm=algorithm, formulation=formulation)

    exact, beside = at(1.0), at(1.0 * (1 + 1e-12))
    r, t = _fresnel(lit, far, 0.0, polarization)
    for result in (exact, beside):
        R, T = np.asarray(result.R)[0], np.asarray(result.T)[0]
        np.testing.assert_allclose(R[2, 2], r, rtol=1e-12)
        np.testing.assert_allclose(T[2, 2], t, rtol=1e-12)
        np.testing.assert_allclose(np.sum(R) + np.sum(T), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.sum(exact.R), np.sum(beside.R), rtol=1e-10)


def test_an_oblique_graze_of_a_diffracted_order_is_the_bare_interface():
    """The (-1, 0) order of an oblique wave, placed on the cover's light line."""
    theta = 0.4
    for polarization in ('s', 'p'):
        problem = _problem((), top=1.0, bottom=2.25, period=(1.0 / (1 + np.sin(theta)), 1.0),
                           theta=theta, polarization=polarization)
        result = _solve(problem)
        r, t = _fresnel(1.0, 2.25, theta, polarization)
        np.testing.assert_allclose(np.sum(result.R), r, rtol=1e-11)
        np.testing.assert_allclose(np.sum(result.T), t, rtol=1e-11)


def test_the_graze_merge_is_reported_and_stays_off_its_band():
    """At the point the layer merges; a 1e-6 step off, it is solved as a layer, as before."""
    layers = _layers('lit medium', 1.0, 2.25)
    exact = _solve(_problem(layers, top=1.0, bottom=2.25, period=(1.0, 1.0)))
    off = _solve(_problem(layers, top=1.0, bottom=2.25, period=(1.0, 1.0),
                          wavelength=1.0 + 1e-6))
    assert exact.diagnostics['half_space_merge'][0]['top'] == 1
    assert off.diagnostics['half_space_merge'] == (None,)


def test_a_coherent_excitation_refuses_the_graze_as_a_graze():
    """The coherent cascade cannot merge a layer, so the exact point refuses -- as the Rayleigh
    point it is, not with advice to raise the Fourier order -- and its neighbour solves."""
    def problem(wavelength):
        sources = (rcwa.PlaneWave(side='top', polarization='s'),
                   rcwa.PlaneWave(side='bottom', polarization='p'))
        return rcwa.Problem(
            stack=rcwa.Stack(period=(1.0, 1.0), top=Material(1.0), bottom=Material(2.25)),
            excitation=rcwa.Excitation(wavelength=wavelength, sources=sources),
            outputs=rcwa.Outputs(diffraction=False, power_region='cell'))

    with pytest.raises(NumericalError, match='light line') as caught:
        _solve(problem(1.0))
    assert caught.value.code == 'rcwa.singular'
    assert 'Fourier order and NVF' not in str(caught.value)
    _solve(problem(1.0 * (1 + 1e-12)))
