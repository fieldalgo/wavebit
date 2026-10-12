"""Gate 4 -- Bloch dispersion: a plane wave under periodic and Bloch faces runs at the discrete frequency.

A 16^3 box carrying ``E = A exp(i k.r)`` with every component of ``k`` nonzero and the
transverse polarisation ``A = (K_y, -K_x, 0)/|..|`` built from the discrete wavenumbers, so
that ``K . A = 0`` exactly and the wave is a discrete eigenmode (``tests/fd_helpers.py``);
``H(-dt/2) = (dt/2) curl E0``. Under ``Periodic`` faces the wave is real, ``A cos(k.r)``
with ``k = 2 pi (l, m, n)/L``, and the run is float64. Under ``Bloch(phase=phi)`` on x and z
the wave is complex, ``k_x = (phi_x + 2 pi l)/L``, and the run is complex128. In both, the
recurrence ``f(n+1) + f(n-1) = 2 cos(w dt) f(n)`` with ``w`` from the discrete dispersion
holds at an E probe and an H probe to round-off (1e-11), on every backend, and the closed
form pins amplitude and phase.

Every axis carries a difference, on purpose. The first version of this gate ran ``E_y
exp(i(k_x x + k_z z))`` with ``k_y = 0``: nothing then drove a y-varying field through the
y-periodic pads, and a y pad reading the wrong layer -- a runtime bug invisible to the
plan's skeleton -- passed this gate and every other (2026-09-18). The plan-level form of
that mutant is pinned below: the y pads copying layer ``N-2`` and ``1`` instead of the edge
layers fail the recurrence by orders of magnitude on both boxes.

The Bloch sign convention is pinned by a mutant: ``k_x = (-phi_x + 2 pi l)/L`` -- the other
sign of the phase on both Bloch axes -- fails the recurrence by orders of magnitude, as does
ignoring the phase. ``field(x + L) = exp(+i phi) field(x)`` is what ``Bloch(phase=phi)``
means, in the ``e^{-i w t}`` convention.

The dtype rule and its refusal are pinned too: ``Periodic`` runs real, any ``Bloch`` face
makes the run complex, and ``Bloch(phase=0)`` is refused naming ``Periodic`` rather than
lowered silently -- a real problem should not pay for a complex run.
"""
import dataclasses
import math

import numpy as np
import pytest

import wavebit.fd as fd
from fd_helpers import (backends, discrete_omega, plane_wave_box, plane_wave_polarization,
                        recurrence_residual, run_plan, to_numpy)

TOL = 1e-11
MUTANT_FLOOR = 1e-4
PHASE = (0.7, -1.1)
L = 1.0


@pytest.fixture(scope='module')
def periodic():
    return plane_wave_box(None, N=16, steps=300)


@pytest.fixture(scope='module')
def bloch():
    return plane_wave_box(PHASE, N=16, steps=300)


def test_periodic_is_real_and_bloch_is_complex(periodic, bloch):
    assert fd.plan(periodic[0]).dtype == 'float64'
    assert fd.plan(bloch[0]).dtype == 'complex128'
    assert fd.plan(periodic[0]).faces['x-'].kind == 'periodic'
    assert fd.plan(bloch[0]).faces['x+'].kind == 'bloch'


@pytest.mark.parametrize('backend', backends())
@pytest.mark.parametrize('probe', ['Ey', 'Hx'])
def test_periodic_recurrence(periodic, backend, probe):
    scene, k, omega = periodic
    res = fd.run(scene, backend=backend)
    values = to_numpy(res[probe].values)
    assert values.dtype == np.float64
    assert recurrence_residual(values, omega, res.plan.dt) < TOL


@pytest.mark.parametrize('backend', backends())
@pytest.mark.parametrize('probe', ['Ey', 'Hx'])
def test_bloch_recurrence(bloch, backend, probe):
    scene, k, omega = bloch
    res = fd.run(scene, backend=backend)
    values = to_numpy(res[probe].values)
    assert values.dtype == np.complex128
    assert recurrence_residual(values, omega, res.plan.dt) < TOL
    # and the wave is genuinely complex at the probe: not a real run in disguise
    assert np.max(np.abs(values.imag)) > 1e-3


@pytest.mark.parametrize('backend', backends())
def test_bloch_closed_form(bloch, backend):
    """``Ey(n) = A_y exp(i k.r) cos(w n dt)`` at the probe: the Bloch pad carries the right phase."""
    scene, k, omega = bloch
    res = fd.run(scene, backend=backend)
    o = next(o for o in res.plan.observers if o.name == 'Ey')
    x, y, z = res.plan.grid.yee_points('E', o.c)
    at = (x[o.i], y[o.j], z[o.k])
    A = plane_wave_polarization(k, res.plan.dx)
    expected = A[1] * np.exp(1j * sum(ki * ai for ki, ai in zip(k, at))) * np.cos(omega * res['Ey'].t)
    assert np.max(np.abs(to_numpy(res['Ey'].values) - expected)) < 1e-10


def test_mutant_the_other_phase_sign_fails(bloch):
    """The recurrence at the frequency of the other sign convention, and of no phase, fails."""
    scene, k, omega = bloch
    res = fd.run(scene)
    dx, dt = res.plan.dx, res.plan.dt
    values = to_numpy(res['Ey'].values)
    assert recurrence_residual(values, omega, dt) < TOL
    flipped = discrete_omega(((-PHASE[0] + 2 * math.pi) / L, 2 * math.pi / L,
                              (-PHASE[1] + 2 * math.pi) / L), dx, dt)
    ignored = discrete_omega((2 * math.pi / L,) * 3, dx, dt)
    assert recurrence_residual(values, flipped, dt) > MUTANT_FLOOR
    assert recurrence_residual(values, ignored, dt) > MUTANT_FLOOR


def _y_pads_read_the_wrong_layer(plan):
    """The mutant: the periodic pads on y copy layer ``N-2`` (low face) and ``1`` (high) instead of the edges."""
    n = plan.ny
    pads = tuple((nm, c, dataclasses.replace(p, layer=n - 2 if nm == 'y-' else 1)
                  if nm[0] == 'y' else p) for nm, c, p in plan.pads)
    return plan.with_static(pads=pads)


@pytest.mark.parametrize('which', ['periodic', 'bloch'])
def test_mutant_a_y_pad_reading_the_wrong_layer_fails_the_recurrence(which, periodic, bloch):
    """A y-only pad defect: invisible while the wave had ``k_y = 0``, O(1e-2) now."""
    scene, k, omega = periodic if which == 'periodic' else bloch
    plan = fd.plan(scene)
    assert plan.pad('y-', 0).kind == plan.pad('y+', 0).kind == 'periodic'
    res = run_plan(scene, _y_pads_read_the_wrong_layer(plan))
    for probe in ('Ey', 'Hx'):
        assert recurrence_residual(to_numpy(res[probe].values), omega, plan.dt) > MUTANT_FLOOR


def test_bloch_k_is_phase_over_the_period(bloch):
    """``Bloch(k=k)`` and ``Bloch(phase=k*L)`` are the same run, bit for bit."""
    scene, k, omega = bloch
    by_k = scene.replace(boundaries=scene.boundaries.replace(
        x=fd.Bloch(k=PHASE[0] / 1.0), z=fd.Bloch(k=PHASE[1] / 1.0)))
    a, b = fd.run(scene), fd.run(by_k)
    assert np.array_equal(to_numpy(a.final.E), to_numpy(b.final.E))


def test_bloch_with_a_concrete_zero_phase_is_refused_naming_periodic():
    with pytest.raises(ValueError, match='Periodic'):
        fd.Bloch(phase=0.0)
    with pytest.raises(ValueError, match='Periodic'):
        fd.Bloch(k=0)
    with pytest.raises(ValueError, match='exactly one'):
        fd.Bloch(phase=0.3, k=0.3)
    with pytest.raises(ValueError, match='exactly one'):
        fd.Bloch()


def test_a_bloch_face_needs_its_pair():
    with pytest.raises(ValueError, match='both faces'):
        fd.Boundaries(x=(fd.Bloch(phase=0.3), fd.PEC()), y=fd.PEC(), z=fd.PEC())
    with pytest.raises(ValueError, match='both faces'):
        fd.Boundaries(x=(fd.Bloch(phase=0.3), fd.Bloch(phase=0.4)), y=fd.PEC(), z=fd.PEC())
    ok = fd.Boundaries(x=(fd.Bloch(phase=0.3), fd.Bloch(phase=0.3)), y=fd.PEC(), z=fd.Periodic())
    assert ok.face('x-') == fd.Bloch(phase=0.3)


def test_a_complex_initial_field_on_a_real_run_is_refused(periodic):
    scene = periodic[0]
    bad = scene.replace(initial=fd.InitialField(E=(None, lambda x, y, z: np.exp(1j * x), None)))
    with pytest.raises(ValueError, match='complex'):
        fd.run(bad)
