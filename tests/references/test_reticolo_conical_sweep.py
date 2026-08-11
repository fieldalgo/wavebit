"""Conical incidence swept over ``(kx, ky)``, against RETICOLO — including complex angles.

``test_reticolo_golden`` covers conical incidence at two fixed angles. This one sweeps the
in-plane wavevector across a grid and, in doing so, reaches a regime nothing else in the
suite touches.

The sweep's angle is ``theta = arcsin(sqrt(kx**2 + ky**2 + 1e-4))``, so wherever
``kx**2 + ky**2 > 1`` the incidence angle is **complex**: the illumination is evanescent.
That is 372 of the 441 grid points, and nothing else in the suite sets a complex ``theta``
at all.

It is worth saying what that blind spot cost, because the shape of the bug is the argument
for keeping this file. The boundary match wrote the incident wave's ``kz`` two ways — once
as ``n_top * cos(theta)`` and once as :func:`~fieldwave._core.conventions.boundary_kz`'s
root — and those are *the same number* for a real angle. For a complex one they are
negatives, the two sides of the match cancelled at the incident order, and the solve
returned ``R = -delta`` with ``T = 0``. Agreement over the propagating 69 points stayed at
1e-14 throughout. No amount of tightening the tolerance on a real angle would have found it.

The golden is the 2024-09-09 RETICOLO run; see ``gen_goldens_reticolo_conical_sweep.py``
for where it came from. RETICOLO's amplitude convention is the conjugate of this solver's
(:mod:`fieldwave._core.conventions`), so every comparison here is on the real part — which
is what the original study compared too.
"""
import pathlib

import numpy as np
import pytest

import fieldwave

GOLDEN_PATH = pathlib.Path(__file__).parent / 'golden_reticolo_conical_sweep.npz'

FTO = 80
# Solving all 441 points takes about a minute. These strides keep the physics — a spread of
# polar and azimuthal angles, on and off the axes — at a couple of seconds.
PROPAGATING_STRIDE = 7
EVANESCENT_STRIDE = 41


def _golden():
    if not GOLDEN_PATH.exists():
        pytest.skip(f'{GOLDEN_PATH.name} is missing — regenerate with '
                    'gen_goldens_reticolo_conical_sweep.py')
    return np.load(GOLDEN_PATH)


def _grid(g):
    """``(propagating, evanescent)`` index pairs, subsampled."""
    kx, ky = g['kx'], g['ky']
    prop, evan = [], []
    for i in range(len(kx)):
        for j in range(len(ky)):
            # The +1e-4 the sweep adds inside the sqrt is what decides the boundary points,
            # so it belongs in the mask too.
            (prop if kx[i] ** 2 + ky[j] ** 2 + 0.0001 <= 1.0 else evan).append((i, j))
    return prop[::PROPAGATING_STRIDE], evan[::EVANESCENT_STRIDE]


def _solve(g, i, j):
    """The (0,0)-order reflected amplitude for TE incidence at grid point ``(i, j)``."""
    kx, ky = g['kx'], g['ky']
    theta = np.arcsin(np.sqrt(kx[i] ** 2 + ky[j] ** 2 + 0.0001, dtype=np.complex128))
    phi = np.arctan(ky[j] / (kx[i] + 0.0001), dtype=np.complex128)
    res = fieldwave.rcwa(backend=0, pol=0, n_top=float(g['n_top']), n_bot=float(g['n_bot']),
                         theta=theta, phi=phi, fto=[FTO], wavelength=float(g['wavelength']),
                         period=list(g['period']), ucell=g['ucell'],
                         thickness=list(g['thickness']),
                         fourier_coefficients='integrated',
                         factorization='li').conv_solve()
    return res.res_te_inc


def test_propagating_incidence_matches_reticolo():
    """Real incidence angles, swept over the light cone.

    ``2e-13`` is round-off room: the measured spread over the full 441-point grid is 2e-14 at
    fto=80. This is the assertion that says the conical path is right; everything else in
    this file is about the regime where it is not.
    """
    g = _golden()
    want = g[f'amp_fto{FTO}'].real
    prop, _ = _grid(g)
    assert len(prop) >= 8, len(prop)

    worst = 0.0
    for i, j in prop:
        got = np.asarray(_solve(g, i, j).R_s).ravel()[FTO].real
        worst = max(worst, abs(got - want[i, j]))
    assert worst < 2e-13, worst


def test_propagating_incidence_conserves_energy():
    """A lossless stack, so the efficiencies must sum to one wherever power is defined.

    Independent of the golden: it would catch a solver that drifted from RETICOLO and from
    physics together.
    """
    g = _golden()
    prop, _ = _grid(g)

    worst = 0.0
    for i, j in prop:
        te = _solve(g, i, j)
        total = sum(float(np.nansum(np.asarray(a).real))
                    for a in (te.de_ri_s, te.de_ri_p, te.de_ti_s, te.de_ti_p))
        worst = max(worst, abs(1.0 - total))
    assert worst < 1e-12, worst


def test_evanescent_incidence_matches_reticolo():
    """Complex incidence angles, outside the light line — the regime this file exists for.

    Every point here has ``Re(theta) == pi/2`` exactly, because ``arcsin`` of a real greater
    than one always lands there. Until the incident order stopped being handed a ``kz`` from
    the outgoing-wave branch (:func:`~fieldwave._core.conventions.incident_kz`) the solve
    returned ``R_s = -1`` at the centre order and exactly zero at every other order, with
    ``T_s = 0`` — a grating that scatters nothing. It plots as a clean flat region, which is
    why the 2024 study that produced this golden did not catch it.

    Same ``2e-13`` as the propagating case, and measured at the same 2e-14: the evanescent
    half is not a softer regime once the branch is right.
    """
    g = _golden()
    want = g[f'amp_fto{FTO}'].real
    _, evan = _grid(g)
    assert len(evan) >= 4, len(evan)

    worst = 0.0
    for i, j in evan:
        got = np.asarray(_solve(g, i, j).R_s).ravel()[FTO].real
        worst = max(worst, abs(got - want[i, j]))
    assert worst < 2e-13, worst


def test_uniform_stack_does_not_diffract_at_a_complex_angle():
    """A layer with no pattern has no diffraction orders, whatever the incidence angle.

    Oracle-free, and it covers the half of the failure the sweep cannot reach: every sweep
    point sits at ``Re(theta) == pi/2``, where the old bug produced a *tidy* wrong answer.
    Away from that value it produced a loud one instead — off-centre amplitudes of ~7e3 here
    — so both symptoms need a case, and this is the cheap one.

    Note that ``|R| <= 1`` is **not** available as an assertion: under evanescent
    illumination the incident wave carries no power, and the analytic Airy result for this
    stack legitimately exceeds unity (1.01 at ``pi/2 + 0.3j``). Orders that must be identically
    zero are the honest invariant.
    """
    uniform = np.full_like(np.asarray(_golden()['ucell'], dtype=float), 2.5)
    g = _golden()
    res = fieldwave.rcwa(backend=0, pol=0, n_top=float(g['n_top']), n_bot=float(g['n_bot']),
                         theta=np.complex128(0.5 + 0.3j), phi=np.complex128(0.3),
                         fto=[FTO], wavelength=float(g['wavelength']),
                         period=list(g['period']), ucell=uniform,
                         thickness=list(g['thickness']),
                         fourier_coefficients='integrated',
                         factorization='li').conv_solve()
    R = np.asarray(res.res_te_inc.R_s).ravel()
    off_centre = np.abs(np.delete(R, FTO)).max()
    assert off_centre < 1e-12, off_centre
