"""A crossed grating against RETICOLO, in both of Li's nestings.

``golden_reticolo.npz`` is all 1D: its gratings have one axis, so they cannot tell the two
nestings of :class:`~wavebit.rcwa._engine.enums.Nesting` apart -- on a y-invariant layer both reduce to
Li's 1D inverse rule. This is the 2D case that can: an axis-aligned rectangle, where Li's rules
are exact for the staircase they are given, so the two codes build the same convolution matrices
and should agree to round-off in *each* nesting while the nestings disagree with each other.

**The numbers.** Measured with RETICOLO V10 under Octave, in a session before this file, with
RETICOLO's own switch for the nesting -- ``parm.res1.li = 1`` for the fixed one and ``li = 0``
for Li's symmetric one -- and its textures given as the exact rectangle, which is what
``fourier_coefficients='integrated'`` builds here. Only the totals were kept, to ten decimals,
so the comparison is at 1e-9 rather than the 1e-11 of the 1D goldens; measured agreement is
4.2e-11 at worst, which is the golden's own rounding. The generator of that session is not in
this repository: the bridge in ``interface/`` does not expose ``li``. What would regenerate them
is ``Reticolo.run_res2`` with that one field set, on the scene below.

**The scene.** ``eps = 6`` rectangle, 200 nm along x by 150 nm along y, centred in a 325 nm
square cell, 600 nm thick, air above, a substrate of ``eps = 1.46**2``, 532 nm at normal
incidence. TE is ``E`` along y (``psi = pi/2``), TM is ``E`` along x (``psi = 0``). The raster is
650 x 650, so every edge sits on a pixel boundary and the raster *is* the rectangle -- which is
also why the sampled path lands on the same numbers.

The nestings differ here by 1.7e-5 to 1.7e-3 in total transmission, so a build that reverted
from one to the other fails by at least 1e4 times the tolerance.
"""

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa
import numpy as np
import pytest


#: (nesting, truncation_order) -> (total T for TE, total T for TM), RETICOLO V10.
GOLDEN = {
    ('fixed', 3): (0.9822061471, 0.8913451356),
    ('fixed', 5): (0.9816211625, 0.8830613359),
    ('symmetric', 3): (0.9821713196, 0.8930579731),
    ('symmetric', 5): (0.9816046632, 0.8841301044),
}

TOL = 1e-9

_N, _PERIOD = 650, 325.0


def _rectangle(along_x=200.0, along_y=150.0):
    """The ``eps = 6`` rectangle, centred, on a 0.5 nm pixel grid; rows are y."""
    pixel = _PERIOD / _N
    x0, x1 = (round((_PERIOD - along_x) / 2 / pixel), round((_PERIOD + along_x) / 2 / pixel))
    y0, y1 = (round((_PERIOD - along_y) / 2 / pixel), round((_PERIOD + along_y) / 2 / pixel))
    eps = np.ones((1, _N, _N))
    eps[0, y0:y1, x0:x1] = 6.0
    return eps


def _total_T(structure, nesting, truncation_order, psi, coefficients='integrated'):
    res = _legacy_wavebit_rcwa(backend='numpy', structure=structure, period=[_PERIOD, _PERIOD],
                       thickness=[600.0], wavelength=532.0, top=1.0, bot=1.46 ** 2,
                       theta=0.0, phi=0.0, psi=psi, truncation_order=[truncation_order] * 2,
                       fourier_coefficients=coefficients, li_nesting=nesting).solve()
    return float(np.sum(np.asarray(res.T)))


_POLARIZATIONS = {'te': np.pi / 2, 'tm': 0.0}


@pytest.mark.parametrize('coefficients', ['integrated', 'sampled'])
@pytest.mark.parametrize('polarization', ['te', 'tm'])
@pytest.mark.parametrize('nesting, truncation_order', list(GOLDEN))
def test_total_transmission_matches_reticolo(nesting, truncation_order, polarization,
                                             coefficients):
    """Each nesting against RETICOLO's own run of it, both polarizations, both series."""
    want = GOLDEN[(nesting, truncation_order)][0 if polarization == 'te' else 1]
    got = _total_T(_rectangle(), nesting, truncation_order, _POLARIZATIONS[polarization],
                   coefficients)
    assert abs(got - want) < TOL, (nesting, truncation_order, polarization, got, want)


def test_the_goldens_can_tell_the_nestings_apart():
    """The premise: were the two nestings closer than the tolerance, the test above could pass
    on either of them and would pin nothing about which one is built."""
    for M in (3, 5):
        gap = np.abs(np.subtract(GOLDEN[('fixed', M)], GOLDEN[('symmetric', M)])).min()
        assert gap > 1e4 * TOL, (M, gap)


def test_turning_the_rectangle_exchanges_te_and_tm_only_in_the_symmetric_nesting():
    """The same rectangle turned a quarter: the covariance the symmetric nesting restores.

    At normal incidence on a square cell, the turned rectangle's TE problem *is* the original's
    TM problem, relabelled -- so its total has to be the golden TM number, and it is, to
    round-off, in the symmetric nesting. The fixed nesting answers the turned problem 1.4e-3
    away from its own golden at ``truncation_order=3``: that gap is the whole of the asymmetry
    ``docs/theory/rcwa/factorization-terms.md`` §6 is about.
    """
    turned = _rectangle(along_x=150.0, along_y=200.0)
    for M in (3, 5):
        te_turned = _total_T(turned, 'symmetric', M, _POLARIZATIONS['te'])
        assert abs(te_turned - GOLDEN[('symmetric', M)][1]) < TOL, (M, te_turned)
        te_turned = _total_T(turned, 'fixed', M, _POLARIZATIONS['te'])
        assert abs(te_turned - GOLDEN[('fixed', M)][1]) > 1e-4, (M, te_turned)
