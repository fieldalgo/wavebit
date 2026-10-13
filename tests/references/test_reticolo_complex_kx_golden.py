"""A complex transverse wavevector against RETICOLO, on both sides of the old split.

A direct ``kx`` / ``ky`` must be real; a complex ``theta`` / ``phi`` is the one door to a complex
one, through the formal ``kx/k0 = n*sin(theta)*cos(phi)``. Nothing else in the suite checks that
path against an outside solver: ``test_reticolo_conical_sweep`` reaches complex *angles*, but
every one of them is evanescent incidence (``Re(theta) = pi/2``), whose ``kx`` is real.

With a complex ``kx`` no root of ``kz`` both carries power away and decays away, so each solver
picks one by its own rule:

* this solver and RETICOLO: Maystre's cut, the root with ``Re(kz) + Im(kz) > 0`` (RETICOLO's
  ``retsqrt(x, -1)``; this solver's :func:`~wavebit._common.conventions.boundary_kz` since
  2026-10-02);
* fmmax: ``Im(kz) > 0``, always.

Until 2026-10-02 this solver took the principal root, continued from normal incidence, which
agrees with Maystre's cut only where every order's ``kz**2`` keeps ``Re(kz) > 0``. The first five
cases were chosen inside that agreement, so they held across the change. The lossless-top grating
cases sit outside it. Their evanescent orders fall below the real axis, and the old rule let them
grow where Maystre's cut keeps them decaying. ``R`` moved by 0.11 there, and
:func:`test_the_lossless_top_cases_reach_the_old_split` fails if an edit walks them back inside.
The interface cases sit where fmmax does *not* agree, which pins that this solver is on
RETICOLO's side of that split rather than fmmax's.

The golden stores per-order reflected and transmitted efficiencies and the zeroth-order
reflection amplitude. Higher-order amplitudes are left out on purpose: they carry a phase
``exp(-i*m*G*x0)`` set by where each solver puts the grating's origin, which the efficiencies
and the zeroth order do not. RETICOLO's default mode selection reports nothing at a complex
``kx``; ``interface/reticolo_res2_all_modes.m`` says how the generator gets past that.

Measured when the golden was made (2026-09-26): efficiencies agree to 1e-16 .. 4e-15, the
zeroth-order amplitudes to 5e-16 .. 1.8e-15. The lossless-top cases were added on 2026-10-02.
The regenerated golden kept the first five cases bit for bit. The new cases agree to
1.2e-15 .. 1.9e-15 in efficiency and 1.4e-15 .. 1.7e-15 in the zeroth-order amplitude. Under
the old rule they missed by 0.003 .. 0.06 in efficiency.
"""

from wavebit.compat import legacy as _legacy_wavebit

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))
from tools.repository_paths import repository_path

import numpy as np
import pytest

from wavebit._common.conventions import boundary_kz

GOLDEN_PATH = repository_path('tests/references/golden_reticolo_complex_kx.npz')

_UNIFORM = np.full((1, 1, 2), 2.25)          # the layer is bot's material: one interface
_HALF = np.array([[[9.0] * 5 + [1.0] * 5]])  # test_reticolo_golden's binary grating
# The grating's kx is chosen outright -- an imaginary part small enough that every retained
# order's kz**2 stays in the upper half plane of both absorbing half-spaces -- and then spelled as
# the angle that gives it, since the angle is the only way in.
_THETA_GRATING = complex(np.arcsin((0.3 + 0.01j) / np.sqrt(2.25 + 1j)))

# name -> solver kwargs. All pure TE/TM, RETICOLO's 1D branch.
CASES = {
    # Chapter 13's interface at a formal complex angle. bot's kz**2 falls below the real axis:
    # this solver and RETICOLO take the root with Re(kz) > 0, fmmax the other one.
    'te_interface_absorbing_top': dict(tetm='te', theta=np.pi / 6 + 0.2j, top=4 + 3j, bot=2.25,
                                       truncation_order=0, period=[1200.], wavelength=1000.,
                                       structure=_UNIFORM, thickness=[100.]),
    'tm_interface_absorbing_top': dict(tetm='tm', theta=np.pi / 6 + 0.2j, top=4 + 3j, bot=2.25,
                                       truncation_order=0, period=[1200.], wavelength=1000.,
                                       structure=_UNIFORM, thickness=[100.]),
    # Lossless top: both half-spaces' kz**2 fall below the axis, so fmmax takes the other root
    # in *both*, and r does not notice. The control for the two above.
    'te_interface_lossless_top': dict(tetm='te', theta=np.pi / 6 + 0.2j, top=4.0, bot=2.25,
                                      truncation_order=0, period=[1200.], wavelength=1000.,
                                      structure=_UNIFORM, thickness=[100.]),
    # A grating: eleven orders, all complex, coupled through the patterned layer.
    'te_grating_absorbing_halfspaces': dict(tetm='te', theta=_THETA_GRATING, top=2.25 + 1j,
                                            bot=2.25 + 0.5j, truncation_order=5, period=[0.7],
                                            wavelength=0.9, structure=_HALF, thickness=[0.15]),
    'tm_grating_absorbing_halfspaces': dict(tetm='tm', theta=_THETA_GRATING, top=2.25 + 1j,
                                            bot=2.25 + 0.5j, truncation_order=5, period=[0.7],
                                            wavelength=0.9, structure=_HALF, thickness=[0.15]),
    # The same grating between lossless half-spaces, at angles the old rule answered differently.
    # Air's evanescent orders fall below the real axis: the positive ones at +Im(theta), the
    # negative ones at -Im(theta).
    'te_grating_lossless_top': dict(tetm='te', theta=0.2 + 0.1j, top=1.0, bot=2.25,
                                    truncation_order=5, period=[0.7], wavelength=0.9,
                                    structure=_HALF, thickness=[0.15]),
    'tm_grating_lossless_top': dict(tetm='tm', theta=0.2 + 0.1j, top=1.0, bot=2.25,
                                    truncation_order=5, period=[0.7], wavelength=0.9,
                                    structure=_HALF, thickness=[0.15]),
    'te_grating_lossless_top_below_axis': dict(tetm='te', theta=0.2 - 0.1j, top=1.0, bot=2.25,
                                               truncation_order=5, period=[0.7], wavelength=0.9,
                                               structure=_HALF, thickness=[0.15]),
}

#: The cases added on 2026-10-02, outside what the old rule agreed with.
_LOSSLESS_TOP = ('te_grating_lossless_top', 'tm_grating_lossless_top',
                 'te_grating_lossless_top_below_axis')


def _solver(name):
    return _legacy_wavebit.rcwa(backend='numpy', fourier_coefficients='integrated', **CASES[name])


def _golden():
    if not GOLDEN_PATH.exists():
        pytest.skip(f'{GOLDEN_PATH.name} is missing -- regenerate with '
                    'gen_goldens_reticolo_complex_kx.py')
    return np.load(GOLDEN_PATH)


@pytest.mark.parametrize('name', list(CASES))
def test_every_case_is_a_complex_kx(name):
    """Oracle-free: a real ``kx`` would pass everything below and prove nothing about this path."""
    kx = np.asarray(_solver(name).order_kxy()[0]).ravel()
    centre = CASES[name]['truncation_order']
    assert abs(kx[centre].imag) > 1e-3, f'{name}: kx = {kx[centre]} is not complex'


@pytest.mark.parametrize('name', _LOSSLESS_TOP)
def test_the_lossless_top_cases_reach_the_old_split(name):
    """Some order there takes a decaying root with ``Re(kz) < 0``, the one the old rule refused.

    The principal root keeps ``Re(kz) >= 0``, and so did this solver until 2026-10-02 wherever
    its continuation fix stayed silent, as it does over a lossless half-space. An order with
    ``Re(kz) < 0`` is therefore one where the two rules differ. Without one, these cases could
    not tell the rules apart, and the golden would pin nothing about the choice.
    """
    kw = CASES[name]
    kx = np.asarray(_solver(name).order_kxy()[0]).ravel()
    kz = np.asarray(boundary_kz(np, complex(kw['top']), kx ** 2)).ravel()
    assert np.any(kz.real < -1e-3), kz


@pytest.mark.parametrize('name', ['te_interface_absorbing_top', 'tm_interface_absorbing_top'])
def test_interface_cases_are_where_fmmax_splits_off(name):
    """The interface cases take a root with ``Im(kz) < 0`` in bot -- the one fmmax refuses.

    Without this the file could drift into cases where all three solvers agree, and stop saying
    anything about which side of the split this solver is on.
    """
    kw = CASES[name]
    kx = np.asarray(_solver(name).order_kxy()[0]).ravel()
    kz = np.asarray(boundary_kz(np, complex(kw['bot']), kx ** 2)).ravel()
    assert np.all(kz.imag < 0) and np.all(kz.real > 0), kz


@pytest.mark.parametrize('name', list(CASES))
def test_orders_match_reticolo(name):
    """Per-order efficiencies and the zeroth-order amplitude, against RETICOLO's own numbers.

    ``atol=1e-11`` is the round-off room ``test_reticolo_golden`` uses; the measured disagreement
    is 1e-16 .. 4e-15. The solve must also say that its R and T are formal: the warning is part
    of the contract at a complex ``kx``, and so is the NaN in ``power_entering``.

    One convention differs, and only in sign. An order whose root decays with ``Re(kz) < 0``
    (the lossless-top cases' evanescent orders) carries its z-flux back toward the stack. This
    solver's formal ratio keeps that sign and comes out negative. RETICOLO reports the same
    magnitude, positive. The amplitudes agree, so the comparison flips those orders rather than
    taking magnitudes everywhere, which would also hide a sign error on every other order.
    """
    g = _golden()
    kw = CASES[name]
    n = kw['truncation_order']
    with pytest.warns(_legacy_wavebit.ComplexIncidenceWarning):
        result = _solver(name).solve()
    kx = np.asarray(_solver(name).order_kxy()[0]).ravel()
    for kind, values, side in (('r', result.R, 'top'), ('t', result.T, 'bot')):
        order = g[f'{name}__order_{kind}']
        assert sorted(order.tolist()) == list(range(-n, n + 1)), order
        kz = np.asarray(boundary_kz(np, complex(kw[side]), kx ** 2)).ravel()
        flux_sign = np.where(kz.real < 0, -1.0, 1.0)[order + n]
        np.testing.assert_allclose(np.asarray(values).ravel()[order + n] * flux_sign,
                                   g[f'{name}__eff_{kind}'], atol=1e-11, rtol=0,
                                   err_msg=f'{name}: {kind}')
    amplitude = result.incident.r_s if kw['tetm'] == 'te' else result.incident.r_p
    # The specular order only, which is why the pure 1D path moving into the per-order s/p
    # frame (2026-09-27) left this untouched: RETICOLO's 1D amplitudes are Cartesian like the
    # old path's, and the two frames differ by sign(Re kx_m) * sign(Re kx_inc) -- 1 here. A
    # per-order comparison against RETICOLO's 1D path would apply that at the comparison.
    np.testing.assert_allclose(np.asarray(amplitude).ravel()[n], g[f'{name}__r0'],
                               atol=1e-11, rtol=0, err_msg=f'{name}: zeroth-order amplitude')
    assert np.isnan(float(np.real(result.power_entering)))
