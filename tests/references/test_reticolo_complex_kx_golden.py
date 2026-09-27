"""A complex transverse wavevector against RETICOLO, where the two solvers' roots coincide.

A direct ``kx`` / ``ky`` must be real; a complex ``theta`` / ``phi`` is the one door to a complex
one, through the formal ``kx/k0 = n*sin(theta)*cos(phi)``. Nothing else in the suite checks that
path against an outside solver: ``test_reticolo_conical_sweep`` reaches complex *angles*, but
every one of them is evanescent incidence (``Re(theta) = pi/2``), whose ``kx`` is real.

What an oracle can say here is limited, and the limit is why the cases look the way they do.
With a complex ``kx`` no root of ``kz`` both carries power away and decays away, so each solver
picks one by its own rule, and the rules differ:

* this solver: the principal root, ``Re(kz) > 0``, moved to the other sheet only where the path
  from normal incidence crosses the cut (:func:`~wavebit._core.conventions._off_principal_branch`);
* RETICOLO: Maystre's cut, ``retsqrt(x, -1)`` -- the root with ``Re(kz) + Im(kz) > 0``;
* fmmax: ``Im(kz) > 0``, always.

Where every order's ``kz**2`` lies in the upper half plane, all three coincide. Outside it they
need not: on a lossless-top grating at ``theta = 0.2+0.1j`` this solver keeps growing evanescent
orders (``Im(kz) < 0``) that RETICOLO's cut replaces with decaying ones, and R moves by 0.11.
No oracle can arbitrate that -- it is a convention, not an error -- so the cases below stay where
this solver and RETICOLO agree, and :func:`test_cases_stay_where_the_roots_agree` fails if an
edit walks one out. The interface cases sit where fmmax does *not* agree, which pins that this
solver is on RETICOLO's side of that split rather than fmmax's.

The golden stores per-order reflected and transmitted efficiencies and the zeroth-order
reflection amplitude. Higher-order amplitudes are left out on purpose: they carry a phase
``exp(-i*m*G*x0)`` set by where each solver puts the grating's origin, which the efficiencies
and the zeroth order do not. RETICOLO's default mode selection reports nothing at a complex
``kx``; ``interface/reticolo_res2_all_modes.m`` says how the generator gets past that.

Measured when the golden was made (2026-09-26): efficiencies agree to 1e-16 .. 4e-15, the
zeroth-order amplitudes to 5e-16 .. 1.8e-15.
"""
import pathlib

import numpy as np
import pytest

import wavebit
from wavebit._core.conventions import boundary_kz

GOLDEN_PATH = pathlib.Path(__file__).parent / 'golden_reticolo_complex_kx.npz'

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
}


def _solver(name):
    return wavebit.rcwa(backend=0, fourier_coefficients='integrated', **CASES[name])


def _golden():
    if not GOLDEN_PATH.exists():
        pytest.skip(f'{GOLDEN_PATH.name} is missing -- regenerate with '
                    'gen_goldens_reticolo_complex_kx.py')
    return np.load(GOLDEN_PATH)


@pytest.mark.parametrize('name', list(CASES))
def test_cases_stay_where_the_roots_agree(name):
    """Oracle-free: every case is a complex kx, and RETICOLO's cut takes this solver's roots.

    The first assertion keeps the file honest about what it tests (a real ``kx`` would pass
    everything below and prove nothing about this path). The second is the scope of the oracle:
    were an order's root to have ``Re(kz) + Im(kz) <= 0``, RETICOLO would take the other one and
    a disagreement would be a convention, not a finding.
    """
    kw = CASES[name]
    kx = np.asarray(_solver(name).order_kxy()[0]).ravel()
    centre = kw['truncation_order']
    assert abs(kx[centre].imag) > 1e-3, f'{name}: kx = {kx[centre]} is not complex'
    for side in ('top', 'bot'):
        kz = np.asarray(boundary_kz(np, complex(kw[side]), kx ** 2)).ravel()
        assert np.all(kz.real + kz.imag > 0), (name, side, kz)


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
    """
    g = _golden()
    kw = CASES[name]
    n = kw['truncation_order']
    with pytest.warns(wavebit.ComplexIncidenceWarning):
        result = _solver(name).solve()
    for kind, values in (('r', result.R), ('t', result.T)):
        order = g[f'{name}__order_{kind}']
        assert sorted(order.tolist()) == list(range(-n, n + 1)), order
        np.testing.assert_allclose(np.asarray(values).ravel()[order + n], g[f'{name}__eff_{kind}'],
                                   atol=1e-11, rtol=0, err_msg=f'{name}: {kind}')
    amplitude = result.incident.r_s if kw['tetm'] == 'te' else result.incident.r_p
    # The specular order only, which is why the pure 1D path moving into the per-order s/p
    # frame (2026-09-27) left this untouched: RETICOLO's 1D amplitudes are Cartesian like the
    # old path's, and the two frames differ by sign(Re kx_m) * sign(Re kx_inc) -- 1 here. A
    # per-order comparison against RETICOLO's 1D path would apply that at the comparison.
    np.testing.assert_allclose(np.asarray(amplitude).ravel()[n], g[f'{name}__r0'],
                               atol=1e-11, rtol=0, err_msg=f'{name}: zeroth-order amplitude')
    assert np.isnan(float(np.real(result.power_entering)))
