"""Numerical stability where RCWA is known to lose it: evanescent orders through thick layers.

An order with ``|kx| > n·k0`` is evanescent, and its ``exp(|kz|·d)`` grows without bound in
the layer thickness. That growth is what makes a naive transfer-matrix formulation blow up,
and it is the open problem named in the roadmap for eig-free GPU modes — so it is worth
knowing precisely where the *current* solver stands.

It stands well, and these tests pin that down. The point is not that the physics is right —
other tests cover that — but that the same physics survives being asked for in the ways
that usually break it: through a stack thick enough for the evanescent tail to overflow,
with enough retained orders to make most of them evanescent, in single precision, and
through both connecting algorithms.

That matters beyond regression. The plan is a fast eig-free mode cross-checked against the
precise one, and a reference is only useful if it holds up exactly where the fast path is
expected to fail.
"""
import numpy as np
import pytest

import wavebit


def _grating(n_hi=4.0, cells=8):
    """Half air, half high index — the contrast is what populates the evanescent orders."""
    half = cells // 2
    return np.array([[[1.0] * half + [n_hi] * half]])


def _solve(*, fto, thickness, n_hi=4.0, algo='ETM', type_complex=np.complex128,
           period=3000.0, wavelength=400.0, n_bot=1.5):
    return wavebit.rcwa(
        backend=0, wavelength=wavelength, period=[period], fto=[fto, 0],
        n_top=1.0, n_bot=n_bot, theta=0.0, phi=0.0, psi=np.pi / 2,
        ucell=_grating(n_hi), thickness=[thickness],
        connecting_algo=algo, type_complex=type_complex,
    ).conv_solve()


def _energy(result):
    return float(result.R.sum() + result.T.sum())


def evanescent_orders(fto, period, wavelength, n=1.0):
    """How many retained orders are evanescent in a medium of index ``n``.

    ``kx_m = n·sinθ + m·λ/Λ`` in this solver's normalisation, so at normal incidence an
    order is evanescent once ``|m|·λ/Λ > n``. Reported so a test that claims to stress
    evanescent growth can show that it does.
    """
    orders = np.arange(-fto, fto + 1)
    return int(np.sum(np.abs(orders * wavelength / period) > n))


# ---------------------------------------------------------------- the regime is real

def test_the_test_conditions_actually_stress_evanescent_orders():
    """Guard the premise: without this the suite below could pass by never being hard."""
    assert evanescent_orders(fto=40, period=3000.0, wavelength=400.0) > 60, (
        'most retained orders should be evanescent, or these tests prove nothing')


# ---------------------------------------------------------------- energy

@pytest.mark.parametrize('thickness', [500.0, 5_000.0, 50_000.0])
@pytest.mark.parametrize('fto', [10, 40])
def test_energy_is_conserved_through_thick_layers(fto, thickness):
    """A lossless stack conserves energy no matter how far the evanescent tail is pushed.

    50 um at fto=40 puts ``exp(|kz|·d)`` far past what a float can hold, so a formulation
    that carried the growing exponential explicitly would have overflowed long before here.
    """
    assert _energy(_solve(fto=fto, thickness=thickness)) == pytest.approx(1.0, abs=1e-9)


def test_absorbing_metal_stays_physical_when_thick():
    """With loss, R + T < 1 — but it must stay in [0, 1] and not drift with thickness."""
    previous = 1.0
    for thickness in (200.0, 2_000.0, 20_000.0):
        total = _energy(_solve(fto=30, thickness=thickness, n_hi=0.15 + 3.5j,
                               period=2000.0, wavelength=500.0))
        assert 0.0 <= total <= 1.0 + 1e-9
        assert total <= previous + 1e-9, 'absorption cannot decrease with thickness'
        previous = total


# ---------------------------------------------------------------- algorithms agree

@pytest.mark.parametrize('thickness', [500.0, 10_000.0])
@pytest.mark.parametrize('fto', [10, 40])
def test_tmm_and_smm_agree(fto, thickness):
    """The two connecting algorithms are independent routes to the same answer.

    Scattering-matrix formulations are the usual answer to evanescent blow-up, so ETM
    tracking SMM to machine precision through a thick stack is the statement that ETM is
    not the fragile one here.
    """
    tmm = _solve(fto=fto, thickness=thickness, algo='ETM')
    smm = _solve(fto=fto, thickness=thickness, algo='SMM')

    assert np.allclose(tmm.R, smm.R, rtol=1e-9, atol=1e-12)
    assert np.allclose(tmm.T, smm.T, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize('thickness', [500.0, 10_000.0])
def test_tmm_and_smm_fields_agree_through_a_thick_layer(thickness):
    """The near field too, not only the efficiencies — the harder half for the SMM.

    SMM does not carry per-layer amplitudes; it recovers them after the cascade, and that
    recovery inverts a matrix built from the layer's interfaces and everything below it. Deep
    in the evanescent regime is where such an inverse would degrade if the reconstruction had
    picked the wrong reference plane, so this is the case that pins the choice.
    """
    cells = []
    for algo in ('ETM', 'SMM'):
        solver = wavebit.rcwa(
            backend=0, wavelength=400.0, period=[3000.0], fto=[40, 0],
            n_top=1.0, n_bot=1.5, theta=0.0, phi=0.0, psi=np.pi / 2,
            ucell=_grating(), thickness=[thickness], connecting_algo=algo,
            answers='field')
        solver.conv_solve()
        cells.append(np.asarray(solver.calculate_field(res_x=8, res_y=1, res_z=6)))

    tmm, smm = cells
    assert np.all(np.isfinite(np.abs(tmm))) and np.all(np.isfinite(np.abs(smm)))
    assert np.max(np.abs(tmm - smm)) < 1e-9 * np.max(np.abs(tmm))


def test_tmm_and_smm_agree_on_a_lossy_stack():
    tmm = _solve(fto=30, thickness=5_000.0, n_hi=0.15 + 3.5j, period=2000.0,
                 wavelength=500.0, algo='ETM')
    smm = _solve(fto=30, thickness=5_000.0, n_hi=0.15 + 3.5j, period=2000.0,
                 wavelength=500.0, algo='SMM')
    assert np.allclose(tmm.R, smm.R, rtol=1e-8, atol=1e-11)
    assert np.allclose(tmm.T, smm.T, rtol=1e-8, atol=1e-11)


# ---------------------------------------------------------------- precision

@pytest.mark.parametrize('thickness', [500.0, 5_000.0])
def test_single_precision_still_conserves_energy(thickness):
    """complex64 overflows at exp(88), so this is where a carried exponential would show."""
    total = _energy(_solve(fto=40, thickness=thickness, type_complex=np.complex64))
    assert total == pytest.approx(1.0, abs=1e-4)


def test_single_precision_tracks_double():
    """32-bit should cost accuracy, not correctness."""
    wide = _solve(fto=40, thickness=5_000.0, type_complex=np.complex128)
    narrow = _solve(fto=40, thickness=5_000.0, type_complex=np.complex64)
    assert np.allclose(np.real(wide.R), np.real(narrow.R), atol=1e-4)
    assert np.allclose(np.real(wide.T), np.real(narrow.T), atol=1e-4)


# ---------------------------------------------------------------- the guard itself

def test_results_are_finite_and_non_negative_at_the_extreme():
    """The two things ``check_physical`` looks for, asserted rather than warned about."""
    result = _solve(fto=40, thickness=50_000.0)
    for name, array in (('R', result.R), ('T', result.T)):
        values = np.real(np.asarray(array))
        assert np.all(np.isfinite(values)), f'{name} went non-finite'
        assert values.min() > -1e-9, f'{name} went negative — negative power'
