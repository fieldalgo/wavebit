"""Numerical stability where RCWA is known to lose it: evanescent orders through thick layers.

An order with ``|kx| > n·k0`` is evanescent, and its ``exp(|kz|·d)`` grows without bound in
the layer thickness. That growth is what makes a naive transfer-matrix formulation blow up,
and it is the failure any eig-free fast path would have to survive — so it is worth knowing
precisely where the *current* solver stands.

It stands well, and these tests pin that down. The point is not that the physics is right —
other tests cover that — but that the same physics survives being asked for in the ways
that usually break it: through a stack thick enough for the evanescent tail to overflow,
with enough retained orders to make most of them evanescent, in single precision, and
through both connecting algorithms.

That matters beyond regression. The plan is a fast eig-free mode cross-checked against the
precise one, and a reference is only useful if it holds up exactly where the fast path is
expected to fail.
"""

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa
import numpy as np
import pytest



def _grating(eps_hi=16.0, cells=8):
    """Half air, half high permittivity — the contrast populates the evanescent orders."""
    half = cells // 2
    return np.array([[[1.0] * half + [eps_hi] * half]])


# eps 16 is the index 4; eps 2.25 the index 1.5.
def _solve(*, truncation_order, thickness, eps_hi=16.0, algo='ETM', complex_dtype=np.complex128,
           period=3000.0, wavelength=400.0, eps_bot=2.25):
    return _legacy_wavebit_rcwa(
        backend='numpy', wavelength=wavelength, period=[period], truncation_order=[truncation_order, 0],
        top=1.0, bot=eps_bot, theta=0.0, phi=0.0, psi=np.pi / 2,
        structure=_grating(eps_hi), thickness=[thickness],
        layer_algorithm=algo, complex_dtype=complex_dtype,
    ).solve()


def _energy(result):
    return float(result.R.sum() + result.T.sum())


def evanescent_orders(truncation_order, period, wavelength, n=1.0):
    """How many retained orders are evanescent in a medium of index ``n``.

    ``kx_m = n·sinθ + m·λ/Λ`` in this solver's normalisation, so at normal incidence an
    order is evanescent once ``|m|·λ/Λ > n``. Reported so a test that claims to stress
    evanescent growth can show that it does.
    """
    orders = np.arange(-truncation_order, truncation_order + 1)
    return int(np.sum(np.abs(orders * wavelength / period) > n))


# ---------------------------------------------------------------- the regime is real

def test_the_test_conditions_actually_stress_evanescent_orders():
    """Guard the premise: without this the suite below could pass by never being hard."""
    assert evanescent_orders(truncation_order=40, period=3000.0, wavelength=400.0) > 60, (
        'most retained orders should be evanescent, or these tests prove nothing')


# ---------------------------------------------------------------- energy

@pytest.mark.parametrize('thickness', [500.0, 5_000.0, 50_000.0])
@pytest.mark.parametrize('truncation_order', [10, 40])
def test_energy_is_conserved_through_thick_layers(truncation_order, thickness):
    """A lossless stack conserves energy no matter how far the evanescent tail is pushed.

    50 um at truncation_order=40 puts ``exp(|kz|·d)`` far past what a float can hold, so a formulation
    that carried the growing exponential explicitly would have overflowed long before here.
    """
    assert _energy(_solve(truncation_order=truncation_order, thickness=thickness)) == pytest.approx(1.0, abs=1e-9)


def test_absorbing_metal_stays_physical_when_thick():
    """With loss, R + T < 1 — but it must stay in [0, 1] and not drift with thickness."""
    previous = 1.0
    for thickness in (200.0, 2_000.0, 20_000.0):
        total = _energy(_solve(truncation_order=30, thickness=thickness, eps_hi=-12.2275 + 1.05j,
                               period=2000.0, wavelength=500.0))
        assert 0.0 <= total <= 1.0 + 1e-9
        assert total <= previous + 1e-9, 'absorption cannot decrease with thickness'
        previous = total


# ---------------------------------------------------------------- algorithms agree

@pytest.mark.parametrize('thickness', [500.0, 10_000.0])
@pytest.mark.parametrize('truncation_order', [10, 40])
def test_tmm_and_smm_agree(truncation_order, thickness):
    """The two connecting algorithms are independent routes to the same answer.

    Scattering-matrix formulations are the usual answer to evanescent blow-up, so ETM
    tracking SMM to machine precision through a thick stack is the statement that ETM is
    not the fragile one here.
    """
    tmm = _solve(truncation_order=truncation_order, thickness=thickness, algo='ETM')
    smm = _solve(truncation_order=truncation_order, thickness=thickness, algo='SMM')

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
        solver = _legacy_wavebit_rcwa(
            backend='numpy', wavelength=400.0, period=[3000.0], truncation_order=[40, 0],
            top=1.0, bot=2.25, theta=0.0, phi=0.0, psi=np.pi / 2,
            structure=_grating(), thickness=[thickness], layer_algorithm=algo,
            answers='field')
        solver.solve()
        cells.append(np.asarray(solver.reconstruct_field(sample_x=8, sample_y=1, sample_z=6)))

    tmm, smm = cells
    assert np.all(np.isfinite(np.abs(tmm))) and np.all(np.isfinite(np.abs(smm)))
    assert np.max(np.abs(tmm - smm)) < 1e-9 * np.max(np.abs(tmm))


def test_tmm_and_smm_agree_on_a_lossy_stack():
    tmm = _solve(truncation_order=30, thickness=5_000.0, eps_hi=-12.2275 + 1.05j, period=2000.0,
                 wavelength=500.0, algo='ETM')
    smm = _solve(truncation_order=30, thickness=5_000.0, eps_hi=-12.2275 + 1.05j, period=2000.0,
                 wavelength=500.0, algo='SMM')
    assert np.allclose(tmm.R, smm.R, rtol=1e-8, atol=1e-11)
    assert np.allclose(tmm.T, smm.T, rtol=1e-8, atol=1e-11)


# ---------------------------------------------------------------- precision

@pytest.mark.parametrize('thickness', [500.0, 5_000.0])
def test_single_precision_still_conserves_energy(thickness):
    """complex64 overflows at exp(88), so this is where a carried exponential would show."""
    total = _energy(_solve(truncation_order=40, thickness=thickness, complex_dtype=np.complex64))
    assert total == pytest.approx(1.0, abs=1e-4)


def test_single_precision_tracks_double():
    """32-bit should cost accuracy, not correctness."""
    wide = _solve(truncation_order=40, thickness=5_000.0, complex_dtype=np.complex128)
    narrow = _solve(truncation_order=40, thickness=5_000.0, complex_dtype=np.complex64)
    assert np.allclose(np.real(wide.R), np.real(narrow.R), atol=1e-4)
    assert np.allclose(np.real(wide.T), np.real(narrow.T), atol=1e-4)


# ---------------------------------------------------------------- the guard itself

def test_results_are_finite_and_non_negative_at_the_extreme():
    """Finite, and no negative power -- asserted on a named scene rather than guessed at.

    These two conditions used to be warned about generically by ``check_physical``, on every
    solve. That check was removed because it could not tell a broken solve from an active
    medium, whose right answer can exceed one or carry a negative order (see
    ``docs/reference/common/result-guards.md``). A scene whose right answer is known, like this one, is where
    the conditions can be asserted instead of guessed at.
    """
    result = _solve(truncation_order=40, thickness=50_000.0)
    for name, array in (('R', result.R), ('T', result.T)):
        values = np.real(np.asarray(array))
        assert np.all(np.isfinite(values)), f'{name} went non-finite'
        assert values.min() > -1e-9, f'{name} went negative — negative power'


# --- a layer whose permittivity is nearly zero everywhere ---------------------------------
#
# SMM inverts the layer's magnetic modes; the p modes' columns are of order eps, so with a y
# wavevector component (s and p mixed) a uniform eps = 1e-13 layer returned R + T - 1 = -1e-6
# and eps = 1e-16 returned +0.09. ETM inverts nothing of the kind and is the reference here.

def _near_zero(eps, *, phi=.4, algorithm='smm', layer=None, precision='double', check=False):
    from wavebit import rcwa
    from wavebit.materials import Material
    from wavebit.units import Units
    layer = layer or rcwa.Layer(thickness=.2, background=Material(eps))
    raster = layer.epsilon is not None
    problem = rcwa.Problem(units=Units(length='um'),
                           stack=rcwa.Stack(period=(.6, .5), layers=(layer,)),
                           excitation=rcwa.Excitation(wavelength=.8, sources=(
                               rcwa.PlaneWave(theta=.3, phi=phi, polarization=(1., .5j)),)))
    numerics = rcwa.Numerics(orders=(2, 2) if raster and np.shape(layer.epsilon)[0] > 1 else (3, 0),
                             algorithm=algorithm)
    if check:
        return rcwa.check(problem, numerics=numerics, precision=precision)
    result = rcwa.run(problem, numerics=numerics, precision=precision,
                      backend='torch' if precision == 'single' else 'numpy')
    return float(np.sum(np.asarray(result.R))), float(np.sum(np.asarray(result.T)))


def _near_zero_layers(eps):
    from wavebit import rcwa
    from wavebit.materials import Material
    return {'uniform': rcwa.Layer(thickness=.2, background=Material(eps)),
            'tensor': rcwa.Layer(thickness=.2, background=Material([eps, eps, eps])),
            'raster_1d': rcwa.Layer(thickness=.2, epsilon=np.full((1, 4), eps)),
            'raster_2d': rcwa.Layer(thickness=.2, epsilon=np.full((3, 4), eps))}


@pytest.mark.parametrize('kind', ['uniform', 'tensor', 'raster_1d', 'raster_2d'])
@pytest.mark.parametrize('eps', [1e-13, -1e-13, 1e-13 + 1e-14j, 1e-16])
def test_smm_refuses_a_near_zero_permittivity_layer_at_conical_incidence(kind, eps):
    from wavebit.errors import UnsupportedFeatureError
    layer = _near_zero_layers(eps)[kind]
    issues = _near_zero(eps, layer=layer, check=True).issues
    assert [(i.code, i.path) for i in issues] == [('rcwa.near_zero_permittivity_smm', 'stack.layers[0]')]
    with pytest.raises(UnsupportedFeatureError) as caught:
        _near_zero(eps, layer=layer)
    assert caught.value.code == 'rcwa.near_zero_permittivity_smm'
    assert caught.value.path == 'stack.layers[0]' and 'algorithm="etm"' in caught.value.message


def test_smm_floor_follows_the_run_precision():
    from wavebit.errors import UnsupportedFeatureError
    pytest.importorskip('torch')
    R, T = _near_zero(1e-3, precision='single')       # above sqrt(float32 eps) = 3.5e-4
    assert abs(R + T - 1) < 1e-5
    for eps in (1e-4, 1e-7):
        with pytest.raises(UnsupportedFeatureError):
            _near_zero(eps, precision='single')


def test_near_zero_layer_has_a_continuous_energy_conserving_limit():
    """ETM approaches the eps -> 0 limit; SMM above the floor and SMM without y mixing agree."""
    etm = {eps: _near_zero(eps, algorithm='etm') for eps in (1e-6, 1e-9, 1e-12, 1e-13, 1e-16)}
    for R, T in etm.values():
        assert abs(R + T - 1) < 1e-14
    limit = etm[1e-16][0]
    slopes = [(etm[eps][0] - limit)/eps for eps in (1e-6, 1e-9, 1e-12)]
    # R moves linearly in eps (slope -0.548); the 1e-12 quotient carries 1e-16/1e-12 of rounding.
    assert all(abs(slope - slopes[1]) < 1e-3 for slope in slopes) and abs(slopes[1]) < 1
    R, T = _near_zero(1e-6)                                 # above the double floor 1.5e-8
    assert abs(R - etm[1e-6][0]) < 1e-10 and abs(R + T - 1) < 1e-10
    for eps in (1e-13, 1e-16):                              # no y component: not refused, exact
        R, T = _near_zero(eps, phi=0.)
        R_etm, T_etm = _near_zero(eps, phi=0., algorithm='etm')
        assert abs(R + T - 1) < 1e-14 and abs(R - R_etm) < 1e-14


# --- a layer whose eps_zz is zero, or whose inverse overflows ------------------------------
#
# RCWA inverts eps_zz, and the inverse rule each sample. eps = 0 raised "SVD did not
# converge" / "Singular matrix" (NumPy, JAX) or a raw torch LinAlgError on every algorithm
# and incidence. Small nonzero values still solve; single-precision ETM fails some of them by
# rounding, which must surface as rcwa.singular.

def _zero_layers(eps):
    from wavebit import rcwa
    from wavebit.materials import Material
    region = rcwa.Region(geometry=rcwa.Rectangle(center=(0., 0.), size=(.2, .2)), material=Material(eps))
    return {'uniform': rcwa.Layer(thickness=.2, background=Material(eps)),
            'tensor_zz': rcwa.Layer(thickness=.2, background=Material([2., 2., eps])),
            'raster_pixel': rcwa.Layer(thickness=.2, epsilon=np.array([[eps, eps, 2.25, 2.25]])),
            'region': rcwa.Layer(thickness=.2, background=Material(2.25), regions=(region,))}


def _zero_problem(layer, phi=.4):
    from wavebit import rcwa
    from wavebit.units import Units
    return rcwa.Problem(units=Units(length='um'), stack=rcwa.Stack(period=(.6, .5), layers=(layer,)),
                        excitation=rcwa.Excitation(wavelength=.8, sources=(
                            rcwa.PlaneWave(theta=.3, phi=phi, polarization=(1., .5j)),)))


@pytest.mark.parametrize('backend', ['numpy', 'torch'])
@pytest.mark.parametrize('algorithm', ['etm', 'smm'])
@pytest.mark.parametrize('phi', [0., .4])
@pytest.mark.parametrize('kind', ['uniform', 'tensor_zz', 'raster_pixel', 'region'])
def test_a_zero_eps_zz_is_refused_at_check_on_every_path(kind, phi, algorithm, backend):
    pytest.importorskip(backend)
    from wavebit import rcwa
    from wavebit.errors import UnsupportedFeatureError
    problem, numerics = _zero_problem(_zero_layers(0.)[kind], phi), rcwa.Numerics(orders=(3, 0), algorithm=algorithm)
    issues = rcwa.check(problem, numerics=numerics, backend=backend).issues
    assert [(i.code, i.path) for i in issues] == [('rcwa.zero_permittivity', 'stack.layers[0]')]
    with pytest.raises(UnsupportedFeatureError) as caught:
        rcwa.run(problem, numerics=numerics, backend=backend)
    assert caught.value.code == 'rcwa.zero_permittivity' and 'nonzero permittivity' in caught.value.message


@pytest.mark.parametrize('precision', ['double', 'single'])
def test_a_permittivity_whose_inverse_overflows_is_refused(precision):
    pytest.importorskip('torch')
    from wavebit import rcwa
    from wavebit.errors import UnsupportedFeatureError
    eps = 1e-320 if precision == 'double' else 1e-40          # 1/eps is not finite there
    with pytest.raises(UnsupportedFeatureError) as caught:
        rcwa.run(_zero_problem(_zero_layers(eps)['uniform']), numerics=rcwa.Numerics(orders=(3, 0), algorithm='etm'),
                 backend='torch', precision=precision)
    assert caught.value.code == 'rcwa.zero_permittivity'


@pytest.mark.parametrize('backend', ['torch', 'jax'])
def test_single_precision_near_zero_either_solves_or_fails_structured(backend):
    """The reported case: single-precision ETM on a uniform |eps| = 1e-13 layer raised a raw
    torch LinAlgError. Whether such a factorization fails depends on rounding; it must either
    conserve energy or say rcwa.singular, never leak a backend error."""
    pytest.importorskip(backend)
    from wavebit import rcwa
    from wavebit.errors import NumericalError
    numerics = rcwa.Numerics(orders=(3, 0), algorithm='etm')
    for eps in (1e-9, 1e-12, 1e-13, 1e-16):
        try:
            result = rcwa.run(_zero_problem(_zero_layers(eps)['uniform']), numerics=numerics,
                              backend=backend, precision='single')
        except NumericalError as error:
            assert error.code == 'rcwa.singular'
        else:
            assert abs(float(np.sum(np.asarray(result.R)) + np.sum(np.asarray(result.T))) - 1) < 1e-6


def test_small_nonzero_permittivity_that_solves_is_not_refused():
    """Not refused: uniform layers at 1e-13 and 1e-18 (below double roundoff), small samples beside ordinary ones,
    and a zero in-plane component with a nonzero eps_zz; each conserves energy."""
    from wavebit import rcwa
    from wavebit.materials import Material
    numerics = rcwa.Numerics(orders=(3, 0), algorithm='etm')
    for layer in (_zero_layers(1e-13)['uniform'], _zero_layers(1e-18)['uniform'], _zero_layers(1e-16)['raster_pixel'],
                  _zero_layers(1e-16)['region'], rcwa.Layer(thickness=.2, background=Material([0., 2., 2.]))):
        result = rcwa.run(_zero_problem(layer), numerics=numerics)
        assert abs(float(np.sum(result.R) + np.sum(result.T)) - 1) < 1e-13
    # Below the double roundoff the answer is the eps -> 0 limit: R moves by 0.548 eps.
    near, tiny = (float(np.sum(rcwa.run(_zero_problem(_zero_layers(eps)['uniform']), numerics=n).R))
                  for eps, n in ((1e-13, numerics), (1e-18, numerics)))
    assert abs(near - tiny) < 1e-13


def test_a_torch_factorization_failure_is_a_structured_numerical_error(monkeypatch):
    torch = pytest.importorskip('torch')
    from wavebit import rcwa
    from wavebit.errors import NumericalError
    from wavebit.rcwa._backends.torch_backend import RCWATorch
    def singular(self, *args, **kwargs):
        raise torch.linalg.LinAlgError('linalg.inv: The diagonal element 8 is zero')
    monkeypatch.setattr(RCWATorch, 'solve', singular)
    with pytest.raises(NumericalError) as caught:
        rcwa.run(_zero_problem(_zero_layers(2.25)['uniform']), numerics=rcwa.Numerics(orders=(3, 0)), backend='torch')
    assert caught.value.code == 'rcwa.singular'
