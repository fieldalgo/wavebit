"""Normal incidence needs no detour: the order azimuth is defined where it used to be NaN.

Every solve that couples s and p rotates each order into its own frame by ``cos(varphi)`` /
``sin(varphi)``, with ``varphi = atan(ky/kx)``. That is undefined wherever ``kx == 0``, which
happens in two places with two *different* answers:

* the ``m == 0`` column of a 2D order grid (``kx == 0``, ``ky != 0``), at any incidence —
  the wavevector lies along y, so the azimuth is a right angle;
* the zeroth order at exactly normal incidence (``kx == ky == 0``), where the in-plane
  wavevector carries no direction at all — and the answer is ``phi``.

The library used to route around both with a ``perturbation`` constructor argument that
replaced ``theta == 0`` with ``1e-20``. That made ``kx_0 = eps*cos(phi)`` and
``ky_0 = eps*sin(phi)``, from which ``atan`` recovered both answers by itself. The knob is
gone and :func:`~wavebit._common.conventions.order_azimuth` fills the singularities directly,
with ``phi`` passed to the solves explicitly rather than smuggled through ``kx`` and ``ky``.

**The trap, which cost a wrong first attempt.** The origin looks like a gauge point — a wave
going straight down has a degenerate plane of incidence, so "any azimuth is a valid s/p
basis" is a tempting argument. It is wrong. The incident *polarization* still needs a
reference direction in the xy plane, and ``phi`` is what supplies it. Filling the origin with
``0`` makes ``F_c = 1``, ``F_s = 0``, and every ``theta = 0`` answer stops depending on
``phi`` at all — silently, and only when ``phi != 0``. It was caught by
``tests/references/test_s4_field_golden.py`` (twelve ``theta=0, phi=35deg`` cases, field
components collapsing from O(1) to 1e-17) and *not* by a 14-case bit-identity battery, whose
normal-incidence cases all used ``phi = 0`` and were therefore right by coincidence.

That is what :func:`test_normal_incidence_still_depends_on_phi` exists to prevent from
recurring; it is the cheapest statement of the failure and it does not need a golden.

Also pinned here: gradients still flow **through** ``varphi``, which needs a derivative with
respect to ``period`` or ``wavelength``, since those feed ``kx = m*wl/period``.
Differentiating the raster instead never touches this code, and an earlier version of these
tests did exactly that and proved nothing.
"""

from wavebit.compat import legacy as _legacy_wavebit
import warnings

import numpy as np
import pytest

from wavebit._common.backend import array_namespace
from wavebit._common.conventions import order_azimuth

HALF_PI = 1.5707963267948966

# Permittivity rasters: 12.25 is the index 3.5 these cells carried before the material flip.
_UC_1D = np.array([[[1.0, 1.0, 1.0, 12.25, 12.25, 12.25, 1.0, 1.0]]])
_UC_2D = np.ones((1, 8, 8))
_UC_2D[0, 2:6, 2:6] = 12.25


def _xp():
    return array_namespace(np.zeros(1, dtype=np.complex128))


def _solve(backend='numpy', structure=None, truncation_order=(6, 0), theta=0.0, phi=0.0, psi=np.pi / 2, **kw):
    base = dict(backend=backend, wavelength=500.0, period=[700.0, 700.0], top=1.0,
                bot=2.25, theta=theta, phi=phi, psi=psi,
                structure=_UC_1D if structure is None else structure, thickness=[220.0], truncation_order=list(truncation_order))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return _legacy_wavebit.rcwa(**{**base, **kw}).solve()


def _total_r(result):
    return float(np.real(np.asarray(result.R, dtype=np.complex128)).sum())


def _energy(result):
    return sum(float(np.real(np.asarray(v, dtype=np.complex128)).sum())
               for v in (result.R, result.T))


# ---------------------------------------------------------------- the singular points

def test_the_origin_carries_the_incidence_azimuth():
    """``kx == ky == 0``: the answer is ``phi``, not zero and not a free gauge.

    ``phi``, with its *direction*. This used to assert ``atan(tan(phi))`` — the wrap into
    ``(-pi/2, pi/2)`` the removed ``perturbation`` detour produced, kept deliberately as the
    limit it reached — which agrees with ``phi`` while ``|phi| < pi/2`` and is ``phi -+ pi``
    beyond it. That is the same ``atan``-of-a-ratio spelling that was a defect on the
    ``Re(kx) < 0`` half-plane, and for the same reason: a ratio has already discarded
    ``sign(cos(phi))``. At the origin the "``kx``" whose sign went missing *is* ``cos(phi)``.

    It survived one round longer than the half-plane because it looked like it bought
    something — ``phi`` and ``phi + pi`` landing on one frame. It does not; see
    :func:`test_the_two_spellings_of_one_incidence_agree`, which is the measurement that
    settles it.

    Asserted on ``cos`` and ``sin`` rather than on the angle, because those are the only two
    things any caller reads: ``phi`` and ``phi + 2pi`` must both pass and ``phi + pi`` must
    not.
    """
    kx = np.array([0.0, 0.5, -0.5], dtype=np.complex128)
    ky = np.zeros(3, dtype=np.complex128)
    for phi in (0.0, 0.3, np.deg2rad(35.0), 1.2, -0.4, 2.0, -2.0, 3.0, -3.0):
        got = np.asarray(order_azimuth(_xp(), ky, kx, phi))
        assert np.all(np.isfinite(got))
        assert np.real(np.cos(got[0])) == pytest.approx(np.cos(phi), abs=1e-14)
        assert np.real(np.sin(got[0])) == pytest.approx(np.sin(phi), abs=1e-14)


def test_the_kx_zero_column_carries_the_right_angle():
    """``kx == 0`` with ``ky != 0``: the wavevector lies along y, so the azimuth is pi/2.

    The sign follows ``Re(ky)``, and a purely imaginary ``ky`` takes ``+pi/2`` — both of which
    are what ``atan(ky / 1e-20)`` returned, rather than a fresh convention invented here.
    This column has nothing to do with ``phi``, and exists at any incidence.
    """
    ky = np.array([0.7, -0.7, 0.7 + 0.3j, -0.7 + 0.3j, 0.5j, -0.5j], dtype=np.complex128)
    kx = np.zeros_like(ky)
    got = np.asarray(order_azimuth(_xp(), ky, kx, 0.9))
    expected = np.array([HALF_PI, -HALF_PI, HALF_PI, -HALF_PI, HALF_PI, HALF_PI])
    assert np.array_equal(np.real(got), expected)
    assert np.all(np.isfinite(got))


def test_ordinary_orders_point_along_their_own_wavevector():
    """Away from ``kx == 0`` the frame is ``s_hat = z_hat x k_hat_par`` — i.e. ``atan2``.

    This used to assert ``atan(ky/kx)`` exactly, "branch and all", and that is the defect
    rather than the contract: ``atan`` reads a *ratio*, which has already discarded
    ``sign(kx)``, so ``(kx, ky)`` and ``(-kx, -ky)`` came back with the same frame instead of
    opposite ones. Every ``Re(kx) < 0`` order was therefore handed ``-s_hat``, ``-p_hat``.

    Stated as the parallelism it actually means — ``(cos, sin)`` must be ``(kx, ky)``
    normalised, *with* its direction — so that it holds for the complex ``kx`` an absorbing
    superstrate produces and not only on the real axis, where ``atan2`` would say the same.
    """
    kx = np.array([0.5, -0.5, 1.0, -1.0, 0.4 + 0.2j, -0.4 - 0.2j], dtype=np.complex128)
    ky = np.array([0.2, 0.2, -0.3, -0.3, 0.6 - 0.1j, 0.6 - 0.1j], dtype=np.complex128)
    got = np.asarray(order_azimuth(_xp(), ky, kx, 0.9))

    norm = np.sqrt(kx ** 2 + ky ** 2)
    assert np.cos(got) == pytest.approx(kx / norm, abs=1e-15)
    assert np.sin(got) == pytest.approx(ky / norm, abs=1e-15)

    # And the half-turn is real: the reversed wavevector must give the reversed frame.
    flipped = np.asarray(order_azimuth(_xp(), -ky, -kx, 0.9))
    assert np.cos(flipped) == pytest.approx(-np.cos(got), abs=1e-15)
    assert np.sin(flipped) == pytest.approx(-np.sin(got), abs=1e-15)


def test_the_frame_is_continuous_where_kx_crosses_zero():
    """The strongest statement that the old half-turn was a defect and not a convention.

    A convention may be surprising; it may not be discontinuous in the inputs. ``atan(ky/kx)``
    was: at ``kx = -1e-12`` it returned ``sin(varphi) = -1`` and at ``kx = 0`` — three lines
    later, in the branch it special-cases — ``+1``. The function disagreed with its own
    special case, which is what marks this as an accident of spelling rather than a choice.
    """
    ky = np.full(5, 0.7 + 0j)
    kx = np.array([-1e-3, -1e-12, 0.0, 1e-12, 1e-3], dtype=np.complex128)
    got = np.asarray(order_azimuth(_xp(), ky, kx, 0.0))

    assert np.real(np.sin(got)) == pytest.approx(np.ones(5), abs=1e-5)
    # cos must be kx/|k| and pass smoothly through zero *with its sign*, which is the whole
    # claim -- `rel` because cos is read back off an angle atan built from a ratio of 700, so
    # it carries ~1e-6 relative round-off, and the exactly-zero entry needs the abs floor.
    assert np.real(np.cos(got)) == pytest.approx(np.real(kx) / 0.7, rel=1e-5, abs=1e-15)


# ---------------------------------------------------------------- the premise

def test_the_2d_grid_at_normal_incidence_hits_both_singularities():
    """Guard the premise: the cases below must actually reach the code they test.

    At ``theta = 0`` the order grid is ``kx_m = m*wl/Px``, ``ky_n = n*wl/Py``, so the whole
    ``m == 0`` column has ``kx == 0`` — the ``(0, 0)`` entry is the origin and every
    ``(0, n != 0)`` entry is the right-angle case. Both singular branches are therefore live
    in one 2D solve, which is why the ``2d`` cases carry most of this file's weight.

    Exactly zero, not nearly: ``m * wl / P`` is a float product with ``m == 0``, so the test
    is ``== 0`` rather than a tolerance, and the branch really is taken.

    The mirror-image check matters too. Oblique incidence looks like it should reach the
    column at ``phi = pi/2``, where ``kx_m = sin(theta)*cos(phi) + m*wl/Px`` would lose its
    first term — but ``cos(pi/2)`` is ``6.1e-17`` in binary floating point, never ``0``, so
    ``kx_0`` lands near the singularity without ever being on it. There is no float ``phi``
    that reaches it. Pinning that keeps someone from adding an oblique "coverage" case that
    silently exercises nothing.
    """
    wl, px, py = 500.0, 700.0, 700.0
    m = np.arange(-4, 5)
    kx = m * wl / px
    ky = m * wl / py
    assert kx[4] == 0.0 and ky[4] == 0.0, 'the origin is not in the retained set'
    assert ky[5] != 0.0, 'the m==0 column has no ky != 0 entry to exercise'

    for theta in (0.3, 0.6, 1.2):
        kx0 = np.sin(theta) * np.cos(np.pi / 2)
        assert kx0 != 0.0, (
            'cos(pi/2) came out exactly zero, so an oblique case can now reach the column '
            'singularity and this file should grow one')


# ---------------------------------------------------------------- the regression

def test_normal_incidence_still_depends_on_phi():
    """The one that would have caught the wrong first attempt, without a golden.

    At ``theta = 0`` every order has ``ky == 0`` and ``kx_0 == 0``, so if the origin is
    filled with a constant then *every* ``varphi`` is zero and the answer stops depending on
    ``phi`` entirely. These four would then be bit-identical rather than distinct.

    Distinctness is the whole assertion — the values themselves are pinned against S4 in
    ``tests/references/test_s4_field_golden.py``, which is where a wrong *value* would show.
    """
    got = [_total_r(_solve(theta=0.0, phi=phi, psi=0.7))
           for phi in (0.0, 0.3, np.deg2rad(35.0), 1.2)]
    for i, a in enumerate(got):
        for b in got[i + 1:]:
            assert abs(a - b) > 1e-6, (
                f'theta=0 gave the same answer for two different phi ({got}) — the incidence '
                'azimuth is being dropped at the origin')


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
@pytest.mark.parametrize('case', ['1d', 'conical', '2d', 'smm', '2d_smm'])
@pytest.mark.parametrize('phi', [0.0, 0.6108652381980153])
def test_normal_incidence_solves_without_a_nudge(backend, case, phi):
    """Every path, every backend, at exactly ``theta = 0`` — and at both azimuths.

    Before :func:`order_azimuth`, removing the nudge made all of these raise
    ``FloatingPointError``: the 2D ones for the ``m == 0`` column, the rest for the origin.
    """
    if backend == 'jax':
        pytest.importorskip('jax')
        _legacy_wavebit.enable_x64()
    elif backend == 'torch':
        pytest.importorskip('torch')

    kw = {'1d': {},
          'conical': dict(psi=0.7),
          '2d': dict(structure=_UC_2D, truncation_order=(4, 4)),
          'smm': dict(layer_algorithm='SMM'),
          '2d_smm': dict(structure=_UC_2D, truncation_order=(4, 4), layer_algorithm='SMM')}[case]
    result = _solve(backend=backend, phi=phi, **kw)
    for array in (result.R, result.T):
        values = np.asarray(array, dtype=np.complex128)
        assert np.all(np.isfinite(values)), f'{backend}/{case} went non-finite at theta=0'
    assert _energy(result) == pytest.approx(1.0, abs=1e-9)


def test_the_first_order_path_gets_the_azimuth_too():
    """``solve_2d_first_order`` has its own ``order_azimuth`` call and its own ``phi`` to thread.

    Forced onto a problem the second-order path can also express, the two must agree — which
    is the only oracle available here, and the one ``tests/rcwa/test_formulation_dispatch.py``
    already relies on. A ``phi`` that reached the second-order path and not the first-order one
    would show up as a disagreement at ``phi != 0`` and nowhere else, so both azimuths are run.
    """
    for phi in (0.0, 0.6108652381980153):
        second = _solve(structure=_UC_2D, truncation_order=(4, 4), phi=phi, psi=0.7)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            solver = _legacy_wavebit.rcwa(backend='numpy', wavelength=500.0, period=[700.0, 700.0],
                                  top=1.0, bot=2.25, theta=0.0, phi=phi, psi=0.7,
                                  structure=_UC_2D, thickness=[220.0], truncation_order=[4, 4])
            solver.formulation = 'first_order'
            first = solver.solve()
        assert str(solver.formulation_assigned).endswith('FIRST_ORDER'), \
            solver.formulation_assigned
        assert _total_r(first) == pytest.approx(_total_r(second), abs=1e-11)


def test_the_two_spellings_of_one_incidence_agree():
    """The measurement that says the origin wrap was a defect, not a convention.

    At ``theta = 0`` the incident field is ``a_s*s_hat(phi) + a_p*p_hat(phi)``, and turning
    the reference azimuth through ``pi`` reverses both basis vectors — so ``(phi + pi, psi)``
    and ``(phi, polarization=(-a_s, -a_p))`` are **the same physical incident field**, written
    two ways. Whatever frame the library reports in, the two spellings must agree wherever
    they share a frame.

    They share a frame everywhere but the origin, and provably: at ``theta = 0`` the order
    grid is ``kx_m = m*wl/Px``, ``ky_n = n*wl/Py``, which does not contain ``phi`` at all. So
    every order except ``(0, 0)`` is reported against the identical basis in both runs, and
    the amplitudes must be equal outright — not equal up to a sign, not equal in modulus.

    That is what the wrap broke. It sent ``phi + pi`` to the same ``varphi`` as ``phi``, which
    means the solve silently answered for ``phi`` — an incident field of the opposite sign to
    the one asked for. The two spellings then came back negated on all 80 off-centre orders
    (measured 2.8e-01 apart, and 1.2e-15 apart after negating), while ``R`` agreed to 2.8e-16
    and hid it.

    The ``(0, 0)`` order *is* allowed to differ by a sign here, and does: it is the one order
    whose own reporting frame is the one being turned. Asserted rather than skipped, because
    "the zeroth order flips too" and "nothing flips" are different states and only one of them
    is right.

    The old form of this test asserted the opposite — that ``phi`` and ``phi + pi`` agree
    outright — on the reasoning that turning the frame flips ``s`` and ``p`` together so the
    sign cancels in every ratio. It cancels for the zeroth order. It does not cancel for any
    other, whose frame does not turn with ``phi``.
    """
    psi = 0.7
    pair = (-np.sin(psi), -np.cos(psi))
    for phi in (0.0, 0.3, 0.6108652381980153, 1.2, -0.4):
        by_azimuth = _solve(structure=_UC_2D, truncation_order=(4, 4), phi=phi + np.pi, psi=psi)
        by_pair = _solve(structure=_UC_2D, truncation_order=(4, 4), phi=phi, psi=None, polarization=pair)
        for name in ('r_s', 'r_p', 't_s', 't_p'):
            x = np.asarray(getattr(by_azimuth.incident, name), dtype=np.complex128)
            y = np.asarray(getattr(by_pair.incident, name), dtype=np.complex128)
            assert np.abs(x).max() > 1e-3, f'{name} came back empty, so this asserts nothing'
            off = np.ones(x.shape, dtype=bool)
            off[4, 4] = False                       # the origin: its own frame is what turned
            assert np.allclose(x[off], y[off], rtol=0, atol=1e-13), (
                f'phi={phi}: {name} disagrees with the same field spelled as an amplitude '
                'pair' + (' by a sign' if np.allclose(x[off], -y[off], rtol=0, atol=1e-13)
                          else ''))
            assert np.allclose(x[4, 4], -y[4, 4], rtol=0, atol=1e-13), (
                f'phi={phi}: {name} at the zeroth order should differ by exactly a sign — '
                'that is the order whose reporting frame the azimuth turned')


def test_phi_plus_pi_negates_the_reconstructed_field():
    """The same statement in the global frame, where there is no basis to hide behind.

    The amplitudes are quoted in each order's own s/p frame, so someone can always argue about
    which frame they are in. The reconstructed field is a vector in the global ``xy`` frame and
    admits no such argument: turning the reference azimuth through ``pi`` reverses ``s_hat``
    and ``p_hat``, hence the incident field, hence — Maxwell being linear — the whole cell.

    So the field must come back **negated**, exactly, and that is the assertion. Measured
    ``|a + b| = 3.7e-14`` against a cell scale of 4.26, with ``|a - b| = 8.5``.

    This test asserted the opposite until 2026-08-16, and was right about what the code did:
    the origin wrap sent ``phi`` and ``phi + pi`` to one ``varphi``, so the solve answered for
    ``phi`` either way and the cell came back identical. Identical is the wrong answer. The
    incident pair ``(a_s, a_p)`` is written *against* the reference direction ``phi`` supplies,
    so reversing that direction reverses the field it describes; a solve that returns the same
    cell for both is quietly ignoring half of what ``phi`` means.

    Six components across a grid rather than four scalars, so a sign confined to one component
    has nowhere to hide.
    """
    def cell(**kw):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            solver = _legacy_wavebit.rcwa(backend='numpy', wavelength=500.0, period=[700.0, 700.0],
                                  top=1.0, bot=2.25, theta=0.0,
                                  structure=_UC_2D, thickness=[220.0], truncation_order=[4, 4],
                                  answers='field', **kw)
            solver.solve()
            return np.asarray(solver.reconstruct_field(sample_x=6, sample_y=6, sample_z=5),
                              dtype=np.complex128)

    psi = 0.7
    for phi in (0.3, 0.6108652381980153, 1.2):
        a, b = cell(phi=phi, psi=psi), cell(phi=phi + np.pi, psi=psi)
        scale = float(np.max(np.abs(a)))
        assert scale > 1e-6, 'the field came back empty, so this asserts nothing'
        assert np.max(np.abs(a + b)) < 1e-12 * scale, (
            f'phi={phi}: the field is not the negative of the phi+pi one'
            + ('  (it is identical -- the azimuth is being read as a line, not a direction)'
               if np.max(np.abs(a - b)) < 1e-12 * scale else ''))

        # And the other spelling of that same reversal must land on the same cell.
        pair = cell(phi=phi, psi=None, polarization=(-np.sin(psi), -np.cos(psi)))
        assert np.max(np.abs(pair - b)) < 1e-12 * scale, (
            f'phi={phi}: the amplitude-pair spelling of the same incident field gives a '
            'different cell')


@pytest.mark.parametrize('label, kw', [
    ('backside', dict(incidence_side='bottom')),
    ('complex64', dict(complex_dtype=np.complex64)),
])
def test_the_azimuth_survives_the_other_solve_variants(label, kw):
    """Backside incidence and single precision, at ``theta = 0`` with a non-zero ``phi``.

    Neither has its own ``varphi``, but both re-enter the same solves by a different route,
    and the mirrored stack in particular reorders arguments that this change touched.
    """
    tol = 1e-9 if label == 'backside' else 1e-4
    for phi in (0.0, 0.6108652381980153):
        result = _solve(phi=phi, psi=0.7, **kw)
        assert np.all(np.isfinite(np.asarray(result.R, dtype=np.complex128)))
        assert _energy(result) == pytest.approx(1.0, abs=tol)


def test_perturbation_argument_is_gone():
    """Removed rather than deprecated, so passing it must fail loudly.

    Silently accepting it would be the worst of the three: the name promises a knob that
    changes the answer, and after this change nothing would.
    """
    with pytest.raises(TypeError):
        _solve(perturbation=1e-14)


def test_the_solves_refuse_to_guess_the_azimuth():
    """``phi`` is keyword-only and required on the core solves, not defaulted to zero.

    A default is what would let the next caller reintroduce exactly the bug this file is
    about, and it would do it silently — a wrong azimuth is finite, physical-looking and
    only wrong when ``phi != 0``.
    """
    import inspect

    from wavebit.rcwa._engine.scattering import smm_1d_conical, smm_2d
    from wavebit.rcwa._engine.solve_1d import solve_1d_conical
    from wavebit.rcwa._engine.solve_2d import solve_2d
    from wavebit.rcwa._engine.solve_2d_first_order import solve_2d_first_order

    for fn in (solve_1d_conical, solve_2d, solve_2d_first_order, smm_1d_conical, smm_2d):
        p = inspect.signature(fn).parameters.get('phi')
        assert p is not None, f'{fn.__name__} lost its phi argument'
        assert p.kind is inspect.Parameter.KEYWORD_ONLY, f'{fn.__name__}: phi went positional'
        assert p.default is inspect.Parameter.empty, f'{fn.__name__}: phi acquired a default'


# ---------------------------------------------------------------- gradients through varphi

def test_gradient_flows_through_the_azimuth():
    """Differentiate ``period``, not the raster — that is what reaches ``varphi``.

    ``kx = n*sin(theta) + m*wl/period``, so a period derivative passes straight through the
    singular point at ``theta = 0``. A raster derivative does not touch this code at all,
    which is why it cannot stand in for this test.
    """
    jax = pytest.importorskip('jax')
    _legacy_wavebit.enable_x64()
    import jax.numpy as jnp

    def loss(period, structure, truncation_order, phi):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = _legacy_wavebit.rcwa(backend='jax', wavelength=500.0, period=[period, period],
                             top=1.0, bot=2.25, theta=0.0, phi=phi, psi=np.pi / 2,
                             structure=jnp.asarray(structure), thickness=[220.0],
                             truncation_order=list(truncation_order)).solve()
        return jnp.real(r.R).sum()

    for structure, truncation_order in ((_UC_1D, (6, 0)), (_UC_2D, (4, 4))):
        for phi in (0.0, 0.6108652381980153):
            grad = float(jax.grad(loss)(jnp.asarray(700.0), structure, truncation_order, phi))
            h = 1e-3
            fd = (float(loss(jnp.asarray(700.0 + h), structure, truncation_order, phi))
                  - float(loss(jnp.asarray(700.0 - h), structure, truncation_order, phi))) / (2 * h)
            assert np.isfinite(grad)
            assert grad == pytest.approx(fd, rel=1e-4, abs=1e-10)


def test_torch_agrees_with_jax_on_the_gradient_through_the_azimuth():
    """The other traced backend, on the same derivative.

    ``order_azimuth`` builds two nested ``where``\\ s and a ``full_like``, and the three
    frameworks do not agree on what those do to a graph by default. Cheap to check, and the
    only thing that says the clamp works under PyTorch's autograd rather than JAX's.
    """
    torch = pytest.importorskip('torch')
    pytest.importorskip('jax')
    _legacy_wavebit.enable_x64()
    import jax
    import jax.numpy as jnp

    def jax_loss(period):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = _legacy_wavebit.rcwa(backend='jax', wavelength=500.0, period=[period, period], top=1.0,
                             bot=2.25, theta=0.0, phi=0.7, psi=np.pi / 2,
                             structure=jnp.asarray(_UC_2D), thickness=[220.0],
                             truncation_order=[4, 4]).solve()
        return jnp.real(r.R).sum()

    p = torch.tensor(700.0, dtype=torch.float64, requires_grad=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = _legacy_wavebit.rcwa(backend='torch', wavelength=500.0, period=[p, p], top=1.0, bot=2.25,
                         theta=0.0, phi=0.7, psi=np.pi / 2,
                         structure=torch.asarray(_UC_2D), thickness=[220.0],
                         truncation_order=[4, 4]).solve()
        torch.real(r.R).sum().backward()

    assert torch.isfinite(p.grad)
    assert float(p.grad) == pytest.approx(float(jax.grad(jax_loss)(jnp.asarray(700.0))),
                                          rel=1e-9, abs=1e-12)


def test_the_denominator_clamp_is_what_saves_the_gradient():
    """Masking the output alone is not enough, and the forward pass cannot tell you so.

    ``where`` evaluates both branches, so an unclamped ``ky/kx`` still builds a NaN in the
    discarded one. It never shows up in the value — only in ``0 * NaN`` on the way back.
    Both spellings run side by side here so the clamp cannot be "simplified" away by someone
    reading only the forward result.
    """
    jax = pytest.importorskip('jax')
    _legacy_wavebit.enable_x64()
    import jax.numpy as jnp

    ky = jnp.asarray([0.0, 0.2, 0.3], dtype=jnp.complex128)

    def unclamped(kx_real):
        kx = jnp.astype(kx_real, jnp.complex128)
        at_origin = (kx == 0) & (ky == 0)
        v = jnp.atan(ky / kx)
        return jnp.sum(jnp.real(jnp.where(at_origin, jnp.zeros_like(v), v)))

    def clamped(kx_real):
        kx = jnp.astype(kx_real, jnp.complex128)
        return jnp.sum(jnp.real(order_azimuth(array_namespace(kx), ky, kx, 0.0)))

    kx = jnp.asarray([0.0, 0.5, 1.0], dtype=jnp.float64)   # first entry is the origin
    assert float(unclamped(kx)) == pytest.approx(float(clamped(kx))), 'forward agrees...'
    assert not np.isfinite(np.asarray(jax.grad(unclamped)(kx))[0]), '...backward does not'
    assert np.all(np.isfinite(np.asarray(jax.grad(clamped)(kx))))


def test_the_column_carries_its_derivative_along_kx():
    """On the ``kx == 0`` column the azimuth is ``+-pi/2`` -- a constant, which has no
    derivative along ``kx``. ``atan2(ky, kx)`` does: ``-1/ky``, the rate at which the frame of a
    ``(0, n)`` order turns as ``kx`` leaves zero. The column is written ``+-pi/2 - kx/ky``, the
    same value to the bit (the term is a signed zero there), and its gradient is pinned against
    the analytic one, for real and complex ``ky`` of either sign."""
    jax = pytest.importorskip('jax')
    _legacy_wavebit.enable_x64()
    import jax.numpy as jnp

    ky = jnp.asarray([0.4, -0.7, 0.3 + 0.2j, -0.5 + 0.1j], dtype=jnp.complex128)

    def column(kx_real):
        kx = jnp.zeros_like(ky) + jnp.astype(kx_real, jnp.complex128)
        return order_azimuth(array_namespace(kx), ky, kx, 0.6)

    value = column(jnp.asarray(0.0))
    np.testing.assert_array_equal(np.asarray(value).real,
                                  np.where(np.real(np.asarray(ky)) < 0, -np.pi / 2, np.pi / 2))
    grad_re = jax.jacfwd(lambda x: jnp.real(column(x)))(jnp.asarray(0.0))
    grad_im = jax.jacfwd(lambda x: jnp.imag(column(x)))(jnp.asarray(0.0))
    np.testing.assert_allclose(np.asarray(grad_re) + 1j * np.asarray(grad_im),
                               -1 / np.asarray(ky), rtol=1e-15, atol=0)


@pytest.mark.parametrize('formulation', [None, 'first_order'])
def test_the_theta_derivative_at_normal_incidence_on_a_2d_grating(formulation):
    """A 2D grating at exactly ``theta = 0``: every ``(0, n)`` order sits on the ``kx == 0``
    column, and ``jax.grad`` of ``R`` with respect to ``theta`` read the column's frame as
    constant -- ``-1.98e-4`` against the ``5.45e-4`` that one-sided differences on either side
    agree on, for every polarization, while ``theta = 1e-4`` was right. Pinned against the
    central difference, which ``R``'s smoothness through ``theta = 0`` makes the reference."""
    jax = pytest.importorskip('jax')
    _legacy_wavebit.enable_x64()
    import jax.numpy as jnp
    grating = np.ones((6, 6), dtype=complex)
    grating[1:4, 2:5] = 4.0
    grating[4, 0:2] = 2.25

    def reflected(theta, backend):
        s = _legacy_wavebit.rcwa(backend=backend, top=2.25, bot=2.25, wavelength=500.0, theta=theta,
                         phi=0.6, psi=0.3, period=(700.0, 630.0), truncation_order=(1, 1),
                         thickness=[180.0], structure=grating[None])
        if formulation:
            s.formulation = formulation
        return s.solve(auto_detune=False).R.sum()

    h = 1e-6
    fd = (float(np.real(reflected(h, 'numpy'))) - float(np.real(reflected(-h, 'numpy')))) / (2 * h)
    ad = float(jax.grad(lambda t: jnp.real(reflected(t, 'jax')))(jnp.asarray(0.0)))
    assert ad == pytest.approx(fd, rel=1e-6)


# ---------------------------------------------------------------- the ky == 0 row, exactly

#: Every kind of kx the pure 1D row can hold: both signed zeros, a subnormal-sized pair that
#: complex64 flushes to a signed zero, ordinary values either side, the complex-theta corners
#: (a real part of either sign with a loss term, and a purely imaginary kx of either sign),
#: and a value that is zero to rounding but positive.
_ROW = (0.0, -0.0, 1e-300, -1e-300, 0.3, -0.3, -0.3 + 0.2j, -0.3 - 0.2j, 0.2j, -0.2j, 1e-17,
        complex(-0.0, 0.2), complex(-0.0, -0.2))


def _row(backend, dtype):
    """``_ROW`` on a backend, as the complex dtype named -- the arrays the kernels are handed."""
    values = np.asarray(_ROW, dtype=dtype)
    if backend == 'jax':
        import jax.numpy as jnp
        return jnp.asarray(values)
    if backend == 'torch':
        import torch
        return torch.asarray(values)
    return values


def _expected_row(dtype):
    """``-1`` where ``Re(kx) < 0``, from the values as ``dtype`` stores them, else ``+1``."""
    return np.where(np.real(np.asarray(_ROW, dtype=dtype)) < 0, -1.0, 1.0)


@pytest.mark.parametrize('dtype', [np.complex128, np.complex64])
@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_the_1d_frame_sign_is_exactly_the_azimuths_cosine(backend, dtype):
    """``order_frame_sign`` is ``cos(order_azimuth)`` on the ``ky == 0`` row, to the bit.

    The pure 1D path cannot call ``order_azimuth`` -- it would carry ``sin(pi)`` into a
    component that is identically zero there -- so the single-source guarantee is this
    comparison instead. Three claims per entry:

    * the value is exactly ``+-1`` (``==``, not a tolerance), real, in ``Re(kx)``'s precision,
      so a ``complex64`` amplitude multiplied by it stays ``complex64``;
    * ``+1`` at both signed zeros and at a purely imaginary ``kx``: the origin rule and the
      ``Re``-keyed half-turn, which ``sign(kx)`` would get wrong on a complex array;
    * equal to ``cos(order_azimuth(0*kx, kx, phi))`` for every ``phi`` ``tetm`` admits, with
      ``|sin|`` no larger than the ``sin(pi)`` it avoids.
    """
    from wavebit._common.conventions import order_frame_sign
    if backend == 'jax':
        pytest.importorskip('jax')
        _legacy_wavebit.enable_x64()
    elif backend == 'torch':
        pytest.importorskip('torch')
    kx = _row(backend, dtype)
    xp = array_namespace(kx)
    got = order_frame_sign(xp, kx)
    host = np.asarray(got.cpu() if hasattr(got, 'cpu') else got)

    real = np.float32 if dtype == np.complex64 else np.float64
    assert host.dtype == real, f'{backend}: {host.dtype}, not the real dtype of {np.dtype(dtype)}'
    want = _expected_row(dtype)
    assert np.array_equal(host, want), (backend, host, want)
    assert (host[:2] == 1).all() and (host[8:10] == 1).all() and (host[11:] == 1).all(), (
        'a signed zero or a purely imaginary kx flipped')
    # -1e-300 is a negative number in double and a signed zero in single precision.
    assert (want == -1).sum() == (4 if dtype == np.complex128 else 3), 'the row lost a sign'

    for phi in (None, 0.0, -0.0):
        angle = order_azimuth(xp, 0 * kx, kx, phi)
        angle = np.asarray(angle.cpu() if hasattr(angle, 'cpu') else angle)
        assert np.array_equal(np.real(np.cos(angle)), host), (backend, phi)
        assert np.abs(np.sin(angle)).max() <= (2e-16 if dtype == np.complex128 else 1e-7)


@pytest.mark.parametrize('dtype', [np.complex128, np.complex64])
@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_the_1d_frame_sign_takes_its_origin_azimuth(backend, dtype):
    """With ``phi = pi`` -- what the engine hands over at normal incidence from a negative-index
    medium (``CoreSolveMixin._frame_azimuth``) -- the entries at ``kx == 0``, both signed zeros,
    turn to ``-1`` and nothing else moves: still exactly ``cos(order_azimuth(0*kx, kx, pi))``.
    ``phi = 0`` is the default to the bit, and a purely imaginary ``kx`` is not the origin."""
    from wavebit._common.conventions import order_frame_sign
    if backend == 'jax':
        pytest.importorskip('jax')
        _legacy_wavebit.enable_x64()
    elif backend == 'torch':
        pytest.importorskip('torch')
    kx = _row(backend, dtype)
    xp = array_namespace(kx)

    def host(v):
        return np.asarray(v.cpu() if hasattr(v, 'cpu') else v)

    stored = np.asarray(_ROW, dtype=dtype)
    want = np.where((np.real(stored) < 0) | (stored == 0), -1.0, 1.0)
    turned = host(order_frame_sign(xp, kx, np.pi))
    assert turned.dtype == host(order_frame_sign(xp, kx)).dtype
    assert np.array_equal(turned, want), (backend, turned, want)
    assert (turned[:2] == -1).all() and (turned[8:10] == 1).all() and (turned[11:] == 1).all()
    angle = host(order_azimuth(xp, 0 * kx, kx, np.pi))
    assert np.array_equal(np.real(np.cos(angle)), turned), backend
    assert np.array_equal(host(order_frame_sign(xp, kx, 0.0)), host(order_frame_sign(xp, kx)))


def test_the_1d_frame_sign_traces_and_differentiates_to_zero():
    """Traced ``kx`` compiles, and the derivative is a clean zero, including at ``kx == 0``.

    The sign is piecewise constant, so zero is the right derivative everywhere it exists. The
    thing to rule out is a NaN from a traced branch or from a division somewhere upstream --
    which ``where`` on a comparison with constant branches cannot produce.
    """
    jax = pytest.importorskip('jax')
    _legacy_wavebit.enable_x64()
    import jax.numpy as jnp
    from wavebit._common.conventions import order_frame_sign

    kx = jnp.asarray(_ROW, dtype=jnp.complex128)
    eager = np.asarray(order_frame_sign(array_namespace(kx), kx))
    traced = np.asarray(jax.jit(lambda k: order_frame_sign(array_namespace(k), k))(kx))
    assert np.array_equal(eager, traced)

    base = jnp.asarray([-0.4, 0.0, 0.25], dtype=jnp.float64)

    def weighted(shift):
        k = jnp.astype(base + shift, jnp.complex128)
        return jnp.sum(order_frame_sign(array_namespace(k), k) * (base + shift) ** 2)

    grad = float(jax.grad(weighted)(jnp.asarray(0.0)))
    # d/ds sum(sigma * (b + s)**2) = sum(2 * sigma * b), with sigma held constant.
    assert grad == pytest.approx(2 * (-1 * -0.4 + 1 * 0.0 + 1 * 0.25), abs=1e-15)
    assert np.isfinite(grad)
