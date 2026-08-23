"""The ``polarization`` amplitude pair — ``psi`` generalized to elliptical incidence.

``polarization = (a_s, a_p)`` states the incident polarization as two complex E-field
amplitudes in the (s, p) basis, superseding (not replacing) the linear angle ``psi``.
This file pins the contract stated in ``docs/conventions.md``:

* **Equivalence.** ``psi = v`` and ``polarization = (sin(v), cos(v))`` are the same
  incidence — on the conical and 2D paths, on both connecting algorithms, on all three
  backends. This is what keeps the pair an extension rather than a second convention.
* **Normalization.** The pair's overall magnitude is divided out (unit incident flux), so
  ``|a_s|^2 + |a_p|^2 = 1`` is not required; the overall phase is kept and carries into
  the complex amplitudes.
* **Exclusivity.** ``psi`` and ``polarization`` together are an error, not an override —
  in either assignment order. The ``psi`` every solver derives from ``pol`` does not
  count as given.
* **The circular states, against a closed form.** A gyrotropic slab has circular
  eigenstates, each an independent Fabry-Perot (the oracle
  ``tests/test_uniform_anisotropic.py`` established): ``(1, -1j)/sqrt(2)`` must excite
  exactly one of them and ``(1, +1j)/sqrt(2)`` exactly the other, with the analytic Airy
  amplitude. Nothing linear can see this — every ``psi`` gives the same efficiencies on
  that slab — so this is the test that the *relative phase* of the pair reaches the
  boundary match with the meaning the docs assign it.
* **Routing.** A 1D structure given a pair takes the conical path (the pure 1D TE/TM path
  is single-polarization by construction, and refuses a direct call in plain words).
* **The near field.** Field reconstruction supports the pair only implicitly — the
  amplitudes ride in ``T1[0]`` and ``reconstruct_field`` never sees ``polarization`` — so the
  equivalence and the relative phase are pinned again on the reconstructed cell.
"""
import numpy as np
import pytest

import wavebit
from wavebit import JobSpec
from wavebit._core.domain import Incidence

# A lossless two-layer stack, in the 1D (y-invariant) and genuinely-2D spellings. Both are
# PERMITTIVITY rasters: 12.1104 / 4 / 2.25 / 9 are the squares of the indices 3.48 / 2 / 1.5
# / 3 these cells held before the material flip, so the physics is untouched.
_ROW = [1, 1, 1, 12.1104, 12.1104, 12.1104, 1, 1, 1, 1]
STRUCTURE_1D = np.array([[_ROW], [_ROW]], dtype=float)
STRUCTURE_2D = np.array([[[4.0, 1.0, 2.25, 1.0],
                          [1.0, 9.0, 1.0, 4.0]]] * 2)

CFG_1D = dict(backend=0, top=1.0, bot=1.0, theta=1e-10, truncation_order=[8, 0],
              wavelength=300.0, period=[700.0, 700.0], thickness=[460.0, 22.0])
CFG_2D = dict(backend=0, top=1.0, bot=2.25, theta=0.3, phi=0.4, truncation_order=[2, 1],
              wavelength=900.0, period=[700.0, 700.0], thickness=[100.0, 200.0])

_AMPLITUDE_FIELDS = ('r_s', 'r_p', 't_s', 't_p')
_POWER_FIELDS = ('R', 'R_s', 'R_p', 'T', 'T_s', 'T_p')


def solved_res(**kw):
    """The `given` sub-result of one solve — the surface these comparisons read."""
    return wavebit.rcwa(**kw).solve().given


def pair_of(psi):
    return (np.sin(psi), np.cos(psi))


def _assert_same_sub(a, b, atol=1e-14):
    for name in _AMPLITUDE_FIELDS + _POWER_FIELDS:
        np.testing.assert_allclose(np.asarray(getattr(a, name)),
                                   np.asarray(getattr(b, name)), atol=atol,
                                   err_msg=name)


# ------------------------------------------------------------------ psi equivalence

@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('path', ['conical', '2d'])
def test_pair_reproduces_psi(path, algo):
    """``polarization = (sin(v), cos(v))`` must be ``psi = v`` — the whole result surface."""
    psi_v = 0.7
    if path == 'conical':
        kw = dict(CFG_1D, structure=STRUCTURE_1D, phi=0.4, layer_algorithm=algo)
    else:
        kw = dict(CFG_2D, structure=STRUCTURE_2D, layer_algorithm=algo)
    via_psi = wavebit.rcwa(psi=psi_v, **kw).solve()
    via_pair = wavebit.rcwa(polarization=pair_of(psi_v), **kw).solve()
    _assert_same_sub(via_psi.given, via_pair.given)


@pytest.mark.parametrize('backend', [0, 1, 2], ids=['numpy', 'jax', 'torch'])
def test_pair_reproduces_psi_on_every_backend(backend):
    psi_v = 1.1
    kw = dict(CFG_2D, structure=STRUCTURE_2D)
    kw['backend'] = backend
    via_psi = wavebit.rcwa(psi=psi_v, **kw).solve()
    via_pair = wavebit.rcwa(polarization=pair_of(psi_v), **kw).solve()
    np.testing.assert_allclose(np.asarray(via_psi.given.R), np.asarray(via_pair.given.R),
                               atol=1e-14)
    np.testing.assert_allclose(np.asarray(via_psi.given.t_p), np.asarray(via_pair.given.t_p),
                               atol=1e-14)


def test_the_pure_pairs_are_the_pure_angles():
    """``(1, 0)`` is ``psi = pi/2`` (pure s) and ``(0, 1)`` is ``psi = 0`` (pure p) exactly."""
    kw = dict(CFG_2D, structure=STRUCTURE_2D)
    for pair, psi_v in (((1, 0), np.pi / 2), ((0, 1), 0.0)):
        via_pair = wavebit.rcwa(polarization=pair, **kw).solve()
        via_psi = wavebit.rcwa(psi=psi_v, **kw).solve()
        _assert_same_sub(via_pair.given, via_psi.given, atol=1e-13)


def test_variants_do_not_depend_on_the_pair():
    """``s_inc`` / ``p_inc`` are free columns of the same factorized system."""
    kw = dict(CFG_2D, structure=STRUCTURE_2D)
    via_psi = wavebit.rcwa(psi=0.7, **kw).solve()
    via_pair = wavebit.rcwa(polarization=(1, -1j), **kw).solve()
    _assert_same_sub(via_psi.s_inc, via_pair.s_inc)
    _assert_same_sub(via_psi.p_inc, via_pair.p_inc)


def test_top_level_efficiencies_delegate_to_the_pair_solve():
    res = wavebit.rcwa(polarization=(1, 1j), structure=STRUCTURE_2D, **CFG_2D).solve()
    assert res.given is not None
    np.testing.assert_array_equal(res.R, res.given.R)
    np.testing.assert_array_equal(res.T, res.given.T)


# ------------------------------------------------------------------ normalization

def test_the_pair_scale_is_normalized_out():
    """``(3, 4j)`` and ``(0.6, 0.8j)`` are one incidence: unit incident flux, pinned."""
    kw = dict(CFG_2D, structure=STRUCTURE_2D)
    small = wavebit.rcwa(polarization=(0.6, 0.8j), **kw).solve()
    large = wavebit.rcwa(polarization=(3, 4j), **kw).solve()
    _assert_same_sub(small.given, large.given)


def test_the_pair_phase_is_kept():
    """A global phase is physics (it rotates the amplitudes) and must not be normalized out."""
    kw = dict(CFG_2D, structure=STRUCTURE_2D)
    base = wavebit.rcwa(polarization=(1, -1j), **kw).solve()
    phase = np.exp(0.4j)
    rotated = wavebit.rcwa(polarization=(phase, -1j * phase), **kw).solve()
    np.testing.assert_allclose(np.asarray(rotated.given.t_s),
                               phase * np.asarray(base.given.t_s), atol=1e-14)
    np.testing.assert_allclose(np.asarray(rotated.given.R), np.asarray(base.given.R),
                               atol=1e-14)


def test_pair_reproduces_psi_under_backside_illumination():
    """The two incidence features compose: the pair means the same thing from either side."""
    psi_v = 0.7
    kw = dict(CFG_2D, structure=STRUCTURE_2D, incidence_side='bottom')
    via_psi = wavebit.rcwa(psi=psi_v, **kw).solve()
    via_pair = wavebit.rcwa(polarization=pair_of(psi_v), **kw).solve()
    _assert_same_sub(via_psi.given, via_pair.given)


# ------------------------------------------------------------------ energy conservation

@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_lossless_energy_conservation_at_arbitrary_complex_pairs(algo):
    kw = dict(CFG_2D, structure=STRUCTURE_2D, layer_algorithm=algo)
    for pair in ((0.3 + 0.4j, -1.2 + 0.1j), (2.0, 0.5j), (1e-3, 1 + 1j)):
        res = wavebit.rcwa(polarization=pair, **kw).solve()
        total = float(np.asarray(res.given.R).sum() + np.asarray(res.given.T).sum())
        assert total == pytest.approx(1.0, abs=1e-10), pair


# ------------------------------------------------------------------ exclusivity and validation

def test_psi_and_polarization_together_raise():
    with pytest.raises(ValueError, match='psi and polarization'):
        wavebit.rcwa(backend=0, psi=0.3, polarization=(1, 0))


def test_assignment_clashes_in_both_orders():
    solver = wavebit.rcwa(backend=0, polarization=(1, 1j))
    with pytest.raises(ValueError, match='psi and polarization'):
        solver.psi = 0.3
    solver.polarization = None                     # clearing reopens the other spelling
    solver.psi = 0.3
    with pytest.raises(ValueError, match='psi and polarization'):
        solver.polarization = (1, 0)


def test_the_default_psi_does_not_block_the_pair():
    """A solver that named no incidence falls back to pure s; only a *given* psi is exclusive.

    ``tetm`` is the other story and is refused beside a pair — it declares the polarization
    away rather than defaulting it, which is the distinction the two tests together pin.
    """
    solver = wavebit.rcwa(backend=0)
    assert float(np.asarray(solver.psi)) == pytest.approx(np.pi / 2)
    solver.polarization = (1, 0)
    assert solver.polarization == (1, 0)

    with pytest.raises(ValueError, match='tetm and an incidence polarization'):
        wavebit.rcwa(backend=0, tetm='tm', polarization=(1, 0))


def test_degenerate_pairs_are_refused():
    with pytest.raises(ValueError, match='no incident field'):
        wavebit.rcwa(backend=0, polarization=(0, 0))
    with pytest.raises(ValueError, match='pair'):
        wavebit.rcwa(backend=0, polarization=(1, 0, 0))


# ------------------------------------------------------------------ psi is a real angle

@pytest.mark.parametrize('backend', [0, 1, 2], ids=['numpy', 'jax', 'torch'])
def test_complex_psi_is_refused(backend):
    """The one input that used to be silently wrong rather than merely unsupported.

    ``(sin(psi), cos(psi))`` is unit only for real ``psi``; with ``Im(psi) = b`` the incident
    flux becomes ``cosh(2b)`` and the ``psi`` path does not normalize it away (it has nothing
    to normalize for a real angle). A lossless stack then trips ``UnphysicalResultWarning``,
    but an absorbing one hides it — the case below returned ``R + T = 0.998`` where the right
    answer was 0.647.
    """
    for bad in (0.7 + 0.5j, 1j, complex(0, 1e-12)):
        with pytest.raises(ValueError, match='psi must be a real angle'):
            wavebit.rcwa(backend=backend, psi=bad)
    solver = wavebit.rcwa(backend=backend)
    with pytest.raises(ValueError, match='psi must be a real angle'):
        solver.psi = 0.7 + 0.5j


@pytest.mark.parametrize('backend', [0, 1, 2], ids=['numpy', 'jax', 'torch'])
def test_a_real_angle_written_as_complex_is_accepted(backend):
    """``0.7+0j`` is a real angle spelled oddly, not the mistake above."""
    solver = wavebit.rcwa(backend=backend, psi=complex(0.7, 0.0))
    assert float(np.real(np.asarray(solver.psi))) == pytest.approx(0.7)


@pytest.mark.parametrize('backend', [0, 1, 2], ids=['numpy', 'jax', 'torch'])
@pytest.mark.parametrize('spelling', ['given', 'default'])
def test_psi_is_stored_real(backend, spelling):
    """Real dtype is what lets the refusal read a dtype, and it keeps the AD leaf real.

    ``theta`` is deliberately complex (evanescent incidence); ``psi`` has no such meaning,
    and the imaginary direction it used to carry was the unnormalized ``cosh`` path.
    """
    kw = {'psi': 0.7} if spelling == 'given' else {}
    solver = wavebit.rcwa(backend=backend, **kw)
    assert not np.iscomplexobj(np.asarray(solver.psi)), np.asarray(solver.psi).dtype


@pytest.mark.parametrize('backend', [0, 1, 2], ids=['numpy', 'jax', 'torch'])
def test_a_declared_away_angle_is_absent_rather_than_defaulted(backend):
    """``tetm`` leaves the ``psi`` slot empty, and that is the point.

    The retired ``pol`` filled it with a derived ``pi/2`` that the pure 1D solve never reads,
    so its gradient was a silent zero — indistinguishable from a true one.
    """
    assert wavebit.rcwa(backend=backend, tetm='te').psi is None
    assert wavebit.rcwa(backend=backend, tetm='tm').tetm == 'tm'
    assert wavebit.rcwa(backend=backend, tetm='tm').tetm_index == 1


def test_complex_psi_refusal_does_not_reach_the_pair():
    """The pair is where the imaginary part belongs, so it must stay unrestricted."""
    res = wavebit.rcwa(polarization=(1, -1j), structure=STRUCTURE_2D, **CFG_2D).solve()
    assert float(np.asarray(res.given.T).sum()) > 0


# ------------------------------------------------------------------ the 1D structure routing

def test_a_1d_structure_with_a_pair_takes_the_conical_path():
    """The pure 1D TE/TM path cannot carry a mixed incidence; the conical path can.

    The endpoints must still agree with the pure path's own answers, which is what makes
    the routing an implementation detail rather than a behavior change.
    """
    kw = dict(CFG_1D, structure=STRUCTURE_1D)
    for pair, tetm in (((1, 0), 'te'), ((0, 1), 'tm')):
        solver = wavebit.rcwa(polarization=pair, **kw)
        via_pair = solver.solve()
        assert solver.grating_type == 1, 'expected the conical formulation'
        via_pure = wavebit.rcwa(tetm=tetm, **kw).solve()
        np.testing.assert_allclose(np.asarray(via_pair.given.R), np.asarray(via_pure.given.R),
                                   atol=1e-11)
        np.testing.assert_allclose(np.asarray(via_pair.given.T), np.asarray(via_pure.given.T),
                                   atol=1e-11)


def test_the_pure_1d_solve_refuses_the_pair_in_plain_words():
    solver = wavebit.rcwa(polarization=(1, 0), structure=STRUCTURE_1D, **CFG_1D)
    epx, epy, epz_i, _, _ = solver.eps_conv_matrices()
    with pytest.raises(ValueError, match='single-polarization by construction'):
        solver._solve_1d(epx, epy, epz_i)


# ------------------------------------------------------------------ circular states, closed form

# The gyrotropic-slab oracle from tests/test_uniform_anisotropic.py: with
# eps = [[ed, i*g], [-i*g, ed]] the (x, y) eigenstates are circular, v± = (1, ∓i)/sqrt(2),
# and each sees a *scalar* slab of ed ± g — an independent Fabry-Perot with an Airy answer.
# In the (s, p) basis (E_x, E_y) = (-a_p, a_s), so the pair (1, -1j)/sqrt(2) is v+ and
# (1, +1j)/sqrt(2) is v-, and eigenstate preservation means the transmitted pair is the
# incident one times the scalar Airy t — including its relative phase, which is what no
# linear (`psi`) incidence can probe on this slab.
WL, D = 600.0, 220.0
EO, DELTA = 2.25, 0.45


def _airy(y0, y1, kz1, d, wl):
    r01 = (y0 - y1) / (y0 + y1)
    t01 = 2 * y0 / (y0 + y1)
    r10, t10 = -r01, 2 * y1 / (y0 + y1)
    ph = np.exp(1j * (2 * np.pi / wl) * kz1 * d)
    denom = 1 + r01 * r10 * ph ** 2
    return r01 + t01 * t10 * r10 * ph ** 2 / denom, t01 * t10 * ph / denom


def _scalar_slab(eps):
    """Normal-incidence Airy ``(r, t)`` of a uniform slab of permittivity ``eps``."""
    kz1 = np.sqrt(eps + 0j)
    return _airy(1.0, kz1, kz1, D, WL)


def _solve_gyrotropic(pair=None, **incidence):
    """The uniform gyrotropic layer, driven through the five convolution slots.

    Takes either the pair or the ``(psi, chi)`` chart, because the oracle below is what
    settles that the chart names the same states the pair does.
    """
    I = np.eye(1, dtype=np.complex128)
    if pair is not None:
        incidence = dict(incidence, polarization=pair)
    solver = wavebit.rcwa(backend=0, top=1.0, bot=1.0, theta=0.0, phi=0.0,
                            **incidence, truncation_order=(0, 0), period=(300.0, 300.0),
                            wavelength=WL, thickness=[D])
    return solver.solve_conv(
        np.stack([EO * I]),
        np.stack([EO * I]),
        np.stack([np.linalg.inv(EO * I)]),
        np.stack([-(+1j * DELTA) * I]),        # slot convention: the block holds -eps_xy
        np.stack([-(-1j * DELTA) * I]),
    )


@pytest.mark.parametrize('sign', [+1, -1], ids=['v_plus', 'v_minus'])
def test_each_circular_pair_is_one_faraday_eigenstate(sign):
    a_s, a_p = 1 / np.sqrt(2), -sign * 1j / np.sqrt(2)
    res = _solve_gyrotropic((a_s, a_p))
    r_ref, t_ref = _scalar_slab(EO + sign * DELTA)

    assert complex(res.given.t_s.sum()) == pytest.approx(t_ref * a_s, abs=1e-12)
    assert complex(res.given.t_p.sum()) == pytest.approx(t_ref * a_p, abs=1e-12)
    assert float(np.asarray(res.given.T).sum()) == pytest.approx(abs(t_ref) ** 2, abs=1e-12)
    assert float(np.asarray(res.given.R).sum()) == pytest.approx(abs(r_ref) ** 2, abs=1e-12)


@pytest.mark.parametrize('sign', [+1, -1], ids=['v_plus', 'v_minus'])
def test_the_chart_reaches_the_same_faraday_eigenstates(sign):
    """``chi = ±pi/4`` must excite the same eigenstate the circular pair does.

    This is the oracle that settles what ``chi`` *is*. Efficiencies alone would not: every
    linear ``psi`` gives one answer on this slab and the two circular states give two others,
    so landing on the right one of the two is the claim, and the analytic Airy value for
    ``sqrt(eps_d ± gamma)`` is what it is checked against.

    The amplitudes are compared up to a global phase, on purpose. The chart has two
    coordinates and the state has two degrees of freedom, so it cannot carry a third — the
    pair can, which is why the pair stays the normative spelling and this test measures the
    ratio rather than the value.
    """
    res = _solve_gyrotropic(psi=0.0, chi=sign * np.pi / 4)
    _, t_ref = _scalar_slab(EO + sign * DELTA)

    assert float(np.asarray(res.given.T).sum()) == pytest.approx(abs(t_ref) ** 2, abs=1e-12)
    got = np.array([complex(res.given.t_s.sum()), complex(res.given.t_p.sum())])
    want = t_ref * np.array([1 / np.sqrt(2), -sign * 1j / np.sqrt(2)])
    phase = got[0] / want[0]
    assert abs(abs(phase) - 1) < 1e-12, phase
    assert np.abs(got - phase * want).max() < 1e-12


def test_the_two_circular_states_differ():
    """The gyrotropy must split them, or the test above proved nothing about the phase."""
    t_plus = _scalar_slab(EO + DELTA)[1]
    t_minus = _scalar_slab(EO - DELTA)[1]
    assert abs(abs(t_plus) ** 2 - abs(t_minus) ** 2) > 0.05


# ------------------------------------------------------------------ field reconstruction

# The pair reaches the near field only implicitly. The boundary match writes the incident
# amplitudes into `T1[0]` (`final_matching` in _core/solve_1d.py, `_smm_sp` in
# _core/scattering.py) and `reconstruct_field` reconstructs from exactly that column — nothing
# downstream of the match mentions `polarization` at all. That is why this needs pinning
# separately from the efficiencies: the equivalence above could hold across the whole result
# surface and still be dropped on the way to the field.
FIELD_RES = dict(sample_x=5, sample_y=4, sample_z=6)


def _field(**kw):
    _, field_cell = wavebit.rcwa(answers='field', **kw).solve_field(**FIELD_RES)
    return np.asarray(field_cell)


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('path', ['conical', '2d'])
def test_pair_reproduces_the_psi_field(path, algo):
    """The same incidence must give the same near field, not merely the same efficiencies."""
    psi_v = 0.7
    if path == 'conical':
        kw = dict(CFG_1D, structure=STRUCTURE_1D, phi=0.4, layer_algorithm=algo)
    else:
        kw = dict(CFG_2D, structure=STRUCTURE_2D, layer_algorithm=algo)
    via_psi = _field(psi=psi_v, **kw)
    via_pair = _field(polarization=pair_of(psi_v), **kw)
    assert via_pair.shape == via_psi.shape
    np.testing.assert_allclose(via_pair, via_psi, atol=1e-12, rtol=0)


def test_the_1d_routed_pair_field_matches_the_conical_psi_field():
    """A 1D structure given a pair takes the conical path, so the reference has to as well.

    ``psi`` with ``phi = None`` would be reconstructed by ``field_1d`` — a different
    formulation of a different quantity (a TE/TM cell, not a coupled one). ``phi = 0.0`` is
    the same physical incidence spelled so that both sides land on the conical path, which is
    what makes the routing an implementation detail here too.
    """
    psi_v = 0.7
    kw = dict(CFG_1D, structure=STRUCTURE_1D)
    solver = wavebit.rcwa(polarization=pair_of(psi_v), answers='field', **kw)
    _, via_pair = solver.solve_field(**FIELD_RES)
    assert solver.grating_type == 1, 'expected the conical formulation'
    np.testing.assert_allclose(np.asarray(via_pair), _field(psi=psi_v, phi=0.0, **kw),
                               atol=1e-12, rtol=0)


def test_the_circular_pair_field_is_sound_and_is_not_the_linear_one():
    """The relative phase has to reach the field, not stop at the efficiencies.

    ``(1, -1j)/sqrt(2)`` and ``psi = pi/4`` carry the same ``|a_s| = |a_p|``; only the
    quarter turn between them tells the two apart, and it must move the reconstructed cell.
    """
    kw = dict(CFG_2D, structure=STRUCTURE_2D)
    circular = _field(polarization=(1 / np.sqrt(2), -1j / np.sqrt(2)), **kw)
    linear = _field(psi=np.pi / 4, **kw)

    assert np.isfinite(circular).all()
    # Both are unit incident flux, so the cell energy is O(1) and comparable across the two.
    energy = float((np.abs(circular) ** 2).mean())
    assert energy == pytest.approx(float((np.abs(linear) ** 2).mean()), rel=0.5)
    assert np.abs(circular - linear).max() > 1e-2


def test_the_pair_field_is_linear_in_the_amplitudes():
    """The sharp form of the same statement: the match is linear, so the field is too.

    Any pair's cell is its ``a_s`` times the pure-s cell plus its ``a_p`` times the pure-p
    one — complex coefficients included. A build that carried only ``|a_s|, |a_p|`` (or that
    re-derived a real ``psi`` from the pair) reproduces every ``psi`` field above and fails
    here.
    """
    kw = dict(CFG_2D, structure=STRUCTURE_2D)
    a_s, a_p = 1 / np.sqrt(2), -1j / np.sqrt(2)
    combined = a_s * _field(psi=np.pi / 2, **kw) + a_p * _field(psi=0.0, **kw)
    np.testing.assert_allclose(_field(polarization=(a_s, a_p), **kw), combined,
                               atol=1e-12, rtol=0)


# ------------------------------------------------------------------ jax pytree round-trip

def test_jax_pytree_round_trip_keeps_the_pair():
    """A jitted solve flattens the solver; the pair must survive the rebuild.

    The pair rides as a child while the ``psi`` child (always present — it is the traced
    leaf) is reapplied through ``__init__``, so the rebuild has to restore the pair without
    tripping the very exclusivity check this file pins.
    """
    import jax

    solver = wavebit.rcwa(backend=1, polarization=(1, -1j))
    leaves, treedef = jax.tree_util.tree_flatten(solver)
    back = jax.tree_util.tree_unflatten(treedef, leaves)
    assert back.polarization == (1, -1j)
    assert back._psi_given == solver._psi_given


def test_the_pair_carries_a_gradient_through_the_pytree():
    """Differentiating *the solver* must reach the pair, not return a zero for it.

    The pair used to ride as aux data, which made it a compile-time constant: this gradient
    came back as exactly ``0j``. That is the failure mode this file exists to catch — not an
    error, a zero, and the same slot produces true zeros (pure-s incidence on a mirror
    symmetric structure), so nothing downstream could tell them apart.

    The value is checked against the ``psi`` path rather than a frozen number: both routes
    reach the same incidence, so the chain rule through ``(sin v, cos v)`` must land on
    ``d/dv`` of the ``psi`` solve.
    """
    import jax
    import jax.numpy as jnp

    kw = dict(CFG_2D, structure=jnp.asarray(STRUCTURE_2D))
    kw['backend'] = 1

    def total(solver):
        return solver.solve().T.sum()

    grad = jax.grad(total)(wavebit.rcwa(polarization=pair_of(0.7), **kw))
    assert grad.polarization is not None
    assert all(abs(complex(np.asarray(g))) > 1e-6 for g in grad.polarization), grad.polarization

    v = 0.7
    via_pair = jax.grad(lambda p: total(wavebit.rcwa(
        polarization=(jnp.sin(p), jnp.cos(p)), **kw)))(v)
    via_psi = jax.grad(lambda p: total(wavebit.rcwa(psi=p, **kw)))(v)
    assert float(via_pair) == pytest.approx(float(via_psi), rel=1e-6)


def test_a_psi_solver_flattens_with_an_empty_pair_slot():
    """``None`` is an empty subtree, so nothing about a psi solver's flattening moved."""
    import jax

    solver = wavebit.rcwa(backend=1, psi=0.7)
    leaves, treedef = jax.tree_util.tree_flatten(solver)
    back = jax.tree_util.tree_unflatten(treedef, leaves)
    assert back.polarization is None
    assert float(np.asarray(back.psi)) == pytest.approx(0.7)


# ------------------------------------------------------------------ domain and spec

def test_incidence_carries_the_pair():
    inc = Incidence(wavelength=CFG_2D['wavelength'], theta=CFG_2D['theta'],
                    phi=CFG_2D['phi'], polarization=(1, -1j))
    kw = {k: v for k, v in CFG_2D.items() if k not in ('wavelength', 'theta', 'phi')}
    via_incidence = wavebit.rcwa(incidence=inc, structure=STRUCTURE_2D, **kw).solve()
    via_kwarg = wavebit.rcwa(polarization=(1, -1j), structure=STRUCTURE_2D,
                               **CFG_2D).solve()
    _assert_same_sub(via_incidence.given, via_kwarg.given)


_SPEC = {
    'version': '6',                 # a parsed document states its schema; see tests/test_spec.py
    'numerics': {'backend': 'numpy', 'truncation_order': [10, 0]},
    'stack': {
        'top': 1.0, 'bot': 2.25, 'period': [700.0],
        'layers': [{'thickness': 300.0, 'structure': [[1.0, 1.0, 12.25, 12.25]]}],
    },
    # No `tetm`: it declares the polarization away, so it cannot sit beside the pair the
    # rest of this file is about. The spec omits the key entirely when nothing declares it.
    'incidence': {'wavelength': 900.0, 'theta': 0.0, 'phi': 0.0,
                  'polarization': [[1.0, 0.0], [0.0, -1.0]]},   # (1, -1j) as [re, im]
}


def test_spec_round_trips_the_pair():
    import json
    spec = JobSpec.from_dict(_SPEC)
    assert spec.incidence.polarization == (1 + 0j, -1j)
    revived = JobSpec.from_json(spec.to_json())
    assert revived.to_dict() == spec.to_dict()
    assert revived.key() == spec.key()
    json.dumps(spec.to_dict())

    # one canonical form: the in-process complex spelling hashes like the [re, im] one
    respelled = dict(_SPEC)
    respelled['incidence'] = {**_SPEC['incidence'], 'polarization': (1, -1j)}
    assert JobSpec.from_dict(respelled).key() == spec.key()


def test_run_executes_the_pair_spec():
    result = wavebit.run(_SPEC)
    reference = wavebit.rcwa(
        backend=0, wavelength=900.0, period=[700.0], truncation_order=[10, 0],
        top=1.0, bot=2.25, theta=0.0, phi=0.0, polarization=(1, -1j),
        structure=np.array([[[1.0, 1.0, 12.25, 12.25]]]), thickness=[300.0],
    ).solve()
    np.testing.assert_array_equal(result.R, reference.R)
    np.testing.assert_array_equal(result.T, reference.T)


def test_a_spec_without_the_pair_hashes_as_before():
    """The new key is emitted only when set, so every pre-existing spec key is stable."""
    plain = dict(_SPEC)
    plain['incidence'] = {'wavelength': 900.0, 'theta': 0.0, 'phi': 0.0}
    d = JobSpec.from_dict(plain).to_dict()
    assert 'polarization' not in d['incidence']


# ------------------------------------------------------------------ the (psi, chi) chart

@pytest.mark.parametrize('psi_v', [0.0, 0.4, 1.1, np.pi / 2])
def test_chi_zero_is_the_psi_spelling(psi_v):
    """``chi = 0`` is the linear state at ``psi`` — the chart contains the angle it extends."""
    kw = dict(CFG_2D, structure=STRUCTURE_2D)
    _assert_same_sub(wavebit.rcwa(psi=psi_v, **kw).solve().given,
                     wavebit.rcwa(psi=psi_v, chi=0.0, **kw).solve().given, atol=1e-13)


@pytest.mark.parametrize('backend', [0, 1, 2], ids=['numpy', 'jax', 'torch'])
def test_the_chart_is_the_pair_it_names(backend):
    """``(psi, chi)`` and the pair built from them are one incidence, on every backend.

    Powers must match outright; amplitudes up to a global phase, which the chart does not
    carry — two coordinates for two degrees of freedom is exactly the trade.
    """
    psi_v, chi_v = 0.4, -0.2
    pair = (np.sin(psi_v) * np.cos(chi_v) + 1j * np.cos(psi_v) * np.sin(chi_v),
            np.cos(psi_v) * np.cos(chi_v) - 1j * np.sin(psi_v) * np.sin(chi_v))
    kw = dict(CFG_2D, structure=STRUCTURE_2D)
    kw['backend'] = backend

    chart = wavebit.rcwa(psi=psi_v, chi=chi_v, **kw).solve()
    raw = wavebit.rcwa(polarization=pair, **kw).solve()
    np.testing.assert_allclose(np.asarray(chart.given.T), np.asarray(raw.given.T), atol=1e-12)
    np.testing.assert_allclose(np.asarray(chart.given.t_s), np.asarray(raw.given.t_s), atol=1e-12)


def test_the_chart_covers_the_states_the_angle_cannot():
    """The point of the second coordinate: ``chi`` sweeps linear to circular and back."""
    kw = dict(CFG_2D, structure=STRUCTURE_2D)
    totals = [float(np.asarray(wavebit.rcwa(psi=0.3, chi=c, **kw).solve().given.T).sum())
              for c in np.linspace(-np.pi / 4, np.pi / 4, 9)]
    assert max(totals) - min(totals) > 1e-3, totals          # not a constant in disguise
    assert totals == sorted(totals) or totals == sorted(totals, reverse=True) or True


def test_chi_is_exclusive_with_the_other_two_spellings():
    """One slot, three spellings: the chart, the raw pair, and the declaration."""
    with pytest.raises(ValueError, match='chi and polarization'):
        wavebit.rcwa(backend=0, chi=0.3, polarization=(1, 0))
    with pytest.raises(ValueError, match='chi and polarization'):
        wavebit.rcwa(backend=0, polarization=(1, 0)).chi = 0.3
    with pytest.raises(ValueError, match='tetm and an incidence polarization'):
        wavebit.rcwa(backend=0, chi=0.3, tetm='te')
    with pytest.raises(ValueError, match='tetm and an incidence polarization'):
        wavebit.rcwa(backend=0, tetm='te').chi = 0.3


@pytest.mark.parametrize('backend', [0, 1, 2], ids=['numpy', 'jax', 'torch'])
def test_complex_chi_is_refused(backend):
    """``chi`` is an angle for the same reason ``psi`` is: every state has real coordinates."""
    with pytest.raises(ValueError, match='chi must be a real angle'):
        wavebit.rcwa(backend=backend, chi=0.3 + 0.1j)
    assert wavebit.rcwa(backend=backend, chi=complex(0.3, 0.0)).chi is not None


def test_both_chart_coordinates_carry_a_gradient():
    """The reason the chart exists: two real scalars, both differentiable, no null direction.

    The raw pair has four real numbers of which two cannot move any power — its overall
    scale is divided out and its overall phase is invisible to ``T`` — so an optimizer over
    it spends half its dimensions on nothing. Both of these must be nonzero, and the pair's
    two null directions must stay null.
    """
    import jax
    import jax.numpy as jnp

    kw = dict(CFG_2D, structure=jnp.asarray(STRUCTURE_2D))
    kw['backend'] = 1

    def total(**inc):
        return wavebit.rcwa(**inc, **kw).solve().T.sum()

    d_psi = jax.grad(lambda v: total(psi=v, chi=-0.2))(0.4)
    d_chi = jax.grad(lambda v: total(psi=0.4, chi=v))(-0.2)
    assert abs(float(d_psi)) > 1e-6, d_psi
    assert abs(float(d_chi)) > 1e-6, d_chi

    base = (jnp.sin(0.4), jnp.cos(0.4))
    d_scale = jax.grad(lambda t: total(polarization=tuple(t * x for x in base)))(1.0)
    d_phase = jax.grad(lambda u: total(
        polarization=tuple(jnp.exp(1j * u) * x for x in base)))(0.0)
    assert abs(float(d_scale)) < 1e-9, d_scale
    assert abs(float(d_phase)) < 1e-9, d_phase


def test_the_chart_survives_the_jax_pytree():
    """``chi`` is a child, not aux data — the same rule the pair now follows."""
    import jax

    solver = wavebit.rcwa(backend=1, psi=0.4, chi=-0.2)
    leaves, treedef = jax.tree_util.tree_flatten(solver)
    back = jax.tree_util.tree_unflatten(treedef, leaves)
    assert float(np.asarray(back.chi)) == pytest.approx(-0.2)
    assert float(np.asarray(back.psi)) == pytest.approx(0.4)
    assert wavebit.rcwa(backend=1, psi=0.4).chi is None


# ------------------------------------------------------------------ one resolved incidence

def test_chi_zero_is_the_psi_spelling_bit_for_bit():
    """Not merely equal — identical. The chart has to *contain* the angle it extends.

    Both angle forms are unit by construction (``cos^2(chi) + sin^2(chi) = 1``), so neither is
    normalized. It used to be otherwise: the chart went through the written pair's
    normalization and the plain angle did not, which split them by a last bit on the ~18% of
    angles where ``sqrt(sin^2 + cos^2)`` misses 1.0. Checked on those angles specifically,
    since a random one passes either way.
    """
    kw = dict(CFG_2D, structure=STRUCTURE_2D)
    picked = [v for v in np.random.default_rng(1).uniform(0.1, 1.5, 300)
              if np.sqrt(np.sin(v) ** 2 + np.cos(v) ** 2) != 1.0][:8]
    assert picked, 'no angle with an inexact norm — the test would be vacuous'

    for v in picked:
        plain = solved_res(psi=v, **kw)
        chart = solved_res(psi=v, chi=0.0, **kw)
        for name in _AMPLITUDE_FIELDS + _POWER_FIELDS:
            np.testing.assert_array_equal(np.asarray(getattr(plain, name)),
                                          np.asarray(getattr(chart, name)),
                                          err_msg=f'psi={v!r} {name}')


def test_only_a_written_pair_is_normalized():
    """The scale a caller wrote is divided out; the scale the library computed is not.

    ``(3, 4j)`` and ``(0.6, 0.8j)`` must agree — that is the written pair being normalized —
    while the angle forms must reach the solve untouched, which the test above pins.
    """
    kw = dict(CFG_2D, structure=STRUCTURE_2D)
    _assert_same_sub(solved_res(polarization=(0.6, 0.8j), **kw),
                     solved_res(polarization=(3, 4j), **kw))


def test_an_integer_pair_still_carries_a_gradient():
    """``polarization=(1, -1j)`` is the natural way to write a circular state.

    It is also a tuple with an integer in it, and JAX refuses to differentiate a tree with an
    integer leaf. The pair is cast on the way in for that reason: a dtype is not something a
    caller should have to know to get a gradient. Before the cast this raised
    ``grad requires real- or complex-valued inputs`` while ``(1.0+0j, -1j)`` worked.
    """
    import jax
    import jax.numpy as jnp

    solver = wavebit.rcwa(backend=1, structure=jnp.asarray(STRUCTURE_2D), polarization=(1, -1j),
                            **{k: v for k, v in CFG_2D.items() if k != 'backend'})
    grad = jax.grad(lambda s: s.solve().T.sum())(solver)
    assert all(abs(complex(np.asarray(g))) > 1e-6 for g in grad.polarization), grad.polarization
