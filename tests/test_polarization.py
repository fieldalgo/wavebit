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
  amplitudes ride in ``T1[0]`` and ``calculate_field`` never sees ``polarization`` — so the
  equivalence and the relative phase are pinned again on the reconstructed cell.
"""
import numpy as np
import pytest

import fieldwave
from fieldwave import JobSpec
from fieldwave._core.domain import Incidence

# A lossless two-layer stack, in the 1D (y-invariant) and genuinely-2D spellings.
_ROW = [1, 1, 1, 3.48, 3.48, 3.48, 1, 1, 1, 1]
UCELL_1D = np.array([[_ROW], [_ROW]], dtype=float)
UCELL_2D = np.array([[[2.0, 1.0, 1.5, 1.0],
                      [1.0, 3.0, 1.0, 2.0]]] * 2)

CFG_1D = dict(backend=0, n_top=1.0, n_bot=1.0, theta=1e-10, fto=[8, 0],
              wavelength=300.0, period=[700.0, 700.0], thickness=[460.0, 22.0])
CFG_2D = dict(backend=0, n_top=1.0, n_bot=1.5, theta=0.3, phi=0.4, fto=[2, 1],
              wavelength=900.0, period=[700.0, 700.0], thickness=[100.0, 200.0])

_AMPLITUDE_FIELDS = ('r_s', 'r_p', 't_s', 't_p')
_POWER_FIELDS = ('R', 'R_s', 'R_p', 'T', 'T_s', 'T_p')


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
        kw = dict(CFG_1D, ucell=UCELL_1D, phi=0.4, connecting_algo=algo)
    else:
        kw = dict(CFG_2D, ucell=UCELL_2D, connecting_algo=algo)
    via_psi = fieldwave.rcwa(psi=psi_v, **kw).conv_solve()
    via_pair = fieldwave.rcwa(polarization=pair_of(psi_v), **kw).conv_solve()
    _assert_same_sub(via_psi.res, via_pair.res)


@pytest.mark.parametrize('backend', [0, 1, 2], ids=['numpy', 'jax', 'torch'])
def test_pair_reproduces_psi_on_every_backend(backend):
    psi_v = 1.1
    kw = dict(CFG_2D, ucell=UCELL_2D)
    kw['backend'] = backend
    via_psi = fieldwave.rcwa(psi=psi_v, **kw).conv_solve()
    via_pair = fieldwave.rcwa(polarization=pair_of(psi_v), **kw).conv_solve()
    np.testing.assert_allclose(np.asarray(via_psi.res.R), np.asarray(via_pair.res.R),
                               atol=1e-14)
    np.testing.assert_allclose(np.asarray(via_psi.res.t_p), np.asarray(via_pair.res.t_p),
                               atol=1e-14)


def test_the_pure_pairs_are_the_pure_angles():
    """``(1, 0)`` is ``psi = pi/2`` (pure s) and ``(0, 1)`` is ``psi = 0`` (pure p) exactly."""
    kw = dict(CFG_2D, ucell=UCELL_2D)
    for pair, psi_v in (((1, 0), np.pi / 2), ((0, 1), 0.0)):
        via_pair = fieldwave.rcwa(polarization=pair, **kw).conv_solve()
        via_psi = fieldwave.rcwa(psi=psi_v, **kw).conv_solve()
        _assert_same_sub(via_pair.res, via_psi.res, atol=1e-13)


def test_variants_do_not_depend_on_the_pair():
    """``res_te_inc`` / ``res_tm_inc`` are free columns of the same factorized system."""
    kw = dict(CFG_2D, ucell=UCELL_2D)
    via_psi = fieldwave.rcwa(psi=0.7, **kw).conv_solve()
    via_pair = fieldwave.rcwa(polarization=(1, -1j), **kw).conv_solve()
    _assert_same_sub(via_psi.res_te_inc, via_pair.res_te_inc)
    _assert_same_sub(via_psi.res_tm_inc, via_pair.res_tm_inc)


def test_top_level_efficiencies_delegate_to_the_pair_solve():
    res = fieldwave.rcwa(polarization=(1, 1j), ucell=UCELL_2D, **CFG_2D).conv_solve()
    assert res.res is not None
    np.testing.assert_array_equal(res.R, res.res.R)
    np.testing.assert_array_equal(res.T, res.res.T)


# ------------------------------------------------------------------ normalization

def test_the_pair_scale_is_normalized_out():
    """``(3, 4j)`` and ``(0.6, 0.8j)`` are one incidence: unit incident flux, pinned."""
    kw = dict(CFG_2D, ucell=UCELL_2D)
    small = fieldwave.rcwa(polarization=(0.6, 0.8j), **kw).conv_solve()
    large = fieldwave.rcwa(polarization=(3, 4j), **kw).conv_solve()
    _assert_same_sub(small.res, large.res)


def test_the_pair_phase_is_kept():
    """A global phase is physics (it rotates the amplitudes) and must not be normalized out."""
    kw = dict(CFG_2D, ucell=UCELL_2D)
    base = fieldwave.rcwa(polarization=(1, -1j), **kw).conv_solve()
    phase = np.exp(0.4j)
    rotated = fieldwave.rcwa(polarization=(phase, -1j * phase), **kw).conv_solve()
    np.testing.assert_allclose(np.asarray(rotated.res.t_s),
                               phase * np.asarray(base.res.t_s), atol=1e-14)
    np.testing.assert_allclose(np.asarray(rotated.res.R), np.asarray(base.res.R),
                               atol=1e-14)


def test_pair_reproduces_psi_under_backside_illumination():
    """The two incidence features compose: the pair means the same thing from either side."""
    psi_v = 0.7
    kw = dict(CFG_2D, ucell=UCELL_2D, incidence_side='bottom')
    via_psi = fieldwave.rcwa(psi=psi_v, **kw).conv_solve()
    via_pair = fieldwave.rcwa(polarization=pair_of(psi_v), **kw).conv_solve()
    _assert_same_sub(via_psi.res, via_pair.res)


# ------------------------------------------------------------------ energy conservation

@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_lossless_energy_conservation_at_arbitrary_complex_pairs(algo):
    kw = dict(CFG_2D, ucell=UCELL_2D, connecting_algo=algo)
    for pair in ((0.3 + 0.4j, -1.2 + 0.1j), (2.0, 0.5j), (1e-3, 1 + 1j)):
        res = fieldwave.rcwa(polarization=pair, **kw).conv_solve()
        total = float(np.asarray(res.res.R).sum() + np.asarray(res.res.T).sum())
        assert total == pytest.approx(1.0, abs=1e-10), pair


# ------------------------------------------------------------------ exclusivity and validation

def test_psi_and_polarization_together_raise():
    with pytest.raises(ValueError, match='psi and polarization'):
        fieldwave.rcwa(backend=0, psi=0.3, polarization=(1, 0))


def test_assignment_clashes_in_both_orders():
    solver = fieldwave.rcwa(backend=0, polarization=(1, 1j))
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
    solver = fieldwave.rcwa(backend=0)
    assert float(np.asarray(solver.psi)) == pytest.approx(np.pi / 2)
    solver.polarization = (1, 0)
    assert solver.polarization == (1, 0)

    with pytest.raises(ValueError, match='tetm and an incidence polarization'):
        fieldwave.rcwa(backend=0, tetm='tm', polarization=(1, 0))


def test_degenerate_pairs_are_refused():
    with pytest.raises(ValueError, match='no incident field'):
        fieldwave.rcwa(backend=0, polarization=(0, 0))
    with pytest.raises(ValueError, match='pair'):
        fieldwave.rcwa(backend=0, polarization=(1, 0, 0))


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
            fieldwave.rcwa(backend=backend, psi=bad)
    solver = fieldwave.rcwa(backend=backend)
    with pytest.raises(ValueError, match='psi must be a real angle'):
        solver.psi = 0.7 + 0.5j


@pytest.mark.parametrize('backend', [0, 1, 2], ids=['numpy', 'jax', 'torch'])
def test_a_real_angle_written_as_complex_is_accepted(backend):
    """``0.7+0j`` is a real angle spelled oddly, not the mistake above."""
    solver = fieldwave.rcwa(backend=backend, psi=complex(0.7, 0.0))
    assert float(np.real(np.asarray(solver.psi))) == pytest.approx(0.7)


@pytest.mark.parametrize('backend', [0, 1, 2], ids=['numpy', 'jax', 'torch'])
@pytest.mark.parametrize('spelling', ['given', 'default'])
def test_psi_is_stored_real(backend, spelling):
    """Real dtype is what lets the refusal read a dtype, and it keeps the AD leaf real.

    ``theta`` is deliberately complex (evanescent incidence); ``psi`` has no such meaning,
    and the imaginary direction it used to carry was the unnormalized ``cosh`` path.
    """
    kw = {'psi': 0.7} if spelling == 'given' else {}
    solver = fieldwave.rcwa(backend=backend, **kw)
    assert not np.iscomplexobj(np.asarray(solver.psi)), np.asarray(solver.psi).dtype


@pytest.mark.parametrize('backend', [0, 1, 2], ids=['numpy', 'jax', 'torch'])
def test_a_declared_away_angle_is_absent_rather_than_defaulted(backend):
    """``tetm`` leaves the ``psi`` slot empty, and that is the point.

    The retired ``pol`` filled it with a derived ``pi/2`` that the pure 1D solve never reads,
    so its gradient was a silent zero — indistinguishable from a true one.
    """
    assert fieldwave.rcwa(backend=backend, tetm='te').psi is None
    assert fieldwave.rcwa(backend=backend, tetm='tm').tetm == 'tm'
    assert fieldwave.rcwa(backend=backend, tetm='tm').tetm_index == 1


def test_complex_psi_refusal_does_not_reach_the_pair():
    """The pair is where the imaginary part belongs, so it must stay unrestricted."""
    res = fieldwave.rcwa(polarization=(1, -1j), ucell=UCELL_2D, **CFG_2D).conv_solve()
    assert float(np.asarray(res.res.T).sum()) > 0


# ------------------------------------------------------------------ the 1D structure routing

def test_a_1d_structure_with_a_pair_takes_the_conical_path():
    """The pure 1D TE/TM path cannot carry a mixed incidence; the conical path can.

    The endpoints must still agree with the pure path's own answers, which is what makes
    the routing an implementation detail rather than a behavior change.
    """
    kw = dict(CFG_1D, ucell=UCELL_1D)
    for pair, tetm in (((1, 0), 'te'), ((0, 1), 'tm')):
        solver = fieldwave.rcwa(polarization=pair, **kw)
        via_pair = solver.conv_solve()
        assert solver.grating_type_assigned == 1, 'expected the conical formulation'
        via_pure = fieldwave.rcwa(tetm=tetm, **kw).conv_solve()
        np.testing.assert_allclose(np.asarray(via_pair.res.R), np.asarray(via_pure.res.R),
                                   atol=1e-11)
        np.testing.assert_allclose(np.asarray(via_pair.res.T), np.asarray(via_pure.res.T),
                                   atol=1e-11)


def test_the_pure_1d_solve_refuses_the_pair_in_plain_words():
    solver = fieldwave.rcwa(polarization=(1, 0), ucell=UCELL_1D, **CFG_1D)
    epx, epy, epz_i, _, _ = solver.conv_matrices()
    with pytest.raises(ValueError, match='single-polarization by construction'):
        solver.solve_1d(CFG_1D['wavelength'], epx, epy, epz_i)


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


def _solve_gyrotropic(pair):
    """The uniform gyrotropic layer, driven through the five convolution slots."""
    I = np.eye(1, dtype=np.complex128)
    solver = fieldwave.rcwa(backend=0, n_top=1.0, n_bot=1.0, theta=0.0, phi=0.0,
                            polarization=pair, fto=(0, 0), period=(300.0, 300.0),
                            wavelength=WL, thickness=[D])
    return solver.solve_for_conv(
        WL,
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

    assert complex(res.res.t_s.sum()) == pytest.approx(t_ref * a_s, abs=1e-12)
    assert complex(res.res.t_p.sum()) == pytest.approx(t_ref * a_p, abs=1e-12)
    assert float(np.asarray(res.res.T).sum()) == pytest.approx(abs(t_ref) ** 2, abs=1e-12)
    assert float(np.asarray(res.res.R).sum()) == pytest.approx(abs(r_ref) ** 2, abs=1e-12)


def test_the_two_circular_states_differ():
    """The gyrotropy must split them, or the test above proved nothing about the phase."""
    t_plus = _scalar_slab(EO + DELTA)[1]
    t_minus = _scalar_slab(EO - DELTA)[1]
    assert abs(abs(t_plus) ** 2 - abs(t_minus) ** 2) > 0.05


# ------------------------------------------------------------------ field reconstruction

# The pair reaches the near field only implicitly. The boundary match writes the incident
# amplitudes into `T1[0]` (`final_matching` in _core/solve_1d.py, `_smm_sp` in
# _core/scattering.py) and `calculate_field` reconstructs from exactly that column — nothing
# downstream of the match mentions `polarization` at all. That is why this needs pinning
# separately from the efficiencies: the equivalence above could hold across the whole result
# surface and still be dropped on the way to the field.
FIELD_RES = dict(res_x=5, res_y=4, res_z=6)


def _field(**kw):
    _, field_cell = fieldwave.rcwa(**kw).conv_solve_field(**FIELD_RES)
    return np.asarray(field_cell)


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('path', ['conical', '2d'])
def test_pair_reproduces_the_psi_field(path, algo):
    """The same incidence must give the same near field, not merely the same efficiencies."""
    psi_v = 0.7
    if path == 'conical':
        kw = dict(CFG_1D, ucell=UCELL_1D, phi=0.4, connecting_algo=algo)
    else:
        kw = dict(CFG_2D, ucell=UCELL_2D, connecting_algo=algo)
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
    kw = dict(CFG_1D, ucell=UCELL_1D)
    solver = fieldwave.rcwa(polarization=pair_of(psi_v), **kw)
    _, via_pair = solver.conv_solve_field(**FIELD_RES)
    assert solver.grating_type_assigned == 1, 'expected the conical formulation'
    np.testing.assert_allclose(np.asarray(via_pair), _field(psi=psi_v, phi=0.0, **kw),
                               atol=1e-12, rtol=0)


def test_the_circular_pair_field_is_sound_and_is_not_the_linear_one():
    """The relative phase has to reach the field, not stop at the efficiencies.

    ``(1, -1j)/sqrt(2)`` and ``psi = pi/4`` carry the same ``|a_s| = |a_p|``; only the
    quarter turn between them tells the two apart, and it must move the reconstructed cell.
    """
    kw = dict(CFG_2D, ucell=UCELL_2D)
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
    kw = dict(CFG_2D, ucell=UCELL_2D)
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

    solver = fieldwave.rcwa(backend=1, polarization=(1, -1j))
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

    kw = dict(CFG_2D, ucell=jnp.asarray(UCELL_2D))
    kw['backend'] = 1

    def total(solver):
        return solver.conv_solve().T.sum()

    grad = jax.grad(total)(fieldwave.rcwa(polarization=pair_of(0.7), **kw))
    assert grad.polarization is not None
    assert all(abs(complex(np.asarray(g))) > 1e-6 for g in grad.polarization), grad.polarization

    v = 0.7
    via_pair = jax.grad(lambda p: total(fieldwave.rcwa(
        polarization=(jnp.sin(p), jnp.cos(p)), **kw)))(v)
    via_psi = jax.grad(lambda p: total(fieldwave.rcwa(psi=p, **kw)))(v)
    assert float(via_pair) == pytest.approx(float(via_psi), rel=1e-6)


def test_a_psi_solver_flattens_with_an_empty_pair_slot():
    """``None`` is an empty subtree, so nothing about a psi solver's flattening moved."""
    import jax

    solver = fieldwave.rcwa(backend=1, psi=0.7)
    leaves, treedef = jax.tree_util.tree_flatten(solver)
    back = jax.tree_util.tree_unflatten(treedef, leaves)
    assert back.polarization is None
    assert float(np.asarray(back.psi)) == pytest.approx(0.7)


# ------------------------------------------------------------------ domain and spec

def test_incidence_carries_the_pair():
    inc = Incidence(wavelength=CFG_2D['wavelength'], theta=CFG_2D['theta'],
                    phi=CFG_2D['phi'], polarization=(1, -1j))
    kw = {k: v for k, v in CFG_2D.items() if k not in ('wavelength', 'theta', 'phi')}
    via_incidence = fieldwave.rcwa(incidence=inc, ucell=UCELL_2D, **kw).conv_solve()
    via_kwarg = fieldwave.rcwa(polarization=(1, -1j), ucell=UCELL_2D,
                               **CFG_2D).conv_solve()
    _assert_same_sub(via_incidence.res, via_kwarg.res)


_SPEC = {
    'numerics': {'backend': 'numpy', 'fto': [10, 0]},
    'stack': {
        'n_top': 1.0, 'n_bot': 1.5, 'period': [700.0],
        'layers': [{'thickness': 300.0, 'eps': [[1.0, 1.0, 3.5, 3.5]]}],
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
    result = fieldwave.run(_SPEC)
    reference = fieldwave.rcwa(
        backend=0, wavelength=900.0, period=[700.0], fto=[10, 0],
        n_top=1.0, n_bot=1.5, theta=0.0, phi=0.0, polarization=(1, -1j),
        ucell=np.array([[[1.0, 1.0, 3.5, 3.5]]]), thickness=[300.0],
    ).conv_solve()
    np.testing.assert_array_equal(result.R, reference.R)
    np.testing.assert_array_equal(result.T, reference.T)


def test_a_spec_without_the_pair_hashes_as_before():
    """The new key is emitted only when set, so every pre-existing spec key is stable."""
    plain = dict(_SPEC)
    plain['incidence'] = {'wavelength': 900.0, 'theta': 0.0, 'phi': 0.0}
    d = JobSpec.from_dict(plain).to_dict()
    assert 'polarization' not in d['incidence']
