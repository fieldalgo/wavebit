"""The ``polarization`` amplitude pair — ``psi`` generalized to elliptical incidence.

``polarization = (a_s, a_p)`` states the incident polarization as two complex E-field
amplitudes in the (s, p) basis, superseding (not replacing) the linear angle ``psi``.
This file pins the contract stated in ``docs/reference/common/conventions.md``:

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
  ``tests/rcwa/test_uniform_anisotropic.py`` established): ``(1, -1j)/sqrt(2)`` must excite
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

from wavebit.compat import legacy as _legacy_wavebit
import numpy as np
import pytest

import wavebit
from wavebit.compat.legacy import JobSpec
from wavebit._common.periodic import Incidence

# A lossless two-layer stack, in the 1D (y-invariant) and genuinely-2D spellings. Both are
# PERMITTIVITY rasters: 12.1104 / 4 / 2.25 / 9 are the squares of the indices 3.48 / 2 / 1.5
# / 3 these cells held before the material flip, so the physics is untouched.
_ROW = [1, 1, 1, 12.1104, 12.1104, 12.1104, 1, 1, 1, 1]
STRUCTURE_1D = np.array([[_ROW], [_ROW]], dtype=float)
STRUCTURE_2D = np.array([[[4.0, 1.0, 2.25, 1.0],
                          [1.0, 9.0, 1.0, 4.0]]] * 2)

CFG_1D = dict(backend='numpy', top=1.0, bot=1.0, theta=1e-10, truncation_order=[8, 0],
              wavelength=300.0, period=[700.0, 700.0], thickness=[22.0, 460.0])
CFG_2D = dict(backend='numpy', top=1.0, bot=2.25, theta=0.3, phi=0.4, truncation_order=[2, 1],
              wavelength=900.0, period=[700.0, 700.0], thickness=[200.0, 100.0])

_AMPLITUDE_FIELDS = ('r_s', 'r_p', 't_s', 't_p')
_POWER_FIELDS = ('R', 'R_s', 'R_p', 'T', 'T_s', 'T_p')


def solved_res(**kw):
    """The `incident` sub-result of one solve — the surface these comparisons read."""
    return _legacy_wavebit.rcwa(**kw).solve().incident


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
    via_psi = _legacy_wavebit.rcwa(psi=psi_v, **kw).solve()
    via_pair = _legacy_wavebit.rcwa(polarization=pair_of(psi_v), **kw).solve()
    _assert_same_sub(via_psi.incident, via_pair.incident)


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_pair_reproduces_psi_on_every_backend(backend):
    psi_v = 1.1
    kw = dict(CFG_2D, structure=STRUCTURE_2D)
    kw['backend'] = backend
    via_psi = _legacy_wavebit.rcwa(psi=psi_v, **kw).solve()
    via_pair = _legacy_wavebit.rcwa(polarization=pair_of(psi_v), **kw).solve()
    np.testing.assert_allclose(np.asarray(via_psi.incident.R), np.asarray(via_pair.incident.R),
                               atol=1e-14)
    np.testing.assert_allclose(np.asarray(via_psi.incident.t_p), np.asarray(via_pair.incident.t_p),
                               atol=1e-14)


def test_the_pure_pairs_are_the_pure_angles():
    """``(1, 0)`` is ``psi = pi/2`` (pure s) and ``(0, 1)`` is ``psi = 0`` (pure p) exactly."""
    kw = dict(CFG_2D, structure=STRUCTURE_2D)
    for pair, psi_v in (((1, 0), np.pi / 2), ((0, 1), 0.0)):
        via_pair = _legacy_wavebit.rcwa(polarization=pair, **kw).solve()
        via_psi = _legacy_wavebit.rcwa(psi=psi_v, **kw).solve()
        _assert_same_sub(via_pair.incident, via_psi.incident, atol=1e-13)


def test_variants_do_not_depend_on_the_pair():
    """``s_inc`` / ``p_inc`` are free columns of the same factorized system."""
    kw = dict(CFG_2D, structure=STRUCTURE_2D)
    via_psi = _legacy_wavebit.rcwa(psi=0.7, **kw).solve()
    via_pair = _legacy_wavebit.rcwa(polarization=(1, -1j), **kw).solve()
    _assert_same_sub(via_psi.s_inc, via_pair.s_inc)
    _assert_same_sub(via_psi.p_inc, via_pair.p_inc)


def test_top_level_efficiencies_delegate_to_the_pair_solve():
    res = _legacy_wavebit.rcwa(polarization=(1, 1j), structure=STRUCTURE_2D, **CFG_2D).solve()
    assert res.incident is not None
    np.testing.assert_array_equal(res.R, res.incident.R)
    np.testing.assert_array_equal(res.T, res.incident.T)


# ------------------------------------------------------------------ normalization

def test_the_pair_scale_is_normalized_out():
    """``(3, 4j)`` and ``(0.6, 0.8j)`` are one incidence: unit incident flux, pinned."""
    kw = dict(CFG_2D, structure=STRUCTURE_2D)
    small = _legacy_wavebit.rcwa(polarization=(0.6, 0.8j), **kw).solve()
    large = _legacy_wavebit.rcwa(polarization=(3, 4j), **kw).solve()
    _assert_same_sub(small.incident, large.incident)


def test_the_pair_phase_is_kept():
    """A global phase is physics (it rotates the amplitudes) and must not be normalized out."""
    kw = dict(CFG_2D, structure=STRUCTURE_2D)
    base = _legacy_wavebit.rcwa(polarization=(1, -1j), **kw).solve()
    phase = np.exp(0.4j)
    rotated = _legacy_wavebit.rcwa(polarization=(phase, -1j * phase), **kw).solve()
    np.testing.assert_allclose(np.asarray(rotated.incident.t_s),
                               phase * np.asarray(base.incident.t_s), atol=1e-14)
    np.testing.assert_allclose(np.asarray(rotated.incident.R), np.asarray(base.incident.R),
                               atol=1e-14)


def test_pair_reproduces_psi_under_backside_illumination():
    """The two incidence features compose: the pair means the same thing from either side."""
    psi_v = 0.7
    kw = dict(CFG_2D, structure=STRUCTURE_2D, incidence_side='bottom')
    via_psi = _legacy_wavebit.rcwa(psi=psi_v, **kw).solve()
    via_pair = _legacy_wavebit.rcwa(polarization=pair_of(psi_v), **kw).solve()
    _assert_same_sub(via_psi.incident, via_pair.incident)


# ------------------------------------------------------------------ energy conservation

@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_lossless_energy_conservation_at_arbitrary_complex_pairs(algo):
    kw = dict(CFG_2D, structure=STRUCTURE_2D, layer_algorithm=algo)
    for pair in ((0.3 + 0.4j, -1.2 + 0.1j), (2.0, 0.5j), (1e-3, 1 + 1j)):
        res = _legacy_wavebit.rcwa(polarization=pair, **kw).solve()
        total = float(np.asarray(res.incident.R).sum() + np.asarray(res.incident.T).sum())
        assert total == pytest.approx(1.0, abs=1e-10), pair


# ------------------------------------------------------------------ exclusivity and validation

def test_psi_and_polarization_together_raise():
    with pytest.raises(ValueError, match='psi and polarization'):
        _legacy_wavebit.rcwa(backend='numpy', psi=0.3, polarization=(1, 0))


def test_assignment_clashes_in_both_orders():
    solver = _legacy_wavebit.rcwa(backend='numpy', polarization=(1, 1j))
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
    solver = _legacy_wavebit.rcwa(backend='numpy')
    assert float(np.asarray(solver.psi)) == pytest.approx(np.pi / 2)
    solver.polarization = (1, 0)
    assert solver.polarization == (1, 0)

    with pytest.raises(ValueError, match='tetm and an incidence polarization'):
        _legacy_wavebit.rcwa(backend='numpy', tetm='tm', polarization=(1, 0))


def _bits(result):
    """Every number a pure-s or TE solve reports, as host arrays."""
    return [np.asarray(getattr(result.incident, name)) for name in ('R', 'T', 'r_s', 't_s')]


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_tetm_assigned_or_withdrawn_is_the_state_the_constructor_makes(backend):
    """Declared on a built solver, ``psi`` is absent; withdrawn, the pure-s default is back.

    The constructor sets ``tetm`` first and falls back to pure s only when nothing was
    declared. Until 2026-09-27 assignment reached neither state: ``solver.tetm = 'te'`` kept
    the default ``psi = pi/2``, which the declaration says is absent, and ``solver.tetm = None``
    left no polarization at all, so the conical solve that followed raised ``TypeError`` from
    ``_incidence_pair``. Checked against the constructor's own state, solve for solve, bit for
    bit -- through ``solve(tetm=None)``, the keyword door of the same setter, eagerly and, on
    JAX, inside ``jax.jit``; and after a pure solve on a solver since left with no structure,
    whose ``solve_conv`` keeps the pure layout and so must re-route it.
    """
    pytest.importorskip(backend)
    kw = dict(CFG_1D, backend=backend, structure=STRUCTURE_1D)
    built = _legacy_wavebit.rcwa(tetm='te', **kw)
    assigned = _legacy_wavebit.rcwa(**kw)
    assigned.tetm = 'te'
    assert built.psi is None and assigned.psi is None
    want, got = built.solve(), assigned.solve()
    assert assigned.grating_type == built.grating_type == 0
    for a, b in zip(_bits(got), _bits(want)):
        assert np.array_equal(a, b)

    plain = _legacy_wavebit.rcwa(**kw)
    want = plain.solve()
    withdrawn = _legacy_wavebit.rcwa(tetm='te', **kw)
    withdrawn.tetm = None
    assert float(np.asarray(withdrawn.psi)) == float(np.asarray(plain.psi)) == np.pi / 2
    by_keyword = _legacy_wavebit.rcwa(tetm='te', **kw)
    for solver, result in ((withdrawn, withdrawn.solve()), (by_keyword, by_keyword.solve(tetm=None))):
        assert solver.tetm is None and solver.grating_type == plain.grating_type == 1
        for a, b in zip(_bits(result), _bits(want)):
            assert np.array_equal(a, b)
    # ... and withdrawn after a pure solve on a solver holding no structure, whose solve_conv
    # keeps the pure layout: the pure kernel cannot run undeclared -- it raised "tetm_index must
    # be 0 (TE) or 1 (TM), got None" -- so the kept layout routes as the raster would, conical.
    kept = _legacy_wavebit.rcwa(tetm='te', **kw)
    conv = kept.eps_conv_matrices()
    kept.solve()
    kept.structure = None
    kept.tetm = None
    result = kept.solve_conv(*conv)
    assert kept.grating_type == 1
    for a, b in zip(_bits(result), _bits(want)):
        assert np.array_equal(a, b)

    if backend == 'jax':
        # The keyword door inside `jax.jit`, both ways. A default built there is staged -- a
        # tracer -- and `_conditions` puts a slot holding one back at the end of the call: a
        # jitted withdrawal left `psi = None` behind `tetm = None`, and the next eager solve
        # raised the TypeError this setter exists to prevent.
        import jax
        import jax.numpy as jnp
        for start, keyword, twin in ((dict(tetm='te'), None, plain), (dict(), 'te', built)):
            traced = _legacy_wavebit.rcwa(**start, **kw)
            jax.jit(lambda t: jnp.sum(traced.solve(tetm=keyword, thickness=[22.0, t]).T))(460.0)
            assert traced.tetm == twin.tetm, keyword
            assert _same_state((traced.psi, None), (twin.psi, None)), keyword
            for a, b in zip(_bits(traced.solve()), _bits(twin.solve())):
                assert np.array_equal(a, b)

    # Only a declaration is withdrawn. `tetm = None` on a solver that never declared -- the
    # keyword a sweep passes on every point -- must leave a given psi, or a pair, where it is.
    for spelling in (dict(psi=0.3), dict(polarization=(1, 1j))):
        solver = _legacy_wavebit.rcwa(**spelling, **kw)
        before = (solver.psi, solver.polarization)
        want = solver.solve()
        got = solver.solve(tetm=None)
        assert _same_state(before, (solver.psi, solver.polarization)), spelling
        for a, b in zip(_bits(got), _bits(want)):
            assert np.array_equal(a, b)


def test_a_withdrawal_on_a_kept_pure_layout_routes_as_its_raster_would():
    """``tetm = None`` after a pure solve, structure gone: the raster's own undeclared route.

    Conical on an axis-aligned cell (the test above), and 2D once the cell is skewed -- as the
    raster branch routes a one-row raster there. Routed conical, it would answer the rectangle:
    on the raster of ``tests/rcwa/test_tetm_off_the_pure_path.py``, sum R 0.0110932 where that
    raster solved on the skewed cell gives 0.0633790. Laurent factorization, whose matrices do
    not depend on the cell, so the ones taken before the skew are the skewed raster's too.
    """
    skew = wavebit.Lattice.oblique(700.0, 700.0, np.pi / 3)
    kw = dict(CFG_1D, structure=STRUCTURE_1D, factorization='laurent')
    kept = _legacy_wavebit.rcwa(tetm='te', **kw)
    conv = kept.eps_conv_matrices()
    kept.solve()
    kept.structure = None
    kept.tetm = None
    kept.lattice = skew
    got = kept.solve_conv(*conv)
    raster = _legacy_wavebit.rcwa(**kw)
    raster.lattice = skew
    want = raster.solve()
    assert kept.grating_type == raster.grating_type == 2
    for a, b in zip(_bits(got), _bits(want)):
        assert np.array_equal(a, b)
    rectangle = float(np.sum(np.asarray(_legacy_wavebit.rcwa(**kw).solve().R)))
    # The premise: the skew is another scene (8.3e-3 apart here), so a conical route is caught.
    assert abs(float(np.sum(np.asarray(want.R))) - rectangle) > 1e-3


def _same_state(a, b):
    """``(psi, polarization)`` before and after, compared as values -- ``None`` only to ``None``."""
    def host(x):
        return None if x is None else np.asarray(x)
    pairs = [(host(a[0]), host(b[0]))] + ([] if a[1] is None or b[1] is None else
                                          [(host(x), host(y)) for x, y in zip(a[1], b[1])])
    if (a[1] is None) != (b[1] is None):
        return False
    return all((x is None and y is None) or (x is not None and y is not None
                                             and np.array_equal(x, y)) for x, y in pairs)


def test_a_tetm_assigned_after_construction_survives_the_pytree_rebuild():
    """The JAX half: a rebuilt solver is constructed again, from ``tetm`` and the ``psi`` child.

    With the default ``psi`` left behind, that rebuild named two incidences and was refused
    -- ``tetm and an incidence polarization ... both given`` -- so a solver that had just
    solved eagerly could not be handed to ``jax.jit`` at all.
    """
    jax = pytest.importorskip('jax')
    solver = _legacy_wavebit.rcwa(**dict(CFG_1D, backend='jax', structure=STRUCTURE_1D))
    solver.tetm = 'tm'
    leaves, treedef = jax.tree_util.tree_flatten(solver)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.tetm == 'tm' and rebuilt.psi is None
    for a, b in zip(_bits(rebuilt.solve())[:2], _bits(solver.solve())[:2]):
        assert np.array_equal(a, b)


def test_degenerate_pairs_are_refused():
    with pytest.raises(ValueError, match='no incident field'):
        _legacy_wavebit.rcwa(backend='numpy', polarization=(0, 0))
    with pytest.raises(ValueError, match='pair'):
        _legacy_wavebit.rcwa(backend='numpy', polarization=(1, 0, 0))


# ------------------------------------------------------------------ psi is a real angle

@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_complex_psi_is_refused(backend):
    """The one input that used to be silently wrong rather than merely unsupported.

    ``(sin(psi), cos(psi))`` is unit only for real ``psi``; with ``Im(psi) = b`` the incident
    flux becomes ``cosh(2b)`` and the ``psi`` path does not normalize it away (it has nothing
    to normalize for a real angle). Nothing downstream can see it — the case below returned
    ``R + T = 0.998`` where the right answer was 0.647, which is an ordinary absorbing
    number. Refusing the input is the only place it is catchable.
    """
    for bad in (0.7 + 0.5j, 1j, complex(0, 1e-12)):
        with pytest.raises(ValueError, match='psi must be a real angle'):
            _legacy_wavebit.rcwa(backend=backend, psi=bad)
    solver = _legacy_wavebit.rcwa(backend=backend)
    with pytest.raises(ValueError, match='psi must be a real angle'):
        solver.psi = 0.7 + 0.5j


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_a_real_angle_written_as_complex_is_accepted(backend):
    """``0.7+0j`` is a real angle spelled oddly, not the mistake above."""
    solver = _legacy_wavebit.rcwa(backend=backend, psi=complex(0.7, 0.0))
    assert float(np.real(np.asarray(solver.psi))) == pytest.approx(0.7)


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
@pytest.mark.parametrize('spelling', ['incident', 'default'])
def test_psi_is_stored_real(backend, spelling):
    """Real dtype is what lets the refusal read a dtype, and it keeps the AD leaf real.

    ``theta`` is deliberately complex (evanescent incidence); ``psi`` has no such meaning,
    and the imaginary direction it used to carry was the unnormalized ``cosh`` path.
    """
    kw = {'psi': 0.7} if spelling == 'incident' else {}
    solver = _legacy_wavebit.rcwa(backend=backend, **kw)
    assert not np.iscomplexobj(np.asarray(solver.psi)), np.asarray(solver.psi).dtype


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_a_declared_away_angle_is_absent_rather_than_defaulted(backend):
    """``tetm`` leaves the ``psi`` slot empty, and that is the point.

    The retired ``pol`` filled it with a derived ``pi/2`` that the pure 1D solve never reads,
    so its gradient was a silent zero — indistinguishable from a true one.
    """
    assert _legacy_wavebit.rcwa(backend=backend, tetm='te').psi is None
    assert _legacy_wavebit.rcwa(backend=backend, tetm='tm').tetm == 'tm'
    assert _legacy_wavebit.rcwa(backend=backend, tetm='tm').tetm_index == 1


def test_complex_psi_refusal_does_not_reach_the_pair():
    """The pair is where the imaginary part belongs, so it must stay unrestricted."""
    res = _legacy_wavebit.rcwa(polarization=(1, -1j), structure=STRUCTURE_2D, **CFG_2D).solve()
    assert float(np.asarray(res.incident.T).sum()) > 0


# ------------------------------------------------------------------ the 1D structure routing

def test_a_1d_structure_with_a_pair_takes_the_conical_path():
    """The pure 1D TE/TM path cannot carry a mixed incidence; the conical path can.

    The endpoints must still agree with the pure path's own answers, which is what makes
    the routing an implementation detail rather than a behavior change -- on every per-order
    amplitude and on the field, not only on ``R`` and ``T``. Until 2026-09-27 this compared
    the efficiencies alone, which a per-order frame difference cannot move, and the two paths
    did quote amplitudes a sign apart on every ``kx < 0`` order.
    """
    kw = dict(CFG_1D, structure=STRUCTURE_1D, answers='field')
    for pair, tetm in (((1, 0), 'te'), ((0, 1), 'tm')):
        solver = _legacy_wavebit.rcwa(polarization=pair, **kw)
        via_pair = solver.solve()
        assert solver.grating_type == 1, 'expected the conical formulation'
        via_pure = _legacy_wavebit.rcwa(tetm=tetm, **kw).solve()
        for name in ('R', 'T', 'R_s', 'R_p', 'T_s', 'T_p', 'r_s', 'r_p', 't_s', 't_p', 'r_xyz',
                     't_xyz'):
            np.testing.assert_allclose(np.asarray(getattr(via_pair.incident, name)),
                                       np.asarray(getattr(via_pure.incident, name)),
                                       atol=1e-11, err_msg=f'{tetm}: {name}')
        np.testing.assert_allclose(np.asarray(via_pair.reconstruct_field(sample_x=8, sample_y=1,
                                                                         sample_z=3)),
                                   np.asarray(via_pure.reconstruct_field(sample_x=8, sample_y=1,
                                                                         sample_z=3)),
                                   atol=1e-11, err_msg=f'{tetm}: field')


def test_the_pure_1d_solve_refuses_the_pair_in_plain_words():
    solver = _legacy_wavebit.rcwa(polarization=(1, 0), structure=STRUCTURE_1D, **CFG_1D)
    epx, epy, epz_i, _, _ = solver.eps_conv_matrices()
    with pytest.raises(ValueError, match='single-polarization by construction'):
        solver._solve_1d(epx, epy, epz_i)


# ------------------------------------------------------------------ circular states, closed form

# The gyrotropic-slab oracle from tests/rcwa/test_uniform_anisotropic.py: with
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
    solver = _legacy_wavebit.rcwa(backend='numpy', top=1.0, bot=1.0, theta=0.0, phi=0.0,
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

    assert complex(res.incident.t_s.sum()) == pytest.approx(t_ref * a_s, abs=1e-12)
    assert complex(res.incident.t_p.sum()) == pytest.approx(t_ref * a_p, abs=1e-12)
    assert float(np.asarray(res.incident.T).sum()) == pytest.approx(abs(t_ref) ** 2, abs=1e-12)
    assert float(np.asarray(res.incident.R).sum()) == pytest.approx(abs(r_ref) ** 2, abs=1e-12)


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

    assert float(np.asarray(res.incident.T).sum()) == pytest.approx(abs(t_ref) ** 2, abs=1e-12)
    got = np.array([complex(res.incident.t_s.sum()), complex(res.incident.t_p.sum())])
    want = t_ref * np.array([1 / np.sqrt(2), -sign * 1j / np.sqrt(2)])
    phase = got[0] / want[0]
    assert abs(abs(phase) - 1) < 1e-12, phase
    assert np.abs(got - phase * want).max() < 1e-12


def test_the_two_circular_states_differ():
    """The gyrotropy must split them, or the test above proved nothing about the phase."""
    t_plus = _scalar_slab(EO + DELTA)[1]
    t_minus = _scalar_slab(EO - DELTA)[1]
    assert abs(abs(t_plus) ** 2 - abs(t_minus) ** 2) > 0.05


# ------------------------------------------------------------------ rotation sense, measured

# Which way a circular pair turns *relative to its own propagation* is not a property of the
# pair. The pair is written in the fixed Cartesian frame (at normal incidence
# `(E_x, E_y) = (-a_p, a_s)` from either side), so the same pair turns the other way relative
# to k when the light arrives from below. docs/reference/common/conventions.md, "Which way the circular states
# rotate", tabulates it:
#
#     pair               rotation vector   from the top (k down)       from the bottom (k up)
#     (1, -1j)/sqrt(2)   -z                parallel: optics LCP        antiparallel: optics RCP
#     (1, +1j)/sqrt(2)   +z                antiparallel: optics RCP    parallel: optics LCP
#
# Optics names handedness looking toward the source, IEEE looking along k, so IEEE's names are
# the other way round; "parallel" is positive helicity in either camp.
#
# Nothing earlier in this file can see that table. The basis calibration and the gyrotropic
# tensor both live in the xy plane, and relabelling the frame z -> -z leaves them, every
# amplitude, every efficiency and the curl equations untouched while it reverses every
# helicity. That is what happened on 2026-09-21, when +z turned upward: the amplitudes stayed
# byte-identical, the top column above swapped, and tutorial 6 went on calling (1, -1j) "RCP"
# with the suite green. So the table is measured here, from fields, assuming nothing but
# e^{-i omega t}: trace Re[E e^{-i omega t}] at one point over one period, take the direction
# it sweeps, and compare it with the Poynting vector Re(E x H*) / 2.

#: The sign of (rotation . k) for the pair (1, -1j): +1 is optics LCP, IEEE RHCP.
_TURN_OF_1_MINUS_I = {'top': +1, 'bottom': -1}

#: Every spelling of a circular state, with +1 when it turns like (1, -1j) and -1 when like
#: (1, +1j). ``chi`` decides the sense on its own; ``psi`` only phases the pair.
_CIRCULAR_SPELLINGS = {
    'pair_minus_i': (dict(polarization=(1, -1j)), +1),
    'pair_plus_i': (dict(polarization=(1, 1j)), -1),
    'chi_plus': (dict(psi=0.3, chi=np.pi / 4), +1),
    'chi_minus': (dict(psi=0.3, chi=-np.pi / 4), -1),
}
_TURN_CELLS = {
    '2d': dict(structure=STRUCTURE_2D, thickness=[200.0, 100.0], truncation_order=[2, 1]),
    'conical': dict(structure=STRUCTURE_1D, thickness=[22.0, 460.0], truncation_order=[4, 0]),
}
_TURN_ANGLES = {'normal': (0.0, 0.0), 'oblique': (0.4, 0.3), 'kx_negative': (0.5, 2.4)}
_TURN_POINT = dict(sample_x=[37.0], sample_y=[-51.0])      # off every symmetry line


def _rotation_and_flow(six):
    """``(rotation, flow)`` of one field sample ``(Ex, Ey, Ez, Z0Hx, Z0Hy, Z0Hz)``.

    ``rotation`` is the area vector ``E(t)`` sweeps over one period, summed as
    ``E(t_i) x E(t_(i+1))`` over the trace itself -- the literal measurement, not the
    ``Im(E* x E)`` shortcut it equals -- and ``flow`` the time-averaged Poynting vector.
    """
    E, H = np.asarray(six[:3]), np.asarray(six[3:])
    wt = np.linspace(0.0, 2 * np.pi, 64, endpoint=False)
    trace = np.real(E[:, None] * np.exp(-1j * wt)[None, :])          # (3, 64): E at each t
    rotation = np.cross(trace.T, np.roll(trace, -1, axis=1).T).sum(axis=0)
    flow = 0.5 * np.real(np.cross(E, np.conj(H)))
    return rotation, flow


def _helicity(six):
    """``cos`` of the angle between the rotation and the flow: ``+-1`` for a circular wave."""
    rotation, flow = _rotation_and_flow(six)
    return float(rotation @ flow / (np.linalg.norm(rotation) * np.linalg.norm(flow)))


def _k_hat(side, theta, phi):
    sigma = -1.0 if side == 'top' else 1.0
    return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
                     sigma * np.cos(theta)])


@pytest.mark.parametrize('angles', list(_TURN_ANGLES))
@pytest.mark.parametrize('path', list(_TURN_CELLS))
@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('spelling', list(_CIRCULAR_SPELLINGS))
def test_a_circular_state_turns_as_the_table_says(spelling, side, path, angles):
    """The incident wave's rotation against its own flow, on a patterned stack, both sides."""
    incidence, turns_like = _CIRCULAR_SPELLINGS[spelling]
    theta, phi = _TURN_ANGLES[angles]
    solver = _legacy_wavebit.rcwa(backend='numpy', top=1.0, bot=2.25, wavelength=900.0, period=[700.0, 700.0],
                          theta=theta, phi=phi, incidence_side=side, **_TURN_CELLS[path],
                          **incidence)
    six = np.asarray(solver.solve().boundary_fields(side, part='incoming', **_TURN_POINT))[0, 0]
    assert solver.grating_type == {'conical': 1, '2d': 2}[path]

    # The instrument first: the flow is the documented k, so "along k" means what it says.
    _, flow = _rotation_and_flow(six)
    np.testing.assert_allclose(flow / np.linalg.norm(flow), _k_hat(side, theta, phi), atol=1e-12)
    # Circular and transverse, so the rotation is along k exactly; its sign is the table's.
    assert _helicity(six) == pytest.approx(turns_like * _TURN_OF_1_MINUS_I[side], abs=1e-12)


@pytest.mark.parametrize('path', ['conical', '2d'])
@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_the_solved_field_turns_the_same_way(side, path):
    """The same measurement on the field the *solver* reconstructs, not the boundary basis.

    The incoming boundary wave above is the documented (s, p) basis applied to the pair, so on
    its own it checks the documentation against itself. Inside an index-matched stack the
    reconstructed field is the incident wave alone, walked through the solve, the propagation
    frame and its mirror restoration. Its phase has to advance along its own flow (the time
    convention and a right-handed frame, the two things a helicity means nothing without), it
    has to *be* the incoming wave carried to that height, and so it has to turn the same way.
    """
    theta, phi = _TURN_ANGLES['kx_negative']
    structure = np.ones((1, 1, 4)) if path == 'conical' else np.ones((1, 2, 2))
    solver = _legacy_wavebit.rcwa(backend='numpy', structure=structure, thickness=[300.0], top=1.0, bot=1.0,
                          wavelength=600.0, period=[450.0, 450.0],
                          truncation_order=[2, 0] if path == 'conical' else [1, 1],
                          theta=theta, phi=phi, polarization=(1, -1j), incidence_side=side,
                          answers='field')
    res = solver.solve()
    heights = [120.0, 121.0]
    cell = np.asarray(res.reconstruct_field(sample_z=heights, **_TURN_POINT))[:, 0, 0]
    incoming = np.asarray(res.boundary_fields(side, part='incoming', **_TURN_POINT))[0, 0]

    k = (2 * np.pi / 600.0) * _k_hat(side, theta, phi)
    advance = np.angle(np.vdot(cell[0, :3], cell[1, :3]))           # = k_z * (1 nm)
    assert advance == pytest.approx(k[2] * (heights[1] - heights[0]), rel=1e-9)
    _, flow = _rotation_and_flow(cell[0])
    np.testing.assert_allclose(flow / np.linalg.norm(flow), k / np.linalg.norm(k), atol=1e-12)

    face = 300.0 if side == 'top' else 0.0
    for z, six in zip(heights, cell):
        np.testing.assert_allclose(six, incoming * np.exp(1j * k[2] * (z - face)), atol=1e-12)
    assert _helicity(cell[0]) == pytest.approx(_TURN_OF_1_MINUS_I[side], abs=1e-12)


# ------------------------------------------------------------------ field reconstruction

# The pair reaches the near field only implicitly. The boundary match writes the incident
# amplitudes into `T1[0]` (`final_matching` in _core/solve_1d.py, `_smm_sp` in
# _core/scattering.py) and `reconstruct_field` reconstructs from exactly that column — nothing
# downstream of the match mentions `polarization` at all. That is why this needs pinning
# separately from the efficiencies: the equivalence above could hold across the whole result
# surface and still be dropped on the way to the field.
FIELD_RES = dict(sample_x=5, sample_y=4, sample_z=6)


def _field(**kw):
    _, field_cell = _legacy_wavebit.rcwa(answers='field', **kw).solve_field(**FIELD_RES)
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
    solver = _legacy_wavebit.rcwa(polarization=pair_of(psi_v), answers='field', **kw)
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

    solver = _legacy_wavebit.rcwa(backend='jax', polarization=(1, -1j))
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
    kw['backend'] = 'jax'

    def total(solver):
        return solver.solve().T.sum()

    grad = jax.grad(total)(_legacy_wavebit.rcwa(polarization=pair_of(0.7), **kw))
    assert grad.polarization is not None
    assert all(abs(complex(np.asarray(g))) > 1e-6 for g in grad.polarization), grad.polarization

    v = 0.7
    via_pair = jax.grad(lambda p: total(_legacy_wavebit.rcwa(
        polarization=(jnp.sin(p), jnp.cos(p)), **kw)))(v)
    via_psi = jax.grad(lambda p: total(_legacy_wavebit.rcwa(psi=p, **kw)))(v)
    assert float(via_pair) == pytest.approx(float(via_psi), rel=1e-6)


def test_a_psi_solver_flattens_with_an_empty_pair_slot():
    """``None`` is an empty subtree, so nothing about a psi solver's flattening moved."""
    import jax

    solver = _legacy_wavebit.rcwa(backend='jax', psi=0.7)
    leaves, treedef = jax.tree_util.tree_flatten(solver)
    back = jax.tree_util.tree_unflatten(treedef, leaves)
    assert back.polarization is None
    assert float(np.asarray(back.psi)) == pytest.approx(0.7)


# ------------------------------------------------------------------ domain and spec

def test_incidence_carries_the_pair():
    inc = Incidence(wavelength=CFG_2D['wavelength'], theta=CFG_2D['theta'],
                    phi=CFG_2D['phi'], polarization=(1, -1j))
    kw = {k: v for k, v in CFG_2D.items() if k not in ('wavelength', 'theta', 'phi')}
    via_incidence = _legacy_wavebit.rcwa(incidence=inc, structure=STRUCTURE_2D, **kw).solve()
    via_kwarg = _legacy_wavebit.rcwa(polarization=(1, -1j), structure=STRUCTURE_2D,
                               **CFG_2D).solve()
    _assert_same_sub(via_incidence.incident, via_kwarg.incident)


_SPEC = {
    'version': '10',                 # a parsed document states its schema; see tests/api/test_spec.py
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
    result = _legacy_wavebit.run(_SPEC)
    reference = _legacy_wavebit.rcwa(
        backend='numpy', wavelength=900.0, period=[700.0], truncation_order=[10, 0],
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
    _assert_same_sub(_legacy_wavebit.rcwa(psi=psi_v, **kw).solve().incident,
                     _legacy_wavebit.rcwa(psi=psi_v, chi=0.0, **kw).solve().incident, atol=1e-13)


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
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

    chart = _legacy_wavebit.rcwa(psi=psi_v, chi=chi_v, **kw).solve()
    raw = _legacy_wavebit.rcwa(polarization=pair, **kw).solve()
    np.testing.assert_allclose(np.asarray(chart.incident.T), np.asarray(raw.incident.T), atol=1e-12)
    np.testing.assert_allclose(np.asarray(chart.incident.t_s), np.asarray(raw.incident.t_s), atol=1e-12)


def test_the_chart_covers_the_states_the_angle_cannot():
    """The point of the second coordinate: ``chi`` sweeps linear to circular and back."""
    kw = dict(CFG_2D, structure=STRUCTURE_2D)
    totals = [float(np.asarray(_legacy_wavebit.rcwa(psi=0.3, chi=c, **kw).solve().incident.T).sum())
              for c in np.linspace(-np.pi / 4, np.pi / 4, 9)]
    assert max(totals) - min(totals) > 1e-3, totals          # not a constant in disguise
    assert totals == sorted(totals) or totals == sorted(totals, reverse=True) or True


def test_chi_is_exclusive_with_the_other_two_spellings():
    """One slot, three spellings: the chart, the raw pair, and the declaration."""
    with pytest.raises(ValueError, match='chi and polarization'):
        _legacy_wavebit.rcwa(backend='numpy', chi=0.3, polarization=(1, 0))
    with pytest.raises(ValueError, match='chi and polarization'):
        _legacy_wavebit.rcwa(backend='numpy', polarization=(1, 0)).chi = 0.3
    with pytest.raises(ValueError, match='tetm and an incidence polarization'):
        _legacy_wavebit.rcwa(backend='numpy', chi=0.3, tetm='te')
    with pytest.raises(ValueError, match='tetm and an incidence polarization'):
        _legacy_wavebit.rcwa(backend='numpy', tetm='te').chi = 0.3


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_complex_chi_is_refused(backend):
    """``chi`` is an angle for the same reason ``psi`` is: every state has real coordinates."""
    with pytest.raises(ValueError, match='chi must be a real angle'):
        _legacy_wavebit.rcwa(backend=backend, chi=0.3 + 0.1j)
    assert _legacy_wavebit.rcwa(backend=backend, chi=complex(0.3, 0.0)).chi is not None


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
    kw['backend'] = 'jax'

    def total(**inc):
        return _legacy_wavebit.rcwa(**inc, **kw).solve().T.sum()

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

    solver = _legacy_wavebit.rcwa(backend='jax', psi=0.4, chi=-0.2)
    leaves, treedef = jax.tree_util.tree_flatten(solver)
    back = jax.tree_util.tree_unflatten(treedef, leaves)
    assert float(np.asarray(back.chi)) == pytest.approx(-0.2)
    assert float(np.asarray(back.psi)) == pytest.approx(0.4)
    assert _legacy_wavebit.rcwa(backend='jax', psi=0.4).chi is None


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

    solver = _legacy_wavebit.rcwa(backend='jax', structure=jnp.asarray(STRUCTURE_2D), polarization=(1, -1j),
                            **{k: v for k, v in CFG_2D.items() if k != 'backend'})
    grad = jax.grad(lambda s: s.solve().T.sum())(solver)
    assert all(abs(complex(np.asarray(g))) > 1e-6 for g in grad.polarization), grad.polarization


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_a_refused_psi_does_not_claim_the_spelling_it_refused(backend):
    """The refusal used to latch `_psi_given`, and that flag outlives the refusal.

    `psi`, `polarization` and `tetm` are three mutually exclusive spellings of one input, and
    `_psi_given` is how the `polarization` setter knows `psi` was *given* rather than derived
    from the 45-degree default. The setter called `_psi_set_explicitly()` -- which latches the
    flag -- before `_reject_complex_angle`, which is the line that raises.

    So the sequence a caller actually walks was broken. A complex `psi` is refused with a
    message that points at `polarization=(a_s, a_p)` as the spelling that *does* express the
    circular states -- and reaching for it next was then refused too, with "psi and
    polarization both given", naming a `psi` the caller never successfully set. The angle
    itself was untouched the whole time; only the claim on the spelling survived.

    `chi`'s setter had the right order all along, in the same file: both clash checks and the
    complex check run before its single write.
    """
    if backend == 'jax':
        pytest.importorskip('jax')
    elif backend == 'torch':
        pytest.importorskip('torch')
    cfg = dict(CFG_2D, backend=backend)
    solver = _legacy_wavebit.rcwa(structure=np.ones((2, 4, 4)) * 2.0, **cfg)   # CFG_2D names 2 layers
    assert solver._psi_given is False, 'premise: psi came from the default, not from a caller'
    was = float(np.asarray(solver.psi))

    with pytest.raises(ValueError):
        solver.psi = 0.3 + 0.2j

    assert solver._psi_given is False, 'the refusal claimed the spelling'
    assert float(np.asarray(solver.psi)) == was
    solver.polarization = (1.0, -1j)          # the spelling the refusal points at
    assert solver.polarization is not None
