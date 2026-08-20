"""``answers`` — the construction-time declaration of what a solve is for.

The declaration exists because the field-reconstruction state is three orders of magnitude
larger than the amplitudes and has to be kept *while the solve runs* or not at all. So these
check three separable things:

* the **ladder** is monotone and each rung delivers exactly its own surface, with the rungs
  agreeing bit-for-bit on everything they share (a cheaper declaration must not be a
  different answer);
* the **refusal** names the declaration, and fires ahead of ``fields._require_solved``, whose
  "run a solve first" would be false advice for a solver whose solves do not keep the state;
* the declaration **survives the JAX pytree round trip**. That one is the regression that
  earns this file: a plain instance attribute is dropped by ``_tree_flatten``, the rebuilt
  solver silently takes the constructor default, and — because the treedef is unchanged —
  ``jit`` does not recompile, so the flag reads *inverted* rather than merely stale.
"""
import numpy as np
import pytest

import wavebit
from wavebit._core.enums import Answers, answers_of

RUNGS = ('efficiency', 'amplitude', 'field')


def _ucell(n_layers=1):
    u = np.ones((n_layers, 8, 8))
    u[:, 2:6, 2:6] = 4.0
    return u


def _solver(backend=0, n_layers=1, **kw):
    xp_ucell = _ucell(n_layers)
    if backend == 1:
        import jax.numpy as jnp
        xp_ucell = jnp.asarray(xp_ucell)
    return wavebit.rcwa(backend=backend, fto=(3, 3), period=(300., 300.), wavelength=500.,
                        thickness=(120.,) * n_layers, n_top=1., n_bot=1.45, ucell=xp_ucell,
                        psi=0.3, theta=0.2, phi=0.1, **kw)


def _owned_bytes(obj, seen=None, out=None):
    """Bytes of every array reachable from ``obj``, charged once to its owning buffer."""
    if seen is None:
        seen, out = set(), {}
    if id(obj) in seen:
        return out
    seen.add(id(obj))
    if isinstance(obj, np.ndarray):
        base = obj
        while getattr(base, 'base', None) is not None:
            base = base.base
        out[id(base)] = base.nbytes
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _owned_bytes(v, seen, out)
    return out


# --------------------------------------------------------------------------- the ladder

def test_the_ladder_is_ordered_and_each_rung_covers_the_ones_below():
    assert [m.value for m in Answers] == list(RUNGS), 'the enum order IS the ladder'
    for i, low in enumerate(RUNGS):
        for j, high in enumerate(RUNGS):
            assert answers_of(high).covers(low) == (j >= i), (high, low)


def test_covers_takes_a_string_or_a_member():
    assert Answers.FIELD.covers('amplitude')
    assert Answers.FIELD.covers(Answers.AMPLITUDE)


@pytest.mark.parametrize('bad', ['fields', 'amplitudes', 'full', True, 2, None])
def test_a_value_off_the_ladder_is_refused_by_name(bad):
    with pytest.raises(ValueError, match='answers='):
        _solver(answers=bad)


def test_the_default_is_amplitude():
    assert _solver().answers is Answers.AMPLITUDE
    assert wavebit._core.enums.DEFAULT_ANSWERS is Answers.AMPLITUDE


# ------------------------------------------------------------- what each rung delivers

@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_each_rung_delivers_exactly_its_own_surface(algo):
    surfaces = {}
    for rung in RUNGS:
        s = _solver(answers=rung, connecting_algo=algo)
        surfaces[rung] = (s.conv_solve(), s)

    eff, _ = surfaces['efficiency']
    assert eff.R is not None and eff.T is not None
    assert eff.res.r_s is None, 'the efficiency rung never formed an amplitude'
    assert eff.res_te_inc is None and eff.res_tm_inc is None

    amp, amp_solver = surfaces['amplitude']
    assert amp.res.r_s is not None and amp.res.t_p is not None
    assert amp.res_te_inc is not None and amp.res_tm_inc is not None
    assert amp_solver.layer_info_list == [] and amp_solver.T1 is None

    fld, fld_solver = surfaces['field']
    assert fld.res.r_s is not None
    assert len(fld_solver.layer_info_list) == 1 and fld_solver.T1 is not None


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_the_rungs_agree_bit_for_bit_on_what_they_share(algo):
    """A cheaper declaration must be a smaller answer, never a different one."""
    out = {r: _solver(answers=r, connecting_algo=algo).conv_solve() for r in RUNGS}
    for field in ('R', 'T'):
        ref = np.asarray(getattr(out['efficiency'], field))
        for rung in ('amplitude', 'field'):
            assert np.array_equal(ref, np.asarray(getattr(out[rung], field))), (field, rung)
    for field in ('r_s', 'r_p', 't_s', 't_p', 'R_s', 'R_p', 'T_s', 'T_p'):
        ref = np.asarray(getattr(out['amplitude'].res, field))
        assert np.array_equal(ref, np.asarray(getattr(out['field'].res, field))), field
    for variant in ('res_te_inc', 'res_tm_inc'):
        a = np.asarray(getattr(getattr(out['amplitude'], variant), 'R'))
        f = np.asarray(getattr(getattr(out['field'], variant), 'R'))
        assert np.array_equal(a, f), variant


# ------------------------------------------------------------------------- the refusal

def test_calculate_field_is_refused_by_naming_the_declaration():
    s = _solver()
    s.conv_solve()
    with pytest.raises(RuntimeError, match=r"answers='field'") as e:
        s.calculate_field(res_x=4, res_y=4, res_z=4)
    assert "answers='amplitude'" in str(e.value), 'the message states what was declared'


def test_the_refusal_beats_require_solved_to_it():
    """Never solved *and* never declared: the declaration is the one to name.

    ``_require_solved``'s "run conv_solve first" is true here but useless — running one on
    this solver keeps nothing, so following that advice reproduces the same failure.
    """
    s = _solver()
    with pytest.raises(RuntimeError, match=r"answers='field'"):
        s.calculate_field(res_x=4, res_y=4, res_z=4)


def test_a_declared_field_solve_reconstructs():
    s = _solver(answers='field')
    s.conv_solve()
    assert s.calculate_field(res_x=4, res_y=4, res_z=4).shape == (1, 4, 4, 4, 6)


def test_the_declaration_is_assignable_and_takes_effect_on_the_next_solve():
    s = _solver()
    s.conv_solve()
    s.answers = 'field'
    with pytest.raises(RuntimeError):
        # The declaration moved, but the solve that would honour it has not run yet, so the
        # state is still absent -- and this is the case _require_solved is for.
        s.calculate_field(res_x=4, res_y=4, res_z=4)
    s.conv_solve()
    assert s.calculate_field(res_x=4, res_y=4, res_z=4).shape == (1, 4, 4, 4, 6)


def test_dropping_back_down_clears_the_state_rather_than_leaving_it_stale():
    s = _solver(answers='field')
    s.conv_solve()
    assert s.T1 is not None
    s.answers = 'amplitude'
    s.conv_solve()
    assert s.layer_info_list == [] and s.T1 is None


# ------------------------------------------------------------------------- retention

@pytest.mark.parametrize('n_layers', [1, 3])
def test_the_amplitude_rung_retains_no_matrices_and_the_field_rung_scales_with_depth(n_layers):
    kept = {}
    for rung in ('amplitude', 'field'):
        s = _solver(answers=rung, n_layers=n_layers)
        s.conv_solve()
        kept[rung] = sum(_owned_bytes([s.layer_info_list, s.T1]).values())
    assert kept['amplitude'] == 0

    # 4.25 blocks of side 2*ff_xy per layer: W, V, big_A_i, big_B whole and epz_conv_i at a
    # quarter. The tolerance covers `q` and `T1`, which are vectors.
    dim = 2 * (2 * 3 + 1) ** 2
    block = dim * dim * 16
    assert kept['field'] / (block * n_layers) == pytest.approx(4.25, abs=0.05)


# ------------------------------------------------ the JAX pytree round trip (the trap)

def test_the_declaration_survives_the_pytree_round_trip():
    jax = pytest.importorskip('jax')
    for rung in ('amplitude', 'field'):
        s = _solver(backend=1, answers=rung)
        leaves, treedef = jax.tree_util.tree_flatten(s)
        assert jax.tree_util.tree_unflatten(treedef, leaves).answers is answers_of(rung)


def test_the_declaration_is_static_so_jit_recompiles_across_it():
    """It must sit in ``aux_data``, not ride as a leaf and not be dropped.

    If it were dropped the two solvers would share a treedef, one compilation would serve
    both, and the rebuilt solver would answer with the constructor default whatever was
    declared — which is how this failed before it was aux data.
    """
    jax = pytest.importorskip('jax')
    _, amplitude = jax.tree_util.tree_flatten(_solver(backend=1, answers='amplitude'))
    _, field = jax.tree_util.tree_flatten(_solver(backend=1, answers='field'))
    assert amplitude != field


def test_the_declaration_holds_across_a_jit_boundary():
    jax = pytest.importorskip('jax')

    @jax.jit
    def solved(solver):
        return solver.conv_solve().R

    ref = None
    for rung in ('amplitude', 'field'):
        R = np.asarray(solved(_solver(backend=1, answers=rung)))
        ref = R if ref is None else ref
        assert np.array_equal(ref, R), rung


# ----------------------------------------------------------------- spec and manifest

def test_the_spec_carries_the_declaration_and_round_trips():
    from wavebit._core.spec import Numerics
    n = Numerics.from_dict({'answers': 'field'})
    assert n.answers is Answers.FIELD
    assert n.as_kwargs()['answers'] == 'field'
    assert n.to_dict()['answers'] == 'field'
    assert Numerics.from_dict(n.to_dict()).answers is Answers.FIELD
    assert Numerics().as_kwargs()['answers'] == 'amplitude'


def test_the_manifest_states_the_ladder_the_code_implements():
    from wavebit.manifest import manifest
    d = manifest()
    assert d['enums']['answers'] == [m.value for m in Answers]
    assert d['results']['answers']['default'] == wavebit._core.enums.DEFAULT_ANSWERS.value
    assert d['results']['answers']['ladder'] == [m.value for m in Answers]
