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

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa
import numpy as np
import pytest

import wavebit
from wavebit.rcwa._engine.enums import Answers, answers_of

RUNGS = ('efficiency', 'amplitude', 'field')


def _structure(n_layers=1):
    u = np.ones((n_layers, 8, 8))
    u[:, 2:6, 2:6] = 16.0                # permittivity: the index 4 squared
    return u


def _solver(backend='numpy', n_layers=1, **kw):
    grid = _structure(n_layers)
    if backend == 'jax':
        import jax.numpy as jnp
        grid = jnp.asarray(grid)
    return _legacy_wavebit_rcwa(backend=backend, truncation_order=(3, 3), period=(300., 300.), wavelength=500.,
                        thickness=(120.,) * n_layers, top=1., bot=2.1025, structure=grid,
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
    assert wavebit.rcwa._engine.enums.DEFAULT_ANSWERS is Answers.AMPLITUDE


# ------------------------------------------------------------- what each rung delivers

@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_each_rung_delivers_exactly_its_own_surface(algo):
    surfaces = {}
    for rung in RUNGS:
        s = _solver(answers=rung, layer_algorithm=algo)
        surfaces[rung] = (s.solve(), s)

    eff, _ = surfaces['efficiency']
    assert eff.R is not None and eff.T is not None
    assert eff.incident.r_s is None, 'the efficiency rung never formed an amplitude'
    assert eff.s_inc is None and eff.p_inc is None

    amp, amp_solver = surfaces['amplitude']
    assert amp.incident.r_s is not None and amp.incident.t_p is not None
    assert amp.s_inc is not None and amp.p_inc is not None
    assert amp_solver.layer_info_list == [] and amp_solver.T1 is None

    fld, fld_solver = surfaces['field']
    assert fld.incident.r_s is not None
    assert len(fld_solver.layer_info_list) == 1 and fld_solver.T1 is not None


#: The efficiency rung against the other two, where the comparison crosses a change in the
#: linear algebra and cannot be exact. Absolute rather than relative because most orders are
#: identically zero and a relative floor says nothing there; efficiencies are O(1), so 1e-12
#: is four orders above the ulp this is absorbing and far below any way of being wrong.
_ACROSS_THE_COLUMN_CHANGE = 1e-12


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_the_rungs_agree_on_what_they_share(algo):
    """A cheaper declaration must be a smaller answer, never a different one.

    Exact where the rungs run the same call, and to a tolerance where they do not, which is
    not a softening of the claim but the shape of it.

    The SMM cascade and shared ETM boundary matching build incidence columns as one,
    or three once the pure-TE and pure-TM variants are wanted, and `answers` decides which. So `'efficiency'`
    asks a one-column question and `'amplitude'` asks a three-column one, of the same matrix.
    Whether a solve with one right-hand side and a solve with three land on the same last bit
    in the column they share is a property of the BLAS -- of gemv against gemm, and of how
    the kernel blocks -- and not of this code. Measured: identical on Accelerate, and 1.11e-16
    apart on some of GitHub's ubuntu runners, 25 times out of 25 on the runner that does it,
    while the same rung repeated against itself agrees 25 out of 25 everywhere. Deterministic
    per machine, which is why it read as a flake across runs that landed on different ones.

    That is the same argument `f150b49` made when it relaxed the four wiring tests -- "a claim
    about the BLAS, not about this code" -- and it left this test exact on the reasoning that
    these rungs do own their bit-identity. They do, but only where the call does not change,
    and it changes at exactly one seam. `'amplitude'` and `'field'` are both three-column, so
    everything compared between those two stays exact and the instrument keeps its edge.
    """
    out = {r: _solver(answers=r, layer_algorithm=algo).solve() for r in RUNGS}
    for field in ('R', 'T'):
        ref = np.asarray(getattr(out['efficiency'], field))
        for rung in ('amplitude', 'field'):
            got = np.asarray(getattr(out[rung], field))
            worst = float(np.abs(got - ref).max())
            assert worst <= _ACROSS_THE_COLUMN_CHANGE, (field, rung, f'{worst:.3e}')
    # Exact from here down: both rungs asked the three-column question.
    for field in ('r_s', 'r_p', 't_s', 't_p', 'R_s', 'R_p', 'T_s', 'T_p'):
        ref = np.asarray(getattr(out['amplitude'].incident, field))
        assert np.array_equal(ref, np.asarray(getattr(out['field'].incident, field))), field
    for variant in ('s_inc', 'p_inc'):
        a = np.asarray(getattr(getattr(out['amplitude'], variant), 'R'))
        f = np.asarray(getattr(getattr(out['field'], variant), 'R'))
        assert np.array_equal(a, f), variant


# ------------------------------------------------------------------------- the refusal

def test_reconstruct_field_is_refused_by_naming_the_declaration():
    s = _solver()
    s.solve()
    with pytest.raises(RuntimeError, match=r"answers='field'") as e:
        s.reconstruct_field(sample_x=4, sample_y=4, sample_z=4)
    assert "answers='amplitude'" in str(e.value), 'the message states what was declared'


def test_the_refusal_beats_require_solved_to_it():
    """Never solved *and* never declared: the declaration is the one to name.

    ``_require_solved``'s "run solve first" is true here but useless — running one on
    this solver keeps nothing, so following that advice reproduces the same failure.
    """
    s = _solver()
    with pytest.raises(RuntimeError, match=r"answers='field'"):
        s.reconstruct_field(sample_x=4, sample_y=4, sample_z=4)


def test_a_declared_field_solve_reconstructs():
    s = _solver(answers='field')
    s.solve()
    assert s.reconstruct_field(sample_x=4, sample_y=4, sample_z=4).shape == (1, 4, 4, 4, 6)


def test_the_declaration_is_assignable_and_takes_effect_on_the_next_solve():
    s = _solver()
    s.solve()
    s.answers = 'field'
    with pytest.raises(RuntimeError):
        # The declaration moved, but the solve that would honour it has not run yet, so the
        # state is still absent -- and this is the case _require_solved is for.
        s.reconstruct_field(sample_x=4, sample_y=4, sample_z=4)
    s.solve()
    assert s.reconstruct_field(sample_x=4, sample_y=4, sample_z=4).shape == (1, 4, 4, 4, 6)


def test_dropping_back_down_clears_the_state_rather_than_leaving_it_stale():
    s = _solver(answers='field')
    s.solve()
    assert s.T1 is not None
    s.answers = 'amplitude'
    s.solve()
    assert s.layer_info_list == [] and s.T1 is None


# ------------------------------------------------------------------------- retention

@pytest.mark.parametrize('algorithm,blocks,vectors', [('ETM', 4.25, 0.05), ('SMM', 2.25, 0.12)])
@pytest.mark.parametrize('n_layers', [1, 3])
def test_the_amplitude_rung_retains_no_matrices_and_the_field_rung_scales_with_depth(
        n_layers, algorithm, blocks, vectors):
    kept = {}
    for rung in ('amplitude', 'field'):
        s = _solver(answers=rung, n_layers=n_layers, layer_algorithm=algorithm)
        s.solve()
        kept[rung] = sum(_owned_bytes([s.layer_info_list, s.T1]).values())
    assert kept['amplitude'] == 0

    # Blocks of side 2*n_orders per layer, `epz_conv_i` counted at a quarter: the ETM keeps `W`,
    # `V`, `big_A_i` and `big_B` whole (4.25); the SMM keeps `W` and `V` and, in place of the two
    # transfer blocks, each layer's `+` and `-` amplitudes (2.25). The tolerance covers `q`,
    # `T1` and those amplitudes, which are vectors: at this size (`dim = 98`, three incidences)
    # 0.04 of a block per layer for the ETM and 0.10 for the SMM, which keeps six of them more.
    dim = 2 * (2 * 3 + 1) ** 2
    block = dim * dim * 16
    assert kept['field'] / (block * n_layers) == pytest.approx(blocks, abs=vectors)


# ------------------------------------------------ the JAX pytree round trip (the trap)

def test_the_declaration_survives_the_pytree_round_trip():
    jax = pytest.importorskip('jax')
    for rung in ('amplitude', 'field'):
        s = _solver(backend='jax', answers=rung)
        leaves, treedef = jax.tree_util.tree_flatten(s)
        assert jax.tree_util.tree_unflatten(treedef, leaves).answers is answers_of(rung)


def test_the_declaration_is_static_so_jit_recompiles_across_it():
    """It must sit in ``aux_data``, not ride as a leaf and not be dropped.

    If it were dropped the two solvers would share a treedef, one compilation would serve
    both, and the rebuilt solver would answer with the constructor default whatever was
    declared — which is how this failed before it was aux data.
    """
    jax = pytest.importorskip('jax')
    _, amplitude = jax.tree_util.tree_flatten(_solver(backend='jax', answers='amplitude'))
    _, field = jax.tree_util.tree_flatten(_solver(backend='jax', answers='field'))
    assert amplitude != field


def test_the_declaration_holds_across_a_jit_boundary():
    jax = pytest.importorskip('jax')

    @jax.jit
    def solved(solver):
        return solver.solve().R

    ref = None
    for rung in ('amplitude', 'field'):
        R = np.asarray(solved(_solver(backend='jax', answers=rung)))
        ref = R if ref is None else ref
        assert np.array_equal(ref, R), rung


# ----------------------------------------------------------------- spec and manifest

def test_the_spec_carries_the_declaration_and_round_trips():
    from wavebit._common._jobs_v1 import Numerics
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
    assert d['results']['answers']['default'] == wavebit.rcwa._engine.enums.DEFAULT_ANSWERS.value
    assert d['results']['answers']['ladder'] == [m.value for m in Answers]
