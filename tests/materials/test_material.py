"""The ``Material`` type itself: the shorthand, the pytree, and the two conversions.

``Material(epsilon, mu=None)`` is what the two *assignment* slots (the parametric layer's
``background`` and the parametric shape's ``material`` key) now spell their contents with, in
place of two different spellings of "what is here". This file tests the type in
isolation -- no solver runs here. Which convention the solver reads is a different question
and is pinned against closed-form Fresnel in ``tests/materials/test_material_convention_anchor.py``;
that separation is on purpose, because a type test and a physics test fail for different
reasons and a file that mixes them tells you neither.

Three properties get most of the attention, because they are the ones whose absence would be
silent:

* **the pytree registration.** An unregistered class is an opaque leaf to JAX, so a
  ``Material`` would sail through ``tree_map`` untouched and ``jax.grad`` would have nothing
  to hang a tangent on. A material is a legitimate design variable, so that is a real defect
  and not a theoretical one.
* **the lazy registration.** It must happen without ``import wavebit`` importing JAX, since
  importing the JAX backend flips ``jax_enable_x64`` process-wide for the caller.
* **the branch in :func:`index_from_eps`.** The principal root is the passive one for
  ``Im(eps) >= 0``, and the negative lip of the cut is where that stops being true.
"""

from wavebit.compat import legacy as _legacy_wavebit
import subprocess
import sys
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import wavebit
from wavebit import Material, eps_from_index, index_from_eps
from wavebit.materials.response import as_material

try:
    import jax
    import jax.numpy as jnp
    _legacy_wavebit.enable_x64()
except ImportError:                             # pragma: no cover - jax is an extra
    jax = jnp = None

# Not `importorskip` at module scope: that would skip the NumPy-only half of this file too,
# and the conversions and the shorthand are the part that has to hold on every install.
requires_jax = pytest.mark.skipif(jax is None, reason='jax is not installed')


# --------------------------------------------------------------- the scalar shorthand

def test_a_bare_number_means_a_non_magnetic_material():
    """``material=12.25`` and ``Material(12.25)`` are one input.

    Non-magnetic is the overwhelming default, so it has to cost nothing to write; if it did
    not, every existing scene would have to be rewritten to say ``mu=1`` out loud.
    """
    assert as_material(12.25) == Material(12.25)
    # Absent, not one: the shorthand names no permeability, and `None` is what "named none"
    # is spelled as. `mu_value` is where the number 1 lives, for the builders that multiply.
    assert as_material(12.25).mu is None
    assert as_material(12.25).mu_value == 1.0
    assert as_material(4 + 0.8j) == Material(4 + 0.8j)


def test_coercion_passes_a_material_through_unchanged():
    """Identity, not a copy -- so a caller can compare against the object it handed in."""
    m = Material(2.25, mu=1.5)
    assert as_material(m) is m


def test_absent_material_is_vacuum():
    """``None`` spells the default. It does not decide whether absence is *legal*.

    A slot where a material is mandatory rejects ``None`` before it reaches the coercion;
    what this fixes is that "unspecified" cannot come to mean two different vacua in two
    different paths.
    """
    assert as_material(None) == Material(1.0)


@pytest.mark.parametrize('bad', ['Si', ['rectangle', 0, 0], (1.0, 2.0), {'epsilon': 2.0}])
def test_containers_are_refused_at_a_material_slot(bad):
    """A string, list, tuple or dict here is a mistake worth naming.

    ``'Si'`` is the tempting one: there is no material-name lookup in wavebit, and quietly
    accepting a string would create the expectation of one. A list or dict is almost always a
    shape or layer recipe that landed one level too deep, and the message says so.
    """
    with pytest.raises(TypeError) as excinfo:
        as_material(bad, where='shapes[2].material')
    assert 'shapes[2].material' in str(excinfo.value)
    assert 'Material' in str(excinfo.value)


def test_arrays_and_tracers_are_not_refused():
    """Deliberately permissive, because the refusal cannot be made precise.

    Shape inspection cannot tell a vmapped tracer from a raster grid, and refusing both to
    catch one would break the design-variable case this type exists for.
    """
    grid = np.full((4, 4), 2.25)
    assert as_material(grid).epsilon is grid


# ------------------------------------------------------- frozen, equality, hashability

def test_material_is_frozen():
    """A material handed to a solve must not change under it."""
    m = Material(2.25)
    with pytest.raises(FrozenInstanceError):
        m.epsilon = 4.0


def test_equality_is_by_value_and_scalar_materials_hash():
    """Value semantics, so two spellings of one medium dedup instead of piling up.

    Identity semantics were the alternative and would have made ``Material(2.0) !=
    Material(2.0)``, defeating any set or dict keyed on materials -- silently, because such a
    dedup still "works", it just never dedups.
    """
    assert Material(2.0) == Material(2.0)
    assert Material(2.0) != Material(3.0)
    assert Material(2.0) != Material(2.0, mu=2.0)

    assert hash(Material(2.0)) == hash(Material(2.0))
    assert len({Material(2.0), Material(2.0), Material(3.0)}) == 2
    assert {Material(2.0): 'si'}[Material(2.0)] == 'si'


def test_a_number_is_not_equal_to_its_material():
    """The shorthand is an input convenience, not a claim that the two are the same thing.

    Making ``Material(2.0) == 2.0`` true would mean a number could be found in a set of
    materials -- and then ``eps`` and ``mu`` could not both be represented, since one number
    would have to stand for the pair.
    """
    assert Material(2.0) != 2.0
    assert 2.0 != Material(2.0)


def test_array_valued_material_inherits_its_values_semantics():
    """Not papered over: a container that answers cleanly for values that cannot would lie.

    ``Material`` passes its values through untouched -- that is what lets it hold a tracer --
    so it is exactly as well-behaved as what is in it. With an array ``epsilon`` you get
    NumPy's ambiguous-truth error from ``==`` and NumPy's unhashability from ``hash``, which
    is the honest answer rather than a coincidence.
    """
    a = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        Material(a) == Material(a.copy())
    with pytest.raises(TypeError):
        hash(Material(a))
    # ... while comparing one against itself short-circuits on identity, as tuples do.
    m = Material(a)
    assert m == m


# --------------------------------------------------------------------- the lossy case

def test_a_lossy_material_holds_its_complex_value_untouched():
    """``Im(eps) > 0`` is loss in ``e^{-iwt}`` (``docs/reference/common/conventions.md``); nothing is coerced.

    No ``complex()``, no ``abs()``, no conjugation on the way in. The conjugation seam that
    used to sit between the front door and the algebra is gone, and this type must not become
    a new one.
    """
    m = Material(4.0 + 0.8j)
    assert m.epsilon == 4.0 + 0.8j
    assert type(m.epsilon) is complex

    n = index_from_eps(m.epsilon)
    assert n.imag > 0, 'a passive permittivity must give a passive index'


def test_is_magnetic_asks_whether_a_permeability_was_named():
    """"Was a ``mu`` given?", which is structural -- **not** "is it different from one?".

    The distinction is not stylistic. A value comparison is unaskable at the only moment the
    answer is needed: under ``jax.jit`` a ``mu`` carried as a pytree child is a tracer, so
    ``m.mu != 1`` is a traced boolean and the ``if`` that reads it raises
    ``TracerBoolConversionError``. That is not hypothetical -- it is what an earlier draft of
    this type did, and because the guard then answered "magnetic" for every traced solve, the
    whole JAX backend's jitted entry point stopped working for any scene at all.

    ``None`` is therefore the *absence* of a permeability and the default, and it is an empty
    pytree subtree, so it survives tracing as ``None`` and the question stays answerable. The
    price is that ``Material(2.25, mu=1.0)`` reads ``True``: naming the value one is not the
    same input as naming nothing, and the two flatten to different treedefs. Keeping that
    crisp is deliberate -- a type whose tree shape depended on whether the caller happened to
    write the default would be the same class of trap one level down.
    """
    assert not Material(2.25).is_magnetic
    assert not as_material(2.25).is_magnetic

    # named, therefore carried -- even when the value it names is one
    assert Material(2.25, mu=1).is_magnetic
    assert Material(2.25, mu=np.float64(1.0)).is_magnetic
    assert Material(2.25, mu=2.0).is_magnetic
    assert Material(2.25, mu=1.0 + 0.1j).is_magnetic

    # and the number is still reachable, for the builders that multiply by it
    assert Material(2.25).mu_value == 1.0
    assert Material(2.25, mu=2.0).mu_value == 2.0


# ------------------------------------------------------------------- the conversions

def test_eps_from_index_is_exactly_the_square():
    """No dtype handling and no namespace, so tracers and gradients pass straight through.

    Exactness matters beyond tidiness here: the suite carries bit-exact assertions, and
    ``3.5**2`` is ``12.25`` on the nose while a hand-written decimal for a less friendly
    index would not be.
    """
    assert eps_from_index(3.5) == 12.25
    assert eps_from_index(1.5) == 2.25
    assert eps_from_index(2.2) == 2.2 * 2.2                 # and not the tidy 4.84
    assert eps_from_index(1.0) == 1.0
    assert eps_from_index(3.5 + 0.1j) == (3.5 + 0.1j) ** 2


def test_index_from_eps_takes_the_passive_root():
    """The principal root, verified against the convention rather than chosen by fiat.

    A passive medium has ``Im(eps) >= 0``, which puts ``eps`` in the closed upper half plane,
    whose principal root is in the first quadrant -- so ``Im(n) >= 0``, loss again. This is
    the same argument that makes ``conventions.boundary_kz`` principal at real ``k_par``.
    """
    assert index_from_eps(12.25) == 3.5
    assert index_from_eps(1.0) == 1.0

    n = index_from_eps(4.0 + 0.8j)
    assert n.real > 0 and n.imag > 0
    assert n ** 2 == pytest.approx(4.0 + 0.8j)

    # A metal: negative real permittivity, evanescent and decaying, not growing.
    assert index_from_eps(-2.0).imag == pytest.approx(2.0 ** 0.5)
    assert index_from_eps(-12.0 + 0.4j).imag > 0


def test_index_from_eps_is_always_complex():
    """Even for ``2.25``. Two reasons, and both are about what the alternative costs.

    A real ``sqrt`` of a metal's negative permittivity is ``nan`` -- silent, and metals are a
    supported material class. And returning a real index whenever the value happens to be
    real and non-negative would make the *dtype* depend on the value: it flips part-way
    through a sweep crossing zero, and under ``jax.jit`` the sign cannot be read at all, so
    eager and traced calls would disagree about the output type.
    """
    assert index_from_eps(2.25) == 1.5 + 0j
    assert isinstance(index_from_eps(2.25), complex)
    assert np.isdtype(index_from_eps(np.asarray(2.25)).dtype, 'complex floating')


def test_index_from_eps_preserves_precision_width():
    """``float32`` in gives ``complex64`` out -- promotion must not widen behind the caller.

    A single-precision solve that silently got a complex128 half-space index would carry the
    widening into every downstream promotion, which is the sort of thing that makes a
    precision comparison unreproducible.
    """
    out32 = index_from_eps(np.asarray([-2.0, 4.0], dtype=np.float32))
    assert out32.dtype == np.complex64
    out64 = index_from_eps(np.asarray([-2.0, 4.0], dtype=np.float64))
    assert out64.dtype == np.complex128
    assert np.asarray(out64).imag[0] > 0                     # still the passive root


def test_index_from_eps_negative_zero_is_the_documented_hazard():
    """The trap, pinned so that it stays *known* rather than becoming a surprise.

    The principal cut runs along the negative real axis and IEEE signed zero decides which lip
    a value sits on. ``complex(-2, 0.0)`` gives ``+1.4142j`` (decaying, correct) and
    ``complex(-2, -0.0)`` gives ``-1.4142j``, a gain medium, from an imaginary part that is
    identically zero with a sign bit set. (The expression ``-2 - 0j`` is not the second before
    Python 3.14: its imaginary part is ``+0.0`` there.) Nothing clamps it here: a helper cannot tell round-off from a deliberately active
    medium, and clamping would make gain media unrepresentable. Callers that build a
    permittivity by subtraction own this.
    """
    assert index_from_eps(complex(-2.0, 0.0)).imag > 0
    assert index_from_eps(complex(-2.0, -0.0)).imag < 0       # the hazard, not a bug here
    assert index_from_eps(complex(4.0, -1e-18)).imag < 0      # round-off scale, same lip


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_a_signed_zero_names_the_same_root_on_every_backend(backend):
    """The lip a signed zero names is the same on every backend, and so is everything read from it.

    ``jax.numpy.sqrt`` put both zeros on the upper lip (jaxlib 0.11.0, eager and under ``jit``,
    complex64 and complex128): ``sqrt(complex(-4, -0.0)) = +2j`` where NumPy, PyTorch and
    :mod:`cmath` give ``-2j``. So a lossless value written with ``-0.0`` and handed in as a JAX
    array got the other root from ``index_from_eps`` and ``signed_index``, the other ``n_bot``
    and the other sign of ``t_p`` (``+t_s`` against ``-t_s`` at normal incidence).
    ``backend.principal_sqrt`` turns it there; off the cut JAX's root is its own to the bit.
    """
    import cmath
    from wavebit._common.conventions import signed_index
    values = [complex(-2, 0.0), complex(-2, -0.0), complex(-4, -0.0), complex(-0.5, -0.0),
              complex(-3, 1e-300), complex(2.25, 0.1)]
    want = np.asarray([cmath.sqrt(v) for v in values])
    if backend == 'numpy':
        convert, back, solver_backend = np.asarray, np.asarray, 'numpy'
    elif backend == 'jax':
        jax = pytest.importorskip('jax')
        _legacy_wavebit.enable_x64()
        convert, back, solver_backend = jax.numpy.asarray, np.asarray, 'jax'
    else:
        torch = pytest.importorskip('torch')
        convert = lambda v: torch.tensor(v, dtype=torch.complex128)   # noqa: E731
        back, solver_backend = (lambda t: t.numpy()), 'torch'
    for dtype in (np.complex128, np.complex64):
        got = back(index_from_eps(convert(np.asarray(values, dtype=dtype))))
        assert np.all(np.sign(got.imag) == np.sign(want.imag)), dtype
        np.testing.assert_allclose(got, want, rtol=1e-6)
    if backend == 'jax':
        eager = np.asarray(index_from_eps(jax.numpy.asarray(values)))
        traced = np.asarray(jax.jit(index_from_eps)(jax.numpy.asarray(values)))
        assert traced.tobytes() == eager.tobytes()
        # On the cut the bits are NumPy's, the real zero included.
        on_cut = slice(0, 4)
        assert eager[on_cut].tobytes() == np.sqrt(np.asarray(values[on_cut])).tobytes()
    assert complex(back(signed_index(convert(complex(-4, -0.0)), convert(1 + 0j)))) == -2j
    outcome = {}
    for tetm in ('te', 'tm'):
        solver = _legacy_wavebit.rcwa(
            backend=solver_backend, top=Material(1 + 0j, mu=1 + 0j),
            bot=Material(convert(complex(-4, -0.0)), mu=convert(1 + 0j)), theta=0.0,
            wavelength=500.0, period=[300.0], truncation_order=0, thickness=[50.0],
            structure=np.full((1, 1, 4), 2.25 + 0j), mu=np.full((1, 1, 4), 1 + 0j), tetm=tetm,
            answers='amplitude')
        result = solver.solve(auto_detune=False)
        amplitude = result.incident.t_s if tetm == 'te' else result.incident.t_p
        outcome[tetm] = complex(np.ravel(back(amplitude))[0])
        assert complex(back(solver.n_bot)) == -2j
    assert abs(outcome['tm'] + outcome['te']) < 1e-13 * abs(outcome['te'])


def test_the_two_conversions_round_trip():
    """``eps -> n -> eps`` closes, which is the only claim either helper makes jointly.

    Written as a test because the pair is what a migrating script uses: an index-shaped
    datasheet goes in through ``eps_from_index`` and a solver-side permittivity comes back out
    through ``index_from_eps`` at the half-spaces.
    """
    for eps in (1.0, 2.25, 12.25, 4.0 + 0.8j, -12.0 + 0.4j):
        assert eps_from_index(index_from_eps(eps)) == pytest.approx(eps)
    for n in (1.0, 1.5, 3.5, 3.5 + 0.1j):
        assert index_from_eps(eps_from_index(n)) == pytest.approx(n)


# ------------------------------------------------------------------ the JAX pytree

def test_import_wavebit_does_not_import_jax():
    """Registration is lazy, and this is what "lazy" has to mean.

    ``src/wavebit/rcwa/_backends/jax_backend.py`` flips ``jax_enable_x64`` process-wide the moment it is
    imported -- a real side effect on the caller's own arrays -- so ``import wavebit`` must
    not drag JAX in on behalf of somebody who will never use it. A module-scope
    ``import jax`` in ``src/wavebit/_common/periodic.py`` would do exactly that, and nothing in the rest of
    the suite would notice, since the tests that matter import JAX themselves.

    Runs in a subprocess because this one cannot be observed in a process that has already
    imported JAX for the tests below.
    """
    code = ('import sys; import wavebit; m = wavebit.Material(2.25); '
            "assert 'jax' not in sys.modules, sorted(k for k in sys.modules if 'jax' in k); "
            'print(m)')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert 'Material(epsilon=2.25' in out.stdout


def test_material_built_after_jax_arrives_is_registered():
    """The ordering the lazy hook exists to cover: JAX imported *after* wavebit.

    Registration is attempted on every construction until it succeeds, so a ``Material`` built
    at any point after JAX lands registers the class -- and because registration is per class
    rather than per instance, it also repairs materials that were built before that.
    """
    code = (
        'import wavebit\n'
        'early = wavebit.Material(2.25)\n'                # built before jax exists
        'import jax, jax.numpy as jnp\n'
        'late = wavebit.Material(4.0)\n'                  # this construction registers
        # one leaf, not two: an unnamed `mu` is `None`, and `None` carries no leaf
        'assert jax.tree_util.tree_leaves(late) == [4.0]\n'
        'assert jax.tree_util.tree_leaves(early) == [2.25]\n'
        'assert jax.tree_util.tree_leaves(wavebit.Material(4.0, mu=1.0)) == [4.0, 1.0]\n'
        'print("ok")\n')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert 'ok' in out.stdout


@requires_jax
def test_registration_is_idempotent():
    """Calling the hook again must not raise -- JAX refuses a duplicate registration.

    It is called from every ``Material.__post_init__``, so "again" is the normal case; and a
    second wavebit import in one process (the worktree/main-checkout duplication this repo
    is prone to) must not turn into a crash at construction time.
    """
    from wavebit.materials import response as domain

    Material(1.0)                                          # ensure the first one has happened
    assert domain._PYTREE_REGISTERED
    domain._PYTREE_REGISTERED = False                      # force the real call again
    try:
        domain._register_pytree_if_jax_loaded()
    finally:
        domain._PYTREE_REGISTERED = True
    assert jnp.asarray(Material(2.0).epsilon) == 2.0       # still usable afterwards


@requires_jax
def test_pytree_round_trip_keeps_both_values_as_children():
    """Children ``(epsilon, mu)``, aux_data ``None``.

    Both are children rather than aux_data on purpose: aux_data is hashed into the jit cache
    key, so a swept permittivity carried there would retrace on every step of an optimisation
    and a traced one could not be carried at all. That is the same defect already fixed twice
    on this branch, in the other direction.
    """
    m = Material(12.25, mu=1.5)
    leaves, treedef = jax.tree_util.tree_flatten(m)
    assert leaves == [12.25, 1.5]
    assert jax.tree_util.tree_unflatten(treedef, leaves) == m

    doubled = jax.tree_util.tree_map(lambda v: v * 2, m)
    assert isinstance(doubled, Material)
    assert doubled == Material(24.5, mu=3.0)


@requires_jax
def test_a_material_survives_a_jit_boundary():
    """As an argument and as a return value, which is what "registered" buys.

    Unregistered, the same call raises ``... is not a valid JAX type`` at the boundary: JAX
    sees one opaque leaf where two numbers should be.
    """
    @jax.jit
    def scale(m, k):
        return Material(m.epsilon * k, mu=m.mu * k)

    out = scale(Material(4.0, mu=2.0), 3.0)
    assert isinstance(out, Material)
    assert float(out.epsilon) == 12.0 and float(out.mu) == 6.0


@requires_jax
def test_grad_flows_to_a_materials_epsilon():
    """A material is a design variable, so the tangent has to reach ``epsilon``.

    This is the property that silently disappears without registration: JAX would treat the
    whole ``Material`` as one leaf, find no float to differentiate, and the gradient of a
    material sweep would come back as nothing at all rather than as an error.

    The cotangent has the same tree shape as the input, which is what lets an optimiser add it
    to the parameters it came from -- and for an unnamed ``mu`` that shape has no ``mu`` slot
    at all, so the tangent is ``None`` rather than a zero. A material that *does* name one gets
    a real zero there, which is the case just below.
    """
    g = jax.grad(lambda m: m.epsilon ** 3)(Material(2.0))
    assert isinstance(g, Material)
    assert float(g.epsilon) == pytest.approx(12.0)          # d(eps^3) = 3 eps^2
    assert g.mu is None

    named = jax.grad(lambda m: m.epsilon ** 3)(Material(2.0, mu=1.0))
    assert float(named.mu) == 0.0
    assert g.epsilon.dtype == jnp.float64, 'x64 must be lifted before this is meaningful'


@requires_jax
def test_a_traced_epsilon_survives_construction_inside_a_trace():
    """The other direction: a ``Material`` *built* from a tracer, mid-solve.

    This is the parametric path's shape-material case -- the value is a tracer by the time the
    scene is assembled. It works only because ``Material`` coerces nothing in
    ``__post_init__``: a ``float()`` or an ``asarray`` there would raise at trace time, or
    worse, drop the tangent and return a plausible zero.
    """
    def loss(x):
        m = Material(eps_from_index(x))                    # tracer in, tracer out
        return jnp.real(m.epsilon) * 10.0

    assert float(jax.grad(loss)(3.0)) == pytest.approx(60.0)   # d(10 x^2) = 20 x


@requires_jax
def test_a_lossy_material_differentiates_holomorphically():
    """The complex case, which needs every *leaf* complex -- and ``mu=None`` is not a leaf.

    ``jax.grad(..., holomorphic=True)`` checks the dtype of all leaves, so what matters is
    which leaves exist. An unnamed ``mu`` contributes none, so a lossy material carrying only
    its permittivity differentiates holomorphically with nothing to declare -- that is a
    consequence of the ``None`` sentinel rather than a separate design choice, and it is worth
    pinning because an earlier draft defaulted ``mu`` to a real ``1.0`` and this call was
    rejected before any arithmetic happened, with an error naming ``float64`` and pointing at
    nothing in particular.

    A material that *names* a real ``mu`` still trips it, and that refusal is correct: the
    caller did put a real number in the tree. The fix there is to spell it complex, which is
    the first case below.
    """
    m = Material(jnp.asarray(4.0 + 0.8j), mu=jnp.asarray(1.0 + 0j))
    g = jax.grad(lambda mm: mm.epsilon ** 2, holomorphic=True)(m)
    assert complex(g.epsilon) == pytest.approx(2 * (4.0 + 0.8j))

    bare = jax.grad(lambda mm: mm.epsilon ** 2,
                    holomorphic=True)(Material(jnp.asarray(4.0 + 0.8j)))
    assert complex(bare.epsilon) == pytest.approx(2 * (4.0 + 0.8j))
    assert bare.mu is None

    with pytest.raises(TypeError):
        jax.grad(lambda mm: mm.epsilon ** 2, holomorphic=True)(
            Material(jnp.asarray(4.0 + 0.8j), mu=jnp.asarray(1.0)))


@requires_jax
def test_is_magnetic_answers_the_same_eagerly_and_under_trace():
    """The property the whole ``None`` sentinel exists for, asserted directly.

    An earlier draft keyed this on *readability* -- concrete ``mu`` compared against one,
    abstract ``mu`` assumed magnetic -- and the eager/traced asymmetry that produced was not a
    subtlety, it was the bug: every jitted solve took the magnetic branch, and the branch then
    compared a tracer against ``1.0`` and raised. Structural means the two must agree, so that
    is what is pinned here rather than either answer on its own.
    """
    for m in (Material(2.25),
              Material(2.25, mu=1.0),
              Material(jnp.asarray(2.25), mu=jnp.asarray(1.0)),
              Material(jnp.asarray(2.25), mu=jnp.asarray(2.0))):
        traced = bool(jax.jit(lambda mm: jnp.asarray(mm.is_magnetic))(m))
        assert traced == m.is_magnetic, f'{m} disagrees across the trace boundary'

    # and the direction is the structural one, not the value one
    assert not Material(jnp.asarray(2.25)).is_magnetic
    assert Material(jnp.asarray(2.25), mu=jnp.asarray(1.0)).is_magnetic


@requires_jax
def test_index_from_eps_runs_on_a_jax_array():
    """Backend-agnostic through the Array-API namespace, tracer included."""
    n = index_from_eps(jnp.asarray(4.0 + 0.8j))
    assert complex(n) == pytest.approx(index_from_eps(4.0 + 0.8j))
    traced = jax.jit(index_from_eps)(jnp.asarray(-2.0))
    assert complex(traced.imag) == pytest.approx(2.0 ** 0.5)


def test_material_is_exported_from_the_package_root():
    """The public surface: the type and both conversions, importable as ``wavebit.X``."""
    assert wavebit.Material is Material
    assert wavebit.eps_from_index is eps_from_index
    assert wavebit.index_from_eps is index_from_eps



@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
@pytest.mark.parametrize('where', ['top', 'bot'])
def test_a_refused_half_space_leaves_neither_the_material_nor_the_index(backend, where):
    """The Material and the index it yields are one write, and they used to be two.

    The setter stored the Material and *then* derived the index, and the derivation is where
    `_refuse_shaped_half_space` lives. So a caller who handed over a shaped epsilon -- a raster
    where a half-space belongs, which is the mistake the guard exists for -- caught the
    `ValueError` and was left with `top` holding the Material the guard had just refused while
    `n_top` still held the previous medium's index.

    The two then disagreed, and the solve reads the *index*, so it went on answering the old
    question under the new label. Naming a `mu` in the refused Material made it worse:
    `is_magnetic` flipped with it, and that selects a different arithmetic path rather than a
    different number.
    """
    if backend == 'jax':
        pytest.importorskip('jax')
    elif backend == 'torch':
        pytest.importorskip('torch')
    solver = _legacy_wavebit.rcwa(backend=backend, structure=np.ones((1, 4, 4)) * 2.0,
                          period=[400.0, 500.0], truncation_order=(2, 2), wavelength=620.,
                          thickness=[150.0], top=2.25, bot=1.5)
    index = f'n_{where}'
    was = (getattr(solver, where).epsilon, getattr(solver, index))

    for refused in (Material(np.ones((4, 4)) * 2.0),          # shaped epsilon
                    Material(2.0, mu=np.ones((4, 4)) * 1.4)):  # scalar epsilon, shaped mu
        with pytest.raises(ValueError):
            setattr(solver, where, refused)
        assert getattr(solver, where).epsilon == was[0], f'{where} kept the refused material'
        assert getattr(solver, index) == was[1], f'{index} and {where} disagree'
