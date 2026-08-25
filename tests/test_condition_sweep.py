"""Theorems that hold for *any* scene, run over a pairwise cover of the conditions.

The suite's shape, measured on 2026-08-24: of 923 test functions, 590 (63.9%) run at a single
fixed operating point, 251 vary one axis, and 82 cross two or more. ``lattice`` was crossed with
nothing at all -- not once in the whole suite was the cell basis parametrized against another
condition. Every bug found this week was a *pair*: ``excitation`` x skew (`b7c4645`), frozen
geometry x skew (`2badc83`), the drawing entry x skew (`c5b061f`), ``complex_dtype`` x skew
(`4316825`). Each half was covered; the product was not, and the product is where the wiring is.

Writing a test per pair does not scale -- there are a dozen axes. What scales is this file's
shape:

* **conditions** are values on a few axes, combined into an *all-pairs* cover. 96 combinations
  reduce to 16 cases in which every pair of values from two axes appears together at least once.
  That is the coverage the bugs above needed, at a sixth of the cost, and it is deterministic --
  a failure is reproducible by its case name.
* **invariants** are theorems, not recorded numbers. Each one internally varies the axis it is
  about (the basis representation, the backend, the algorithm, the declared width) and compares
  a scene against *itself*. Nothing here is a golden, so nothing here goes stale.

Adding an axis or a value re-crosses every invariant against it for free. That is the point: the
next feature does not need anyone to remember to cross it with the cell.

**Energy conservation is not among the invariants.** NVF assembly is Hermitian-symmetrised
whatever the normal field says, so ``R + T == 1`` to 1e-15 while the cell is being read wrongly;
it is asserted only as a liveness check, to catch a scene that stopped solving at all.
"""
import itertools

import numpy as np
import pytest

import wavebit

P, WL, H, ROD = 400.0, 900.0, 150.0, 110.0

LATTICE = {
    'square': np.diag([P, P]),
    'rect': np.diag([P, 1.6 * P]),
    'hex': np.column_stack(([P, 0.0], [P / 2, P * np.sqrt(3) / 2])),
    'shear': np.column_stack(([P, 0.0], [0.35 * P, 1.2 * P])),
}
INCIDENCE = {
    'normal': dict(theta=0.0, phi=0.0, psi=0.0),
    'oblique': dict(theta=0.35, phi=0.0, psi=0.4),
    'azimuthal': dict(theta=0.35, phi=0.9, psi=0.4),
    'circular': dict(theta=0.35, phi=0.9, psi=0.0, chi=np.pi / 4),
}
MEDIA = {
    'lossless': dict(top=1.0, bot=1.0, eps=5.76),
    'lossy': dict(top=1.0, bot=2.25, eps=5.76 + 0.35j),
    'magnetic': dict(top=1.0, bot=1.0, eps=5.76, mu=1.8),
}
SIDE = ['top', 'bottom']

AXES = {'lattice': list(LATTICE), 'incidence': list(INCIDENCE),
        'media': list(MEDIA), 'side': SIDE}


def _all_pairs(axes):
    """Greedy all-pairs cover: every value pair from two different axes appears in some case.

    Max-gain greedy rather than a first-fit scan of the product -- first-fit gives 39 cases here
    and front-loads them onto one axis's first value, which is exactly the bias this file exists
    to remove. Ties break on the product's own order, so the list is deterministic.
    """
    names = list(axes)
    values = [list(axes[n]) for n in names]
    idx = list(itertools.combinations(range(len(names)), 2))
    need = {(i, a, j, b) for i, j in idx for a in values[i] for b in values[j]}
    chosen = []
    while need:
        best, best_gain, best_got = None, 0, None
        for cand in itertools.product(*values):
            got = {(i, cand[i], j, cand[j]) for i, j in idx}
            gain = len(got & need)
            if gain > best_gain:
                best, best_gain, best_got = cand, gain, got
        chosen.append(best)
        need -= best_got
    return [dict(zip(names, c)) for c in chosen]


CASES = _all_pairs(AXES)
IDS = ['-'.join(c[k] for k in AXES) for c in CASES]


def _rod_mask(basis, n=32, radius=ROD):
    """A disc, stated in the cell's own coordinates so a change of basis carries the same object."""
    a1, a2 = np.asarray(basis)[:, 0], np.asarray(basis)[:, 1]
    t = (np.arange(n) + 0.5) / n
    U, V = np.meshgrid(t, t)
    pts = U[..., None] * a1 + V[..., None] * a2
    centre = 0.5 * (a1 + a2)
    inside = np.zeros(pts.shape[:2], bool)
    for i in (-1, 0, 1):                              # minimum image over the lattice
        for j in (-1, 0, 1):
            s = i * a1 + j * a2
            inside |= (((pts[..., 0] - centre[0] - s[0]) ** 2
                        + (pts[..., 1] - centre[1] - s[1]) ** 2) <= radius ** 2)
    return inside


def scene(case, basis=None, **over):
    """One solver for a condition case. ``basis`` overrides the cell without moving the object."""
    basis = LATTICE[case['lattice']] if basis is None else np.asarray(basis)
    medium = MEDIA[case['media']]
    mask = _rod_mask(basis, n=over.pop('n', 32))
    kw = dict(backend=0, structure=np.where(mask, medium['eps'], 1.0)[None, ...],
              period=[P, P], lattice=basis.tolist(), truncation_order=[4, 4], thickness=[H],
              wavelength=WL, top=medium['top'], bot=medium['bot'], factorization='nvf',
              incidence_side=case['side'], **INCIDENCE[case['incidence']])
    if 'mu' in medium:
        kw['mu'] = np.where(mask, medium['mu'], 1.0)[None, ...]
    kw.update(over)
    return wavebit.rcwa(**kw)


def totals(result):
    return np.asarray(result.R), np.asarray(result.T)


@pytest.mark.parametrize('case', CASES, ids=IDS)
def test_the_scene_solves_at_all(case):
    """Liveness, and the one place ``R + T`` is allowed to be the assertion.

    A dead or degenerate scene passes every comparison below by comparing nothing to nothing, so
    each case states once that it is a real solve. On the lossy medium the sum is bounded rather
    than pinned, which is all absorption permits.
    """
    R, T = totals(scene(case).solve())
    assert np.all(np.isfinite(R)) and np.all(np.isfinite(T))
    assert float(np.sum(R) + np.sum(T)) > 1e-6
    if case['media'] == 'lossy':
        assert float(np.sum(R) + np.sum(T)) < 1.0 + 1e-9
    else:
        assert float(np.sum(R) + np.sum(T)) == pytest.approx(1.0, abs=1e-9)


# ------------------------------------------------------------------ the cell's own description

def shape_scene(case, basis=None, **over):
    """The same disc, stated as a *shape* rather than sampled onto the cell's grid.

    Needed by the re-basis invariant and by nothing else. A raster is not the same object under
    a change of basis: re-basing re-discretizes it, and the staircase difference converges only
    as ``O(1/n)`` -- measured on the hex cell, 7.4% relative at ``n=32`` and still 2.0% at
    ``n=512``, which is far above anything a wiring bug would move. Stating the disc exactly
    removes that term and leaves the truncation as the only difference between the two spellings.
    """
    basis = LATTICE[case['lattice']] if basis is None else np.asarray(basis)
    medium = MEDIA[case['media']]
    centre = 0.5 * (np.asarray(basis)[:, 0] + np.asarray(basis)[:, 1])
    material = (wavebit.Material(medium['eps'], mu=medium['mu']) if 'mu' in medium
                else medium['eps'])
    kw = dict(backend=0, period=[P, P], lattice=basis.tolist(), truncation_order=[6, 6],
              thickness=[H], wavelength=WL, top=medium['top'], bot=medium['bot'],
              factorization='nvf', incidence_side=case['side'], normal_grid=256,
              structure=[{'background': 1.0,
                          'shapes': [{'type': 'circle', 'cx': float(centre[0]),
                                      'cy': float(centre[1]), 'lx': 2 * ROD,
                                      'material': material}]}],
              **INCIDENCE[case['incidence']])
    kw.update(over)
    return wavebit.rcwa(**kw)


@pytest.mark.parametrize('case', CASES, ids=IDS)
def test_a_unimodular_rebasis_is_the_same_crystal(case):
    """``A @ U`` with ``det U = +-1`` spells the same point lattice, so every total must agree.

    Any place that reads ``period`` where it needed the basis reads ``(|a1|, |a2|)`` -- which a
    re-basis *does* change while the crystal does not -- and parts here. That is the shape of
    every lattice bug found this week.

    Not bit-identity, and not the raster: the two spellings retain a rectangle of orders in
    different index coordinates, so they agree to the truncation. 2e-3 is above the measured
    residual at ``truncation_order=6`` and far below the 6e-2 a dropped off-diagonal produces.
    """
    rebased = LATTICE[case['lattice']] @ np.array([[1, 0], [1, 1]])     # a1 -> a1 + a2
    R0, T0 = totals(shape_scene(case).solve())
    R1, T1 = totals(shape_scene(case, basis=rebased).solve())
    scale = max(float(np.sum(R0) + np.sum(T0)), 1e-12)
    assert abs(float(np.sum(R1) + np.sum(T1)) - float(np.sum(R0) + np.sum(T0))) / scale < 2e-3, case
    assert abs(float(np.sum(R1)) - float(np.sum(R0))) / scale < 2e-3, case


@pytest.mark.parametrize('case', CASES, ids=IDS)
def test_turning_the_cell_and_the_azimuth_together_changes_nothing(case):
    """A change of observer. Rotate ``A`` and ``phi`` by the same angle; the powers are fixed.

    47 degrees, not 90: a right angle maps ``x`` onto ``y`` and so survives an axis-aligned
    assumption unharmed -- it agreed to 1e-14 while `b7c4645` was live.
    """
    angle = np.radians(47.0)
    c, s = np.cos(angle), np.sin(angle)
    basis = np.array([[c, -s], [s, c]]) @ LATTICE[case['lattice']]
    phi = INCIDENCE[case['incidence']].get('phi', 0.0)
    R0, T0 = totals(scene(case).solve())
    R1, T1 = totals(scene(case, basis=basis, phi=phi + angle).solve())
    for a, b, name in ((R0, R1, 'R'), (T0, T1, 'T')):
        assert float(np.max(np.abs(a - b))) < 1e-11, (name, case)


# --------------------------------------------------------------- the machinery under the scene

@pytest.mark.parametrize('case', CASES, ids=IDS)
def test_the_two_layer_algorithms_agree(case):
    """ETM and SMM are two ways to cascade one stack, so they answer the same question.

    A stack this thin is far from where the transfer form loses its conditioning, which is what
    makes the comparison a statement about the *wiring* rather than about stability.
    """
    a = totals(scene(case, layer_algorithm='ETM').solve())
    b = totals(scene(case, layer_algorithm='SMM').solve())
    for x, y, name in ((a[0], b[0], 'R'), (a[1], b[1], 'T')):
        assert float(np.max(np.abs(x - y))) < 1e-9, (name, case)


@pytest.mark.parametrize('case', CASES, ids=IDS)
def test_the_backends_agree(case):
    """NumPy, JAX and Torch solve the same scene. Differences here are wiring, not physics."""
    ref = totals(scene(case, backend=0).solve())
    for backend in (1, 2):
        got = totals(scene(case, backend=backend).solve())
        for x, y, name in ((ref[0], got[0], 'R'), (ref[1], got[1], 'T')):
            assert float(np.max(np.abs(np.asarray(x) - np.asarray(y)))) < 1e-9, \
                (backend, name, case)


@pytest.mark.parametrize('case', CASES, ids=IDS)
def test_the_declared_width_is_the_width_answered_in(case):
    """``complex_dtype`` is a declaration, and until `4316825` a named cell ignored it.

    The basis is read rather than converted, so it arrives at float64 and NEP 50 widened the
    whole return with it -- but only once the cell was skewed, because the rectangular path came
    back through a build that closes with its own cast. Asserted as *the same width whatever the
    cell*, which is the claim that broke.
    """
    for complex_dtype, want in ((0, np.float64), (1, np.float32)):
        R, T = totals(scene(case, complex_dtype=complex_dtype).solve())
        assert np.asarray(R).dtype == want, (complex_dtype, case)
        assert np.asarray(T).dtype == want, (complex_dtype, case)


# ------------------------------------------------ the clamp that was three bodies and one lie

#: `theta` values the incident wavevector has to survive, including the two the clamp exists
#: for. `pi/2` is where `sin` in single precision walks off the branch numpy documents at
#: numpy/numpy#27306; `float32(pi/2)` is the threshold itself, one ulp the other side.
CLAMP_THETAS = [0.0, 0.3, 0.4 + 0.3j, np.pi / 2, np.pi / 2 + 0.5j, 1.4 + 2j, 1.9, -0.7,
                float(np.float32(np.pi / 2)) + 0.5j]


@pytest.mark.parametrize('theta', CLAMP_THETAS, ids=lambda t: f'{t!r}')
def test_the_three_backends_clamp_a_grazing_theta_the_same_way(theta):
    """One body serves all three, and the sweep above cannot see it.

    ``_incident_kxy`` lived three times, under a docstring saying the ``sin(theta)`` clamp
    "differs between NumPy, JAX and Torch for tracing reasons". It does not. NumPy and Torch
    branched with ``if`` and JAX with ``jnp.where`` on the *result*; putting the ``where`` on
    the *argument* reproduces all three byte for byte. The sentence was in four places and was
    never checked -- it was copied, including by the commit that split the method out.

    Nothing else here reaches this. Every scene in the condition sweep is at a propagating
    angle, so the clamped branch is never taken and the 96-scene digest is blind to it: a
    change that broke only ``theta >= pi/2`` would pass the whole file. These points are the
    part of the input space the sweep does not cover, and they are cheap because no solve runs.

    Cross-backend and not merely self-consistent, because that is the claim that was false.
    ``1e-13`` rather than byte equality: the three ``sin`` implementations differ by an ulp on
    some of these already, and pinning to the byte would fail for a reason that has nothing to
    do with the clamp.

    Agreement alone is not enough, and finding that out is why the second assertion is here: a
    cross-backend check is blind to any change that moves all three the same way. Deleting the
    clamp outright passed it. So the clamp's *own* property is stated too --
    :func:`test_a_theta_past_grazing_saturates` below.
    """
    reference = None
    for backend in (0, 1, 2):
        solver = wavebit.rcwa(backend=backend, structure=np.ones((1, 4, 4)), period=[P, P],
                              truncation_order=[1, 1], thickness=[H], wavelength=WL,
                              top=1.7, bot=1.0, theta=theta, phi=0.4, psi=0.1)
        got = tuple(complex(np.asarray(v)) for v in solver._incident_kxy(WL))
        assert np.all(np.isfinite([[v.real, v.imag] for v in got])), (backend, got)
        if reference is None:
            reference = got
            # A zero result would satisfy the comparison below trivially -- but at normal
            # incidence it is the right answer, so the guard applies only where sin(theta) is
            # not itself zero.
            if theta != 0.0:
                assert abs(got[0]) + abs(got[1]) > 1e-12, got
        else:
            for a, b in zip(reference, got):
                assert abs(a - b) < 1e-13, (backend, a, b)


@pytest.mark.parametrize('imag', [0.0, 0.5, 2.0])
def test_a_theta_past_grazing_saturates(imag):
    """Above ``pi/2`` the real part stops mattering, and that is what the clamp is *for*.

    It is not a rounding fix. It replaces the real part with ``pi/2`` (one float32 ulp below),
    so ``sin(2.0)`` returns 1.0 rather than 0.909: any incidence at or past grazing is treated
    as grazing, and the imaginary part -- the decay of an evanescent incidence -- rides through
    untouched.

    Stated as an identity rather than a table: above the threshold the answer does not depend
    on the real part at all. That catches the clamp being deleted, which cross-backend
    agreement cannot -- removing it moves all three backends together, so they still agree, on
    the wrong number.

    The threshold is ``float32(pi/2)``, computed here rather than imported, and it sits
    *above* ``pi/2`` because float32 rounds up: 1.5707963705062866 against 1.5707963267948966.
    So ``theta = pi/2`` does not trigger the clamp, and using it as the reference is what the
    first version of this test did -- it failed, correctly, for a reason that was the test's
    and not the code's.
    """
    threshold = float(np.float32(np.pi / 2))
    reference = None
    for backend in (0, 1, 2):
        for real in (threshold, 1.9, 2.6, 3.0):
            solver = wavebit.rcwa(backend=backend, structure=np.ones((1, 4, 4)), period=[P, P],
                                  truncation_order=[1, 1], thickness=[H], wavelength=WL,
                                  top=1.7, bot=1.0, theta=real + 1j * imag, phi=0.4, psi=0.1)
            got = tuple(complex(np.asarray(v)) for v in solver._incident_kxy(WL))
            if reference is None:
                reference = got
                assert abs(got[0]) > 1e-12, got            # a zero would pass anything
            for a, b in zip(reference, got):
                assert abs(a - b) < 1e-13, (backend, real, a, b)

    # The real part stops mattering; the imaginary part must not. Saturating is what the clamp
    # is for, but it saturates *only* the real part -- an evanescent incidence carries its decay
    # in `Im(theta)`, and a clamp that dropped it would leave every one of them looking like a
    # grazing propagating wave. The identity above cannot see that on its own: with the
    # imaginary part gone, all four real parts still agree, so the check still passes.
    grazing = float(np.float32(np.pi / 2))
    def _at(imag_part):
        solver = wavebit.rcwa(backend=0, structure=np.ones((1, 4, 4)), period=[P, P],
                              truncation_order=[1, 1], thickness=[H], wavelength=WL,
                              top=1.7, bot=1.0, theta=grazing + 1j * imag_part,
                              phi=0.4, psi=0.1)
        return complex(np.asarray(solver._incident_kxy(WL)[0]))
    if imag != 0.0:
        assert abs(_at(imag) - _at(0.0)) > 1e-6, (imag, _at(imag), _at(0.0))


# --------------------------------------------------- the scope no digest on this machine sees

def test_every_backend_declares_where_its_arrays_are_placed():
    """`_device_scope` is a declared member, and that is the whole point of it.

    JAX is told where to build things by `jax.default_device`, which is a *scope* rather than
    an argument -- so five methods carried a `@jax_device_set` decorator that named
    `jax.default_device` directly. All five have counterparts on the other two backends, so the
    decorator had to stop belonging to one backend before those bodies could become one body.

    A missing device scope is invisible in every other check here. Everything still computes,
    on the wrong device, and the 96-scene digest is CPU-only on this machine -- it cannot tell
    a scoped call from an unscoped one. So the contract is asserted directly: every backend
    answers `_device_scope()` with a context manager, and the five methods that need one carry
    the decorator that opens it.

    `self.device` is a list on JAX (`jax.devices('cpu')`) and a scalar elsewhere, which is why
    this is an override rather than a value the base could hold.
    """
    import contextlib
    import inspect

    scoped = ('solve_conv', 'reconstruct_field', 'solve_field', 'solve_field_eager',
              '_separable_kxy')

    for backend in (0, 1, 2):
        solver = wavebit.rcwa(backend=backend, structure=np.ones((1, 4, 4)), period=[P, P],
                              truncation_order=[1, 1], thickness=[H], wavelength=WL,
                              top=1.0, bot=1.0)
        scope = solver._device_scope()
        assert hasattr(scope, '__enter__') and hasattr(scope, '__exit__'), (backend, scope)
        with scope:                                    # it has to be usable, not merely typed
            pass

    jax_solver = wavebit.rcwa(backend=1, structure=np.ones((1, 4, 4)), period=[P, P],
                              truncation_order=[1, 1], thickness=[H], wavelength=WL,
                              top=1.0, bot=1.0)
    assert not isinstance(jax_solver._device_scope(), contextlib.nullcontext), (
        'the JAX override is gone: its scope is the only one that does anything, and a '
        '`nullcontext` here means every array is placed wherever JAX defaults to.')

    for name in scoped:
        method = inspect.unwrap(getattr(type(jax_solver), name))
        assert method is not getattr(type(jax_solver), name), (
            f'{name} is not wrapped: it needs `@device_scoped`, and without it JAX builds its '
            f'arrays outside the solver\'s device scope.')
