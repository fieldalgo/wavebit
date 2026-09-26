"""Choosing between the two formulations — and what each choice must not disturb.

``tests/test_solve_2d_4ff.py`` pins the first-order engine itself, and
``tests/test_anisotropic_berreman.py`` pins the physics it now answers through the public API.
This file is about the *dispatch*: which path a given call takes, what happens at the edges of
that decision, and — the half that is easy to lose — that the ordinary path is untouched by the
existence of the other one.

The four claims worth stating up front, because each was a way this could have gone wrong
silently:

* **A z-decoupled problem still runs on the second-order path**, and gives what it always gave.
  Widening a signature with five optional arguments is exactly the change that quietly reroutes
  everything; ``formulation_assigned`` is checked, not just the number.
* **Forced ``'4ff'`` reproduces the ``'2ff'`` answer** on problems both can express — including a
  patterned grating, where the two formulations genuinely disagree about what a layer *is* and
  agree only about the answer. No external oracle provides this cross-check.
* **Forced ``'2ff'`` refuses a z row** rather than dropping it.
* **The transmitted amplitude is checked, not only the reflected one.** ``big_T`` feeds ``T_s`` /
  ``T_p`` directly, so a per-mode scale that cancels in ``R`` survives in ``T``; a route that got
  that wrong would match on reflection and be off by a constant factor on transmission, with
  ``R + T`` still looking sane on a lossless stack. Every comparison below is on amplitudes as
  well as powers.
"""
import numpy as np
import pytest

import wavebit

WL = 620.0
PERIOD = (400.0, 380.0)


def uniform_slots(eps_list, n_orders, z_row=True):
    """The nine (or five) convolution stacks for a list of uniform tensors.

    Slot convention: every off-diagonal slot holds the **negative** of its tensor component,
    the z row included; ``epz_i`` is the inverse of the ``zz`` block. See
    ``docs/off-diagonal-slot-convention.md``.
    """
    I = np.eye(n_orders, dtype=complex)
    e = [np.asarray(x, dtype=complex) for x in eps_list]
    stacks = [np.stack([x[0, 0] * I for x in e]),
              np.stack([x[1, 1] * I for x in e]),
              np.stack([np.linalg.inv(x[2, 2] * I) for x in e]),
              np.stack([-x[0, 1] * I for x in e]),
              np.stack([-x[1, 0] * I for x in e]),
              np.stack([-x[0, 2] * I for x in e]),
              np.stack([-x[1, 2] * I for x in e]),
              np.stack([-x[2, 0] * I for x in e]),
              np.stack([-x[2, 1] * I for x in e])]
    return stacks if z_row else stacks[:5]


def solver(truncation_order=(0, 0), backend=0, **kw):
    kw.setdefault('theta', np.deg2rad(28.0))
    kw.setdefault('phi', np.deg2rad(37.0))
    kw.setdefault('psi', 0.6)
    return wavebit.rcwa(backend=backend, top=1.0, bot=2.3104, truncation_order=truncation_order, period=PERIOD,
                          wavelength=WL, **kw)


def channels(res):
    """Every public number one incidence produces — powers and amplitudes alike."""
    out = {}
    for tag, sub in (('psi', res.incident), ('te', res.s_inc), ('tm', res.p_inc)):
        for name in ('R', 'T', 'R_s', 'R_p', 'T_s', 'T_p', 'r_s', 'r_p', 't_s', 't_p'):
            out[f'{tag}.{name}'] = np.asarray(getattr(sub, name))
    return out


def worst_gap(a, b):
    """Largest relative-to-scale gap over every channel, and which channel it was."""
    worst, where = 0.0, ''
    for key in a:
        gap = float(np.abs(a[key] - b[key]).max())
        scale = max(float(np.abs(a[key]).max()), 1e-12)
        if gap / scale > worst:
            worst, where = gap / scale, key
    return worst, where


# --------------------------------------------------------------- the ordinary path is intact

DIAGONAL = np.diag([2.4, 3.1, 2.7]).astype(complex)
IN_PLANE = np.array([[2.4, 0.35, 0], [0.35, 3.1, 0], [0, 0, 2.7]], dtype=complex)
VOIGT = np.array([[2.4, 0.2j, -0.31j], [-0.2j, 3.1, 0], [0.31j, 0, 2.7]], dtype=complex)


@pytest.mark.parametrize('eps', [DIAGONAL, IN_PLANE], ids=['diagonal', 'in-plane pair'])
def test_a_tensor_without_a_z_row_stays_on_the_second_order_path(eps):
    """The default reads the slots, and five slots mean the cheap formulation.

    Checking ``formulation_assigned`` rather than only the numbers: the two paths agree to
    ~1e-13 on this problem, so a dispatch that sent everything to the first-order solve would
    pass every numerical assertion in this file while costing 8x.
    """
    mee = solver(thickness=[180.0])
    res = mee.solve_conv(*uniform_slots([eps], 1, z_row=False))
    assert mee.formulation_assigned == '2ff'
    assert np.isfinite(np.asarray(res.incident.R)).all()


def test_passing_the_z_slots_is_what_selects_the_first_order_path():
    """Given-ness decides, not the values in them.

    The branch decides control flow, so it has to be answerable without reading an array — on a
    JAX tracer there is nothing to read. "Passed all-zero matrices" is a claim about the caller
    rather than about the tensor, and the cheap path is one keyword away.
    """
    zero_z = np.zeros((1, 1, 1), dtype=complex)
    mee = solver(thickness=[180.0])
    five = uniform_slots([DIAGONAL], 1, z_row=False)
    res_2ff = mee.solve_conv(*five)
    assert mee.formulation_assigned == '2ff'

    mee = solver(thickness=[180.0])
    res_4ff = mee.solve_conv(*five, zero_z, zero_z, zero_z, zero_z)
    assert mee.formulation_assigned == '4ff'

    gap, where = worst_gap(channels(res_2ff), channels(res_4ff))
    assert gap < 1e-11, f'an empty z row changed the answer by {gap:.2e} on {where}'


def test_the_default_never_reaches_the_first_order_path_on_a_grating():
    """A raster solve — the ordinary case — must be untouched by any of this."""
    structure = np.ones((2, 20, 20))
    structure[1, 5:15, 5:15] = 11.559999999999999           # permittivity: 3.4**2
    mee = solver(truncation_order=(3, 3), thickness=[90.0, 120.0], structure=structure)
    mee.solve()
    assert mee.formulation_assigned == '2ff'


def test_the_result_surface_is_the_same_object_the_other_paths_return():
    """A z-coupled solve must be indistinguishable downstream from any other.

    Three claims at once, because they fail together and are cheap to check together:

    * the **layout** — ``(n_orders_v, n_orders_u)``, y first, incident order at the centre, and
      ``result.order(m, n)`` indexing it by the order indices;
    * the **naming contract** (``docs/conventions.md``) — uppercase real (power), lowercase
      complex (amplitude), both incidence variants present;
    * that the extra orders are *real* orders rather than shape padding. A uniform layer
      cannot diffract, so every order but ``(0, 0)`` must be empty while ``R + T`` sums to one
      — a layout error would spread the power instead.

    "Empty" is round-off, not an identical zero: the off-orders come out of the same linear
    solve as the specular one and land at ~1e-16 in amplitude, hence ~1e-32 in power. The
    bound is set on the amplitude, where the error actually lives.
    """
    truncation_order = (2, 1)
    n_orders = (2 * truncation_order[0] + 1) * (2 * truncation_order[1] + 1)
    mee = solver(truncation_order=truncation_order, thickness=[180.0])
    res = mee.solve_conv(*uniform_slots([VOIGT], n_orders))

    assert mee.formulation_assigned == '4ff'
    assert res.s_inc is not None and res.p_inc is not None
    R, T = np.asarray(res.incident.R), np.asarray(res.incident.T)
    assert R.shape == (2 * truncation_order[1] + 1, 2 * truncation_order[0] + 1)
    assert R.dtype == np.float64 and np.asarray(res.incident.r_s).dtype == np.complex128

    assert float(R.sum() + T.sum()) == pytest.approx(1.0, abs=1e-12)
    assert res.order(0, 0).R + res.order(0, 0).T == pytest.approx(1.0, abs=1e-12)
    for m in range(-truncation_order[0], truncation_order[0] + 1):
        for n in range(-truncation_order[1], truncation_order[1] + 1):
            if (m, n) == (0, 0):
                continue
            o = res.order(m, n)
            assert max(abs(o.r_s), abs(o.r_p), abs(o.t_s), abs(o.t_p)) < 1e-14, \
                f'a uniform layer diffracted into order ({m}, {n})'


# ------------------------------------------------------------ the two formulations cross-check

def test_forcing_the_first_order_path_reproduces_a_uniform_anisotropic_answer():
    """Same scene, both formulations, on a tensor with an in-plane pair and a stack of three."""
    layers = [IN_PLANE.conj().T, np.diag([4.0, 4.0, 2.9]).astype(complex), IN_PLANE]
    thickness = [210.0, 75.0, 140.0]

    a = solver(thickness=thickness)
    res_a = a.solve_conv(*uniform_slots(layers, 1, z_row=False))
    b = solver(thickness=thickness)
    b.formulation = '4ff'
    res_b = b.solve_conv(*uniform_slots(layers, 1, z_row=False))

    assert (a.formulation_assigned, b.formulation_assigned) == ('2ff', '4ff')
    gap, where = worst_gap(channels(res_a), channels(res_b))
    assert gap < 1e-11, f'the two formulations disagree by {gap:.2e} on {where}'


@pytest.mark.parametrize('truncation_order', [(1, 1), (2, 2)], ids=['9 orders', '25 orders'])
def test_forcing_the_first_order_path_reproduces_a_patterned_grating(truncation_order):
    """The cross-check that matters: full convolution matrices, so the orders genuinely couple.

    A uniform layer's convolution matrices are scaled identities, and the two formulations
    could agree there for reasons that have nothing to do with a grating — every order
    decoupled, one mode pair each. Here they are dense and the second-order path's ``eig``
    sees a completely different matrix from the first-order path's.

    Conical incidence, since a plane of incidence off both axes is what makes the s/p rotation
    non-trivial, and that rotation is shared code fed from two different places.
    """
    structure = np.ones((2, 24, 24))
    structure[1, 6:18, 4:20] = 12.1104                      # 3.48**2
    structure[0, 10:14, :] = 4.41                           # 2.1**2

    a = solver(truncation_order=truncation_order, thickness=[85.0, 130.0], structure=structure)
    res_a = a.solve()
    b = solver(truncation_order=truncation_order, thickness=[85.0, 130.0], structure=structure)
    b.formulation = '4ff'
    res_b = b.solve()

    assert (a.formulation_assigned, b.formulation_assigned) == ('2ff', '4ff')
    gap, where = worst_gap(channels(res_a), channels(res_b))
    assert gap < 1e-10, f'the two formulations disagree by {gap:.2e} on {where}'


def test_the_transmitted_amplitude_is_not_merely_proportional():
    """The trap this route was warned about, made into an assertion.

    ``big_T`` feeds ``T_s`` / ``T_p`` directly, so a per-mode scale that cancels in ``R``
    survives in ``T``. The failure signature is transmission off by a *constant factor* — which
    ``R + T`` on a lossy stack would not flag and a same-shape comparison would not either. So
    the ratio is checked against 1 explicitly, on a stack with absorption so that ``R + T`` is
    not itself pinning the scale.
    """
    lossy = np.diag([2.4 + 0.05j, 3.1 + 0.05j, 2.7 + 0.05j]).astype(complex)
    layers = [lossy, IN_PLANE]
    thickness = [260.0, 140.0]

    a = solver(thickness=thickness)
    res_a = a.solve_conv(*uniform_slots(layers, 1, z_row=False))
    b = solver(thickness=thickness)
    b.formulation = '4ff'
    res_b = b.solve_conv(*uniform_slots(layers, 1, z_row=False))

    assert float(np.asarray(res_a.incident.R).sum() + np.asarray(res_a.incident.T).sum()) < 0.99
    for name in ('t_s', 't_p'):
        u = np.asarray(getattr(res_a.incident, name)).ravel()
        v = np.asarray(getattr(res_b.incident, name)).ravel()
        assert abs(u).max() > 1e-3, f'{name} is too small to carry the test'
        assert v[np.argmax(abs(u))] / u[np.argmax(abs(u))] == pytest.approx(1.0, abs=1e-10)


#: The half-spaces as PERMITTIVITY: (1.46 + 0.35j)**2 and (1.2 + 0.02j)**2, the same two media
#: this case has always used.
@pytest.mark.parametrize('eps_top,eps_bot', [(1.0, 2.0091 + 1.022j),
                                             (1.4396 + 0.048j, 2.0091 + 1.022j)],
                         ids=['absorbing substrate', 'absorbing on both sides'])
def test_the_two_formulations_agree_across_absorbing_half_spaces(eps_top, eps_bot):
    """The substrate seed carries a complex ``eps`` and a complex ``kz``, so it gets its own case.

    The first-order path writes that seed by hand rather than eigendecomposing the half-space,
    which means its square-root branch is taken from ``boundary_kz`` — the same function every
    other solve uses — and not from the flux rule the in-layer mode split applies. Getting the
    branch from somewhere else is exactly the mistake an absorbing half-space would expose and
    a lossless one would not.

    With an absorbing *superstrate* the incident power is convention-dependent and ``R``/``T``
    are not comparable to 1 (``docs/conventions.md``); the amplitudes still are, and
    :func:`channels` compares both.
    """
    layers = [np.diag([2.4 + 0.05j, 3.1 + 0.05j, 2.7 + 0.05j]).astype(complex), IN_PLANE]
    kw = dict(top=eps_top, bot=eps_bot, thickness=[260.0, 140.0])

    a = wavebit.rcwa(backend=0, truncation_order=(0, 0), period=PERIOD, wavelength=WL,
                       theta=np.deg2rad(28.0), phi=np.deg2rad(37.0), psi=0.6, **kw)
    b = wavebit.rcwa(backend=0, truncation_order=(0, 0), period=PERIOD, wavelength=WL,
                       theta=np.deg2rad(28.0), phi=np.deg2rad(37.0), psi=0.6, **kw)
    b.formulation = '4ff'

    res_a = a.solve_conv(*uniform_slots(layers, 1, z_row=False))
    res_b = b.solve_conv(*uniform_slots(layers, 1, z_row=False))
    gap, where = worst_gap(channels(res_a), channels(res_b))
    assert gap < 1e-11, f'the two formulations disagree by {gap:.2e} on {where}'


# ------------------------------------------------------------------------------ the refusals

def test_forcing_the_second_order_path_on_a_z_row_raises():
    """Naming the slot that cannot be held, rather than reporting a shape mismatch."""
    mee = solver(thickness=[180.0])
    mee.formulation = '2ff'
    with pytest.raises(ValueError, match='epxz_conv_all'):
        mee.solve_conv(*uniform_slots([VOIGT], 1))


def test_an_unknown_formulation_is_refused_at_the_setter():
    with pytest.raises(ValueError, match='formulation'):
        solver(thickness=[180.0]).formulation = '3ff'


def test_the_first_order_scattering_path_does_not_run_etm(monkeypatch):
    """An SMM request must use its cascade, including field reconstruction."""
    import importlib
    module = importlib.import_module('wavebit._core.solve_2d_4ff')
    def unexpected(*args, **kwargs):
        raise AssertionError('SMM called the ETM recursion')
    monkeypatch.setattr(module, '_etm_recursion', unexpected)
    mee = solver(thickness=[180.0], layer_algorithm='SMM', answers='field')
    result = mee.solve_conv(*uniform_slots([VOIGT], 1))
    assert mee.formulation_assigned == '4ff'
    assert np.isfinite(result.R).all()
    assert np.isfinite(mee.reconstruct_field(sample_x=2, sample_y=2, sample_z=3)).all()
    assert mee.scattering.S11.shape == (2, 2)


@pytest.mark.parametrize('z_row', [True, False], ids=['with a z row', 'without one'])
def test_the_unstabilised_transfer_cannot_be_reached_from_the_public_path(z_row):
    """``'TMM'`` is a diagnostic inside the 4ff module and must not be selectable.

    It is the plain ``Psi exp(...) Psi^-1`` transfer, kept there only to show that the
    stabilisation is load-bearing — it holds the growing and decaying halves in one matrix and
    loses relative precision like ``eps*exp(2x)``. Both routes refuse it, and they refuse it
    through their respective algorithm checks,
    which is why both are checked.
    """
    mee = solver(thickness=[180.0], layer_algorithm='TMM')
    slots = uniform_slots([VOIGT if z_row else IN_PLANE], 1, z_row=z_row)
    with pytest.raises(ValueError, match='TMM'):
        mee.solve_conv(*slots)


def test_the_scattering_algorithm_still_works_without_a_z_row():
    """The control: the ordinary two-field SMM path still works."""
    mee = solver(thickness=[180.0], layer_algorithm='SMM')
    res = mee.solve_conv(*uniform_slots([IN_PLANE], 1, z_row=False))
    assert mee.formulation_assigned == '2ff'
    assert np.isfinite(np.asarray(res.incident.R)).all()


def test_field_reconstruction_is_dispatched_on_the_formulation_not_the_grating_type(monkeypatch):
    """The branch ``reconstruct_field`` has to make, and the reason it cannot be the usual one.

    The first-order path sets ``_grating_type = 2`` as well — a z row is a choice of
    *layout* as much as of formulation — so a ``reconstruct_field`` that read only the grating
    type would hand a 4ff solve to the 2ff ``field_2d``. That is not a crash but a wrong
    answer: the two reconstructions read the same two state slots and one of them would
    silently reconstruct with the ``E_z`` line that assumes the z row is empty.

    Asserted by watching which reconstruction actually runs, rather than by trusting the
    numbers — ``tests/test_field_4ff.py`` is where the numbers are checked.
    """
    # `reconstruct_field` folded into `_core.engine` in C8, so the names it dispatches on
    # are looked up there now. Patching the backend module would patch a name nothing
    # reads any more -- the spy would see nothing and the assertion would read as a
    # dispatch failure rather than as a stale patch target.
    import wavebit._core.engine as dispatch_site

    mee = solver(thickness=[180.0], answers='field')
    mee.solve_conv(*uniform_slots([VOIGT], 1))
    assert mee.formulation_assigned == '4ff'
    assert mee.grating_type == 2, 'the premise: the grating type cannot tell them apart'

    seen = []

    def spy(name):
        real = getattr(dispatch_site, name)

        def wrapper(*args, **kwargs):
            seen.append(name)
            return real(*args, **kwargs)
        monkeypatch.setattr(dispatch_site, name, wrapper)

    spy('field_2d')
    spy('field_2d_4ff')

    cell = mee.reconstruct_field(sample_x=4, sample_y=4, sample_z=4)
    assert seen == ['field_2d_4ff'], seen
    assert np.isfinite(np.asarray(cell)).all()


def test_field_reconstruction_still_works_on_the_second_order_path():
    """The control for the guard above — it must not have cost the ordinary path anything."""
    structure = np.ones((1, 16, 16))
    structure[0, 4:12, 4:12] = 9.0
    mee = solver(truncation_order=(1, 1), thickness=[150.0], structure=structure, answers='field')
    mee.solve()
    cell = mee.reconstruct_field(sample_x=4, sample_y=4, sample_z=4)
    assert np.isfinite(np.asarray(cell)).all()


# --------------------------------------------------------------------------------- backends

@pytest.mark.parametrize('backend', [1, 2], ids=['jax', 'torch'])
def test_the_first_order_path_gives_the_same_answer_on_every_backend(backend):
    """All three agree — which is more than the module originally promised.

    ``split_forward_backward`` used to select modes with a boolean mask, and the note beside it
    read "numpy-only for now". Two things turned out to be wrong with that. Masks were already
    fine on all three backends eagerly and under ``jax.grad``; and the one case they genuinely
    broke, ``jax.jit``, was fixable rather than inherent — the mask is now a stable-sort
    permutation, which keeps the shapes static. See
    :func:`test_the_first_order_path_traces_under_jit`.
    """
    pytest.importorskip({1: 'jax', 2: 'torch'}[backend])
    ref = solver(thickness=[180.0]).solve_conv(*uniform_slots([VOIGT], 1))

    # Solver first, arrays second: importing the JAX backend is what lifts `jax_enable_x64`,
    # and an array built before that is silently complex64 (measured 6.6e-5 out, which reads
    # as a solver disagreement rather than as the single-precision result it is).
    mee = solver(thickness=[180.0], backend=backend)
    xp = pytest.importorskip({1: 'jax.numpy', 2: 'torch'}[backend])
    to = xp.asarray if backend == 1 else xp.as_tensor
    got = mee.solve_conv(*[to(s) for s in uniform_slots([VOIGT], 1)])

    assert mee.formulation_assigned == '4ff'
    gap, where = worst_gap(channels(ref), channels(got))
    assert gap < 1e-12, f'backend {backend} is {gap:.2e} from NumPy on {where}'


def test_the_first_order_path_traces_under_jit():
    """``jax.jit`` — the one thing the boolean-mask mode split genuinely could not do.

    A mask makes the split's output shape depend on the values being masked, and a tracer
    cannot answer that. The stable-sort permutation gives the same partition with the shapes
    fixed at trace time, so the whole route traces.

    Both compositions are checked, because they fail differently: ``jit`` alone traces with
    abstract values (which is what broke), and ``jit(grad)`` also has to get an eig gradient
    through the same permutation.
    """
    jax = pytest.importorskip('jax')
    solver(thickness=[180.0], backend=1)          # imports the backend, which lifts x64
    jnp = pytest.importorskip('jax.numpy')
    slots = [jnp.asarray(s) for s in uniform_slots([VOIGT], 1)]

    def run(thickness):
        mee = solver(thickness=[thickness], backend=1)
        return mee.solve_conv(*slots).incident.T.sum()

    eager = float(run(180.0))
    assert float(jax.jit(run)(jnp.asarray(180.0))) == pytest.approx(eager, abs=1e-12)
    assert float(jax.jit(jax.grad(run))(jnp.asarray(180.0))) == pytest.approx(
        float(jax.grad(run)(180.0)), rel=1e-9)


def test_the_gradient_through_the_first_order_path_is_the_real_one():
    """Tracing is not the same as differentiating correctly — check against a difference.

    ``jax.grad`` returning a number says the permutation did not break the tape. Whether it
    returns the *right* number is a separate question, and the mode split is exactly the sort
    of place a wrong one could come from: the sort is a data-dependent reordering, and a
    gradient that ignored it would still be finite and plausible.
    """
    jax = pytest.importorskip('jax')
    solver(thickness=[180.0], backend=1)
    jnp = pytest.importorskip('jax.numpy')
    slots = [jnp.asarray(s) for s in uniform_slots([VOIGT], 1)]

    def run(thickness):
        mee = solver(thickness=[thickness], backend=1)
        return mee.solve_conv(*slots).incident.T.sum()

    h = 1e-4
    fd = (float(run(180.0 + h)) - float(run(180.0 - h))) / (2 * h)
    assert float(jax.grad(run)(180.0)) == pytest.approx(fd, rel=1e-6)


def test_the_degenerate_mode_guard_still_fires_where_it_can():
    """The count check survived the rewrite — on concrete values, which is where it can run.

    ``M = i·I`` gives four modes with no flux at all, so every one of them falls through to the
    decay rule and lands forward: 4 where 2 were expected. That is the ambiguous layer the
    guard exists for, and refusing beats guessing which half a fluxless mode belongs to.

    Under ``jit`` there is nothing concrete to count and the guard is skipped — a real
    consequence of tracing, stated here rather than left to be discovered.
    """
    from wavebit._core.solve_2d_4ff import _forward_count, split_forward_backward

    with pytest.raises(ValueError, match='mode split found 4 forward modes'):
        split_forward_backward(np, np.eye(4, dtype=complex) * 1j, 1)

    assert _forward_count(np, np.array([1.0, 1.0, 0.0, 0.0])) == 2

    jax = pytest.importorskip('jax')
    jnp = pytest.importorskip('jax.numpy')
    import array_api_compat as ac
    traced = jax.jit(lambda k: 0 if _forward_count(ac.array_namespace(k), k) is None else 1)
    assert int(traced(jnp.array([1.0, 1.0, 0.0, 0.0]))) == 0, 'the guard must skip on a tracer'


# ---------------------------------------------------------------------------- the 1D layouts

def test_a_z_row_pulls_a_one_dimensional_structure_onto_the_2d_layout():
    """The 1D paths have no z-row slots, so a z row is also a choice of grating type.

    Without this the tensor would be handed to ``solve_1d_conical``, which takes three
    convolution stacks and would drop the other six on the floor — the silent reduction the
    whole dispatch exists to prevent.
    """
    structure = np.ones((1, 1, 20))
    structure[0, 0, 5:15] = 11.559999999999999
    mee = solver(truncation_order=(3, 0), thickness=[150.0], structure=structure)
    mee._assign_grating_type()
    assert mee.grating_type in (0, 1)      # what it would have taken on its own

    n_orders = 2 * 3 + 1
    mee.solve_conv(*uniform_slots([VOIGT], n_orders))
    assert (mee.grating_type, mee.formulation_assigned) == (2, '4ff')


# ------------------------------------------------- asking a 1D cell for y diffraction orders

_Y_INVARIANT = np.array([[[9.0, 9.0, 1.0, 1.0]]])          # one row: nothing varies in y
_Y_ROWS = np.repeat(_Y_INVARIANT, 2, axis=1)               # the same physics, on the 2D path
_1D_KW = dict(backend=0, top=1.0, bot=2.25, theta=0.3, phi=0.4, thickness=[0.2],
              period=[0.7, 0.7], wavelength=0.9)


@pytest.mark.parametrize('incidence', [
    dict(psi=0.4), dict(polarization=(1, -1j)), dict(psi=0.4, chi=0.2),
], ids=['psi', 'pair', 'chart'])
def test_y_orders_on_a_y_invariant_cell_are_refused_in_plain_words(incidence):
    """The refusal is about the structure, so it must not depend on how the light is written.

    It used to reach ``layer_modes_1d_conical`` and surface as ``operands could not be
    broadcast together with shapes (5,5) (15,15)`` — the ``Ky2`` block is built over
    ``n_orders_u * n_orders_v`` and the convolution matrices over ``n_orders_u``. Nothing in that names the
    mistake, and it fired inside the solve rather than at the request.
    """
    with pytest.raises(ValueError, match='asks for y diffraction orders'):
        wavebit.rcwa(structure=_Y_INVARIANT, truncation_order=[2, 1], **incidence, **_1D_KW).solve()


def test_the_refused_orders_are_the_ones_that_are_zero():
    """Why refusing is right rather than merely convenient: there is nothing there to return.

    Written with two identical rows the same structure takes the 2D path, which *does* keep
    the y orders — and returns them empty. So the 1D formulations are not withholding an
    answer, and clamping ``truncation_order[1]`` silently would hand back a different shape than asked for.
    """
    res = wavebit.rcwa(structure=_Y_ROWS, truncation_order=[2, 1], psi=0.4, **_1D_KW).solve()
    T = np.asarray(res.incident.T)
    assert T.shape == (3, 5)
    assert T[[0, 2], :].sum() < 1e-25, T                     # every n != 0 order is empty

    flat = np.asarray(wavebit.rcwa(structure=_Y_ROWS, truncation_order=[2, 0], psi=0.4,
                                   **_1D_KW).solve().incident.T).ravel()
    assert np.abs(T[1] - flat).max() < 1e-13                 # and n = 0 is the 1D answer


def test_max_order_y_zero_is_untouched():
    """The guard must not cost the ordinary 1D call anything."""
    res = wavebit.rcwa(structure=_Y_INVARIANT, truncation_order=[2, 0], psi=0.4, **_1D_KW).solve()
    assert np.asarray(res.incident.T).shape == (1, 5)
