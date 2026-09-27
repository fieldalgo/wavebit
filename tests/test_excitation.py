"""Multimode incidence — several orders lit at once, in one solve.

``excitation={0: 1.0, 1: 0.6-0.8j}`` says the illumination arrives on more than one
diffraction order, coherently. What it is *for* is the superposition: a single non-specular
order is the same plane wave as the specular one at the angle whose ``kx`` matches, so lighting
one order buys nothing but a relabelling. Several at once cannot be had from one ordinary
solve, and the alternative — one solve per beam, indices realigned, amplitudes added — costs a
solve per beam for the same answer.

The oracle is the SMM's scattering matrix. ``S11 @ c_inc`` is by definition the reflected
amplitude for the incident column ``c_inc``, built by machinery that shares nothing with the
ETM's boundary match, so the two agreeing is a real check rather than a restatement. Beside it:
energy on a lossless stack, and the two connecting algorithms against each other.

What is *not* tested here is that a superposition is more accurate than solving beam by beam —
it is not. Measured at two and three beams across seven basis sizes, neither is systematically
better; the truncation error of each dominates their difference from `truncation_order` 8 up. The gain
is the solve count.
"""
import numpy as np
import pytest

import wavebit

WL, PERIOD, THICK = 600.0, 2000.0, [130.0, 220.0]
#: `lambda/period = 0.3`, so orders -3..+3 are all inside the light line at this angle --
#: which is what makes a three-beam excitation legal here at all (see the refusal below).
COMMON = dict(backend=0, truncation_order=(6, 0), period=(PERIOD, PERIOD), wavelength=WL,
              theta=0.25, thickness=THICK, top=1.0, bot=2.25, answers='amplitude')
BEAMS = {-1: 0.4 + 0.2j, 0: 1.0, 1: 0.6 - 0.8j}
#: Deliberately *not* symmetric about the specular order, so the widened window's centre is
#: non-zero. A symmetric excitation leaves it at 0, where a `Result` that ignored the centre
#: entirely would still index correctly -- and two mutants proved exactly that.
OFF_CENTRE = {0: 1.0, 2: 0.5 - 0.3j}


def _structure():
    eps = np.ones((2, 1, 64), dtype=complex)
    eps[1, 0, :26] = 6.25
    eps[0, 0, 20:] = 4.0
    return eps


def _solve(excitation=None, algo='ETM', phi=None, tetm='te', **over):
    kw = dict(COMMON, phi=phi, tetm=tetm, layer_algorithm=algo, excitation=excitation)
    kw.update(over)
    solver = wavebit.rcwa(**kw)
    solver.structure = _structure()
    return solver, solver.solve()


def _column(solver, excitation):
    """The incident column, on whatever window the solver actually retained.

    Not `order + truncation_order`: an excitation widens the window and can move its centre, so the
    position of order `m` is `m - centre + half_width`. Reading it off the solver rather than
    recomputing it is what keeps this an oracle instead of a second implementation.
    """
    (centre, half), _ = solver.order_window
    column = np.zeros((2 * half + 1, 1), dtype=complex)
    for order, amplitude in excitation.items():
        column[order - centre + half, 0] = amplitude
    return column


# ------------------------------------------------------------------------------ the oracle

@pytest.mark.parametrize('tetm', ['te', 'tm'])
@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_a_superposition_is_the_scattering_matrix_applied_to_it(algo, tetm):
    """``S11 @ c_inc`` and ``S21 @ c_inc``, which is what an incident column *means*.

    The strongest oracle available, and an independent one: the SMM builds ``S`` by cascading
    layer matrices and never performs the ETM's boundary match, so an agreement here is two
    constructions meeting rather than one restated. Parametrised over the algorithm because
    the excitation has to reach both incident-column builders, which are separate code.

    TM as well: excitation x scattering matrix x the p port is the crossing where a lost
    per-order sign or a lost port minus would live. The p block holds port variables, so it
    returns ``-r_p`` and ``t_p`` times ``(n/mu)_exit / (n/mu)_lit`` (``docs/conventions.md``).
    """
    # The reference carries the same excitation, so its window -- and therefore its S -- is
    # the one the ETM solve is built on. `S` itself does not depend on the excitation, only on
    # the window it widened to.
    reference, _ = _solve(BEAMS, algo='SMM', tetm=tetm)
    S = reference.scattering
    _, got = _solve(BEAMS, algo=algo, tetm=tetm)
    column = _column(reference, BEAMS)
    # top = 1.0 (lit), bot = 2.25: n/mu = 1 and 1.5.
    readings = ((('S11', 'r_s', 1.0), ('S21', 't_s', 1.0)) if tetm == 'te' else
                (('S11', 'r_p', -1.0), ('S21', 't_p', 1.5)))
    for block, reported, factor in readings:
        want = (np.asarray(getattr(S, block)) @ column).ravel()
        have = factor * np.asarray(getattr(got.incident, reported)).ravel()
        assert np.abs(have - want).max() < 1e-12, block


@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('tetm', ['te', 'tm'])
def test_tetm_weights_mean_what_the_coupled_weights_mean(tetm, algo, side):
    """A weight multiplies its order's own ``s_hat`` (TE) or ``p_hat`` (TM), on every path.

    ``BEAMS`` lights order ``-1`` at ``kx < 0`` beside orders ``0`` and ``+1`` at ``kx > 0``
    (asserted), so the lit set straddles ``kx = 0``: on the pure 1D path that makes the ``-1``
    beam arrive polarized along ``-y`` (TE), because that is its ``s_hat``. Until 2026-09-27 the
    pure path multiplied ``+y`` into every lit order instead, which is different light -- the
    two solves then disagreed in efficiency, not only in phase.

    Held to the conical path at ``phi = 0`` per order and on the field (incoming boundary wave
    included, where a column signed twice would show), and anchored to a number: the conical
    top-lit ``sum(R)``, 0.370091 for TE and 0.158687 for TM, measured on the conical path
    before this change and unchanged by it. The pure path's old answer was 0.493324 and
    0.217085.
    """
    # Lit from the glass below, `theta = 0.25` puts order -1 at kx = +0.071; 0.15 straddles.
    over = dict(incidence_side=side, answers='field', theta=0.25 if side == 'top' else 0.15)
    pure, a = _solve(BEAMS, algo=algo, tetm=tetm, **over)
    _, b = _solve(BEAMS, algo=algo, tetm=None, phi=0.0,
                  polarization=(1, 0) if tetm == 'te' else (0, 1), **over)
    kx = np.real(np.asarray(pure.order_kxy()[0])).ravel()
    (centre, half), _ = pure.order_window
    assert kx[-1 - centre + half] < 0 < kx[0 - centre + half] < kx[1 - centre + half], kx

    for field in ('R', 'T', 'r_s', 'r_p', 't_s', 't_p', 'power_entering'):
        x, y = np.asarray(getattr(a.incident, field)), np.asarray(getattr(b.incident, field))
        assert np.abs(x - y).max() < 1e-12, (field, np.abs(x - y).max())
    points = dict(sample_x=9, sample_y=1, sample_z=3)
    x, y = np.asarray(a.reconstruct_field(**points)), np.asarray(b.reconstruct_field(**points))
    assert np.abs(x - y).max() < 1e-12 * np.abs(y).max(), 'reconstruct_field'
    for face in ('top', 'bottom'):
        x = np.asarray(a.boundary_fields(face, sample_x=9, part='incoming'))
        y = np.asarray(b.boundary_fields(face, sample_x=9, part='incoming'))
        assert np.abs(x - y).max() < 1e-12, f'incoming at {face}'
    if side == 'top':
        anchor = 0.370091 if tetm == 'te' else 0.158687
        assert float(np.asarray(a.incident.R).sum()) == pytest.approx(anchor, abs=1e-6)


@pytest.mark.parametrize('phi, tetm, name', [(None, 'te', '1d_te'), (None, 'tm', '1d_tm'),
                                             (0.3, None, 'conical')])
def test_the_two_connecting_algorithms_agree_on_a_superposition(phi, tetm, name):
    """The ETM matches a boundary and the SMM cascades; a superposition must not part them."""
    _, a = _solve(BEAMS, algo='ETM', phi=phi, tetm=tetm)
    _, b = _solve(BEAMS, algo='SMM', phi=phi, tetm=tetm)
    for field in ('R', 'T', 'r_s', 't_s', 'r_p', 't_p'):
        assert np.abs(np.asarray(getattr(a.incident, field))
                      - np.asarray(getattr(b.incident, field))).max() < 1e-12, (name, field)


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('phi, tetm', [(None, 'te'), (None, 'tm'), (0.3, None)])
def test_a_lossless_stack_still_closes_under_a_superposition(algo, phi, tetm):
    """``R + T == 1``, which is the whole point of getting the denominator right.

    Every efficiency is divided by the *incident* flux, and for a superposition that is the sum
    over the lit orders rather than one wave's. Getting it wrong leaves a number that looks
    like an efficiency and is not; energy is what notices.
    """
    _, r = _solve(BEAMS, algo=algo, phi=phi, tetm=tetm)
    total = float(np.asarray(r.incident.R).sum() + np.asarray(r.incident.T).sum())
    assert total == pytest.approx(1.0, abs=1e-12)


def test_lighting_only_the_specular_order_is_the_ordinary_solve():
    """The general path has to reduce, or the declaration would change the physics.

    Not bit-identical, and deliberately so: the undeclared case keeps ``n_top*cos(theta)`` and
    this one sums ``Re(kz)`` over one order, which are the same quantity spelled two ways and
    differ by an ulp on half of all angles. Agreement to 1e-14 is the claim; bit-identity is
    the *other* test, and it is about the undeclared path staying untouched.
    """
    _, plain = _solve()
    _, spelled = _solve({0: 1.0})
    for field in ('R', 'T', 'r_s', 't_s'):
        a, b = np.asarray(getattr(plain.incident, field)), np.asarray(getattr(spelled.incident, field))
        assert np.abs(a - b).max() < 1e-14, field


def test_declaring_nothing_leaves_the_solve_byte_for_byte_what_it_was():
    """No excitation named, no expression changed — the reason for the structural branch.

    The incident column is built by one function now, but the *normalisation* is not: the
    undeclared case keeps the spelling it had, because a golden should not move by a rounding
    step for a feature the scene does not use (the argument
    :func:`~wavebit._core.engine._half_space_index` makes for an unnamed ``mu``). This pins
    that, on the paths and angles where the two spellings differ.
    """
    for tetm, phi in (('te', None), ('tm', None), (None, 0.3)):
        for theta in (0.0, 0.3, 1.2):
            solver = wavebit.rcwa(**dict(COMMON, theta=theta, phi=phi, tetm=tetm))
            solver.structure = _structure()
            named = wavebit.rcwa(**dict(COMMON, theta=theta, phi=phi, tetm=tetm,
                                        excitation=None))
            named.structure = _structure()
            a, b = solver.solve(), named.solve()
            for field in ('R', 'T', 'r_s', 't_s'):
                assert np.array_equal(np.asarray(getattr(a.incident, field)),
                                      np.asarray(getattr(b.incident, field))), (tetm, theta, field)


def test_the_result_indexes_by_order_on_a_widened_window():
    """``order(m)`` reads the array, so it has to describe the array and not the declaration.

    The window widens past the declared ``truncation_order``, which means the array grows and order 0
    stops sitting at its middle. A ``Result`` carrying the *declared* half-width then indexed
    one order out -- and returned a plausible number rather than raising, which is the shape of
    defect this repo hunts. Checked against the raw array, which is the only thing that cannot
    be wrong the same way.
    """
    solver, result = _solve(OFF_CENTRE)
    (centre, half), _ = solver.order_window
    assert result.order_window[0] == (centre, half)
    assert half > COMMON['truncation_order'][0]                 # the window really did widen
    assert centre != 0                                   # ...and it is not centred on 0

    raw = np.asarray(result.incident.R).ravel()
    for order in range(centre - half, centre + half + 1):
        assert float(result.order(order).R) == pytest.approx(
            float(raw[order - centre + half]), abs=0, rel=0), order


def test_an_order_outside_the_widened_window_is_still_refused():
    """And the bound moves with the window, rather than staying at the declaration."""
    solver, result = _solve(OFF_CENTRE)
    (centre, half), _ = solver.order_window
    result.order(centre + half)                          # the edge is in
    with pytest.raises(IndexError, match='outside the retained set'):
        result.order(centre + half + 1)


# -------------------------------------------------------------------------------- refusals

def test_lighting_an_evanescent_order_is_refused():
    """It carries no power, so it would drive the stack while paying nothing into the flux.

    Left in, ``R + T`` exceeds one on a passive stack -- measured 8.87 on a thirteen-order
    excitation before this refusal existed -- because the denominator counts only the
    propagating part of an input the numerator counts all of. That is not a wrong number so
    much as a different quantity wearing the name "efficiency". An evanescent excitation is a
    near-field source, which this library does not model.
    """
    with pytest.raises(ValueError, match='evanescent'):
        _solve({0: 1.0, 5: 0.5}, period=(600.0, 600.0))


def test_lighting_an_order_outside_the_declared_max_order_is_refused():
    """The window widens around the lit orders, so this rule is what bounds the widening.

    Without it, `{0: 1.0, 9: 0.5}` at `truncation_order=6` would not be an error at all -- it would
    quietly retain `[-6, 15]`, twenty-two orders where six were asked for. Refusing keeps
    `truncation_order` meaning the orders being modelled, and caps the widening at twice its
    half-width.
    """
    with pytest.raises(ValueError, match='outside the declared'):
        _solve({0: 1.0, 9: 0.5})


def test_an_out_of_plane_order_on_a_one_dimensional_solve_is_refused():
    """``truncation_order[1] == 0`` keeps the single ``n = 0`` order, so ``(0, 1)`` names nothing."""
    with pytest.raises(ValueError, match='n = 0'):
        _solve({(0, 0): 1.0, (0, 1): 0.5})


@pytest.mark.parametrize('bad, match', [
    ({}, 'empty'),
    ([1.0, 0.5], 'dict'),
    ({'first': 1.0}, 'diffraction order'),
])
def test_an_excitation_that_is_not_orders_and_amplitudes_is_refused(bad, match):
    with pytest.raises((TypeError, ValueError), match=match):
        _solve(bad)


# ------------------------------------------------------------------------ design variables

def test_an_incident_amplitude_carries_a_gradient():
    """The weights are values, not structure, so they differentiate like any other input.

    Which is the reason the keys and the values are read differently: a key decides which
    entries of the incident column exist, and structure cannot be traced, while an amplitude
    is an ordinary design variable. Checked against a difference quotient because an autodiff
    number agreeing with itself proves nothing.
    """
    jax = pytest.importorskip('jax')
    jnp = jax.numpy

    def reflectance(scale):
        solver = wavebit.rcwa(**dict(COMMON, backend=1, phi=None, tetm='te',
                                     excitation={0: 1.0, 1: scale * (0.6 - 0.8j)}))
        solver.structure = jnp.asarray(_structure())
        return jnp.sum(solver.solve().incident.R)

    x = jnp.asarray(1.0)
    step = 1e-6
    analytic = float(jax.grad(reflectance)(x))
    quotient = float((reflectance(x + step) - reflectance(x - step)) / (2 * step))
    assert analytic == pytest.approx(quotient, rel=1e-5)
    assert abs(analytic) > 1e-6                      # and it is not trivially zero


def test_a_superposition_survives_a_jit_boundary():
    """The keys are static and the values are traced, which is what makes this compile."""
    jax = pytest.importorskip('jax')
    jnp = jax.numpy

    def reflectance(a1):
        solver = wavebit.rcwa(**dict(COMMON, backend=1, phi=None, tetm='te',
                                     excitation={0: 1.0, 1: a1}))
        solver.structure = jnp.asarray(_structure())
        return jnp.sum(solver.solve().incident.R)

    a1 = jnp.asarray(0.6 - 0.8j)
    assert float(jax.jit(reflectance)(a1)) == pytest.approx(float(reflectance(a1)), rel=1e-10)


def test_a_superposition_survives_the_solver_itself_crossing_the_boundary():
    """The boundary the two tests above do not cross, and the one production uses.

    Both of them build the solver *inside* the traced function, so the only thing that crosses
    the jit boundary is the amplitude. The solver is constructed within the trace and holds the
    tracer directly -- it is never flattened, so the pytree is never consulted, and both tests
    compile and pass while `_tree_flatten` carries no `excitation` at all.

    That is how it was: `excitation` was in neither `CHILD_NAMES` nor `aux_data`. The path that
    matters is the other direction -- `RCWAJax.solve_field` is `jax.jit`-decorated at class
    level with `self` at argnum 0 and not static, so every call flattens the solver and rebuilds
    it through `__init__`. A multimode scene came back with `excitation = None` and answered the
    single-wave problem: on a 1D grating, R read 0.148814934 against eager's 0.164651118, and
    0.148814934 is exactly what that scene answers with no excitation at all.

    So: pass the solver, do not build it inside.
    """
    jax = pytest.importorskip('jax')
    jnp = jax.numpy

    solver, eager = _solve(BEAMS, backend=1, phi=None, tetm='te')

    @jax.jit
    def reflectance(s):                                # `s` is flattened on the way in
        return jnp.sum(s.solve().incident.R)

    assert float(reflectance(solver)) == pytest.approx(float(jnp.sum(eager.incident.R)), rel=1e-10)

    # And the same scene with the excitation dropped must NOT agree -- otherwise the assertion
    # above is satisfied by a solver that ignored the superposition on both sides.
    plain, _ = _solve(None, backend=1, phi=None, tetm='te')
    assert float(reflectance(plain)) != pytest.approx(float(jnp.sum(eager.incident.R)), rel=1e-6), (
        'the lit and unlit scenes answer the same number, so this test cannot see the drop')


def test_the_jitted_field_path_keeps_the_superposition():
    """The production path, end to end: `solve_field` is where the drop actually bit.

    `solve_field_eager` and `solve_field` differ only in whether the solver crosses a jit
    boundary, so they are each other's oracle. With the excitation dropped they parted by 1.22
    in the field cell.
    """
    jax = pytest.importorskip('jax')

    kw = dict(COMMON, backend=1, phi=None, tetm='te', answers='field', excitation=BEAMS)
    solver = wavebit.rcwa(**kw)
    solver.structure = _structure()

    sample = dict(sample_x=8, sample_y=1, sample_z=4)
    _, eager = solver.solve_field_eager(**sample)
    _, jitted = solver.solve_field(geometry=solver.freeze_geometry(), **sample)

    a = [np.asarray(x) for x in jax.tree_util.tree_leaves(eager)]
    b = [np.asarray(x) for x in jax.tree_util.tree_leaves(jitted)]
    assert [x.shape for x in a] == [x.shape for x in b]
    assert max(np.max(np.abs(x - y)) for x, y in zip(a, b)) < 1e-10
