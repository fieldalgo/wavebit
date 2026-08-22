"""The Popov-Neviere factorization for a *patterned anisotropic* layer.

``_nvf_assemble`` cannot serve a tensor. Its entire input is ``<eps>`` and ``<1/eps>``, and
``1/eps`` is not a thing a tensor has: Li's rule needs one discontinuous factor times one
continuous one, while ``D_x = exx Ex + exy Ey + exz Ez`` is a sum of three such products with no
per-term rule. :func:`~wavebit._core.factorization.pn_eps_conv_matrices` rearranges instead —
eliminate the discontinuous ``n.E`` in favour of the continuous ``n.D``, then every product has
the shape Laurent converges on. That function's docstring carries the algebra.

The external check lives in ``tests/references/test_fmmax_patterned_golden.py``: fmmax is the
only reference solver that both accepts a patterned complex off-diagonal and demonstrably uses
it. **This file is the part that needs no oracle**, and it is not the lesser half — the
uniform-layer gate below is what caught the first version of this code, which fmmax would have
passed within tolerance while being wrong.
"""
import numpy as np
import pytest

import wavebit
from wavebit._core.factorization import (
    fourier2d_piecewise_constant, fourier2d_pointwise, normal_vector_field,
    nvf_conv_matrices, pn_eps_conv_matrices)

WL, PERIOD, THICK = 620.0, 340.0, 185.0
N = 96


def pillar(value, background=1.0, grid=N):
    """Deliberately not square: wider in x than in y.

    A square pillar is invariant under the x/y swap, so it cannot see an axis mix-up — and an
    axis mix-up is the single most likely way to get this wrong (it is how the first comparison
    against fmmax failed, with the s/p split reversed).
    """
    a = np.full((grid, grid), background, dtype=complex)
    a[grid // 3:2 * grid // 3, grid // 4:2 * grid // 3] = value
    return a


def tensor(eps, gyro=None, ezz=None):
    """Pointwise ``(Ny, Nx, 3, 3)`` from scalar rasters. The *physical* tensor, unnegated."""
    e = np.zeros(np.shape(eps) + (3, 3), dtype=complex)
    e[..., 0, 0] = e[..., 1, 1] = eps
    e[..., 2, 2] = eps if ezz is None else ezz
    if gyro is not None:
        e[..., 0, 1], e[..., 1, 0] = gyro, -gyro
    return e


def solve(slots, eps, truncation_order, theta=0.0, phi=0.0, psi=0.0):
    epx, epy, epz_i, epxy, epyx = slots
    mee = wavebit.rcwa(backend=0, top=1.0, bot=1.0, theta=np.deg2rad(theta),
                       phi=np.deg2rad(phi), psi=np.deg2rad(psi), truncation_order=(truncation_order, truncation_order),
                       period=(PERIOD, PERIOD), wavelength=WL, thickness=[THICK],
                       structure=np.expand_dims(np.sqrt(eps), axis=0))
    return mee.solve_conv(np.stack([epx]), np.stack([epy]), np.stack([epz_i]),
                              np.stack([epxy]), np.stack([epyx]))


def pn(eps33, eps_scalar, truncation_order):
    nx, ny = normal_vector_field(eps_scalar)
    return pn_eps_conv_matrices(np, eps33, nx, ny, truncation_order, truncation_order)


def rt(res):
    return float(res.R.sum()), float(res.T.sum())


# ------------------------------------------------------------------------------- the backends

@pytest.mark.parametrize('name', ['jax', 'torch'])
def test_it_builds_and_differentiates_on_every_backend(name):
    """Same numbers and a finite gradient under jax and torch, not just numpy.

    Pinned because the failure is silent under numpy and loud everywhere else. The normal field
    arrives real and the tensor complex; NumPy promotes them on contact, so a version that
    never casts works perfectly on the backend it was written against and raises "expected
    scalar type Double but found ComplexDouble" the first time torch reaches the ``einsum``.
    That is exactly what the first version of ``_pn_fields`` did.

    The gradient is checked too, not just the forward build: this factorization exists to be
    optimised through, and a build that traces but detaches would pass a values-only test.
    """
    pytest.importorskip(name)
    from wavebit._core.backend import array_namespace

    eps = pillar(1.6 ** 2, grid=48)
    g = pillar(0.30j, 0.0, grid=48)
    e33 = tensor(eps, g)
    nx, ny = normal_vector_field(eps)
    ref = [np.asarray(m) for m in pn_eps_conv_matrices(np, e33, nx, ny, 4, 4)]

    if name == 'jax':
        import jax
        import jax.numpy as jnp
        jax.config.update('jax_enable_x64', True)
        args = (jnp.asarray(e33), jnp.asarray(nx), jnp.asarray(ny))
        xp = array_namespace(args[0])
        grad = jax.grad(lambda e: jnp.abs(
            pn_eps_conv_matrices(xp, e, args[1], args[2], 2, 2)[0]).sum())(args[0])
    else:
        import torch
        args = (torch.as_tensor(e33).clone().requires_grad_(True),
                torch.as_tensor(np.asarray(nx)), torch.as_tensor(np.asarray(ny)))
        xp = array_namespace(args[0])
        pn_eps_conv_matrices(xp, args[0], args[1], args[2], 2, 2)[0].abs().sum().backward()
        grad = args[0].grad

    def numpy(a):                     # torch refuses .numpy() on a grad-tracking tensor
        return np.asarray(a.detach() if hasattr(a, 'detach') else a)

    got = pn_eps_conv_matrices(xp, args[0], args[1], args[2], 4, 4)
    assert max(np.abs(numpy(a) - b).max() for a, b in zip(got, ref)) < 1e-12
    assert np.isfinite(numpy(grad)).all() and np.abs(numpy(grad)).max() > 0


# --------------------------------------------------------------- the gate that caught the bug

@pytest.mark.parametrize('truncation_order', [2, 4, 6])
@pytest.mark.parametrize('gyro', [None, 0.30j], ids=['isotropic', 'gyrotropic'])
def test_a_uniform_layer_comes_back_as_the_tensor_it_was_given(truncation_order, gyro):
    """No interface, so the factorization has nothing to do and must do nothing — exactly.

    This is the check that rejected the first version of this code, and it is worth
    understanding why it had teeth when an energy check and an fmmax comparison would not have.

    The rejected form built the projectors ``[[T]]``, ``[[N]]`` and the transfer ``[[M]]`` as
    separate convolution matrices and multiplied them. That looks like the same algebra, and
    pointwise it is. But the normal field is a *unit* field extended into the bulk, so it is
    discontinuous itself — ``[[T]] @ [[M]]`` is Laurent applied to two factors that jump in the
    same place, the one product Li's rule forbids. It conserved energy, it was rotation
    covariant, and it sat plausibly near fmmax. It was also **wrong by 0.5 on a uniform slab**,
    which no oracle-based test would have phrased that sharply.

    Doing the projector algebra pointwise and transforming once fixes it, and then this is
    exact rather than close: for ``eps = e I`` the two transformed fields satisfy
    ``[[A]] = e [[B]]`` identically, for *any* normal field whatsoever.
    """
    uni = np.full((N, N), 2.25, dtype=complex)
    g = None if gyro is None else np.full((N, N), gyro)
    epx, epy, epz_i, epxy, epyx = pn(tensor(uni, g), uni, truncation_order)

    n_orders = (2 * truncation_order + 1) ** 2
    I = np.eye(n_orders)
    assert np.abs(epx - 2.25 * I).max() < 1e-13
    assert np.abs(epy - 2.25 * I).max() < 1e-13
    assert np.abs(epz_i - I / 2.25).max() < 1e-13
    # slots hold the negation: `epxy` is -eps_xy
    want_xy = 0.0 if gyro is None else -gyro
    assert np.abs(epxy - want_xy * I).max() < 1e-13
    assert np.abs(epyx + want_xy * I).max() < 1e-13


def test_the_uniform_gate_is_not_vacuous():
    """The gate above must fail on a factorization that gets a patterned layer wrong.

    Stated because "uniform layer returns the tensor" reads like a triviality, and a check that
    cannot fail is worse than no check — it reports confidence it has not earned. Feeding a
    *patterned* layer through the same assertions has to break them.
    """
    eps = pillar(1.6 ** 2)
    epx = pn(tensor(eps), eps, 4)[0]
    n_orders = (2 * 4 + 1) ** 2
    assert np.abs(epx - np.mean(eps) * np.eye(n_orders)).max() > 1e-3


# ------------------------------------------------------- agreement with the path it generalizes

@pytest.mark.parametrize('n_in', [1.15, 1.30, 1.60])
def test_an_isotropic_tensor_reproduces_the_shipped_scalar_nvf(n_in):
    """With a scalar permittivity the new path and the old one solve the same problem.

    They are not the same formula — ``_nvf_assemble`` is a 1/2-symmetrized ``Delta @ C`` form
    and this one carries whole products through the transform — so they agree by *converging*,
    not identically. The tolerance scales with the contrast because that is what sets both
    truncation errors; a single number across all three would be slack on two of them.
    """
    eps = pillar(n_in ** 2)
    truncation_order = 8
    a = rt(solve(pn(tensor(eps), eps, truncation_order), eps, truncation_order, theta=8.0))
    nx, ny = normal_vector_field(eps)
    b = rt(solve(nvf_conv_matrices(np, eps, nx, ny, truncation_order, truncation_order), eps, truncation_order, theta=8.0))
    tol = 3e-5 * (n_in ** 2 - 1.0) ** 2
    assert max(abs(a[0] - b[0]), abs(a[1] - b[1])) < tol


def test_the_two_paths_agree_in_the_limit_where_they_have_to():
    """The claim above is agreement, so something has to watch a trend, not just one number.

    **Not the trend in truncation_order.** Measured, that gap does not fall: it sits near 1.3e-5 from truncation_order=6
    to truncation_order=14 on the strongest contrast here. Nor should it — two vector formulations differ at
    any finite truncation, and 1.3e-5 is an order tighter than the 1.8e-4 between fmmax's own
    two formulations on the same structure. An earlier version of this test asserted the truncation_order
    trend and was simply wrong about the physics.

    The limit that *is* forced is the contrast. As ``n_in -> 1`` the layer becomes uniform,
    where both paths reduce to ``<eps>`` exactly, so the gap must vanish with it — measured, as
    roughly the square of ``n^2 - 1``, falling from 2.3e-5 to 1.4e-8 across the range below.

    Which on its own would prove little, because the reflectance falls too. What carries the
    claim is that the gap falls *faster* than the answer does: 1600x against 95x over the same
    range, a factor of 17. A defect that scaled with the field rather than with the
    factorization would leave that ratio at 1, which is why the assertion is on the ratio and
    not on either number alone.
    """
    truncation_order = 8
    relative = []
    for n_in in (1.05, 1.6):
        eps = pillar(n_in ** 2)
        nx, ny = normal_vector_field(eps)
        a = rt(solve(pn(tensor(eps), eps, truncation_order), eps, truncation_order, theta=8.0))
        b = rt(solve(nvf_conv_matrices(np, eps, nx, ny, truncation_order, truncation_order), eps, truncation_order, theta=8.0))
        relative.append(abs(a[0] - b[0]) / abs(b[0]))

    assert relative[0] < relative[1] / 10, (
        f'the paths do not close faster than the field does: relative gaps {relative} '
        f'(ratio {relative[1] / relative[0]:.1f}, measured 17 when this was written)')


# ------------------------------------------------------------------------------- the z slot

def test_the_z_slot_is_the_direct_rule_inverted_not_the_inverse_rule():
    """``epz_i = <eps_zz>^-1``, never ``<1/eps_zz>``. The two are different rules.

    ``E_z`` is *tangential* to a vertical sidewall — the interface normal lies in the plane — so
    it is continuous there, and a continuous factor times a discontinuous one is exactly what
    Laurent's rule handles. The inverse rule answers a different continuity argument and does
    not apply.

    This test exists because a mutation survey said it had to. Swapping the rule changes the
    answer by 3.5e-6 at normal incidence and 2.4e-5 at 40 degrees — real, growing with angle,
    and invisible to every other test in this file: it preserves energy, it is *identical* on a
    uniform layer (``<1/e> = (1/e) I = <e>^-1`` when ``e`` is constant), and at the 8 degrees
    the rest of the file uses it lands well inside the fmmax band.

    Asserted against the two spellings directly rather than against ``nvf_conv_matrices``,
    which would be the obvious reference and is the wrong one: it obtains its permittivity
    coefficients the same way this slot does but closes over the projector as well, so a
    disagreement there would not name this slot. Which *transform* the two spellings are taken
    with is the separate question
    :func:`test_the_z_slot_reads_the_raster_as_the_staircase_it_is` settles; both are in closed
    form here so that this test sees only the rule.
    """
    eps = pillar(1.6 ** 2)
    nx, ny = normal_vector_field(eps)
    for truncation_order in (4, 8):
        got = pn_eps_conv_matrices(np, tensor(eps), nx, ny, truncation_order, truncation_order)[2]
        direct = np.linalg.inv(fourier2d_piecewise_constant(
            np, eps.astype(complex), truncation_order, truncation_order, inverse_rule=None))
        inverse = fourier2d_piecewise_constant(
            np, (1.0 / eps).astype(complex), truncation_order, truncation_order, inverse_rule=None)
        assert np.abs(got - direct).max() < 1e-12
        assert np.abs(got - inverse).max() > 1e-3, 'the two rules have stopped differing'


def test_the_z_slot_reads_the_raster_as_the_staircase_it_is():
    """The closed form, not the plain DFT — per-term accuracy, and nothing pulling against it.

    ``eps_zz`` on its own is a staircase, so ``fourier2d_piecewise_constant`` gives its
    coefficients *exactly*. A and B cannot have that (they mix eps with the sampled normal
    field), but they do not constrain this slot either: with ``nz = 0`` the projectors are block
    diagonal, so the z slot is never multiplied against them. It decouples, and the reading it
    has to match is the rest of the package's — every other permittivity raster in wavebit is
    read as the staircase it is, ``nvf_conv_matrices`` included, which mixes these same two
    transforms for these same reasons.

    **This line held the DFT until 2026-08-17, on a measurement that was reading a bug.** The
    argument was that the two transforms described different layers and that mixing them cost
    rotation covariance about z — 1.9e-14 against 5.2e-8. They did disagree, but about *where in
    the pixel* a value sits: the closed form places it at the cell centre (the
    ``exp(-i*pi*k/n)`` half of ``_pixel_correction``) and a bare DFT at the cell's left edge.
    That was a defect in the sampled transform, corrected in ``da4762a`` by
    :func:`~wavebit._core.factorization._half_pixel_phase`, and it took the closed-form arm to
    1.5e-14. Both arms hold the quarter turn now, so covariance cannot choose between them, and
    neither can energy conservation.

    What is left between them is the ``sinc`` alone — the box transform of a value held across
    its pixel, which a point sample never had. The first assertion below is what that means, and
    it is the reason for the switch: pixel-repeating a raster respells the *same* staircase, so
    an exact transform of it cannot move. The closed form does not (8.4e-17); the DFT moves by
    2.7e-4, because a finer grid of point samples is a different set of samples. That gap is the
    ``sinc`` at this truncation -- ``sinc(12/96)`` is 0.9974 -- so it widens with truncation_order and the
    1e-4 floor below is not a tight one to have picked.

    The quarter turn is kept on both arms, still as the ``_half_pixel_phase`` guard: strip that
    phase from either transform and the two origins part company again, the two arms land ~1e-8
    apart, and this fires.

    Accuracy at the solve is measured in ``docs/patterned-anisotropic-factorization.md`` §7
    rather than here — it needs a raster sweep against a fine-grid reference, which is far too
    slow for this file. The verdict: at fixed truncation_order and a frozen geometry the DFT arm carries 3.5x
    to 4.8x the closed form's error from 45 to 55 degrees, crossing over near 38 degrees, below
    which it leads by 5% of a term 30x smaller.
    """
    eps = pillar(1.6 ** 2)
    g = pillar(0.30j, 0.0)
    truncation_order = 6
    e33 = tensor(eps, g)

    # Exactness, stated as the invariance it *is*: the same staircase spelled on a 2x finer grid.
    def z_slot(raster, transform):
        return transform(np, raster[..., 2, 2], truncation_order, truncation_order)

    fine = np.repeat(np.repeat(e33, 2, axis=0), 2, axis=1)
    closed = [z_slot(r, lambda *a: fourier2d_piecewise_constant(*a, inverse_rule=None))
              for r in (e33, fine)]
    sampled = [z_slot(r, fourier2d_pointwise) for r in (e33, fine)]
    moved_closed = np.abs(closed[0] - closed[1]).max()
    moved_sampled = np.abs(sampled[0] - sampled[1]).max()
    assert moved_closed < 1e-13, (
        f'the closed form moved by {moved_closed:.2e} when the same staircase was respelled on '
        f'a finer grid; it is meant to be exact for it, so this is not a tolerance to relax')
    assert moved_sampled > 1e-4, (
        f'the DFT arm no longer moves under pixel repetition ({moved_sampled:.2e}), so the two '
        f'transforms have stopped differing and this test has stopped comparing anything')

    # The shipped slot is the closed-form one.
    shipped = pn(e33, eps, truncation_order).epz_i
    assert np.abs(shipped - np.linalg.inv(closed[0])).max() < 1e-13

    def rotated_pair(slot_transform):
        out = []
        for e, gg, psi in ((eps, g, 30.0), (np.rot90(eps), np.rot90(g), 120.0)):
            slots = list(pn(tensor(e, gg), e, truncation_order))
            slots[2] = np.linalg.inv(z_slot(tensor(e, gg), slot_transform))
            out.append(rt(solve(slots, e, truncation_order, psi=psi)))
        return max(abs(out[0][0] - out[1][0]), abs(out[0][1] - out[1][1]))

    shipped_turn = rotated_pair(lambda *a: fourier2d_piecewise_constant(*a, inverse_rule=None))
    sampled_turn = rotated_pair(fourier2d_pointwise)
    assert shipped_turn < 1e-12, f'the shipped z slot broke the quarter turn: {shipped_turn:.2e}'
    assert sampled_turn < 1e-12, (
        f'the DFT z slot breaks the quarter turn by {sampled_turn:.2e}; the two arms agreed to '
        f'1.5e-14 once both transforms placed a pixel value at the same point, so this is the '
        f'half-pixel cell-centre phase having gone missing from one of them')


def test_the_z_slot_reaches_the_answer_at_a_steep_angle():
    """The equality above is a unit check; this is what says the slot is wired to the physics.

    ``epz_i`` enters only through ``Kx epz_i Kx`` and its partners, so at normal incidence it
    barely moves and a test sited there would pass with the slot mis-built. At 40 degrees it
    carries several times more weight.
    """
    eps = pillar(1.6 ** 2)
    g = pillar(0.30j, 0.0)
    truncation_order = 8
    slots = list(pn(tensor(eps, g), eps, truncation_order))
    good = rt(solve(slots, eps, truncation_order, theta=40.0, psi=30.0))

    slots[2] = np.linalg.inv(np.eye(slots[2].shape[0]) * np.mean(eps))   # a plausible wrong slot
    bad = rt(solve(slots, eps, truncation_order, theta=40.0, psi=30.0))
    assert abs(good[0] - bad[0]) > 1e-5, 'the z slot is not reaching the answer here'


# ------------------------------------------------------------------------ theorems, not oracles

@pytest.mark.parametrize('truncation_order, tol', [(4, 5e-5), (8, 5e-6), (12, 2e-6)])
def test_a_hermitian_tensor_conserves_energy_however_it_is_patterned(truncation_order, tol):
    """``eps_xy = +ig`` with ``eps_yx = -ig`` is Hermitian, hence lossless.

    Losslessness follows from the tensor being Hermitian at every pixel and has nothing to do
    with the pattern, so this holds no matter what the factorization does to the geometry —
    which is what makes it a check the factorization cannot talk its way out of.

    The tolerance is per-order and measured (the residual runs 1.7e-5 / 1.7e-6 / 5.8e-7 at
    these three), not one loose number reused. A single tolerance wide enough for truncation_order=4 would
    leave truncation_order=12 free to regress by an order without anything noticing.
    """
    eps = pillar(1.6 ** 2)
    g = pillar(0.30j, 0.0)
    r, t = rt(solve(pn(tensor(eps, g), eps, truncation_order), eps, truncation_order, theta=8.0, psi=30.0))
    assert abs(r + t - 1.0) < tol


def test_energy_closes_better_as_the_truncation_is_relaxed():
    """The residual above is truncation, so it has to shrink. If it plateaus it is a defect.

    Worth pinning separately: an early arrangement of this factorization conserved energy to
    ~1e-2 at every order, and a fixed tolerance would have called that a pass.
    """
    eps = pillar(1.6 ** 2)
    g = pillar(0.30j, 0.0)
    drift = []
    for truncation_order in (2, 4, 8):
        r, t = rt(solve(pn(tensor(eps, g), eps, truncation_order), eps, truncation_order, theta=8.0, psi=30.0))
        drift.append(abs(r + t - 1.0))
    assert drift[-1] < drift[0] / 10, f'not improving with truncation_order: {drift}'


def test_it_is_covariant_under_a_quarter_turn_about_z():
    """Rotate structure and illumination together; the answer must follow.

    A theorem-oracle, and the one that reaches where no solver can: it needs no reference
    implementation, only that the same physics was set up twice. Normal incidence, because
    there the azimuth drops out and the rotation is unambiguous — at oblique incidence the
    raster rotation has to be matched against the ``phi`` convention, and the *shipped* NVF path
    shows the same residual there, so that combination measures the harness rather than this
    code.

    A gyrotropic ``g`` along z is invariant under a rotation about z, so the tensor components
    are unchanged and rotating the raster is the whole transformation. The pillar is not square,
    so the rotated problem is genuinely a different array.
    """
    eps = pillar(1.6 ** 2)
    g = pillar(0.30j, 0.0)
    truncation_order = 6

    a = rt(solve(pn(tensor(eps, g), eps, truncation_order), eps, truncation_order, psi=30.0))
    eps_r = np.rot90(eps)
    b = rt(solve(pn(tensor(eps_r, np.rot90(g)), eps_r, truncation_order), eps_r, truncation_order, psi=30.0 + 90.0))

    assert max(abs(a[0] - b[0]), abs(a[1] - b[1])) < 1e-12


# ------------------------------------------------------------------------------- the n = 0 pixels

def test_the_normal_field_vanishing_is_handled_rather_than_divided_by():
    """``normal_vector_field`` returns ``(0, 0)`` on the skeleton of the distance transform.

    The rearrangement divides by ``a = n.eps.n``, which is zero there, and the first working
    version produced NaN across the whole matrix from those few pixels. The fix is not a patch:
    with ``n = 0`` the numerators ``p``, ``q`` and ``n n^T`` all vanish and ``T = I``, so
    substituting ``a = 1`` makes the formulas return ``A = eps`` and ``B = I`` on their own —
    the direct rule, which is exactly what a zero projector does in ``_nvf_assemble``.

    Pinned with a shape whose skeleton is unmissable: concentric squares put a ridge in the
    middle of every arm.
    """
    eps = np.full((N, N), 1.0, dtype=complex)
    eps[N // 6:5 * N // 6, N // 6:5 * N // 6] = 2.56
    eps[N // 3:2 * N // 3, N // 3:2 * N // 3] = 1.0

    nx, ny = normal_vector_field(eps)
    assert (nx ** 2 + ny ** 2 == 0).any(), 'this shape was supposed to have a skeleton'

    for mat in pn(tensor(eps, pillar(0.20j, 0.0)), eps, 6):
        assert np.isfinite(mat).all()


def test_a_null_direction_of_the_tensor_is_not_quietly_patched():
    """The ``a = 1`` substitution is guarded on the *field* being zero, not on ``a`` being zero.

    They are different failures. A vanishing normal is geometry, expected, and has an exact
    limit. An ``a = n.eps.n`` that vanishes under a *nonzero* normal is a null direction of the
    tensor itself — a real singularity of the method — and substituting there would bury it.
    So it must come through as a non-finite result rather than a plausible number.
    """
    eps = pillar(1.6 ** 2)
    e33 = tensor(eps)
    nx, ny = normal_vector_field(eps)

    # Make n.eps.n vanish where the normal points along +x, without zeroing the normal.
    col = np.argmax(np.abs(nx).max(axis=0))
    e33[:, col, 0, 0] = 0.0
    e33[:, col, 1, 1] = 0.0

    hit = (nx[:, col] ** 2 > 0.99)
    assert hit.any(), 'this column was supposed to carry an x-pointing normal'

    with np.errstate(divide='ignore', invalid='ignore'):    # the division is the point here
        out = pn_eps_conv_matrices(np, e33, nx, ny, 4, 4)
    assert not all(np.isfinite(m).all() for m in out)
