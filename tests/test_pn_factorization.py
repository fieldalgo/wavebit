"""The Popov-Neviere factorization for a *patterned anisotropic* layer.

``_nvf_assemble`` cannot serve a tensor. Its entire input is ``<eps>`` and ``<1/eps>``, and
``1/eps`` is not a thing a tensor has: Li's rule needs one discontinuous factor times one
continuous one, while ``D_x = exx Ex + exy Ey + exz Ez`` is a sum of three such products with no
per-term rule. :func:`~wavebit._core.factorization.pn_conv_matrices` rearranges instead —
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
    normal_vector_field, nvf_conv_matrices, pn_conv_matrices)

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


def solve(slots, eps, fto, theta=0.0, phi=0.0, psi=0.0):
    epx, epy, epxy, epyx, epz_i = slots
    mee = wavebit.rcwa(backend=0, n_top=1.0, n_bot=1.0, theta=np.deg2rad(theta),
                       phi=np.deg2rad(phi), psi=np.deg2rad(psi), fto=(fto, fto),
                       period=(PERIOD, PERIOD), wavelength=WL, thickness=[THICK],
                       ucell=np.expand_dims(np.sqrt(eps), axis=0))
    return mee.solve_for_conv(WL, np.stack([epx]), np.stack([epy]), np.stack([epz_i]),
                              np.stack([epxy]), np.stack([epyx]))


def pn(eps33, eps_scalar, fto):
    nx, ny = normal_vector_field(eps_scalar)
    return pn_conv_matrices(np, eps33, nx, ny, fto, fto)


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
    ref = [np.asarray(m) for m in pn_conv_matrices(np, e33, nx, ny, 4, 4)]

    if name == 'jax':
        import jax
        import jax.numpy as jnp
        jax.config.update('jax_enable_x64', True)
        args = (jnp.asarray(e33), jnp.asarray(nx), jnp.asarray(ny))
        xp = array_namespace(args[0])
        grad = jax.grad(lambda e: jnp.abs(
            pn_conv_matrices(xp, e, args[1], args[2], 2, 2)[0]).sum())(args[0])
    else:
        import torch
        args = (torch.as_tensor(e33).clone().requires_grad_(True),
                torch.as_tensor(np.asarray(nx)), torch.as_tensor(np.asarray(ny)))
        xp = array_namespace(args[0])
        pn_conv_matrices(xp, args[0], args[1], args[2], 2, 2)[0].abs().sum().backward()
        grad = args[0].grad

    def numpy(a):                     # torch refuses .numpy() on a grad-tracking tensor
        return np.asarray(a.detach() if hasattr(a, 'detach') else a)

    got = pn_conv_matrices(xp, args[0], args[1], args[2], 4, 4)
    assert max(np.abs(numpy(a) - b).max() for a, b in zip(got, ref)) < 1e-12
    assert np.isfinite(numpy(grad)).all() and np.abs(numpy(grad)).max() > 0


# --------------------------------------------------------------- the gate that caught the bug

@pytest.mark.parametrize('fto', [2, 4, 6])
@pytest.mark.parametrize('gyro', [None, 0.30j], ids=['isotropic', 'gyrotropic'])
def test_a_uniform_layer_comes_back_as_the_tensor_it_was_given(fto, gyro):
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
    epx, epy, epxy, epyx, epz_i = pn(tensor(uni, g), uni, fto)

    ff = (2 * fto + 1) ** 2
    I = np.eye(ff)
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
    ff = (2 * 4 + 1) ** 2
    assert np.abs(epx - np.mean(eps) * np.eye(ff)).max() > 1e-3


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
    fto = 8
    a = rt(solve(pn(tensor(eps), eps, fto), eps, fto, theta=8.0))
    nx, ny = normal_vector_field(eps)
    b = rt(solve(nvf_conv_matrices(np, eps, nx, ny, fto, fto), eps, fto, theta=8.0))
    tol = 3e-5 * (n_in ** 2 - 1.0) ** 2
    assert max(abs(a[0] - b[0]), abs(a[1] - b[1])) < tol


def test_the_two_paths_agree_in_the_limit_where_they_have_to():
    """The claim above is agreement, so something has to watch a trend, not just one number.

    **Not the trend in fto.** Measured, that gap does not fall: it sits near 1.3e-5 from fto=6
    to fto=14 on the strongest contrast here. Nor should it — two vector formulations differ at
    any finite truncation, and 1.3e-5 is an order tighter than the 1.8e-4 between fmmax's own
    two formulations on the same structure. An earlier version of this test asserted the fto
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
    fto = 8
    relative = []
    for n_in in (1.05, 1.6):
        eps = pillar(n_in ** 2)
        nx, ny = normal_vector_field(eps)
        a = rt(solve(pn(tensor(eps), eps, fto), eps, fto, theta=8.0))
        b = rt(solve(nvf_conv_matrices(np, eps, nx, ny, fto, fto), eps, fto, theta=8.0))
        relative.append(abs(a[0] - b[0]) / abs(b[0]))

    assert relative[0] < relative[1] / 10, (
        f'the paths do not close faster than the field does: relative gaps {relative} '
        f'(ratio {relative[1] / relative[0]:.1f}, measured 17 when this was written)')


# ------------------------------------------------------------------------ theorems, not oracles

@pytest.mark.parametrize('fto, tol', [(4, 5e-5), (8, 5e-6), (12, 2e-6)])
def test_a_hermitian_tensor_conserves_energy_however_it_is_patterned(fto, tol):
    """``eps_xy = +ig`` with ``eps_yx = -ig`` is Hermitian, hence lossless.

    Losslessness follows from the tensor being Hermitian at every pixel and has nothing to do
    with the pattern, so this holds no matter what the factorization does to the geometry —
    which is what makes it a check the factorization cannot talk its way out of.

    The tolerance is per-order and measured (the residual runs 1.7e-5 / 1.7e-6 / 5.8e-7 at
    these three), not one loose number reused. A single tolerance wide enough for fto=4 would
    leave fto=12 free to regress by an order without anything noticing.
    """
    eps = pillar(1.6 ** 2)
    g = pillar(0.30j, 0.0)
    r, t = rt(solve(pn(tensor(eps, g), eps, fto), eps, fto, theta=8.0, psi=30.0))
    assert abs(r + t - 1.0) < tol


def test_energy_closes_better_as_the_truncation_is_relaxed():
    """The residual above is truncation, so it has to shrink. If it plateaus it is a defect.

    Worth pinning separately: an early arrangement of this factorization conserved energy to
    ~1e-2 at every order, and a fixed tolerance would have called that a pass.
    """
    eps = pillar(1.6 ** 2)
    g = pillar(0.30j, 0.0)
    drift = []
    for fto in (2, 4, 8):
        r, t = rt(solve(pn(tensor(eps, g), eps, fto), eps, fto, theta=8.0, psi=30.0))
        drift.append(abs(r + t - 1.0))
    assert drift[-1] < drift[0] / 10, f'not improving with fto: {drift}'


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
    fto = 6

    a = rt(solve(pn(tensor(eps, g), eps, fto), eps, fto, psi=30.0))
    eps_r = np.rot90(eps)
    b = rt(solve(pn(tensor(eps_r, np.rot90(g)), eps_r, fto), eps_r, fto, psi=30.0 + 90.0))

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
        out = pn_conv_matrices(np, e33, nx, ny, 4, 4)
    assert not all(np.isfinite(m).all() for m in out)
