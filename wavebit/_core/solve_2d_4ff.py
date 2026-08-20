"""First-order (4ff) layer solve — the formulation a z-coupled tensor needs.

**Status: reachable. A z-coupled tensor handed to the public solver arrives here** —
:func:`solve_2d_4ff` is the entry, and ``formulation='4ff'`` forces it on a problem that does not
need it. Field reconstruction landed 2026-08-14. **What is still missing is a patterned
*anisotropic* layer and ``mu``**; see "What is not here" at the bottom before building on it.

Pinned by ``tests/test_solve_2d_4ff.py`` against the 2ff path and a Berreman 4x4, and by
``tests/references/test_4ff_external_goldens.py`` against RETICOLO and fmmax — 21 cases, worst
gap 2.4e-15, no convention fitting, now run through the public route as well as this module's
own entry. Note what the goldens do *not* cover: every case in both has an empty z row, because
both were frozen for a 2ff consumer. The geometry this module exists for is judged by our own
Berreman — which since dispatch landed judges the public route directly, on stacks, on a tilted
magnetisation (``xy`` and ``yz`` at once, which RETICOLO cannot arbitrate), and from the back
side (``tests/test_anisotropic_berreman.py``).

Why it exists. :mod:`wavebit._core.solve_2d` reduces a layer to the second-order form
``d2E_t/dz2 = Omega2 E_t``, which exists only when the permittivity has an empty z row and
column. With ``eps_xz`` or ``eps_yz`` populated the elimination of ``E_z`` leaves a
first-derivative term, the reduction fails, and the 2ff mode operator has nothing to be —
``docs/anisotropic-materials.md`` §6. Physically that is the Voigt geometry: a magnetisation
lying in the plane rather than along the stack normal, which is what a magnetised film does on
its own unless it has perpendicular anisotropy or an applied field.

What the first-order form keeps instead is the full transverse state,

    d/dz (Ex, Ey, Hx, Hy) = i k0 M (Ex, Ey, Hx, Hy)

with ``M`` of size ``4*n_orders``. It costs about 8x the 2ff path (``eig`` is cubic and the problem
doubles), measured at 9.5x on RETICOLO, which is why choosing between them belongs to the
solver and not the caller — ``docs/anisotropic-materials.md`` §8.

The derivation, for ``e^{-iwt}`` with ``d/dx -> i k0 Kx`` and ``d/dy -> i k0 Ky``::

    curl E = i k0 H,   curl H = -i k0 eps E

    z components:   Hz = Kx Ey - Ky Ex,      Dz = Ky Hx - Kx Hy
    and Dz = ezx Ex + ezy Ey + ezz Ez, so
                    Ez = ezz^-1 (Ky Hx - Kx Hy - ezx Ex - ezy Ey)

    leaving         Ex' = i k0 (Kx Ez + Hy)
                    Ey' = i k0 (Ky Ez - Hx)
                    Hx' = i k0 (Kx Hz - Dy)
                    Hy' = i k0 (Ky Hz + Dx)

Every product is a convolution-matrix product and ``Kx`` / ``Ky`` are the same diagonals the 2ff
path uses, so this is the Fourier-space statement rather than a per-order one.

**Forward and backward modes are not a sign apart here.** In 2ff, ``exp(+q z)`` and
``exp(-q z)`` solve the same second-order equation with the *same* eigenvector, so the pair came
for free and :func:`wavebit._core.conventions.mode_q` only had to choose a root. A
first-order system has ``4*n_orders`` eigenvalues that are not ``+-`` pairs in general, and the two
directions are genuinely different mode sets. Measured: for a tilted magnetisation at 35
degrees the backward set departs from the negated forward set by 9e-2, while at normal
incidence the two coincide to 1e-15 — so the pairing survives exactly where nobody would have
noticed it failing.

**What makes the stack solve stable is the direction split, not the algorithm on top of it.**
The split is what lets a layer be propagated by ``exp(i k0 lam_f d)`` forward and
``exp(-i k0 lam_b d)`` backward. **In a passive medium** both have modulus at most one for any
thickness — a mode carrying energy towards ``+z`` cannot gain amplitude on the way, so
``Im(lam_f) >= 0``, and the mirrored statement holds backwards. Two of the three connecting
algorithms are then arrangements that keep only those factors:

==========  ==================================  ====================  ================
``algo``    form                                thickness             cost, 40 layers
==========  ==================================  ====================  ================
``'ETM'``   enhanced transmittance (default)    unbounded             0.94x
``'SMM'``   scattering matrices, Redheffer      unbounded             1.13x
``'TMM'``   ``Psi exp(...) Psi^-1``, plain      refuses past x = 9    1.00x
==========  ==================================  ====================  ================

``'ETM'`` is the same option, under the same name, as the 2ff path's default — the string has
to mean one thing across both solvers. ``'TMM'`` here is the plain unstabilised transfer,
which the 2ff path does not offer at all; it exists only to show the stabilisation is
load-bearing.

The two stable ones agree to 1.2e-15 over eight tensors, three angles and both polarisations,
and both hold ~1e-13 relative accuracy out to a modal exponent of 2500, where they reproduce the
closed-form single-interface Fresnel reflectance to 4e-16.

``'TMM'`` is kept as a third opinion whose failure mode is understood: it holds the growing
and decaying halves in one matrix and inverts it, so its relative error grows like
``eps*exp(2x)``. Its bound is set by *accuracy*, not by overflow — see ``_MAX_EXPONENT``.

**None of this covers a gain medium**, where a forward mode amplifies by construction
(``Im(lam_f) < 0``; measured -0.108 for ``n = 1.5 - 0.1j``) and the propagators stop being
contractions. Measured anyway: the two stable algorithms still track the Airy closed form to
~1e-13 out to an exponent of 113, and past roughly 700 return NaN. Nothing guards that, because
there the growth is the physics rather than an artefact — but the stability claim is about
passive media and should not be read further.

**Field reconstruction landed 2026-08-14** and is no longer on the list below.
:func:`_etm_recursion` keeps a ``layer_info_list`` — ``[eps_blocks, psi_f, lam_f, psi_b,
lam_b, d, P_i, QP_i]`` per layer, bottom-to-top, the counterpart of the 2ff path's
``[epz_conv_i, W, V, q, d, big_A_i, big_B]`` — and
:func:`wavebit._core.fields.field_2d_4ff` propagates it. Three notes worth keeping:

* the transverse half is *simpler* here, because the state vector already **is**
  ``(Ex, Ey, Hx, Hy)`` and there is no ``W`` / ``V`` split to reassemble;
* ``E_z`` is the one formula that could not be copied — with a populated z row the
  elimination keeps two more terms, ``Ez = ezz^-1 [ (Ky Hx - Kx Hy) - ezx Ex - ezy Ey ]``,
  and ``fields.py``'s ``Sz = 1j * epz_conv_i @ (Kx @ Uy - Ky @ Ux)`` is that expression at
  ``ezx = ezy = 0`` (with ``U = i H``), i.e. a special case rather than a shared line;
* ``reconstruct_field`` now branches on the **formulation**. It had to: this path sets
  ``_grating_type_assigned`` to 2 as well, so the grating type cannot tell the two apart, and
  the 2ff reconstruction reads the same two state slots — it would have returned a plausible
  cell built on the wrong ``E_z``.

The oracle was internal and exact, as scoped: forcing ``formulation='4ff'`` on a problem with
an empty z row makes the two paths solve the same problem, and the reconstructed cells agree to
1e-11 relative *per component* over four geometries, three angles, both incidence sides and all
three incidence variants. The z-coupled case has no such twin and is judged by Maxwell (central
differences down a layer, converging at the second order the step demands), by exact transverse
continuity at the interfaces, and — because both of those are homogeneous — by tying the cell's
``z`` flux between lossless spacers to the reported ``T / (1 - R)``.
See ``tests/test_field_4ff.py``.

What is not here, in the order it will be wanted:

* **``mu`` on this path.** Permeability landed on the 2ff path on 2026-08-14
  (``docs/magnetic-materials.md``) and this one still refuses it — ``engine.py``'s 2D solve
  raises when a ``mu`` reaches the first-order formulation, by either route in (a z-coupled
  ``eps``, or ``formulation='4ff'`` asked for directly), rather than dropping it. That refusal
  covers the *field* too: :func:`wavebit._core.fields.field_2d_4ff` spells ``H_z`` as
  ``Kx Ey - Ky Ex``, Faraday's z row at ``mu = 1``, where its 2ff twin carries the general
  ``muz_conv_i (Kx Ey - Ky Ex)``. Adding ``mu`` here means both — and, this being the formulation
  that can express a z row at all, the ``mu_zx H_x`` / ``mu_zy H_y`` terms as well, exactly as
  ``E_z`` already keeps ``ezx`` / ``ezy``.
* **A patterned *and* z-coupled layer.** The factorization for a patterned anisotropic layer
  now exists (``factorization.py::pn_conv_matrices``, 2026-08-14) and this module takes
  convolution matrices, so the in-plane case needs no change here. What is still missing is the
  3x3 closing step for a patterned layer that also fills the z row — and, more to the point,
  any way to check it: fmmax was the only solver that reached the patterned case and it has no
  z-row arguments.

How dispatch is built, kept because the two measurements behind it are why
:func:`solve_2d_4ff` is shaped the way it is:

**The mode correspondence is exact and simple.** For a layer both formulations can express,
the 2ff triple is recoverable from the 4ff state with no fitting::

    W = psi_f[:2*n_orders]          the transverse E
    V = 1j * psi_f[2*n_orders:]     the partner field, i times the transverse H
    q  = -1j * lam_f

Verified to 1e-15 on biaxial and rotated-uniaxial layers at three incidences; the ``1j`` is a
clean scalar, identical in every case.

**But the ETM recursion cannot be reused as it stands.** It writes the two directions as
``big_F = big_W (I + Y)`` and ``big_G = big_V (I - Y)``, which assumes the backward modes are
the forward ones with the partner field negated. Measured, that assumption holds to 1e-15 for
a z-decoupled layer *and at normal incidence*, and fails at 6.9e-1 (E) / 1.1 (H) for a Voigt
tensor at 35 degrees — the same ``+-`` pairing this module exists because of.

So dispatch is not a branch inside :func:`wavebit._core.solve_2d.solve_2d`. The 4ff path
carries its own stack solve and hands the shared
:func:`wavebit._core.solve_1d.final_matching` a ``(big_F, big_G, big_T)`` built from its own
``G`` and ``T_acc`` — rotated into the s/p basis by ``varphi`` and carrying the ``1j`` above.

The normalisation that warning is about is settled by *naming the seed's coordinates*:
:func:`_half_space_state` writes the substrate wave with its two columns already the 2ff
path's ``(t_s, t_p)``, so ``big_T`` is ``T_acc`` with nothing applied to it. Eigendecomposing
the half-space instead — the obvious alternative — leaves ``eig``'s arbitrary column scale
between ``big_F`` and ``big_T``, and that error is invisible in ``R``.
"""
from .backend import array_namespace, as_array, eig
from .conventions import boundary_kz, order_azimuth
from .enums import Answers, answers_of
from .solve_1d import _block, _diag, _is_backside, _mirrored_scene, final_matching
from .solve_2d import order_grid_kxy, sp_modes_2d

# Flux below this, on a mode normalised to unit energy, is taken as none — the mode is
# evanescent and direction is decided by decay instead. See split_forward_backward.
_FLAT = 1e-12

# The transfer inverts a matrix holding exp(+x) and exp(-x) at once, so it loses relative
# precision like eps*exp(2x) -- measured 1.2e-8 at x = 9.1 against a predicted 1.8e-8, and 8.5e-4
# by x = 15.9. This bound is where that prediction passes ~1e-9, which is where it stops being
# usable as a cross-check. It is deliberately far below the overflow point: the earlier value of
# 300 only caught the crash, and let three decades of silently wrong answers through first
# (5.1e+25 relative error at x = 34, no complaint).
_MAX_EXPONENT = 9.0


def layer_matrix_4ff(xp, Kx, Ky, exx, exy, exz, eyx, eyy, eyz, ezx, ezy, ezz_i):
    """``M`` in ``dpsi/dz = i k0 M psi`` for ``psi = (Ex, Ey, Hx, Hy)``.

    All arguments are ``n_orders x n_orders`` convolution matrices except ``Kx`` / ``Ky``, which
    are the diagonal order matrices the 2ff path builds. ``ezz_i`` is the *inverse* of the zz
    block, matching how :func:`wavebit._core.solve_2d.layer_modes_2d` already carries it.
    """
    n_orders = Kx.shape[0]
    I = xp.astype(xp.eye(n_orders), Kx.dtype)

    ez_ex, ez_ey = -(ezz_i @ ezx), -(ezz_i @ ezy)      # Ez in terms of the state
    ez_hx, ez_hy = ezz_i @ Ky, -(ezz_i @ Kx)
    hz_ex, hz_ey = -Ky, Kx                             # Hz likewise

    dx_ex, dx_ey = exx + exz @ ez_ex, exy + exz @ ez_ey
    dx_hx, dx_hy = exz @ ez_hx, exz @ ez_hy
    dy_ex, dy_ey = eyx + eyz @ ez_ex, eyy + eyz @ ez_ey
    dy_hx, dy_hy = eyz @ ez_hx, eyz @ ez_hy

    return _block(xp, [
        [Kx @ ez_ex, Kx @ ez_ey, Kx @ ez_hx, Kx @ ez_hy + I],
        [Ky @ ez_ex, Ky @ ez_ey, Ky @ ez_hx - I, Ky @ ez_hy],
        [Kx @ hz_ex - dy_ex, Kx @ hz_ey - dy_ey, -dy_hx, -dy_hy],
        [Ky @ hz_ex + dx_ex, Ky @ hz_ey + dx_ey, dx_hx, dx_hy],
    ])


def mode_flux_z(xp, psi, n_orders):
    """Time-averaged Poynting ``z`` component of each column of ``psi``.

    This is the raw flux carried by the column as given, which is what a mode *amplitude*
    multiplies — see :func:`solve_uniform_stack`. For deciding direction, divide by the column
    norm first; :func:`split_forward_backward` does.
    """
    Ex, Ey = psi[:n_orders], psi[n_orders:2 * n_orders]
    Hx, Hy = psi[2 * n_orders:3 * n_orders], psi[3 * n_orders:]
    return 0.5 * xp.real(xp.sum(Ex * xp.conj(Hy) - Ey * xp.conj(Hx), axis=0))


def split_forward_backward(xp, M, n_orders):
    """Eigenmodes of ``M`` split by direction: ``(lam_f, psi_f, lam_b, psi_b)``.

    Forward means energy flowing towards ``+z``, falling back to decay in ``+z`` where the mode
    carries no real flux. That is the same rule
    :func:`wavebit._core.conventions.boundary_kz` applies to a half-space **for a real
    in-plane wavevector**, which is the only kind this path sees: its half-spaces are vacuum,
    so ``k_par`` is real and the flux and decay conditions select the same mode (they are
    separate demands only when ``k_par`` is complex — see ``docs/time-convention-audit.md``
    §3). ``psi`` goes as ``exp(i k0 lam z)``, so ``lam`` sits exactly where that function's
    ``kz`` does. If this path ever grows an absorbing superstrate, the correction
    ``boundary_kz`` carries has to come with it, and this comment is the pointer to that.

    Flux is the primary test rather than ``Re(lam)`` because an absorbing or gyrotropic layer
    can put a propagating mode on the wrong side of the real axis; the ``Im`` fallback only
    decides modes that carry no flux to speak of.

    "No flux to speak of" is measured *per mode*, against its own norm. Scaling instead by the
    largest flux in the set — the obvious thing — holds up until every mode is evanescent, at
    which point that scale is itself noise and dust at 1e-17 is promoted to real energy flow.
    Every order above the light line is that case, so the set-relative version breaks exactly
    where the multi-order path will live.

    **Selection is a stable sort, not a boolean mask**, and that is what lets this path run
    under ``jax.jit``. A mask makes the output shape depend on the values being masked, which a
    tracer cannot answer; sorting on a two-valued key and slicing at ``2*n_orders`` gives the
    same partition with shapes fixed at trace time. Stability is what makes it *the same*: equal
    keys keep their original order, so ``lam[forward]`` and ``take(lam, order[:2*n_orders])``
    are the same vector entry for entry, not merely the same set. Verified bit-identical across
    every shipped path (``tests/test_formulation_dispatch.py``).

    The count check is the one thing that cannot be traced, so it runs only where the values
    are concrete — the same rule :func:`wavebit._core.backend.check_finite` and
    ``check_physical`` follow, and for the same reason. Note that "concrete" is the whole of
    it: those two once keyed the exemption on the *namespace* instead, which skipped eager JAX
    and eager PyTorch and cost a silently wrong answer on both. Under ``jit`` a degenerate
    layer still gets no complaint; see :func:`_forward_count`.
    """
    lam, psi = eig(M)
    flux = mode_flux_z(xp, psi, n_orders) / xp.sum(xp.abs(psi) ** 2, axis=0)
    forward = xp.where(xp.abs(flux) > _FLAT, flux > 0, xp.imag(lam) > 0)

    # 1.0 forward / 0.0 backward, in `flux`'s own real dtype -- asking for int64 here is what
    # made JAX warn and silently narrow to int32 when x64 is off.
    key = xp.astype(forward, flux.dtype)
    n_f = _forward_count(xp, key)
    if n_f is not None and n_f != 2 * n_orders:
        raise ValueError(
            f'mode split found {n_f} forward modes, expected {2 * n_orders}. A degenerate layer can '
            'leave a mode with no flux and an ambiguous decay; this refuses rather than '
            'guessing which half it belongs to.')

    order = xp.argsort(-key, stable=True)
    fwd, bwd = order[:2 * n_orders], order[2 * n_orders:]
    return (xp.take(lam, fwd), xp.take(psi, fwd, axis=1),
            xp.take(lam, bwd), xp.take(psi, bwd, axis=1))


def _forward_count(xp, key):
    """How many modes went forward, or ``None`` when nothing concrete can be read.

    ``None`` means a tracer: ``jax.jit`` traces with abstract values and ``int()`` on one
    raises rather than returning a number. ``jax.grad`` is *not* that case — its tracer carries
    a concrete primal, so the guard still runs under a gradient. That asymmetry is the same one
    every eager decision in this library leans on, and it is tabulated in
    ``docs/backend-choice.md`` §3: a grad tracer refuses a whole-array conversion but permits
    narrowing one value; a jit tracer refuses both.

    Catching broadly rather than naming the exception keeps this backend-agnostic, which is the
    same trade :func:`wavebit._core.engine._concrete_period` makes two modules over.
    """
    try:
        return int(xp.sum(key))
    except Exception:
        return None


def vacuum_sp_states(xp, kx, ky, n_orders, order=0):
    """``(psi_s, psi_p)``: unit-amplitude incident plane waves in vacuum, as ``4ff`` states.

    The incident field has to be named rather than picked out of the vacuum eigenbasis. In
    vacuum both forward modes share ``lam = kz``, so that eigenspace is degenerate and *any*
    basis of it is a valid answer from ``eig`` — LAPACK happens to return clean s and p at
    ``ky = 0`` because ``M`` separates there, which is an artefact of the block structure and
    not a promise. Off the plane of incidence, or on another backend, the two would mix and
    "the first forward mode" would silently stop meaning p.

    With ``khat = (kx, ky, kz)`` and ``shat`` the unit vector perpendicular to the plane of
    incidence, ``curl E = i k0 H`` gives ``H = khat x E``, so::

        s:  E = shat,          H = khat x shat
        p:  E = shat x khat,   H = shat

    At normal incidence the plane of incidence is undefined and ``shat`` is taken as ``y``,
    which is the limit the oblique cases approach.
    """
    # `complex`, not `float`. The cast that was here dropped `Im(kx)`, which is round-off at a
    # real angle and not remotely round-off at a complex one: it moved the returned state by
    # 0.12 at `theta = 0.4 + 0.3j` and by 0.47 at `0.2 + 0.9j`, and what came back was no longer
    # a plane wave at all -- `H != khat x E` by the same amount. Everything below is already
    # complex-clean, so keeping the imaginary part costs nothing and changes nothing at a real
    # angle. What a complex `kt` has no answer for is the guard, immediately below.
    kx0, ky0 = complex(xp.reshape(xp.asarray(kx), (-1,))[order]), \
        complex(xp.reshape(xp.asarray(ky), (-1,))[order])
    kt = (kx0 ** 2 + ky0 ** 2) ** 0.5
    kz = (1.0 - kt ** 2) ** 0.5

    # The guard asks about `order`, not about the angle. Order m carries `k_par + G_m`, and past
    # the light line that order is evanescent in vacuum, so there is no incident plane wave for
    # this function to name -- which is the one thing it exists to do. At a real angle
    # `kt ** 2 >= 1` states exactly that.
    #
    # Two different things get called evanescent here, and only the second is an open question.
    #
    # * Real `kx`, `kt > 1`, imaginary `kz`. Refused below, and the refusal is not a missing
    #   toggle: `solve_uniform_stack` normalises by `flux(incident)`, and a purely evanescent
    #   state carries no time-averaged z-flux in a lossless medium -- measured 4.3e-01 at
    #   `kt = 0.5` against 2.0e-17 at `kt = 1.2`. Lifting the guard gives 0/0, so R and T stop
    #   being defined before the solve is even reached. The band is worth reaching -- in an
    #   emitting stack it is where the substrate, waveguided and surface-plasmon channels live
    #   -- but reaching it means a source inside the stack and a power spectrum over `k_par`,
    #   not an incident plane wave and an efficiency. A separate feature, not this guard.
    #
    # * Complex `kx` (a complex incidence angle). TODO: undefined here. Under `e^{-iwt}` with
    #   `Im(n) > 0` a complex theta is an inhomogeneous wave by construction, so "evanescent,
    #   therefore refuse" stops being the rule -- `|kt| > 1` would reject the legitimate case
    #   rather than the unnameable one. `docs/first-order-4ff.md` does not scope complex
    #   incidence and no test reaches it, so this refuses rather than guessing. The `float()`
    #   cast that used to sit above admitted it silently and returned the wrong state, and it
    #   disarmed the line below too: `theta = 0.2 + 0.9j` has a true `|kt| = 1.046` and the
    #   cast read it as 0.285.
    #
    #   The S4 field oracle added in `dacc134` does not close this either, and it is worth
    #   saying so here because it now looks like it might: it excites propagating plane waves
    #   only. Scoping a complex angle would still mean scoping it with no external check
    #   behind it.
    if abs(kt.imag) > 1e-12 * max(abs(kt), 1.0):
        raise NotImplementedError(
            f'complex incidence angle (kt = {kt:.4g}): whether an order counts as evanescent is '
            'undefined here, so the vacuum s and p states have no scoped meaning yet.')
    if kt.real ** 2 >= 1.0:
        raise ValueError(f'order {order} is evanescent in vacuum (|kt| = {kt.real:.4f}); there '
                         'is no incident plane wave to define.')

    khat = (kx0, ky0, kz)
    shat = (-ky0 / kt, kx0 / kt, 0.0) if abs(kt) > 1e-14 else (0.0, 1.0, 0.0)
    cross = lambda a, b: (a[1] * b[2] - a[2] * b[1],   # noqa: E731
                          a[2] * b[0] - a[0] * b[2],
                          a[0] * b[1] - a[1] * b[0])

    out = []
    for E, H in ((shat, cross(khat, shat)), (cross(shat, khat), shat)):
        psi = xp.zeros((4 * n_orders,), dtype=xp.complex128)
        for slot, val in ((0, E[0]), (1, E[1]), (2, H[0]), (3, H[1])):
            psi[slot * n_orders + order] = val
        out.append(psi)
    return tuple(out)


def _interface_s(xp, psi1_f, psi1_b, psi2_f, psi2_b):
    """Scattering matrix of the boundary between two media, in their own mode bases.

    The whole transverse state is continuous across the interface, so with amplitudes ``a``,

        Psi1_f a1_f + Psi1_b a1_b = Psi2_f a2_f + Psi2_b a2_b

    and grouping the two *outgoing* sets on the left gives the scattering form directly. No
    ``kz`` sign flip and no gap medium is needed here — the 2ff path needs both because its two
    directions share one eigenvector, and these do not.
    """
    n = psi1_f.shape[1]
    out = xp.concat([psi1_b, -psi2_f], axis=1)
    inn = xp.concat([-psi1_f, psi2_b], axis=1)
    S = xp.linalg.solve(out, inn)
    return S[:n, :n], S[:n, n:], S[n:, :n], S[n:, n:]


def _propagate_s(xp, lam_f, lam_b, k0, d, n_orders):
    """Scattering matrix of a uniform layer, forward amplitudes referenced at its top.

    **This is the whole reason for the direction split.** Forward modes decay towards ``+z``
    (``Im(lam_f) >= 0``) so ``|exp(i k0 lam_f d)| <= 1``, and backward modes decay towards ``-z``
    so their propagator is bounded too. Both entries are contractions for any thickness, which is
    precisely what the eigen-decomposition transfer cannot promise: it holds the growing and
    decaying halves in one matrix and inverts it, so the growing half sets the conditioning.
    """
    zero = xp.astype(xp.zeros((2 * n_orders, 2 * n_orders)), lam_f.dtype)
    X_f = _diag_of(xp, xp.exp(1j * k0 * lam_f * d))
    X_b = _diag_of(xp, xp.exp(-1j * k0 * lam_b * d))
    return zero, X_b, X_f, zero


def _solve_smm(xp, k0, Kx, Ky, layers, n_orders, psi_vf, psi_vb):
    """Cascade interface and propagation scattering matrices with the Redheffer star."""
    from .scattering import redheffer_star

    I = xp.astype(xp.eye(2 * n_orders), Kx.dtype)
    S, prev_f, prev_b = None, psi_vf, psi_vb
    for eps_blocks, d in layers:
        lam_f, psi_f, lam_b, psi_b = split_forward_backward(
            xp, layer_matrix_4ff(xp, Kx, Ky, **eps_blocks), n_orders)
        for part in (_interface_s(xp, prev_f, prev_b, psi_f, psi_b),
                     _propagate_s(xp, lam_f, lam_b, k0, d, n_orders)):
            S = part if S is None else redheffer_star(xp, I, S, part)
        prev_f, prev_b = psi_f, psi_b
    tail = _interface_s(xp, prev_f, prev_b, psi_vf, psi_vb)
    return tail if S is None else redheffer_star(xp, I, S, tail)


def _forward_amplitudes(xp, incident, psi_vf, psi_vb, n_orders):
    """Mode amplitudes of an incident state, and a check that it is purely incident.

    Decomposing over the *full* basis rather than solving against ``psi_vf`` alone keeps the
    system square and costs nothing, since the backward half is the thing worth testing: a state
    with any backward content is not an incident wave, and would otherwise be silently projected.
    """
    coef = xp.linalg.solve(xp.concat([psi_vf, psi_vb], axis=1),
                           xp.reshape(incident, (-1, 1)))[:, 0]
    leak = float(xp.max(xp.abs(coef[2 * n_orders:])))
    if leak > 1e-10 * max(float(xp.max(xp.abs(coef[:2 * n_orders]))), 1e-300):
        raise ValueError(f'the incident state has backward content ({leak:.2e}); it is not a '
                         'wave arriving from the top. Build it with vacuum_sp_states.')
    return coef[:2 * n_orders]


def _solve_etm(xp, k0, Kx, Ky, layers, n_orders, incident, psi_vf, psi_vb):
    """Enhanced transmittance matrix — the 2ff path's stable transfer, carried over to 4ff.

    Same idea as :func:`wavebit._core.solve_2d.solve_2d`'s ``big_A_i`` / ``big_T`` recursion
    (Moharam et al. 1995), which is why the 2ff transfer is stable while the naive one in
    :func:`_solve_naive` is not. The growth is not removed by better conditioning; it is arranged
    never to appear.

    Sweeping upward, suppose the state below a layer is ``psi = G t`` for an unknown ``t``.
    Decomposing in that layer's modes gives ``a_f = X_f^-1 P t`` — and ``X_f^-1`` is exactly the
    growing factor. ETM absorbs it into the unknown, ``t' = X_f^-1 P t``, leaving

        G_above = Psi_f + Psi_b @ X_b @ (Q P^-1) @ X_f

    in which ``X_f`` and ``X_b`` appear only as the bounded, decaying factors they are. The
    redefinition is bookkept separately as ``P^-1 X_f`` per layer, so the transmitted amplitude
    can be recovered at the end.
    """
    G, T_acc, _ = _etm_recursion(xp, k0, Kx, Ky, layers, n_orders, psi_vf, keep_field=False)

    # incident + reflected on the top side must be whatever the recursion says arrives there
    sol = xp.linalg.solve(xp.concat([psi_vb, -G], axis=1),
                          xp.reshape(-incident, (-1, 1)))[:, 0]
    return psi_vb @ sol[:2 * n_orders], psi_vf @ (T_acc @ sol[2 * n_orders:])


def _etm_recursion(xp, k0, Kx, Ky, layers, n_orders, G, keep_field=True):
    """Sweep the ETM recursion up the stack: ``(G_above, T_acc, layer_info_list)``.

    Split out of :func:`_solve_etm` because the recursion is the part
    :func:`solve_2d_4ff` shares — it seeds a half-space of index ``n_bot`` rather than vacuum
    and hands the result to the s/p matching instead of matching against a vacuum basis here.
    The arithmetic is untouched by the split.

    ``G`` maps an unknown amplitude vector to the transverse state; whatever coordinates the
    seed's columns are in, ``T_acc`` maps the *returned* ``G``'s unknown back into them. The two
    are only meaningful together, which is why they are returned together: ``big_T`` feeds the
    transmitted amplitude directly, so a per-mode scale that cancels in the reflection does not
    cancel there.

    ``layer_info_list`` is what field reconstruction propagates — the counterpart of the
    ``[epz_conv_i, W, V, q, d, big_A_i, big_B]`` records
    :func:`wavebit._core.solve_2d.solve_2d` leaves behind, appended **bottom-to-top** for the
    same reason (:func:`wavebit._core.fields.field_2d_4ff` walks it in reverse). Each entry is
    ``[eps_blocks, psi_f, lam_f, psi_b, lam_b, d, P_i, QP_i]``, and the three pieces that are
    not simply the layer's modes are exactly the two amplitude maps this loop already forms:

    * ``P_i @ X_f`` carries the unknown *down* one layer (``t = P_i X_f t'``), which is what
      makes ``T_acc`` the product of them;
    * ``QP_i @ X_f`` turns the same unknown into the backward amplitudes referenced at the
      layer's **bottom** face, which is where ``X_b`` in the ``G`` update starts from.

    ``eps_blocks`` rides along because ``E_z`` needs ``ezz_i``, ``ezx`` and ``ezy`` — the one
    part of the reconstruction the 2ff path cannot lend.
    """
    T_acc = xp.astype(xp.eye(2 * n_orders), Kx.dtype)
    layer_info_list = []
    for eps_blocks, d in reversed(list(layers)):
        lam_f, psi_f, lam_b, psi_b = split_forward_backward(
            xp, layer_matrix_4ff(xp, Kx, Ky, **eps_blocks), n_orders)
        X_f = _diag_of(xp, xp.exp(1j * k0 * lam_f * d))
        X_b = _diag_of(xp, xp.exp(-1j * k0 * lam_b * d))

        PQ = xp.linalg.solve(xp.concat([psi_f, psi_b], axis=1), G)
        P_i = xp.linalg.inv(PQ[:2 * n_orders])
        QP_i = PQ[2 * n_orders:] @ P_i
        G = psi_f + psi_b @ X_b @ QP_i @ X_f
        T_acc = T_acc @ (P_i @ X_f)

        # `keep_field=False` lets each layer's eight-slot state be collected as the loop
        # moves on; only `G` and `T_acc` have to survive an iteration. `_solve_etm` above
        # discards the list outright and so never builds it.
        if keep_field:
            layer_info_list.append([eps_blocks, psi_f, lam_f, psi_b, lam_b, d, P_i, QP_i])
    return G, T_acc, layer_info_list


def _solve_naive(xp, k0, Kx, Ky, layers, n_orders, incident, psi_vf, psi_vb):
    """Unstabilised eigen-decomposition transfer, kept only as a third opinion.

    Not ``layer_algorithm='ETM'``, which is the enhanced form in :func:`_solve_etm` and is what
    :mod:`wavebit._core.solve_2d` uses. This one exists so the stabilisation can be shown to
    be doing something, and refuses where it would be inaccurate.
    """
    P = xp.astype(xp.eye(4 * n_orders), Kx.dtype)
    for eps_blocks, d in layers:
        lam_f, psi_f, lam_b, psi_b = split_forward_backward(
            xp, layer_matrix_4ff(xp, Kx, Ky, **eps_blocks), n_orders)
        lam = xp.concat([lam_f, lam_b])
        exponent = k0 * float(xp.max(xp.abs(xp.imag(lam)))) * float(d)
        if exponent > _MAX_EXPONENT:
            raise ValueError(
                f'layer exponent {exponent:.1f} is past what the unstabilised transfer can '
                "carry in float64. Use layer_algorithm='ETM' or 'SMM', either of "
                'which is bounded for any thickness.')
        Psi = xp.concat([psi_f, psi_b], axis=1)
        E = _diag_of(xp, xp.exp(1j * k0 * lam * d))
        P = (Psi @ E @ xp.linalg.inv(Psi)) @ P

    # Matching at both interfaces at once: incident + reflected on the input side must transfer
    # to transmitted only on the output side, so P (a + psi_vb r) = psi_vf t.
    A = xp.concat([P @ psi_vb, -psi_vf], axis=1)
    sol = xp.linalg.solve(A, xp.reshape(-(P @ incident), (-1, 1)))[:, 0]
    return psi_vb @ sol[:2 * n_orders], psi_vf @ sol[2 * n_orders:]


def solve_uniform_stack(wavelength, kx, ky, layers, n_orders, incident, layer_algorithm='ETM'):
    """``(R, T)`` for one incident state, through a stack of uniform anisotropic layers.

    ``layers`` is ``[(eps_blocks, thickness), ...]`` where ``eps_blocks`` is the mapping
    :func:`layer_matrix_4ff` takes. ``incident`` is a ``4ff`` state — see
    :func:`vacuum_sp_states`. Vacuum on both sides, and note it is *not* special-cased: vacuum
    is a layer with ``eps = I`` whose forward and backward halves come from the same mode split
    as any other, which is what would make an anisotropic substrate free here. The 2ff path
    cannot say that, and neither can RETICOLO.

    ``R`` and ``T`` are fluxes of the *reconstructed* reflected and transmitted fields, not sums
    of per-mode fluxes. Those agree only when the modes are flux-orthogonal, which holds for a
    lossless medium and stops holding exactly where it would be least noticed.

    ``layer_algorithm`` selects how layers are joined. Three, and the choice matters here in a
    way it does not in the 2ff path, where both shipped options are already stable:

    * ``'ETM'`` (default) is the enhanced-transmittance recursion — **the same option, under
      the same name, as the 2ff path's default**. Growth never appears rather than being
      cancelled. Measured the cheapest of the three — 0.94x the plain transfer on a 40-layer
      stack, against SMM's 1.13x — and identical to SMM at 1.2e-15.
    * ``'SMM'`` cascades per-layer scattering matrices with the Redheffer star. Same stability,
      slightly dearer, and the formulation most of the literature states its results in. Its
      per-layer matrices are also independent of their neighbours, which is what would let them
      be cached or reordered.
    * ``'TMM'`` propagates ``Psi exp(i k0 lam d) Psi^-1``, holding the growing and decaying
      halves in one matrix and inverting it. Deliberately kept in its naive form as a third
      opinion, and it **raises** past the point where it is accurate — see ``_MAX_EXPONENT``.
    """
    xp = array_namespace(kx)
    k0 = 2 * 3.141592653589793 / wavelength
    Kx, Ky = _diag_of(xp, kx), _diag_of(xp, ky)

    vac = _identity_blocks(xp, n_orders, Kx.dtype)
    _, psi_vf, _, psi_vb = split_forward_backward(xp, layer_matrix_4ff(xp, Kx, Ky, **vac), n_orders)

    if layer_algorithm == 'SMM':
        S11, _, S21, _ = _solve_smm(xp, k0, Kx, Ky, layers, n_orders, psi_vf, psi_vb)
        a_inc = _forward_amplitudes(xp, incident, psi_vf, psi_vb, n_orders)
        reflected, transmitted = psi_vb @ (S11 @ a_inc), psi_vf @ (S21 @ a_inc)
    elif layer_algorithm == 'ETM':
        reflected, transmitted = _solve_etm(xp, k0, Kx, Ky, layers, n_orders, incident, psi_vf, psi_vb)
    elif layer_algorithm == 'TMM':
        reflected, transmitted = _solve_naive(xp, k0, Kx, Ky, layers, n_orders, incident,
                                              psi_vf, psi_vb)
    else:
        raise ValueError(
            f"layer_algorithm must be 'ETM', 'SMM' or 'TMM', got {layer_algorithm!r}")

    def flux(v):
        return float(mode_flux_z(xp, xp.reshape(v, (-1, 1)), n_orders)[0])

    inc = flux(incident)
    return -flux(reflected) / inc, flux(transmitted) / inc


def solve_2d_4ff(wavelength, kx, ky, n_top, n_bot, theta, incidence, thickness,
                 epx_conv_all, epy_conv_all, epz_conv_i_all,
                 epxy_conv_all=None, epyx_conv_all=None,
                 epxz_conv_all=None, epyz_conv_all=None,
                 epzx_conv_all=None, epzy_conv_all=None,
                 *, phi, incidence_side='top', answers=Answers.EFFICIENCY,
                 layer_algorithm='ETM'):
    """Solve a 2D stack on the first-order formulation — the public 4ff route.

    Twin of :func:`wavebit._core.solve_2d.solve_2d`: same arguments, same returns, four more
    convolution slots. ``epxz_conv_all`` / ``epyz_conv_all`` / ``epzx_conv_all`` /
    ``epzy_conv_all`` are the ``z`` row and column the second-order form cannot express;
    ``None`` (the default) is zero, and a call giving none of them is the same physical problem
    :func:`~wavebit._core.solve_2d.solve_2d` solves, at about 8x the cost.

    **All four carry the same slot convention as the in-plane pair: the slot holds the
    negative of the tensor component.** ``epxz = -eps_xz``, and so on for the other three, so
    that one rule covers every off-diagonal slot in the signature rather than two rules split
    down the middle of it. ``epz_conv_i`` is still the *inverse* of the ``zz`` block. See
    ``docs/off-diagonal-slot-convention.md``.

    Three pieces are shared with the 2ff path rather than re-derived, and this is where the
    mode correspondence at the top of this module is spent:

    * the **substrate seed** — :func:`_half_space_state` writes the same downward half-space
      wave the 2ff seed ``(big_F, big_G) = ([[I, 0], [0, -i Kz/eps]], [[-i Kz, 0], [0, I]])``
      spells, as a ``4ff`` state whose two columns are that path's ``(t_s, t_p)``;
    * the **s/p rotation** — :func:`~wavebit._core.solve_2d.sp_modes_2d`, given the
      recursion's ``G`` split as ``W = G[:2ff]`` and ``V = 1j*G[2ff:]``. That ``1j`` is the
      whole correspondence: with it the four boundary variables come out in the basis
      :func:`~wavebit._core.solve_1d.final_matching` already reads;
    * the **matching** — ``final_matching`` itself, so the incidence, the efficiencies, the
      Fresnel normalisation and the pure-s/pure-p variants are the same code the rest of the
      library uses, not a second opinion about them.

    ``big_T`` is ``T_acc`` unchanged, with no rescaling, *because* the seed's columns are
    already ``(t_s, t_p)``: the transmitted amplitude is read in the coordinates the seed
    named, so the per-mode scale that would cancel in ``R`` and survive in ``T`` never enters.
    Feeding ``eig``'s half-space modes instead — the obvious alternative — is what puts an
    arbitrary column scale between the two, and it is invisible in reflection.

    ``incidence_side='bottom'`` is the same mirrored scene the other solves use, **plus a sign
    flip on all four z-row slots**: ``z -> -z`` sends ``eps_xz -> -eps_xz`` (and likewise
    ``zx``, ``yz``, ``zy``) while leaving the in-plane block and ``eps_zz`` alone, which is why
    the shared :func:`~wavebit._core.solve_1d._mirrored_scene` cannot be the whole of it
    here. Pinned against a Berreman stack illuminated from below with no mirror at all
    (``tests/test_anisotropic_berreman.py``).

    Only ``layer_algorithm='ETM'`` is wired: this route needs its stack solve to hand
    ``final_matching`` a ``(big_F, big_G, big_T)``, and the SMM cascade in :func:`_solve_smm`
    produces a scattering matrix in the two half-spaces' own mode bases instead. ``'SMM'``
    raises here rather than silently running the other algorithm.
    """
    xp = array_namespace(epx_conv_all)
    if layer_algorithm != 'ETM':
        raise ValueError(
            f"layer_algorithm={layer_algorithm!r} is not available on the first-order (4ff) "
            "path, which is what a populated z row selects; only 'ETM' is wired. The 4ff "
            'stack solve has to hand the shared final matching a (big_F, big_G, big_T), and '
            'the scattering cascade produces a scattering matrix in the half-spaces\' own '
            "mode bases instead. Use layer_algorithm='ETM' for a z-coupled tensor.")

    per_layer = [epx_conv_all, epy_conv_all, epz_conv_i_all, epxy_conv_all, epyx_conv_all,
                 epxz_conv_all, epyz_conv_all, epzx_conv_all, epzy_conv_all]
    if _is_backside(incidence_side):
        n_top, n_bot, thickness, *per_layer = _mirrored_scene(n_top, n_bot, thickness,
                                                              *per_layer)
        # The mirror is z -> -z, under which the z row and column change sign. The in-plane
        # block and eps_zz do not, which is why the shared mirror can carry those untouched.
        per_layer = per_layer[:5] + [None if s is None else [-m for m in s]
                                     for s in per_layer[5:]]

    kx = as_array(xp, kx)
    ky = as_array(xp, ky)
    if not xp.isdtype(kx.dtype, 'complex floating'):
        kx = xp.astype(kx, xp.complex128)
        ky = xp.astype(ky, xp.complex128)
    C = kx.dtype

    kx_g, ky_g = order_grid_kxy(xp, kx, ky)
    n_orders_y, n_orders_x = kx_g.shape
    n_orders = n_orders_x * n_orders_y
    k0 = 2 * 3.141592653589793 / wavelength
    theta = 0.0 if theta is None else theta

    k_par_sq = kx_g ** 2 + ky_g ** 2
    kz_top = xp.reshape(boundary_kz(xp, n_top ** 2, k_par_sq), (-1,))
    kz_bot = xp.reshape(boundary_kz(xp, n_bot ** 2, k_par_sq), (-1,))
    varphi = order_azimuth(xp, ky_g, kx_g, phi)

    Kx = _diag(xp, xp.reshape(kx_g, (-1,)))
    Ky = _diag(xp, xp.reshape(ky_g, (-1,)))
    F_c = _diag(xp, xp.cos(varphi))
    F_s = _diag(xp, xp.sin(varphi))

    layers = _conv_layers(xp, per_layer, thickness, n_orders, C)
    seed = _half_space_state(xp, _diag(xp, kz_bot), n_bot ** 2, F_c, F_s, n_orders, C)
    keep_field = answers_of(answers).covers(Answers.FIELD)
    G, big_T, layer_info_list = _etm_recursion(xp, k0, Kx, Ky, layers, n_orders, seed,
                                               keep_field=keep_field)
    big_F, big_G = sp_modes_2d(xp, G[:2 * n_orders], 1j * G[2 * n_orders:], F_c, F_s, n_orders)

    out = final_matching(xp, C, big_F, big_G, big_T, kz_top, kz_bot, incidence, theta,
                         n_top, n_bot, n_orders_x, n_orders_y, answers=answers)
    if keep_field:
        # `sp_modes_2d` recombines the *rows* of `G` into the s/p boundary variables and leaves
        # its columns alone, so `final_matching`'s `T1` is the unknown of the `G` above the top
        # layer unchanged -- which is that layer's forward amplitudes at its own top face. That
        # is what `field_2d_4ff` starts from; nothing has to be undone here.
        out['layer_info_list'] = layer_info_list
    return out


def _half_space_state(xp, Kz, eps, F_c, F_s, n_orders, C):
    """The downward half-space wave as a ``4ff`` state, columns ``(t_s, t_p)``.

    This is the 2ff substrate seed, written in the variables the first-order system carries.
    Reading it off rather than deriving it: the seed must satisfy
    ``sp_modes_2d(seed[:2ff], 1j*seed[2ff:]) == (big_F, big_G)`` for the pair
    :func:`wavebit._core.solve_2d.solve_2d` starts from, which fixes the four transverse
    components per column with nothing left to choose ::

        E_s = t_s              E_p = -i kz t_p / eps
        H_p = -kz t_s          H_s = -i t_p

    and the ``(s, p) -> (x, y)`` rotation ``Ax = F_c A_p - F_s A_s``,
    ``Ay = F_s A_p + F_c A_s`` puts them in the state's own basis. ``Kz`` comes from
    :func:`~wavebit._core.conventions.boundary_kz`, so this path takes the half-space
    branch from the same place every other solve does rather than from the flux rule
    :func:`split_forward_backward` applies inside a layer.

    Building the seed instead of eigendecomposing the half-space is not only cheaper. An
    isotropic half-space has both forward modes at the same ``kz``, so that eigenspace is
    degenerate and ``eig`` may return any basis of it — including one that mixes s and p, and
    one whose column scales would land straight in ``big_T``.
    """
    I = xp.astype(xp.eye(n_orders), C)
    E_s, E_p = I, -1j * Kz / eps          # per unit t_s, per unit t_p
    H_p, H_s = -Kz, -1j * I
    return _block(xp, [
        [-F_s @ E_s, F_c @ E_p],          # Ex = F_c E_p - F_s E_s
        [F_c @ E_s, F_s @ E_p],           # Ey = F_s E_p + F_c E_s
        [F_c @ H_p, -F_s @ H_s],          # Hx = F_c H_p - F_s H_s
        [F_s @ H_p, F_c @ H_s],           # Hy = F_s H_p + F_c H_s
    ])


def _conv_layers(xp, per_layer, thickness, n_orders, C):
    """``[(eps_blocks, d), ...]`` for :func:`layer_matrix_4ff`, from the nine slot stacks.

    Where the slot convention is undone: every off-diagonal slot holds the negative of its
    tensor component (``docs/off-diagonal-slot-convention.md``), and ``layer_matrix_4ff``
    reads the tensor itself. A missing stack is zero, matching how the 2ff path treats
    ``epxy_conv_all`` / ``epyx_conv_all``.
    """
    epx, epy, epz_i, epxy, epyx, epxz, epyz, epzx, epzy = per_layer
    O = xp.astype(xp.zeros((n_orders, n_orders)), C)
    n_layers = len(thickness) if not hasattr(thickness, 'shape') else thickness.shape[0]

    def slot(stack, i):
        return O if stack is None else -stack[i]

    return [(dict(exx=epx[i], eyy=epy[i], ezz_i=epz_i[i],
                  exy=slot(epxy, i), eyx=slot(epyx, i),
                  exz=slot(epxz, i), eyz=slot(epyz, i),
                  ezx=slot(epzx, i), ezy=slot(epzy, i)),
             thickness[i])
            for i in range(n_layers)]


def _diag_of(xp, v):
    return _diag(xp, xp.reshape(xp.asarray(v), (-1,)))


def _identity_blocks(xp, n_orders, dtype):
    I = xp.astype(xp.eye(n_orders), dtype)
    O = xp.astype(xp.zeros((n_orders, n_orders)), dtype)
    return dict(exx=I, exy=O, exz=O, eyx=O, eyy=I, eyz=O, ezx=O, ezy=O, ezz_i=I)
