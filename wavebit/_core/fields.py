"""Single-source near-field reconstruction.

Fields come back in the same ``e^{-iωt}`` convention as everything else (see
:mod:`wavebit._core.conventions`): a wave travelling in ``+z`` carries ``exp(+i*k0*kz*z)``,
and the in-plane kernel matches the order-index sign the convolution matrices are built with.

Axes follow the array, not the picture: row ``0`` of the returned cell is ``y = 0``, the same
row of ``structure`` that the factorization reads as ``y = 0``. Turning that into a conventional
image is :func:`field_plot`'s job, and it flips there.

**The vector components are in the library's own frame, where ``+z`` points down, into the
stack** (``docs/conventions.md``) — the same ``+z`` the phase, the layer order and the
``Im(kz) > 0`` decay rule are stated against. Until 2026-08-14 they were not: every path
returned ``E_z``, ``H_x`` and ``H_y`` with the opposite sign.

It is worth being precise about what that was, because it was not an undeclared convention.
``docs/formulation.md`` §1 already carried the answer as its sanity anchor — a vacuum wave
``E = y exp(i k0 z)`` travelling ``+z`` has ``Hx = -exp(i k0 z)``, the pair ``(Ey, Hx) =
(1, -kz)`` — and ``tests/test_anisotropic_berreman.py::_vacuum_modes`` writes that pair, and
:mod:`wavebit._core.solve_2d_4ff` derives in it. The formulation, the oracle and the solver
agreed; **only this module disagreed with them.** Measured on a three-layer slab at 33
degrees, the curl-equation residual was 2.8 as shipped and 8.0e-4 (converging like the step)
once the parity was undone, and a vacuum slab returned a Poynting vector pointing back out of
the stack.

The three signs are applied at each entry point below rather than in the mode algebra, because
the mode algebra is shared with the solve and the solve was never wrong. Pinned by
``tests/test_field_core.py::test_every_path_returns_a_maxwell_field_in_the_declared_frame``
(the curl equations and their convergence rate) and, for the first time on this subsystem, by
an outside solver: ``tests/references/test_s4_field_golden.py`` agrees with S4 entry for entry
at ~1e-15 on uniform stacks, on every path.

Three more outside opinions have arrived since, and they divide by path.
``test_fmmax_field_golden.py`` reaches a *patterned* layer through the coupled 2D route, at
~1e-4. ``test_reticolo_field_golden.py`` covers :func:`field_1d` specifically — the pure TE/TM
layouts, which neither of the other two exercises — at 6.3e-14 on gratings, two-layer stacks
and absorbing ones alike, with no fitted amplitude.
``test_reticolo_conical_field_golden.py`` does the same for :func:`field_1d_conical` on a
patterned layer, at 9.8e-14, under the *same* frame transform — which is itself the result, the
coupled path carrying no frame quirk of its own. Both share one exception: the field component
normal to the grating's vertical interfaces (TM's ``E_x``, and ``E_x`` again in the conical
layout) is discontinuous there and resolved differently by any two truncations.

Turns a solved stack back into the field inside it. Each layer's mode amplitudes are
propagated to a set of z-slices, then inverse-Fourier-transformed onto an (x, y) grid.

**Where it is sampled is the caller's choice.** ``res_x`` / ``res_y`` / ``res_z`` each take
either a count — the uniform grid this has always used, unchanged and bit-identical — or the
coordinates themselves, as a 1-D array. Nothing here has grid structure to respect: the
transverse kernel is a pointwise ``exp(i k.r)`` and each z-slice is an independent
propagation, so arbitrary points cost the same per point as grid ones and need no
interpolation. Refining a grid leaves the shared points bit-identical, which is what says the
sampling is genuinely pointwise rather than merely appearing to be.

Explicit ``res_x`` / ``res_y`` are lengths in ``period``'s units and are unconstrained —
unsorted, non-uniform and outside the cell are all fine, the last being a periodic
continuation. Explicit ``res_z`` is absolute, measured from the **top of the stack**, and is
the one with rules: it must lie within the stack, an interface belongs to the layer below it,
and the layer thicknesses have to be concrete. :func:`_z_plan` states each of those and why.

Inputs are the ``layer_info_list`` and ``T1`` the solve leaves behind — see
:mod:`wavebit._core.solve_1d`. Layers arrive bottom-to-top and are walked in reverse, so the
returned cell runs top-to-bottom in z.

The entry points mirror the solve paths: :func:`field_1d` (pure TE/TM, 3 field components),
:func:`field_1d_conical` and :func:`field_2d` (coupled s/p, 6 components, with a leading axis
selecting the incidence variants requested via ``set_field_input``), and
:func:`field_2d_4ff` for the first-order path, which carries a different per-layer state and a
different ``E_z``. ``calculate_field`` picks between the last two on the *formulation*, not on
the grating type — the first-order path sets that to 2 as well.
"""
import numpy as _np

from .backend import array_namespace


def _dev(x):
    """Device of ``x``, so arrays the core creates land beside the ones it was given."""
    return x.device if hasattr(x, 'device') else None


def _require_solved(T1, layer_info_list):
    """Reject the states that cannot produce a field, with the reason.

    Both connecting algorithms fill this state (the SMM recovers it from its cascade), so an
    empty one means no solve has run — or a stack with no layers, which has no interior.

    A solver that declared ``answers='amplitude'`` never reaches here: the backends'
    ``calculate_field`` refuses first, naming the declaration
    (:meth:`wavebit._core.engine.CoreSolveMixin._require_field_declaration`). That split is
    deliberate — "run a solve first" is false advice for a solver whose solves are not
    keeping the state, and running another one would change nothing. This message is for the
    two cases where a solve genuinely has not happened yet, plus the direct core caller who
    passed a lower rung to a solve and then walked its output.
    """
    if T1 is None or len(layer_info_list) == 0:
        raise RuntimeError(
            'field reconstruction needs the per-layer amplitudes a solve leaves behind: '
            'run conv_solve (or solve) first, on a stack with at least one layer, and with '
            "answers='field' declared so the solve keeps them.")


def _diag_exp(xp, diag_vals, n):
    """``exp`` of a diagonal matrix stack, given its diagonals — the legacy ``d_exp``.

    Off-diagonal entries stay exactly zero (the legacy built this by assigning into a zero
    array; multiplying an identity achieves the same without in-place mutation).
    """
    eye = xp.eye(n, dtype=diag_vals.dtype, device=_dev(diag_vals))
    return eye * xp.exp(diag_vals)[..., None, :]


def _is_count(spec):
    """Is this sampling argument a *number of points* rather than the points themselves?

    Only a Python/NumPy integer counts. Anything with a length -- list, tuple, ndarray, backend
    tensor -- is read as explicit coordinates. A bare float is neither, and is rejected by
    :func:`_check_spec` rather than quietly becoming one of them.
    """
    return isinstance(spec, (int, _np.integer)) and not isinstance(spec, bool)


def _check_spec(spec, name):
    """Reject the one spelling that would otherwise mean something the caller did not ask for.

    ``res_z=5`` is five samples and ``res_z=[5.0]`` is one sample at ``z = 5``; ``res_z=5.0``
    reads like the first and would silently behave like the second. It is a typo often enough
    that guessing either way is worse than saying so.
    """
    if _is_count(spec):
        return
    if _np.ndim(spec) != 1:
        raise TypeError(
            f'{name} must be an integer count or a 1-D array of coordinates; got '
            f'{spec!r}. Write {name}=5 for five samples, or {name}=[5.0] for one at 5.')


def _n_samples(spec, name='res_*'):
    _check_spec(spec, name)
    return int(spec) if _is_count(spec) else len(spec)


def _axis_samples(xp, spec, extent, C, dev):
    """Sample coordinates along one in-plane axis: a count, or the coordinates themselves.

    A count reproduces the uniform inclusive grid this has always used, spelled exactly as it
    was so those samples stay bit-identical. Explicit coordinates are taken as given, in the
    same length units as ``period`` -- they need not be uniform, sorted, or inside the cell,
    because the kernel below is a plain pointwise ``exp(i k.r)`` with no grid structure to
    respect. That is also why this is cheap: cost is points x orders either way.
    """
    _check_spec(spec, 'res_x/res_y')
    if _is_count(spec):
        return xp.linspace(0, extent, spec, dtype=C, device=dev)
    return xp.astype(xp.asarray(spec, device=dev), C)


def _inv_fourier(xp, kx, ky, period, res_x, res_y, k0, C, lattice=None):
    """Inverse-transform kernel ``exp(-i k.r)`` sampled on the cell, flattened over orders.

    With ``lattice`` given (and ``kx`` / ``ky`` therefore per-order, ``(ff_y, ff_x)``) the
    samples are taken on the cell's own parallelogram, ``r = u*a1 + v*a2`` for ``u, v`` in
    ``[0, 1]``, and the returned grid is indexed by ``(v, u)`` rather than ``(y, x)``.

    Separability survives that, which is the whole reason this is a few lines rather than a
    rewrite::

        k.r = k.(u*a1 + v*a2) = u*(k.a1) + v*(k.a2)

    so the kernel is still an outer product, just of ``exp(i*k0*u*(k.a1))`` and
    ``exp(i*k0*v*(k.a2))`` instead of ``exp(i*kx*x)`` and ``exp(i*ky*y)``. Nothing extra has to
    be carried: the incident offset is already inside ``kx`` / ``ky``, so ``k.a1`` and ``k.a2``
    absorb it. On an axis-aligned cell ``k.a1 = kx*Lx`` and ``u = x/Lx``, which is the 1-D
    branch below, and that branch is left alone so its samples stay bit-identical.
    """
    dev = _dev(kx)
    nx, ny = _n_samples(res_x, 'res_x'), _n_samples(res_y, 'res_y')
    # Branch on the *layout*, not on whether a basis was supplied: an axis-aligned cell may
    # carry an explicit `diag(period)` and still hand over separable 1-D vectors, in which case
    # the old path is both correct and bit-identical.
    if kx.ndim == 2:
        a1x, a2x = float(lattice[0][0]), float(lattice[0][1])
        a1y, a2y = float(lattice[1][0]), float(lattice[1][1])
        kx_f = xp.reshape(kx, (-1,))
        ky_f = xp.reshape(ky, (-1,))
        proj_a1 = (kx_f * a1x + ky_f * a1y) * k0          # k . a1, per order
        proj_a2 = (kx_f * a2x + ky_f * a2y) * k0          # k . a2
        # This branch samples the cell's own parallelogram in fractional `(u, v)`, so a count
        # is `linspace(0, 1, n)` -- unchanged, and stated per axis so that mixing a count with
        # explicit coordinates on the other axis leaves this one bit-identical. Explicit
        # coordinates are lengths and have to be converted, and `x` alone does not determine
        # `u` on a sheared cell: refuse rather than guess. The solver does not support an
        # oblique lattice anyway, so `u = x / a1x` is exact everywhere this is reachable --
        # and it divides by the *lattice* vector rather than by `period`, which need not be
        # the same number.
        def _frac(spec, a_par, a_perp, axis):
            if _is_count(spec):
                return xp.linspace(0, 1, _n_samples(spec), dtype=C, device=dev)
            if abs(a_perp) > 1e-12 * max(abs(a_par), 1.0):
                raise NotImplementedError(
                    f'explicit {axis} coordinates need an axis-aligned cell: on a sheared '
                    'lattice an (x, y) does not determine the (u, v) this path samples in. '
                    'Pass a count instead, or describe the cell as a rectangular supercell.')
            return xp.astype(xp.asarray(spec, device=dev), C) / a_par

        _check_spec(res_x, 'res_x')
        _check_spec(res_y, 'res_y')
        u_1d = _frac(res_x, a1x, a1y, 'res_x')
        v_1d = _frac(res_y, a2y, a2x, 'res_y')
        u_ph = xp.exp(1j * xp.reshape(u_1d, (1, nx, 1)) * proj_a1)
        v_ph = xp.exp(1j * xp.reshape(v_1d, (ny, 1, 1)) * proj_a2)
        return xp.broadcast_to(u_ph * v_ph, (ny, nx, kx_f.shape[0]))
    # `period` stays an array rather than a Python float: all three backends take an array
    # endpoint and give bit-identical samples, and narrowing it would break tracing whenever
    # the solver itself is jitted (period is a traced child of the JAX solver pytree).
    x_1d = _axis_samples(xp, res_x, period[0], C, dev)
    x_2d = xp.reshape(x_1d, (1, nx, 1)) * kx * k0
    x_2d = xp.reshape(xp.broadcast_to(x_2d, (ny, nx, kx.shape[0])),
                      (ny, nx, 1, kx.shape[0]))
    if ky is None:
        return xp.reshape(xp.exp(1j * x_2d), (ny, nx, -1))

    y_1d = _axis_samples(xp, res_y, period[1], C, dev)
    y_2d = xp.reshape(y_1d, (ny, 1, 1)) * ky * k0
    y_2d = xp.reshape(xp.broadcast_to(y_2d, (ny, nx, ky.shape[0])),
                      (ny, nx, ky.shape[0], 1))
    return xp.reshape(xp.exp(1j * x_2d) * xp.exp(1j * y_2d), (ny, nx, -1))


def _z_1d(xp, res_z, d, C, dev=None):
    """Local z samples through one layer, as a flat vector.

    A count keeps the original spelling — ``linspace(0, res_z, res_z) / res_z * d`` rather than
    ``linspace(0, d, res_z)`` — because the two are not identical in floating point and every
    committed field golden was captured with this one.
    """
    if _is_count(res_z):
        return xp.reshape(xp.linspace(0, res_z, res_z, dtype=C, device=dev), (-1,)) / res_z * d
    return xp.astype(xp.asarray(res_z, device=dev), C)


def _z_slices(xp, res_z, d, C, dev=None):
    """Sample points through one layer's thickness, shaped for broadcasting over modes."""
    return xp.reshape(_z_1d(xp, res_z, d, C, dev), (-1, 1, 1))


def _z_plan(res_z, layer_info_list, d_index):
    """Turn a z request into per-layer local samples, in the order the layers are walked.

    ``d_index`` is where the thickness sits in one layer's entry, which differs by path and is
    passed rather than sniffed: it used to be read off the entry's *length*, which was only
    ever a proxy for "is this the 4ff layout", and adding one slot to a second layout would
    have silently started reading a transfer block as a thickness.

    Returns ``(per_layer, unshuffle)``. ``per_layer[k]`` is what layer ``k`` of
    ``layer_info_list[::-1]`` should be evaluated at — either the caller's count, passed
    straight through, or that layer's own local depths. ``unshuffle`` is ``None`` for a count
    and otherwise the permutation that puts the concatenated layers back into the caller's
    order.

    **A count means what it always did**: every layer gets ``res_z`` samples inclusive of both
    faces, and the layers concatenate, so the z axis is ``res_z * n_layers`` long and every
    internal interface appears twice — once as one layer's bottom, once as the next layer's
    top. Nothing about that path changes.

    **Explicit coordinates are absolute and measured from the top of the stack**, and the
    returned z axis is exactly as long as the request, in the caller's order. Three rules,
    which are choices rather than consequences:

    * A point is assigned to the layer whose span contains it, half-open ``[top, bottom)``, so
      an internal interface belongs to the layer *below* it. This only matters for ``E_z``,
      which is genuinely two-valued there; ask for ``z - eps`` to get the other limit.
    * The bottom face of the last layer is included, so the closed end of the stack is
      reachable.
    * A point outside the stack is an error, not an extrapolation. The reconstruction only
      knows the layers — the half-spaces above and below carry different mode sets that this
      state does not contain.

    Bucketing needs the interfaces as numbers, so this is the one place a *traced* thickness
    cannot be used. That is reported rather than worked around: which layer a point falls in
    is genuinely data-dependent, and silently picking a layer would differentiate the wrong
    geometry.
    """
    _check_spec(res_z, 'res_z')
    if _is_count(res_z):
        return [res_z] * len(layer_info_list), None

    z_req = _np.asarray(res_z, dtype=float).reshape(-1)
    ds = []
    for entry in layer_info_list[::-1]:          # walk order: top of the stack first
        d = entry[d_index]
        try:
            ds.append(float(d))
        except Exception as e:                   # a JAX/torch tracer, typically
            raise TypeError(
                'explicit z coordinates need the layer thicknesses as numbers, and this '
                'stack carries them as traced values. Which layer a z falls in is a '
                'data-dependent choice, so it cannot be traced through; pass a count for '
                'res_z, or reconstruct outside the transform.') from e

    edges = _np.concatenate([[0.0], _np.cumsum(ds)])
    total = edges[-1]
    if z_req.size and (z_req.min() < -1e-9 or z_req.max() > total + 1e-9):
        raise ValueError(
            f'z coordinates must lie inside the stack, 0 to {total:g} measured from its top; '
            f'got {z_req.min():g} to {z_req.max():g}. The half-spaces outside carry mode sets '
            'this reconstruction does not hold.')

    # `right=False` gives the half-open [top, bottom) rule; the -1 turns a boundary hit into
    # the layer below it, and the clip puts the stack's closed bottom face on the last layer.
    owner = _np.clip(_np.searchsorted(edges, z_req, side='right') - 1, 0, len(ds) - 1)

    per_layer, order = [], []
    for k in range(len(ds)):
        sel = _np.flatnonzero(owner == k)
        order.append(sel)
        per_layer.append(z_req[sel] - edges[k])          # absolute -> local depth
    order = _np.concatenate(order) if order else _np.zeros(0, dtype=int)
    return per_layer, _np.argsort(order)


def _unshuffle(xp, cell, unshuffle, axis):
    """Put concatenated per-layer samples back into the caller's z order."""
    if unshuffle is None:
        return cell
    return xp.take(cell, xp.asarray(unshuffle, device=_dev(cell)), axis=axis)


def field_1d(wavelength, kx, T1, layer_info_list, period, tetm_index,
             res_x=20, res_y=1, res_z=20):
    """Field of a pure TE/TM 1D stack.

    Returns ``(nz, res_y, res_x, 3)``, where ``nz`` is ``res_z * n_layers`` for a count and
    ``len(res_z)`` for explicit coordinates. See the module docstring on sampling.
    """
    _require_solved(T1, layer_info_list)
    xp = array_namespace(T1)
    C = T1.dtype
    k0 = 2 * 3.141592653589793 / wavelength
    ff = kx.shape[0]
    Kx = xp.diag(kx)

    z_per_layer, unshuffle = _z_plan(res_z, layer_info_list, 4)

    T_layer = T1
    cells = []
    for spec, (epz_conv_i, W, V, q, d, A_i, B) in zip(z_per_layer, layer_info_list[::-1]):
        nz = _n_samples(spec)
        X = xp.diag(xp.exp(-k0 * q * d))
        c1 = T_layer[:, None]
        c2 = B @ A_i @ X @ T_layer[:, None]
        z = _z_slices(xp, spec, d, C, _dev(T1))
        Qz = xp.reshape(q, (1, 1, ff)) * z            # diagonals of Q*z, shape (nz, 1, ff)
        Qz = xp.reshape(Qz, (nz, ff))
        Qd = xp.reshape(q, (1, ff)) * xp.reshape(xp.reshape(z, (nz,)) - d, (nz, 1))

        fwd = _diag_exp(xp, -k0 * Qz, ff)
        bwd = _diag_exp(xp, k0 * Qd, ff)
        My = W @ (fwd @ c1 + bwd @ c2)
        Mx = V @ (-fwd @ c1 + bwd @ c2)
        # TE's ``H_z = Kx Ey`` is Faraday's z row and would carry ``muz_i`` on a magnetic
        # layer, as the coupled path's does. It has none to carry: the pure TE/TM operator
        # has no permeability slot, and `solve_for_conv` refuses a `mu` that reaches this
        # path by name ("2D path only") rather than dropping it here.
        Mz = 1j * (epz_conv_i @ Kx @ My if tetm_index else Kx @ My)

        inv_f = _inv_fourier(xp, kx, None, period, res_x, res_y, k0, C)
        sgn = -1j if tetm_index == 0 else 1j
        Fy = inv_f[:, :, None, :] @ My[:, None, None, :, :]
        Fx = sgn * (inv_f[:, :, None, :] @ Mx[:, None, None, :, :])
        Fz = sgn * (inv_f[:, :, None, :] @ Mz[:, None, None, :, :])
        # Into the declared frame (module docstring): E_z, H_x, H_y flip, E_t and H_z do not.
        # The two layouts fill the same three slots with different components, so the pattern
        # is not the same one twice -- TE is (Ey, Hx, Hz) and only its middle slot is magnetic
        # and transverse; TM is (Hy, Ex, Ez) and its *outer* two are what the parity moves.
        par_y, par_x, par_z = (1, -1, 1) if tetm_index == 0 else (-1, 1, -1)
        cells.append(xp.concat([par_y * Fy[..., 0], par_x * Fx[..., 0],
                                par_z * Fz[..., 0]], axis=-1))

        T_layer = A_i @ X @ T_layer

    return _unshuffle(xp, xp.concat(cells, axis=0), unshuffle, axis=0)


def _field_coupled(wavelength, kx, ky, T1, layer_info_list, period, res_x, res_y, res_z,
                   set_field_input, conical, lattice=None):
    """Shared body of the conical and 2D reconstructions.

    They differ only in how the layer's mode matrices assemble into ``(Sx, Sy, Ux, Uy)``; the
    propagation, the z-sampling and the inverse transform are identical.

    **The two longitudinal components are algebraic, and each divides by the z slot of its own
    tensor.** ``E_z`` comes from Ampere's z row and carries ``epz_conv_i``; ``H_z`` comes from
    Faraday's z row and carries ``muz_i``, which is the layer's ``mu_zz^-1`` or ``None`` where
    the layer is non-magnetic. Written out, in the ``U = i H`` variables this function works
    in and with ``Kx`` / ``Ky`` normalised by ``k0``::

        eps_zz E_z =   Ky Hx - Kx Hy          ->  Sz = 1j epz_conv_i (Kx Uy - Ky Ux)
        mu_zz  H_z =   Kx Ey - Ky Ex          ->  Uz = 1j muz_i      (Kx Sy - Ky Sx)

    which is one line under the other's duality ``(E, H, eps, mu) -> (H, -E, mu, eps)`` --
    the same duality :func:`wavebit._core.solve_2d.layer_modes_2d` places ``mu`` by, and the
    reason ``Omega_R`` (which eliminates ``H_z``) is the block that carries ``muz_i`` there.

    ``muz_i`` is a **convolution**, not a scalar: on a patterned magnetic layer ``mu_zz``
    varies over the cell, so dividing it out is a matrix solve against the same inverse-rule
    slot the solve used, exactly as ``epz_conv_i`` is for ``E_z``. Until 2026-08-20 this line
    carried the ``mu = 1`` form and a magnetic layer's ``H_z`` came back scaled by ``mu_zz``
    -- silently, because every other component was right and energy still conserved. See
    ``tests/test_magnetic_layer.py``.
    """
    _require_solved(T1, layer_info_list)
    xp = array_namespace(T1)
    C = T1.dtype
    k0 = 2 * 3.141592653589793 / wavelength
    if kx.ndim == 2:
        ff_y, ff_x = kx.shape
    else:
        ff_x, ff_y = kx.shape[0], ky.shape[0]
    ff_xy = ff_x * ff_y

    # Same two spellings the solve accepts, resolved by the same helper — 1-D separable for an
    # axis-aligned cell (broadcast exactly as before), per-order for any lattice.
    from .solve_2d import order_grid_kxy
    kx_g, ky_g = order_grid_kxy(xp, kx, ky)
    Kx, Ky = xp.diag(xp.reshape(kx_g, (-1,))), xp.diag(xp.reshape(ky_g, (-1,)))

    dev = _dev(T1)
    keep = [i for i, on in enumerate(set_field_input) if on]
    T_layer = xp.take(T1, xp.asarray(keep, device=dev), axis=0)
    big_I = xp.eye(T1.shape[1], dtype=C, device=dev)

    z_per_layer, unshuffle = _z_plan(res_z, layer_info_list, 4)

    cells = []
    for spec, (epz_conv_i, W, V, q, d, big_A_i, big_B, muz_i) in zip(z_per_layer,
                                                                     layer_info_list[::-1]):
        nz = _n_samples(spec)
        q_1, q_2 = q[:ff_xy], q[ff_xy:]
        big_X = xp.diag(xp.exp(-k0 * q * d))

        c = xp.concat([big_I, big_B @ big_A_i @ big_X], axis=0) @ T_layer
        c1p, c2p = c[:, 0:ff_xy], c[:, ff_xy:2 * ff_xy]
        c1m, c2m = c[:, 2 * ff_xy:3 * ff_xy], c[:, 3 * ff_xy:4 * ff_xy]

        z = _z_1d(xp, spec, d, C, dev)
        def prop(qi):
            fwd = _diag_exp(xp, -k0 * xp.reshape(qi, (1, ff_xy)) * xp.reshape(z, (nz, 1)), ff_xy)
            bwd = _diag_exp(xp, k0 * xp.reshape(qi, (1, ff_xy)) * xp.reshape(z - d, (nz, 1)), ff_xy)
            return fwd[:, None, :, :], bwd[:, None, :, :]

        f1, b1 = prop(q_1)
        f2, b2 = prop(q_2)
        a1p, a1m = f1 @ c1p, b1 @ c1m          # + / - going, mode family 1
        a2p, a2m = f2 @ c2p, b2 @ c2m

        if conical:
            W_1, W_2 = W[:, :ff_xy], W[:, ff_xy:]
            V_11, V_12 = V[:ff_xy, :ff_xy], V[:ff_xy, ff_xy:]
            V_21, V_22 = V[ff_xy:, :ff_xy], V[ff_xy:, ff_xy:]
            Sx = W_2 @ (a2p + a2m)
            Sy = V_11 @ (a1p + a1m) + V_12 @ (a2p + a2m)
            Ux = W_1 @ (-a1p + a1m)
            Uy = V_21 @ (-a1p + a1m) + V_22 @ (-a2p + a2m)
        else:
            W_11, W_12 = W[:ff_xy, :ff_xy], W[:ff_xy, ff_xy:]
            W_21, W_22 = W[ff_xy:, :ff_xy], W[ff_xy:, ff_xy:]
            V_11, V_12 = V[:ff_xy, :ff_xy], V[:ff_xy, ff_xy:]
            V_21, V_22 = V[ff_xy:, :ff_xy], V[ff_xy:, ff_xy:]
            Sx = W_11 @ (a1p + a1m) + W_12 @ (a2p + a2m)
            Sy = W_21 @ (a1p + a1m) + W_22 @ (a2p + a2m)
            Ux = V_11 @ (-a1p + a1m) + V_12 @ (-a2p + a2m)
            Uy = V_21 @ (-a1p + a1m) + V_22 @ (-a2p + a2m)

        Sz = 1j * epz_conv_i @ (Kx @ Uy - Ky @ Ux)
        Uz = 1j * (Kx @ Sy - Ky @ Sx)
        # Skipped rather than multiplied through by an identity where the layer is
        # non-magnetic, so a non-magnetic cell stays bit-identical; and asked structurally
        # (`is not None`), which is the one form a traced `muz_i` can answer.
        if muz_i is not None:
            Uz = muz_i @ Uz

        inv_f = _inv_fourier(xp, kx, ky, period, res_x, res_y, k0, C,
                             lattice)[:, :, None, :]
        # `-1j` converts out of `U = i H`; the three minus signs on top of it are the parity
        # into the declared frame (module docstring), so `H_z` keeps the bare conversion.
        comps = [inv_f @ M[:, :, None, None, :, :] for M in (Sx, Sy, -Sz)]
        comps += [-1j * (inv_f @ M[:, :, None, None, :, :]) for M in (-Ux, -Uy, Uz)]
        val = xp.concat([c[..., 0] for c in comps], axis=-1)
        cells.append(xp.moveaxis(val, 1, 0))

        T_layer = big_A_i @ big_X @ T_layer

    return _unshuffle(xp, xp.concat(cells, axis=1), unshuffle, axis=1)


def field_2d_4ff(wavelength, kx, ky, T1, layer_info_list, period,
                 res_x=20, res_y=20, res_z=20, set_field_input=(True, False, False),
                 lattice=None):
    """Field of a stack solved on the first-order (4ff) path. Same shape as :func:`field_2d`.

    The transverse half is *simpler* than the second-order reconstruction, not harder. There
    the layer's state has to be reassembled from ``W`` (the transverse ``E``) and ``V`` (its
    partner) with a sign per direction; here the eigenvector **is** the state — reading
    ``(Ex, Ey, Hx, Hy)`` off the four row blocks of ``psi`` is the whole of it. The two
    directions are separate mode sets rather than one set with a sign, so each is propagated
    from its own reference face: forward amplitudes are carried at the layer's top, backward
    ones at its bottom, which is the same split that makes the stack solve stable
    (:func:`wavebit._core.solve_2d_4ff._propagate_s`).

    ``E_z`` is the one component that cannot be carried over. Eliminating it from the curl
    equations with a **populated** permittivity ``z`` row keeps two terms the second-order form
    never has::

        Ez = ezz^-1 [ (Ky Hx - Kx Hy) - ezx Ex - ezy Ey ]

    :func:`_field_coupled`'s ``Sz = 1j * epz_conv_i @ (Kx @ Uy - Ky @ Ux)`` is that expression
    at ``ezx = ezy = 0`` — the same line with ``U = i H`` substituted — so the 2ff spelling is a
    special case of this one, not a shared one.

    ``H_z = Kx Ey - Ky Ex`` is Faraday's z row at ``mu = 1``, and ``mu = 1`` is the only thing
    this path can be handed: :meth:`wavebit._core.engine.CoreSolveMixin.solve_2d` refuses a
    permeability before routing here, by name. That refusal is what makes this line correct,
    so it is not an assumption to be checked here — it is a precondition held elsewhere, and
    the two move together. The 2ff twin carries the general form, ``H_z = muz_i (Kx Ey - Ky
    Ex)``; should ``mu`` ever land on this path it needs that factor *and* the two z-row terms
    ``mu_zx H_x`` / ``mu_zy H_y`` this formulation would then be the one able to express,
    exactly as ``E_z`` above keeps ``ezx`` / ``ezy``.

    The returned components are ``(Ex, Ey, Ez, Hx, Hy, Hz)``, matching :func:`field_2d`. This
    path needs **no** parity at the output, and it is the only one that does not:
    :mod:`wavebit._core.solve_2d_4ff` already derives everything with ``+z`` down along the
    propagation direction — which is what makes ``Im(lam_f) >= 0`` mean "does not grow" — and
    that is the library's declared frame. The state vector is therefore already in it.

    This is the reverse of how it shipped for one day. When reconstruction landed on this path
    (2026-08-14) the other three paths were returning ``E_z``, ``H_x``, ``H_y`` in the opposite
    frame, so this one applied the parity to match them rather than break a caller who flips
    ``formulation``. The vacuum slab that measurement rested on is what then showed the other
    three were wrong rather than merely different, and the parity moved to where it belongs.
    """
    _require_solved(T1, layer_info_list)
    xp = array_namespace(T1)
    C = T1.dtype
    k0 = 2 * 3.141592653589793 / wavelength
    if kx.ndim == 2:
        ff_y, ff_x = kx.shape
    else:
        ff_x, ff_y = kx.shape[0], ky.shape[0]
    ff = ff_x * ff_y

    from .solve_2d import order_grid_kxy
    kx_g, ky_g = order_grid_kxy(xp, kx, ky)
    Kx, Ky = xp.diag(xp.reshape(kx_g, (-1,))), xp.diag(xp.reshape(ky_g, (-1,)))

    dev = _dev(T1)
    keep = [i for i, on in enumerate(set_field_input) if on]
    T_layer = xp.take(T1, xp.asarray(keep, device=dev), axis=0)

    z_per_layer, unshuffle = _z_plan(res_z, layer_info_list, 5)

    cells = []
    for spec, (eps, psi_f, lam_f, psi_b, lam_b, d, P_i, QP_i) in zip(z_per_layer,
                                                                     layer_info_list[::-1]):
        nz = _n_samples(spec)
        # The layer's unknown is its forward amplitude at its own top face -- that is what the
        # ETM redefinition `t' = X_f^-1 P t` bought. The backward half is referenced at the
        # bottom face, so `QP_i @ X_f` is the map to it.
        X_f = _diag_exp(xp, 1j * k0 * lam_f * d, 2 * ff)
        a_f = T_layer
        a_b = (QP_i @ X_f) @ T_layer

        z = _z_1d(xp, spec, d, C, dev)
        fwd = _diag_exp(xp, 1j * k0 * xp.reshape(lam_f, (1, 2 * ff))
                        * xp.reshape(z, (nz, 1)), 2 * ff)
        bwd = _diag_exp(xp, 1j * k0 * xp.reshape(lam_b, (1, 2 * ff))
                        * xp.reshape(z - d, (nz, 1)), 2 * ff)
        psi = psi_f @ (fwd[:, None] @ a_f) + psi_b @ (bwd[:, None] @ a_b)

        Ex, Ey = psi[:, :, :ff], psi[:, :, ff:2 * ff]
        Hx, Hy = psi[:, :, 2 * ff:3 * ff], psi[:, :, 3 * ff:]
        Ez = eps['ezz_i'] @ (Ky @ Hx - Kx @ Hy - eps['ezx'] @ Ex - eps['ezy'] @ Ey)
        Hz = Kx @ Ey - Ky @ Ex

        inv_f = _inv_fourier(xp, kx, ky, period, res_x, res_y, k0, C,
                             lattice)[:, :, None, :]
        # No parity here, deliberately: the 4ff state vector is already in the declared frame.
        comps = [inv_f @ M[:, :, None, None, :, :]
                 for M in (Ex, Ey, Ez, Hx, Hy, Hz)]
        val = xp.concat([c[..., 0] for c in comps], axis=-1)
        cells.append(xp.moveaxis(val, 1, 0))

        T_layer = (P_i @ X_f) @ T_layer

    return _unshuffle(xp, xp.concat(cells, axis=1), unshuffle, axis=1)


def field_1d_conical(wavelength, kx, ky, T1, layer_info_list, period,
                     res_x=20, res_y=20, res_z=20, set_field_input=(True, False, False),
                     lattice=None):
    """Field of a general 1D (conical) stack. ``(n_selected, nz, res_y, res_x, 6)``.

    ``nz`` is ``res_z * n_layers`` for a count, ``len(res_z)`` for explicit coordinates.
    """
    return _field_coupled(wavelength, kx, ky, T1, layer_info_list, period,
                          res_x, res_y, res_z, set_field_input, conical=True, lattice=lattice)


def field_2d(wavelength, kx, ky, T1, layer_info_list, period,
             res_x=20, res_y=20, res_z=20, set_field_input=(True, False, False),
             lattice=None):
    """Field of a 2D stack. ``(n_selected, nz, res_y, res_x, 6)``.

    ``nz`` is ``res_z * n_layers`` for a count, ``len(res_z)`` for explicit coordinates.
    """
    return _field_coupled(wavelength, kx, ky, T1, layer_info_list, period,
                          res_x, res_y, res_z, set_field_input, conical=False, lattice=lattice)


def field_plot(field_cell, tetm_index=0, plot_indices=(1, 1, 1, 1, 1, 1), y_slice=0, z_slice=-1,
               zx=True, yx=True):
    """Plot |field|^2 slices of a reconstructed cell (side and/or top view).

    Presentation rather than physics, but there is no reason for it to exist per backend:
    it only needs the array as NumPy.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError('field_plot() needs matplotlib installed') from e
    import numpy as np

    field_cell = np.asarray(field_cell)

    if field_cell.shape[-1] == 6:                     # 2D / conical grating
        title = ['2D Ex', '2D Ey', '2D Ez', '2D Hx', '2D Hy', '2D Hz']
    elif tetm_index == 0:                             # 1D TE
        title = ['1D Ey', '1D Hx', '1D Hz']
    else:                                             # 1D TM
        title = ['1D Hy', '1D Ex', '1D Ez']

    for enabled, sl, view, ylabel in ((zx, (slice(None), y_slice), 'Side View', 'Z'),
                                      (yx, (z_slice,), 'Top View', 'Y')):
        if not enabled:
            continue
        for idx, name in enumerate(title):
            if not plot_indices[idx]:
                continue
            image = abs(field_cell[sl + (slice(None), idx)]) ** 2
            if ylabel == 'Y':
                # the cell's y runs with the array index; an image reads top-down, so the
                # flip that used to live in the reconstruction lives here instead
                image = image[::-1]
            plt.imshow(image, cmap='jet', aspect='auto')
            plt.colorbar()
            plt.title(f'{name}, {view}')
            plt.xlabel('X')
            plt.ylabel(ylabel)
            plt.show()
