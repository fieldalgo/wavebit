"""Single-source near-field reconstruction.

Fields come back in the same ``e^{-iωt}`` convention as everything else (see
:mod:`wavebit._core.conventions`): a wave travelling in ``+z`` carries ``exp(+i*k0*kz*z)``,
and the in-plane kernel matches the order-index sign the convolution matrices are built with.

Axes follow the array, not the picture: row ``0`` of the returned cell is ``y = 0``, the same
row of ``structure`` that the factorization reads as ``y = 0``. Turning that into a conventional
image is :func:`plot_field`'s job, and it flips there.

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

**Where it is sampled is the caller's choice.** ``sample_x`` / ``sample_y`` / ``sample_z`` each
take either a count — the uniform grid this has always used, unchanged and bit-identical — or
the coordinates themselves, as a 1-D array. Nothing here has grid structure to respect: the
transverse kernel is a pointwise ``exp(i k.r)`` and each z-slice is an independent propagation,
so arbitrary points cost the same per point as grid ones and need no interpolation. Refining a
grid leaves the shared points bit-identical, which is what says the sampling is genuinely
pointwise rather than merely appearing to be.

Explicit ``sample_x`` / ``sample_y`` are lengths in ``period``'s units and are unconstrained —
unsorted, non-uniform and outside the cell are all fine, the last being a periodic
continuation. Explicit ``sample_z`` is absolute, measured from the **top of the stack**, and is
the one with rules: it must lie within the stack, an interface belongs to the layer below it,
and the layer thicknesses have to be concrete. :func:`_z_plan` states each of those and why.

Inputs are the ``layer_info_list`` and ``T1`` the solve leaves behind — see
:mod:`wavebit._core.solve_1d`. Layers arrive bottom-to-top and are walked in reverse, so the
returned cell runs top-to-bottom in z.

The entry points mirror the solve paths: :func:`field_1d` (pure TE/TM, 3 field components),
:func:`field_1d_conical` and :func:`field_2d` (coupled s/p, 6 components, with a leading axis
selecting the incidence variants named by ``incidences``), and
:func:`field_2d_4ff` for the first-order path, which carries a different per-layer state and a
different ``E_z``. ``reconstruct_field`` picks between the last two on the *formulation*, not on
the grating type — the first-order path sets that to 2 as well.
"""
import numpy as _np

from .backend import array_namespace


#: The three incidences a coupled solve leaves stacked in ``T1``, in the order it stacks
#: them: ``'given'`` is the incidence actually asked for (``psi``, or the ``polarization``
#: pair), and ``'s'`` / ``'p'`` are the pure variants the coupled solves produce for free.
#: The same three the result object calls ``given`` / ``s_inc`` / ``p_inc``.
INCIDENCE_NAMES = ('given', 's', 'p')


def incidence_indices(incidences):
    """``incidences`` as rows of ``T1``, in the caller's own order.

    Names rather than the positional ``(True, False, False)`` triple this took until
    2026-08-20. The triple had the failure mode a boolean flag always has -- nothing at the
    call site said which slot meant what, and a reconstruction of the wrong incidence is a
    complete, plausible field cell -- and it could not express the thing a caller most often
    wants to say, which is *which* incidence, in what order the slabs come back.

    Booleans are refused by name for that reason: the old triple is still a legal Python
    tuple, and read as names it would otherwise produce an error about ``True`` far from
    where it was written.
    """
    if isinstance(incidences, str):
        raise ValueError(
            f'incidences: expected a tuple of names, got the bare string {incidences!r} -- '
            f'write ({incidences!r},). One name is still a tuple.')
    names = tuple(incidences)
    if not names:
        raise ValueError(
            'incidences: nothing requested, so there is no field to reconstruct. Pass one or '
            f'more of {INCIDENCE_NAMES}.')
    for name in names:
        if not isinstance(name, str) or isinstance(name, bool):
            raise ValueError(
                f'incidences: expected names from {INCIDENCE_NAMES}, got {name!r}. This was '
                f'a tuple of booleans until 2026-08-20; the names replaced it so the request '
                f'says which incidence it means, and so the returned slabs can be asked for '
                f'in any order.')
        if name not in INCIDENCE_NAMES:
            raise ValueError(
                f'incidences: {name!r} is not an incidence this solve carries; expected names '
                f'from {INCIDENCE_NAMES}.')
    if len(set(names)) != len(names):
        raise ValueError(f'incidences: {names} repeats an incidence.')
    return [INCIDENCE_NAMES.index(name) for name in names]


def _dev(x):
    """Device of ``x``, so arrays the core creates land beside the ones it was given."""
    return x.device if hasattr(x, 'device') else None


def _require_solved(T1, layer_info_list):
    """Reject the states that cannot produce a field, with the reason.

    Both connecting algorithms fill this state (the SMM recovers it from its cascade), so an
    empty one means no solve has run — or a stack with no layers, which has no interior.

    A solver that declared ``answers='amplitude'`` never reaches here: the backends'
    ``reconstruct_field`` refuses first, naming the declaration
    (:meth:`wavebit._core.engine.CoreSolveMixin._require_field_declaration`). That split is
    deliberate — "run a solve first" is false advice for a solver whose solves are not
    keeping the state, and running another one would change nothing. This message is for the
    two cases where a solve genuinely has not happened yet, plus the direct core caller who
    passed a lower rung to a solve and then walked its output.
    """
    if T1 is None or len(layer_info_list) == 0:
        raise RuntimeError(
            'field reconstruction needs the per-layer amplitudes a solve leaves behind: '
            'run solve() -- or solve_conv(), if you built the convolution matrices '
            'yourself -- first, on a stack with at least one layer, and with '
            "answers='field' declared so the solve keeps them.")


#: Vector parity of the mirror ``diag(1, 1, -1)``, per layout, in each one's component order.
#: Not a convention: ``E`` is a polar vector and transforms as the coordinates do, so only
#: ``E_z`` changes sign; ``H`` is axial and picks up the determinant as well, so it is
#: ``H_x`` / ``H_y`` that change and ``H_z`` that does not. The two 1D layouts fill three slots
#: with different components, so the same rule reads as two different vectors.
_MIRROR_PARITY = {
    'coupled': (1.0, 1.0, -1.0, -1.0, -1.0, 1.0),   # (Ex, Ey, Ez, Hx, Hy, Hz)
    'te': (1.0, -1.0, 1.0),                          # (Ey, Hx, Hz)
    'tm': (-1.0, 1.0, -1.0),                         # (Hy, Ex, Ez)
}


def to_mirrored_z(sample_z, thickness):
    """A caller's ``sample_z`` restated in the frame a backside solve actually walks.

    Explicit coordinates are absolute in **scene** coordinates -- that is the contract on both
    sides, and it is what makes ``incidence_side`` a statement about the light rather than
    about the axis. A backside solve walks the z-mirror of that scene, so each coordinate
    becomes ``total - z`` before it is bucketed.

    Reversing the *output* instead would not do, and that is what was happening until
    2026-08-22: :func:`_z_plan` hands explicit samples back in the caller's own order, so the
    blind flip that follows returned the request backwards, on top of having bucketed a scene
    coordinate against the mirror's thicknesses. Measured against the hand-reversed scene, both
    together were worth O(1) at every requested z -- 2.86 on a two-layer TE grating.

    A **count** passes straight through. It is a shape, not a position: "six samples per layer"
    means the same thing in either frame, and the returned axis is reversed at the end instead.

    A traced thickness passes through untouched as well, so that the refusal a caller sees is
    :func:`_z_plan`'s -- which says why a data-dependent layer choice cannot be traced -- rather
    than a numpy conversion error raised here first.
    """
    if is_count(sample_z):
        return sample_z
    try:
        total = float(_np.sum(_np.asarray(thickness, dtype=float)))
    except Exception:
        return sample_z
    return total - _np.asarray(sample_z, dtype=float)

def to_scene_frame(cell, pure_1d, tetm_index=None, flip_z=True):
    """A backside cell, put back into scene coordinates -- both axis and components.

    A backside solve *is* the top-side solve of the z-mirrored scene
    (:func:`~wavebit._core.solve_1d._mirrored_scene`), so everything it leaves behind is
    expressed in that mirror. Returning it as-is would make ``incidence_side`` change the frame
    the answer is written in, which is the one thing the scene labels promise it does not:
    index 0 of the z axis is the top of ``structure[0]`` whichever side the light came from.

    Two halves, and the second was missing until 2026-08-22. Reversing the z axis is the
    obvious one. The other is that ``z -> -z`` is *improper*, so the field components do not
    all follow the coordinates: see :data:`_MIRROR_PARITY`. Without it a backside cell comes
    back with ``E_z``, ``H_x`` and ``H_y`` negated -- which conserves energy, leaves every
    efficiency untouched, and reverses the net Poynting vector. Measured on a TE grating lit
    from below: ``S_z`` came back ``+0.343`` where the light travels upward and it must be
    negative, bit-identical to the top-side solve rather than its mirror.

    The parity is the same vector the reconstructions already apply once to reach the library's
    declared frame (``+z`` down, see the module docstring). That is not a coincidence and not a
    double-application: both are mirrors in ``z``, and this one composes on top of a cell that
    is already in the declared frame.

    ``flip_z`` is false when the caller asked for explicit coordinates. Those were restated in
    the mirror's frame on the way in (:func:`to_mirrored_z`) and come back in the caller's own
    order, so there is no axis left to reverse -- only the components. A count did not move on
    the way in, and its axis is what carries the reversal.
    """
    xp = array_namespace(cell)
    which = 'coupled' if not pure_1d else ('te' if tetm_index == 0 else 'tm')
    parity = xp.asarray(_MIRROR_PARITY[which], dtype=cell.dtype, device=_dev(cell))
    if flip_z:
        cell = xp.flip(cell, axis=0 if pure_1d else 1)
    return cell * parity

def _diag_exp(xp, diag_vals, n):
    """``exp`` of a diagonal matrix stack, given its diagonals — the legacy ``d_exp``.

    Off-diagonal entries stay exactly zero (the legacy built this by assigning into a zero
    array; multiplying an identity achieves the same without in-place mutation).
    """
    eye = xp.eye(n, dtype=diag_vals.dtype, device=_dev(diag_vals))
    return eye * xp.exp(diag_vals)[..., None, :]


def is_count(spec):
    """Is this sampling argument a *number of points* rather than the points themselves?

    Only a Python/NumPy integer counts. Anything with a length -- list, tuple, ndarray, backend
    tensor -- is read as explicit coordinates. A bare float is neither, and is rejected by
    :func:`_check_spec` rather than quietly becoming one of them.
    """
    return isinstance(spec, (int, _np.integer)) and not isinstance(spec, bool)


def _check_spec(spec, name):
    """Reject the one spelling that would otherwise mean something the caller did not ask for.

    ``sample_z=5`` is five samples and ``sample_z=[5.0]`` is one sample at ``z = 5``;
    ``sample_z=5.0`` reads like the first and would silently behave like the second. It is a
    typo often enough that guessing either way is worse than saying so.
    """
    if is_count(spec):
        return
    if _np.ndim(spec) != 1:
        raise TypeError(
            f'{name} must be an integer count or a 1-D array of coordinates; got '
            f'{spec!r}. Write {name}=5 for five samples, or {name}=[5.0] for one at 5.')


def _n_samples(spec, name='res_*'):
    _check_spec(spec, name)
    return int(spec) if is_count(spec) else len(spec)


def _axis_samples(xp, spec, extent, C, dev):
    """Sample coordinates along one in-plane axis: a count, or the coordinates themselves.

    A count reproduces the uniform inclusive grid this has always used, spelled exactly as it
    was so those samples stay bit-identical. Explicit coordinates are taken as given, in the
    same length units as ``period`` -- they need not be uniform, sorted, or inside the cell,
    because the kernel below is a plain pointwise ``exp(i k.r)`` with no grid structure to
    respect. That is also why this is cheap: cost is points x orders either way.
    """
    _check_spec(spec, 'sample_x/sample_y')
    if is_count(spec):
        return xp.linspace(0, extent, spec, dtype=C, device=dev)
    return xp.astype(xp.asarray(spec, device=dev), C)


def _inv_fourier(xp, kx, ky, period, sample_x, sample_y, k0, C, lattice=None):
    """Inverse-transform kernel ``exp(-i k.r)`` sampled on the cell, flattened over orders.

    With ``lattice`` given (and ``kx`` / ``ky`` therefore per-order, ``(n_orders_v,
    n_orders_u)``) the samples are taken on the cell's own parallelogram, ``r = u*a1 + v*a2``
    for ``u, v`` in ``[0, 1]``, and the returned grid is indexed by ``(v, u)`` rather than ``(y,
    x)``.

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
    n_u, n_v = _n_samples(sample_x, 'sample_x'), _n_samples(sample_y, 'sample_y')
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
            if is_count(spec):
                return xp.linspace(0, 1, _n_samples(spec), dtype=C, device=dev)
            if abs(a_perp) > 1e-12 * max(abs(a_par), 1.0):
                raise NotImplementedError(
                    f'explicit {axis} coordinates need an axis-aligned cell: on a sheared '
                    'lattice an (x, y) does not determine the (u, v) this path samples in. '
                    'Pass a count instead, or describe the cell as a rectangular supercell.')
            return xp.astype(xp.asarray(spec, device=dev), C) / a_par

        _check_spec(sample_x, 'sample_x')
        _check_spec(sample_y, 'sample_y')
        u_1d = _frac(sample_x, a1x, a1y, 'sample_x')
        v_1d = _frac(sample_y, a2y, a2x, 'sample_y')
        u_ph = xp.exp(1j * xp.reshape(u_1d, (1, n_u, 1)) * proj_a1)
        v_ph = xp.exp(1j * xp.reshape(v_1d, (n_v, 1, 1)) * proj_a2)
        return xp.broadcast_to(u_ph * v_ph, (n_v, n_u, kx_f.shape[0]))
    # `period` stays an array rather than a Python float: all three backends take an array
    # endpoint and give bit-identical samples, and narrowing it would break tracing whenever
    # the solver itself is jitted (period is a traced child of the JAX solver pytree).
    x_1d = _axis_samples(xp, sample_x, period[0], C, dev)
    x_2d = xp.reshape(x_1d, (1, n_u, 1)) * kx * k0
    x_2d = xp.reshape(xp.broadcast_to(x_2d, (n_v, n_u, kx.shape[0])),
                      (n_v, n_u, 1, kx.shape[0]))
    if ky is None:
        return xp.reshape(xp.exp(1j * x_2d), (n_v, n_u, -1))

    y_1d = _axis_samples(xp, sample_y, period[1], C, dev)
    y_2d = xp.reshape(y_1d, (n_v, 1, 1)) * ky * k0
    y_2d = xp.reshape(xp.broadcast_to(y_2d, (n_v, n_u, ky.shape[0])),
                      (n_v, n_u, ky.shape[0], 1))
    return xp.reshape(xp.exp(1j * x_2d) * xp.exp(1j * y_2d), (n_v, n_u, -1))


def _z_1d(xp, sample_z, d, C, dev=None):
    """Local z samples through one layer, as a flat vector.

    A count keeps the original spelling — ``linspace(0, sample_z, sample_z) / sample_z * d``
    rather than ``linspace(0, d, sample_z)`` — because the two are not identical in floating
    point and every committed field golden was captured with this one.
    """
    if is_count(sample_z):
        return xp.reshape(xp.linspace(0, sample_z, sample_z, dtype=C, device=dev), (-1,)) / sample_z * d
    return xp.astype(xp.asarray(sample_z, device=dev), C)


def _z_slices(xp, sample_z, d, C, dev=None):
    """Sample points through one layer's thickness, shaped for broadcasting over modes."""
    return xp.reshape(_z_1d(xp, sample_z, d, C, dev), (-1, 1, 1))


def _z_plan(sample_z, layer_info_list, d_index):
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

    **A count means what it always did**: every layer gets ``sample_z`` samples inclusive of both
    faces, and the layers concatenate, so the z axis is ``sample_z * n_layers`` long and every
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
    _check_spec(sample_z, 'sample_z')
    if is_count(sample_z):
        return [sample_z] * len(layer_info_list), None

    z_req = _np.asarray(sample_z, dtype=float).reshape(-1)
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
                'sample_z, or reconstruct outside the transform.') from e

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


def refuse_extra_incidences(incidences):
    """Refuse anything but ``('given',)`` on the path that carries exactly one incidence.

    The pure 1D TE/TM solve is single-polarization by construction: ``tetm`` declares the
    other degree of freedom away, so ``T1`` there is one column rather than the stack of three
    the coupled paths leave (:func:`wavebit._core.solve_1d.solve_1d`). ``'s'`` and ``'p'`` name
    variants that solve never computed.

    Refused rather than ignored, which is what it was until 2026-08-20 -- the argument was read
    on the coupled paths and silently dropped here, so ``incidences=('s', 'p')`` returned the
    declared incidence, once, with no axis to say so. That is the unread-option defect
    :meth:`~wavebit._core.engine.CoreFactorizationMixin._reject_unread_normal_method` exists to
    refuse one layer up, and the rename from ``set_field_input`` made it worse rather than
    better: three anonymous booleans read as plumbing, while a name reads as a request.
    """
    if incidence_indices(incidences) == [0]:
        return
    raise ValueError(
        f'incidences={tuple(incidences)!r}: this solve took the pure 1D TE/TM path, which '
        f'carries exactly one incidence -- the one `tetm` declared -- so '
        f"{INCIDENCE_NAMES[1]!r} and {INCIDENCE_NAMES[2]!r} name variants it never computed, "
        f'and the cell it returns has no incidence axis to put them on. Reconstruct with '
        f"incidences=('given',), or drop `tetm` and name the incidence with `psi` or "
        f'`polarization`: the same 1D structure then takes the conical path, which carries '
        f'all three for one extra solve.')

def field_1d(wavelength, kx, T1, layer_info_list, period, tetm_index,
             sample_x=20, sample_y=1, sample_z=20, incidences=('given',)):
    """Field of a pure TE/TM 1D stack.

    Returns ``(nz, sample_y, sample_x, 3)``, where ``nz`` is ``sample_z * n_layers`` for a count
    and ``len(sample_z)`` for explicit coordinates. See the module docstring on sampling.

    No leading incidence axis, unlike every other entry point here: ``tetm`` declared the
    polarization away, so this path solved one incidence and ``T1`` holds one column. Asking
    for another is refused rather than ignored -- see :func:`refuse_extra_incidences`.
    """
    refuse_extra_incidences(incidences)
    _require_solved(T1, layer_info_list)
    xp = array_namespace(T1)
    C = T1.dtype
    k0 = 2 * 3.141592653589793 / wavelength
    n_orders = kx.shape[0]
    Kx = xp.diag(kx)

    z_per_layer, unshuffle = _z_plan(sample_z, layer_info_list, 4)

    T_layer = T1
    cells = []
    for spec, (epz_conv_i, W, V, q, d, A_i, B) in zip(z_per_layer, layer_info_list[::-1]):
        nz = _n_samples(spec)
        X = xp.diag(xp.exp(-k0 * q * d))
        c1 = T_layer[:, None]
        c2 = B @ A_i @ X @ T_layer[:, None]
        z = _z_slices(xp, spec, d, C, _dev(T1))
        Qz = xp.reshape(q, (1, 1, n_orders)) * z       # diagonals of Q*z, (nz, 1, n_orders)
        Qz = xp.reshape(Qz, (nz, n_orders))
        Qd = xp.reshape(q, (1, n_orders)) * xp.reshape(xp.reshape(z, (nz,)) - d, (nz, 1))

        fwd = _diag_exp(xp, -k0 * Qz, n_orders)
        bwd = _diag_exp(xp, k0 * Qd, n_orders)
        My = W @ (fwd @ c1 + bwd @ c2)
        Mx = V @ (-fwd @ c1 + bwd @ c2)
        # TE's ``H_z = Kx Ey`` is Faraday's z row and would carry ``muz_conv_i`` on a magnetic
        # layer, as the coupled path's does. It has none to carry: the pure TE/TM operator
        # has no permeability slot, and `solve_conv` refuses a `mu` that reaches this
        # path by name ("2D path only") rather than dropping it here.
        Mz = 1j * (epz_conv_i @ Kx @ My if tetm_index else Kx @ My)

        inv_f = _inv_fourier(xp, kx, None, period, sample_x, sample_y, k0, C)
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


def _field_coupled(wavelength, kx, ky, T1, layer_info_list, period, sample_x, sample_y, sample_z,
                   incidences, conical, lattice=None):
    """Shared body of the conical and 2D reconstructions.

    They differ only in how the layer's mode matrices assemble into ``(Sx, Sy, Ux, Uy)``; the
    propagation, the z-sampling and the inverse transform are identical.

    **The two longitudinal components are algebraic, and each divides by the z slot of its own
    tensor.** ``E_z`` comes from Ampere's z row and carries ``epz_conv_i``; ``H_z`` comes from
    Faraday's z row and carries ``muz_conv_i``, which is the layer's ``mu_zz^-1`` or ``None``
    where the layer is non-magnetic. Written out, in the ``U = i H`` variables this function works
    in and with ``Kx`` / ``Ky`` normalised by ``k0``::

        eps_zz E_z =   Ky Hx - Kx Hy          ->  Sz = 1j epz_conv_i (Kx Uy - Ky Ux)
        mu_zz  H_z =   Kx Ey - Ky Ex          ->  Uz = 1j muz_conv_i (Kx Sy - Ky Sx)

    which is one line under the other's duality ``(E, H, eps, mu) -> (H, -E, mu, eps)`` --
    the same duality :func:`wavebit._core.solve_2d.layer_modes_2d` places ``mu`` by, and the
    reason ``Omega_R`` (which eliminates ``H_z``) is the block that carries ``muz_conv_i``
    there.

    ``muz_conv_i`` is a **convolution**, not a scalar: on a patterned magnetic layer ``mu_zz``
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
        n_orders_v, n_orders_u = kx.shape
    else:
        n_orders_u, n_orders_v = kx.shape[0], ky.shape[0]
    n_orders = n_orders_u * n_orders_v

    # Same two spellings the solve accepts, resolved by the same helper — 1-D separable for an
    # axis-aligned cell (broadcast exactly as before), per-order for any lattice.
    from .solve_2d import order_grid_kxy
    kx_g, ky_g = order_grid_kxy(xp, kx, ky)
    Kx, Ky = xp.diag(xp.reshape(kx_g, (-1,))), xp.diag(xp.reshape(ky_g, (-1,)))

    dev = _dev(T1)
    keep = incidence_indices(incidences)
    T_layer = xp.take(T1, xp.asarray(keep, device=dev), axis=0)
    big_I = xp.eye(T1.shape[1], dtype=C, device=dev)

    z_per_layer, unshuffle = _z_plan(sample_z, layer_info_list, 4)

    cells = []
    for spec, (epz_conv_i, W, V, q, d, big_A_i, big_B, muz_conv_i) in zip(
            z_per_layer, layer_info_list[::-1]):
        nz = _n_samples(spec)
        q_1, q_2 = q[:n_orders], q[n_orders:]
        big_X = xp.diag(xp.exp(-k0 * q * d))

        c = xp.concat([big_I, big_B @ big_A_i @ big_X], axis=0) @ T_layer
        c1p, c2p = c[:, 0:n_orders], c[:, n_orders:2 * n_orders]
        c1m, c2m = c[:, 2 * n_orders:3 * n_orders], c[:, 3 * n_orders:4 * n_orders]

        z = _z_1d(xp, spec, d, C, dev)
        def prop(qi):
            fwd = _diag_exp(xp, -k0 * xp.reshape(qi, (1, n_orders)) * xp.reshape(z, (nz, 1)), n_orders)
            bwd = _diag_exp(xp, k0 * xp.reshape(qi, (1, n_orders)) * xp.reshape(z - d, (nz, 1)), n_orders)
            return fwd[:, None, :, :], bwd[:, None, :, :]

        f1, b1 = prop(q_1)
        f2, b2 = prop(q_2)
        a1p, a1m = f1 @ c1p, b1 @ c1m          # + / - going, mode family 1
        a2p, a2m = f2 @ c2p, b2 @ c2m

        if conical:
            W_1, W_2 = W[:, :n_orders], W[:, n_orders:]
            V_11, V_12 = V[:n_orders, :n_orders], V[:n_orders, n_orders:]
            V_21, V_22 = V[n_orders:, :n_orders], V[n_orders:, n_orders:]
            Sx = W_2 @ (a2p + a2m)
            Sy = V_11 @ (a1p + a1m) + V_12 @ (a2p + a2m)
            Ux = W_1 @ (-a1p + a1m)
            Uy = V_21 @ (-a1p + a1m) + V_22 @ (-a2p + a2m)
        else:
            W_11, W_12 = W[:n_orders, :n_orders], W[:n_orders, n_orders:]
            W_21, W_22 = W[n_orders:, :n_orders], W[n_orders:, n_orders:]
            V_11, V_12 = V[:n_orders, :n_orders], V[:n_orders, n_orders:]
            V_21, V_22 = V[n_orders:, :n_orders], V[n_orders:, n_orders:]
            Sx = W_11 @ (a1p + a1m) + W_12 @ (a2p + a2m)
            Sy = W_21 @ (a1p + a1m) + W_22 @ (a2p + a2m)
            Ux = V_11 @ (-a1p + a1m) + V_12 @ (-a2p + a2m)
            Uy = V_21 @ (-a1p + a1m) + V_22 @ (-a2p + a2m)

        Sz = 1j * epz_conv_i @ (Kx @ Uy - Ky @ Ux)
        Uz = 1j * (Kx @ Sy - Ky @ Sx)
        # Skipped rather than multiplied through by an identity where the layer is
        # non-magnetic, so a non-magnetic cell stays bit-identical; and asked structurally
        # (`is not None`), which is the one form a traced `muz_conv_i` can answer.
        if muz_conv_i is not None:
            Uz = muz_conv_i @ Uz

        inv_f = _inv_fourier(xp, kx, ky, period, sample_x, sample_y, k0, C,
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
                 sample_x=20, sample_y=20, sample_z=20, incidences=('given',),
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
    the two move together. The 2ff twin carries the general form,
    ``H_z = muz_conv_i (Kx Ey - Ky Ex)``; should ``mu`` ever land on this path it needs that
    factor *and* the two z-row terms
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
        n_orders_v, n_orders_u = kx.shape
    else:
        n_orders_u, n_orders_v = kx.shape[0], ky.shape[0]
    n_orders = n_orders_u * n_orders_v

    from .solve_2d import order_grid_kxy
    kx_g, ky_g = order_grid_kxy(xp, kx, ky)
    Kx, Ky = xp.diag(xp.reshape(kx_g, (-1,))), xp.diag(xp.reshape(ky_g, (-1,)))

    dev = _dev(T1)
    keep = incidence_indices(incidences)
    T_layer = xp.take(T1, xp.asarray(keep, device=dev), axis=0)

    z_per_layer, unshuffle = _z_plan(sample_z, layer_info_list, 5)

    cells = []
    for spec, (eps, psi_f, lam_f, psi_b, lam_b, d, P_i, QP_i) in zip(z_per_layer,
                                                                     layer_info_list[::-1]):
        nz = _n_samples(spec)
        # The layer's unknown is its forward amplitude at its own top face -- that is what the
        # ETM redefinition `t' = X_f^-1 P t` bought. The backward half is referenced at the
        # bottom face, so `QP_i @ X_f` is the map to it.
        X_f = _diag_exp(xp, 1j * k0 * lam_f * d, 2 * n_orders)
        a_f = T_layer
        a_b = (QP_i @ X_f) @ T_layer

        z = _z_1d(xp, spec, d, C, dev)
        fwd = _diag_exp(xp, 1j * k0 * xp.reshape(lam_f, (1, 2 * n_orders))
                        * xp.reshape(z, (nz, 1)), 2 * n_orders)
        bwd = _diag_exp(xp, 1j * k0 * xp.reshape(lam_b, (1, 2 * n_orders))
                        * xp.reshape(z - d, (nz, 1)), 2 * n_orders)
        psi = psi_f @ (fwd[:, None] @ a_f) + psi_b @ (bwd[:, None] @ a_b)

        Ex, Ey = psi[:, :, :n_orders], psi[:, :, n_orders:2 * n_orders]
        Hx, Hy = psi[:, :, 2 * n_orders:3 * n_orders], psi[:, :, 3 * n_orders:]
        Ez = eps['ezz_i'] @ (Ky @ Hx - Kx @ Hy - eps['ezx'] @ Ex - eps['ezy'] @ Ey)
        Hz = Kx @ Ey - Ky @ Ex

        inv_f = _inv_fourier(xp, kx, ky, period, sample_x, sample_y, k0, C,
                             lattice)[:, :, None, :]
        # No parity here, deliberately: the 4ff state vector is already in the declared frame.
        comps = [inv_f @ M[:, :, None, None, :, :]
                 for M in (Ex, Ey, Ez, Hx, Hy, Hz)]
        val = xp.concat([c[..., 0] for c in comps], axis=-1)
        cells.append(xp.moveaxis(val, 1, 0))

        T_layer = (P_i @ X_f) @ T_layer

    return _unshuffle(xp, xp.concat(cells, axis=1), unshuffle, axis=1)


def field_1d_conical(wavelength, kx, ky, T1, layer_info_list, period,
                     sample_x=20, sample_y=20, sample_z=20, incidences=('given',),
                     lattice=None):
    """Field of a general 1D (conical) stack. ``(n_selected, nz, sample_y, sample_x, 6)``.

    ``nz`` is ``sample_z * n_layers`` for a count, ``len(sample_z)`` for explicit coordinates.
    """
    return _field_coupled(wavelength, kx, ky, T1, layer_info_list, period,
                          sample_x, sample_y, sample_z, incidences, conical=True, lattice=lattice)


def field_2d(wavelength, kx, ky, T1, layer_info_list, period,
             sample_x=20, sample_y=20, sample_z=20, incidences=('given',),
             lattice=None):
    """Field of a 2D stack. ``(n_selected, nz, sample_y, sample_x, 6)``.

    ``nz`` is ``sample_z * n_layers`` for a count, ``len(sample_z)`` for explicit coordinates.
    """
    return _field_coupled(wavelength, kx, ky, T1, layer_info_list, period,
                          sample_x, sample_y, sample_z, incidences, conical=False, lattice=lattice)


def plot_field(field_cell, tetm_index=0, plot_indices=(1, 1, 1, 1, 1, 1), y_slice=0, z_slice=-1,
               zx=True, yx=True):
    """Plot |field|^2 slices of a reconstructed cell (side and/or top view).

    Presentation rather than physics, but there is no reason for it to exist per backend:
    it only needs the array as NumPy.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError('plot_field() needs matplotlib installed') from e
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
