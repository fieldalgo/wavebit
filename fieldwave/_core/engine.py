"""Bridge from the solver classes to the single-source core.

``CoreSolveMixin`` reimplements the three ``_BaseRCWA`` solve methods on top of
:mod:`fieldwave._core.solve_1d` / :mod:`fieldwave._core.solve_2d`. It keeps their exact contract —
same arguments, same returned result mapping, and the same ``layer_info_list`` / ``T1``
side-effects that field reconstruction reads — so mixing it in swaps the *implementation*
without touching the public API.

Because the core is backend-agnostic, one mixin serves NumPy, JAX and PyTorch alike; the
per-backend ``transfer_method`` chains it replaces existed in triplicate.
"""
from . import analytic as _analytic
from . import boundary as _boundary
from . import factorization as _fac
from . import lattice as _lattice
from .backend import _device_of, array_namespace
from .enums import (DEFAULT_FACTORIZATION, DEFAULT_FORMULATION, DEFAULT_FOURIER_COEFFICIENTS,
                    Factorization, Formulation, FourierCoefficients, coefficients_of,
                    factorization_of, formulation_of, resolve_fourier_options)
from .geometry import StaticGeometry
from .scattering import smm_1d as _smm_1d, smm_1d_conical as _smm_1d_conical, smm_2d as _smm_2d
from .solve_1d import solve_1d as _solve_1d, solve_1d_conical as _solve_1d_conical
from .solve_2d import solve_2d as _solve_2d
from .solve_2d_4ff import solve_2d_4ff as _solve_2d_4ff

# The four convolution slots that hold the permittivity's z row and column, in the order the
# solve methods take them. Naming them once keeps the dispatch message specific about which
# one was given -- see `CoreSolveMixin._formulation_for`.
_Z_ROW_SLOTS = ('epxz_conv_all', 'epyz_conv_all', 'epzx_conv_all', 'epzy_conv_all')

# Connecting algorithm -> the core solve for each grating type. ETM propagates an enhanced
# transmittance matrix (Moharam et al. 1995); SMM cascades per-layer scattering matrices. Same modes, same boundary
# conventions, same answers (~1e-14), both able to reconstruct fields; SMM costs ~1.1-1.4x on
# efficiencies and a little more once it also recovers the per-layer amplitudes. The two
# solves in each row take identical arguments, which is what makes the dispatch a lookup.
_SOLVES = {
    'ETM': {'1d': _solve_1d, 'conical': _solve_1d_conical, '2d': _solve_2d},
    'SMM': {'1d': _smm_1d, 'conical': _smm_1d_conical, '2d': _smm_2d},
}


def _concrete_period(value):
    """``period`` as two plain floats, or ``None`` if it has none to give.

    ``period`` is a differentiable parameter (see ``tests/test_gradients.py``), so
    it arrives in three shapes and all three are legitimate. Plain numbers convert outright. A
    grad-tracking torch tensor has to be detached first. A ``jax.grad`` tracer refuses both a
    whole-array conversion *and* ``float()`` — unlike the boolean narrowing
    :func:`~fieldwave._core.factorization._as_bools` relies on, there is no way to read a
    scalar out of one — so the honest answer there is ``None``.

    ``None`` costs nothing when a frozen geometry is present, because
    :meth:`CoreFactorizationMixin._geometry_lattice_for` recovers it from the record. Without
    one it falls back to the legacy index-space normal, unchanged.
    """
    import numpy as np
    if value is None:
        return None
    if hasattr(value, 'detach'):
        value = value.detach()
    try:
        p = np.atleast_1d(np.asarray(value, dtype=float))
    except Exception:
        try:
            items = value if isinstance(value, (list, tuple)) else list(value)
            p = np.asarray([float(v.detach() if hasattr(v, 'detach') else v) for v in items],
                           dtype=float)
        except Exception:             # a jax tracer: no primal to read
            return None
    if p.size == 1:
        return float(p[0]), float(p[0])
    return float(p[0]), float(p[1])


def _recorded_lattice(normal_field_source):
    """The cell basis a frozen record was taken on, or ``None`` if it does not name one.

    The record stores the basis scaled to ``|a1| == 1``
    (:func:`~fieldwave._core.factorization._lattice_key`), because that is all the normal field
    depends on — a uniform rescale of the cell leaves it identical. That scaled basis is itself
    a perfectly good basis to hand back: it re-derives to exactly the key the staleness check
    compares against, so a replay can never be refused for having reconstructed it differently.

    Layers disagreeing, or a record predating the basis being an input, both answer ``None``:
    neither names a single cell, and inventing one would be worse than falling back.
    """
    import numpy as np
    if not normal_field_source:
        return None
    entries = [e for e in normal_field_source if len(e) > 2]
    if len(entries) != len(normal_field_source):
        return None
    keys = {e[2] for e in entries}
    if len(keys) != 1:
        return None
    key = keys.pop()
    if key is None:
        return None
    return np.asarray(key, dtype=float).reshape(2, 2)


#: One message for both directions of the same mistake — the ``psi`` setters raise it when a
#: pair is already set, the ``polarization`` setter when ``psi`` was given explicitly.
_PSI_POLARIZATION_CLASH = (
    'psi and polarization both given: they specify the same thing (the incident '
    'polarization), so one must be dropped rather than silently overridden. '
    'psi = v is exactly polarization = (sin(v), cos(v)).')


class CoreSolveMixin:
    """Provides ``solve_1d`` / ``solve_1d_conical`` / ``solve_2d`` backed by the core."""

    _incidence_side = 'top'

    @property
    def incidence_side(self):
        """Which medium the illumination arrives from: ``'top'`` (default) or ``'bottom'``.

        ``n_top`` / ``n_bot`` / ``ucell`` order are *scene* labels and never move — this
        selects the illumination only. Results are incidence-relative (the port model):
        ``R`` is power back into the incidence medium, ``T`` power through the far side,
        the Fresnel amplitudes keep their contract, and ``theta`` / ``phi`` are measured
        against the incidence medium's normal (``n_bot``'s, for ``'bottom'``). User
        post-processing is therefore identical for both sides. See ``docs/conventions.md``.
        """
        return self._incidence_side

    @incidence_side.setter
    def incidence_side(self, value):
        if value not in ('top', 'bottom'):
            raise ValueError(
                f"incidence_side must be 'top' or 'bottom', got {value!r}")
        self._incidence_side = value

    @property
    def n_inc(self):
        """The incidence medium's index — the one ``theta`` / ``phi`` / ``kx`` live in.

        The in-plane wavevector build (each backend's ``get_kx_ky_vector``) reads this
        rather than ``n_top``, so ``kx_m = n_inc*sin(theta)*cos(phi) + m*wl/period`` holds
        for either illumination side with one spelling.
        """
        return self.n_bot if self._incidence_side == 'bottom' else self.n_top

    def _psi_set_explicitly(self):
        """Called by the backend ``psi`` setters on a non-``None`` assignment.

        Refuses the assignment while a ``polarization`` pair is set (the two spell the same
        physical input), and otherwise records that ``psi`` was *given* rather than derived
        from ``pol`` — the distinction the ``polarization`` setter's own refusal reads.
        """
        if self._polarization is not None:
            raise ValueError(_PSI_POLARIZATION_CLASH)
        self._psi_given = True

    def _solve_fn(self, grating_type):
        """The core solve for this instance's ``connecting_algo`` and the given grating type."""
        try:
            return _SOLVES[self.connecting_algo][grating_type]
        except KeyError:
            raise ValueError(
                f'connecting_algo {self.connecting_algo!r} is not supported '
                f'(expected one of {sorted(_SOLVES)})') from None

    _frozen_geometry = None
    _lattice_basis = None
    _polarization = None
    #: Whether ``psi`` was set explicitly (rather than derived from ``pol``); the backend
    #: ``psi`` setters maintain it. It is what lets ``polarization`` refuse to coexist with an
    #: explicit ``psi`` without also refusing the ``psi`` every solver derives from ``pol``.
    _psi_given = False

    @property
    def polarization(self):
        """The incident polarization as a complex amplitude pair ``(a_s, a_p)`` — or ``None``.

        The pair lives in the same ``(s, p)`` basis as ``psi`` and the returned amplitudes:
        ``(1, 0)`` is pure s (``psi = pi/2``), ``(0, 1)`` pure p (``psi = 0``), and
        ``psi = v`` is exactly ``(sin(v), cos(v))``. What the pair adds is the relative
        phase ``psi`` cannot spell — ``(1, -1j)/sqrt(2)`` and ``(1, +1j)/sqrt(2)`` are the
        circular states. The overall magnitude is divided out (efficiencies are per unit
        incident flux, amplitudes per unit incident E-field); the overall phase is kept and
        carries into the complex amplitudes. See
        :func:`~fieldwave._core.conventions.incident_sp_amplitudes`.

        ``polarization`` and an explicit ``psi`` are mutually exclusive — setting both is an
        error, not an override. A 1D structure given a pair is solved on the conical path,
        which carries both polarizations; the pure 1D TE/TM path is single-polarization by
        construction.
        """
        return self._polarization

    @polarization.setter
    def polarization(self, polarization):
        if polarization is None:
            self._polarization = None
            return
        if self._psi_given:
            raise ValueError(_PSI_POLARIZATION_CLASH)
        pair = tuple(polarization)
        if len(pair) != 2:
            raise ValueError(
                f'polarization must be a pair (a_s, a_p) of complex amplitudes, '
                f'got {len(pair)} value(s)')
        try:
            if complex(pair[0]) == 0 and complex(pair[1]) == 0:
                raise ValueError(
                    'polarization = (0, 0) carries no incident field; give at least one '
                    'nonzero amplitude (the pair is normalized to unit flux, so its overall '
                    'scale is free)')
        except TypeError:
            pass                      # traced values: concreteness is not theirs to prove
        self._polarization = pair

    @property
    def lattice(self):
        """The unit cell's ``(2, 2)`` basis, columns ``a1`` / ``a2`` — or ``None``.

        ``None`` is the default and means the axis-aligned cell ``diag(period)``. It is a
        distinct value rather than a materialised ``diag(period)`` on purpose: ``period`` may
        be a JAX tracer inside a jitted solve, and a basis has to be concrete (it decides
        control flow and reaches SciPy). Leaving the default un-materialised keeps every
        existing call off that conversion entirely, so the orthogonal path is untouched.

        Assigning a basis selects the non-orthogonal path; ``period`` then reports the
        basis-vector lengths (see :func:`fieldwave._core.lattice.periods_of`), and the raster
        is read in lattice coordinates — ``ucell[layer][j, i]`` sampling
        ``r = (i+0.5)/Nx * a1 + (j+0.5)/Ny * a2``.
        """
        return self._lattice_basis

    @lattice.setter
    def lattice(self, value):
        if value is None:
            self._lattice_basis = None
            return
        A = _lattice.as_lattice(lattice=value)
        self._lattice_basis = A
        self.period = _lattice.periods_of(A)

    @property
    def lattice_is_axis_aligned(self):
        """Whether the cell is ``diag(Lx, Ly)`` — i.e. whether the separable k-layout applies."""
        return self._lattice_basis is None or _lattice.is_axis_aligned(self._lattice_basis)

    def _geometry_lattice_for(self, geometry):
        """The cell basis the *geometry* steps should use — live, or recovered from the record.

        One accessor rather than two, because a skewed basis and a non-square pixel are the
        same question asked twice: both ask what one raster step means in real space, and a 2x2
        answers it in either case. An axis-aligned cell gets ``diag(period)`` — it is *not*
        withheld, because the metric it enables fixes a real defect there too (a raster whose
        pixels are not square in real space had its normal up to ~18 degrees off; see
        :func:`~fieldwave._core.factorization.normal_vector_field`).

        ``period`` is readable except when it is the variable under differentiation, which is
        exactly when a caller is most likely to also be replaying a frozen geometry. Freezing
        happens eagerly on a concrete ``period``, so the record was built on a real pixel shape;
        letting an unreadable one arrive as ``None`` would make the same structure look like a
        different raster and the replay would be refused
        (:func:`~fieldwave._core.factorization._check_normal_field_still_fits`). Taking it back
        out of the record rather than storing it twice keeps the two in step, and is the honest
        answer besides — the field being replayed *is* the one built on that pixel shape,
        whatever ``period`` has since become.
        """
        if not self.lattice_is_axis_aligned:
            return self._lattice_basis
        period = _concrete_period(self.period)
        if period is not None:
            return _lattice.as_lattice(period=period)
        return None if geometry is None else _recorded_lattice(geometry.normal_field_source)

    def _kxy_2d(self, wavelength):
        """Per-order ``(kx, ky)`` for the 2D solve, in whichever layout the lattice admits.

        Axis-aligned cells keep the two 1-D vectors untouched — same objects, same values, so
        nothing downstream can tell this method was added. A skewed cell cannot be written that
        way and gets the ``(ff_y, ff_x)`` Cartesian components instead; both spellings are what
        :func:`fieldwave._core.solve_2d.order_grid_kxy` accepts.

        The incident components are read back out of the 1-D vectors at ``m = n = 0`` rather
        than recomputed. That entry *is* ``k_inc`` by construction, so the backend's own
        ``sin(theta)`` handling — which differs between NumPy, JAX and Torch for tracing
        reasons — stays the single source for incidence, and the lattice basis is the only new
        arithmetic here.
        """
        kx, ky = self.get_kx_ky_vector(wavelength)
        A = self._lattice_basis
        if A is None or _lattice.is_axis_aligned(A):
            return kx, ky
        fto_x, fto_y = int(self.fto[0]), int(self.fto[1])
        xp = array_namespace(kx)
        return _lattice.order_kxy(xp, A, fto_x, fto_y, wavelength,
                                  kx[fto_x], ky[fto_y],
                                  dtype=kx.dtype, device=_device_of(kx))

    def _finish(self, out):
        """Store the field-reconstruction state and return just the result mapping."""
        self.layer_info_list = out.pop('layer_info_list')
        self.T1 = out.pop('T1')
        return out

    _formulation = DEFAULT_FORMULATION
    _formulation_assigned = None

    @property
    def formulation(self):
        """Which layer formulation the 2D solve uses: ``'auto'`` (default), ``'2ff'``, ``'4ff'``.

        ``'2ff'`` is the second-order form the library has always used, one field pair per
        order. ``'4ff'`` carries the full transverse state and is what a populated
        permittivity ``z`` row — the Voigt / magneto-optic geometry — needs; it costs about 8x,
        because the eigenproblem doubles and ``eig`` is cubic.

        The default reads the problem rather than asking: a solve handed any of the four
        ``ep{xz,yz,zx,zy}_conv_all`` slots takes ``'4ff'``, everything else ``'2ff'``. That
        decision is the solver's because a caller with a Faraday stack has no way to know that
        one populated ``eps_yz`` is what would cost them eight times more.

        Setting it overrides that, in the two directions that mean something:

        * ``'4ff'`` forces the first-order path on a problem that does not need it. This is
          how the two formulations are cross-checked against each other — the same scene run
          both ways must agree, and no external oracle provides that.
        * ``'2ff'`` refuses a z row instead of dropping it. Dropping it is the dangerous
          failure: the reduced tensor is a perfectly good medium, so the solve succeeds and
          ``R + T`` stays at 1 to machine precision while the efficiencies have moved by 2e-2
          to 6e-2.

        Which one actually ran is on :attr:`formulation_assigned` afterwards.
        See ``docs/anisotropic-materials.md`` §8.
        """
        return self._formulation

    @formulation.setter
    def formulation(self, value):
        self._formulation = formulation_of(value)

    @property
    def formulation_assigned(self):
        """The formulation the last solve actually ran on — ``None`` before the first.

        The companion to :attr:`grating_type_assigned`, and exposed for the same reason: a
        silent 8x is indistinguishable from a slow machine, and a benchmark that cannot say
        which formulation it measured is not a benchmark.

        Every solve sets it, including the 1D ones, which are second-order and have no choice
        to make. Leaving those alone would be the worse answer: the attribute would then keep
        whatever a *previous* 2D solve put there, and a stale ``'4ff'`` reads exactly like a
        fresh one.
        """
        return self._formulation_assigned

    def _formulation_for(self, z_row):
        """Settle :attr:`formulation` against the z-row slots this call was handed.

        ``z_row`` is the four ``ep{xz,yz,zx,zy}_conv_all`` arguments. The test is whether a
        slot was *given*, not whether it is nonzero: the branch decides control flow, so it has
        to be answerable on a JAX tracer, and "passed all-zero matrices" is a claim about the
        caller rather than about the tensor. Leave them ``None`` — the default — to stay on the
        cheap path.
        """
        given = [name for name, stack in zip(_Z_ROW_SLOTS, z_row) if stack is not None]
        if self._formulation == Formulation.SECOND and given:
            raise ValueError(
                f"formulation='2ff' cannot express the z row, and {', '.join(given)} "
                f"{'was' if len(given) == 1 else 'were'} given. "
                'The second-order form d2E/dz2 = Omega2 E exists only while the '
                'permittivity z row and column are empty; dropping them would answer for the '
                'reduced tensor, which conserves energy to machine precision while the '
                "efficiencies move by 2e-2 to 6e-2. Use formulation='auto' (the default) or "
                "'4ff'. See docs/anisotropic-materials.md section 6.")
        if self._formulation == Formulation.AUTO:
            return Formulation.FIRST if given else Formulation.SECOND
        return self._formulation

    def _require_field_state(self):
        """Refuse field reconstruction where the last solve left no per-layer state behind.

        Only the 4ff route does: it has no field reconstruction yet, and returns ``None`` in
        place of the ``layer_info_list`` the 2ff solves fill.

        ``None`` rather than ``[]`` because the two states are different questions and both
        deserve their own answer. An *empty* list is a solver that has not solved yet (or a
        stack with no interior), which :func:`~fieldwave._core.fields._require_solved` already
        reports — and ``_base`` initialises the attribute to ``[]``, so that is the state a
        fresh solver is in. ``None`` is the one this method exists for.
        """
        if getattr(self, 'layer_info_list', None) is None:
            raise NotImplementedError(
                'field reconstruction is not implemented on the first-order (4ff) path, '
                'which is the one a populated permittivity z row selects. The efficiencies '
                'and amplitudes it returns are complete; only the in-cell field is missing. '
                'See fieldwave/_core/solve_2d_4ff.py, "What is not here".')

    def solve_1d(self, wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all):
        if self._polarization is not None:
            # Unreachable through conv_solve — `_assign_grating_type` sends a structure with a
            # pair to the conical path, which carries both polarizations — so this only
            # answers a direct call.
            raise ValueError(
                'the pure 1D TE/TM path is single-polarization by construction and cannot '
                'take a polarization pair; use solve_1d_conical, which a full solve '
                '(conv_solve) selects on its own when polarization is set')
        solve = self._solve_fn('1d')
        self._formulation_assigned = Formulation.SECOND
        kx, _ = self.get_kx_ky_vector(wavelength)
        return self._finish(solve(
            self.pol, wavelength, kx, self.n_top, self.n_bot, self.theta, self.thickness,
            epx_conv_all, epy_conv_all, epz_conv_i_all,
            incidence_side=self.incidence_side, full=True))

    def solve_1d_conical(self, wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all):
        solve = self._solve_fn('conical')
        self._formulation_assigned = Formulation.SECOND
        kx, ky = self.get_kx_ky_vector(wavelength)
        return self._finish(solve(
            wavelength, kx, ky, self.n_top, self.n_bot, self.theta, self.psi, self.thickness,
            epx_conv_all, epy_conv_all, epz_conv_i_all,
            incidence_side=self.incidence_side,
            polarization=self._polarization, full=True))

    def solve_2d(self, wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all,
                 epxy_conv_all=None, epyx_conv_all=None,
                 epxz_conv_all=None, epyz_conv_all=None,
                 epzx_conv_all=None, epzy_conv_all=None):
        kx, ky = self._kxy_2d(wavelength)
        z_row = (epxz_conv_all, epyz_conv_all, epzx_conv_all, epzy_conv_all)
        self._formulation_assigned = self._formulation_for(z_row)
        if self._formulation_assigned == Formulation.FIRST:
            return self._finish(_solve_2d_4ff(
                wavelength, kx, ky, self.n_top, self.n_bot, self.theta, self.psi,
                self.thickness, epx_conv_all, epy_conv_all, epz_conv_i_all,
                epxy_conv_all=epxy_conv_all, epyx_conv_all=epyx_conv_all,
                epxz_conv_all=epxz_conv_all, epyz_conv_all=epyz_conv_all,
                epzx_conv_all=epzx_conv_all, epzy_conv_all=epzy_conv_all,
                incidence_side=self.incidence_side, polarization=self._polarization,
                full=True, connecting_algo=self.connecting_algo))
        solve = self._solve_fn('2d')
        return self._finish(solve(
            wavelength, kx, ky, self.n_top, self.n_bot, self.theta, self.psi, self.thickness,
            epx_conv_all, epy_conv_all, epz_conv_i_all,
            epxy_conv_all=epxy_conv_all, epyx_conv_all=epyx_conv_all,
            incidence_side=self.incidence_side,
            polarization=self._polarization, full=True))


# The rule each modeling path actually implements, for the two paths that implement exactly
# one. The raster path (0) is absent because it implements both and so takes the option as
# given; on these two the option is a claim to check rather than a choice to make.
_FACTORIZATION_IMPLEMENTED_BY = {
    1: Factorization.LI,
    2: Factorization.NVF,
}

# Why each of those two is the only rule its path has, phrased for the person who asked for
# the other one. The distinction matters: one is unbuilt, the other is unbuildable.
_FACTORIZATION_REASON = {
    1: ("the block path (modeling_type=1) applies Li's rules to the rectangles its modeler "
        "paints (fieldwave._core.factorization.to_conv_mat_block); no normal field is computed "
        "there, so NVF is not implemented on it. Use a raster ucell (modeling_type=0) or "
        "smooth shapes (modeling_type=2), which both have it"),
    2: ("the smooth shape path (modeling_type=2) builds eps from an analytic normal field -- "
        "the shapes' own (fieldwave._core.analytic.to_conv_mat_smooth_nvf), or, for a painted "
        "layer whose shapes overlap, that of the disjoint polygons the painting resolves into "
        "(fieldwave._core.boundary.to_conv_mat_boundary_layer). There is no raster "
        "axis to run Li's nested 1D rule along, so Li's rules are not something that path has "
        "yet to grow -- they are not something it can express. NVF is a superset of Li anyway "
        "(a constant normal collapses the projector onto Li's rules exactly), so nothing is "
        "lost: drop the keyword, or pass factorization='nvf' to say so"),
}

# The other Fourier axis, exactly the same shape of fact. Both shape paths describe a layer as
# geometry rather than as samples, and integrate the permittivity over that geometry in closed
# form; neither has a raster underneath to run a discrete series along. The raster path (0) is
# again absent because it is the one that implements both.
_COEFFICIENTS_IMPLEMENTED_BY = {
    1: FourierCoefficients.INTEGRATED,
    2: FourierCoefficients.INTEGRATED,
}

# Why, phrased for the person who asked for 'sampled'. Same distinction as above: the block
# path's missing series is unbuilt, the smooth path's is unbuildable.
_COEFFICIENTS_REASON = {
    1: ("the block path (modeling_type=1) hands the factorizer explicit rectangles and their "
        "edges, and integrates eps over them in closed form "
        "(fieldwave._core.factorization.fourier2d_integrated, via to_conv_mat_block). The edges are the "
        "geometry, at whatever real coordinates the modeler put them, so there is no raster "
        "underneath to sample -- rasterising one first is what modeling_type=0 is. Use a "
        "raster ucell, which implements both"),
    2: ("the smooth shape path (modeling_type=2) transforms geometry analytically -- each "
        "shape in turn (fieldwave._core.analytic.shape_ft), or, for a painted layer whose "
        "shapes overlap, the disjoint polygons the painting resolves into "
        "(fieldwave._core.boundary.to_conv_mat_boundary_layer). Either transform *is* the "
        "integral of eps over the cell, evaluated in closed form. Parametric geometry has no "
        "samples to take a discrete "
        "series of, so the sampled coefficients are not a series that path has yet to grow -- "
        "they are not one it can form. The integrated coefficients are the exact ones anyway, "
        "so nothing is lost: drop the keyword, or pass fourier_coefficients='integrated' to "
        "say so"),
}


class CoreFactorizationMixin:
    """Provides ``conv_solve`` — factorization through the core, then the core solve.

    Dispatches the same way the per-backend ``conv_solve`` implementations did (modeling type,
    then the Fourier options) but builds every convolution matrix with
    :mod:`fieldwave._core.factorization`. Two consequences beyond de-duplication:

    * NVF works on every backend — the per-backend builders had it only in NumPy, so JAX and
      PyTorch used to raise here.
    * The eager geometry steps (NVF normal field, the integrated path's block compression) are computed inside
      the builders, which needs concrete values. For JAX-traced use, call the core builders
      directly with a precomputed ``normal_field`` / ``compression``.

    The smooth shape path (``modeling_type == 2``) is built from parametric shapes rather
    than an array, so there is no input array to read the namespace from — it comes from the
    backend code instead.

    Also owns the two Fourier option attributes, so the three backend constructors resolve them
    identically. *Neither* axis is a free choice on every path — see :attr:`factorization` and
    :attr:`fourier_coefficients` — so both are stored as *what was asked for* (``None`` when
    nobody asked) and resolved against the modeling path on read.
    """

    #: Which of the three modeling paths the current ``ucell`` selected; the concrete solver
    #: classes assign it. Defaulted here because both option properties resolve against it, so
    #: the mixin would otherwise only work inside a class that happens to set it first.
    _modeling_type_assigned = None
    #: What was *asked for*, with ``None`` meaning "nobody said". Read through the
    #: :attr:`factorization` property, never directly: only two of the three modeling paths
    #: implement both rules, so the requested value and the effective one are not the same
    #: thing, and telling an explicit request from an untouched default is what decides
    #: whether the mismatch is worth refusing.
    _factorization = None
    #: Likewise, and for the same reason — see :attr:`fourier_coefficients`. Read through the
    #: property.
    _fourier_coefficients = None

    def _set_fourier_options(self, fourier_coefficients, factorization):
        """Settle ``fourier_coefficients`` / ``factorization`` from a constructor's arguments."""
        self._fourier_coefficients, self._factorization = resolve_fourier_options(
            fourier_coefficients, factorization, keep_unset=True)

    @property
    def fourier_coefficients(self):
        """How each layer's Fourier coefficients are obtained: ``'sampled'`` | ``'integrated'``.

        Resolved against the modeling path, exactly like :attr:`factorization` and for the same
        reason. Only the raster path takes both: the two shape paths state a layer as geometry
        and integrate over it in closed form, so the integrated series is the only one either
        can form, and the keyword there is a claim about the build rather than a choice within
        it.

        Raises:
            NotImplementedError: when the series was *explicitly* asked for and this modeling
                path cannot form it. Same exemption for the default, and for the same reason:
                ``'sampled'`` is the module-wide default, so a block- or smooth-path user who
                never touched the keyword is not asking for anything and keeps the closed-form
                series both paths always ran.
        """
        implemented = _COEFFICIENTS_IMPLEMENTED_BY.get(self._modeling_type_assigned)
        if implemented is None:                       # raster, or no ucell yet: free choice
            return (DEFAULT_FOURIER_COEFFICIENTS if self._fourier_coefficients is None
                    else self._fourier_coefficients)
        if self._fourier_coefficients is not None and self._fourier_coefficients != implemented:
            # `!=` rather than `is not`, and `coefficients_of` in the message, for the reason
            # spelled out in `factorization`: this is a str enum, so a raw string written onto
            # the private attribute compares equal and must still format.
            raise NotImplementedError(
                f"fourier_coefficients="
                f"{coefficients_of(self._fourier_coefficients).value!r} is not available "
                f"here: {_COEFFICIENTS_REASON[self._modeling_type_assigned]}.")
        return implemented

    @fourier_coefficients.setter
    def fourier_coefficients(self, value):
        self._fourier_coefficients = coefficients_of(value)

    @property
    def factorization(self):
        """Which rule closes the discontinuous products: ``'li'`` | ``'nvf'``.

        This reports the rule the *current structure* will be built with, which is not always
        the one that was asked for, because only the raster path implements both. The block
        path has only Li's rules and the smooth shape path only NVF, so on those two the
        keyword is a claim about the build rather than a choice within it, and this resolves
        it against the modeling path instead of repeating it back.

        Raises:
            NotImplementedError: when the rule was *explicitly* asked for and this modeling
                path does not implement it. Refusing rather than substituting is the point --
                a smooth-path circle used to answer to ``factorization='li'`` with NVF numbers
                and no way to tell. The default is exempt because it is not a request: it is
                resolved per path, so a user who never touched the keyword gets Li on a raster
                and NVF on smooth shapes, which is what each path always did.
        """
        implemented = _FACTORIZATION_IMPLEMENTED_BY.get(self._modeling_type_assigned)
        if implemented is None:                       # raster, or no ucell yet: free choice
            return DEFAULT_FACTORIZATION if self._factorization is None else self._factorization
        if self._factorization is not None and self._factorization != implemented:
            # `!=`, not `is not`: `Factorization` is a str enum, so a value written straight
            # onto the private attribute compares equal to the member it spells, and the
            # formatting below tolerates one rather than dying inside the error path.
            raise NotImplementedError(
                f"factorization={factorization_of(self._factorization).value!r} is not "
                f"available here: {_FACTORIZATION_REASON[self._modeling_type_assigned]}.")
        return implemented

    @factorization.setter
    def factorization(self, value):
        self._factorization = factorization_of(value)

    def freeze_geometry(self, **kwargs):
        """Take this structure's discrete geometry decisions once, eagerly.

        Captures whatever the active Fourier strategy decides from the concrete ``ucell``.
        Hand the result back to :meth:`conv_solve` as ``geometry=`` to make the call
        traceable. The default ('sampled', 'li') build decides nothing, so its frozen geometry
        is empty — passing it is still what keeps ``conv_solve`` off the eager path.

        The record describes *this* structure. Replaying it onto one whose geometry has moved
        is refused rather than answered, eagerly — see :mod:`fieldwave._core.geometry` for what
        each record checks and why a traced replay cannot.
        """
        [setattr(self, k, v) for k, v in kwargs.items()]
        # Same two refusals as `_conv_matrices`, in the same order, for the same reason: a
        # record of a build that cannot happen is worth nothing, and a structure that gets one
        # diagnosis from `freeze_geometry` and a different one from `conv_solve` sends the
        # reader to the wrong place depending on which they called.
        self._reject_unsupported_lattice()
        factorization = self.factorization
        coefficients = self.fourier_coefficients
        fto_x, fto_y = self.fto[0], self.fto[1]

        # The two axes decide independently what has to be frozen: NVF needs the normal
        # field, the integrated series needs the block compression, and ('integrated', 'nvf')
        # -- the combination the old integer could not spell -- needs both.
        normal_field = normal_field_source = compression = None
        if self._modeling_type_assigned == 0:
            if factorization == Factorization.NVF:
                basis = self._geometry_lattice_for(None)
                normal_field = _fac.normal_field_for_ucell(
                    self.ucell, fto_x, fto_y, lattice=basis)
                normal_field_source = _fac.normal_field_source_for_ucell(
                    self.ucell, fto_x, fto_y, lattice=basis)
            if coefficients == FourierCoefficients.INTEGRATED:
                compression = [_fac.compression_indices(self.ucell[i])
                               for i in range(self.ucell.shape[0])]

        geometry = StaticGeometry(normal_field=normal_field, compression=compression,
                                  normal_field_source=normal_field_source)
        if self._modeling_type_assigned == 1:
            # The block modeler records its plans while it builds, so run the build once to
            # get them -- the *modeler*, not the whole factorization. Freezing used to go
            # through `_conv_matrices` and throw the convolution matrices away, which it paid
            # for only because the same-material detector read them; that detector is gone.
            # Dropping them takes a freeze from 25.6 ms to 10.4 ms, which is what makes
            # re-freezing every step of a fit cheaper than nursing one along.
            self.modeling_block_instruction(self.ucell)
            geometry.block_grid = self.modeler.grid_plans
            geometry.shape_choices = self.modeler.shape_choices
        if self._modeling_type_assigned == 2:
            # Only a painted layer that actually overlaps has a plan to freeze. A summed one
            # has no discrete decision in it, and a painted one with nothing overlapping means
            # the same structure either way -- both stay on the closed-form path, so their
            # slots stay None and the replay never looks.
            plans = [_boundary.plan_layer(layer.get('shapes', []), self.period)
                     if _analytic.layer_needs_resolving(
                         layer, self.period, self._geometry_lattice_for(None)) else None
                     for layer in self.ucell]
            # Recorded whether or not anything needed a plan. The entry *is* the decision:
            # `None` here means "looked at while the numbers were concrete, and it does not
            # overlap". Setting the attribute only when something overlapped made those two
            # cases indistinguishable downstream, so a disjoint painted layer under a trace
            # had to re-take the overlap test it had already passed -- on values that can no
            # longer answer it.
            geometry.boundary_plans = plans
        return geometry

    def conv_solve(self, geometry=None, **kwargs):
        """Build the convolution matrices, then solve.

        ``geometry`` is an optional :class:`~fieldwave._core.geometry.StaticGeometry` — the
        discrete decisions frozen by :meth:`freeze_geometry`. Supply it to keep the whole call
        traceable under ``jax.jit`` / ``jax.grad``; omit it and each decision is auto-detected
        eagerly, which needs concrete values.

        Raises:
            StaleGeometryError: eagerly, when ``geometry`` no longer describes the structure it
                is being replayed onto. A traced call cannot check — the comparison is the
                thing the record exists to avoid — so a long fit should re-freeze on a schedule
                rather than wait to be told.
        """
        [setattr(self, k, v) for k, v in kwargs.items()]  # needed for optimization sweeps
        self._frozen_geometry = geometry
        return self.solve_for_conv(self.wavelength, *self._conv_matrices(geometry))

    def conv_matrices(self, geometry=None, **kwargs):
        """The convolution matrices for this structure, without solving.

        Returns ``(epx, epy, epz_i, epxy, epyx)`` — exactly the tuple :meth:`solve_for_conv`
        takes after the wavelength, so ``solver.solve_for_conv(wl, *solver.conv_matrices())``
        is ``conv_solve()``. Each is ``(n_layers, ff_xy, ff_xy)`` with ``ff_xy =
        (2*fto_x+1) * (2*fto_y+1)``; ``epxy`` and ``epyx`` are ``None`` on the paths that
        produce three rather than five (see :func:`~._core.factorization.nvf_conv_matrices`).

        **This exists because the boundary was one-way.** ``solve_for_conv`` has always accepted
        these matrices, so a caller could put them *in*; nothing returned them, so anyone who
        wanted to look at what a modeling path actually built had to reach into ``_core`` and
        reassemble it. That is not hypothetical: three separate attempts to hand-rebuild this
        quantity against internals produced three wrong numbers in one day, because the
        reassembly has to reproduce the doubled reciprocal grid's layout and the order-index
        sign convention, and both fail silently when guessed.

        It is the right object to compare two modeling paths with, for the same reason it is the
        one the solve consumes: every path produces it, so a comparison is between the same
        quantity rather than between two reconstructions of it. Note what it *includes*, though
        — on an NVF path these carry the normal-field projector as well as the geometry, so a
        difference between two paths is not attributable to their geometry alone. Isolating the
        geometry means comparing the permittivity coefficients instead, which is a narrower
        question and is left to the caller that has it.

        ``geometry`` behaves as in :meth:`conv_solve`.
        """
        [setattr(self, k, v) for k, v in kwargs.items()]
        return self._conv_matrices(geometry)

    def _reject_unsupported_lattice(self):
        """Refuse the build paths a skewed cell has no implementation for — before they answer.

        Every one of these *runs* on a non-orthogonal cell and returns a plausible, wrong
        number: the raster is still a raster, the transforms are still transforms, and the NVF
        assembly stays Hermitian so energy still conserves to 1e-15. Nothing downstream is in a
        position to notice, which is why the refusal has to live here.

        What is left is specific, not general. ``factorization='li'`` applies the inverse rule
        along the *raster axes* (see
        :func:`~fieldwave._core.factorization.to_conv_mat_raster_sampled`), and the solve
        consumes the result as Cartesian ``eps_xx`` / ``eps_yy``. On a skewed cell those axes
        are ``a1`` / ``a2``, not ``x`` / ``y``. Repairing that is Li's covariant formulation
        (JOSA A 14(10), 2758) — a separate piece of work, and one worth costing before doing,
        since NVF already converges faster than Li on the curved boundaries this feature exists
        for.

        The block modeler (``modeling_type=1``) rasterises shapes onto a rectangular grid of
        its own and is not ported.

        Two things that were refused here are not any more, both because the assumption behind
        the refusal turned out to be wrong rather than because they were implemented:

        * The **integrated** coefficients are computed by ``fourier2d_integrated`` as an integral over the
          *parametric* cell, where the Jacobian cancels against the ``1/area`` normalisation.
          There is no metric in it, so it was always lattice-general.
        * The **smooth shape** path (``modeling_type=2``) states shape lengths as absolute
          Cartesian quantities, so nothing about a shape deforms under a skew — a rectangle
          stays a rectangle. Only the grid of frequencies its transform is sampled at moves,
          which is one expression in :func:`~fieldwave._core.analytic.shape_ft`. Overlapping
          *painted* layers were refused inside that path for a while, on the belief that
          resolving an overlap clips against a rectangular cell. It does not: the clipper does
          Cartesian polygon booleans and never sees the cell, and ``exp(-i G.r)`` carries the
          periodicity for a polygon straddling an edge. What actually needed the basis was the
          periodic-image enumeration in
          :func:`~fieldwave._core.overlap.overlapping_pairs`, which decides whether a layer has
          an overlap at all. See ``test_a_painted_overlap_survives_a_unimodular_shear``.

        The Li check below reads the *resolved* :attr:`factorization`, which is why a skewed
        run on the smooth shape path no longer has to spell ``factorization='nvf'`` out: that
        path resolves to NVF on its own, and demanding the keyword was this refusal reading the
        unresolved default and mistaking it for a request.
        """
        if self.lattice_is_axis_aligned:
            return
        common = ('The cell basis is non-orthogonal (set via .lattice). ')
        if self._modeling_type_assigned == 1:
            raise NotImplementedError(
                common + 'the block modeler (modeling_type=1) rasterises onto a rectangular '
                'grid, which is not implemented yet for a skewed cell. Use a raster ucell '
                '(modeling_type=0) or smooth shapes (modeling_type=2).')
        factorization = self.factorization
        if factorization != Factorization.NVF:
            raise NotImplementedError(
                common + f"factorization={factorization.value!r} applies the inverse "
                f'rule along the raster axes, which are not x and y once the cell is skewed, '
                f"so it would answer plausibly and wrongly. Use factorization='nvf'.")

    def _conv_matrices(self, geometry=None):
        """Convolution matrices for the current structure: ``(epx, epy, epz_i, epxy, epyx)``."""
        # Lattice first, then the rule -- the order is load-bearing. A skewed cell on the block
        # path is unbuildable whatever rule is named, so reporting the *keyword* there would
        # send the user to change the one thing that cannot help. The other way round is safe:
        # `_reject_unsupported_lattice` reads the resolved `factorization` itself, so a skewed
        # cell asking for a rule its path cannot express still gets told about the rule.
        self._reject_unsupported_lattice()
        # Then read both, unconditionally: this is the one call every build goes through, so it
        # is where an option the chosen path cannot express has to be refused rather than
        # quietly replaced by the one that path does implement. The factorization is read first
        # so that a smooth-path call naming *both* unavailable values is told about the rule --
        # the axis a user is far likelier to have meant something by.
        factorization = self.factorization
        coefficients = self.fourier_coefficients
        fto_x, fto_y = self.fto[0], self.fto[1]
        epxy_conv_all = epyx_conv_all = None

        if self._modeling_type_assigned == 0:                      # raster
            # The two axes are dispatched independently. Splitting on the factorization
            # first is not arbitrary: Li's rules return three matrices and NVF returns five
            # (it couples x and y, so epxy / epyx stop being zero).
            sampled = coefficients == FourierCoefficients.SAMPLED
            normal_field = None if geometry is None else geometry.normal_field
            normal_field_source = None if geometry is None else geometry.normal_field_source
            compression = None if geometry is None else geometry.compression

            if factorization == Factorization.LI:
                if sampled:
                    epx_conv_all, epy_conv_all, epz_conv_i_all = \
                        _fac.to_conv_mat_raster_sampled(
                            self.ucell, fto_x, fto_y)
                else:
                    epx_conv_all, epy_conv_all, epz_conv_i_all = \
                        _fac.to_conv_mat_raster_integrated(
                            self.ucell, fto_x, fto_y, compression=compression)
            else:
                if sampled:
                    (epx_conv_all, epy_conv_all, epxy_conv_all, epyx_conv_all,
                     epz_conv_i_all) = _fac.to_conv_mat_raster_nvf(
                        self.ucell, fto_x, fto_y,
                        normal_field=normal_field,
                        normal_field_source=normal_field_source,
                        lattice=self._geometry_lattice_for(geometry))
                else:
                    (epx_conv_all, epy_conv_all, epxy_conv_all, epyx_conv_all,
                     epz_conv_i_all) = _fac.to_conv_mat_raster_integrated_nvf(
                        self.ucell, fto_x, fto_y,
                        normal_field=normal_field,
                        normal_field_source=normal_field_source,
                        compression=compression,
                        lattice=self._geometry_lattice_for(geometry))

        elif self._modeling_type_assigned == 1:      # block -- Li on integrated coefficients
            # No branch on either option here, and that is not an oversight: this path has one
            # rule and one series (see _FACTORIZATION_IMPLEMENTED_BY and
            # _COEFFICIENTS_IMPLEMENTED_BY) -- `to_conv_mat_block` integrates eps over the
            # painted rectangles with `fourier2d_integrated`, unconditionally. Asking it for the other of
            # either was refused above rather than answered with this.
            ucell_block = self.modeling_block_instruction(
                self.ucell, plans=None if geometry is None else geometry.block_grid,
                shape_choices=None if geometry is None else geometry.shape_choices)
            epx_conv_all, epy_conv_all, epz_conv_i_all = _fac.to_conv_mat_block(
                ucell_block, fto_x, fto_y)

        elif self._modeling_type_assigned == 2:      # smooth shapes -- NVF on exact transforms
            # Likewise fixed on both axes: the epsilon tensor comes from the shapes' analytic
            # normal field, so five matrices always, and each shape's transform is the exact
            # integral over the cell, so the coefficients are integrated ones. Naming
            # `factorization='li'` or `fourier_coefficients='sampled'` was refused above.
            xp = _namespace_for_backend(self.backend)
            (epx_conv_all, epy_conv_all, epxy_conv_all, epyx_conv_all,
             epz_conv_i_all) = _analytic.to_conv_mat_smooth_layers(
                self.ucell, fto_x, fto_y, self.period,
                normal_grid=self.normal_grid,
                type_complex=xp.complex128, xp=xp,
                boundary_plans=None if geometry is None else geometry.boundary_plans,
                lattice=self._geometry_lattice_for(geometry))

        else:
            raise ValueError("Check 'modeling_type' in 'conv_solve'.")

        return epx_conv_all, epy_conv_all, epz_conv_i_all, epxy_conv_all, epyx_conv_all


def _namespace_for_backend(backend):
    """Array namespace for a ``Backend`` code.

    The analytic builder starts from parametric shapes, not arrays, so unlike every other
    entry point it cannot infer the namespace from its input.
    """
    from .backend import array_namespace
    if backend == 1:
        import jax.numpy as jnp
        return array_namespace(jnp.zeros(1))
    if backend == 2:
        import torch
        return array_namespace(torch.zeros(1))
    import numpy as np
    return array_namespace(np.zeros(1))
