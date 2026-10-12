"""The face types, and the six-face declaration every scene must make.

**Boundaries are required per face.** There is no default: a box with an unnamed face is a
box whose physics nobody chose, and the run would answer a question the user did not ask.
``Boundaries(x=..., y=..., z=...)`` takes, per axis, one face type for both faces or a
``(low, high)`` pair. Faces are named ``'x-'``, ``'x+'``, ``'y-'``, ``'y+'``, ``'z-'``,
``'z+'`` and never ordered.

Stage 0 knows five face types. What each one *is* on the Yee grid -- the neighbour rule and
the mask it lowers to, and where the wall physically sits -- is decided in
``src/wavebit/fd/_kernel/faces.py``; this module only says what a user may declare:

``Periodic()``
    Both faces of the axis wrap. Must be given to both faces (one object for the axis).
``Bloch(phase=phi)`` or ``Bloch(k=k)``
    Both faces wrap with the phase ``phi`` radians across the period: the field at ``x + L``
    is ``exp(+i*phi)`` times the field at ``x``, in the ``e^{-i w t}`` time convention this
    project uses everywhere (a wave travelling toward ``+x`` is ``exp(+i k x)``). ``k`` is the
    wavevector component along the axis, so ``phi = k * L`` with ``L`` the snapped box size.
    A Bloch face makes the whole run complex, doubling every byte and every flop, so a phase
    that is concretely ``0`` is refused in favour of ``Periodic()`` rather than lowered to it
    silently -- a user who wrote ``Bloch`` and got a real run would rightly ask why. A
    ``Bloch`` phase on an axis one cell long is refused (stage 3), naming ``Axis1D``.
``Axis1D(k=k, k_max=...)``
    (Stage 3.) A one-cell axis whose derivative is ``i k`` exactly -- the invariant axis of a
    2-D (conical) problem, without the one-cell difference's artificial dispersion. Complex.
``PEC()``
    Perfect electric conductor: tangential E vanishes on the wall.
``PMC()``
    Perfect magnetic conductor: tangential H vanishes on the wall.
``PML(cells=12, wavelength=...)``
    An absorbing face: a PEC wall with a complex-frequency-shifted stretched layer in front
    of it. See the class for what it is in the kernel, why ``wavelength`` is required and
    where the 12 comes from.
``Absorber(cells=12)``
    The fallback where a PML cannot go: a graded matched conductivity instead of a stretch.
``Symmetry(parity=+1)`` / ``Symmetry(parity=-1)``
    A mirror plane on that face; the box is one half of a symmetric problem. ``parity`` is
    the eigenvalue of the field under the mirror, stated for the **tangential E**: ``+1``
    means tangential E is even across the plane, which makes the plane PMC-like (tangential
    H odd, hence zero, on it); ``-1`` means tangential E is odd, PEC-like. Allowed on at most
    one face of an axis, with ``PEC`` or ``PMC`` on the other. Where the plane sits relative
    to the face is fixed by the Yee layout (a node plane on a low face, a half-cell plane on
    a high face) and recorded per face by the plan; ``faces.py`` derives it and the
    symmetry gate is the proof.

Presets: ``Boundaries.periodic(z=...)`` for a periodic cell with the given z faces, and
``Boundaries.open(fd.PML(cells=12, wavelength=0.55))`` for absorbing faces all round, with
per-axis overrides (``Boundaries.open(pml, z=fd.Periodic())``).
"""
import math
import numbers

from ._spec import is_jax_typed, spec, static_float, static_int, unknown_keyword_message

__all__ = ['Boundaries', 'Periodic', 'Bloch', 'Axis1D', 'PEC', 'PMC', 'Symmetry', 'PML',
           'Absorber', 'Absorbing', 'FACES', 'AXES', 'PML_CELLS', 'PML_REFLECTION']

#: Default PML depth in cells, and the line below which a ``PML`` is refused without
#: ``allow_thin=True``. Re-measured for this kernel by the doubling study that *is*
#: ``tests/test_fd_pml.py`` -- its module docstring carries the table the refusal cites.
PML_CELLS = 12

#: Round-trip amplitude reflection the conductivity grading is aimed at (the legacy
#: ``PML_TARGET_REFLECTION``). It is the nominal normal-incidence figure of the *continuous*
#: profile, not a bound on the discrete reflection, and least of all at grazing incidence.
PML_REFLECTION = 1e-8

#: The amplitude reflection of this kernel's CPML, re-measured for stage 1 by the doubling
#: study that *is* ``tests/test_fd_pml.py`` -- read its module docstring for the method and
#: the caveats. Rows are ``cells``, columns the nominal angle of incidence. Quoted verbatim
#: by :data:`THIN_PML_MESSAGE`, and asserted against a live re-measurement by that gate, so
#: the number a user is refused with is a number this kernel actually produced.
PML_REFLECTION_TABLE = """\
    cells      0 deg     30 deg     60 deg     80 deg
        6    2.6e-02    2.1e-02    1.0e-02    1.9e-02
        8    4.6e-03    3.3e-03    8.8e-04    2.6e-02
       12    9.2e-05    4.7e-05    1.1e-04    3.4e-02
       16    1.1e-06    5.9e-07    1.3e-04    3.9e-02
       24    2.2e-08    1.4e-07    1.2e-04    4.4e-02
   the target's own law, reflection**cos(theta):
             1.0e-08    1.2e-07    1.0e-04    4.1e-02"""

#: The refusal below :data:`PML_CELLS`, quoting the table its own gate measured. Kept here
#: rather than inline so that the test that re-measures the table can assert the message
#: still carries what it found (``tests/test_fd_pml.py``).
THIN_PML_MESSAGE = (
    'PML(cells={cells}) is thinner than the 12 cells this kernel was measured at. The '
    'doubling study in tests/test_fd_pml.py -- a Gaussian packet launched at a z face, the '
    'same packet in a longer box as the reference, the amplitude ratio at the carrier '
    'frequency at a probe three cells in front of the layer -- measures, at the default '
    'reflection=1e-8:\n'
    + PML_REFLECTION_TABLE + '\n'
    'Below 12 the number is the grading\'s own discretization and it rises about a decade '
    'every two cells; at 12 and above the layer already delivers its design target wherever '
    'that target dominates, and the last row is why. So more cells buy accuracy near normal '
    'incidence, and at grazing a lower reflection= is what buys it -- not more cells. Pass '
    'allow_thin=True to take fewer anyway (the scene runs, it just reflects more), or raise '
    'cells. The absorber is dx-independent, so cells and not a thickness is the knob.')

#: The three axes in the order the public API spells a triple, and the six face names.
AXES = ('x', 'y', 'z')
FACES = ('x-', 'x+', 'y-', 'y+', 'z-', 'z+')


class FaceType:
    """Base of the five face types; carries nothing, exists so ``isinstance`` has a name."""

    __slots__ = ()


@spec
class Periodic(FaceType):
    """Both faces of the axis wrap; real dtype."""


def _concrete_scalar(what, v, where='Bloch'):
    """``v`` as a float when its value is readable now; ``None`` for a JAX tracer.

    The phase is a traced leaf, so a tracer must pass; but a tracer is the *only* thing that
    passes without being read. Everything else ``float()`` cannot read -- a string, a list,
    a complex number, an array of two phases -- is refused here by name, not left for the
    lowering to trip over with a NumPy message that names neither ``Bloch`` nor ``phase``.
    """
    if isinstance(v, (bool, complex, str)) or v is None:
        raise ValueError(f'{where}({what}={v!r}): a real scalar (or a traced scalar of jax '
                         f'or torch); got {type(v).__name__}.')
    shape = getattr(v, 'shape', None)
    if shape is not None and tuple(shape) != ():
        raise ValueError(f'{where}({what}=...): one real scalar per axis; got shape '
                         f'{tuple(shape)}. A phase per axis is one Bloch(...) per axis.')
    detach = getattr(v, 'detach', None)
    if detach is not None:
        v = detach()
    try:
        return float(v)
    except Exception as err:
        if is_jax_typed(v):
            return None
        raise ValueError(f'{where}({what}={v!r}): a real scalar (or a traced scalar of jax '
                         f'or torch); got {type(v).__name__}.') from err


@spec
class Bloch(FaceType):
    """Both faces wrap with a phase; complex dtype. Exactly one of ``phase`` or ``k``.

    ``phase`` is in radians across the period; ``k`` is the wavevector component along the
    axis and is turned into a phase with the snapped box size when the scene is lowered.
    Either may be a traced scalar (a ``jax`` or ``torch`` value); it is a leaf of the traced
    parameters, not part of the static plan, so differentiating through it is allowed.
    """

    phase: object = None
    k: object = None

    def __post_init__(self):
        if (self.phase is None) == (self.k is None):
            raise ValueError('Bloch takes exactly one of phase= (radians across the period) '
                             'or k= (the wavevector component along the axis).')
        what = 'phase' if self.k is None else 'k'
        value = _concrete_scalar(what, self.phase if self.k is None else self.k)
        if value is not None and value == 0.0:
            raise ValueError(
                f'Bloch({what}=0) is Periodic(): the field repeats with no phase. Say '
                f'Periodic() -- a Bloch face makes the whole run complex, which doubles every '
                f'byte and every flop, and a phase of zero buys nothing for that price.')


    @staticmethod
    def from_source(source, background=None, axis1d=()):
        """The transverse faces a :class:`~wavebit.fd.PlaneWave` needs, as ``{axis: face}``.

        ``fd.Boundaries(**fd.Bloch.from_source(pw), z=fd.PML(...))`` for a wave along z. The
        transverse wavevector is taken at the band centre in the medium the sheet sits in,
        ``k_par = n (2 pi / wavelength) sin(theta)`` split by ``phi`` over the two transverse
        axes, with ``n = sqrt(eps mu)`` of ``background`` (a material, a bare permittivity,
        or ``None`` for vacuum -- pass the scene's ``background`` when it is not vacuum). An
        axis whose component is zero gets ``Periodic()``, because a zero Bloch phase is
        refused; a concrete ``theta = 0`` therefore returns ``Periodic`` on both.

        The phase is fixed-k: every frequency in the band shares this ``k_par``, so the angle
        of incidence varies across the band as ``sin(theta(f)) = k_par / (2 pi n f)``, which is
        what a Bloch-periodic time-domain run can carry. The band centre must be concrete.

        ``axis1d`` (stage 3) names transverse axes -- ``'y'`` or ``('x', 'y')`` -- that get
        :class:`Axis1D` with that ``k`` (and ``k_max = |k|``) instead of ``Bloch``/``Periodic``:
        the one-cell invariant axis of a conical problem, whose domain is one cell there.
        """
        import math
        from .materials import as_material
        from .sources import PlaneWave
        if not isinstance(source, PlaneWave):
            raise TypeError(f'Bloch.from_source({type(source).__name__}): the phases come '
                            f'from an fd.PlaneWave; a Dipole or a TFSF box needs none.')
        wavelength = source.band.concrete_wavelength
        if wavelength is None:
            raise ValueError('Bloch.from_source(pw): the band centre is traced, and a Bloch '
                             'phase is fixed by the plan; give the band a concrete '
                             'wavelength, or write Bloch(k=...) yourself.')
        n = as_material(background, 'Bloch.from_source(background=)').index_bound()
        if n is None:
            raise ValueError('Bloch.from_source(background=<traced>): the index must be '
                             'concrete to fix a phase.')
        names = (axis1d,) if isinstance(axis1d, str) else tuple(axis1d)
        for name in names:
            if name not in AXES or AXES.index(name) == source.axis:
                raise ValueError(f'Bloch.from_source(axis1d={axis1d!r}): name transverse axes of '
                                 f'the wave, from {[a for a in AXES if AXES.index(a) != source.axis]}.')
        k, _, _ = source.vectors()
        k0 = n * 2.0 * math.pi / wavelength
        out = {}
        for a, axis in enumerate(AXES):
            if a == source.axis:
                continue
            ka = k0 * k[a] if abs(k[a]) >= 1e-12 else 0.0
            if axis in names:
                out[axis] = Axis1D(k=ka)
            else:
                out[axis] = Periodic() if ka == 0.0 else Bloch(k=ka)
        return out

@spec
class Axis1D(FaceType):
    """A one-cell axis whose derivative is ``i k`` exactly: the invariant axis of a 2-D problem.

    ``fd.Boundaries(x=fd.Bloch(k=kx), y=fd.Axis1D(k=ky, k_max=...), z=fd.PML(...))`` is a
    conical 1-D grating: the structure is uniform along y, the field is ``F(x, z) exp(i ky
    y)``, and the grid holds one cell of it along y. **Stage 3.**

    **Why it is not a one-cell Bloch face.** A ``Bloch(k=k)`` axis differences a field
    ``exp(i k y)`` exactly as the Yee grid does on any number of cells: ``(exp(i k dx) - 1)/dx``,
    whose modulus ``2 sin(k dx/2)/dx`` is the *discrete* wavenumber. On a multi-cell axis that
    is the grid's own dispersion and every other difference shares it; on an axis the
    structure never varies along, it is an artificial dispersion the problem does not have --
    the wave sees ``K_y = 2 sin(k dx/2)/dx`` instead of ``k``, a relative error of ``(k dx)^2 /
    24``, and the conical grating it models is a slightly different grating. ``Axis1D``
    replaces that difference by the exact derivative: ``D_y F = i k dx F`` in the kernel's
    units (the curl carries no ``1/dx``; ``src/wavebit/fd/_kernel/curl.py``), on both the forward
    and the backward difference, so every component's y dependence is ``exp(i k y)`` with no
    stagger at all. So a user ``Bloch`` phase on a one-cell axis is refused, naming this
    (the stage-2 PlaneWave and TFSF *columns* are one cell across and Bloch by construction
    -- they must reproduce the main grid's multi-cell difference exactly -- and are not user
    faces; they are untouched).

    ``k``
        The wavevector along the axis, a real scalar, concrete or traced (a gradient may flow
        through it). ``k = 0`` is allowed: a 2-D problem with no dependence along the axis.
    ``k_max``
        The largest ``|k|`` the plan is made for (static). The derivative is ``i k`` and not a
        bounded difference, so the CFL bound depends on it: the sum ``sum_a max K_a^2 / 4``
        that the stable ``dt`` is ``n_min / sqrt(...)`` of has ``1/dx^2`` for a differenced
        axis and ``k_max^2 / 4`` for this one (``src/wavebit/fd/_lower/checks.py::cfl_bound``).
        Default ``|k|`` when ``k`` is concrete; required when it is traced. A concrete ``|k| >
        k_max`` is refused here; a traced one is checked at run time and makes the run
        ``nan`` with ``res.diagnostics['stable'] = False``.
    ``algebraic``
        Must be ``True``: the derivative is the algebraic ``i k``. There is no other kind --
        the one-cell difference is exactly what is being replaced -- so ``False`` is refused
        rather than quietly meaning ``Bloch``.

    A run with an ``Axis1D`` face is always complex (``k = 0`` included: one dtype per plan,
    and a traced ``k`` could become nonzero). The domain must be one cell along the axis
    (``Domain(size=...)`` equal to ``dx`` there); sources whose support spans cells along it
    (a ``Dipole``'s stamp, a ``TFSF`` box) are refused, and a ``Diffraction`` monitor has the
    single order ``0`` along it.
    """

    k: object = 0.0
    k_max: object = None
    algebraic: bool = True

    def __post_init__(self):
        if self.algebraic is not True:
            raise ValueError(
                f'Axis1D(algebraic={self.algebraic!r}): the derivative along a one-cell axis is '
                f'the algebraic i k and nothing else; the alternative -- the one-cell Bloch '
                f'difference (exp(i k dx) - 1)/dx -- is the artificial dispersion Axis1D exists '
                f'to remove. Leave algebraic=True.')
        if self.k is None:
            raise ValueError('Axis1D(k=None): k is the real wavevector along the axis (0 for '
                             'a field uniform along it).')
        value = _concrete_scalar('k', self.k, 'Axis1D')
        if self.k_max is None:
            if value is None:
                raise ValueError(
                    'Axis1D(k=<traced>): a traced k needs k_max=..., the largest |k| the plan '
                    'is made for. The CFL bound depends on it (the derivative is i k, not a '
                    'bounded difference) and the plan is static.')
            object.__setattr__(self, 'k_max', abs(value))
        else:
            k_max = static_float('Axis1D', 'k_max', self.k_max)
            if not (k_max >= 0.0 and k_max == k_max and k_max != float('inf')):
                raise ValueError(f'Axis1D(k_max={self.k_max!r}): a non-negative finite '
                                 f'wavevector.')
            object.__setattr__(self, 'k_max', k_max)
        if value is not None and abs(value) > self.k_max * (1.0 + 1e-12):
            raise ValueError(
                f'Axis1D(k={value!r}, k_max={self.k_max!r}): |k| is above k_max, and the '
                f'plan\'s CFL bound is made for k_max. Raise k_max to at least {abs(value)!r}.')


@spec
class PEC(FaceType):
    """Perfect electric conductor: tangential E vanishes on the wall."""


@spec
class PMC(FaceType):
    """Perfect magnetic conductor: tangential H vanishes on the wall."""


@spec
class Symmetry(FaceType):
    """A mirror plane on this face. ``parity`` is the tangential-E eigenvalue: +1 or -1."""

    parity: int

    def __post_init__(self):
        # Normalised to a Python int at the door: the parity is a value in the static plan
        # and its skeleton, so ``1``, ``1.0`` and ``numpy.int64(1)`` must be the one mirror
        # (the skeleton encodes the type of a value, and a float ``1.0`` would hash apart).
        # A bool is not one: ``True`` is not a mirror eigenvalue anyone meant.
        p = self.parity
        if isinstance(p, float) and p in (1.0, -1.0):
            p = int(p)
        if isinstance(p, bool) or not isinstance(p, numbers.Integral) or int(p) not in (1, -1):
            raise ValueError(
                f'Symmetry(parity={self.parity!r}): parity is +1 (tangential E even across '
                f'the plane, PMC-like) or -1 (tangential E odd, PEC-like); nothing else is a '
                f'mirror eigenvalue.')
        object.__setattr__(self, 'parity', int(p))


class Absorbing(FaceType):
    """Base of the two absorbing faces, :class:`PML` and :class:`Absorber`.

    Both are, in the kernel, a **PEC wall with something in front of it**: the face's
    neighbour rule is the zero pad and the wall mask that ``PEC()`` lowers to, and the
    absorption is a slab of ``cells`` cells inside the face. Neither is a face *rule* -- the
    kernel has four of those (``zero``, ``periodic``, ``bloch``, ``mirror``) and stage 1 adds
    none. What they differ in is what the slab does:

    ``PML``       stretches the derivative along the axis (four psi arrays per face, carried
                  in the state), matched to a background it must be told about.
    ``Absorber``  grades a matched electric and magnetic conductivity into the update
                  coefficients. No memory, no kernel code, and no matching condition -- which
                  is why it is the fallback where a stretch cannot be matched.

    An absorbing face is allowed on either or both faces of an axis, and mixes freely with
    the other four types on the other axes.
    """

    __slots__ = ()


@spec
class PML(Absorbing):
    """A complex-frequency-shifted perfectly matched layer (Roden & Gedney 2000).

    ``fd.PML(cells=12, wavelength=0.55)``. Inside the ``cells`` cells behind this face the
    derivative along the face's axis is stretched,

        psi <- b * psi + a * d           (d is the bare difference, in the slab)
        d   <- d / kappa + psi

    with ``b``, ``a`` and ``kappa`` per-cell profiles of the depth into the layer, evaluated
    at the E points and the H points separately. The formulas and their unit audit are
    ported from ``src/wavebit/_core/fdtd.py::cpml_coefficients`` (a kernel seed -- ported, never
    imported); ``src/wavebit/fd/_lower/cpml.py`` carries them and the audit.

    ``cells``
        The depth, **in cells and not in a length**: a correctly graded CPML is
        ``dx``-independent, which the legacy engine measured (a 1-D probe in glass at a fixed
        20-cell depth reflects 2.06e-6 / 1.63e-6 / 1.42e-6 / 1.37e-6 as ``dz`` runs 20 / 10 /
        5 / 2.5 nm). Fewer than 12 is refused unless ``allow_thin=True``; the refusal quotes
        the table ``tests/test_fd_pml.py`` measured on *this* kernel.
    ``wavelength``
        **Required, with no default.** It sets ``alpha_max``, the CFS pole, which is a
        frequency and is therefore meaningless without one to be relative to. Stage 1 has no
        source to infer it from, so it comes from here or from
        ``Resolution(wavelength=...)``; a PML with neither is refused by name.
    ``background``
        The medium the stretch is matched to: a material, a bare permittivity, or ``None``
        for the scene's own ``background=``. A structure that reaches into this face must be
        uniform along the axis there and must equal this, or the scene is refused naming
        ``Absorber``.
    ``reflection``
        The nominal normal-incidence round-trip amplitude the conductivity grading is aimed
        at (default 1e-8). It is a grading parameter, not a bound: the discrete reflection is
        set by the grading *and* the cell count *and* the angle, which is what the doubling
        table measures.
    ``allow_thin``
        Take fewer than 12 cells anyway. The scene still runs; it just reflects more.
    ``stabilize``
        A stabilizing loss ``rho`` (static, ``>= 0``): a matched loss ``rho * sigma(depth)`` on
        the field component normal to this face. ``None`` (the default) and ``0`` are the pure
        stretch; ``3e-3`` (``wavebit.fd._lower.cpml.STABILIZING_LOSS``) is the measured value
        that suffices at the default ``reflection=1e-8`` (and stronger targets). A weaker target
        needs more -- 5e-3 at 1e-6, 1e-2 at 1e-4, 2e-2 at 1e-3
        (``wavebit.fd._lower.cpml.MEASURED_STABILIZE``) -- and the plan warns
        (``PMLStabilityWarning``) when a face asks for less than its target's value, or sets a
        target weaker than any measured. **Why it exists**: a stretch backed by a PEC wall is an *active*
        termination for evanescent waves, so a lossless structure with a bound mode just past
        the light line -- a grating's guided resonance at a Rayleigh wavelength, a slab under
        an ``Axis1D`` column -- grows exponentially under the pure stretch, whatever its
        constants; with enough loss it decays on every slab family it was measured on. That is
        a measured balance, not a guarantee: the loss does not make the layer passive, so a
        structure unlike the sampled ones can still grow. **What it costs**: nothing at normal
        incidence and, at an angle, a reflection that replaces the target's
        ``reflection ** cos(theta)`` law (12 cells: 2.1e-4 at 30 degrees against 4.7e-5, 3.7e-3
        at 60 against 1.1e-4). That is why it is opt-in: on by default it broke the Airy,
        closed-form PlaneWave, RCWA-stack and grazing-rule gates at 20-40 degrees.
        ``src/wavebit/fd/_lower/cpml.py::stabilizing_rate`` carries the mechanism and the numbers.
        The grazing rule judges a stabilized face by its exact return on every propagating
        order (``src/wavebit/fd/_lower/layer_return.py``) and refuses an order it returns above
        both 1e-3 and what its pure stretch returns: at 3e-3 in vacuum at 20 cells per
        wavelength, below exit ``cos theta`` 0.660 (TE) / 0.657 (TM) on 12 cells and 0.472 /
        0.475 on 24. At normal incidence the loss is inert and the face is judged as its pure
        stretch. What a refusal offers is judged on every order, medium and wavelength of the
        faces it replaces.
        When the energy of the scene's interior still rises after every source has ended,
        the run warns ``fd.GrowingFieldWarning``, which names this knob with the value measured
        for the face's target; a run that is merely unsettled is only told to run longer.
    """

    cells: int = PML_CELLS
    wavelength: object = None
    background: object = None
    reflection: float = PML_REFLECTION
    allow_thin: bool = False
    stabilize: object = None

    def __post_init__(self):
        from .materials import as_material
        cells = static_int('PML', 'cells', self.cells)
        if cells < 1:
            raise ValueError(f'PML(cells={self.cells!r}): a layer is at least one cell deep.')
        if cells < PML_CELLS and not self.allow_thin:
            raise ValueError(THIN_PML_MESSAGE.format(cells=cells))
        object.__setattr__(self, 'cells', cells)
        if self.wavelength is not None:
            w = static_float('PML', 'wavelength', self.wavelength)
            if not w > 0:
                raise ValueError(f'PML(wavelength={self.wavelength!r}): a positive length.')
            object.__setattr__(self, 'wavelength', w)
        r = static_float('PML', 'reflection', self.reflection)
        if not 0 < r < 1:
            raise ValueError(
                f'PML(reflection={self.reflection!r}): the target is an amplitude ratio in '
                f'(0, 1); 1e-8 is the default and the legacy engine\'s.')
        object.__setattr__(self, 'reflection', r)
        object.__setattr__(self, 'allow_thin', bool(self.allow_thin))
        if self.stabilize is not None:
            rho = static_float('PML', 'stabilize', self.stabilize)
            if not (rho >= 0.0 and math.isfinite(rho)):
                raise ValueError(
                    f'PML(stabilize={self.stabilize!r}): the stabilizing loss is a finite '
                    f'factor >= 0 -- 0 (or None, the default) is the pure stretch and 3e-3 the '
                    f'measured value that stops a bound mode from growing at the default '
                    f'reflection target (a weaker target needs more).')
            object.__setattr__(self, 'stabilize', rho)
        if self.background is not None:
            object.__setattr__(self, 'background',
                               as_material(self.background, 'PML(background=)'))


@spec
class Absorber(Absorbing):
    """A graded matched conductivity instead of a stretch: the fallback where a PML cannot go.

    A PML is matched to one ``(eps, mu)``; a conductive or dispersive medium crossing the
    face has no such match, and a stretch that is not matched is a reflector with extra
    steps. ``Absorber`` grades an electric rate and an equal magnetic rate into ``ae``/``be``
    and ``ah``/``bh`` over ``cells`` cells, with the ``damping_profile`` shape
    ``6 u**3 / thickness`` ported from ``src/wavebit/_core/grid_fdtd.py`` (a kernel seed).

    **What it absorbs by.** The two rates are equal, and equal rates set the attenuation
    constant ``r`` rather than doubling it (substituting ``E, H -> e^{-rt}E', e^{-rt}H'`` into
    both curl equations returns the lossless pair). The integral of ``6 u**3`` over a
    unit-width layer is ``3/2``, so one pass through the layer costs ``exp(-1.5)`` and the
    round trip a normally incident wave makes -- in, off the PEC wall behind the layer, and
    out -- is ``exp(-3) = 4.98e-2``. That number does not depend on ``cells``: the profile is
    normalised by its own thickness, so a deeper layer spreads the same total and buys nothing
    at normal incidence. ``tests/test_fd_absorber.py`` measures 4.81e-2 to 5.04e-2 from 8 to 32
    cells and settles the scaling law behind it.

    It absorbs far worse than a PML at any angle and hopelessly at grazing -- it is matched
    only at normal incidence -- so it is the fallback and never the default. It carries no
    memory and no kernel code: ``Absorber`` is a change to four coefficient arrays.

    There is no ``background=``: the grading is matched **by construction**, because the
    electric and magnetic rates are equal whatever the medium is, so there is nothing for a
    background to change. ``PML(background=...)`` is the one that reads it, and naming it here
    is refused by ``fd.Absorber``'s own keyword check rather than silently ignored.
    """

    cells: int = PML_CELLS

    def __post_init__(self):
        cells = static_int('Absorber', 'cells', self.cells)
        if cells < 2:
            raise ValueError(
                f'Absorber(cells={self.cells!r}): the profile 6*u**3/thickness is zero at the '
                f'inner face and rises across the layer, so one cell is a step and not a '
                f'grading; two is the minimum that grades at all.')
        object.__setattr__(self, 'cells', cells)


_WRAPPING = (Periodic, Bloch, Axis1D)


def _pair(axis, value):
    """The ``(low, high)`` pair an axis declaration means, validated."""
    if isinstance(value, FaceType):
        if isinstance(value, Symmetry):
            raise ValueError(
                f'Boundaries({axis}=Symmetry(...)): a mirror on both faces of one axis is not '
                f'a half box, and stage 0 has no rule for it. Give a (Symmetry, PEC) or '
                f'(Symmetry, PMC) pair, mirror side first or last as the geometry needs.')
        return (value, value)
    if isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, FaceType) for v in value):
        lo, hi = value
        if isinstance(lo, _WRAPPING) or isinstance(hi, _WRAPPING):
            try:
                same = type(lo) is type(hi) and (lo is hi or bool(lo == hi))
            except Exception:          # a traced phase cannot be compared; one object can
                same = False
            if not same:
                raise ValueError(
                    f'Boundaries({axis}=({lo}, {hi})): Periodic, Bloch and Axis1D wrap the axis, so '
                    f'both faces must carry the same declaration. Give one object for the '
                    f'axis, {axis}={lo if isinstance(lo, _WRAPPING) else hi}.')
        if isinstance(lo, Symmetry) and isinstance(hi, Symmetry):
            raise ValueError(
                f'Boundaries({axis}=(Symmetry, Symmetry)): a mirror on both faces of one axis '
                f'is not a half box; Symmetry goes on at most one face, with PEC or PMC on '
                f'the other.')
        return (lo, hi)
    raise TypeError(
        f'Boundaries({axis}={value!r}): an axis takes one face type (both faces) or a '
        f'(low, high) pair of face types -- Periodic(), Bloch(phase=...), PEC(), PMC(), '
        f'Symmetry(parity=...), PML(cells=..., wavelength=...), Absorber(cells=...). Every '
        f'axis is required; there is no default face.')


@spec
class Boundaries:
    """The six faces, declared per axis as one face type or a ``(low, high)`` pair."""

    x: object
    y: object
    z: object

    def __post_init__(self):
        for axis in AXES:
            object.__setattr__(self, axis, _pair(axis, getattr(self, axis)))

    def face(self, name):
        """The face type on ``name`` (``'x-'`` .. ``'z+'``)."""
        if name not in FACES:
            raise ValueError(f'no face named {name!r}; the faces are {FACES}')
        return getattr(self, name[0])[0 if name[1] == '-' else 1]

    @classmethod
    def periodic(cls, *, z):
        """``x`` and ``y`` Periodic, ``z`` as given: the periodic-cell preset."""
        return cls(x=Periodic(), y=Periodic(), z=z)

    @classmethod
    def open(cls, face=None, *, x=None, y=None, z=None, **later):
        """Absorbing faces all round: ``Boundaries.open(fd.PML(cells=12, wavelength=0.55))``.

        ``face`` is the absorbing face type every axis gets; ``x=``, ``y=`` and ``z=``
        override one axis each with anything a ``Boundaries`` axis takes, so a periodic cell
        with absorbing top and bottom is ``Boundaries.open(pml, x=fd.Periodic(),
        y=fd.Periodic())`` and a half box is ``Boundaries.open(pml, z=(fd.Symmetry(parity=-1),
        pml))``.

        The default ``face`` is ``PML(cells=12)`` with no wavelength, which the scene refuses
        unless ``Resolution(wavelength=...)`` supplies one -- so the preset never silently
        picks a CFS pole for you.
        """
        if later:
            raise TypeError(unknown_keyword_message('Boundaries.open', list(later),
                                                    ('face', 'x', 'y', 'z')))
        if face is None:
            face = PML()
        if not isinstance(face, Absorbing):
            raise TypeError(
                f'Boundaries.open({face!r}): the preset takes an absorbing face -- '
                f'fd.PML(...) or fd.Absorber(...) -- and puts it on every axis. For anything '
                f'else, name the faces: fd.Boundaries(x=..., y=..., z=...).')
        return cls(x=face if x is None else x, y=face if y is None else y,
                   z=face if z is None else z)
