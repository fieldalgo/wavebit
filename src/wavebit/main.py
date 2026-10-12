from ._core.backend import eager_under_torch_compile as _eager_under_torch_compile


def _merge_domain(kwargs, domain_kwargs, source, solver='rcwa'):
    """Fold a domain object's keyword arguments into ``kwargs``, rejecting duplicates.

    Passing e.g. both ``stack=`` and ``structure=`` is ambiguous, so it is an explicit error
    rather than a silent override.

    ``solver`` names the door in the message. It is a parameter rather than a constant because
    :func:`fdtd` is the second caller, and a duplicate keyword on the time-domain entry point
    was being reported as a mistake made to ``rcwa``.
    """
    clash = set(kwargs) & set(domain_kwargs)
    if clash:
        raise ValueError(
            f"{source} already sets {sorted(clash)}; do not also pass them to {solver}.")
    merged = dict(kwargs)
    merged.update(domain_kwargs)
    return merged


@_eager_under_torch_compile
def rcwa(backend='numpy', *args, stack=None, incidence=None, **kwargs):
    """
    decide backend and return RCWA solver instance

    The scene is stated in **relative permittivity**: ``structure`` is a permittivity raster
    or a per-layer list of shape recipes naming a :class:`wavebit.Material` per region, and
    ``top`` / ``bot`` are the two half-space materials. ``mu`` is the raster path's
    permeability field; on the shape paths a permeability lives inside each ``Material``. The
    retired spellings (``ucell``, ``n_top``, ``n_bot``) are refused by name, because the
    *quantity* changed with them -- see ``docs/conventions.md``.

    Args:
        backend: ``'numpy'`` (the default), ``'jax'`` or ``'torch'``, or the
            :class:`wavebit.Backend` member. Anything else -- the integers ``0`` / ``1`` /
            ``2`` it took until 2026-10-10 among them -- is refused with the list of names.
        stack: optional typed ``Stack`` supplying ``structure`` / ``thickness`` / ``top`` /
            ``bot`` / ``period`` (and ``mu``, when its layers carry one); adapts in, so do
            not also pass those keywords.
        incidence: optional typed ``Incidence`` supplying ``wavelength`` / ``theta`` /
            ``phi`` / ``kx`` / ``ky`` / ``tetm`` / ``psi`` / ``chi`` / ``polarization`` / ``incidence_side``
            (adapts in; do not also pass those keywords).
        *args: passed to RCWA instance
        **kwargs: passed to RCWA instance

    Returns:
        RCWA solver instance, or :class:`wavebit.CoherentSolver` when ``excitation``
        contains typed sources (``PlaneWave``, ``GaussianBeam``, ``ElectricDipole``),
        or is keyed by ``'top'`` / ``'bottom'`` for external sources.
        ``power_region`` selects the averaging area. Internal electric dipoles
        require ``length_unit``: metres per scene length unit.

    """
    from ._core.enums import Backend, backend_of
    backend = backend_of(backend)
    if stack is not None:
        kwargs = _merge_domain(kwargs, stack.as_kwargs(), 'stack')
    if incidence is not None:
        kwargs = _merge_domain(kwargs, incidence.as_kwargs(), 'incidence')

    from ._core.illumination import CoherentSolver
    from ._core.sources import is_source_excitation
    if is_source_excitation(kwargs.get('excitation')):
        return CoherentSolver(backend, args, kwargs)

    if backend is Backend.NUMPY:
        from .backends.numpy_backend import SolverNumpy
        solver = SolverNumpy(backend=backend, *args, **kwargs)
    elif backend is Backend.JAX:
        from .backends.jax_backend import SolverJax
        solver = SolverJax(backend=backend, *args, **kwargs)
    else:
        from .backends.torch_backend import SolverTorch
        solver = SolverTorch(backend=backend, *args, **kwargs)
    return solver


@_eager_under_torch_compile
def fdtd(stack=None, incidence=None, **kwargs):
    """Solve the same scene in the time domain; returns a :class:`wavebit._core.result.Result`.

    The point of having this beside :func:`rcwa` is that its errors are different errors --
    RCWA truncates a Fourier series, FDTD discretises space and time -- so a defect that
    survives both is unlikely to belong to either method. It reads the same ``Stack`` and
    ``Incidence``, and it returns the same ``Result``, so the two are compared without a
    translation layer.

    Unlike :func:`rcwa` this hands back the answer rather than a solver. There is no reusable
    per-instance state to keep. Explicit backend selection enables material gradients
    through the common native-grid engine.

    It answers pure TE / TM at normal and oblique incidence, linear or elliptical incidence through
    ``psi``/``chi`` or a complex ``polarization`` pair at any ``theta`` and ``phi``,
    and a crossed grating; dispersive and magnetic media are
    carried on all of them. With no backend specified, the legacy NumPy scene solver
    preserves time records and graded meshes. Explicit ``backend='numpy'/'jax'/'torch'`` selects the shared scene engine, requires
    explicit resolution, and differentiates raster materials and electric poles.
    ``transverse_resolution`` optionally sets x/y cells per length independently
    of the z resolution, as a scalar or an ``(x, y)`` pair.
    Both paths support time records and user frequency planes; source and
    normalization conventions are documented in the time-domain guide.
    Explicit backends also support bottom incidence and static z grading.
    They accept axis-aligned rectangle/square layer recipes with the same
    centre, absolute length and Material conventions as RCWA. Subpixel cell
    integration preserves shape gradients and periodic painting. Optional
    shape_smoothing averages transverse translations over static (x,y) widths;
    shape contacts can still have topology kinks. Two periods are required.
    ``shape_integration='area'`` accepts multiple rectangles or one circle/ellipse per layer,
    with projected extents within both periods. Circle/disk lx (or l) is the diameter.
    Rectangle/ellipse angles retain derivatives in radians; ellipse lx/ly are
    full principal-axis lengths. Ellipses use fixed 16-point angular integration
    for sharp and averaged cells. Positive shape_smoothing
    averages translations by exact polygon moments or stable disk/ellipse arc strips
    with fixed 16-point Gaussian integration.
    Multiple rectangles retain later-shape painting through live boundary events
    and exact cubic boundary integrals; mixed or multiple curved shapes are refused.
    Cell integrals give arithmetic transverse averages, with full y sampling even at angle zero. It is not
    a full rotated effective tensor or a second-order accuracy guarantee.
    Their real scalar ``theta`` and ``phi`` can retain autodiff graphs when an
    explicit ``truncation_order`` fixes the output window. Array ``phi`` uses
    the full y-period mesh even at zero and requires Jones/psi polarization.
    Both angles change the Bloch wavevector shared by the wavelength band.
    ``align_layers=True`` places every layer boundary on a z node, independently
    of padding grading; geometry is static. Half-space epsilon/mu retain
    gradients with explicit ``mesh_indices=(n_top,n_bottom)`` and an order
    window. These static nominal refractive indices set padding/grading;
    actual half-space coefficients determine propagation and CPML.
    With explicit ``mesh_thickness``, ``variable_depth=True`` permits positive
    layer lengths with a changing total. Laboratory coordinates remain fixed
    and complex amplitudes follow the live physical input/output faces.
    Layers and smoothing supports must clear the fixed extraction planes.
    ``pulse_width`` controls their Gaussian source envelope in c=1
    time units; ``pulse_notches`` optionally suppresses selected positive
    frequencies with causal delayed copies of that pulse.
    ``diffraction_distance`` sets extraction-plane distances from
    the incident and outgoing interfaces without resizing the mesh. Diagnostics
    report physical and Yee transverse cutoff classifications for each order.
    ``diffraction_method='directional'`` separates travelling E/H components
    and reports incoming residuals; its measurement planes need locally uniform z cells.
    ``source_distance`` places the incident current sheet independently of
    padding and extraction planes, using the nearest incident-side z node.
    Layers are bottom-to-top, +z is up, and the bottom interface is z=0.
    Order amplitudes use the cell centre and each order's own s/p frame, as RCWA.
    Cartesian order amplitudes r_xyz/t_xyz and RCWA field reconstruction state
    are not supplied; time-domain fields are available through diagnostics.
    Cartesian kx/ky incidence is currently unsupported; use real theta/phi.
    See the time-domain guide for scope and coordinate conventions.

    Args:
        stack: optional typed ``Stack`` supplying ``structure`` / ``thickness`` / ``top`` /
            ``bot`` / ``period`` (adapts in; do not also pass those keywords).
        incidence: optional typed ``Incidence`` supplying ``wavelength`` / ``theta`` /
            ``phi`` / ``tetm`` / ``incidence_side`` (adapts in, same rule).
        **kwargs: passed to :func:`~wavebit._core.fdtd.solve_fdtd`.

    Returns:
        Result: the same container an RCWA solve returns.
    """
    from ._core.periodic_fdtd import solve_periodic_fdtd
    if stack is not None:
        kwargs = _merge_domain(kwargs, stack.as_kwargs(), 'stack', 'fdtd')
    if incidence is not None:
        kwargs = _merge_domain(kwargs, incidence.as_kwargs(), 'incidence', 'fdtd')
    # The typed carrier names the whole polarization family; omit its unset values.
    for unset in ('psi', 'chi', 'polarization'):
        if kwargs.get(unset) is None:
            kwargs.pop(unset, None)
    for name in ('kx', 'ky'):
        if kwargs.pop(name, None) is not None:
            raise ValueError('fdtd: Cartesian kx/ky incidence is not implemented; use real theta/phi in radians.')
    # The shared declaration is an IntEnum; private Yee kernels use names.
    from ._core.enums import Tetm
    tetm = kwargs.get('tetm')
    if isinstance(tetm, Tetm):
        kwargs['tetm'] = tetm.name.lower()
    elif isinstance(tetm, str):
        kwargs['tetm'] = tetm.lower()
    backend = kwargs.pop('backend', None)
    if backend is not None:
        from ._core.enums import backend_of
        backend = backend_of(backend)
    return solve_periodic_fdtd(kwargs, backend=backend)
