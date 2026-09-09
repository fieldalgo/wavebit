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


def rcwa(backend=0, *args, stack=None, incidence=None, **kwargs):
    """
    decide backend and return RCWA solver instance

    The scene is stated in **relative permittivity**: ``structure`` is a permittivity raster
    or a per-layer list of shape recipes naming a :class:`wavebit.Material` per region, and
    ``top`` / ``bot`` are the two half-space materials. ``mu`` is the raster path's
    permeability field; on the shape paths a permeability lives inside each ``Material``. The
    retired spellings (``ucell``, ``n_top``, ``n_bot``) are refused by name, because the
    *quantity* changed with them -- see ``docs/conventions.md``.

    Args:
        backend: decide backend. 0 is numpy, 1 is JAX, 2 is PyTorch.
        stack: optional typed ``Stack`` supplying ``structure`` / ``thickness`` / ``top`` /
            ``bot`` / ``period`` (and ``mu``, when its layers carry one); adapts in, so do
            not also pass those keywords.
        incidence: optional typed ``Incidence`` supplying ``wavelength`` / ``theta`` /
            ``phi`` / ``tetm`` / ``psi`` / ``chi`` / ``polarization`` / ``incidence_side``
            (adapts in; do not also pass those keywords).
        *args: passed to RCWA instance
        **kwargs: passed to RCWA instance

    Returns:
        RCWA: RCWA solver instance

    """
    if stack is not None:
        kwargs = _merge_domain(kwargs, stack.as_kwargs(), 'stack')
    if incidence is not None:
        kwargs = _merge_domain(kwargs, incidence.as_kwargs(), 'incidence')

    if backend == 0:
        from .backends.numpy_backend import SolverNumpy
        solver = SolverNumpy(backend=backend, *args, **kwargs)
    elif backend == 1:
        from .backends.jax_backend import SolverJax
        solver = SolverJax(backend=backend, *args, **kwargs)
    elif backend == 2:
        from .backends.torch_backend import SolverTorch
        solver = SolverTorch(backend=backend, *args, **kwargs)
    else:
        raise ValueError
    return solver


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
    preserves time records and graded meshes. Explicit ``backend=0/1/2`` (or
    ``numpy/jax/torch``) selects the shared scene engine, requires
    explicit resolution, and differentiates raster materials and electric poles.
    ``transverse_resolution`` optionally sets x/y cells per length independently
    of the z resolution, as a scalar or an ``(x, y)`` pair.
    Both paths support time records and user frequency planes; source and
    normalization conventions are documented in the time-domain guide.
    Explicit backends also support bottom incidence and static z grading.
    Their real scalar ``theta`` can retain an autodiff graph when an explicit
    ``truncation_order`` fixes the output window. Azimuth ``phi`` remains static;
    theta changes the Bloch wavevector shared by the wavelength band.
    ``align_layers=True`` places every layer boundary on a z node, independently
    of padding grading; geometry and half spaces are static.
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
    from ._core.fdtd import solve_fdtd
    if stack is not None:
        kwargs = _merge_domain(kwargs, stack.as_kwargs(), 'stack', 'fdtd')
    if incidence is not None:
        kwargs = _merge_domain(kwargs, incidence.as_kwargs(), 'incidence', 'fdtd')
    # The typed carrier names the whole polarization family; omit its unset values.
    for unset in ('psi', 'chi', 'polarization'):
        if kwargs.get(unset) is None:
            kwargs.pop(unset, None)
    backend = kwargs.pop("backend", None)
    if backend is not None:
        from ._core.scene_fdtd import solve_scene_fdtd
        return solve_scene_fdtd(backend=backend, **kwargs)
    return solve_fdtd(**kwargs)
