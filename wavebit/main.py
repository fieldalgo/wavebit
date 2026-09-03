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
    per-instance state to keep: no eigen-decomposition to cache, no frozen geometry, no
    gradient to take. A solver object would be a place to put the state this method does not
    have.

    It answers pure TE / TM at normal and oblique incidence, a conical mount through ``psi``
    at any ``theta`` and ``phi``, and a crossed grating; dispersive and magnetic media are
    carried on all of them. Everything outside that is refused by name. There is no ``backend``
    argument; ``wavebit/_core/fdtd.py``'s docstring says why.

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
    # `Incidence.as_kwargs` always names the whole polarization family; the three this
    # formulation cannot carry are dropped only when they are unset, and refused otherwise.
    for unset in ('psi', 'chi', 'polarization'):
        if kwargs.get(unset) is None:
            kwargs.pop(unset, None)
    return solve_fdtd(**kwargs)
