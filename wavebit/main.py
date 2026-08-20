def _merge_domain(kwargs, domain_kwargs, source):
    """Fold a domain object's keyword arguments into ``kwargs``, rejecting duplicates.

    Passing e.g. both ``stack=`` and ``structure=`` is ambiguous, so it is an explicit error
    rather than a silent override.
    """
    clash = set(kwargs) & set(domain_kwargs)
    if clash:
        raise ValueError(
            f"{source} already sets {sorted(clash)}; do not also pass them to rcwa.")
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
        from .on_numpy.solver import SolverNumpy
        solver = SolverNumpy(backend=backend, *args, **kwargs)
    elif backend == 1:
        from .on_jax.solver import SolverJax
        solver = SolverJax(backend=backend, *args, **kwargs)
    elif backend == 2:
        from .on_torch.solver import SolverTorch
        solver = SolverTorch(backend=backend, *args, **kwargs)
    else:
        raise ValueError
    return solver
