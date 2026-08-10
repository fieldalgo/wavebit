def _merge_domain(kwargs, domain_kwargs, source):
    """Fold a domain object's keyword arguments into ``kwargs``, rejecting duplicates.

    Passing e.g. both ``stack=`` and ``ucell=`` is ambiguous, so it is an explicit error
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

    Args:
        backend: decide backend. 0 is numpy and 1 is JAX.
        stack: optional typed ``Stack`` supplying ``ucell`` / ``thickness`` / ``n_top`` /
            ``n_bot`` / ``period`` (adapts in; do not also pass those keywords).
        incidence: optional typed ``Incidence`` supplying ``wavelength`` / ``theta`` /
            ``phi`` / ``pol`` / ``psi`` (adapts in; do not also pass those keywords).
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
