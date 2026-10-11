from ._core.backend import eager_under_torch_compile as _eager_under_torch_compile


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
