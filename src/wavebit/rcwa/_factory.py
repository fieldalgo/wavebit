"""Legacy RCWA factory, used by the typed lifecycle."""
from wavebit._common.backend import eager_under_torch_compile as _eager_under_torch_compile
from wavebit._common.declarations import _merge_domain

@_eager_under_torch_compile
def rcwa(backend='numpy', *args, stack=None, incidence=None, **kwargs):
    """
    decide backend and return RCWA solver instance

    The scene is stated in **relative permittivity**: ``structure`` is a permittivity raster
    or a per-layer list of shape recipes naming a :class:`wavebit.Material` per region, and
    ``top`` / ``bot`` are the two half-space materials. ``mu`` is the raster path's
    permeability field; on the shape paths a permeability lives inside each ``Material``. The
    retired spellings (``ucell``, ``n_top``, ``n_bot``) are refused as unknown keywords, which
    matters because the *quantity* changed with them -- see
    ``docs/reference/common/conventions.md``.

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
    from wavebit._common.enums import Backend
    from wavebit._common.enums import backend_of
    backend = backend_of(backend)
    if stack is not None:
        kwargs = _merge_domain(kwargs, stack.as_kwargs(), 'stack')
    if incidence is not None:
        kwargs = _merge_domain(kwargs, incidence.as_kwargs(), 'incidence')

    from wavebit.rcwa._engine.illumination import CoherentSolver
    from wavebit.rcwa._engine.sources import is_source_excitation
    if is_source_excitation(kwargs.get('excitation')):
        return CoherentSolver(backend, args, kwargs)

    if backend is Backend.NUMPY:
        from wavebit.rcwa._backends.numpy_backend import SolverNumpy
        solver = SolverNumpy(backend=backend, *args, **kwargs)
    elif backend is Backend.JAX:
        from wavebit.rcwa._backends.jax_backend import SolverJax
        solver = SolverJax(backend=backend, *args, **kwargs)
    else:
        from wavebit.rcwa._backends.torch_backend import SolverTorch
        solver = SolverTorch(backend=backend, *args, **kwargs)
    return solver
