from functools import partial

import jax

from ._base import _BaseRCWA
from ..._core.engine import CoreFactorizationMixin, device_scoped
from ..._core.result import Result, ResultSub


@jax.tree_util.register_pytree_node_class
class ResultJax(Result):
    """Shared container, registered as a pytree so a jitted method can return one."""


@jax.tree_util.register_pytree_node_class
class ResultSubJax(ResultSub):
    """Shared container; a pytree for the same reason as :class:`ResultJax`."""


class RCWAJax(CoreFactorizationMixin, _BaseRCWA):
    #: The pytree-registered pair, so a jitted `solve_field` can return one. See the
    #: declaration in `CoreFactorizationMixin`.
    _RESULT = ResultJax
    _RESULT_SUB = ResultSubJax

    @partial(jax.jit, static_argnums=(1, 2, 3, 4, 5),
             static_argnames=('sample_x', 'sample_y', 'sample_z', 'incidences', 'geometry'))
    @device_scoped
    def solve_field(self, sample_x=20, sample_y=20, sample_z=20, incidences=('given',),
                    geometry=None, **kwargs):
        """Solve and reconstruct the field in one jit-compiled call.

        ``geometry`` is required here, unlike on :meth:`solve`. This method jits over
        ``self``, so the structure arrives as tracers and the discrete decisions inside the
        factorization and the solve have nothing concrete to branch on; a
        :class:`~wavebit._core.geometry.StaticGeometry` from
        :meth:`~wavebit._core.engine.CoreFactorizationMixin.freeze_geometry` supplies them as
        static data. It is a static argument -- reuse one freeze across calls and you compile
        once.

        Because this method always traces, a stale ``geometry`` is replayed on trust here: the
        comparison that would catch it is the thing the freeze exists to avoid. ``solve``
        refuses one eagerly, so check a long fit there, or re-freeze on a schedule.

        ``sample_x`` / ``sample_y`` / ``sample_z`` and ``incidences`` are static too: they set the
        output shape and select which incidences to reconstruct.

        Use :meth:`solve_field_eager` for the eager equivalent that auto-detects.
        """
        if geometry is None:
            raise ValueError(
                'solve_field is jit-compiled and needs the geometry frozen up front:\n'
                '    geometry = solver.freeze_geometry()\n'
                '    result, field_cell = solver.solve_field(..., geometry=geometry)\n'
                'Reuse that one geometry across calls to compile only once. '
                'For the eager, auto-detecting path use solver.solve_field_eager(...).')
        return self._solve_and_field(sample_x, sample_y, sample_z, incidences, geometry,
                                     **kwargs)


# The eager path, which is `CoreFactorizationMixin.solve_field` under its other name -- one
# function, two names, rather than a body written twice. JAX is the only backend that needs
# both, because it is the only one whose `solve_field` is jitted and so demands a frozen
# geometry; the other two have nothing to be eager *relative to*. Bound here rather than inside
# the class because the name it aliases is the one this class overrides.
RCWAJax.solve_field_eager = CoreFactorizationMixin.solve_field
