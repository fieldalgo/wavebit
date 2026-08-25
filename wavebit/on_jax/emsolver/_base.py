import jax

import jax.numpy as jnp
import jax.numpy as _xp

from ..._core.engine import CoreSolveMixin


class _BaseRCWA(CoreSolveMixin):
    #: The array namespace this solver's arrays live in. A class attribute rather than a
    #: lookup, because the one caller that needs it -- the analytic builder, which starts from
    #: parametric shapes and so cannot infer a namespace from its input -- used to reach a
    #: `_namespace_for_backend(self.backend)` living in `_core` that imported jax and torch by
    #: name and fell through to numpy on an unrecognised code, with no `else`. The solver
    #: already knows which namespace it is; asking it is cheaper than deciding again, and it
    #: keeps `_core` from naming an optional dependency.
    xp = _xp

    def _device_scope(self):
        """`jax.default_device`, which is how JAX is told where to place what it builds.

        `self.device` is a *list* here -- `jax.devices('cpu')` -- so it is indexed. That is
        the one thing about it that is not obvious, and it is why this is an override rather
        than a value the base could hold.
        """
        return jax.default_device(self.device[0])

    #: A JAX array -- and a `DynamicJaxprTracer` too, which is what lets a period be
    #: differentiated or closed over inside `jax.jit`.
    _ARRAY_TYPES = (jnp.ndarray,)

    def _resolve_device(self, value):
        """A list of :class:`jax.Device`, which is what `jax.default_device` takes.

        A single already-resolved device is wrapped rather than refused, so
        `solver.device = solver.device` round-trips against the getter. The membership test is
        against the public `jax.Device`: matching the implementation class's repr instead broke
        silently when jaxlib moved it from `xla_extension` to `_jax`.
        """
        if value is None or value in (0, 'cpu'):
            return jax.devices('cpu')
        if value in (1, 'gpu', 'cuda'):
            return jax.devices('gpu')
        if isinstance(value, jax.Device):
            return [value]
        if (isinstance(value, (list, tuple)) and value
                and all(isinstance(d, jax.Device) for d in value)):
            return list(value)
        raise ValueError(f'unsupported jax device: {value!r}')

