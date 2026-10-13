"""Shared diffraction result pytrees, loaded only when JAX is requested."""
import jax
from wavebit._common.backend import eager_under_torch_compile as _eager_under_torch_compile
from wavebit.observations.diffraction import Result, ResultSub

@_eager_under_torch_compile
@jax.tree_util.register_pytree_node_class
class ResultJax(Result):
    """Shared container, registered as a pytree so a jitted method can return one."""


@_eager_under_torch_compile
@jax.tree_util.register_pytree_node_class
class ResultSubJax(ResultSub):
    """Shared container; a pytree for the same reason as :class:`ResultJax`."""
