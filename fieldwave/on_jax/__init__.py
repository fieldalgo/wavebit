import jax
from jax import tree_util

# RCWA needs double precision: with x64 off, jax silently truncates float64 and
# complex128 at array creation, which costs ~8 significant digits and surfaces
# only as a cross-backend mismatch. Enabling it changes the caller's jax
# process-wide, so it happens here — reached only when the jax backend is
# actually imported — and not on `import fieldwave`.
jax.config.update('jax_enable_x64', True)

from fieldwave.on_jax.mee import MeeJax

tree_util.register_pytree_node(MeeJax,
                               MeeJax._tree_flatten,
                               MeeJax._tree_unflatten)

