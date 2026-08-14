import jax
from jax import tree_util

# RCWA needs double precision: with x64 off, jax silently truncates float64 and
# complex128 at array creation, which costs ~8 significant digits and surfaces
# only as a cross-backend mismatch. Enabling it changes the caller's jax
# process-wide, so it happens here — reached only when the jax backend is
# actually imported — and not on `import wavebit`.
#
# It is *also* done earlier when the caller already had jax loaded, by
# `wavebit/__init__.py::_enable_jax_x64_if_already_imported`. Doing it only
# here left a window: an array built between `import wavebit` and the first
# solve is complex64, and gets widened later without recovering anything. This
# line still covers the case where jax arrives after wavebit.
jax.config.update('jax_enable_x64', True)

from wavebit.on_jax.solver import SolverJax

tree_util.register_pytree_node(SolverJax,
                               SolverJax._tree_flatten,
                               SolverJax._tree_unflatten)

