"""Shared backend and polarization selectors."""
from enum import Enum, IntEnum

class Backend(str, Enum):
    """Array backend for :func:`wavebit.rcwa`, named by the library it computes in.

    Each value is the name that library is imported by, and it is the only spelling taken:
    ``'JAX'``, ``'pytorch'`` and an integer are refused by :func:`backend_of`, which every door
    that takes a backend calls. ``solver.backend`` holds the member, so
    ``solver.backend == 'jax'`` reads as written.

    The values were the integers ``0`` / ``1`` / ``2`` until 2026-10-11 and changed because a
    number says nothing about which library it means, the spec documents already stored the
    names, and one vocabulary is what a caller should have to learn.
    """
    NUMPY = 'numpy'   # no automatic differentiation; the default
    JAX = 'jax'       # differentiable; jit with a frozen geometry
    TORCH = 'torch'   # differentiable, eagerly


DIFFERENTIABLE_BACKENDS = frozenset({Backend.JAX, Backend.TORCH})


class Tetm(IntEnum):
    """The two states of the ``tetm`` declaration; the value is ``tetm_index``."""
    TE = 0
    TM = 1


def backend_of(value):
    """Validate a ``backend`` value: a :class:`Backend` member or its name, exactly.

    The one parser behind every door that takes a backend -- :func:`wavebit.rcwa`, the solver
    classes, :class:`~wavebit.CoherentSolver`, the ``meshgrid`` helpers and the spec's
    ``numerics.backend``. Exact rather than case-folded like :func:`_one_of`: each name is the
    one ``import`` takes, so there is one spelling per backend, and a near spelling is refused
    as the retired integers are.
    """
    if isinstance(value, Backend):
        return value
    if isinstance(value, str):
        try:
            return Backend(value)
        except ValueError:
            pass
    raise ValueError(
        f"backend={value!r} is not valid; expected one of {[m.value for m in Backend]}")
