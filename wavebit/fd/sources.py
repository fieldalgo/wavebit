"""What drives a run: in stage 0, an initial field only -- and the protocol every later term obeys.

Stage 0 has no sources. The excitation is ``InitialField``: the state the leapfrog starts
from, E at ``t = 0`` and H at ``t = -dt/2``. It is not a term at all -- nothing is added into
the fields during the run -- which is exactly why it is the stage-0 excitation: it exercises
the kernel's slots with an empty term list, so the first gates test the curl, the faces and
the masks and nothing else. Sources arrive in stage 2 as terms.

**The Term protocol** is fixed here because the kernel step is written against it now.
A term adds into a slab of the curl that feeds the next update, in one of two phase-tagged
slots, and may carry state of its own (``mem``) that it advances once per step:

    before_H(cE, E, mem, n, params, rt) -> cE     # cE feeds  H = ah*H - bh*cE
    before_E(cH, H, mem, n, params, rt) -> cH     # cH feeds  E = ae*E + be*cH
    advance(mem, E, H, n, params, rt)   -> mem
    init_mem(plan, rt)                  -> mem

``rt`` is the runtime whose ``slab_add`` a term must use for the write (the only primitive
whose cost was measured, ``studies/fdtd_slab_add_cost.py``), and ``mem`` is the term's own
entry, not the whole memory. Terms within one slot must commute to round-off: the
permutation gate applies two of them in either order. On Torch the write is in place, so a
term must not read a view of the array it is about to write into -- the order gate's mutant
does exactly that and must raise. A term is a frozen dataclass of static values (it is part
of the plan and of its skeleton hash); anything traced belongs in ``params``.
"""
from typing import Protocol, runtime_checkable

from ._spec import spec

__all__ = ['InitialField', 'Term']


@runtime_checkable
class Term(Protocol):
    """The phase-tagged term the kernel step calls; see the module docstring."""

    name: str

    def init_mem(self, plan, rt): ...

    def before_H(self, cE, E, mem, n, params, rt): ...

    def before_E(self, cH, H, mem, n, params, rt): ...

    def advance(self, mem, E, H, n, params, rt): ...


def _check_field(name, value):
    if value is None:
        return
    if callable(value):
        return
    shape = getattr(value, 'shape', None)
    if shape is not None:
        if len(shape) != 4 or shape[0] != 3:
            raise ValueError(
                f'InitialField({name}=array): the array is [component, z, y, x] with three '
                f'components, so its shape is (3, nz, ny, nx); got {tuple(shape)}.')
        return
    if isinstance(value, (tuple, list)):
        if len(value) != 3:
            raise ValueError(
                f'InitialField({name}=(...)): a tuple gives one entry per component, (x, y, z), '
                f'so it has three entries; got {len(value)}.')
        for c, v in zip('xyz', value):
            if not (v is None or callable(v) or hasattr(v, 'shape') or isinstance(v, (int, float, complex))):
                raise TypeError(
                    f'InitialField({name}=...): the {c} entry is {type(v).__name__}; each entry '
                    f'is a callable fn(x, y, z), an array, a number, or None for zero.')
        return
    raise TypeError(
        f'InitialField({name}={type(value).__name__}): give an array (3, nz, ny, nx), a '
        f'callable fn(x, y, z) applied to every component, or a tuple of three per-component '
        f'entries (callable, array, number or None).')


@spec
class InitialField:
    """The state the leapfrog starts from: E at ``t = 0``, H at ``t = -dt/2``.

    Each of ``E`` and ``H`` is one of

    * an array ``(3, nz, ny, nx)`` in the ``[component, z, y, x]`` layout;
    * a tuple ``(fx, fy, fz)`` of per-component entries, each a callable ``fn(x, y, z)``, an
      array ``(nz, ny, nx)``, a number, or ``None`` for zero;
    * a single callable ``fn(x, y, z)``, applied to every component.

    A callable receives the three coordinate grids of **that component's Yee points**,
    each ``(nz, ny, nx)``, in the user's length unit -- so ``lambda x, y, z: np.cos(k*x)``
    samples ``Ex`` half a cell further along x than ``Ey``, as the staggering requires.
    Nothing is interpolated: the caller is stating a field on the grid, not in the continuum.

    ``amplitude`` scales both fields and is a traced parameter, so a gradient with respect
    to the excitation's size flows through it without rebuilding the pattern (the AD gate
    differentiates final energy with respect to it).
    """

    E: object = None
    H: object = None
    amplitude: object = 1.0

    def __post_init__(self):
        if self.E is None and self.H is None:
            raise ValueError('InitialField(): neither E nor H given, so the run would start '
                             'from nothing and stay there. Give at least one.')
        _check_field('E', self.E)
        _check_field('H', self.H)
