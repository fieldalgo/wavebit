"""Face rules: what each boundary type is on the Yee grid, and where its wall really sits.

**Which field a face pads.** ``curl E`` (which feeds the H update) is a forward difference
and needs the E neighbour on the HIGH side of each axis; ``curl H`` (feeding E) is a backward
difference and needs the H neighbour on the LOW side. So a high face pads E and only E, a
low face pads H and only H, and every wall is one missing layer plus, where the missing
layer is not the wall itself, a mask on the edge layer after the update. No ``N + 1`` arrays.

**The four neighbour rules** (``Pad.kind``), per face, per component of the padded field:

    zero        the missing layer is 0                      PEC, PMC
    periodic    the opposite edge layer                     Periodic
    bloch       the opposite edge layer times exp(+-i phi)  Bloch (phi traced, in params)
    mirror      +-1 times an interior layer                  Symmetry

**Wall positions**, which the plan records per face and the eigenmode gates check (a wall
one cell off changes the mode frequency by ``1/N`` and fails the recurrence by ~1e-2
against a round-off tolerance):

    PEC   low face: x_0        the zero pad is H at x_{-1/2}; tangential E on the node
                                layer x_0 is masked to zero -- the mask is the wall. The
                                normal H on that layer is masked too: n.B = 0 there, and
                                its update reads only the masked E, so it is dead
          high face: x_N       the zero pad IS the wall: tangential E at x_N is the pad
    PMC   low face: x_{-1/2}   the zero pad IS the wall: tangential H at x_{-1/2} is the pad
          high face: x_{N-1/2} tangential H on the edge layer x_{N-1/2} is masked, and the
                                dead normal E with it; the E pad at x_N feeds only the
                                masked components and is irrelevant

The masks run on the initial state as well as after every update (``step.py``), so an
initial field that violates a wall is projected onto it at ``t = 0``.

**Mirror planes.** A mirror on a face reflects the interior onto the missing layer with the
per-component signs of the parity. Under a mirror across x, E is a polar vector and H an
axial one, so for parity ``p`` (the tangential-E eigenvalue)

    E_normal -> -p     E_tangential -> +p
    H_normal -> +p     H_tangential -> -p

(``p = +1``: tangential E even, tangential H odd hence zero on the plane -- PMC-like;
``p = -1``: tangential E odd, PEC-like.) Where the plane can sit is decided by which field
the face pads and where that field's components lie, not chosen:

    low face  (pads H): the tangential H at x_{-1/2} is the image of x_{+1/2}, so the plane
              is the NODE plane x_0 -- the face itself, no redundant layer. The normal H at
              x_{-1} images x_{+1} (layer 1; never read by the curl, kept for generality).
    high face (pads E): the tangential E at x_N is the image of x_{N-1}, so the plane is the
              HALF plane x_{N-1/2}, half a cell inside the face. The normal E at x_{N+1/2}
              images x_{N-3/2} (layer N-2; never read).

The other placements are impossible in this layout: a half plane on the low face at
``x_{-1/2}`` puts the padded tangential H on the plane itself, where for ``p = -1`` it is an
unknown of its own (an ``N + 1`` array); a node plane on the high face at ``x_N`` does the
same to tangential E for ``p = +1``. A user who needs the other placement mirrors the
scene. The masks follow from the placement: a component lying on the plane whose sign is
``-1`` is zeroed there, and that is the whole table

    low  node plane, p = -1:  E_tangential and H_normal on layer 0
    low  node plane, p = +1:  nothing (the odd components sit half a cell off the plane)
    high half plane, p = +1:  E_normal and H_tangential on layer N-1
    high half plane, p = -1:  nothing

Two corollaries, both tested: ``Symmetry(parity=-1)`` on a low face is PEC on that face
(same mask; the pad it differs by feeds only masked components), and ``Symmetry(parity=+1)``
on a high face is PMC there. The two cases that are new are ``+1`` low (a PMC-like node
plane) and ``-1`` high (a PEC-like half plane), and the symmetry gate holds all four against
the full box.
"""
import dataclasses

from ..boundaries import PEC, PMC, Bloch, Periodic, Symmetry
from . import AX
from .grid import OFFSETS
from .masks import Mask, Slab

__all__ = ['Pad', 'FaceRule', 'lower_face', 'pad_neighbor', 'PADDED_FIELD']

#: Which field a side pads: E on high faces (forward differences), H on low (backward).
PADDED_FIELD = {'-': 'H', '+': 'E'}


@dataclasses.dataclass(frozen=True)
class Pad:
    """One component's missing layer: ``kind``, the interior ``layer`` it copies, its ``sign``."""

    kind: str
    layer: int = -1
    sign: int = 1


@dataclasses.dataclass(frozen=True)
class FaceRule:
    """What the plan records for a face: its kind, where its wall or plane is, its parity."""

    name: str
    kind: str
    position: float = None
    plane: str = None
    parity: int = None


def _mirror_pad(field, c, axis, side, parity, n):
    """The mirror pad of component ``c`` of ``field`` on ``side`` of ``axis`` (0, 1, 2)."""
    o = OFFSETS[field][c][axis]                       # 0 at a node, 1 half a cell along axis
    if side == '-':
        plane = 0                                     # x_0, in half cells
        q = -2 + o                                    # the missing layer's position
    else:
        plane = 2 * n - 1                             # x_{N-1/2}
        q = 2 * n + o
    layer = (2 * plane - q - o) // 2                  # the image, as a layer index
    normal = (c == axis)
    if field == 'E':
        sign = -parity if normal else parity
    else:
        sign = parity if normal else -parity
    return Pad(kind='mirror', layer=layer, sign=sign)


def lower_face(name, face, grid):
    """``(FaceRule, pads, masks)`` for ``face`` on the face ``name`` of ``grid``.

    ``pads`` is a tuple of three ``Pad`` for the components of the padded field; ``masks``
    a tuple of ``Mask``.
    """
    axis = 'xyz'.index(name[0])
    side = name[1]
    n = grid.n[axis]
    dx = grid.dx
    lo, hi = grid.low[axis], grid.high[axis]
    field = PADDED_FIELD[side]
    tangential = [c for c in range(3) if c != axis]
    layer = 0 if side == '-' else n - 1

    if isinstance(face, Periodic):
        src = n - 1 if side == '-' else 0
        return (FaceRule(name, 'periodic'),
                tuple(Pad('periodic', src) for _ in range(3)), ())
    if isinstance(face, Bloch):
        src = n - 1 if side == '-' else 0
        return (FaceRule(name, 'bloch'),
                tuple(Pad('bloch', src) for _ in range(3)), ())
    if isinstance(face, PEC):
        position = lo if side == '-' else hi
        masks = ()
        if side == '-':
            # Tangential E on the wall, and the normal H on it: n.B = 0 on a conductor, and
            # the update keeps it (its curl reads only the masked E), so a stray initial
            # value would sit there for the whole run otherwise.
            masks = (tuple(Mask('E', Slab.layer(c, axis, 0, grid.n)) for c in tangential)
                     + (Mask('H', Slab.layer(axis, axis, 0, grid.n)),))
        return (FaceRule(name, 'pec', position=position),
                tuple(Pad('zero') for _ in range(3)), masks)
    if isinstance(face, PMC):
        position = lo - dx / 2 if side == '-' else hi - dx / 2
        masks = ()
        if side == '+':
            # The dual: tangential H on the wall and the normal E on it.
            masks = (tuple(Mask('H', Slab.layer(c, axis, n - 1, grid.n)) for c in tangential)
                     + (Mask('E', Slab.layer(axis, axis, n - 1, grid.n)),))
        return (FaceRule(name, 'pmc', position=position),
                tuple(Pad('zero') for _ in range(3)), masks)
    if isinstance(face, Symmetry):
        p = face.parity
        if side == '-':
            plane, position = 'node', lo
            odd = [('E', c) for c in tangential] + [('H', axis)] if p == -1 else []
        else:
            plane, position = 'half', hi - dx / 2
            odd = [('E', axis)] + [('H', c) for c in tangential] if p == +1 else []
        if n < 2:
            raise ValueError(f'Symmetry on {name!r}: a mirror reads the layer next to the '
                             f'edge, so the axis needs at least two cells; it has {n}.')
        pads = tuple(_mirror_pad(field, c, axis, side, p, n) for c in range(3))
        masks = tuple(Mask(f, Slab.layer(c, axis, layer, grid.n)) for f, c in odd)
        return (FaceRule(name, 'symmetry', position=position, plane=plane, parity=p),
                pads, masks)
    raise TypeError(f'face {name!r}: {type(face).__name__} is not a stage-0 face type.')


def pad_neighbor(rt, a, axis, side, pad, factor):
    """The missing layer of single-component array ``a`` on ``side`` of ``axis`` (a name).

    ``factor`` is the Bloch factor for this face (already conjugated for a low face) and is
    read only when ``pad.kind == 'bloch'``. The returned layer has size 1 along the axis.
    """
    ax = AX[axis]
    if pad.kind == 'zero':
        return rt.zero_layer(a, ax)
    layer = rt.take_layer(a, ax, pad.layer)
    if pad.kind == 'periodic':
        return layer
    if pad.kind == 'bloch':
        return layer * factor
    if pad.kind == 'mirror':
        return layer if pad.sign == 1 else -layer
    raise ValueError(f'unknown pad kind {pad.kind!r}')
