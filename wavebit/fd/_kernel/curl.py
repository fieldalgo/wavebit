"""The discrete curl on the Yee grid, without the ``1/dx``.

``curl(F, 'E', ...)`` takes forward differences (the value one index up minus the value
here), which lands ``curl E`` on the H points; ``curl(F, 'H', ...)`` takes backward ones and
lands ``curl H`` on the E points. The missing neighbour at the last (forward) or first
(backward) index is the face's pad (``faces.py``), so the one function serves every
boundary type: the boundary is data, not a branch in the stencil.

The ``1/dx`` is left out and carried by the update coefficients (``bh = be = dt/dx`` in
vacuum). One uniform ``dx`` makes that a single scalar; folding it into the coefficients
saves a full-volume multiply per curl, and keeping the curl a pure difference is what stage
1's CPML stretch will scale, axis by axis, at this one place.

Component by component, with ``D_a`` the difference along axis ``a``::

    (curl F)_x = D_y F_z - D_z F_y
    (curl F)_y = D_z F_x - D_x F_z
    (curl F)_z = D_x F_y - D_y F_x

so component ``c`` is ``D_a F_b - D_b F_a`` with ``a = c+1``, ``b = c+2`` (mod 3). Each
difference is one call on the runtime; on NumPy the calls write into the curl scratch and
one temporary layer of it, so a step allocates nothing at full volume.
"""
from . import AXIS_NAMES, AX
from .faces import pad_neighbor

__all__ = ['curl']


def curl(F, field, plan, params, rt):
    """``curl F`` at the other field's points; ``field`` names what ``F`` is, ``'E'`` or ``'H'``."""
    side = '+' if field == 'E' else '-'
    out = rt.scratch('curl_' + field, F.shape, F.dtype)
    tmp = rt.scratch('diff_' + field, F.shape[1:], F.dtype)
    parts = []
    for c in range(3):
        a, b = (c + 1) % 3, (c + 2) % 3
        d1 = _diff(F, b, a, side, plan, params, rt, None if out is None else out[c])
        d2 = _diff(F, a, b, side, plan, params, rt, tmp)
        parts.append(rt.sub(d1, d2, out=None if out is None else out[c]))
    return rt.assemble(parts, out)


def _diff(F, comp, axis, side, plan, params, rt, out):
    """``D_axis F[comp]`` with the face's pad closing the stencil at the edge."""
    name = AXIS_NAMES[axis]
    face = name + side
    pad = plan.pad(face, comp)
    factor = None
    if pad.kind == 'bloch':
        f = params['bloch'][name]
        factor = f if side == '+' else rt.conj(f)
    layer = pad_neighbor(rt, F[comp], name, side, pad, factor)
    return rt.diff(F[comp], AX[name], side, layer, out=out)
