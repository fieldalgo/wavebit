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

**Stage 1's CPML stretches a difference here**, between the bare difference and the
subtraction, which is the one place in the kernel where a single component's derivative
along a single axis exists as a value. ``psi`` -- the stretch's memory, one entry per
(face, differenced field, component) -- is threaded through because a recurrence cannot be
recomputed; ``cpml.py`` says what it is and where it lives. A scene with no PML face has an
empty ``plan.cpml``, so the loop over faces costs one iteration of nothing and a vacuum run
is the stage-0 arithmetic exactly.
"""
from wavebit.fdtd._kernel import AXIS_NAMES
from wavebit.fdtd._kernel import AX
from wavebit.fdtd._kernel.cpml import stretch
from wavebit.fdtd._kernel.faces import pad_neighbor

__all__ = ['curl']


def curl(F, field, plan, params, psi, rt):
    """``curl F`` at the other field's points; ``field`` names what ``F`` is, ``'E'`` or ``'H'``.

    Returns ``(curl, psi)``: the CPML advances its memory inside the differences, so the
    curl is where the new ``psi`` comes back from. With no PML face the same dict comes
    straight back.
    """
    side = '+' if field == 'E' else '-'
    out = rt.scratch('curl_' + field, F.shape, F.dtype)
    tmp = rt.scratch('diff_' + field, F.shape[1:], F.dtype)
    parts, updates = [], {}
    for c in range(3):
        a, b = (c + 1) % 3, (c + 2) % 3
        d1 = _diff(F, b, a, side, field, plan, params, psi, updates, rt,
                   None if out is None else out[c])
        d2 = _diff(F, a, b, side, field, plan, params, psi, updates, rt, tmp)
        parts.append(rt.sub(d1, d2, out=None if out is None else out[c]))
    return rt.assemble(parts, out), ({**psi, **updates} if updates else psi)


def _diff(F, comp, axis, side, field, plan, params, psi, updates, rt, out):
    """``D_axis F[comp]``, with the face's pad closing the stencil and the CPML stretching it."""
    name = AXIS_NAMES[axis]
    face = name + side
    pad = plan.pad(face, comp)
    if pad.kind == 'axis1d':
        # An Axis1D axis (one cell): the derivative is i k exactly, forward and backward alike,
        # so the "difference" is the product i k dx F (``faces.py``). No pad, no CPML.
        f = params['axis1d'][name]
        if out is not None:
            return rt.xp.multiply(F[comp], f, out=out)
        return F[comp] * f
    factor = None
    if pad.kind == 'bloch':
        f = params['bloch'][name]
        factor = f if side == '+' else rt.conj(f)
    layer = pad_neighbor(rt, F[comp], name, side, pad, factor)
    d = rt.diff(F[comp], AX[name], side, layer, out=out)
    if plan.cpml:
        d, advanced = stretch(d, field, comp, name, plan, params, psi, rt)
        updates.update(advanced)
    return d
