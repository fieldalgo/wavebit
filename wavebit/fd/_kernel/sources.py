"""The source terms: what a Dipole, a PlaneWave and a TFSF box are inside the kernel step.

Every sign and half step below is read off the kernel's own update (``step.py``), with ``c = 1``,
``eps0 = mu0 = 1`` and H stored as ``Z0 H``:

    H(n+1/2) = ah H(n-1/2) - bh * cE,   cE = curl E(n)     + [before_H terms]
    E(n+1)   = ae E(n)     + be * cH,   cH = curl H(n+1/2) + [before_E terms]

with ``bh = dt/(dx mu)`` and ``be = dt/(dx eps)`` in a lossless medium -- the curl is the bare
difference and the ``1/dx`` lives in the coefficient.

**The waveform table.** Every source reads its time signal from one table ``W`` of shape
``(rows, steps)``, built once per run *inside* the traced function (``waveform_table``) from
the traced amplitude, phase, delay and band leaves, so all of them differentiate. A term owns
one or more rows, each sampled at ``t = (n + shift) dt`` with the shift at which its slot
reads it: a ``before_E`` term acts across ``E(n) -> E(n+1)``, so it is sampled at the half
step ``(n + 1/2) dt``; a ``before_H`` term acts across ``H(n-1/2) -> H(n+1/2)``, centred on
``n dt``; a value imposed on ``E(n+1)`` is sampled at ``(n + 1) dt``. The carrier is real;
in a Bloch (complex) run the table stays real in time unless the amplitude is complex, and
the spatial Bloch factor carries the phase.

**Dipole.** Ampere's law with a current, ``eps dE/dt = curl H - J``, adds ``-dt J/eps`` to the
E update, which is ``be * (-dx J)``; with ``J = w(t) S / dx^3`` (``S`` the stamp, summing to
1, so ``sum J dV = w``) the ``before_E`` slot adds ``-w S / dx^2`` into ``cH`` over the stamp's
slab, and the table row is sampled at ``(n + 1/2) dt``. The magnetic dual,
``mu dH/dt = -curl E - M``, adds ``-dt M/mu = -bh (dx M)``, so ``before_H`` adds ``+w S /
dx^2`` into ``cE`` (the H update *subtracts* ``bh cE``), sampled at ``n dt``. The duality
``E -> H, H -> -E, J -> M, M -> -J`` maps one onto the other exactly, which is what the
duality gate checks.

*The stamp*, measured (``studies/fdtd_dipole_stamp.py``): the tent and the cubic B-spline,
both half-width 2 cells, both with exact sums and centroids; a 16 cells-per-wavelength ``Ez``
dipole in an open vacuum box, its radiation resistance at the band centre (from its own drive
and field series) as ``center`` moves by 0, 1/4 and 1/2 cell along the diagonal. The
position spread is **1.2e-4 for the B-spline against 2.4e-2 for the tent** (and the
B-spline sits at -3.5% of the continuum ``k^2/(6 pi)`` against the tent's -8.8%), so
``fd.Dipole`` defaults to ``stamp='bspline'`` (:data:`STAMP_PROBE`).

**The work series.** ``res.sources[name].work`` is the power the source delivers to the
field, ``-sum J.E dV = -w sum S E`` (Poynting's theorem: ``d(energy)/dt = -int J.E``), with
``E`` at the time of the current -- the mean of the two E levels it straddles, ``E(n)`` and
``E(n+1)`` for ``J(n+1/2)`` (the H levels ``n -+ 1/2`` for ``M(n)``). ``drive`` is ``w`` at
its own time and ``field`` the stamp-weighted field ``sum S F`` at the field's own time.

**PlaneWave and TFSF are one construction.** The stored field is ``F = F_scat + chi F_inc``
with ``chi`` the total-field indicator of a region; wherever a curl stencil straddles the
region's surface, the difference must read its neighbour in the target's own region, and
the correction is ``+- sgn * F_inc`` of that neighbour. Written out for the box ``[lo, hi]``
of node indices on each axis, component ``c`` of a curl being ``D_a F_b - ...`` with sign
``sgn`` for the term whose derivative is along ``a`` and whose source component is ``b`` (and
``a`` is then the third axis, so the target and source components sit at the same position
on the two other axes -- the correction slabs are rectangles):

    before_H, into cE[c] (H targets, forward differences, E_inc[b] read at E(n)):
        at a = lo - 1:  -sgn * E_inc[b](a = lo)      at a = hi:  +sgn * E_inc[b](a = hi)
    before_E, into cH[c] (E targets, backward differences, H_inc[b] read at H(n+1/2)):
        at a = lo:      -sgn * H_inc[b](a = lo - 1)  at a = hi:  +sgn * H_inc[b](a = hi)

with the slab spanning, on axis ``c``, the target's own extent inside the box (``[lo, hi]`` of
nodes for an H target, ``[lo, hi)`` of half cells for an E target) and on axis ``b`` the other
one. A TFSF box is all six faces of that; a ``PlaneWave`` is the one face across its
propagation axis with the transverse box unbounded -- a one-sided TFSF plane, which is the
Huygens sheet: the ``before_E`` correction is an electric current on the E plane and the
``before_H`` one a magnetic current on the H plane half a cell upstream, and their ratio is
the discrete one because both come from the same discrete solution.

**The incident field is a column.** ``F_inc`` must be an exact discrete solution for the
corrections to cancel at round-off, and for a laterally uniform medium the full grid's
solution is ``f(P) exp(i k_par . r)``, which a 1 x 1 transverse column with the same Bloch
phase per cell reproduces exactly (the same discrete transverse derivative, ``(exp(i phi) -
1)/dx``; on a stage-3 ``Axis1D`` axis the column carries the scene's own ``Axis1D`` face, so
the derivative there is the same exact ``i k``). So each ``PlaneWave`` and ``TFSF`` term carries a column in its memory -- a
sub-plan along the propagation axis stepped by the kernel's own curl, update, masks, CPML
and pole terms -- driven by a hard source one cell upstream of the surface, so every value
the corrections read lies in the column's forward region. The hard source fixes the tangential
E, and the column radiates whatever discrete waves carry it; a PlaneWave's drive is therefore
the grid's own s or p wave -- transverse to the discrete ``K_par`` and with each stored
component's stagger phase -- or the column launches both
(``_lower/sources.py::plane_wave_drive``). For ``TFSF`` the column has the scene's own (absorbing) faces on that axis and every structure that is a slab across it (the
layered background). For ``PlaneWave`` it has the uniform background, so waves reflected
back through the sheet by the scene's layers pass it as scattered field, and its two ends
absorb whatever the scene's do: a PEC, PMC, Symmetry or periodic face along the axis is
replaced in the column by a PML on cells padded past the scene's
(``wavebit/fd/_lower/sources.py::column_pads``), because a column that reflects is a cavity
behind a hard source. The column's field is the reference series each returns. At a
concrete ``theta = 0`` the discrete impedance is exactly 1 (``Z0 H = -E`` at the staggered
points), so the column is not needed to find it there -- but the sheet is not a closed form
at any other angle, because with a fixed ``k_par`` the ratio depends on frequency, and the
column is what carries it at every frequency at once.

**A layered TFSF column is fed through a sheet, not driven by a hard source.** A hard source
is a perfect mirror (``r = -1``) for a wave coming back up the column, and over a layered
background the layers' reflection does come back: the hard-driven column was a cavity
between the source and the interface, and the box injected that cavity as "incident" -- off
the layered background's own (Airy) solution by ``O(|r|)``, with the leak still at round-off
because the box is consistent with whatever its column holds. So a TFSF column with
structures carries a second column, the **driver**: uniform in the medium the wave arrives
from, with the scene's faces, hard-driven as above -- it has no layer to echo from. The
layered column is fed from it through a one-sided sheet on the box's entry plane (the
``PlaneWave`` construction, ``sheet``), so upstream of the entry plane it holds the scattered
field only and the reflection passes the sheet and dies in the column's CPML. The one value
the box reads upstream of the entry plane -- the tangential H half a cell outside it -- is the
layered column's (the reflection) plus the driver's (the incident); that is the
``Correction.driver`` flag. The hand-over is exact when the layered column is the driver's
medium in every cell upstream of the entry plane (the sheet then launches the same wave into
it that it launches into the driver, and the layers are downstream contrast, which is the
physics); ``wavebit/fd/_lower/sources.py::_tfsf_driver`` refuses any other incident side.
With no structure in the column the driver would be the column itself, and the single
hard-driven column is kept, so a vacuum TFSF is what it was to the bit.

**The half-step shift.** ``before_H`` of step ``n`` needs the column's ``E(n)`` and
``before_E`` needs its ``H(n+1/2)``, but a term's memory changes only in ``advance``, once per
step and after the E update. So the column runs half a step ahead: at the start of step ``n``
its memory holds ``E(n)`` and ``H(n+1/2)``, and ``advance`` does the E half of step ``n`` (to
``E(n+1)``, then the hard source) followed by the H half of step ``n+1``. It starts from rest
(all zeros), which satisfies the invariant because the first H half acts on a zero field.
A driver runs on the same shift, and the layered column reads it at the level the main grid
would: its E half (``before_E``) reads the driver's ``H(n+1/2)``, before the driver's H half
moves it on, and its H half (``before_H``) the driver's ``E(n+1)``.
"""
import dataclasses
import math

from . import AX
from .curl import curl
from .grid import OFFSETS

__all__ = ['DipoleTerm', 'ColumnTerm', 'Correction', 'waveform_table', 'has_sources',
           'STAMP_PROBE']

#: The stamp measurement, 2026-09-19 (``studies/fdtd_dipole_stamp.py``): the spread of the
#: radiation resistance over three sub-cell positions at f/f0 = 0.8 / 1.0 / 1.2.
STAMP_PROBE = {'bspline': (3.2e-4, 1.2e-4, 2.0e-4), 'tent': (1.5e-2, 2.4e-2, 3.5e-2)}


def has_sources(plan):
    """Whether any term of ``plan`` reads the waveform table (a static question)."""
    return any(getattr(t, 'rows', ()) for t in plan.terms)


def waveform_table(plan, params, rt):
    """``W``, ``(rows, steps)``: every source term's waveform rows, from traced leaves.

    ``w(t) = A exp(-s^2 / (2 tau^2)) sin(2 pi f0 s + phi)``, ``s = t - t0 - delay``, sampled at
    ``t = (n + shift) dt`` for each row's shift (see the module docstring).
    """
    import numpy as np
    n = rt.array(np.arange(plan.steps, dtype=np.float64), 'float64')
    rows = []
    for term in plan.terms:
        for shift in getattr(term, 'rows', ()):
            p = params['sources'][term.name]
            s = (n + shift) * plan.dt - p['t0'] - p['delay']
            envelope = rt.xp.exp(-0.5 * (s / p['tau']) ** 2)
            w = envelope * rt.xp.sin((2.0 * math.pi) * p['f0'] * s + p['phase'])
            rows.append(p['amplitude'] * w)
    if plan.dtype == 'complex128':
        one = rt.array(1.0 + 0.0j, 'complex128')
        rows = [r * one for r in rows]
    return rt.xp.stack(rows, axis=0)


# ---------------------------------------------------------------------------------------------
# Dipole
# ---------------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class DipoleTerm:
    """One point current: its field (``'E'`` for J, ``'H'`` for M), its slab, its table row.

    Static: the slab is the support the ``range`` (or the concrete centre) fixes; the stamp's
    weights are traced (they depend on the position) and live in
    ``params['sources'][name]['stamp']``, shaped like the slab.
    """

    name: str
    field: str
    slab: object            # masks.Slab of the driven component
    row: int
    dx: float
    stamp: str

    @property
    def rows(self):
        return (0.5,) if self.field == 'E' else (0.0,)

    def init_mem(self, plan, rt):
        return {'drive': rt.zeros((plan.steps,), plan.dtype),
                'field': rt.zeros((plan.steps,), plan.dtype),
                'work': rt.zeros((plan.steps,), plan.dtype),
                'last': rt.zeros((), plan.dtype)}

    def _value(self, n, params, sign):
        w = params['waveform'][self.row, n]
        return (w * (sign / self.dx ** 2)) * params['sources'][self.name]['stamp']

    def before_H(self, cE, E, mem, n, params, rt):
        if self.field == 'H':
            cE = rt.slab_add(cE, self.slab, self._value(n, params, 1.0))
        return cE

    def before_E(self, cH, H, mem, n, params, rt):
        if self.field == 'E':
            cH = rt.slab_add(cH, self.slab, self._value(n, params, -1.0))
        return cH

    def advance(self, mem, E, H, n, params, rt):
        F = E if self.field == 'E' else H
        region = F[self.slab.slices]
        field = rt.xp.sum(region * params['sources'][self.name]['stamp'])
        w = params['waveform'][self.row, n]
        work = -w * (0.5 * (mem['last'] + field))
        return {'drive': rt.record(mem['drive'], n, w),
                'field': rt.record(mem['field'], n, field),
                'work': rt.record(mem['work'], n, work),
                'last': field}

    def series(self, plan, mem, rt):
        import numpy as np
        from ..results import SourceSeries
        steps = np.arange(plan.steps)
        if self.field == 'E':
            t, t_field = (steps + 0.5) * plan.dt, (steps + 1.0) * plan.dt
        else:
            t, t_field = steps * plan.dt, (steps + 0.5) * plan.dt
        return SourceSeries(name=self.name, kind=f'Dipole({self.field})', t=t,
                            drive=mem['drive'], work=mem['work'], t_field=t_field,
                            field=mem['field'])


def stamp_weights(kind, u, xp):
    """The 1-D stamp at offsets ``u`` (in cells) from the centre; each sums to 1 on a lattice.

    ``bspline``: the cubic B-spline, ``2/3 - u^2 + |u|^3/2`` for ``|u| < 1`` and ``(2 - |u|)^3/6``
    for ``1 <= |u| < 2``. ``tent``: ``(1 - |u|/2)/2`` for ``|u| < 2``. Both are partitions of
    unity and reproduce linear functions, so the moment and the centroid are exact.
    """
    a = xp.abs(u)
    if kind == 'bspline':
        inner = 2.0 / 3.0 - a ** 2 + 0.5 * a ** 3
        outer = (2.0 - a) ** 3 / 6.0
        return xp.where(a < 1.0, inner, xp.where(a < 2.0, outer, 0.0 * a))
    return xp.where(a < 2.0, 0.5 * (1.0 - 0.5 * a), 0.0 * a)


def stamp(term, center, grid, rt):
    """The stamp array of ``term`` at ``center`` (traced or not), shaped like its slab."""
    import numpy as np
    field = term.field
    c = term.slab.c
    ranges = {0: term.slab.x, 1: term.slab.y, 2: term.slab.z}
    parts = []
    for a in range(3):
        lo, hi = ranges[a]
        offset = 0.5 * OFFSETS[field][c][a]
        position = grid.low[a] + (np.arange(lo, hi, dtype=np.float64) + offset) * grid.dx
        u = (rt.array(position, 'float64') - center[a]) / grid.dx
        parts.append(stamp_weights(term.stamp, u, rt.xp))
    wx, wy, wz = parts
    return (wz[:, None, None] * wy[None, :, None]) * wx[None, None, :]


# ---------------------------------------------------------------------------------------------
# PlaneWave and TFSF: the correction surface and the column behind it
# ---------------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Correction:
    """One rectangle of the TFSF surface: ``slab`` of the target, fed from the column.

    ``slot`` is ``'H'`` (``before_H``, into ``cE``, reading the column's E) or ``'E'``
    (``before_E``, into ``cH``, reading the column's H). ``src`` is the source component and
    ``p0:p1`` the column's range along the propagation axis the value is read over -- one
    index when the slab is a plane across that axis, the slab's own range when it runs along
    it. ``sign`` is ``+-1`` and ``pattern`` says whether the transverse Bloch phase
    multiplies it (a PlaneWave in a Bloch cell). ``driver`` says the read lies upstream of a
    layered column's sheet, where the incident field is the column plus its driver.
    """

    slot: str
    slab: object
    src: int
    p0: int
    p1: int
    sign: float
    pattern: bool = False
    driver: bool = False


class _Scoped:
    """The runtime, with the NumPy scratch keys of a column kept apart from the main grid's.

    The curl and the CPML name their scratch buffers by field and psi key; a column shares
    those names with the main grid at a different shape, and NumPy would reallocate the
    buffer on every call if the two were not kept apart.
    """

    def __init__(self, rt, prefix):
        self._rt, self._prefix = rt, prefix

    def __getattr__(self, name):
        return getattr(self._rt, name)

    def scratch(self, key, shape, dtype):
        return self._rt.scratch(self._prefix + key, shape, dtype)

    def cpml_stretch(self, d, region, psi, b, a, ik, key):
        return self._rt.cpml_stretch(d, region, psi, b, a, ik, self._prefix + key)


@dataclasses.dataclass(frozen=True)
class ColumnTerm:
    """A PlaneWave sheet or a TFSF box: its correction rectangles, and its incident column.

    ``sub`` is the column's own plan (1 x 1 across, the scene's cells along ``axis``);
    ``drive`` is ``((component, weight), ...)`` -- the tangential E the hard source imposes
    at column index ``ks``; a PlaneWave's weights are the grid's own s or p wave, each with
    its component's half-cell stagger phase, so complex off normal incidence
    (``wavebit/fd/_lower/sources.py::plane_wave_drive`` derives them) -- and ``entry_E``/``entry_H`` the column indices the reference
    series is read at: ``entry_E`` is the node plane where the total field begins (the
    tangential E there) and ``entry_H`` the index of the half plane just inside it (the
    tangential H, and the normal E; the normal H sits on that index's node). ``kind`` is
    ``'PlaneWave'`` or ``'TFSF'``. Every index along the axis is the column's own: the main
    grid's plus the column's low padding (``wavebit/fd/_lower/sources.py::column_pads``).

    ``driver`` is ``None`` for a column hard-driven itself. For a layered TFSF column it is
    the uniform driver's plan, hard-driven at ``ks``, and ``sheet`` the one-sided corrections
    (column slabs, read off the driver) that feed the column from it.
    """

    name: str
    kind: str
    row: int
    axis: int
    sub: object
    drive: tuple
    ks: int
    corrections: tuple
    entry_E: int
    entry_H: int
    plane_E: float
    plane_H: float
    driver: object = None
    sheet: tuple = ()

    rows = (1.0,)

    def _point(self, index):
        """The column index tuple of the node ``index`` along the axis (transverse 0)."""
        idx = [0, 0, 0]
        idx[AX['xyz'[self.axis]]] = index
        return tuple(idx)

    def _range(self, p0, p1):
        idx = [slice(0, 1)] * 3
        idx[AX['xyz'[self.axis]]] = slice(p0, p1)
        return tuple(idx)

    def _drive_slab(self, c):
        from .masks import Slab
        spans = {0: (0, 1), 1: (0, 1), 2: (0, 1)}
        spans[self.axis] = (self.ks, self.ks + 1)
        return Slab(c=c, z=spans[2], y=spans[1], x=spans[0])

    @staticmethod
    def _column_state(sub, rt):
        return {'E': rt.zeros(sub.shape, sub.dtype), 'H': rt.zeros(sub.shape, sub.dtype),
                'psi': {key: rt.zeros(face.region.shape, sub.dtype)
                        for face in sub.cpml for key in face.keys()},
                'mem': {t.name: t.init_mem(sub, rt) for t in sub.terms}}

    def init_mem(self, plan, rt):
        sub = self.sub
        out = {'col': self._column_state(sub, rt), 'E': rt.zeros((plan.steps, 3), sub.dtype),
               'H': rt.zeros((plan.steps, 3), sub.dtype)}
        if self.driver is not None:
            out['drv'] = self._column_state(self.driver, rt)
        return out

    def _read(self, source, corr):
        return source[(corr.src,) + self._range(corr.p0, corr.p1)]

    def _add(self, target, slot, mem, params, rt):
        which = 'E' if slot == 'H' else 'H'
        source = mem['col'][which]
        own = params['sources'][self.name]
        for corr in self.corrections:
            if corr.slot != slot:
                continue
            v = self._read(source, corr)
            if corr.driver:
                v = v + self._read(mem['drv'][which], corr)
            if corr.sign != 1.0:
                v = v * corr.sign
            if corr.pattern:
                v = v * own['pattern']
            v = rt.xp.broadcast_to(v, corr.slab.shape)
            target = rt.slab_add(target, corr.slab, v)
        return target

    def _feed(self, target, slot, source, rt):
        """The sheet's corrections into the layered column, read off the driver's ``source``."""
        for corr in self.sheet:
            if corr.slot != slot:
                continue
            v = self._read(source, corr)
            if corr.sign != 1.0:
                v = v * corr.sign
            target = rt.slab_add(target, corr.slab, rt.xp.broadcast_to(v, corr.slab.shape))
        return target

    def before_H(self, cE, E, mem, n, params, rt):
        return self._add(cE, 'H', mem, params, rt)

    def before_E(self, cH, H, mem, n, params, rt):
        return self._add(cH, 'E', mem, params, rt)

    def _hard(self, Ec, w, crt):
        """The hard source at ``ks``: the tangential E there set to ``w * weight``."""
        for c, weight in self.drive:
            slab = self._drive_slab(c)
            Ec = crt.slab_mul(Ec, slab, 0.0)
            Ec = crt.slab_add(Ec, slab, w * weight)
        return Ec

    def advance(self, mem, E, H, n, params, rt):
        sub = self.sub
        own = params['sources'][self.name]
        sp = own['column']
        crt = _Scoped(rt, f'col[{self.name}]:')
        col = mem['col']
        w = params['waveform'][self.row, n]
        feed_H = None
        if self.driver is None:
            # The E half of step n: E(n) -> E(n+1), with the hard source at (n+1) dt.
            Ec, psi = _half_E(sub, sp, col, n, crt)
            Ec = self._hard(Ec, w, crt)
        else:
            # The driver's E half and hard source, then the layered column's E half, fed
            # from the driver's H(n+1/2) -- read before the driver's H half moves it on.
            dsub, dp, drv = self.driver, own['driver'], mem['drv']
            drt = _Scoped(rt, f'drv[{self.name}]:')
            Ed, dpsi = _half_E(dsub, dp, drv, n, drt)
            Ed = self._hard(Ed, w, drt)
            dmem = {t.name: t.advance(drv['mem'][t.name], Ed, drv['H'], n, dp, drt)
                    for t in dsub.terms}
            Ec, psi = _half_E(sub, sp, col, n, crt,
                              lambda cH: self._feed(cH, 'E', drv['H'], crt))

            def feed_H(cE):
                return self._feed(cE, 'H', Ed, crt)
        Hc, cmem = col['H'], col['mem']
        cmem = {t.name: t.advance(cmem[t.name], Ec, Hc, n, sp, crt) for t in sub.terms}
        # Every component is read inside the total-field region: the tangential E on the
        # entry node plane, the normal E on the half cell next to it on the inside -- which
        # is the half plane ``entry_H`` indexes, whichever way the wave goes.
        rec_E = rt.record(mem['E'], n, rt.xp.stack(
            [Ec[(c,) + self._point(self.entry_H if c == self.axis else self.entry_E)]
             for c in range(3)]))
        rec_H = rt.record(mem['H'], n, Hc[(slice(None),) + self._point(self.entry_H)])
        if self.driver is not None:
            Hd, dpsi = _half_H(dsub, dp, Ed, drv['H'], dpsi, dmem, n, drt)
            driven = {'E': Ed, 'H': Hd, 'psi': dpsi, 'mem': dmem}
        # The H half of step n+1: H(n+1/2) -> H(n+3/2), fed from the driver's E(n+1).
        Hc, psi = _half_H(sub, sp, Ec, Hc, psi, cmem, n, crt, feed_H)
        # The keys in init_mem's order: Torch's checkpointed scan rebuilds the carry from a
        # flat list in the order the first carry had.
        out = {'col': {'E': Ec, 'H': Hc, 'psi': psi, 'mem': cmem}, 'E': rec_E, 'H': rec_H}
        if self.driver is not None:
            out['drv'] = driven
        return out

    def series(self, plan, mem, rt):
        import numpy as np
        from ..results import SourceSeries
        steps = np.arange(plan.steps)
        return SourceSeries(name=self.name, kind=self.kind, t=(steps + 1.0) * plan.dt,
                            incident=mem['E'], t_H=(steps + 0.5) * plan.dt,
                            incident_H=mem['H'], plane=self.plane_E, plane_H=self.plane_H)


def _half_E(sub, sp, col, n, crt, feed=None):
    """The E half of step ``n`` of column ``col``: ``E(n) -> E(n+1)``, before any hard source."""
    Ec, Hc, cmem = col['E'], col['H'], col['mem']
    cH, psi = curl(Hc, 'H', sub, sp, col['psi'], crt)
    for t in sub.terms:
        cH = t.before_E(cH, Hc, cmem[t.name], n, sp, crt)
    if feed is not None:
        cH = feed(cH)
    Ec = crt.update(Ec, cH, sp['ae'], sp['be'])
    for m in sub.masks:
        Ec = m.after_E(Ec, crt)
    return Ec, psi


def _half_H(sub, sp, Ec, Hc, psi, cmem, n, crt, feed=None):
    """The H half of step ``n + 1`` of a column: ``H(n+1/2) -> H(n+3/2)``."""
    cE, psi = curl(Ec, 'E', sub, sp, psi, crt)
    for t in sub.terms:
        cE = t.before_H(cE, Ec, cmem[t.name], n, sp, crt)
    if feed is not None:
        cE = feed(cE)
    Hc = crt.update(Hc, cE, sp['ah'], sp['minus_bh'])
    for m in sub.masks:
        Hc = m.after_H(Hc, crt)
    return Hc, psi


def box_corrections(lo, hi, e_src, h_src, n, pattern=False, faces=None):
    """The correction rectangles of the box ``[lo, hi]`` (node indices per axis, x y z).

    ``e_src`` / ``h_src`` are the incident E / H components that can be nonzero; ``faces``
    restricts to ``((axis, side), ...)`` -- a PlaneWave is one face with the transverse box
    unbounded (``lo = 0``, ``hi = n`` on those axes, which also makes the slab full-width).
    The column range ``p0:p1`` is filled in by the caller's ``column_range``.
    """
    from .masks import Slab
    out = []
    for c in range(3):
        for src, a, sgn in ((( c + 2) % 3, (c + 1) % 3, 1.0), ((c + 1) % 3, (c + 2) % 3, -1.0)):
            for slot, sources in (('H', e_src), ('E', h_src)):
                if src not in sources:
                    continue
                for side in ('-', '+'):
                    if faces is not None and (a, side) not in faces:
                        continue
                    spans = [None, None, None]
                    if slot == 'H':
                        at = lo[a] - 1 if side == '-' else hi[a]
                        read = lo[a] if side == '-' else hi[a]
                        spans[c] = (lo[c], min(hi[c] + 1, n[c]))       # nodes along c
                        spans[src] = (lo[src], hi[src])                # half cells along src
                    else:
                        at = lo[a] if side == '-' else hi[a]
                        read = lo[a] - 1 if side == '-' else hi[a]
                        spans[c] = (lo[c], hi[c])                      # half cells along c
                        spans[src] = (lo[src], min(hi[src] + 1, n[src]))   # nodes along src
                    spans[a] = (at, at + 1)
                    sign = -sgn if side == '-' else sgn
                    slab = Slab(c=c, z=spans[2], y=spans[1], x=spans[0])
                    out.append((slot, slab, src, a, read, sign))
    return out
