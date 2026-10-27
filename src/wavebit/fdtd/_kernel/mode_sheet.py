"""The kernel term of a waveguide mode source: a one-sided TFSF plane fed by a mode.

A :class:`ModeSheetTerm` is the PlaneWave construction (``sources.py`` beside this one) with a
waveguide mode for its incident field. The stored field is ``F_scat + chi F_inc`` with ``chi``
the indicator of the total-field side; wherever a curl stencil straddles the sheet it must read
its neighbour in its own region, and the correction is ``+- sgn F_inc`` of that neighbour
(the same table the ``sources.py`` docstring writes out for a TFSF box). For a sheet on the
node plane ``k0`` across ``w``, with ``u = w+1`` and ``v = w+2``:

    travelling +w (total field at w >= w_k0, the box's low face):
        before_H, H half plane k0 - 1/2:   cE_u += +E_inc_v(k0)      cE_v += -E_inc_u(k0)
        before_E, E node plane k0:         cH_u += +H_inc_v(k0-1/2)  cH_v += -H_inc_u(k0-1/2)
    travelling -w (total field at w <= w_k0, the box's high face):
        before_H, H half plane k0 + 1/2:   cE_u += -E_inc_v(k0)      cE_v += +E_inc_u(k0)
        before_E, E node plane k0:         cH_u += -H_inc_v(k0+1/2)  cH_v += +H_inc_u(k0+1/2)

The incident field is separable: a static transverse pattern (the mode's E_t on the node plane,
its H_t on the half plane upstream, from ``src/wavebit/fdtd/_modes.py``, already in the kernel's
units and signed for the direction) times one waveform row each. Every pattern lives in
``params['sources'][name]['pattern']``, shaped like its slab.

**The rows.** E rows are read at ``n dt`` (the level ``curl E(n)`` uses), H rows at
``(n + 1/2) dt``. A mode source has one of each; a Gaussian beam's patterns are complex, and
each of its two fields has a cosine and a sine row (``_beam_source.py``). For the mode source:
the E row is read at ``n dt``, the H row at ``(n + 1/2) dt``. The H row is the E row's waveform with the carrier advanced by the mode's
phase over half a cell, ``beta dx/2``, and the envelope advanced by its group delay over half a
cell, ``dx / (2 v_g)``: the half plane is upstream of the node plane by half a cell. At the
pulse's centre frequency the pair is then the discrete mode itself, so the sheet launches it
one way only, exactly; elsewhere in the band the profile and the half-cell phase are the
centre's, and what the sheet injects is computed exactly by
:func:`wavebit.fdtd._modes.injected_amplitudes`. ``row_offsets`` carries the two rows'
``(phase, delay)`` offsets for :func:`wavebit.fdtd._kernel.sources.waveform_table`.
"""
import dataclasses

__all__ = ['ModeSheetTerm']


@dataclasses.dataclass(frozen=True)
class ModeSheetTerm:
    """One sheet: correction slabs, each fed by a pattern times one of the term's waveform rows.

    ``corrections`` is ``((slot, slab, label, sign, row), ...)``: ``slot`` ``'H'`` adds into
    ``cE`` before the H update, ``'E'`` into ``cH`` before the E update; ``label`` names the
    pattern in ``params['sources'][name]['pattern']`` and ``row`` the term's own row (an index
    into ``rows``). ``rows`` are the rows' time shifts in steps and ``row_offsets`` their
    ``(phase, delay)`` offsets.
    """

    name: str
    row: int
    corrections: tuple
    row_offsets: tuple
    rows: tuple = (0.0, 0.5)

    def init_mem(self, plan, rt):
        return {}

    def _add(self, out, slot, n, params, rt):
        pattern = params['sources'][self.name]['pattern']
        for kind, slab, label, sign, row in self.corrections:
            if kind == slot:
                w = params['waveform'][self.row + row, n]
                out = rt.slab_add(out, slab, (w * sign) * pattern[label])
        return out

    def before_H(self, cE, E, mem, n, params, rt):
        return self._add(cE, 'H', n, params, rt)

    def before_E(self, cH, H, mem, n, params, rt):
        return self._add(cH, 'E', n, params, rt)

    def advance(self, mem, E, H, n, params, rt):
        return mem
