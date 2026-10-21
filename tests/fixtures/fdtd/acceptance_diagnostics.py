"""Distinguish missing growth evidence from a measured quiet energy floor."""
import numpy as np


def diagnostic_basis(diagnostics, record_end, period, growing_warning):
    """Return evidence type or ``None``; mirror documented diagnostic floor semantics.

    A None growth ratio has several causes. Only a complete, finite post-source
    window with a nonzero original signal and a verified quiet last quarter
    can substitute for a reported ratio. These are the existing engine's 4-period
    observation minimum and 1e-7 relative energy floor, not accuracy tolerances.
    """
    if not diagnostics.get('stable') or growing_warning:
        return None
    start = diagnostics.get('sources_end')
    trace = diagnostics.get('interior_energy')
    if start is None or not np.isfinite(start) or trace is None:
        return None
    t, energy = [np.asarray(v, float) for v in trace]
    if (t.ndim != 1 or energy.shape != t.shape or t.size < 2
            or not np.isfinite(t).all() or not np.isfinite(energy).all()
            or np.any(energy < 0) or np.any(np.diff(t) <= 0)
            or not record_end > start or not period > 0):
        return None
    ratio = diagnostics.get('interior_growth')
    if ratio is not None:
        return 'reported_growth' if np.isfinite(ratio) else None
    hi = float(t[-1])
    if hi-start < 4*period or not energy.max() > 0:
        return None
    quarter = (hi-start)/4
    edges = [hi-2*quarter+i*quarter/2 for i in range(4)] + [float('inf')]
    if any(not np.any((t >= a) & (t < b)) for a, b in zip(edges, edges[1:])):
        return None
    last = energy[t >= hi-quarter]
    return 'quiet_floor' if last.max() <= 1e-7*energy.max() else None
