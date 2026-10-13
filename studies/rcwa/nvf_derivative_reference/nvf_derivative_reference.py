"""Finite-difference references selected without consulting an AD derivative."""
import numpy as np


def central_reference(function, point, direction, step, *, atol=1e-9, rtol=1e-6, max_doublings=3):
    """Resolve two Richardson estimates within a bounded step window.

    Tiny steps amplify forward roundoff. Enlarge the window only when the
    independent references disagree, with a fixed upper limit. No derivative
    being tested participates in step selection. Failure remains explicit.
    The caller must ensure the whole bounded window is in the physical domain.
    """
    if (not np.isfinite(step) or step <= 0 or not isinstance(max_doublings, int)
            or max_doublings < 0 or not np.isfinite([atol, rtol]).all() or min(atol, rtol) < 0):
        raise ValueError('reference steps and tolerances must be finite, with a positive step')
    point, direction = np.asarray(point), np.asarray(direction)
    if point.shape != direction.shape or not (np.isfinite(point).all() and np.isfinite(direction).all()):
        raise ValueError('reference point and direction must be finite and have matching shapes')
    cache, history = {}, []
    for level in range(max_doublings+1):
        for exponent in (level, level-1, level-2):
            if exponent not in cache:
                h = step*2.**exponent
                plus, minus = np.asarray(function(point+h*direction)), np.asarray(function(point-h*direction))
                if not (np.isfinite(plus).all() and np.isfinite(minus).all()):
                    raise FloatingPointError('nonfinite central reference value')
                cache[exponent] = (plus-minus)/(2*h)
        first = (4*cache[level-1]-cache[level])/3
        second = (4*cache[level-2]-cache[level-1])/3
        error = np.abs(first-second)
        threshold = atol+rtol*np.maximum(np.abs(first), np.abs(second))
        history.append(dict(step=step*2.**level, references=np.array([first, second]).tolist(),
                            difference=error.tolist()))
        if np.all(error <= threshold):
            return dict(references=np.array([first, second]), step=step*2.**level, history=history)
    raise FloatingPointError(f'central references remain unresolved in the bounded step window: {history}')
