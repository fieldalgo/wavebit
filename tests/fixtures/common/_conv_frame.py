"""Independent test-side restatement of Fourier operators between the two origins."""
import numpy as np

from wavebit._common.backend import array_namespace, as_array, _device_of


def about_centre(matrix, order=None):
    """Apply the half-cell translation without using the production parity helper.

    With no order given, odd window widths make flat-index parity equivalent:
    the global sign of D cancels from D A D. That spelling also preserves the
    existing random-entry research objectives when their public inputs move.
    """
    if matrix is None:
        return None
    if order is None:
        odd = np.arange(matrix.shape[-1]) % 2 == 1
    else:
        u, v = order
        odd = np.array([(m+n) % 2 for n in range(-v, v+1)
                        for m in range(-u, u+1)], dtype=bool)
        assert matrix.shape[-1] % odd.size == 0
        odd = np.tile(odd, matrix.shape[-1] // odd.size)
    xp = array_namespace(matrix)
    pair = as_array(xp, odd[:, None] != odd[None, :], device=_device_of(matrix))
    return xp.where(pair, -matrix, matrix)


def slots_about_centre(slots, order=None):
    values = [about_centre(a, order) for a in slots]
    return type(slots)(*values) if hasattr(slots, '_fields') else tuple(values)
