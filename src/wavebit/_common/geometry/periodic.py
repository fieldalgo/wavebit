"""Pure periodic coordinate operations."""

def fold_into_cell(xp, P, lattice):
    """``P`` (N, 2) carried by lattice vectors into the fundamental parallelogram of ``lattice``.

    ``floor`` of the fractional coordinates, not ``round``: this names the cell a point sits in,
    and the bounded sweep of :func:`periodic_segments` covers the candidate images. Its derivative is
    zero, which is correct -- the choice of image is locally constant, and moving a point does
    not move which cell it is in except on a measure-zero seam. A point already inside the
    parallelogram comes back with its coordinates untouched (``x - 0.0 * a`` is ``x``), so on a
    basis that is reduced to begin with the sample grid is the sample grid, to the bit.
    """
    a1x, a2x = lattice[0][0], lattice[0][1]
    a1y, a2y = lattice[1][0], lattice[1][1]
    det = a1x * a2y - a2x * a1y
    su = xp.floor((a2y * P[:, 0] - a2x * P[:, 1]) / det)
    sv = xp.floor((-a1y * P[:, 0] + a1x * P[:, 1]) / det)
    return xp.stack([P[:, 0] - su * a1x - sv * a2x, P[:, 1] - su * a1y - sv * a2y], axis=-1)
