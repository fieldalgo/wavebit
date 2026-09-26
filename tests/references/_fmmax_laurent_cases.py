"""Cases shared by ``gen_goldens_fmmax_laurent.py`` and ``test_fmmax_laurent_golden.py``.

Same split as ``_fmmax_cases.py``: the generator runs in ``fwref`` (fmmax, no wavebit) and the
test in the project env (the reverse), so what they share imports numpy and nothing else.

**What this oracle is for.** ``factorization='laurent'`` closes every product with the plain
convolution matrix ``[[eps]]`` and the z slot with ``[[eps]]^-1``; fmmax's ``Formulation.FFT``
builds exactly that (``fmmax.fmm_matrices.transverse_permittivity_fft`` for the transverse
block, ``script_k_matrix_patterned`` solving against ``[[eps]]`` for the z slot). So the two
solvers assemble the same operator, and the comparison can be one of identity rather than of
convergence -- once the one difference that remains is removed.

**That difference is the coefficients, and it has a closed form.** A raster here is a staircase
and its Fourier coefficients are the exact integral over each pixel. fmmax takes the DFT of its
array at pixel-centred coordinates, which is the midpoint rule for the same integral. Hand fmmax
each pixel as ``R x R`` equal samples and, per axis, the midpoint rule over one pixel of
``exp(-2 pi i k x)`` is exact up to the factor::

    y / sin(y),   y = pi k / (N R)        (N pixels per axis, order k)
                = 1 + y^2/6 + 7 y^4/360 + 31 y^6/15120 + ...

-- real, no phase, and **even in 1/R**. So fmmax's answer at ``R`` is Wavebit's plus a series
in ``1/R^2``, and Richardson extrapolation over ``R, 2R, 4R`` removes it through ``1/R^4``. The
test checks the premise before it uses it: the raw gap has to fall fourfold per doubling of
``R``, which is what says nothing but the quadrature separates the two.

**Transpose, and not square.** This project stores a layer ``A[v, u]``; fmmax's axis 0 runs
along its first lattice vector, so the same layer is ``A.T`` to it -- ``_fmmax_patterned_cases.py``
records the s/p split coming out reversed without it. Every feature below is off-centre and not
square, and the incidence is oblique, so a transposed, mirrored or order-flipped layer changes
the answer; the test measures the mirrored order index and requires it to miss.

**The order set** is ``|m|, |n| <= ORDER`` -- Wavebit's ``truncation_order=(ORDER, ORDER)`` --
handed to fmmax as an explicit ``Expansion``. Its own parallelogramic truncation balances the two
sides in k-space, which on the 700 x 560 cell keeps 17 x 13 orders instead of 15 x 15.
"""
import numpy as np

WL = 500.0            # short enough that four or five orders propagate, asymmetrically
THETA = 11.0          # degrees, in the plane of the first lattice vector (phi = 0)
THICK = 240.0
ORDER = 7
PIXELS = 16           # per axis, the raster Wavebit is given
LEVELS = (16, 32, 64, 128)   # fmmax samples per Wavebit pixel, per axis


def _rod(eps_rod, eps_bg, rows, cols):
    a = np.full((PIXELS, PIXELS), eps_bg, dtype=complex)
    a[rows[0]:rows[1], cols[0]:cols[1]] = eps_rod
    return a


def _cross(eps_in, eps_bg):
    a = np.full((PIXELS, PIXELS), eps_bg, dtype=complex)
    a[3:13, 5:8] = eps_in          # a tall arm
    a[6:9, 2:14] = eps_in          # a wide arm, off the tall one's centre
    return a


#: name -> (lattice columns (a1, a2) in nm, raster [v, u], top eps, bottom eps).
#: A rectangular cell with a silicon-like rod; a lossy metal film with a hole (Laurent's worst
#: case, where the in-plane field jumps hardest); and a hexagonal cell, where Li's rules are
#: refused and Laurent's -- which names no axis -- is the rule a comparison can use.
CASES = {
    'rod_eps12_rect': (np.array([[700.0, 0.0], [0.0, 560.0]]),
                       _rod(12.25, 1.0, (4, 11), (2, 12)), 1.0, 1.0),
    'hole_metal_rect': (np.array([[600.0, 0.0], [0.0, 600.0]]),
                        _rod(1.0, -16.49 + 1.06j, (5, 12), (3, 9)), 1.0, 1.0),
    'cross_eps6_hex': (np.array([[640.0, 320.0], [0.0, 320.0 * np.sqrt(3.0)]]),
                       _cross(6.0, 1.0), 1.0, 1.0),
}


def lossless(name):
    """Whether the case conserves energy -- every material real."""
    _, raster, top, bot = CASES[name]
    return bool(np.all(np.isreal(raster)) and np.isreal(top) and np.isreal(bot))


def upsampled(raster, samples):
    """The raster with every pixel repeated ``samples x samples`` times -- the same staircase."""
    return np.repeat(np.repeat(raster, samples, axis=0), samples, axis=1)


def richardson(values, levels):
    """Extrapolate ``values`` sampled at ``levels`` (samples per pixel, each double the last)
    to infinitely many, removing the ``1/R^2`` and ``1/R^4`` terms. Three levels in, one out.
    """
    a, b, c = values
    if not (levels[1] == 2 * levels[0] and levels[2] == 2 * levels[1]):
        raise ValueError(f'levels must double: {levels}')
    return (64.0 * c - 20.0 * b + a) / 45.0
