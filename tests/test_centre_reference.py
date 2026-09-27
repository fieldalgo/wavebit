"""One lateral phase reference for every per-order quantity: the cell centre, ``x = y = 0``.

Every public coordinate is measured from the centre of the cell (2026-08-31). The kernels still
integrate from the corner -- a convolution matrix is a set of Fourier coefficients about it --
and the corner is also where every per-order phase used to be referenced. Moving a plane wave's
reference from the corner to the centre ``o = (a1 + a2)/2`` multiplies order ``(m, n)`` by
``exp(1j * k_mn . o) = exp(1j * k_inc . o) * (-1)**(m + n)``, on any cell, because
``b_i . a_j = 2 pi delta_ij``. The scalar is a normalisation; the sign is what an order carries.

1. **the rule** -- ``conventions.centre_parity`` is that sign, by order label, on a rectangle, a
   60-degree cell and a shear, windows moved by an excitation included; ``corner_column`` is an
   exact sign bit and leaves an undeclared solve alone.
"""
import numpy as np
import pytest

import wavebit
from wavebit._core.conventions import (centre_parity, corner_column, incident_column,
                                       lit_orders, order_window)
from wavebit._core.solve_2d import order_grid_kxy

_HEX = np.column_stack(([1.6, 0.0], [0.8, 1.6 * np.sqrt(3) / 2]))
_SHEAR = np.array([[1.6, 0.5], [0.0, 1.7]])
_CELLS = {'rectangle': dict(period=(1.3, 1.7)),
          'hexagonal': dict(lattice=_HEX, factorization='nvf'),
          'shear': dict(lattice=_SHEAR, factorization='nvf')}


def _host(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu()
    return np.asarray(value)


# ------------------------------------------------------------------------------ 1. the rule

@pytest.mark.parametrize('excitation', [None, {(1, 0): 1.0, (0, 1): 0.5j},
                                        {(2, 1): 1.0, (1, 1): 0.3}],
                         ids=['default', 'moved-by-one', 'moved-by-two-and-one'])
@pytest.mark.parametrize('cell', sorted(_CELLS))
def test_the_centre_sign_is_the_phase_the_centre_adds(cell, excitation):
    """``(-1)**(m + n) == exp(1j * G_mn . (a1 + a2)/2)``, read off the solver's own wavevectors.

    ``G_mn`` is each retained order's in-plane wavevector minus the specular order's, from
    ``order_kxy()`` -- the grid the solve is built on, not a second spelling of it -- and the
    centre from ``solver.lattice``. On the two skewed cells ``G . o`` is ``pi (m + n)`` only
    because ``b_i . a_j = 2 pi delta_ij`` holds for the *pair*; a rectangle-only formula for the
    centre would pass the first cell and fail these. The excitations move the window by an odd
    step on each axis, where parity by array index and parity by label part company.
    """
    solver = wavebit.rcwa(backend=0, wavelength=0.9, theta=0.3, phi=0.4, top=1.0, bot=2.25,
                          structure=np.full((1, 8, 8), 2.25), thickness=[0.2],
                          truncation_order=(2, 2), excitation=excitation, **_CELLS[cell])
    window_u, window_v = solver.order_window
    gx, gy = order_grid_kxy(np, *(_host(k) for k in solver.order_kxy()))
    specular = (0 - window_v[0] + window_v[1], 0 - window_u[0] + window_u[1])
    k0 = 2 * np.pi / solver.wavelength
    G = k0 * np.stack([gx - gx[specular], gy - gy[specular]], axis=-1)
    o = _host(solver.lattice.columns).sum(axis=1) / 2
    phase = np.exp(1j * G @ o)
    sign = np.where(centre_parity(window_u, window_v), -1.0, 1.0)
    assert phase.shape == sign.shape
    assert np.abs(phase - sign).max() < 1e-13, np.abs(phase - sign).max()
    if excitation is not None:
        # The premise: this window's first label is odd on at least one axis, so an index
        # parity would disagree with the label parity here.
        assert (window_u[0] - window_u[1]) % 2 or (window_v[0] - window_v[1]) % 2


def test_the_parity_is_read_by_label():
    """Label ``(m, n)``, not the array position: a window moved by one flips every entry."""
    assert centre_parity((0, 1), (0, 0)).tolist() == [[True, False, True]]
    assert centre_parity((1, 1), (0, 0)).tolist() == [[False, True, False]]
    assert centre_parity((0, 1), (0, 1)).tolist() == [[False, True, False],
                                                      [True, False, True],
                                                      [False, True, False]]
    assert centre_parity((0, 1), (-1, 1)).tolist() == [[True, False, True],
                                                       [False, True, False],
                                                       [True, False, True]]
    assert centre_parity((0, 2), (0, 0)).dtype == bool


def test_the_corner_column_is_a_sign_bit():
    """Odd weights negated -- bit for bit, signed zeros and all -- even ones untouched.

    ``None`` stays ``None``, so an undeclared solve keeps the unit column it always had.
    """
    assert corner_column(None, (0, 3), (0, 0)) is None
    weights = {-1: 0.4 + 0.2j, 0: 1.0, 1: 0.6 - 0.8j, 2: complex(-0.0, 0.3), 3: -0.0}
    window = order_window(3, 0, weights)
    lit = lit_orders(weights, *window)
    corner = corner_column(lit, *window)
    odd = centre_parity(*window).reshape(-1)
    assert odd[[one.index for one in lit]].tolist() == [True, False, True, False, True]
    for public, kernel in zip(lit, corner):
        assert (kernel.index, kernel.order) == (public.index, public.order)
        want = np.asarray(-public.amplitude if odd[public.index] else public.amplitude)
        got = np.asarray(kernel.amplitude)
        assert (got.dtype, got.tobytes()) == (want.dtype, want.tobytes())
    n = centre_parity(*window).size
    public_column = incident_column(np, np.complex128, lit, n, None)
    kernel_column = incident_column(np, np.complex128, corner, n, None)
    # `==`, not bytes: a real weight takes its `+0j` after the negation, so its imaginary zero
    # is unsigned in the kernels' column and signed in the negated public one.
    assert np.array_equal(kernel_column, np.where(odd, -public_column, public_column))
    # The specular order is even, so the one-wave default a direct kx names is untouched.
    default = lit_orders({(0, 0): 1.0}, *order_window(3, 2, None))
    assert corner_column(default, *order_window(3, 2, None)) == default
