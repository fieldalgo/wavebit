"""
Naming the two order axes, at both ends: no order to remember in, labels out.

Every array axis in this package runs ``v`` first and every hand-written pair runs ``x``
first. Each rule is consistent on its own -- one is numpy's, one is what people type -- and
neither is wrong. They meet at ``truncation_order``, and the meeting is where readers go
wrong: ``[4, 1]`` gives ``R.shape == (3, 9)``, so taking the leading ``4`` to mean the leading
axis indexes the wrong entry and returns a plausible number.

Two changes, at the two ends:

* a mapping input (``dict(u=4, v=1)``) has no position to misread. ``dict(x=, y=)`` is
  accepted where it is true and **refused on a skewed cell**, where the order axes follow the
  reciprocal vectors and ``x`` names nothing -- measured, a 60-degree hex sends ``m += 1`` to
  ``(dkx, dky) = (+0.600, -0.346)``.
* ``orders_u`` / ``orders_v`` label the axes on the way out, so a reader who zips them the
  wrong way gets a length mismatch instead of a number.
"""
import numpy as np
import pytest

import wavebit
from wavebit._core.conventions import parse_order_pair

RECT = [700.0, 1100.0]
HEX = wavebit.hexagonal(400.0)


def _eps():
    e = np.ones((1, 16, 16))
    e[0, 3:8, 2:7] = 12.25
    return e


def _solver(truncation_order, lattice=None, period=None):
    kw = dict(backend=0, wavelength=900.0, truncation_order=truncation_order, top=1.0, bot=1.0,
              theta=0.1, phi=0.3, psi=0.2, structure=_eps(), thickness=[300.0],
              factorization='nvf')
    if lattice is not None:
        kw['lattice'] = lattice
    else:
        kw['period'] = period if period is not None else RECT
    return wavebit.rcwa(**kw)


@pytest.mark.parametrize('spelling', [[4, 1], (4, 1), np.array([4, 1]),
                                      dict(u=4, v=1), dict(x=4, y=1)],
                         ids=['list', 'tuple', 'ndarray', 'uv', 'xy'])
def test_every_spelling_of_the_same_pair_is_the_same_solve(spelling):
    """A mapping is a spelling and not a mode: byte-identical to the pair it replaces."""
    want = np.asarray(_solver([4, 1]).solve().R)
    got = np.asarray(_solver(spelling).solve().R)
    assert got.tobytes() == want.tobytes()


def test_a_mapping_says_which_is_which_and_a_pair_does_not():
    assert parse_order_pair(dict(u=4, v=1)) == ([4, 1], 'uv')
    assert parse_order_pair(dict(x=4, y=1)) == ([4, 1], 'xy')
    assert parse_order_pair([4, 1]) == ([4, 1], None)
    assert parse_order_pair(7) == ([7, 0], None)
    for bad in (dict(u=4), dict(u=4, v=1, w=0), dict(a=1, b=2), [1, 2, 3], 'x'):
        with pytest.raises(ValueError):
            parse_order_pair(bad)


def test_xy_is_refused_on_a_skewed_cell_and_uv_is_not():
    """``x`` and ``y`` do not name the order axes off a rectangular cell, so they are refused.

    Not reinterpreted as ``u``/``v``: that would be right in exactly the case where the caller
    did not need telling and wrong in the case where they did. The same choice
    ``inverse_rule`` and the explicit ``sample_x`` coordinates already make.
    """
    _solver(dict(u=4, v=1), lattice=HEX).solve()          # always true, always fine
    _solver([4, 1], lattice=HEX).solve()                  # a bare pair names nothing to check
    with pytest.raises(ValueError, match='not axis-aligned'):
        _solver(dict(x=4, y=1), lattice=HEX).solve()
    _solver(dict(x=4, y=1)).solve()                       # rectangular: true, so allowed


def test_period_takes_the_axis_aligned_names_only():
    """``period`` is the axis-aligned spelling of a cell, so ``x``/``y`` is what it has.

    A skewed cell arrives as ``lattice=`` instead, which is why this one needs no cell-shape
    check: there is no cell it can describe where the names would be false.
    """
    a = np.asarray(_solver([4, 1], period=RECT).solve().R)
    b = np.asarray(_solver([4, 1], period=dict(x=700.0, y=1100.0)).solve().R)
    assert a.tobytes() == b.tobytes()
    with pytest.raises(ValueError, match=r"needs exactly \{'x', 'y'\}"):
        _solver([4, 1], period=dict(u=700.0, v=1100.0))


def test_the_labels_name_the_axis_they_belong_to():
    r = _solver(dict(u=4, v=1)).solve()
    R = np.asarray(r.R)
    assert R.shape == (len(r.orders_v), len(r.orders_u))
    assert list(r.orders_u) == list(range(-4, 5))
    assert list(r.orders_v) == list(range(-1, 2))

    # every entry agrees with the accessor that hides the layout
    for i, v in enumerate(r.orders_v):
        for j, u in enumerate(r.orders_u):
            assert R[i, j] == pytest.approx(float(np.real(r.order(int(u), int(v)).R)))


def test_reading_the_table_the_wrong_way_round_stops():
    """The service the labels exist for: a length mismatch instead of a plausible number."""
    r = _solver(dict(u=4, v=1)).solve()
    R = np.asarray(r.R)
    assert len(r.orders_u) != R.shape[0]           # 9 against 3 -- the mismatch is the point
    with pytest.raises(ValueError):
        np.vstack([r.orders_u, R[:, 0]])


def test_the_labels_follow_a_moved_window_not_the_declaration():
    """With an ``excitation`` the retained window widens, and the labels describe the array.

    A result labelled from ``truncation_order`` would be describing something it is not -- the
    same defect that once made ``order(0)`` return order -1.
    """
    e = _eps()
    # long enough that orders 0 and 2 both propagate: |kx_m| = |m| * wl / period < 1
    s = wavebit.rcwa(backend=0, wavelength=900.0, period=[2400.0, 1100.0],
                     truncation_order=dict(u=6, v=2), top=1.0, bot=1.0, theta=0.0, phi=0.0,
                     psi=0.2, structure=e, thickness=[300.0],
                     excitation={0: 1.0, 2: 0.5 - 0.3j})
    r = s.solve()
    R = np.asarray(r.R)
    assert R.shape == (len(r.orders_v), len(r.orders_u))
    assert 0 in list(r.orders_u) and 2 in list(r.orders_u)
    for j, u in enumerate(r.orders_u):
        assert R[len(r.orders_v) // 2, j] == pytest.approx(
            float(np.real(r.order(int(u), int(r.orders_v[len(r.orders_v) // 2])).R)))
