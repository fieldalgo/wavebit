"""Polygon regions against S4's ``SetRegionPolygon``, from a checked-in golden.

``rcwa.Polygon`` is checked here against an implementation that shares none of its code: S4
transforms its own polygons and solves with its own operators. Its direct rule on a matched
order set is the operator ``factorization='laurent'`` builds, so the two must agree to rounding
on every diffraction order -- and they do, to 5.8e-13 at worst over the seven cases and both
orders (measured when the golden was frozen; the tolerance below is ten times that). What the
cases reach that a primitive cannot: a reflex corner, an outline crossing the cell edge, one
wider than the period that misses its own images only through its notches, two regions painted
one inside the other, and a sheared cell.

The default NVF factorization is not arbitrated here: its normal field is this repository's,
and S4's vector formulations do not settle on a corner-rich layer (``README.md``). The polygon's
NVF solve is checked against the closed-form rectangle and against its own Laurent geometry in
``tests/rcwa/test_polygon_region.py``.

Values are frozen in ``golden_s4_polygon.npz`` by ``gen_goldens_s4_polygon.py``, which needs
the S4 extension. This file needs only numpy.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))
from tools.repository_paths import repository_path

import numpy as np
import pytest

from wavebit import Material, rcwa
from tests.references.gen_goldens_s4_polygon import (CASES, EPS_SUBSTRATE, ORDERS, PHI, THETA,
                                                     THICKNESS, WAVELENGTH)

GOLDEN_PATH = repository_path('tests/references') / 'golden_s4_polygon.npz'

#: Ten times the largest per-order gap measured when the golden was frozen (5.8e-13, the sheared
#: cell at order 6; 1e-14 to 1e-13 elsewhere). An identity, so anything that moves the geometry
#: -- a vertex read against the wrong origin, a winding taken as given -- lands far above it.
TOLERANCE = 6e-12


@pytest.fixture(scope='module')
def golden():
    if not GOLDEN_PATH.exists():
        pytest.skip(f'{GOLDEN_PATH.name} missing; regenerate with gen_goldens_s4_polygon.py')
    return np.load(GOLDEN_PATH)


def solve(regions, columns, order, polarization, reverse=False):
    """Per-order ``(R, T)`` from the public lifecycle, as ``[v, u]`` arrays."""
    layer = rcwa.Layer(thickness=THICKNESS, background=Material(1.0), regions=tuple(
        rcwa.Region(geometry=rcwa.Polygon(vertices=vertices[::-1] if reverse else vertices),
                    material=Material(epsilon)) for vertices, epsilon in regions))
    stack = rcwa.Stack(layers=(layer,), bottom=Material(EPS_SUBSTRATE),
                       lattice=np.asarray(columns, float).T)
    problem = rcwa.Problem(stack=stack, excitation=rcwa.Excitation(
        wavelength=WAVELENGTH,
        sources=(rcwa.PlaneWave(theta=THETA, phi=PHI, polarization=polarization),)))
    result = rcwa.run(problem, numerics=rcwa.Numerics(orders=(order, order),
                                                      factorization='laurent'))
    ou = np.asarray(result.reflection.orders_u)
    ov = np.asarray(result.reflection.orders_v)
    assert list(ou) == list(range(-order, order + 1)) and list(ov) == list(ou)
    return np.asarray(result.R)[0], np.asarray(result.T)[0]


@pytest.mark.parametrize('order', ORDERS)
@pytest.mark.parametrize('name', list(CASES))
def test_every_order_matches_s4(golden, name, order):
    regions, columns = CASES[name]
    for polarization in ('s', 'p'):
        R, T = solve(regions, columns, order, polarization)
        gap = max(np.abs(R - golden[f'{name}__{order}__{polarization}__R']).max(),
                  np.abs(T - golden[f'{name}__{order}__{polarization}__T']).max())
        assert gap < TOLERANCE, f'{name} order {order} {polarization}: {gap:.2e} from S4'


def test_a_clockwise_outline_is_the_same_region(golden):
    """S4 needs counter-clockwise vertices; Wavebit reads the winding and takes either.

    The golden was frozen counter-clockwise because S4 solves a clockwise list as another
    structure. Fed reversed, every outline here must still land on S4's numbers -- the
    painting included, where the hole is a clockwise outline painted over a counter-clockwise
    one and not a hole in the bookkeeping sense.
    """
    for name in ('l_shape', 'l_with_an_inclusion'):
        regions, columns = CASES[name]
        R, T = solve(regions, columns, 3, 'p', reverse=True)
        gap = max(np.abs(R - golden[f'{name}__3__p__R']).max(),
                  np.abs(T - golden[f'{name}__3__p__T']).max())
        assert gap < TOLERANCE, f'{name}: {gap:.2e} from S4 with the outlines reversed'
