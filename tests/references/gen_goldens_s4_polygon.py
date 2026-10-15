"""Freeze S4's per-order efficiencies for polygon regions into ``golden_s4_polygon.npz``.

Run once, in the project env with the S4 extension built; ``test_s4_polygon_golden.py`` then
runs anywhere. Rerun it only to add cases, never to make a failing test pass.

    python tests/references/gen_goldens_s4_polygon.py

**What it arbitrates.** ``rcwa.Polygon``: the vertices as an outline, the exact polygon
transform, the painting of one region over another, an outline reaching past the cell edge,
one wider than the period and one on a sheared cell. S4's ``SetRegionPolygon`` is an
independent implementation of the same geometry, and its *direct* rule -- no
``PolarizationDecomposition`` -- on ``LatticeTruncation='Parallelogramic'`` with
``NumBasis=(2N+1)**2`` is the operator ``factorization='laurent'`` builds on the same order set.
So the comparison is an identity, not a convergence study: measured before freezing, every
order of every case below agrees to 6e-13 or better. It cannot speak for the default NVF
factorization, whose normal field S4 does not build the same way (S4's vector formulations do
not even settle on a corner-rich layer -- see ``README.md``).

Three S4 conventions, each established by a probe before any number here was trusted:

* ``Vertices`` are relative to ``Center`` and must run **counter-clockwise**. A clockwise list
  is accepted silently and solves a different structure: a rotated rectangle given clockwise
  read ``sum(T)`` 0.859 against the 0.974 of the same list counter-clockwise, which matched
  ``SetRegionRectangle`` to 1e-13. :func:`s4_orders` therefore refuses a clockwise outline;
  Wavebit takes either winding, and the test feeds it the declared order.
* S4 drops a region whose boundary crosses a later one's -- no error, the earlier region simply
  vanishes. Regions here are therefore nested or disjoint, never crossing; a crossing painting
  is the clipper's, and ``tests/rcwa/test_polygon_region.py`` checks it against a raster.
* ``GetBasisSet`` labels ``(m, n)`` are this repository's ``(orders_u, orders_v)``, and
  ``IncidenceAngles`` is ``(theta, phi)`` in degrees, as ``gen_goldens_s4.py`` found.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))
from tools.repository_paths import repository_path

import numpy as np

OUT = repository_path('tests/references') / 'golden_s4_polygon.npz'

WAVELENGTH, THICKNESS = 1.3, 0.4
EPS_INCLUSION, EPS_SUBSTRATE = 6.25, 2.25
THETA, PHI = 0.35, 0.6                    # radians: oblique, and off both cell axes
ORDERS = (3, 6)
SQUARE = ((1.0, 0.0), (0.0, 1.0))         # columns a1, a2

TRIANGLE = ((-.3, -.25), (.32, -.2), (.05, .33))
#: Non-convex: the reflex corner at (-0.1, -0.1) is what a convex-only test would get wrong.
L_SHAPE = ((-.35, -.35), (.3, -.35), (.3, -.1), (-.1, -.1), (-.1, .3), (-.35, .3))
#: The same kind of L, reaching past x = +0.5 into the next cell: the transform has to carry the
#: wrap by itself, since nothing cuts the outline at the cell edge.
L_ACROSS = ((.1, -.35), (.8, -.35), (.8, -.1), (.35, -.1), (.35, .3), (.1, .3))
#: 1.2 wide on a period of 1: wider than the cell, yet clear of its own images, because each
#: arm runs into the notch the neighbouring copy leaves. A bounding box or a convex hull would
#: call it overlapping; the outline is what decides.
Z_WIDE = ((-.5, -.4), (.2, -.4), (.2, .2), (.7, .2), (.7, .4), (0., .4), (0., -.2), (-.5, -.2))
#: Painted inside the L's corner, nested rather than crossing -- the one kind of painting S4 reads.
INNER = ((-.3, -.3), (-.18, -.3), (-.22, -.15))
SHEAR = ((1.0, 0.0), (0.35, 0.9))         # columns a1, a2: a1 = (1, 0), a2 = (0.35, 0.9)

#: name -> (regions as (vertices, permittivity), painted in order; cell columns)
CASES = {
    'triangle': ([(TRIANGLE, EPS_INCLUSION)], SQUARE),
    'l_shape': ([(L_SHAPE, EPS_INCLUSION)], SQUARE),
    'l_across_the_cell_edge': ([(L_ACROSS, EPS_INCLUSION)], SQUARE),
    'z_wider_than_the_period': ([(Z_WIDE, EPS_INCLUSION)], SQUARE),
    'l_with_a_hole': ([(L_SHAPE, EPS_INCLUSION), (INNER, 1.0)], SQUARE),
    'l_with_an_inclusion': ([(L_SHAPE, EPS_INCLUSION), (INNER, 3.0)], SQUARE),
    'l_on_a_sheared_cell': ([(L_SHAPE, EPS_INCLUSION)], SHEAR),
}


def _signed_area(vertices):
    v = np.asarray(vertices, float)
    return 0.5 * float(np.sum(v[:, 0] * np.roll(v[:, 1], -1) - np.roll(v[:, 0], -1) * v[:, 1]))


def s4_orders(order, regions, columns, polarization):
    """``(R, T)``, each ``(2*order+1, 2*order+1)`` indexed ``[v, u]``, from S4's direct rule."""
    import S4
    a1, a2 = np.asarray(columns, float)
    S = S4.New(Lattice=(tuple(a1), tuple(a2)), NumBasis=(2 * order + 1) ** 2)
    S.SetOptions(LatticeTruncation='Parallelogramic')
    S.SetMaterial(Name='air', Epsilon=1.0)
    S.SetMaterial(Name='substrate', Epsilon=EPS_SUBSTRATE)
    S.AddLayer(Name='top', Thickness=0.0, Material='air')
    S.AddLayer(Name='film', Thickness=THICKNESS, Material='air')
    for k, (vertices, epsilon) in enumerate(regions):
        if _signed_area(vertices) <= 0:
            raise SystemExit('S4 takes its polygons as given: a clockwise outline would be '
                             'solved as a different structure. Do not freeze this.')
        S.SetMaterial(Name=f'region{k}', Epsilon=epsilon)
        S.SetRegionPolygon(Layer='film', Material=f'region{k}', Center=(0.0, 0.0), Angle=0.0,
                           Vertices=tuple(tuple(float(c) for c in p) for p in vertices))
    S.AddLayer(Name='bottom', Thickness=0.0, Material='substrate')
    S.SetExcitationPlanewave(IncidenceAngles=(np.rad2deg(THETA), np.rad2deg(PHI)),
                             sAmplitude=1.0 if polarization == 's' else 0.0,
                             pAmplitude=0.0 if polarization == 's' else 1.0)
    S.SetFrequency(1.0 / WAVELENGTH)
    basis = S.GetBasisSet()
    expected = {(m, n) for m in range(-order, order + 1) for n in range(-order, order + 1)}
    if set(basis) != expected:
        raise SystemExit('S4 retained another order set than the square window; not a reference')
    incident, _ = S.GetPowerFlux(Layer='top')
    R = np.zeros((2 * order + 1, 2 * order + 1))
    T = np.zeros_like(R)
    for (m, n), (_, back) in zip(basis, S.GetPowerFluxByOrder(Layer='top')):
        R[n + order, m + order] = -back.real / incident.real
    for (m, n), (forward, _) in zip(basis, S.GetPowerFluxByOrder(Layer='bottom')):
        T[n + order, m + order] = forward.real / incident.real
    return R, T


def main():
    out = {}
    for name, (regions, columns) in CASES.items():
        for order in ORDERS:
            for polarization in ('s', 'p'):
                R, T = s4_orders(order, regions, columns, polarization)
                # Every case is lossless, so a bridge wired wrong shows up here first.
                if abs(R.sum() + T.sum() - 1.0) > 1e-9:
                    raise SystemExit(f'{name} order {order} {polarization}: R + T = '
                                     f'{R.sum() + T.sum():.12f}; do not freeze this')
                out[f'{name}__{order}__{polarization}__R'] = R
                out[f'{name}__{order}__{polarization}__T'] = T
                print(f'  {name:<26} order {order} {polarization}: sum R {R.sum():.12f}  '
                      f'sum T {T.sum():.12f}')
    np.savez(OUT, **out)
    print(f'\nwrote {OUT} ({len(out)} arrays)')


if __name__ == '__main__':
    main()
