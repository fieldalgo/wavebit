"""Tapered line gratings shared by ``gen_goldens_reticolo_taper.py`` and its test.

The staircase handed to RETICOLO is built here, from first principles, without importing
``wavebit.rcwa._profile``: a sidewall is the straight line through its two straight-wall
corners, and each rounding is the circle whose centre is the intersection of the two lines
offset by its radius from the faces it joins. Slices are uniform, each taking the outline at
its midpoint height. The test checks that the solver's own slices land on the same widths.

All lengths in micrometres; materials are relative permittivities.
"""
import math

import numpy as np

WAVELENGTH = 0.6
PERIOD = 0.5
ORDERS = 15
#: A lossy silicon-like line on fused silica, under air. The substrate is lossless so that RETICOLO
#: reports transmitted efficiencies: it gives none into an absorbing half-space.
EPS_LINE = 15.2 + 0.16j
EPS_SUBSTRATE = 2.1025

#: name -> (cd at the bottom, (left, right) sidewall angles in radians, top rounding, footing, height)
PROFILES = {
    'taper': (0.20, (math.radians(80.), math.radians(80.)), 0., 0., 0.20),
    'rounded': (0.22, (math.radians(82.), math.radians(86.)), 0.03, 0.02, 0.20),
}
#: (theta in radians, polarization) pairs.
LIGHT = ((0., 'te'), (math.radians(20.), 'te'), (0., 'tm'), (math.radians(20.), 'tm'))
#: Slice counts compared on the same staircase, and the ladder extrapolated to the limit.
SAME_STAIRCASE = (8, 32)
LADDER = (64, 128, 256)


def _offset_intersection(p1, d1, p2, d2, r, n1, n2):
    """Intersection of two lines, each moved by ``r`` along its unit normal ``n``."""
    a = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
    rhs = (np.asarray(p2) + r*np.asarray(n2)) - (np.asarray(p1) + r*np.asarray(n1))
    s, _ = np.linalg.solve(a, rhs)
    return np.asarray(p1) + r*np.asarray(n1) + s*np.asarray(d1)


def edge(name, z, side):
    """x of the left (``side=-1``) or right (``side=+1``) outline at height ``z``, centre at 0."""
    cd, (left, right), top, foot, h = PROFILES[name]
    swa = right if side > 0 else left
    base = side*cd/2                                  # straight-wall corner on the base
    up = np.array([-side*math.cos(swa), math.sin(swa)])   # along the wall, upward
    crest = np.array([base, 0.]) + (h/math.sin(swa))*up   # straight-wall corner on the top
    inward = np.array([-side*math.sin(swa), -math.cos(swa)])  # wall normal into the material
    if top > 0:
        centre = _offset_intersection(crest, up, crest, np.array([1., 0.]), top, inward, np.array([0., -1.]))
        # The wall tangent point is the foot of the perpendicular from the centre.
        tangent_z = (centre - top*inward)[1]
        if z >= tangent_z:
            return centre[0] + side*math.sqrt(max(top**2 - (z - centre[1])**2, 0.))
    if foot > 0:
        outward = -inward
        centre = _offset_intersection(np.array([base, 0.]), up, np.array([base, 0.]), np.array([1., 0.]),
                                      foot, outward, np.array([0., 1.]))
        tangent_z = (centre - foot*outward)[1]
        if z <= tangent_z:
            return centre[0] - side*math.sqrt(max(foot**2 - (z - centre[1])**2, 0.))
    return base + (z/up[1])*up[0]


def staircase(name, count):
    """Uniform slices: ``(thickness, left, right)`` per slice, bottom first."""
    h = PROFILES[name][4]
    out = []
    for k in range(count):
        z = (k + 0.5)*h/count
        out.append((h/count, edge(name, z, -1), edge(name, z, 1)))
    return out


def key(name, count, theta, pol):
    return f'{name}__n{count}__th{round(math.degrees(theta))}__{pol}'
