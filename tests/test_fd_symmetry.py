"""Gate 3 -- symmetry: a half box with a mirror equals the matching half of the full box.

Four cases on each of the three axes, each on every backend: the mirror on the low face (a
NODE plane, on the face) and on the high face (a HALF plane, half a cell inside), each with
parity +1 (tangential E even, PMC-like) and -1 (tangential E odd, PEC-like). The half box
starts from random fields with the on-plane odd components zeroed; the full box starts
from their exact mirror image with the parity's signs (``tests/fd_helpers.py``); after 200
steps every component of every cell of the half box is held against the corresponding cell
of the full box, and two probes are held step by step. The arithmetic is the same in mirror
image -- ``(-a) - (-b)`` is exactly ``-(a - b)`` -- so the tolerance is round-off, 1e-13
relative. Each case also asserts that the run *moved*: the probe series are not constant.
The equality alone is blind to a kernel that ran zero steps, since the full box starts
from the bit-exact mirror of the half box (an identity step passed 22 of 26 tests here on
2026-09-18; only the mutant saw it).

The parity table the kernel implements (``src/wavebit/fd/_kernel/faces.py``): under a mirror
across an axis with parity ``p``, the normal E goes to ``-p``, tangential E to ``+p``, the
normal H to ``+p``, tangential H to ``-p``. The low face pads H with the image of the layer
next to the plane (tangential H from layer 0, normal H from layer 1); the high face pads E
(tangential E from layer N-1, normal E from N-2). Masks: low node plane, ``p = -1``:
tangential E and normal H on layer 0; high half plane, ``p = +1``: normal E and tangential H
on layer N-1; nothing otherwise.

Mutants. On x: the tangential pair ``(E_y, H_z)`` given the opposite parity -- the pad sign
of the padded member flipped, the mask of the other toggled -- breaks the equality by O(1)
in every case. On y and z: the mirror's normal component hard-wired to x (``normal = (c ==
0)`` in ``_mirror_pad``), which the x cases cannot see and which passed the whole suite when
the gate was x-only (2026-09-18); it breaks the two non-degenerate cases per axis. Two
corollaries are pinned as well: ``Symmetry(-1)`` on a low face runs bit-for-bit as PEC
there, and ``Symmetry(+1)`` on a high face as PMC; in those two the pad and the mask each
enforce the same zero on their own, so no single sign is load-bearing, which is why the
first mutant flips the pair and the second is asserted on the other two cases.
"""
import dataclasses

import numpy as np
import pytest

import wavebit.fd as fd
from wavebit.fd._kernel import faces
from wavebit.fd._kernel.masks import Mask, Slab
from fd_helpers import backends, run_plan, symmetry_pair, to_numpy

TOL = 1e-13
#: A probe series must vary by at least this fraction of its peak: the guard against a
#: kernel that did nothing. Under an identity step the series is constant (0), and a
#: comparison of the final field with the initial one would not see that -- the far PEC
#: walls project the random initial data at t = 0, which is O(1) of movement by itself.
MOVED = 0.1
CASES = [('-', +1), ('-', -1), ('+', +1), ('+', -1)]
AXES = [0, 1, 2]


def _max_rel(a, b):
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def _assert_moved(res):
    """The probe series are not constant: the kernel ran."""
    for name in ('Ey', 'Hx'):
        v = to_numpy(res[name].values)
        assert np.ptp(v) > MOVED * np.max(np.abs(v)), f'{name} did not move'


@pytest.mark.parametrize('axis', AXES)
@pytest.mark.parametrize('side,parity', CASES)
def test_the_plan_places_the_plane_where_the_layout_allows(axis, side, parity):
    half, full, take = symmetry_pair(side, parity, axis=axis)
    plan = fd.plan(half)
    rule = plan.faces['xyz'[axis] + side]
    assert rule.kind == 'symmetry' and rule.parity == parity
    dx = plan.dx
    n = plan.grid.n[axis]
    if side == '-':
        assert rule.plane == 'node'
        assert rule.position == pytest.approx(plan.low[axis])
    else:
        assert rule.plane == 'half'
        assert rule.position == pytest.approx(plan.low[axis] + n * dx - dx / 2)
    # the plane coincides with the full box's centre plane
    full_plan = fd.plan(full)
    assert rule.position == pytest.approx(full_plan.low[axis] + full_plan.grid.n[axis] * dx / 2)


@pytest.mark.parametrize('backend', backends())
@pytest.mark.parametrize('axis', AXES)
@pytest.mark.parametrize('side,parity', CASES)
def test_half_box_equals_the_half_of_the_full_box(backend, axis, side, parity):
    half, full, take = symmetry_pair(side, parity, axis=axis)
    rh = fd.run(half, backend=backend)
    rf = fd.run(full, backend=backend)
    _assert_moved(rh)
    for name in ('Ey', 'Hx'):
        assert _max_rel(to_numpy(rh[name].values), to_numpy(rf[name].values)) < TOL
    for field in ('E', 'H'):
        h = to_numpy(getattr(rh.final, field))
        f = take(to_numpy(getattr(rf.final, field)))
        assert h.shape == f.shape
        assert _max_rel(h, f) < TOL


def _flip_pair(plan, face):
    """The mutant: the ``(E_y, H_z)`` pair gets the opposite parity on ``face`` (an x face).

    The x-difference at the plane couples exactly that pair: ``H_z``'s update reads the
    ``E_y`` pad on a high face, ``E_y``'s reads the ``H_z`` pad on a low one. So the mutant
    flips the pad sign of the padded member and toggles the mask of the other on the plane
    layer. Both, because in the two degenerate cases -- ``p = -1`` low, ``p = +1`` high --
    the pad and the mask each enforce the same zero on their own (that is the corollary
    that those cases *are* PEC and PMC), so no single sign there is load-bearing; in the
    other two the pad sign alone already breaks it.
    """
    axis = 'xyz'.index(face[0])
    n = (plan.nx, plan.ny, plan.nz)
    if face[1] == '-':
        padded_c, masked = 2, ('E', 1, 0)                   # H_z pad; E_y mask on layer 0
    else:
        padded_c, masked = 1, ('H', 2, n[axis] - 1)         # E_y pad; H_z mask on layer N-1
    pads = tuple((nm, comp, dataclasses.replace(p, sign=-p.sign)
                  if (nm, comp) == (face, padded_c) else p) for nm, comp, p in plan.pads)
    field, c, layer = masked
    mask = Mask(field, Slab.layer(c, axis, layer, n))
    masks = tuple(m for m in plan.masks if m != mask)
    if len(masks) == len(plan.masks):
        masks = masks + (mask,)
    return plan.with_static(pads=pads, masks=masks)


@pytest.mark.parametrize('side,parity', CASES)
def test_mutant_one_flipped_sign_breaks_the_equality(side, parity):
    half, full, take = symmetry_pair(side, parity)
    plan = fd.plan(half)
    face = 'x' + side
    assert plan.pad(face, 1).kind == 'mirror'
    rh = run_plan(half, _flip_pair(plan, face))
    rf = fd.run(full)
    worst = max(_max_rel(to_numpy(rh[name].values), to_numpy(rf[name].values))
                for name in ('Ey', 'Hx'))
    assert worst > 1e-3, f'flipping the (E_y, H_z) parity on {face} moved nothing: {worst}'


def _mirror_pad_with_x_hard_wired(field, c, axis, side, parity, n):
    """``faces._mirror_pad`` with ``normal = (c == 0)``: the sign table of an x mirror on every axis."""
    pad = _MIRROR_PAD(field, c, axis, side, parity, n)
    normal = (c == 0)
    sign = (-parity if normal else parity) if field == 'E' else (parity if normal else -parity)
    return dataclasses.replace(pad, sign=sign)


_MIRROR_PAD = faces._mirror_pad


@pytest.mark.parametrize('axis', [1, 2])
@pytest.mark.parametrize('side,parity', [('-', +1), ('+', -1)])
def test_mutant_a_mirror_whose_normal_is_always_x_breaks_the_y_and_z_cases(monkeypatch, axis,
                                                                           side, parity):
    """The two non-degenerate cases per axis; on x the mutant is the code itself.

    On a y or z mirror the mutant swaps the signs of ``c = 0`` (treated as normal) and
    ``c = axis`` (treated as tangential) in the padded field. In the degenerate cases
    (``-1`` low, ``+1`` high) the flipped pads feed only masked components and the mutant is
    provably invisible, as ``faces.py`` says of those cases; here it is O(1).
    """
    monkeypatch.setattr(faces, '_mirror_pad', _mirror_pad_with_x_hard_wired)
    half, full, take = symmetry_pair(side, parity, axis=axis)
    plan = fd.plan(half)                                   # built with the mutant in place
    face = 'xyz'[axis] + side
    assert plan.pad(face, 0).sign != _MIRROR_PAD(faces.PADDED_FIELD[side], 0, axis, side, parity,
                                                 plan.grid.n[axis]).sign
    rh = run_plan(half, plan)
    rf = fd.run(full)                                      # no mirror: the mutant is not reached
    worst = max(_max_rel(to_numpy(rh.final.E), take(to_numpy(rf.final.E))),
                _max_rel(to_numpy(rh.final.H), take(to_numpy(rf.final.H))))
    assert worst > 1e-3, f'an x-only sign table on {face} moved nothing: {worst}'
    # and on x the mutant is the code: the same plan as without it
    x_half, _, _ = symmetry_pair(side, parity, axis=0)
    assert fd.plan(x_half).skeleton == _X_SKELETONS[(side, parity)]


#: The x-mirror plans built with the real ``_mirror_pad``, before any test patches it.
_X_SKELETONS = {case: fd.plan(symmetry_pair(*case, axis=0)[0]).skeleton for case in CASES}


def test_symmetry_minus_one_on_a_low_face_is_pec_there():
    half, _, _ = symmetry_pair('-', -1)
    pec = half.replace(boundaries=half.boundaries.replace(x=fd.PEC()))
    a, b = fd.run(half), fd.run(pec)
    _assert_moved(a)
    assert np.array_equal(to_numpy(a.final.E), to_numpy(b.final.E))
    assert np.array_equal(to_numpy(a.final.H), to_numpy(b.final.H))


def test_symmetry_plus_one_on_a_high_face_is_pmc_there():
    half, _, _ = symmetry_pair('+', +1)
    pmc = half.replace(boundaries=half.boundaries.replace(x=(fd.PEC(), fd.PMC())))
    a, b = fd.run(half), fd.run(pmc)
    _assert_moved(a)
    assert np.array_equal(to_numpy(a.final.E), to_numpy(b.final.E))
    assert np.array_equal(to_numpy(a.final.H), to_numpy(b.final.H))


@pytest.mark.parametrize('axis', AXES)
@pytest.mark.parametrize('side,parity', CASES)
def test_the_far_walls_may_be_pmc(axis, side, parity):
    """The full box's PMC walls sit half a cell outside, so it needs one cell more than
    with PEC to be symmetric about the same plane (``fd_helpers.symmetry_pair`` counts)."""
    half, full, take = symmetry_pair(side, parity, far=fd.PMC(), axis=axis)
    rh, rf = fd.run(half), fd.run(full)
    _assert_moved(rh)
    assert _max_rel(to_numpy(rh.final.E), take(to_numpy(rf.final.E))) < TOL
    assert _max_rel(to_numpy(rh.final.H), take(to_numpy(rf.final.H))) < TOL


def test_two_mirrors_on_one_axis_are_refused():
    with pytest.raises(ValueError, match='at most one face'):
        fd.Boundaries(x=(fd.Symmetry(parity=1), fd.Symmetry(parity=1)), y=fd.PEC(), z=fd.PEC())
    with pytest.raises(ValueError, match='both faces'):
        fd.Boundaries(x=fd.Symmetry(parity=1), y=fd.PEC(), z=fd.PEC())
