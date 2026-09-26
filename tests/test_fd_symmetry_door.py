"""Stage 4's C5: what a structure may be where it meets a ``Symmetry`` face.

A mirror face stands for a scene that is its own mirror image: the run carries the domain side,
so it models every structure's domain part and that part's image. ``checks.check_symmetry``
refuses the two declarations that disagree with that silently -- a structure crossing the plane
that is not its own mirror image, and a smoothed body within half a cell of the plane where
the parity leaves the plane's components free -- and accepts the rest, above all the symmetric
body and the dimer a mirror exists to halve. The plane is the node plane on the wall at a
``-`` face and ``dx/2`` inside the wall at a ``+`` face.

The oracles are the geometry, written by hand in each case, and one pair of runs: the half box
with its mirror against the full box, which agree for an accepted scene and do not for the
refused one (with the door held open).
"""
import warnings

import numpy as np
import pytest

import wavebit.fd as fd
import wavebit.fd._lower.checks as checks

DX, N = 0.05, 12


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*args, **kwargs)


def _mirror(structures, side='-', parity=1):
    """A 0.6 box with a mirror on x: ``x in [0, 0.6]`` for ``-``, ``[-0.6, 0]`` for ``+``.

    The ``-`` plane is then at ``x = 0`` and the ``+`` plane at ``x = -0.025``.
    """
    mirror = fd.Symmetry(parity=parity)
    x = (mirror, fd.PEC()) if side == '-' else (fd.PEC(), mirror)
    centre = N * DX / 2 if side == '-' else -N * DX / 2
    return fd.Scene(domain=fd.Domain(size=(N * DX,) * 3, center=(centre, 0.0, 0.0)),
                    boundaries=fd.Boundaries(x=x, y=fd.PEC(), z=fd.PEC()),
                    resolution=fd.Resolution(dx=DX), run=fd.Run(steps=4),
                    structures=tuple(fd.Structure(s, 4.0) for s in structures))


def _box(lo, hi):
    """An unturned ``fd.Box`` spanning ``x in [lo, hi]``, 0.2 square across."""
    return fd.Box(size=(hi - lo, 0.2, 0.2), center=((lo + hi) / 2, 0.0, 0.0))


def _region(lo, hi):
    return fd.Region(size=(hi - lo, 0.2, 0.2), center=((lo + hi) / 2, 0.0, 0.0))


def _plans(scene):
    return _quiet(fd.plan, scene)


def _refused(scene, *phrases):
    with pytest.raises(ValueError) as caught:
        _quiet(fd.plan, scene)
    message = str(caught.value)
    for phrase in phrases:
        assert phrase in message, message
    return message


@pytest.mark.parametrize('parity', (1, -1))
def test_a_box_symmetric_about_the_plane_plans_on_either_face(parity):
    """The one body stage 4 lets cross a mirror, centred on the plane, on both faces."""
    _plans(_mirror((_box(-0.2, 0.2),), '-', parity))
    _plans(_mirror((_box(-0.225, 0.175),), '+', parity))


def test_a_box_crossing_the_plane_unmirrored_is_refused():
    """Declared ``[-0.1, 0.2]`` across the ``-`` plane: the run would model ``[-0.2, 0.2]``.

    The second case reaches further past the plane than into the domain, which a band one cell
    deep does not see: fill 1 on both sides of the plane there.
    """
    message = _refused(_mirror((_box(-0.1, 0.2),)), 'is not its own mirror image',
                       'x- Symmetry plane at x = 0')
    assert 'studies/fdtd_symmetry_mirror.py' in message
    _refused(_mirror((_box(-0.3, 0.1),)), 'is not its own mirror image')


def test_a_box_centred_on_a_plus_wall_is_refused_naming_where_the_plane_is():
    """The half-cell trap: a ``+`` plane is ``dx/2`` inside the wall, not on it."""
    _refused(_mirror((_box(-0.2, 0.2),), '+'), 'is not its own mirror image',
             'half a cell INSIDE the wall', 'x = -0.025')


@pytest.mark.parametrize('side, lo, hi, free', (('-', 0.0, 0.2, 1), ('+', -0.225, -0.025, -1)))
def test_a_flush_half_is_refused_only_where_the_parity_leaves_the_plane_free(side, lo, hi, free):
    """Flush against the plane, the plane's voxels read the body's half and not its image's.

    At the other parity the mirror zeroes exactly those components, and the flush half is
    exact: not refused.
    """
    _refused(_mirror((_box(lo, hi),), side, free), 'within half a cell',
             f'parity={free:+d}')
    _plans(_mirror((_box(lo, hi),), side, -free))


def test_half_a_cell_of_clearance_is_what_a_free_plane_needs():
    """0.6 cells clear of a ``parity=+1`` plane plans; 0.4 cells clear is refused.

    By hand: the plane's voxels span ``[-0.025, 0.025]``; a face at 0.02 puts 0.1 of the box
    in them, a face at 0.03 none.
    """
    _plans(_mirror((_box(0.03, 0.23),)))
    message = _refused(_mirror((_box(0.02, 0.22),)), 'within half a cell')
    assert 'fills 0.1000' in message


def test_a_dimer_across_the_plane_plans():
    """A body wholly on the domain side and clear of the plane: it and its image are the scene."""
    _plans(_mirror((fd.Sphere(radius=0.1, center=(0.3, 0.0, 0.0)),)))


def test_a_cell_snapped_region_crosses_by_its_cells_within_half_a_cell():
    """A Region is judged on its snapped cells, with the snap's own half-cell contract.

    ``-`` face: symmetric plans, flush plans (the mirror joins it to its image), an edge 0.4 of
    a cell past the plane snaps onto it and plans, ``[-0.1, 0.2]`` is refused. ``+`` face, plane
    at cell centre 11.5: 5 cells about it plan exactly, 4 cells about it plan within the half
    cell a cell centre allows, 4 cells about the wall are refused (their mirror edge is a whole
    cell off).
    """
    _plans(_mirror((_region(-0.2, 0.2),)))
    _plans(_mirror((_region(0.0, 0.2),)))
    _plans(_mirror((_region(-0.02, 0.2),)))
    _refused(_mirror((_region(-0.1, 0.2),)), 'crosses the x- Symmetry plane',
             'runs as (-0.2, 0.2)', '2.00 cells')
    _plans(_mirror((_region(-0.15, 0.1),), '+'))
    _plans(_mirror((_region(-0.125, 0.075),), '+'))
    _refused(_mirror((_region(-0.1, 0.1),), '+'), 'crosses the x+ Symmetry plane',
             'half a cell INSIDE the wall')


def test_a_slab_is_judged_across_its_own_axis_only():
    _refused(_mirror((fd.Slab(x=(-0.1, 0.2)),)), 'crosses the x- Symmetry plane')
    _plans(_mirror((fd.Slab(z=(-0.1, 0.1)),)))


def test_a_symmetric_ball_meets_the_mirror_adjoint_refusal_not_the_wall_door():
    """Door 3 no longer answers at a mirror: a symmetric ball is refused for its cross terms.

    Before C5 it met the minus-face message about material outside the domain, which is not
    true at a mirror. An unmirrored ball is this door's.
    """
    message = _refused(_mirror((fd.Sphere(radius=0.1, center=(0.0, 0.0, 0.0)),)),
                       'mirror-weighted adjoint')
    assert 'reaches past' not in message and 'only the unturned fd.Box' in message
    _refused(_mirror((fd.Sphere(radius=0.1, center=(0.02, 0.0, 0.0)),)),
             'is not its own mirror image')


#: The run pair's bar. Its x faces sit on no voxel boundary of any component: a box face lying
#: exactly on one enters that voxel's normal moment or not by round-off (``w_x`` jumps from 0 to
#: 1/3 on the rim voxels when the face moves 1e-12 inside), so the half box and the full box --
#: the same geometry, different floats -- lowered eps 2.105 against 2.5 there and differed by
#: 4.5e-3 of the peak in 300 steps with faces at +-0.2. That is the C4 moment's, not the mirror's.
BAR = _box(-0.185, 0.185)


def _series(side_half, box):
    """``Ez`` at a probe in the bar: the mirrored half box, or the full box it stands for."""
    length = 2 * N * DX

    def mode(x, y, z):
        return np.cos(np.pi * x / length) * np.cos(np.pi * y / (N * DX)) * np.ones_like(z)
    if side_half:
        scene = _mirror((box,), '-', 1)
    else:
        scene = fd.Scene(domain=fd.Domain(size=(length, N * DX, N * DX)),
                         boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
                         resolution=fd.Resolution(dx=DX), run=fd.Run(steps=4),
                         structures=(fd.Structure(BAR, 4.0),))
    scene = scene.replace(run=fd.Run(steps=300, courant=0.5),
                          initial=fd.InitialField(E=(None, None, mode)),
                          monitors=(fd.Record('Ez', at=(0.1125, 0.0125, 0.0), component='Ez'),))
    return np.asarray(_quiet(fd.run, scene)['Ez'].values)


def test_the_accepted_mirror_is_the_full_box_and_the_refused_one_is_not(monkeypatch):
    """What the door protects, run: the symmetric box's half agrees with the full box to
    round-off (6.2e-16 of the peak, measured 2026-09-26); its flush half, with the door held
    open, does not (6.6e-2: its plane voxels read eps 2.5 where the full box holds 4). See
    :data:`BAR` for where its faces are, and why."""
    full = _series(False, None)
    assert np.max(np.abs(_series(True, BAR) - full)) <= 1e-12 * np.max(np.abs(full))
    monkeypatch.setattr(checks, 'check_symmetry', lambda scene, grid: None)
    flush = _series(True, _box(0.0, 0.185))             # exactly BAR's domain half
    assert np.max(np.abs(flush - full)) >= 1e-2 * np.max(np.abs(full))
