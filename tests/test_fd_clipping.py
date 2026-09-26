"""Stage 4's C5: a structure that reaches past a face of the box, and what the report says.

The box clips geometry. At a PEC or PMC face the part past the wall is not in the run, which
may be meant (a guide ending on a wall) or not (a quarter of a box that vanished), and before C5
nothing said which: the report printed ``<- snapped`` for a clipped ``fd.Region`` and nothing at
all for a clipped smoothed body. Now a ``GeometryClippedWarning`` names the share lost and the
faces, the report's rows say ``clipped`` or ``snapped`` for what really happened, and a
cell-snapped shape across one periodic seam without spanning the period -- clipped where it
should have wrapped -- is refused. What runs on out of the box on purpose does not warn: through
an absorbing face, or spanning a periodic axis.

The oracles are lengths by hand: the share lost is the length past the wall over the declared
length, multiplied over the axes.
"""
import warnings

import pytest

import wavebit.fd as fd
from wavebit.fd._check import report as describe


def _box(structures, boundaries=None, size=1.0, dx=0.05):
    return fd.Scene(domain=fd.Domain(size=(size,) * 3),
                    boundaries=boundaries or fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
                    resolution=fd.Resolution(dx=dx), run=fd.Run(steps=4),
                    structures=tuple(fd.Structure(s, 2.25) for s in structures))


def _clipping(scene):
    """The ``GeometryClippedWarning`` messages ``fd.plan`` emits for ``scene``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fd.plan(scene)
    return [str(w.message) for w in caught if issubclass(w.category, fd.GeometryClippedWarning)]


def _rows(scene):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return [line for line in str(describe(scene)).splitlines()
                if line.startswith('structure')]


def test_a_region_past_a_pec_face_warns_with_the_share_lost():
    """``(0.4, 0.4, 0.2)`` at ``x = 0.4`` in a box to ``x = 0.5``: 0.1 of 0.4 is past the wall."""
    scene = _box((fd.Region(size=(0.4, 0.4, 0.2), center=(0.4, 0.0, 0.0)),))
    message, = _clipping(scene)
    assert '25.0%' in message and "'x+' (PEC)" in message and 'structures[0]' in message
    x, y, z = _rows(scene)
    assert x.endswith('<- clipped') and 'box.x=(0.2, 0.6) -> (0.2, 0.5)' in x
    assert not y.endswith('snapped') and not z.endswith('snapped'), (y, z)


def test_a_slab_past_a_pmc_face_warns_on_its_own_axis():
    """``x=(0.2, 0.9)`` against a wall at 0.5: 0.4 of 0.7 is lost, 57.1%."""
    scene = _box((fd.Slab(x=(0.2, 0.9)),),
                 fd.Boundaries(x=fd.PMC(), y=fd.PEC(), z=fd.PEC()))
    message, = _clipping(scene)
    assert '57.1%' in message and "'x+' (PMC)" in message


def test_a_smoothed_substrate_past_four_walls_warns_and_its_rows_say_clipped():
    """``fd.Box(size=(3, 3, 0.3))`` in a 2.4 box: ``(2.4/3)^2`` is kept, 36.0% lost."""
    scene = _box((fd.Box(size=(3.0, 3.0, 0.3)),), size=2.4, dx=0.1)
    message, = _clipping(scene)
    assert '36.0%' in message
    for face in ('x-', 'x+', 'y-', 'y+'):
        assert f"'{face}' (PEC)" in message
    x, y, z = _rows(scene)
    assert 'cuboid.x=(-1.5, 1.5) -> (-1.2, 1.2)   <- clipped' in x
    assert 'cuboid.y=(-1.5, 1.5) -> (-1.2, 1.2)   <- clipped' in y
    assert z.endswith('(-0.15, 0.15)'), z


def test_running_out_through_an_absorbing_face_or_across_a_period_does_not_warn():
    pml = fd.PML(cells=3, wavelength=2.0, allow_thin=True)
    through = _box((fd.Box(size=(3.0, 3.0, 0.3)),),
                   fd.Boundaries(x=pml, y=pml, z=fd.PEC()), size=2.4, dx=0.1)
    assert _clipping(through) == []
    lamellar = _box((fd.Region(size=(2.0, 0.2, 0.2)),),
                    fd.Boundaries(x=fd.Periodic(), y=fd.PEC(), z=fd.PEC()))
    assert _clipping(lamellar) == []


@pytest.mark.parametrize('size, x, lost', (((0.4, 0.2, 0.2), 0.5, '50.0%'),
                                           ((1.2, 0.2, 0.2), 0.2, '25.0%')))
def test_a_snapped_shape_across_one_periodic_seam_is_refused(size, x, lost):
    """Clipped, not wrapped: cells 16..24 of 20 lose 4 of 8; cells 2..26 lose 6 of 24.

    The second is wider than the period, so wrapped it would cover every cell; clipped it
    misses the first two. Both are refused, naming the share and the spellings that work.
    """
    scene = _box((fd.Region(size=size, center=(x, 0.0, 0.0)),),
                 fd.Boundaries(x=fd.Periodic(), y=fd.PEC(), z=fd.PEC()))
    with pytest.raises(ValueError) as caught:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fd.plan(scene)
    message = str(caught.value)
    assert 'Periodic x seam without spanning the period' in message and lost in message
    assert 'two structures' in message and 'span the cell' in message


@pytest.mark.parametrize('size', (1.0, 2.0))
def test_a_snapped_shape_that_spans_the_period_plans(size):
    """Exactly the cell (cells 0..20) and wider (-10..30) both span it: the lamellar spelling."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fd.plan(_box((fd.Region(size=(size, 0.2, 0.2)),),
                     fd.Boundaries(x=fd.Periodic(), y=fd.PEC(), z=fd.PEC())))


def test_the_rows_tell_a_snap_from_a_clip_and_do_not_mistake_round_off_for_either():
    """``-0.2`` asked comes back ``-0.20000000000000004`` from the cell arithmetic.

    The exact comparison printed ``<- snapped`` there (the C5 audit). A face between cells
    snaps; one past the wall clips; one both, both.
    """
    rows = _rows(_box((fd.Region(size=(0.4, 0.4, 0.4)),)))
    assert all(not row.endswith('snapped') and not row.endswith('clipped') for row in rows), rows
    x = _rows(_box((fd.Region(size=(0.446, 0.4, 0.4)),)))[0]
    assert x.endswith('<- snapped'), x
    x = _rows(_box((fd.Region(size=(0.446, 0.4, 0.4), center=(0.4, 0.0, 0.0)),)))[0]
    assert x.endswith('<- clipped and snapped'), x


def test_the_warning_is_public():
    assert 'GeometryClippedWarning' in fd.__all__
    assert issubclass(fd.GeometryClippedWarning, UserWarning)
