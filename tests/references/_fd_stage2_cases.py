"""The stage-2 scenes, samples and lowered arrays that ``fd_golden_stage2.npz`` and its test share.

One module, imported by both ``interface/freeze_fd_stage2.py`` (which writes the golden) and
``tests/test_fd_stage2_backends.py`` (which reads it), so the two cannot disagree about which
scene a key names. Nothing here solves anything. The stage-0 and stage-1 case lists are
imported rather than copied: the capture, the content digest and the relative gap are the same
definitions, and a second copy of them would be a second thing to keep in step. The sampling
rule is the one thing that is *not* shared, and :func:`sample_indices` says why.

**The scenes are built here and not by the ``tests/fd_helpers.py`` builders** (``dipole_box``,
``bloch_plane_wave``, ``tfsf_box``). Those helpers are shared with the physics gates and are
edited as those gates grow; a golden whose scenes moved every time a helper gained a default
would read as kernel drift. Every number below is spelled out once, here.

**The four scenes**, one per stage-2 source path, each under a second on every backend and long
enough that the source has actually driven the grid (the band centre's pulse peaks at
``t0 = 3.94``, step 273; every run reaches at least 60% of that envelope):

``dipole``
    Two point currents in a 10-cell vacuum box with an 8-cell PML all round (26^3): an electric
    ``Ez`` dipole on the default cubic B-spline stamp, off every node, and a magnetic ``Hx``
    dipole on the tent stamp with its own amplitude, phase and delay. Both slots (``before_E``
    and ``before_H``), both stamps, both row shifts (0.5 and 0.0) and the ``drive``, ``work``
    and ``field`` series. Real.
``plane_wave``
    A p-polarised ``PlaneWave`` at 20 degrees, ``phi = 30``, in a Bloch cell whose phases come
    from ``Bloch.from_source`` (both transverse axes Bloch, so the correction pattern is a
    genuine 2-D phase), with a lossless layer across z below the sheet so that a reflected
    wave crosses it as scattered field. Complex; a complex drive through the column. **Re-frozen
    2026-09-19 with ``--rebaseline``**: the first freeze pinned a launch defect -- the column's
    hard source wrote the continuum polarisation into the stored ``E_x`` and ``E_y`` without
    the half-cell stagger phase each carries along its own axis, and not transverse to the
    grid's discrete ``K``, so this conical (``phi = 30``) p wave was launched with an s part
    first order in ``dx``. The drive now writes the grid's own p wave
    (``wavebit/fd/_lower/sources.py::plane_wave_drive``). Only this scene moved: its fields,
    ``Record`` series and reference series by 2.7e-2 to 4.7e-2 of their peaks (the stagger
    phase ``k_x dx/2`` is 0.048 rad here, and the s part is gone), its energy by 7.9e-6; the
    lowered rows (pattern, waveform) and the other three scenes reproduced bit for bit, and
    the skeleton changed because the drive weights are now complex.
``tfsf``
    A ``TFSF`` box in vacuum travelling ``+x`` with E along z: all six faces of the box, and a
    column along x, so the propagation axis is not the one every other scene uses.
``tfsf_substrate``
    A ``-z`` TFSF box on a glass half-space (``eps = 2.25``) whose top lies inside the box and
    which runs out through the low z PML: the column's layered background and its 1-D CPML,
    and the driver column and one-sided sheet that feed a layered column. **Re-frozen
    2026-09-19 with ``--rebaseline``**: the first freeze pinned the hard-driven layered column,
    whose hard source reflected the substrate's echo back into the box (a known defect,
    ``xfail`` at the time), and the glass top one cell inside the 6-cell box put that echo in
    every array of this scene. Only this scene's arrays moved (13-29% of their peaks); the
    other three reproduced bit for bit, with new skeletons only because ``ColumnTerm`` and
    ``Correction`` gained fields.

**What is frozen.** Per scene and backend, the stage-0 set -- every ``Record`` series, the final
E and H at eight cells per component (:func:`sample_indices`), the whole-volume energy -- and, new here, **every array of
every source's reference series** (``incident``/``incident_H`` for a sheet or a box, ``drive``,
``work``, ``field`` for a dipole): those are what stage 3 will normalise by, and a moved
reference is a moved answer even when the field is unchanged. Per scene and *once*, from the
NumPy lowering, the numbers stage 2 added to the params: each dipole's stamp weights, the Bloch
phase pattern of a PlaneWave, each source's three waveform scalars (``f0``, ``tau``, ``t0``),
and the waveform table the kernel reads its drive from. They are backend-independent by
construction, so a failure that touches only them says the lowering moved, and one that touches
only the per-backend rows says the step moved under an unchanged lowering.
"""
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                          # tests/references, for the earlier cases
sys.path.insert(0, str(HERE.parent))                   # tests/, for fd_helpers

import wavebit.fd as fd                                                       # noqa: E402
from _fd_stage0_cases import (BACKENDS, capture as _capture_fields,          # noqa: E402,F401
                              content_digest, relative_gap)
from fd_helpers import to_numpy                                              # noqa: E402
from wavebit.fd._kernel.sources import waveform_table                        # noqa: E402
from wavebit.fd._runtime import runtime                                      # noqa: E402

#: The committed golden.
GOLDEN = HERE / 'fd_golden_stage2.npz'

#: Gate 6's budget, the same 1e-12 of an array's peak the stage-0 and stage-1 scenes are held
#: to. Measured 2026-09-19 on these four scenes, worst array of each (fields, ``Record`` series
#: and source series alike): Torch is bit-identical to NumPy on the two TFSF scenes and 2.1e-15
#: / 3.3e-15 off on the dipole / PlaneWave; JAX sits 8.3e-16 to 4.0e-15 off NumPy. The budget is
#: 250x above the worst of that and twelve decades below what the two mutants in the gate do
#: (a stamp normalised by its sum of squares: 1.9; one TFSF face with its sign flipped: 1.2).
TOL = 1e-12

#: The grid, and the one band every source uses: centre wavelength 1, spectral sigma 0.15 f0,
#: so the shortest wavelength is 0.642 -- 12.8 cells in vacuum at this dx.
DX = 0.05
BAND = dict(wavelength=1.0, df=0.15)
PML_CELLS = 8

#: The substrate of the two layered scenes. At the band's short edge glass is 8.6 cells per
#: wavelength, which the mesh check warns about; the warning is asserted by the gate rather
#: than avoided with a finer grid, because these scenes are a drift detector and cost is the
#: constraint.
GLASS = 2.25

#: What each layered scene warns about on purpose, asserted by the gate.
EXPECTED_WARNINGS = {
    'dipole': (),
    'plane_wave': ('CoarseMeshWarning',),
    'tfsf': (),
    'tfsf_substrate': ('CoarseMeshWarning', 'AbsorberProximityWarning'),
}

#: Which source series a kind of source fills.
SERIES_FIELDS = ('incident', 'incident_H', 'drive', 'work', 'field')


def band():
    return fd.Band(**BAND)


def _layer():
    # 8 cells rather than the measured 12: every scene here is a drift detector, and the
    # thinner layer halves the cost of the dipole and TFSF boxes. The wavelength is left out on
    # purpose -- stage 2 infers it from the sources' shared band centre.
    return fd.PML(cells=PML_CELLS, allow_thin=True)


def _cube(interior):
    return fd.Domain(size=((interior + 2 * PML_CELLS) * DX,) * 3)


def dipole_scene():
    """Two dipoles, electric and magnetic, bspline and tent, in a 26^3 PML box."""
    j = fd.Dipole('j', center=(0.013, -0.02, 0.007), component='Ez', band=band())
    m = fd.Dipole('m', center=(-0.1, 0.08, -0.05), component='Hx', kind='magnetic',
                  band=band(), stamp='tent', amplitude=0.5, phase=0.7, delay=0.3)
    return fd.Scene(domain=_cube(10), boundaries=fd.Boundaries.open(_layer()),
                    resolution=fd.Resolution(dx=DX), run=fd.Run(steps=250), sources=(j, m),
                    monitors=(fd.Record('Ez', at=(0.15, 0.0, 0.05), component='Ez'),
                              fd.Record('Hy', at=(0.1, -0.1, 0.0), component='Hy')))


def plane_wave_source():
    return fd.PlaneWave('pw', z=0.25, direction='-z', theta=20.0, phi=30.0, polarization='p',
                        band=band(), amplitude=0.8, phase=0.4)


def plane_wave_scene():
    """A 20-degree p PlaneWave in a 4 x 3 x 36 Bloch cell, over a glass layer."""
    pw = plane_wave_source()
    return fd.Scene(domain=fd.Domain(size=(4 * DX, 3 * DX, (20 + 2 * PML_CELLS) * DX)),
                    boundaries=fd.Boundaries(**fd.Bloch.from_source(pw), z=_layer()),
                    resolution=fd.Resolution(dx=DX), run=fd.Run(steps=300), sources=(pw,),
                    structures=(fd.Structure(fd.Slab(z=(-0.25, -0.1)), fd.Material(eps=GLASS)),),
                    monitors=(fd.Record('above', at=(0.0, 0.0, 0.4), component='Ex'),
                              fd.Record('between', at=(0.0, 0.0, 0.0), component='Ey'),
                              fd.Record('below', at=(0.0, 0.0, -0.4), component='Hy')))


def tfsf_box():
    return fd.Region(size=(0.3, 0.3, 0.3))


def tfsf_scene(substrate):
    """A TFSF box 6 cells on a side in a 28^3 PML box: ``+x`` in vacuum, or ``-z`` on glass."""
    if substrate:
        source = fd.TFSF('t', box=tfsf_box(), direction='-z', polarization='x', band=band())
        structures = (fd.Structure(fd.Slab(z=(-2.0, -0.05)), fd.Material(eps=GLASS)),)
        monitors = (fd.Record('above', at=(0.0, 0.0, 0.1), component='Ex'),
                    fd.Record('inside', at=(0.0, 0.0, -0.1), component='Hy'))
    else:
        source = fd.TFSF('t', box=tfsf_box(), direction='+x', polarization='z', band=band())
        structures = ()
        monitors = (fd.Record('inside', at=(0.0, 0.0, 0.0), component='Ez'),
                    fd.Record('H', at=(0.05, 0.0, 0.0), component='Hy'))
    return fd.Scene(domain=_cube(12), boundaries=fd.Boundaries.open(_layer()),
                    resolution=fd.Resolution(dx=DX), run=fd.Run(steps=250), sources=(source,),
                    structures=structures, monitors=monitors)


def scenes():
    """``{name: scene}`` -- the four stage-2 scenes; see the module docstring for each."""
    return {'dipole': dipole_scene(), 'plane_wave': plane_wave_scene(),
            'tfsf': tfsf_scene(False), 'tfsf_substrate': tfsf_scene(True)}


def sample_indices(shape):
    """``(S, 4)`` cell indices ``(c, k, j, i)`` to freeze of a ``(3, nz, ny, nx)`` field.

    **Not the stage-1 rule, on purpose.** That rule picks the ends of the long axis, cells
    inside each absorbing layer and a third of the way in -- in a TFSF scene every one of
    those is in the scattered-field region, where the field is exactly 0.0 on NumPy and Torch
    and 1e-16 on JAX. A frozen array that is nothing but round-off is a pin on round-off: its
    own peak is the scale, so a harmless reordering moves it by O(1) (measured: the first
    freeze of this archive had exactly that, on all eight final-field arrays of both TFSF
    scenes). So the picks here cluster on the cell nearest the domain centre -- inside every
    TFSF box, next to every dipole, across the sheet's line -- seven cells within two of it
    along each axis, and one far cell three from the low corner of the long axis (inside the
    PML), clipped into range on a line. Eight per component, as the earlier stages freeze.
    """
    _, nz, ny, nx = shape

    def take(n, offsets, far):
        c = n // 2
        return [min(max(c + o, 0), n - 1) for o in offsets] + [min(far, n - 1)]

    ks = take(nz, (0, 1, 0, -1, 2, -2, 1), 3)
    js = take(ny, (0, 0, 1, -1, -2, 1, 2), ny // 2)
    xs = take(nx, (0, 0, 0, 1, -1, 2, -2), nx // 3)
    return np.array([(c, k, j, i) for c in range(3) for k, j, i in zip(ks, js, xs)],
                    dtype=np.int64)


def capture(res, samples):
    """The stage-0 capture, plus every array of every source's reference series."""
    out = _capture_fields(res, samples)
    for name, series in res.sources.items():
        for field in SERIES_FIELDS:
            value = getattr(series, field)
            if value is not None:
                out[f'source__{name}__{field}'] = np.ascontiguousarray(to_numpy(value))
    return out


def lowered(scene, plan):
    """The arrays frozen once per scene from the NumPy lowering: what stage 2 added to params.

    The stamp weights, the Bloch pattern, the waveform scalars and the waveform table. All of
    it is small: the table, ``(rows, steps)``, is the largest at ``(2, 300)``.
    """
    rt = runtime('numpy')
    params = fd.lower(scene, plan, rt)
    out = {}
    for name, entry in params['sources'].items():
        for key in ('f0', 'tau', 't0', 'amplitude', 'phase', 'delay'):
            out[f'lowered__{name}__{key}'] = np.array(np.asarray(entry[key]))
        for key in ('stamp', 'pattern'):
            if key in entry:
                out[f'lowered__{name}__{key}'] = np.ascontiguousarray(np.asarray(entry[key]))
    out['lowered__waveform'] = np.ascontiguousarray(np.asarray(waveform_table(plan, params, rt)))
    return out
