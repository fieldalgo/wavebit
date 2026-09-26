"""Stage 2, gate 4 -- the TFSF box: the leak, the interior, the reference series, the substrate.

**What a TFSF box promises.** Inside the box the stored field is the total field, outside it
the scattered field alone, and the incident field comes from a column along the propagation
axis stepped by the kernel's own update (``wavebit/fd/_kernel/sources.py``). On an axis the
column IS the 3-D grid restricted to a line -- the same ``dx``, ``dt``, curl, coefficients and
CPML -- so in a laterally uniform (or layered-along-the-axis) background the incident field is
an exact discrete solution and the corrections on the six faces cancel it outside the box to
round-off. That is the gate:

1. **The leak.** An empty box in vacuum, every axis direction and both polarisations: the peak
   of every component outside the box's one-cell correction layer, over every step (a ring of
   ``fd.Record`` probes) and in the final fields, is at round-off relative to the peak inside.
   Measured 0.0 exactly on NumPy. "Inside" is read off the **main grid** (a ``Record`` at the
   box centre), not off the column's own series, and it must carry the incident pulse: a box
   that injects nothing leaks nothing too, and a leak gate read from the column could not
   tell (a review found every one of these cases passing with every correction zeroed).
2. **The interior is the column.** Every E component inside the box equals the column's E
   broadcast across, at the same step; every H component equals the column's H of the run one
   step shorter -- because the column runs half a step ahead (its memory holds ``E(n)`` and
   ``H(n+1/2)`` at the start of step ``n``), which this reads directly off the state.
3. **The reference series.** ``res.sources['t'].incident`` / ``incident_H`` equal a
   ``fd.Record`` of the main grid's total field at the entry plane and the half plane inside
   it, at the same times.
4. **The substrate.** A ``Slab`` across the propagation axis crossing the box (running out
   through the low PML) is the column's layered background, and the box still leaks at
   round-off on every backend.
5. **The discrete Airy field.** Inside the box over the substrate the field is the layered
   background's own solution: incident plus the half-space reflection above the interface,
   the transmitted wave below, with ``r`` and ``T`` from the semi-discrete Helmholtz matching
   of ``tests/test_fd_airy.py`` (its ``discrete_wavenumber``, imported, with the two-node
   matching restated for a half space). This was a kernel defect until the layered column got
   its driver: the column was driven by a *hard* source one cell upstream of the entry plane
   (``E_t(ks) := w``), a hard source is a perfect mirror (r = -1) for the column's own upward
   wave -- the substrate's reflection -- so the column was a cavity between the source and
   the substrate and the interior missed the Airy field by ``O(|r|)`` (0.28 at worst, with
   ``|r| = 0.21``), while the leak stayed at round-off (the box is self-consistent with
   whatever its column holds). Now a layered column is fed through a one-sided sheet from a
   uniform, hard-driven driver column (``wavebit/fd/_kernel/sources.py``), and the interior
   is the ``PlaneWave``-driven physical field at every step to round-off (measured 6.7e-16):
   the two scenes launch the same wave through the same sheet on the same plane. The old
   column is kept as a mutant below, and the Airy gate catches it.
6. **The refusal.** A structure that crosses a TFSF face and is not a slab across the
   propagation axis -- not in the column's background -- is refused; the same slab moved clear
   of the box is accepted and the empty box still leaks nothing onto it.
   Over a substrate the wave must arrive from one uniform medium -- the layered column's
   driver is that medium -- so a layer between the entry plane and the incident side's face
   is refused, and the same box with the layer moved past the exit side is accepted.
7. **A Bloch cell.** A TFSF box in a cell with a Bloch face runs complex, and so does its
   column: ``amplitude=1j`` gives ``1j`` times the real-amplitude run on every backend. (The
   column used to stay real, and the complex waveform written into it failed on NumPy and
   Torch and dropped its imaginary part on JAX -- a source that vanished with no error.)
8. **Oblique.** Refused at ``fd.TFSF(...)`` citing the measurement (the build did not allow
   any oblique angle, so there is no leakage table to pin); the refusal is asserted here.

**The mutants**, each measured by the gate that exists to catch it:

``one face's correction with the wrong sign``
    every one of the six faces, in turn: the leak rises to O(1) of the incident peak (0.85 on
    ``x-``, 0.78 on the entry face ``z+``, 1.1 on the exit face ``z-``, for a 0.97 peak).
``the column stepped half a step late``
    its advance does the H half of step ``n`` before the E half, so ``before_E`` reads
    ``H(n-1/2)`` instead of ``H(n+1/2)``: the leak rises to 4.1e-2 of the incident peak.
``the column's background missing the substrate``
    the column's params lowered from the same scene with no slab: the substrate leak rises to
    0.48 of the incident peak.
``the layered column hard-driven`` (the defect the driver fixed)
    the driver and the sheet removed, so the column holding the substrate is driven by the hard
    source itself: the Airy error rises from below 1e-3 to 0.28, and the interior departs from
    the PlaneWave-driven field by O(0.1) once the echo off the hard source is back.

Cost: 34^3 cells for a few hundred steps per run (about half a second on NumPy); the Airy
scenes are 6 x 6 x 84 with a Periodic transverse cell. JAX and Torch run one substrate case.
"""
import dataclasses
import math
import warnings

import numpy as np
import pytest

import wavebit.fd as fd
from fd_helpers import (backends, band, box_node_range, peak_of, ring_monitors, run_mutated,
                        scattered_region_peak, tfsf_box, to_numpy)
from test_fd_airy import discrete_wavenumber, spectrum
from wavebit.fd._kernel.curl import curl
from wavebit.fd._kernel.grid import OFFSETS
from wavebit.fd._kernel.sources import ColumnTerm, _Scoped, box_corrections
from wavebit.fd._lower import lower
from wavebit.fd._lower.sources import column_scene
from wavebit.fd._run import execute
from wavebit.fd._runtime import runtime

#: The small box every leak case runs: a 0.3 box, 0.1 of scattered-field clearance, 12 PML
#: cells, 20 cells per centre wavelength -- 34^3 cells.
SMALL = dict(box=(0.3, 0.3, 0.3), margin=0.1, pml=12)

#: Steps for the leak cases: the pulse (centre at about 3.7 time units) is inside the box.
STEPS = 300

#: Round-off, relative to the peak incident field inside the box. Measured 0.0 on NumPy.
LEAK_TOL = 1e-12

#: How far a mutant must rise above :data:`LEAK_TOL` to be called caught.
CAUGHT = 1e-3

#: Every axis direction with both transverse polarisations.
CASES = [(d, p) for d in ('-z', '+z', '-x', '+x', '-y', '+y')
         for p in 'xyz' if p != d[1]]


def _scene(direction='-z', polarization='x', steps=STEPS, substrate=None, **kw):
    """A small TFSF box with a ring of probes just outside its correction layer."""
    kw = dict(SMALL, **kw)
    dx = band().wavelength / 20
    out = kw['box'][0] / 2 + 1.6 * dx
    points = []
    for a in range(3):
        for s in (-1, 1):
            p = [0.0, 0.0, 0.0]
            p[a] = s * out
            points.append(p)
    points += [[s * out, t * out, u * out] for s in (-1, 1) for t in (-1, 1) for u in (-1, 1)]
    centre = fd.Record('centre', at=(0.0, 0.0, 0.0), component='E' + polarization)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return tfsf_box(direction=direction, polarization=polarization, steps=steps,
                        substrate=substrate,
                        monitors=ring_monitors(points, prefix='ring') + (centre,), **kw)


def _leak(res, source):
    """``(outside peak over time and final fields, inside peak)``, both off the main grid.

    ``inside`` is the peak of the polarisation's E at the box centre over every step -- the
    main grid's, so a box that injects nothing reads 0 here and fails each gate's ``inside``
    floor instead of passing its leak line (the column's own series, which this read before,
    is full whatever the box does with it).
    """
    outside = max(peak_of(res, 'ring'), scattered_region_peak(res, source))
    inside = float(np.max(np.abs(to_numpy(res['centre'].values))))
    assert np.isfinite(inside)
    return outside, inside


def _run(scene, backend='numpy', **kw):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fd.run(scene, backend=backend, **kw)


# ---------------------------------------------------------------------------------------------
# 1. The leak
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('direction,polarization', CASES)
def test_an_empty_box_leaks_at_round_off_for_every_axis_direction_and_polarisation(
        direction, polarization):
    scene = _scene(direction, polarization)
    res = _run(scene)
    outside, inside = _leak(res, scene.sources[0])
    assert inside > 0.5, f'the pulse is not inside the box ({inside:.3g})'
    assert outside <= LEAK_TOL * inside, (direction, polarization, outside, inside)


# ---------------------------------------------------------------------------------------------
# 2. The interior is the column
# ---------------------------------------------------------------------------------------------

def _state(scene, plan=None):
    """The final state of a NumPy run, copied out of the runtime's buffers."""
    rt = runtime('numpy')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        p = fd.plan(scene) if plan is None else plan
        state = execute(p, lower(scene, p, rt), rt)
    col = state['mem'][scene.sources[0].name]['col']
    return p, {'E': np.array(state['E']), 'H': np.array(state['H']),
               'col_E': np.array(col['E']), 'col_H': np.array(col['H'])}


def _inside(F, field, c, lo, hi, P=None):
    """Component ``c`` of ``F`` at its Yee points inside the box ``[lo, hi]`` (node indices).

    A node coordinate is inside for ``lo <= i <= hi``, a half-cell one for ``lo <= i < hi``.
    With ``P`` given, ``F`` is a column: the transverse axes are its single cell.
    """
    index = []
    for a in (2, 1, 0):
        if P is not None and a != P:
            index.append(slice(0, 1))
            continue
        half = OFFSETS[field][c][a]
        index.append(slice(lo[a], hi[a] + (0 if half else 1)))
    return F[c][tuple(index)]


@pytest.mark.parametrize('direction,polarization', [('-z', 'x'), ('+x', 'y'), ('-y', 'z')])
def test_the_total_field_inside_the_box_is_the_column_at_every_yee_point(
        direction, polarization):
    scene = _scene(direction, polarization)
    longer = _scene(direction, polarization, steps=STEPS + 1)
    plan, s = _state(scene)
    _, s1 = _state(longer)
    source = scene.sources[0]
    lo, hi = box_node_range(plan, source)
    P = source.axis
    scale = float(np.abs(s['col_E']).max())
    assert scale > 0.5
    for c in range(3):
        # E: both hold E(steps).
        main = _inside(s['E'], 'E', c, lo, hi)
        col = _inside(s['col_E'], 'E', c, lo, hi, P)
        assert np.abs(main - col).max() <= 1e-13 * scale, ('E', c)
        # H: the column is half a step ahead, so its H(steps + 1/2) is the main grid's
        # final H of the run one step longer.
        main = _inside(s1['H'], 'H', c, lo, hi)
        col = _inside(s['col_H'], 'H', c, lo, hi, P)
        assert np.abs(main - col).max() <= 1e-13 * scale, ('H', c)
    # and it is not the same step: the main grid's own final H lags the column by one step.
    main = _inside(s['H'], 'H', 3 - P - source.pol, lo, hi)
    col = _inside(s['col_H'], 'H', 3 - P - source.pol, lo, hi, P)
    assert np.abs(main - col).max() > 1e-3 * scale


# ---------------------------------------------------------------------------------------------
# 3. The reference series
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('direction,polarization', [('-z', 'x'), ('+z', 'y'), ('-x', 'y'),
                                                    ('+y', 'x')])
def test_the_reference_series_is_the_total_field_at_the_entry_plane(direction, polarization):
    probe = _scene(direction, polarization)
    source = probe.sources[0]
    P, pol = source.axis, source.pol
    h = 3 - P - pol
    res0 = _run(probe)
    ref = res0.sources['t']
    at_E, at_H = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    at_E[P], at_H[P] = ref.plane, ref.plane_H
    E_name, H_name = 'E' + 'xyz'[pol], 'H' + 'xyz'[h]
    scene = _scene(direction, polarization)
    scene = dataclasses.replace(scene, monitors=(
        fd.Record('e', at=tuple(at_E), component=E_name),
        fd.Record('h', at=tuple(at_H), component=H_name)))
    res = _run(scene)
    ref = res.sources['t']
    assert ref.kind == 'TFSF'
    np.testing.assert_array_equal(res['e'].t, ref.t)
    np.testing.assert_array_equal(res['h'].t, ref.t_H)
    incident = to_numpy(ref.incident)
    incident_H = to_numpy(ref.incident_H)
    assert incident.shape == (res.plan.steps, 3)
    scale = float(np.abs(incident[:, pol]).max())
    assert scale > 0.5
    assert np.abs(to_numpy(res['e'].values) - incident[:, pol]).max() <= 1e-13 * scale
    assert np.abs(to_numpy(res['h'].values) - incident_H[:, h]).max() <= 1e-13 * scale
    # In vacuum on an axis only the polarisation and its partner carry anything.
    others = [c for c in range(3) if c != pol]
    assert np.abs(incident[:, others]).max() == 0.0
    # the H partner: h = k x e, with Z0 H = E for a travelling wave -- same sign as E when
    # (k, e, h) is right-handed, read off the series at its peak (half a step apart, so a
    # loose check of the sign only).
    k = np.zeros(3)
    k[P] = source.sign
    e = np.zeros(3)
    e[pol] = 1.0
    partner = np.cross(k, e)[h]
    i = int(np.argmax(np.abs(incident[:, pol])))
    assert np.sign(incident_H[i, h]) == np.sign(partner * incident[i, pol])



# ---------------------------------------------------------------------------------------------
# 4. The substrate: part of the column's background, and the box still leaks nothing
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('direction,polarization', [('-z', 'x'), ('+z', 'y'), ('-x', 'z')])
def test_a_substrate_crossing_the_box_is_the_columns_background_and_leaks_nothing(
        direction, polarization):
    scene = _scene(direction, polarization, substrate=(2.25, 0.0))
    res = _run(scene)
    outside, inside = _leak(res, scene.sources[0])
    assert inside > 0.3
    assert outside <= LEAK_TOL * inside, (direction, polarization, outside, inside)


@pytest.mark.parametrize('backend', backends())
def test_the_substrate_box_leaks_nothing_on_every_backend(backend):
    scene = _scene('-z', 'x', steps=200, substrate=(2.25, 0.0))
    res = _run(scene, backend=backend)
    outside, inside = _leak(res, scene.sources[0])
    assert inside > 0.1
    assert outside <= LEAK_TOL * inside, (backend, outside, inside)
    reference = _run(scene)
    assert np.abs(to_numpy(res.final.E) - to_numpy(reference.final.E)).max() <= 1e-13


# ---------------------------------------------------------------------------------------------
# 5. The discrete Airy field inside the box
# ---------------------------------------------------------------------------------------------

AIRY_DX = 0.05
AIRY_WAVELENGTHS = (0.8, 0.9, 1.0, 1.15, 1.3)
#: Measured 3.5e-4 to 7.8e-4 for the PlaneWave-driven control: the floor is the 12-cell PML
#: (the substrate runs out through it) and the pulse's truncation.
AIRY_TOL = 2e-3


def _airy_scene(kind, eps, steps=1600):
    """A 6 x 6 Periodic cell, PML on z, a wave in -z from z = 1 onto a half space below z = 0.

    ``kind='tfsf'``: a TFSF box ``z in [-1, 1]`` (two cells across); ``kind='pw'``: a
    ``PlaneWave`` sheet at ``z = 1``, whose column is uniform -- the physical control. Both
    drive ``Ey`` with the same waveform at the same column index, so their incident waves are
    identical. Probes ``Ey`` on the 31 nodes ``z = -0.75 .. 0.75``.
    """
    dx, pml = AIRY_DX, 12
    nz = int(round(3.0 / dx)) + 2 * pml
    b = band()
    if kind == 'tfsf':
        src = fd.TFSF('t', box=fd.Region(size=(2 * dx, 2 * dx, 2.0), center=(0.0, 0.0, 0.0)),
                      direction='-z', polarization='y', band=b)
    else:
        src = fd.PlaneWave('t', z=1.0, direction='-z', polarization='s', band=b)
    structures = () if eps is None else (
        fd.Structure(fd.Slab(z=(-nz * dx, 0.0)), fd.Material(eps=eps)),)
    monitors = tuple(fd.Record(f'z{k}', at=(0.0, 0.0, k * dx), component='Ey')
                     for k in range(-15, 16))
    return fd.Scene(domain=fd.Domain(size=(6 * dx, 6 * dx, nz * dx)),
                    boundaries=fd.Boundaries(x=fd.Periodic(), y=fd.Periodic(),
                                             z=fd.PML(cells=pml)),
                    resolution=fd.Resolution(dx=dx), run=fd.Run(steps=steps),
                    structures=structures, sources=(src,), monitors=monitors)


def airy_half_space(omega, n, dx, dt):
    """``(r, T)`` of a half space of index ``n`` on the semi-discrete Yee grid.

    The recurrence of ``tests/test_fd_airy.py``'s :func:`airy_discrete`, with the eps step at
    node ``j = 0`` (the first node inside the medium) and no far face: ``u^j + r u^-j`` for
    ``j <= 0`` and ``T v^j`` for ``j >= -1``, matched at ``j = -1, 0``::

        u^-1 + r u = T v^-1,   1 + r = T   ->   r = (1/v - 1/u) / (u - 1/v)

    ``j`` counts nodes along the propagation direction.
    """
    u = np.exp(1j * discrete_wavenumber(omega, 1.0, dx, dt) * dx)
    v = np.exp(1j * discrete_wavenumber(omega, n, dx, dt) * dx)
    r = (1 / v - 1 / u) / (u - 1 / v)
    return r, 1 + r, u, v


def _airy_error(kind, run=None):
    """``max |measured / incident - closed form|`` over the probes and the band.

    ``run(scene)`` runs a scene (``_run`` by default; the mutant passes its own). A run that
    drives nothing would make every ratio ``0/0``, and ``max(0.0, nan)`` is ``0.0`` -- so the
    vacuum spectrum is held above a tenth of its band peak and every ratio to be finite.
    """
    run = _run if run is None else run
    sub, vac = run(_airy_scene(kind, 2.25)), run(_airy_scene(kind, None))
    plan = sub.plan
    first_medium = fd.Slab(z=(-plan.nz * AIRY_DX, 0.0)).cells(plan.grid)[1] - 1
    omegas = [2 * math.pi / wavelength for wavelength in AIRY_WAVELENGTHS]
    worst = 0.0
    for name in sub:
        observer = next(o for o in plan.observers if o.name == name)
        j = first_medium - observer.k                         # along -z
        incident = spectrum(vac[name], omegas)
        measured = spectrum(sub[name], omegas)
        assert np.abs(incident).max() > 0.0, name
        assert np.all(np.abs(incident) > 0.1 * np.abs(incident).max()), name
        for omega, a, b in zip(omegas, measured, incident):
            r, T, u, v = airy_half_space(omega, 1.5, plan.dx, plan.dt)
            ratio = a / b
            assert np.isfinite(ratio), (name, omega)
            expected = 1 + r * u ** (-2 * j) if j <= 0 else T * (v / u) ** j
            worst = max(worst, abs(ratio - expected))
    return worst


def test_the_airy_closed_form_holds_for_the_physical_control():
    """The same measurement driven by a PlaneWave sheet (a uniform column that cannot echo):
    the half-space formula and the probe reading are right, so the TFSF case measures TFSF."""
    assert _airy_error('pw') <= AIRY_TOL


def test_inside_the_box_the_field_over_the_substrate_is_the_discrete_airy_field():
    assert _airy_error('tfsf') <= AIRY_TOL


def _hard_driven(scene):
    """The mutant plan: the layered column driven by the hard source itself (the old column)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        plan = fd.plan(scene)
    term = next(t for t in plan.terms if t.name == 't')
    assert term.driver is not None and term.sheet
    old = dataclasses.replace(term, driver=None, sheet=(), corrections=tuple(
        dataclasses.replace(c, driver=False) for c in term.corrections))
    return _replace_term(plan, 't', old)


def _run_hard_driven(scene):
    if not scene.structures:                   # the vacuum run has no driver to take away
        return _run(scene)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return run_mutated(scene, plan=_hard_driven(scene))


def test_the_tfsf_interior_is_the_physical_field_at_every_step():
    """TFSF and the PlaneWave-driven control agree to round-off at every step and probe.

    Both launch the same wave (the same waveform, hard-driven at the same column index in a
    uniform column) through the same one-sided sheet on the same plane -- the TFSF's inside its
    layered column, the PlaneWave's on the main grid -- so they are the same discrete field.
    The hard-driven mutant localises what the driver removed: it agrees until the substrate's
    reflection, bounced off the hard source one cell above the box, can reach the probe, and
    departs at O(0.1) afterwards. The bounce path from the start of the drive at
    ``z_s = 1 + dx``: down to the interface, up to ``z_s``, down to the probe.
    """
    scene = _airy_scene('tfsf', 2.25)
    tf, pw = _run(scene), _run(_airy_scene('pw', 2.25))
    hard = _run_hard_driven(scene)
    dt = tf.plan.dt
    z_s = 1.0 + AIRY_DX
    for k in (15, 10, 0, -10):
        a = to_numpy(tf[f'z{k}'].values)
        b = to_numpy(pw[f'z{k}'].values)
        m = to_numpy(hard[f'z{k}'].values)
        assert np.all(np.isfinite(a)) and np.abs(b).max() > 0.3
        assert np.abs(a - b).max() <= 1e-12 * np.abs(b).max(), k
        z = k * AIRY_DX
        down = z_s - z if z >= 0 else z_s + 1.5 * (-z)
        echo = 2 * z_s + down
        t = (np.arange(a.size) + 1.0) * dt
        early = t < echo - 0.8
        assert early.sum() > 50
        assert np.abs(m - b)[early].max() <= 1e-12, k
        assert np.abs(m - b).max() > 0.05, k


def test_mutant_the_layered_column_hard_driven_misses_the_airy_field():
    assert _airy_error('tfsf', run=_run_hard_driven) > 0.1


# ---------------------------------------------------------------------------------------------
# 6. The refusal: a structure on the surface that the column does not carry
# ---------------------------------------------------------------------------------------------

def test_a_structure_crossing_a_face_outside_the_column_background_is_refused():
    base = _scene('-z', 'x')
    dx = base.resolution.dx
    glass = fd.Material(eps=2.25)
    crossing = fd.Structure(fd.Slab(x=(0.1, 1.0)), glass)      # across x, out through x+
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(ValueError) as e:
            fd.plan(dataclasses.replace(base, structures=(crossing,)))
    message = str(e.value)
    assert "crosses the faces of TFSF('t')" in message and 'slab across z' in message
    # The accepted spellings: the slab across the propagation axis (the background) ...
    across = fd.Structure(fd.Slab(z=(-0.1, 0.05)), glass)
    res = _run(dataclasses.replace(base, structures=(across,)))
    outside, inside = _leak(res, base.sources[0])
    assert inside > 0.3 and outside <= LEAK_TOL * inside
    # ... or the same x slab clear of the correction layer, onto which nothing leaks.
    clear = fd.Structure(fd.Slab(x=(0.15 + 2 * dx, 1.0)), glass)
    res = _run(dataclasses.replace(base, structures=(clear,)))
    outside, inside = _leak(res, base.sources[0])
    assert inside > 0.5 and outside <= LEAK_TOL * inside


def test_a_layer_on_the_incident_side_is_refused():
    """The driver is one medium: a slab between the entry plane and the incident face is not."""
    base = _scene('-z', 'x')
    glass = fd.Material(eps=2.25)
    above = fd.Structure(fd.Slab(z=(0.15, 0.2)), glass)           # between the box and z+
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(ValueError) as e:
            fd.plan(dataclasses.replace(base, structures=(above,)))
    message = str(e.value)
    assert "TFSF('t') propagates in -z" in message and 'structures[0]' in message
    assert 'the background' in message
    # the accepted spelling: the same slab on the exit side, where it is downstream contrast
    below = fd.Structure(fd.Slab(z=(-0.2, -0.15)), glass)
    res = _run(dataclasses.replace(base, structures=(below,)))
    outside, inside = _leak(res, base.sources[0])
    assert inside > 0.5 and outside <= LEAK_TOL * inside


def test_a_wave_arriving_from_inside_the_substrate_is_driven_in_the_substrate():
    """A ``+z`` box over a substrate that runs out through z-: the driver is glass."""
    scene = _scene('+z', 'y', substrate=(2.25, 0.0))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        plan = fd.plan(scene)
    (term,) = plan.terms
    assert term.driver is not None and not term.driver.spans     # uniform: no structure
    params = lower(scene, plan, runtime('numpy'))
    be = np.asarray(params['sources']['t']['driver']['be'])
    assert np.allclose(be, plan.dt / plan.dx / 2.25)             # glass everywhere


def _bloch_tfsf(amplitude, steps=320):
    dx = band().wavelength / 20
    src = fd.TFSF('t', box=fd.Region(size=(0.3, 0.3, 0.3)), direction='-z', polarization='x',
                  band=band(), amplitude=amplitude)
    return fd.Scene(domain=fd.Domain(size=(0.6, 0.6, 0.5 + 24 * dx)),
                    boundaries=fd.Boundaries(x=fd.Bloch(k=1.0), y=fd.Periodic(),
                                             z=fd.PML(cells=12)),
                    resolution=fd.Resolution(dx=dx), run=fd.Run(steps=steps), sources=(src,),
                    monitors=(fd.Record('centre', at=(0.0, 0.0, 0.0), component='Ex'),))


@pytest.mark.parametrize('backend', backends())
def test_a_tfsf_box_in_a_bloch_cell_is_linear_in_a_complex_amplitude(backend):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        plan = fd.plan(_bloch_tfsf(1.0))
    (term,) = plan.terms
    assert plan.dtype == 'complex128' and term.sub.dtype == 'complex128'
    scene = _bloch_tfsf(1j)
    real, imag = _run(_bloch_tfsf(1.0), backend), _run(scene, backend)
    E1, Ej = to_numpy(real.final.E), to_numpy(imag.final.E)
    scale = float(np.abs(to_numpy(real['centre'].values)).max())
    assert scale > 0.5                                        # the box injected
    assert np.abs(Ej - 1j * E1).max() <= 1e-13 * scale
    assert np.abs(to_numpy(imag['centre'].values)
                  - 1j * to_numpy(real['centre'].values)).max() <= 1e-13 * scale
    assert scattered_region_peak(imag, scene.sources[0]) <= LEAK_TOL * scale


def test_oblique_tfsf_is_refused_so_there_is_no_leakage_table_to_pin():
    b = band()
    box = fd.Region(size=(0.3, 0.3, 0.3))
    for kwargs in (dict(direction='-z', theta=45.0), dict(direction=(0.6, 0.0, -0.8))):
        with pytest.raises(ValueError) as e:
            fd.TFSF('t', box=box, polarization='x', band=b, **kwargs)
        assert 'oblique TFSF box is refused in v1' in str(e.value)
        assert 'fd.PlaneWave' in str(e.value)


# ---------------------------------------------------------------------------------------------
# 7. The mutants
# ---------------------------------------------------------------------------------------------

def _replace_term(plan, name, term):
    return plan.with_static(terms=tuple(term if t.name == name else t for t in plan.terms))


def _flipped(scene, face):
    """The plan with every correction of one face (``'x-'`` ... ``'z+'``) sign-flipped."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        plan = fd.plan(scene)
    source = scene.sources[0]
    term = next(t for t in plan.terms if t.name == source.name)
    lo, hi = box_node_range(plan, source)
    listed = box_corrections(lo, hi, (source.pol,), (3 - source.axis - source.pol,),
                             plan.grid.n)
    assert len(listed) == len(term.corrections)
    axis, side = 'xyz'.index(face[0]), face[1]
    out, flipped = [], 0
    for corr, (slot, slab, _, a, _, _) in zip(term.corrections, listed):
        assert corr.slab == slab
        at = {0: slab.x, 1: slab.y, 2: slab.z}[a][0]
        this = '+' if at == hi[a] else '-'
        if a == axis and this == side:
            corr = dataclasses.replace(corr, sign=-corr.sign)
            flipped += 1
        out.append(corr)
    assert flipped > 0
    return _replace_term(plan, source.name, dataclasses.replace(term, corrections=tuple(out)))


@pytest.mark.parametrize('face', ['x-', 'x+', 'y-', 'y+', 'z-', 'z+'])
def test_mutant_one_face_with_the_wrong_sign_leaks(face):
    scene = _scene('-z', 'x')
    res = run_mutated(scene, plan=_flipped(scene, face))
    outside, inside = _leak(res, scene.sources[0])
    assert outside > CAUGHT * inside, (face, outside, inside)


@dataclasses.dataclass(frozen=True)
class _LateColumn(ColumnTerm):
    """The mutant column: the H half of step ``n`` first, then the E half.

    At the start of step ``n`` its memory holds ``E(n)`` and ``H(n-1/2)`` -- the H half a step
    late, which is the order a column stepped "like the main grid" would have.
    """

    def advance(self, mem, E, H, n, params, rt):
        sub = self.sub
        sp = params['sources'][self.name]['column']
        crt = _Scoped(rt, f'col[{self.name}]:')
        col = mem['col']
        Ec, Hc, psi, cmem = col['E'], col['H'], col['psi'], col['mem']
        cE, psi = curl(Ec, 'E', sub, sp, psi, crt)
        for t in sub.terms:
            cE = t.before_H(cE, Ec, cmem[t.name], n, sp, crt)
        Hc = crt.update(Hc, cE, sp['ah'], sp['minus_bh'])
        for m in sub.masks:
            Hc = m.after_H(Hc, crt)
        cH, psi = curl(Hc, 'H', sub, sp, psi, crt)
        for t in sub.terms:
            cH = t.before_E(cH, Hc, cmem[t.name], n, sp, crt)
        Ec = crt.update(Ec, cH, sp['ae'], sp['be'])
        for m in sub.masks:
            Ec = m.after_E(Ec, crt)
        w = params['waveform'][self.row, n]
        for c, weight in self.drive:
            slab = self._drive_slab(c)
            Ec = crt.slab_mul(Ec, slab, 0.0)
            Ec = crt.slab_add(Ec, slab, w * weight)
        cmem = {t.name: t.advance(cmem[t.name], Ec, Hc, n, sp, crt) for t in sub.terms}
        rec_E = rt.record(mem['E'], n, rt.xp.stack(
            [Ec[(c,) + self._point(self.entry_H if c == self.axis else self.entry_E)]
             for c in range(3)]))
        rec_H = rt.record(mem['H'], n, Hc[(slice(None),) + self._point(self.entry_H)])
        return {'col': {'E': Ec, 'H': Hc, 'psi': psi, 'mem': cmem}, 'E': rec_E, 'H': rec_H}


def test_mutant_the_column_stepped_half_a_step_late_leaks():
    scene = _scene('-z', 'x')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        plan = fd.plan(scene)
    term = next(t for t in plan.terms if t.name == 't')
    late = _LateColumn(**{f.name: getattr(term, f.name) for f in dataclasses.fields(term)})
    res = run_mutated(scene, plan=_replace_term(plan, 't', late))
    outside, inside = _leak(res, scene.sources[0])
    assert outside > CAUGHT * inside, (outside, inside)
    # The control: the real column through the same hook leaks nothing.
    res = run_mutated(scene, plan=plan)
    outside, inside = _leak(res, scene.sources[0])
    assert inside > 0.5 and outside <= LEAK_TOL * inside


def test_mutant_the_column_background_missing_the_substrate_leaks():
    scene = _scene('-z', 'x', substrate=(2.25, 0.0))
    bare = _scene('-z', 'x')

    def drop_substrate(params, scene_, plan, rt):
        term = next(t for t in plan.terms if t.name == 't')
        params['sources']['t']['column'] = lower(column_scene(bare, bare.sources[0], plan),
                                                 term.sub, rt)
        return params

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = run_mutated(scene, mutate_params=drop_substrate)
    outside, inside = _leak(res, scene.sources[0])
    assert outside > CAUGHT * inside, (outside, inside)
