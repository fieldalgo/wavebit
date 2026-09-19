"""Stage 1, gate 1 -- the CPML absorbs: the doubling table, and the law behind its floor.

**The method.** A Gaussian packet travelling in +z is launched in a box whose ``z+`` face is
a ``PML``; the reference is the *same packet on the same cells* in a box whose ``z+`` PML sits
``delay * W`` further away (``fd_helpers.doubling_pair``). Everything up to that layer -- the
excitation, the ``z-`` layer, the probe's cell -- is one problem, so the difference at a probe
three cells in front of the layer is that layer's reflection and nothing else; round-off
between two runs of different lengths is ~1e-14 of the peak. The reference's own reflection
arrives ``2 * delay * W / cos(theta)`` later, outside the window by construction.

The angle is set by a Bloch phase on x with a **one-cell** transverse box, which is exact
rather than cheap: the discrete transverse operator of a one-cell period is the same
``2 sin(phi/2)/dx`` a many-cell period gives. The number reported is the ratio of the
amplitudes **at the carrier frequency**, not the worst in time, because the packet has a fixed
``kx`` and a band of ``kz``, so its angle varies across its band; one frequency pins one angle
(``sin theta = kx / w``) and a worst-in-time figure is the worst angle in the band instead of
the angle in the column heading. (The first version of this file used worst-in-time and read
the *rise* of the grazing tail as a defect in the layer.)

**The table**, measured 2026-09-18 on this kernel at the default ``reflection=1e-8``,
``dx = wavelength/20``, ``courant = 0.5``, amplitude ratio at the carrier:

    cells      0 deg     30 deg     60 deg     80 deg
        6    2.6e-02    2.1e-02    1.0e-02    1.9e-02
        8    4.6e-03    3.3e-03    8.8e-04    2.6e-02
       12    9.2e-05    4.7e-05    1.1e-04    3.4e-02
       16    1.1e-06    5.9e-07    1.3e-04    3.9e-02
       24    2.2e-08    1.4e-07    1.2e-04    4.4e-02

**This is what re-measures the 12-cell refusal line**, and it says something the row labels
alone do not. The reflection is the larger of two things: the *discretization* of the
conductivity grading, which falls about a decade every two cells, and the layer's own
**design target at that angle**, which is ``reflection ** cos(theta)`` and does not depend on
the depth at all -- the grading is normalised to hit the target whatever the thickness is. At
the default target those floors are 1.0e-8 / 1.2e-7 / 1.0e-4 / 4.1e-2 for the four columns,
and every saturated entry above sits within a factor of 2.2 of one.
:func:`test_the_floor_is_the_targets_own_angular_law` measures that law directly, over three
decades of target, and it is the gate's oracle: a self-captured table cannot say a layer
absorbs, and this law can.

**Both z faces are measured, because they are different code.** ``_lower/cpml.profiles`` has a
branch per side, and every measurement above launches its packet at the ``z+`` layer, so the
low branch was reached only by the frozen golden -- which is a pin and not an oracle.
:func:`test_the_low_face_layer_absorbs_what_the_high_face_one_does` runs the same doubling pair
mirrored (``fd_helpers.doubling_pair(toward='-')``) and puts the low branch under the table.

So: below 12 cells the grading is under-resolved and the number is the grid's. From 12 on,
the columns that have reached their target stay there, and **more cells buy accuracy near
normal incidence while a lower ``reflection=`` is what buys it at grazing**. Twelve is the
line because it is the first depth at which no column is limited by the grid rather than by
what was asked for.

**Two things the plan expected of this gate that the measurement contradicts.** The stage-1
design names ``kappa = 1`` and ``alpha = 0`` as the mutants that must raise the grazing number
by much more than ten times. Measured here, both *lower* it -- ``alpha = 0`` by 127x at normal
incidence, ``kappa = 1`` by 1.4x -- because this probe carries only propagating waves, and for
a propagating wave the CFS pole is pure cost. That is the same thing the legacy engine's own
comment says about ``PML_KAPPA_MAX`` ("classically justified by grazing and evanescent
absorption, which the probe cannot see") and it is now measured rather than asserted:
:func:`test_the_cfs_pole_and_kappa_are_not_settled_by_this_probe` pins it, so that nobody
reads the two constants as optima this file established. What a *structure's* evanescent near
field does to them is not measured anywhere in this repository yet.

The mutants that do work are the three that break the layer rather than tune it: dropping the
recurrence (``a = 0``, leaving a real coordinate stretch that reflects 0.89-0.98), grading the
profile from the wrong end (0.37-0.59) and swapping the node and half-cell profiles, a
half-cell error in where the stretch is sampled (5.5e-2 to 9.6e-2, 600 to 840 times the real
layer). All three are below.

**The opt-in stabilizing loss** (``PML(stabilize=3e-3)``, ``tests/test_fd_pml_stability.py``
for why it exists) is measured here by the same method, because what it costs is a reflection.
It sits on the component normal to the face, which a normally incident wave lacks, so **its
0-degree column is the table's to the bit**
(:func:`test_at_normal_incidence_the_stabilized_layer_is_the_pure_stretch`). At an angle it
reflects in proportion to its strength
(:func:`test_the_stabilizing_loss_reflects_in_proportion_to_its_strength`), and that replaces
the target's law as the oblique floor. Measured 2026-09-19, :data:`STABILIZED`:

    cells      0 deg     30 deg     60 deg     80 deg
        6    2.6e-02    2.1e-02    1.8e-02    1.7e-01
        8    4.6e-03    3.6e-03    6.8e-03    1.4e-01
       12    9.2e-05    2.1e-04    3.7e-03    1.1e-01
       16    1.1e-06    8.2e-05    2.3e-03    9.8e-02
       24    2.2e-08    1.7e-05    9.2e-04    7.9e-02

so more cells lower its oblique columns, and a lower ``reflection=`` does not (the loss is a
fraction of ``sigma_max`` and grows with it: 24 cells at 60 degrees, 9.2e-4 at 1e-8 and 1.1e-3
at 1e-12). The table above, re-measured the same day with the loss off, reproduced every entry
to its printed digits.

**Cost.** The grazing rows are the expensive ones: the packet has to be long enough in z that
its band stays clear of ``kz = 0``, which at 80 degrees is 7.3 wavelengths and 13 000 steps.
The full table and the full law sweep are marked ``slow``; a three-by-two corner of the table,
the law at one angle, the mutants and the doors run in the default tier.
"""
import math
import re

import numpy as np
import pytest

import wavebit.fd as fd
from wavebit.fd._lower.params import _cpml, _cpml_loss
from fd_helpers import backends, doubling_reflection, to_numpy

#: What this file measured, at the default ``reflection=1e-8``. Rows are cells, columns are
#: the nominal angle of incidence, values the amplitude ratio at the carrier.
TABLE = {
    6: {0: 2.56e-02, 30: 2.13e-02, 60: 9.98e-03, 80: 1.87e-02},
    8: {0: 4.64e-03, 30: 3.33e-03, 60: 8.84e-04, 80: 2.62e-02},
    12: {0: 9.19e-05, 30: 4.68e-05, 60: 1.15e-04, 80: 3.43e-02},
    16: {0: 1.11e-06, 30: 5.87e-07, 60: 1.26e-04, 80: 3.92e-02},
    24: {0: 2.17e-08, 30: 1.36e-07, 60: 1.23e-04, 80: 4.44e-02},
}
#: The same doubling table for ``PML(stabilize=3e-3)``, the opt-in stabilizing loss.
STABILIZED = {
    6: {0: 2.56e-02, 30: 2.13e-02, 60: 1.78e-02, 80: 1.71e-01},
    8: {0: 4.64e-03, 30: 3.65e-03, 60: 6.81e-03, 80: 1.43e-01},
    12: {0: 9.19e-05, 30: 2.10e-04, 60: 3.70e-03, 80: 1.13e-01},
    16: {0: 1.11e-06, 30: 8.22e-05, 60: 2.35e-03, 80: 9.77e-02},
    24: {0: 2.17e-08, 30: 1.72e-05, 60: 9.23e-04, 80: 7.86e-02},
}
#: The value the stability gate sized (``wavebit/fd/_lower/cpml.py::STABILIZING_LOSS``).
RHO = 3e-3
#: How far a re-measurement may drift from the table before this gate calls it a change. A
#: factor of three: the entries span six decades and a defect moves them by three or more
#: (the smallest mutant below is 600x), while a libm or a platform moves the smallest entry
#: -- 2.2e-8, which is the discretization floor of a 24-cell layer -- by tens of percent.
DRIFT = 3.0
#: The corner of the table the default tier runs; the rest is ``slow``.
FAST = [(6, 0), (12, 0), (24, 0), (6, 30), (12, 30), (24, 30)]
#: A mutant of the layer must reflect at least this much, against 9.2e-5 for the real one.
MUTANT_FLOOR = 1e-2


def _cells(angle, only_fast):
    return [pytest.param(n, angle, marks=() if (n, angle) in FAST else pytest.mark.slow)
            for n in sorted(TABLE) if (not only_fast or (n, angle) in FAST)]


def table_cases():
    out = []
    for angle in (0, 30, 60, 80):
        out += _cells(angle, only_fast=False)
    return out


# ---------------------------------------------------------------------------------------------
# The table, and the law under its floor
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('cells, angle', table_cases())
def test_the_doubling_table_is_what_this_kernel_measures(cells, angle):
    """Every entry of the module docstring's table, re-measured, within a factor of three."""
    measured, worst, layout = doubling_reflection(cells, float(angle))
    want = TABLE[cells][angle]
    assert want / DRIFT < measured < want * DRIFT, (cells, angle, measured, want)
    # The worst-in-time figure is the one a user suffers, and it is never below the carrier's
    # -- but it can be well above it, by up to 15x on the deepest layers at 30 degrees, and
    # that gap IS the band's grazing tail: once the carrier's own reflection has fallen to
    # 6e-7 the biggest thing left in the difference is the part of the packet that arrived at
    # a worse angle than the column heading. Which is the whole reason the table is spectral.
    assert measured / 2 < worst < 20 * measured + 1e-9, (measured, worst)


@pytest.mark.parametrize('angle', [pytest.param(a, marks=() if a == 60 else pytest.mark.slow)
                                   for a in (0, 30, 60, 80)])
@pytest.mark.parametrize('target', [1e-4, 1e-8, 1e-12])
def test_the_floor_is_the_targets_own_angular_law(angle, target):
    """``R = reflection ** cos(theta)`` wherever the target is above the grid's own floor.

    The gate's oracle, and the reason the table is not merely a self-captured golden. The
    conductivity grading is normalised so that the *continuous* profile attenuates a normally
    incident wave to ``reflection`` on a round trip; at angle ``theta`` the path through the
    layer carries a factor ``cos(theta)`` in the exponent, so the layer's design reflection is
    ``reflection ** cos(theta)``, independent of the depth. Measured at 24 cells over three
    decades of target and four angles, every point where that law is above the 24-cell
    discretization floor (~2e-8) lands within a factor of 2.2 of it -- and every point where
    it is *below* saturates at that floor instead, which is the other half of the statement.

    **Both branches are two-sided, and the saturating one has to be.** ``measured`` is a ratio
    of two runs, so an identity kernel -- fields frozen, observers still recording -- returns
    exactly 0.0 from it, and ``measured < floor`` on its own is a check that cannot fail.
    Measured with such a kernel patched in, the four cases whose law falls below the floor all
    passed while the eight two-sided ones failed. Round-off between the two boxes of a real run
    is ~1e-14 of the peak, so zero separates cleanly; the reference amplitude is checked
    besides, which is what tells "the layer absorbed everything" from "nothing was launched".
    """
    measured, _, layout = doubling_reflection(24, float(angle),
                                              mutate_params=_retarget(target))
    assert layout['reference_peak'] > 100 * layout['reference_initial'], \
        'nothing reached the probe: the measurement below would be zero for that reason'
    law = target ** math.cos(math.radians(angle))
    #: The 24-cell layer's own discretization floor, with room for the angle: measured
    #: 1.0e-8 at 0 degrees and 7.7e-8 at 30 when the target is far below it.
    floor = 2e-7
    if law > floor:
        assert law / 2.5 < measured < law * 2.5, (angle, target, measured, law)
    else:
        assert 0.0 < measured < floor, (angle, target, measured,
                                        'saturates at the grid floor; it does not vanish')


@pytest.mark.parametrize('cells, angle', [(12, 0), (24, 0),
                                          pytest.param(12, 30, marks=pytest.mark.slow)])
def test_the_low_face_layer_absorbs_what_the_high_face_one_does(cells, angle):
    """The ``z-`` branch of ``profiles()``, under the same oracle as the ``z+`` one.

    **Why this exists as a separate case.** ``_lower/cpml.profiles`` has a branch per side --
    ``depth = (cells - index - offset) * dx`` on a low face and ``(index + offset) * dx`` on a
    high one -- and every other absorbing measurement in stage 1 launches its packet at the
    high face, so the low branch was reached only by the frozen golden. A golden is a pin and
    not an oracle: a wrong low profile would have been frozen just as happily.

    **What this catches, measured with the low face broken and the high one left alone.** The
    low branch spelled like the high one (``depth = (index + offset) * dx``, the copy-paste
    this code is one keystroke from) reflects 6.5e-1 -- 7000x. The node and half offsets
    exchanged on that face alone is 5.5e-2, 596x. Both are four decades clear of the assertion
    below.

    **What it does not catch, and that is a statement about the layer rather than a gap.**
    Displacing the whole low profile half a cell deeper -- both offsets moved together -- is a
    1.2% change in the reflection, because it neither unmatches the two sample points from each
    other nor rescales the grading: it only makes the layer slightly deeper than it says it is.
    No reflection measurement at this resolution can see 1.2%, and the frozen lowering in
    ``tests/test_fd_stage1_backends.py`` stays the right instrument for it. The distinction
    worth carrying away is that a *relative* half-cell error between the node and half profiles
    is 600x where a *common* one is 1.2%.

    The two faces are not forced symmetric and must not be: on a low face the PEC wall is the
    node ``x_0``, which the node profile reaches at ``depth = thickness``, and on a high face it
    is the pad ``x_N``, which no node occupies, so the node profile stops at
    ``thickness - dx``. That asymmetry is the Yee grid's. Measured, it costs nothing at normal
    incidence -- the two numbers agree to every digit -- and 1.3% at 30 degrees, where the
    Bloch phase across x is not mirrored with the geometry.
    """
    high, _, _ = doubling_reflection(cells, float(angle))
    low, _, layout = doubling_reflection(cells, float(angle), toward='-')
    assert layout['reference_peak'] > 100 * layout['reference_initial'], 'nothing arrived'
    want = TABLE[cells][angle]
    assert want / DRIFT < low < want * DRIFT, (cells, angle, low, want)
    assert abs(low - high) < 0.1 * high, (cells, angle, low, high)


@pytest.mark.parametrize('cells, angle', [
    pytest.param(n, a, marks=() if (n, a) in ((12, 30),) else pytest.mark.slow)
    for n in sorted(STABILIZED) for a in (30, 60, 80)])
def test_the_stabilized_layer_reflects_what_its_table_says(cells, angle):
    """The opt-in loss's doubling table, re-measured within :data:`DRIFT`."""
    measured, _, _ = doubling_reflection(cells, float(angle), stabilize=RHO)
    want = STABILIZED[cells][angle]
    assert want / DRIFT < measured < want * DRIFT, (cells, angle, measured, want)


def test_at_normal_incidence_the_stabilized_layer_is_the_pure_stretch():
    """The stabilizing loss acts on the normal component, which a normally incident wave lacks.

    So the stabilized layer and the default record the same probe series to the bit -- the loss
    multiplies zeros -- and the 0-degree column of :data:`STABILIZED` is :data:`TABLE`'s.
    """
    from fd_helpers import doubling_pair, run_mutated
    series = []
    for stabilize in (RHO, None):
        res = run_mutated(doubling_pair(12, 0.0, stabilize=stabilize)[0])
        series.append([to_numpy(res[f'E{a}'].values) for a in 'xyz'])
    assert np.max(np.abs(series[0][0])) > 0.0
    for a, b in zip(*series):
        assert np.array_equal(a, b)
    assert all(STABILIZED[n][0] == TABLE[n][0] for n in TABLE)


@pytest.mark.parametrize('angle', [30, pytest.param(60, marks=pytest.mark.slow)])
def test_the_stabilizing_loss_reflects_in_proportion_to_its_strength(angle):
    """The stabilized layer's oblique floor is its loss's, and it is first order in ``rho``.

    Measured at 12 cells: 2.1e-4 / 6.2e-4 at 30 degrees and 3.7e-3 / 1.28e-2 at 60 for
    ``rho`` = 3e-3 / 1e-2 (x2.95 and x3.46 for x3.33), against 4.7e-5 and 1.1e-4 for the pure
    stretch.
    """
    pure, _, _ = doubling_reflection(12, float(angle))
    weak, _, _ = doubling_reflection(12, float(angle), stabilize=RHO)
    strong, _, _ = doubling_reflection(12, float(angle), stabilize=1e-2)
    assert weak > 3 * pure, (angle, weak, pure)
    assert 2.5 < strong / weak < 4.0, (angle, weak, strong)


def _retarget(reflection):
    """Rebuild the layer for a different ``reflection`` target, nothing else moved.

    Both halves of a layer read the target -- the stretch's ``sigma_max`` and, on a stabilized
    face, the loss that is a fraction of it -- so both are rebuilt (the second is empty on the
    default pure stretch).
    """
    def mutate(params, scene, plan, rt):
        z = tuple(f.replace(reflection=reflection) for f in scene.boundaries.z)
        moved = scene.replace(boundaries=fd.Boundaries(x=scene.boundaries.x,
                                                       y=scene.boundaries.y, z=z))
        return {**params, 'cpml': _cpml(moved, plan, rt),
                'cpml_loss': _cpml_loss(moved, plan, rt)}
    return mutate


# ---------------------------------------------------------------------------------------------
# The mutants: three ways to break the layer, and two constants that are not settled here
# ---------------------------------------------------------------------------------------------

def _drop_the_recurrence(params, scene, plan, rt):
    """``a = 0``: psi is never charged, so the layer is a real coordinate stretch and a mirror."""
    return {**params, 'cpml': {k: {**v, 'a': rt.xp.zeros_like(v['a'])}
                               for k, v in params['cpml'].items()}}


def _grade_from_the_wrong_end(params, scene, plan, rt):
    """The profile reversed along the absorbing axis: sigma largest at the inner face."""
    def flip(v):
        axis = next((i for i, s in enumerate(v.shape) if s > 1), 0)
        return rt.xp.flip(v, axis=axis)
    return {**params, 'cpml': {k: {kk: flip(vv) for kk, vv in v.items()}
                               for k, v in params['cpml'].items()}}


def _swap_the_node_and_half_profiles(params, scene, plan, rt):
    """Each difference reads the other one's profile: a half-cell error in where sigma sits."""
    out = {}
    for key in params['cpml']:
        face, field, c = key.split(':')
        out[key] = params['cpml'][f'{face}:{"H" if field == "E" else "E"}:{c}']
    return {**params, 'cpml': out}


@pytest.mark.parametrize('angle', [0, 30, 60])
@pytest.mark.parametrize('mutant, least', [
    pytest.param(_drop_the_recurrence, 0.5, id='a=0, no recurrence'),
    pytest.param(_grade_from_the_wrong_end, 0.2, id='graded from the wrong end'),
    pytest.param(_swap_the_node_and_half_profiles, MUTANT_FLOOR, id='node/half swapped'),
])
def test_mutant_a_broken_layer_reflects_orders_more(angle, mutant, least):
    """Measured: 0.89-0.98, 0.37-0.59 and 5.5e-2 to 9.6e-2 against 4.7e-5 to 1.1e-4 real."""
    if angle != 0:
        pytest.importorskip('numpy')
    real, _, _ = doubling_reflection(12, float(angle))
    broken, _, _ = doubling_reflection(12, float(angle), mutate_params=mutant)
    assert broken > least, (angle, broken)
    assert broken > 300 * real, (angle, broken, real)


def test_the_cfs_pole_and_kappa_are_not_settled_by_this_probe():
    """``alpha = 0`` and ``kappa_max = 1`` *improve* this measurement, and that is the finding.

    The stage-1 design expected both to be mutants that raise the grazing number. They do the
    opposite, and the reason is the probe: it carries propagating plane waves only, and for a
    propagating wave the complex-frequency shift is pure cost (it moves the pole away from the
    operating frequency and makes the stretch more real) while ``kappa > 1`` costs a little as
    the legacy engine's own comment already says. What the two constants are *for* -- absorbing
    the evanescent near field a structure radiates into the layer -- is not measured anywhere
    in this repository, so this pins the surprise rather than the optimum, and the constants
    stay at the legacy values until something that can see evanescent fields moves them.
    """
    import wavebit.fd._lower.cpml as cpml_module

    def without(name, value):
        def mutate(params, scene, plan, rt):
            saved = getattr(cpml_module, name)
            setattr(cpml_module, name, value)
            try:
                return {**params, 'cpml': _cpml(scene, plan, rt)}
            finally:
                setattr(cpml_module, name, saved)
        return mutate

    real, _, _ = doubling_reflection(12, 0.0)
    no_alpha, _, _ = doubling_reflection(12, 0.0, mutate_params=without('ALPHA_RELATIVE', 0.0))
    no_kappa, _, _ = doubling_reflection(12, 0.0, mutate_params=without('KAPPA_MAX', 1.0))
    assert no_alpha < real / 20, (no_alpha, real)      # measured 7.2e-7 against 9.2e-5
    assert no_kappa < real, (no_kappa, real)           # measured 6.7e-5 against 9.2e-5
    assert (cpml_module.ALPHA_RELATIVE, cpml_module.KAPPA_MAX) == (1.5, 2.0)


# ---------------------------------------------------------------------------------------------
# What a PML face IS: the plan it lowers to, and the memory it carries
# ---------------------------------------------------------------------------------------------

@pytest.fixture(scope='module')
def scene():
    from fd_helpers import doubling_pair
    return doubling_pair(12, 30.0)[0]


def test_a_pml_face_is_a_pec_wall_with_a_slab_in_front(scene):
    """The face rule is the PEC one; the absorber is the slab and the psi, not a new rule."""
    plan = fd.plan(scene)
    pec = fd.plan(scene.replace(boundaries=fd.Boundaries(
        x=scene.boundaries.x, y=scene.boundaries.y, z=fd.PEC())))
    for name in ('z-', 'z+'):
        rule, plain = plan.faces[name], pec.faces[name]
        assert rule.kind == 'pml' and plain.kind == 'pec'
        assert rule.position == plain.position
    assert [(n, c, p) for n, c, p in plan.pads] == [(n, c, p) for n, c, p in pec.pads]
    assert plan.masks == pec.masks
    assert plan.pml_cells == (('z-', 12), ('z+', 12)) and plan.absorber_cells == ()


def test_the_psi_lives_in_the_slab_and_there_are_four_per_face(scene):
    """Four psi per face -- two curls times the two transverse components -- at slab shape.

    The layout the measurement chose (``studies/fdtd_cpml_slab_cost.py``): psi is
    ``(cells, ny, nx)`` and not ``(nz, ny, nx)``, which at 128^3 with six faces is 36 MB
    against 192 MB, and faster on all three backends. Here it is the shape that is pinned,
    because the bytes are the part of that finding which transfers to a GPU.
    """
    from wavebit.fd._kernel.step import initial_state
    from wavebit.fd._runtime import runtime
    plan = fd.plan(scene)
    rt = runtime('numpy')
    state = initial_state(plan, fd.lower(scene, plan, rt), rt)
    assert len(plan.cpml) == 2
    assert set(state['psi']) == {k for face in plan.cpml for k in face.keys()}
    assert len(state['psi']) == 8
    for face in plan.cpml:
        for key in face.keys():
            assert key.split(':')[2] in ('0', '1'), 'never the component along the axis'
            assert state['psi'][key].shape == (12, plan.ny, plan.nx)
    report = fd.check(scene)
    item = np.dtype(plan.dtype).itemsize            # complex: a Bloch face sets the angle
    assert report.bytes_per_state == (2 * 3 * plan.cells + 8 * 12 * plan.ny * plan.nx) * item
    assert [a[:2] for a in report.absorbers] == [('z-', 'PML'), ('z+', 'PML')]


def test_a_vacuum_box_with_no_pml_carries_no_psi_and_the_stage_0_coefficients():
    """The condition every stage-0 golden still holding depends on, stated as a test."""
    from fd_helpers import cavity_box
    from wavebit.fd._kernel.step import initial_state
    from wavebit.fd._runtime import runtime
    box = cavity_box('pec', N=8, steps=4)[0]
    plan = fd.plan(box)
    rt = runtime('numpy')
    params = fd.lower(box, plan, rt)
    assert plan.cpml == () and plan.absorbing_faces == ()
    assert initial_state(plan, params, rt)['psi'] == {}
    assert float(params['ae']) == 1.0 and float(params['ah']) == 1.0
    assert float(params['be']) == plan.dt / plan.dx == float(params['bh'])
    assert params['cpml'] == {} and params['poles'] == {}


@pytest.mark.parametrize('backend', backends())
def test_the_three_backends_absorb_alike(backend):
    """One kernel body, three runtimes: the same layer, the same number to 1e-9.

    The CPML write is a *different* primitive per backend -- four fused in-place operations on
    NumPy and Torch, a dynamic-update-slice on JAX -- so this is the check that the three
    writings compute one recurrence. The JAX form adds ``d (1/kappa - 1) + psi`` where the
    fused one evaluates ``d/kappa + psi``, which differs by one rounding per cell per step;
    over 452 steps that stays far inside 1e-9 of a 9e-5 reflection.
    """
    reference, _, _ = doubling_reflection(12, 0.0, backend='numpy')
    got, _, _ = doubling_reflection(12, 0.0, backend=backend)
    assert abs(got - reference) < 1e-9 + 1e-6 * reference, (backend, got, reference)


# ---------------------------------------------------------------------------------------------
# The door: the refusal below twelve cells quotes what this file measured
# ---------------------------------------------------------------------------------------------

def test_the_thin_pml_refusal_quotes_this_files_table():
    """The number a user is refused with is a number this kernel produced, and still produces."""
    with pytest.raises(ValueError) as e:
        fd.PML(cells=6, wavelength=0.55)
    message = str(e.value)
    assert 'tests/test_fd_pml.py' in message and 'allow_thin=True' in message
    for cells in sorted(TABLE):
        assert f'{cells:>9}' in message or f'\n{cells:>9}' in message or str(cells) in message
    for cells, row in TABLE.items():
        for angle, value in row.items():
            assert f'{value:.1e}'.replace('e-0', 'e-0') in message, (cells, angle, value)
    assert fd.PML(cells=6, wavelength=0.55, allow_thin=True).cells == 6
    assert fd.PML(cells=12, wavelength=0.55).cells == 12


def test_the_refusal_is_about_the_default_target_and_says_so():
    """The message has to carry the second half of the finding, or the reader raises cells."""
    with pytest.raises(ValueError) as e:
        fd.PML(cells=11, wavelength=0.55)
    message = str(e.value)
    assert 'reflection**cos(theta)' in message
    assert 'lower reflection=' in message and 'not more cells' in message


# ---------------------------------------------------------------------------------------------
# The writing, per backend: the contract the cost study fixed, and the one gap it left
# ---------------------------------------------------------------------------------------------

def _compiled_body(scene):
    """The optimized HLO of the scan body JAX compiles for ``scene``, and its plan.

    Read exactly as ``tests/test_fd_terms.py`` reads it (and as
    ``studies/fdtd_cpml_slab_cost.py`` read it): the ``while`` body and every computation it
    calls, recursively.
    """
    jax = pytest.importorskip('jax')
    from wavebit.fd import _runtime
    from wavebit.fd._run import execute
    rt = _runtime.JaxRuntime()
    captured = {}

    def jit(fn):
        compiled = jax.jit(fn)

        def wrapped(*args):
            captured['lowered'] = compiled.lower(*args)
            return compiled(*args)
        return wrapped
    rt.jit = jit
    plan = fd.plan(scene)
    execute(plan, fd.lower(scene, plan, rt), rt)
    text = captured['lowered'].compile().as_text()
    pattern = r'^(?:ENTRY )?(\S+) \([^' + chr(10) + r']*\{' + chr(10) + r'(.*?)' + chr(10) + r'\}'
    blocks = dict(re.findall(pattern, text, re.M | re.S))
    todo, seen, parts = re.findall(r'body=(\S+?)[,\s]', text), set(), []
    assert todo, 'no while loop: the scan did not lower to one'
    while todo:
        name = todo.pop()
        if name in seen or name not in blocks:
            continue
        seen.add(name)
        parts.append(blocks[name])
        todo.extend(re.findall(r'(?:calls|to_apply)=(\S+?)[,\s)]', blocks[name]))
    return chr(10).join(parts), plan


def test_the_jax_cpml_write_is_a_dynamic_update_slice_and_not_the_scatter():
    """``rt.slab_add`` and the CPML write are two different primitives; this is why.

    The stage-0 study rejected ``a.at[slices].add`` for a small cube at an arbitrary origin,
    which lowers to a full-volume select loop LLVM unswitches only on aligned origins, and
    made ``slab_add`` a 4-D index scatter. A CPML slab is not that cube: it has full extent on
    two axes, is contiguous on the third and is anchored at 0 or ``N - cells``, so the loop
    bounds are exact and there is nothing to unswitch. ``studies/fdtd_cpml_slab_cost.py``
    measured the dynamic-update-slice cheapest at every N and on every axis (+6585..+8161 us
    at N = 128 against +9503..+12910 for the scatter), the only writing that beats full-volume
    psi at every N, and the only one needing no index arrays -- a scatter's three int32 per
    cell cost 1.5x the float64 psi they address.

    So the body must carry one DUS per stretched difference and **no scatter for the CPML**,
    and no full-volume copy the baseline lacks.
    """
    from fd_helpers import doubling_pair
    pytest.importorskip('jax')
    scene = doubling_pair(12, 0.0)[0].replace(monitors=(), run=fd.Run(steps=4))
    body, plan = _compiled_body(scene)
    without = scene.replace(boundaries=fd.Boundaries(x=scene.boundaries.x,
                                                     y=scene.boundaries.y, z=fd.PEC()))
    baseline, _ = _compiled_body(without)
    full = r'\w+\[3,%d,%d,%d\]\{[^}]*\} ' % (plan.nz, plan.ny, plan.nx)
    assert body.count(' scatter(') == baseline.count(' scatter('), \
        'the CPML must not lower to the slab-add scatter'
    assert len(re.findall(full + r'copy\(', body)) == len(re.findall(full + r'copy\(', baseline))
    # One dynamic-update-slice per stretched difference: two faces x two curls x two
    # components. Each is on a single component's (nz, ny, nx), never on the 4-D carry --
    # the full-volume ones the body does carry are the curl's own assembly and are in the
    # baseline too.
    single = r'\w+\[%d,%d,%d\]\{[^}]*\} dynamic-update-slice\(' % (plan.nz, plan.ny, plan.nx)
    assert len(re.findall(single, baseline)) == 0
    assert len(re.findall(single, body)) == 8, body.count(' dynamic-update-slice(')
    assert (len(re.findall(full + r'dynamic-update-slice\(', body))
            == len(re.findall(full + r'dynamic-update-slice\(', baseline)))
    # The opt-in stabilizing loss is a broadcast multiply that fuses into the update around
    # it: the stabilized body has exactly the pure stretch's scatters, copies, update-slices
    # and fusions. Written as an update-slice of the scaled slab -- the CPML's own writing --
    # it cost two to four full-volume copies per step; as ``.at[].multiply``, a scatter-mul.
    stabilized, _ = _compiled_body(doubling_pair(12, 0.0, stabilize=RHO)[0].replace(
        monitors=(), run=fd.Run(steps=4)))
    for op in (' scatter(', ' copy(', ' dynamic-update-slice(', ' fusion('):
        assert stabilized.count(op) == body.count(op), op


def test_a_run_with_a_pml_differentiates_on_jax_and_on_torch():
    """The psi is a carry entry, so a gradient has to flow through the recurrence, twice a step.

    A smoke gate, not an accuracy one -- gate 8 of the stage-1 plan is the AD gate and
    compares against central differences. What is pinned here is that the CPML does not break
    either autodiff: on JAX the stretch is functional and the scan transposes; on Torch the
    in-place writing is **replaced by an out-of-place one under autograd**, because a psi is
    read as well as written and the ``mark_dirty`` Function that makes a slab *add* O(slab) in
    backward is not the one this needs. That fallback costs a full-volume copy per stretch per
    step and is the one measurement stage 1 owes stage 6; without it a differentiated Torch
    run would either raise a version-counter error or differentiate the wrong graph.
    """
    from fd_helpers import doubling_pair
    scene = doubling_pair(12, 0.0)[0].replace(monitors=(), run=fd.Run(steps=8))

    jax = pytest.importorskip('jax')
    import wavebit
    # Lifted before ``jax.grad`` traces its argument: when this test ran first in its process
    # the argument was float32 and the gradient off by 4e-8 relative (seen under -n 6).
    wavebit.enable_x64()
    from wavebit.fd._run import execute
    from wavebit.fd._runtime import runtime
    plan = fd.plan(scene)

    def loss(amplitude, backend):
        moved = scene.replace(initial=scene.initial.replace(amplitude=amplitude))
        rt = runtime(backend)
        return (execute(plan, fd.lower(moved, plan, rt), rt)['E'] ** 2).sum()

    g_jax = float(jax.grad(lambda a: loss(a, 'jax'))(1.0))
    assert abs(g_jax) > 0.0

    torch = pytest.importorskip('torch')
    amplitude = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    g_torch = torch.autograd.grad(loss(amplitude, 'torch'), amplitude)[0].item()
    assert abs(g_torch - g_jax) <= 1e-9 * abs(g_jax), (g_torch, g_jax)
    # and the value is the derivative of a quadratic in the amplitude: 2 * loss(1)
    value = float(loss(1.0, 'numpy'))
    assert abs(g_jax - 2 * value) <= 1e-9 * abs(g_jax)
