"""A pure PML feeds a bound mode; ``PML(stabilize=3e-3)`` does not. The loss, and the growth.

**The defect.** A stretch backed by a PEC wall is an *active* termination for evanescent
waves (``src/wavebit/fdtd/_prepare/cpml.py::stabilizing_rate`` derives it). A lossless structure whose
bound mode sits just past the light line -- a grating tooth's guided resonance at a Rayleigh
wavelength, or simply a dielectric slab under a fixed transverse ``k`` -- therefore grows
exponentially under the pure CPML, whatever its constants. It is not the discretization: a
continuum transfer-matrix root of slab + gap + graded CFS-PML + PEC grows at the rate this
kernel converges to as ``dx -> 0``.

**The smallest scene that shows it** (:func:`slab_column`): an ``Axis1D`` column at
``k_x = 2 pi / 0.8`` -- one cell across, so there is no grating and no source -- through a
0.08-thick ``eps = 1.5`` slab, with 0.24 of vacuum on each side and a 12-cell PML on each z
face, kicked by a Gaussian ``E_z``. Nothing in it can gain energy. Measured, the total
``sum |E|^2 + |H|^2`` over the array:

    t                     100       300       E(300)/E(100)
    PML() (the default)   6.74      2.90e3    430          the pure stretch
    PML(stabilize=3e-3)   1.68      1.66      0.983        the stabilizing loss
    PEC                   3.82      3.55      0.930        a closed lossless box: the control
    eps = 1               1.41      1.41      0.9999       no slab, no bound mode: the control

identical on NumPy, JAX and Torch to a few ulps.
:func:`test_the_stabilized_slab_column_does_not_grow` is the time-domain gate, and the default
layer is its mutant: the code path from before the loss existed, which is bit for bit what a
pure-stretch face still runs (the kernel skips the loss on the plan's static face type, and
the frozen goldens reproduce).

**Why the loss is opt-in** is in ``src/wavebit/fdtd/_prepare/cpml.py::stabilizing_rate``: at oblique
incidence it reflects in proportion to its strength (12 cells: 2.1e-4 at 30 degrees against
4.7e-5, ``tests/fdtd/test_fd_pml.py::STABILIZED``), and on by default it broke the Airy, the
closed-form PlaneWave, the RCWA-stack and the grazing-rule gates.

**The eigen gate** (:func:`test_no_mode_of_the_one_step_matrix_grows`) is the same question
without a time window. The kernel's one-step map of the column is linear, so it is a matrix,
built by stepping each unit state once (N ~ 300-600, a tenth of a second); an eigenvalue with
``|lambda| > 1`` is a mode that grows at ``log|lambda| / dt`` e-folds per unit time. Over 24
slab modes (eps 1.05 / 1.5 / 2.25 / 4 x thickness 0.08 / 0.24 x ``k_x = 2 pi/1.6, 2 pi/1.2,
2 pi/0.8``), measured:

    rho                12 cells                    24 cells
    0 (the default)    worst +1.6e-2, 19/24 grow   worst +9.8e-3, 20/24 grow
    1e-3               worst +4.1e-3,  3/24        worst +3.8e-3,  8/24
    2e-3               0/24                        worst +6.7e-4,  2/24
    3e-3 (stabilize=)  0/24, worst +4.8e-13        0/24, worst +8.1e-13

so the stabilizing value's margin over the 24-cell layer is 1.5x, which
:func:`test_a_third_of_the_stabilizing_loss_is_not_enough` states. The positive ``1e-13`` is
round-off on the static (``omega = 0``) null space the Yee curl
carries; the gate line is 1e-9.

**The table is for the default reflection target, 1e-8.** ``rho`` is relative to the stretch's
``sigma_max``, and at a weaker ``reflection=`` the loss that suffices rises, in ``rho`` and
in absolute rate alike. Measured on the same family at 12 / 24 cells, the smallest value of
the grid ``3e-3, 5e-3, 7e-3, 1e-2, 1.5e-2, 2e-2`` with no growing mode:

    reflection    enough     3e-3 there
    1e-8          3e-3       0/24 and 0/24
    1e-6          5e-3       2/24 and 6/24 grow
    1e-4          1e-2       10/24 and 11/24
    1e-3          2e-2       14/24 and 14/24

(``src/wavebit/fdtd/_prepare/cpml.py::MEASURED_STABILIZE``). The eigen gate runs at every row with
its own value, :func:`test_the_default_targets_value_is_not_enough_at_a_weaker_target` is
the third column, and the plan warns (``PMLStabilityWarning``) below a target's value. Enough
here is a measured balance on the sampled families, not passivity: cancelling the stretch's
active normal component point by point would take about two decades more loss.

**The growth is reported, not silent.** The default stays the pure stretch, so a run that grows
must say so: when the energy of the scene's interior (every cell but the layers') still
rises after every source has ended (``src/wavebit/fdtd/_assemble.py::decay_report``,
``res.diagnostics['interior_growth']``), every monitor that sees it warns
``fd.GrowingFieldWarning`` instead of the plain "run longer", naming ``PML(stabilize=...)`` with
the value measured for the face's target (the larger one when two faces ask for different
targets). On the column at t = 200 the interior reads 3.95 (the line is 1.1), a Record in the
slab and a Flux plane in the gap see it (2.69 and 4.86), and both warn ``PML(stabilize=3e-3)``;
at ``reflection=1e-4``, 7.41 and ``1e-2``. The stabilized, PEC and slab-less columns read
0.9985, 1.0000 and 1.0005 and do not warn. NumPy, JAX and Torch read the same statistics; under
``jax.jit`` they are ``None``; under a bare ``jax.grad`` a Record-only column still warns, for
the interior, though the Record's own series is on the differentiated path and unread. The
mutant that reads the span's first two quarters instead of its last two half hides the growth
under the kick's ring-down (1.39), and the Record, whose own series falls there, no longer
warns.

**The lamellar grating** that first showed it is :mod:`test_fd_decay`'s (slow): the default
layer grows there and warns ``GrowingFieldWarning`` naming ``PML(stabilize=3e-3)``;
``stabilize=3e-3`` is its twin that does not grow.
"""

from wavebit.compat import legacy as _legacy_wavebit
import math
import warnings

import numpy as np
import pytest

import wavebit.fdtd._legacy as fd
from wavebit.fdtd import _post
from wavebit.fdtd._assemble import GROWTH_THRESHOLD, _growth_cause
from wavebit.fdtd._prepare import lower, plan as make_plan
from wavebit.fdtd._prepare.cpml import MEASURED_STABILIZE, STABILIZING_LOSS, sufficient_stabilize
from wavebit.fdtd._legacy_boundaries import PML_REFLECTION
from wavebit.fdtd._runtime import runtime
from tests.fixtures.fdtd.fd_helpers import backends, one_step_matrix, to_numpy

#: An eigenvalue at or below this growth rate (e-folds per unit time) is not a growing mode:
#: the omega = 0 null space of the Yee curl sits at a few 1e-13.
GROWTH_LINE = 1e-9

#: The 24-mode family the stabilizing loss was sized on (``src/wavebit/fdtd/_prepare/cpml.py``).
SLABS = [(1.05, 0.08), (1.05, 0.24), (1.5, 0.08), (1.5, 0.24),
         (2.25, 0.08), (2.25, 0.24), (4.0, 0.08), (4.0, 0.24)]
KX = [2 * math.pi / 1.6, 2 * math.pi / 1.2, 2 * math.pi / 0.8]


def slab_column(eps=1.5, thickness=0.08, kx=2 * math.pi / 0.8, dx=0.04, gap=0.24, cells=12,
                time=400.0, z=None, stabilize=STABILIZING_LOSS, kick=True,
                reflection=PML_REFLECTION):
    """The minimal scene: an ``Axis1D`` column through a slab, the z faces ``PML(cells)``.

    Stabilized unless told otherwise: ``stabilize=None`` is the default layer, the pure stretch.
    """
    if z is None:
        z = fd.PML(cells=cells, wavelength=1.0, stabilize=stabilize, reflection=reflection)
    Lz = thickness + 2 * gap + 2 * cells * dx
    initial = None
    if kick:
        initial = fd.InitialField(E=(None, None, lambda x, y, zz:
                                     np.exp(-((zz + thickness / 2) / 0.1) ** 2)))
    return fd.Scene(
        domain=fd.Domain(size=(dx, dx, Lz), center=(0.0, 0.0, -thickness / 2)),
        boundaries=fd.Boundaries(x=fd.Axis1D(k=kx), y=fd.Axis1D(k=0.0, k_max=1.0), z=z),
        resolution=fd.Resolution(dx=dx), run=fd.Run(time=time),
        structures=(fd.Structure(fd.Slab(z=(-thickness, 0.0)), fd.Material(eps=eps)),),
        initial=initial)


def _energy(scene, backend='numpy'):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = fd.run(scene, backend=backend)
    return float(sum(np.sum(np.abs(to_numpy(a)) ** 2) for a in (res.final.E, res.final.H)))


def _ratio(backend='numpy', **kwargs):
    early = _energy(slab_column(time=100.0, **kwargs), backend)
    late = _energy(slab_column(time=300.0, **kwargs), backend)
    return early, late


def growth_rate(scene):
    """The largest ``log|lambda| / dt`` of the one-step matrix: e-folds per unit time."""
    M, dt = one_step_matrix(scene)
    return float(np.max(np.log(np.abs(np.linalg.eigvals(M)))) / dt)


def family_rates(**kwargs):
    return [growth_rate(slab_column(eps=eps, thickness=th, kx=kx, kick=False, **kwargs))
            for kx in KX for eps, th in SLABS]


# ---------------------------------------------------------------------------------------------
# The gates
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('backend', backends())
def test_the_stabilized_slab_column_does_not_grow(backend):
    early, late = _ratio(backend)
    assert late <= early, (backend, early, late)                  # measured 1.68 -> 1.66
    assert late > 0.5 * early, 'the kick did not survive to be measured'


@pytest.mark.parametrize('stabilize', [None, 0.0])
def test_mutant_the_default_pure_stretch_grows_on_the_same_column(stabilize):
    """The code path from before the loss -- the default layer runs it, bit for bit -- grows
    ~430x over 200 time units; ``stabilize=0`` is the same layer spelled out."""
    early, late = _ratio(stabilize=stabilize)
    assert late > 100 * early, (early, late)                      # measured 6.74 -> 2.90e3


def test_the_controls_do_not_grow():
    """PEC in place of the PML, and the pure PML with no slab: nothing to feed, nothing grows."""
    for kwargs in (dict(z=fd.PEC()), dict(eps=1.0, stabilize=None)):
        early, late = _ratio(**kwargs)
        assert late <= 1.01 * early, (kwargs, early, late)        # 0.930 and 0.9999


# ---------------------------------------------------------------------------------------------
# The growth warning: a growing column says so, and names the knob
# ---------------------------------------------------------------------------------------------

#: A Record in the slab and a Flux plane in the gap above it: the two kinds of series a
#: growing scene names when they see the growth (``|v|^2``, and a monitor's region energy).
COLUMN_MONITORS = (fd.Record('rec', at=(0.0, 0.0, -0.04), component='Ez'),
                   fd.Flux('flux', plane=fd.Plane(z=0.15), wavelengths=(1.0,)))


def _monitored(backend='numpy', time=200.0, monitors=COLUMN_MONITORS, **kwargs):
    """``(res, growing, unsettled)``: the column with ``monitors``, and the texts of the
    ``GrowingFieldWarning``\\ s and of the plain ``UnsettledFieldWarning``\\ s it raised."""
    scene = slab_column(time=time, **kwargs).replace(monitors=monitors)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('ignore')
        warnings.simplefilter('always', fd.UnsettledFieldWarning)
        res = fd.run(scene, backend=backend)
    growing = [str(w.message) for w in caught if w.category is fd.GrowingFieldWarning]
    unsettled = [str(w.message) for w in caught if w.category is fd.UnsettledFieldWarning]
    return res, growing, unsettled


@pytest.mark.parametrize('reflection, knob', [(PML_REFLECTION, 'PML(stabilize=3e-3)'),
                                              (1e-4, 'PML(stabilize=1e-2)')])
def test_a_growing_column_warns_growing_and_names_the_value_for_its_target(reflection, knob):
    """The pure stretch, t = 200: the interior's energy rises 3.95 times from quarter to
    quarter at the default target and 7.41 at 1e-4; both monitors see it (their own series rise
    2.69 and 4.86 times, and 5.44 and 7.70) and warn ``GrowingFieldWarning`` -- instead of the
    plain unsettled warning, not beside it -- naming the stabilized layer with the value
    measured for the face's own target (``MEASURED_STABILIZE``)."""
    res, growing, unsettled = _monitored(stabilize=None, reflection=reflection)
    interior, growth = res.diagnostics['interior_growth'], res.diagnostics['growth']
    assert interior > GROWTH_THRESHOLD, interior
    assert all(growth[m.name] > 1.0 for m in COLUMN_MONITORS), growth
    assert len(growing) == 2 and unsettled == [], (growing, unsettled)
    for name, message in zip(('rec', 'flux'), growing):
        assert f"monitor {name!r}" in message and knob in message, message
        assert "['z-', 'z+']" in message and 'stabilize=0' in message
        assert f'{interior:.3g} times higher' in message
        assert f'its own peak rose {growth[name]:.3g} times' in message


@pytest.mark.parametrize('kwargs', [dict(), dict(z=fd.PEC()), dict(eps=1.0, stabilize=None)],
                         ids=['stabilized', 'pec', 'no-slab'])
def test_the_stabilized_and_the_control_columns_do_not_warn_growing(kwargs):
    """The same monitors on the stabilized column, the PEC one and the slab-less one: nothing
    warns growing, and the interior's statistic is concrete and at or under 1 (measured
    0.9985 stabilized, 1.0000 PEC, 1.0005 without the slab)."""
    res, growing, _ = _monitored(**kwargs)
    interior = res.diagnostics['interior_growth']
    assert growing == [], growing
    assert interior is not None and interior < 1.001, interior


def test_a_growing_scene_whose_only_monitor_is_a_record_warns_under_jax_grad():
    """Under a bare ``jax.grad`` a Record's values are on the differentiated path and its own
    series cannot be read (``growth`` is ``None`` for it) -- but the interior's energy is not,
    so the growing column still warns, once, for the interior. A gradient loop over a scene
    whose only monitor is a Record is exactly where a silent growth would do its harm."""
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    _legacy_wavebit.enable_x64()
    scene = slab_column(time=200.0, stabilize=None).replace(monitors=COLUMN_MONITORS[:1])
    seen = []

    def total(amplitude):
        res = fd.run(scene.replace(initial=scene.initial.replace(amplitude=amplitude)),
                     backend='jax')
        seen.append(res.diagnostics)
        return jnp.sum(jnp.abs(res['rec'].values) ** 2)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('ignore')
        warnings.simplefilter('always', fd.UnsettledFieldWarning)
        jax.grad(total)(jnp.asarray(1.0, dtype=jnp.float64))
    growing = [str(w.message) for w in caught if w.category is fd.GrowingFieldWarning]
    assert seen[-1]['growth'] == {'rec': None}, seen[-1]['growth']
    assert seen[-1]['interior_growth'] > GROWTH_THRESHOLD, seen[-1]['interior_growth']
    assert len(growing) == 1 and 'no monitor sees it yet' in growing[0], growing
    assert 'PML(stabilize=3e-3)' in growing[0]


def test_the_backends_agree_on_the_growth():
    """NumPy, JAX and Torch read the same statistics off the same growing column: the series
    agree to round-off, and each statistic is a ratio of two of their samples."""
    values = {}
    for backend in ('numpy', 'jax', 'torch'):
        try:
            __import__(backend)
        except ImportError:
            continue
        if backend == 'jax':
            _legacy_wavebit.enable_x64()
        d = _monitored(backend, stabilize=None)[0].diagnostics
        values[backend] = dict(d['growth'], interior=d['interior_growth'])
    reference = values.pop('numpy')
    for backend, growth in values.items():
        for name, g in growth.items():
            assert abs(g - reference[name]) <= 1e-9 * reference[name], (backend, name, g,
                                                                         reference[name])


def test_under_jit_the_growth_is_none_and_nothing_warns():
    """Under ``jax.jit`` the series are tracers: every growth is ``None`` and the growing
    column does not warn -- the diagnostic reads values, and a trace has none yet. Outside the
    jit the same column warns (the tests above)."""
    jax = pytest.importorskip('jax')
    _legacy_wavebit.enable_x64()
    scene = slab_column(time=200.0, stabilize=None).replace(monitors=COLUMN_MONITORS)
    seen = []

    def total(amplitude):
        res = fd.run(scene.replace(initial=scene.initial.replace(amplitude=amplitude)),
                     backend='jax')
        seen.append((res.diagnostics['growth'], res.diagnostics['interior_growth']))
        return res['rec'].values.sum()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        warnings.simplefilter('error', fd.UnsettledFieldWarning)
        jax.jit(total)(1.0)
    assert seen == [({'rec': None, 'flux': None}, None)], seen


def _cause(scene):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return _growth_cause(scene, make_plan(scene))


def test_the_warning_names_each_stabilized_face_or_a_defect():
    """With every PML face stabilized, the cause names each face's value against the one
    measured for its target and says the structure is outside the sampled families; with no
    PML face, that nothing can feed the field and it is a defect to report."""
    cause = _cause(slab_column(reflection=1e-4, stabilize=4e-3))
    defect = _cause(slab_column(z=fd.PEC()))
    assert 'Every PML face is stabilized' in cause and 'Raise stabilize' in cause
    assert ("'z-' stabilize=4e-3 at reflection=1e-4 (1e-2 was measured to suffice there)"
            in cause), cause
    assert 'outside the families' in cause
    assert 'No face of this scene is a PML' in defect and 'defect' in defect


def test_with_two_targets_the_knob_is_the_value_for_the_weaker_one():
    """Two pure-stretch faces at different targets: the value named is the larger one measured
    -- the one the weaker target needs (1e-2 at 1e-4, where 3e-3 suffices only at 1e-8)."""
    layer = dict(cells=12, wavelength=1.0)
    cause = _cause(slab_column(z=(fd.PML(reflection=1e-8, **layer),
                                  fd.PML(reflection=1e-4, **layer))))
    assert "['z-', 'z+']" in cause and 'PML(stabilize=1e-2)' in cause, cause


#: The statistic itself, kept for the mutant below to call around.
_REAL_GROWTH = _post.growth_ratio


def _first_quarters(t, energy, start, window, min_windows, floor):
    """Mutant: the judged span's second quarter against its first, instead of its last against
    the one before -- the growth read where the kick is still ringing down."""
    got = _REAL_GROWTH(t, energy, start, window, min_windows, floor)
    if got is None:
        return None
    t, e = np.asarray(t), np.asarray(energy)
    _, lo, mid, hi, spread = got
    q = hi - mid
    first, second = e[(t >= start) & (t < start + q)], e[(t >= start + q) & (t < start + 2 * q)]
    return float(second.mean() / first.mean()), start, start + q, start + 2 * q, spread


def test_mutant_the_first_quarters_are_caught_by_the_growing_gate(monkeypatch):
    """Read over the first two quarters of the span, the growth is half hidden under the
    kick's ring-down (measured 1.39 at t = 200, against 3.94 read where it should be), and
    over those quarters the Record's own series falls (0.93): it no longer warns, and the
    growing gate above -- both monitors see it and warn -- fails."""
    monkeypatch.setattr(_post, 'growth_ratio', _first_quarters)
    res, growing, _ = _monitored(stabilize=None)
    assert res.diagnostics['growth']['rec'] < 1.0, res.diagnostics['growth']
    assert len(growing) < len(COLUMN_MONITORS), growing


@pytest.mark.parametrize('reflection', [target for target, _ in MEASURED_STABILIZE])
@pytest.mark.parametrize('cells', [12, 24])
def test_no_mode_of_the_one_step_matrix_grows(cells, reflection):
    """At every tabulated target, the value :func:`sufficient_stabilize` names for it: 3e-3 at
    the default 1e-8, 5e-3 at 1e-6, 1e-2 at 1e-4, 2e-2 at 1e-3 (measured worst +5e-13 to
    +1.2e-12, 0/24 each)."""
    rho = sufficient_stabilize(reflection)
    assert (rho == STABILIZING_LOSS) is (reflection == PML_REFLECTION)
    rates = family_rates(cells=cells, reflection=reflection, stabilize=rho)
    assert max(rates) <= GROWTH_LINE, (cells, reflection, max(rates),
                                       sum(r > GROWTH_LINE for r in rates))


@pytest.mark.parametrize('reflection, measured', [(1e-6, 2), (1e-4, 10), (1e-3, 14)])
def test_the_default_targets_value_is_not_enough_at_a_weaker_target(reflection, measured):
    """``rho`` is relative to ``sigma_max``, and ``sigma_max`` falls with ``log(1/reflection)``,
    but the loss that suffices does not fall with it: 3e-3 at a weaker target leaves modes
    growing (measured 2, 10 and 14 of 24 at 12 cells). This is why the plan warns
    (:func:`test_a_face_below_its_targets_value_warns`)."""
    rates = family_rates(reflection=reflection, stabilize=STABILIZING_LOSS)
    grow = sum(r > GROWTH_LINE for r in rates)
    assert grow >= max(1, measured // 2), (reflection, max(rates), grow)


@pytest.mark.parametrize('cells', [12, 24])
def test_mutant_the_default_pure_stretch_has_growing_modes(cells):
    rates = family_rates(cells=cells, stabilize=None)
    grow = sum(r > GROWTH_LINE for r in rates)
    assert max(rates) > 5e-3 and grow >= 15, (cells, max(rates), grow)   # 19/24 and 20/24


#: A harsher family the value was also sized on: thin high-index slabs, a finer grid, and a
#: fourth ``k_x`` closer to the light line.
STRESS = [(12.0, 0.02), (12.0, 0.04), (6.0, 0.04), (6.0, 0.08), (3.0, 0.04), (1.02, 0.04)]
KX_STRESS = KX + [2 * math.pi / 2.5]


@pytest.mark.parametrize('cells', [12, 16])
def test_no_mode_of_the_stress_family_grows(cells):
    """Measured: the pure stretch 24/24 growing (worst +2.3e-2 at 12 cells); stabilized 0/24."""
    rates = [growth_rate(slab_column(eps=eps, thickness=th, kx=kx, dx=0.02, cells=cells,
                                     kick=False))
             for kx in KX_STRESS for eps, th in STRESS]
    assert max(rates) <= GROWTH_LINE, (cells, max(rates), sum(r > GROWTH_LINE for r in rates))


def test_a_third_of_the_stabilizing_loss_is_not_enough():
    """The margin, stated: ``rho = 1e-3`` still leaves growing modes (measured 3/24 at 12)."""
    rates = family_rates(stabilize=1e-3)
    assert max(rates) > GROWTH_LINE, max(rates)


# ---------------------------------------------------------------------------------------------
# What the loss is, statically
# ---------------------------------------------------------------------------------------------

def test_the_loss_acts_on_the_normal_component_and_only_inside_the_slab():
    from wavebit.fdtd._kernel.cpml import CpmlFace, StabilizedFace
    scene = slab_column(kick=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        p = make_plan(scene)
        p0 = make_plan(slab_column(kick=False, stabilize=0))
        p_default = make_plan(slab_column(kick=False, stabilize=None))
    assert [f.stabilize for f in p.cpml] == [STABILIZING_LOSS] * 2 == [3e-3] * 2
    assert all(type(f) is StabilizedFace for f in p.cpml)
    for face in p.cpml:
        slab = face.normal_slab
        assert slab.c == face.axis == 2
        assert slab.shape == (12, 1, 1)
    params = lower(scene, p, runtime('numpy'))
    assert set(params['cpml_loss']) == {'z-:E', 'z-:H', 'z+:E', 'z+:H'}
    for key, v in params['cpml_loss'].items():
        keep, lift = np.ravel(v['keep']), np.ravel(v['lift'])
        assert np.all((0 < keep) & (keep <= 1) & (lift >= 1)), key
        # keep * lift is the matched update's drive, 1 / (1 + h), exactly as an Absorber's be
        h = (1 - keep) / (1 + keep)
        assert np.allclose(keep * lift, 1 / (1 + h), rtol=1e-14, atol=0), key
        assert keep.min() < 0.999, key                      # it does something
        deep = keep if key.startswith('z+') else keep[::-1]
        assert np.all(np.diff(deep) <= 0), 'graded: the loss rises with depth'
    # The pure stretch -- the default, or 0 spelled out -- is the pre-stabilization face type,
    # so its plan and skeleton are the ones from before the loss existed, and it lowers none.
    assert p0 == p_default and p0.skeleton == p_default.skeleton != p.skeleton
    assert all(type(f) is CpmlFace for f in p0.cpml) and [f.stabilize for f in p0.cpml] == [0, 0]
    assert lower(slab_column(kick=False, stabilize=0), p0, runtime('numpy'))['cpml_loss'] == {}


def _box(stabilize, faces='xyz'):
    """A vacuum box, PML on the axes in ``faces`` and Periodic on the rest, a curl-free kick."""
    dx, n = 0.05, 32
    pml = fd.PML(cells=12, wavelength=1.0, stabilize=stabilize)
    axes = {a: (pml if a in faces else fd.Periodic()) for a in 'xyz'}

    def blob(x, y, z):
        return np.exp(-((x - 0.07) ** 2 + (y + 0.05) ** 2 + (z - 0.03) ** 2) / 0.02)
    return fd.Scene(domain=fd.Domain(size=(n * dx,) * 3), boundaries=fd.Boundaries(**axes),
                    resolution=fd.Resolution(dx=dx), run=fd.Run(steps=60),
                    initial=fd.InitialField(E=(blob, lambda x, y, z: -blob(x, y, z), blob),
                                            H=(None, blob, None)),
                    monitors=(fd.Record('p', at=(0.0, 0.0, 0.0), component='Ez'),))


def _folded(rho):
    """``mutate_params``: the loss folded into ``ae``/``be``/``ah``/``bh`` as an ``Absorber``'s is.

    The independent writing of the same physics -- full-volume coefficient arrays built here
    from the Yee offsets and ``stabilizing_rate``, applied to a pure-stretch plan whose kernel
    runs no ``damp`` -- so a loss at the wrong Yee point, on the wrong side, on the wrong
    component or in the wrong order around the update disagrees with it.
    """
    from wavebit.fdtd._kernel.grid import OFFSETS
    from wavebit.fdtd._prepare.cpml import stabilizing_rate

    def mutate(params, scene, plan, rt):
        n = (plan.nx, plan.ny, plan.nz)
        rates = {f: np.zeros((3, plan.nz, plan.ny, plan.nx)) for f in 'EH'}
        for face in plan.cpml:
            a, cells = face.axis, face.cells
            thickness = cells * plan.dx
            for f in 'EH':
                index = np.arange(n[a]) + 0.5 * OFFSETS[f][a][a]
                depth = (cells - index if face.side == '-' else index - (n[a] - cells)) * plan.dx
                r = np.where(depth > 0, stabilizing_rate(np.maximum(depth, 0), thickness, 1.0,
                                                         1e-8, rho), 0.0)
                shape = [1, 1, 1]
                shape[2 - a] = n[a]
                rates[f][a] += r.reshape(shape)             # the normal component only
        out = dict(params)
        for f, alpha, beta in (('E', 'ae', 'be'), ('H', 'ah', 'bh')):
            h = 0.5 * plan.dt * rates[f]
            out[alpha] = rt.array((1 - h) / (1 + h) * np.asarray(params[alpha]), 'float64')
            out[beta] = rt.array(np.asarray(params[beta]) / (1 + h), 'float64')
        out['minus_bh'] = -out['bh']
        return out
    return mutate


@pytest.mark.parametrize('backend', backends())
def test_the_loss_is_the_folded_matched_loss_on_every_face_and_backend(backend):
    """Six stabilized faces -- every axis, both sides, the corners -- against the same loss
    written independently as full-volume coefficients on a pure-stretch plan (NumPy): the
    fields after 60 steps agree to round-off (measured 1.0e-15 / 5.8e-16 of the peak for E /
    H), where the pure stretch is 8.6e-4 / 6.9e-4 away and the loss's node and half profiles
    swapped 1.9e-4 / 2.2e-4 away."""
    from tests.fixtures.fdtd.fd_helpers import run_mutated
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        got = run_mutated(_box(STABILIZING_LOSS), backend=backend)
        want = run_mutated(_box(None), mutate_params=_folded(STABILIZING_LOSS))
        pure = run_mutated(_box(None))
    for F in ('E', 'H'):
        g, w, p = (to_numpy(getattr(r.final, F)) for r in (got, want, pure))
        peak = np.max(np.abs(w))
        assert np.max(np.abs(g - w)) < 1e-13 * peak, (backend, F, np.max(np.abs(g - w)) / peak)
        assert np.max(np.abs(p - w)) > 1e-6 * peak, (F, np.max(np.abs(p - w)) / peak)


def test_a_stabilized_run_differentiates_on_jax_and_on_torch():
    """The loss is two slab multiplies per face per update -- ``SlabMul`` under Torch autograd,
    a broadcast multiply on JAX -- and a gradient flows through both. The six-face box's kick
    reaches every layer within the 30 steps, so the loss acts on the differentiated field; the
    energy is quadratic in the kick's amplitude, so its derivative at 1 is twice its value."""
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')
    from wavebit.fdtd._execution import execute
    _legacy_wavebit.enable_x64()
    scene = _box(STABILIZING_LOSS).replace(run=fd.Run(steps=30), monitors=())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        p = make_plan(scene)

    def energy(amplitude, backend):
        moved = scene.replace(initial=scene.initial.replace(amplitude=amplitude))
        rt = runtime(backend)
        state = execute(p, lower(moved, p, rt), rt)
        return (state['E'] ** 2).sum() + (state['H'] ** 2).sum()

    value = float(energy(1.0, 'numpy'))
    g_jax = float(jax.grad(lambda a: energy(a, 'jax'))(1.0))
    amplitude = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    g_torch = torch.autograd.grad(energy(amplitude, 'torch'), amplitude)[0].item()
    assert abs(g_jax - 2 * value) <= 1e-12 * abs(value), (g_jax, value)
    assert abs(g_torch - 2 * value) <= 1e-12 * abs(value), (g_torch, value)


def _plan_warnings(**pml):
    scene = slab_column(kick=False, z=fd.PML(cells=12, wavelength=1.0, **pml))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        make_plan(scene)
    return [str(w.message) for w in caught if issubclass(w.category, fd.PMLStabilityWarning)]


def test_a_face_below_its_targets_value_warns():
    """``PMLStabilityWarning``, once per stabilized face, when it asks for less than the value
    measured at its target or sets a target weaker than any measured; never for the pure
    stretch (the growth warning speaks for that, when a run grows) or for enough loss."""
    for quiet in (dict(), dict(stabilize=0), dict(reflection=1e-3),
                  dict(stabilize=STABILIZING_LOSS), dict(stabilize=3e-3, reflection=1e-12),
                  dict(stabilize=5e-3, reflection=1e-6), dict(stabilize=1e-2, reflection=1e-4),
                  dict(stabilize=2e-2, reflection=1e-3), dict(stabilize=5e-2, reflection=3e-4)):
        assert _plan_warnings(**quiet) == [], quiet
    low = _plan_warnings(stabilize=3e-3, reflection=1e-4)
    assert len(low) == 2 and all('is 1e-2' in m and 'reflection=0.0001' in m for m in low), low
    assert len(_plan_warnings(stabilize=1e-3)) == 2                     # below 3e-3 at 1e-8
    assert len(_plan_warnings(stabilize=4e-3, reflection=1e-7)) == 2    # 5e-3 applies
    weak = _plan_warnings(stabilize=1.0, reflection=1e-2)
    assert len(weak) == 2 and all('weaker than any' in m for m in weak), weak
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        report = fd.check(slab_column(kick=False, stabilize=3e-3, reflection=1e-4))
    assert any('is 1e-2' in str(note) for note in report.warnings), report


def test_the_door_takes_a_non_negative_static_number():
    from wavebit.fdtd._prepare.cpml import DEFAULT_STABILIZE
    assert DEFAULT_STABILIZE == 0.0                     # opt-in: the default is the pure stretch
    assert fd.PML(cells=12, wavelength=1.0).stabilize is None
    assert fd.PML(cells=12, wavelength=1.0, stabilize=0).stabilize == 0.0
    assert fd.PML(cells=12, wavelength=1.0, stabilize=np.float64(1e-2)).stabilize == 1e-2
    for bad in (-1e-3, float('nan'), float('inf'), 'x', True):
        with pytest.raises(ValueError, match='stabilize'):
            fd.PML(cells=12, wavelength=1.0, stabilize=bad)
    # A value so large that the update stops being a loss is refused at lowering, by name.
    with pytest.raises(ValueError, match='a conductor, not a stabilizer'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fd.run(slab_column(time=1.0, stabilize=1e3))
