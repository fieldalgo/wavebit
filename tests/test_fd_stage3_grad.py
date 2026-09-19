"""Gate 8, second half -- the stage-3 derivatives: ``jax.grad`` and ``torch.autograd`` against
central differences, and the static doors that are still not leaves.

Stage 3's results are post-processing of the ``DFTRegion`` accumulators in the backend's own
arrays, after the scan: nothing in them is a new leaf, but every one of them is a new *route*
from a leaf to a number a design loop differentiates, and each is measured here:

    flux        d(sum of a Flux plane's flux)/d(eps of a Slab) -- the accumulator carried
                through the scan, the end correction, the 2-cell pairing and the trapezoid
    R0          d(sum of the zero order's R)/d(eps of a Box tooth) -- the index-DFT, the
                direction split, the s/p frame and the division by the PlaneWave's own
                incident spectrum (read from its column, which the tooth's eps does not reach)
    amplitude   d(far-field intensity)/d(dipole amplitude) -- the near-to-far transform
    phase       d(far-field intensity)/d(dipole phase)
    axis1d      d(sum of the zero order's R)/d(k of an fd.Axis1D face) -- the ``i k dx``
                product in the curl (main grid and PlaneWave column), the exact-``K`` plane-
                wave algebra and the Axis1D order extraction (no stagger along that axis)

each at three points on both autodiff backends, against a central difference of the same loss
on NumPy. The far-field scene has **two** dipoles and the leaf is the second one's: a lone
dipole's intensity does not depend on its own phase at all (``|e^{i phi}|^2 = 1``), so its
phase gradient would be an exact zero on every backend and a test of nothing. With two, the
intensity is ``a + b cos(phi - c) A + d A^2`` in the second dipole's amplitude ``A`` and phase
``phi`` -- exactly quadratic in ``A``, so a central difference in the amplitude has no
truncation error at all, and :func:`test_the_intensity_is_exactly_quadratic_in_the_amplitude`
holds the autodiff to it at a step a thousand times larger than the budget's.

**The plan is built once per family from a concrete scene** and every point is lowered against
it, as a sweep would -- the stage-1 and stage-2 rule: a gradient compared against a difference of
two *different plans* would be measuring the planner. ``Run(courant=0.5)`` fixes ``dt`` apart
from the material, and the Diffraction window is given (``orders=(1, 0)``), because a default
window is refused over a traced medium.

**Each run is short** (200 to 400 steps): the DFT of a field that has not decayed is still a
differentiable function of the leaf, and the gate is about the route, not the physics. Every
loss is order ``1e-2`` and every gradient order ``1e-2``, well away from zero.

**The step.** Measured 2026-09-19 at the middle point of each family (``jax.grad`` against the
central difference, relative error, ``h`` relative to ``max(1, |p|)``):

    h =                  1e-3     1e-4     1e-5     1e-6     1e-7
    flux   eps 2.25      1.2e-7   1.2e-9   1.1e-11  2.6e-10  7.6e-9
    R0     eps 2.25      5.6e-8   5.6e-10  1.3e-11  7.9e-11  3.6e-9
    phase      0.7       1.1e-7   1.1e-9   2.5e-11  4.0e-10  2.1e-9
    amplitude  0.8       2.6e-13  5.7e-13  9.6e-13  1.8e-10  3.6e-10
    axis1d k   2.655     5.0e-7   5.0e-9   7.3e-11  8.0e-10  1.6e-8

-- truncation falling as ``h^2`` to a valley at ``1e-5`` and cancellation rising below it, except
the amplitude, which has no truncation (the loss is a quadratic) and is best at the largest
step. So :data:`H_STEP` is ``1e-5`` for the four curved leaves and ``1e-3`` for the amplitude.
(The far-field rows were measured on the scene before its box grew from 0.2 to 0.4 to enclose
both stamps; the Axis1D row at ``k`` 2.0 and 3.2 reads 3.5e-11 and 2.2e-10 at ``1e-5``, and JAX
and Torch agree to 2e-15 there.)

**The budget** is the stage-2 one, ``|g_ad - g_fd| <= 1e-7 |g_fd| + 1e-10 |f|``, some 1e4 above
the valley, and :func:`test_the_budget_rejects_a_gradient_a_part_in_a_million_wrong` shows it
fails a gradient a part in a million wrong. JAX and Torch equal each other to 5e-16 relative at
every middle point (measured), held at 1e-10.

**The checkpoint schedule changes no gradient** through a DFT accumulator either: the
accumulators are carried state, and a checkpointed block that moved one in place under its own
recompute would give a different gradient (or, on Torch, a version-counter error).

**The static doors are not leaves.** A monitor's wavelengths fix the accumulator's shape and the
phase each step multiplies by; a ``Plane``'s position and a ``Diffraction`` plane's (and its
reference and incident planes) pick node indices; a ``Box``'s size and centre fix which cells a
structure fills; ``Run(time=)`` is a step count; ``Axis1D(k_max=)`` fixes the CFL bound. Each is
refused by name when a tracer or a ``requires_grad`` tensor arrives, and the concrete spelling
is taken.
"""
import warnings

import numpy as np
import pytest

import wavebit.fd as fd
from wavebit.fd._lower import lower
from wavebit.fd._run import _results, execute
from wavebit.fd._runtime import runtime

DX = 0.05
PML_CELLS = 8
WAVELENGTHS = (0.9, 1.0, 1.1)
GRATING_WAVELENGTHS = (0.95, 1.0, 1.05)

#: The finite-difference step per leaf, relative to ``max(1, |point|)``: near the bottom of each
#: leaf's own valley, measured; see the module docstring.
H_STEP = {'flux': 1e-5, 'R0': 1e-5, 'amplitude': 1e-3, 'phase': 1e-5, 'axis1d': 1e-5}

#: ``|g_ad - g_fd| <= RTOL |g_fd| + FTOL |f|``; see the module docstring.
RTOL, FTOL = 1e-7, 1e-10

#: The Axis1D family's operating point: the transverse k of a 25-degree wave at the band
#: centre, which the plan checks the PlaneWave against (the other points are lowered through
#: that plan, as a sweep would), and the static bound its CFL is checked with.
AXIS1D_K = 2.0 * np.pi * np.sin(np.radians(25.0))
AXIS1D_K_MAX = 4.0

#: Three points per leaf, spread so that a gradient right at one operating point is not
#: mistaken for a gradient. The permittivity stops at 2.5: at ``dx = 0.05`` the band's shortest
#: wavelength (0.642) is 8.1 cells in it, and the lowering refuses a structure under 8 (``eps =
#: 4`` is 6.4 cells, refused by name -- the stage-1 gate's 4.0 had a finer grid).
POINTS = {
    'flux': (1.44, 2.25, 2.5),
    'R0': (1.44, 2.25, 2.5),
    'amplitude': (0.5, 0.8, 1.6),
    'phase': (0.3, 0.7, 2.2),
    'axis1d': (2.0, AXIS1D_K, 3.2),
}

#: The concrete operating point each family's plan is built from.
DEFAULTS = {'flux': 2.25, 'R0': 2.25, 'amplitude': 0.8, 'phase': 0.7, 'axis1d': AXIS1D_K}

#: The far-field directions the intensity loss sums over, degrees.
THETA = np.array([30.0, 90.0, 140.0])
PHI = np.array([20.0, 100.0, 250.0])


def _band():
    return fd.Band(wavelength=1.0, df=0.15)


def _layer():
    return fd.PML(cells=PML_CELLS, wavelength=1.0, allow_thin=True)


def stack_scene(eps):
    """A 25-degree s PlaneWave in a 4 x 3 Bloch cell through a 3-cell slab of ``eps``, a Flux
    plane below it (the transmitted flux, negative: the plane's normal is +z)."""
    pw = fd.PlaneWave('pw', z=0.3, direction='-z', theta=25.0, phi=30.0, polarization='s',
                      band=_band())
    return fd.Scene(domain=fd.Domain(size=(4 * DX, 3 * DX, (24 + 2 * PML_CELLS) * DX)),
                    boundaries=fd.Boundaries(**fd.Bloch.from_source(pw), z=_layer()),
                    resolution=fd.Resolution(dx=DX), run=fd.Run(steps=400, courant=0.5),
                    sources=(pw,),
                    structures=(fd.Structure(fd.Slab(z=(-3 * DX, 0.0)), fd.Material(eps=eps)),),
                    monitors=(fd.Flux('t', plane=fd.Plane(z=-0.35), wavelengths=WAVELENGTHS),))


def grating_scene(eps):
    """A cell-snapped Box tooth of ``eps`` (12 x 4 cells in a 24-cell period, one cell in y)
    under a normally incident s PlaneWave, with a reflect Diffraction monitor over three orders."""
    pw = fd.PlaneWave('pw', z=0.3, direction='-z', polarization='s', band=_band())
    height = 4 * DX
    tooth = fd.Box(size=(12 * DX, 10.0, height), center=(0.0, 0.0, -height / 2))
    return fd.Scene(domain=fd.Domain(size=(24 * DX, DX, (26 + 2 * PML_CELLS) * DX),
                                     center=(0.0, 0.0, -0.05)),
                    boundaries=fd.Boundaries(**fd.Bloch.from_source(pw), z=_layer()),
                    resolution=fd.Resolution(dx=DX), run=fd.Run(steps=400, courant=0.5),
                    sources=(pw,), structures=(fd.Structure(tooth, fd.Material(eps=eps)),),
                    monitors=(fd.Diffraction('r', z=0.45, side='reflect',
                                             wavelengths=GRATING_WAVELENGTHS,
                                             reference_plane=-0.5 * DX, orders=(1, 0)),))


def farfield_scene(amplitude, phase):
    """Two ``Ez`` dipoles in a 28^3 PML box inside a FarField box; the second carries the leaf.

    The box (0.4, eight cells) encloses both stamps with the cell to spare a flux surface must
    keep, which is what sizes the interior at twelve cells."""
    a = fd.Dipole('a', center=(0.013, -0.02, 0.007), component='Ez', band=_band())
    b = fd.Dipole('b', center=(-0.06, 0.04, -0.03), component='Ez', band=_band(),
                  amplitude=amplitude, phase=phase)
    return fd.Scene(domain=fd.Domain(size=((12 + 2 * PML_CELLS) * DX,) * 3),
                    boundaries=fd.Boundaries.open(_layer()),
                    resolution=fd.Resolution(dx=DX), run=fd.Run(steps=200, courant=0.5),
                    sources=(a, b),
                    monitors=(fd.FarField('ff', box=fd.Box(size=(0.4,) * 3),
                                          wavelengths=WAVELENGTHS),))


def axis1d_scene(k):
    """A 25-degree s PlaneWave travelling in the y-z plane (``phi = 90``) through a 3-cell slab
    of ``eps = 2.25``, in a cell whose y axis is an ``fd.Axis1D`` carrying the leaf ``k``; a
    reflect Diffraction plane reads the zero order. The route is the ``i k dx`` product in the
    curl (main grid and column alike), the exact-``K`` plane-wave algebra and the Axis1D order
    extraction with no stagger along y."""
    pw = fd.PlaneWave('pw', z=0.3, direction='-z', theta=25.0, phi=90.0, polarization='s',
                      band=_band())
    return fd.Scene(domain=fd.Domain(size=(2 * DX, DX, (24 + 2 * PML_CELLS) * DX)),
                    boundaries=fd.Boundaries(x=fd.Periodic(),
                                             y=fd.Axis1D(k=k, k_max=AXIS1D_K_MAX), z=_layer()),
                    resolution=fd.Resolution(dx=DX), run=fd.Run(steps=400, courant=0.5),
                    sources=(pw,),
                    structures=(fd.Structure(fd.Slab(z=(-3 * DX, 0.0)),
                                             fd.Material(eps=2.25)),),
                    monitors=(fd.Diffraction('r', z=0.45, side='reflect',
                                             wavelengths=WAVELENGTHS,
                                             reference_plane=-0.5 * DX, orders=0),))


def scene_of(family, value):
    if family == 'axis1d':
        return axis1d_scene(value)
    if family == 'flux':
        return stack_scene(value)
    if family == 'R0':
        return grating_scene(value)
    if family == 'amplitude':
        return farfield_scene(value, DEFAULTS['phase'])
    return farfield_scene(DEFAULTS['amplitude'], value)


def _quiet(fn, *args):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', fd.CoarseMeshWarning)
        warnings.simplefilter('ignore', fd.UnsettledFieldWarning)
        return fn(*args)


_PLANS = {}


def plan_of(family):
    """The plan of ``family``, built once from the concrete operating point."""
    if family not in _PLANS:
        _PLANS[family] = _quiet(fd.plan, scene_of(family, DEFAULTS[family]))
    return _PLANS[family]


def _run(family, value, backend, schedule=None):
    rt = runtime(backend)
    plan = plan_of(family)
    if schedule is not None:
        plan = plan.with_static(schedule=schedule)
    scene = scene_of(family, value)
    params = lower(scene, plan, rt)
    state = execute(plan, params, rt)
    return _quiet(_results, scene, plan, state, rt, params)


def loss(family, value, backend, schedule=None):
    """The family's result, summed to one number, differentiable in ``value``."""
    res = _run(family, value, backend, schedule)
    if family == 'flux':
        return res['t'].flux.sum()
    if family == 'R0':
        return res['r'].orders.given.R[:, 0, 1].sum()          # (nf, n_v, n_u): m = 0
    if family == 'axis1d':
        return res['r'].orders.given.R[:, 0, 0].sum()          # the single order (0, 0)
    return res['ff'].intensity(THETA, PHI).sum()


def central_difference(family, point, relative=None):
    """``(f(x+h) - f(x-h)) / 2h`` on NumPy, with ``h`` relative to ``max(1, |point|)``."""
    h = (H_STEP[family] if relative is None else relative) * max(1.0, abs(point))
    return (float(loss(family, point + h, 'numpy'))
            - float(loss(family, point - h, 'numpy'))) / (2.0 * h)


def jax_gradient(family, point, schedule=None):
    jax = pytest.importorskip('jax')
    runtime('jax')              # x64 lifted before jax.grad makes the leaf (the stage-2 trap)
    return float(jax.grad(lambda v: loss(family, v, 'jax', schedule))(point))


def torch_gradient(family, point, schedule=None):
    torch = pytest.importorskip('torch')
    value = torch.tensor(point, dtype=torch.float64, requires_grad=True)
    return float(torch.autograd.grad(loss(family, value, 'torch', schedule), value)[0])


def close(got, want, scale):
    """The budget of the module docstring, as a predicate, so a mutant can be run through it."""
    return abs(got - want) <= RTOL * abs(want) + FTOL * abs(scale)


def _cases():
    return [pytest.param(family, index, id=f'{family}-{index}')
            for family in POINTS for index in range(3)]


# ---------------------------------------------------------------------------------------------
# Four routes, both autodiff backends, three points each
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('family,index', _cases())
def test_the_gradient_is_the_central_difference(family, index):
    """``jax.grad`` and ``torch.autograd`` both reproduce the central difference.

    The reference is asserted to be well away from zero: a chain that detached somewhere (an
    accumulator read through ``numpy()``, a result built from a detached copy) gives 0.0, which
    a relative tolerance would pass against a reference that was zero too.
    """
    point = POINTS[family][index]
    f = float(loss(family, point, 'numpy'))
    reference = central_difference(family, point)
    assert abs(reference) > 1e-2 * abs(f), (reference, f)
    for name, gradient in (('jax', jax_gradient(family, point)),
                           ('torch', torch_gradient(family, point))):
        assert close(gradient, reference, f), (family, point, name, gradient, reference)


@pytest.mark.parametrize('family', list(POINTS))
def test_jax_and_torch_agree_on_the_gradient(family):
    """The two autodiffs held to each other at the middle point, with no finite difference's
    round-off floor in between."""
    point = POINTS[family][1]
    a, b = jax_gradient(family, point), torch_gradient(family, point)
    assert abs(a - b) <= 1e-10 * abs(a), (family, a, b)


def test_the_intensity_is_exactly_quadratic_in_the_amplitude():
    """The one closed form in this file: the far field is linear in each dipole's amplitude, so
    the intensity is a quadratic in it and a central difference at *any* step is its
    derivative. At ``h = 0.1`` -- a hundred times the budget's step, where a curved loss would
    carry a 1e-3 truncation -- the autodiff is held to 1e-11."""
    point = POINTS['amplitude'][1]
    wide = central_difference('amplitude', point, relative=0.1)
    for gradient in (jax_gradient('amplitude', point), torch_gradient('amplitude', point)):
        assert abs(gradient - wide) <= 1e-11 * abs(wide), (gradient, wide)


def test_the_step_sits_in_the_valley():
    """The evidence for ``H_STEP['flux'] = 1e-5``, measured rather than quoted: from ``1e-3`` to
    ``1e-4`` the error falls by the ``h^2`` law's two decades, ``1e-5`` is well under the
    budget, and at ``1e-7`` cancellation has pushed it back up."""
    point = POINTS['flux'][1]
    gradient = jax_gradient('flux', point)
    errors = {h: abs(central_difference('flux', point, h) - gradient) / abs(gradient)
              for h in (1e-3, 1e-4, 1e-5, 1e-7)}
    assert 50.0 < errors[1e-3] / errors[1e-4] < 200.0, errors            # the h^2 law
    assert errors[1e-3] > RTOL, errors                                    # truncation above
    assert errors[1e-5] < 0.1 * RTOL, errors                              # the valley
    assert errors[1e-7] > errors[1e-5], errors                            # cancellation below


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('family', ['flux', 'phase'])
def test_a_checkpoint_schedule_does_not_change_the_gradient(family, backend):
    """``Plan.schedule = 7`` (checkpointed blocks of 7 steps, which divide neither 400 nor 200,
    so the remainder path runs too) gives the plain scan's gradient through a DFT accumulator."""
    pytest.importorskip(backend)
    gradient = jax_gradient if backend == 'jax' else torch_gradient
    point = POINTS[family][1]
    plain = gradient(family, point)
    blocked = gradient(family, point, schedule=7)
    assert plain != 0.0
    assert abs(blocked - plain) <= 1e-12 * abs(plain), (plain, blocked)


def test_the_budget_rejects_a_gradient_a_part_in_a_million_wrong():
    """A tolerance nothing can fail is not a tolerance."""
    point = POINTS['R0'][1]
    f = float(loss('R0', point, 'numpy'))
    reference = central_difference('R0', point)
    assert close(reference, reference, f)
    assert not close(reference * 1.000001, reference, f)


# ---------------------------------------------------------------------------------------------
# The other half: a traced value in a static door, refused by name
# ---------------------------------------------------------------------------------------------

def _static_doors():
    """``(build, pattern, concrete)`` for each stage-3 static door."""
    return (
        pytest.param(lambda v: fd.Flux('f', plane=fd.Plane(z=0.0), wavelengths=(v, 1.0)),
                     r'Flux\(wavelengths=\.\.\.\) is static', 0.9, id='flux-wavelengths'),
        pytest.param(lambda v: fd.Field('f', plane=fd.Plane(z=0.0), components=('Ex',),
                                        wavelengths=(v,)),
                     r'Field\(wavelengths=\.\.\.\) is static', 0.9, id='field-wavelengths'),
        pytest.param(lambda v: fd.FarField('f', box=fd.Box(size=(0.2,) * 3), wavelengths=(v,)),
                     r'FarField\(wavelengths=\.\.\.\) is static', 0.9,
                     id='farfield-wavelengths'),
        pytest.param(lambda v: fd.Diffraction('r', z=0.4, wavelengths=(v,)),
                     r'Diffraction\(wavelengths=\.\.\.\) is static', 0.9,
                     id='diffraction-wavelengths'),
        pytest.param(lambda v: fd.Diffraction('r', z=v, wavelengths=(1.0,)),
                     r'Diffraction\(z=\.\.\.\) is static', 0.4, id='diffraction-plane'),
        pytest.param(lambda v: fd.Diffraction('r', z=0.4, wavelengths=(1.0,),
                                              reference_plane=v),
                     r'Diffraction\(reference_plane=\.\.\.\) is static', -0.025,
                     id='diffraction-reference-plane'),
        pytest.param(lambda v: fd.Diffraction('t', z=-0.4, side='transmit', wavelengths=(1.0,),
                                              incident_plane=v),
                     r'Diffraction\(incident_plane=\.\.\.\) is static', -0.025,
                     id='diffraction-incident-plane'),
        pytest.param(lambda v: fd.Plane(z=v), r'Plane\(z=\.\.\.\) is static', 0.2,
                     id='plane-position'),
        pytest.param(lambda v: fd.Plane(z=0.2, x=(v, 0.3)), r'Plane\(x=\.\.\.\) is static',
                     -0.3, id='plane-extent'),
        pytest.param(lambda v: fd.Structure(fd.Box(size=(v, 0.2, 0.2)), fd.Material(eps=2.25)),
                     r'Box\(size=\.\.\.\) is static', 0.3, id='box-structure-size'),
        pytest.param(lambda v: fd.Structure(fd.Box(size=(0.2, 0.2, 0.2), center=(0.0, 0.0, v)),
                                            fd.Material(eps=2.25)),
                     r'Box\(center=\.\.\.\) is static', -0.1, id='box-structure-edge'),
        pytest.param(lambda v: fd.Run(time=v), r'Run\(time=\.\.\.\) is static', 12.0,
                     id='run-time'),
        pytest.param(lambda v: fd.Axis1D(k=1.0, k_max=v), r'Axis1D\(k_max=\.\.\.\) is static',
                     2.0, id='axis1d-k-max'),
    )


@pytest.mark.parametrize('build,pattern,concrete', _static_doors())
def test_a_jax_tracer_in_a_static_door_is_refused_by_name(build, pattern, concrete):
    """Inside ``jax.grad`` the value is a tracer, and the door says which field refused it."""
    jax = pytest.importorskip('jax')

    def through(v):
        build(v)
        return v * 2.0

    with pytest.raises(ValueError, match=pattern) as raised:
        jax.grad(through)(float(concrete))
    assert 'not a leaf' in str(raised.value)
    assert build(concrete) is not None                        # and the concrete spelling is taken


@pytest.mark.parametrize('build,pattern,concrete', _static_doors())
def test_a_torch_leaf_in_a_static_door_is_refused_by_name(build, pattern, concrete):
    """The case the refusal exists for: ``float()`` on a live tensor succeeds and loses the graph."""
    torch = pytest.importorskip('torch')
    live = torch.tensor(float(concrete), dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValueError, match=pattern):
        build(live)
    assert build(concrete) is not None


def test_the_stage_three_leaves_pass_the_same_doors():
    """The refusals above are a decision about *which* numbers are static, not about tracers:
    the same tracer and the same live tensor go through a Box structure's material, a Slab's
    material and a dipole's amplitude and phase in a monitored scene without a word."""
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')
    built = []

    def through(v):
        built.append((stack_scene(v), grating_scene(v), farfield_scene(v, v)))
        return v * 2.0

    jax.grad(through)(1.7)
    assert len(built) == 1
    live = torch.tensor(1.7, dtype=torch.float64, requires_grad=True)
    assert all(e.requires_grad for e in grating_scene(live).structures[0].material.eps)
    assert farfield_scene(live, 0.0).sources[1].amplitude.requires_grad
