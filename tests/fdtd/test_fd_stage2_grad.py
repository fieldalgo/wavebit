"""Gate 6, second half -- the source derivatives: ``jax.grad`` and ``torch.autograd`` against
central differences, and the static doors that are still not leaves.

Stage 2 made five kinds of number into leaves a gradient may flow through, and each reaches the
answer by a different route:

    amplitude   a factor on the waveform row (and, through the column, on every correction)
    phase       inside the carrier's sine
    delay       inside the envelope and the carrier, through ``s = t - t0 - delay``
    centre      the dipole's stamp weights, ``xp.where`` of the traced offset from each Yee point
    wavelength  the band centre: ``f0``, and nothing else (``tau`` and ``t0`` come from ``df``)

and three paths carry them into the grid: a dipole's stamp (``before_E``), a TFSF box's six
faces fed from an incident column carried in the term's memory, and a PlaneWave sheet in a
Bloch cell, which is complex and multiplies its corrections by a traced phase pattern. Each
leaf is measured here at three points on both autodiff backends, against a central difference
of the same loss computed on NumPy: ``d/dp sum(|E|^2 + |H|^2)`` at the last step.

    dipole      amplitude, phase, delay, centre (x, inside a declared range), band wavelength
    tfsf        amplitude, phase, delay
    plane_wave  amplitude, phase, delay

**The plan is built once per family from a concrete scene**, and every point is lowered against
it, as a sweep would. That is also what makes the band-centre gradient a gradient of the
*waveform*: the plan fixes ``dx``, ``dt``, the step count and the PML's graded profile (its
``alpha_max`` is set from the band centre the concrete scene had), so a central difference in
the wavelength moves ``f0`` and nothing static. A gradient compared against a difference of
two *different plans* would be measuring the planner.

**Each run is short on purpose**: the loss is taken while the source is still driving, at 92%
of its envelope peak (``delay = -1.5`` brings the peak from step 273 to step 169; the runs stop
at 140), so every leaf moves the loss by order one and none of the gradients is the derivative
of a field that has mostly left through the PML.

**How the step was chosen, per leaf.** A central difference carries a truncation error
``h^2 f'''/6`` and a round-off error of order ``eps_machine |f| / h``, and the two balance where
``h`` is about ``1e-5`` of the length over which the loss *curves* in that leaf -- a radian for
the phase, but a fraction of a period (``1/(2 pi f0) = 0.16``) for the delay and the band centre,
and a small fraction of a cell for the dipole's position. Measured on these scenes at the middle
point of each leaf (``jax.grad`` against the central difference, relative error, ``h`` relative
to ``max(1, |p|)``):

    h =                        1e-3     1e-4     1e-5     1e-6     1e-7
    dipole phase       1.0     6.7e-7   6.7e-9   1.2e-11  6.1e-10  5.4e-9
    dipole delay      -1.5     6.2e-5   6.2e-7   6.2e-9   4.3e-11  2.9e-10
    dipole centre      0.004   4.2e-3   4.2e-5   4.2e-7   4.2e-9   1.8e-9
    dipole wavelength  1.0     2.3e-4   2.3e-6   2.3e-8   7.2e-10  1.3e-8
    tfsf delay        -1.5     7.7e-5   7.7e-7   7.6e-9   7.5e-11  1.4e-10
    plane_wave delay  -1.5     2.4e-3   2.4e-5   2.4e-7   3.2e-9   2.0e-8

-- truncation falling as ``h^2`` (a decade of ``h``, two of error) until cancellation takes over
and rises as ``1/h``. So :data:`H_STEP` takes each leaf near the bottom of its own valley:
``1e-5`` for the phase, ``1e-6`` for the delay and the band centre, ``1e-7`` for the position. One
step for every leaf would not do: ``1e-5`` leaves the position at 4e-7 and the PlaneWave's delay
at 2.4e-7, truncation passing for disagreement. The amplitude is in no row because the loss is
exactly quadratic in it -- the central difference has no truncation at all, and
:func:`test_the_amplitude_gradient_is_twice_the_loss_over_the_amplitude` holds it to the closed
form as well. :func:`test_the_step_sits_in_the_valley` measures the band centre's row rather
than only quoting it.

**The budget** is ``|g_ad - g_fd| <= 1e-7 |g_fd| + 1e-10 |f|``. Measured 2026-09-19 at the chosen
steps over all 33 points: 3.2e-9 relative at worst (the PlaneWave delay at -1.5, where the loss
is nearly stationary in the delay, ``dL/dp = 0.04 L``), then 1.8e-9 (the dipole centre), and
JAX and Torch equal to each other to 1.5e-14 everywhere. So the budget is 30x above the worst
point, and it fails a gradient a part in a million wrong (:func:`test_the_budget_rejects_a_gradient_a_part_in_a_million_wrong`). The absolute part
is scaled by the loss because a phase or delay point can sit near where the loss is stationary
in that leaf, and there the difference of two runs cancels to the round-off of the loss itself.

**JAX is started before the first gradient.** ``jax.grad(f)(0.5)`` turns ``0.5`` into a JAX
array *before* ``f`` builds a runtime, so in a fresh process -- x64 not yet lifted -- the leaf is
float32 and so is its gradient: measured 4.2e-8 off Torch on the first point of the first run
of this file, and exact on every later one (the x64 import-order trap; the lowering says so with
a precision warning). :func:`jax_gradient` builds the JAX runtime first.

**A checkpoint schedule changes no gradient**: ``Plan.schedule`` blocks the scan into
checkpointed pieces on JAX and Torch, and the gradient of every family (and of a TFSF over a
substrate, whose column carries a driver) is the plain scan's to 1e-12.

**The static doors are not leaves**, and the other half of this file is that they still say
so, naming every leaf that is: a TFSF ``Region``, a dipole's ``range``, the step count (``Run(steps=)``, ``Run(periods=)``, and
a ``periods`` run whose band centre is traced), a PlaneWave's sheet position and angle and a
band's ``cutoff`` fix the plan, so a tracer or a ``requires_grad`` tensor in one of them is
refused by the field's name.
"""
import pytest

import wavebit.fdtd._legacy as fd
from wavebit.fdtd._execution import execute
from wavebit.fdtd._runtime import runtime

from tests.fixtures.fdtd.fd_helpers import torch_forward_paths

DX = 0.05
PML_CELLS = 8
STEPS = 140

#: Brings the pulse peak from ``t0 = 3.94`` (step 273) to step 169, so the runs, which stop at
#: 140, end with the source still driving at 92% of its envelope peak.
DELAY = -1.5

#: The finite-difference step per leaf, relative to ``max(1, |point|)``: near the bottom of
#: each leaf's own valley, measured; see the module docstring.
H_STEP = {'amplitude': 1e-5, 'phase': 1e-5, 'delay': 1e-6, 'wavelength': 1e-6, 'center': 1e-7}

#: ``|g_ad - g_fd| <= RTOL |g_fd| + FTOL |f|``; measured worst 3.2e-9, see the module docstring.
RTOL, FTOL = 1e-7, 1e-10

#: The dipole's declared range along x; the centre moves inside it and y, z stay put.
RANGE_X = (-0.025, 0.025)
CENTER = (0.004, -0.013, 0.007)

#: The three points per leaf, spread so that a gradient that is right at one operating point is
#: not mistaken for a gradient. The centre points are off every node and every half node.
POINTS = {
    ('dipole', 'amplitude'): (0.5, 1.0, 2.0),
    ('dipole', 'phase'): (0.3, 1.0, 2.2),
    ('dipole', 'delay'): (-1.8, -1.5, -1.1),
    ('dipole', 'center'): (-0.021, 0.004, 0.018),
    ('dipole', 'wavelength'): (0.95, 1.0, 1.06),
    ('tfsf', 'amplitude'): (0.5, 1.0, 2.0),
    ('tfsf', 'phase'): (0.3, 1.0, 2.2),
    ('tfsf', 'delay'): (-1.8, -1.5, -1.1),
    ('plane_wave', 'amplitude'): (0.5, 1.0, 2.0),
    ('plane_wave', 'phase'): (0.3, 1.0, 2.2),
    ('plane_wave', 'delay'): (-1.8, -1.5, -1.1),
}

#: Each family's concrete operating point: the value every leaf not being varied takes.
DEFAULTS = {'amplitude': 1.0, 'phase': 0.0, 'delay': DELAY, 'center': CENTER[0],
            'wavelength': 1.0}


def _values(parameter, value):
    out = dict(DEFAULTS)
    out[parameter] = value
    return out


def _band(wavelength):
    return fd.Band(wavelength=wavelength, df=0.15)


def dipole_scene(amplitude, phase, delay, center, wavelength):
    """An ``Ez`` dipole in a 22^3 box (6 cells of vacuum, an 8-cell PML all round)."""
    source = fd.Dipole('d', center=(center, CENTER[1], CENTER[2]), component='Ez',
                       band=_band(wavelength), amplitude=amplitude, phase=phase, delay=delay,
                       range=(RANGE_X, (CENTER[1],) * 2, (CENTER[2],) * 2))
    side = (6 + 2 * PML_CELLS) * DX
    return fd.Scene(domain=fd.Domain(size=(side,) * 3),
                    boundaries=fd.Boundaries.open(fd.PML(cells=PML_CELLS, wavelength=1.0,
                                                         allow_thin=True)),
                    resolution=fd.Resolution(dx=DX), run=fd.Run(steps=STEPS), sources=(source,))


def tfsf_scene(amplitude, phase, delay, center=None, wavelength=1.0):
    """A 4-cell ``-z`` TFSF box with two cells of scattered region, in a 24^3 PML box."""
    source = fd.TFSF('t', box=fd.Region(size=(4 * DX,) * 3), direction='-z', polarization='y',
                     band=_band(wavelength), amplitude=amplitude, phase=phase, delay=delay)
    side = (8 + 2 * PML_CELLS) * DX
    return fd.Scene(domain=fd.Domain(size=(side,) * 3),
                    boundaries=fd.Boundaries.open(fd.PML(cells=PML_CELLS, wavelength=1.0,
                                                         allow_thin=True)),
                    resolution=fd.Resolution(dx=DX), run=fd.Run(steps=STEPS), sources=(source,))


def plane_wave_scene(amplitude, phase, delay, center=None, wavelength=1.0):
    """A 25-degree s PlaneWave in a 4 x 3 x 36 Bloch cell: complex, with a traced pattern."""
    pw = fd.PlaneWave('pw', z=0.25, direction='-z', theta=25.0, phi=40.0, polarization='s',
                      band=_band(wavelength), amplitude=amplitude, phase=phase, delay=delay)
    return fd.Scene(domain=fd.Domain(size=(4 * DX, 3 * DX, (20 + 2 * PML_CELLS) * DX)),
                    boundaries=fd.Boundaries(**fd.Bloch.from_source(pw),
                                             z=fd.PML(cells=PML_CELLS, wavelength=1.0,
                                                      allow_thin=True)),
                    resolution=fd.Resolution(dx=DX), run=fd.Run(steps=STEPS), sources=(pw,))


BUILDERS = {'dipole': dipole_scene, 'tfsf': tfsf_scene, 'plane_wave': plane_wave_scene}
_PLANS = {}


def plan_of(family):
    """The plan of ``family``, built once from the concrete operating point."""
    if family not in _PLANS:
        _PLANS[family] = fd.plan(BUILDERS[family](**DEFAULTS))
    return _PLANS[family]


def loss(family, parameter, value, backend):
    """``sum(|E|^2) + sum(|H|^2)`` of the final state: one number, differentiable in ``value``."""
    rt = runtime(backend)
    plan = plan_of(family)
    scene = BUILDERS[family](**_values(parameter, value))
    state = execute(plan, fd.lower(scene, plan, rt), rt)
    return (abs(state['E']) ** 2).sum() + (abs(state['H']) ** 2).sum()


def central_difference(family, parameter, point, relative=None):
    """``(f(x+h) - f(x-h)) / 2h`` on NumPy, with ``h`` relative to ``max(1, |point|)``."""
    h = (H_STEP[parameter] if relative is None else relative) * max(1.0, abs(point))
    return (float(loss(family, parameter, point + h, 'numpy'))
            - float(loss(family, parameter, point - h, 'numpy'))) / (2.0 * h)


def jax_gradient(family, parameter, point):
    jax = pytest.importorskip('jax')
    runtime('jax')              # x64 lifted before jax.grad makes the leaf; see the module
    return float(jax.grad(lambda v: loss(family, parameter, v, 'jax'))(point))


def torch_gradient(family, parameter, point):
    torch = pytest.importorskip('torch')
    value = torch.tensor(point, dtype=torch.float64, requires_grad=True)
    return float(torch.autograd.grad(loss(family, parameter, value, 'torch'), value)[0])


def close(got, want, scale):
    """The budget of the module docstring, as a predicate, so a mutant can be run through it."""
    return abs(got - want) <= RTOL * abs(want) + FTOL * abs(scale)


def _cases():
    return [pytest.param(family, parameter, index, id=f'{family}-{parameter}-{index}')
            for (family, parameter) in POINTS for index in range(3)]


# ---------------------------------------------------------------------------------------------
# Eleven leaves, both autodiff backends, three points each
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('family,parameter,index', _cases())
def test_the_gradient_is_the_central_difference(family, parameter, index):
    """``jax.grad`` and ``torch.autograd`` both reproduce ``d(final energy)/d(leaf)``.

    One test per point rather than a loop, so a failure names the leaf and the operating point.
    The reference is asserted to be well away from zero: a chain that quietly detached
    somewhere gives 0.0, which a relative tolerance would pass against a reference that was
    zero too.
    """
    point = POINTS[family, parameter][index]
    f = float(loss(family, parameter, point, 'numpy'))
    reference = central_difference(family, parameter, point)
    assert abs(reference) > 1e-3 * f, (reference, f)
    for name, gradient in (('jax', jax_gradient(family, parameter, point)),
                           ('torch', torch_gradient(family, parameter, point))):
        assert close(gradient, reference, f), (family, parameter, point, name, gradient,
                                               reference)


@pytest.mark.parametrize('family,parameter', list(POINTS))
def test_jax_and_torch_agree_on_the_gradient(family, parameter):
    """The two autodiffs held to each other at the middle point, with no round-off floor of a
    finite difference in between -- where a backend-specific defect (Torch's out-of-place
    column CPML, its ``SlabMul`` backward for the hard source) shows first."""
    point = POINTS[family, parameter][1]
    a, b = jax_gradient(family, parameter, point), torch_gradient(family, parameter, point)
    assert abs(a - b) <= 1e-10 * abs(a), (family, parameter, a, b)


def test_the_amplitude_gradient_is_twice_the_loss_over_the_amplitude():
    """The one closed form in this file: the field is linear in the amplitude, so the energy is
    quadratic in it and ``dL/dA = 2 L / A`` exactly, on every path. It needs no finite
    difference, so it holds the autodiff to 1e-12 rather than to the budget."""
    for family in ('dipole', 'tfsf', 'plane_wave'):
        point = POINTS[family, 'amplitude'][2]
        f = float(loss(family, 'amplitude', point, 'numpy'))
        want = 2.0 * f / point
        for gradient in (jax_gradient(family, 'amplitude', point),
                         torch_gradient(family, 'amplitude', point)):
            assert abs(gradient - want) <= 1e-12 * abs(want), (family, gradient, want)


def test_the_step_sits_in_the_valley():
    """The evidence for ``H_STEP['wavelength'] = 1e-6``, measured rather than quoted.

    The band centre, where ``f0`` multiplies a time of order a hundred steps, is a leaf with a
    large third derivative: from ``1e-4`` to ``1e-5`` the error falls by the ``h^2`` law's two
    decades, at ``1e-6`` it is well under the budget, and at ``1e-7`` cancellation has already
    pushed it back up.
    """
    point = POINTS['dipole', 'wavelength'][1]
    gradient = jax_gradient('dipole', 'wavelength', point)
    errors = {h: abs(central_difference('dipole', 'wavelength', point, h) - gradient)
              / abs(gradient) for h in (1e-4, 1e-5, 1e-6, 1e-7)}
    assert 50.0 < errors[1e-4] / errors[1e-5] < 200.0, errors             # the h^2 law
    assert errors[1e-4] > RTOL, errors                                     # truncation above
    assert errors[1e-6] < 0.1 * RTOL, errors                               # the valley
    assert errors[1e-7] > errors[1e-6], errors                             # cancellation below


def _substrate_scene(amplitude):
    """The TFSF family's box over a glass half-space: the layered column and its driver."""
    scene = tfsf_scene(amplitude, 0.0, DELAY)
    glass = fd.Structure(fd.Slab(z=(-2.0, -DX)), fd.Material(eps=2.25))
    return scene.replace(structures=(glass,))


#: The families the checkpoint gate runs, each a function of the amplitude leaf.
SCHEDULED = {'dipole': lambda a: dipole_scene(a, 0.0, DELAY, CENTER[0], 1.0),
             'tfsf': lambda a: tfsf_scene(a, 0.0, DELAY),
             'tfsf_substrate': _substrate_scene,
             'plane_wave': lambda a: plane_wave_scene(a, 0.0, DELAY)}


def _scheduled_gradient(family, backend, schedule):
    """``d loss / d amplitude`` at 1.3 with the plan's checkpoint block size ``schedule``."""
    import warnings
    make = SCHEDULED[family]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        plan = fd.plan(make(1.0))
    if schedule is not None:
        plan = plan.with_static(schedule=schedule)
    rt = runtime(backend)

    def f(a):
        state = execute(plan, fd.lower(make(a), plan, rt), rt)
        # the series every source records are in the state, and a Record-less loss still
        # has to differentiate through them: they are what the checkpoint used to break
        return (abs(state['E']) ** 2).sum() + (abs(state['H']) ** 2).sum()

    if backend == 'jax':
        import jax
        return float(jax.grad(f)(1.3))
    import torch
    value = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
    return float(torch.autograd.grad(f(value), value)[0])


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('family', list(SCHEDULED))
def test_a_checkpoint_schedule_does_not_change_the_gradient(family, backend):
    """``Plan.schedule = K`` (checkpointed blocks of K steps) gives the plain scan's gradient.

    On Torch a scheduled run could not be differentiated at all when this was written: the
    checkpoint keeps each block's input state to recompute it, and the step writes in place
    wherever nothing is differentiated yet (a CPML psi before the field is live, a series
    accumulator), so an input moved under the recompute and backward raised a version-counter
    error -- on every source run, since every source records a series, and on a stage-0
    ``fd.Record``. 7 does not divide the 140 steps, so the remainder path runs too.
    """
    pytest.importorskip(backend)
    runtime(backend)
    plain = _scheduled_gradient(family, backend, None)
    blocked = _scheduled_gradient(family, backend, 7)
    assert plain != 0.0
    assert abs(blocked - plain) <= 1e-12 * abs(plain), (plain, blocked)


def test_the_budget_rejects_a_gradient_a_part_in_a_million_wrong():
    """A tolerance nothing can fail is not a tolerance: the budget fails a 1.000001x gradient.

    The scale a real defect produces is far above it -- a centre gradient taken through the
    stamp's value but not through the ``xp.where`` branch, or a delay whose sign is flipped in
    the table, moves a gradient by O(1).
    """
    point = POINTS['tfsf', 'phase'][1]
    f = float(loss('tfsf', 'phase', point, 'numpy'))
    reference = central_difference('tfsf', 'phase', point)
    assert close(reference, reference, f)
    assert not close(reference * 1.000001, reference, f)


# ---------------------------------------------------------------------------------------------
# The other half: a traced value in a static door, refused by name
# ---------------------------------------------------------------------------------------------

def _static_doors():
    """``(build, pattern, concrete)`` for each stage-2 static door."""
    def dipole_range(v):
        return fd.Dipole('d', center=(0.0, 0.0, 0.0), component='Ez', band=_band(1.0),
                         range=((v, 0.1), (-0.1, 0.1), (-0.1, 0.1)))

    def pw_position(v):
        return fd.PlaneWave('pw', z=v, direction='-z', band=_band(1.0))

    def pw_theta(v):
        return fd.PlaneWave('pw', z=0.0, direction='-z', theta=v, band=_band(1.0))

    return (
        pytest.param(lambda v: fd.Region(size=(v, 0.3, 0.3)),
                     r'Region\(size=\.\.\.\) is static', 0.3, id='tfsf-box-size'),
        pytest.param(lambda v: fd.Region(size=(0.3, 0.3, 0.3), center=(0.0, v, 0.0)),
                     r'Region\(center=\.\.\.\) is static', 0.05, id='tfsf-box-center'),
        pytest.param(dipole_range, r'Dipole\(range=\.\.\.\) is static', -0.1,
                     id='dipole-range'),
        pytest.param(lambda v: fd.Run(steps=v), r'Run\(steps=\.\.\.\) is static', 140,
                     id='run-steps'),
        pytest.param(lambda v: fd.Run(periods=v), r'Run\(periods=\.\.\.\) is static', 8.0,
                     id='run-periods'),
        pytest.param(pw_position, r'PlaneWave\(z=\.\.\.\) is static', 0.25,
                     id='plane-wave-position'),
        pytest.param(pw_theta, r'PlaneWave\(theta=\.\.\.\) is static', 20.0,
                     id='plane-wave-theta'),
        pytest.param(lambda v: fd.Band(wavelength=1.0, df=0.15, cutoff=v),
                     r'Band\(cutoff=\.\.\.\) is static', 1e-3, id='band-cutoff'),
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


@pytest.mark.parametrize('build,pattern,concrete', _static_doors())
def test_a_torch_forward_mode_value_in_a_static_door_is_refused_by_name(build, pattern, concrete):
    """Forward mode needs the door as much as a leaf does: ``float()`` reads a dual's primal.

    Measured on ``b24a708`` through ``fd.Sphere(radius=)`` and public ``fd.run`` (third review
    of the stencil's preparation): ``torch.func.jvp`` read a derivative of exactly 0.0 and
    ``forward_ad`` no tangent at all, where the central difference is 1.97e-7, because the
    static door asked only for ``requires_grad``. Every static door shares that function, so
    every one of them is asked here.
    """
    torch = pytest.importorskip('torch')

    def through(v):
        build(v)
        return v * 2.0

    for name, path in torch_forward_paths(torch).items():
        with pytest.raises(ValueError, match=pattern) as raised:
            path(through, float(concrete))
        assert 'not a leaf' in str(raised.value), name
    assert build(concrete) is not None


def test_a_traced_band_centre_cannot_count_periods():
    """``Run(periods=)`` is a step count read off the band centre, so a traced centre -- a leaf
    everywhere else -- is refused at the plan, by name, and ``Run(steps=)`` is the spelling that
    carries the same leaf."""
    jax = pytest.importorskip('jax')
    seen = []

    def through(w):
        scene = dipole_scene(1.0, 0.0, DELAY, CENTER[0], w).replace(run=fd.Run(periods=4))
        with pytest.raises(ValueError, match=r'Run\(periods=\.\.\.\) needs a concrete band '
                                             r'centre'):
            fd.plan(scene)
        seen.append(dipole_scene(1.0, 0.0, DELAY, CENTER[0], w))
        return w * 2.0

    jax.grad(through)(1.0)
    assert len(seen) == 1


def test_the_source_leaves_pass_the_same_two_doors():
    """The refusals above are a decision about *which* numbers are static, not about tracers:
    the same tracer and the same live tensor go through a source's amplitude, phase, delay,
    centre (with a range) and band centre without a word."""
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')
    built = []

    def through(v):
        built.append((dipole_scene(v, v, v, 0.0, 1.0 + 0.0 * v),
                      dipole_scene(1.0, 0.0, DELAY, 0.01 * v, 1.0),
                      tfsf_scene(v, v, v), plane_wave_scene(v, v, v)))
        return v * 2.0

    jax.grad(through)(0.9)
    assert len(built) == 1
    live = torch.tensor(0.9, dtype=torch.float64, requires_grad=True)
    assert tfsf_scene(live, 0.0, DELAY).sources[0].amplitude.requires_grad
    assert _band(live).wavelength.requires_grad


def test_the_static_door_refusal_names_the_source_leaves():
    jax = pytest.importorskip('jax')

    def through(v):
        fd.Region(size=(v, 0.3, 0.3))
        return v

    with pytest.raises(ValueError) as raised:
        jax.grad(through)(0.3)
    message = str(raised.value)
    assert 'amplitude' in message and 'delay' in message and 'Band' in message, message
    # every stage's leaves, read from the one tuple the refusal is built from
    from wavebit._common.declarations import TRACED_LEAVES
    for leaf in TRACED_LEAVES:
        assert leaf in message, leaf
    assert 'center' in message and 'material' in message
