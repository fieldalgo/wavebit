"""Gate 8 -- the derivatives: ``jax.grad`` and ``torch.autograd`` against central differences.

Stage 1 made three kinds of number into leaves a gradient may flow through -- a permittivity,
a conductivity and a pole strength -- and each reaches the answer by a different route:

    eps       ``be = (dt/dx)/eps/(1 + dt rate/2)``, a coefficient array the step multiplies by
    sigma     ``ae`` *and* ``be``, through ``rate_e = sigma/eps``
    strength  ``drive`` of the ADE recurrence, whose memory is carried across every step

So there are three things that can be differentiably wrong, and one of them (the pole strength)
is only reachable through state that the scan carries. All three are measured here at three
points each, on both autodiff backends, against a central difference of the same loss computed
on NumPy -- ``d/dp sum(|E|^2 + |H|^2)`` at the last step of a small box whose ``z`` faces are a
PML, so every gradient also passes through the CPML recurrence (functional on JAX, the
out-of-place fallback on Torch).

**How the step was chosen.** A central difference carries a truncation error ``h^2 f'''/6`` and
a round-off error of order ``eps_machine |f| / h``; the two balance at
``h ~ (3 eps_machine |f| / |f'''|)^(1/3)``, which for an ``|f|`` of order 1 and a smooth
``f'''`` is ~1e-5. Measured on this scene, the agreement with ``jax.grad`` over
``h = 1e-3 ... 1e-7`` (relative to the point) runs

    h =              1e-3     1e-4     1e-5     1e-6     1e-7
    eps = 2.25       6.6e-7   6.6e-9   3.5e-11  5.7e-10  1.1e-9
    sigma = 0.5      1.6e-8   1.9e-10  3.5e-10  6.3e-9   4.7e-8
    strength = 60    1.0e-6   1.0e-8   3.1e-11  2.0e-9   4.4e-8

-- a plateau two to three decades wide with 1e-5 in the middle of it, truncation dominating
above and cancellation below. :func:`test_the_step_sits_in_the_plateau` measures that spread
rather than only asserting it, because "the finite difference agrees" means nothing if the
finite difference was taken at a step where it agrees with anything.

**The budget** is ``|g_ad - g_fd| <= 1e-6 |g_fd| + 1e-8``. The absolute floor is doing real
work at one point: at ``eps = 1.44`` the loss is nearly flat (``g = -1.5e-2`` against a loss of
order 1), so the difference of two runs cancels four digits, the finite difference's own noise
floors at ~1.5e-9 and does not fall with ``h`` -- and that is also the one point where JAX and
Torch part company, by the same 1.6e-9. Everywhere else the three numbers agree to 1e-10
relative or better, and Torch is bit-identical to JAX to 1e-16.
:func:`test_the_budget_rejects_a_gradient_a_part_in_ten_thousand_wrong` is the check that the
budget is a measurement and not a formality.

**The static doors are not leaves**, and the other half of this gate is that they still say so:
``PML(cells=)``, ``PML(wavelength=)`` and a ``Slab`` edge fix the grid, the psi shapes and the
skeleton, so a tracer or a ``requires_grad`` tensor in one of them is refused by the field's
name. Torch is the case that makes this necessary rather than tidy: ``float()`` on a tensor
that requires a gradient *succeeds*, the run completes, and the gradient is simply missing.
"""
import math

import numpy as np
import pytest

import wavebit.fd as fd
from wavebit.fd._run import execute
from wavebit.fd._runtime import runtime

from fd_helpers import torch_forward_paths

#: The differentiated scene: a 1 x 1 x 44 line, a slab in cells 20..26, PML on both z faces.
#: Small on purpose -- a gradient gate is about the derivative, not about the physics, and
#: every point below costs one JAX trace, one Torch backward and two NumPy runs. The 44 is
#: what keeps the slab half a wavelength clear of the layers at every point below, so no
#: scene here carries a proximity warning it would then have to silence.
DX = 0.05
CELLS = 44
STEPS = 60
FRONT, BACK = 20 * DX, 26 * DX

#: The finite-difference step, relative to the point; see the module docstring for the choice.
H_RELATIVE = 1e-5

#: ``|g_ad - g_fd| <= RTOL |g_fd| + ATOL``; both measured, see the module docstring.
RTOL, ATOL = 1e-6, 1e-8

#: The three points per parameter. Spread over a factor of ~3 so that a gradient that happens
#: to be right at one operating point is not mistaken for a gradient.
POINTS = {'eps': (1.44, 2.25, 4.0), 'sigma': (0.2, 0.5, 1.0), 'strength': (20.0, 60.0, 120.0)}


def _pulse(x, y, z):
    return np.exp(-((z - 14 * DX) / 0.15) ** 2)


def scene_with(material):
    """The one scene, with ``material`` in the slab."""
    return fd.Scene(
        domain=fd.Domain(size=(DX, DX, CELLS * DX), center=(0.0, 0.0, CELLS * DX / 2)),
        boundaries=fd.Boundaries(x=fd.Periodic(), y=fd.Periodic(),
                                 z=fd.PML(cells=8, wavelength=1.0, allow_thin=True)),
        resolution=fd.Resolution(cells_per_wavelength=20.0, wavelength=1.0),
        run=fd.Run(steps=STEPS, courant=0.5),
        initial=fd.InitialField(E=(None, _pulse, None)),
        structures=(fd.Structure(fd.Slab(z=(FRONT, BACK)), material),))


def material_of(parameter, value):
    """The material of each family, with ``value`` in the place the gradient is taken at."""
    if parameter == 'eps':
        return fd.Material(eps=value)
    if parameter == 'sigma':
        return fd.Conductor(eps=2.25, sigma=value)
    return fd.Dispersive(eps_inf=1.5,
                         poles=(fd.Lorentz(omega_0=12.0, gamma=1.0, strength=value),))


_PLANS = {}


def plan_of(parameter):
    """The plan of ``parameter``'s family, built once from a **concrete** scene.

    Built once for two reasons. It is what a caller differentiating a sweep does (the plan is
    static, the material is a leaf, and that separation is the whole point of the pytree rule),
    and it keeps the finite difference and the autodiff on exactly the same static content --
    a gradient compared against a difference of two *different plans* would be measuring the
    planner.
    """
    if parameter not in _PLANS:
        _PLANS[parameter] = fd.plan(scene_with(material_of(parameter, POINTS[parameter][1])))
    return _PLANS[parameter]


def loss(parameter, value, backend):
    """``sum(|E|^2) + sum(|H|^2)`` of the final state: one number, differentiable in ``value``."""
    rt = runtime(backend)
    plan = plan_of(parameter)
    state = execute(plan, fd.lower(scene_with(material_of(parameter, value)), plan, rt), rt)
    return (abs(state['E']) ** 2).sum() + (abs(state['H']) ** 2).sum()


def central_difference(parameter, point, relative=H_RELATIVE):
    """``(f(x+h) - f(x-h)) / 2h`` on NumPy, with ``h`` relative to the point."""
    h = relative * max(1.0, abs(point))
    return (float(loss(parameter, point + h, 'numpy'))
            - float(loss(parameter, point - h, 'numpy'))) / (2.0 * h)


def jax_gradient(parameter, point):
    jax = pytest.importorskip('jax')
    return float(jax.grad(lambda v: loss(parameter, v, 'jax'))(point))


def torch_gradient(parameter, point):
    torch = pytest.importorskip('torch')
    value = torch.tensor(point, dtype=torch.float64, requires_grad=True)
    return float(torch.autograd.grad(loss(parameter, value, 'torch'), value)[0])


def close(got, want):
    """The budget of the module docstring, as a predicate, so a mutant can be run through it."""
    return abs(got - want) <= RTOL * abs(want) + ATOL


# ---------------------------------------------------------------------------------------------
# The three parameters, both autodiff backends, three points each
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('parameter', sorted(POINTS))
@pytest.mark.parametrize('index', (0, 1, 2))
def test_the_gradient_is_the_central_difference(parameter, index):
    """``jax.grad`` and ``torch.autograd`` both reproduce ``d(final energy)/d(parameter)``.

    One test per point rather than a loop, so a failure names the parameter and the operating
    point rather than the first of nine. The gradient is also asserted to be nonzero: a chain
    that quietly detached somewhere would give 0.0, which agrees with nothing but would sail
    through a relative tolerance if the reference were zero too.
    """
    point = POINTS[parameter][index]
    reference = central_difference(parameter, point)
    assert abs(reference) > 1e-4, reference
    for name, gradient in (('jax', jax_gradient(parameter, point)),
                           ('torch', torch_gradient(parameter, point))):
        assert gradient != 0.0, name
        assert close(gradient, reference), (parameter, point, name, gradient, reference)


@pytest.mark.parametrize('parameter', sorted(POINTS))
def test_jax_and_torch_agree_on_the_gradient(parameter):
    """The two autodiffs are also held to each other, which the finite difference cannot do.

    A shared error -- a coefficient that is differentiated the same wrong way in both, because
    the wrongness is in ``wavebit.fd`` and not in the autodiff -- shows against the central
    difference above and not here. A backend-specific one (Torch's out-of-place CPML fallback
    is exactly such a place) shows here first, and against a reference with no round-off floor
    of its own.
    """
    for point in POINTS[parameter]:
        a, b = jax_gradient(parameter, point), torch_gradient(parameter, point)
        assert close(a, b), (parameter, point, a, b)


def test_the_step_sits_in_the_plateau():
    """The evidence for ``h = 1e-5``: the difference is flat across two decades around it.

    Truncation above, cancellation below, and ``jax.grad`` inside the flat part. Without this
    the agreement above would be a coincidence of one step size, which is the usual way a
    finite-difference check passes a wrong derivative.
    """
    point = POINTS['eps'][1]
    gradient = jax_gradient('eps', point)
    estimates = {h: central_difference('eps', point, h) for h in (1e-4, 1e-5, 1e-6)}
    spread = max(estimates.values()) - min(estimates.values())
    assert spread < 1e-6 * abs(gradient), estimates
    for h, value in estimates.items():
        assert close(gradient, value), (h, value, gradient)
    coarse = central_difference('eps', point, 1e-2)
    assert not close(gradient, coarse), coarse          # truncation, above the plateau


def test_the_budget_rejects_a_gradient_a_part_in_ten_thousand_wrong():
    """The budget is a measurement, not a formality: it fails a derivative that is 1.0001x.

    A tolerance nothing can fail is not a tolerance. The scale is the one a real defect would
    produce: dropping the ``dx`` from the pole term's contribution, or differentiating through
    ``eps`` but not through the ``1 + dt rate/2`` denominator, moves a gradient by percents,
    four decades above this.
    """
    point = POINTS['sigma'][1]
    reference = central_difference('sigma', point)
    assert close(reference, reference)
    assert not close(reference * 1.0001, reference)
    assert not close(reference + 1e-6, reference)


def test_the_gradient_reaches_the_pole_memory_and_not_only_the_coefficients():
    """A pole strength enters only through state the scan carries; a run of one step cannot see it.

    The ADE memory starts at rest and ``advance`` runs at the *end* of a step, so the first
    step's polarisation increment is exactly zero and the strength has no effect at all on a
    one-step run. That is what makes this parameter the interesting one for autodiff: its
    gradient is zero unless the cotangent is carried back through the scan, and it grows with
    the number of steps. Both are asserted, because a gradient that is merely nonzero could
    still be leaking in through a coefficient that has no business depending on the strength.
    """
    jax = pytest.importorskip('jax')
    point = POINTS['strength'][1]

    one = fd.plan(scene_with(material_of('strength', point)).replace(
        run=fd.Run(steps=1, courant=0.5)))

    def one_step(value):
        scene = scene_with(material_of('strength', value)).replace(
            run=fd.Run(steps=1, courant=0.5))
        rt = runtime('jax')
        state = execute(one, fd.lower(scene, one, rt), rt)
        return (abs(state['E']) ** 2).sum() + (abs(state['H']) ** 2).sum()

    assert float(jax.grad(one_step)(point)) == 0.0
    assert abs(jax_gradient('strength', point)) > 1e-4


def test_the_public_run_door_carries_a_traced_material_end_to_end():
    """``fd.run(scene)`` itself, not the plan-once path the rest of this file uses.

    The tests above build the plan from a concrete scene and lower by hand, which is what a
    sweep does. A user who writes ``fd.run(scene_with(Material(eps=tensor)))`` goes through
    ``plan(scene)`` as well -- every stage-1 check runs against a value it cannot read -- and
    the answer, and its gradient, must be the same. That a check *did* give up is the other
    half of the contract and is asserted rather than silenced: the mesh check says in a
    warning that it could not read this permittivity, which is what ``wavebit/fd/_lower/checks.py``
    promises a traced value costs.
    """
    torch = pytest.importorskip('torch')
    point = POINTS['eps'][1]
    value = torch.tensor(point, dtype=torch.float64, requires_grad=True)
    with pytest.warns(fd.CoarseMeshWarning, match='traced value'):
        result = fd.run(scene_with(fd.Material(eps=value)), backend='torch')
    total = (abs(result.final.E) ** 2).sum() + (abs(result.final.H) ** 2).sum()
    gradient = float(torch.autograd.grad(total, value)[0])
    assert close(gradient, central_difference('eps', point))
    assert math.isfinite(float(total.detach()))


# ---------------------------------------------------------------------------------------------
# The other half: a traced value in a static door, refused by name
# ---------------------------------------------------------------------------------------------

def _traced_doors():
    """``(id, build, pattern)`` for each static door, with the concrete spelling that is taken."""
    return (
        pytest.param(lambda v: fd.PML(cells=v, wavelength=1.0),
                     r'PML\(cells=\.\.\.\) is static', 12, id='pml-cells'),
        pytest.param(lambda v: fd.PML(cells=12, wavelength=v),
                     r'PML\(wavelength=\.\.\.\) is static', 0.55, id='pml-wavelength'),
        pytest.param(lambda v: fd.Slab(z=(v, 2.0)),
                     r'Slab\(z=\.\.\.\) is static', 1.0, id='slab-edge'),
    )


@pytest.mark.parametrize('build,pattern,concrete', _traced_doors())
def test_a_jax_tracer_in_a_static_door_is_refused_by_name(build, pattern, concrete):
    """Inside ``jax.grad`` the value is a tracer, and the door says which field refused it."""
    jax = pytest.importorskip('jax')

    def through(v):
        build(v)
        return v * 2.0

    with pytest.raises(ValueError, match=pattern) as raised:
        jax.grad(through)(float(concrete))
    assert 'not a leaf' in str(raised.value)
    assert 'InitialField.amplitude' in str(raised.value)      # what the leaves actually are
    assert build(concrete) is not None                        # and the concrete spelling is taken


@pytest.mark.parametrize('build,pattern,concrete', _traced_doors())
def test_a_torch_leaf_in_a_static_door_is_refused_by_name(build, pattern, concrete):
    """The case the refusal exists for: ``float()`` on a live tensor succeeds and loses the graph.

    Without the check the door would take the tensor, read its value, and return a scene whose
    run completes and whose gradient with respect to that tensor is silently absent -- the
    failure mode a ``ConcretizationTypeError`` saves JAX from and nothing saves Torch from.
    """
    torch = pytest.importorskip('torch')
    live = torch.tensor(float(concrete), dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValueError, match=pattern):
        build(live)
    # ``float(live)`` would have *worked* here -- Torch reads the value and warns once per
    # process, which is not a refusal -- so the door has to be the thing that says no.
    assert float(live.detach()) == float(concrete)
    assert build(concrete) is not None


@pytest.mark.parametrize('build,pattern,concrete', _traced_doors())
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


def test_a_material_value_is_a_leaf_and_passes_the_same_two_doors():
    """The refusals above are a decision about *which* numbers are static, not about tracers.

    The same tracer and the same live tensor go through ``Material``, ``Conductor`` and a pole
    without a word, which is what the gradients at the top of this file then flow through.
    """
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')
    seen = []

    def through(v):
        seen.append((fd.Material(eps=v), fd.Conductor(eps=2.25, sigma=v),
                     fd.Lorentz(omega_0=12.0, gamma=1.0, strength=v)))
        return v * 2.0

    jax.grad(through)(2.25)
    assert len(seen) == 1
    live = torch.tensor(2.25, dtype=torch.float64, requires_grad=True)
    assert fd.Material(eps=live).eps[0].requires_grad
    assert fd.Conductor(eps=2.25, sigma=live).sigma[1].requires_grad
    assert fd.Lorentz(omega_0=12.0, gamma=1.0, strength=live).strength.requires_grad
