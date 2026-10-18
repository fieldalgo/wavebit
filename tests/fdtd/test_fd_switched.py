"""A temporal boundary: a medium whose permittivity jumps once, against its closed form.

**The closed form.** In a uniform, lossless, non-magnetic medium whose permittivity jumps from
``eps1`` to ``eps2`` at ``t0``, ``D = eps E`` and ``B`` are continuous across the jump and the
wavenumber is kept (Morgenthaler, IRE Trans. Microwave Theory Tech. 6, 167 (1958); Xiao, Maywar
and Agrawal, Opt. Lett. 39, 574 (2014)). With ``n = sqrt(eps)`` and ``Z0 H = n E`` for a
forward wave, matching both gives the forward and backward field amplitudes

    E_f / E_0 = n1 (n1 + n2) / (2 n2^2),     E_b / E_0 = n1 (n1 - n2) / (2 n2^2)

and the frequency becomes ``w1 n1 / n2``. Because ``k`` is kept, each wave after the jump has
the incident wave's spatial profile, so an amplitude is an L2 norm over the wave's window.

**The mutant.** Writing the jump as ``eps(t) dE/dt = curl H`` keeps ``E`` continuous instead
of ``D``. It is the right equation away from the jump and the wrong one at it: at ``eps 1 -> 4``
both amplitudes come out exactly twice the closed form, at every resolution. Zeroing the
``dx/dt`` leaf is exactly that mutant (``src/wavebit/fdtd/_kernel/switched.py``).
"""
import math
import warnings

import numpy as np
import pytest

import wavebit.fdtd._legacy as fd
from wavebit import Units, fdtd
from wavebit.errors import UnsupportedFeatureError, ValidationError
from wavebit.materials.time_domain import Switched
from wavebit.units import C0
from tests.fixtures.fdtd.fd_helpers import Packet, run_mutated, to_numpy

COURANT = 0.5


def closed_form(eps1, eps2):
    n1, n2 = math.sqrt(eps1), math.sqrt(eps2)
    return n1 * (n1 + n2) / (2 * n2 ** 2), n1 * (n1 - n2) / (2 * n2 ** 2)


def packet_scene(cells, eps2, length=24.0, z0=6.0, t_switch=1.0, travel=5.0, material=None):
    """A forward packet in vacuum on a periodic line, switched to ``eps2`` while inside it."""
    dx = length / cells
    dt = COURANT * dx / math.sqrt(3)
    packet = Packet(1.0, 0.0, dx, dt, z0=z0, real=True)
    steps = int(round((t_switch + travel * math.sqrt(eps2)) / dt))
    if material is None:
        material = Switched(eps=1.0, after=eps2, time=t_switch)
    return fd.Scene(
        domain=fd.Domain(size=(dx, dx, length), center=(0.0, 0.0, length / 2)),
        boundaries=fd.Boundaries(x=fd.Periodic(), y=fd.Periodic(), z=fd.Periodic()),
        resolution=fd.Resolution(dx=dx), run=fd.Run(steps=steps, courant=COURANT),
        initial=packet.initial(),
        structures=(fd.Structure(fd.Slab(z=(0.0, length)), material),))


def signed_amplitudes(cells, eps2, mutate=None, backend='numpy'):
    """``(E_f, E_b) / E_0`` read from the final field, both signed by the incident profile."""
    length, z0, t_switch, travel = 24.0, 6.0, 1.0, 5.0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        start = fd.run(_initial_only(cells, eps2))
        scene = packet_scene(cells, eps2)
        final = run_mutated(scene, backend=backend, mutate_params=mutate)
    E0 = to_numpy(start.final.E)[1, :, 0, 0].real
    E = to_numpy(final.final.E)[1, :, 0, 0].real
    dx = length / cells
    z = (np.arange(cells) + 0.5) * dx

    def window(centre):
        return np.abs(((z - centre) + length / 2) % length - length / 2) < 4.0

    reference = np.sqrt(np.sum(E0[window(z0)] ** 2))

    def amplitude(centre):
        part = E[window(centre)]
        shifted = np.roll(E0, int(round((centre - z0) / dx)))[window(centre)]
        return np.sign(np.sum(part * shifted)) * np.sqrt(np.sum(part ** 2)) / reference

    centre = z0 + t_switch
    return amplitude(centre + travel), amplitude(centre - travel)


def _initial_only(cells, eps2):
    """The packet after one vacuum step: the incident profile the amplitudes are read against."""
    scene = packet_scene(cells, eps2, material=fd.Material(eps=1.0))
    return fd.Scene(domain=scene.domain, boundaries=scene.boundaries, resolution=scene.resolution,
                    run=fd.Run(steps=1, courant=COURANT), initial=scene.initial,
                    structures=scene.structures)


def test_a_temporal_boundary_splits_a_packet_by_the_closed_form_to_second_order():
    """Forward and backward amplitudes at ``eps 1 -> 4``, three resolutions, order 2.

    Measured: errors 4.9e-5, 1.2e-5, 3.1e-6 at 960, 1920 and 3840 cells -- order 2.00 at both
    refinements. The gate asks for a 1e-4 error at the finest and an order above 1.8.
    """
    forward, backward = closed_form(1.0, 4.0)
    errors = []
    for cells in (960, 1920, 3840):
        f, b = signed_amplitudes(cells, 4.0)
        errors.append(abs(f - forward) + abs(b - backward))
    assert errors[-1] < 1e-4
    orders = [math.log2(errors[i] / errors[i + 1]) for i in range(2)]
    assert min(orders) > 1.8, (errors, orders)


def test_the_e_continuous_mutant_is_twice_the_closed_form_at_every_resolution():
    """``eps dE/dt`` keeps E continuous: both waves come out at twice their closed form."""
    forward, backward = closed_form(1.0, 4.0)

    def e_continuous(params, scene, plan, rt):
        params = dict(params)
        params['switched'] = {name: dict(v, dx_over_dt=v['dx_over_dt'] * 0.0)
                              for name, v in params['switched'].items()}
        return params

    for cells in (960, 1920):
        f, b = signed_amplitudes(cells, 4.0, mutate=e_continuous)
        assert abs(f / forward - 2.0) < 1e-3 and abs(b / backward + 2.0) < 1e-3, (f, b)


# ---------------------------------------------------------------------------------------------
# The public declaration
# ---------------------------------------------------------------------------------------------

def standing_problem(material, spacing):
    return fdtd.Problem(units=Units(length='um'), scene=fdtd.Scene(
        domain=fdtd.Domain(size=(spacing, spacing, 1.)),
        boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.Periodic()),
        structures=(fdtd.Structure('m', fdtd.Slab(z=(-.5, .5)), material),)),
        initial=fdtd.InitialField(E=(None, lambda x, y, z: np.cos(2 * np.pi * z), None)),
        outputs=fdtd.Outputs(monitors=(fdtd.Record('p', at=(0., 0., 0.), component='Ey'),)))


def standing_amplitude(spacing, eps1=1.0, eps2=4.0, t0=0.3, backend='numpy'):
    """``max |Ey|`` after the jump at an antinode of ``cos(2 pi z)``, and its closed form.

    Before the jump ``E = cos(kz) cos(w1 t)`` and ``Z0 H = n1 sin(kz) sin(w1 t)``. Matching
    ``D`` and ``B`` at ``t0`` gives ``E = cos(kz) (a cos(w2 s) + b sin(w2 s))`` after it, with
    ``a = (eps1/eps2) cos(w1 t0)`` and ``|b| = (n1/n2) |sin(w1 t0)|``, so the amplitude is
    ``sqrt(a^2 + b^2)``; ``t0`` is the E level the jump lands on (``k dt``).
    """
    material = fdtd.SwitchedMaterial(epsilon_before=eps1, epsilon_after=eps2,
                                     switch_time=t0 * 1e-6 / C0)
    dt = COURANT * spacing / math.sqrt(3)
    steps = int(round(2.5 / dt))
    numerics = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=spacing),
                             duration=fdtd.Duration(steps=steps))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = fdtd.run(standing_problem(material, spacing), numerics=numerics, backend=backend)
    values = to_numpy(result['p'].values).real.ravel()
    k = int(math.ceil(t0 / dt - 1e-6))
    phase = 2 * math.pi * math.sqrt(1 / eps1) * k * dt
    expected = math.hypot((eps1 / eps2) * math.cos(phase),
                          math.sqrt(eps1 / eps2) * math.sin(phase))
    return float(np.max(np.abs(values[k + 5:]))), expected, values, k, dt


def test_the_public_material_matches_the_standing_wave_closed_form_to_second_order():
    """Measured 2.0e-4, 5.2e-5, 1.3e-5 at spacings 0.02, 0.01, 0.005: order 1.9-2.0."""
    errors = []
    for spacing in (0.02, 0.01, 0.005):
        got, expected, *_ = standing_amplitude(spacing)
        errors.append(abs(got - expected))
    assert errors[-1] < 5e-5
    assert min(math.log2(errors[i] / errors[i + 1]) for i in range(2)) > 1.8, errors


def test_a_drop_in_permittivity_matches_the_closed_form_too():
    """``eps 4 -> 1``: a faster medium after the jump, whose CFL bound the plan read."""
    errors = []
    for spacing in (0.01, 0.005):
        got, expected, *_ = standing_amplitude(spacing, eps1=4.0, eps2=1.0, t0=0.45)
        errors.append(abs(got - expected) / expected)
    assert errors[-1] < 1e-4 and errors[-1] < errors[0], errors


def test_the_frequency_after_the_jump_is_the_incident_one_times_n1_over_n2():
    """Zero crossings of the standing wave: period 1 in vacuum, 2 in ``eps 4``."""
    _, _, values, k, dt = standing_amplitude(0.005)
    after = values[k + 1:]
    crossings = np.nonzero(np.diff(np.sign(after)))[0]
    period = 2 * np.mean(np.diff(crossings)) * dt
    assert abs(period - 2.0) < 1.5e-3, period


def test_both_permittivities_differentiate_and_agree_with_finite_differences():
    jax = pytest.importorskip('jax')
    jnp = jax.numpy
    jax.config.update('jax_enable_x64', True)
    cells, length = 120, 6.0
    dx = length / cells
    dt = COURANT * dx / math.sqrt(3)
    packet = Packet(1.0, 0.0, dx, dt, z0=2.0, real=True)
    weight = np.zeros(cells)
    weight[30:60] = 1.0

    def scene(after, before):
        return fd.Scene(
            domain=fd.Domain(size=(dx, dx, length), center=(0.0, 0.0, length / 2)),
            boundaries=fd.Boundaries(x=fd.Periodic(), y=fd.Periodic(), z=fd.Periodic()),
            resolution=fd.Resolution(dx=dx), run=fd.Run(steps=200, courant=COURANT),
            initial=packet.initial(),
            structures=(fd.Structure(fd.Slab(z=(0.0, length)),
                                     Switched(eps=before, after=after, time=0.4)),))

    def objective(after, before, backend='numpy'):
        E = fd.run(scene(after, before), backend=backend).final.E[1, :, 0, 0]
        if backend == 'jax':
            return jnp.sum(jnp.asarray(weight) * E ** 2)
        return float(np.sum(weight * to_numpy(E) ** 2))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        grad_after, grad_before = jax.grad(lambda a, b: objective(a, b, 'jax'),
                                           argnums=(0, 1))(3.0, 1.2)
        h = 1e-5
        fd_after = (objective(3.0 + h, 1.2) - objective(3.0 - h, 1.2)) / (2 * h)
        fd_before = (objective(3.0, 1.2 + h) - objective(3.0, 1.2 - h)) / (2 * h)
    # Measured 7.70886787 against 7.70886788, and 56.2514176 against 56.2514176.
    assert abs(float(grad_after) / fd_after - 1) < 1e-6
    assert abs(float(grad_before) / fd_before - 1) < 1e-6

    torch = pytest.importorskip('torch')
    a = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        E = fd.run(scene(a, b), backend='torch').final.E[1, :, 0, 0]
    torch.sum(torch.as_tensor(weight) * E ** 2).backward()
    assert abs(a.grad.item() / fd_after - 1) < 1e-6
    assert abs(b.grad.item() / fd_before - 1) < 1e-6


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_the_backends_agree_with_numpy(backend):
    pytest.importorskip(backend)
    scene = packet_scene(240, 3.0, length=6.0, z0=2.0, t_switch=0.4, travel=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        reference = to_numpy(fd.run(scene).final.E)
        other = to_numpy(fd.run(scene, backend=backend).final.E)
    # Measured: JAX 1.4e-14 of a 5.7 peak, Torch bit-identical.
    assert np.max(np.abs(other - reference)) < 1e-12 * np.max(np.abs(reference))


def test_a_scene_with_no_switched_structure_plans_no_switched_term():
    from wavebit.fdtd._kernel.switched import SwitchedTerm
    plan = fd.plan(packet_scene(96, 4.0, material=fd.Material(eps=2.0)))
    assert not any(isinstance(t, SwitchedTerm) for t in plan.terms)


# ---------------------------------------------------------------------------------------------
# Refusals: each is a scene where the D correction would not be exact
# ---------------------------------------------------------------------------------------------

def thin_scene(structures, sources=(), boundaries=None, monitors=()):
    dx = 0.05
    return fd.Scene(
        domain=fd.Domain(size=(dx, dx, 3.0), center=(0.0, 0.0, 0.0)),
        boundaries=boundaries or fd.Boundaries(x=fd.Periodic(), y=fd.Periodic(),
                                               z=fd.PML(cells=12, wavelength=1.0)),
        resolution=fd.Resolution(dx=dx), run=fd.Run(steps=10, courant=COURANT),
        structures=tuple(structures), sources=tuple(sources), monitors=tuple(monitors))


SWITCHED = Switched(eps=1.0, after=4.0, time=0.2)


def plan_error(scene):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(ValueError) as caught:
            fd.plan(scene)
    return str(caught.value)


def test_a_smoothed_switched_shape_is_refused():
    box = fd.Box(size=(0.05, 0.05, 0.5), center=(0.0, 0.0, 0.0))
    assert 'smoothed shape' in plan_error(thin_scene([fd.Structure(box, SWITCHED)]))


def test_an_overlap_with_another_structure_is_refused():
    message = plan_error(thin_scene([fd.Structure(fd.Slab(z=(-0.5, 0.5)), SWITCHED),
                                     fd.Structure(fd.Slab(z=(0.0, 0.8)), fd.Material(eps=2.0))]))
    assert 'share cells' in message


def test_a_switched_region_in_an_absorbing_layer_is_refused():
    assert 'absorbing layer' in plan_error(
        thin_scene([fd.Structure(fd.Slab(z=(-1.5, 0.0)), SWITCHED)]))


def test_a_source_inside_the_switched_region_is_refused():
    """A dipole is not refused by any other door, so this one names it.

    (A plane wave inside any structure is already refused by the sheet's own check.)
    """
    source = fd.Dipole('d', center=(0.0, 0.0, 0.1), component='Ey',
                       band=fd.Band(wavelength=1.0, df=0.2))
    scene = thin_scene([fd.Structure(fd.Slab(z=(-0.5, 0.5)), SWITCHED)], sources=[source])
    scene = fd.Scene(domain=fd.Domain(size=(0.4, 0.4, 3.0)), boundaries=scene.boundaries,
                     resolution=scene.resolution, run=scene.run, structures=scene.structures,
                     sources=scene.sources)
    assert 'sits inside' in plan_error(scene)


def test_a_jump_before_the_second_step_is_refused():
    early = Switched(eps=1.0, after=4.0, time=1e-4)
    assert 'E level' in plan_error(thin_scene([fd.Structure(fd.Slab(z=(-0.5, 0.5)), early)]))


def test_a_switched_background_is_refused():
    scene = fd.Scene(
        domain=fd.Domain(size=(0.05, 0.05, 1.0)),
        boundaries=fd.Boundaries(x=fd.Periodic(), y=fd.Periodic(), z=fd.Periodic()),
        resolution=fd.Resolution(dx=0.05), run=fd.Run(steps=4, courant=COURANT),
        background=SWITCHED)
    assert 'background' in plan_error(scene)


def test_the_public_declaration_validates_its_inputs():
    with pytest.raises(ValidationError):
        fdtd.SwitchedMaterial(epsilon_before=1.0, epsilon_after=-2.0, switch_time=1e-15)
    with pytest.raises(ValidationError):
        fdtd.SwitchedMaterial(epsilon_before=1.0, epsilon_after=2.0, switch_time=0.0)
    material = fdtd.SwitchedMaterial(epsilon_before=1.0, epsilon_after=2.0, switch_time=1e-15)
    problem = fdtd.Problem(units=Units(length='um'), scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.05, .05, 1.)), background=material,
        boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.Periodic())))
    report = fdtd.check(problem, numerics=fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.05),
                                                        duration=fdtd.Duration(steps=4)))
    assert [issue.code for issue in report.issues] == ['switched_background']
    with pytest.raises(UnsupportedFeatureError):
        fdtd.run(problem, numerics=fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.05),
                                                 duration=fdtd.Duration(steps=4)))
