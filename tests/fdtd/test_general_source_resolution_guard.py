"""Live pole-free source bands retain the eager spatial-resolution floor."""
import numpy as np
import pytest

from tests.fdtd import _pulse_yee_reference as pulse
from wavebit import fdtd
from wavebit.errors import ValidationError


@pytest.mark.parametrize('trajectory', ['wavelength', 'bandwidth'])
@pytest.mark.parametrize('device,block', [('cpu', None), ('cuda:0', None), ('cuda:0', 7)],
                         ids=['cpu_plain', 'gpu_plain', 'gpu_k7'])
def test_same_band_leaves_undersampled_then_recover(device, block, trajectory):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(1)
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('actual CUDA device required')

    def problem(wavelength, bandwidth):
        return pulse.fixture((wavelength, bandwidth, .4, .12), steps=17, block=block)

    # The carrier stays positive: this refusal belongs to spatial resolution.
    seed = (1., 1.) if trajectory == 'wavelength' else (.5, 1.)
    bad = (.4, 1.) if trajectory == 'wavelength' else (.5, 2.)
    changed = 0 if trajectory == 'wavelength' else 1
    frequency = 1./(2*bad[0])
    cutoff_width = pulse.KAPPA*.08*bad[1]
    shortest = 1./(frequency+cutoff_width)
    cells = shortest/(.05*np.sqrt(pulse.EPSILON*pulse.MU))
    assert frequency > cutoff_width and 7. < cells < 8.
    with pytest.raises(ValidationError, match='cells per wavelength'):
        twin, numerics = problem(*bad)
        fdtd.prepare(twin, numerics=numerics, backend='torch', device=device)

    initial, numerics = problem(*seed)
    prepared = fdtd.prepare(initial, numerics=numerics, backend='torch', device=device)
    leaves = (torch.tensor(seed[0], dtype=torch.float64, requires_grad=True),
              torch.tensor(seed[1], dtype=torch.float64, device=device, requires_grad=True))
    identities = tuple(id(leaf) for leaf in leaves)
    for leaf in leaves:
        leaf.grad = torch.full_like(leaf, .125)
    gradients = tuple(leaf.grad for leaf in leaves)
    grad_values = tuple(leaf.grad.clone() for leaf in leaves)

    def run():
        versions = tuple(leaf._version for leaf in leaves)
        values = tuple(leaf.detach().clone() for leaf in leaves)
        grad_versions = tuple(leaf.grad._version for leaf in leaves)
        try:
            current, _ = problem(*leaves)
            assert current.sources[0].center == initial.sources[0].center
            assert current.sources[0].delay_seconds == initial.sources[0].delay_seconds
            return prepared.run(current)
        finally:
            assert tuple(id(leaf) for leaf in leaves) == identities
            assert tuple(leaf._version for leaf in leaves) == versions
            for i, leaf in enumerate(leaves):
                assert leaf.is_leaf and leaf.requires_grad and leaf.dtype == torch.float64
                assert str(leaf.device) == ('cpu' if i == 0 else device)
                assert leaf.grad is gradients[i] and leaf.grad._version == grad_versions[i]
                torch.testing.assert_close(leaf, values[i], rtol=0, atol=0)
                torch.testing.assert_close(leaf.grad, grad_values[i], rtol=0, atol=0)

    def outputs(result):
        return (*result.final_fields.values.values(),
                *(result[name].values for name in pulse.LABELS),
                *(result.sources['d'][key] for key in ('drive', 'field', 'delivered_power')))

    first = run()
    assert bool(first.validity.checks['runtime_stability'])
    assert bool((first.final_fields.values['E'] != 0).any())
    held = tuple(t.detach().clone() for t in outputs(first))
    with torch.no_grad():
        leaves[changed].fill_(bad[changed])
    refused, invalid_has_nan, invalid_stable = False, False, None
    try:
        invalid = run()
    except ValidationError as exc:
        assert 'cells per wavelength' in str(exc)
        refused = True
    else:
        invalid_has_nan = bool(torch.isnan(invalid.final_fields.values['E']).any())
        invalid_stable = bool(invalid.validity.checks['runtime_stability'])
    print(dict(cells_per_wavelength=cells, structured_refusal=refused,
               output_has_nan=invalid_has_nan, runtime_stability=invalid_stable), flush=True)
    with torch.no_grad():
        leaves[changed].fill_(seed[changed])
    recovered = run()
    assert bool(recovered.validity.checks['runtime_stability'])
    for actual, original, expected in zip(outputs(recovered), outputs(first), held):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(original, expected, rtol=0, atol=0)
    assert refused or (invalid_has_nan and invalid_stable is False), (
        'a live band below the eager spatial-resolution floor returned valid fields',
        invalid_has_nan, invalid_stable)


def guard_scene(wavelength, df=.125, *, background=None, structures=(), siblings=(),
                monitors=()):
    from types import SimpleNamespace
    from wavebit.fdtd import _legacy as fd
    bands = ((wavelength, df),)+tuple(siblings)
    return SimpleNamespace(background=fd.Material(eps=1.) if background is None else background,
        structures=tuple(SimpleNamespace(material=m, shape=None) for m in structures),
        sources=tuple(SimpleNamespace(band=SimpleNamespace(wavelength=w, df=b, sigmas=2.,
            shortest_wavelength=None if hasattr(w, 'dtype') or hasattr(b, 'dtype')
            else 1./(1./w+2*b))) for w, b in bands), monitors=monitors)


def guard_plan(dx=.125):
    from types import SimpleNamespace
    return SimpleNamespace(dx=dx, grid=None, pml_wavelengths=())


def evaluate_guard(backend, fn, point):
    from wavebit.fdtd._runtime import runtime
    rt = runtime(backend)
    if backend == 'jax':
        jax = pytest.importorskip('jax')
        import jax.numpy as jnp
        return np.asarray(jax.jit(lambda x: fn(x, rt))(jnp.asarray(point, dtype=jnp.float64)))
    torch = pytest.importorskip('torch')
    torch.set_num_threads(1)
    return fn(torch.tensor(point, dtype=torch.float64, requires_grad=True), rt).detach().numpy()


@pytest.mark.parametrize('backend', ['torch', 'jax'])
@pytest.mark.parametrize('dx', [np.nextafter(.125, 0.), .125, np.nextafter(.125, np.inf)],
                         ids=['above-eight', 'exactly-eight', 'below-eight'])
def test_live_band_inclusive_floor_and_nearest_dx(backend, dx):
    from wavebit.fdtd._prepare.shape_ranges import resolution_guard
    # f0=.75, k*df=.25, shortest=1, index=1. No tolerance band at eight.
    value = evaluate_guard(backend, lambda x, rt: resolution_guard(
        guard_scene(x), guard_plan(dx), rt, None), 4/3)
    assert bool(np.isfinite(value)) == (1./dx >= 8.)


@pytest.mark.parametrize('backend', ['torch', 'jax'])
@pytest.mark.parametrize('case', ['background', 'static-body', 'concrete-sibling', 'live-material'])
def test_live_band_checks_every_medium_and_concrete_source(backend, case):
    from wavebit.fdtd import _legacy as fd
    from wavebit.fdtd._prepare.shape_ranges import resolution_guard

    def run(x, rt):
        kw = {}
        if case == 'background':
            kw['background'] = fd.Material(eps=4.)
        elif case == 'static-body':
            kw['structures'] = (fd.Material(eps=(1., 4., 1.), mu=(1., 1., 4.)),)
        elif case == 'concrete-sibling':
            kw['siblings'] = ((.5, .125),)
        else:
            kw['background'] = fd.Material(eps=x*3.)
        return resolution_guard(guard_scene(x, **kw), guard_plan(), rt, None)

    assert np.isnan(evaluate_guard(backend, run, 4/3))


@pytest.mark.parametrize('backend', ['torch', 'jax'])
def test_live_band_retains_static_diffraction_and_prior_failure(backend):
    from wavebit.fdtd import _legacy as fd
    from wavebit.fdtd._prepare.shape_ranges import resolution_guard
    monitor = fd.Diffraction('T', z=0., side='transmit', wavelengths=(.5,), orders=0)
    for monitors, previous, accepted in [((), 2., True), ((monitor,), 2., False),
                                         ((), float('nan'), False)]:
        value = evaluate_guard(backend, lambda x, rt: resolution_guard(
            guard_scene(x, monitors=monitors), guard_plan(), rt,
            rt.array(previous, 'float64')), 4/3)
        assert bool(np.isfinite(value)) is accepted
        if accepted:
            assert value == 2.


@pytest.mark.parametrize('backend', ['torch', 'jax'])
def test_any_declared_pole_preserves_entire_old_path(backend):
    from wavebit.fdtd import _legacy as fd
    from wavebit.fdtd._prepare.shape_ranges import resolution_guard
    # Even a concretely inactive pole leaves the dynamic-band gap unchanged.
    pole = fd.Dispersive(eps_inf=1., poles=(fd.Lorentz(omega_0=2., strength=0.),))
    monitor = fd.Diffraction('T', z=0., side='transmit', wavelengths=(.5,), orders=0)

    def run(x, rt, monitors):
        s = guard_scene(x, background=pole, structures=(fd.Material(eps=x*3.),),
                        monitors=monitors)
        return resolution_guard(s, guard_plan(), rt, rt.array(.75, 'float64'))

    # A wholesale early return would incorrectly omit the old live-material
    # Diffraction check; applying the new band guard would reject the first case.
    assert evaluate_guard(backend, lambda x, rt: run(x, rt, ()), 4/3) == .75
    assert np.isnan(evaluate_guard(backend, lambda x, rt: run(x, rt, (monitor,)), 4/3))


@pytest.mark.parametrize('backend', ['torch', 'jax'])
def test_dynamic_conductivity_reading_keeps_descriptor_scope(backend, monkeypatch):
    from wavebit.fdtd import _legacy as fd
    from wavebit.fdtd._prepare import conductive_film
    from wavebit.fdtd._prepare.checks import _index_at
    from wavebit.fdtd._prepare.shape_ranges import resolution_guard
    material = fd.Conductor(eps=1., sigma=1000.)
    for enabled in (False, True):
        monkeypatch.setattr(conductive_film, 'descriptor',
                            lambda scene, grid: object() if enabled else None)
        n, skin = _index_at(material, 1., conductive=enabled)
        assert skin is None
        expected = 1./(n*.125) >= 8.
        value = evaluate_guard(backend, lambda x, rt: resolution_guard(
            guard_scene(x, background=material), guard_plan(), rt, None), 4/3)
        assert bool(np.isfinite(value)) == expected


def test_concrete_band_keeps_previous_guard_identity():
    torch = pytest.importorskip('torch')
    from wavebit.fdtd._prepare.shape_ranges import resolution_guard
    from wavebit.fdtd._runtime import runtime
    rt = runtime('torch')
    previous = torch.tensor(.75, dtype=torch.float64)
    scene = guard_scene(4/3)
    assert resolution_guard(scene, guard_plan(), rt, previous) is previous
    assert resolution_guard(scene, guard_plan(), rt, None) is None


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'], ids=['cpu', 'gpu'])
def test_torch_live_resolution_no_host_read_and_constant_certificate(device, monkeypatch):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(1)
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('actual CUDA device required')
    from wavebit.fdtd._prepare.shape_ranges import resolution_guard
    from wavebit.fdtd._runtime import runtime
    rt = runtime('torch', device)
    plan = guard_plan(.0625)

    def certificate(point):
        return resolution_guard(guard_scene(point[0], point[1]), plan, rt, None)

    def objective(point):
        return point.square().sum()*certificate(point)

    def forbidden(*args, **kwargs):
        raise AssertionError('resolution certificate read a backend leaf on the host')

    point = torch.tensor([4/3, .125], dtype=torch.float64, device=device)
    direction = torch.tensor([.3, -.2], dtype=point.dtype, device=device)
    with monkeypatch.context() as patch:
        for name in ('__float__', '__bool__', 'item', 'numpy', 'tolist', 'cpu'):
            patch.setattr(torch.Tensor, name, forbidden)
        gradient = torch.func.jacfwd(objective)(point)
        hessian = torch.func.hessian(objective)(point)
        batch = torch.func.vmap(certificate)(torch.stack((point, point/4)))
        with torch.autograd.forward_ad.dual_level():
            dual = torch.autograd.forward_ad.make_dual(point, direction)
            guard, guard_tangent = torch.autograd.forward_ad.unpack_dual(certificate(dual))
        live = point.detach().clone().requires_grad_()
        first = torch.autograd.grad(objective(live), live, create_graph=True)[0]
        second = torch.stack([torch.autograd.grad(first[i], live, retain_graph=True)[0]
                              for i in range(2)])
    assert guard_tangent is None and guard.item() == 1.
    torch.testing.assert_close(gradient, 2*point, rtol=0, atol=0)
    torch.testing.assert_close(first, 2*point, rtol=0, atol=0)
    expected = 2*torch.eye(2, dtype=point.dtype, device=device)
    torch.testing.assert_close(hessian, expected, rtol=0, atol=0)
    torch.testing.assert_close(second, expected, rtol=0, atol=0)
    assert batch[0].item() == 1. and bool(torch.isnan(batch[1]))


def test_jax_live_resolution_jit_hessian_and_batch():
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from wavebit.fdtd._prepare.shape_ranges import resolution_guard
    from wavebit.fdtd._runtime import runtime
    rt = runtime('jax')

    def certificate(point):
        return resolution_guard(guard_scene(point[0], point[1]), guard_plan(.0625), rt, None)

    def objective(point):
        return jnp.sum(point**2)*certificate(point)

    point = jnp.asarray([4/3, .125])
    np.testing.assert_array_equal(jax.grad(jax.jit(objective))(point), 2*np.asarray(point))
    np.testing.assert_array_equal(jax.hessian(jax.jit(objective))(point), 2*np.eye(2))
    np.testing.assert_array_equal(jax.jacfwd(jax.jit(certificate))(point), np.zeros(2))
    batch = np.asarray(jax.jit(jax.vmap(certificate))(jnp.stack((point, point/4))))
    assert batch[0] == 1. and np.isnan(batch[1])
