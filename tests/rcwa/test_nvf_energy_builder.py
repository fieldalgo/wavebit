"""Energy closure: constitutive recovery, basis invariance and actual engine use."""

from wavebit.compat import legacy as _legacy_wavebit
import numpy as np
import pytest

from wavebit.compat.legacy import ConvMatrices


@pytest.fixture(params=['numpy', 'jax', 'torch'])
def arrays(request):
    if request.param == 'jax':
        jax = pytest.importorskip('jax')
        jax.config.update('jax_enable_x64', True)
        import jax.numpy as jnp
        return jnp.asarray, np.asarray
    if request.param == 'torch':
        torch = pytest.importorskip('torch')
        return torch.as_tensor, lambda a: a.detach().cpu().numpy()
    return np.asarray, np.asarray


@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_recovers_prescribed_nonhermitian_constitutive_tensor_and_preserves_inputs(arrays, batched, dtype):
    put, read = arrays
    rng = np.random.default_rng(430)
    shape = (2, 6, 6) if batched else (6, 6)
    q = (3*np.eye(6)+.2*(rng.normal(size=shape)+1j*rng.normal(size=shape))).astype(dtype)
    e = (2*np.eye(6)+.1*(rng.normal(size=shape)+1j*rng.normal(size=shape))).astype(dtype)
    g = q.conj().swapaxes(-1, -2)@e@q
    z = np.broadcast_to(np.diag([2.+.1j, 3.+.2j, 4.+.3j]), (*shape[:-2], 3, 3)).copy().astype(dtype)
    inputs = [put(a.copy()) for a in (q, g, z)]
    result = ConvMatrices.from_energy(*inputs)
    expected = (e[..., :3, :3], e[..., 3:, 3:], np.linalg.inv(z), -e[..., :3, 3:], -e[..., 3:, :3])
    tolerance = 2e-6 if dtype == np.complex64 else 3e-14
    for got, want in zip(result, expected):
        np.testing.assert_allclose(read(got), want, atol=tolerance, rtol=tolerance)
        assert read(got).dtype == dtype
    for got, want in zip(inputs, (q, g, z)):
        np.testing.assert_array_equal(read(got), want)
    assert np.max(abs(read(result.epxy)-read(result.epyx).conj().swapaxes(-1, -2))) > .01


def _basis_invariant_objective(parameters, xp):
    a, epsilon = parameters[0], parameters[1]
    q = xp.stack([xp.stack([2.+a+0j, .3j+0*a]),
                  xp.stack([-.4+0j*a, 1.+.2j*a])])
    physical = xp.diag(xp.stack([epsilon+0j, 2*epsilon+0j]))
    g = q.conj().T@physical@q
    z = xp.reshape(epsilon+0j, (1, 1))
    slots = ConvMatrices.from_energy(q, g, z)
    return (slots.epx+2*slots.epy+3*slots.epz_i).real.sum()


def test_jit_preserves_material_derivative_and_cancels_basis_derivative():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    value, derivative = jax.jit(jax.value_and_grad(lambda p: _basis_invariant_objective(p, jnp)))(
        jnp.array([.27, 3.]))
    assert float(value) == pytest.approx(16., abs=2e-13)
    np.testing.assert_allclose(derivative, [0., 5-3/9], atol=2e-13, rtol=0)


def test_torch_preserves_input_autograd_graph_and_material_derivative():
    torch = pytest.importorskip('torch')
    parameters = torch.tensor([.27, 3.], dtype=torch.float64, requires_grad=True)
    value = _basis_invariant_objective(parameters, torch)
    value.backward()
    assert parameters.requires_grad
    assert value.item() == pytest.approx(16., abs=2e-13)
    np.testing.assert_allclose(parameters.grad.numpy(), [0., 5-3/9], atol=2e-13, rtol=0)


@pytest.mark.parametrize('q,g,z', [
    (np.eye(3), np.eye(3), np.eye(1)),
    (np.eye(2), np.eye(4), np.eye(1)),
    (np.eye(2), np.eye(2), np.eye(2)),
    (np.zeros((0, 2, 2)), np.zeros((0, 2, 2)), np.zeros((0, 1, 1))),
    (np.empty((0, 0)), np.empty((0, 0)), np.empty((0, 0))),
    (np.ones((1, 2, 2)), np.ones((1, 2, 2)), np.eye(1)),
    (np.array([[np.inf, 0.], [0., 1.]]), np.eye(2), np.eye(1)),
])
def test_invalid_shapes_and_nonfinite_inputs_are_refused(q, g, z):
    with pytest.raises(ValueError):
        ConvMatrices.from_energy(q, g, z)


def test_integer_matrices_and_singular_solves_are_refused():
    with pytest.raises(TypeError, match='dtype'):
        ConvMatrices.from_energy(np.eye(2, dtype=int), np.eye(2, dtype=int), np.eye(1, dtype=int))
    with pytest.raises(np.linalg.LinAlgError):
        ConvMatrices.from_energy(np.zeros((2, 2)), np.eye(2), np.eye(1))
    with pytest.raises(np.linalg.LinAlgError):
        ConvMatrices.from_energy(np.eye(2), np.eye(2), np.zeros((1, 1)))


def test_invalid_jit_solve_poisoning_covers_every_slot():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    build = jax.jit(lambda scale: ConvMatrices.from_energy(
        jnp.eye(2), jnp.eye(2), jnp.reshape(scale, (1, 1))))
    for value in (0., np.nan):
        assert all(np.isnan(np.asarray(slot)).all() for slot in build(value))


def _uniform_energy(epsilon, order, layers=1):
    count = (2*order+1)**2
    q = np.kron(np.array([[1.3+.2j, .3j], [-.2, 1.1-.1j]]), np.eye(count))
    q = np.broadcast_to(q, (layers, 2*count, 2*count))
    g = epsilon*(q.conj().swapaxes(-1, -2)@q)
    z = np.broadcast_to(epsilon*np.eye(count, dtype=complex), (layers, count, count))
    return ConvMatrices.from_energy(q, g, z)


def _airy(epsilon, wavelength, thickness, theta, polarization):
    eps = np.array([1., epsilon, 1.44], dtype=complex)
    kz = np.sqrt(eps-np.sin(theta)**2)
    top, body, bottom = eps/kz if polarization == 'p' else kz
    phase = np.exp(2j*np.pi*thickness/wavelength*kz[1])
    transmission = 4*top*body*phase/((top+body)*(body+bottom)+(top-body)*(body-bottom)*phase**2)
    return float((bottom.real/top.real)*abs(transmission)**2)


@pytest.mark.parametrize('epsilon,wavelength,thickness', [(4., 830., 177.), (3.+.2j, 910., 243.)])
def test_real_engine_matches_independent_lossless_and_absorbing_slab(epsilon, wavelength, thickness):
    solver = _legacy_wavebit.rcwa(period=(700., 700.), wavelength=wavelength, theta=.23, phi=.37,
                          thickness=[thickness], top=1., bot=1.44, truncation_order=(1, 1))
    result = solver.solve_conv(*_uniform_energy(epsilon, 1))
    for polarization in ('p', 's'):
        actual = np.asarray(getattr(result, polarization+'_inc').T).sum()
        assert actual == pytest.approx(_airy(epsilon, wavelength, thickness, .23, polarization), abs=2e-11)


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_conforming_nvf_export_runs_in_the_real_engine_for_both_polarizations(backend):
    if backend == 'jax':
        pytest.importorskip('jax')
        _legacy_wavebit.enable_x64()
    elif backend == 'torch':
        pytest.importorskip('torch')
    from studies.rcwa.nvf_polynomial_body.nvf_polynomial_body import PolynomialBodyGapModel
    from studies.rcwa.nvf_scattering_doubling.nvf_scattering_doubling import doubling_powers
    model = PolynomialBodyGapModel(1, rows=2, body_degree=2, gap_degree=2)
    conv = model.conv_matrices(17., dict(seams=[31.], values=[4., 9.]))
    reference = doubling_powers(1, tuple(a[0] for a in conv), theta=.2, azimuth=.37)
    solver = _legacy_wavebit.rcwa(backend=backend, period=(700., 700.), wavelength=900.,
                          thickness=[300.], theta=.2, phi=.37, top=1., bot=2.25,
                          truncation_order=(1, 1))
    slots = ConvMatrices(*(solver.xp.asarray(a, dtype=solver.xp.complex128) for a in conv))
    result = solver._solve_conv(*slots)
    for index, polarization in enumerate(('p', 's')):
        answer = getattr(result, polarization+'_inc')
        for key, expected in zip(('T', 'R'), reference[:2]):
            got = getattr(answer, key)
            got = got.detach().cpu().numpy() if hasattr(got, 'detach') else np.asarray(got)
            assert got.sum() == pytest.approx(expected[index], abs=2e-10)


def test_splitting_a_conforming_layer_preserves_the_engine_scattering_result():
    from studies.rcwa.nvf_polynomial_body.nvf_polynomial_body import PolynomialBodyGapModel
    conv = PolynomialBodyGapModel(1, rows=2, gap_degree=2).conv_matrices(17.)
    common = dict(period=(700., 700.), wavelength=875., theta=.19, phi=.31,
                  top=1., bot=1.7, truncation_order=(1, 1))
    single = _legacy_wavebit.rcwa(thickness=[300.], **common).solve_conv(*conv)
    split = _legacy_wavebit.rcwa(thickness=[110., 190.], **common).solve_conv(
        *(np.concatenate([slot, slot], axis=0) for slot in conv))
    for polarization in ('p_inc', 's_inc'):
        for key in ('T', 'R'):
            np.testing.assert_allclose(getattr(getattr(split, polarization), key),
                                       getattr(getattr(single, polarization), key), atol=2e-11, rtol=0)
