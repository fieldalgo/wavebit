"""Bounded SnappedBox CUDA gates against independent indexed SI Yee tangents.

These fixed cell-snapped boxes establish discrete heterogeneous evolution and
material derivatives, not smooth geometry derivatives or continuum convergence.
"""
import numpy as np
import pytest

from wavebit import fdtd
from wavebit.units import Z0
from tests.fdtd._region_yee_reference import (
    DT, LABELS, POINTS, TOL, geometry, mask, numerics, objectives, patterns,
    problem, reference, weights,
)


def host(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def scaled(actual, expected):
    assert np.isfinite(actual).all()
    peak = np.max(abs(expected))
    assert peak > 1e-8
    return float(np.max(abs(actual-expected))/peak)


def verify(result, expected, steps):
    assert bool(result.validity.checks['runtime_stability'])
    errors = []
    for field in ('E', 'H'):
        errors.append(scaled(host(result.final_fields.values[field])*(Z0 if field == 'H' else 1),
                             expected[field]))
    for c, label in enumerate(LABELS):
        errors.append(scaled(host(result[label].values)*(Z0 if c >= 3 else 1), expected['records'][c]))
        times = (np.arange(steps)+(1. if c < 3 else .5))*DT
        np.testing.assert_allclose(host(result[label].times_seconds), times, rtol=2e-14, atol=0)
    np.testing.assert_allclose(result.provenance['dt_seconds'], DT, rtol=2e-14, atol=0)
    assert max(errors) < TOL, errors
    return max(errors)


def snapshot(result):
    return {key: host(value).copy() for key, value in (
        [(f, result.final_fields.values[f]) for f in ('E', 'H')]
        +[(label, result[label].values) for label in LABELS])}


def unchanged(result, saved):
    for key, value in saved.items():
        actual = result.final_fields.values[key] if key in ('E', 'H') else result[key].values
        np.testing.assert_array_equal(host(actual), value)


@pytest.mark.parametrize('rotation', range(3))
def test_reference_geometry_tangents_and_mutants(rotation, record_property):
    geo = geometry(rotation)
    assert tuple(geo['cells']) == tuple(np.roll((13, 10, 15), rotation))
    assert np.sum(mask(rotation)) == 168
    # Independently evaluate the documented nearest-boundary rule on declared faces.
    for sign, key in ((-1, 'lo'), (1, 'hi')):
        face = geo['region_center']+sign*geo['region_size']/2
        np.testing.assert_array_equal(np.floor((face-geo['low'])/.05e-6+.5), geo[key])
    exact = reference(rotation=rotation)
    values = objectives(exact)
    for value, gradient in values:
        assert np.all(abs(gradient*np.array([2.25, 1.3, 600., 1.2]))/value > 1e-6)
    secant_errors = []
    for p in range(4):
        for relative_step in (2e-4, 1e-4):
            step = POINTS[0][p]*relative_step
            plus, minus = list(POINTS[0]), list(POINTS[0])
            plus[p] += step
            minus[p] -= step
            op, om = objectives(reference(plus, rotation)), objectives(reference(minus, rotation))
            for i in range(2):
                secant = (op[i][0]-om[i][0])/(2*step)
                error = abs(secant/values[i][1][p]-1)
                assert error < 2e-6
                secant_errors.append(error)
    separations = {}
    for mutant in ('shift', 'floor', 'wrong_axis', 'wrong_wrap', 'omit_mu',
                   'omit_sigma_denominator', 'wrong_curl', 'omit_dq'):
        wrong = reference(rotation=rotation, mutant=mutant)
        if mutant == 'omit_dq':
            separation = max(abs(objectives(wrong)[i][1][2]/values[i][1][2]-1) for i in range(2))
        else:
            separation = max(scaled(wrong[f], exact[f]) for f in ('E', 'H'))
        assert separation > 1e-4, (mutant, separation)
        separations[mutant] = separation
    record_property('mutant_separations', {name: float(value) for name, value in separations.items()})
    record_property('maximum_secant_relative_error', float(max(secant_errors)))


@pytest.mark.parametrize('rotation', range(3))
@pytest.mark.parametrize('steps', [17, 41])
@pytest.mark.parametrize('backend,device', [('numpy', 'cpu'), ('torch', 'cpu')], ids=['numpy', 'cpu'])
def test_region_forward_cpu(backend, device, steps, rotation):
    result = fdtd.run(problem(rotation=rotation), numerics=numerics(steps), backend=backend, device=device)
    verify(result, reference(rotation=rotation, steps=steps), steps)


@pytest.mark.parametrize('rotation', range(3))
@pytest.mark.parametrize('steps', [17, 41])
@pytest.mark.parametrize('device,block', [('cpu', None), ('cuda:0', None), ('cuda:0', 7)],
                         ids=['cpu-plain', 'cuda-plain', 'cuda-k7'])
def test_region_material_tangents_and_reuse(device, block, steps, rotation, record_property):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(1)
    if device != 'cpu' and not torch.cuda.is_available():
        pytest.skip('requires actual CUDA')
    arrays = patterns(rotation)
    saved_arrays = tuple(a.copy() for a in arrays)
    prepared = fdtd.prepare(problem(rotation=rotation, arrays=arrays), numerics=numerics(steps, block),
                            backend='torch', device=device)
    assert prepared.contract['resolved_steps'] == steps
    assert prepared.contract['resolved_checkpoint_steps'] == block
    old = None
    for index, point in enumerate(POINTS):
        leaves = tuple(torch.tensor(value, dtype=torch.float64, device='cpu' if p == 3 else device,
                                    requires_grad=True) for p, value in enumerate(point))
        leaf_copies = tuple(a.detach().clone() for a in leaves)
        live = problem(leaves, rotation, arrays=arrays)
        assert prepared.validate(live).ok
        result = prepared.run(live)
        exact = reference(point, rotation, steps)
        error = verify(result, exact, steps)
        actual_values = torch.cat((result.final_fields.values['E'], Z0*result.final_fields.values['H']))
        actual_records = torch.stack([result[label].values*(Z0 if c >= 3 else 1)
                                      for c, label in enumerate(LABELS)])
        fw, rw = weights(exact['E'].shape, steps)
        losses = ((torch.as_tensor(fw, device=device)*actual_values.square()).mean(),
                  (torch.as_tensor(rw, device=device)*actual_records.square()).mean())
        for i, (loss, (value, expected_grad)) in enumerate(zip(losses, objectives(exact))):
            gradients = torch.autograd.grad(loss, leaves, retain_graph=i == 0)
            actual_grad = np.array([float(g.detach().cpu()) for g in gradients])
            np.testing.assert_allclose(float(loss.detach().cpu()), value, rtol=TOL, atol=0)
            np.testing.assert_allclose(actual_grad, expected_grad, rtol=TOL, atol=0)
            np.testing.assert_allclose(actual_grad[3], 2*value/point[3], rtol=TOL, atol=0)
            assert np.all(abs(expected_grad*np.asarray(point))/value > 1e-6)
            for leaf, gradient, saved in zip(leaves, gradients, leaf_copies):
                assert leaf.is_leaf and leaf.grad is None and leaf.requires_grad
                assert gradient.device == leaf.device and gradient.dtype == torch.float64
                assert torch.equal(leaf, saved)
            record_property(f'point_{index}_objective_{i}_gradient', actual_grad.tolist())
        for f in ('E', 'H'):
            assert result.final_fields.values[f].device == torch.device(device)
            assert result.final_fields.values[f].dtype == torch.float64
        if old is not None:
            unchanged(*old)
        for original, saved in zip(arrays, saved_arrays):
            np.testing.assert_array_equal(original, saved)
        old = result, snapshot(result)
        record_property(f'point_{index}_field_record_error', error)


@pytest.mark.parametrize('rotation', range(3))
@pytest.mark.parametrize('device', ['cpu', 'cuda:0'], ids=['cpu', 'cuda'])
def test_region_lossless_material(device, rotation):
    torch = pytest.importorskip('torch')
    if device != 'cpu' and not torch.cuda.is_available():
        pytest.skip('requires actual CUDA')
    point = (2.25, 1.3, 0., 1.2)
    with torch.no_grad():
        result = fdtd.run(problem(point, rotation, lossless=True), numerics=numerics(17),
                          backend='torch', device=device)
    verify(result, reference(point, rotation), 17)


@pytest.mark.parametrize('device,block', [('cpu', None), ('cuda:0', None), ('cuda:0', 7)],
                         ids=['cpu-plain', 'cuda-plain', 'cuda-k7'])
def test_region_background_tangents_same_si_units(device, block, record_property):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(1)
    if device != 'cpu' and not torch.cuda.is_available():
        pytest.skip('requires actual CUDA')
    bg = (1.4, 1.1, 80.)
    point, steps, rotation = POINTS[0], 41, 1
    expected = reference(point, rotation, steps, bg, differentiate='background')
    previous = None
    for unit in ('um', 'nm'):
        leaves = tuple(torch.tensor(value, dtype=torch.float64, device='cpu' if p == 3 else device,
                                    requires_grad=True) for p, value in enumerate((*bg, point[3])))
        copies = tuple(a.detach().clone() for a in leaves)
        live = problem((*point[:3], leaves[3]), rotation, unit, leaves[:3])
        result = fdtd.run(live, numerics=numerics(steps, block, unit), backend='torch', device=device)
        verify(result, expected, steps)
        fw, rw = weights(expected['E'].shape, steps)
        values = torch.cat((result.final_fields.values['E'], Z0*result.final_fields.values['H']))
        records = torch.stack([result[label].values*(Z0 if c >= 3 else 1)
                               for c, label in enumerate(LABELS)])
        losses = ((torch.as_tensor(fw, device=device)*values.square()).mean(),
                  (torch.as_tensor(rw, device=device)*records.square()).mean())
        for i, (loss, (value, derivative)) in enumerate(zip(losses, objectives(expected))):
            gradients = torch.autograd.grad(loss, leaves, retain_graph=i == 0)
            got = np.array([float(g.detach().cpu()) for g in gradients])
            np.testing.assert_allclose(float(loss.detach().cpu()), value, rtol=TOL, atol=0)
            np.testing.assert_allclose(got, derivative, rtol=TOL, atol=0)
            assert np.all(abs(derivative*np.array([*bg, point[3]]))/value > 1e-6)
            for leaf, grad, saved in zip(leaves, gradients, copies):
                assert grad.device == leaf.device and leaf.is_leaf and leaf.grad is None
                assert torch.equal(leaf, saved)
            record_property(f'{unit}_objective_{i}_gradient', got.tolist())
        current = snapshot(result)
        if previous is not None:
            for key in current:
                assert scaled(current[key], previous[key]) < TOL
        previous = current
