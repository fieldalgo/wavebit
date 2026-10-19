"""Bounded CUDA ADE storage selection, reuse and independent derivative gates.

CPU/NumPy/JAX retain ordinary three-history plans. The compact CUDA plan has
its own static identity and may not be executed through another backend.
"""
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from wavebit import fdtd
from wavebit.fdtd import _api, _execution
from wavebit.fdtd._kernel.poles import CompactPoleTerm, PoleTerm
from wavebit.fdtd._prepare.compact import compact_plan
from wavebit.fdtd._runtime import runtime
from wavebit.units import Z0

from tests.fdtd._homogeneous_ade_reference import (
    UNITS, fourier_mode, host, pole_parameters, uniform_history,
)
from tests.fdtd.test_general_ade_ad import (
    CELLS, POINTS, live_problem, oracle_states, raw_outputs, reconstruct,
)
from tests.fdtd.test_general_ade_contract import numerics, problem

# The ordinary pre-compaction plan at dad313c0. Material values and pole family
# are live data; this fixture's topology and exact static arrays fix its hash.
ORDINARY_SKELETON = 'ada66b0a2cf7e62e4a70ce95f466e9547ae37b58e4021736c580a7ecf07db75c'


def available(backend, device):
    if backend == 'jax':
        pytest.importorskip('jax').config.update('jax_enable_x64', True)
    if backend == 'torch':
        torch = pytest.importorskip('torch')
        torch.set_num_threads(1)
        if device != 'cpu' and not torch.cuda.is_available():
            pytest.skip('requires an actual CUDA device')


@pytest.mark.parametrize('kind', ['lorentz', 'drude'])
@pytest.mark.parametrize('backend,device', [('numpy', 'cpu'), ('torch', 'cpu'),
    ('jax', 'cpu'), ('torch', 'cuda:0')], ids=['cpu-numpy', 'cpu-torch', 'cpu-jax', 'gpu-torch'])
def test_selected_state_bytes_and_concrete_reuse(backend, device, kind):
    available(backend, device)
    p, n = problem(kind), numerics()
    check = fdtd.check(p, numerics=n, backend=backend, device=device)
    prepared = fdtd.prepare(p, numerics=n, backend=backend, device=device)
    compact = device != 'cpu'
    expected_type = CompactPoleTerm if compact else PoleTerm
    components = 12 if compact else 15
    assert check.ok and prepared.validate(p).ok
    assert check.estimates['state_bytes'] == prepared.report.estimates['state_bytes'] == components*8**3*8
    term, = prepared._plan.terms
    assert type(term) is expected_type
    assert (prepared._plan.skeleton != ORDINARY_SKELETON) is compact
    memory = term.init_mem(prepared._plan, runtime(backend, device))
    assert set(memory) == ({'P', 'previous'} if compact else {'P', 'previous', 'delta'})
    assert sum(host(a).nbytes for group in memory.values() for a in group) == (components-6)*8**3*8
    assert all(np.count_nonzero(host(a)) == 0 for group in memory.values() for a in group)
    del memory
    original = prepared.run(p)
    snapshots = {key: host(value).copy() for key, value in original.final_fields.values.items()}
    old_record = host(original['electric'].values).copy()
    dt = .5/8/math.sqrt(3)*UNITS.time_unit_seconds
    omega0, gamma, strength = pole_parameters(kind, True)
    expected = uniform_history(17, dt, omega0, gamma, strength)
    np.testing.assert_allclose(host(original['electric'].values), expected[1:], rtol=1e-11, atol=0.)
    changed = problem(kind, epsilon_inf=1.9, mu=1.1, gamma=.4/UNITS.time_unit_seconds,
                      omega_0=4.2/UNITS.time_unit_seconds)
    assert prepared.validate(changed).ok
    result = prepared.run(changed)
    reference = uniform_history(17, dt, 0. if kind == 'drude' else 4.2/UNITS.time_unit_seconds,
                                .4/UNITS.time_unit_seconds, strength, epsilon=1.9)
    np.testing.assert_allclose(host(result['electric'].values), reference[1:], rtol=1e-11, atol=0.)
    np.testing.assert_allclose(host(result.final_fields.values['E'])[1], reference[-1], rtol=1e-11, atol=0.)
    assert np.max(abs(host(result.final_fields.values['H']))) == 0
    assert bool(result.validity.checks['runtime_stability'])
    assert type(prepared._plan.terms[0]) is expected_type
    for key, value in original.final_fields.values.items():
        np.testing.assert_array_equal(host(value), snapshots[key])
    np.testing.assert_array_equal(host(original['electric'].values), old_record)


def objective_reference(kind, amplitude, parameter, dt, discrete_k, initial, spatial, steps):
    value, first, second = [reconstruct(state, spatial) for state in
        oracle_states(kind, parameter, dt, discrete_k, initial, steps=steps)]
    q, qp, qpp = np.dot(value, value), 2*np.dot(value, first), 2*(np.dot(first, first)+np.dot(value, second))
    loss = amplitude**2*q
    gradient = np.array([2*amplitude*q, amplitude**2*qp])
    hessian = np.array([[2*q, 2*amplitude*qp], [2*amplitude*qp, amplitude**2*qpp]])
    assert loss > 1 and np.all(abs(gradient) > 1e-5*loss) and np.all(abs(hessian) > 1e-5*loss)
    return loss, gradient, hessian


@pytest.mark.parametrize('kind', ['lorentz', 'drude'])
@pytest.mark.parametrize('block', [8, 32], ids=['k8-tail3', 'k32-tail3'])
@pytest.mark.parametrize('warm_mode', ['no_grad', 'inference_mode'])
def test_gpu_checkpoint_reuse_after_nograd_full_hessian(kind, block, warm_mode):
    torch = pytest.importorskip('torch')
    available('torch', 'cuda:0')
    steps = 35
    _, discrete_k, initial, electric, magnetic, spatial = fourier_mode(CELLS, 0)
    patterns = electric.copy(), magnetic.copy()
    from tests.fdtd._homogeneous_ade_reference import numerics as mode_numerics
    policy = mode_numerics(CELLS, steps).replace(checkpoint_steps=block)
    p = live_problem(kind, *POINTS[0], electric, magnetic)
    prepared = fdtd.prepare(p, numerics=policy, backend='torch', device='cuda:0')
    assert type(prepared._plan.terms[0]) is CompactPoleTerm
    assert prepared.contract['resolved_steps'] == steps
    assert prepared.contract['resolved_checkpoint_steps'] == block and steps % block == 3
    dt = .5/CELLS/math.sqrt(3)*UNITS.time_unit_seconds
    with getattr(torch, warm_mode)():
        warm = prepared.run(p)
        warm_loss = warm.final_fields.values['E'].square().sum()+(Z0*warm.final_fields.values['H']).square().sum()
    wanted = objective_reference(kind, *POINTS[0], dt, discrete_k, initial, spatial, steps)
    np.testing.assert_allclose(host(warm_loss), wanted[0], rtol=2e-12, atol=0.)
    old = tuple((value, value.detach().clone()) for value in raw_outputs(warm))
    for amplitude, parameter in POINTS:
        leaves = (torch.tensor(amplitude, dtype=torch.float64, requires_grad=True),
                  torch.tensor(parameter, dtype=torch.float64, device='cuda:0', requires_grad=True))
        saved = tuple(value.detach().clone() for value in leaves)
        live = live_problem(kind, *leaves, electric, magnetic)
        assert prepared.validate(live).ok
        result = prepared.run(live)
        fields = result.final_fields.values
        loss = fields['E'].square().sum()+(Z0*fields['H']).square().sum()
        gradient = torch.autograd.grad(loss, leaves, create_graph=True)
        rows = [torch.autograd.grad(value, leaves, retain_graph=index == 0)
                for index, value in enumerate(gradient)]
        actual = (host(loss), np.array([float(host(x)) for x in gradient]),
                  np.array([[float(host(x)) for x in row] for row in rows]))
        expected = objective_reference(kind, amplitude, parameter, dt, discrete_k, initial, spatial, steps)
        for value, wanted in zip(actual, expected):
            np.testing.assert_allclose(value, wanted, rtol=2e-12, atol=0.)
        for group in (gradient, *rows):
            for leaf, derivative, snapshot in zip(leaves, group, saved):
                assert leaf.is_leaf and leaf.grad is None and torch.equal(leaf, snapshot)
                assert derivative.device == leaf.device and derivative.dtype == leaf.dtype
        for value, snapshot in old:
            assert torch.equal(value, snapshot)
        old = tuple((value, value.detach().clone()) for value in raw_outputs(result))
        assert all(value.device.type == 'cuda' and value.dtype == torch.float64 for value in raw_outputs(result))
        assert bool(result.validity.checks['runtime_stability'])
        np.testing.assert_array_equal(electric, patterns[0])
        np.testing.assert_array_equal(magnetic, patterns[1])


def ordinary():
    scene, _ = _api.lower_problem(problem(), numerics())
    plan, _ = _execution._planned(scene)
    assert type(plan.terms[0]) is PoleTerm and plan.skeleton == ORDINARY_SKELETON
    return scene, plan


@pytest.mark.parametrize('backend', ['numpy', 'torch', 'jax'])
def test_cpu_compact_plan_rejects_wrong_backend_before_execution(backend):
    available(backend, 'cpu')
    scene, base = ordinary()
    compact = compact_plan(scene, base)
    with pytest.raises(ValueError, match='CompactPoleTerm requires Torch CUDA execution'):
        _execution.run(scene, backend=backend, device='cpu', plan=compact)
    with pytest.raises(ValueError, match='CompactPoleTerm requires Torch CUDA execution'):
        _execution.execute(compact, {}, runtime(backend, 'cpu'))


@pytest.mark.parametrize('bad', ['decay', 'source', 'wall', 'partial', 'multipole',
                                  'subclass', 'film', 'monitor', 'complex', 'extra'])
def test_cpu_internal_compaction_refuses_unqualified_scope(bad):
    from wavebit.fdtd import _legacy as fd
    scene, base = ordinary()
    structure, = scene.structures
    term, = base.terms
    if bad == 'decay':
        scene = replace(scene, run=replace(scene.run, decay=fd.Decay(threshold=.1, window=.2)))
    elif bad == 'source':
        scene = replace(scene, sources=(fd.Dipole('drive', center=(0., 0., 0.), component='Ey',
                                                  band=fd.Band(wavelength=1., df=.2)),))
    elif bad == 'wall':
        scene = replace(scene, boundaries=replace(scene.boundaries, z=(fd.PEC(), fd.PEC())))
    elif bad == 'partial':
        base = base.with_static(terms=(replace(term, region=replace(term.region, z=(0, 7))),))
    elif bad == 'multipole':
        scene = replace(scene, structures=(replace(structure, material=replace(
            structure.material, poles=structure.material.poles*2)),))
    elif bad == 'subclass':
        base = base.with_static(terms=(CompactPoleTerm(term.name, term.region, term.slabs, 1),))
    elif bad == 'film':
        from tests.fdtd.test_fd_single_pole_film import film_scene
        scene = film_scene()
        base = fd.plan(scene)
    elif bad == 'monitor':
        scene = replace(scene, monitors=(fd.Field('dft', plane=fd.Plane(z=0.), wavelengths=(1.,)),))
    elif bad == 'complex':
        base = base.with_static(dtype='complex128')
    else:
        base = base.with_static(terms=(term, term))
    with pytest.raises(ValueError, match='compact ADE requires'):
        compact_plan(scene, base)


def test_cpu_internal_replan_preserves_compact_identity_without_global_mutation():
    scene, base = ordinary()
    compact, _ = _execution._planned(scene, compact_poles=True)
    assert compact.skeleton != base.skeleton and type(compact.terms[0]) is CompactPoleTerm
    rebuilt, _ = _execution._planned(scene, reuse=compact, compact_poles=True)
    assert rebuilt.skeleton == compact.skeleton
    ordinary_again, _ = _execution._planned(scene)
    assert ordinary_again.skeleton == base.skeleton and type(ordinary_again.terms[0]) is PoleTerm
    changed = replace(scene, structures=(replace(scene.structures[0], material=replace(
        scene.structures[0].material, eps_inf=1.9)),))
    reused, _ = _execution._planned(changed, reuse=compact, compact_poles=True)
    assert reused.skeleton == compact.skeleton


def test_cpu_metadata_hashes_are_deterministic_across_fresh_processes():
    # Skip only device probing: check/prepare allocate no CUDA runtime. A real
    # device is exercised by the separate gpu-torch and derivative cases.
    source = '''import json
from wavebit import fdtd
from wavebit._common.execution import Execution
from tests.fdtd.test_general_ade_contract import problem, numerics
Execution.validate = lambda self: self
rows={}
for device in ('cpu','cuda:0'):
    p=fdtd.prepare(problem(),numerics=numerics(),backend='torch',device=device)
    rows[device]=dict(skeleton=p._plan.skeleton,term=type(p._plan.terms[0]).__name__)
print(json.dumps(rows))
'''
    root = Path(__file__).resolve().parents[2]
    outputs = []
    for seed in ('13', '71'):
        env = dict(os.environ, PYTHONPATH=str(root/'src'), PYTHONHASHSEED=seed)
        process = subprocess.run([sys.executable, '-c', source], cwd=root, env=env,
                                 capture_output=True, text=True, check=True)
        outputs.append(json.loads(process.stdout.strip().splitlines()[-1]))
    assert outputs[0] == outputs[1]
    assert outputs[0]['cpu'] == dict(skeleton=ORDINARY_SKELETON, term='PoleTerm')
    assert outputs[0]['cuda:0']['term'] == 'CompactPoleTerm'
    assert outputs[0]['cuda:0']['skeleton'] != ORDINARY_SKELETON
