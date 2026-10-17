"""Homogeneous Lorentz forward/reverse allocator and elapsed-time study.

One fresh process per policy. Independent roots and a four-state Frechet matrix
validate the complete Record, objective and both derivatives outside timing.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback

STEPS = 163
AMPLITUDE, PARAMETER = 1.2, .8
EPSILON, MU = 2.25, 1.3
DX = .05
GIB = 2**30


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cuda:0')
    parser.add_argument('--size', type=int, choices=(8, 32, 64), required=True)
    parser.add_argument('--checkpoint-steps', type=int, choices=(8, 32), default=None)
    parser.add_argument('--forecast-from', type=Path)
    parser.add_argument('--profile', action='store_true', help='separate profile-only process, no timing samples')
    args = parser.parse_args(argv)
    if args.device == 'cpu' and (args.size != 8 or args.checkpoint_steps is not None):
        parser.error('CPU qualification is size 8 / plain only')
    if args.device != 'cpu' and args.size == 8:
        parser.error('CUDA measurements are size 32 or 64')
    if (args.size == 64) != (args.forecast_from is not None):
        parser.error('exactly size 64 requires --forecast-from authentic corresponding size-32 attempt.json')
    args.source_root, args.output_dir = args.source_root.resolve(), args.output_dir.resolve()
    if args.output_dir.is_relative_to(args.source_root):
        parser.error('artifacts must be outside the source repository')
    return args


def oracle(size, dt, time_scale, parameter=PARAMETER, amplitude=AMPLITUDE, steps=STEPS):
    import numpy as np
    omega0, gamma, strength0 = 5/time_scale, .7/time_scale, 12/time_scale**2
    denominator = 1+gamma*dt/2
    g, h, drive0 = (2-omega0**2*dt**2)/denominator, (1-gamma*dt/2)/denominator, strength0*dt**2/denominator
    drive = parameter*drive0
    # x=(E, Q=P/eps0, Q_previous, delta); memory starts entirely at rest.
    matrix = np.zeros((4, 4))
    matrix[0] = (1, 0, 0, -1/EPSILON)
    matrix[1] = drive*matrix[0]+np.array([0, g, -h, 0])
    matrix[2, 1] = 1
    matrix[3] = matrix[1]-matrix[2]
    derivative = np.zeros((4, 4))
    derivative[1] = derivative[3] = drive0*matrix[0]
    block = np.block([[matrix, derivative], [np.zeros((4, 4)), matrix]])
    start = np.zeros(8)
    start[4] = 1
    tangent, state = np.split(np.linalg.matrix_power(block, steps)@start, 2)
    expected = size**3*amplitude**2*state[0]**2
    gradients = np.array([2*expected/amplitude, 2*size**3*amplitude**2*state[0]*tangent[0]])
    equilibrium = omega0**2/(omega0**2+parameter*strength0/EPSILON)
    roots = np.roots([1., -(g-drive/EPSILON), h]).astype(complex)
    assert abs(roots[0]-roots[1]) > 1e-5
    first = (1-equilibrium)*(1-roots[1])/(roots[0]-roots[1])
    second = 1-equilibrium-first
    values = equilibrium+first*roots[0]**np.arange(steps+1)+second*roots[1]**np.arange(steps+1)
    assert np.max(abs(values.imag)) < 1e-13
    assert expected > 0 and np.all(abs(gradients) > 1e-5*expected)
    return dict(value=float(expected), gradients=gradients, record=amplitude*values.real,
                final_e=amplitude*state[0], g=g, h=h, drive=drive)


def forecast(reference, report):
    """Match an authentic same-instrument 32-cell ADE receipt, never conductor data."""
    config = reference['config']
    required = dict(report['config'], size=32, profile=False)
    for key in ('device', 'size', 'checkpoint_steps', 'steps', 'repeat', 'warmup', 'profile'):
        if config.get(key) != required[key]:
            raise ValueError('forecast configuration mismatch: '+key)
    if reference.get('status') != 'passed' or len(reference.get('samples', [])) != 5:
        raise ValueError('forecast requires a passed five-repetition ADE receipt')
    for key in ('script_sha256', 'source_sha256', 'fixture_identity', 'product_head'):
        if reference.get(key) != report.get(key):
            raise ValueError('forecast source/fixture mismatch: '+key)
    for key in ('torch', 'cuda_runtime', 'numpy', 'gpu', 'gpu_total_bytes', 'precision',
                'executable', 'python', 'platform', 'cpu_threads', 'torch_file', 'wavebit_file'):
        if reference['environment'].get(key) != report['environment'].get(key):
            raise ValueError('forecast environment mismatch: '+key)
    if not reference.get('source_integrity', {}).get('unchanged'):
        raise ValueError('forecast requires unchanged before/after source identity')
    if not all(sample.get('checks_passed') for sample in reference['samples']):
        raise ValueError('forecast numerical checks are incomplete')
    observed = max(sample['retained']['peak_reserved_bytes'] for sample in reference['samples'])
    if not isinstance(observed, int) or observed <= 0:
        raise ValueError('forecast requires a positive CUDA reserved peak')
    predicted = observed*8*3//2
    if predicted > 4*GIB:
        raise ValueError('ADE reserved-memory forecast exceeds 4 GiB budget')
    return dict(reference_peak_reserved_bytes=observed, cell_multiplier=8, margin_multiplier=1.5,
                forecast_reserved_bytes=predicted, budget_bytes=4*GIB, headroom_bytes=GIB)


def main(argv=None):
    args = arguments(argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    path = args.output_dir/'attempt.json'
    report = dict(status='running', stage='imports', started_utc=datetime.now(timezone.utc).isoformat(),
        command=sys.argv, script_sha256=digest(__file__),
        config=dict(device=args.device, size=args.size,
                    checkpoint_steps=args.checkpoint_steps, steps=STEPS, repeat=5, warmup=1,
                    profile=args.profile), samples=[],
        timing='synchronized live leaf creation + prepared.run (including live validation/lowering/adaptation) + sum(E squared) + autograd.grad; excludes prepare, host copies, numerical checks, allocator queries, cleanup and profiling',
        allocation_policy='empty cache after warmup and after all repetitions; retain cache between repetitions; release all returned leaves, outputs, loss and derivatives before next baseline',
        limitations=['Uniform source-free closed-box Lorentz workload, not optical convergence.',
            'Forecast is a same-policy 32-cell empirical estimate, not an allocation guarantee.',
            'Torch allocator excludes driver/display/other processes.',
            'CPU qualification uses the ordinary three-history policy; CPU seconds are not CUDA evidence.'])

    def save(stage=None):
        if stage:
            report['stage'] = stage
        write_json(path, report)

    save()
    try:
        sys.path.insert(0, str(args.source_root/'src'))
        import numpy as np
        import torch
        import wavebit
        from wavebit import Units, fdtd
        from wavebit.materials import DispersiveMaterial, Lorentz
        from wavebit.fdtd import _runtime
        from wavebit.fdtd._kernel.poles import CompactPoleTerm, PoleTerm
        if Path(wavebit.__file__).resolve() != (args.source_root/'src/wavebit/__init__.py').resolve():
            raise RuntimeError('wrong Wavebit source checkout imported')
        torch.set_num_threads(1)
        cuda = args.device != 'cpu'
        report['environment'] = dict(executable=sys.executable, python=platform.python_version(),
            platform=platform.platform(), torch=torch.__version__, torch_file=torch.__file__,
            numpy=np.__version__, cuda_runtime=torch.version.cuda, wavebit_file=wavebit.__file__,
            precision='float64', cpu_threads=torch.get_num_threads())
        print(json.dumps(report['environment']), flush=True)
        report['product_head'] = subprocess.check_output(
            ['git', '-C', str(args.source_root), 'rev-parse', 'HEAD'], text=True).strip()
        names = ('_api.py', '_runtime.py', '_execution.py', '_kernel/poles.py', '_kernel/step.py',
                 '_kernel/observe.py', '_prepare/plan.py', '_prepare/params.py',
                 '_prepare/coefficients.py', '_prepare/report.py', '_prepare/compact.py')
        report['source_sha256'] = {name: digest(args.source_root/'src/wavebit/fdtd'/name) for name in names}
        identity_before = {key: report[key] for key in ('script_sha256', 'source_sha256')}

        def source_identity():
            return dict(script_sha256=digest(__file__),
                source_sha256={name: digest(args.source_root/'src/wavebit/fdtd'/name) for name in names})
        if cuda:
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA device required')
            properties = torch.cuda.get_device_properties(args.device)
            report['environment'].update(gpu=properties.name, gpu_total_bytes=properties.total_memory)

        def sync():
            if cuda:
                torch.cuda.synchronize(args.device)

        def memory():
            if not cuda:
                return dict(allocated_bytes=None, reserved_bytes=None,
                            peak_allocated_bytes=None, peak_reserved_bytes=None)
            return dict(allocated_bytes=torch.cuda.memory_allocated(args.device),
                reserved_bytes=torch.cuda.memory_reserved(args.device),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device))

        def cleanup(empty=False):
            gc.collect()
            sync()
            if cuda and empty:
                torch.cuda.empty_cache()
            return memory()

        units = Units(length='um')
        dt = .5*DX/math.sqrt(3)*units.time_seconds
        report['fixture_identity'] = dict(kind='homogeneous-lorentz', spacing_um=DX, courant=.5,
            epsilon=EPSILON, mu=MU, omega0_si=5/units.time_seconds, gamma_si=.7/units.time_seconds,
            strength0_si=12/units.time_seconds**2, amplitude=AMPLITUDE, parameter=PARAMETER,
            initial='real uniform Ey=1; H=None', boundaries='six Periodic',
            outputs='Ey Record and final E/H', dt_seconds=dt)
        expected = oracle(args.size, dt, units.time_seconds)
        report['oracle'] = dict(value=expected['value'], gradients=expected['gradients'].tolist(),
            method='independent four-state matrix and eight-state Frechet power; separate damped quadratic roots')
        save('preflight')
        cleanup(empty=True)
        if args.size == 64:
            reference = json.loads(args.forecast_from.read_text(encoding='utf-8'))
            report['forecast'] = forecast(reference, report)
            report['forecast']['reference_path'] = str(args.forecast_from.resolve())
            report['forecast']['reference_sha256'] = digest(args.forecast_from)
        if cuda:
            free, total = torch.cuda.mem_get_info(args.device)
            report['preconstruction'] = dict(free_bytes=free, total_bytes=total, allocator=memory())
            save()
            required = report.get('forecast', {}).get('forecast_reserved_bytes', 0)+GIB
            if free < required:
                raise RuntimeError('insufficient VRAM for forecast plus 1 GiB headroom')
        electric = np.zeros((3, args.size, args.size, args.size), dtype=np.float64)
        electric[1] = 1.
        saved_pattern = electric.copy()
        side = args.size*DX
        pole = Lorentz(omega_0=5/units.time_seconds, gamma=.7/units.time_seconds,
                       strength=PARAMETER*12/units.time_seconds**2)
        material = DispersiveMaterial(epsilon_inf=EPSILON, mu=MU, poles=(pole,))
        problem = fdtd.Problem(units=units, scene=fdtd.Scene(
            domain=fdtd.Domain(size=(side,)*3),
            boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.Periodic()),
            structures=(fdtd.Structure('body', fdtd.Slab(z=(-side/2, side/2)), material),)),
            initial=fdtd.InitialField(E=electric, amplitude=AMPLITUDE),
            outputs=fdtd.Outputs(final_fields=True, monitors=(fdtd.Record(
                'probe', at=(0., 0., 0.), component='Ey'),)))
        policy = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=DX),
            duration=fdtd.Duration(steps=STEPS), courant=.5, checkpoint_steps=args.checkpoint_steps)
        save('prepare')
        prepared = fdtd.prepare(problem, numerics=policy, backend='torch', device=args.device)
        assert prepared.report.ok and prepared.contract['resolved_steps'] == STEPS
        assert prepared.contract['resolved_checkpoint_steps'] == args.checkpoint_steps
        np.testing.assert_allclose(prepared.contract['resolved_dt_seconds'], dt, rtol=2e-15, atol=0.)
        term, = prepared._plan.terms
        assert type(term) is (CompactPoleTerm if cuda else PoleTerm)
        rt = _runtime.runtime('torch', args.device)
        assert type(rt) is _runtime.TorchRuntime
        arrays_per_pole = getattr(term, 'memory_arrays_per_pole', 3)
        vector_bytes = 3*args.size**3*8
        report['payload'] = dict(state_bytes_report=int(prepared.report.estimates['state_bytes']),
            parameter_bytes_report=int(prepared.report.estimates['parameter_bytes']),
            electric_magnetic_bytes=2*vector_bytes, pole_arrays=arrays_per_pole,
            pole_bytes=arrays_per_pole*vector_bytes,
            main_fields_pole_bytes=(2+arrays_per_pole)*vector_bytes,
            record_bytes=STEPS*8, initial_host_pattern_bytes=electric.nbytes,
            scope='logical array payload, not autograd graph or allocator peak')
        assert report['payload']['state_bytes_report'] == report['payload']['main_fields_pole_bytes']
        actual_memory = term.init_mem(prepared._plan, rt)
        measured_pole_bytes = sum(value.numel()*value.element_size()
                                  for values in actual_memory.values() for value in values)
        assert measured_pole_bytes == report['payload']['pole_bytes']
        report['payload']['actual_init_mem_bytes'] = measured_pole_bytes
        report['payload']['actual_init_mem_keys'] = list(actual_memory)
        del actual_memory
        report['term_class'] = type(term).__module__+'.'+type(term).__qualname__
        report['runtime_class'] = type(rt).__module__+'.'+type(rt).__qualname__

        def forward():
            leaves = tuple(torch.tensor(value, dtype=torch.float64, device=args.device, requires_grad=True)
                           for value in (AMPLITUDE, PARAMETER))
            live_pole = pole.replace(strength=leaves[1]*12/units.time_seconds**2)
            live_material = material.replace(poles=(live_pole,))
            live_structure = problem.scene.structures[0].replace(material=live_material)
            live = problem.replace(scene=problem.scene.replace(structures=(live_structure,)),
                                   initial=problem.initial.replace(amplitude=leaves[0]))
            return prepared.run(live), leaves

        def solve():
            result, leaves = forward()
            value = result.final_fields.values['E'].square().sum()
            derivatives = torch.autograd.grad(value, leaves)
            return result, value, derivatives, leaves

        def host(value):
            return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)

        def validate(bundle):
            result, value, derivatives, leaves = bundle
            assert bool(result.validity.checks['runtime_stability'])
            for field in result.final_fields.values.values():
                assert field.device == torch.device(args.device) and field.dtype == torch.float64
                assert np.isfinite(host(field)).all()
            e, h = (host(result.final_fields.values[key]) for key in ('E', 'H'))
            np.testing.assert_allclose(e[1], expected['final_e'], rtol=1e-11, atol=0.)
            assert np.max(abs(e[[0, 2]])) == 0 and np.max(abs(h)) == 0
            record = host(result['probe'].values)
            assert result['probe'].values.device == torch.device(args.device)
            record_error = float(np.max(abs(record-expected['record'][1:]))/np.max(abs(expected['record'])))
            assert record_error < 1e-11
            series = np.r_[AMPLITUDE, record]
            residual = series[2:]-(expected['g']-expected['drive']/EPSILON)*series[1:-1] + expected['h']*series[:-2] - AMPLITUDE*(1-expected['g']+expected['h'])
            recurrence = float(np.max(abs(residual)))
            assert recurrence < 1e-12
            np.testing.assert_allclose(host(result['probe'].times), np.arange(1, STEPS+1)*dt, rtol=2e-14, atol=0.)
            actual = float(host(value))
            gradients = [float(host(v)) for v in derivatives]
            np.testing.assert_allclose(actual, expected['value'], rtol=2e-12, atol=0.)
            np.testing.assert_allclose(gradients, expected['gradients'], rtol=2e-12, atol=0.)
            for leaf, derivative, original in zip(leaves, derivatives, (AMPLITUDE, PARAMETER)):
                assert leaf.is_leaf and leaf.grad is None and float(host(leaf)) == original
                assert derivative.device == leaf.device and derivative.dtype == leaf.dtype
            np.testing.assert_array_equal(electric, saved_pattern)
            return dict(checks_passed=True, value=actual, gradients=gradients,
                        record_peak_relative_error=record_error, recurrence_max_abs=recurrence,
                        recurrence_relative_initial_amplitude=recurrence/AMPLITUDE)

        cleanup(empty=True)
        if cuda:
            free, total = torch.cuda.mem_get_info(args.device)
            report['prewarmup'] = dict(free_bytes=free, total_bytes=total, allocator=memory())
            save()
            required = report.get('forecast', {}).get('forecast_reserved_bytes', 0)+GIB
            if free < required:
                raise RuntimeError('insufficient VRAM immediately before warmup')
        save('warmup')
        bundle = solve()
        sync()
        report['warmup'] = validate(bundle)
        np.savez_compressed(args.output_dir/'warmup-record.npz',
                            values=host(bundle[0]['probe'].values), reference=expected['record'][1:])
        report['warmup_arrays_sha256'] = digest(args.output_dir/'warmup-record.npz')
        del bundle
        report['after_warmup_cleanup'] = cleanup(empty=True)
        if not args.profile:
            save('measurement')
            for index in range(5):
                sync()
                if cuda:
                    torch.cuda.reset_peak_memory_stats(args.device)
                baseline = memory()
                report['active_repetition'] = index
                save()
                sync()
                start = time.perf_counter()
                bundle = solve()
                sync()
                elapsed = time.perf_counter()-start
                retained = memory()
                sample = validate(bundle)
                sample.update(repetition=index, seconds=elapsed, baseline=baseline, retained=retained,
                    incremental_peak_allocated_bytes=retained['peak_allocated_bytes']-baseline['allocated_bytes'] if cuda else None,
                    incremental_peak_reserved_bytes=retained['peak_reserved_bytes']-baseline['reserved_bytes'] if cuda else None)
                del bundle
                sample['post_release'] = cleanup()
                report['samples'].append(sample)
                save()
            report['measurement_cleanup'] = cleanup(empty=True)
            timings = [row['seconds'] for row in report['samples']]
            report['measurement'] = dict(seconds=timings, median_seconds=statistics.median(timings),
                min_seconds=min(timings), max_seconds=max(timings))
            save('untimed_lifecycle')
            if cuda:
                torch.cuda.reset_peak_memory_stats(args.device)
            lifecycle = dict(baseline=memory(), timing='not measured; synchronization and allocator queries between forward and reverse')
            result, leaves = forward()
            sync()
            lifecycle['after_forward'] = memory()
            value = result.final_fields.values['E'].square().sum()
            derivatives = torch.autograd.grad(value, leaves)
            sync()
            lifecycle['after_backward'] = memory()
            lifecycle['validation'] = validate((result, value, derivatives, leaves))
            del result, value, derivatives, leaves
            lifecycle['after_release'] = cleanup(empty=True)
            report['lifecycle'] = lifecycle
            save('untimed_ownership')
            first = solve()
            originals = tuple(first[0].final_fields.values.values())+(first[0]['probe'].values,)
            snapshots = tuple(np.array(host(v), copy=True) for v in originals)
            second = solve()
            validate(first)
            validate(second)
            next_values = tuple(second[0].final_fields.values.values())+(second[0]['probe'].values,)
            for old, snapshot, new in zip(originals, snapshots, next_values):
                np.testing.assert_array_equal(host(old), snapshot)
                assert old.untyped_storage().data_ptr() != new.untyped_storage().data_ptr()
            report['ownership'] = dict(passed=True, separate_runs=2)
            del first, second, originals, snapshots, next_values, old, new, snapshot
            report['ownership_cleanup'] = cleanup(empty=True)
        else:
            save('profile_only')
            activities = [torch.profiler.ProfilerActivity.CPU]
            if cuda:
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(activities=activities, profile_memory=True,
                    record_shapes=False, with_stack=False) as profiler:
                bundle = solve()
                sync()
            report['profile_validation'] = validate(bundle)
            rows = [dict(name=e.key, count=e.count, cpu_time_total_us=e.cpu_time_total,
                self_cpu_time_total_us=e.self_cpu_time_total, device_time_total_us=e.device_time_total,
                self_device_time_total_us=e.self_device_time_total,
                self_cpu_memory_usage_bytes=e.self_cpu_memory_usage,
                self_device_memory_usage_bytes=e.self_device_memory_usage) for e in profiler.key_averages()]
            write_json(args.output_dir/'operators.json', rows)
            profiler.export_chrome_trace(str(args.output_dir/'trace.json'))
            report['profile'] = dict(operators_sha256=digest(args.output_dir/'operators.json'),
                trace_sha256=digest(args.output_dir/'trace.json'), separate_runs=1,
                cuda_events=sum(str(e.device_type).endswith('.CUDA') for e in profiler.events()))
            if cuda:
                assert report['profile']['cuda_events'] > 0
            del bundle, profiler
        report['final_cleanup'] = cleanup(empty=True)
        identity_after = source_identity()
        report['source_integrity'] = dict(before=identity_before, after=identity_after,
                                          unchanged=identity_before == identity_after)
        if identity_before != identity_after:
            raise RuntimeError('source identity changed during measurement; reject receipt')
        report['status'] = 'passed'
        save('complete')
        print(json.dumps(dict(status='passed', receipt=str(path), measurement=report.get('measurement'))), flush=True)
    except BaseException as exc:
        report['status'] = 'failed'
        report['failure'] = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        save()
        raise


if __name__ == '__main__':
    main()
