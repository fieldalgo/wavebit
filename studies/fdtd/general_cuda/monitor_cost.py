"""Bounded general FDTD monitor cost and optional CPU/CUDA operator profiling.

Two Ey-only FieldMonitors and two FluxMonitors observe a short plane-wave
film run. The finite window is a cost fixture, not an Airy accuracy claim.
Uninstrumented timings and instrumented operator attribution remain separate.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--transverse-cells', type=int, default=4)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--checkpoint-steps', type=int, default=None,
                        help='optional positive Torch CUDA checkpoint block length')
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cuda:0')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not 32 <= args.steps <= 256 or not 2 <= args.transverse_cells <= 16:
        parser.error('this bounded instrument requires 32..256 steps and 2..16 transverse cells')
    if not 1 <= args.repeat <= 20:
        parser.error('repeat must be 1..20')
    if args.checkpoint_steps is not None and args.checkpoint_steps < 1:
        parser.error('checkpoint-steps must be positive')
    return args


def main():
    args = arguments()
    # Preserve failed attempts as well as successful measurements. This receipt
    # precedes third-party imports, device admission, and all solver allocation.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    path = args.output_dir / 'attempt.json'
    report = dict(status='running', stage='imports',
        started_utc=datetime.now(timezone.utc).isoformat(),
        config=dict(steps=args.steps, transverse_cells=args.transverse_cells,
                    repeat=args.repeat, device=args.device, profile=args.profile,
                    checkpoint_steps=args.checkpoint_steps),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='prepared general FDTD no-grad forward with Ey-only FieldMonitor and E/H FluxMonitor monitors',
        timing='synchronized prepared.run including live validation, lowering and result adaptation; excludes initial preparation, result checks, host copies and profiler',
        warmup_runs=1, profiled_runs=int(args.profile), attempts=[], samples=[],
        limitations=[
            'Short-window field and flux data establish neither convergence nor Airy accuracy.',
            'CPU mode checks the instrument and is not evidence of CUDA performance.',
            'Allocator peaks exclude driver, display and complete GPU process memory.',
            'Profiler shape/memory recording changes cost and tensor lifetimes.',
            'Inclusive operator times overlap; CPU dispatch and device activity must not be added.',
            'Operator counts include preparation performed by prepared.run and all source/monitor work.'])

    def save():
        write_json(path, report)

    def stage(name):
        report['stage'] = name
        save()

    save()
    try:
        import numpy as np
        import torch
        import wavebit
        from wavebit import Material, Units, fdtd
        from wavebit.fdtd import _api, _runtime
        from wavebit.fdtd._kernel import observe

        torch.set_num_threads(1)
        cuda = args.device.startswith('cuda')
        report['environment'] = dict(python=platform.python_version(),
            platform=platform.platform(), torch=torch.__version__, numpy=np.__version__,
            cuda_runtime=torch.version.cuda, wavebit=getattr(wavebit, '__version__', None),
            cpu_threads=torch.get_num_threads(), precision='float64/complex128')
        report['source_sha256'] = {name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for name, module in (('_kernel/observe.py', observe), ('_runtime.py', _runtime), ('_api.py', _api))}
        stage('device_admission')
        activities = [torch.profiler.ProfilerActivity.CPU]
        if cuda:
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA-enabled Torch and an NVIDIA GPU are required')
            props = torch.cuda.get_device_properties(args.device)
            report['environment'].update(gpu=props.name, gpu_total_bytes=props.total_memory)
            if args.profile:
                if torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
                    raise RuntimeError('this Torch profiler does not support CUDA activity collection')
                activities.append(torch.profiler.ProfilerActivity.CUDA)

        stage('case_construction')
        units = Units(length='um')
        band = (.9, 1., 1.1)
        monitors = tuple(m for name, z in (('front', -.6), ('back', .6)) for m in (
            fdtd.FieldMonitor(name, plane=fdtd.Plane(z=z), wavelengths=band, components=('Ey',)),
            fdtd.FluxMonitor(name + '_flux', plane=fdtd.Plane(z=z), wavelengths=band)))
        problem = fdtd.Problem(units=units, scene=fdtd.Scene(
            domain=fdtd.Domain(size=(args.transverse_cells*.05, args.transverse_cells*.05, 3.)),
            background=Material(epsilon=1.), boundaries=fdtd.Boundaries.periodic(z=fdtd.Open()),
            structures=(fdtd.Structure(name='film', shape=fdtd.SnappedSlab(z=(-.1, .1)),
                                       material=Material(epsilon=2.25)),)),
            sources=(fdtd.PlaneWave('pw', plane=fdtd.Plane(z=-.9), direction='+z',
                polarization='s', pulse=fdtd.GaussianPulse(wavelength=1.,
                    frequency_sigma_hz=.15/units.time_unit_seconds)),),
            outputs=fdtd.Outputs(monitors=monitors))
        numerics = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.05),
            duration=fdtd.Duration(steps=args.steps), courant=.5,
            absorber=fdtd.PML(cells=12, wavelength=1.), checkpoint_steps=args.checkpoint_steps)
        prepared = fdtd.prepare(problem, numerics=numerics, backend='torch', device=args.device)
        if not prepared.report.ok:
            raise RuntimeError('fixture preparation failed')
        report['fixture'] = dict(grid_shape=prepared.report.resolved['grid_shape'],
            dt_seconds=prepared.report.resolved['time_step_seconds'], wavelengths=list(band),
            field_monitors=2, field_components=['Ey'], flux_monitors=2, pml_cells=12,
            checkpoint_steps=prepared.report.resolved['checkpoint_steps'])

        def sync():
            if cuda:
                torch.cuda.synchronize(args.device)

        def solve():
            with torch.no_grad():
                return prepared.run(problem)

        def inspect(result):
            checks = dict(runtime_stability=bool(result.validity.checks['runtime_stability']))
            sample = {}
            for name in ('front', 'back'):
                field = result[name].values['Ey']
                flux = result[name + '_flux'].spectral_energy
                checks[name + '_placement_dtype'] = bool(
                    field.device == torch.device(args.device) and field.dtype == torch.complex128
                    and flux.device == torch.device(args.device) and flux.dtype == torch.float64)
                field_host = field.detach().cpu().numpy()
                flux_host = flux.detach().cpu().numpy()
                checks[name + '_finite'] = bool(np.isfinite(field_host).all() and np.isfinite(flux_host).all())
                # Keep JSON valid even when numerical validation fails.
                sample[name] = dict(field_abs_max=float(np.max(np.abs(field_host)))
                    if np.isfinite(field_host).all() else 'nonfinite',
                    flux=flux_host.tolist() if np.isfinite(flux_host).all() else 'nonfinite')
            return dict(checks=checks, observations=sample)

        stage('warmup')
        warm = solve()
        sync()
        report['warmup'] = inspect(warm)
        del warm
        save()
        if not all(report['warmup']['checks'].values()):
            raise RuntimeError('warmup numerical checks failed')
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(args.device)
        report['allocator'] = dict(baseline_allocated_bytes=torch.cuda.memory_allocated(args.device)
            if cuda else None, baseline_reserved_bytes=torch.cuda.memory_reserved(args.device) if cuda else None)
        stage('measurement')
        times = []
        for index in range(args.repeat):
            attempt = dict(repetition=index, status='running')
            report['attempts'].append(attempt)
            save()
            sync()
            start = time.perf_counter()
            result = solve()
            sync()
            times.append(time.perf_counter()-start)
            sample = inspect(result)
            sample['seconds'] = times[-1]
            report['samples'].append(sample)
            del result
            attempt['status'] = 'passed' if all(sample['checks'].values()) else 'failed_numerical_check'
            save()
            if attempt['status'] != 'passed':
                raise RuntimeError('measurement numerical checks failed; failed sample retained')
        report['measurement'] = dict(seconds=times, median_seconds=statistics.median(times),
            min_seconds=min(times), max_seconds=max(times))
        report['allocator'].update(peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device)
            if cuda else None, peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device) if cuda else None)
        save()

        if args.profile:
            stage('profiling')
            report['profiler_options'] = dict(record_shapes=True, profile_memory=True, with_stack=False)
            save()
            with torch.profiler.profile(activities=activities, record_shapes=True,
                                       profile_memory=True, with_stack=False) as profiler:
                with torch.profiler.record_function('general.monitor_forward'):
                    result = solve()
                sync()
            stage('exporting')
            profiler.export_chrome_trace(str(args.output_dir/'trace.json'))
            events = profiler.events()
            rows = [dict(name=e.key, count=e.count, input_shapes=e.input_shapes,
                device_type=str(e.device_type), is_user_annotation=bool(e.is_user_annotation),
                cpu_time_total_us=e.cpu_time_total, self_cpu_time_total_us=e.self_cpu_time_total,
                device_time_total_us=e.device_time_total, self_device_time_total_us=e.self_device_time_total,
                cpu_memory_usage_bytes=e.cpu_memory_usage, self_cpu_memory_usage_bytes=e.self_cpu_memory_usage,
                device_memory_usage_bytes=e.device_memory_usage, self_device_memory_usage_bytes=e.self_device_memory_usage)
                for e in profiler.key_averages(group_by_input_shape=True)]
            rows.sort(key=lambda r:r['self_device_time_total_us'] if cuda else r['self_cpu_time_total_us'], reverse=True)
            write_json(args.output_dir/'operators.json', rows)
            counts = Counter()
            for row in rows:
                if row['device_type'].endswith('.CPU') and not row['is_user_annotation']:
                    counts[row['name']] += row['count']
            report['profile'] = dict(activities=[str(a) for a in activities],
                event_devices=dict(Counter(str(e.device_type) for e in events)),
                cpu_operator_counts=dict(sorted(counts.items())), operator_groups=len(rows),
                files=dict(trace='trace.json', operators='operators.json'), numerical=inspect(result))
            del result
            checks = dict(operator_events=bool(rows), **report['profile']['numerical']['checks'])
            if cuda:
                checks['cuda_activity_captured'] = any(str(e.device_type).endswith('.CUDA')
                    and not e.is_user_annotation for e in events)
            report['profile']['checks'] = checks
            save()
            if not all(checks.values()):
                raise RuntimeError('profile checks failed; available trace and operator data retained')
        report.update(status='complete', stage='complete', finished_utc=datetime.now(timezone.utc).isoformat())
        save()
        print(json.dumps(dict(status='complete', output_dir=str(args.output_dir),
            median_seconds=report['measurement']['median_seconds'], profiled=args.profile)))
        return 0
    except BaseException as exc:
        report.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
            finished_utc=datetime.now(timezone.utc).isoformat(),
            error=dict(type=type(exc).__name__, message=str(exc)))
        if report['attempts'] and report['attempts'][-1]['status'] == 'running':
            report['attempts'][-1]['status'] = report['status']
        save()
        print(json.dumps(dict(status=report['status'], stage=report['stage'],
            output_dir=str(args.output_dir), error=report['error'])), file=sys.stderr)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == '__main__':
    raise SystemExit(main())
