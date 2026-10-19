"""Bounded native FDTD operator profiling; profiler times are not benchmark times.

One warm-up and one profiled solve use the cost study's explicit vacuum/CPML
fixture. CUDA records both CPU dispatch and GPU activities. CPU mode is provided
for instrument checks and does not establish CUDA behavior. Generated reports
and the Chrome trace belong in a new, user-selected output directory.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, choices=(32, 64), default=32)
    parser.add_argument('--steps', type=int, default=24)
    parser.add_argument('--checkpoint-steps', type=int, default=8)
    parser.add_argument('--mode', choices=('forward', 'reverse'), default='forward')
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cuda:0')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    # This instrument deliberately bounds trace size and runs exactly once.
    # These are profiling limits, not solver admission restrictions.
    if not 2 <= args.steps <= 128 or not 1 <= args.checkpoint_steps <= 128:
        parser.error('profiling requires 2..128 steps and 1..128 checkpoint steps')
    return args


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    args = arguments()
    # Refusing an existing directory preserves every prior attempt, including
    # failures. The first receipt precedes imports, device checks and allocation.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report_path = args.output_dir / 'attempt.json'
    report = dict(
        status='running', stage='imports',
        started_utc=datetime.now(timezone.utc).isoformat(),
        config=dict(size=args.size, steps=args.steps,
                    checkpoint_steps=args.checkpoint_steps,
                    mode=args.mode, device=args.device),
        scope='one instrumented end-to-end solve; operator attribution, not timing claims',
        warmup_runs=1, profiled_runs=1,
        profiler_options=dict(record_shapes=True, profile_memory=True, with_stack=False),
        limitations=[
            'Shape/memory recording changes cost and tensor lifetimes.',
            'CPU dispatch times and asynchronous device durations are different quantities.',
            'Inclusive operator times overlap and must not be summed as wall time.',
            'CPU aten attribution and CUDA kernel rows describe the same device work; do not add them.',
            'User annotations describe ranges, including gaps, rather than active kernel time.',
            'Allocator peaks exclude driver, display and complete GPU process memory.',
            'The fixture has no spectral monitors, recording or material poles.',
            'Two CPML cells are a cost fixture, not an absorption-accuracy recommendation.'])
    write_json(report_path, report)
    try:
        import numpy as np
        import torch
        import wavebit
        from wavebit import Units
        from wavebit.fdtd import native as n

        torch.set_num_threads(1)
        cuda = args.device.startswith('cuda')
        activities = [torch.profiler.ProfilerActivity.CPU]
        report['environment'] = dict(
            python=platform.python_version(), platform=platform.platform(),
            torch=torch.__version__, cuda_runtime=torch.version.cuda,
            numpy=np.__version__, wavebit=getattr(wavebit, '__version__', None),
            cpu_threads=torch.get_num_threads(), precision='float64/complex128')
        report['stage'] = 'device_admission'
        write_json(report_path, report)
        if cuda:
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA-enabled Torch and an NVIDIA GPU are required')
            if torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
                raise RuntimeError('this Torch profiler does not support CUDA activity collection')
            activities.append(torch.profiler.ProfilerActivity.CUDA)
            props = torch.cuda.get_device_properties(args.device)
            report['environment'].update(gpu=props.name, gpu_total_bytes=props.total_memory)
        report['activities'] = [str(activity) for activity in activities]
        report['stage'] = 'case_construction'
        write_json(report_path, report)

        units = Units(length='um')
        shape = (args.size,) * 3
        current = np.zeros((3,) + shape)
        current[0, args.size // 2, args.size // 2, args.size // 2] = 1e7
        density = torch.tensor(current, dtype=torch.float64, device=args.device)
        problem = n.Problem(
            units=units, medium=n.Medium(epsilon=np.ones(shape)),
            current=n.Current(density=current, samples=np.sin(np.arange(args.steps) * .15)))
        numerics = n.Numerics(
            grid=n.Grid(spacing=(.1,) * 3),
            time_step_seconds=.03 * units.time_unit_seconds, steps=args.steps,
            checkpoint_steps=args.checkpoint_steps,
            absorber=n.PML(cells=2, wavelength=1.))
        reverse = args.mode == 'reverse'

        def synchronize():
            if cuda:
                torch.cuda.synchronize(args.device)

        def solve():
            with torch.set_grad_enabled(reverse):
                with torch.profiler.record_function('native.input_construction'):
                    amplitude = torch.tensor(1.2, dtype=torch.float64, device=args.device,
                                             requires_grad=reverse)
                    live = problem.replace(current=problem.current.replace(density=density * amplitude))
                with torch.profiler.record_function('native.run'):
                    result = n.run(live, numerics=numerics, backend='torch', device=args.device)
                with torch.profiler.record_function('native.objective'):
                    objective = result.E.abs().square().sum()
                gradient = None
                if reverse:
                    with torch.profiler.record_function('native.backward'):
                        gradient = torch.autograd.grad(objective, amplitude)[0]
                return objective, gradient

        report['stage'] = 'warmup'
        write_json(report_path, report)
        solve()
        synchronize()
        if cuda:
            torch.cuda.reset_peak_memory_stats(args.device)
            report['baseline_allocated_bytes'] = torch.cuda.memory_allocated(args.device)
            report['baseline_reserved_bytes'] = torch.cuda.memory_reserved(args.device)
        report['stage'] = 'profiling'
        write_json(report_path, report)
        with torch.profiler.profile(
                activities=activities, record_shapes=True, profile_memory=True,
                with_stack=False) as profiler:
            with torch.profiler.record_function('native.profiled_solve'):
                value, gradient = solve()
            synchronize()
        report['stage'] = 'exporting'
        if cuda:
            report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(args.device)
            report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved(args.device)
        write_json(report_path, report)
        # Export before checking numerical results so a failed solve keeps its
        # available trace and operator evidence rather than discarding them.
        profiler.export_chrome_trace(str(args.output_dir / 'trace.json'))
        events = profiler.events()
        event_devices = Counter(str(event.device_type) for event in events)
        report['event_devices'] = dict(event_devices)
        report['event_types'] = dict(Counter(
            str(event.device_type) + ('/annotation' if event.is_user_annotation else '/activity')
            for event in events))
        rows = []
        for event in profiler.key_averages(group_by_input_shape=True):
            rows.append(dict(
                name=event.key, count=event.count, input_shapes=event.input_shapes,
                device_type=str(event.device_type),
                is_user_annotation=bool(event.is_user_annotation), scope=event.scope,
                cpu_time_total_us=event.cpu_time_total,
                self_cpu_time_total_us=event.self_cpu_time_total,
                device_time_total_us=event.device_time_total,
                self_device_time_total_us=event.self_device_time_total,
                cpu_memory_usage_bytes=event.cpu_memory_usage,
                self_cpu_memory_usage_bytes=event.self_cpu_memory_usage,
                device_memory_usage_bytes=event.device_memory_usage,
                self_device_memory_usage_bytes=event.self_device_memory_usage))
        rows.sort(key=lambda row: row['self_device_time_total_us'] if cuda
                  else row['self_cpu_time_total_us'], reverse=True)
        write_json(args.output_dir / 'operators.json', rows)
        objective_value = float(value.detach().cpu())
        gradient_value = None if gradient is None else float(gradient.detach().cpu())
        checks = dict(finite_objective=bool(np.isfinite(objective_value)),
                      nonzero_objective=objective_value > 0,
                      operator_events=bool(rows))
        if reverse:
            checks['finite_gradient'] = bool(np.isfinite(gradient_value))
            checks['quadratic_source_gradient'] = bool(np.isclose(
                gradient_value, 2 * objective_value / 1.2, rtol=2e-12, atol=0.))
        if cuda:
            checks['cuda_activity_captured'] = any(
                str(event.device_type).endswith('.CUDA') and not event.is_user_annotation
                for event in events)
        report['checks'] = checks
        # JSON remains valid even when the failed solve returned NaN or Inf.
        report['objective'] = objective_value if np.isfinite(objective_value) else str(objective_value)
        report['gradient'] = (gradient_value if gradient_value is None or np.isfinite(gradient_value)
                              else str(gradient_value))
        report['files'] = dict(trace='trace.json', operators='operators.json')
        report['operator_groups'] = len(rows)
        report['stage'] = 'validation'
        write_json(report_path, report)
        if not all(checks.values()):
            raise RuntimeError('profile checks failed; trace, operator data and checks were retained')
        report.update(status='complete', stage='complete',
                      finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(report_path, report)
        print(json.dumps(dict(status='complete', output_dir=str(args.output_dir), checks=checks)))
        return 0
    except BaseException as exc:
        report.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                      finished_utc=datetime.now(timezone.utc).isoformat(),
                      error=dict(type=type(exc).__name__, message=str(exc)))
        write_json(report_path, report)
        print(json.dumps(dict(status=report['status'], stage=report['stage'],
                              output_dir=str(args.output_dir), error=report['error'])), file=sys.stderr)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == '__main__':
    raise SystemExit(main())
