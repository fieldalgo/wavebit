"""General CUDA reverse cost and allocator peaks versus checkpoint interval.

A uniform conductor supplies an independent SI recurrence and exact amplitude
derivative. This bounded fixture measures storage policy, not optical accuracy.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import torch
from wavebit import Units, fdtd
from wavebit.materials import ConductiveMaterial
from wavebit.units import EPSILON0


# Authentic 32^3/163-step conductivity-gradient reserved peaks, in bytes.
# Scale by cell count, then add 50%; this is a forecast, not an allocation cap.
_RESERVED_32 = {None: 350224384, 8: 69206016, 32: 88080384}
_GIB = 1024**3


def memory_forecast(size, block):
    if size != 64:
        return None
    observed = _RESERVED_32[block]
    return dict(reference_size=32, reference_steps=163,
                reference_parameters='material', reference_peak_reserved_bytes=observed,
                cell_multiplier=8, margin_multiplier=1.5,
                forecast_reserved_bytes=observed * 8 * 3 // 2,
                forecast_budget_bytes=4 * _GIB, required_headroom_bytes=_GIB)


def measure(size, steps, block, repeat, parameters='amplitude', record_preflight=None):
    units = Units(length='um')
    problem = fdtd.Problem(units=units, scene=fdtd.Scene(
        domain=fdtd.Domain(size=(size*.05,)*3),
        background=ConductiveMaterial(epsilon=2.25, conductivity=600.),
        boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.Periodic())),
        initial=fdtd.InitialField(E=(0., 1., 0.)),
        outputs=fdtd.Outputs(monitors=(fdtd.TimeSeriesMonitor('probe', at=(0., 0., 0.), component='Ey'),),
                             final_fields=True))
    numerics = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.05),
        duration=fdtd.Duration(steps=steps), checkpoint_steps=block)
    prepared = fdtd.prepare(problem, numerics=numerics, backend='torch', device='cuda:0')
    dt = prepared.report.resolved['dt_seconds']
    half = 600.*dt/(2*EPSILON0*2.25)
    decay = (1-half)/(1+half)
    expected = size**3*1.2**2*decay**(2*steps)
    factor = dt/(2*EPSILON0*2.25)
    expected_material = size**3*1.2**2*2*steps*decay**(2*steps-1)*(-2*factor/(1+600.*factor)**2)

    def solve():
        amplitude = torch.tensor(1.2, dtype=torch.float64, device='cuda:0', requires_grad=True)
        live = problem.replace(initial=problem.initial.replace(amplitude=amplitude))
        leaves = (amplitude,)
        if parameters == 'material':
            sigma = torch.tensor(600., dtype=torch.float64, device='cuda:0', requires_grad=True)
            live = live.replace(scene=live.scene.replace(background=ConductiveMaterial(
                epsilon=2.25, conductivity=sigma)))
            leaves += (sigma,)
        result = prepared.run(live)
        value = result.final_fields.values['E'].square().sum()
        gradients = torch.autograd.grad(value, leaves)
        return value, gradients

    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    prewarmup = dict(free_bytes=free, total_bytes=total,
                    allocated_bytes=torch.cuda.memory_allocated(),
                    reserved_bytes=torch.cuda.memory_reserved(),
                    forecast=memory_forecast(size, block))
    if record_preflight is not None:
        record_preflight(prewarmup)
    forecast = prewarmup['forecast']
    if forecast is not None:
        required = forecast['forecast_reserved_bytes']
        if required > forecast['forecast_budget_bytes']:
            raise RuntimeError('reserved-memory forecast exceeds the 4 GiB study budget')
        if free < required + forecast['required_headroom_bytes']:
            raise RuntimeError('insufficient free VRAM for forecast plus 1 GiB headroom')
    solve()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    times, values, gradients, material_gradients = [], [], [], []
    post_repetition_allocated = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = time.perf_counter()
        value, derivatives = solve()
        torch.cuda.synchronize()
        times.append(time.perf_counter()-start)
        values.append(float(value.detach().cpu()))
        gradients.append(float(derivatives[0].detach().cpu()))
        if parameters == 'material':
            material_gradients.append(float(derivatives[1].detach().cpu()))
        del value, derivatives
        post_repetition_allocated.append(torch.cuda.memory_allocated())
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    post_cleanup_allocated = torch.cuda.memory_allocated()
    post_cleanup_reserved = torch.cuda.memory_reserved()
    checks = dict(finite=bool(np.isfinite(values).all() and np.isfinite(gradients).all()),
                  si_recurrence=bool(np.allclose(values, expected, rtol=2e-12, atol=0.)),
                  analytic_gradient=bool(np.allclose(gradients, 2*expected/1.2, rtol=2e-12, atol=0.)))
    if parameters == 'material':
        checks['analytic_material_gradient'] = bool(np.allclose(
            material_gradients, expected_material, rtol=2e-12, atol=0.))
    return dict(size=size, steps=steps, checkpoint_steps=block, parameters=parameters,
        checkpoint_blocks=0 if block is None else steps//block,
        tail_steps=steps if block is None else steps%block,
        seconds=times, median_seconds=statistics.median(times),
        min_seconds=min(times), max_seconds=max(times),
        values=values, gradients=gradients, analytic_value=expected,
        material_gradients=material_gradients, analytic_material_gradient=expected_material,
        checks=checks,
        prewarmup=prewarmup,
        baseline_allocated_bytes=baseline, baseline_reserved_bytes=baseline_reserved,
        peak_allocated_bytes=peak_allocated, peak_reserved_bytes=peak_reserved,
        incremental_peak_allocated_bytes=peak_allocated-baseline,
        incremental_peak_reserved_bytes=peak_reserved-baseline_reserved,
        post_repetition_allocated_bytes=post_repetition_allocated,
        post_cleanup_allocated_bytes=post_cleanup_allocated,
        post_cleanup_reserved_bytes=post_cleanup_reserved)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', nargs='+', type=int, default=[16, 32])
    parser.add_argument('--steps', type=int, default=163)
    parser.add_argument('--blocks', nargs='+', type=int, default=[0, 8, 32, 200],
                        help='zero selects the unchanged plain scan')
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--parameters', nargs='+', choices=('amplitude','material'),
                        default=['amplitude','material'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    legacy = all(4 <= size <= 32 for size in args.sizes) and 1 <= args.steps <= 512
    extended = (args.sizes == [64] and args.steps == 163
                and args.parameters == ['material'] and len(args.blocks) == 1
                and args.blocks[0] in (0, 8, 32))
    if not (legacy or extended):
        parser.error('requires sizes 4..32 and steps 1..512, or size 64 / steps 163 / '
                     'parameters material / exactly one block from 0, 8, 32 per process')
    if min(args.blocks) < 0 or args.repeat < 1:
        parser.error('blocks must be nonnegative; repeat must be positive')
    if args.output.exists():
        parser.error('output already exists; preserve it and choose a new path')
    return args


def main():
    from wavebit.fdtd import _api, _execution, _runtime
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status='running', python=platform.python_version(),
        platform=platform.platform(), torch=torch.__version__, cuda=torch.version.cuda,
        numpy=np.__version__, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256={name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                       for name, module in (('api', _api), ('execution', _execution), ('runtime', _runtime))},
        scope='prepared general CUDA forward plus amplitude/optional conductivity reverse; uniform conductor with one point record',
        timing='live leaf creation, prepared run, objective and backward; excludes preparation and host scalar copies',
        warmup_runs=1, repeat=args.repeat,
        allocator_scope='Torch allocator only; excludes driver/display and other processes',
        forecast_scope='64-cubed only: 8x observed 32-cubed/163-step reserved peak, '
                       'with 1.5x margin and 1 GiB free headroom; estimate, not a guarantee',
        rows=[], attempts=[])

    def save():
        args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')

    save()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA-enabled Torch and an NVIDIA GPU are required')
        torch.set_num_threads(1)
        report.update(gpu=torch.cuda.get_device_name(0),
                      gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory)
        configurations = ((size, block, parameters) for parameters in args.parameters
                          for size in args.sizes for block in args.blocks)
        for size, block, parameters in configurations:
            attempt = dict(size=size, steps=args.steps, checkpoint_steps=block or None,
                           parameters=parameters, status='running')
            report['attempts'].append(attempt)
            save()
            def record_preflight(info):
                attempt['prewarmup'] = info
                save()

            row = measure(size, args.steps, block or None, args.repeat, parameters,
                          record_preflight=record_preflight)
            report['rows'].append(row)
            attempt['status'] = 'passed' if all(row['checks'].values()) else 'failed_numerical_check'
            save()
            print(json.dumps(row), flush=True)
            if not all(row['checks'].values()):
                raise RuntimeError('numerical check failed; failed row retained')
        report['status'] = 'complete'
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        if report['attempts']:
            report['attempts'][-1]['status'] = 'failed'
        save()
        raise
    save()


if __name__ == '__main__':
    main()
