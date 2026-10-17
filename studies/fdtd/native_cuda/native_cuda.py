"""Synchronized native FDTD CPU/CUDA timing and CUDA allocator peaks."""
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
from wavebit import Units
from wavebit.fdtd import native as n


def measure(size, steps, device, reverse, repeat, checkpoint_steps=8):
    units = Units(length='um')
    shape = (size,)*3
    current = np.zeros((3,)+shape)
    current[0, size//2, size//2, size//2] = 1e7
    density = torch.tensor(current, dtype=torch.float64, device=device)
    problem = n.Problem(units=units, medium=n.Medium(epsilon=np.ones(shape)),
        current=n.Current(density=current, samples=np.sin(np.arange(steps)*.15)))
    nums = n.Numerics(grid=n.Grid(spacing=(.1,)*3),
        time_step_seconds=.03*units.time_seconds, steps=steps,
        checkpoint_steps=checkpoint_steps, absorber=n.PML(cells=2, wavelength=1.))
    def run():
        amplitude = torch.tensor(1.2, dtype=torch.float64, device=device, requires_grad=reverse)
        p = problem.replace(current=problem.current.replace(density=density*amplitude))
        result = n.run(p, numerics=nums, backend='torch', device=device)
        value = result.E.abs().square().sum()
        gradient = torch.autograd.grad(value, amplitude)[0] if reverse else None
        return value, gradient
    cuda = device.startswith('cuda')
    def sync():
        if cuda:
            torch.cuda.synchronize(device)
    with torch.set_grad_enabled(reverse):
        run()
    sync()
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device) if cuda else None
    times, values, gradients = [], [], []
    for _ in range(repeat):
        sync()
        start = time.perf_counter()
        with torch.set_grad_enabled(reverse):
            value, gradient = run()
        sync()
        times.append(time.perf_counter()-start)
        result = float(value.detach().cpu())
        derivative = None if gradient is None else float(gradient.detach().cpu())
        values.append(result)
        gradients.append(derivative)
        del value, gradient
    return dict(size=size, steps=steps, device=device, reverse=reverse,
        checkpoint_steps=checkpoint_steps,
        seconds=times, median_seconds=statistics.median(times), value=result,
        gradient=derivative, values=values, gradients=gradients,
        baseline_allocated_bytes=baseline,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if cuda else None,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if cuda else None)


def main():
    from wavebit.fdtd._periodic import grid, runtime
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', type=int, nargs='+', default=[8, 16, 32])
    parser.add_argument('--steps', type=int, default=24)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--checkpoint-steps', type=int, default=8)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.sizes) < 8 or min(args.steps, args.repeat, args.checkpoint_steps) < 1:
        parser.error('sizes must be >=8; steps, repeat and checkpoint-steps must be positive')
    if not torch.cuda.is_available():
        parser.error('CUDA-enabled Torch and an NVIDIA GPU are required')
    torch.set_num_threads(1)
    report = dict(python=platform.python_version(), platform=platform.system(),
        platform_version=platform.version(), processor=platform.processor(),
        torch=torch.__version__, cuda=torch.version.cuda, numpy=np.__version__,
        gpu=torch.cuda.get_device_name(0), gpu_index=0,
        gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        cpu_threads=torch.get_num_threads(), warmup_runs=1,
        precision='float64/complex128', checkpoint_steps=args.checkpoint_steps,
        forward_grad_enabled=False,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256={name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                       for name, module in (('native', n), ('grid', grid), ('runtime', runtime))},
        rows=[], attempts=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for size in args.sizes:
        for reverse in (False, True):
            for device in ('cpu', 'cuda:0'):
                attempt = dict(size=size, steps=args.steps, device=device, reverse=reverse,
                    checkpoint_steps=args.checkpoint_steps, status='running')
                report['attempts'].append(attempt)
                args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
                try:
                    row = measure(size, args.steps, device, reverse, args.repeat, args.checkpoint_steps)
                except Exception as exc:
                    attempt.update(status='failed', error_type=type(exc).__name__, error=str(exc))
                    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
                    raise
                report['rows'].append(row)
                checks = {'finite': bool(np.isfinite(row['values']).all())}
                if reverse:
                    checks['quadratic_gradient'] = bool(np.allclose(
                        row['gradients'], np.array(row['values'])*2/1.2, rtol=2e-12, atol=0))
                if device.startswith('cuda'):
                    cpu = report['rows'][-2]
                    checks['cpu_gpu_value'] = bool(np.allclose(
                        row['values'], cpu['values'], rtol=2e-12, atol=0))
                    if reverse:
                        checks['cpu_gpu_gradient'] = bool(np.allclose(
                            row['gradients'], cpu['gradients'], rtol=2e-12, atol=0))
                row['checks'] = checks
                attempt['status'] = 'passed' if all(checks.values()) else 'failed_numerical_check'
                args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
                print(json.dumps(row), flush=True)
                if not all(checks.values()):
                    raise RuntimeError('numerical check failed; the output retains the failed row')


if __name__ == '__main__':
    main()
