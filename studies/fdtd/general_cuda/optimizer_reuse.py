"""Bounded same-leaf SGD reuse, independent basis derivatives and CUDA lifetimes.

The NumPy basis is a derivative oracle for the same discrete solver, not an
independent Maxwell/CPML accuracy reference. Instrumented wall times are not
benchmark comparisons. Torch is optional until a run is requested.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np

STEPS = 41
UPDATES = 16
RATE = .125
TARGET = (1.1, .2)
SEEDS = ((.8, -.3), (1.4, .7))
RTOL = 2e-12
GIB = 2**30


def problem(amplitude=1., phase=0.):
    from wavebit import Material, Units, fdtd
    units = Units(length='um')
    pulse = fdtd.GaussianPulse(wavelength=2., frequency_sigma_hz=.08/units.time_unit_seconds)
    peak = np.sqrt(2*np.log(1/pulse.cutoff))/(2*np.pi*pulse.frequency_sigma_hz)
    return fdtd.Problem(units=units, scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.6,)*3), background=Material(epsilon=1.),
        boundaries=fdtd.Boundaries(x=fdtd.Open(), y=fdtd.Open(), z=fdtd.Open())),
        sources=(fdtd.CurrentDipole('d', center=(-.031, .017, -.10), component='Ey',
            pulse=pulse, current_moment=1e-16*amplitude, phase=phase,
            delay_seconds=-peak+.12*units.time_unit_seconds),), outputs=fdtd.Outputs(final_fields=True))


def numerics(block=None):
    from wavebit import fdtd
    return fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.05),
        duration=fdtd.Duration(steps=STEPS), checkpoint_steps=block,
        absorber=fdtd.PML(cells=4, wavelength=2., allow_thin=True))


def fields(result):
    from wavebit.units import Z0
    return result.final_fields.values['E'], Z0*result.final_fields.values['H']


def oracle(theta, gram):
    amplitude, phase = theta
    u = np.array([np.cos(phase), np.sin(phase)])
    du = np.array([-np.sin(phase), np.cos(phase)])
    target = TARGET[0]*np.array([np.cos(TARGET[1]), np.sin(TARGET[1])])
    residual = amplitude*u-target
    gradient = np.array([u@gram@residual, amplitude*du@gram@residual])
    return .5*residual@gram@residual, gradient


def trajectory(seed, gram):
    theta = np.asarray(SEEDS[seed], dtype=np.float64)
    rows = []
    for index in range(UPDATES):
        value, gradient = oracle(theta, gram)
        after = theta-RATE*gradient
        if not (np.isfinite(value) and value > 0 and np.isfinite(gradient).all()
                and np.all(gradient != 0) and np.isfinite(after).all() and np.all(after != 0)):
            raise ValueError('fixed oracle trajectory has a nonfinite or zero relative-comparison quantity')
        rows.append(dict(iteration=index, before=theta.tolist(), value=float(value),
                         gradient=gradient.tolist(), after=after.tolist()))
        theta = after
    return rows


def source_files(root):
    paths = set((root/'src/wavebit').rglob('*.py'))
    paths.update((root/'studies/fdtd/general_cuda/optimizer_reuse.py',
                  root/'tests/fdtd/test_general_cuda_optimizer.py'))
    return {p.relative_to(root).as_posix(): p for p in sorted(paths)}


def identity(root):
    return {name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in source_files(root).items()}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def exception_record(error):
    return dict(type=type(error).__name__, message=str(error),
                traceback=''.join(traceback.format_exception(type(error), error, error.__traceback__)))


def begin(root, output, config):
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError('raw study output must be outside the product checkout')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', config=config, source_before=None, source_audit_complete=False,
                  environment=dict(python=platform.python_version(), numpy=np.__version__,
                      executable=sys.executable, platform=platform.platform(), machine=platform.machine(),
                      cpu_thread_environment={key: os.environ.get(key) for key in
                          ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}),
                  command=[sys.executable, *sys.argv], audit_errors=[], inputs_unchanged=None)
    # An attempt exists before any source read/copy can fail. Never overwrite an
    # existing attempt, including a previous incomplete or failed one.
    write_json(output/'receipt.json', report)
    try:
        before = identity(root)
        report['source_before'] = before
        for name, path in source_files(root).items():
            destination = output/'source'/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
            if hashlib.sha256(destination.read_bytes()).hexdigest() != before[name]:
                raise RuntimeError('source changed while its exact bytes were being saved')
        report['source_audit_complete'] = True
        write_json(output/'receipt.json', report)
        return report
    except BaseException as error:
        report.update(status='failed', inputs_unchanged=False, failure=exception_record(error))
        report['audit_errors'].append(dict(phase='initial_source_snapshot', **exception_record(error)))
        try:
            write_json(output/'receipt.json', report)
        except BaseException as receipt_error:
            error.add_note('Writing the failed receipt also failed: '+repr(receipt_error))
        raise


def finish(root, output, report, error=None):
    audit_errors = report.setdefault('audit_errors', [])
    first_audit_error = None
    unchanged = report.get('source_audit_complete', False)
    try:
        after = identity(root)
        report['source_after'] = after
        unchanged = unchanged and report['source_before'] == after
    except BaseException as audit_error:
        first_audit_error = audit_error
        unchanged = False
        audit_errors.append(dict(phase='final_source_identity', **exception_record(audit_error)))
    if 'reference' in report:
        reference = Path(report['reference']['receipt'])
        try:
            report['reference_after'] = dict(
                receipt_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                basis_sha256=hashlib.sha256((reference.parent/'basis.npz').read_bytes()).hexdigest())
            report['reference_unchanged'] = all(
                report['reference'][key] == value for key, value in report['reference_after'].items())
        except BaseException as audit_error:
            if first_audit_error is None:
                first_audit_error = audit_error
            report['reference_unchanged'] = False
            audit_errors.append(dict(phase='final_reference_identity', **exception_record(audit_error)))
        unchanged = unchanged and report['reference_unchanged']
    report['inputs_unchanged'] = bool(unchanged)
    failure = error if error is not None else first_audit_error
    if failure is None and not unchanged:
        failure = RuntimeError('source changed or reference artifact changed during execution; receipt rejected')
    report['status'] = 'passed' if failure is None else 'failed'
    if failure is not None:
        report['failure'] = exception_record(failure)
    try:
        write_json(output/'receipt.json', report)
    except BaseException as receipt_error:
        # The original solver/audit error remains primary even if the filesystem
        # cannot persist its receipt. A caller's except block re-raises `error`.
        if failure is not None:
            failure.add_note('Writing the final receipt also failed: '+repr(receipt_error))
            if error is not None:
                return
            raise failure from receipt_error
        raise
    if error is None and failure is not None:
        raise failure


def finite_refusal(error):
    """An unrelated structured refusal is not evidence that NaN was detected."""
    import re
    from wavebit.errors import ValidationError
    if not isinstance(error, ValidationError) or not re.search(
            r'finite|\bnan\b', str(error)+' '+error.code, re.IGNORECASE):
        raise AssertionError('invalid source refusal must identify nonfinite input') from error
    return dict(kind='structured_refusal', code=error.issue.code, path=error.issue.path,
                message=error.issue.message)


def check_import(root):
    import wavebit
    if Path(wavebit.__file__).resolve() != (root/'src/wavebit/__init__.py').resolve():
        raise ValueError('imported Wavebit is not the requested source checkout')
    return str(Path(wavebit.__file__).resolve())


def build_basis(root, output):
    report = begin(root, output, dict(mode='basis', steps=STEPS))
    try:
        report['environment']['wavebit_import'] = check_import(root)
        from wavebit import fdtd
        basis = [tuple(np.array(x, copy=True) for x in fields(
            fdtd.run(problem(phase=phase), numerics=numerics()))) for phase in (0., np.pi/2)]
        raw = np.array([[sum(np.sum(a*b) for a, b in zip(left, right))
                         for right in basis] for left in basis])
        scale = float(np.trace(raw))
        gram = raw/scale
        eigenvalues = np.linalg.eigvalsh(gram)
        if not (np.isfinite(gram).all() and scale > 0 and np.all(eigenvalues > 0)):
            raise ValueError('basis Gram matrix is not finite positive definite')
        report.update(gram=gram.tolist(), scale=scale, eigenvalues=eigenvalues.tolist(),
                      condition_number=float(eigenvalues[-1]/eigenvalues[0]),
                      trajectories=[trajectory(i, gram) for i in range(2)])
        np.savez(output/'basis.npz', e0=basis[0][0], h0=basis[0][1],
                 e1=basis[1][0], h1=basis[1][1], gram=gram, scale=scale)
        report['basis_sha256'] = hashlib.sha256((output/'basis.npz').read_bytes()).hexdigest()
    except BaseException as error:
        finish(root, output, report, error)
        raise
    finish(root, output, report)
    return output/'receipt.json'


def read_basis(root, reference):
    receipt = json.loads(reference.read_text(encoding='utf-8'))
    if receipt['status'] != 'passed' or receipt['inputs_unchanged'] is not True:
        raise ValueError('basis receipt was not a successful unchanged-source attempt')
    if receipt['source_before'] != identity(root):
        raise ValueError('basis source differs from this run; build a new basis explicitly')
    path = reference.parent/'basis.npz'
    if hashlib.sha256(path.read_bytes()).hexdigest() != receipt['basis_sha256']:
        raise ValueError('basis artifact bytes changed')
    with np.load(path, allow_pickle=False) as values:
        basis = tuple((values['e'+str(i)].copy(), values['h'+str(i)].copy()) for i in (0, 1))
        return basis, values['gram'].copy(), float(values['scale'])


def _cycle(torch, device, block, seed, basis, gram, scale, report, save):
    from wavebit import fdtd
    from wavebit.errors import ValidationError
    cuda = device.startswith('cuda')

    def memory():
        if not cuda:
            return None
        torch.cuda.synchronize(device)
        free, total = torch.cuda.mem_get_info(device)
        return dict(allocated=torch.cuda.memory_allocated(device),
                    reserved=torch.cuda.memory_reserved(device), free=free, total=total,
                    peak_allocated=torch.cuda.max_memory_allocated(device),
                    peak_reserved=torch.cuda.max_memory_reserved(device))

    def clean():
        gc.collect()
        return memory()

    u_target = TARGET[0]*np.array([np.cos(TARGET[1]), np.sin(TARGET[1])])
    target = tuple(torch.tensor(u_target[0]*a+u_target[1]*b, dtype=torch.float64, device=device)
                   for a, b in zip(*basis))
    leaves = (torch.tensor(SEEDS[seed][0], dtype=torch.float64, device='cpu', requires_grad=True),
              torch.tensor(SEEDS[seed][1], dtype=torch.float64, device=device, requires_grad=True))
    initial_identity = tuple((id(x), x.data_ptr(), x._version, str(x.dtype), str(x.device)) for x in leaves)
    optimizer = torch.optim.SGD(leaves, lr=RATE, foreach=False)
    prepared = fdtd.prepare(problem(), numerics=numerics(block), backend='torch', device=device)
    report['grid_shape'] = list(prepared.report.resolved['grid_shape'])
    assert report['grid_shape'] == [20, 20, 20]
    assert prepared.report.resolved['steps'] == STEPS
    assert prepared.contract['resolved_checkpoint_steps'] == block
    if block is not None:
        assert STEPS % block == 6
    report['persistent_payload'] = dict(
        target_bytes=sum(x.numel()*x.element_size() for x in target),
        leaf_devices=[str(x.device) for x in leaves], leaf_dtypes=[str(x.dtype) for x in leaves],
        leaf_bytes=[x.numel()*x.element_size() for x in leaves])
    report['persistent_baseline'] = clean()
    reference_rows = trajectory(seed, gram)
    report['memory_peak_scope'] = (
        'Cumulative since pre-setup reset through iteration 1; cumulative since the reset '
        'after iteration 1 thereafter. Peaks are not per-iteration allocations.')
    report['iterations'] = []
    report['negative_recovery'] = []
    old = None
    snapshots = None
    plateau = None

    def host(value):
        return value.detach().cpu().numpy().copy()

    def check_fields(result, theta):
        amplitude, phase = theta
        actual = fields(result)
        for got, a, b in zip(actual, *basis):
            expected = amplitude*(np.cos(phase)*a+np.sin(phase)*b)
            received = host(got)
            peak = np.max(np.abs(expected))
            if not np.isfinite(received).all() or not peak > 0:
                raise AssertionError('nonfinite or empty field comparison')
            assert np.max(np.abs(received-expected))/peak <= RTOL

    def advance(index, expected):
        optimizer.zero_grad(set_to_none=True)
        assert all(x.grad is None for x in leaves)
        before = [float(x.detach().cpu()) for x in leaves]
        np.testing.assert_allclose(before, expected['before'], rtol=RTOL, atol=0.)
        result = prepared.run(problem(*leaves))
        assert bool(result.validity.checks['runtime_stability'])
        assert result.provenance['actual_steps'] == STEPS
        assert result.provenance['checkpoint_steps'] == block
        check_fields(result, before)
        loss = sum(((value-want)**2).sum() for value, want in zip(fields(result), target))/(2*scale)
        after_forward = memory()
        value = float(loss.detach().cpu())
        np.testing.assert_allclose(value, expected['value'], rtol=RTOL, atol=0.)
        loss.backward()
        after_backward = memory()
        gradient = [float(x.grad.detach().cpu()) for x in leaves]
        np.testing.assert_allclose(gradient, expected['gradient'], rtol=RTOL, atol=0.)
        for leaf in leaves:
            assert leaf.grad.dtype == leaf.dtype and leaf.grad.device == leaf.device
        optimizer.step()
        after = [float(x.detach().cpu()) for x in leaves]
        np.testing.assert_allclose(after, expected['after'], rtol=RTOL, atol=0.)
        optimizer.zero_grad(set_to_none=True)
        for leaf, original in zip(leaves, initial_identity):
            assert (id(leaf), leaf.data_ptr(), str(leaf.dtype), str(leaf.device)) == (
                original[0], original[1], original[3], original[4])
            assert leaf.is_leaf and leaf.requires_grad and leaf.grad is None
            assert leaf._version == original[2]+index+1
        return result, dict(iteration=index, before=before, value=value, gradient=gradient, after=after,
                            versions=[x._version for x in leaves], after_forward=after_forward,
                            after_backward=after_backward)

    started = time.perf_counter()
    for index, expected in enumerate(reference_rows):
        result, row = advance(index, expected)
        if index == 0:
            old = result
            snapshots = tuple(host(x) for x in fields(result))
        elif index == 1:
            for previous, snapshot in zip(fields(old), snapshots):
                np.testing.assert_array_equal(host(previous), snapshot)
            del previous, snapshot
            old = snapshots = None
            row['old_outputs_unchanged'] = True
        del result
        row['after_release'] = clean()
        if cuda and index == 1:
            plateau = row['after_release']['allocated']
            report['plateau_baseline'] = plateau
            # There are no persistent device buffers in the Torch runtime for this
            # fixture. Do not excuse an unexplained offset as allocator caching.
            assert plateau == report['persistent_baseline']['allocated']
            torch.cuda.reset_peak_memory_stats(device)
        if cuda and index >= 1:
            assert row['after_release']['allocated'] == plateau
        report['iterations'].append(row)
        save()
    report['instrumented_loop_seconds'] = time.perf_counter()-started
    final = reference_rows[-1]['after']
    for parameter, name in enumerate(('amplitude', 'phase')):
        with torch.no_grad():
            leaves[parameter].fill_(float('nan'))
        invalid = None
        try:
            try:
                invalid = prepared.run(problem(*leaves))
            except ValidationError as error:
                verdict = finite_refusal(error)
            else:
                invalid_fields = fields(invalid)
                nonfinite = any(not bool(torch.isfinite(x).all()) for x in invalid_fields)
                stable = bool(invalid.validity.checks['runtime_stability'])
                if stable or not nonfinite:
                    raise AssertionError('invalid source must refuse or return nonfinite fields with failed validity')
                verdict = dict(kind='failed_runtime_validity', nonfinite=True)
                del invalid_fields
        finally:
            del invalid
            with torch.no_grad():
                leaves[parameter].fill_(final[parameter])
        recovered = prepared.run(problem(*leaves))
        assert bool(recovered.validity.checks['runtime_stability'])
        check_fields(recovered, final)
        del recovered
        released = clean()
        if cuda:
            assert released['allocated'] == plateau
        report['negative_recovery'].append(dict(parameter=name, verdict=verdict,
                                                restored=final, after_release=released))
        save()
    report['leaf_identity_preserved'] = True
    report['versions_after_recovery'] = [x._version for x in leaves]
    assert all(x.grad is None for x in leaves)
    optimizer = prepared = leaves = target = None
    gc.collect()


def run(root, output, reference, device='cpu', block=None, seed=0):
    if device not in ('cpu', 'cuda:0') or block not in (None, 7) or seed not in (0, 1):
        raise ValueError('bounded run requires cpu/cuda:0, plain/K7 and seed0/1')
    if device == 'cpu' and block is not None:
        raise ValueError('the public checkpoint route requires CUDA')
    report = begin(root, output, dict(mode='run', device=device, checkpoint_steps=block,
        seed=seed, updates=UPDATES, steps=STEPS, learning_rate=RATE, target=list(TARGET)))
    try:
        report['environment']['wavebit_import'] = check_import(root)
        import torch
        torch.set_num_threads(1)
        basis, gram, scale = read_basis(root, reference)
        cuda = device.startswith('cuda')
        report['environment'].update(torch=torch.__version__, cuda_runtime=torch.version.cuda,
            torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads())
        report['reference'] = dict(receipt=str(reference.resolve()),
            receipt_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
            basis_sha256=hashlib.sha256((reference.parent/'basis.npz').read_bytes()).hexdigest())
        if cuda:
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA is unavailable; CPU substitution is forbidden')
            free, total = torch.cuda.mem_get_info(device)
            if free < 5*GIB or total < 4*GIB:
                raise RuntimeError('bounded study requires 5 GiB free and a 4 GiB allocator budget')
            torch.cuda.set_per_process_memory_fraction(4*GIB/total, device)
            report['environment'].update(gpu=torch.cuda.get_device_name(device),
                                          free_before=free, total_bytes=total, allocator_cap_bytes=4*GIB)
            gc.collect()
            torch.cuda.synchronize(device)
            report['process_baseline_allocated'] = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        _cycle(torch, device, block, seed, basis, gram, scale, report,
               lambda: write_json(output/'receipt.json', report))
        gc.collect()
        if cuda:
            torch.cuda.synchronize(device)
            allocated = torch.cuda.memory_allocated(device)
            report['final_cleanup'] = dict(allocated=allocated, reserved=torch.cuda.memory_reserved(device))
            assert allocated == report['process_baseline_allocated']
            torch.cuda.empty_cache()
            report['after_empty_cache'] = dict(allocated=torch.cuda.memory_allocated(device),
                reserved=torch.cuda.memory_reserved(device), free=torch.cuda.mem_get_info(device)[0])
    except BaseException as error:
        finish(root, output, report, error)
        raise
    finish(root, output, report)
    return report


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('basis', 'run'), required=True)
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cpu')
    parser.add_argument('--block', type=int, choices=(0, 7), default=0)
    parser.add_argument('--seed', type=int, choices=(0, 1), default=0)
    args = parser.parse_args(argv)
    if args.output_dir.resolve().is_relative_to(args.source_root.resolve()):
        parser.error('raw output directory must be outside the product checkout')
    if args.mode == 'run' and args.reference is None:
        parser.error('--reference is required for run mode')
    if args.mode == 'basis' and (args.reference is not None or args.device != 'cpu' or args.block != 0):
        parser.error('basis mode is CPU/ordinary and creates its own reference')
    if args.device == 'cpu' and args.block != 0:
        parser.error('public checkpoint execution requires CUDA')
    return args


def main(argv=None):
    args = arguments(argv)
    root = args.source_root.resolve()
    output = args.output_dir.resolve()
    if args.mode == 'basis':
        result = build_basis(root, output)
        print(json.dumps(dict(status='passed', reference=str(result))), flush=True)
    else:
        result = run(root, output, args.reference.resolve(), args.device, args.block or None, args.seed)
        print(json.dumps(dict(status=result['status'], receipt=str(output/'receipt.json'))), flush=True)


if __name__ == '__main__':
    main()
