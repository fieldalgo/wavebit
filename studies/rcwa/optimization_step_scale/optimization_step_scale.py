"""
optimization_step_scale.py -- one optimization step at the maintainer's real scale, per backend.

Does ``jax.jit`` still pay for a differentiated RCWA step once the step is dominated by LAPACK?
The earlier measurements (docs/validation/rcwa/backend-choice.md, the lifecycle study beside
this one) stopped at ``orders=(13, 13)`` on the retired interface and ``(9, 9)`` on the current
one. This one times the workload a metasurface optimization actually runs: ``orders`` (15, 15)
and (20, 20), one and two patterned layers, through ``rcwa.prepare`` / ``prepared.run``.

One step is a forward run and the gradient of the zeroth-order transmittance with respect to the
design variables. The arms, all on the same machine, process and thread count:

    torch          eager PyTorch autograd
    torch-compile  torch.compile around the loss, then .backward(); wavebit's public callables
                   are torch.compiler.disable doors, so this measures the doors (expect ~1x)
    jax-grad       eager jax.grad, no jit
    jax-jit        jax.jit(jax.value_and_grad(loss)), compiled once on the first call and reused

Two design paths, run in the same process sweep:

    shape    one Circle per patterned layer, its radius the design variable (one per layer);
             default modeling (NVF, default normal-field grid) with topology_check='host', the
             policy that admits a traced circle under jit. **The radius moves every step**: the
             normal field and Fourier coefficients are cached on the shape parameters, so a
             fixed shape would time a cache hit and make the modeling half look free.
    raster   the control: a fixed binary 128 x 128 pattern per layer, the permittivity array the
             design variable. Its high value moves every step (the phase partition the
             prepared guard checks stays the same).

Each step is split in two by a second, modeling-only loss timed the same way: the gradient of a
fixed linear functional of the permittivity convolution matrices, ``_eps_conv_matrices`` of the
engine scene ``prepared.run`` would solve. That is the shape's Fourier transform and normal
field, forward and backward; ``solve`` is the full step minus it (eig, S-matrix, their backward).
The modeling functional reaches into the engine through private names, which is why this is a
study and not a test.

Arms are interleaved: every round runs one step of each arm, full and modeling-only, at the
same parameter value, and the parameter moves between rounds. The first round is reported apart
(compilation, first-call caches); the median and minimum come from the remaining rounds. Peak
memory is the largest resident set sampled (every 2 ms, from /proc/self/statm) during an arm's
calls, minus the resident set just before the call -- the step's own working set on top of
whatever the process already held -- with the process's whole-run ``ru_maxrss`` beside it.

One configuration per process, so allocator pools and JAX compilations are freed between them:

    python studies/rcwa/optimization_step_scale/optimization_step_scale.py --orders 15 --layers 1 --path shape
    python studies/rcwa/optimization_step_scale/optimization_step_scale.py --sweep    # every configuration, in turn

``--sweep`` alternates shape and raster at each (orders, layers), runs each configuration in a
child process with the same thread pinning, and appends JSON lines to ``--out``.
"""
import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--orders', type=int, default=15)
    ap.add_argument('--layers', type=int, default=1)
    ap.add_argument('--path', choices=('shape', 'raster'), default='shape')
    ap.add_argument('--rounds', type=int, default=4, help='rounds after the first (each moves the design)')
    ap.add_argument('--arms', default='torch,torch-compile,jax-grad,jax-jit')
    ap.add_argument('--no-split', action='store_true', help='skip the modeling-only arms')
    ap.add_argument('--threads', type=int, default=os.cpu_count())
    ap.add_argument('--sweep', action='store_true')
    ap.add_argument('--configs', default='15:1,15:2,20:1,20:2', help='orders:layers list for --sweep')
    ap.add_argument('--compile-configs', default='15:1',
                    help='orders:layers at which --sweep keeps the torch-compile arm')
    ap.add_argument('--mem-limit-gb', type=float, default=None,
                    help='abort and record the peak once the resident set passes this (default 95%% of RAM or of the memory cgroup limit, whichever is lower)')
    ap.add_argument('--kernels', action='store_true',
                    help='time the dense kernels alone (eig, solve, matmul) at each order\'s eig size')
    ap.add_argument('--out', default=str(ROOT / 'artifacts' / 'studies' / 'optimization_step_scale.jsonl'))
    return ap.parse_args()


ARGS = _parse()


def _memory_ceiling():
    """RAM, or the memory cgroup's limit when that is lower (a container's real ceiling)."""
    ceiling = next(int(line.split()[1]) * 1024 for line in open('/proc/meminfo')
                   if line.startswith('MemTotal'))
    for line in open('/proc/self/cgroup'):
        _, controllers, path = line.strip().split(':', 2)
        for limit_file in ([f'/sys/fs/cgroup/memory{path}/memory.limit_in_bytes'] if 'memory' in controllers.split(',')
                           else [f'/sys/fs/cgroup{path}/memory.max'] if controllers == '' else []):
            try:
                ceiling = min(ceiling, int(open(limit_file).read()))
            except (OSError, ValueError):          # absent, or 'max'
                pass
    return ceiling


# The cgroup counts every process in it, not only this one, so stop with some room to spare.
MEM_LIMIT = (ARGS.mem_limit_gb * 2**30 if ARGS.mem_limit_gb else 0.95 * _memory_ceiling())

# Thread pinning must precede the first import of numpy/torch/jax. Torch's LAPACK is MKL
# (OpenMP); jaxlib's CPU LAPACK kernels come from SciPy's OpenBLAS; XLA's own ops use its
# Eigen pool, which follows the core count.
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = str(ARGS.threads)


def machine():
    import numpy, scipy, torch, jax
    cpu = next((line.split(':', 1)[1].strip() for line in open('/proc/cpuinfo')
                if line.startswith('model name')), platform.processor())
    mem = next(int(line.split()[1]) for line in open('/proc/meminfo') if line.startswith('MemTotal'))
    commit = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', '--short', 'HEAD'],
                            capture_output=True, text=True).stdout.strip()
    return dict(cpu=cpu, cores=os.cpu_count(), ram_gb=round(mem / 2**20, 1), threads=ARGS.threads,
                memory_ceiling_gb=round(_memory_ceiling() / 2**30, 2),
                python=platform.python_version(), numpy=numpy.__version__, scipy=scipy.__version__,
                torch=torch.__version__, jax=jax.__version__, commit=commit,
                torch_lapack='MKL' if torch.backends.mkl.is_available() else 'other')


def sweep():
    out = Path(ARGS.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    configs = [tuple(int(x) for x in c.split(':')) for c in ARGS.configs.split(',')]
    compiled = {tuple(int(x) for x in c.split(':')) for c in ARGS.compile_configs.split(',') if c}
    for orders, layers in configs:
        arms = ','.join(a for a in ARGS.arms.split(',')
                        if a != 'torch-compile' or (orders, layers) in compiled)
        for path in ('shape', 'raster'):
            cmd = [sys.executable, __file__, '--orders', str(orders), '--layers', str(layers),
                   '--path', path, '--rounds', str(ARGS.rounds), '--arms', arms,
                   '--threads', str(ARGS.threads), '--out', str(out)]
            if ARGS.no_split:
                cmd.append('--no-split')
            print('#', ' '.join(cmd[1:]), flush=True)
            done = subprocess.run(cmd)
            if done.returncode:
                with open(out, 'a') as fh:
                    fh.write(json.dumps(dict(orders=orders, layers=layers, path=path,
                                             failed=done.returncode)) + '\n')


# ------------------------------------------------------------------ memory

PAGE = os.sysconf('SC_PAGE_SIZE')
CURRENT = {}      # what is running, for the memory-limit record


def rss():
    with open('/proc/self/statm') as fh:
        return int(fh.read().split()[1]) * PAGE


class Peak:
    """The largest resident set seen while a block runs, sampled from a thread."""

    def __enter__(self):
        self.before = self.peak = rss()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def _poll(self):
        while not self._stop.wait(0.002):
            self.peak = max(self.peak, rss())
            if self.peak > MEM_LIMIT:
                # Out of memory is the measurement: record how far it got, then stop before
                # the kernel's OOM killer picks a process for us.
                with open(ARGS.out, 'a') as fh:
                    fh.write(json.dumps(dict(CURRENT, exceeded_gb=self.peak / 2**30,
                                             before_gb=self.before / 2**30,
                                             limit_gb=MEM_LIMIT / 2**30)) + '\n')
                print(f'memory limit: {self.peak / 2**30:.2f} GB resident during {CURRENT}', flush=True)
                os._exit(3)

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, rss())


# ------------------------------------------------------------------ scenes

def main():
    import numpy as np
    import torch
    import jax
    jax.config.update('jax_enable_x64', True)      # before any array is built
    import jax.numpy as jnp
    torch.set_num_threads(ARGS.threads)
    from wavebit import Material, Units, rcwa
    from wavebit.rcwa._lifecycle import _engine_scene

    n, layers = ARGS.orders, ARGS.layers
    THICKNESS = 200.
    RASTER = 128
    rng = np.random.default_rng(0)
    masks = [rng.random((RASTER, RASTER)) < 0.35 for _ in range(layers)]

    def problem(stack_layers):
        return rcwa.Problem(units=Units(length='nm'),
            stack=rcwa.Stack(period=(600., 600.), bottom=Material(2.25), layers=tuple(stack_layers)),
            excitation=rcwa.Excitation(wavelength=850., sources=(rcwa.PlaneWave(polarization='s'),)))

    if ARGS.path == 'shape':
        # Radii differ per layer so two layers are two distinct scenes, not one repeated.
        x0 = np.array([160., 120.][:layers])
        dx = np.array([0.37, -0.29][:layers])
        numerics = rcwa.Numerics(orders=(n, n), topology_check='host')

        def build(r):
            return problem(rcwa.Layer(thickness=THICKNESS, background=Material(1.),
                                      regions=(rcwa.Region(geometry=rcwa.Circle(radius=r[i]),
                                                           material=Material(12.25)),))
                           for i in range(layers))

        def point(k):
            return x0 + dx * k
    else:
        x0 = np.stack([np.where(m, 12.25, 1.) for m in masks])
        numerics = rcwa.Numerics(orders=(n, n))

        def build(eps):
            return problem(rcwa.Layer(thickness=THICKNESS, epsilon=eps[i]) for i in range(layers))

        def point(k):
            return np.stack([np.where(m, 12.25 + 0.013 * k, 1.) for m in masks])

    def loss_of(prepared, xp):
        def loss(v):
            return prepared.run(build(v)).T[0, n, n].real            # zeroth-order transmittance
        return loss

    def modeling_of(prepared, xp):
        # A fixed linear functional of every convolution slot: forward and backward of the
        # modeling half only. Weights are fixed so the functional itself costs the same each call.
        def modeling(v):
            problem_ = build(v)
            total = 0.
            with prepared.execution._scope():
                solvers = prepared._solvers(problem_)
                for solver, geometry in zip(solvers, prepared._geometry):
                    for slot in _engine_scene(solver)._eps_conv_matrices(geometry):
                        if slot is not None:
                            total = total + xp.sum(slot.real) + 0.5 * xp.sum(slot.imag)
            return total
        return modeling

    # ------------------------------------------------------------------ arms
    arms = {}
    wanted = ARGS.arms.split(',')
    if any(a.startswith('torch') for a in wanted):
        prepared_t = rcwa.prepare(build(torch.as_tensor(x0, dtype=torch.float64)),
                                  numerics=numerics, backend='torch')

        def torch_arm(fn):
            def step(x):
                v = torch.tensor(x, dtype=torch.float64, requires_grad=True)
                value = fn(v)
                value.backward()
                return float(value), v.grad.numpy().copy()
            return step

        loss_t, model_t = loss_of(prepared_t, torch), modeling_of(prepared_t, torch)
        if 'torch' in wanted:
            arms['torch'] = (torch_arm(loss_t), torch_arm(model_t))
        if 'torch-compile' in wanted:
            arms['torch-compile'] = (torch_arm(torch.compile(loss_t)), torch_arm(torch.compile(model_t)))
    if any(a.startswith('jax') for a in wanted):
        prepared_j = rcwa.prepare(build(x0), numerics=numerics, backend='jax')

        def jax_arm(fn, jit):
            g = jax.value_and_grad(fn)
            if jit:
                g = jax.jit(g)

            def step(x):
                value, grad = g(jnp.asarray(x))
                grad = np.asarray(jax.block_until_ready(grad))
                return float(value), grad
            return step

        loss_j, model_j = loss_of(prepared_j, jnp), modeling_of(prepared_j, jnp)
        if 'jax-grad' in wanted:
            arms['jax-grad'] = (jax_arm(loss_j, False), jax_arm(model_j, False))
        if 'jax-jit' in wanted:
            arms['jax-jit'] = (jax_arm(loss_j, True), jax_arm(model_j, True))

    # ------------------------------------------------------------------ interleaved rounds
    record = {name: dict(step=[], model=[], step_mem=[], model_mem=[], value=[], grad=[], errors=[])
              for name in arms}
    names = list(arms)
    for k in range(ARGS.rounds + 1):
        x = point(k + 1)
        # Rotate which arm goes first, so no arm always runs on the heels of the same one.
        for name in names[k % len(names):] + names[:k % len(names)]:
            full, model = arms[name]
            parts = [('step', full)] + ([] if ARGS.no_split else [('model', model)])
            for part, fn in parts:
                CURRENT.update(orders=n, layers=layers, path=ARGS.path, arm=name, part=part, round=k)
                with Peak() as peak:
                    t0 = time.perf_counter()
                    try:
                        value, grad = fn(x)
                    except Exception as exc:
                        # A step that raises is a result too (a solver failing at one design
                        # point): record it and keep the other arms and rounds.
                        value = grad = None
                        record[name]['errors'].append(dict(round=k, part=part,
                                                           error=f'{type(exc).__name__}: {str(exc)[:200]}'))
                        print(f'round {k} {name:13s} {part} FAILED: {type(exc).__name__}: {str(exc)[:200]}',
                              flush=True)
                    seconds = time.perf_counter() - t0
                record[name][part].append(None if value is None else seconds)
                record[name][part + '_mem'].append((peak.peak - peak.before) / 2**30)
                if part == 'step':
                    record[name]['value'].append(value)
                    # Raster gradients are 128x128 per layer: keep a fixed projection of them.
                    g = None if grad is None else np.asarray(grad, dtype=np.float64).ravel()
                    record[name]['grad'].append(
                        None if g is None else g.tolist() if g.size <= 4 else
                        [float(g @ np.random.default_rng(1).standard_normal(g.size))])
            fmt = lambda t: '  failed ' if t is None else f'{t:8.3f}'
            print(f'round {k} {name:13s} step {fmt(record[name]["step"][-1])} s'
                  + ('' if ARGS.no_split else f'  modeling {fmt(record[name]["model"][-1])} s')
                  + f'  +{record[name]["step_mem"][-1]:.2f} GB  T0 {record[name]["value"][-1]}',
                  flush=True)

    # ------------------------------------------------------------------ summary
    import resource
    summary = dict(orders=n, layers=layers, path=ARGS.path, rounds=ARGS.rounds,
                   harmonics=(2 * n + 1) ** 2, machine=machine(),
                   maxrss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, arms={})
    ref = next(iter(record.values()))
    ok = lambda xs: [t for t in xs if t is not None]
    for name, r in record.items():
        warm = ok(r['step'][1:])
        pairs = [(a, b) for a, b in zip(r['value'], ref['value']) if a is not None and b is not None]
        gpairs = [(a, b) for ga, gb in zip(r['grad'], ref['grad']) if ga is not None and gb is not None
                  for a, b in zip(ga, gb)]
        entry = dict(first=r['step'][0], median=statistics.median(warm) if warm else None,
                     min=min(warm) if warm else None, steps=r['step'], errors=r['errors'],
                     peak_step_gb=max(r['step_mem']),
                     value_spread=max((abs(a - b) for a, b in pairs), default=None),
                     grad_spread=max((abs(a - b) / max(abs(b), 1e-300) for a, b in gpairs), default=None))
        if not ARGS.no_split:
            mwarm = ok(r['model'][1:])
            entry.update(model_first=r['model'][0],
                         model_median=statistics.median(mwarm) if mwarm else None,
                         model_min=min(mwarm) if mwarm else None, peak_model_gb=max(r['model_mem']))
        summary['arms'][name] = entry
    print(json.dumps(summary['arms'], indent=1))
    with open(ARGS.out, 'a') as fh:
        fh.write(json.dumps(summary) + '\n')


def kernels():
    """The dense kernels a step is made of, alone, at the size of each order's eigenproblem.

    A 2D layer's eigenproblem is ``2*(2n+1)**2`` square. Each kernel is timed on the library
    each backend calls -- torch on its MKL, jax through XLA (eig and solve via jaxlib's LAPACK
    kernels, matmul via XLA's own dot) -- with the same thread count.
    """
    import numpy as np
    import torch
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    torch.set_num_threads(ARGS.threads)
    rows = []
    for n in sorted({int(c.split(':')[0]) for c in ARGS.configs.split(',')}):
        m = 2 * (2 * n + 1) ** 2
        rng = np.random.default_rng(0)
        a = rng.standard_normal((m, m)) + 1j * rng.standard_normal((m, m))
        ta, ja = torch.tensor(a), jnp.asarray(a)
        jeig, jsolve, jmm = jax.jit(jnp.linalg.eig), jax.jit(jnp.linalg.solve), jax.jit(jnp.matmul)
        cases = {'eig': (lambda: torch.linalg.eig(ta), lambda: jax.block_until_ready(jeig(ja))),
                 'solve': (lambda: torch.linalg.solve(ta, ta), lambda: jax.block_until_ready(jsolve(ja, ja))),
                 'matmul': (lambda: ta @ ta, lambda: jax.block_until_ready(jmm(ja, ja)))}
        for kernel, (tf, jf) in cases.items():
            row = dict(orders=n, size=m, kernel=kernel)
            for who, f in (('torch', tf), ('jax', jf)):
                f()
                samples = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    f()
                    samples.append(time.perf_counter() - t0)
                row[who] = min(samples)
            print(json.dumps(row), flush=True)
            rows.append(row)
    with open(ARGS.out, 'a') as fh:
        fh.write(json.dumps(dict(kernels=rows, machine=machine())) + '\n')


if __name__ == '__main__':
    kernels() if ARGS.kernels else sweep() if ARGS.sweep else main()
