"""
benchmark_backends.py — which backend, and whether to freeze, for a *differentiated* run.

Wavebit can produce the same gradient three ways, and they are not interchangeable:

    torch      eager autograd. Decides its geometry on every call, so nothing is frozen and
               nothing can go stale.
    jax eager  jax.grad with no frozen geometry. Available only where the discrete decision
               can be narrowed inside a grad trace (see docs/backend-choice.md).
    jax jit    jax.jit over a frozen geometry. Fastest per step where it applies, but a
               recompilation lands whenever the frozen record's *content* changes.

The choice is not a matter of taste: measured below, the ordering flips with problem size and
with how often the structure's discrete description moves. This benchmark measures

    * forward and gradient wall time for each mode, over a scale sweep;
    * the compile and *re*-compile cost that jit adds;
    * the resulting break-even -- how many steps a fit must hold its discrete description
      before jit repays the recompilation it costs;
    * and, as a gate on all of the above, whether the three modes agree on the gradient.

Two paths are swept, because they differentiate different things:

    vector     a shape's *length* is the design variable (OCD: CD, sidewall angle, pitch).
               This is what the closed-form series exists for -- the segment edges enter the
               Fourier transform analytically, so d(efficiency)/d(length) is available.
    raster     the permittivity grid is the design variable (inverse design). Swept across
               all three Fourier strategies, since they freeze different things.

Every material below is a **relative permittivity**: ``structure`` (raster pixels, or a block
layer's background and each object's own slot) and the ``top`` / ``bot`` half-spaces alike.
The structures are written from refractive indices through ``wavebit.eps_from_index`` so the
scenes stay the ones these timings were taken on -- the quantity changed with the keyword
names, and a datasheet index left in a material slot is a legal, different medium.

Run from the repo root::

    python studies/benchmark_backends.py                    # ~2 h, the full sweep
    python studies/benchmark_backends.py --quick            # ~5 min
    python studies/benchmark_backends.py --budget 3600      # stop adding work after 1 h

Results stream to ``studies/results/backends.json`` as they land, so an interrupted run
still leaves everything it finished. Markdown tables go to stdout.
"""
import argparse
import importlib
import json
import os
import statistics
import sys
import time

import numpy as np

# Ensure the local (worktree) wavebit is used rather than any installed copy.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wavebit                                                        # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'backends.json')

STRATEGIES = {
    'sampled': dict(fourier_coefficients='sampled', factorization='li'),
    'integrated': dict(fourier_coefficients='integrated', factorization='li'),
    'NVF': dict(fourier_coefficients='sampled', factorization='nvf'),
}

# The three media every structure below is built from, as **relative permittivity** -- named
# by the refractive index they came from, because that is how they were written before the
# flip and keeping the timings comparable means keeping the same scenes. `eps_from_index` is
# `n**2` and nothing else; it is spelled out rather than inlined as a literal because the
# rounding matters elsewhere in this repository and a tidy decimal is not the same number.
EPS_HIGH = wavebit.eps_from_index(3.5)          # 12.25
EPS_MID = wavebit.eps_from_index(2.0)           # 4.0
EPS_LOW = wavebit.eps_from_index(1.0)           # 1.0

#: TE, as a polarization angle. This was `pol=0`, which additionally declared a solver path;
#: `psi` says only the angle, which is what these 2D scenes can use.
TE_PSI = np.pi / 2


# ------------------------------------------------------------------ timing helpers

def _time(fn, reps, warmup=1):
    """Median wall time of ``fn`` in ms, after ``warmup`` untimed calls."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


class Budget:
    """A wall-clock allowance, so a long sweep degrades by dropping work rather than by dying."""

    def __init__(self, seconds):
        self.limit = seconds
        self.start = time.perf_counter()

    def left(self):
        return self.limit - (time.perf_counter() - self.start)

    def can_afford(self, estimate_s):
        return self.limit is None or self.left() > estimate_s



# ------------------------------------------------------------------ machine state

def _probe_step():
    """One fixed, mid-sized solve. Not a result — a ruler for the machine itself."""
    import torch
    rng = np.random.default_rng(0)
    pattern = np.where(rng.random((1, 64, 64)) < 0.35, EPS_HIGH, EPS_LOW)
    cfg = dict(period=[300.0, 300.0], thickness=[205.0], fto=[5, 5], top=1.0, bot=1.0,
               theta=0.15, phi=0.35, psi=0.3, wavelength=900.0)

    def step():
        t = torch.tensor(pattern, requires_grad=True)
        out = wavebit.rcwa(backend=2, structure=t, **STRATEGIES['sampled'], **cfg).solve()
        out.res.T.real.sum().backward()

    return step


def warm_up(seconds=45):
    """Run the probe until the machine settles, before anything is timed.

    Without this the first sweep is measured on a cold machine and the later ones on a warm
    one, which showed up as the raster tables being systematically slower than the vector ones
    in the same run — a property of the ordering, not of the code under test.
    """
    step = _probe_step()
    print(f'[warm-up] {seconds} s ...', flush=True)
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        step()


def drift(label, out):
    """Time the probe and record it, so a run states its own repeatability."""
    ms = _time(_probe_step(), reps=3)
    out.append({'at': label, 'ms': ms})
    print(f'[drift] {label}: {ms:.0f} ms', flush=True)
    checkpoint({'drift': out})


# ------------------------------------------------------------------ the three modes

def _fingerprint(grad):
    """One number standing for a whole gradient, so modes can be compared for agreement.

    A fixed pseudo-random projection rather than a norm: two different gradients can share a
    maximum, but they will not share a projection.
    """
    g = np.asarray(grad, dtype=np.float64).ravel()
    probe = np.random.default_rng(12345).standard_normal(g.size)
    return float(g @ probe)


def torch_grad(build, seed):
    """(fingerprint, callable) for eager PyTorch autograd — no frozen geometry."""
    import torch

    def step():
        v = torch.tensor(np.asarray(seed, dtype=np.float64), requires_grad=True)
        out = build(2, v).solve().res.T.real.sum()
        out.backward()
        return _fingerprint(v.grad.detach().numpy())

    return step(), step


def jax_eager_grad(build, seed):
    """(fingerprint, callable) for jax.grad with no frozen geometry."""
    import jax
    import jax.numpy as jnp

    def loss(v):
        return jnp.sum(build(1, v).solve().res.T).real

    x = jnp.asarray(np.asarray(seed, dtype=np.float64))
    g = jax.grad(loss)
    return _fingerprint(g(x)), lambda: np.asarray(g(x))


def jax_jit_grad(build, seed):
    """jax.jit over a frozen geometry, measured the two ways a fit can be written.

    ``refreeze`` retakes the freeze every step — the only shape of loop that cannot carry a
    stale record forward, and the one whose cost depends on whether the record can say what it
    contains. ``reuse`` holds one freeze across every step: the fastest possible number, and
    the one that goes silently wrong the moment the structure outgrows the record.
    """
    import jax
    import jax.numpy as jnp

    def loss(v, geometry):
        return jnp.sum(build(1, v).solve(geometry=geometry).res.T).real

    x = jnp.asarray(np.asarray(seed, dtype=np.float64))
    freeze = lambda: build(1, x).freeze_geometry()
    g = jax.jit(jax.grad(loss), static_argnums=1)

    jax.clear_caches()
    t0 = time.perf_counter()
    first = np.asarray(g(x, freeze()))
    compile_ms = (time.perf_counter() - t0) * 1e3

    held = freeze()
    return (_fingerprint(first), lambda: np.asarray(g(x, freeze())),
            lambda: np.asarray(g(x, held)), compile_ms,
            _time(freeze, reps=5))


# ------------------------------------------------------------------ the two paths

def vector_builder(fto, split, layers):
    """A stack of ellipse layers whose ``lx`` is the design variable."""
    cfg = dict(period=[300.0, 300.0], thickness=[205.0] * layers, fto=[fto, fto],
               top=1.0, bot=1.0, theta=0.15, phi=0.35, psi=TE_PSI, wavelength=900.0)

    def build(backend, lx):
        structure = [[EPS_HIGH,
                      [['ellipse', 150.0, 150.0, lx, 55.0, EPS_MID, 0.27, split, split]]]] * layers
        return wavebit.rcwa(backend=backend, structure=structure, **cfg)

    return build


def raster_builder(n, fto, strategy, layers=1):
    """A binary raster stack whose permittivity grid *is* the design variable."""
    rng = np.random.default_rng(0)
    pattern = np.where(rng.random((layers, n, n)) < 0.35, EPS_HIGH, EPS_LOW)
    cfg = dict(period=[300.0, 300.0], thickness=[205.0] * layers, fto=[fto, fto],
               top=1.0, bot=1.0, theta=0.15, phi=0.35, psi=0.3, wavelength=900.0)

    def build(backend, structure):
        return wavebit.rcwa(backend=backend, structure=structure, **STRATEGIES[strategy], **cfg)

    return build, pattern


# ------------------------------------------------------------------ one measurement

def measure(build, seed, reps, budget, estimate_s):
    """Run every mode that applies and return a dict of times, fingerprints and failures."""
    row = {}
    for name, fn in (('torch', lambda: torch_grad(build, seed)),
                     ('jax_eager', lambda: jax_eager_grad(build, seed))):
        if not budget.can_afford(estimate_s):
            row[name] = {'skipped': 'budget'}
            continue
        try:
            fingerprint, step = fn()
            row[name] = {'grad': fingerprint, 'grad_ms': _time(step, reps)}
        except Exception as exc:                      # a mode that cannot run says so
            row[name] = {'error': f'{type(exc).__name__}: {str(exc)[:120]}'}

    if budget.can_afford(estimate_s * 4):
        try:
            fingerprint, refreeze, reuse, compile_ms, freeze_ms = jax_jit_grad(build, seed)
            row['jax_jit_reuse'] = {'grad': fingerprint, 'grad_ms': _time(reuse, reps),
                                    'compile_ms': compile_ms, 'freeze_ms': freeze_ms}
            row['jax_jit_refreeze'] = {'grad_ms': _time(refreeze, reps, warmup=0)}
        except Exception as exc:
            row['jax_jit_reuse'] = {'error': f'{type(exc).__name__}: {str(exc)[:120]}'}
    else:
        row['jax_jit_reuse'] = {'skipped': 'budget'}

    grads = [m['grad'] for m in row.values() if isinstance(m, dict) and 'grad' in m]
    if len(grads) > 1:
        row['grad_agreement'] = max(abs(g - grads[0]) for g in grads) / max(abs(grads[0]), 1e-30)
    return row


# ------------------------------------------------------------------ sweeps

def sweep_vector(budget, reps, quick):
    grid = ([(3, 4, 1), (5, 12, 1)] if quick else
            [(3, 4, 1), (3, 12, 1), (3, 20, 1), (5, 4, 1), (5, 12, 1), (5, 20, 1),
             (5, 12, 3), (7, 4, 1), (7, 12, 1), (7, 20, 1), (7, 12, 3), (7, 20, 3),
             (9, 12, 1), (9, 20, 1), (9, 20, 3), (11, 20, 3)])
    out = {}
    for fto, split, layers in grid:
        key = f'fto={fto},split={split},L={layers}'
        estimate = 0.05 * (2 * fto + 1) ** 3 / 100 * layers * 6    # rough, for budgeting only
        print(f'[vector] {key} ...', flush=True)
        out[key] = measure(vector_builder(fto, split, layers), 85.0, reps, budget, estimate)
        out[key]['fto'], out[key]['split'], out[key]['layers'] = fto, split, layers
        checkpoint({'vector': out})
    return out


def sweep_raster(budget, reps, quick):
    grid = ([(16, 3)] if quick else
            [(16, 3), (32, 3), (32, 5), (64, 3), (64, 5), (64, 7), (128, 3), (128, 5)])
    out = {}
    for n, fto in grid:
        for strategy in STRATEGIES:
            key = f'N={n},fto={fto},{strategy}'
            estimate = 0.05 * (2 * fto + 1) ** 3 / 100 * 6
            print(f'[raster] {key} ...', flush=True)
            build, pattern = raster_builder(n, fto, strategy)
            out[key] = measure(build, pattern, reps, budget, estimate)
            out[key].update(n=n, fto=fto, strategy=strategy)
            checkpoint({'raster': out})
    return out


def sweep_recompile(budget, reps):
    """How much a *content* change costs, against the steady step it interrupts."""
    import jax
    import jax.numpy as jnp
    cfg = dict(period=[300.0, 300.0], thickness=[205.0], fto=[3, 3], top=1.0, bot=1.0,
               theta=0.15, phi=0.35, psi=TE_PSI, wavelength=900.0)
    make = lambda lx: [[EPS_HIGH, [['ellipse', 150.0, 150.0, lx, 55.0, EPS_MID, 0.27, 4, 4]]]]

    def loss(v, geometry):
        return jnp.sum(wavebit.rcwa(backend=1, structure=make(v), **cfg)
                       .solve(geometry=geometry).T).real

    g = jax.jit(jax.grad(loss), static_argnums=1)
    freeze = lambda lx: wavebit.rcwa(backend=1, structure=make(jnp.asarray(lx)),
                                     **cfg).freeze_geometry()
    jax.clear_caches()
    out = []
    for lx in [85.0, 90.0, 95.0, 100.0, 103.0, 106.0, 110.0, 115.0, 120.0, 125.0]:
        t0 = time.perf_counter()
        grad = float(g(jnp.asarray(lx), freeze(lx)))
        out.append({'lx': lx, 'ms': (time.perf_counter() - t0) * 1e3, 'grad': grad})
        print(f'[recompile] lx={lx}: {out[-1]["ms"]:.0f} ms', flush=True)
    checkpoint({'recompile_sweep': out})
    return out


def sweep_stale_window(budget):
    """How far a *held* freeze stays valid — bisected, per shape and slab count.

    Only matters if a fit holds one record rather than retaking it; on the block path that is
    no longer the recommended loop, but the raster records give no choice.
    """
    import jax.numpy as jnp
    from wavebit import StaleGeometryError
    cfg = dict(period=[300.0, 300.0], thickness=[205.0], fto=[3, 3], top=1.0, bot=1.0,
               theta=0.15, phi=0.35, psi=TE_PSI, wavelength=900.0)

    def accepts(make, value, frozen):
        try:
            wavebit.rcwa(backend=1, structure=make(jnp.asarray(value)),
                         **cfg).solve(geometry=frozen)
            return True
        except StaleGeometryError:
            return False

    def window(make, nominal, lo_probe, hi_probe, tol=1e-4):
        frozen = wavebit.rcwa(backend=1, structure=make(jnp.asarray(nominal)),
                              **cfg).freeze_geometry()

        def edge(probe):
            if accepts(make, probe, frozen):
                return None                      # unbounded in this direction
            good, bad = nominal, probe
            while abs(bad - good) > tol:
                mid = 0.5 * (good + bad)
                good, bad = (mid, bad) if accepts(make, mid, frozen) else (good, mid)
            return good

        return edge(lo_probe), edge(hi_probe)

    cases = [
        ('ellipse', 'lx', 85.0, 5.0, 295.0, [3, 4, 6, 9, 12, 20],
         lambda s: (lambda v: [[EPS_HIGH,
                                [['ellipse', 150., 150., v, 55., EPS_MID, 0.27, s, s]]]])),
        ('rectangle', 'angle', 0.27, -1.5, 1.5, [1, 2, 4, 8, 12, 20],
         lambda s: (lambda v: [[EPS_HIGH,
                                [['rectangle', 150., 150., 85., 55., EPS_MID, v, s, s]]]])),
        ('rectangle', 'lx (angle=0)', 85.0, 5.0, 295.0, [1, 4, 12, 20],
         lambda s: (lambda v: [[EPS_HIGH,
                                [['rectangle', 150., 150., v, 55., EPS_MID, 0.0, s, s]]]])),
    ]
    out = []
    for shape, param, nominal, lo_probe, hi_probe, splits, make_for in cases:
        for split in splits:
            if not budget.can_afford(30):
                break
            lo, hi = window(make_for(split), nominal, lo_probe, hi_probe)
            out.append({'shape': shape, 'parameter': param, 'nominal': nominal,
                        'n_split': split, 'lo': lo, 'hi': hi})
            print(f'[window] {shape} {param} split={split}: {lo} .. {hi}', flush=True)
            checkpoint({'stale_window': out})
    return out


# ------------------------------------------------------------------ reporting

_STATE = {}


def checkpoint(update):
    _STATE.update(update)
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, 'w') as fh:
        json.dump(_STATE, fh, indent=1, sort_keys=True)


def _cell(entry, field='grad_ms'):
    if 'error' in entry:
        return entry['error'].split(':')[0]
    if 'skipped' in entry:
        return '—'
    return f'{entry[field]:.0f}'


def table(rows, title, columns):
    print(f'\n### {title}\n')
    print('| ' + ' | '.join(columns) + ' |')
    print('|' + '|'.join(['---'] * len(columns)) + '|')
    for row in rows:
        print('| ' + ' | '.join(row) + ' |')


def report(state):
    for section, title, unit in (
            ('vector', 'Vector path — d(efficiency)/d(length)', 'length'),
            ('raster', 'Raster path — d(efficiency)/d(permittivity)', 'permittivity')):
        if section not in state:
            continue
        rows = []
        for key, r in state[section].items():
            reuse = r.get('jax_jit_reuse', {})
            rows.append([
                key,
                _cell(r.get('torch', {})),
                _cell(r.get('jax_eager', {})),
                _cell(reuse),
                _cell(r.get('jax_jit_refreeze', {})),
                _cell(reuse, 'compile_ms') if 'compile_ms' in reuse else '—',
                _cell(reuse, 'freeze_ms') if 'freeze_ms' in reuse else '—',
                f'{r.get("grad_agreement", float("nan")):.1e}',
            ])
        table(rows, f'{title}, ms per step',
              ['config', 'torch', 'jax eager', 'jit (one freeze)', 'jit (re-freeze)',
               'compile', 'freeze', 'grad spread'])

    if 'recompile_sweep' in state:
        rows = [[f'{e["lx"]:.0f}', f'{e["ms"]:.0f}', f'{e["grad"]:+.3e}']
                for e in state['recompile_sweep']]
        table(rows, 'Re-freezing every step while lx moves — where the recompilation lands',
              ['lx', 'ms', 'grad'])

    if 'drift' in state:
        rows = [[e['at'], f'{e["ms"]:.0f}'] for e in state['drift']]
        first = state['drift'][0]['ms']
        rows.append(['spread', f'{max(e["ms"] for e in state["drift"]) / min(e["ms"] for e in state["drift"]):.2f}x'])
        table(rows, 'Machine drift over the run — one fixed configuration, timed three times',
              ['at', 'ms'])

    if 'stale_window' in state:
        rows = []
        for e in state['stale_window']:
            lo, hi, nominal = e['lo'], e['hi'], e['nominal']
            if lo is None and hi is None:
                span = 'unbounded'
            elif lo is None or hi is None:
                span = 'one-sided'
            else:
                span = (f'{hi - lo:.3f}  ({(lo - nominal) / nominal * 100:+.1f}% .. '
                        f'{(hi - nominal) / nominal * 100:+.1f}%)')
            rows.append([e['shape'], f'{e["parameter"]} @ {nominal}', str(e['n_split']),
                         'none' if lo is None else f'{lo:.3f}',
                         'none' if hi is None else f'{hi:.3f}', span])
        table(rows, 'How far a *held* freeze stays valid',
              ['shape', 'parameter', 'n_split', 'lo', 'hi', 'width'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='a few points, for a smoke test')
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--budget', type=float, default=7200, help='wall-clock allowance, seconds')
    args = ap.parse_args()

    importlib.import_module('wavebit.on_jax')
    importlib.import_module('wavebit.on_torch')

    budget = Budget(args.budget)
    print(f'# Wavebit backend benchmark  (budget {args.budget:.0f} s, reps {args.reps})\n',
          flush=True)

    warm_up(5 if args.quick else 45)
    marks = []
    drift('start', marks)
    sweep_vector(budget, args.reps, args.quick)
    drift('after vector', marks)
    sweep_raster(budget, args.reps, args.quick)
    drift('after raster', marks)
    sweep_recompile(budget, args.reps)
    if not args.quick:
        sweep_stale_window(budget)

    print(f'\nfinished with {budget.left():.0f} s of budget left\n')
    report(_STATE)
    print(f'\nraw results: {RESULTS}')


if __name__ == '__main__':
    main()
