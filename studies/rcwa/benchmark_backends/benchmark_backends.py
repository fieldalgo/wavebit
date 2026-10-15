"""
benchmark_backends.py — which backend, and whether to freeze, for a *differentiated* run.

Wavebit can produce the same gradient three ways, and they are not interchangeable:

    torch      eager autograd. Decides its geometry on every call, so nothing is frozen and
               nothing can go stale.
    jax eager  jax.grad with no frozen geometry. Available only where the discrete decision
               can be narrowed inside a grad trace (see docs/validation/rcwa/backend-choice.md).
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

    shapes     a shape's *length* is the design variable (OCD: CD, sidewall angle, pitch).
               This is what the closed-form series exists for -- the shape's outline enters
               the Fourier transform analytically, so d(efficiency)/d(length) is available.
    raster     the permittivity grid is the design variable (inverse design). Swept across
               all three Fourier strategies, since they freeze different things.

Every material below is a **relative permittivity**: ``structure`` (raster pixels, or a shape
layer's background and each shape's own ``material``) and the ``top`` / ``bot`` half-spaces
alike.
The structures are written from refractive indices through ``wavebit.eps_from_refractive_index`` so
the scenes stay the ones these timings were taken on -- the quantity changed with the keyword
names, and a datasheet index left in a material slot is a legal, different medium.

Run from the repo root::

    python studies/rcwa/benchmark_backends/benchmark_backends.py                    # ~2 h, the full sweep
    python studies/rcwa/benchmark_backends/benchmark_backends.py --quick            # ~5 min
    python studies/rcwa/benchmark_backends/benchmark_backends.py --budget 3600      # stop adding work after 1 h

Results stream to ``artifacts/studies/backends.json`` as they land, so an interrupted run
still leaves everything it finished. Markdown tables go to stdout.
"""

import argparse
from pathlib import Path
import importlib
import json
import os
import statistics
import sys
import time

import numpy as np

# Select the checkout before importing any wavebit module. Archived comparisons may
# use the historical flat package; the maintained tree uses src/.
ROOT = str(Path(__file__).resolve().parents[3])
_root_parser = argparse.ArgumentParser(add_help=False)
_root_parser.add_argument('--root', default=ROOT)
_SELECTED_ROOT = Path(_root_parser.parse_known_args()[0].root).expanduser().resolve()
_PACKAGE_PARENT = next((path for path in (_SELECTED_ROOT / 'src', _SELECTED_ROOT)
                        if (path / 'wavebit' / '__init__.py').is_file()), None)
if _PACKAGE_PARENT is None:
    raise RuntimeError(f'No wavebit source package in selected root: {_SELECTED_ROOT}')
sys.path.insert(0, str(_PACKAGE_PARENT))
_selected_wavebit = importlib.import_module('wavebit')
_expected_origin = (_PACKAGE_PARENT / 'wavebit' / '__init__.py').resolve()
if Path(_selected_wavebit.__file__).resolve() != _expected_origin:
    raise RuntimeError(
        f'Selected root {_SELECTED_ROOT} requires {_expected_origin}; '
        f'wavebit was already imported from {_selected_wavebit.__file__}. '
        'Run this study in a fresh Python process.')
_CURRENT_LAYOUT = (_PACKAGE_PARENT / 'wavebit/rcwa/_factory.py').is_file()


def _study_module(current, historical):
    """Load the matching private implementation in the selected checkout."""
    return importlib.import_module(
        current if _CURRENT_LAYOUT else historical)



# Only an explicitly selected historical tree may supply the retired facade.
_HISTORICAL_API = (None if _CURRENT_LAYOUT else
    importlib.import_module('wavebit.compat.legacy')
    if (_PACKAGE_PARENT / 'wavebit/compat/legacy.py').is_file() else _selected_wavebit)


def _study_symbol(module, name):
    if not _CURRENT_LAYOUT:
        return getattr(_HISTORICAL_API, 'enable_x64' if name == '_enable_x64' else name)
    owner = importlib.import_module(module)
    if name == '_enable_x64' and not hasattr(owner, name):
        # Src-layout snapshots before retirement kept this helper in the facade.
        # Reproduce its explicit setup without requiring that facade to be installed.
        def enable():
            importlib.import_module('jax')
            owner.lift_jax_x64(required=True)
            scattering = importlib.import_module('wavebit.rcwa._engine.scattering')
            scattering._register_pytree_if_jax_loaded()
        return enable
    return getattr(owner, name)

_study_enable_x64 = _study_symbol('wavebit._common.backend', '_enable_x64')
_study_eps_from_refractive_index = _study_symbol('wavebit.materials.response', 'eps_from_refractive_index')
_study_rcwa = _study_symbol('wavebit.rcwa._factory', 'rcwa')


def _backend_arg(name):
    Backend = _study_module('wavebit._common.enums', 'wavebit._core.enums').Backend
    return {'numpy': 0, 'jax': 1, 'torch': 2}[name] if issubclass(Backend, int) else name


RESULTS = os.path.join(ROOT, 'artifacts', 'studies', 'backends.json')

STRATEGIES = {
    'sampled': dict(fourier_coefficients='sampled', factorization='li'),
    'integrated': dict(fourier_coefficients='integrated', factorization='li'),
    'NVF': dict(fourier_coefficients='sampled', factorization='nvf'),
}

# The three media every structure below is built from, as **relative permittivity** -- named
# by the refractive index they came from, because that is how they were written before the
# flip and keeping the timings comparable means keeping the same scenes. `eps_from_refractive_index`
# is `n**2` and nothing else; it is spelled out rather than inlined as a literal because the
# rounding matters elsewhere in this repository and a tidy decimal is not the same number.
EPS_HIGH = _study_eps_from_refractive_index(3.5)  # 12.25
EPS_MID = _study_eps_from_refractive_index(2.0)   # 4.0
EPS_LOW = _study_eps_from_refractive_index(1.0)   # 1.0

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
    cfg = dict(period=[300.0, 300.0], thickness=[205.0], truncation_order=[5, 5], top=1.0, bot=1.0,
               theta=0.15, phi=0.35, psi=0.3, wavelength=900.0)

    def step():
        t = torch.tensor(pattern, requires_grad=True)
        out = _study_rcwa(backend=_backend_arg('torch'), structure=t, **STRATEGIES['sampled'], **cfg).solve()
        out.incident.T.real.sum().backward()

    return step


def warm_up(seconds=45):
    """Run the probe until the machine settles, before anything is timed.

    Without this the first sweep is measured on a cold machine and the later ones on a warm
    one, which showed up as the raster tables being systematically slower than the shape ones
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
        out = build('torch', v).solve().incident.T.real.sum()
        out.backward()
        return _fingerprint(v.grad.detach().numpy())

    return step(), step


def jax_eager_grad(build, seed):
    """(fingerprint, callable) for jax.grad with no frozen geometry."""
    import jax
    import jax.numpy as jnp

    def loss(v):
        return jnp.sum(build('jax', v).solve().incident.T).real

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
        return jnp.sum(build('jax', v).solve(geometry=geometry).incident.T).real

    x = jnp.asarray(np.asarray(seed, dtype=np.float64))
    freeze = lambda: build('jax', x).freeze_geometry()
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

def shape_builder(truncation_order, layers):
    """A stack of ellipse layers whose ``lx`` is the design variable.

    One shape per layer, deliberately: a traced layer of more than one shape needs a frozen
    geometry (which builder takes it is a question about whether they overlap), and this sweep
    is about the *step* cost, not about the freeze. `sweep_recompile` below is where the freeze
    is timed.

    This took a `split` argument until 2026-08-27 -- the block modeller's `n_split`, which set
    how many axis-aligned rectangles a shape was cut into. There is no successor knob on the
    surviving paths, and `shape_nvf_grid_size` is not one: it samples a normal field rather than
    approximating the outline. So the sweep's grid lost that axis rather than renaming it.
    """
    cfg = dict(period=[300.0, 300.0], thickness=[205.0] * layers, truncation_order=[truncation_order, truncation_order],
               top=1.0, bot=1.0, theta=0.15, phi=0.35, psi=TE_PSI, wavelength=900.0)

    def build(backend, lx):
        structure = [dict(background=EPS_HIGH,
                          shapes=[dict(type='ellipse', cx=0.0, cy=0.0, lx=lx, ly=55.0,
                                       angle=0.27, material=EPS_MID)])] * layers
        return _study_rcwa(backend=_backend_arg(backend), structure=structure, **cfg)

    return build


def raster_builder(n, truncation_order, strategy, layers=1):
    """A binary raster stack whose permittivity grid *is* the design variable."""
    rng = np.random.default_rng(0)
    pattern = np.where(rng.random((layers, n, n)) < 0.35, EPS_HIGH, EPS_LOW)
    cfg = dict(period=[300.0, 300.0], thickness=[205.0] * layers, truncation_order=[truncation_order, truncation_order],
               top=1.0, bot=1.0, theta=0.15, phi=0.35, psi=0.3, wavelength=900.0)

    def build(backend, structure):
        return _study_rcwa(backend=_backend_arg(backend), structure=structure, **STRATEGIES[strategy], **cfg)

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

def sweep_shapes(budget, reps, quick):
    grid = ([(3, 1), (5, 1)] if quick else
            [(3, 1), (5, 1), (7, 1), (9, 1), (11, 1), (13, 1),
             (5, 3), (7, 3), (9, 3), (11, 3)])
    out = {}
    for truncation_order, layers in grid:
        key = f'truncation_order={truncation_order},L={layers}'
        estimate = 0.05 * (2 * truncation_order + 1) ** 3 / 100 * layers * 6    # rough, for budgeting only
        print(f'[shapes] {key} ...', flush=True)
        out[key] = measure(shape_builder(truncation_order, layers), 85.0, reps, budget, estimate)
        out[key]['truncation_order'], out[key]['layers'] = truncation_order, layers
        checkpoint({'shapes': out})
    return out


def sweep_raster(budget, reps, quick):
    grid = ([(16, 3)] if quick else
            [(16, 3), (32, 3), (32, 5), (64, 3), (64, 5), (64, 7), (128, 3), (128, 5)])
    out = {}
    for n, truncation_order in grid:
        for strategy in STRATEGIES:
            key = f'N={n},truncation_order={truncation_order},{strategy}'
            estimate = 0.05 * (2 * truncation_order + 1) ** 3 / 100 * 6
            print(f'[raster] {key} ...', flush=True)
            build, pattern = raster_builder(n, truncation_order, strategy)
            out[key] = measure(build, pattern, reps, budget, estimate)
            out[key].update(n=n, truncation_order=truncation_order, strategy=strategy)
            checkpoint({'raster': out})
    return out


def sweep_recompile(budget, reps):
    """How much a *content* change costs, against the steady step it interrupts."""
    import jax
    import jax.numpy as jnp
    cfg = dict(period=[300.0, 300.0], thickness=[205.0], truncation_order=[3, 3], top=1.0, bot=1.0,
               theta=0.15, phi=0.35, psi=TE_PSI, wavelength=900.0)
    make = lambda lx: [dict(background=EPS_HIGH,
                            shapes=[dict(type='ellipse', cx=0.0, cy=0.0, lx=lx, ly=55.0,
                                         angle=0.27, material=EPS_MID)])]

    def loss(v, geometry):
        return jnp.sum(_study_rcwa(backend=_backend_arg('jax'), structure=make(v), **cfg)
                       .solve(geometry=geometry).T).real

    g = jax.jit(jax.grad(loss), static_argnums=1)
    freeze = lambda lx: _study_rcwa(backend=_backend_arg('jax'), structure=make(jnp.asarray(lx)),
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


def decompose(backend, truncation_order, shape_nvf_grid_size, reps):
    """Modeling and solve, timed apart, for one backend and one configuration.

    `eps_conv_matrices()` is the modeling half -- the shapes' analytic transform and the normal
    field they carry -- and `solve_conv` is what is left of a forward call. Timed separately
    because the two halves have opposite backend preferences, which a single per-step number
    hides.
    """
    cfg = dict(period=[300.0, 300.0], thickness=[205.0],
               truncation_order=[truncation_order, truncation_order],
               top=1.0, bot=1.0, theta=0.15, phi=0.35, psi=TE_PSI, wavelength=900.0,
               shape_nvf_grid_size=shape_nvf_grid_size)

    def build(lx):
        return _study_rcwa(backend=_backend_arg(backend), structure=[
            dict(background=EPS_HIGH,
                 shapes=[dict(type='ellipse', cx=0.0, cy=0.0, lx=lx, ly=55.0,
                              angle=0.27, material=EPS_MID)])], **cfg)

    # **The shape moves between calls, and it has to.** `normal_projector_coeffs` is cached on
    # the shape parameters, so timing the same call twice measures a cache hit -- flat in
    # `shape_nvf_grid_size` at 1.1 ms, which is the one number a fit never sees. A fit moves the design
    # variable every step and misses every time.
    step = [0]

    def modeling():
        step[0] += 1
        build(85.0 + 0.037 * step[0]).eps_conv_matrices()

    mats = build(85.0).eps_conv_matrices()
    solver = build(85.0)
    return {'modeling_ms': _time(modeling, reps),
            'solve_ms': _time(lambda: solver.solve_conv(*mats), reps)}


def sweep_decomposition(budget, reps):
    """Where each backend's time goes, and how that moves with `shape_nvf_grid_size`.

    This replaces a table keyed on the removed block path's `n_split`, which set how many small
    array operations that modeller emitted. The shape path emits no such stream; what sets its
    modeling cost is `shape_nvf_grid_size`, the resolution its analytic normal field is sampled at.
    """
    out = []
    for truncation_order, shape_nvf_grid_size in ((3, 256), (3, 1024), (3, 2048),
                                          (7, 1024), (11, 1024)):
        if not budget.can_afford(90):
            break
        row = {'truncation_order': truncation_order, 'shape_nvf_grid_size': shape_nvf_grid_size}
        for name in ('torch', 'jax'):
            try:
                row[name] = decompose(name, truncation_order, shape_nvf_grid_size, reps)
            except Exception as exc:                   # a backend that cannot run says so
                row[name] = {'error': f'{type(exc).__name__}: {str(exc)[:80]}'}
        out.append(row)
        print(f'[decompose] truncation_order={truncation_order}, shape_nvf_grid_size={shape_nvf_grid_size}: '
              + '  '.join(f'{k} {v.get("modeling_ms", float("nan")):.1f}/'
                          f'{v.get("solve_ms", float("nan")):.1f} ms'
                          for k, v in row.items() if isinstance(v, dict)), flush=True)
        checkpoint({'decomposition': out})
    return out


def sweep_recompile_shapes(budget, reps):
    """Where a re-freeze recompiles on the shape path, and where it never does.

    The block path recompiled when two of its grid edges crossed. A shape layer's plan holds
    the combinatorics of the *painting*, so a lone shape has nothing that can change, and a
    two-shape layer changes exactly once -- when they start or stop overlapping. Both are here,
    because "never" is the more surprising half and only the pair shows the boundary.
    """
    import jax
    import jax.numpy as jnp
    cfg = dict(period=[300.0, 300.0], thickness=[205.0], truncation_order=[3, 3], top=1.0,
               bot=1.0, theta=0.15, phi=0.35, psi=TE_PSI, wavelength=900.0, shape_nvf_grid_size=256)

    def one(make, values, label):
        def loss(v, geometry):
            return jnp.sum(_study_rcwa(backend=_backend_arg('jax'), structure=make(v), **cfg)
                           .solve(geometry=geometry).T).real

        g = jax.jit(jax.grad(loss), static_argnums=1)
        freeze = lambda v: _study_rcwa(backend=_backend_arg('jax'), structure=make(jnp.asarray(v)),
                                        **cfg).freeze_geometry()
        jax.clear_caches()
        rows = []
        for v in values:
            t0 = time.perf_counter()
            grad = float(g(jnp.asarray(v), freeze(v)))
            rows.append({'case': label, 'value': v,
                         'ms': (time.perf_counter() - t0) * 1e3, 'grad': grad})
            print(f'[recompile:{label}] {v}: {rows[-1]["ms"]:.0f} ms', flush=True)
        return rows

    lone = lambda lx: [dict(background=EPS_HIGH,
                            shapes=[dict(type='ellipse', cx=0.0, cy=0.0, lx=lx, ly=55.0,
                                         angle=0.27, material=EPS_MID)])]
    # Two rods 120 apart, each `lx` wide: they meet as `lx` passes 120.
    pair = lambda lx: [dict(background=EPS_HIGH, shapes=[
        dict(type='rectangle', cx=-60.0, cy=0.0, lx=lx, ly=55.0, angle=0.0,
             material=EPS_MID),
        dict(type='rectangle', cx=60.0, cy=0.0, lx=lx, ly=55.0, angle=0.0,
             material=EPS_LOW)])]

    out = one(lone, [85.0, 90.0, 95.0, 100.0, 103.0, 106.0, 110.0, 115.0, 120.0, 125.0],
              'one shape')
    if budget.can_afford(180):
        out += one(pair, [100.0, 105.0, 110.0, 115.0, 118.0, 122.0, 126.0, 130.0, 140.0],
                   'two shapes')
    checkpoint({'recompile_shapes': out})
    return out


# `sweep_stale_window` lived here until 2026-08-27: it bisected how far a *held* freeze stays
# valid, per shape and per `n_split`. The block modeller it measured is gone, and `n_split`
# with it, so the sweep could not be renamed -- its whole abscissa was that parameter. The
# question it asked is still real on the boundary route, where the plan holds combinatorics
# rather than grid lines and what makes it stale is shapes starting or stopping overlapping;
# re-earning the answer there is a measurement, not a rename, and it has not been taken.


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
            ('shapes', 'Shape path — d(efficiency)/d(length)', 'length'),
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

    if 'decomposition' in state:
        rows = []
        for e in state['decomposition']:
            cells = []
            for who in ('torch', 'jax'):
                m = e.get(who, {})
                cells += ['—' if 'error' in m else f'{m["modeling_ms"]:.1f}',
                          '—' if 'error' in m else f'{m["solve_ms"]:.1f}']
            rows.append([str(e['truncation_order']), str(e['shape_nvf_grid_size'])] + cells)
        table(rows, 'Where the time goes — one forward call, split in two',
              ['truncation_order', 'shape_nvf_grid_size', 'torch modeling', 'torch solve',
               'jax modeling', 'jax solve'])

    if 'recompile_shapes' in state:
        rows = [[e['case'], f'{e["value"]:.0f}', f'{e["ms"]:.0f}', f'{e["grad"]:+.3e}']
                for e in state['recompile_shapes']]
        table(rows, 'Re-freezing every step — where a recompilation lands, and where none does',
              ['case', 'lx', 'ms', 'grad'])

    if 'drift' in state:
        rows = [[e['at'], f'{e["ms"]:.0f}'] for e in state['drift']]
        first = state['drift'][0]['ms']
        rows.append(['spread', f'{max(e["ms"] for e in state["drift"]) / min(e["ms"] for e in state["drift"]):.2f}x'])
        table(rows, 'Machine drift over the run — one fixed configuration, timed three times',
              ['at', 'ms'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='a few points, for a smoke test')
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--budget', type=float, default=7200, help='wall-clock allowance, seconds')
    ap.add_argument('--root', help='import wavebit from this checkout (src or flat layout)')
    args = ap.parse_args()

    _study_enable_x64()
    _study_module('wavebit.rcwa._backends.torch_backend', 'wavebit.backends.torch_backend')

    budget = Budget(args.budget)
    print(f'# Wavebit backend benchmark  (budget {args.budget:.0f} s, reps {args.reps})\n',
          flush=True)

    warm_up(5 if args.quick else 45)
    marks = []
    drift('start', marks)
    sweep_shapes(budget, args.reps, args.quick)
    drift('after shapes', marks)
    sweep_raster(budget, args.reps, args.quick)
    drift('after raster', marks)
    sweep_recompile(budget, args.reps)
    sweep_decomposition(budget, args.reps)
    if not args.quick:
        sweep_recompile_shapes(budget, args.reps)

    print(f'\nfinished with {budget.left():.0f} s of budget left\n')
    report(_STATE)
    print(f'\nraw results: {RESULTS}')


if __name__ == '__main__':
    main()
