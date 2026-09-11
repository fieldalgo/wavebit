"""A JAX solve runs on the device it was given -- every array, not only `solver.device`.

`solver.device` is the one thing that cannot answer this: it reports what was asked for.
Measured 2026-09-10 on a CUDA jaxlib (jax 0.11.0): `rcwa(backend=1, device='cpu')` reported
`[CpuDevice(id=0)]` while its structure, its convolution matrices and the `R` it returned all
lived on `cuda:0`, and T differed from a CPU-only jaxlib's in the 13th digit
(8.77000489428177987e-01 against 8.77000489428339414e-01 on a 2D raster). Omitting `device`,
whose documented default is the CPU, did the same. On a machine with one device every
placement is correct, so nothing on a CPU-only box could see it, and an A/B driven by
`device=` compared the GPU with itself.

Two instruments, because each reaches what the other cannot:

* **Two host devices, in a subprocess.** `--xla_force_host_platform_device_count=2` gives any
  machine a second device, so "landed on JAX's default instead of the requested one" is
  observable without a GPU. The flag is read when JAX initialises, hence the subprocess. Asked
  for `cpu:1`, everything must be on `cpu:1`; with JAX's default moved to `cpu:1`, omitting
  `device` must still put everything on `cpu:0`, which is what `0` -- the CPU -- resolves to.
* **The GPU's allocation counter**, where there is a GPU. Listing arrays checks the ones
  somebody thought to list; `memory_stats()['num_allocs']` counts every buffer any entry point
  put on the card, listed or not. A CPU solve must leave it where it was.
"""
import os
import pathlib
import subprocess
import sys
import textwrap

import numpy as np
import pytest

pytest.importorskip('jax')

P, WL, H = 700.0, 900.0, 300.0


def _foreign():
    """Values a caller built with JAX on its *current* default device, not the solver's.

    They enter through doors no setter's placement reached before: a half-space Material, a
    shape layer's background, a shape's coordinate. `jax.grad` builds exactly these when it is
    handed a Python float, which is how `test_analytic_shape_gradients[background-jax]` crashed
    once the solver's own arrays were placed and a caller's GPU background was not.
    """
    import wavebit
    wavebit.enable_x64()
    import jax.numpy as jnp
    return dict(top=jnp.asarray(1.0), background=jnp.asarray(1.0), lx=jnp.asarray(250.0),
                wavelength=jnp.asarray(WL))


def _solvers(device_kw, foreign):
    """Three scenes, chosen for the routes an array takes into a solve rather than for physics.

    * ``1d`` -- the 1D operator, and the scene ``grad`` differentiates.
    * ``magnetic`` -- a 2D raster with a permeability field, a magnetic half-space, a
      multimode excitation and ``answers='field'``: both convolution builders and both field
      paths, the jitted one included.
    * ``shapes`` -- the shape path, whose coefficients are built from Python floats and so
      have no stored array to follow: the device scope is the only thing that places them.
    """
    import wavebit
    rng = np.random.default_rng(0)
    eps = np.where(rng.random((2, 8, 8)) < 0.4, 5.76, 1.0)
    mu = np.where(rng.random((2, 8, 8)) < 0.3, 1.8, 1.0)
    shapes = [dict(background=foreign['background'], shapes=[
        dict(type='ellipse', cx=0.0, cy=0.0, lx=foreign['lx'], ly=150.0, angle=0.7,
             material=5.76),
        dict(type='circle', cx=-200.0, cy=-250.0, lx=80.0, material=9.0)])]
    return {
        '1d': wavebit.rcwa(backend=1, structure=np.array([[[1.0, 1.0, 12.25, 12.25]]]),
                           period=[P], truncation_order=dict(x=6, y=0),
                           wavelength=foreign['wavelength'],
                           top=1.0, bot=2.25, theta=0.2, phi=0.0, tetm='te', thickness=[H],
                           **device_kw),
        'magnetic': wavebit.rcwa(backend=1, structure=eps, mu=mu, period=[P, 0.9 * P],
                                 truncation_order=[2, 2], thickness=[H, 0.5 * H],
                                 wavelength=WL, top=wavebit.Material(2.25, mu=1.4),
                                 bot=wavebit.Material(4.0), theta=0.31, phi=0.47, psi=0.29,
                                 layer_algorithm='SMM', answers='field',
                                 excitation={(0, 0): 1.0, (-1, 0): 0.5j}, **device_kw),
        'shapes': wavebit.rcwa(backend=1, structure=shapes, period=[P, 1.2 * P],
                               truncation_order=[3, 3], thickness=[H], wavelength=WL,
                               top=foreign['top'], bot=2.25, theta=0.1, phi=0.3, psi=0.4,
                               **device_kw),
    }


def _touch(device_kw, foreign=None):
    """Drive every public entry point; return ``({tag: array}, {tag: what it raised})``.

    The solver itself is read as a pytree before and after, because its children are what
    the setters built -- the arrays a constructor makes before any decorated method runs.

    A raise is recorded rather than propagated, because a misplacement does not always stay
    quiet: with two devices in play the unfixed code *crashed* in `reconstruct_field`
    (`Received incompatible devices for jitted computation`, one operand committed to each),
    and on a CUDA box it did so for every 2D field reconstruction that did not ask for the GPU.
    Stopping at the first crash would hide what every other entry point did.
    """
    import jax
    import jax.numpy as jnp
    found, raised = {}, {}

    def take(tag, build):
        try:
            tree = build()
        except Exception as exc:
            raised[tag] = f'{type(exc).__name__}: {str(exc).splitlines()[0][:200]}'
            return
        for i, leaf in enumerate(jax.tree_util.tree_leaves(tree)):
            if isinstance(leaf, jax.Array):
                found[f'{tag}[{i}]'] = leaf

    solvers = _solvers(device_kw, _foreign() if foreign is None else foreign)
    for name, s in solvers.items():
        take(f'{name}/constructed', lambda: s)
        take(f'{name}/eps_conv_matrices', lambda: tuple(s.eps_conv_matrices()))
        take(f'{name}/mu_conv_matrices', s.mu_conv_matrices)
        take(f'{name}/order_kxy', s.order_kxy)
        take(f'{name}/solve', s.solve)
        take(f'{name}/after-solve', lambda: s)
    s = solvers['magnetic']
    take('magnetic/reconstruct_field', lambda: s.reconstruct_field(4, 4, 2))
    take('magnetic/solve_field(jit)',
         lambda: s.solve_field(4, 4, 2, geometry=s.freeze_geometry()))
    s = solvers['shapes']
    take('shapes/solve(geometry=)', lambda: s.solve(geometry=s.freeze_geometry()))
    s = solvers['1d']
    take('1d/grad', lambda: s.grad(['thickness'], s.solve, lambda r: jnp.sum(jnp.real(r.T))))
    jax.block_until_ready(list(found.values()))
    return found, raised


def misplaced(device_kw, expected, foreign=None):
    """Every array `_touch` produced off ``expected``, and every entry point that raised."""
    found, raised = _touch(device_kw, foreign)
    return ([f'{tag} on {sorted(str(d) for d in a.devices())}'
             for tag, a in found.items() if a.devices() != {expected}]
            + [f'{tag} raised {msg}' for tag, msg in raised.items()])


def poisoned(first, second):
    """Whether a solve that mixed two devices leaves a cache entry that breaks the next solver.

    The shape path's projector cache is keyed on the device a fresh array lands on -- the
    scope's -- while the build follows whatever committed operand it reads. A shape value
    committed to ``first``, read under ``second``'s scope, filed an entry keyed on ``second``
    that held ``first``'s arrays. The solve that filed it raised at once, but the entry stayed,
    and every later solve of that geometry on ``second`` raised `Received incompatible devices`
    too, on any solver in the process.

    The trigger here is a device change on a live solver, which keeps its stored arrays where
    they were. Whether that solve itself answers is not asked: moving a live solver is a
    separate question. What is asked is only that a fresh solver on ``second`` is not handed
    ``first``'s arrays.
    """
    import wavebit
    wavebit.enable_x64()
    import jax
    import jax.numpy as jnp
    from wavebit._core import parametric

    def solver(lx, device):
        shapes = [dict(background=1.0, shapes=[
            dict(type='ellipse', cx=0.0, cy=0.0, lx=lx, ly=150.0, angle=0.7, material=5.76),
            dict(type='circle', cx=-200.0, cy=-250.0, lx=80.0, material=9.0)])]
        return wavebit.rcwa(backend=1, structure=shapes, period=[P, 1.2 * P],
                            truncation_order=[2, 2], thickness=[H], wavelength=WL, top=1.0,
                            bot=2.25, theta=0.1, phi=0.3, psi=0.4, normal_grid=64,
                            device=device)

    parametric._PROJECTOR_CACHE.clear()
    moved = solver(jnp.asarray(250.0), first)          # the shape value is committed to `first`
    moved.solve()
    moved.device = second
    try:
        moved.solve()
    except Exception:                                  # noqa: BLE001 - not what is measured
        pass
    try:
        result = solver(250.0, second).solve()
    except Exception as exc:                           # noqa: BLE001 - reported, not hidden
        return (f'a fresh solver on {second} raised {type(exc).__name__}: '
                f'{str(exc).splitlines()[0][:200]}')
    leaves = [leaf for leaf in jax.tree_util.tree_leaves(result) if isinstance(leaf, jax.Array)]
    off = sorted({str(d) for leaf in leaves for d in leaf.devices()} - {str(second)})
    return f'a fresh solver on {second} answered on {off}' if off else 'OK'


_CHILD = """
    import sys
    sys.path.insert(0, {tests!r})
    import jax
    import jax.numpy as jnp
    cpus = jax.devices('cpu')
    assert len(cpus) == 2, f'the forced second host device did not appear: {{cpus}}'
    from test_jax_device_placement import misplaced

    # Asked for the second device by object; JAX's default is still the first.
    bad = misplaced(dict(device=cpus[1]), cpus[1])

    # JAX's default moved to the second device. The documented default and 'cpu' both
    # resolve to jax.devices('cpu')[0], so everything must stay on the first -- the CUDA
    # box's situation, where the default device is the GPU and the request is the CPU.
    jax.config.update('jax_default_device', cpus[1])
    assert jnp.zeros(1).devices() == {{cpus[1]}}, 'the default did not move: vacuous check'
    bad += misplaced({{}}, cpus[0]) + misplaced(dict(device='cpu'), cpus[0])
    print('\\n'.join(bad) if bad else 'OK')
"""

_POISON_CHILD = """
    import sys
    sys.path.insert(0, {tests!r})
    import jax
    cpus = jax.devices('cpu')
    assert len(cpus) == 2, f'the forced second host device did not appear: {{cpus}}'
    from test_jax_device_placement import poisoned
    print(poisoned(cpus[1], cpus[0]))
"""


def _two_host_devices(child):
    """Run ``child`` in a fresh interpreter with two host devices; return its stdout lines."""
    tests = pathlib.Path(__file__).resolve().parent
    env = {k: v for k, v in os.environ.items() if not k.startswith('JAX')}
    env['JAX_PLATFORMS'] = 'cpu'
    env['XLA_FLAGS'] = (env.get('XLA_FLAGS', '') +
                        ' --xla_force_host_platform_device_count=2').strip()
    env['PYTHONPATH'] = str(tests.parent)
    out = subprocess.run([sys.executable, '-c', textwrap.dedent(child.format(tests=str(tests)))],
                         capture_output=True, text=True, env=env, cwd=str(tests.parent))
    assert out.returncode == 0, out.stderr[-3000:]
    return out.stdout.strip().splitlines()



def test_every_array_lands_on_the_requested_device_not_on_jaxs_default():
    """Two host devices, so the requested device and JAX's default can differ anywhere.

    Before 2026-09-11 both halves failed with every array on the default device: the setters
    placed nothing, and the builders then *committed* the convolution matrices to wherever the
    structure happened to be, which overruled the scope `solve_conv` did open.
    """
    lines = _two_host_devices(_CHILD)
    assert lines and lines[-1] == 'OK', 'placed on the wrong device:\n' + '\n'.join(lines)[-3000:]


def test_the_projector_cache_hands_no_solver_another_devices_arrays():
    """A solve that mixed devices must not leave an entry a correctly placed solver is served.

    Measured 2026-09-11 on two host devices: a solver built on `cpu:1` with a shape value
    committed there, moved to `cpu:0` and solved, raised -- and left an entry keyed `cpu:0`
    holding `cpu:1` arrays, so a fresh `cpu:0` solver of the same geometry raised too until
    `_PROJECTOR_CACHE` was cleared by hand. The GPU half of this is the next test.
    """
    lines = _two_host_devices(_POISON_CHILD)
    assert lines and lines[-1] == 'OK', '\n'.join(lines)[-3000:]


def test_the_projector_cache_hands_no_solver_another_devices_arrays_cpu_and_gpu():
    """The same, between the CPU and the GPU, in both directions -- where a GPU is visible."""
    import jax
    gpus = [d for d in jax.devices() if d.platform == 'gpu']
    if not gpus:
        pytest.skip('no GPU visible to JAX; the two-host-device test covers this here')
    cpu = jax.devices('cpu')[0]
    assert poisoned(gpus[0], cpu) == 'OK'
    assert poisoned(cpu, gpus[0]) == 'OK'



def test_a_cpu_solve_puts_nothing_on_the_gpu():
    """Where a GPU is visible: `device='cpu'` and an omitted `device` allocate nothing on it.

    The counter is proven live in the same test -- a `device='gpu'` solve must move it and
    land there -- so a reading that cannot change is not mistaken for a pass.
    """
    import jax
    gpus = [d for d in jax.devices() if d.platform == 'gpu']
    if not gpus:
        pytest.skip('no GPU visible to JAX; the two-host-device test covers placement here')

    def allocations():
        return (gpus[0].memory_stats() or {}).get('num_allocs')

    if allocations() is None:
        pytest.skip('this jaxlib reports no GPU allocation count')
    cpu = jax.devices('cpu')[0]
    # The caller's own arrays, built on the GPU before counting starts: they are the caller's
    # allocations, and what is measured is whether the solver makes any more of its own.
    foreign = _foreign()

    before = allocations()
    bad = misplaced({}, cpu, foreign) + misplaced(dict(device='cpu'), cpu, foreign)
    during_cpu = allocations() - before
    assert not bad, 'a CPU solve placed arrays elsewhere:\n' + '\n'.join(bad)
    assert during_cpu == 0, f'{during_cpu} GPU allocations during solves asked to run on the CPU'

    before = allocations()
    bad = misplaced(dict(device='gpu'), gpus[0], foreign)
    assert not bad, 'a GPU solve placed arrays elsewhere:\n' + '\n'.join(bad)
    assert allocations() > before, 'the counter did not move for a GPU solve: it measures nothing'
