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
        'magnetic': wavebit.rcwa(backend=1, structure=eps[::-1], mu=mu[::-1], period=[P, 0.9 * P],
                                 truncation_order=[2, 2], thickness=[0.5 * H, H],
                                 wavelength=WL, top=wavebit.Material(2.25, mu=1.4),
                                 bot=wavebit.Material(4.0), theta=0.31, phi=0.47, psi=0.29,
                                 layer_algorithm='SMM', answers='field',
                                 excitation={(0, 0): 1.0, (-1, 0): 0.5j}, **device_kw),
        'shapes': wavebit.rcwa(backend=1, structure=shapes, period=[P, 1.2 * P],
                               truncation_order=[3, 3], thickness=[H], wavelength=WL,
                               top=foreign['top'], bot=2.25, theta=0.1, phi=0.3, psi=0.4,
                               **device_kw),
    }


def _touch(device_kw, foreign=None, move_to=None):
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
    if move_to is not None:
        # Built on one device and then moved: a live solver's arrays follow `device`, or every
        # entry point below reads two devices at once.
        for s in solvers.values():
            s.device = move_to
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


def misplaced(device_kw, expected, foreign=None, move_to=None):
    """Every array `_touch` produced off ``expected``, and every entry point that raised."""
    found, raised = _touch(device_kw, foreign, move_to)
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
    they were. Whether that solve itself answers is :func:`moved`'s question. What is asked
    here is only that a fresh solver on ``second`` is not handed ``first``'s arrays.
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
                            bot=2.25, theta=0.1, phi=0.3, psi=0.4, shape_nvf_grid_size=64,
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


def moved(first, second):
    """What a solver built on ``first`` and moved to ``second`` gets wrong, against one born there.

    Moving is a promise about the arrays: the moved solver holds nothing on ``first``, and its
    solve is the one a solver built on ``second`` gives -- bit for bit, since the two run the
    same programs on the same device. Both routes are asked: the attribute, and the
    `solve(device=...)` keyword, whose call opens its device scope before the keyword is
    assigned -- eagerly, inside `jax.jit`, and under `jax.grad` with a differentiated keyword
    given *before* ``device``. What the solver holds is read off the whole instance, the
    solve's own state included, right after the move and after each call; a cell basis built
    from JAX arrays is asked about separately, because no other scene stores one. A field
    reconstruction from a solve taken on ``first`` is refused as a changed scene, the way a
    changed width is, and answers on ``second`` once solved again.
    """
    import jax
    bad = []

    def on(tree):
        leaves = [x for x in jax.tree_util.tree_leaves(tree) if isinstance(x, jax.Array)]
        return {str(d) for x in leaves for d in x.devices()}, leaves

    def held_off(s):
        """Each slot of the instance holding something off ``second``, solve state included.

        The pytree's children are not enough: the solve's scattering matrix, its modes and `T1`
        are not children, and a partial move that skipped them passed a children-only check.
        Named by slot, so that one solver's findings can be compared with another's.
        """
        import dataclasses
        found = set()
        for slot, value in vars(s).items():
            stack = [value]
            while stack:
                v = stack.pop()
                if isinstance(v, jax.core.Tracer):
                    found.add(f'{slot} holds a tracer')
                elif isinstance(v, jax.Array):
                    found |= {f'{slot} on {d}' for d in v.devices() if str(d) != str(second)}
                elif isinstance(v, dict):
                    stack.extend(v.values())
                elif isinstance(v, (list, tuple)):
                    stack.extend(v)
                elif dataclasses.is_dataclass(v) and not isinstance(v, type):
                    stack.extend(getattr(v, f.name) for f in dataclasses.fields(v))
        return sorted(found)

    def same(a, b):
        return len(a) == len(b) and all(
            x.shape == y.shape and x.dtype == y.dtype
            and np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b))

    want = {name: on(s.solve())[1]
            for name, s in _solvers(dict(device=second), _foreign()).items()}
    by_attribute = _solvers(dict(device=first), _foreign())
    by_keyword = _solvers(dict(device=first), _foreign())
    for name in want:
        by_attribute[name].solve()
        by_attribute[name].device = second
        off = held_off(by_attribute[name])
        if off:
            bad.append(f'{name}: right after the move the solver still holds arrays on {off}')
        routes = (('attribute', by_attribute[name].solve),
                  ('keyword', lambda n=name: by_keyword[n].solve(device=second)))
        for route, run in routes:
            try:
                got = run()
            except Exception as exc:                   # noqa: BLE001 - reported, not hidden
                bad.append(f'{name}/{route} raised {type(exc).__name__}: '
                           f'{str(exc).splitlines()[0][:160]}')
                continue
            devices, leaves = on(got)
            if devices != {str(second)}:
                bad.append(f'{name}/{route} answered on {sorted(devices)}')
            elif not same(leaves, want[name]):
                bad.append(f'{name}/{route} differs from a solver born on {second}')
        for route, s in (('attribute', by_attribute[name]), ('keyword', by_keyword[name])):
            off = held_off(s)
            if off:
                bad.append(f'{name}/{route}: after its solve the solver holds arrays on {off}')

    s = _solvers(dict(device=first), _foreign())['magnetic']
    s.solve()
    s.device = second
    try:
        s.reconstruct_field(4, 4, 2)
        bad.append('magnetic: a field was reconstructed from a solve taken on the old device')
    except RuntimeError as exc:
        if 'scene has changed' not in str(exc):
            bad.append(f'magnetic: refused for another reason: {str(exc).splitlines()[0][:160]}')
    except Exception as exc:                           # noqa: BLE001 - reported, not hidden
        bad.append(f'magnetic: reconstruct raised {type(exc).__name__}: '
                   f'{str(exc).splitlines()[0][:160]}')
    try:
        s.solve()
        devices = on(s.reconstruct_field(4, 4, 2))[0]
        if devices != {str(second)}:
            bad.append(f'magnetic: the field after re-solving is on {sorted(devices)}')
    except Exception as exc:                           # noqa: BLE001 - reported, not hidden
        bad.append(f'magnetic: re-solve and reconstruct raised {type(exc).__name__}: '
                   f'{str(exc).splitlines()[0][:160]}')

    import jax.numpy as jnp
    import wavebit

    def oblique(device):
        rng = np.random.default_rng(1)
        eps = np.where(rng.random((1, 8, 8)) < 0.4, 5.76, 1.0)
        basis = wavebit.Lattice(jnp.asarray([P, 0.0]), jnp.asarray([0.2 * P, 0.9 * P]), False)
        return wavebit.rcwa(backend=1, structure=eps, lattice=basis, truncation_order=[2, 2],
                            thickness=[H], wavelength=WL, factorization='nvf', device=device)

    try:
        s = oblique(first)
        s.solve()
        s.device = second
        off = held_off(s)
        if off:
            bad.append(f'oblique: right after the move the solver still holds arrays on {off}')
        if not same(on(s.solve())[1], on(oblique(second).solve())[1]):
            bad.append(f'oblique: differs from a solver born on {second}')
    except Exception as exc:                           # noqa: BLE001 - reported, not hidden
        bad.append(f'oblique raised {type(exc).__name__}: {str(exc).splitlines()[0][:160]}')

    def total(r):
        return jnp.sum(jnp.real(r.T))

    def exact(a, b):
        return np.array_equal(np.asarray(a), np.asarray(b))

    # Inside `jax.jit`: the move is a fact about the solver, carried out when the trace runs.
    for name in ('1d', 'magnetic', 'shapes'):
        try:
            s = _solvers(dict(device=first), _foreign())[name]
            born = _solvers(dict(device=second), _foreign())[name]
            got = jax.jit(lambda w: total(s.solve(device=second, wavelength=w)))(WL)
            ref = jax.jit(lambda w: total(born.solve(wavelength=w)))(WL)
            if {str(d) for d in got.devices()} != {str(second)}:
                bad.append(f'{name}/jit answered on {sorted(str(d) for d in got.devices())}')
            elif not exact(got, ref):
                bad.append(f'{name}/jit gave {float(got)!r} against {float(ref)!r} born there')
            # A jitted solve that keeps its field state leaves that state traced on ANY solver,
            # moved or not -- `T1`, the modes and the scattering matrix of the magnetic scene,
            # measured on develop e8fcf67 with no move involved, and kept from use by the
            # reconstruction's refusal of a traced scene. What is asked here is only that the
            # move adds nothing to it.
            off = sorted(set(held_off(s)) - set(held_off(born)))
            if off:
                bad.append(f'{name}/jit: after the call the moved solver, unlike one born '
                           f'there, holds {off}')
            if not same(on(s.solve())[1], on(born.solve())[1]):
                bad.append(f'{name}/jit: the eager solve after it differs from one born there')
        except Exception as exc:                       # noqa: BLE001 - reported, not hidden
            bad.append(f'{name}/jit raised {type(exc).__name__}: {str(exc).splitlines()[0][:160]}')

    # Under `jax.grad`, a differentiated keyword given before `device`: its setter placed it
    # on the old device, so the move has to take the tracer as well.
    try:
        s = _solvers(dict(device=first), _foreign())['1d']
        born = _solvers(dict(device=second), _foreign())['1d']
        got = jax.grad(lambda w: total(s.solve(wavelength=w, device=second)))(WL)
        ref = jax.grad(lambda w: total(born.solve(wavelength=w)))(WL)
        if not exact(got, ref):
            bad.append(f'1d/grad gave {float(got)!r} against {float(ref)!r} born there')
        off = held_off(s)
        if off:
            bad.append(f'1d/grad: after the call the solver holds arrays on {off}')
        if not same(on(s.solve())[1], on(born.solve())[1]):
            bad.append('1d/grad: the eager solve after it differs from one born there')
    except Exception as exc:                           # noqa: BLE001 - reported, not hidden
        bad.append(f'1d/grad raised {type(exc).__name__}: {str(exc).splitlines()[0][:160]}')
    return bad


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
    # Built on the first device and moved to the second: everything follows the move.
    bad += misplaced(dict(device=cpus[0]), cpus[1], move_to=cpus[1])

    # JAX's default moved to the second device. The documented default and 'cpu' both
    # resolve to jax.devices('cpu')[0], so everything must stay on the first -- the CUDA
    # box's situation, where the default device is the GPU and the request is the CPU.
    jax.config.update('jax_default_device', cpus[1])
    assert jnp.zeros(1).devices() == {{cpus[1]}}, 'the default did not move: vacuous check'
    bad += misplaced({{}}, cpus[0]) + misplaced(dict(device='cpu'), cpus[0])
    bad += misplaced(dict(device=cpus[1]), cpus[0], move_to=cpus[0])
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

_MOVE_CHILD = """
    import sys
    sys.path.insert(0, {tests!r})
    import jax
    cpus = jax.devices('cpu')
    assert len(cpus) == 2, f'the forced second host device did not appear: {{cpus}}'
    from test_jax_device_placement import moved
    bad = moved(cpus[1], cpus[0]) + moved(cpus[0], cpus[1])
    print('; '.join(bad) if bad else 'OK')
"""


_FIELD_DOOR_CHILD = """
    import jax
    import numpy as np
    import wavebit
    wavebit.enable_x64()
    cpus = jax.devices('cpu')
    assert len(cpus) == 2, f'the forced second host device did not appear: {{cpus}}'
    eps = np.where(np.random.default_rng(0).random((1, 6, 6)) < 0.4, 5.76, 1.0)
    kw = dict(structure=eps, period=[700.0, 700.0], truncation_order=[1, 1], thickness=[300.0],
              wavelength=900.0, theta=0.2, phi=0.1, psi=0.3, answers='field')
    bad = []
    for first, second in ((cpus[0], cpus[1]), (cpus[1], cpus[0])):
        solver = wavebit.rcwa(backend=1, device=first, **kw)
        built = wavebit.rcwa(backend=1, device=second, **kw)
        geometry = built.freeze_geometry()
        got = solver.solve_field(3, 3, 2, geometry=geometry, device=second)
        want = built.solve_field(3, 3, 2, geometry=geometry)
        if solver.device != [second]:
            bad.append(f'{{first}}->{{second}}: solver.device reads {{solver.device}}')
        if not all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in
                   zip(jax.tree_util.tree_leaves(got), jax.tree_util.tree_leaves(want))):
            bad.append(f'{{first}}->{{second}}: answer differs from a solver built there')
        on = {{str(d) for x in jax.tree_util.tree_leaves((got, solver))
              if isinstance(x, jax.Array) for d in x.devices()}}
        if on != {{str(second)}}:
            bad.append(f'{{first}}->{{second}}: arrays on {{sorted(on)}}')
    print('; '.join(bad) if bad else 'OK')
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


def test_a_moved_solver_answers_as_one_built_where_it_was_moved():
    """`solver.device = X` and `solve(device=X)` move a live solver, both ways, on two host devices.

    Measured 2026-09-11 before the move existed: the setter rebound the device and left every
    stored array where the setters had committed it, so the next solve raised `Received
    incompatible devices` -- a new-device order range against an old-device period -- by
    either route. The keyword also opened the old device's scope, before it was assigned.
    """
    lines = _two_host_devices(_MOVE_CHILD)
    assert lines and lines[-1] == 'OK', '; '.join(lines)[-3000:]


def test_the_jitted_field_door_moves_the_solver_it_is_given():
    """`solve_field(device=X, geometry=...)` answers on X, as a solver built there, and stays.

    The device is a condition like any other at this door: assigned on the live solver before
    the compiled call, kept after it, and part of the call's static data, so the new device's
    program is the one that runs. It used to be checked as `device='cpu'` in
    `tests/test_solve_keywords.py`, on a solver already on the CPU, which asserted nothing.
    """
    lines = _two_host_devices(_FIELD_DOOR_CHILD)
    assert lines and lines[-1] == 'OK', '; '.join(lines)[-3000:]


def test_a_moved_solver_answers_as_one_built_where_it_was_moved_cpu_and_gpu():
    """The same, between the CPU and the GPU, in both directions -- where a GPU is visible."""
    import jax
    gpus = [d for d in jax.devices() if d.platform == 'gpu']
    if not gpus:
        pytest.skip('no GPU visible to JAX; the two-host-device test covers this here')
    cpu = jax.devices('cpu')[0]
    bad = moved(cpu, gpus[0]) + moved(gpus[0], cpu)
    assert not bad, '; '.join(bad)

