"""The stage-0 public contract of ``wavebit.fd``: what it takes, what it refuses, and by name.

Not a physics gate. This pins the doors: every constructor is keyword-only and refuses an
unknown keyword with the nearest valid name; a keyword of a later stage is refused naming
the stage; every axis of ``Boundaries`` is required; ``run(plan=)`` refuses a plan built
for another scene; the skeleton hash is stable, moves on a static change and not on a
traced one; ``import wavebit`` and ``import wavebit.fd`` load no JAX; and the test hooks
the later gates lean on (``fd._testing``) do what they say.

Two runtime contracts from the design are pinned here as well, because no physics gate can
see them. **NumPy works in preallocated buffers**: H, E and the curl scratch live across
steps and every full-volume result is written with ``out=``, so one step's transient
allocation is a fraction of a volume (measured 0.26 volumes at N = 32: NumPy's fixed-size
ufunc buffer on the strided views, ~197 kB whatever N, not a temporary array; a mutant that
updates out of place costs 4.0 volumes per step and one that stacks the curl 3.7). **JAX
compiles the executor once per plan**: a fresh ``jax.jit`` per ``fd.run`` retraced the whole
scan every call (250 ms against 5 ms for NumPy on a 16^3, 20-step box), which a sweep or a
gradient loop would pay every iteration.
"""
import dataclasses
import subprocess
import sys
import tracemalloc

import numpy as np
import pytest

import wavebit
import wavebit.fd as fd
from wavebit.fd import _testing
from wavebit.fd._kernel.step import initial_state, step
from wavebit.fd._lower import lower
from wavebit.fd._runtime import runtime
from fd_helpers import cavity_box, to_numpy

PY = sys.executable
#: One NumPy step may allocate at most this fraction of a field volume, transiently
#: (measured 0.26; the out-of-place mutants sit at 3.7 and 4.0).
NUMPY_TRANSIENT_VOLUMES = 0.5


def _scene(**changes):
    scene, _, _ = cavity_box('pec', N=8, steps=20)
    return scene.replace(**changes) if changes else scene


def test_unknown_keywords_are_refused_with_the_nearest_name():
    with pytest.raises(TypeError, match="did you mean 'size'"):
        fd.Domain(sizes=(1, 1, 1))
    with pytest.raises(TypeError, match="did you mean 'steps'"):
        fd.Run(step=10)
    with pytest.raises(TypeError, match="did you mean 'parity'"):
        fd.Symmetry(parities=1)
    with pytest.raises(TypeError, match="did you mean 'monitors'"):
        _scene().replace(monitor=())


def test_positional_arguments_are_refused_except_a_records_name():
    with pytest.raises(TypeError, match='keyword arguments only'):
        fd.Domain((1, 1, 1))
    r = fd.Record('probe', at=(0, 0, 0))
    assert r.name == 'probe' and r.component == 'Ey'


def test_later_stages_are_refused_by_name():
    base = _scene()
    kw = dict(domain=base.domain, boundaries=base.boundaries, resolution=base.resolution,
              run=base.run)
    # Stage 2 made ``sources=`` and ``Run(periods=)`` legal; their refusals live in
    # tests/test_fd_sources_doors.py. The stage-7 keyword is still refused by its stage.
    with pytest.raises(NotImplementedError, match='stage 7'):
        fd.Scene(decay=1e-6, **kw)
    # ``materials=`` and ``pml=`` are stage-1 features spelled another way, so the refusal
    # now points at the spelling rather than at the stage; ``Boundaries.open`` is the
    # absorber preset as of stage 1 and is pinned in tests/test_fd_scene.py instead.
    with pytest.raises(NotImplementedError, match='stage 1'):
        fd.Scene(materials=(), **kw)
    with pytest.raises(NotImplementedError, match='Boundaries'):
        fd.Scene(pml=0.5, **kw)
    with pytest.raises(NotImplementedError, match='stage 7'):
        fd.GaussianBeam()


def test_every_axis_is_required():
    with pytest.raises(TypeError, match="'z'"):
        fd.Boundaries(x=fd.PEC(), y=fd.PEC())
    with pytest.raises(TypeError, match='face type'):
        fd.Boundaries(x='pec', y=fd.PEC(), z=fd.PEC())


def test_the_declaration_is_validated_at_construction():
    with pytest.raises(ValueError, match='positive'):
        fd.Domain(size=(1, -1, 1))
    with pytest.raises(ValueError, match='CFL'):
        fd.Run(steps=10, courant=1.5)
    with pytest.raises(ValueError, match='outside'):
        _scene(monitors=(fd.Record('far', at=(3.0, 0, 0)),))
    with pytest.raises(ValueError, match='unique'):
        _scene(monitors=(fd.Record('a', at=(0, 0, 0)), fd.Record('a', at=(0, 0, 0))))
    with pytest.raises(ValueError, match='at least one'):
        fd.InitialField()
    with pytest.raises(ValueError, match='mirror eigenvalue'):
        fd.Symmetry(parity=2)


def test_backends_are_named_and_the_enum_is_accepted():
    scene = _scene()
    with pytest.raises(ValueError, match="Did you mean 'torch'"):
        fd.run(scene, backend='torhc')
    a = fd.run(scene, backend=wavebit.Backend.NUMPY)
    b = fd.run(scene, backend='numpy')
    assert a.backend == b.backend == 'numpy'
    assert np.array_equal(a.final.E, b.final.E)


def test_run_refuses_a_plan_built_for_another_scene():
    scene = _scene()
    other = fd.plan(scene.replace(run=fd.Run(steps=21)))
    with pytest.raises(ValueError, match='different scene'):
        fd.run(scene, plan=other)
    res = fd.run(scene, plan=fd.plan(scene))
    assert res.plan.skeleton == fd.plan(scene).skeleton


def test_the_skeleton_moves_on_static_changes_and_not_on_traced_ones():
    scene = _scene()
    s = fd.plan(scene).skeleton
    assert fd.plan(scene).skeleton == s
    assert fd.plan(scene.replace(run=fd.Run(steps=21))).skeleton != s
    assert fd.plan(scene.replace(boundaries=fd.Boundaries(
        x=fd.PMC(), y=fd.PEC(), z=fd.PEC()))).skeleton != s
    # a traced leaf: the amplitude, and the values of the initial field
    assert fd.plan(scene.replace(initial=scene.initial.replace(amplitude=2.0))).skeleton == s
    assert fd.plan(scene.replace(initial=fd.InitialField(
        E=np.ones(fd.plan(scene).shape), H=np.zeros(fd.plan(scene).shape)))).skeleton == s
    # None-versus-present is static
    assert fd.plan(scene.replace(initial=fd.InitialField(E=np.ones(fd.plan(scene).shape)))
                   ).skeleton != s


def test_the_skeleton_is_the_same_in_another_process():
    code = ('import sys; sys.path.insert(0, "tests")\n'
            'import wavebit.fd as fd\nfrom fd_helpers import cavity_box\n'
            'print(fd.plan(cavity_box("pec", N=8, steps=20)[0]).skeleton)')
    out = subprocess.run([PY, '-c', code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == fd.plan(_scene()).skeleton


def test_importing_wavebit_or_wavebit_fd_does_not_import_jax():
    code = ('import sys, wavebit, wavebit.fd\n'
            "assert 'jax' not in sys.modules and 'torch' not in sys.modules\n"
            'print("ok")')
    out = subprocess.run([PY, '-c', code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == 'ok'


def test_the_domain_is_snapped_up_and_the_report_says_so(capsys):
    scene = _scene(domain=fd.Domain(size=(1.0, 1.0, 1.01)))
    report = fd.check(scene)
    assert report.cells == (8, 8, 9)
    assert report.size == pytest.approx((1.0, 1.0, 9 / 8))
    assert any('snapped' in w for w in report.warnings)
    assert 'snapped' in capsys.readouterr().out
    assert report.bytes_per_state == 2 * 3 * 8 * 8 * 9 * 8


def test_results_carry_series_times_and_final_fields():
    scene = _scene(monitors=(fd.Record('e', at=(0.1, 0.0, -0.05), component='Ey'),
                             fd.Record('h', at=(0.1, 0.0, -0.05), component='Hz')))
    res = fd.run(scene)
    dt = res.plan.dt
    assert np.allclose(res['e'].t, (np.arange(20) + 1) * dt)
    assert np.allclose(res['h'].t, (np.arange(20) + 0.5) * dt)
    assert res.final.E.shape == res.final.H.shape == (3, 8, 8, 8)
    assert sorted(res) == ['e', 'h']
    with pytest.raises(KeyError, match='no monitor named'):
        res['x']


def test_a_records_yee_point_is_the_nearest_of_its_component():
    plan = fd.plan(_scene(monitors=(fd.Record('a', at=(0.13, 0.0, -0.05), component='Ex'),
                                    fd.Record('b', at=(0.13, 0.0, -0.05), component='Ey'))))
    a, b = plan.observers
    x_a = plan.grid.yee_points('E', 0)[0][a.i]
    x_b = plan.grid.yee_points('E', 1)[0][b.i]
    assert abs(x_a - 0.13) <= plan.dx / 2 + 1e-12
    assert abs(x_b - 0.13) <= plan.dx / 2 + 1e-12
    assert x_a != x_b                       # Ex sits half a cell along x from Ey


def test_the_numpy_kernel_steps_in_preallocated_buffers():
    """One step allocates under half a volume, transiently, and E and H keep their identity.

    The 0.26 volumes the kernel does allocate is NumPy's ufunc buffer on the non-contiguous
    views the differences write into (a constant ~197 kB, the same at N = 64), not an
    array of the field's size; a reader who sees it in a profile should not chase it.
    """
    scene, _, _ = cavity_box('pec', N=32, steps=8)            # stepped by hand below, 4 times
    plan = fd.plan(scene)
    rt = runtime('numpy')
    params = lower(scene, plan, rt)
    state = initial_state(plan, params, rt)
    E, H = state['E'], state['H']
    volume = 3 * plan.cells * np.dtype(plan.dtype).itemsize
    state = step(plan, params, state, 0, rt)                  # warm: the scratch exists now
    tracemalloc.start()
    try:
        before, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        for n in range(1, 4):
            state = step(plan, params, state, n, rt)
        after, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert (peak - before) < NUMPY_TRANSIENT_VOLUMES * volume, (peak - before) / volume
    assert (after - before) < 0.01 * volume                    # nothing kept per step
    assert state['E'] is E and state['H'] is H


def test_the_preallocation_holds_when_the_coefficients_are_arrays():
    """The same budget with ``ae``/``be``/``ah``/``bh`` at full ``(3, nz, ny, nx)`` shape.

    The vacuum cavity above cannot see the thing this exists for: there every coefficient is a
    Python float, so an expression in the step body that allocates one array per coefficient
    per step costs nothing. Stage 1 makes them arrays as soon as an ``Absorber`` (or a set of
    structures that varies on all three axes) says so, and one such expression -- ``-bh``,
    negated inside the step because the H update subtracts -- took the transient from 0.14
    volumes a step to 1.00. It is lowered once as ``minus_bh`` now
    (``wavebit/fd/_lower/coefficients.py``), and this is the case that would have objected.
    """
    # 40^3, not the 32^3 above: the transient the kernel legitimately carries is a *constant*
    # ~200 kB of ufunc buffer, so the budget below is only a statement about volumes once the
    # volume is well past it. At 24^3 the constant alone is 0.61 volumes.
    scene = fd.Scene(domain=fd.Domain(size=(1.0, 1.0, 1.0)),
                     boundaries=fd.Boundaries.open(fd.Absorber(cells=12)),
                     resolution=fd.Resolution(cells_per_wavelength=40, wavelength=1.0),
                     run=fd.Run(steps=8), initial=None, monitors=())
    plan = fd.plan(scene)
    rt = runtime('numpy')
    params = lower(scene, plan, rt)
    assert all(np.asarray(params[k]).shape == plan.shape
               for k in ('ae', 'be', 'ah', 'bh', 'minus_bh')), 'the premise: full-volume arrays'
    state = initial_state(plan, params, rt)
    E, H = state['E'], state['H']
    volume = 3 * plan.cells * np.dtype(plan.dtype).itemsize
    for n in range(3):
        state = step(plan, params, state, n, rt)              # warm: every scratch exists
    tracemalloc.start()
    try:
        before, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        for n in range(3, 6):
            state = step(plan, params, state, n, rt)
        after, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert (peak - before) < NUMPY_TRANSIENT_VOLUMES * volume, (peak - before) / volume
    assert (after - before) < 0.01 * volume
    assert state['E'] is E and state['H'] is H


@dataclasses.dataclass(frozen=True)
class _Tally:
    """A term that counts how often the kernel body is traced: once per compile on JAX."""

    name: str
    traces = []                                               # shared, on purpose

    def init_mem(self, plan, rt):
        return None

    def before_H(self, cE, E, mem, n, params, rt):
        self.traces.append(n)
        return cE

    def before_E(self, cH, H, mem, n, params, rt):
        return cH

    def advance(self, mem, E, H, n, params, rt):
        return mem


def test_the_jax_executor_is_compiled_once_per_plan():
    """Two runs of one scene trace the body once; a different plan traces it again."""
    pytest.importorskip('jax')
    scene = _scene()
    tally = _Tally('tally')
    tally.traces.clear()
    _testing.run_with_terms(scene, [tally], backend='jax')
    first = len(tally.traces)
    assert first >= 1
    _testing.run_with_terms(scene, [tally], backend='jax')
    assert len(tally.traces) == first, 'the second run of the same plan retraced the body'
    _testing.run_with_terms(scene.replace(run=fd.Run(steps=21)), [tally], backend='jax')
    assert len(tally.traces) > first, 'a different plan did not compile'
    # and the same runtime comes back, which is where the compiled executors live
    assert runtime('jax') is runtime('jax') and runtime('jax').name == 'jax'
    assert runtime('numpy') is runtime('numpy') and runtime('torch') is runtime('torch')
    assert runtime('jax') is not runtime('numpy')


def test_the_testing_hook_injects_terms_that_the_kernel_applies():
    scene = _scene()
    plan = fd.plan(scene)
    slab = _testing.Slab(c=1, z=(2, 5), y=(2, 5), x=(2, 5))
    term = _testing.SlabTerm(name='t', slab=slab, value=0.25, phase='before_E')
    mutated = _testing.with_terms(plan, [term])
    assert mutated.skeleton != plan.skeleton and mutated.terms == (term,)
    with pytest.raises(ValueError, match='different scene'):
        fd.run(scene, plan=mutated)
    plain = fd.run(scene)
    for backend in ('numpy', 'jax', 'torch'):
        res = _testing.run_with_terms(scene, [term], backend=backend)
        assert not np.allclose(to_numpy(res.final.E), to_numpy(plain.final.E))
        assert np.allclose(to_numpy(res.final.E), to_numpy(
            _testing.run_with_terms(scene, [term], backend='numpy').final.E), rtol=0, atol=1e-12)
