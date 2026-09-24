"""The scene text that asks for the extended-stencil material discretisation, and its doors.

What this file gates is the *routing*, not the physics. The operator's numbers are the
research study's and are measured in ``docs/fd-symmetric-stencil.md`` and
``tests/test_fd_symmetric_stencil.py``; what is new here is that a public scene can reach
them, that reaching them costs the default nothing, and that every scene the operator cannot
be prepared for is **refused by name** rather than quietly given the other rule.

**The oracles, and why each is independent of the thing it judges**

* *The research driver itself.* ``studies/fdtd_subpixel_symmetric_stencil.py::replace`` is the
  path every published number of this operator was measured through. The public preparation
  and it call **one** ``geometry``, so the comparison is of what each hands it and of the four
  arrays each ends with -- digested, bit for bit, with no tolerance anywhere
  (:func:`test_the_public_preparation_prepares_the_research_operator_bit_for_bit`). That is
  owner decision (C)'s hook: it makes the claim that the public path runs the operator the
  sweep was measured on a *proof about arrays* and not a second sweep.
* *The scene without the selector.* For "the default did not move" the oracle is the same
  scene with ``discretisation='voxel'``, whose plan, skeleton and coefficient arrays are
  compared against the ones built before this commit existed. No tolerance: ``array_equal``.
* *The fit's own report.* For the memory door's row count the oracle is
  ``report['active_edges']`` -- the number the expensive path arrives at -- against the cheap
  screen the door reads before spending anything.
* *The coefficient arrays.* For "the effective medium reaches the diagnostic" the oracle is
  ``InteriorEnergy._weight``, which is the code the run's energy reading really uses, asked
  for the region the operator owns.

**The fixture** is deliberately small: a ball of radius 0.14 at 18 cells per unit wavelength
in a 36-cell box, whose preparation is about three seconds and 1830 active rows. Every
resolution here is a routing question, and a routing question does not get bigger by being
asked on a bigger grid.
"""
import functools
import subprocess
import sys
import warnings

import numpy as np
import pytest

import wavebit.fd as fd
from wavebit.fd._kernel.stencil import StencilTerm
from wavebit.fd._lower import lower as lower_params, plan as build_plan, stencil


def scene(cpw=18, radius=.14, eps=2.25, pml=6, periods=2., discretisation='stencil',
          center=(0., 0., 0.), background=None, **changes):
    """A ball in vacuum with a TFSF box and one closed flux surface around it.

    The half widths are the ones the doors need and are written here rather than derived, so
    that a test that moves one of them says which: the ball's support plus the constitutive
    halo plus the operator's three nodes has to clear the TFSF injection band, which has to
    clear the flux surface, which has to clear the absorbing layer.
    """
    dx = 1 / cpw
    size = 2 * (.65 + pml * dx)
    return fd.Scene(domain=fd.Domain(size=(size,) * 3),
                    boundaries=fd.Boundaries.open(fd.PML(cells=pml, allow_thin=True)),
                    resolution=fd.Resolution(dx=dx), run=fd.Run(periods=periods),
                    background=background,
                    structures=(fd.Structure(fd.Sphere(radius=radius, center=center), eps,
                                             discretisation=discretisation),),
                    sources=(fd.TFSF('t', box=fd.Box(size=(1.11,) * 3), direction='+z',
                                     polarization='x', band=fd.Band(wavelength=1., df=.1)),),
                    monitors=(fd.Flux('sca', box=fd.Box(size=(1.2,) * 3),
                                      wavelengths=(1.,)),), **changes)


def quiet(call, *args, **kwargs):
    """``call`` with this scene's own accuracy and mesh warnings silenced.

    They are the subject of other files and every plan here would print all three.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return call(*args, **kwargs)


@functools.lru_cache(maxsize=4)
def prepared(discretisation='stencil'):
    """The fixture's plan, prepared once per worker -- the assembly costs about three seconds."""
    s = scene(discretisation=discretisation)
    return s, quiet(build_plan, s)


def stencil_term(plan):
    term, = (t for t in plan.terms if isinstance(t, StencilTerm))
    return term


@functools.lru_cache(maxsize=1)
def assembled():
    """The fixture's operator as an assembled sparse matrix, beside the arguments that made it.

    The uncompressed matrix is the oracle for what ``padded`` stored and for what the kernel
    gathers: those two are a storage format and a loop over it, and a format is only worth
    anything while something else can say what it was supposed to hold.
    """
    args = stencil.arguments(scene(), prepared('voxel')[1], 0)
    return args, stencil.geometry(*args)


class _Reached(Exception):
    """Raised in place of the solve: every door in front of it has passed."""


def reaches_assembly(s):
    """Whether ``fd.plan(s)`` passes every door and arrives at the solve, without solving.

    The nearest legal scene beside a refusal is asked this rather than prepared, because the
    doors are the question and the solve is three seconds that answer a different one. The
    fixture itself is prepared in full by :func:`prepared`, so the solve behind the doors is
    exercised too.
    """
    real = stencil._assemble

    def stop(*args, **kwargs):
        raise _Reached()
    stencil._assemble = stop
    try:
        quiet(build_plan, s)
    except _Reached:
        return True
    finally:
        stencil._assemble = real
    return False


def refusal(s, call=None):
    """The message ``fd.plan(s)`` (or ``call()``) refuses with; fails if it does not refuse."""
    with pytest.raises(ValueError) as err:
        quiet(build_plan, s) if call is None else quiet(call)
    return str(err.value)


def with_structure(s, shape=None, material=2.25, **kwargs):
    """``s`` with its one structure replaced; ``discretisation='stencil'`` unless given."""
    kwargs.setdefault('discretisation', 'stencil')
    shape = fd.Sphere(radius=.14) if shape is None else shape
    return s.replace(structures=(fd.Structure(shape, material, **kwargs),))


# -- the selector ----------------------------------------------------------------------------

def test_the_selector_is_static_scene_text_with_two_spellings_and_a_typo_is_refused():
    """``discretisation=`` is a static string on the structure, and a third value is refused.

    The refusal is the point rather than the default: a value nobody reads is silently
    ignored, and here "silently ignored" means the scene runs the *other* discretisation,
    which converges at a different order. It would read as a physics result.
    """
    ball = fd.Sphere(radius=.1)
    assert fd.Structure(ball, 2.25).discretisation == 'voxel'
    assert fd.Structure(ball, 2.25, discretisation='stencil').discretisation == 'stencil'
    with pytest.raises(ValueError) as err:
        fd.Structure(ball, 2.25, discretisation='stencl')
    assert "'voxel'" in str(err.value) and "'stencil'" in str(err.value)
    # And the spelling of the keyword itself is the spec door's, with the nearest name.
    with pytest.raises(TypeError) as err:
        fd.Structure(ball, 2.25, discretization='stencil')
    assert 'discretisation' in str(err.value)


def test_the_selector_is_never_a_function_of_whether_a_value_is_a_tracer():
    """A traced leaf on a structure that asked for the stencil is refused, not re-routed.

    This is the objection ``geometry.py``'s module docstring raises against a ``smoothing=``
    keyword, answered rather than dodged: the one thing that must never happen is that the
    same scene takes one discretisation when a radius is concrete and another when it is
    traced, because then ``jax.grad`` and a finite difference compute two different
    functions. So the traced case is a named refusal and the selector's value is read from
    the structure and from nothing else.
    """
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')

    def ask(rule, value):
        return scene().replace(structures=(fd.Structure(
            fd.Sphere(radius=.14), fd.Material(eps=value), discretisation=rule),))

    # A JAX tracer: the leaf arrives traced exactly as it does under jax.grad, which is the
    # case that matters -- fd.plan runs inside the transformation there.
    seen = []

    def build(eps):
        seen.append(quiet(build_plan, ask('stencil', eps)))
        return eps
    with pytest.raises(ValueError) as err:
        jax.jit(build)(2.25)
    assert 'StencilNumberRefused' in str(err.value) and not seen
    # A Torch leaf that asked for a gradient, which float() would read and silently detach.
    with pytest.raises(ValueError) as err:
        quiet(build_plan, ask('stencil', torch.tensor(2.25, requires_grad=True)))
    assert 'StencilNumberRefused' in str(err.value)
    # The other side: the same scene with the leaf concrete prepares, so what the refusal
    # turns away is the tracer and not the scene. (The default rule refuses a traced leaf on a
    # *smoothed* structure too, with its own older sentence from constitutive.validate -- so
    # this is not a case where one discretisation works and the other does not. What matters
    # here is that the stencil's own door fires, by its own name, before that one.)
    assert stencil_term(prepared()[1]) is not None
    with pytest.raises(ValueError) as other:
        quiet(build_plan, ask('voxel', torch.tensor(2.25, requires_grad=True)))
    assert 'StencilNumberRefused' not in str(other.value)


def test_the_selector_reaches_the_plan_the_skeleton_and_the_report():
    """Its value is in a plan term, in the skeleton, and printed by ``fd.check``.

    It is on the *term* and not a new ``Plan`` field on purpose. A field would encode as
    ``['tuple', []]`` on every scene that does not use it and would have moved all 23 frozen
    reference skeletons for no change of scene; a term exists only where one was prepared.
    """
    _, plan = prepared()
    term = stencil_term(plan)
    assert term.discretisation == 'stencil' and term.components == (0, 1, 2)
    assert plan.skeleton != prepared('voxel')[1].skeleton
    from wavebit.fd._check import report as build_report
    report = quiet(build_report, *prepared())
    rows = report.discretisations
    assert [(n, rule) for n, rule, *_ in rows] == [(term.name, 'stencil')]
    assert 'discretisation=stencil' in str(report)
    assert not quiet(build_report, *prepared('voxel')).discretisations


def test_the_prepared_arrays_move_the_skeleton_by_their_content():
    """The operator's pattern, diagonal and coefficients are hashed, by content.

    Both halves matter. An operator that did not move the skeleton would let ``fd.run(scene,
    plan=other)`` drive one scene with another's material; a skeleton that read an address
    would make every freeze disagree with itself (memory ``a-digest-you-did-not-validate``).

    Every variant below carries the **same** ``source`` digest, so what is under test is the
    ``metadata={'digest': True}`` path over the arrays themselves and not the string beside
    them. (That is not a hypothetical distinction: when the variants differed by a string
    alone, the mutant that makes ``_digest`` ignore array content survived the whole file.)
    ``value`` joined the digested fields when the module-level coefficient store was removed,
    so it is nudged here too.
    """
    _, plan = prepared()
    term = stencil_term(plan)

    def variant(**changes):
        fields = dict(components=term.components, index=term.index, output=term.output,
                      diagonal=term.diagonal, value=term.value, structure=term.structure,
                      source=term.source, spectral_upper_bound=term.spectral_upper_bound)
        fields.update(changes)
        return plan.with_static(terms=tuple(
            t.__class__(t.name, t.field, t.region, t.ratio, t.masks, **fields)
            if t is term else t for t in plan.terms))

    assert variant().skeleton == plan.skeleton
    # The diagonal and the coefficients, by value: same dtype, same shape, one number apart.
    for name in ('diagonal', 'value'):
        nudged = np.array(getattr(term, name), copy=True)
        nudged.flat[0] = np.nextafter(nudged.flat[0], np.inf)
        assert variant(**{name: nudged}).skeleton != plan.skeleton
    # The gather pattern, by value, likewise.
    shifted = np.array(term.index, copy=True)
    shifted[0, 0] += 1
    assert variant(index=shifted).skeleton != plan.skeleton
    # And the selector's own word is in it as text.
    assert variant(discretisation='voxel').skeleton != plan.skeleton


def test_two_processes_agree_on_a_prepared_skeleton():
    """A prepared plan's skeleton is the same number in two fresh interpreters.

    Both sides of the comparison are **child processes with the same explicit thread
    environment** -- OMP, OpenBLAS, MKL and NumExpr all pinned to 2 -- and different
    ``PYTHONHASHSEED`` values. The parent is deliberately not one of them: the assembled matrix's
    data is known to differ between 1, 2 and 4 BLAS threads (commit c939ed3), and the parent
    runs at whatever the caller set (CI sets 1), so a comparison with it would test the
    caller's environment and not the code.

    What this proves: the same code and the same thread count give the same skeleton in two
    interpreters, whatever the hash seed and the addresses -- which is what the freezers'
    two-freeze check and ``fd.run``'s skeleton comparison need. What it does **not** prove:
    that the operator is independent of the thread count. It is not, and a plan prepared at
    another count is a different plan (``docs/fd-symmetric-stencil.md``).
    """
    import os
    import pathlib
    root = str(pathlib.Path(fd.__file__).resolve().parents[2])
    here = str(pathlib.Path(__file__).parent)
    code = ('import sys, warnings\n'
            f'sys.path[:0] = [{root!r}, {here!r}]\n'
            'warnings.simplefilter("ignore")\n'
            'import test_fd_stencil as t\n'
            'print(t.prepared()[1].skeleton)\n')
    threads = dict(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
                   NUMEXPR_NUM_THREADS='2', PYTHONUTF8='1')
    children = [subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True, cwd=root,
                                 env=dict(os.environ, PYTHONHASHSEED=seed, **threads))
                for seed in ('0', '1')]
    out = [child.communicate(timeout=900) for child in children]
    for child, (_, err) in zip(children, out):
        assert child.returncode == 0, err
    first, second = (o.strip() for o, _ in out)
    assert len(first) == 64 and first == second


# -- plan time, not lowering -----------------------------------------------------------------

def test_the_operator_is_assembled_at_plan_time_and_lowering_only_reads_it():
    """``lower`` calls nothing in SciPy: it moves arrays and patches coefficients.

    That is the whole reason preparation is where it is. ``fit`` is a sparse least-squares
    solve, a symmetric eigenpair and a sparse factorization, none of which has a JAX or a
    Torch spelling -- so a lowering that reached them would be a NumPy-only run, or three
    different runs. The witness is a counter on ``geometry`` itself: it is called once by
    ``plan`` and never by ``lower``, on any backend.
    """
    calls = []
    real = stencil.geometry
    stencil.geometry = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
    try:
        s = scene(radius=.145)              # not the cached fixture: this one must assemble
        plan = quiet(build_plan, s)
        assert len(calls) == 1
        for backend in ('numpy', 'jax', 'torch'):
            lower_params(s, plan, backend)
            assert len(calls) == 1
    finally:
        stencil.geometry = real


def test_run_takes_the_operator_its_plan_already_carries():
    """``fd.run(scene, plan=p)`` does not solve the operator ``p`` already holds.

    ``run`` plans the scene again to check the caller's plan against it and to take the
    grazing survey, and until this commit that re-plan re-solved the operator: measured on
    this fixture, one ``geometry`` call per run, 3.0 of a 4.0 second NumPy run (the 56-cell
    research sphere's preparation is 493 s). Now the re-plan runs every door and takes the
    caller's operator when it was prepared from the same arguments. The other side is the
    one that matters as much: a plan whose operator was prepared for a *different* scene is
    not reused -- the scene's own is solved, its skeleton differs, and the run is refused.
    """
    calls = []
    real = stencil.geometry
    stencil.geometry = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
    try:
        s = scene(radius=.1415)             # not the cached fixture: this one must assemble
        plan = quiet(build_plan, s)
        assert len(calls) == 1
        quiet(fd.run, s, 'numpy', plan=plan)
        assert len(calls) == 1
        # Without a plan there is nothing to reuse, so the run prepares its own.
        quiet(fd.run, s, 'numpy')
        assert len(calls) == 2
        other = with_structure(s, fd.Sphere(radius=.1415), 2.3)
        with pytest.raises(ValueError) as err:
            quiet(fd.run, other, 'numpy', plan=plan)
        assert 'built for a different scene' in str(err.value)
        assert len(calls) == 3
    finally:
        stencil.geometry = real


def test_lowering_twice_gives_the_same_arrays_and_three_backends_agree():
    """The same plan lowered twice, and on three backends, is the same operator.

    Nothing in the lowering solves, so this is not a tolerance question; the only thing that
    could move is a cast.
    """
    s, plan = prepared()
    a = lower_params(s, plan, 'numpy')
    b = lower_params(s, plan, 'numpy')
    name = stencil_term(plan).name
    for key in ('index', 'value', 'output'):
        assert np.array_equal(a['stencil'][name][key], b['stencil'][name][key])
    assert np.array_equal(np.asarray(a['be']), np.asarray(b['be']))
    for backend in ('jax', 'torch'):
        pytest.importorskip(backend)
        other = lower_params(s, plan, backend)
        assert np.array_equal(np.asarray(other['stencil'][name]['value']),
                              np.asarray(a['stencil'][name]['value']))
        assert np.array_equal(np.asarray(other['stencil'][name]['index']),
                              np.asarray(a['stencil'][name]['index']))


def test_the_prepared_diagonal_reaches_be_and_the_energy_weight():
    """The effective medium goes into ``be``, which is what the interior energy reads.

    An operator whose diagonal lived only in ``params['stencil']`` would run correctly and
    report the energy of the medium it no longer has: ``InteriorEnergy._weight`` recovers
    ``eps`` from ``ae``/``be``, so a private key is invisible to it. This asserts both ends --
    the patch is exactly ``diagonal * ratio`` over the term's own region, and the diagnostic's
    own code recovers the prepared medium there and **not** the diagonal rule's.
    """
    s, plan = prepared()
    term = stencil_term(plan)
    params = lower_params(s, plan, 'numpy')
    view = (slice(None),) + term.region.slices
    assert np.array_equal(np.asarray(params['be'])[view], term.diagonal * term.ratio)
    # Outside the region nothing moved: compare with the same scene on the default.
    voxel_s, voxel_plan = prepared('voxel')
    voxel = np.asarray(lower_params(voxel_s, voxel_plan, 'numpy')['be'])
    be = np.asarray(params['be'])
    outside = np.ones(be.shape, dtype=bool)
    outside[view] = False
    assert np.array_equal(be[outside], voxel[outside])
    assert not np.array_equal(be[view], voxel[view])
    # And the diagnostic's own reading of the region follows the patch rather than the default.
    from wavebit.fd._runtime import runtime
    rt = runtime('numpy')
    energy = plan.energy
    # Distinct ``key``s: NumPy's ``energy_weight`` writes into a scratch buffer the runtime
    # keeps per key, so asking twice under one key would compare an array with itself -- which
    # is exactly what this assertion is for.
    weight = np.array(energy._weight(params, 'E', rt, slice(None), term.box, key='prepared'))
    default = np.array(energy._weight({**params, 'be': voxel}, 'E', rt, slice(None), term.box,
                                      key='default'))
    assert not np.array_equal(weight, default)
    # And it reads the prepared medium: ratio (1 + ae) / (2 be) with ae = 1 is ratio / be.
    assert np.allclose(weight, term.ratio / (term.diagonal * term.ratio))


def test_the_prepared_operator_changes_what_the_run_returns_and_all_backends_agree():
    """The selector is not decoration: the same scene returns a different number.

    And the four ways of running it return the same one. NumPy against JAX and Torch is the
    package's standing rule; here it also says that the gather the kernel does -- an integer
    ``take``, a multiply, a sum and a scatter -- is the same four lines on all three.
    """
    s, plan = prepared()
    flux = {}
    for backend in ('numpy', 'jax', 'torch'):
        pytest.importorskip(backend)
        flux[backend] = float(np.asarray(quiet(fd.run, s, backend, plan=plan)['sca'].flux)[0])
    assert abs(flux['jax'] / flux['numpy'] - 1) < 1e-12
    assert abs(flux['torch'] / flux['numpy'] - 1) < 1e-12
    voxel_s, voxel_plan = prepared('voxel')
    default = float(np.asarray(quiet(fd.run, voxel_s, 'numpy', plan=voxel_plan)['sca'].flux)[0])
    assert abs(flux['numpy'] / default - 1) > 1e-6


# -- the bit-identity hook -------------------------------------------------------------------

def test_the_kernel_gather_is_the_assembled_operators_off_diagonal_action():
    """The four lines the kernel runs are the operator's off-diagonal matvec, and nothing else.

    This is the gate the "does the answer move" test could not be: patching ``be`` alone
    already moves a scattering flux by far more than any tolerance, so a kernel that dropped
    the off-diagonal increment entirely would still return a different number from the
    default and look like it was doing the work. The mutant is exactly that -- multiply the
    returned ``delta`` by zero -- and it survived until this test existed (memory
    ``mutate-the-code-to-test-the-tests``).

    The oracle is the **assembled** matrix, not the padded arrays: the sparse matvec sums in a
    different order and knows nothing about the compression, the zero output slot or the
    gather, so it can say what those were supposed to produce.
    """
    from scipy.sparse import diags as sparse_diags

    from wavebit.fd._runtime import runtime
    (nodes, dx, center, radius, eps, components), (matrix, _) = assembled()
    s, plan = prepared()
    term = stencil_term(plan)
    params = lower_params(s, plan, 'numpy')
    rt = runtime('numpy')
    curl = np.random.default_rng(20260924).standard_normal((3, plan.nz, plan.ny, plan.nx))
    mem = np.zeros((3,) + term.region.shape)
    flux, delta = term.prepare(curl, mem, params, rt)
    local = curl[term.view]
    compact = np.stack([local[c] for c in term.components]).reshape(-1)
    off = (matrix - sparse_diags(matrix.diagonal())).tocsr()
    expected = (off @ compact).reshape((len(term.components),) + term.region.shape)
    assert np.abs(expected).max() > 1e-3          # the comparison is not vacuous
    assert np.allclose(np.asarray(delta), term.ratio * expected, rtol=1e-12, atol=0)
    assert np.array_equal(np.asarray(flux), mem + term.ratio * local)


def test_the_public_preparation_prepares_the_research_operator_bit_for_bit():
    """Owner decision (C)'s hook: the public path's operator IS the research path's.

    The expensive 40/56/80 convergence sweep is run once, on the research path. What lets its
    result be quoted for a public scene is this: for one scene, both paths reduce to the same
    four arrays, digested, with no tolerance. They call one ``geometry`` -- so what is really
    under test is that the public preparation hands it the same node lattice, the same centre,
    radius and permittivity, and grows the same region -- and this is cheap enough to run in
    the fast tier, which is the other half of what the decision asked for.
    """
    study = pytest.importorskip('studies.fdtd_subpixel_symmetric_stencil')
    ball = fd.Sphere(radius=.14)
    base = scene(discretisation='voxel', background=fd.Material(eps=1.))
    research_plan = quiet(build_plan, base)
    params = lower_params(base, research_plan, 'numpy')
    _, patched, _ = study.replace(base, research_plan, params, tuple(ball.center),
                                  ball.radius, 2.25, (0, 1, 2))
    asked = scene(background=fd.Material(eps=1.))
    plan = quiet(build_plan, asked)
    term = stencil_term(plan)
    public = lower_params(asked, plan, 'numpy')
    view = (slice(None),) + term.region.slices
    mine = stencil.prepared_digest(term.index, public['stencil'][term.name]['value'],
                                   term.output, np.asarray(public['be'])[view])
    theirs = stencil.prepared_digest(patched['stencil_index'], patched['stencil_value'],
                                     patched['stencil_output'],
                                     np.asarray(patched['be'])[view])
    assert mine == theirs
    # And the whole coefficient array, not only the region: the two paths patch the same cells
    # and leave the same ones alone.
    assert np.array_equal(np.asarray(patched['be']), np.asarray(public['be']))


def test_the_operator_digest_is_a_function_of_content_and_of_nothing_else():
    """The hook's own gate: two assemblies of one fixture digest alike, two fixtures do not.

    A digest that read an address would compare equal to nothing and would make the hook above
    pass or fail for reasons that have nothing to do with the operator.
    """
    from scipy.sparse import csr_matrix
    a = csr_matrix(np.diag([1., 2., 3.]))
    b = csr_matrix(np.diag([1., 2., 3.]))
    c = csr_matrix(np.diag([1., 2., 3.0000000000000004]))
    assert stencil.operator_digest(a) == stencil.operator_digest(b)
    assert stencil.operator_digest(a) != stencil.operator_digest(c)
    index = np.zeros((2, 3), dtype=np.int32)
    value = np.zeros((2, 3))
    out = np.zeros(4, dtype=np.int32)
    diag = np.ones(4)
    first = stencil.prepared_digest(index, value, out, diag)
    assert first == stencil.prepared_digest(index.copy(), value.copy(), out.copy(), diag.copy())
    assert first != stencil.prepared_digest(index, value + 1e-300, out, diag)


# -- the doors -------------------------------------------------------------------------------

def test_a_cell_snapped_shape_cannot_ask_for_the_stencil():
    """Both sides: the ball prepares, the cell-snapped box is refused by name.

    A cell-snapped shape's edge *is* a cell boundary, so there is no interface inside a voxel
    for the operator's edge and dual-face constraints to be written across; the refusal says
    that rather than preparing an operator of identity rows. It also has to be heard: a
    cell-snapped structure makes no smoothed region at all, so a preparation that only ran
    where one existed would wave this scene through on the default and answer the question it
    was not asked.
    """
    s = scene().replace(structures=(fd.Structure(fd.Box(size=(.28,) * 3), 2.25,
                                                 discretisation='stencil'),))
    with pytest.raises(ValueError) as err:
        quiet(build_plan, s)
    # Its own sentence and not the not-a-sphere one beside it, which a Box also fails: with
    # only the shape name asserted, removing this door survived (both arms share the name).
    assert 'StencilShapeRefused' in str(err.value) and 'cell-snapped' in str(err.value)
    assert isinstance(stencil_term(prepared()[1]), StencilTerm)


def test_a_structure_whose_support_reaches_an_absorbing_layer_is_refused():
    """Both sides: clear of the layer it prepares, inside it the scene is refused.

    The operator is fitted on constant interface states in a lossless medium. A graded
    absorber inside the fitted neighbourhood is a material the fit never saw, so the rows it
    returns there are not the rows the layer needs -- and the run would complete.
    """
    with pytest.raises(ValueError) as err:
        quiet(build_plan, scene(radius=.42))
    assert 'StencilLayerRefused' in str(err.value)
    assert stencil_term(prepared()[1]) is not None


def test_a_source_or_monitor_surface_that_cuts_the_nonlocal_support_is_refused():
    """Both sides, on the rule the research code already had rather than a new one.

    A TFSF face injects into a one-cell band; a closed flux surface has to lie wholly inside
    or wholly outside the rows the operator touches, because the surface integral assumes the
    medium is the same on both of its sides. Moving either onto the support is the mutant the
    scene cannot see for itself.
    """
    tight = scene().replace(sources=(fd.TFSF('t', box=fd.Box(size=(.42,) * 3), direction='+z',
                                             polarization='x',
                                             band=fd.Band(wavelength=1., df=.1)),))
    with pytest.raises(ValueError) as err:
        quiet(build_plan, tight)
    assert 'StencilSourceRefused' in str(err.value)
    cut = scene().replace(monitors=(fd.Flux('sca', box=fd.Box(size=(.56,) * 3),
                                            wavelengths=(1.,)),))
    with pytest.raises(ValueError) as err:
        quiet(build_plan, cut)
    assert 'StencilSurfaceRefused' in str(err.value)


def test_a_predicted_preparation_above_its_bound_is_refused_before_it_starts(monkeypatch):
    """Both sides, both bounds, and the prediction is shut *before* the solve.

    The bound is declared in bytes and the prediction is two measured constants times a row
    count. The refusal has to come first: the solve is what would take the memory, so a door
    behind it protects nothing (memory ``laptop-ram-panics-not-just-slows``). The fixture is
    predicted at about 265 MiB, so a 128 MiB bound refuses it and a 512 MiB one admits it --
    whether the bound is the default or the one a structure declares with ``memory_bound=``.
    """
    calls = []
    real = stencil.geometry
    monkeypatch.setattr(stencil, 'geometry',
                        lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    tight = with_structure(scene(), memory_bound=128 * 1024 ** 2)
    message = refusal(tight)
    assert 'StencilMemoryRefused' in message and 'prediction' in message
    assert 'bound this structure declared' in message and not calls
    assert reaches_assembly(with_structure(scene(), memory_bound=512 * 1024 ** 2))
    monkeypatch.setattr(stencil, 'MEMORY_BOUND_BYTES', 128 * 1024 ** 2)
    message = refusal(scene())
    assert 'StencilMemoryRefused' in message and 'default bound' in message and not calls
    assert 'memory_bound=' in message
    assert reaches_assembly(with_structure(scene(), memory_bound=512 * 1024 ** 2))
    # The spelling is refused where nobody would read it, and a bound has to be a size.
    with pytest.raises(ValueError, match='prepares nothing'):
        fd.Structure(fd.Sphere(radius=.1), 2.25, memory_bound=2 ** 30)
    for bad in (0, -1., float('inf'), True, '1GB'):
        with pytest.raises(ValueError, match='positive, finite number of bytes'):
            fd.Structure(fd.Sphere(radius=.1), 2.25, discretisation='stencil', memory_bound=bad)


def test_the_default_bound_admits_the_resolutions_the_orders_were_measured_at():
    """Owner decision (C)'s public spot check has to be *possible*: the bound admits them.

    The scenes are the research driver's own -- ``studies/fdtd_subpixel_mie_sphere.py``'s
    ``sphere_scene``, the epsilon-12, radius/wavelength .25 ball -- at the resolutions whose
    preparation peak was measured (``docs/fd-symmetric-stencil.md``), asked for through the
    public selector. Every door, the memory door included, has to pass; nothing is solved.
    """
    mie = pytest.importorskip('studies.fdtd_subpixel_mie_sphere')
    transform = pytest.importorskip('studies.fdtd_subpixel_mie_transform')
    # The placements are the research's own tuple, not a copy of it.
    assert stencil.RESEARCH_PLACEMENTS == transform.PLACEMENTS
    for cpw in stencil.MEASURED_RESOLUTIONS:
        for placement in stencil.RESEARCH_PLACEMENTS:
            s = mie.sphere_scene(cpw, 12., 48., placement, radius=.25, propagation='yee')
            # The research's own .45 absorption half-width: at .4 the placed 40-cell spheres
            # cut the grown region, which the occupancy door refuses by name.
            monitors = tuple(m.replace(box=fd.Box(size=(.9,) * 3)) if m.name == 'abs' else m
                             for m in s.monitors)
            s = s.replace(monitors=monitors, structures=(fd.Structure(
                s.structures[0].shape, 12., discretisation='stencil'),))
            assert reaches_assembly(s), (cpw, placement)


def test_the_predicted_active_rows_are_the_ones_the_fit_reports():
    """The cheap screen the memory door reads agrees with the expensive path's own count.

    The screen is a second spelling of four lines of ``geometry``; that is only honest while
    it is checked against the thing it predicts, which is what this is. ``active_edges`` comes
    from ``fit``, after the neighbour search, on the operator that was actually assembled.
    """
    s, plan = prepared()
    term = stencil_term(plan)
    (nodes, dx, center, radius, eps, components), (_, report) = assembled()
    rows = stencil.active_rows(nodes, dx, center, radius, components)
    assert rows == report['active_edges'] == len(term.index)


def test_a_scene_with_no_stencil_structure_is_untouched_by_the_preparation():
    """The default path costs one generator expression and nothing else.

    Not a claim about speed: the assertion is that ``prepare`` returns the identical object,
    so no ``with_static`` runs, no skeleton is recomputed and no plan field moves. That is why
    every stage-0..3 reference skeleton is byte-identical across this commit.
    """
    s, plan = prepared('voxel')
    assert stencil.prepare(s, plan) is plan
    assert not any(isinstance(t, StencilTerm) for t in plan.terms)
    assert 'stencil' not in lower_params(s, plan, 'numpy')
    vacuum = fd.Scene(domain=fd.Domain(size=(.5, .5, .5)),
                      boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
                      resolution=fd.Resolution(dx=.05),
                      run=fd.Run(steps=2))
    p = quiet(build_plan, vacuum)
    assert stencil.prepare(vacuum, p) is p


def test_two_prepared_scenes_in_one_process_keep_their_own_coefficients():
    """Preparing a second scene must not make the first one lower the second's operator.

    The coefficients are a field of the prepared term, so each plan carries its own and there
    is no store for a second preparation to overwrite. (There was one: a module-level dict,
    first keyed by the term's *name* -- a position, ``constitutive[E0]`` in every such scene --
    then by digest, and never released.) Both plans are lowered *after* both preparations,
    which is the order that made the name-keyed defect visible.
    """
    small, small_plan = prepared()
    big = scene(radius=.145)
    big_plan = quiet(build_plan, big)
    a, b = stencil_term(small_plan), stencil_term(big_plan)
    assert a.name == b.name and a.source != b.source
    first = lower_params(small, small_plan, 'numpy')
    second = lower_params(big, big_plan, 'numpy')
    assert not np.array_equal(np.asarray(first['stencil'][a.name]['value']),
                              np.asarray(second['stencil'][b.name]['value']))
    assert np.array_equal(np.asarray(first['stencil'][a.name]['value']), a.value)


def test_a_prepared_operator_is_released_with_its_plan():
    """Dropping the plan drops the operator: nothing in the module keeps it alive.

    The coefficients used to live in a module-level dict that nothing evicted, so every
    operator prepared in a process lived until the process did -- 0.9 MB here, 51 MB on the
    56-cell research sphere, per distinct operator, in a sweep. They are a field of the
    prepared term now. The witness is a weak reference to each of the term's arrays, dead once
    the plan, the lowered params and the term are.
    """
    import gc
    import weakref
    s = scene(radius=.1395)
    plan = quiet(build_plan, s)
    term = stencil_term(plan)
    params = lower_params(s, plan, 'numpy')
    alive = [weakref.ref(a) for a in (term.value, term.index, term.output, term.diagonal)]
    assert all(r() is not None for r in alive)
    del plan, term, params
    gc.collect()
    assert [r() is None for r in alive] == [True] * 4


def test_a_prepared_operator_is_released_with_its_plan_after_a_jax_run():
    """The same on JAX, over a three-radius sweep: the compile cache keeps the program only.

    ``JaxRuntime`` compiles the executor once per plan and keeps it on a runtime that lives as
    long as the process, and both its key and the compiled closure held the plan -- so the
    NumPy-only gate above could not see that, after one ``fd.run(scene, 'jax')``, the operator
    was never released. Measured on ``199463e`` (review 1): a three-radius sweep with the
    caller keeping nothing left 1, 2 and 3 live operators, 2.4, 4.9 and 7.5 MB on a 20-cell
    fixture and about 77 MB each at 56 cells. ``_run._program`` strips the digested arrays out
    of what is keyed and closed over. The other side: a second run of the same plan hits the
    cache (one entry per plan, not per run), so the program is still compiled once.
    """
    import gc
    import weakref
    pytest.importorskip('jax')
    from wavebit.fd._runtime import runtime
    rt = runtime('jax')
    for radius in (.1375, .139, .1405):
        s = scene(radius=radius).replace(run=fd.Run(steps=4))
        plan = quiet(build_plan, s)
        term = stencil_term(plan)
        alive = [weakref.ref(a) for a in (term.value, term.index, term.output, term.diagonal)]
        before = len(rt._compiled)
        quiet(fd.run, s, 'jax', plan=plan)
        quiet(fd.run, s, 'jax', plan=plan)
        assert len(rt._compiled) == before + 1
        del plan, term
        gc.collect()
        assert [r() is None for r in alive] == [True] * 4, radius


# -- the scene around the operator: what the research driver refused, refused again ----------

def test_a_non_vacuum_background_is_refused_not_replaced_by_vacuum():
    """The operator was fitted with vacuum outside the body; another background is refused.

    Measured before this door existed, on this fixture: ``Scene(background=1.44)`` was
    accepted, and ``be / ratio`` at the grown region's corner read 1.0 on the stencil against
    0.6944 (1/1.44, the background the scene declared) on the default -- the prepared diagonal
    had pasted a cube of vacuum round the sphere. The research driver refused this scene by
    name; the public preparation had dropped the condition. Both sides: a declared vacuum
    background, spelt as a material, reaches the solve.
    """
    message = refusal(scene(background=fd.Material(eps=1.44)))
    assert 'StencilBackgroundRefused' in message and "discretisation='voxel'" in message
    # The spelling it names works: the same scene on the voxel rule plans.
    assert quiet(build_plan, scene(background=fd.Material(eps=1.44), discretisation='voxel'))
    # A background that names mu=1 is vacuum's values with a permeability path. The refusal
    # used to tell this user to use vacuum -- what they had written. It now says what is
    # different, and names a spelling that is not the one they used, which reaches the solve.
    message = refusal(scene(background=fd.Material(eps=1., mu=1.)))
    assert 'StencilBackgroundRefused' in message and 'names a permeability' in message
    assert 'Scene(background=fd.Material(eps=1.))' in message
    assert reaches_assembly(scene(background=fd.Material(eps=1.)))


def test_a_second_structure_anywhere_is_refused():
    """One structure per scene, as the research driver required; no other is erased.

    Measured before this door existed, both accepted with no warning: a cell-snapped
    ``fd.Box`` beside the ball, inside the operator's grown region, kept 864 cells at 1/1.44
    on the default and 0 on the stencil; a second smoothed ``fd.Sphere`` beside it left the
    whole lowered ``be`` ``np.array_equal`` to the scene without it, while ``fd.check`` still
    listed it. A structure outside the region is refused too: nothing has measured the
    operator beside another medium. The nearest legal scene is the fixture.
    """
    dx = 1 / 18
    s = scene()
    box = fd.Structure(fd.Box(size=(2 * dx, 12 * dx, 12 * dx), center=(4 * dx, 0, 0)), 1.44)
    ball = fd.Structure(fd.Sphere(radius=.045, center=(.27, 0, 0)), 2.25)
    for extra in (box, ball):
        message = refusal(s.replace(structures=s.structures + (extra,)))
        assert 'StencilStructuresRefused' in message and "discretisation='voxel'" in message
    # The spelling it names for the scene as written: the ball on the voxel rule plans.
    voxel = s.structures[0].replace(discretisation='voxel')
    assert quiet(build_plan, s.replace(structures=(voxel, box)))
    assert stencil_term(prepared()[1]) is not None
    # Far from the region too, in a scene with nothing else in it to object first.
    bare = s.replace(sources=(), monitors=(), run=fd.Run(steps=4), boundaries=fd.Boundaries.open(
        fd.PML(cells=6, wavelength=1., allow_thin=True)))
    far = fd.Structure(fd.Box(size=(2 * dx,) * 3, center=(.5, .5, .5)), 1.44)
    assert 'StencilStructuresRefused' in refusal(bare.replace(
        structures=bare.structures + (far,)))
    assert reaches_assembly(bare)


def test_a_complex_run_is_refused_and_a_real_periodic_one_is_not():
    """A Bloch face makes the run complex128; the operator was measured in float64 only."""
    bloch = scene().replace(boundaries=fd.Boundaries(
        x=fd.Bloch(phase=.3), y=fd.Periodic(), z=fd.PML(cells=6, allow_thin=True)))
    message = refusal(bloch)
    assert 'StencilDtypeRefused' in message
    assert "discretisation='voxel'" in message and 'fd.Periodic()' in message
    voxel = bloch.replace(structures=(bloch.structures[0].replace(discretisation='voxel'),))
    assert quiet(build_plan, voxel).dtype == 'complex128'
    assert reaches_assembly(bloch.replace(boundaries=fd.Boundaries(
        x=fd.Periodic(), y=fd.Periodic(), z=fd.PML(cells=6, allow_thin=True))))


def test_an_initial_field_is_refused_on_either_half():
    """InitialField.E and InitialField.H are both refused, as the research driver refused them.

    The E half has a reason of its own (the term stores D); the H half is refused because no
    run of this operator has ever started from a field, not because a failure was measured.
    """
    pulse = (None, None, lambda x, y, z: 1e-3 * np.exp(-(x * x + y * y + z * z) / .01))
    for field in ('E', 'H'):
        message = refusal(scene(initial=fd.InitialField(**{field: pulse})))
        assert 'StencilScopeRefused' in message and f'InitialField.{field}' in message
    assert reaches_assembly(scene())


def test_only_fd_sphere_itself_can_ask():
    """A subclass of fd.Sphere may answer its own fills; the operator uses the closed forms."""
    class Sphere(fd.Sphere):
        pass
    message = refusal(with_structure(scene(), Sphere(radius=.14)))
    assert 'StencilShapeRefused' in message and 'not a subclass' in message
    assert "discretisation='voxel'" in message
    assert quiet(build_plan, with_structure(scene(), Sphere(radius=.14), discretisation='voxel'))
    assert reaches_assembly(with_structure(scene(), fd.Sphere(radius=.14)))


def test_a_magnetic_structure_is_told_it_is_magnetic():
    """One magnetic ball makes an E and an H term; the refusal names the permeability.

    Before this commit the region count ran first and told this scene it carried two smoothed
    regions, while the accurate sentence beside it could not be reached.
    """
    message = refusal(with_structure(scene(), material=fd.Material(eps=1.44, mu=1.05)))
    assert 'StencilScopeRefused' in message and 'permeability' in message
    assert 'region' not in message.split('.')[0]
    assert "discretisation='voxel'" in message and 'Leave mu unnamed' in message
    assert quiet(build_plan, with_structure(scene(), material=fd.Material(eps=1.44, mu=1.05),
                                            discretisation='voxel'))
    assert reaches_assembly(with_structure(scene(), material=fd.Material(eps=1.44)))
    # mu=1 named is refused too, and told why: a named mu is magnetic to these doors.
    message = refusal(with_structure(scene(), material=fd.Material(eps=1.44, mu=1.)))
    assert 'whatever its value' in message


def test_a_wall_or_the_domain_edge_inside_the_region_is_refused():
    """A PEC wall in the region, and a grown region that leaves the box, are refused by name."""
    dx = 1 / 18

    def cell(x, boundary):
        return fd.Scene(domain=fd.Domain(size=(1., 1., 1.)),
                        boundaries=fd.Boundaries(x=boundary, y=fd.Periodic(), z=fd.Periodic()),
                        resolution=fd.Resolution(dx=dx), run=fd.Run(steps=4),
                        structures=(fd.Structure(fd.Sphere(radius=.14, center=(x, 0, 0)), 2.25,
                                                 discretisation='stencil'),))
    message = refusal(cell(-.3, fd.PEC()))
    assert 'StencilScopeRefused' in message and 'PEC or PMC wall' in message
    assert 'StencilRegionRefused' in refusal(cell(.33, fd.PEC()))
    assert reaches_assembly(cell(0., fd.PEC()))


def test_a_source_the_clearance_rule_cannot_read_is_refused():
    """Only a TFSF box and a PlaneWave plane have a clearance rule; a Dipole is refused."""
    dipole = scene().replace(monitors=(), sources=(fd.Dipole(
        'd', center=(0., 0., .5), component='Ex', band=fd.Band(wavelength=1., df=.1)),))
    message = refusal(dipole)
    assert 'StencilSourceRefused' in message and "discretisation='voxel'" in message
    assert 'fd.TFSF' in message
    assert quiet(build_plan, dipole.replace(
        structures=(dipole.structures[0].replace(discretisation='voxel'),)))
    assert reaches_assembly(scene())


def test_nothing_else_may_occupy_the_grown_region():
    """No source and no monitor surface in the region the operator owns, including its halo.

    The grown region is the smoothed support plus three nodes a side, and the prepared term
    owns all of it: its diagonal replaces ``be`` there and its gather reads the curl there.
    Each case below is refused where the research clearance rule -- which judged only the
    rows the operator changes -- would have let it through, and each has a nearest legal
    twin one step further out. The fixture's own TFSF box is 1.11 wide for this reason: the
    0.9 it had before put its faces on the region's outermost node.
    """
    s = scene()
    tfsf = lambda size: s.replace(sources=(fd.TFSF(
        't', box=fd.Box(size=(size,) * 3), direction='+z', polarization='x',
        band=fd.Band(wavelength=1., df=.1)),))
    assert 'StencilSourceRefused' in refusal(tfsf(.95))
    assert reaches_assembly(tfsf(1.11))
    flux = lambda size: s.replace(monitors=(fd.Flux('f', box=fd.Box(size=(size,) * 3),
                                                    wavelengths=(1.,)),))
    assert 'StencilSurfaceRefused' in refusal(flux(.8))
    assert reaches_assembly(flux(1.2))
    plane = lambda z: s.replace(monitors=(fd.Flux(
        'f', plane=fd.Plane(z=z, x=(-.3, .3), y=(-.3, .3)), wavelengths=(1.,)),))
    assert 'StencilSurfaceRefused' in refusal(plane(.38))
    assert reaches_assembly(plane(.5))
    periodic = s.replace(boundaries=fd.Boundaries(x=fd.Periodic(), y=fd.Periodic(),
                                                  z=fd.PML(cells=6, allow_thin=True)))
    wave = lambda z, monitors=(): periodic.replace(monitors=monitors, sources=(fd.PlaneWave(
        'p', z=z, direction='+z', polarization='s', band=fd.Band(wavelength=1., df=.1)),))
    assert 'StencilSourceRefused' in refusal(wave(-.47))
    # One node outside the region is still inside the injection band: -.5 is node 9 and the
    # region starts at node 10. The review of 199463e found that -.47 (node 10) and -.6
    # (node 7) could not tell a band of one node from a band of none; this case can.
    assert 'StencilSourceRefused' in refusal(wave(-.5))
    assert reaches_assembly(wave(-.6))
    orders = lambda z: (fd.Diffraction('r', z=z, side='transmit', wavelengths=(1.,)),)
    assert 'StencilSurfaceRefused' in refusal(wave(-.6, orders(.38)))
    assert reaches_assembly(wave(-.6, orders(.5)))


def test_a_plane_is_judged_against_the_region_on_its_own_axis():
    """The occupancy span is read along the plane's own axis, not a transposed one.

    Every case above is a centred ball, whose grown region is the same range on x, y and z,
    so reading the region's z range for an x plane could not be told from the right one: the
    review of 199463e transposed the span (``box[a]`` for ``box[2 - a]``) and all 105 cases
    passed. Here the ball is off centre along x, so the region is nodes x 11..27 and y, z
    8..25, and three planes each sit where the two readings disagree: an x plane at node 26
    (inside the x range, outside the z range) crosses the region; an x plane at node 10 and
    a z plane at node 26 (outside their own range, inside the other one) miss it.
    """
    def ball(monitors=()):
        return fd.Scene(domain=fd.Domain(size=(1.8,) * 3),
                        boundaries=fd.Boundaries.open(fd.PML(cells=6, wavelength=1.,
                                                             allow_thin=True)),
                        resolution=fd.Resolution(dx=1 / 18), run=fd.Run(steps=2),
                        structures=(fd.Structure(fd.Sphere(radius=.14, center=(.15, 0., 0.)),
                                                 2.25, discretisation='stencil'),),
                        monitors=monitors)
    plan = quiet(build_plan, ball().replace(
        structures=(ball().structures[0].replace(discretisation='voxel'),)))
    old, = plan.constitutive
    assert stencil._stencil_box(old, (0, 1, 2), plan) == ((8, 25), (8, 25), (11, 27))

    def plane(axis, node):
        at = {a: (-.2, .2) for a in 'xyz'}
        at[axis] = plan.low['xyz'.index(axis)] + node * plan.dx
        return (fd.Flux('f', plane=fd.Plane(**at), wavelengths=(1.,)),)
    message = refusal(ball(plane('x', 26)))
    assert 'StencilSurfaceRefused' in message and 'node 26 on x' in message
    assert reaches_assembly(ball(plane('x', 10)))
    assert reaches_assembly(ball(plane('z', 26)))


def test_a_studies_checkout_that_is_missing_is_named():
    """Where ``studies/`` cannot be imported, the triplet baseline's seam is refused by name."""
    import contextlib

    class Hide:
        def find_spec(self, name, path=None, target=None):
            if name == 'studies' or name.startswith('studies.'):
                raise ImportError(f'No module named {name!r} (hidden by the test)')

    @contextlib.contextmanager
    def hidden():
        saved = {n: m for n, m in sys.modules.items()
                 if n == 'studies' or n.startswith('studies.')}
        for n in saved:
            del sys.modules[n]
        finder = Hide()
        sys.meta_path.insert(0, finder)
        try:
            yield
        finally:
            sys.meta_path.remove(finder)
            sys.modules.update(saved)
    with hidden():
        assert 'StencilSeamRefused' in refusal(scene(radius=.1385))


def test_the_operators_own_certificates_refuse_by_name(monkeypatch):
    """Consistency, energy, halo and time step: each refuses, each by its own name.

    The operator the solve returns is served from the fixture's own assembly, with one
    thing changed per case, so every case costs no solve; the time-step case is a real scene
    -- ``courant=0.99``, which the default accepts and this operator's spectral bound
    (1.027 on the fixture) does not. The fixture's unchanged report is the nearest legal case.
    """
    _, (matrix, report) = assembled()

    def serve(served=matrix, **changes):
        monkeypatch.setattr(stencil, 'geometry',
                            lambda *a, **k: (served, {**report, **changes}))
    for changes, name in (({'residual_max': 1e-9}, 'StencilConsistencyRefused'),
                          ({'lsmr_stop': 7}, 'StencilConsistencyRefused'),
                          ({'minimum_eigenvalue': -1e-3}, 'StencilEnergyRefused')):
        serve(**changes)
        assert name in refusal(scene())
    moved = matrix.copy()
    moved[0, 0] = 2.
    serve(moved)
    assert 'StencilHaloRefused' in refusal(scene())
    serve()
    message = refusal(scene().replace(run=fd.Run(periods=2., courant=.99)))
    assert 'StencilCFLRefused' in message and 'after the preparation' in message
    # The courant number it names is the one that passes, applied as written -- and it is the
    # largest such to its four places, so one step above it is refused again.
    limit = float(message.split('fd.Run(courant=...) below ')[1].split(' ')[0])
    assert .9 < limit < .99
    assert stencil_term(quiet(build_plan, scene().replace(
        run=fd.Run(periods=2., courant=limit)))) is not None
    assert 'StencilCFLRefused' in refusal(scene().replace(
        run=fd.Run(periods=2., courant=limit + 1e-4)))
    assert stencil_term(quiet(build_plan, scene())) is not None


# -- lowering: where every traced path passes -------------------------------------------------

def _paths(jax, torch):
    """Every transform this file can build, as ``name -> (backend, call(f, x))``.

    Reverse mode, forward mode and batching, on both libraries. The forward-mode Torch
    spellings are the ones the review of ``199463e`` found passing both doors -- a dual reads
    as a plain number through ``float()`` -- and returning exactly 0.0.
    """
    t = lambda v: torch.tensor(v, dtype=torch.float64)

    def autograd(f, x):
        leaf = t(x).requires_grad_()
        f(leaf).backward()
        return float(leaf.grad)

    def forward_ad(f, x):
        import torch.autograd.forward_ad as fwAD
        with fwAD.dual_level():
            tangent = fwAD.unpack_dual(f(fwAD.make_dual(t(x), t(1.)))).tangent
            return None if tangent is None else float(tangent)
    return {
        'jax.grad': ('jax', lambda f, x: float(jax.grad(f)(x))),
        'jax.jit(jax.grad)': ('jax', lambda f, x: float(jax.jit(jax.grad(f))(x))),
        'jax.vmap': ('jax', lambda f, x: np.asarray(jax.vmap(f)(jax.numpy.asarray([x, x])))),
        'jax.jvp': ('jax', lambda f, x: float(jax.jvp(f, (x,), (1.,))[1])),
        'jax.jacfwd': ('jax', lambda f, x: float(jax.jacfwd(f)(x))),
        'torch autograd': ('torch', autograd),
        'torch.func.grad': ('torch', lambda f, x: float(torch.func.grad(f)(t(x)))),
        'torch.func.jvp': ('torch', lambda f, x: float(torch.func.jvp(f, (t(x),), (t(1.),))[1])),
        'torch.func.jacfwd': ('torch', lambda f, x: float(torch.func.jacfwd(f)(t(x)))),
        'torch forward_ad': ('torch', forward_ad),
    }


def test_lowering_takes_plain_numbers_only_on_every_path():
    """Every transform meets the plain-number door at lowering, and the recipe it names works.

    The repository's gradient recipe builds the plan from plain numbers and lowers a traced
    scene against it (``tests/test_fd_constitutive.py``), which never passes ``fd.plan``'s
    door. Measured before the first lowering door existed: ``d(sum be)/d(eps)`` was -14.3487
    by autodiff and by central difference on the default and **exactly 0** by both on the
    stencil. The door that replaced it asked "is it traced" and let a Torch forward-mode dual
    through (``torch.func.jvp``, ``torch.func.jacfwd``, ``forward_ad``: 0.0 again). The door now
    asks "is it a plain number", so every path below is refused by one name, an untraced array
    of any library is refused the same way, and a NumPy scalar is not. The other side is the
    recipe the refusal names, applied as written -- voxel in both scenes -- returning the
    central difference on every derivative path, forward mode included.
    """
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')
    from wavebit.fd._runtime import runtime
    runtime('jax')                                   # x64 before any tracer is made

    def loss(rule, plan, backend):
        def f(eps):
            s = with_structure(scene(), material=fd.Material(eps=eps), discretisation=rule)
            return lower_params(s, plan, backend)['be'].sum()
        return f
    plan = prepared()[1]
    paths = _paths(jax, torch)
    for name, (backend, path) in paths.items():
        with pytest.raises(ValueError) as err:
            path(loss('stencil', plan, backend), 2.25)
        message = str(err.value)
        assert 'StencilNumberRefused' in message and 'fd.lower' in message, name
        assert "discretisation='voxel'" in message and 'BOTH' in message, name
    # A traced background is refused as well: the operator was fitted for vacuum around it.
    def background_loss(eps):
        return lower_params(scene(background=fd.Material(eps=eps)), plan, 'jax')['be'].sum()
    with pytest.raises(ValueError) as err:
        jax.grad(background_loss)(1.)
    assert 'StencilNumberRefused' in str(err.value) and 'Scene(background=)' in str(err.value)
    # Not traced at all, and still an array: refused the same way, on any library.
    for value in (jax.numpy.asarray(2.25), torch.tensor(2.25, dtype=torch.float64),
                  np.array(2.25)):
        other = with_structure(scene(), material=value)
        message = refusal(None, lambda: lower_params(other, plan, 'numpy'))
        assert 'StencilNumberRefused' in message, type(value)
    # A NumPy scalar is a plain number: it lowers to the fixture's own arrays, bit for bit.
    be = np.asarray(lower_params(scene(), plan, 'numpy')['be'])
    for value in (np.float64(2.25), np.float32(2.25), 2.25):
        other = lower_params(with_structure(scene(), material=value), plan, 'numpy')
        assert np.array_equal(np.asarray(other['be']), be), type(value)
    # The recipe the refusal names, as written: discretisation='voxel' in both scenes, the
    # plan from plain numbers, fd.lower of the traced scene -- on every derivative path.
    voxel = prepared('voxel')[1]
    numpy = loss('voxel', voxel, 'numpy')
    central = float((numpy(2.25 + 1e-6) - numpy(2.25 - 1e-6)) / 2e-6)
    assert central < -1
    for name, (backend, path) in paths.items():
        if name == 'jax.vmap':
            continue
        derivative = path(loss('voxel', voxel, backend), 2.25)
        assert abs(derivative / central - 1) < 1e-5, (name, derivative, central)


def test_fd_run_takes_plain_numbers_only_before_it_prepares_anything(monkeypatch):
    """The same door at ``fd.plan``, reached through ``fd.run`` itself, before any solve.

    Measured on ``199463e`` (review 2): ``fd.run(scene, 'torch')`` with ``eps`` a forward-mode
    dual returned the scattered flux with a tangent of **exactly 0.0** on the stencil, where
    the stencil's own central difference is 5.64e-9 and the voxel rule's forward mode matches
    its difference to 7e-10. Every path is refused now, by name, with no operator solved; and
    a plain-array scene that never meets a transform is refused too, because the door is a
    type and not a question about tracing.
    """
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')
    from wavebit.fd._runtime import runtime
    runtime('jax')
    calls = []
    real = stencil.geometry
    monkeypatch.setattr(stencil, 'geometry', lambda *a, **k: (calls.append(1), real(*a, **k))[1])

    def flux(backend):
        def f(eps):
            s = with_structure(scene(), material=fd.Material(eps=eps))
            return quiet(fd.run, s, backend)['sca'].flux[0]
        return f
    for name, (backend, path) in _paths(jax, torch).items():
        with pytest.raises(ValueError) as err:
            path(flux(backend), 2.25)
        assert 'StencilNumberRefused' in str(err.value) and 'fd.plan' in str(err.value), name
    for value in (jax.numpy.asarray(2.25), torch.tensor(2.25, dtype=torch.float64),
                  np.array(2.25)):
        assert 'StencilNumberRefused' in refusal(with_structure(scene(), material=value))
    background = scene(background=fd.Material(eps=jax.numpy.asarray(1.)))
    assert 'StencilNumberRefused' in refusal(background)
    assert not calls
    assert reaches_assembly(with_structure(scene(), material=np.float64(2.25)))


def test_the_discretisation_door_at_lowering_works_both_ways():
    """A scene and a plan that disagree about the stencil are refused in either direction.

    Measured on ``199463e`` (review 1): a stencil scene lowered against its voxel twin's plan
    -- the plan the traced-leaf refusal's own recipe builds -- was accepted and painted with
    the voxel rule, and on the traced path returned the voxel gradient, -12.667 on
    ``jax.grad``, ``jax.jit(jax.grad)`` and Torch alike: the two-different-functions outcome
    that refusal exists to prevent. A plan built with no structure at all did the same. Only
    the other direction (a voxel scene against a stencil plan) was refused. Both are now; the
    nearest legal lowering is the fixture against its own plan, and the recipe the refusal
    names (voxel in both scenes) is applied as written in
    :func:`test_lowering_takes_plain_numbers_only_on_every_path`.
    """
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')
    from wavebit.fd._runtime import runtime
    runtime('jax')
    s, plan = prepared()
    v, voxel_plan = prepared('voxel')
    bare = quiet(build_plan, s.replace(structures=()))
    for other in (voxel_plan, bare):
        message = refusal(None, lambda: lower_params(s, other, 'numpy'))
        assert 'StencilMismatchRefused' in message and 'carries no operator' in message
        assert "discretisation='voxel' in the scene you lower" in message

    def loss(eps, backend='jax'):
        traced = with_structure(scene(), material=fd.Material(eps=eps))
        return lower_params(traced, voxel_plan, backend)['be'].sum()
    for path in (jax.grad(loss), jax.jit(jax.grad(loss))):
        with pytest.raises(ValueError) as err:
            path(2.25)
        assert 'StencilMismatchRefused' in str(err.value)
    with pytest.raises(ValueError) as err:
        loss(torch.tensor(2.25, dtype=torch.float64, requires_grad=True), 'torch')
    assert 'StencilMismatchRefused' in str(err.value)
    # The other direction: the plan carries an operator the scene does not ask for.
    for other, said in ((v, "asks for discretisation='voxel'"),
                        (s.replace(structures=()), 'has no structures[0]')):
        message = refusal(None, lambda: lower_params(other, plan, 'numpy'))
        assert 'StencilMismatchRefused' in message and said in message
    assert 'stencil' in lower_params(s, plan, 'numpy')
    assert 'stencil' not in lower_params(v, voxel_plan, 'numpy')


def test_loss_and_anisotropy_meet_check_smoothed_at_plan_and_the_stencil_door_at_lowering():
    """Which door answers a lossy or an anisotropic stencil ball, and where.

    At plan time ``checks.check_smoothed`` refuses both for a smoothed shape on either
    discretisation (its doors 1 and 4) before the stencil's preparation runs, so the
    stencil's own arms for them could never fire there -- the review of ``199463e`` found them
    unreachable from ``fd.plan`` -- and they are the lowering door only. This pins the plan
    side to ``check_smoothed``'s sentences, so a change that let either past it would show;
    :func:`test_lowering_refuses_a_scene_the_operator_was_not_prepared_for` pins the lowering
    side, whose mutants (either arm removed) lower the wrong scene silently.
    """
    for material, sentence in ((fd.Conductor(eps=2.25, sigma=.01), 'fd.Conductor'),
                               (fd.Material(eps=(2.25, 2.25, 2.4)), 'anisotropic eps')):
        for rule in ('stencil', 'voxel'):
            message = refusal(with_structure(scene(), material=material, discretisation=rule))
            assert sentence in message and 'Stencil' not in message, (rule, message[:80])


def test_lowering_refuses_a_scene_the_operator_was_not_prepared_for():
    """A concrete scene that is not the prepared one is refused at lowering, by name.

    Lowered, the prepared diagonal would replace whatever the scene now says: a different
    permittivity or radius, a loss, an anisotropy, a permeability, a background, a second
    structure. Each is refused; the prepared scene itself lowers.
    """
    s, plan = prepared()
    lower = lambda other: (lambda: lower_params(other, plan, 'numpy'))
    cases = ((with_structure(s, material=2.3), 'StencilMismatchRefused'),
             (with_structure(s, fd.Sphere(radius=.145)), 'StencilMismatchRefused'),
             (with_structure(s, discretisation='voxel'), 'StencilMismatchRefused'),
             (s.replace(structures=s.structures + (fd.Structure(
                 fd.Box(size=(.1,) * 3, center=(.5, .5, .5)), 1.44),)),
              'StencilMismatchRefused'),
             (with_structure(s, material=fd.Conductor(eps=2.25, sigma=.1)),
              'lossless fd.Material(eps=...)'),
             (with_structure(s, material=fd.Material(eps=(2.25, 2.25, 2.4))),
              'isotropic fd.Material(eps=...)'),
             (with_structure(s, material=fd.Material(eps=2.25, mu=1.05)), 'permeability'),
             (scene(background=1.44), 'StencilBackgroundRefused'))
    for other, name in cases:
        assert name in refusal(None, lower(other)), name
    assert 'stencil' in lower_params(s, plan, 'numpy')


def test_an_off_centre_sphere_is_patched_where_it_is():
    """The patch lands on the region's own cells when the region is not the same on every axis.

    Every other fixture here is a ball centred in a cubic box, whose grown region is the same
    range on z, y and x -- so a patch box written in the wrong axis order could not be told
    from the right one (review: the transposed-box mutant survived all twenty cases). Here the
    ball sits off centre along x, so the three ranges differ, and ``be`` has to be exactly the
    prepared diagonal over the region and exactly the default's everywhere else.
    """
    def ball(rule):
        return fd.Scene(domain=fd.Domain(size=(1.8,) * 3),
                        boundaries=fd.Boundaries.open(fd.PML(cells=6, wavelength=1.,
                                                             allow_thin=True)),
                        resolution=fd.Resolution(dx=1 / 18), run=fd.Run(steps=2),
                        structures=(fd.Structure(fd.Sphere(radius=.10, center=(.08, 0., 0.)),
                                                 2.25, discretisation=rule),))
    s, v = ball('stencil'), ball('voxel')
    plan, voxel_plan = quiet(build_plan, s), quiet(build_plan, v)
    term = stencil_term(plan)
    assert term.region.x != term.region.y == term.region.z
    be = np.asarray(lower_params(s, plan, 'numpy')['be'])
    voxel = np.asarray(lower_params(v, voxel_plan, 'numpy')['be'])
    view = (slice(None),) + term.region.slices
    assert np.array_equal(be[view], term.diagonal * term.ratio)
    outside = np.ones(be.shape, dtype=bool)
    outside[view] = False
    assert np.array_equal(be[outside], voxel[outside])
    # And the ball is where the default puts it: its deepest cell is inside on both.
    deepest = np.unravel_index(np.argmin(voxel), voxel.shape)
    assert be[deepest] < .9 * term.ratio
