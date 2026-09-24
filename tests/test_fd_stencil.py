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
                    sources=(fd.TFSF('t', box=fd.Box(size=(.9,) * 3), direction='+z',
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
    assert 'StencilTracedRefused' in str(err.value) and not seen
    # A Torch leaf that asked for a gradient, which float() would read and silently detach.
    with pytest.raises(ValueError) as err:
        quiet(build_plan, ask('stencil', torch.tensor(2.25, requires_grad=True)))
    assert 'StencilTracedRefused' in str(err.value)
    # The other side: the same scene with the leaf concrete prepares, so what the refusal
    # turns away is the tracer and not the scene. (The default rule refuses a traced leaf on a
    # *smoothed* structure too, with its own older sentence from constitutive.validate -- so
    # this is not a case where one discretisation works and the other does not. What matters
    # here is that the stencil's own door fires, by its own name, before that one.)
    assert stencil_term(prepared()[1]) is not None
    with pytest.raises(ValueError) as other:
        quiet(build_plan, ask('voxel', torch.tensor(2.25, requires_grad=True)))
    assert 'StencilTracedRefused' not in str(other.value)


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
    """The operator's pattern and diagonal are hashed, and the hash is a function of content.

    Both halves matter. An operator that did not move the skeleton would let ``fd.run(scene,
    plan=other)`` drive one scene with another's material; a skeleton that read an address
    would make every freeze disagree with itself (memory ``a-digest-you-did-not-validate``).

    Every variant below carries the **same** ``key``, so what is under test is the
    ``metadata={'digest': True}`` path over the arrays themselves and not the string beside
    them. (That is not a hypothetical distinction: with the key left at its default the
    variants differed by the key alone and the mutant that makes ``_digest`` ignore array
    content survived the whole file.)
    """
    _, plan = prepared()
    term = stencil_term(plan)

    def variant(**changes):
        fields = dict(components=term.components, index=term.index, output=term.output,
                      diagonal=term.diagonal, key=term.key)
        fields.update(changes)
        return plan.with_static(terms=tuple(
            t.__class__(t.name, t.field, t.region, t.ratio, t.masks, **fields)
            if t is term else t for t in plan.terms))

    assert variant().skeleton == plan.skeleton
    # The diagonal, by value: same dtype, same shape, one number apart.
    nudged = np.array(term.diagonal, copy=True)
    nudged[0, 0, 0, 0] = np.nextafter(nudged[0, 0, 0, 0], np.inf)
    assert variant(diagonal=nudged).skeleton != plan.skeleton
    # The gather pattern, by value, likewise.
    shifted = np.array(term.index, copy=True)
    shifted[0, 0] += 1
    assert variant(index=shifted).skeleton != plan.skeleton
    # And the selector's own word is in it as text.
    assert variant().skeleton == plan.skeleton
    assert plan.with_static(terms=tuple(
        t.__class__(t.name, t.field, t.region, t.ratio, t.masks, components=t.components,
                    discretisation='voxel', index=t.index, output=t.output,
                    diagonal=t.diagonal, key=t.key) if t is term else t
        for t in plan.terms)).skeleton != plan.skeleton


def test_two_processes_agree_on_a_prepared_skeleton():
    """A prepared plan's skeleton is the same number in a fresh interpreter.

    The freezers' two-freeze check rests on this, and a digest over arrays is exactly where it
    could stop being true. Thread count is held at the value the rest of the suite runs at:
    the assembled matrix's data is known to differ between 1, 2 and 4 BLAS threads (commit
    c939ed3), so this says "the same code and the same threads give the same number", which is
    what a skeleton has to promise, and not that the operator is thread independent.
    """
    root = str(__import__('pathlib').Path(fd.__file__).resolve().parents[2])
    code = ('import sys, warnings\n'
            f'sys.path[:0] = [{root!r}, {str(__import__("pathlib").Path(__file__).parent)!r}]\n'
            'warnings.simplefilter("ignore")\n'
            'import test_fd_stencil as t\n'
            'print(t.prepared()[1].skeleton)\n')
    env = dict(__import__('os').environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               MKL_NUM_THREADS='2', PYTHONUTF8='1')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                         timeout=900, env=env, cwd=root)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == prepared()[1].skeleton


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
    assert 'StencilShapeRefused' in str(err.value)
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


def test_a_predicted_preparation_above_the_declared_bound_is_refused_before_it_starts():
    """Both sides, and the prediction is shut *before* the solve rather than after it.

    The bound is declared in bytes and the prediction is two measured constants times a row
    count. The refusal has to come first: the solve is what would take the memory, so a door
    behind it protects nothing (memory ``laptop-ram-panics-not-just-slows``).
    """
    big = scene(radius=.3, cpw=20)
    with pytest.raises(ValueError) as err:
        quiet(build_plan, big)
    message = str(err.value)
    assert 'StencilMemoryRefused' in message and 'prediction' in message
    # The bound is what decides it, not the scene: raise it and the same scene passes the door
    # and goes on to fail somewhere else entirely (or succeed), which is what "this door and
    # no other" means.
    original = stencil.MEMORY_BOUND_BYTES
    stencil.MEMORY_BOUND_BYTES = 8 * 1024 ** 3
    try:
        with pytest.raises(ValueError) as later:
            quiet(build_plan, big)
        assert 'StencilMemoryRefused' not in str(later.value)
    finally:
        stencil.MEMORY_BOUND_BYTES = original
    # And the fixture is under the bound, which is the other side of the door.
    assert stencil_term(prepared()[1]) is not None


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

    The off-diagonal coefficients are held beside the plan, so they need a key, and the term's
    *name* is not one: ``constitutive[E0]`` is a position in a scene, not an identity, and
    every scene with one smoothed region calls its term that. Keyed by name, the second
    preparation would overwrite the first and the first plan would then lower the second's
    numbers under its own pattern -- the same array shapes, a plausible run, a different
    medium. The key is the prepared operator's own digest instead.

    Both plans are lowered *after* both preparations, which is the order that makes the defect
    visible at all.
    """
    small, small_plan = prepared()
    big = scene(radius=.17)
    big_plan = quiet(build_plan, big)
    a, b = stencil_term(small_plan), stencil_term(big_plan)
    assert a.name == b.name and a.key != b.key
    first = lower_params(small, small_plan, 'numpy')
    second = lower_params(big, big_plan, 'numpy')
    assert not np.array_equal(np.asarray(first['stencil'][a.name]['value']),
                              np.asarray(second['stencil'][b.name]['value']))
    assert np.array_equal(np.asarray(first['stencil'][a.name]['value']),
                          stencil._prepared[a.key])


def test_a_plan_prepared_elsewhere_is_refused_rather_than_lowered_without_its_coefficients():
    """The off-diagonal coefficients are held beside the plan, and their absence is named.

    They are floats the kernel multiplies by, so they belong in ``params`` and not in a
    hashable plan; the plan carries their pattern and its own diagonal by digest instead.
    What must never happen is a lowering that finds them missing and runs the diagonal alone,
    which is the extended stencil's diagonal without its coupling -- a third discretisation
    nobody asked for.
    """
    s, plan = prepared()
    term = stencil_term(plan)
    held = stencil._prepared.pop(term.key)
    try:
        with pytest.raises(ValueError) as err:
            lower_params(s, plan, 'numpy')
        assert 'StencilUnpreparedRefused' in str(err.value)
    finally:
        stencil._prepared[term.key] = held
