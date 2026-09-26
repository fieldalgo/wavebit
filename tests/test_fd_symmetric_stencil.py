"""Independent controls of the experimental extended material stencil."""
import contextlib
from functools import lru_cache
import pathlib
import subprocess
import sys
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from studies import fdtd_subpixel_symmetric_stencil as study
import wavebit
from wavebit.fd._runtime import runtime
from wavebit.fd._kernel.masks import Region

ROOT = pathlib.Path(wavebit.__file__).resolve().parents[1]


def test_the_moved_operator_is_an_fd_module_that_imports_with_no_cycle():
    """The five assembly functions are ``wavebit.fd._lower.stencil``'s, and it stands alone.

    The move put them in the package while leaving every caller in the study, so two things
    have to hold at once. The names the study exposes have to be the moved ones and not
    copies -- ``__module__`` says which -- and the new module has to import on its own, with
    no study pulled in behind it and no cycle through ``wavebit.fd``. The subprocess runs
    with the checkout as its working directory, so ``studies`` *is* importable and simply is
    not imported; that is the claim, not an accident of the path.

    None of them reaches into ``studies/`` any more: ``geometry`` takes its exact edge and face
    fractions from ``wavebit/fd/_shapes`` and ``triplet_baseline`` its maps from
    ``wavebit/fd/_lower/triplet.py`` (2026-09-26). The subprocess shows it by assembling the
    default operator, triplet baseline and all, on a 2-D and a 3-D fixture and finding no study
    in ``sys.modules`` afterwards, and by running ``energy_bounds`` and ``padded`` beside it.
    What happens where ``studies/`` is not importable at all is
    :func:`test_the_operator_builds_with_no_study_and_is_the_same_to_the_bit`.

    ``surface_clearance`` moved the same way in the commit that gave a public scene the
    selector, because the public preparation has to apply the **same** clearance rule and not
    a second spelling of it. It is checked here with the other five: what the study exposes
    under that name has to be the moved function and not a copy.
    """
    from wavebit.fd._lower import stencil
    moved = (study.fit, study.energy_bounds, study.triplet_baseline, study.geometry,
             study.padded, study.surface_clearance)
    assert [f.__module__ for f in moved] == ['wavebit.fd._lower.stencil'] * 6
    # Exactly the moved functions and the public preparation's own names -- an equality, not
    # a subset, so that a name added to ``__all__`` that is neither has to be added here on
    # purpose (review: the subset form let ``__all__`` grow unseen).
    preparation = {'DEFAULT', 'MEMORY_BOUND_BYTES', 'REACH', 'STENCIL', 'active_rows',
                   'arguments', 'lower', 'operator_digest', 'predicted_bytes', 'prepare',
                   'prepared_digest', 'refuse_discretisation', 'source_digest'}
    assert set(stencil.__all__) == {f.__name__ for f in moved} | preparation
    assert len(stencil.__all__) == len(set(stencil.__all__))
    assert list(stencil.__all__) == sorted(stencil.__all__)
    code = ('import sys\n'
            'import numpy as np\n'
            'from scipy.sparse import csr_matrix\n'
            'import wavebit.fd._lower.stencil as s\n'
            'assert s.fit.__module__ == "wavebit.fd._lower.stencil", s.fit.__module__\n'
            'q = np.arange(-3, 4) * .1\n'
            'z, y = np.meshgrid(q, q, indexing="ij")\n'
            'plane = np.stack((y, z), axis=-1)[:, :, None, :]\n'
            'z, y, x = np.meshgrid(q, q, q, indexing="ij")\n'
            'space = np.stack((x, y, z), axis=-1)\n'
            'for points, parts, centre in ((plane, (1, 2), (.017, -.023)),\n'
            '                              (space, (0, 1, 2), (.017, -.023, .009))):\n'
            '    assert s.geometry(points, .1, centre, .16, 12., parts)[0].nnz\n'
            'assert s.energy_bounds(csr_matrix([[2., .25], [.25, .5]]))["energy_lower_bound"] > 0\n'
            'diagonal = s.padded(csr_matrix(np.eye(3)))[0].tolist()\n'
            'leaked = sorted(m for m in sys.modules if m.split(".")[0] == "studies")\n'
            'assert not leaked, leaked\n'
            'print(diagonal)\n')
    out = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == '[1.0, 1.0, 1.0]'


class _NoStudies:
    """A finder that hides ``studies`` the way an installed wheel does: it is simply not there.

    A meta-path entry rather than a path edit, because the checkout *is* on ``sys.path`` while
    the tests run and the thing under test is what happens when the package is installed
    without it. Raising from ``find_spec`` is how the import machinery is told "no", and it
    arrives at the caller as the ``ImportError`` an absent module gives.
    """

    def find_spec(self, name, path=None, target=None):
        if name == 'studies' or name.startswith('studies.'):
            raise ImportError(f'No module named {name!r} (hidden by the test)')
        return None


@contextlib.contextmanager
def no_studies():
    """``studies`` unimportable, and every study module already imported put back afterwards."""
    finder = _NoStudies()
    saved = {name: module for name, module in sys.modules.items()
             if name == 'studies' or name.startswith('studies.')}
    for name in saved:
        del sys.modules[name]
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
        sys.modules.update(saved)


def test_the_operator_builds_with_no_study_and_is_the_same_to_the_bit():
    """Where ``studies`` cannot be imported, every route of this module builds, and unchanged.

    ``geometry``'s exact edge and face fractions are ``wavebit/fd/_shapes``'s -- the
    ``Sphere.exact_edge_fill`` and ``exact_face_fill`` the study helpers wrap -- and since
    2026-09-26 so are ``triplet_baseline``'s orientation table and local maps, moved out of the
    two triplet studies into ``wavebit/fd/_lower/triplet.py`` unchanged. So with ``studies``
    hidden the way an installed wheel hides it, the default triplet baseline and the diagonal
    one both build, and each operator is compared with the one built while ``studies`` is
    importable **bit for bit**, on both the 2-D and the 3-D fixture, as is the triplet baseline
    alone. The comparison is what says nothing falls back when the checkout is missing: a route
    that answered some other way there would still build, and would differ. (The two ARPACK
    report scalars are held out for the reason ``docs/fd-symmetric-stencil.md`` records:
    ``eigsh`` draws its own starting vector, and two runs of one unchanged tree already
    disagree on them.)

    Before that date the triplet baseline refused here by name (``StencilStudyRefused``), and
    the name went with the seam.

    Mutant: ``triplet_baseline`` taking ``sphere_maps`` from ``studies`` again through a
    deferred import. Measured 2026-09-26 over the 138 fast cases of the four files that gate
    this operator: three go red -- this one on the hidden import's ``ImportError``,
    :func:`test_the_moved_operator_is_an_fd_module_that_imports_with_no_cycle` on the seven
    study modules its subprocess then finds loaded, and ``tests/test_fd_stencil.py``'s
    ``test_a_stencil_scene_prepares_where_studies_cannot_be_imported`` -- and nothing else
    moves.
    """
    from wavebit.fd._lower import stencil
    q = np.arange(-3, 4) * .1
    z, y = np.meshgrid(q, q, indexing='ij')
    plane = np.stack((y, z), axis=-1)[:, :, None, :]
    z, y, x = np.meshgrid(q, q, q, indexing='ij')
    space = np.stack((x, y, z), axis=-1)
    fixtures = ((plane, (1, 2), (.017, -.023)), (space, (0, 1, 2), (.017, -.023, .009)))
    held = ('minimum_eigenvalue', 'eigenpair_residual')

    def same(matrix, want):
        assert matrix.shape == want.shape and matrix.nnz and np.isfinite(matrix.data).all()
        assert np.array_equal(matrix.data, want.data)
        assert np.array_equal(matrix.indices, want.indices)
        assert np.array_equal(matrix.indptr, want.indptr)

    def build(kind):
        return [stencil.geometry(points, .1, centre, .16, 12., components, baseline_kind=kind)
                for points, components, centre in fixtures]

    def baselines():
        return [stencil.triplet_baseline(points, .1, centre, .16, 12., components)
                for points, components, centre in fixtures]
    with_study = {kind: build(kind) for kind in ('triplet', None)}
    with_study_baselines = baselines()
    with no_studies():
        with pytest.raises(ImportError, match='No module named'):
            import studies                                                    # noqa: F401
        for kind, wanted in with_study.items():
            for (matrix, report), (want, want_report) in zip(build(kind), wanted):
                same(matrix, want)
                assert ({k: v for k, v in report.items() if k not in held}
                        == {k: v for k, v in want_report.items() if k not in held})
        for got, want in zip(baselines(), with_study_baselines):
            same(got, want)


def test_the_triplet_studies_hand_out_the_packages_maps():
    """The two triplet studies' maps are ``wavebit/fd/_lower/triplet.py``'s objects, not copies.

    The move left the studies' names in place for their own drivers and tests, which is the same
    arrangement ``test_the_moved_operator_is_an_fd_module_that_imports_with_no_cycle`` checks
    for the stencil's assembly, with the same two ways to go wrong: a study growing its own copy
    again -- its research tests would then judge a body the public baseline no longer reads --
    and the package module growing a name nobody put there on purpose. ``is`` rules out the
    first, an exact sorted ``__all__`` the second.

    Mutant: the sphere study defining its own ``sphere_maps`` again, a wrapper over the moved
    one. Measured 2026-09-26: this case goes red, and it is the only one of the 138 fast cases
    in the four files that gate the operator to move -- every number is still the same, which
    is exactly why nothing else can see it.
    """
    from studies import fdtd_subpixel_triplet_cylinder as cylinder
    from studies import fdtd_subpixel_triplet_sphere as sphere
    from wavebit.fd._lower import triplet
    pairs = ((sphere.ORIENTATIONS, triplet.TRIPLETS), (sphere.local_map, triplet.triplet_map),
             (sphere.sphere_maps, triplet.sphere_maps), (cylinder.ORIENTATIONS, triplet.PAIRS),
             (cylinder.local_map, triplet.pair_map),
             (cylinder.circle_fraction, triplet.circle_fraction),
             (cylinder.circle_maps, triplet.circle_maps))
    assert [a is b for a, b in pairs] == [True] * len(pairs)
    assert triplet.__all__ == sorted(triplet.__all__)
    assert set(triplet.__all__) == {'PAIRS', 'TRIPLETS', 'circle_fraction', 'circle_maps',
                                    'pair_map', 'sphere_maps', 'triplet_map'}
    assert len(set(triplet.TRIPLETS)) == 8 and len(set(triplet.PAIRS)) == 4


def test_a_complex_permittivity_is_refused_where_the_buffers_are_real():
    """A lossy ``eps`` has no operator here, and the refusal says so instead of truncating it.

    ``fit`` assembles into preallocated ``float64`` buffers (``rhs``, ``vv``, ``values``), and a
    complex ``eps`` reaches every one of them through ``gamma``, ``pis`` and ``pi``. NumPy does
    not refuse that assignment: it casts, with a ``ComplexWarning`` that a caller filtering
    warnings never sees. Measured on the fixture below with the guard removed: twenty
    ``ComplexWarning``s, and the matrix that comes back is ``complex128`` whose **fitted
    correction is exactly real** -- ``(matrix - baseline).data.imag`` is zero everywhere, at
    ``eps = 12 + 1e-9j`` and equally at ``eps = 12 + 5j``, where the loss is 40% of the
    permittivity. All of the loss that survives is the baseline's diagonal ``1/eps``; the
    interface correction, which is the entire point of this operator, is the lossless one.

    The choice is a refusal and not complex buffers because the routine is real symmetric
    throughout: ``lsmr`` solves a real least-squares problem, ``eigsh(which='SA')`` asks for a
    smallest *algebraic* eigenvalue, and ``energy_bounds`` certifies an LDL with positive
    pivots. Complex buffers would carry the imaginary part into arithmetic that has no meaning
    for it. The public route cannot deliver a lossy material today -- ``fd.Material`` refuses a
    complex permittivity and ``prepare`` refuses a Conductor or a Dispersive -- and this says
    so by name as well, rather than by the absence of a caller.

    All five inputs the buffers read are doored, because ``eps`` is not the only way in -- a
    complex ``ell``, ``area``, ``normals`` or baseline reaches the same three buffers.

    The mutant is the refusal deleted. Measured 2026-09-24: this case goes red on the first
    ``DID NOT RAISE`` and the other 67 in this file pass, which is the defect exactly -- a
    truncation nothing else in the suite is in a position to notice.
    """
    points = np.array([[.5, 0.], [0., .5], [1.5, 0.], [1., .5]])
    axes = np.array([0, 1, 0, 1])
    normals = np.broadcast_to(np.array([.6, .8]), points.shape)
    signed = np.array([-.1, .1, .2, -.2])
    ell = np.array([.3, .7, .4, .6])
    area = np.array([.7, .3, .6, .4])
    real = dict(points=points, axes=axes, normals=normals, signed=signed, ell=ell, area=area)
    matrix, report = study.fit(eps=12., **real)
    assert matrix.shape == (4, 4) and np.isfinite(matrix.data).all()
    assert (matrix - matrix.T).nnz == 0 and report['unknowns'] > 0
    for name, changed in (('eps', dict(eps=12. + 1e-9j)),
                          ('eps', dict(eps=complex(12.))),
                          ('ell', dict(eps=12., ell=ell.astype(complex))),
                          ('area', dict(eps=12., area=area.astype(complex))),
                          ('normals', dict(eps=12., normals=normals.astype(complex)))):
        with pytest.raises(ValueError, match='StencilComplexRefused') as caught:
            study.fit(**{**real, **changed})
        assert name in str(caught.value)
    with pytest.raises(ValueError, match='StencilComplexRefused'):
        study.fit(eps=12., baseline=csr_matrix(np.eye(len(points)) * (1 + 0j)), **real)
    # The control: the same baseline spelled out in real arithmetic is accepted, and it is
    # the one the default builds -- so the door refuses the dtype and not the argument.
    from scipy.sparse import diags
    spelled = diags(np.where(signed < 0, 1 / 12., 1.)).tocsr()
    plain, _ = study.fit(eps=12., baseline=spelled, **real)
    assert np.array_equal(plain.data, matrix.data) and np.array_equal(plain.indices,
                                                                      matrix.indices)


def planar_fixture(size, normal, eps, offset=.23):
    """Exact periodic piecewise-constant E_t/D_n states on parallel slabs."""
    z,y=np.meshgrid(np.arange(size),np.arange(size),indexing='ij')
    p=np.stack((y,z),axis=-1).reshape(-1,2)
    points=np.concatenate((p+[.5,0.],p+[0.,.5]))
    axes=np.repeat([0,1],len(p));n=np.array(normal)
    integer=np.array([1.,0.]) if normal[0]==1 else np.array([3.,4.])
    phase=(points@integer-offset)%size
    signed=np.where(phase<size/2,-np.minimum(phase,size/2-phase),
                    np.minimum(phase-size/2,size-phase))/np.linalg.norm(integer)
    def average(axis):
        half=abs(integer[axis])/2
        if half==0:return (phase<size/2).astype(float)
        lo,hi=phase-half,phase+half
        return sum(np.maximum(0,np.minimum(hi,k*size+size/2)-np.maximum(lo,k*size))
                   for k in (-1,0,1))/(2*half)
    ell=np.choose(axes,[average(0),average(1)])
    area=np.choose(axes,[average(1),average(0)])
    matrix,report=study.fit(points,axes,np.broadcast_to(n,points.shape),signed,ell,area,eps,periodic=size)
    tangent=np.array([-n[1],n[0]])
    electric=np.stack((tangent[axes],(ell/eps+1-ell)*n[axes]),axis=-1)
    displacement=np.stack(((area*eps+1-area)*tangent[axes],n[axes]),axis=-1)
    return matrix,report,electric,displacement


@lru_cache(maxsize=2)
def small_geometry(dim):
    q=np.arange(-4,5)*.1
    if dim==2:
        z,y=np.meshgrid(q,q,indexing='ij')
        points=np.stack((y,z),axis=-1)[:,:,None,:]
        components=(1,2);center=(.017,-.023)
    else:
        z,y,x=np.meshgrid(q,q,q,indexing='ij')
        points=np.stack((x,y,z),axis=-1)
        components=(0,1,2);center=(.017,-.023,.009)
    matrix,report=study.geometry(points,.1,center,.16,12.,components)
    return points,components,center,matrix,report


def increment_gap(dim,backend,diagonal_only=False):
    points,components,_,matrix,_=small_geometry(dim)
    shape=points.shape[:-1]
    if diagonal_only:
        from scipy.sparse import diags
        matrix=diags(matrix.diagonal()).tocsr()
    diagonal,index,value,output=study.padded(matrix)
    rt=runtime('jax' if backend=='jax_jit' else backend)
    region=Region(z=(0,shape[0]),y=(0,shape[1]),x=(0,shape[2]))
    # A reproducible nonzero increment and memory are experiment inputs.
    term=study.StencilTerm('s','E',region,.037,(),components)
    local=np.sin(np.arange(3*np.prod(shape))).reshape((3,)+shape)
    memory=np.cos(np.arange(3*np.prod(shape))).reshape((3,)+shape)
    def step(curl,mem,idx,val,rows):
        return term.prepare(curl,mem,dict(stencil_index=idx,stencil_value=val,stencil_output=rows),rt)
    if backend=='jax_jit':
        import jax
        step=jax.jit(step)
    args=[rt.array(local,'float64'),rt.array(memory,'float64'),rt.array(index),rt.array(value,'float64'),rt.array(output)]
    flux,delta=step(*args);flux=rt.numpy(flux);delta=rt.numpy(delta)
    selected=np.stack(tuple(local[c] for c in components)).ravel()
    actual=diagonal*np.stack(tuple(flux[c]-memory[c] for c in components)).ravel()
    actual+=np.stack(tuple(delta[c] for c in components)).ravel()
    return float(np.max(abs(actual-.037*(matrix@selected))))


@pytest.mark.parametrize('normal',((1.,0.),(.6,.8)))
def test_exact_static_planar_states(normal):
    for eps in (1.,2.25,12.):
        for size in (16,24,32):
            matrix,report,electric,displacement=planar_fixture(size,normal,eps)
            # Eighteen preliminary cases measured at most 6.85563e-15.
            np.testing.assert_allclose(matrix@displacement,electric,rtol=0,atol=1e-13)
            assert report['minimum_eigenvalue']>0
            assert (matrix-matrix.T).nnz==0


def test_thirty_planar_interface_offsets():
    # Thirty distinct subcell phases; .37 avoids the aligned endpoint.
    for i in range(30):
        matrix,report,electric,displacement=planar_fixture(16,(.6,.8),12.,(i+.37)/30)
        # The preceding contrast/size audit measured 6.85563e-15.
        np.testing.assert_allclose(matrix@displacement,electric,rtol=0,atol=1e-13)
        assert report['minimum_eigenvalue']>0


@pytest.mark.parametrize('dim',(2,3))
@pytest.mark.parametrize('backend',('numpy','jax','jax_jit','torch'))
def test_increment_matches_independent_sparse_product(dim,backend):
    # Eight premeasurements, nonzero flux memory: maximum 1.08421e-16.
    assert increment_gap(dim,backend)<2e-15


@pytest.mark.parametrize('dim',(2,3))
@pytest.mark.parametrize('backend',('numpy','jax','jax_jit','torch'))
def test_empty_stencil_matches_diagonal_increment(dim,backend):
    # The same nonzero-memory arithmetic measured <=1.08421e-16 above.
    assert increment_gap(dim,backend,diagonal_only=True)<2e-15


@pytest.mark.parametrize('empty',(False,True))
def test_padding_keeps_only_nonzero_rows_and_an_exact_zero_slot(empty):
    from scipy.sparse import diags
    matrix=diags(np.arange(1.,6.)).tocsr()
    if not empty:
        matrix+=csr_matrix(([.125,.125,-.25,-.25],([0,3,1,2],[3,0,2,1])),shape=(5,5))
    diagonal,index,value,output=study.padded(matrix)
    cross=matrix-diags(diagonal);cross.eliminate_zeros()
    active=np.flatnonzero(np.diff(cross.indptr))
    assert index.shape==value.shape==(len(active),0 if empty else 1)
    assert np.array_equal(output[active],np.arange(1,len(active)+1))
    assert np.array_equal(output[np.setdiff1d(np.arange(5),active)],0*np.ones(5-len(active)))
    vector=np.arange(1.,6.)
    # All fixture coefficients are binary fractions: the reconstruction
    # is exact, and no measured floating tolerance is needed here.
    actual=np.concatenate(([0.],np.sum(vector[index]*value,axis=1)))[output]
    np.testing.assert_array_equal(actual,cross@vector)


def test_rigid_translation_keeps_the_material_operator():
    q=np.arange(-10,11)/40
    z,y=np.meshgrid(q,q,indexing='ij')
    points=np.stack((y,z),axis=-1)[:,:,None,:]
    matrix,_=study.geometry(points,.025,(0.,0.),.15,12.,(1,2))
    for shift in ((.3,.7),(1.1,-.7)):
        other,_=study.geometry(points+shift,.025,shift,.15,12.,(1,2))
        # Grazing cuts measured 4.04286e-9; twice that rounded upward.
        # The rejected hard support measured 1.41418e-3 on these inputs.
        assert abs(matrix-other).max()<1e-8


def test_energy_bounds_enclose_independent_eigenvalues():
    matrix=csr_matrix([[2.,.25],[.25,.5]])
    bounds=study.energy_bounds(matrix)
    low,high=np.linalg.eigvalsh(matrix.toarray())
    assert 0<bounds['energy_lower_bound']<=low
    assert bounds['spectral_upper_bound']>=high
    # Measured Cholesky reconstruction <=7.82e-16 on the full sphere.
    assert bounds['factor_relative_residual']<1e-14


def test_sparse_certificate_encloses_nonlocal_positive_matrix():
    # A known SPD Gram matrix with nonlocal couplings and mixed signs.
    rng=np.random.default_rng(10913)
    basis=np.eye(17)+np.tril(rng.uniform(-.4,.4,size=(17,17)),-1)
    matrix=basis@basis.T
    eigenvalues=np.linalg.eigvalsh(matrix)
    failure=None
    try:
        bounds=study.energy_bounds(csr_matrix(matrix))
    except ValueError as error:
        failure=str(error)
    assert failure is None, failure
    assert 0<bounds['energy_lower_bound']<=eigenvalues[0]
    assert bounds['spectral_upper_bound']>=eigenvalues[-1]
    # Radius-4/10 sparse screens measured reconstruction <=1.337e-15.
    assert bounds['factor_relative_residual']<1e-14


@pytest.mark.parametrize('broken,reason',(
    ([[-1.,0.],[0.,1.]],'StencilEnergyRefused'),
    ([[2.,1.],[0.,2.]],'StencilSymmetryRefused'),
    ([[float('nan'),0.],[0.,1.]],'StencilFiniteRefused'),
))
def test_invalid_energy_is_refused_before_stepping(broken,reason):
    with pytest.raises(ValueError,match=reason):
        study.energy_bounds(csr_matrix(broken))


@pytest.mark.parametrize('broken,reason', (
    ([[1., 2.], [2., 1.]], 'nonpositive LDL pivot'),
    ([[1., 1.], [1., 1.]], 'sparse factorization failed'),
))
def test_an_indefinite_or_singular_coupled_block_is_refused_by_its_factorization(broken, reason):
    """The LDL of the coupled block: a negative pivot and an exactly singular block.

    Every row of these two is coupled, so the diagonal screen in front of the factorization
    sees nothing wrong and the refusal has to come from the factorization itself: pivots
    1 and 1 - 4 = -3 for the first, 1 and 1 - 1 = 0 (singular) for the second. The public
    preparation calls this same function (``wavebit/fd/_lower/stencil.py::_assemble``).
    """
    with pytest.raises(ValueError, match=f'StencilEnergyRefused: {reason}'):
        study.energy_bounds(csr_matrix(broken))
    assert study.energy_bounds(csr_matrix([[2., .25], [.25, .5]]))['energy_lower_bound'] > 0


def test_the_factorization_certificate_refuses_what_it_cannot_certify(monkeypatch):
    """The two refusals no matrix reaches honestly, reached by replacing SciPy's answers.

    ``splu`` in symmetric mode with no pivoting threshold orders rows and columns alike, and
    the comparison-inverse solves of a unit triangular factor are finite -- so a symmetric
    positive matrix cannot make either door fire. They are guards against SciPy doing
    otherwise, and the only way to see them is to make it do otherwise.
    """
    import scipy.sparse.linalg as linalg
    good = csr_matrix([[2., .25], [.25, .5]])
    real = linalg.splu

    class Unordered:
        def __init__(self, factor):
            self.factor = factor

        def __getattr__(self, name):
            if name == 'perm_c':
                return self.factor.perm_c[::-1].copy()
            return getattr(self.factor, name)
    monkeypatch.setattr(linalg, 'splu', lambda *a, **k: Unordered(real(*a, **k)))
    with pytest.raises(ValueError, match='not symmetrically ordered'):
        study.energy_bounds(good)
    monkeypatch.setattr(linalg, 'splu', real)
    monkeypatch.setattr(linalg, 'spsolve_triangular', lambda *a, **k: np.full(2, np.nan))
    with pytest.raises(ValueError, match='nonfinite comparison-inverse bound'):
        study.energy_bounds(good)


@pytest.mark.parametrize('factor',(2.,float('nan'),0.))
def test_invalid_time_step_is_refused_before_stepping(factor):
    import dataclasses
    from studies.fdtd_subpixel_coupled_cylinder import prepare
    from studies.fdtd_subpixel_mie_cylinder import scene_for,SCHEDULE
    from wavebit.fd._lower import plan,lower
    empty=scene_for(40,'TE',*SCHEDULE[12.])
    scene,p=prepare(empty,plan(empty),12.,(0.,0.))
    # The original dt^2 rate is .282553; doubling dt makes it >1.
    bad=dataclasses.replace(p,dt=factor*p.dt)
    with pytest.raises(ValueError,match='StencilCFLRefused'):
        study.replace(scene,bad,lower(scene,p,'numpy'),(0.,0.),.15,12.,(1,2))


@pytest.mark.slow
@pytest.mark.parametrize('shape',('cylinder','sphere'))
def test_same_operator_physical_controls(shape):
    row=(study.cylinder if shape=='cylinder' else study.sphere)()
    # Preserve the previous independent physical-instrument limits:
    # cylinder 5%, sphere 3%. This candidate measured 3.51206%/2.46813%.
    assert abs(row['error'])<(.05 if shape=='cylinder' else .03)
    decay=row['decay'] if shape=='cylinder' else max(row['decay'].values())
    # Existing cylinder settling control, measured here <=3.729e-9.
    assert decay<1e-7
    assert row['preparation']['energy_lower_bound']>0
    assert row['preparation']['dt_squared_rate']<1



@lru_cache(maxsize=1)
def scope_fixture():
    from studies.fdtd_subpixel_coupled_cylinder import prepare
    from studies.fdtd_subpixel_mie_cylinder import scene_for,SCHEDULE
    from wavebit.fd._lower import plan
    empty=scene_for(40,'TE',*SCHEDULE[12.])
    return prepare(empty,plan(empty),12.,(0.,0.))


@pytest.mark.parametrize('fault',(
    'empty_scene','two_structures','empty_terms','magnetic_term','masked_term',
    'background','magnetic_background','conductivity','anisotropy','magnetic_material',
    'material_mismatch','center_mismatch','radius_mismatch','components','invariant_axis',
    'dtype','initial_field','nan_radius','nan_center','jax_parameter','torch_parameter',
))
def test_unsupported_research_scope_is_refused(fault):
    import dataclasses
    import wavebit.fd as fd
    from wavebit.fd._kernel.masks import Slab
    scene,p=scope_fixture();old=p.constitutive[0]
    args=dict(center=(0.,0.),radius=.15,eps=12.,components=(1,2))
    if fault=='empty_scene':scene=scene.replace(structures=())
    elif fault=='two_structures':scene=scene.replace(structures=scene.structures*2)
    elif fault=='empty_terms':p=p.with_static(terms=tuple(t for t in p.terms if t is not old))
    elif fault in ('magnetic_term','masked_term'):
        bad=dataclasses.replace(old,**({'field':'H'} if fault=='magnetic_term'
            else {'masks':(Slab(1,old.region.z,old.region.y,old.region.x),)}))
        p=p.with_static(terms=tuple(bad if t is old else t for t in p.terms))
    elif fault=='background':scene=scene.replace(background=2.)
    elif fault=='magnetic_background':scene=scene.replace(background=fd.Material(eps=1.,mu=1.))
    elif fault in ('conductivity','anisotropy','magnetic_material','material_mismatch'):
        material={'conductivity':fd.Conductor(eps=12.,sigma=.1),
                  'anisotropy':fd.Material(eps=(12.,12.,13.)),
                  'magnetic_material':fd.Material(eps=12.,mu=1.),
                  'material_mismatch':fd.Material(eps=2.25)}[fault]
        scene=scene.replace(structures=(fd.Structure(scene.structures[0].shape,material),))
    elif fault=='center_mismatch':args['center']=(.01,0.)
    elif fault=='radius_mismatch':args['radius']=.16
    elif fault=='components':args['components']=(0,2)
    elif fault=='invariant_axis':p=dataclasses.replace(p,nx=2)
    elif fault=='dtype':p=dataclasses.replace(p,dtype='float32')
    elif fault=='initial_field':p=dataclasses.replace(p,has_E0=True)
    elif fault=='nan_radius':args['radius']=float('nan')
    elif fault=='nan_center':args['center']=(float('nan'),0.)
    elif fault=='jax_parameter':
        import jax.numpy as jnp
        args['eps']=jnp.asarray(12.)
    elif fault=='torch_parameter':
        import torch
        args['eps']=torch.tensor(12.,requires_grad=True)
    with pytest.raises(ValueError,match='StencilScopeRefused'):
        study.validate_scope(scene,p,**args)


def test_replacement_refuses_scope_before_assembly(monkeypatch):
    scene,p=scope_fixture()
    def forbidden(*args,**kwargs):
        assert False, 'Unsupported material reached numerical assembly'
    monkeypatch.setattr(study,'geometry',forbidden)
    with pytest.raises(ValueError,match='StencilScopeRefused'):
        study.replace(scene.replace(background=2.),p,{},(0.,0.),.15,12.,(1,2))


@lru_cache(maxsize=3)
def sphere_surface_fixture(placement):
    import warnings
    from studies.fdtd_subpixel_mie_sphere import sphere_scene
    from wavebit.fd._lower import plan
    from wavebit.fd._lower.constitutive import common_points
    offsets=((0.,0.,0.),(.5,1/3,.2),(.25,2/3,.4))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        scene=sphere_scene(20,2.25,24.,offsets[placement],propagation='yee')
        p=plan(scene)
    box=tuple((lo-3,hi+3) for lo,hi in p.constitutive[0].box)
    nodes=common_points(p.grid,tuple(reversed(box)),'E')
    matrix,report=study.geometry(nodes,p.dx,scene.structures[0].shape.center,.25,2.25,(0,1,2))
    return scene,p,box,matrix,report


@pytest.mark.parametrize('placement',(1,2))
def test_cut_nonlocal_flux_surface_is_refused(placement):
    scene,p,box,matrix,_=sphere_surface_fixture(placement)
    # Measured outside coefficients: 1.60439e-8 / 3.88092e-7; exact
    # nonzero support is checked, with no fitted small-coefficient waiver.
    with pytest.raises(ValueError,match='StencilSurfaceRefused'):
        study.surface_clearance(scene,p,matrix,box,(0,1,2))


@pytest.mark.parametrize('placement,half',((0,.4),(2,.45)))
def test_flux_surface_encloses_the_complete_stencil(placement,half):
    import dataclasses
    import wavebit.fd as fd
    scene,p,box,matrix,_=sphere_surface_fixture(placement)
    scene=scene.replace(monitors=tuple(dataclasses.replace(m,box=fd.Region(size=(2*half,)*3))
        if m.name=='abs' else m for m in scene.monitors))
    failure=None
    try:reports=study.surface_clearance(scene,p,matrix,box,(0,1,2))
    except ValueError as error:failure=str(error)
    assert failure is None,failure
    assert dict((r['name'],r['relation']) for r in reports)['abs']=='encloses'


def test_replacement_checks_surfaces_before_energy_certification(monkeypatch):
    from wavebit.fd._lower import lower
    scene,p,box,matrix,report=sphere_surface_fixture(2)
    monkeypatch.setattr(study,'geometry',lambda *a,**k:(matrix,dict(report)))
    def forbidden(*args):
        assert False,'Cut surface reached the energy certificate'
    monkeypatch.setattr(study,'energy_bounds',forbidden)
    with pytest.raises(ValueError,match='StencilSurfaceRefused'):
        study.replace(scene,p,lower(scene,p,'numpy'),scene.structures[0].shape.center,.25,2.25,(0,1,2))


@pytest.mark.parametrize('kind',('tfsf','plane_wave'))
def test_source_injection_requires_a_vacuum_band(kind):
    import dataclasses
    import wavebit.fd as fd
    if kind=='tfsf':
        scene,p,box,matrix,_=sphere_surface_fixture(0);components=(0,1,2)
        source=dataclasses.replace(scene.sources[0],box=fd.Region(size=(.8,)*3))
    else:
        from wavebit.fd._lower.constitutive import common_points
        scene,p=scope_fixture();components=(1,2)
        box=tuple((lo-3,hi+3) if axis<2 else (lo,hi)
                  for axis,(lo,hi) in enumerate(p.constitutive[0].box))
        nodes=common_points(p.grid,tuple(reversed(box)),'E')[...,components]
        matrix,_=study.geometry(nodes,p.dx,(0.,0.),.15,12.,components)
        source=dataclasses.replace(scene.sources[0],z=0.)
    with pytest.raises(ValueError,match='StencilSourceRefused'):
        study.surface_clearance(scene.replace(sources=(source,)),p,matrix,box,components)


def test_interior_field_samples_are_allowed():
    import wavebit.fd as fd
    scene,p,box,matrix,_=sphere_surface_fixture(2)
    scene=scene.replace(monitors=(fd.Field('interior',box=fd.Region(size=(.3,)*3),wavelengths=(1.,)),))
    failure=None
    try:study.surface_clearance(scene,p,matrix,box,(0,1,2))
    except ValueError as error:failure=str(error)
    assert failure is None,failure


def test_research_source_scope_is_named():
    import wavebit.fd as fd
    scene,p,box,matrix,_=sphere_surface_fixture(0)
    source=fd.Dipole('point',component='Ex',center=(0.,0.,0.),band=fd.Band(wavelength=1.,df=.1))
    with pytest.raises(ValueError,match='StencilScopeRefused'):
        study.surface_clearance(scene.replace(sources=(source,)),p,matrix,box,(0,1,2))


def test_research_plane_flux_scope_is_named():
    import wavebit.fd as fd
    scene,p,box,matrix,_=sphere_surface_fixture(0)
    scene=scene.replace(monitors=(fd.Flux('plane',plane=fd.Plane(z=.5),wavelengths=(1.,)),))
    failure=None
    try:study.surface_clearance(scene,p,matrix,box,(0,1,2))
    except Exception as error:failure=error
    assert isinstance(failure,ValueError) and 'StencilScopeRefused' in str(failure)


def test_explicit_absorption_surface_reaches_preparation(monkeypatch):
    scene,p,box,matrix,_=sphere_surface_fixture(2)
    class PreparationObserved(Exception):pass
    def inspect(actual,actual_plan,*args):
        study.surface_clearance(actual,actual_plan,matrix,box,(0,1,2))
        raise PreparationObserved
    monkeypatch.setattr(study,'replace',inspect)
    observed=False
    try:study.sphere(20,2.25,24.,.25,(.25,2/3,.4),absorption_half=.45)
    except PreparationObserved:observed=True
    except ValueError:pass
    assert observed,'The explicitly enclosing surface never reached preparation'
