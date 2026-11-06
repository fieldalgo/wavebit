"""A traced transverse ``k`` is held to the grazing rule (issue #31).

``grazing._survey`` counts the diffraction orders and the PlaneWave's own incidence from the
transverse ``k`` on the host. A traced Bloch or Axis1D ``k`` (or a traced Bloch phase) had no
host value, so the survey skipped every row: a scene the concrete plan refuses ran with
``stable`` True. A PlaneWave cell is a plane wave only where its faces carry the wave's own
``k``, which is concrete (a concrete angle, band centre and background), so the survey now
reads a traced face's ``k`` as the wave's (``grazing._transverse_k``) and refuses by name
exactly as it does for the concrete scene. ``sources.bloch_match_guard`` then holds the live
``k`` to the wave's at run time, the check ``fd.plan`` makes of a concrete face. A traced
``k`` that does not match had run, silently, a sheet launched with one ``k`` into a cell
imposing another.

The cell is the traced-material probe's substrate cell (period .6, 8-cell PMLs, glass of
eps 2.25 filling the bottom layer) without its pillar, lit at ``phi = 0`` so that ``k`` lies
along x (along y, one cell, for Axis1D).
"""
import contextlib

import numpy as np
import pytest

import wavebit.fdtd._legacy as fd
from studies.fdtd.traced_material.fdtd_traced_material_probe import SUBSTRATE
from studies.fdtd.traced_material.fdtd_traced_material_probe import substrate_scene
from tests.fdtd.test_fd_sphere_range import quiet

#: The incidence the seed plan is built at: every row accepted.
SEED_THETA = 5.0
REFUSED = 'Grazing order refused'


def cell(theta=SEED_THETA, face=None, eps=2.25, monitors=True, axis1d=False, rho=None):
    """The cell lit at ``theta``, its faces from ``fd.Bloch.from_source``; ``face`` replaces
    the x faces (the y faces with ``axis1d``, which lights along y and makes y one cell),
    ``rho`` is the z faces' stabilizing loss."""
    base = substrate_scene(list(SUBSTRATE[:4]) + [eps], steps=20)
    pw = base.sources[0].replace(theta=theta, phi=90.0 if axis1d else 0.0)
    faces = fd.Bloch.from_source(pw, axis1d='y' if axis1d else ())
    b = base.boundaries.replace(x=(faces['x'],) * 2, y=(faces['y'],) * 2)
    s = base.replace(structures=base.structures[:1], sources=(pw,), boundaries=b)
    if axis1d:
        size = s.domain.size
        s = s.replace(domain=s.domain.replace(size=(size[0], s.resolution.dx, size[2])))
    if face is not None:
        s = s.replace(boundaries=s.boundaries.replace(**{'y' if axis1d else 'x': (face,) * 2}))
    if not monitors:
        s = s.replace(monitors=())
    if rho is not None:
        s = s.replace(boundaries=s.boundaries.replace(
            z=tuple(f.replace(stabilize=rho) for f in s.boundaries.z)))
    return s


def k_of(theta, axis1d=False):
    pw = cell(theta, axis1d=axis1d).sources[0]
    return fd.Bloch.from_source(pw, axis1d='y' if axis1d else ())['y' if axis1d else 'x'].k


def leaf(x):
    import torch
    return torch.tensor(x, dtype=torch.float64, requires_grad=True)


#: The grazing rule's refusals: a pure-stretch row, and a stabilized face's exact return.
REFUSALS = {REFUSED: 'refused', 'Stabilized face refused': 'refused on its exact return'}


def verdict(function, *args, **kwargs):
    """``True`` (ran stable), ``False`` (ran unstable), or which grazing refusal it met."""
    try:
        result = quiet(function, *args, **kwargs)
    except ValueError as error:
        kinds = [kind for start, kind in REFUSALS.items() if str(error).startswith(start)]
        assert kinds, error
        return kinds[0]
    return True if function is fd.plan else bool(result.diagnostics['stable'])


def traced(theta, plan, **kwargs):
    """The run of the cell with its k a torch leaf, against ``plan``."""
    axis1d = kwargs.get('axis1d', False)
    k = k_of(theta, axis1d)
    face = (fd.Axis1D(k=leaf(k), k_max=abs(k), algebraic=True) if axis1d
            else fd.Bloch(k=leaf(k)))
    return verdict(fd.run, cell(theta, face=face, **kwargs), 'torch', plan=plan)


def concrete(theta, **kwargs):
    return verdict(fd.plan, cell(theta, **kwargs))


def _edge(lo, hi):
    """The angle where the concrete verdict changes between ``lo`` and ``hi``, to 1e-9 deg."""
    at_lo = concrete(lo)
    while hi - lo > 1e-9:
        mid = (lo + hi) / 2.0
        lo, hi = (mid, hi) if concrete(mid) == at_lo else (lo, mid)
    return (lo + hi) / 2.0


def test_a_traced_bloch_k_is_refused_where_the_concrete_plan_refuses():
    """The refused band of incidences here (the (-1, 0) order grazing out through the glass,
    from 5.0324 to 21.3654 degrees, measured): the traced ``k`` is refused by name, with the
    concrete plan's message, at every angle the concrete plan refuses, and runs stable at every
    angle it accepts -- across the band and 1e-6 degree either side of each end."""
    p = quiet(fd.plan, cell())
    low, high = _edge(SEED_THETA, 10.0), _edge(20.0, 24.0)
    assert 5.03 < low < 5.04 and 21.36 < high < 21.37, (low, high)
    angles = (low - 1e-6, low + 1e-6, high - 1e-6, high + 1e-6) + tuple(np.linspace(0.5, 30., 8))
    seen = set()
    for theta in angles:
        want = concrete(theta)
        seen.add(want)
        assert traced(theta, p) == want, (theta, want)
    assert seen == {True, 'refused'}


def test_a_traced_k_must_be_the_waves_own():
    """``bloch_match_guard``: a traced ``k`` (or phase) within ``1e-9`` of the wave's runs, as
    a concrete one plans; one ``1e-6`` away is ``nan``, where a concrete one is refused."""
    p = quiet(fd.plan, cell())
    k = k_of(SEED_THETA)
    L = p.grid.size[0]
    for scale, want in ((1.0, True), (1.0 + 1e-12, True), (1.0 + 1e-6, False),
                        (1.0 - 1e-6, False)):
        with pytest.raises(ValueError, match='carries k') if not want else contextlib.nullcontext():
            quiet(fd.plan, cell(face=fd.Bloch(k=k * scale)))
        for face in (fd.Bloch(k=leaf(k * scale)), fd.Bloch(phase=leaf(k * scale * L))):
            assert verdict(fd.run, cell(face=face), 'torch', plan=p) is want, (scale, face)
    k = k_of(SEED_THETA, axis1d=True)
    p = quiet(fd.plan, cell(axis1d=True))
    for scale, want in ((1.0, True), (1.0 + 1e-6, False)):
        face = fd.Axis1D(k=leaf(k * scale), k_max=abs(k) * 1.1, algebraic=True)
        assert verdict(fd.run, cell(face=face, axis1d=True), 'torch', plan=p) is want, scale



def test_a_traced_bloch_phase_is_read_as_the_waves_k():
    p = quiet(fd.plan, cell())
    L = p.grid.size[0]
    for theta in (SEED_THETA, 15.0, 24.0):
        want = concrete(theta)
        face = fd.Bloch(phase=leaf(k_of(theta) * L))
        assert verdict(fd.run, cell(theta, face=face), 'torch', plan=p) == want, theta
    assert concrete(15.0) == 'refused'


def test_the_incidence_alone_is_held_without_a_diffraction_monitor():
    """No Diffraction monitor: no order is judged, but the PlaneWave's own incidence is, on
    its column's two faces; it is refused from about 70 degrees here."""
    p = quiet(fd.plan, cell(monitors=False))
    seen = []
    for theta in (40.0, 60.0, 70.0, 80.0):
        want = concrete(theta, monitors=False)
        seen.append(want)
        assert traced(theta, p, monitors=False) == want, theta
    assert seen == [True, True, 'refused', 'refused']


def test_a_traced_axis1d_k_is_held():
    """y Axis1D (one cell, no orders of its own) with its ``k`` traced, without and with the
    Diffraction monitors. Here the x orders other than (0, 0) are evanescent (g = 10.5 against
    n k0 <= 9.9 in the glass) and (0, 0) leaves the glass at cos >= 0.74, so it is the
    incidence that is refused, from about 70 degrees, in both."""
    for monitors, thetas in ((False, (40.0, 60.0, 70.0, 80.0)), (True, (5.0, 40.0, 70.0, 80.0))):
        p = quiet(fd.plan, cell(axis1d=True, monitors=monitors))
        seen = set()
        for theta in thetas:
            want = concrete(theta, axis1d=True, monitors=monitors)
            seen.add(want)
            assert traced(theta, p, axis1d=True, monitors=monitors) == want, (monitors, theta)
        assert seen == {True, 'refused'}, (monitors, seen)


def test_a_traced_k_and_a_traced_substrate_are_held_together():
    """Both leaves traced: the host bands of refused indices (``grazing.refused_bands``) are
    counted at the wave's ``k``, so ``grazing_guard`` holds the substrate's live index to them;
    without a ``k`` they were empty."""
    p = quiet(fd.plan, cell())
    seen = set()
    for theta in (SEED_THETA, 3.0):
        k = k_of(theta)
        for eps in (1.3, 2.0, 2.25, 2.4, 2.55):
            want = concrete(theta, eps=eps) == True        # noqa: E712 -- True or 'refused'
            seen.add(want)
            got = verdict(fd.run, cell(theta, face=fd.Bloch(k=leaf(k)), eps=leaf(eps)),
                          'torch', plan=p)
            assert got is want, (theta, eps)
    assert seen == {True, False}


def test_a_jitted_run_holds_a_traced_k():
    """JAX: ``k`` a jit argument, one program; the wave's own ``k`` runs stable, one 1e-6 away
    is ``nan``; a refused angle is refused by name while the program is traced."""
    from wavebit.fdtd._runtime import runtime
    runtime('jax')
    import jax
    import jax.numpy as jnp
    p = quiet(fd.plan, cell())

    def stable(k, theta=SEED_THETA):
        return quiet(fd.run, cell(theta, face=fd.Bloch(k=k)), 'jax', plan=p).diagnostics['stable']
    compiled = jax.jit(stable)
    k = k_of(SEED_THETA)
    assert bool(compiled(jnp.asarray(k)))
    assert not bool(compiled(jnp.asarray(k * (1 + 1e-6))))
    assert compiled._cache_size() == 1
    with pytest.raises(ValueError, match=REFUSED):
        jax.jit(lambda k: stable(k, 15.0))(jnp.asarray(k_of(15.0)))


def test_a_traced_k_beside_a_stabilized_face_is_judged_on_its_exact_return():
    """A stabilized face is judged on its exact discrete return as well, on the host: with
    the wave's ``k`` read for the traced one, the traced and the concrete verdicts agree. On
    these 8-cell faces the loss takes even the normal order past the rule, and inside the
    band of grazing angles the grazing order is the worse row; the traced scene used to run
    (planned from itself, its rows skipped)."""
    seen = []
    for theta in (SEED_THETA, 15.0, 30.0):
        want = concrete(theta, rho=.05)
        seen.append(want)
        assert traced(theta, None, rho=.05) == want, theta
    assert seen == ['refused on its exact return', 'refused', 'refused on its exact return']
