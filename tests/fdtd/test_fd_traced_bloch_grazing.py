"""A traced transverse ``k`` is held to the grazing rule at run time (issue #31).

``grazing._survey`` counts the diffraction orders and the PlaneWave's own incidence from the
transverse ``k`` on the host. A traced Bloch or Axis1D ``k`` (or a traced Bloch phase) has no
host value, so the survey skipped every row: a scene the concrete plan refuses ran with
``stable`` True. A traced ``k`` lowered through a plan is the run's ``k`` -- the PlaneWave column
is driven with the face's live value, which ``tests/fdtd/test_fd_stage3_grad.py`` sweeps through
one plan -- so ``shape_ranges.grazing_guard`` now judges those rows at the live ``k``
(``grazing.live_valid``): the run is ``nan`` exactly where the plan of the concrete scene at that
``k`` refuses a grazing order or incidence. A stabilized face, judged on its exact return on the
host only, is refused by name beside a traced ``k``.

The cell is the traced-material probe's substrate cell (period .6, 8-cell PMLs, glass of
eps 2.25 filling the bottom layer) without its pillar, lit at ``phi = 0`` so that ``k`` lies
along x (along y, one cell, for Axis1D).
"""
import math

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


def planned(scene):
    """Whether the concrete scene plans; any refusal but the grazing rule's is a failure."""
    try:
        quiet(fd.plan, scene)
        return True
    except ValueError as error:
        assert REFUSED in str(error), error
        return False


def stable(scene, plan, backend='torch'):
    return bool(quiet(fd.run, scene, backend, plan=plan).diagnostics['stable'])


def face_at(k, axis1d=False, k_max=None):
    if axis1d:
        return fd.Axis1D(k=k, k_max=k_max, algebraic=True)
    return fd.Bloch(k=k)


def _edge(lo, hi):
    """The angle where the concrete verdict changes between ``lo`` and ``hi``, to 1e-9 deg."""
    at_lo = planned(cell(lo))
    while hi - lo > 1e-9:
        mid = (lo + hi) / 2.0
        lo, hi = (mid, hi) if planned(cell(mid)) == at_lo else (lo, mid)
    return (lo + hi) / 2.0


def test_a_traced_k_runs_nan_where_the_concrete_plan_refuses():
    """The refused band of incidences here (the (-1, 0) order grazing out through the glass,
    from 5.0324 to 21.3654 degrees, measured): the traced ``k`` of the wave at each angle runs
    stable exactly where the concrete plan at that angle plans -- across the band and 1e-6
    degree either side of each end -- planned once at the seed."""
    p = quiet(fd.plan, cell())
    low, high = _edge(SEED_THETA, 10.0), _edge(20.0, 24.0)
    assert 5.03 < low < 5.04 and 21.36 < high < 21.37, (low, high)
    angles = (low - 1e-6, low + 1e-6, high - 1e-6, high + 1e-6) + tuple(np.linspace(0.5, 30., 8))
    seen = set()
    for theta in angles:
        want = planned(cell(theta))
        seen.add(want)
        assert stable(cell(theta, face=face_at(leaf(k_of(theta)))), p) is want, (theta, want)
    assert seen == {True, False}


def test_a_k_swept_through_one_plan_is_judged_at_its_live_value():
    """The PlaneWave declared at the seed angle and only ``k`` swept, as a sweep through one
    plan runs (the column is driven with the live ``k``): the verdict follows ``k``. ``k``
    reduced by whole ``g`` first, so ``k + m g`` -- the same orders, relabelled, and an
    evanescent incidence -- holds where ``k`` holds; ``3 g`` is past the window's own margin."""
    p = quiet(fd.plan, cell())
    g = 2 * math.pi / p.grid.size[0]
    for theta in (2.0, SEED_THETA, 10.0, 15.0, 20.0, 24.0):
        want = planned(cell(theta))
        for shift in (0.0, g, -g, 3 * g):
            k = leaf(k_of(theta) + shift)
            assert stable(cell(face=face_at(k)), p) is want, (theta, shift, want)
    assert not planned(cell(15.0)) and planned(cell(24.0))


def test_a_traced_bloch_phase_is_held_like_a_traced_k():
    p = quiet(fd.plan, cell())
    L = p.grid.size[0]
    for theta in (SEED_THETA, 15.0, 24.0):
        want = planned(cell(theta))
        face = fd.Bloch(phase=leaf(k_of(theta) * L))
        assert stable(cell(theta, face=face), p) is want, theta
    assert not planned(cell(15.0))


def test_the_incidence_alone_is_held_without_a_diffraction_monitor():
    """No Diffraction monitor: no order is judged, but the PlaneWave's own incidence is, on
    its column's two faces; it is refused from about 70 degrees here."""
    p = quiet(fd.plan, cell(monitors=False))
    seen = []
    for theta in (40.0, 60.0, 70.0, 80.0):
        want = planned(cell(theta, monitors=False))
        seen.append(want)
        got = stable(cell(theta, face=face_at(leaf(k_of(theta))), monitors=False), p)
        assert got is want, theta
    assert seen == [True, True, False, False]


def test_a_traced_axis1d_k_is_held():
    """y Axis1D (one cell, no orders of its own) with its ``k`` traced, without and with the
    Diffraction monitors. Here the x orders other than (0, 0) are evanescent (g = 10.5 against
    n k0 <= 9.9 in the glass) and (0, 0) leaves the glass at cos >= 0.74, so it is the
    incidence that is refused, from about 70 degrees, in both."""
    for monitors, thetas in ((False, (40.0, 60.0, 70.0, 80.0)), (True, (5.0, 40.0, 70.0, 80.0))):
        p = quiet(fd.plan, cell(axis1d=True, monitors=monitors))
        seen = set()
        for theta in thetas:
            k = k_of(theta, axis1d=True)
            want = planned(cell(theta, axis1d=True, monitors=monitors))
            seen.add(want)
            scene = cell(theta, face=face_at(leaf(k), True, abs(k)), axis1d=True,
                         monitors=monitors)
            assert stable(scene, p) is want, (monitors, theta)
        assert seen == {True, False}, (monitors, seen)


def test_a_traced_k_and_a_traced_substrate_are_held_together():
    """Both leaves traced: the orders are counted at the substrate's live index, which the host
    bands (``grazing.refused_bands``) cannot do without a concrete ``k``."""
    p = quiet(fd.plan, cell())
    seen = set()
    for theta in (SEED_THETA, 3.0):
        for eps in (1.3, 2.0, 2.25, 2.4, 2.55):
            want = planned(cell(theta, eps=eps))
            seen.add(want)
            scene = cell(theta, face=face_at(leaf(k_of(theta))), eps=leaf(eps))
            assert stable(scene, p) is want, (theta, eps)
    assert seen == {True, False}


def test_a_jitted_run_holds_a_traced_k():
    """JAX: ``k`` a jit argument, one program, the concrete plan's verdict on either side of
    the band."""
    from wavebit.fdtd._runtime import runtime
    runtime('jax')
    import jax
    import jax.numpy as jnp
    p = quiet(fd.plan, cell())

    def run(k):
        return quiet(fd.run, cell(face=fd.Bloch(k=k)), 'jax', plan=p).diagnostics['stable']
    compiled = jax.jit(run)
    for theta in (SEED_THETA, 10.0, 20.0, 24.0):
        assert bool(compiled(jnp.asarray(k_of(theta)))) is planned(cell(theta)), theta
    assert compiled._cache_size() == 1


def test_a_traced_k_beside_a_stabilized_face_is_refused_by_name():
    """The survey judges a stabilized face on its exact return, on the host only."""
    with pytest.raises(ValueError, match='traced Bloch k .* stabilized PML face'):
        quiet(fd.run, cell(face=face_at(leaf(k_of(SEED_THETA))), rho=.05), 'torch')
    with pytest.raises(ValueError, match='Stabilized face refused'):
        quiet(fd.plan, cell(rho=.05))          # the concrete scene: refused on its return
