"""Field reconstruction on the first-order (4ff) path.

``tests/test_formulation_dispatch.py`` covers which path a call takes; this file covers the
in-cell field the first-order path now returns, and — the harder half — what measures it.

**The z-decoupled half has an exact internal oracle.** Forcing ``formulation='4ff'`` on a
problem whose permittivity ``z`` row is empty makes the two paths solve the same physical
problem by different algebra, so their reconstructed fields must agree entry for entry. Their
efficiencies already do to 1e-14, patterned and unpatterned, which is what makes a *field*
comparison measure the reconstruction and nothing else. That is the workhorse below.

**The z-coupled half has no twin at all** — it is the geometry the path exists for, and the
one no other solver in the survey reaches. Three checks with exact answers stand in, chosen so
that between them nothing is left free:

* **Maxwell.** The reconstructed six components must satisfy ``curl E = i k0 H`` and
  ``curl H = -i k0 D`` for the tensor that was handed in, ``z`` row included. Checked by
  central differences down a layer, so the *rate* is an assertion too: the field is analytic
  inside a layer, so halving the step must quarter the residual, and a residual that is merely
  "small" without converging is a different thing from one that is discretisation error.
  This is the only check that reaches the two terms ``- ezx Ex - ezy Ey`` that the second-order
  path's ``E_z`` line does not have.
* **Continuity.** The four transverse components are continuous across a layer interface, and
  the sampling lands on it exactly (``linspace(0, sample_z, sample_z)`` is inclusive, so a layer's
  last slice *is* its bottom face and the next layer's first slice is the same plane). Exact,
  not asymptotic. ``E_z`` must jump there, which is asserted too — a reconstruction that
  quietly returned a continuous ``E_z`` would be returning something other than ``E_z``.
* **Absolute scale.** Maxwell and continuity are both homogeneous: multiply the whole cell by a
  constant and neither notices. So the cell is tied to the *reported efficiencies* by sitting
  the stack between two lossless vacuum spacers, where the ``z`` flux is constant and equals
  ``(1 - R)`` above and ``T`` below. Their ratio is the one number that catches a wrong
  normalisation of ``T1``, which is exactly the failure a plausible-looking field hides.

A note on signs. The vacuum slab in :func:`test_a_vacuum_slab_comes_back_as_the_incident_plane_wave`
below is what found the convention defect this path briefly inherited: for one day every
reconstruction returned ``E_z``, ``H_x`` and ``H_y`` in the frame whose ``+z`` points *out* of
the stack, which ``docs/conventions.md`` explicitly does not declare, and this path applied a
parity on output to match the other three rather than break a caller who flips
``formulation``. The parity now lives where it belongs — on the three second-order paths — and
this one needs none, because :mod:`wavebit._core.solve_2d_4ff` already derives in the declared
frame. So the checks below read the returned cell directly. The frame itself is pinned for
every path at once by ``tests/test_field_core.py``.
"""
import numpy as np
import pytest

import wavebit
# One residual, shared: it takes the full 3x3 tensor precisely so the z-coupled cases here and
# the isotropic frame check there measure the same thing the same way.
from test_field_core import maxwell_residual

WL = 620.0
PERIOD = (400.0, 380.0)

VAC = np.eye(3, dtype=complex)
# Voigt: magnetisation along x, so the gyrotropy fills the y-z block and the z row with it.
VOIGT = np.array([[4.0 + 0.05j, 0, 0],
                  [0, 4.0 + 0.05j, 0.9j],
                  [0, -0.9j, 4.0 + 0.05j]], dtype=complex)
# Tilted, so that all four z-row slots are populated at once and none can hide behind a zero.
TILTED = np.array([[5.2, 0.3j, -0.4j],
                   [-0.3j, 4.1, 0.7j],
                   [0.4j, -0.7j, 3.3]], dtype=complex)

COMPONENTS = ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')


def uniform_slots(eps_list):
    """The nine convolution stacks for a list of uniform tensors, at one retained order.

    Slot convention: every off-diagonal slot holds the **negative** of its tensor component,
    the z row included; ``epz_i`` is the inverse of the ``zz`` block. See
    ``docs/off-diagonal-slot-convention.md``.
    """
    I = np.eye(1, dtype=complex)
    e = [np.asarray(x, dtype=complex) for x in eps_list]
    return [np.stack([x[0, 0] * I for x in e]),
            np.stack([x[1, 1] * I for x in e]),
            np.stack([np.linalg.inv(x[2, 2] * I) for x in e]),
            np.stack([-x[0, 1] * I for x in e]),
            np.stack([-x[1, 0] * I for x in e]),
            np.stack([-x[0, 2] * I for x in e]),
            np.stack([-x[1, 2] * I for x in e]),
            np.stack([-x[2, 0] * I for x in e]),
            np.stack([-x[2, 1] * I for x in e])]


def z_solver(thickness, theta_deg=28.0, backend=0):
    """A solver for a stack of uniform tensors, at one retained order.

    The tensors themselves arrive through :func:`uniform_slots`, not through a ``structure``: a
    z-coupled permittivity is not something the modeler can paint.
    """
    return wavebit.rcwa(backend=backend, top=1.0, bot=2.3104, max_order=(0, 0), period=PERIOD,
                        wavelength=WL, theta=np.deg2rad(theta_deg), phi=np.deg2rad(37.0),
                        psi=0.6, thickness=thickness, answers='field')


def _to_backend(a, backend):
    """A convolution stack in the backend's own array type — ``thickness`` is already in it."""
    if backend == 0:
        return a
    if backend == 1:
        return pytest.importorskip('jax.numpy').asarray(a)
    return pytest.importorskip('torch').as_tensor(a)


def z_coupled_cell(eps_list, thickness, theta_deg=28.0, sample_z=64, backend=0):
    """``(result, cell, kx, ky)`` for a stack of uniform tensors.

    ``sample_x = sample_y = 1`` on purpose: with a single retained order the field is one plane wave,
    so the sample at ``(x, y) = (0, 0)`` **is** its Fourier coefficient and the checks below
    can be stated on the coefficients without inverting the transform by hand.
    """
    # Solver first, arrays second: importing the JAX backend is what lifts `jax_enable_x64`,
    # and an array built before that is silently complex64 (see test_formulation_dispatch).
    mee = z_solver(thickness, theta_deg, backend)
    res = mee.solve_conv(*[_to_backend(s, backend) for s in uniform_slots(eps_list)])
    assert mee.formulation_assigned == '4ff'
    cell = np.asarray(mee.reconstruct_field(sample_x=1, sample_y=1, sample_z=sample_z))
    kx, ky = mee.order_kxy()
    return (res, np.asarray(cell, dtype=complex)[0, :, 0, 0, :],
            complex(np.asarray(kx)[0]), complex(np.asarray(ky)[0]))


def z_flux(F):
    """Time-averaged Poynting ``z``, in the declared frame (``+z`` down, into the stack)."""
    return 0.5 * np.real(F[..., 0] * np.conj(F[..., 4]) - F[..., 1] * np.conj(F[..., 3]))


# ------------------------------------------------- the exact oracle: the same scene both ways

#: Permittivity rasters -- each entry is the square of the index the pre-flip file wrote, so
#: the twins compare the same two formulations on the same four structures as before.
TWINS = {
    'a uniform slab': (np.full((1, 1, 1), 5.0625), (0, 0), [180.0]),
    'two uniform slabs': (np.array([[[5.0625]], [[16.0]]]), (0, 0), [110.0, 90.0]),
    'a patterned layer': (np.random.default_rng(0).uniform(1, 4, (1, 8, 8)) ** 2, (1, 1),
                          [150.0]),
    'two patterned layers': (np.random.default_rng(1).uniform(1, 4, (2, 8, 8)) ** 2, (2, 1),
                             [80.0, 120.0]),
}


def twin_cell(formulation, name, theta_deg, side, backend=0):
    structure, max_order, thickness = TWINS[name]
    mee = wavebit.rcwa(backend=backend, top=1.0, bot=2.3104, max_order=max_order, period=PERIOD,
                       wavelength=WL, theta=np.deg2rad(theta_deg), phi=np.deg2rad(37.0),
                       psi=0.6, thickness=thickness, structure=structure, incidence_side=side,
                       answers='field')
    mee.formulation = formulation
    mee.solve()
    assert mee.formulation_assigned == formulation
    return np.asarray(mee.reconstruct_field(sample_x=7, sample_y=5, sample_z=6,
                                          incidences=('given', 's', 'p')))


@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('theta_deg', [0.0, 8.0, 35.0])
@pytest.mark.parametrize('name', list(TWINS))
def test_the_two_formulations_reconstruct_the_same_field(name, theta_deg, side):
    """The oracle: an empty z row makes the two paths the same problem, so the cells must agree.

    Every component is scaled by **its own** magnitude rather than by the cell's. ``E_z`` runs
    two orders below ``H_y`` at these incidences, so a wholly wrong ``E_z`` measured against
    the cell's largest entry would come back at 1e-2 and read like a tolerance question rather
    than a broken formula.

    All three incidence variants are compared at once (``incidences=('given', 's', 'p')``):
    they ride in separate columns of ``T1``, and a reconstruction that dropped the leading axis
    would still reproduce the first of them.
    """
    got = twin_cell('4ff', name, theta_deg, side)
    want = twin_cell('2ff', name, theta_deg, side)
    assert got.shape == want.shape, (got.shape, want.shape)

    cell_scale = np.max(np.abs(want))
    assert np.max(np.abs(want - got)) < 1e-11 * cell_scale
    for i, comp in enumerate(COMPONENTS):
        scale = np.max(np.abs(want[..., i]))
        if scale < 1e-6 * cell_scale:
            # identically zero in this geometry (E_z and H_z at normal incidence on a uniform
            # slab); the absolute comparison above is what covers it. That the loop does bite
            # somewhere is test_every_component_is_actually_present_somewhere.
            continue
        gap = np.max(np.abs(want[..., i] - got[..., i])) / scale
        assert gap < 1e-11, f'{comp}: {gap:.3e}'


def test_every_component_is_actually_present_somewhere():
    """The companion to the skip above: all six are non-negligible in the oblique cases.

    Without this the per-component loop could quietly degrade into the absolute check — which
    is the whole thing it was written to avoid, since ``E_z`` runs two orders below ``H_y``.
    """
    for name in TWINS:
        cell = twin_cell('2ff', name, 35.0, 'top')
        scale = np.max(np.abs(cell))
        smallest = min(np.max(np.abs(cell[..., i])) for i in range(6))
        assert smallest > 1e-4 * scale, f'{name}: {smallest / scale:.2e}'


def test_the_two_formulations_agree_on_the_variants_taken_one_at_a_time():
    """``incidences`` selects columns of ``T1``, and the selection must survive the swap.

    Asking for the variants singly and together must give the same cells; the leading axis is
    the only place a reconstruction can get the *incidence* right and the *bookkeeping* wrong.
    """
    structure, max_order, thickness = TWINS['a patterned layer']
    cells = {}
    for formulation in ('2ff', '4ff'):
        mee = wavebit.rcwa(backend=0, top=1.0, bot=2.3104, max_order=max_order, period=PERIOD,
                           wavelength=WL, theta=np.deg2rad(21.0), phi=np.deg2rad(37.0),
                           psi=0.6, thickness=thickness, structure=structure, answers='field')
        mee.formulation = formulation
        mee.solve()
        cells[formulation] = [
            np.asarray(mee.reconstruct_field(sample_x=4, sample_y=4, sample_z=5,
                                             incidences=which))
            for which in (('given',), ('s',), ('p',), ('given', 's', 'p'))]
    for k, (want, got) in enumerate(zip(cells['2ff'], cells['4ff'])):
        assert want.shape == got.shape
        assert np.max(np.abs(want - got)) / np.max(np.abs(want)) < 1e-11, k
    # and on the new path the three single calls are the three slices of the combined one
    single, combined = cells['4ff'][:3], cells['4ff'][3]
    for k in range(3):
        assert np.allclose(single[k][0], combined[k], rtol=0, atol=1e-12), k


# ------------------------------------------------------- the z-coupled half, which has no twin

@pytest.mark.parametrize('eps,tag', [(VOIGT, 'voigt'), (TILTED, 'tilted')])
def test_the_z_coupled_cell_satisfies_maxwell_and_converges_like_it_should(eps, tag):
    """Both curl equations hold inside a z-coupled layer, at second order in the step.

    The rate is the load-bearing half. The field is analytic inside a layer, so a correct
    reconstruction differenced centrally must lose a factor of four per doubling; a formula
    that is wrong by a *term* leaves a residual that stops converging and sits on a floor.
    Only the two ``x`` / ``y`` rows carry the ``z`` derivative — the two ``z`` rows are
    algebraic identities the reconstruction builds ``H_z`` and ``D_z`` from, and they come back
    at 1e-16, which is worth seeing but proves nothing on its own.
    """
    thickness = [90.0, 210.0, 90.0]
    worst = []
    for sample_z in (32, 64, 128):
        _, F, kx, ky = z_coupled_cell([VAC, eps, VAC], thickness, sample_z=sample_z)
        r = maxwell_residual(F, kx, ky, eps, thickness[1], sample_z, layer=1)
        assert np.max(r[[2, 5]]) < 1e-13, f'the algebraic rows should be exact: {r}'
        worst.append(np.max(r[[0, 1, 3, 4]]))

    assert worst[0] < 1e-2, worst
    assert worst[-1] < 1e-3, worst
    for coarse, fine in zip(worst, worst[1:]):
        assert 3.4 < coarse / fine < 4.6, f'{tag}: rate {coarse / fine:.2f}, residuals {worst}'


@pytest.mark.parametrize('eps,tag', [(VOIGT, 'voigt'), (TILTED, 'tilted')])
def test_the_transverse_field_is_continuous_across_a_z_coupled_interface(eps, tag):
    """Exact, not asymptotic: a layer's last slice and the next layer's first are one plane.

    ``E_z`` is asserted to *jump* there in the same breath. It is the component the z row
    changes, it is discontinuous wherever ``eps_zz`` is, and a reconstruction that returned a
    continuous one would be returning something that is not ``E_z`` — the shape of error a
    smoothness check alone rewards.
    """
    sample_z = 16
    _, F, _, _ = z_coupled_cell([VAC, eps, VAC], [90.0, 210.0, 90.0], sample_z=sample_z)
    scale = np.max(np.abs(F))
    for k in (1, 2):
        above, below = F[k * sample_z - 1], F[k * sample_z]
        transverse = np.max(np.abs(above[[0, 1, 3, 4]] - below[[0, 1, 3, 4]]))
        assert transverse / scale < 1e-13, f'{tag} interface {k}: {transverse:.3e}'
        assert abs(above[2] - below[2]) / scale > 1e-2, f'{tag} interface {k}: E_z did not jump'


@pytest.mark.parametrize('theta_deg', [0.0, 28.0, 55.0])
@pytest.mark.parametrize('eps,tag', [(VOIGT, 'voigt'), (TILTED, 'tilted'), (VAC, 'vacuum')])
def test_the_flux_the_cell_carries_is_the_efficiency_the_solve_reported(eps, tag, theta_deg):
    """The one check that fixes the absolute scale, and so the ``T1`` normalisation.

    Maxwell and continuity are homogeneous — scale the whole cell and neither moves — so on
    their own they would accept a reconstruction that is right in shape and wrong by a
    constant. That constant is exactly what a mis-normalised ``T1`` would be, and ``big_T``
    feeds the transmitted amplitude directly, so it is not a hypothetical.

    Sitting the stack between two **lossless** vacuum spacers is what makes the check exact
    rather than a limit: inside a lossless layer the ``z`` flux does not vary, so the spacers
    read ``(1 - R)`` above and ``T`` below at every slice, and their ratio is scale-free.
    The vacuum row is the control — nothing there but the spacers, so the ratio must be 1.
    """
    res, F, _, _ = z_coupled_cell([VAC, eps, VAC], [90.0, 210.0, 90.0],
                                  theta_deg=theta_deg, sample_z=16)
    S = z_flux(F)
    above, below = S[:16], S[32:]
    for name, band in (('above', above), ('below', below)):
        assert np.ptp(band) / abs(np.mean(band)) < 1e-12, f'{tag} {name} spacer is not flat'

    R = float(np.sum(np.asarray(res.given.R)))
    T = float(np.sum(np.asarray(res.given.T)))
    got = np.mean(below) / np.mean(above)
    assert abs(got - T / (1 - R)) < 1e-12, f'{tag}: {got} vs {T / (1 - R)}'


def test_a_z_coupled_cell_is_not_quietly_the_z_decoupled_one():
    """The z row has to *do* something to the field, or none of the above is measuring it.

    Dropping the four z-row slots is the failure mode the formulation guard exists for
    (``docs/anisotropic-materials.md`` §6); it conserves energy to machine precision, so this
    states the positive claim instead: the same stack with the z row zeroed reconstructs a
    visibly different cell.
    """
    stack = [VAC, TILTED, VAC]
    flat = [np.diag(np.diag(TILTED)) if e is TILTED else e for e in stack]
    _, coupled, _, _ = z_coupled_cell(stack, [90.0, 210.0, 90.0], sample_z=16)
    _, decoupled, _, _ = z_coupled_cell(flat, [90.0, 210.0, 90.0], sample_z=16)
    gap = np.max(np.abs(coupled - decoupled)) / np.max(np.abs(coupled))
    assert gap > 1e-2, gap


# ------------------------------------------------------------------------- the sign convention

@pytest.mark.parametrize('formulation', ['2ff', '4ff'])
@pytest.mark.parametrize('psi_v,tag', [(np.pi / 2, 's'), (0.0, 'p')])
def test_a_vacuum_slab_comes_back_as_the_incident_plane_wave(formulation, psi_v, tag):
    """A layer of vacuum in vacuum: the field inside is the incident wave, exactly.

    Every entry is known in closed form, which is what made this the check that found the
    convention defect — and now the one that keeps it fixed. Three facts, and they agree:

    * the phase **advances** with the cell's ``z`` index, so the wave travels that way
      (``e^{-iwt}``, ``exp(+i k0 kz z)``);
    * ``H = khat x E`` with ``khat`` along the index, so the pair is right-handed about it;
    * the Poynting vector points **along** the index, i.e. into the stack.

    Until 2026-08-14 the second fact failed and the third came back reversed, on every path.
    Both formulations are checked here so the claim is about the library rather than one of
    them; ``tests/test_field_core.py`` makes the same claim for the two 1D layouts and by
    Maxwell rather than by a closed form.
    """
    mee = wavebit.rcwa(backend=0, top=1.0, bot=1.0, max_order=(0, 0), period=PERIOD,
                       wavelength=WL, theta=0.0, phi=0.0, psi=psi_v,
                       thickness=[WL], structure=np.ones((1, 1, 1)), answers='field')
    mee.formulation = formulation
    mee.solve()
    cell = np.asarray(mee.reconstruct_field(sample_x=1, sample_y=1, sample_z=8))[0, :, 0, 0, :]

    # khat = +z (the index).  s: E = +y, so H = z x y = -x.  p: E = -x, so H = z x (-x) = -y.
    (e_idx, e_sign), (h_idx, h_sign) = (((1, 1.0), (3, -1.0)) if tag == 's'
                                        else ((0, -1.0), (4, -1.0)))
    assert abs(cell[0, e_idx] - e_sign) < 1e-12, cell[0]
    assert abs(cell[0, h_idx] - h_sign) < 1e-12, cell[0]
    quiet = [i for i in range(6) if i not in (e_idx, h_idx)]
    assert np.max(np.abs(cell[:, quiet])) < 1e-12, cell[0]

    # the phase advances with the index -- one full wavelength over a slab one wavelength thick,
    # sampled inclusively, so slice j sits at j/7 of a period
    for j in (1, 3, 5):
        want = e_sign * np.exp(2j * np.pi * j / 7)
        assert abs(cell[j, e_idx] - want) < 1e-12, (j, cell[j, e_idx], want)

    # and the energy flows the way the wave does
    assert z_flux(cell)[0] > 0


# --------------------------------------------------------------------------------- backends

@pytest.mark.parametrize('backend', [1, 2], ids=['jax', 'torch'])
def test_the_reconstruction_is_the_same_on_every_backend(backend):
    """One implementation, so jax and torch reconstruct the same cell as numpy.

    Both the twin route (a patterned layer forced onto ``'4ff'``) and the z-coupled route,
    because they reach the reconstruction through different halves of the solver.
    """
    pytest.importorskip('jax' if backend == 1 else 'torch')
    want = twin_cell('4ff', 'a patterned layer', 21.0, 'top')
    got = twin_cell('4ff', 'a patterned layer', 21.0, 'top', backend=backend)
    assert np.max(np.abs(want - got)) / np.max(np.abs(want)) < 1e-10

    _, want_z, _, _ = z_coupled_cell([VAC, TILTED, VAC], [90.0, 210.0, 90.0], sample_z=16)
    _, got_z, _, _ = z_coupled_cell([VAC, TILTED, VAC], [90.0, 210.0, 90.0], sample_z=16,
                                    backend=backend)
    assert np.max(np.abs(want_z - got_z)) / np.max(np.abs(want_z)) < 1e-10
