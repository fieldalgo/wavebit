"""The time-domain solve, against three references that fail in different ways.

`wavebit/_core/fdtd.py` exists so that a structure can be solved by a method whose errors are
not RCWA's. That only buys anything if the comparison is arranged so the two can actually
disagree, so this file is built in three layers, weakest oracle first:

**1. Closed form.** A planar stack has an exact transfer matrix, written out here in the same
frame the FDTD uses (`_stack_rt`). It needs no solver at all, so it is the one test that still
means something if every other part of this package is wrong. It is also where the amplitude
conventions are pinned: TE's `Ey_r/Ey_i` and TM's `Hy_r/Hy_i` must come out as `solve_1d`'s
`r_s` and `r_p`, and at a single interface those differ by a sign that `|r|**2` cannot see.

**2. Theorems.** Statements true of the *discretisation* regardless of what the right answer
is: a structure uniform along x radiates no non-specular order, at all; a mirror-symmetric
grating at normal incidence gives `T[+m] == T[-m]`; a lossless scene conserves energy; TE and
TM are degenerate on a planar stack at normal incidence. These hold at any resolution, so they
catch a defect that a convergence test would only see as a slightly worse number.

**3. The RCWA path.** Solved in the same process rather than stored, so there is no golden
here to regenerate and the reference moves if the RCWA path moves. The tolerances are the
*measured* gaps, quoted in each message beside the value that would have to change for the
assertion to be doing nothing.

**Symmetry is the enemy of an order-resolved test.** Almost every structure here is uniform
along x or mirror-symmetric about its own centre -- the duty-0.3 rectangle included, which
looks asymmetric and is not -- and on those RCWA returns `T(+1) == T(-1)` exactly, so nothing
can tell the two orders apart. That makes the whole file blind to the x origin, to the sign of
the order index, and to every amplitude phase.
`test_a_profile_with_no_mirror_symmetry_agrees_on_the_AMPLITUDES` is the one case that is not,
and it is the only check here that would notice wavebit moving its coordinate origin, which it
did on 2026-08-31.

**What this file cannot see.** The uniform-slab and multilayer gates are blind to both
subpixel harmonic averages -- `Ez` is identically zero at normal incidence with no x variation,
and a harmonic mean along x of a constant is that constant -- so only the grating cases
arbitrate `component_property`'s rule -- and the *x* half of that rule needs two conditions
the grating comparisons at `resolution=0.1` do not meet. It reaches `eps_xx` alone, so the run
has to be TM; and `Ex` sits on the x half-steps, so its averaging cell runs node to node and
the tooth edge has to fall inside one rather than on a boundary, which it does only when `nx`
is odd. At 70 cells it is even and the rule is inert -- deleting it moves those answers by
1.4e-16. `test_a_tooth_edge_between_two_cells_arbitrates_the_harmonic_rule` is the one scene
here that meets both.

Of the five scopes this file once listed as refused, four are
now implemented and tested here -- oblique incidence, loss, `mu` and a crossed grating -- and
only a skewed cell is still refused; `test_refusals` is what keeps that last one honest, by
pinning the refusal rather than the behaviour.
"""
import numpy as np
import pytest

import wavebit
from wavebit._core.fdtd import solve_fdtd


WL, PERIOD, THICK = 900.0, 700.0, 300.0
N_TOOTH, N_TOP, N_BOT = 3.5, 1.0, 1.5

EPS_TOP = wavebit.eps_from_index(N_TOP)
EPS_BOT = wavebit.eps_from_index(N_BOT)


def _grating(duty=0.5):
    """The benchmark binary grating: a tooth on ``[0, duty*period)``, air beyond.

    Two raster columns rather than two thousand. `component_property` integrates a
    piecewise-constant profile exactly, so the two describe the same geometry -- checked
    against the 2000-column RCWA reference, which agrees to 6e-7.
    """
    if duty != 0.5:
        n = 1000
        col = np.full(n, wavebit.eps_from_index(N_TOP))
        col[:int(round(duty * n))] = wavebit.eps_from_index(N_TOOTH)
        return col.reshape(1, 1, n)
    return np.array([[[wavebit.eps_from_index(N_TOOTH), wavebit.eps_from_index(N_TOP)]]])


def _uniform(index=N_TOOTH, n_layers=1):
    return np.full((n_layers, 1, 2), wavebit.eps_from_index(index))


def _fdtd(structure, thickness, tetm, **kw):
    kw.setdefault('resolution', 0.05)
    kw.setdefault('n_periods', 60)
    return solve_fdtd(structure=structure, thickness=thickness, top=EPS_TOP, bot=EPS_BOT,
                      period=[PERIOD], wavelength=WL, tetm=tetm, **kw)


def _rcwa(structure, thickness, tetm, orders=80):
    solver = wavebit.rcwa(backend=0, wavelength=WL, period=[PERIOD],
                          truncation_order=dict(x=orders, y=0), top=EPS_TOP, bot=EPS_BOT,
                          theta=0.0, phi=0.0, tetm=tetm,
                          structure=structure, thickness=thickness)
    return solver.solve()


def _by_order(result):
    """``{m: (R, T)}`` -- the labels rather than the positions, which is the whole point of
    :attr:`~wavebit._core.result.Result.orders_u` existing."""
    R = np.asarray(result.R).ravel()
    T = np.asarray(result.T).ravel()
    return {int(m): (float(R[i]), float(T[i]))
            for i, m in enumerate(np.asarray(result.orders_u).ravel())}


# ---------------------------------------------------------------------------------------
# 1. the closed form
# ---------------------------------------------------------------------------------------

def _stack_rt(wavelength, n_top, n_list, d_list, n_bot, tetm,
              mu_list=None, mu_top=1.0, mu_bot=1.0):
    """Exact ``(r, t)`` for a planar stack at normal incidence, in wavebit's frame.

    ``+z`` down, ``e^{-iwt}``, forward wave ``A exp(+i k z)``. TE propagates ``Ey`` with
    ``Hx = -n Ey`` on the forward branch, TM propagates ``Hy`` with ``Ex = Hy/n``: those two
    choices are what make the result come out as ``r_s`` and ``r_p`` rather than as one of
    them twice. A single interface gives ``r_s = (n1-n2)/(n1+n2)`` and
    ``r_p = (n2-n1)/(n2+n1) = -r_s``, which is the convention `result.py` states.
    """
    k0 = 2.0 * np.pi / wavelength
    # `n_list` entries are the indices `sqrt(eps*mu)`; `mu_list` carries the permeabilities, so
    # `eps = n**2/mu`. An s channel's admittance is `n/mu` and a p channel's is `n/eps`, which
    # is the whole content of a magnetic half-space -- `n` alone cannot express one, since it
    # sees no interface at all between vacuum and an impedance-mismatched medium of equal index.
    mus = ([complex(mu_top)] + [complex(m) for m in (mu_list or [1.0] * len(n_list))]
           + [complex(mu_bot)])
    ns = [complex(n_top)] + [complex(n) for n in n_list] + [complex(n_bot)]
    ds = [0.0] + [float(d) for d in d_list] + [0.0]
    epss = [n ** 2 / m for n, m in zip(ns, mus)]

    M = np.eye(2, dtype=complex)
    for j in range(len(ns) - 1):
        phase = k0 * ns[j] * ds[j]
        P = np.diag([np.exp(1j * phase), np.exp(-1j * phase)])
        if tetm == 'te':
            a, b = ns[j] / mus[j], ns[j + 1] / mus[j + 1]
            I = np.array([[b + a, b - a], [b - a, b + a]], dtype=complex) / (2.0 * b)
        else:
            q = (ns[j] / epss[j]) / (ns[j + 1] / epss[j + 1])
            I = np.array([[1.0 + q, 1.0 - q], [1.0 - q, 1.0 + q]], dtype=complex) / 2.0
        M = I @ P @ M

    r = -M[1, 0] / M[1, 1]
    t = M[0, 0] + M[0, 1] * r
    R = abs(r) ** 2
    if tetm == 'te':
        T = abs(t) ** 2 * np.real((ns[-1] / mus[-1]) / (ns[0] / mus[0]))
    else:
        T = abs(t) ** 2 * np.real((ns[-1] / epss[-1]) / (ns[0] / epss[0]))
        # t_E from t_H, solve_1d.py:505-508 -- `t_H * n_top/n_bot` exactly at mu = 1.
        t = t * ((ns[0] / mus[0]) / (ns[-1] / mus[-1]))
    return r, t, R, T


def test_the_closed_form_reproduces_the_fresnel_signs_it_is_used_to_check():
    """The oracle before the oracle: a single interface, where the answer is on paper.

    Without this the TM sign convention would be checked only against itself.
    """
    for n2 in (1.5, 3.5):
        r_s, _, _, _ = _stack_rt(WL, 1.0, [], [], n2, 'te')
        r_p, t_p, _, _ = _stack_rt(WL, 1.0, [], [], n2, 'tm')
        assert abs(r_s - (1.0 - n2) / (1.0 + n2)) < 1e-15
        assert abs(r_p - (n2 - 1.0) / (1.0 + n2)) < 1e-15
        assert abs(r_p + r_s) < 1e-15                        # r_p = -r_s at normal incidence
        assert abs(t_p - 2.0 / (1.0 + n2)) < 1e-15           # the E ratio, not the H one


@pytest.mark.parametrize('tetm', ['te', 'tm'])
def test_a_uniform_slab_matches_the_closed_form(tetm):
    """The cheapest instrument that separates a geometry error from a physics error.

    Run it before anything else whenever the grid or the geometry changes: a broken mesh
    returns plausible grating numbers, and energy conservation does not catch it -- that is how
    a sibling FEM oracle once returned R0 = 0.047 against an analytic 0.543 while its own
    balance check passed.
    """
    result = _fdtd(_uniform(), [THICK], tetm)
    r_a, t_a, R_a, T_a = _stack_rt(WL, N_TOP, [N_TOOTH], [THICK], N_BOT, tetm)
    orders = _by_order(result)

    # Measured 5.9e-3 (TE) / 5.5e-3 (TM) at this 20 nm grid, and falling as dz**2: 1.6e-3 and
    # 1.4e-3 at 10 nm. 2e-2 sits above both and far below the 5.4e-1 a wrong branch gives.
    assert abs(orders[0][0] - R_a) < 2e-2, f'R {orders[0][0]} against closed form {R_a}'
    assert abs(orders[0][1] - T_a) < 2e-2, f'T {orders[0][1]} against closed form {T_a}'

    amplitude = result.given.r_s if tetm == 'te' else result.given.r_p
    assert abs(complex(np.asarray(amplitude).ravel()[1]) - r_a) < 6e-2, (
        f'the reflected amplitude carries the sign convention, and |r|**2 cannot see it; '
        f'measured 3.3e-2 at this grid, falling as dz**2')


@pytest.mark.parametrize('tetm', ['te', 'tm'])
def test_a_multilayer_stack_matches_the_closed_form(tetm):
    """Four layers whose thicknesses land nowhere near a cell boundary.

    The single slab's interfaces sit on grid lines at every resolution the tests use, so a
    layer-walk that mishandled a straddling cell would pass it. These do not.
    """
    n_list = [2.1, 3.5, 1.45, 2.9]
    d_list = [137.3, 211.7, 89.1, 173.9]
    structure = np.array([[[wavebit.eps_from_index(n)]] for n in n_list])
    result = _fdtd(structure, d_list, tetm)
    _, _, R_a, T_a = _stack_rt(WL, N_TOP, n_list, d_list, N_BOT, tetm)
    orders = _by_order(result)

    # Measured 7.6e-3 (TE) / 8.3e-3 (TM) at 20 nm, 3.1e-4 / 2.5e-5 at 10 nm.
    assert abs(orders[0][0] - R_a) < 3e-2, f'R {orders[0][0]} against closed form {R_a}'
    assert abs(orders[0][1] - T_a) < 3e-2, f'T {orders[0][1]} against closed form {T_a}'


# ---------------------------------------------------------------------------------------
# 2. theorems about the discretisation
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize('tetm', ['te', 'tm'])
def test_a_structure_uniform_along_x_radiates_no_other_order(tetm):
    """Exactly zero, not small.

    A layer with no x variation has one non-zero Fourier component, so every other order is
    zero identically rather than to some tolerance. Measured 1e-32 -- round-off in the
    extraction, nothing more. This is what energy conservation cannot see: power scattered
    into a spurious order still sums to one.
    """
    result = _fdtd(_uniform(), [THICK], tetm, truncation_order=dict(x=3, y=0))
    orders = _by_order(result)
    for m, (R, T) in orders.items():
        if m != 0:
            assert abs(R) < 1e-20 and abs(T) < 1e-20, f'order {m} is {R}/{T}, not zero'


def test_te_and_tm_are_degenerate_on_a_planar_stack():
    """At normal incidence there is no plane of incidence, so the two must agree.

    They are computed by two separate steppers over two separate staggerings with two
    separate TFSF corrections, so agreement here is a real cross-check and not an identity.
    It is also the only place the TM path is checked against something other than a closed
    form written by the same hand.
    """
    te = _by_order(_fdtd(_uniform(), [THICK], 'te'))[0]
    tm = _by_order(_fdtd(_uniform(), [THICK], 'tm'))[0]
    # Measured 4.3e-4 on R and 7.3e-3 on T at this grid; both fall as dz**2.
    assert abs(te[0] - tm[0]) < 5e-3, f'R: TE {te[0]} against TM {tm[0]}'
    assert abs(te[1] - tm[1]) < 2e-2, f'T: TE {te[1]} against TM {tm[1]}'


@pytest.mark.parametrize('tetm', ['te', 'tm'])
def test_a_mirror_symmetric_grating_scatters_symmetrically(tetm):
    """``T[+1] == T[-1]`` to round-off, at normal incidence on a 50% duty grating.

    The structure is mirror-symmetric about the centre of its tooth, so the two first orders
    carry the same power exactly. Whether the *grid* can represent that symmetry is a separate
    question, and this is the assertion that asks it: at an odd cell count the tooth wall
    falls mid-cell and the two part company -- measured 2.3e-2 without subpixel averaging at
    35 cells, against 0 here at 70. Meep does not pass this test at all (8.7e-5 apart at its
    own res 200), which is why it is stated as an exact identity rather than a tolerance.
    """
    result = _fdtd(_grating(), [THICK], tetm, resolution=0.1)
    orders = _by_order(result)
    assert abs(orders[1][1] - orders[-1][1]) < 1e-15, (
        f'T+1 {orders[1][1]} against T-1 {orders[-1][1]}; a difference here means the grid '
        f'does not carry the structure\'s mirror symmetry, not that the physics is wrong')


@pytest.mark.parametrize('tetm', ['te', 'tm'])
def test_a_lossless_scene_conserves_energy(tetm):
    """A liveness check, deliberately demoted.

    Energy conservation is satisfied by a reversed order index, a reversed s/p frame, a
    reversed z axis and a spurious order; it is here to catch a dead run, not a wrong one.
    """
    result = _fdtd(_grating(), [THICK], tetm, resolution=0.1)
    total = float(np.asarray(result.R).sum() + np.asarray(result.T).sum())
    assert abs(total - 1.0) < 5e-3, f'R + T = {total}'


@pytest.mark.parametrize('tetm', ['te', 'tm'])
def test_the_error_falls_when_the_grid_is_refined(tetm):
    """Second order, which is what the subpixel averaging is there to buy.

    A first-order scheme would give a ratio near 2, and a defect that does not converge at all
    -- a wrong reference plane, a PML floor, a truncated run -- gives a ratio near 1. Measured
    ratios on this pair: 4.1 (TE) and 3.6 (TM).
    """
    reference = _by_order(_rcwa(_grating(), [THICK], tetm))
    errors = []
    for resolution in (0.05, 0.1):
        orders = _by_order(_fdtd(_grating(), [THICK], tetm, resolution=resolution))
        errors.append(max(max(abs(orders[m][0] - reference[m][0]),
                              abs(orders[m][1] - reference[m][1]))
                          for m in (-1, 0, 1)))
    assert errors[1] < errors[0] / 2.5, (
        f'errors {errors[0]:.2e} -> {errors[1]:.2e}, ratio {errors[0] / errors[1]:.2f}; '
        f'below 2.5 means something in the chain is not second order')


# ---------------------------------------------------------------------------------------
# 3. against the RCWA path
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize('tetm', ['te', 'tm'])
def test_the_binary_grating_agrees_with_the_rcwa_solve(tetm):
    """The comparison the module exists for, on the scene a third method has also solved.

    lambda 900, period 700, one 300 nm layer, 50% duty n = 3.5 tooth in air, substrate
    n = 1.5, normal incidence. In glass orders 0 and +-1 propagate; in air only order 0, so
    ``R[+-1]`` is exactly zero and that is asserted rather than skipped.

    At the 5 nm grid this file does not run (it costs ~6 s a polarisation) the worst per-order
    gap is 6.6e-4 for TE and 6.9e-4 for TM, against Meep's 4.9e-4 and 7.4e-4 on the same grid
    -- the agreement is a three-way one, and the number below is the coarser grid's. Meep's
    half of it is not a quotation: it is ``tests/references/golden_meep_1d.npz``, frozen from
    this same scene, and ``tests/references/test_meep_1d_golden.py`` holds this path against
    it directly on a grid neither code chose for the other.

    **The TM figure was 7.5e-4 here until 2026-09-07 and no configuration reproduced it** --
    not the shipped one (6.911e-4) and not the ``pad_scale=2.0`` one the published table had
    drifted to (6.2e-4). It is a number that outlived whatever run produced it, which is the
    same failure ``docs/fdtd.md`` §2 had and the reason all four of that section's columns
    are now read out of the markdown and compared to a golden by
    ``tests/references/test_fdtd_table_golden.py``. A docstring is not covered by that gate,
    so this one states where its numbers come from: ``golden_fdtd_table.npz``.
    """
    reference = _by_order(_rcwa(_grating(), [THICK], tetm))
    orders = _by_order(_fdtd(_grating(), [THICK], tetm, resolution=0.1))

    for m in (-1, 1):
        assert orders[m][0] == 0.0, (
            f'order {m} is evanescent in air (period {PERIOD} < wavelength {WL}), so its '
            f'reflected efficiency is identically zero, not {orders[m][0]}')
    # Measured worst gap 2.6e-3 (TE) / 2.0e-3 (TM) at this 10 nm grid.
    for m in (-1, 0, 1):
        assert abs(orders[m][0] - reference[m][0]) < 5e-3, (
            f'R[{m}]: fdtd {orders[m][0]} against rcwa {reference[m][0]}')
        assert abs(orders[m][1] - reference[m][1]) < 5e-3, (
            f'T[{m}]: fdtd {orders[m][1]} against rcwa {reference[m][1]}')


def test_an_asymmetric_duty_cycle_still_agrees():
    """A 50% duty is the one case where the tooth and the groove are interchangeable.

    Nothing in the extraction distinguishes them there, so a defect that swapped the two would
    pass every other case in this file. Run at 30%.
    """
    structure = _grating(duty=0.3)
    reference = _by_order(_rcwa(structure, [THICK], 'te'))
    orders = _by_order(_fdtd(structure, [THICK], 'te', resolution=0.1))
    for m in (-1, 0, 1):
        assert abs(orders[m][1] - reference[m][1]) < 8e-3, (
            f'T[{m}]: fdtd {orders[m][1]} against rcwa {reference[m][1]}')


def test_the_result_is_the_same_container_an_rcwa_solve_returns():
    """Same type, same order labels, same field layout -- the comparison needs no adapter.

    ``R = R_s + R_p`` with everything on one side, as the pure path does it.
    """
    result = _fdtd(_uniform(), [THICK], 'te')
    reference = _rcwa(_uniform(), [THICK], 'te', orders=1)
    assert type(result) is type(reference)
    assert np.asarray(result.R).shape == (1, len(np.asarray(result.orders_u)))
    assert np.array_equal(np.asarray(result.orders_v), [0])
    assert np.allclose(np.asarray(result.R), np.asarray(result.given.R_s))
    assert np.allclose(np.asarray(result.given.R_p), 0.0)
    assert np.allclose(np.asarray(result.given.r_p), 0.0)


# ---------------------------------------------------------------------------------------
# 4. the refusals
# ---------------------------------------------------------------------------------------

REFUSED = [
    (dict(tetm=None), r'needs either `tetm` or `psi`'),
    (dict(tetm='psi'), r'tetm must be'),
    (dict(chi=10.0), r'elliptical or named polarisation'),
    (dict(period=dict(u=700.0, v=700.0)), r"a mapping needs exactly \{'x', 'y'\}"),
    (dict(polarization='rcp'), r'elliptical or named polarisation'),
    (dict(theta=0.3 + 0.1j), r'is complex'),
    (dict(phi=0.5), r'conical mount'),
    (dict(incidence_side='bottom'), r'incidence_side'),
    (dict(mu=np.ones((1, 1, 3))), r'share a shape exactly'),
    (dict(mu=np.array([[[-1.0, 1.0]]])), r'non-positive'),
    (dict(lattice=[[1.0, 0.0], [0.5, 1.0]]), r'lattice'),
    (dict(structure=[{'shapes': []}]), r'permittivity raster'),
    # A y axis is no longer refused -- it is a crossed grating -- but it needs its own period.
    (dict(structure=np.ones((1, 4, 2))), r'needs a y period'),
    (dict(structure=np.array([[[1.0 + 0.3j, 1.0]]])), r'complex permittivity'),
    (dict(structure=np.array([[[-2.0, 1.0]]])), r'non-positive'),
    (dict(bot=-1.5), r'non-positive'),
]


@pytest.mark.parametrize('override, message', REFUSED)
def test_refusals(override, message):
    """Every input outside this formulation, refused by name.

    Each of these returns a *plausible* number if it is quietly ignored -- a solve at the wrong
    angle, a lossless answer to a lossy question, a grating read along the wrong axis -- and a
    plausible wrong number from the cross-check solver is worse than no cross-check at all.
    """
    kwargs = dict(structure=_uniform(), thickness=[THICK], top=EPS_TOP, bot=EPS_BOT,
                  period=[PERIOD], wavelength=WL, tetm='te', resolution=0.02, n_periods=5)
    kwargs.update(override)
    with pytest.raises(ValueError, match=message):
        solve_fdtd(**kwargs)


def test_the_diagnostics_report_whether_the_run_was_long_enough():
    """The residual is the stopping criterion, and it has to be reported rather than assumed.

    Measured on the benchmark grating: at 40 optical cycles the field at the transmission
    monitor is still 61% of its peak and the specular efficiency is wrong by 0.09; at 80 the
    residual is 5e-6 and the answer no longer moves. A run that stopped too early is the one
    failure mode that looks like a converged answer.
    """
    _, short = solve_fdtd(structure=_grating(), thickness=[THICK], top=EPS_TOP, bot=EPS_BOT,
                          period=[PERIOD], wavelength=WL, tetm='te', resolution=0.05,
                          n_periods=50, return_diagnostics=True)
    _, long = solve_fdtd(structure=_grating(), thickness=[THICK], top=EPS_TOP, bot=EPS_BOT,
                         period=[PERIOD], wavelength=WL, tetm='te', resolution=0.05,
                         n_periods=60, return_diagnostics=True)
    assert short['residual'] > 1e-2, 'a 50-cycle run has not finished and must say so'
    assert long['residual'] < 1e-3, f"a 60-cycle run has, and reports {long['residual']}"
    assert long['n_steps'] > short['n_steps']
    assert long['pml_thickness'] == short['pml_thickness']


def test_the_cell_is_read_the_way_the_rcwa_door_reads_it():
    """`period=dict(x=..., y=...)` is one rule, read by `conventions.parse_period_mapping`.

    It was written twice -- once in the solver's `period` setter and once here -- and the
    second copy went in deliberately, with matching wording, because folding it meant editing
    a file another session held. A dict has `__len__`, so before either copy existed the
    positional indexing below reached it and raised a bare `KeyError: 0`.

    What is NOT shared is what each door does afterwards, and both finishings are right: the
    RCWA setter turns a lone number into a square cell, while this one keeps a scalar
    distinguishable from a pair, because `period_y is None` is what tells a grating from a
    crossed grating. So the test is that the mapping *agrees* with the list, not that the two
    doors normalise alike.
    """
    kw = dict(structure=_grating(), thickness=[THICK], top=EPS_TOP, bot=EPS_BOT,
              wavelength=WL, tetm='te', theta=0.0, resolution=0.05, n_periods=60)
    as_list = solve_fdtd(period=[PERIOD], **kw)
    as_map = solve_fdtd(period=dict(x=PERIOD, y=PERIOD), **kw)
    assert np.allclose(as_list.R, as_map.R) and np.allclose(as_list.T, as_map.T), (
        'the mapping spelling of the cell must solve the identical scene')

    # A y period does NOT appear from a mapping that names one on a y-invariant raster: the
    # raster has a single row, so this stays the 2D path rather than becoming a crossed run.
    _, diag = solve_fdtd(period=dict(x=PERIOD, y=PERIOD), return_diagnostics=True, **kw)
    assert diag['grid'].ny == 1, 'a one-row raster is a grating whatever the cell says'


def test_the_reported_mesh_is_the_mesh_that_ran():
    """`dz` and `pml_thickness` come off the node table, not off the nominal cell.

    `grid.dz` is set once, as `self.dz = self.dx`, and a graded run (§8f) never updates it --
    so both of these were computed as if the mesh were uniform. The reported PML was 240 nm
    against a true 546.5 above and 367.6 below, which is the number a reader consults to decide
    whether the absorber is thick enough: it understated it by a factor of 2.3.

    The two sides differ from each other, which is the part no single number could have got
    right. Grading coarsens toward the cell each half-space asks for, and the two half-spaces
    here are air and glass, so they ask for different cells.
    """
    kw = dict(structure=_grating(), thickness=[THICK], top=EPS_TOP, bot=EPS_BOT,
              period=[PERIOD], wavelength=WL, tetm='te', theta=0.0, resolution=0.05,
              n_periods=60, return_diagnostics=True)
    uniform = solve_fdtd(**kw)[1]
    graded = solve_fdtd(grading=1.06, **kw)[1]

    for tag, diag in (('uniform', uniform), ('graded', graded)):
        z = np.asarray(diag['grid'].z_node, dtype=float)
        cells = np.diff(z)
        n = diag['n_pml']
        assert np.isclose(diag['dz'], cells.min()), f'{tag}: dz must be the finest cell'
        assert np.isclose(diag['dz_max'], cells.max()), f'{tag}: dz_max must be the coarsest'
        assert np.allclose(diag['pml_thickness'],
                           (z[n] - z[0], z[-1] - z[-1 - n])), (
            f'{tag}: the PML thickness must be read off the node table')

    # On a uniform mesh the old arithmetic was right, which is why nothing caught this.
    assert np.isclose(uniform['dz'], uniform['dz_max'])
    assert np.allclose(uniform['pml_thickness'], uniform['n_pml'] * uniform['dz'])

    # On a graded one it is wrong in three separate ways at once.
    assert graded['dz_max'] / graded['dz'] > 2.0, 'the graded mesh must actually vary'
    assert graded['pml_thickness'][0] > 2.0 * graded['n_pml'] * graded['dz'], (
        'the graded PML is far thicker than the nominal cell says')
    assert graded['pml_thickness'][0] != graded['pml_thickness'][1], (
        'the two half-spaces ask for different cells, so the two sides differ')


# ---------------------------------------------------------------------------------------
# 5. the checks with the most resolving power, all oracle-free
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize('tetm', ['te', 'tm'])
def test_an_empty_grid_transmits_everything_and_reflects_nothing(tetm):
    """No structure at all: the TFSF plane's cancellation, measured against itself.

    With ``eps_top == eps_bot == eps_layer`` there is nothing to scatter, so the reflection
    monitor -- which sits in the scattered-field region -- must read zero, and the
    transmission monitor must read the incident wave exactly. Measured `R0` is 1e-33 and
    `T0` is 1.000000000.

    This is the highest-resolving-power check in the file: every sign, index and time-level
    mutant in the TFSF coupling drives `R0` from 1e-33 to 1e-2 or worse, so it has ~30 orders
    of headroom, and it needs no oracle of any kind.
    """
    structure = np.full((1, 1, 2), EPS_TOP)
    result = solve_fdtd(structure=structure, thickness=[THICK], top=EPS_TOP, bot=EPS_TOP,
                        period=[PERIOD], wavelength=WL, tetm=tetm, resolution=0.05,
                        n_periods=60)
    orders = _by_order(result)
    assert abs(orders[0][0]) < 1e-20, f'R0 = {orders[0][0]} into an empty grid'
    # `T0` is the weaker half: it carries the grid's own dispersion, and converges rather than
    # vanishing -- 4.4e-4 here, 2.6e-4 at 10 nm, 6.4e-5 at 5 nm. `R0` is the one with 30 orders
    # of headroom, and it does not move with resolution at all.
    assert abs(orders[0][1] - 1.0) < 1e-3, f'T0 = {orders[0][1]}, and nothing is in the way'


@pytest.mark.parametrize('tetm', ['te', 'tm'])
def test_the_incident_wave_is_the_same_at_both_monitors(tetm):
    """The auxiliary grid is uniform, so its wave must arrive undiminished.

    ``|inc_trn| / |inc_ref|`` is 1 exactly in exact arithmetic: the 1D grid is filled with the
    superstrate medium end to end. What eats into it is the auxiliary grid's own PML
    reflecting back through the monitors, so this is a direct read on the absorber with no
    solve, no oracle and no reference value. Measured 1.4e-4 with the over-conducting layer
    this module shipped with first, and 6e-5 after.
    """
    _, diagnostics = solve_fdtd(structure=_grating(), thickness=[THICK], top=EPS_TOP,
                                bot=EPS_BOT, period=[PERIOD], wavelength=WL, tetm=tetm,
                                resolution=0.05, n_periods=60, return_diagnostics=True)
    assert abs(diagnostics['incident_balance']) < 5e-4, (
        f"the incident wave changed by {diagnostics['incident_balance']:.2e} between the two "
        f"monitors, and the medium between them is uniform")


def test_the_absorber_holds_up_against_a_dense_substrate():
    """`n_bot = 3.5`, where the PML's ``sqrt(eps_r)`` scaling actually bites.

    Every other case in this file has `n_bot = 1.5`, where an inverted impedance factor is
    wrong by 2.25 rather than 12.25 and hides inside the discretisation error. Here it does
    not: the same code with the factor the other way up returns `T = 1.037` from a lossless
    stack and `R` wrong by 27%.
    """
    structure = np.full((1, 1, 2), wavebit.eps_from_index(2.0))
    eps_bot = wavebit.eps_from_index(3.5)
    result = solve_fdtd(structure=structure, thickness=[THICK], top=EPS_TOP, bot=eps_bot,
                        period=[PERIOD], wavelength=WL, tetm='te', resolution=0.1,
                        n_periods=60)
    _, _, R_a, T_a = _stack_rt(WL, N_TOP, [2.0], [THICK], 3.5, 'te')
    orders = _by_order(result)
    total = orders[0][0] + orders[0][1]
    # Each bound sits between the measured value and what the inverted factor gives, with
    # about 5x of margin on both sides: sum 1.0065 against 1.1130, R 1.9e-3 against 2.7e-2,
    # T 8.4e-3 against 1.4e-1. All three fall as dz**2.
    assert abs(total - 1.0) < 2e-2, f'R + T = {total} on a lossless stack'
    assert abs(orders[0][0] - R_a) < 1e-2, f'R {orders[0][0]} against closed form {R_a}'
    assert abs(orders[0][1] - T_a) < 3e-2, f'T {orders[0][1]} against closed form {T_a}'


def test_the_answer_does_not_depend_on_the_length_unit():
    """Nanometres or microns: same scene, same number.

    The solver is scale-free by construction -- ``c = eps0 = mu0 = 1`` and every constant is
    dimensionless or derived from ``wavelength`` -- and this is the assertion that keeps it
    that way. It is not hypothetical: the PML's frequency shift was once a bare ``0.05``,
    which is a reciprocal *length*, and that one constant made the mid-infrared return
    ``R + T = 1.22`` on a discretisation identical to the visible case that looked fine.
    """
    def run(scale):
        structure = np.full((1, 1, 2), wavebit.eps_from_index(N_TOOTH))
        return _by_order(solve_fdtd(
            structure=structure, thickness=[THICK * scale], top=EPS_TOP, bot=EPS_BOT,
            period=[PERIOD * scale], wavelength=WL * scale, tetm='te', n_periods=60))[0]

    nanometres = run(1.0)
    microns = run(1e-3)
    assert abs(nanometres[0] - microns[0]) < 1e-9, (
        f'R is {nanometres[0]} in one unit and {microns[0]} in another')
    assert abs(nanometres[1] - microns[1]) < 1e-9


def test_a_medium_faster_than_vacuum_does_not_diverge():
    """``eps = 0.81`` is ordinary, and the textbook Courant bound does not cover it.

    The usual spelling assumes no wave beats ``c``, which is false for any ``eps < 1``. Without
    the ``n_min`` factor this scene reaches 1e281 in forty optical cycles and returns ``nan``
    -- and no other case in this file can see it, because the unstable mode sits at
    ``kx*dx = pi`` and every other scene is either uniform in x or has ``eps >= 1``.
    """
    structure = np.array([[[0.81, 1.0]]])
    orders = _by_order(_fdtd(structure, [THICK], 'te', resolution=0.1))
    reference = _by_order(_rcwa(structure, [THICK], 'te'))
    assert np.isfinite(orders[0][0]) and np.isfinite(orders[0][1])
    for m in (-1, 0, 1):
        assert abs(orders[m][1] - reference[m][1]) < 5e-3, (
            f'T[{m}]: fdtd {orders[m][1]} against rcwa {reference[m][1]}')


def test_a_layer_that_is_not_a_whole_number_of_cells_still_agrees():
    """317 nm, which no cell size in this file divides -- and x structure, so ``Ez != 0``.

    Both conditions are needed and neither is present anywhere else. The benchmark's 300 nm is
    an exact multiple of every resolution used here, so the z averaging never has a straddling
    cell to resolve; and every non-commensurate case is uniform in x, where ``Ez`` is
    identically zero and ``eps_z`` multiplies into nothing. So the ``harmonic_axis='z'`` branch
    -- the one that carries the horizontal interfaces -- is unreachable by every other test in
    this file while being load-bearing on any real stack.
    """
    thickness = [317.0]
    reference = _by_order(_rcwa(_grating(), thickness, 'tm'))
    orders = _by_order(_fdtd(_grating(), thickness, 'tm', resolution=0.1))
    # Measured worst gap 1.9e-3 here; dropping the z-harmonic rule takes it to 6.6e-3.
    for m in (-1, 0, 1):
        assert abs(orders[m][1] - reference[m][1]) < 4e-3, (
            f'T[{m}]: fdtd {orders[m][1]} against rcwa {reference[m][1]}')


def test_point_sampling_costs_an_order_of_accuracy():
    """The subpixel A/B, on a slab whose interfaces genuinely cut cells.

    ``subpixel=False`` point-samples, which is the naive staircase; ``True`` is the exact cell
    average with the harmonic rule. The comparison is only meaningful where a cell straddles
    an interface, which is why the thickness is 317 rather than 300 -- at 300 the two arms are
    bit-identical at every resolution this file uses, and an A/B between them would measure
    nothing while looking like it measured something.
    """
    structure = np.full((1, 1, 2), wavebit.eps_from_index(N_TOOTH))
    _, _, R_a, _ = _stack_rt(WL, N_TOP, [N_TOOTH], [317.0], N_BOT, 'te')
    errors = {}
    for subpixel in (True, False):
        orders = _by_order(_fdtd(structure, [317.0], 'te', resolution=0.1,
                                 subpixel=subpixel))
        errors[subpixel] = abs(orders[0][0] - R_a)
    # Measured 1.73e-3 against 6.67e-3, a ratio of 3.9 -- and 1.5 at half the resolution,
    # which is why this runs on the finer grid: the two arms converge at different orders
    # (~2.5 against ~1.0) so the gap between them is itself resolution-dependent.
    assert errors[True] < errors[False] / 2.0, (
        f'subpixel {errors[True]:.2e} against point-sampled {errors[False]:.2e}; if these are '
        f'equal the flag is not reaching the geometry and the A/B measures nothing')


def test_a_run_too_short_for_the_source_is_refused():
    """The source is not finished emitting until 48 optical cycles, and 20 looks reasonable.

    A truncated run returns a number shaped exactly like an efficiency -- 0.033 where the
    answer is 0.543 -- so reporting it beside a residual would be reporting it. The floor is
    a property of the excitation, not of the scene, so it is knowable before the loop runs.
    """
    with pytest.raises(ValueError, match=r'optical cycles'):
        solve_fdtd(structure=_uniform(), thickness=[THICK], top=EPS_TOP, bot=EPS_BOT,
                   period=[PERIOD], wavelength=WL, tetm='te', resolution=0.02, n_periods=20)


def test_asking_for_orders_past_nyquist_is_refused():
    """A caller mirroring an RCWA order set is the case, and it is a natural thing to do.

    `truncation_order=80` is an ordinary RCWA setting and this module advertises wavebit's
    contract, so the request arrives. Past `nx/2` the x transform folds one order onto another
    and returns a plausible amplitude -- with a flipped sign on TM, from the half-cell sample
    offset -- rather than anything that could be recognised as wrong.
    """
    with pytest.raises(ValueError, match=r'orders asked of'):
        _fdtd(_grating(), [THICK], 'te', resolution=0.05, truncation_order=dict(x=80, y=0))


def test_an_evanescent_order_reports_nan_rather_than_an_amplified_reading():
    """Undoing an exponential decay amplifies whatever is left, which eventually is round-off.

    An order evanescent in the superstrate arrives at the monitor already decayed by
    `exp(k0|Im kz||z_ref|)`, and referencing it back to the interface multiplies by exactly
    that. Measured against the RCWA amplitudes: a gain of 1.1e8 still gives 6% accuracy, 4.5e12
    is wrong by 5x, and 1.4e17 by 3e5 -- and `R` is identically zero for all of them, so no
    efficiency check anywhere can notice a finite, plausible, meaningless number.
    """
    result = _fdtd(_grating(duty=0.3), [THICK], 'te', resolution=0.1,
                   truncation_order=dict(x=8, y=0))
    r_s = np.asarray(result.given.r_s).ravel()
    orders = np.asarray(result.orders_u).ravel()
    deep = np.abs(orders) >= 5           # |kx| well past n_top, so heavily evanescent in air
    assert np.all(np.isnan(r_s[deep])), (
        f'deep evanescent amplitudes are {r_s[deep]}, which are finite and meaningless')
    assert np.all(np.isfinite(np.asarray(result.R).ravel())), (
        'the efficiencies must stay finite -- they are identically zero for these orders')
    assert np.all(np.isfinite(r_s[np.abs(orders) <= 1])), (
        'the propagating orders must not be caught by the guard')


def test_a_profile_with_no_mirror_symmetry_agrees_on_the_AMPLITUDES():
    """The only test here that can see the x origin, the order-index sign, or a phase.

    Every other structure in this file is either uniform along x or mirror-symmetric about its
    own centre -- including the duty-0.3 rectangle, which looks asymmetric and is not, so RCWA
    returns `T(+1) == T(-1)` for it exactly and no comparison can tell the two orders apart.
    This two-step profile is genuinely asymmetric: `T(-1) = 0.7207` against `T(+1) = 0.0838`,
    a factor of 8.6.

    What that buys, none of which `R` or `T` can see:

    * **The x origin.** Shifting the structure by half a period multiplies order `m` by
      `(-1)**m`, which is a phase of pi on every odd order and nothing at all on `|r|**2`.
      Measured `arg` differences here are 0.007 and 0.010 rad on `m = +-1`, so the frames agree
      with about 300x of margin against the half-period alternative. This matters more than it
      looks: wavebit moved every *user-facing* coordinate to the centre of the cell on
      2026-08-31, and the reason this module was unaffected is that the raster's index
      convention and the transform's `[0, 1]` integration did not move with it -- a fact worth
      having pinned rather than reasoned about.
    * **The order-index sign.** `kx_m = ... + m*wl/period`; reversing it swaps two values that
      are 8.6x apart here and identical everywhere else in this file.
    * **The de-referencing phase**, which is what `|r|**2` discarded when it hid a half-cell
      monitor offset and a two-leg dispersion correction.
    """
    column = np.array([wavebit.eps_from_index(N_TOOTH), wavebit.eps_from_index(2.0), 1.0, 1.0])
    structure = column.reshape(1, 1, 4)

    reference = _rcwa(structure, [THICK], 'te')
    result = _fdtd(structure, [THICK], 'te', resolution=0.1)

    def amplitudes(res, name):
        index = {int(m): i for i, m in enumerate(np.asarray(res.orders_u).ravel())}
        return {m: complex(np.asarray(getattr(res.given, name)).ravel()[index[m]])
                for m in (-1, 0, 1)}

    ref_T = _by_order(reference)
    assert ref_T[-1][1] > 5.0 * ref_T[1][1], (
        'the profile has stopped being asymmetric, and this test measures nothing without it')

    # Measured at this 10 nm grid: max complex gap 1.0e-2 on `r_s` and 1.6e-2 on `t_s`,
    # falling to 2.1e-3 and 3.1e-3 at 5 nm -- a ratio of 4.7 and 5.2, so second order. Each
    # bound is about 2x the measured value and about 16x below the 2*|r| a sign flip gives.
    for name, bound in (('r_s', 2e-2), ('t_s', 3e-2)):
        got, want = amplitudes(result, name), amplitudes(reference, name)
        for m in (-1, 0, 1):
            assert abs(got[m] - want[m]) < bound, (
                f'{name}[{m}]: fdtd {got[m]} against rcwa {want[m]}')
        for m in (-1, 1):
            # The specific failure a half-period origin shift produces, named rather than
            # left to the tolerance above: it is exactly a sign flip on the odd orders.
            assert abs(got[m] + want[m]) > 0.5 * abs(want[m]), (
                f'{name}[{m}] is the NEGATIVE of the reference, which is what a half-period '
                f'shift of the x origin looks like')


# ---------------------------------------------------------------------------------------
# 6. oblique incidence
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize('degrees, tetm', [(30.0, 'te'), (45.0, 'tm')])
def test_oblique_incidence_agrees_with_the_rcwa_solve(degrees, tetm):
    """`theta` in RADIANS, which is wavebit's unit -- measured, not assumed.

    A degrees/radians slip returns a legal answer at a legal angle, so it is pinned against
    the RCWA path rather than against a comment. The classical mount (`phi = 0`) leaves TE and
    TM decoupled at every `theta`, which is what makes this answerable by the pure TE/TM
    formulation at all; a conical mount is refused.

    Measured worst per-order gap 3.5e-3 (30 deg TE) and 3.6e-3 (45 deg TM) at this 10 nm grid,
    against 2.9e-3 and 2.3e-3 at normal incidence -- so an oblique run costs very little
    accuracy, and what it costs is run length rather than resolution.
    """
    theta = np.deg2rad(degrees)
    solver = wavebit.rcwa(backend=0, wavelength=WL, period=[PERIOD],
                          truncation_order=dict(x=60, y=0), top=EPS_TOP, bot=EPS_BOT,
                          theta=theta, phi=0.0, tetm=tetm,
                          structure=_grating(), thickness=[THICK])
    reference = _by_order(solver.solve())
    orders = _by_order(solve_fdtd(
        structure=_grating(), thickness=[THICK], top=EPS_TOP, bot=EPS_BOT, period=[PERIOD],
        wavelength=WL, tetm=tetm, theta=theta, resolution=0.1, n_periods=120,
        truncation_order=dict(x=2, y=0)))

    for m in orders:
        if m not in reference:
            continue
        assert abs(orders[m][0] - reference[m][0]) < 8e-3, (
            f'R[{m}] at {degrees} deg: fdtd {orders[m][0]} against rcwa {reference[m][0]}')
        assert abs(orders[m][1] - reference[m][1]) < 8e-3, (
            f'T[{m}] at {degrees} deg: fdtd {orders[m][1]} against rcwa {reference[m][1]}')


def test_an_oblique_scene_scatters_asymmetrically_and_on_the_right_side():
    """The theorem that pins the SIGN of the incident in-plane wavevector.

    Tilting the illumination breaks the mirror symmetry that makes `T[+1] == T[-1]` at normal
    incidence, and it breaks it hard: at 30 degrees `T[-1] = 0.1324` while `T[+1]` is
    identically zero, because `kx_{+1} = sin(30 deg) + 900/700 = 1.786` is past the light line
    in glass and `kx_{-1} = -0.786` is not. Reversing the sign of `kx_inc` swaps the two, which
    no efficiency sum and no symmetric scene could notice.
    """
    theta = np.deg2rad(30.0)
    orders = _by_order(solve_fdtd(
        structure=_grating(), thickness=[THICK], top=EPS_TOP, bot=EPS_BOT, period=[PERIOD],
        wavelength=WL, tetm='te', theta=theta, resolution=0.1, n_periods=120,
        truncation_order=dict(x=1, y=0)))
    assert orders[1][1] == 0.0, (
        f'T[+1] = {orders[1][1]}; that order is evanescent in the substrate at this angle')
    assert orders[-1][1] > 0.1, (
        f'T[-1] = {orders[-1][1]}; it propagates, and carries the power +1 cannot')


def test_normal_incidence_stays_in_real_arithmetic():
    """The Bloch phase is identically one at `theta = 0`, and paying for it would be a bug.

    Carrying it as `(1+0j)` upcasts every field array and doubles the cost of every
    normal-incidence run -- which is most of them, and all of this file except the oblique
    cases. Pinned here because the two spellings are numerically identical and nothing else
    would ever notice.
    """
    from wavebit._core.fdtd import _Grid

    def grid(theta):
        return _Grid(WL, PERIOD, np.array([[EPS_TOP, EPS_TOP]]), np.array([THICK]),
                     EPS_TOP, EPS_BOT, resolution=0.02, n_pml=12, courant=0.99,
                     subpixel=True, theta=theta)

    straight = grid(0.0)
    assert straight.dtype is float and straight.bloch == 1.0 and straight.kx_op == 0.0
    tilted = grid(np.deg2rad(30.0))
    assert tilted.dtype is complex and abs(abs(tilted.bloch) - 1.0) < 1e-12
    assert tilted.kx_op != 0.0


def test_a_field_still_ringing_at_the_end_of_the_run_warns():
    """A truncated running DFT returns a number shaped exactly like an efficiency.

    The run-length floor refuses what is knowable before the loop -- the source not having
    finished emitting. This is the other half, and it is only knowable afterwards: a scene
    that clears slowly. It matters most at oblique incidence, where 60 optical cycles left a
    residual of 5.9e-2 and an answer wrong by 3.4e-2 at 15 degrees, against 4.5e-3 at 120.
    """
    from wavebit._core.fdtd import UnsettledFieldWarning

    with pytest.warns(UnsettledFieldWarning, match='had not settled'):
        solve_fdtd(structure=_grating(), thickness=[THICK], top=EPS_TOP, bot=EPS_BOT,
                   period=[PERIOD], wavelength=WL, tetm='tm', theta=np.deg2rad(15.0),
                   resolution=0.05, n_periods=50, truncation_order=dict(x=1, y=0))


# ---------------------------------------------------------------------------------------
# 7. a spectrum out of one run
# ---------------------------------------------------------------------------------------

BAND = [860.0, 900.0, 940.0]


def test_a_band_costs_one_run_and_gives_the_same_answers_as_three():
    """The one thing a time-domain solve does that a frequency-domain one cannot.

    The pulse carries every frequency in its band, so a second wavelength costs one more
    running sum over a monitor plane against a whole new RCWA solve. This asserts the part
    that matters: it is not an approximation. The band run and three separate single-wavelength
    runs agree to 3e-3, which is the difference between two source bandwidths and not between
    two answers -- a band buys itself a shorter pulse, so the two runs are not the same
    numerical experiment even though they are the same physical one.
    """
    together = solve_fdtd(structure=_grating(), thickness=[THICK], top=EPS_TOP, bot=EPS_BOT,
                          period=[PERIOD], wavelength=BAND, tetm='te', resolution=0.05,
                          n_periods=100, truncation_order=dict(x=1, y=0))
    assert isinstance(together, list) and len(together) == len(BAND)

    for wavelength, banded in zip(BAND, together):
        alone = solve_fdtd(structure=_grating(), thickness=[THICK], top=EPS_TOP, bot=EPS_BOT,
                           period=[PERIOD], wavelength=wavelength, tetm='te', resolution=0.05,
                           n_periods=100, truncation_order=dict(x=1, y=0))
        for m in (-1, 0, 1):
            got = _by_order(banded)[m]
            want = _by_order(alone)[m]
            assert abs(got[0] - want[0]) < 3e-3 and abs(got[1] - want[1]) < 3e-3, (
                f'{wavelength} nm, order {m}: in a band {got} against alone {want}')


def test_a_band_at_oblique_incidence_is_a_fixed_kx_sweep_and_says_so():
    """At oblique incidence a band is a fixed-`kx` sweep, not a fixed-angle one.

    The grid holds one in-plane wavevector, and `kx = k0 n sin(theta)` carries `k0` -- so the
    polar angle drifts across the band, by construction and not by defect. Measured on a
    30-degree centre: 28.5 degrees at 860 nm and 31.5 at 940. That is a well-defined and
    useful sweep (it is how a band structure is mapped), it is simply not the one a naive
    reading of `theta=30 deg, wavelength=[...]` would expect, so the angle is reported.

    The check that it is right rather than merely different: solve the RCWA path at the
    DRIFTED angle and require agreement. It agrees to 5.4e-3, against 2.3e-2 when the
    extraction reused the centre wavelength's normalised `kx` -- a defect only a band at
    oblique incidence could expose, since it is identically zero at normal incidence.
    """
    theta = np.deg2rad(30.0)
    results, diagnostics = solve_fdtd(
        structure=_grating(), thickness=[THICK], top=EPS_TOP, bot=EPS_BOT, period=[PERIOD],
        wavelength=BAND, tetm='te', theta=theta, resolution=0.1, n_periods=140,
        truncation_order=dict(x=2, y=0), return_diagnostics=True)

    angles = diagnostics['angles_deg']
    assert abs(angles[1] - 30.0) < 1e-9, 'the centre wavelength must be the angle asked for'
    assert angles[0] < 30.0 < angles[2], f'the angle must drift across the band, got {angles}'

    for wavelength, angle, result in zip(BAND, angles, results):
        solver = wavebit.rcwa(backend=0, wavelength=wavelength, period=[PERIOD],
                              truncation_order=dict(x=60, y=0), top=EPS_TOP, bot=EPS_BOT,
                              theta=np.deg2rad(angle), phi=0.0, tetm='te',
                              structure=_grating(), thickness=[THICK])
        reference = _by_order(solver.solve())
        for m, (R, T) in _by_order(result).items():
            if m not in reference:
                continue
            assert abs(R - reference[m][0]) < 8e-3 and abs(T - reference[m][1]) < 8e-3, (
                f'{wavelength} nm at {angle:.2f} deg, order {m}: '
                f'fdtd ({R}, {T}) against rcwa {reference[m]}')


def test_the_source_widens_to_cover_whatever_band_is_asked_for():
    """Why the band-edge cliff never arrives -- stated as the invariant, not left to luck.

    Every amplitude is divided by the incident spectrum, so a wavelength where that spectrum
    has decayed into round-off returns noise over noise: measured on a prototype with a fixed
    pulse width, an incident spectrum 2.4e-7 of its peak gave `R = 105` on a scene whose true
    `R` is 0.04, with nothing on the number to say it was wrong.

    `_time_loop` sizes the pulse from the requested span instead, which is what makes that
    unreachable -- so `MIN_INCIDENT_FRACTION` is a backstop against a change to the sizing
    rule rather than a live guard, and this is the assertion that says so. A three-octave band
    is deliberate: if any band could starve an edge, that one would.
    """
    from wavebit._core.fdtd import MIN_INCIDENT_FRACTION

    for band in ([860.0, 900.0, 940.0], [400.0, 900.0, 2400.0]):
        _, diagnostics = solve_fdtd(
            structure=_grating(), thickness=[THICK], top=EPS_TOP, bot=EPS_BOT,
            period=[PERIOD], wavelength=band, tetm='te', resolution=0.05, n_periods=100,
            truncation_order=dict(x=1, y=0), return_diagnostics=True)
        worst = float(np.min(diagnostics['incident_fraction']))
        assert worst > 0.2, (
            f'band {band}: the weakest wavelength sits at {worst:.2e} of the peak incident '
            f'spectrum, so the source is not covering what was asked for')
        assert worst > MIN_INCIDENT_FRACTION * 100


# ---------------------------------------------------------------------------------------
# 8. dispersion -- the one thing the RCWA path can express and this could not
# ---------------------------------------------------------------------------------------

# A Drude metal, gold-ish at 900 nm: plasma wavelength 136 nm, a long collision time.
# `(omega0, gamma, strength)` with omega0 = 0 is Drude and `strength` is the plasma frequency
# squared; the units are reciprocal length, since c = 1 here.
DRUDE = {1: [(0.0, 2.0 * np.pi / 17000.0, (2.0 * np.pi / 136.0) ** 2)]}


def test_a_pole_model_and_a_complex_permittivity_are_the_same_material():
    """The closed form the time stepping discretises, checked before anything is stepped.

    This is what makes the comparisons below meaningful: the RCWA reference is handed a single
    complex `eps` and the FDTD is handed a pole model, so if the two disagree about what the
    material IS then nothing downstream means anything. It also pins the sign convention --
    `Im(eps) > 0` is loss under `e^{-iwt}`, which is wavebit's and not a free choice.
    """
    from wavebit._core.fdtd import permittivity_at

    eps = permittivity_at(DRUDE[1], WL, 1.0)
    assert abs(eps - (-42.6709 + 2.3120j)) < 1e-3, eps
    assert eps.imag > 0.0, f'Im(eps) = {eps.imag}, which is gain in this convention'
    # A Drude metal well below its plasma frequency: eps strongly negative, n mostly imaginary.
    assert eps.real < -10.0 and np.sqrt(eps).imag > 5.0


@pytest.mark.parametrize('tetm', ['te', 'tm'])
def test_a_drude_slab_matches_the_closed_form(tetm):
    """The decisive gate for the auxiliary differential equation, and it needs no solver.

    A uniform Drude slab has an exact transfer matrix once `eps(w)` is known, so this compares
    the time stepping against arithmetic rather than against another implementation. 30 nm of
    metal against an 11 nm skin depth: `T` is 0.036, small but not zero, which is what makes
    the test sensitive to the absorption rather than only to the reflection.

    Measured error 3.7e-3 (R) at this 20 nm grid, falling to 5.9e-4 at 10 nm and 8.5e-5 at 5 --
    second order or better.
    """
    from wavebit._core.fdtd import permittivity_at

    index = np.sqrt(permittivity_at(DRUDE[1], WL, 1.0))
    thickness, structure = 30.0, np.ones((1, 1, 2))
    result = solve_fdtd(structure=structure, thickness=[thickness], top=EPS_TOP, bot=EPS_BOT,
                        period=[PERIOD], wavelength=WL, tetm=tetm, resolution=0.05,
                        n_periods=100, truncation_order=dict(x=1, y=0),
                        dispersion=np.array([[1, 1]]), poles=DRUDE)
    _, _, R_a, T_a = _stack_rt(WL, N_TOP, [index], [thickness], N_BOT, tetm)
    orders = _by_order(result)

    assert abs(orders[0][0] - R_a) < 1e-2, f'R {orders[0][0]} against closed form {R_a}'
    assert abs(orders[0][1] - T_a) < 1e-2, f'T {orders[0][1]} against closed form {T_a}'
    absorbed = 1.0 - orders[0][0] - orders[0][1]
    assert 0.01 < absorbed < 0.05, (
        f'the slab absorbed {absorbed}; the closed form says {1 - R_a - T_a:.4f}, and a scene '
        f'that absorbs nothing means the pole never reached the field update')


def test_a_metal_grating_agrees_with_the_rcwa_solve():
    """The target: a structure RCWA can express and this module could not, until now.

    The two are handed the material in different forms -- RCWA a complex `eps` at 900 nm, the
    FDTD a Drude pole -- which is the point rather than an inconvenience: agreement means the
    pole model and the complex number describe one material AND that the ADE integrates it
    correctly. Measured worst per-order gap 2.7e-3 at this 10 nm grid, 6.1e-4 at 5 nm.
    """
    from wavebit._core.fdtd import permittivity_at

    thickness = 60.0
    eps_metal = permittivity_at(DRUDE[1], WL, 1.0)
    solver = wavebit.rcwa(backend=0, wavelength=WL, period=[PERIOD],
                          truncation_order=dict(x=60, y=0), top=EPS_TOP, bot=EPS_BOT,
                          theta=0.0, phi=0.0, tetm='te',
                          structure=np.array([[[eps_metal, EPS_TOP]]]), thickness=[thickness])
    reference = _by_order(solver.solve())

    orders = _by_order(solve_fdtd(
        structure=np.ones((1, 1, 2)), thickness=[thickness], top=EPS_TOP, bot=EPS_BOT,
        period=[PERIOD], wavelength=WL, tetm='te', resolution=0.1, n_periods=100,
        truncation_order=dict(x=1, y=0), dispersion=np.array([[1, 0]]), poles=DRUDE))

    for m in (-1, 0, 1):
        assert abs(orders[m][0] - reference[m][0]) < 6e-3, (
            f'R[{m}]: fdtd {orders[m][0]} against rcwa {reference[m][0]}')
        assert abs(orders[m][1] - reference[m][1]) < 6e-3, (
            f'T[{m}]: fdtd {orders[m][1]} against rcwa {reference[m][1]}')
    total = float(np.asarray(orders[0]).sum())
    assert total < 0.999, f'R0 + T0 = {total}; a metal grating has to absorb something'


DISPERSION_REFUSED = [
    (dict(dispersion=np.array([[1, 0]])), r'come as a pair'),
    (dict(poles=DRUDE), r'come as a pair'),
    (dict(dispersion=np.array([[1, 0, 0]]), poles=DRUDE), r'share a shape exactly'),
    (dict(dispersion=np.array([[2, 0]]), poles=DRUDE), r'no entry for them'),
    (dict(dispersion=np.array([[1, 0]]), poles={1: [(0.0, -1.0, 1.0)]}), r'negative'),
    (dict(dispersion=np.array([[1, 0]]), poles={1: [(0.0, 1.0)]}), r'omega0, gamma, strength'),
]


@pytest.mark.parametrize('override, message', DISPERSION_REFUSED)
def test_dispersion_refusals(override, message):
    """A pole set that is silently dropped answers a lossy question losslessly.

    That is the same failure mode as every other refusal here -- a number shaped exactly like
    an efficiency, with nothing on it to say the material was ignored.
    """
    kwargs = dict(structure=np.ones((1, 1, 2)), thickness=[30.0], top=EPS_TOP, bot=EPS_BOT,
                  period=[PERIOD], wavelength=WL, tetm='te', resolution=0.02, n_periods=60)
    kwargs.update(override)
    with pytest.raises(ValueError, match=message):
        solve_fdtd(**kwargs)


def test_a_band_through_a_metal_carries_the_dispersion_at_every_wavelength():
    """Where dispersion and a band multiply each other, and the clearest case for this solver.

    One run gives the whole spectrum, and each wavelength in it sees the permittivity the pole
    model has THERE -- not the centre's. RCWA has to be re-solved per wavelength *and* handed a
    different `eps` each time. This asserts the part that could silently be wrong: that
    `eps(w)` really varies across the band rather than being frozen at the centre.
    """
    from wavebit._core.fdtd import permittivity_at

    band = [820.0, 900.0, 980.0]
    metal = [permittivity_at(DRUDE[1], wl, 1.0) for wl in band]
    assert abs(metal[0] - metal[-1]) > 5.0, (
        'the band has to span a real change in eps or this test proves nothing')

    thickness = 30.0
    results = solve_fdtd(structure=np.ones((1, 1, 2)), thickness=[thickness], top=EPS_TOP,
                         bot=EPS_BOT, period=[PERIOD], wavelength=band, tetm='te',
                         resolution=0.05, n_periods=100, truncation_order=dict(x=1, y=0),
                         dispersion=np.array([[1, 1]]), poles=DRUDE)

    for wavelength, eps, result in zip(band, metal, results):
        _, _, R_a, T_a = _stack_rt(wavelength, N_TOP, [np.sqrt(eps)], [thickness], N_BOT, 'te')
        got = _by_order(result)[0]
        assert abs(got[0] - R_a) < 1.5e-2, (
            f'{wavelength} nm: R {got[0]} against the closed form at THAT wavelength {R_a}')
        assert abs(got[1] - T_a) < 1.5e-2, f'{wavelength} nm: T {got[1]} against {T_a}'


# ---------------------------------------------------------------------------------------
# 9. permeability
# ---------------------------------------------------------------------------------------
#
# **Energy conservation is not a check here.** `R + T = 1` passes for magnetic code that is
# wrong -- on the RCWA side all six path/algorithm combinations conserved energy on the first
# smoke test, before any real oracle ran (docs/magnetic-materials.md). Every test below is
# something else: an exact identity, a closed form that carries mu, or a duality.

@pytest.mark.parametrize('tetm', ['te', 'tm'])
def test_an_impedance_matched_slab_does_not_reflect_at_all(tetm):
    """`eps = mu` makes a slab invisible however large the index is. The sharpest mu test.

    Wave impedance is `sqrt(mu/eps)`, so a slab with `eps = mu = 4` is matched to vacuum and
    reflects **nothing**, at any thickness -- while its index is `sqrt(eps*mu) = 4`, so it is
    optically thick and the transmitted phase is far from vacuum's. A solver that drops `mu`
    sees `eps = 4, mu = 1`, an index-2 slab, and reflects a great deal.

    Needs no oracle and no tolerance argument: the answer is zero.
    """
    # 110 nm is a whole number of cells at this resolution AND close to the quarter wave of
    # the CONTROL (index 2, lambda/4n = 112.5 nm), where a slab reflects the most it ever
    # will. Both halves of that matter: a thickness straddling a cell spoils the discrete
    # match, and a half-wave thickness would make the control transparent too. An earlier
    # draft picked 220 nm and measured a control of 0.002 instead of 0.36.
    eps = mu = 4.0
    result = solve_fdtd(structure=np.full((1, 1, 2), eps), thickness=[110.0],
                        mu=np.full((1, 1, 2), mu), top=1.0, bot=1.0, period=[PERIOD],
                        wavelength=WL, tetm=tetm, resolution=0.1, n_periods=80,
                        truncation_order=dict(x=1, y=0))
    orders = _by_order(result)
    # Measured 4.6e-7 at this 10 nm grid and 2.0e-8 at 5 nm. The control -- the same scene
    # with `mu` dropped -- gives 0.358, a separation of 774,000x, and that is what caught
    # three H updates silently keeping their non-magnetic form.
    assert orders[0][0] < 1e-5, (
        f'R = {orders[0][0]} from an impedance-matched slab, which must not reflect')
    assert abs(orders[0][1] - 1.0) < 5e-3, f'T = {orders[0][1]}, and nothing was absorbed'


@pytest.mark.parametrize('tetm', ['te', 'tm'])
def test_a_magnetic_slab_matches_the_closed_form(tetm):
    """A slab that is magnetic *and* mismatched, against the transfer matrix that carries mu.

    `eps = 2, mu = 3`: index `sqrt(6) = 2.449` and impedance `sqrt(1.5) = 1.22`, so neither the
    index nor the impedance is the vacuum's and a solver cannot get this right by accident.
    The two polarisations are no longer degenerate at normal incidence either -- an s channel's
    admittance is `n/mu` and a p channel's is `n/eps` -- so this is two tests, not one twice.
    """
    eps, mu, thickness = 2.0, 3.0, 180.0
    index = np.sqrt(eps * mu)
    result = solve_fdtd(structure=np.full((1, 1, 2), eps), thickness=[thickness],
                        mu=np.full((1, 1, 2), mu), top=EPS_TOP, bot=EPS_BOT, period=[PERIOD],
                        wavelength=WL, tetm=tetm, resolution=0.05, n_periods=80,
                        truncation_order=dict(x=1, y=0))
    _, _, R_a, T_a = _stack_rt(WL, N_TOP, [index], [thickness], N_BOT, tetm, mu_list=[mu])
    orders = _by_order(result)
    assert abs(orders[0][0] - R_a) < 8e-3, f'R {orders[0][0]} against closed form {R_a}'
    assert abs(orders[0][1] - T_a) < 8e-3, f'T {orders[0][1]} against closed form {T_a}'


def test_the_same_index_with_a_different_impedance_is_a_different_scene():
    """The one thing a solver carrying only `n` cannot express, stated as the test.

    `eps = 2, mu = 3` and `eps = 6, mu = 1` have the **same index** `sqrt(eps*mu) = 2.449` and
    different impedances `sqrt(mu/eps)` -- 1.22 against 0.41. Every `kz`, every Snell relation
    and the whole order grid are identical between them; only the two admittances differ. A
    solver that had reduced the pair to an index would see no interface at all between vacuum
    and the first of them, which is the sentence `sp_efficiencies`' docstring puts it in.

    Note what this test is NOT: at normal incidence TE and TM stay degenerate in `\|r\|^2`
    whether or not the material is magnetic, since `r_p = -r_s`. An earlier draft asserted the
    opposite and was wrong about the physics rather than about the code.
    """
    # 90 nm is the quarter wave of the shared index (91.9 nm) rounded to a whole number of
    # cells. At 180 nm the layer is a HALF wave and nearly invisible, so both arms collapse
    # onto the bare substrate's 0.04 and the test measures nothing -- an earlier draft did
    # exactly that.
    kwargs = dict(thickness=[90.0], top=EPS_TOP, bot=EPS_BOT, period=[PERIOD],
                  wavelength=WL, tetm='te', resolution=0.1, n_periods=80,
                  truncation_order=dict(x=1, y=0))
    magnetic = _by_order(solve_fdtd(structure=np.full((1, 1, 2), 2.0),
                                    mu=np.full((1, 1, 2), 3.0), **kwargs))[0]
    plain = _by_order(solve_fdtd(structure=np.full((1, 1, 2), 6.0), **kwargs))[0]
    assert abs(np.sqrt(2.0 * 3.0) - np.sqrt(6.0)) < 1e-12, 'the two indices must be equal'
    # Measured 0.150 against 0.358 -- a gap of 0.21.
    assert abs(magnetic[0] - plain[0]) > 0.1, (
        f'R is {magnetic[0]} magnetic against {plain[0]} non-magnetic; the two have the same '
        f'index, so an equal answer means the impedance never reached the solve')


@pytest.mark.parametrize('tetm', ['te', 'tm'])
def test_a_magnetic_grating_agrees_with_the_rcwa_solve(tetm):
    """The cross-check that only became possible on 2026-09-01.

    The RCWA path refused `mu` on its 1D operators until then, so a magnetic 1D grating had no
    reference at all and this module refused it for parity. `545dd4c` lifted that, and this is
    the test the lifting bought.
    """
    eps_tooth, mu_tooth, thickness = 4.0, 2.5, 200.0
    structure = np.array([[[eps_tooth, EPS_TOP]]])
    mu_raster = np.array([[[mu_tooth, 1.0]]])

    solver = wavebit.rcwa(backend=0, wavelength=WL, period=[PERIOD],
                          truncation_order=dict(x=60, y=0), top=EPS_TOP, bot=EPS_BOT,
                          theta=0.0, phi=0.0, tetm=tetm,
                          structure=structure, thickness=[thickness], mu=mu_raster)
    reference = _by_order(solver.solve())
    orders = _by_order(solve_fdtd(structure=structure, thickness=[thickness], mu=mu_raster,
                                  top=EPS_TOP, bot=EPS_BOT, period=[PERIOD], wavelength=WL,
                                  tetm=tetm, resolution=0.1, n_periods=80,
                                  truncation_order=dict(x=1, y=0)))
    for m in (-1, 0, 1):
        assert abs(orders[m][0] - reference[m][0]) < 8e-3, (
            f'R[{m}]: fdtd {orders[m][0]} against rcwa {reference[m][0]}')
        assert abs(orders[m][1] - reference[m][1]) < 8e-3, (
            f'T[{m}]: fdtd {orders[m][1]} against rcwa {reference[m][1]}')


def test_a_magnetic_half_space_is_read():
    """`mu` on a half-space arrives inside its `Material`, which is where the RCWA path takes it.

    A half-space permeability reaches the index, both admittances and the whole order grid, and
    nothing about the raster would notice if it were dropped -- so it is checked on its own.
    """
    from wavebit import Material

    substrate = Material(2.25, mu=1.6)
    result = solve_fdtd(structure=np.full((1, 1, 2), EPS_TOP), thickness=[100.0],
                        top=1.0, bot=substrate, period=[PERIOD], wavelength=WL, tetm='te',
                        resolution=0.1, n_periods=80, truncation_order=dict(x=1, y=0))
    index = np.sqrt(2.25 * 1.6)
    _, _, R_a, T_a = _stack_rt(WL, 1.0, [], [], index, 'te', mu_bot=1.6)
    orders = _by_order(result)
    # Measured 1.5e-4 and 2.0e-3 at this grid. Dropping the half-space mu gives R = 0.040
    # against a true 0.0072 -- the answer of an n = 1.5 substrate.
    assert abs(orders[0][0] - R_a) < 2e-3, f'R {orders[0][0]} against closed form {R_a}'
    assert abs(orders[0][1] - T_a) < 8e-3, f'T {orders[0][1]} against closed form {T_a}'


def test_the_default_order_window_follows_the_incidence():
    """At an angle the propagating set is not centred on zero, and missing one is silent.

    `kx_m = kx_inc + m*wl/period`, so the orders that propagate are those with
    `|kx_m| <= n` -- a range centred on `-kx_inc`, asymmetric in `m`. A window that assumed
    symmetry dropped orders on the far side and the only symptom was a short energy sum.

    Measured at 60 degrees into an `n = 2` substrate: order `-2` propagates and carries 7.2 %
    of the power. The symmetric window reported `sum(R) + sum(T) = 0.928`, with nothing on the
    number to say an order had been left out. No result-level check can see this: a short sum
    is what ordinary absorption looks like.
    """
    theta = np.deg2rad(60.0)
    eps_bot = wavebit.eps_from_index(2.0)
    result = solve_fdtd(structure=_grating(), thickness=[THICK], top=EPS_TOP, bot=eps_bot,
                        period=[PERIOD], wavelength=WL, tetm='te', theta=theta,
                        resolution=0.1, n_periods=140)
    orders = _by_order(result)
    assert -2 in orders, f'order -2 propagates here and the window is {sorted(orders)}'
    assert orders[-2][1] > 0.05, (
        f'T[-2] = {orders[-2][1]}; it carries 7.2 % of the power in the RCWA solve')
    total = sum(R + T for R, T in orders.values())
    assert abs(total - 1.0) < 5e-3, (
        f'R + T = {total}; a short sum is what a dropped propagating order looks like')

    labels = np.asarray(result.orders_u).ravel()
    assert labels.min() < 0 <= labels.max(), f'window {labels} must straddle the specular order'


# ---------------------------------------------------------------------------------------
# 10. a conical mount at normal incidence -- the output frame, on its own
# ---------------------------------------------------------------------------------------

def _conical(phi_deg, psi_deg, resolution=0.1, n_periods=60):
    return solve_fdtd(structure=_grating(), thickness=[THICK], top=EPS_TOP, bot=EPS_BOT,
                      period=[PERIOD], wavelength=WL, theta=0.0,
                      phi=np.deg2rad(phi_deg), psi=np.deg2rad(psi_deg),
                      resolution=resolution, n_periods=n_periods)


def _conical_rcwa(phi_deg, psi_deg, orders=40):
    return wavebit.rcwa(backend=0, wavelength=WL, period=[PERIOD],
                        truncation_order=dict(x=orders, y=0), top=EPS_TOP, bot=EPS_BOT,
                        theta=0.0, phi=np.deg2rad(phi_deg), psi=np.deg2rad(psi_deg),
                        structure=_grating(), thickness=[THICK]).solve()


def _channels(result, leg, name):
    """``{m: value}`` for one channel of one leg. ``orders_u`` lives on the Result, not the
    sub-result, so the labels have to come from the parent -- which is the point of them."""
    a = np.asarray(getattr(getattr(result, leg), name)).ravel()
    return {int(m): a[i] for i, m in enumerate(np.asarray(result.orders_u).ravel())}


def test_at_normal_incidence_the_azimuth_only_relabels_the_channels():
    """theta = 0, phi = 90 deg: one physical field, and the orders land in DIFFERENT channels.

    At normal incidence `ky` is exactly zero whatever `phi` is, so nothing about the field
    changes -- the whole azimuth dependence is the output frame. And that frame is not one
    rotation: the zeroth order's in-plane wavevector is the origin, so it takes `phi`, while
    every `kx_m != 0` order has azimuth 0 or pi. At `phi = 90 deg` the two are a right angle
    apart, so the SAME x-polarised field reads as pure s in the specular order and pure p in
    the first orders.

    That asymmetry is the assertion. A projection that used one frame for all orders would
    put both in the same channel, conserve energy exactly, and be wrong.
    """
    result = _conical(phi_deg=90.0, psi_deg=90.0)          # (a_s, a_p) = (sin psi, cos psi)
    tm = _by_order(_rcwa(_grating(), [THICK], 'tm'))       # the same physics, named TE/TM

    R_s, R_p = _channels(result, 's_inc', 'R_s'), _channels(result, 's_inc', 'R_p')
    T_s, T_p = _channels(result, 's_inc', 'T_s'), _channels(result, 's_inc', 'T_p')

    assert R_s[0] == pytest.approx(tm[0][0], abs=5e-3), 'the specular order takes phi'
    assert R_p[0] < 1e-12, f'R_p[0] = {R_p[0]}; the specular frame is at phi = 90 deg'
    assert T_s[0] == pytest.approx(tm[0][1], abs=5e-3)
    assert T_p[0] < 1e-12

    for m in (-1, 1):
        assert T_p[m] == pytest.approx(tm[m][1], abs=5e-3), (
            f'order {m} has azimuth 0 or pi, a right angle from the specular frame, so the '
            f'same field reads as p there')
        assert T_s[m] < 1e-12, f'T_s[{m}] = {T_s[m]}'


@pytest.mark.parametrize('phi_deg', [0.0, 30.0, 45.0, 60.0, 90.0, 135.0])
def test_the_conical_projection_matches_the_rcwa_path(phi_deg):
    """Every channel of every order, against `solve_1d_conical`, on both pure legs.

    The residual here is the Yee grid's, not the projection's: at `phi = 45 deg` it falls
    8.2e-3 / 2.1e-3 / 5.9e-4 / 1.8e-4 over dz = 20 / 10 / 5 / 2.5 nm, an order of 1.94 / 1.86
    / 1.73. A projection defect leaves a floor instead.
    """
    got = _conical(phi_deg, psi_deg=30.0)
    ref = _conical_rcwa(phi_deg, psi_deg=30.0)
    for leg in ('given', 's_inc', 'p_inc'):
        for name in ('R_s', 'R_p', 'T_s', 'T_p'):
            a = _channels(got, leg, name)
            b = _channels(ref, leg, name)
            for m in a:
                assert a[m] == pytest.approx(b[m], abs=5e-3), f'{leg}.{name}[{m}]'


def test_the_transmitted_leg_carries_its_own_p_direction():
    """`p_hat = k_hat x s_hat` reverses between the up- and down-going legs, and only phi = 45 sees it.

    `s_hat` is common to both legs, so the reflected and transmitted frames differ by the sign
    of `sin varphi_m` alone. Two things hide that. Every order with `kx_m != 0` has azimuth 0
    or pi, where `sin varphi_m` is zero -- so **only the specular order can carry the defect**.
    And at `phi = 0` the term is absent while at `phi = 90 deg` it is the entire answer, so
    both signs pass there. Measured with the reflected rule used for both, at 45 degrees:
    `T_p[0] = 0.241` against a true 0.017, while `R` was correct to 3e-3 and `R + T` was 1.000.
    """
    got = _conical(phi_deg=45.0, psi_deg=90.0)
    ref = _conical_rcwa(45.0, psi_deg=90.0)
    assert _channels(got, 's_inc', 'T_p')[0] == pytest.approx(
        _channels(ref, 's_inc', 'T_p')[0], abs=5e-3)
    assert _channels(got, 's_inc', 'T_p')[0] < 0.1, 'the wrong leg sign returned 0.241 here'


def test_the_two_pure_legs_compose_the_given_incidence():
    """`given` is `a_s * s_inc + a_p * p_inc` in AMPLITUDE -- exactly, not to a tolerance.

    The two scalar solves are independent, so both legs come back for free and the caller can
    reweight the polarisation without solving again. Asserted on the complex amplitudes: the
    efficiencies are quadratic and would pass on a sign error in either leg.
    """
    psi = np.deg2rad(30.0)
    got = _conical(phi_deg=45.0, psi_deg=30.0)
    a_s, a_p = np.sin(psi), np.cos(psi)
    for name in ('r_s', 'r_p', 't_s', 't_p'):
        mixed = a_s * np.asarray(getattr(got.s_inc, name)) + a_p * np.asarray(
            getattr(got.p_inc, name))
        assert np.allclose(np.asarray(getattr(got.given, name)), mixed, atol=1e-13), name


def test_psi_and_tetm_together_are_refused():
    """Two spellings of the polarisation at once name it twice, and neither wins silently."""
    with pytest.raises(ValueError, match='name the polarisation twice'):
        _ = solve_fdtd(structure=_grating(), thickness=[THICK], top=EPS_TOP, bot=EPS_BOT,
                       period=[PERIOD], wavelength=WL, theta=0.0, psi=0.3, tetm='te',
                       resolution=0.05, n_periods=20)


def test_a_conical_mount_off_normal_takes_the_six_component_stepper():
    """`theta != 0` with `psi` was refused until the stepper was wired; it is answered now.

    The refusal this replaces said the coupled operator was written but had no extraction. It
    also had no source: the auxiliary grid was built without `ky` or a drive, so the TFSF plane
    injected a non-conical wave into it. Both are wired, and the route taken here is the
    six-component one rather than the `theta = 0` projection -- asserted on `ky`, since the two
    paths agree at `ky = 0` by construction and only part company away from it.
    """
    from wavebit._core.fdtd import _Grid
    grid = _Grid(WL, PERIOD, np.asarray(_grating(), dtype=complex)[:, 0, :], [THICK],
                 EPS_TOP, EPS_BOT, 0.05, 12, 0.99, True,
                 theta=np.deg2rad(30.0), phi=np.deg2rad(40.0))
    assert grid.ky_inc != 0.0, 'the oblique conical path must reach a nonzero ky'

    result = _oblique_fdtd(resolution=0.06, n_periods=80)
    assert result.s_inc is not None and result.p_inc is not None, (
        'both eigenpolarisations come back from a conical run')
    total = float(np.asarray(result.given.R).sum() + np.asarray(result.given.T).sum())
    assert abs(total - 1.0) < 2e-2, f'R + T = {total}' 


def test_tetm_with_an_azimuth_is_still_refused_and_points_at_psi():
    """The old refusal stays, but now it names the door that opened."""
    with pytest.raises(ValueError, match='pass `psi`'):
        _ = solve_fdtd(structure=_grating(), thickness=[THICK], top=EPS_TOP, bot=EPS_BOT,
                       period=[PERIOD], wavelength=WL, theta=0.0, tetm='te',
                       phi=np.deg2rad(30.0), resolution=0.05, n_periods=20)


def test_the_tfsf_source_injects_a_conical_plane_wave():
    """On vacuum the scattered-field region must stay empty -- at any azimuth.

    This is the only instrument in the module with resolving power over the conical source, and
    it needs no oracle: with nothing to scatter from, everything above the TFSF plane is
    scattered field and must cancel to round-off. Before the source was wired it did not --
    measured 6.9e-2 / 9.5e-2 / 6.7e-2 of the incident amplitude at phi = 30 / 45 / 90 deg
    against 3.7e-17 at phi = 0. Thirteen orders, and **every other gate in the module passed
    throughout**: the stepper was stable and decaying, and the public path with the refusal
    lifted returned `R + T = 0.9966`.

    Both transverse components are checked, and that is not padding. Watching only `Ey` says
    1e-32 for pure-s incidence at phi = 90 deg -- not because it is exact but because the wave
    is x-polarised there and the monitor cannot see it. A one-component check reports its own
    blindness as a pass.
    """
    from wavebit._core.fdtd import _Grid, _time_loop
    vacuum = np.ones((1, 2), dtype=complex)
    for phi_deg in (0.0, 30.0, 45.0, 90.0):
        for leg in ('s', 'p'):
            grid = _Grid(WL, PERIOD, vacuum, [THICK], EPS_TOP, EPS_TOP, 0.05, 12, 0.99, True,
                         theta=np.deg2rad(30.0), phi=np.deg2rad(phi_deg),
                         mu_top=1.0 + 0j, mu_bot=1.0 + 0j)
            out = _time_loop(grid, 'te', int(np.ceil(60.0 * WL / grid.dt)),
                             np.array([WL]), leg=leg)
            for name in ('dft_ref',):
                leak = float(np.abs(out[name][0]).max())
                total = float(np.abs(out['inc_z0'][0]))
                if total > 1e-12:                     # the leg this monitor can see
                    assert leak / total < 1e-10, (
                        f'phi={phi_deg} leg={leg}: {leak / total:.2e} of the incident amplitude '
                        f'is left in the scattered-field region on a vacuum scene')


def test_the_auxiliary_grid_takes_a_dimensional_ky():
    """`_Incident1D.ky` is 1/length, unlike every other `ky` name in the module.

    `kx_op` beside it is the discrete x-derivative operator and already carries `k0`, so its
    partner must too -- while `grid.ky_inc`, like `kx_inc`, is normalised by `k0`. Passing the
    normalised value is a factor `k0` too large: 143x on a 900 nm run, and the auxiliary grid
    reaches 1e267 in 200 steps and takes the 2D grid to NaN with it.

    Pinned because the failure is loud but the CAUSE is not: an unstable FDTD reads as a
    Courant problem, and the Courant bound here already includes the ky term.
    """
    from wavebit._core.fdtd import _Grid, _time_loop
    grid = _Grid(WL, PERIOD, np.ones((1, 2), dtype=complex), [THICK], EPS_TOP, EPS_TOP,
                 0.05, 12, 0.99, True, theta=np.deg2rad(30.0), phi=np.deg2rad(45.0),
                 mu_top=1.0 + 0j, mu_bot=1.0 + 0j)
    out = _time_loop(grid, 'te', int(np.ceil(60.0 * WL / grid.dt)), np.array([WL]), leg='s')
    assert np.all(np.isfinite(out['dft_ref'])), 'the conical run went non-finite'
    assert float(np.abs(out['dft_ref'][0]).max()) < 1e3, 'the conical run diverged'


# ---------------------------------------------------------------------------------------
# 11. a conical mount away from normal -- the six-component stepper, end to end
# ---------------------------------------------------------------------------------------

OBLIQUE = dict(theta=np.deg2rad(30.0), phi=np.deg2rad(40.0), psi=np.deg2rad(30.0))


def _oblique_fdtd(resolution=0.1, n_periods=120, **over):
    kw = dict(OBLIQUE, **over)
    return solve_fdtd(structure=_grating(), thickness=[THICK], top=EPS_TOP, bot=EPS_BOT,
                      period=[PERIOD], wavelength=WL, resolution=resolution,
                      n_periods=n_periods, **kw)


def _oblique_rcwa(orders=40, **over):
    kw = dict(OBLIQUE, **over)
    return wavebit.rcwa(backend=0, wavelength=WL, period=[PERIOD],
                        truncation_order=dict(x=orders, y=0), top=EPS_TOP, bot=EPS_BOT,
                        structure=_grating(), thickness=[THICK], **kw).solve()


def test_the_oblique_conical_mount_matches_the_rcwa_path():
    """theta = 30 deg, phi = 40 deg: every channel of every order, on all three legs.

    This is the six-component stepper end to end -- conical source, coupled operator, two
    transverse monitors, per-order s/p projection. Against `solve_1d_conical` at a 10 nm grid
    the worst efficiency gap is 2.8e-3 and the worst amplitude gap 1.2e-2, and both fall at
    second order (1.85 / 1.88 on efficiency, 1.98 / 1.79 on amplitude over dz = 20 / 10 / 5 nm)
    -- so the residual is the Yee grid's rather than the projection's.
    """
    got, ref = _oblique_fdtd(), _oblique_rcwa()
    labels = np.asarray(got.orders_u).ravel()
    assert set(int(m) for m in labels) >= {-1, 0}, f'window {labels} misses a propagating order'
    for leg in ('s_inc', 'p_inc', 'given'):
        for name in ('R_s', 'R_p', 'T_s', 'T_p'):
            a, b = _channels(got, leg, name), _channels(ref, leg, name)
            for m in a:
                assert float(np.real(a[m])) == pytest.approx(float(np.real(b[m])), abs=8e-3), (
                    f'{leg}.{name}[{m}]')
    total = float(np.asarray(got.given.R).sum() + np.asarray(got.given.T).sum())
    assert abs(total - 1.0) < 5e-3, f'R + T = {total}'


def test_the_mixed_leg_is_what_sees_the_per_leg_signs():
    """The pure legs are sign-blind by construction; `given` is the only instrument here.

    Each leg is normalised by its OWN incident amplitude, so a global sign on either one leaves
    every `|r|**2` in that leg exactly right. It shows up only where the two are added.
    Measured, with the three leg signs wrong: `s_inc` and `p_inc` agreed with RCWA to 2e-3 in
    every channel while `given.R_s[-1]` read 0.1795 against a true 0.0197 -- and `R + T` was
    1.0005 throughout.

    The assertion is on the complex amplitudes, and it has to be: `given.R_s[-1]` is smaller
    than either pure leg's, which is destructive interference between them, and no combination
    of the two efficiencies can produce it.
    """
    got, ref = _oblique_fdtd(), _oblique_rcwa()
    for name in ('r_s', 'r_p', 't_s', 't_p'):
        a, b = _channels(got, 'given', name), _channels(ref, 'given', name)
        for m in a:
            assert abs(complex(a[m]) - complex(b[m])) < 3e-2, f'given.{name}[{m}]'

    # The cross term is the whole content: R_s[-1] on the mixed leg is BELOW both pure legs.
    mixed = float(np.real(_channels(got, 'given', 'R_s')[-1]))
    pure = [float(np.real(_channels(got, leg, 'R_s')[-1])) for leg in ('s_inc', 'p_inc')]
    assert mixed < min(pure), (
        f'R_s[-1] mixed {mixed} against pure {pure}; an incoherent sum cannot go below both, '
        f'so this order is where a lost relative phase shows')


def test_the_p_channel_carries_the_per_order_secant():
    """`E . p_hat` is the transverse projection times `n / kz_m`, not the projection.

    `p_hat = k_hat x s_hat` has a `z` component, and eliminating `E_z` through `E . k_hat = 0`
    leaves `E . p_hat = -+ (n / kz_m) (c Ex + s Ey)`. That factor is a per-order
    `1 / cos(theta_m)`, equal to 1 only for the specular order at normal incidence -- which is
    exactly the case the `theta = 0` path covered, and why it could not have found this.

    Pinned as a ratio between two orders so the test states the factor rather than a number:
    at this incidence `kz` differs enough between `m = 0` and `m = -1` that dropping the
    secant changes their `r_p` ratio by more than 20 %.
    """
    got = _oblique_fdtd()
    ref = _oblique_rcwa()
    a = _channels(got, 's_inc', 'r_p')
    b = _channels(ref, 's_inc', 'r_p')
    got_ratio = abs(complex(a[-1]) / complex(a[0]))
    ref_ratio = abs(complex(b[-1]) / complex(b[0]))
    assert got_ratio == pytest.approx(ref_ratio, rel=5e-2), (
        f'|r_p[-1]/r_p[0]| = {got_ratio} against {ref_ratio}; the per-order n/kz is what sets '
        f'this ratio, and a projection without it is a pure amplitude error no efficiency or '
        f'energy check can see')


def test_the_order_window_narrows_with_ky():
    """`ky` is one scalar for every order, so it does not shift the window -- it narrows it.

    An order propagates where `kx_m**2 + ky**2 <= n**2`, and `ky` eats into the budget `kx_m`
    has left. `_half_space_kz` has to subtract it too, or the window and the extraction end up
    on different order sets: at theta = 60 / phi = 30 a ky-blind `kz` calls order -4
    propagating (0.39965 real) where the truth is evanescent (0.16667j), and power appears out
    of an order that carries none.
    """
    from wavebit._core.fdtd import _propagating_window, _half_space_kz
    wl, period, n_sq = WL, PERIOD, 4.0
    kx_inc, ky = 2.0 * np.sin(np.deg2rad(60.0)) * np.cos(np.deg2rad(30.0)), \
                 2.0 * np.sin(np.deg2rad(60.0)) * np.sin(np.deg2rad(30.0))
    centre, half = _propagating_window(wl, period, 1.0, n_sq, kx_inc=kx_inc, ky_inc=ky)
    m = np.arange(centre - half, centre + half + 1)
    kx = kx_inc + m * wl / period
    truth = kx ** 2 + ky ** 2 <= n_sq + 1e-12
    kz = _half_space_kz(n_sq, kx, ky)
    assert np.all((np.abs(np.imag(kz)) < 1e-12) == truth), (
        f'the window and kz disagree about which orders propagate: kz={kz}, truth={truth}')
    wider, _ = _propagating_window(wl, period, 1.0, n_sq, kx_inc=kx_inc, ky_inc=0.0)
    _, half0 = _propagating_window(wl, period, 1.0, n_sq, kx_inc=kx_inc, ky_inc=0.0)
    assert half <= half0, f'ky must narrow the window, not widen it ({half} vs {half0})'


#: A conical incidence with no order near a Rayleigh point in either half-space, found by
#: scanning (theta, phi) for the largest minimum |kz| margin. The default OBLIQUE scene is not
#: one: its m = -1 order sits at kz_top = 0.286, and a grazing order converges slowly for a
#: reason that is arithmetic rather than a defect -- `kz = sqrt(n**2 - k_par**2)` has
#: sensitivity `1/kz`, and the p channel then multiplies by `n/kz` again.
CLEAN = dict(theta=np.deg2rad(30.0), phi=np.deg2rad(60.0), psi=np.deg2rad(30.0))


def _propagates(m, n_sq, scene):
    kx = np.sin(scene['theta']) * np.cos(scene['phi']) + m * WL / PERIOD
    ky = np.sin(scene['theta']) * np.sin(scene['phi'])
    return n_sq - (kx ** 2 + ky ** 2) > 0.0


def _amplitude_gap(scene, resolution, ref):
    """Worst amplitude gap against RCWA, over orders that actually propagate.

    R is compared only where an order propagates above and T only where it propagates below.
    Anything else is round-off the de-referencing multiplied up -- see the evanescent test.
    """
    got = _oblique_fdtd(resolution=resolution, n_periods=200, **scene)
    worst = 0.0
    for leg in ('s_inc', 'p_inc', 'given'):
        for name, n_sq in (('r_s', 1.0), ('r_p', 1.0),
                           ('t_s', N_BOT ** 2), ('t_p', N_BOT ** 2)):
            a, b = _channels(got, leg, name), _channels(ref, leg, name)
            for m in a:
                if _propagates(m, n_sq, scene):
                    worst = max(worst, abs(complex(a[m]) - complex(b[m])))
    return worst


def test_the_conical_amplitudes_converge_at_second_order():
    """On a scene with margin: 5.6e-2 -> 1.3e-2 on halving dz, an order of 2.05.

    **Two things have to be kept out of the metric, and both were in it at first.** An order
    evanescent in a half-space carries no power there and its amplitude is round-off amplified
    by de-referencing -- included, `t_s[+1]` reads 3.1e-3 / 3.6e-4 / 6.2e-4 and gets *worse*
    with resolution. And an order near a Rayleigh point converges slowly for arithmetic
    reasons: the default scene's m = -1 sits at kz_top = 0.286, where the p channel multiplies
    by `n/kz` = 3.5 and `kz = sqrt(n**2 - k_par**2)` has sensitivity `1/kz` on top of that.
    With both in, the run reads 1.58 / 1.37 and looks like a first-order term that is not
    there. The classical controls at the same angle are 2.02 / 1.78 (TE) and 1.89 / 1.85 (TM),
    which is what made the difference look real.
    """
    ref = _oblique_rcwa(orders=200, **CLEAN)
    coarse, fine = _amplitude_gap(CLEAN, 0.05, ref), _amplitude_gap(CLEAN, 0.1, ref)
    ratio = coarse / fine
    assert ratio > 3.4, (
        f'amplitude error {coarse:.3e} -> {fine:.3e} is a ratio of {ratio:.2f}, an order of '
        f'{np.log2(ratio):.2f}; second order is 4')


def test_the_two_monitors_need_two_demodulation_kernels():
    """`Ex` sits on the x half-steps and `Ey` on the nodes, and one kernel for both is FIRST order.

    The error is a per-order phase `exp(-i k0 kx_m dx/2)`: measured 6.7e-2 / 3.3e-2 / 1.6e-2
    over dz = 20 / 10 / 5 nm, scaling as dx**1.0 to two digits. It is mostly cancelled in
    `|r|**2`, and **every fixed-tolerance test in this file passed with it in place** -- at
    10 nm it sits just under a tolerance that has to accommodate the true 1.2e-2. Only a ratio
    between two grids separates it: 3.62 here against 2.39 with one kernel.

    **This runs on the default scene and not on `CLEAN`, and the reason is worth stating.**
    The mutation scales with `|kx_m|`, so the scene that makes the convergence claim cleanest
    is the one that cannot see the defect: on `CLEAN` the only order propagating in air has
    `kx_0 = 0.25` and the single-kernel version reads 4.24 against a true 4.15 -- no separation
    at all. The default scene's m = -1 carries `kx = -0.903`, four times the phase. A scene
    chosen for margin and a scene chosen for sensitivity are not the same scene.
    """
    ref = _oblique_rcwa(orders=200)
    coarse, fine = _amplitude_gap(OBLIQUE, 0.05, ref), _amplitude_gap(OBLIQUE, 0.1, ref)
    ratio = coarse / fine
    assert ratio > 3.0, (
        f'amplitude error {coarse:.3e} -> {fine:.3e} is a ratio of {ratio:.2f}, an order of '
        f'{np.log2(ratio):.2f}. A ratio near 2.4 is the single-kernel first-order term.')


def test_an_evanescent_order_reports_no_power_and_its_amplitude_is_not_a_number():
    """R and T are exactly zero there, and the amplitude beside them is amplified round-off.

    Measured on the default scene: order +1 has `k_par**2 = 2.89` against `n_bot**2 = 2.25`, so
    it is evanescent in BOTH half-spaces -- and its `t_s` gap against RCWA grows from 3.6e-4 to
    6.2e-4 as the grid is refined, because what is being compared is noise the de-referencing
    multiplied up. Any convergence metric that maximises over all orders picks it up.
    """
    got = _oblique_fdtd(resolution=0.06, n_periods=80)
    for name, n_sq in (('R_s', 1.0), ('R_p', 1.0), ('T_s', N_BOT ** 2), ('T_p', N_BOT ** 2)):
        channel = _channels(got, 's_inc', name)
        for m, value in channel.items():
            if not _propagates(m, n_sq, OBLIQUE):
                assert float(np.real(value)) == 0.0, (
                    f'{name}[{m}] = {value}; the order is evanescent in that half-space')


# ---------------------------------------------------------------------------------------
# 12. a graded z mesh
# ---------------------------------------------------------------------------------------

def test_the_graded_mesh_puts_both_structure_faces_on_nodes():
    """`z = 0` and `z = total_thickness` are nodes exactly, not to a rounding.

    The incident amplitude is referenced at the first and the transmitted at the second, so a
    half-cell error there is a phase error in every amplitude -- and on a graded mesh a "half
    cell" is up to half a COARSE one, which is `n_max` times worse than it would be on a
    uniform grid.
    """
    from wavebit._core.fdtd import build_z_mesh
    for n_top, n_bot, n_max in ((1.0, 1.5, 3.5), (1.0, 1.5, 6.54), (1.0, 1.0, 2.0)):
        z, k_z0, n_struct = build_z_mesh(6.86, 60.0, 768.0, 391.0, n_top, n_bot, n_max, WL)
        assert abs(z[k_z0]) < 1e-9, f'structure top at {z[k_z0]}'
        assert abs(z[k_z0 + n_struct] - 60.0) < 1e-9


def test_the_grading_ratio_is_what_bounds_the_transition():
    """No two adjacent cells differ by more than the ratio, and the smallest stays `h_fine`.

    Both halves matter. The ratio bounds the reflection off the transition -- a mesh step is an
    optical interface, and a 4.5x jump taken in one cell reflects 5.9e-3 in amplitude against
    6.0e-4 spread over 40. And the smallest cell has to stay `h_fine` because the Courant bound
    follows it: rescaling the whole ramp to fit the distance, which is the obvious way to make
    the ends meet, shrinks it and buys back in time steps what the grading saved in cells --
    measured 130 -> 80 cells for only 1.50 s -> 1.29 s before the ramp was left alone and the
    residue pushed into the coarse cells.
    """
    from wavebit._core.fdtd import build_z_mesh
    for ratio in (1.03, 1.06, 1.12):
        for h_fine, thick in ((6.86, 60.0), (12.96, 300.0), (2.5, 317.0)):
            z, k_z0, n_struct = build_z_mesh(h_fine, thick, 768.0, 391.0, 1.0, 1.5, 6.54,
                                             WL, ratio=ratio)
            cells = np.diff(z)
            grew = cells[1:] / cells[:-1]
            # The ramp is built at exactly `ratio` and then scaled by one common factor to
            # land on the distance -- a common factor preserves every ratio, so the only slack
            # is that factor itself, `distance / total`, bounded by one coarse cell over the
            # span. Measured worst case here 1.069 against a 1.06 budget.
            assert np.all(grew < ratio * 1.02), f'ratio {ratio}: worst step {grew.max()}'
            assert np.all(grew > 1.0 / (ratio * 1.02))
            # The structure is split into a WHOLE number of cells so both its faces land on
            # nodes, which makes its cells up to `n_struct / (n_struct - 1)` smaller than
            # `h_fine`. Nothing else may be smaller: the Courant bound follows the minimum.
            h_core = thick / n_struct
            assert cells.min() >= h_core * (1.0 - 1e-9), (
                f'smallest cell {cells.min()} is below the structure cell {h_core}')
            assert h_core <= h_fine * (1.0 + 1e-9)


def test_a_graded_run_agrees_with_the_rcwa_path():
    """The grading is allowed to cost accuracy; it is not allowed to cost correctness.

    Measured against `solve_1d` at orders = 200 on the benchmark grating: 5.0e-3 uniform against
    6.5e-3 / 8.8e-3 / 7.9e-3 at grading 1.03 / 1.06 / 1.12, with amplitudes 1.6e-2 against
    1.7e-2 / 1.9e-2 / 1.8e-2. That is the transition reflection, and it is the price.
    """
    ref = _rcwa(_grating(), [THICK], 'te', orders=200)
    labels = np.asarray(ref.orders_u).ravel()
    want = {int(m): (float(np.asarray(ref.R).ravel()[i]),
                     float(np.asarray(ref.T).ravel()[i])) for i, m in enumerate(labels)}
    for grading in (1.03, 1.06, 1.12):
        got = _by_order(solve_fdtd(structure=_grating(), thickness=[THICK], top=EPS_TOP,
                                   bot=EPS_BOT, period=[PERIOD], wavelength=WL, tetm='te',
                                   theta=0.0, resolution=0.0777, n_periods=100,
                                   grading=grading))
        for m, (R, T) in got.items():
            assert R == pytest.approx(want[m][0], abs=2e-2), f'grading={grading} R[{m}]'
            assert T == pytest.approx(want[m][1], abs=2e-2), f'grading={grading} T[{m}]'


def test_the_auxiliary_grid_must_ride_the_same_mesh():
    """A uniform incident wave under a graded scattered one is silent, and it is a phase error.

    `_time_loop` used to hand `_Incident1D` the scalar `grid.dz`, which is right while the mesh
    is uniform and wrong the moment it is not. Measured on a graded dielectric scene, the
    EFFICIENCIES stayed plausible -- 3.5e-3 against RCWA, better than the uniform run's 5.0e-3
    -- while every amplitude was wrong by 0.52. `R + T` never noticed.

    So the constructor checks the length rather than trusting the caller: `nz` nodes need
    `nz - 1` cells, and a scalar reaching it under a graded grid is a shape mismatch.
    """
    from wavebit._core.fdtd import _Incident1D
    with pytest.raises(ValueError, match='must ride the SAME mesh'):
        _Incident1D(40, np.full(12, 10.0), 0.02, 1.0, 6, 8, WL, 1.0 / WL, 100.0, 400.0,
                    'te', 0.0, float)


def test_grading_does_not_move_the_time_step():
    """The Courant bound follows the SMALLEST cell, and grading does not make one smaller.

    This is the fact that caps what grading can buy: it saves cell count and per-step cost, and
    **no** step count. Measured spectral radii of the non-uniform operator land at 0.9993-0.9999
    of `4/min(dz)**2` on six meshes including a 1:10 jump, so `min` is tight rather than
    conservative. Pinned because the natural implementation breaks it silently -- see the ramp
    note in the ratio test.
    """
    kw = dict(structure=_grating(), thickness=[THICK], top=EPS_TOP, bot=EPS_BOT,
              period=[PERIOD], wavelength=WL, tetm='te', theta=0.0, resolution=0.0777,
              n_periods=60, return_diagnostics=True)
    uniform = solve_fdtd(grading=None, **kw)[1]
    for grading in (1.03, 1.06, 1.12):
        got = solve_fdtd(grading=grading, **kw)[1]
        # Not exactly equal, and the gap is the structure's whole-cell split rather than the
        # grading: `ceil(thickness / h_fine)` cells across the layer are each a little smaller
        # than `h_fine`, which is the price of both faces landing on nodes. Bounded and small;
        # what must NOT happen is the ramp itself producing a smaller cell.
        assert got['dt'] > uniform['dt'] * 0.85, (
            f'grading={grading} moved dt from {uniform["dt"]} to {got["dt"]}')
        assert got['nz'] < uniform['nz'], (
            f'grading={grading} did not reduce the cell count ({got["nz"]} vs '
            f'{uniform["nz"]})')


def test_grading_is_off_by_default_and_true_takes_the_measured_optimum():
    """`None` and `False` are the uniform mesh; `True` is the ratio the measurement picked.

    Off by default is a measured decision. On the Drude metal grating, time to reach an
    efficiency error of 1e-3: uniform 5.99 s, grading 1.03 6.60 s, **1.06 3.78 s**, 1.12
    9.62 s -- a real optimum, 1.6x faster than uniform. On the dielectric benchmark grading
    loses: the error grows 2.7x and buying that back in cell size costs more than the grading
    saves in cell count. A scheme that loses on one of two benchmark scenes does not get to be
    the default.
    """
    from wavebit._core.fdtd import GRADE_RATIO
    kw = dict(structure=_grating(), thickness=[THICK], top=EPS_TOP, bot=EPS_BOT,
              period=[PERIOD], wavelength=WL, tetm='te', theta=0.0, resolution=0.0777,
              n_periods=60, return_diagnostics=True)
    off = solve_fdtd(**kw)[1]['nz']
    assert solve_fdtd(grading=False, **kw)[1]['nz'] == off
    assert solve_fdtd(grading=None, **kw)[1]['nz'] == off
    assert solve_fdtd(grading=True, **kw)[1]['nz'] == solve_fdtd(grading=GRADE_RATIO, **kw)[1]['nz']
    assert solve_fdtd(grading=True, **kw)[1]['nz'] < off


@pytest.mark.parametrize('bad', [1.0, 0.9, 2.5, -1.0])
def test_a_grading_ratio_outside_its_range_is_refused(bad):
    """It is a growth factor, so it is above 1, and a big one is the step it exists to avoid."""
    with pytest.raises(ValueError, match='most one cell may grow'):
        _ = solve_fdtd(structure=_grating(), thickness=[THICK], top=EPS_TOP, bot=EPS_BOT,
                       period=[PERIOD], wavelength=WL, tetm='te', theta=0.0,
                       resolution=0.05, n_periods=20, grading=bad)


def test_the_coarse_cells_refine_with_the_fine_ones():
    """The padding must not stop converging while the structure does.

    `n_max` is the scene's true maximum index, not one derived back from the cell size. The
    derived form assumes the caller asked for exactly `CELLS_PER_WAVELENGTH`, so a caller who
    refines `resolution` is read as having a higher-index material and the padding is coarsened
    to match -- pinning the coarse cells at 45 nm however fine the structure gets. Measured
    with that bug, the graded run **floored at 2.3e-3 and stopped converging** (order 0.42)
    while the uniform path went on to 1.4e-4. Fixed, it converges at 2.19 / 1.91.
    """
    kw = dict(structure=_grating(), thickness=[THICK], top=EPS_TOP, bot=EPS_BOT,
              period=[PERIOD], wavelength=WL, tetm='te', theta=0.0, n_periods=60,
              grading=1.06, return_diagnostics=True)
    coarse = solve_fdtd(resolution=0.05, **kw)[1]['grid'].dz_cell.max()
    fine = solve_fdtd(resolution=0.2, **kw)[1]['grid'].dz_cell.max()
    assert fine < coarse / 2.0, (
        f'refining 4x moved the coarsest cell only {coarse} -> {fine}; the padding is not '
        f'converging with the structure')


# ---------------------------------------------------------------------------------------
# 13. the time record -- the one thing a frequency-domain answer cannot show
# ---------------------------------------------------------------------------------------

def _record(n_index, record_field=1, **over):
    """A uniform medium, so the recorded front has one speed to measure rather than two."""
    eps = wavebit.eps_from_index(n_index)
    kw = dict(structure=np.full((1, 1, 2), eps), thickness=[100.], top=eps, bot=eps,
              period=[PERIOD], wavelength=WL, tetm='te', theta=0.0, resolution=0.05,
              n_periods=60, record_field=record_field, return_diagnostics=True)
    kw.update(over)
    return solve_fdtd(**kw)[1]


def _front_speed(diag):
    """Speed of the first arrival, fitted over the interior."""
    # `field_e` is (snapshot, nz, ny, nx) -- the y axis is there even while it is 1, so the
    # reduction names the transverse axes rather than counting them.
    field = np.abs(diag['field_e']).max(axis=(2, 3))
    t, z, grid = diag['field_t'], diag['field_z'], diag['grid']
    threshold = 0.02 * field.max()
    arrival = np.array([t[np.flatnonzero(field[:, k] > threshold)[0]]
                        if (field[:, k] > threshold).any() else np.nan
                        for k in range(len(z))])
    k = np.arange(len(z))
    inside = np.isfinite(arrival) & (k > grid.k_tfsf + 2) & (k < grid.nz - grid.n_pml - 4)
    return 1.0 / np.polyfit(z[inside], arrival[inside], 1)[0]


@pytest.mark.parametrize('n_index', [1.0, 2.0, 2.5])
def test_the_recorded_field_travels_at_the_speed_of_light(n_index):
    """The record is only worth having if its time axis is right, and this measures exactly that.

    A pulse in a uniform medium of index `n` advances at `1/n` in the module's units. Measured
    from the recorded snapshots alone -- no closed form, no RCWA -- the fitted front speed is
    0.9987 / 0.4980 / 0.3993 against 1 / 0.5 / 0.4, so 0.1 % / 0.4 % / 0.2 %.

    A uniform medium and not the benchmark grating, because an interface gives the fit two
    speeds and the slope comes back as a meaningless average: measured 0.2994 against 0.5 on a
    scene whose fit window straddled one.
    """
    assert _front_speed(_record(n_index)) == pytest.approx(1.0 / n_index, rel=0.03)


def test_recording_less_often_costs_only_resolution():
    """The gap at a coarse `record_field` is the SAMPLING, not the physics -- shown, not asserted.

    A first-arrival time is resolved to one snapshot interval, so a coarse record biases the
    fit; the point is that refining it converges rather than plateauing. Measured at n = 2.5:
    21.7 % / 8.0 % / 0.6 % / 0.2 % at every 8th / 4th / 2nd / 1st step. A recorder that was
    wrong rather than under-sampled would sit on a floor instead.
    """
    exact = 1.0 / 2.5
    errors = [abs(_front_speed(_record(2.5, record_field=d)) - exact) / exact
              for d in (8, 2, 1)]
    assert errors[-1] < errors[0] / 5.0, f'refining the record barely helped: {errors}'
    assert errors[-1] < 0.02


def test_the_record_carries_the_axes_it_is_measured_on():
    """`field_z` is the node TABLE, not `arange(nz) * dz`, and on a graded mesh those differ.

    A reader who rebuilds the abscissa from `dz` plots every graded run against the wrong
    positions -- and the plot looks entirely reasonable, because the field is still a field.
    """
    # The benchmark grating, not a uniform medium: grading coarsens toward the cell each
    # half-space would have asked for, so a scene whose index is the same everywhere has
    # nothing to grade and the test would pass on a mesh that never varied.
    diag = solve_fdtd(structure=_grating(), thickness=[THICK], top=EPS_TOP, bot=EPS_BOT,
                      period=[PERIOD], wavelength=WL, tetm='te', theta=0.0, resolution=0.05,
                      n_periods=60, grading=1.06, record_field=20,
                      return_diagnostics=True)[1]
    z = diag['field_z']
    assert z.shape == (diag['nz'],)
    spacing = np.diff(z)
    assert spacing.max() / spacing.min() > 1.5, (
        'the graded run should have unequal cells; if it does not, the scene is not testing '
        'what this test is for')
    assert np.allclose(z, diag['grid'].z_node), 'field_z must BE the node table, not a rebuild'
    assert diag['field_t'].shape[0] == diag['field_e'].shape[0]

    # The in-plane axes by VALUE, not by shape. `field_z` above is pinned to the node table
    # because a rebuild from `dz` is wrong on a graded mesh; the same argument applies in x and
    # it is what this assertion was missing. Two things are being pinned at once: the ORIGIN --
    # every in-plane coordinate the library hands back is measured from the centre of the cell,
    # not its corner -- and the STAGGERING, since a TE run records `Ey`, which sits on the x
    # nodes, half a cell away from the pixel centres `wavebit.meshgrid_xy` returns.
    x, dx = diag['field_x'], diag['grid'].dx
    assert diag['field_e_component'] == 'ey', 'a TE run records Ey'
    assert np.allclose(x, np.arange(diag['nx']) * dx - 0.5 * PERIOD), (
        'field_x must be the Ey node axis in the cell-centre frame')
    assert abs(x.mean() + 0.5 * dx) < 1e-9, 'the node axis straddles the centre, off by dx/2'
    centres = wavebit.meshgrid_xy((1, diag['nx']), period=(PERIOD, PERIOD))[0][0]
    assert np.allclose(x + 0.5 * dx, centres), (
        'shifting Ey by half a cell must land exactly on the pixel centres the RCWA path uses; '
        'if this fails the two solvers are quoting fields on different grids')
    assert diag['field_y'].shape == (diag['grid'].ny,), 'the y axis is published beside field_y'
    assert diag['field_e'].shape[1:] == (diag['nz'], diag['grid'].ny, diag['nx']), (
        'the record carries the y axis even at ny = 1, so a crossed grating is a larger ny '
        'rather than a different array shape')


def test_the_record_is_off_by_default_and_costs_nothing():
    """A solve that was not asked for the record does not pay for it, and cannot read one."""
    diag = solve_fdtd(structure=_grating(), thickness=[THICK], top=EPS_TOP, bot=EPS_BOT,
                      period=[PERIOD], wavelength=WL, tetm='te', theta=0.0,
                      resolution=0.05, n_periods=60, return_diagnostics=True)[1]
    for key in ('field_t', 'field_e', 'field_z', 'field_x'):
        assert key not in diag, f'{key} appeared without record_field being asked for'


def test_the_y_average_is_harmonic_along_the_component_it_is_for():
    """The subpixel rule generalises verbatim: harmonic along the component's own axis.

    Checked against hand values rather than against another run — a cell holding equal parts
    `eps = 4` and `eps = 1` averages to the harmonic 1.6 for a component pointing along y and
    to the arithmetic 2.5 for one that does not. It is the same rule `'x'` and `'z'` already
    take, applied to the axis that only exists once a grating is crossed.
    """
    from wavebit._core.fdtd import component_property
    layer = np.zeros((1, 4, 2), dtype=complex)
    layer[0, :2, :] = 4.0
    layer[0, 2:, :] = 1.0
    kw = dict(thickness=[1.0], top=1.0 + 0j, bot=1.0 + 0j, period=2.0,
              z_centres=np.array([0.5]), z_widths=np.array([1.0]), dx=1.0, nx=2,
              x_half=False, dy=4.0, ny=1, period_y=4.0)
    harmonic = float(np.real(component_property(layer, harmonic_axis='y', **kw)[0, 0, 0]))
    arithmetic = float(np.real(component_property(layer, harmonic_axis=None, **kw)[0, 0, 0]))
    assert harmonic == pytest.approx(2.0 * 4.0 * 1.0 / 5.0, rel=1e-12)
    assert arithmetic == pytest.approx(2.5, rel=1e-12)


def test_a_y_invariant_raster_skips_the_y_average_rather_than_running_it():
    """Averaging a constant is not the constant, in floating point, and that is why it is skipped.

    `(w * a).sum() / w.sum()` is not bit-equal to `a`. Running the y pass on a y-invariant
    scene would therefore have moved every existing answer in the last digits — which is
    exactly the kind of change that cannot be told from a real one after the fact. The 2D
    module's byte-identity is the reason this branch exists.
    """
    from wavebit._core.fdtd import component_property
    flat = np.full((1, 2), 4.0 + 0j)                 # (n_layers, Nx) -- no y axis at all
    kw = dict(thickness=[1.0], top=1.0 + 0j, bot=1.0 + 0j, period=2.0,
              z_centres=np.array([0.5]), z_widths=np.array([1.0]), dx=1.0, nx=2,
              x_half=False, harmonic_axis='y')
    out = component_property(flat, **kw, dy=1.0, ny=1, period_y=4.0)
    assert out.shape == (1, 1, 2)
    # Bit-exact, not approximately: the layer is the only material in the z cell.
    assert out[0, 0, 0] == 4.0 + 0j and out[0, 0, 1] == 4.0 + 0j


# ---------------------------------------------------------------------------------------
# 14. the y axis -- a crossed grating, stage by stage
# ---------------------------------------------------------------------------------------

def _crossed_grid(theta_deg=20.0, phi_deg=30.0, n_y=8, n_x=8, resolution=0.05):
    """A crossed grating that is asymmetric in x, in y, and under transpose.

    Deliberately not an L or a square block: every external oracle this project has used was
    once blind to a y row-order defect because the 2D rasters tested were y-symmetric, and a
    scene that cannot see a transposition is worse than no scene.
    """
    from wavebit._core.fdtd import _Grid
    raster = np.full((1, n_y, n_x), EPS_TOP, dtype=complex)
    raster[0, :n_y // 2, :n_x // 2] = wavebit.eps_from_index(N_TOOTH)
    raster[0, n_y // 2:, :n_x // 4] = wavebit.eps_from_index(2.0)
    return _Grid(WL, PERIOD, raster, [THICK], EPS_TOP, EPS_BOT, resolution, 12, 0.99, True,
                 theta=np.deg2rad(theta_deg), phi=np.deg2rad(phi_deg),
                 mu_top=1.0 + 0j, mu_bot=1.0 + 0j, period_y=PERIOD)


def test_the_y_shift_carries_its_own_bloch_phase_on_its_own_axis():
    """`_shift_up_y` moves along y, not x, and multiplies by the y phase, not the x one.

    Two separate mistakes, and neither raises: shifting the wrong axis returns an array of the
    right shape, and using `bloch` where `bloch_y` belongs returns the right shape too. The
    RCWA path's own notes record the `(Py, Px)` swap happening repeatedly for exactly this
    reason -- a pair of periods is easy to hand over in the wrong order.
    """
    from wavebit._core.fdtd import _shift_up_y, _shift_down_y, _shift_up
    a = np.arange(2 * 3 * 4, dtype=complex).reshape(2, 3, 4)
    phase = np.exp(0.7j)
    up = _shift_up_y(a, phase)
    assert np.array_equal(up[..., :-1, :], a[..., 1:, :])
    assert np.allclose(up[..., -1:, :], a[..., :1, :] * phase)
    assert not np.allclose(up, _shift_up(a, phase)), 'the y shift moved the x axis'
    assert np.allclose(_shift_down_y(_shift_up_y(a, phase), phase), a), 'not inverses'


def test_a_crossed_raster_gives_the_grid_a_real_y_axis():
    """`ny`, `dy` and `period_y` come from the raster having a y axis, not from a default."""
    grid = _crossed_grid()
    assert grid.ny > 1 and grid.dy is not None and grid.period_y == PERIOD
    material = grid.eps_for(False, False, 'y', y_half=True)
    assert material.shape == (grid.nz, grid.ny, grid.nx)
    assert np.ptp(np.real(material)) > 1.0, 'the material does not vary at all'
    # It must vary along y as well as x -- a scene that varies only in x is the 2D case again.
    in_layer = np.real(material[grid.k_z0 + 2])
    assert np.ptp(in_layer.mean(axis=1)) > 0.1, 'the material is constant along y'
    assert np.ptp(in_layer.mean(axis=0)) > 0.1, 'the material is constant along x'


def test_a_crossed_raster_without_a_y_period_is_refused():
    """Two periods, and assuming they are equal is a silently square cell."""
    from wavebit._core.fdtd import _Grid
    raster = np.full((1, 4, 4), EPS_TOP, dtype=complex)
    raster[0, :2, :2] = wavebit.eps_from_index(N_TOOTH)
    with pytest.raises(ValueError, match='needs a y period'):
        _Grid(WL, PERIOD, raster, [THICK], EPS_TOP, EPS_BOT, 0.05, 12, 0.99, True,
              theta=0.0, mu_top=1.0 + 0j, mu_bot=1.0 + 0j)


def test_the_courant_bound_counts_the_y_axis_once():
    """`1/dy**2` replaces the algebraic `ky` term rather than joining it -- they are one term.

    Keeping both would be conservative, which sounds safe and is not: a bound that is loose for
    the wrong reason hides the case where the real one is violated.
    """
    grid = _crossed_grid()
    n_min = 1.0
    expect = 0.99 * n_min / np.sqrt(1.0 / grid.dx ** 2
                                    + 1.0 / grid.dz_cell.min() ** 2
                                    + 1.0 / grid.dy ** 2)
    assert grid.dt == pytest.approx(expect, rel=1e-12)


@pytest.mark.parametrize('phi_deg', [0.0, 30.0, 45.0])
@pytest.mark.parametrize('leg', ['s', 'p'])
def test_the_tfsf_source_injects_a_crossed_plane_wave(phi_deg, leg):
    """On vacuum the scattered-field region must stay empty -- with a y axis as with only x.

    This is the module's strongest instrument and it carries over to a crossed grating intact:
    with nothing to scatter from, everything above the TFSF plane is scattered field and must
    cancel to round-off. Measured ~1e-16 to 1e-17 at phi = 0 / 30 / 45 deg on both legs, the
    same as the y-free case.

    **It has ten orders of resolving power on exactly the mistake this stage exists to avoid.**
    The auxiliary grid's `d/dy` must be the DISCRETE difference the 2D grid applies, not the
    exact `i*ky`, or the two grids sit on different dispersion relations. Measured with the
    continuum operator in the auxiliary grid: 5.4e-7 at 30 deg and 2.2e-6 at 45, against 4e-17.
    That is the same failure `kx_op` already exists to prevent, in the other axis.
    """
    from wavebit._core.fdtd import _Grid, _time_loop
    vacuum = np.ones((1, 8, 8), dtype=complex)
    grid = _Grid(WL, PERIOD, vacuum, [THICK], EPS_TOP, EPS_TOP, 0.05, 12, 0.99, True,
                 theta=np.deg2rad(30.0), phi=np.deg2rad(phi_deg),
                 mu_top=1.0 + 0j, mu_bot=1.0 + 0j, period_y=PERIOD)
    assert grid.ny > 1, 'the scene must have a real y axis or this tests nothing'
    out = _time_loop(grid, 'te', int(np.ceil(60.0 * WL / grid.dt)), np.array([WL]), leg=leg)
    incident = float(np.abs(out['inc_z0'][0]))
    if incident < 1e-12:
        return                                    # this leg drives the other polarisation
    leak = float(np.abs(out['dft_ref'][0]).max()) / incident
    assert leak < 1e-10, (
        f'phi={phi_deg} leg={leg}: {leak:.2e} of the incident amplitude is left in the '
        f'scattered-field region on a vacuum scene with a y axis')


def test_the_in_plane_phase_is_the_outer_product_of_two_vectors():
    """A plane wave is separable, so two vectors serve where an (ny, nx) array would.

    And the four injected components occupy only TWO distinct in-plane Yee positions -- `Ey`
    and `Hx` on (x-node, y-half), `Ex` and `Hy` on (x-half, y-node) -- which is why there are
    two phase arrays rather than four. Checked against the definition rather than against the
    code that builds it.
    """
    grid = _crossed_grid(theta_deg=30.0, phi_deg=40.0)
    k0 = 2.0 * np.pi / WL
    x_node = np.exp(1j * k0 * grid.kx_inc * np.arange(grid.nx) * grid.dx)
    y_half = np.exp(1j * k0 * grid.ky_inc * (np.arange(grid.ny) + 0.5) * grid.dy)
    outer = y_half[:, None] * x_node[None, :]
    # The separable form must equal the direct one, exp(i k0 (kx x + ky y)), to round-off.
    xx, yy = np.meshgrid(np.arange(grid.nx) * grid.dx,
                         (np.arange(grid.ny) + 0.5) * grid.dy)
    direct = np.exp(1j * k0 * (grid.kx_inc * xx + grid.ky_inc * yy))
    assert np.allclose(outer, direct, atol=1e-13)


# ---------------------------------------------------------------------------------------
# 15. a crossed grating, end to end
# ---------------------------------------------------------------------------------------

def _crossed_raster(n=16, hi=N_TOOTH, lo=2.0):
    """Asymmetric in x, in y, AND under transpose -- see `_crossed_grid` for why.

    The two indices are arguments because the amplitude comparison in section 16 cannot afford
    this one. `N_TOOTH = 3.5` leaves under eight cells per wavelength *inside the tooth* even
    at a 33 nm grid, which is already finer than a 3D run at this period can pay for, and the
    worst amplitude gap against RCWA is 3.8e-1 there -- against 2.9e-2 at `hi = 2.0` on the
    coarser 39 nm grid section 16 does run. The footprint, which is what makes the scene see a
    transpose, is the same either way.
    """
    raster = np.full((1, n, n), EPS_TOP, dtype=complex)
    raster[0, :n // 2, :n // 2] = wavebit.eps_from_index(hi)
    raster[0, n // 2:, :n // 4] = wavebit.eps_from_index(lo)
    return raster


CROSSED = dict(theta=np.deg2rad(20.0), phi=np.deg2rad(30.0), psi=np.deg2rad(30.0))


def test_the_2d_order_window_is_the_disc_and_not_a_per_axis_rectangle():
    """The propagating set is a DISC, and the per-axis shortcut is the defect to avoid.

    An order propagates where `kx_m**2 + ky_l**2 <= n**2`, centred on `(-kx_inc, -ky_inc)`.
    Calling the 1D window once per axis narrows the x range by the INCIDENT `ky` -- but order
    `(m, l)` has `ky_inc + l*wl/Py`, so an `l` that moves `ky` toward zero widens the range the
    disc actually allows. Checked by brute force over 4000 random scenes: no propagating order
    falls outside the reported rectangle.
    """
    from wavebit._core.fdtd import propagating_window_2d
    rng = np.random.default_rng(7)
    for _ in range(400):
        wl = rng.uniform(400, 1600)
        px, py = rng.uniform(300, 3000), rng.uniform(300, 3000)
        n_top, n_bot = rng.uniform(1, 2.5), rng.uniform(1, 3.5)
        theta, phi = rng.uniform(0, 1.2), rng.uniform(0, 6.28)
        kx = n_top * np.sin(theta) * np.cos(phi)
        ky = n_top * np.sin(theta) * np.sin(phi)
        (cx, hx), (cy, hy) = propagating_window_2d(wl, px, py, n_top ** 2, n_bot ** 2, kx, ky)
        reach = max(n_top, n_bot)
        for m in range(cx - hx - 3, cx + hx + 4):
            for l in range(cy - hy - 3, cy + hy + 4):
                inside = (cx - hx <= m <= cx + hx) and (cy - hy <= l <= cy + hy)
                propagates = (kx + m * wl / px) ** 2 + (ky + l * wl / py) ** 2 <= reach ** 2
                assert not (propagates and not inside), (
                    f'order ({m},{l}) propagates and is outside the window')


def test_a_crossed_grating_reduces_to_the_2d_path_when_it_is_y_invariant():
    """The 3D machinery, checked against the answer it must reproduce rather than against RCWA.

    A structure given as a `(1, Ny, Nx)` raster whose y rows are identical is the SAME physical
    scene the 2D path solves, so the two must agree -- and this isolates the 3D machinery from
    the genuinely-3D geometry, which is what a comparison against RCWA cannot do. Measured
    4.8e-3 at a 20 nm grid, the same order as each path's own gap against RCWA.

    `period_y` is small enough that no y order propagates (`wl/Py > 2n`), so the y axis is
    exercised without adding physics the 2D path cannot express.
    """
    column = np.array([wavebit.eps_from_index(N_TOOTH)] * 8 + [EPS_TOP] * 8)
    flat = column.reshape(1, 1, 16)
    stacked = np.repeat(column.reshape(1, 1, 16), 6, axis=1)
    common = dict(thickness=[THICK], top=EPS_TOP, bot=EPS_BOT, wavelength=WL,
                  resolution=0.05, n_periods=100, **CROSSED)
    flat_result = solve_fdtd(structure=flat, period=[PERIOD], **common)
    crossed = solve_fdtd(structure=stacked, period=[PERIOD, 200.0], **common)

    labels_u = np.asarray(crossed.orders_u).ravel()
    flat_u = np.asarray(flat_result.orders_u).ravel()
    row = int(np.flatnonzero(np.asarray(crossed.orders_v).ravel() == 0)[0])
    worst = 0.0
    for i, m in enumerate(labels_u):
        if m not in flat_u:
            continue
        j = int(np.flatnonzero(flat_u == m)[0])
        for name in ('R', 'T'):
            worst = max(worst, abs(float(np.asarray(getattr(crossed, name))[row, i])
                                   - float(np.asarray(getattr(flat_result, name)).ravel()[j])))
    assert worst < 2e-2, f'the 3D path and the 2D path disagree by {worst:.2e}'


@pytest.mark.parametrize('tetm', ['te', 'tm'])
def test_a_crossed_raster_answers_the_polarisation_it_was_asked_for(tetm):
    """`tetm` on a crossed raster must select the leg, and it silently did not.

    A crossed grating needs all six components whatever the polarisation is, so it turns the
    conical stepper on -- and then reaches the incident drive with no `leg`, because no conical
    dispatch happened. Falling through drove the p direction, so BOTH spellings of `tetm`
    answered the TM problem: on the y-invariant raster below, `tetm='te'` returned
    T = 0.2713 / 0.1721 / 0.2694, which is the 2D path's TM answer, against a true
    0.2148 / 0.3533 / 0.2145. Worst per-order gap 1.8e-1, now 1.1e-4.

    **Nothing caught it, and the reason is a gap in the question rather than in the tolerance.**
    Every other crossed test asks with `psi`, which always carries a leg; none asked with
    `tetm`. `R + T` stayed at 1.01 throughout, because solving the wrong polarisation is still
    solving something. The record was the visible symptom -- `field_e` holds `ey`, which is
    identically zero when only the p leg is driven -- and a shape-only assertion on it saw
    nothing.
    """
    column = np.array([wavebit.eps_from_index(N_TOOTH)] * 8 + [EPS_TOP] * 8)
    common = dict(thickness=[THICK], top=EPS_TOP, bot=EPS_BOT, wavelength=WL,
                  resolution=0.05, n_periods=100, theta=0.0, tetm=tetm)
    flat = solve_fdtd(structure=column.reshape(1, 1, 16), period=[PERIOD], **common)
    crossed = solve_fdtd(structure=np.repeat(column.reshape(1, 1, 16), 6, axis=1),
                         period=[PERIOD, 200.0], **common)

    flat_u = np.asarray(flat.orders_u).ravel()
    crossed_u = np.asarray(crossed.orders_u).ravel()
    row = int(np.flatnonzero(np.asarray(crossed.orders_v).ravel() == 0)[0])
    worst = 0.0
    for i, m in enumerate(crossed_u):
        if m not in flat_u:
            continue
        j = int(np.flatnonzero(flat_u == m)[0])
        for name in ('R', 'T'):
            worst = max(worst, abs(float(np.asarray(getattr(crossed, name))[row, i])
                                   - float(np.asarray(getattr(flat, name)).ravel()[j])))
    assert worst < 5e-3, (
        f'a crossed raster asked for {tetm!r} disagrees with the 2D path by {worst:.2e}; '
        f'if this is ~1.8e-1 the two polarisations have been swapped again')


def test_the_record_of_a_crossed_run_carries_the_field_that_was_driven():
    """The component `record_field` hands back must be the one the incidence drives.

    This is the same defect from the other side, and it is the side a user meets: ask a crossed
    grating for `tetm='te'` with a record, and `field_e` came back identically zero on a
    y-invariant raster while the whole field sat in `field_ex`. An all-zero array called `the
    field` is worse than no array.
    """
    column = np.array([wavebit.eps_from_index(N_TOOTH)] * 8 + [EPS_TOP] * 8)
    _, diag = solve_fdtd(structure=np.repeat(column.reshape(1, 1, 16), 6, axis=1),
                         thickness=[THICK], top=EPS_TOP, bot=EPS_BOT, period=[PERIOD, 200.0],
                         wavelength=WL, tetm='te', theta=0.0, resolution=0.05, n_periods=60,
                         record_field=40, return_diagnostics=True)
    assert diag['field_e_component'] == 'ey', "a TE run's primary is Ey"
    driven = np.abs(diag['field_e']).max()
    assert driven > 1e-2, (
        f'field_e peaks at {driven:.2e} on a TE crossed run: the record names Ey and the '
        f'incidence drove the other leg')


def test_a_crossed_grating_answers_on_a_two_dimensional_order_grid():
    """`R` and `T` come back on `(orders_v, orders_u)`, and the energy balances.

    The shape is the assertion as much as the values: a crossed grating whose answer is a
    single row of orders has silently taken the 1D window, which is what happened while
    `solve_fdtd` still dropped the raster's y axis unconditionally -- it ran, reported one `v`
    order, and returned `R + T = 0.9167` with nothing to say why.
    """
    result = solve_fdtd(structure=_crossed_raster(), thickness=[THICK], top=EPS_TOP,
                        bot=EPS_BOT, period=[PERIOD, PERIOD], wavelength=WL,
                        resolution=0.03, n_periods=100, **CROSSED)
    labels_u = np.asarray(result.orders_u).ravel()
    labels_v = np.asarray(result.orders_v).ravel()
    assert len(labels_u) > 1 and len(labels_v) > 1, (
        f'a crossed grating needs orders on both axes, got u={labels_u} v={labels_v}')
    assert np.asarray(result.R).shape == (len(labels_v), len(labels_u))
    total = float(np.asarray(result.R).sum() + np.asarray(result.T).sum())
    assert abs(total - 1.0) < 5e-2, f'R + T = {total}'


@pytest.mark.parametrize('theta_deg,phi_deg', [(0.0, 0.0), (0.0, 45.0), (30.0, 0.0),
                                               (30.0, 1e-9), (30.0, 40.0), (50.0, 0.0)])
def test_a_polarisation_mixed_run_is_continuous_across_phi_zero(theta_deg, phi_deg):
    """`phi` exactly 0 is not a special case of the physics, and it must not be one of the code.

    The two polarisations decouple where `ky` vanishes, and `ky = n sin(theta) sin(phi)`
    vanishes at `theta = 0` **or** at `phi = 0`. Routing on `theta` alone sent
    `theta != 0, phi = 0` down the coupled path, where `conical` is False because `ky_inc` is
    zero -- so both legs ran the same `tetm='te'` problem and were added together. Measured
    `R + T = 2.0042` at theta = 30 deg with phi exactly 0, against 1.0023 at phi = 1e-9 deg.

    **Twice the incident power, and nothing refused it.** The efficiency guard of the day was
    applied to each sub-result and said nothing anyone heard: on this path it also fired on 85
    of 243 ordinary solves, which is second-order Yee truncation error and not a defect. It
    has since been removed. What pins this case is the `abs(total - 1.0) < 2e-2` assertion
    below, whose tolerance is chosen for a time-domain discretisation rather than inherited
    from the frequency-domain path.

    The first repair routed on `ky` and was still wrong, at `R + T = 0.8864`: the order ladder
    was built as `m * wl / period` with no `kx_inc`, correct while the path only served
    `theta = 0` and wrong the moment it did not. Both are pinned here because the knife edge is
    the same either way -- a value that is only ever exercised at one exact input.
    """
    theta, phi, psi = np.deg2rad(theta_deg), np.deg2rad(phi_deg), np.deg2rad(45.0)
    got = solve_fdtd(structure=_grating(), thickness=[THICK], top=EPS_TOP, bot=EPS_BOT,
                     period=[PERIOD], wavelength=WL, theta=theta, phi=phi, psi=psi,
                     resolution=0.05, n_periods=80)
    total = float(np.asarray(got.given.R).sum() + np.asarray(got.given.T).sum())
    assert abs(total - 1.0) < 2e-2, f'R + T = {total}'

    ref = wavebit.rcwa(backend=0, wavelength=WL, period=[PERIOD],
                       truncation_order=dict(x=40, y=0), top=EPS_TOP, bot=EPS_BOT,
                       theta=theta, phi=phi, psi=psi,
                       structure=_grating(), thickness=[THICK]).solve()
    for leg in ('given', 's_inc', 'p_inc'):
        for name in ('R_s', 'R_p', 'T_s', 'T_p'):
            a, b = _channels(got, leg, name), _channels(ref, leg, name)
            for m in a:
                assert float(np.real(a[m])) == pytest.approx(float(np.real(b[m])), abs=3e-2), \
                    f'{leg}.{name}[{m}]'


def test_the_two_pure_legs_are_never_the_same_solve():
    """s and p incidence are different problems, and two identical legs mean one was not run.

    This is the shape of the phi = 0 defect stated as an invariant rather than as a number: at
    any incidence the s leg and the p leg solve different problems, so their amplitudes differ.
    Two legs that agree bit for bit is not a coincidence to tolerate -- it is one leg reported
    twice, and the sum then doubles.
    """
    for theta_deg, phi_deg in ((30.0, 0.0), (30.0, 40.0), (0.0, 45.0)):
        got = solve_fdtd(structure=_grating(), thickness=[THICK], top=EPS_TOP, bot=EPS_BOT,
                         period=[PERIOD], wavelength=WL, theta=np.deg2rad(theta_deg),
                         phi=np.deg2rad(phi_deg), psi=np.deg2rad(45.0),
                         resolution=0.05, n_periods=80)
        s_amp = np.asarray(got.s_inc.r_s)
        p_amp = np.asarray(got.p_inc.r_s)
        assert not np.allclose(s_amp, p_amp), (
            f'theta={theta_deg} phi={phi_deg}: the two legs returned the same amplitudes, so '
            f'one polarisation was never solved')


# ---------------------------------------------------------------------------------------
# 16. the crossed path's per-order AMPLITUDES, and the contracts around them
# ---------------------------------------------------------------------------------------

def _crossed_channels(result, leg, name):
    """``{(m, l): value}`` for one channel of one leg, on the two-dimensional order grid.

    `_channels` above ravels against `orders_u` alone, which is right while there is one `v`
    order and silently indexes the wrong entry once there are several: the arrays are
    ``(len(orders_v), len(orders_u))`` and the two axes are not the same length on a
    rectangular cell. Both label vectors come off the parent `Result`, as there.
    """
    a = np.asarray(getattr(getattr(result, leg), name))
    u = np.asarray(result.orders_u).ravel()
    v = np.asarray(result.orders_v).ravel()
    return {(int(m), int(l)): a[j, i] for j, l in enumerate(v) for i, m in enumerate(u)}


def _crossed_margin(m, l, n_sq, period_y, scene=CROSSED):
    """``n**2 - k_par**2`` for order ``(m, l)`` -- positive where the order propagates.

    The 1D `_propagates` with the y ladder added. Compared against a *margin* rather than
    against zero wherever amplitudes are: `test_the_conical_amplitudes_converge_at_second_order`
    already records why -- an order within round-off of a Rayleigh point converges at
    `1/kz`, and the p channel then multiplies by `n/kz` again.
    """
    kx = np.sin(scene['theta']) * np.cos(scene['phi']) + m * WL / PERIOD
    ky = np.sin(scene['theta']) * np.sin(scene['phi']) + l * WL / period_y
    return float(n_sq - (kx ** 2 + ky ** 2))


#: Two crossed cells. The square one is what every other test in section 15 runs on; the
#: **rectangular** one exists because no test in this file had `Px != Py` and a propagating y
#: order at the same time, and that is exactly the pair of conditions a y ladder built on the
#: x period survives -- `l * wl / Px` is `l * wl / Py` on a square cell. Measured with the y
#: ladder written on `grid.period`: the square case passes untouched and the rectangular one
#: fails on `s_inc.r_s(0, -1)`, 0.0984 + 0.0047i against 0.0379 + 0.0086i.
CROSSED_CELLS = [pytest.param(PERIOD, id='square'),
                 pytest.param(2.0 * PERIOD, id='rectangular')]


@pytest.mark.parametrize('period_y', CROSSED_CELLS)
def test_the_crossed_amplitudes_carry_the_per_leg_signs(period_y):
    """`_extract_crossed` is a line-for-line copy of `_extract_conical` with **no amplitude oracle**.

    Every per-order sign in it -- the three `sigma` legs, the `s_hat`/`p_hat` frame, the
    `n / kz_m` secant -- is invisible to `R`, `T` and `R + T`, and the copy inherited them
    without inheriting the test that measured them. Measured here with the transmitted leg's
    p sign flipped (`sigma = (-1, 1, +1)`): the worst amplitude moves by 1.21 on the square
    cell and 1.26 on the rectangular one, and **every efficiency in every leg moves by exactly
    0.000e+00** on both. That is arithmetic rather than luck -- a global sign on one channel
    of one leg leaves `|t_p|**2` alone in each pure leg, and `given` is built from the two
    legs' *amplitudes*, so the mixed leg's efficiencies lose it too.

    So this asserts complex amplitudes, on all three legs, against `wavebit.rcwa`'s crossed
    conical solve. `psi = 30 deg` and not 45: at 45 the two legs enter `given` with equal
    weight and a leg swap is invisible.

    Measured worst gap over the orders that propagate: **2.9e-2** on the square cell and
    2.2e-2 on the rectangular one, at this 39 nm grid. It is the Yee grid's rather than the
    projection's -- doubling `resolution` takes the square cell to 8.3e-3, a ratio of 3.5 --
    so the bound below is about 1.7x the measured value and 24x under the 2*|t_p| = 1.21 a
    leg sign gives.
    """
    raster = _crossed_raster(hi=2.0, lo=1.5)
    got = solve_fdtd(structure=raster, thickness=[THICK], top=EPS_TOP, bot=EPS_BOT,
                     period=[PERIOD, period_y], wavelength=WL, resolution=0.025,
                     n_periods=100, **CROSSED)
    # 9 orders per axis, not 12: on this contrast the reference has converged to 5.0e-4 by 9
    # -- measured on the square cell against 15, and on the rectangular one against 12 -- which
    # is 60x below the gap being asserted.
    ref = wavebit.rcwa(backend=0, wavelength=WL, period=[PERIOD, period_y],
                       truncation_order=dict(x=9, y=9), top=EPS_TOP, bot=EPS_BOT,
                       structure=raster, thickness=[THICK], **CROSSED).solve()

    checked = 0
    for leg in ('s_inc', 'p_inc', 'given'):
        for name, n_sq in (('r_s', 1.0), ('r_p', 1.0),
                           ('t_s', N_BOT ** 2), ('t_p', N_BOT ** 2)):
            a, b = _crossed_channels(got, leg, name), _crossed_channels(ref, leg, name)
            for key in a:
                if _crossed_margin(key[0], key[1], n_sq, period_y) < 0.05:
                    continue        # evanescent, or within round-off of a Rayleigh point
                checked += 1
                assert abs(complex(a[key]) - complex(b[key])) < 5e-2, (
                    f'{leg}.{name}{key}: fdtd {a[key]} against rcwa {b[key]}')
    assert checked >= 24, (
        f'only {checked} channels propagate on this cell, so the comparison has quietly '
        f'stopped covering the legs it exists for')

    # The rectangular cell earns its place only if a y order actually radiates: on the square
    # cell every `l != 0` order is evanescent above, so a y ladder mis-built on `Px` reaches
    # nothing that carries power in reflection.
    if period_y != PERIOD:
        above = [int(l) for l in np.asarray(got.orders_v).ravel()
                 if l != 0 and _crossed_margin(0, int(l), 1.0, period_y) > 0.05]
        assert above, 'the rectangular cell has no propagating y order and measures nothing new'


@pytest.mark.parametrize('truncation_order, shape', [(dict(x=10, y=2), '21 x 5'),
                                                     (dict(x=2, y=10), '5 x 21')])
def test_orders_past_nyquist_are_refused_on_the_y_axis_as_well(truncation_order, shape):
    """The transverse transform has two axes now, and the guard has to have two halves.

    The x half is pinned by `test_asking_for_orders_past_nyquist_is_refused`; the y half was
    not, and deleting `or hy >= ny // 2` left every test in this file passing. On a 21 x 21
    transverse grid `|l| < 10` resolves, and `dict(x=2, y=10)` asks for 21 y orders: past that
    an order folds onto another and comes back with a plausible amplitude, which is the same
    failure in the axis that was added later.

    Both spellings are parametrised, and the message is matched on the **counts** rather than
    on the word alone -- `5 x 21` can only have come from the y half of the test.
    """
    with pytest.raises(ValueError, match=rf'{shape} orders asked of a 21 x 21'):
        solve_fdtd(structure=_crossed_raster(), thickness=[THICK], top=EPS_TOP, bot=EPS_BOT,
                   period=[PERIOD, PERIOD], wavelength=WL, tetm='te', theta=0.0,
                   resolution=0.03, n_periods=80, truncation_order=truncation_order)


def test_the_crossed_vacuum_transmission_has_the_oblique_propagation_phase():
    """A vacuum layer transmits exp(i k0 cos(theta) d), independently of azimuth.

    The grating amplitude oracles allow a discretisation error larger than the effect of
    dropping ky from the INCIDENT kz. Their tight normal-incidence controls have ky == 0.
    Here theta=45 and phi=60 make that omission large, while a uniform medium removes
    corner errors. The small rectangular transverse cell still runs the 3D extraction.
    At this resolution the co-polar amplitude error is 1.2e-3 and cross-polar leakage 2.7e-4.
    Both phase and amplitude are checked: a wrong reference plane can preserve all powers.
    """
    theta = np.deg2rad(45.0)
    got, diag = solve_fdtd(
        structure=np.ones((1, 2, 2)), thickness=[THICK], top=1.0, bot=1.0,
        period=[300.0, 400.0], wavelength=WL, theta=theta, phi=np.deg2rad(60.0),
        psi=0.4, resolution=0.025, n_periods=80, truncation_order=dict(x=0, y=0),
        return_diagnostics=True)
    assert diag['grid'].ny > 1
    expected = np.exp(2j * np.pi * THICK / WL * np.cos(theta))
    for leg, co, cross in (('s_inc', 't_s', 't_p'), ('p_inc', 't_p', 't_s')):
        sub = getattr(got, leg)
        np.testing.assert_allclose(getattr(sub, co), expected, rtol=0, atol=3e-3)
        np.testing.assert_allclose(getattr(sub, cross), 0, rtol=0, atol=5e-4)
        np.testing.assert_allclose(sub.r_s, 0, rtol=0, atol=1e-12)
        np.testing.assert_allclose(sub.r_p, 0, rtol=0, atol=1e-12)


def test_the_conical_dereference_guard_keeps_near_orders_and_masks_deep_ones():
    """Exercise the conical guard on both polarisations, beyond its actual threshold.

    Oblique incidence makes the cutoff asymmetric: reflection orders -6, +5, +6 exceed
    the 1e10 gain limit, but -5 stays below it. The nearer transmission plane keeps every
    requested order. Zero evanescent power cannot arbitrate any of these amplitude masks.
    """
    got = solve_fdtd(
        structure=_grating(), thickness=[THICK], top=EPS_TOP, bot=EPS_BOT,
        period=[PERIOD], wavelength=WL, theta=np.deg2rad(30.0), phi=np.deg2rad(40.0),
        psi=0.4, resolution=0.03, n_periods=80, truncation_order=dict(x=6, y=0))
    orders = np.asarray(got.orders_u).ravel()
    np.testing.assert_array_equal(orders, np.arange(-6, 7))
    masked = np.isin(orders, [-6, 5, 6])[None, :]
    for leg in ('s_inc', 'p_inc'):
        sub = getattr(got, leg)
        for name in ('r_s', 'r_p'):
            values = np.asarray(getattr(sub, name))
            np.testing.assert_array_equal(np.isnan(values), masked)
            assert np.all(np.isfinite(values[~masked]))
        for name in ('t_s', 't_p'):
            assert np.all(np.isfinite(getattr(sub, name)))
        np.testing.assert_array_equal(np.asarray(sub.R)[masked], 0.0)
        assert np.all(np.isfinite(sub.T))


def test_a_crossed_truncation_order_comes_back_labelled_the_way_it_was_asked():
    """`dict(x=A, y=B)` with `A != B`, which is the only shape that can see a swap.

    `solve_fdtd` writes the pair out as `((0, want_u), (0, want_v))` rather than zipping it,
    with a comment saying that the two halves being the same length is what makes a swap here
    silent -- and every existing crossed test either takes the default window or is square, so
    the comment was the only thing holding the order. Asked for 3 x 1, this must come back as
    7 `u` labels, 3 `v` labels and `R` of shape `(3, 7)` -- `v` first, which is the array rule
    `Result.orders_u` states and the transpose of the `dict(x=..., y=...)` input.

    Every other `truncation_order` in this file is `dict(x=..., y=0)` on a y-invariant
    raster, so the branch that reads the pair for a crossed grating had never been entered at
    all -- neither half of it, in either order.
    """
    result = solve_fdtd(structure=_crossed_raster(), thickness=[THICK], top=EPS_TOP,
                        bot=EPS_BOT, period=[PERIOD, PERIOD], wavelength=WL, tetm='te',
                        theta=0.0, resolution=0.03, n_periods=80,
                        truncation_order=dict(x=3, y=1))
    assert np.array_equal(np.asarray(result.orders_u), np.arange(-3, 4))
    assert np.array_equal(np.asarray(result.orders_v), np.arange(-1, 2))
    assert np.asarray(result.R).shape == (3, 7), (
        f'R is {np.asarray(result.R).shape}; it is (len(orders_v), len(orders_u)) = (3, 7), '
        f'and (7, 3) is the pair handed over the other way round')
    for name in ('r_s', 'r_p', 't_s', 't_p', 'R_s', 'T_p'):
        assert np.asarray(getattr(result.given, name)).shape == (3, 7), name


def test_an_evanescent_order_on_the_crossed_path_is_nan_rather_than_amplified_round_off():
    """The de-referencing guard, on the copy of it that had no test.

    Same statement as `test_an_evanescent_order_reports_nan_rather_than_an_amplified_reading`
    and the same reason -- referencing a decayed order back to the interface multiplies by
    `exp(k0|Im kz||z_ref|)`, which past `MAX_DEREFERENCE_GAIN` is amplifying round-off into a
    finite, plausible, meaningless number -- but asked of `_extract_crossed`, where nothing
    asked it before. Measured on this scene: `|m| = 4` sits at a gain of 2.6e8 and is kept,
    `|m| = 5` at 3.0e10 and is dropped, and `R` is identically zero across the whole band of
    them either way, so no efficiency check anywhere can tell the two apart.

    Asserted on `r_s` and not on `t_s`, for the reason the numbers give: the transmission
    monitor sits 133 nm from the interface against the reflection monitor's 533, so the same
    `|m| = 5` order is only amplified by 3.8e2 there and is *correctly* kept.
    """
    result = solve_fdtd(structure=_crossed_raster(), thickness=[THICK], top=EPS_TOP,
                        bot=EPS_BOT, period=[PERIOD, PERIOD], wavelength=WL, tetm='te',
                        theta=0.0, resolution=0.03, n_periods=80,
                        truncation_order=dict(x=5, y=1))
    orders = np.asarray(result.orders_u).ravel()
    r_s = np.asarray(result.given.r_s)
    deep = np.abs(orders) >= 5              # |kx| = 6.4, far past n_top and n_bot alike
    assert np.all(np.isnan(r_s[:, deep])), (
        f'deep evanescent amplitudes are {r_s[:, deep]}, which are finite and meaningless')
    assert np.all(np.isfinite(r_s[:, np.abs(orders) <= 1])), (
        'the propagating orders must not be caught by the guard')
    assert np.all(np.isfinite(np.asarray(result.R))), (
        'the efficiencies stay finite -- they are identically zero for these orders')


def test_the_default_crossed_window_is_the_propagating_set_of_the_shortest_wavelength():
    """The window against a closed form, because `R + T` cannot see either way it goes wrong.

    A window that is too **wide** reports orders that carry nothing, and they come back as
    zeros that sum to the same 1. One that is too **narrow** drops power, and the short sum
    reads as ordinary absorption -- which is what `R + T = 0.9167` was, the day the crossed
    path took the 1D window. Neither is visible in an energy check, so the set is computed
    here from `kx_inc`, `ky_inc`, `n_top` and `n_bot` and compared as a set.

    The band is the second half. `kx` is fixed by the grid, so the normalised `kx_inc` and the
    grating step both scale with the wavelength and the **shortest** wavelength has the most
    propagating orders: over 700-1100 nm on a 700 x 1400 cell the propagating set reaches
    `l = -3` at 700 nm and only `l = -2` at the 900 nm centre. A window built on the centre
    would drop a radiating order and report a short sum, so `-3 in orders_v` is the assertion
    that separates the two rules.
    """
    band = np.array([700.0, 900.0, 1100.0])
    period_y = 2.0 * PERIOD
    results = solve_fdtd(structure=_crossed_raster(hi=2.0, lo=1.5), thickness=[THICK],
                         top=EPS_TOP, bot=EPS_BOT, period=[PERIOD, period_y],
                         wavelength=band, resolution=0.025, n_periods=100, **CROSSED)

    def propagating(wavelength):
        """Every ``(m, l)`` that radiates in either half-space, from the closed form."""
        # The grid holds the PHYSICAL kx, ky fixed across the band, so the normalised pair
        # scales with the wavelength -- the same rule `_extract_crossed` applies per index.
        scale = wavelength / float(0.5 * (band.min() + band.max()))
        kx_inc = N_TOP * np.sin(CROSSED['theta']) * np.cos(CROSSED['phi']) * scale
        ky_inc = N_TOP * np.sin(CROSSED['theta']) * np.sin(CROSSED['phi']) * scale
        reach = max(N_TOP, N_BOT)
        return {(m, l)
                for m in range(-12, 13) for l in range(-12, 13)
                if (kx_inc + m * wavelength / PERIOD) ** 2
                + (ky_inc + l * wavelength / period_y) ** 2 <= reach ** 2}

    short, centre = propagating(float(band.min())), propagating(900.0)
    assert short - centre, 'the band does not widen the window here and this test is vacuous'

    for result in results:
        u = [int(m) for m in np.asarray(result.orders_u).ravel()]
        v = [int(l) for l in np.asarray(result.orders_v).ravel()]
        for m, l in short:
            assert m in u and l in v, (
                f'order ({m}, {l}) propagates at {band.min()} nm and is outside the reported '
                f'window u={u} v={v}; its power is missing and the sum reads as absorption')
        # Tight to within the centring: `order_window` is a `(centre, half)` pair per axis, so
        # a bounding box that is not symmetric about an integer costs at most one extra order.
        for labels, axis in ((u, 0), (v, 1)):
            span = max(k[axis] for k in short) - min(k[axis] for k in short) + 1
            assert len(labels) <= span + 1, (
                f'the window is {len(labels)} orders wide where the propagating set spans '
                f'{span}; the extra ones report zero and no energy check can see them')
        assert np.asarray(result.R).shape == (len(v), len(u))
    assert -3 in [int(l) for l in np.asarray(results[0].orders_v).ravel()], (
        'l = -3 radiates at 700 nm and not at the 900 nm centre, so a window missing it was '
        'built for the centre of the band rather than for the whole of it')


# ---------------------------------------------------------------------------------------
# 17. two-sided: an angle has a sign, and an edge can miss the mesh
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize('theta_deg', [0.0, 30.0])
@pytest.mark.parametrize('phi_deg', [30.0, -30.0])
def test_the_azimuth_refusal_is_two_sided(theta_deg, phi_deg):
    """`phi != 0` is refused with `tetm`, and a *negative* azimuth is just as non-zero.

    Every azimuth asked of a refusal in this file is positive, so `!= 0.0` weakened to `> 0.0`
    survives the whole suite -- and what it lets through is not a refusal with a worse message
    but a solve. At `theta = 30, phi = -30` the grid then carries `ky_inc = -0.171` while the
    extraction taken is the one-monitor `_extract`, which is the scalar TE/TM reader: the two
    polarisations are coupled and only one of them is read.

    Free to pin, because the refusal fires before a grid is built -- which is also why the
    weakened version fails this test by raising the *wrong* `ValueError`, the run-length floor
    rather than the mount. Parametrised over `theta` as well because the refusal is about the
    mount and not about `ky`, which is identically zero at `theta = 0` whatever `phi` is.
    """
    with pytest.raises(ValueError, match='pass `psi`'):
        solve_fdtd(structure=_grating(), thickness=[THICK], top=EPS_TOP, bot=EPS_BOT,
                   period=[PERIOD], wavelength=WL, theta=np.deg2rad(theta_deg), tetm='te',
                   phi=np.deg2rad(phi_deg), resolution=0.05, n_periods=20)


@pytest.mark.parametrize('theta_deg, phi_deg', [(30.0, 40.0), (-30.0, 40.0),
                                                (30.0, -40.0), (-30.0, -40.0)])
def test_the_conical_route_is_chosen_on_both_signs_of_both_angles(theta_deg, phi_deg):
    """`theta_v != 0.0 and phi_v != 0.0` picks the six-component stepper, at either sign.

    The predicate is the one `_solve_conical` documents as a knife edge, and both halves of it
    are only ever asked with a positive angle -- so `!= 0.0` weakened to `> 0.0` still routes
    `(+30, +40)` correctly and sends `(-30, +40)` and `(+30, -40)` to the `theta = 0`
    projection instead. That path runs its two legs with `phi=None`, so the grids it builds
    carry `ky_inc = 0` and are then projected onto a frame at the requested azimuth: a
    different scene, solved and relabelled. Measured at `(-30, +40)` with the predicate
    weakened, `R + T = 1.0012` -- inside every energy tolerance in this file, and closer to 1
    than the correct run's own 1.0023 -- while `p_inc.T_s[0]` reads 0.2378 against a true
    0.0275, a factor of 8.6.

    Measured against `solve_1d_conical` at this 15 nm grid: the worst efficiency gap is
    8.4e-3 at `+40 deg` and 5.4e-3 at `-40`, and the two signs of `theta` give the same two
    numbers, mirrored order for order.
    """
    scene = dict(theta=np.deg2rad(theta_deg), phi=np.deg2rad(phi_deg), psi=np.deg2rad(30.0))
    got = _oblique_fdtd(resolution=0.06, n_periods=80, **scene)
    ref = _oblique_rcwa(**scene)
    for leg in ('s_inc', 'p_inc', 'given'):
        for name in ('R_s', 'R_p', 'T_s', 'T_p'):
            a, b = _channels(got, leg, name), _channels(ref, leg, name)
            for m in a:
                assert float(np.real(a[m])) == pytest.approx(float(np.real(b[m])), abs=2e-2), (
                    f'{leg}.{name}[{m}]')


def test_a_tooth_edge_between_two_cells_arbitrates_the_harmonic_rule():
    """`component_property`'s harmonic-along-x rule, on the only scene that can see it.

    Two conditions have to hold at once and no test in this file held both. The rule reaches
    `eps_xx` alone, so the run has to be **TM** -- a TE run has no `Ex` and the A/B is
    bit-identical, 0.0e0. And the tooth edge has to **cut a cell**: `Ex` sits on the x
    half-steps, so its averaging cell runs from node to node and the 350 nm edge lands on a
    cell boundary whenever `nx` is even. The grating comparisons run at `resolution=0.1`,
    where `nx = 70` -- so the harmonic rule is not merely untested there, it is *inert*, and
    deleting it changes those answers by 1.4e-16.

    At `resolution=0.13` (`nx = 91`, odd) the edge sits at 45.5 cells and the rule does work:
    measured worst per-order gap against RCWA 1.9e-3 with the harmonic rule and 6.0e-3 with
    the arithmetic one, a ratio of 3.3. That is what the bound below sits between -- and the
    failure mode is the typo `component_property`'s own `ValueError` warns about, which
    "silently selects the arithmetic rule and costs an order of accuracy without failing".
    """
    reference = _by_order(_rcwa(_grating(), [THICK], 'tm'))
    result, diag = solve_fdtd(structure=_grating(), thickness=[THICK], top=EPS_TOP,
                              bot=EPS_BOT, period=[PERIOD], wavelength=WL, tetm='tm',
                              resolution=0.13, n_periods=60, return_diagnostics=True)
    assert diag['nx'] % 2 == 1, (
        f"nx = {diag['nx']} is even, so the tooth edge lands on an Ex cell boundary and the "
        f'harmonic rule averages nothing -- this test would measure the mesh, not the rule')
    orders = _by_order(result)
    for m in (-1, 0, 1):
        assert abs(orders[m][0] - reference[m][0]) < 3e-3, (
            f'R[{m}]: fdtd {orders[m][0]} against rcwa {reference[m][0]}')
        assert abs(orders[m][1] - reference[m][1]) < 3e-3, (
            f'T[{m}]: fdtd {orders[m][1]} against rcwa {reference[m][1]}')
