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
arbitrate `component_property`'s rule. Nothing here tests oblique incidence, loss, `mu`, a
crossed grating or a skewed cell, because all five are refused; `test_refusals` is what keeps
that honest, by pinning the refusal rather than the behaviour.
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


def _rcwa(structure, thickness, tetm, fto=80):
    solver = wavebit.rcwa(backend=0, wavelength=WL, period=[PERIOD],
                          truncation_order=dict(x=fto, y=0), top=EPS_TOP, bot=EPS_BOT,
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

def _stack_rt(wavelength, n_top, n_list, d_list, n_bot, tetm):
    """Exact ``(r, t)`` for a planar stack at normal incidence, in wavebit's frame.

    ``+z`` down, ``e^{-iwt}``, forward wave ``A exp(+i k z)``. TE propagates ``Ey`` with
    ``Hx = -n Ey`` on the forward branch, TM propagates ``Hy`` with ``Ex = Hy/n``: those two
    choices are what make the result come out as ``r_s`` and ``r_p`` rather than as one of
    them twice. A single interface gives ``r_s = (n1-n2)/(n1+n2)`` and
    ``r_p = (n2-n1)/(n2+n1) = -r_s``, which is the convention `result.py` states.
    """
    k0 = 2.0 * np.pi / wavelength
    ns = [complex(n_top)] + [complex(n) for n in n_list] + [complex(n_bot)]
    ds = [0.0] + [float(d) for d in d_list] + [0.0]

    M = np.eye(2, dtype=complex)
    for j in range(len(ns) - 1):
        phase = k0 * ns[j] * ds[j]
        P = np.diag([np.exp(1j * phase), np.exp(-1j * phase)])
        a, b = ns[j], ns[j + 1]
        if tetm == 'te':
            I = np.array([[b + a, b - a], [b - a, b + a]], dtype=complex) / (2.0 * b)
        else:
            q = b / a
            I = np.array([[1.0 + q, 1.0 - q], [1.0 - q, 1.0 + q]], dtype=complex) / 2.0
        M = I @ P @ M

    r = -M[1, 0] / M[1, 1]
    t = M[0, 0] + M[0, 1] * r
    R = abs(r) ** 2
    if tetm == 'te':
        T = abs(t) ** 2 * np.real(complex(n_bot) / complex(n_top))
    else:
        nt, nb = complex(n_top), complex(n_bot)
        T = abs(t) ** 2 * np.real((nb / nb ** 2) / (nt / nt ** 2))
        t = t * (nt / nb)                      # t_E = t_H * n_top/n_bot, solve_1d.py:339-341
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

    At the 5 nm grid this file does not run (it costs ~12 s a polarisation) the worst per-order
    gap is 6.5e-4 for TE and 7.5e-4 for TM, against Meep's 7.4e-4 and 6.9e-4 on the same grid
    -- the agreement is a three-way one, and the number below is the coarser grid's.
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
    reference = _rcwa(_uniform(), [THICK], 'te', fto=1)
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
    (dict(tetm=None), r'tetm must be'),
    (dict(tetm='psi'), r'tetm must be'),
    (dict(theta=0.3 + 0.1j), r'is complex'),
    (dict(phi=0.5), r'conical mount'),
    (dict(incidence_side='bottom'), r'incidence_side'),
    (dict(mu=[[[1.0, 1.0]]]), r'mu is not read here'),
    (dict(lattice=[[1.0, 0.0], [0.5, 1.0]]), r'lattice'),
    (dict(structure=[{'shapes': []}]), r'permittivity raster'),
    (dict(structure=np.ones((1, 4, 2))), r'invariant along y'),
    (dict(structure=np.array([[[1.0 + 0.3j, 1.0]]])), r'complex permittivity'),
    (dict(structure=np.array([[[-2.0, 1.0]]])), r'non-positive permittivity'),
    (dict(bot=-1.5), r'non-positive permittivity'),
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
