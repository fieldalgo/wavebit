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
    (dict(mu=np.ones((1, 1, 3))), r'share a shape exactly'),
    (dict(mu=np.array([[[-1.0, 1.0]]])), r'non-positive'),
    (dict(lattice=[[1.0, 0.0], [0.5, 1.0]]), r'lattice'),
    (dict(structure=[{'shapes': []}]), r'permittivity raster'),
    (dict(structure=np.ones((1, 4, 2))), r'invariant along y'),
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
    reference at all and this module refused it for parity. `644f881` lifted that, and this is
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
    number to say an order had been left out -- and `check_physical` cannot see it either,
    since it only refuses a sum ABOVE one.
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


def _conical_rcwa(phi_deg, psi_deg, fto=40):
    return wavebit.rcwa(backend=0, wavelength=WL, period=[PERIOD],
                        truncation_order=dict(x=fto, y=0), top=EPS_TOP, bot=EPS_BOT,
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


def _oblique_rcwa(fto=40, **over):
    kw = dict(OBLIQUE, **over)
    return wavebit.rcwa(backend=0, wavelength=WL, period=[PERIOD],
                        truncation_order=dict(x=fto, y=0), top=EPS_TOP, bot=EPS_BOT,
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
    ref = _oblique_rcwa(fto=200, **CLEAN)
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
    ref = _oblique_rcwa(fto=200)
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

    Measured against `solve_1d` at fto = 200 on the benchmark grating: 5.0e-3 uniform against
    6.5e-3 / 8.8e-3 / 7.9e-3 at grading 1.03 / 1.06 / 1.12, with amplitudes 1.6e-2 against
    1.7e-2 / 1.9e-2 / 1.8e-2. That is the transition reflection, and it is the price.
    """
    ref = _rcwa(_grating(), [THICK], 'te', fto=200)
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
