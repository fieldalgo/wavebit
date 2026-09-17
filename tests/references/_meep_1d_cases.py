"""The 1D benchmark scene, **and the full record of the conditions it is solved under**.

Read by ``gen_goldens_meep_1d.py`` (the meep env), ``gen_goldens_fdtd_table.py`` (the
project env) and both of their tests.

It lives here rather than in a test module for the reason the crossed pair gives: those
run in *different environments*. Meep is GPL and brings its own compiled stack, so the
meep generator runs in a separate ``meep`` env that has no ``wavebit`` in it, and the
tests run in the project env that has no ``meep``. Neither can import the other's module.
This one imports numpy and nothing else, so all four can read it.

**What this file is for, in one sentence: a reader regenerating ``docs/fdtd.md`` §2's
three-way table in five years reads THIS, not the document.** The document prints six
decimals; six decimals are a claim about a configuration, and the configuration is here.

**Why it had to become that.** §2's table was published with three of its conditions
recorded -- the scene, the cell size, and (in the caption of a *different* table) the PML
depth and the pad. Everything else was carried by the phrase "at the shipped defaults".
Then ``PAD_WAVELENGTHS`` was halved, the sentence stayed true-looking, and the table went
stale: measured 2026-09-07, ``pad_scale=2.0`` reproduces the previously published wavebit
column to the last printed digit in all eight entries, ``Σ`` included, while the shipped
default gives numbers up to 7.93e-5 away. A record that names *values* cannot drift that
way; a record that names *the defaults* drifts every time a default moves. So nothing
below says "the default" -- every determinant is written out, and
``test_fdtd_table_golden.py`` asserts that what is written out is still what the module
ships.

**Two spellings that no result can tell apart.** ``pad_scale`` and ``PAD_WAVELENGTHS``
multiply into one product, and ``PAD_WAVELENGTHS=1.0`` reproduces ``pad_scale=2.0``
bit-for-bit; so do ``n_pml`` / ``PML_CELLS`` and ``n_periods`` / ``N_PERIODS``. Naming
either spelling alone leaves the record ambiguous, which is why
:data:`WAVEBIT_DERIVED` states the clearance and the absorber **in length** and the run
**in steps** -- quantities that are unambiguous whichever spelling produced them, and that
survive a future default change.

The scene is **not a new one**. It is the benchmark ``tests/test_fdtd.py::_grating()``
runs everywhere, read off that module's own constants:

    tests/test_fdtd.py          here
    WL       = 900.0 nm         wavelength = 0.90
    PERIOD   = 700.0 nm         period     = 0.70
    THICK    = 300.0 nm         thickness  = (0.30,)
    N_TOOTH  = 3.5              index column 0
    N_TOP    = 1.0              n_top
    N_BOT    = 1.5              n_bot
    _grating(duty=0.5)          a two-column raster, tooth then air
                                theta = 0, phi = 0 -- normal incidence

**Lengths are micrometres** (:data:`UNITS`), which is a deliberate departure from the
nanometres ``tests/test_fdtd.py`` uses -- the same departure ``_meep_crossed_cases.py``
makes and for the same reason. Both solvers are scale-free and the FDTD module has had
exactly one defect that was not (a bare ``0.05`` in the CPML that was a reciprocal
*length*, ``docs/fdtd.md`` §3), so stating this oracle in the other unit costs nothing and
is the one thing that would notice such a constant coming back. The unit is also
load-bearing for the record itself: ``resolution=200`` means a 5 nm cell in micrometres
and a 5 µm cell in millimetres, so the keyword cannot be reconstructed from §2's printed
"5 nm" without it.

**Materials are stated as a refractive index**, not as a permittivity. Meep's
``mp.Medium`` takes ``index=`` and wavebit takes ``eps``; a case file that stated one of
the two would have to be squared or rooted on one side, and the direction is easy to get
wrong silently. ``eps_of`` below is the one conversion, read by both generators.

**Two scenes, and the first one is the calibration.** ``slab_uniform`` is the same 0.30
layer with the tooth's index everywhere -- a planar stack, whose exact answer
``airy_rt`` below writes out with no solver of any kind involved. It runs first, through
the *same* bridge as the grating, and the meep generator refuses to write a golden if it
does not reproduce the closed form. That ordering is this repository's recorded rule that
a reference must be shown able to answer a question before its answer is believed
(``memory: proxy oracles, and when to stop retreating``). It also earns its keep as a
frozen row: a uniform layer must radiate **no** non-specular order, and the two orders
that propagate in the substrate are there to receive whatever a broken bridge puts in
them.
"""
import numpy as np

#: The length unit every number in this file is stated in. Not decoration: it is what
#: makes ``meep_resolution`` and ``fdtd_resolution`` mean a cell size, and it is the one
#: thing §2's printed "5 nm" cannot be turned back into a keyword without.
UNITS = 'micrometre'


def eps_of(index):
    """The one index -> permittivity conversion. Meep takes the index, wavebit the
    permittivity; a case file that stated only one of the two invites a stray square."""
    return np.asarray(index, dtype=float) ** 2


def _raster(n_x, indices):
    """``(1, 1, n_x)`` refractive index: one layer, invariant along y, ``n_x`` columns.

    The array is **v-first** -- ``(layer, y, x)`` -- which is the rule
    ``docs/conventions.md`` states for arrays, and ``Ny == 1`` is what both solvers read
    as "a grating invariant along y" rather than as a crossed one.

    **The column count is part of the record, and so is the ORDER.** The count is free
    only while it divides the duty exactly: wavebit's averaging is an exact integral over
    segment overlaps, so 2 columns and 140 columns of the same profile agree to round-off
    (measured 3.3e-16), while 7 columns cannot express a 50 % duty and describe a
    different grating (1.9e-1). The order is not free at all on the meep side --
    ``[1.0, 3.5]`` is this raster translated by half a period, which is the same grating,
    and meep returns an answer 8.1e-3 different at resolution 100. See
    :data:`SENSITIVITY`; that is a defect in the bridge rather than a knob, and until it
    is understood the raster written here is a *condition* of the frozen column.
    """
    return np.asarray(indices, dtype=float).reshape(1, 1, n_x)


#: name -> the scene, and every setting either solver is *given*.
#:
#: ``orders_u`` is the propagating window both solvers report on, written out rather than
#: derived so that a change in either solver's window is a *visible* diff against a frozen
#: list rather than a quiet reshape. ``orders_v`` is ``(0,)`` throughout: a y-invariant
#: grating has one row of orders, and it is spelled out so the arrays carry the same rank
#: as the crossed golden's.
#:
#: ``meep_*`` and ``fdtd_*`` are the settings each golden was produced with. **Three of
#: the meep ones moved here on 2026-09-07 from function defaults in
#: ``gen_goldens_meep_1d.py``** -- ``meep_gap`` and ``meep_df_frac`` from ``_run``'s
#: signature and ``meep_src_frac`` from ``_z_layout``'s body. Their values are unchanged
#: and the regenerated golden is identical, but a number that lives only in a default is a
#: number a regeneration can move silently, which is the failure this whole file exists to
#: close. Of the three only ``meep_gap`` measurably matters, and only at zero.
CASES = {
    'slab_uniform': dict(
        wavelength=0.90, period=(0.70,), thickness=(0.30,),
        n_top=1.0, n_bot=1.5, theta_deg=0.0, phi_deg=0.0,
        # One column of the tooth's own index: the grating's material and thickness with
        # its patterning removed, so the two scenes differ in exactly one thing.
        index=_raster(1, [3.5]),
        orders_u=(-1, 0, 1),
        meep_resolution=200, meep_until=120.0, meep_dpml=0.6, meep_pad=0.7,
        meep_gap=0.10, meep_df_frac=0.2, meep_src_frac=0.30),
    'binary_grating': dict(
        wavelength=0.90, period=(0.70,), thickness=(0.30,),
        n_top=1.0, n_bot=1.5, theta_deg=0.0, phi_deg=0.0,
        # `tests/test_fdtd.py::_grating(duty=0.5)`, spelled as an index rather than as a
        # permittivity: a 50 % tooth of n = 3.5, then air. Two raster columns rather than
        # two thousand, which describes the same geometry exactly -- that module's own
        # docstring records it agreeing with a 2000-column reference to 6e-7.
        index=_raster(2, [3.5, 1.0]),
        orders_u=(-1, 0, 1),
        meep_resolution=200, meep_until=120.0, meep_dpml=0.6, meep_pad=0.7,
        meep_gap=0.10, meep_df_frac=0.2, meep_src_frac=0.30),
}


# =======================================================================================
# THE RECORD -- part 1, the meep column
# =======================================================================================

#: The build, and it is load-bearing far beyond its length. Eight settings that move this
#: column are never passed by ``gen_goldens_meep_1d.py`` at all (:data:`MEEP_UNPASSED`),
#: four of them by 5e-4 to 1.5e-2 -- so the *only* thing pinning them is this. ``grep
#: eps_averaging`` over the generator returns nothing.
#:
#: Determinism *within* the build is measured, not assumed: two separate process
#: invocations of one configuration returned identical float reprs on all eight reported
#: values, and ``OMP_NUM_THREADS`` 1 against 4 is bit-identical. Across builds, compilers,
#: architectures or an MPI build it is **unestablished** -- one build is installed here.
MEEP_BUILD = dict(
    package='pymeep', version='1.34.0', channel='conda-forge',
    conda_build='nompi_py312h46debac_101',
    python='3.12.14', parallel=False,          # mp.count_processors() == 1
    platform='macOS-26.5.2-arm64', numpy='2.5.2')

#: Settings the harness never passes, whose values are therefore meep's own defaults at
#: :data:`MEEP_BUILD`. ``name -> (value, worst per-order |delta| if it changed, note)``.
#: The delta is what a *deviation* costs, so a reader can see which of these would matter
#: on a different build. Measured one knob at a time from the shipped configuration,
#: 2026-09-07, at resolution 100 unless the note says otherwise.
#:
#: This block is the answer to "the Meep column cannot be reproduced at all": it is not
#: enough to record what the harness passes, because the four largest determinants of the
#: column are things it does not pass.
MEEP_UNPASSED = {
    'eps_averaging': (True, 4.6e-2,
                      'subpixel smoothing; 1.5e-2 at the frozen resolution 200. The '
                      'largest determinant of this column and it appears nowhere in the '
                      'harness. It bites even though every interface lands on a whole '
                      'pixel, because a face at a cell boundary sits at the CENTRE of '
                      'the cell of whichever Yee component points across it'),
    'ensure_periodicity': (True, 1.9e-2, 'energy-conserving when wrong, so no gate here '
                                         'would see it'),
    'Courant': (0.5, 5.2e-5, 'at Courant 0.25; below 1e-4 but above the 1e-5 a '
                             'reproduction should aim at, so it has to be named'),
    'pml_profile_exponent': (2.0, 6.9e-4, 'meep\'s quadratic default; a u^1 profile costs '
                                          'this much, a u^3 profile 1.6e-5'),
    'pml_R_asymptotic': (1e-15, 3.2e-4, 'at 1e-8. 1e-25 moves 6e-7, so the default is '
                                        'converged'),
    'gaussian_cutoff': (5.0, 4.8e-4, 'at cutoff 3. cutoff 10 moves 1.3e-7, so 5 is '
                                     'converged and 3 is not'),
    'decimation_factor': ('automatic', 1.3e-7, 'meep\'s automatic decimation is safe here'),
    'subpixel_tol_maxeval': ('meep defaults', 0.0,
                             'measured EXACTLY zero: the averages this geometry needs are '
                             'exact rather than sampled. Recorded as measured-inert so '
                             'nobody tunes it hoping to move the number'),
}

#: What the harness constructs, stated as rules rather than as code, because a reader
#: reimplementing the bridge needs these and cannot get them from the case dict.
MEEP_CONSTRUCTION = dict(
    cell='three real axes, y a single pixel; cell_size = (period, 0, z_extent)',
    absorber='mp.PML(dpml, direction=mp.Z) -- mp.Absorber instead moves the answer 1.5e-1 '
             'and returns R + T = 0.81',
    geometry='a full-period background block of the LAST raster run, the other runs '
             'overlaid (the later object wins), a run equal to the background skipped, '
             'blocks placed from the CELL CENTRE',
    substrate='sized exactly to the +z cell edge -- overhanging it wraps through the PML '
              'axis and returns R + T = 1.83',
    source='an electric sheet along y for TE and a magnetic sheet along y for TM, one '
           'period wide, is_integrated=True, GaussianSource(fcen, fwidth=df_frac*fcen)',
    normalisation='a second run of the SAME cell with the geometry removed entirely, i.e. '
                  'a homogeneous n_top -- not a substrate-with-no-grating run, which '
                  'would be wrong by the substrate Fresnel factor',
    extraction="meep's own abscissae AND weights from get_array_metadata; reflection is "
               'structure minus empty at the ref plane, transmission is unsubtracted; the '
               'incident order is de-referenced to z = 0, not to the monitor',
    frame='origin factor (-1)**m, leg signs SIGMA below, Re(kz) >= 0, per-order s_hat',
    reading='the E-projection with the analytic kz (see MEEP_READING_FLOOR)',
    grid_refusal='period, period/n_columns and every thickness must be a whole number of '
                 'pixels or the generator exits -- on this scene that restricts the '
                 'resolution to multiples of 20',
)

#: Derived meep layout, per resolution: what the recorded settings actually build. A
#: reproduction that lands on different numbers here is not reproducing this column,
#: whatever its case dict says. ``(cell z extent, z pixels, x pixels, z_min, z_max,
#: z_src, z_ref, z_trn)``.
MEEP_DERIVED = {
    200: dict(z_extent=2.9, nz_pixels=580, nx_pixels=140, z_min=-1.3, z_max=1.6,
              z_src=-0.49, z_ref=-0.10, z_trn=0.40),
    100: dict(z_extent=2.9, nz_pixels=290, nx_pixels=70, z_min=-1.3, z_max=1.6,
              z_src=-0.49, z_ref=-0.10, z_trn=0.40),
}

#: **The irreducible ambiguity in the meep column, and it is smaller than §2 used to say.**
#: One resolution-200 run read three legitimate ways -- the shipped E-projection with the
#: analytic ``kz``, a per-order ``Re(E x conj(H))`` from the H field meep actually stepped,
#: and meep's own ``add_flux`` -- differ by this much. Per order it is 3.0e-5 to 7.2e-5;
#: on the *total* it is 2.06e-4 (TE) and 1.17e-4 (TM). The two Poynting readings agree to
#: 5e-6 (TM) and 8.0e-5 (TE); it is the shipped E-projection that sits apart from both.
#:
#: §2 inherited "two readings of a transmitted order differ by 2.5e-4 (TE) and 3.4e-4 (TM)"
#: from the 2026-08-25 lab run. That reproduces on TE only as a statement about the TOTAL,
#: and is 2.9x too large on TM. Per order the floor is four times smaller than the figure
#: it was quoted as. ``get_eigenmode_coefficients`` was NOT among the three readings, so
#: this is a lower bound on the ambiguity rather than necessarily the same pairing.
MEEP_READING_FLOOR = dict(per_order_te=6.7e-5, per_order_tm=7.2e-5,
                          total_te=2.06e-4, total_tm=1.17e-4)


# =======================================================================================
# THE RECORD -- part 2, the wavebit column
# =======================================================================================

#: The grid §2's wavebit column is taken on, in cells per :data:`UNITS`. 200 is the 5 nm
#: cell the table prints -- and "5 nm" is only a cell size once :data:`UNITS` is known,
#: which is the half of it §2 never stated.
FDTD_PUBLISHED = 200

#: The four cells §2's **convergence** table runs, in cells per :data:`UNITS`:
#: 20 / 10 / 5 / 2.5 nm. That series is also what makes the frozen 5 nm column carry its
#: own error bar rather than be one number taken on faith -- the same job the meep
#: golden's two tiers do, done with four. The 5 nm row is *shared* with the three-way
#: table above, which is what pins the two tables to one configuration: at HEAD on
#: 2026-09-06 they had been taken on either side of the ``PAD_WAVELENGTHS`` change and
#: only one of them was re-derived, so the document disagreed with itself.
FDTD_CONVERGENCE = (50, 100, 200, 400)

#: Truncation order for the RCWA reference both gaps are measured against, and it is
#: §2's own choice. Its own uncertainty is measured rather than assumed: against
#: ``truncation_order = 400`` the worst per-order move is **1.21e-6 (TE)** and
#: **1.30e-5 (TM)**, so every gap in the table clears the reference by 18x or better.
RCWA_ORDERS = 80
RCWA_ORDERS_CONVERGED = 400

#: **Every keyword ``solve_fdtd`` is given for §2's column**, beyond the scene itself. The
#: call is deliberately this short: a table taken at settings no user types is a table
#: about nothing. What makes it reproducible is not the shortness but
#: :data:`WAVEBIT_SHIPPED` below, which writes out what the short call inherits.
WAVEBIT_CALL = dict(resolution=200)

#: **What ``WAVEBIT_CALL`` inherits, by value.** Every determinant the module supplies
#: rather than the caller, each measured to move §2's printed digits by 1e-4 or worse (or,
#: for the last four, to be reachable only through something that does).
#: ``test_fdtd_table_golden.py`` asserts each of these against the live module, so halving
#: a clearance again is a failing test that names this table instead of a silent 8e-5.
#:
#: Read as ``name -> value``; the sources are ``wavebit._core.fdtd`` module attributes
#: except the four marked, which are ``solve_fdtd`` signature defaults. **The distinction
#: matters and is not cosmetic**: ``PML_CELLS`` and ``COURANT`` are bound into the
#: signature at import, so patching the module attribute is inert, while
#: ``PAD_WAVELENGTHS``, ``N_PERIODS``, ``CELLS_PER_WAVELENGTH`` and the PML profile constants
#: are read at call time and do bite. ``PML_TARGET_REFLECTION`` has since moved to the first
#: kind: it is a bound default of ``cpml_coefficients``, ``cpml_profiles`` and ``solve_fdtd``'s
#: ``pml_reflection``, so patching it is inert and ``pml_reflection=`` is the live knob. Its
#: ``SENSITIVITY`` rows were measured before that, while it was still read at call time.
#: Anyone re-deriving these by monkeypatching needs to know which is which or they will
#: report a live knob as inert.
WAVEBIT_SHIPPED = {
    # -- signature defaults of solve_fdtd --
    'pad_scale': 1.0,
    'n_pml': 12,
    'courant': 0.99,
    'subpixel': True,
    'grading': None,
    'n_periods': None,          # -> N_PERIODS
    'truncation_order': None,   # -> every order propagating at the shortest wavelength
    'mu': None,
    'dispersion': None,
    'poles': None,
    'theta': 0.0,
    'phi': None,
    'lattice': None,
    'incidence_side': 'top',
    # -- module attributes, read at call time --
    'PAD_WAVELENGTHS': 0.5,
    'N_PERIODS': 80.0,
    'CELLS_PER_WAVELENGTH': 20.0,
    'SOURCE_CYCLES': 8.0,
    'SOURCE_DELAY': 4.0,
    'SOURCE_BAND_WIDTHS': 1.2,
    'PML_ORDER': 4.0,
    'PML_KAPPA_MAX': 2.0,
    'PML_ALPHA_RELATIVE': 1.5,
    'PML_TARGET_REFLECTION': 1e-8,  # now bound at import; see the note above the table
    'SETTLED_RESIDUAL': 1e-2,
    'MIN_INCIDENT_FRACTION': 1e-3,
    'MAX_DEREFERENCE_GAIN': 1e10,
    'GRADE_RATIO': 1.06,        # unreachable here: grading is off. Recorded, not omitted
    # -- module attributes bound into the signature at import; listed so that the two
    #    spellings of one quantity are both on the record --
    'PML_CELLS': 12,
    'COURANT': 0.99,
}

#: **The same run stated in lengths and counts**, which is the form that survives a
#: default change and the form a reproduction should check itself against. Every entry is
#: read straight out of ``solve_fdtd(..., return_diagnostics=True)`` or off the grid it
#: builds, at ``resolution=200`` on the grating; ``test_fdtd_table_golden.py`` asserts
#: them.
#:
#: The point of this block is the two-spellings problem: ``pad_scale=2.0`` and
#: ``PAD_WAVELENGTHS=1.0`` are indistinguishable in any result, and both give
#: ``clearance_top = 0.90``. A record stating ``pad_scale=1.0`` alone is therefore not a
#: record. A record stating ``0.45 above and 0.30 below`` is.
WAVEBIT_DERIVED = dict(
    nx=140, dx=0.005, dz=0.005,             # dz follows dx, which follows period/resolution
    nz=281,
    n_max=3.5, n_min=1.0,
    dt=3.5001785668734102e-3,               # courant * n_min / sqrt(dx^-2 + dz^-2)
    n_steps=20571,                          # ceil(n_periods * centre / dt) = 80.0024 cycles
    clearance_top=0.45,                     # PAD_WAVELENGTHS * pad_scale * wl / n_top
    clearance_bot=0.30,                     # ... / n_bot
    pml_thickness=0.060,                    # both ends; 12 cells x 5 nm, uniform mesh only
    z0=-0.74,                               # the top of the grid, structure top at z = 0
    z_source=-0.67, z_ref=-0.56, z_tfsf=-0.45,
    z_top_interface=0.0, z_bot_interface=0.30, z_trn=0.45,
    source_width=7.2, source_delay=28.8, source_emission_floor_cycles=48.0,
    residual_te=4.37e-6, residual_tm=9.77e-8,
    dereference_gain=23.5,                  # 8.6 orders under MAX_DEREFERENCE_GAIN
)

#: The environment the wavebit column was taken in. NumPy only, by refusal -- ``solve_fdtd``
#: has no backend keyword. Measured bit-identical across repeat calls in one process and
#: across separate processes: no RNG, no threading, no BLAS in the time loop. That
#: determinism is what makes six printed decimals a testable claim rather than a tolerance.
WAVEBIT_BUILD = dict(numpy='2.5.1', python='3.12.9', platform='macOS-26.5.2-arm64',
                     backend='numpy only (no backend keyword exists)')


# =======================================================================================
# THE RECORD -- part 3, what a deviation costs
# =======================================================================================

#: ``(side, knob) -> [(value, worst per-order |delta|, note)]``. Worst ``|delta|`` over the
#: per-order ``R`` and ``T`` at ``m = -1, 0, +1`` against the shipped configuration, one
#: knob at a time, on ``binary_grating``. TE unless a note says otherwise.
#:
#: **Provenance.** Measured 2026-09-07. Six rows were taken twice, by harnesses written
#: independently, and agree to every figure printed here: the shipped baseline itself (both
#: sides), ``pad_scale=2.0``, ``subpixel=False``, ``n_pml=6``, ``meep_dpml=0.9`` at the frozen
#: resolution, and ``eps_of``'s two-column raster against 140 columns. The rest are single
#: measurements. The meep rows marked "at resolution 100" were taken on the golden's coarse
#: tier because a res-200 run is ~6 minutes; where both were taken the res-200 figure is
#: smaller (``eps_averaging`` 4.6e-2 -> 1.5e-2, the background choice 4.9e-3 -> 1.2e-3,
#: ``meep_dpml`` 6.5e-3 -> 1.6e-3) but never below 1e-3, so a res-100 row is an upper bound
#: on the frozen column's sensitivity and not a substitute for measuring it there.
#:
#: **This is what makes the record usable rather than merely long.** A determinant list
#: without sizes cannot tell a reader whether their deviation matters, and it cannot tell a
#: maintainer which entries to defend. §2's whole published agreement is 4.9e-4 to 7.4e-4,
#: so anything here at 1e-4 or worse is the same size as the claim.
#:
#: Two facts that come out of the table rather than out of any single row:
#:
#: * **``R + T`` is blind to the biggest knob on either side.** ``subpixel=False`` moves
#:   per-order efficiencies by 3.0e-2 while ``Σ`` moves 4.5e-6. An energy check would have
#:   passed every configuration in this sweep except a 6-cell PML.
#: * **Amplitudes are 3-10x more sensitive than the efficiencies §2 prints.** Worst
#:   ``|delta|`` on ``(r_s, r_p, t_s, t_p)`` is 8.3e-2 for ``subpixel=False`` against
#:   3.0e-2 on ``R``/``T``, and 6.2e-2 for ``PML_ALPHA_RELATIVE=0`` against 1.8e-4. If an
#:   amplitude from this scene is ever published, several courtesy entries become
#:   determinants.
SENSITIVITY = {
    ('wavebit', 'pad_scale'): [
        (0.25, 2.24e-4, ''), (0.5, 1.04e-4, ''),
        (2.0, 7.93e-5, 'THE row this file exists for: reproduces the column published '
                       'until 2026-09-06 to the last printed digit, Sigma included'),
        (4.0, 4.1e-5, 'not monotone -- 4.0 is nearer the default than 0.25 is, so this '
                      'is not a convergence parameter and cannot be argued away'),
    ],
    ('wavebit', 'PAD_WAVELENGTHS'): [
        (1.0, 7.93e-5, 'BIT-IDENTICAL to pad_scale=2.0; they multiply into one product '
                       'and no result can separate them'),
    ],
    ('wavebit', 'resolution'): [
        (None, 4.4e-3, 'the module default, 20 cells per wavelength in n_max = 77.78 per '
                       'um: 6x the gap section 2 publishes'),
        (100, 2.1e-3, ''), (400, 5.2e-4, ''),
        (201, 3.3e-4, 'the trap: resolution is NOT 1/dx. dx = period/round(period*'
                      'resolution), so 201 gives nx = 141 and dx = 4.9645 nm'),
    ],
    ('wavebit', 'n_pml'): [
        (6, 2.61e-2, 'the only row in this sweep an energy check would catch'),
        (8, 4.30e-3, ''), (16, 6.4e-5, ''),
        (20, 6.0e-5, 'saturates: 16 and 20 agree to 5e-6, so 12 is 6e-5 short of the '
                     'converged absorber -- 10% of the published gap'),
    ],
    ('wavebit', 'n_periods'): [
        (47, None, 'REFUSED: the source is not finished emitting until 48 cycles'),
        (48, 8.27e-3, ''), (60, 1.36e-4, ''), (100, 4.2e-7, ''),
        (160, 4.1e-7, 'converged by 80 AT NORMAL INCIDENCE only; 15 degrees needs 120'),
    ],
    ('wavebit', 'courant'): [
        (0.5, 8.9e-5, 'moves the answer toward MORE error as the step shrinks -- temporal '
                      'dispersion partly cancels the spatial one at the stability bound, '
                      'so playing safe costs accuracy and 2x the wall time'),
        (0.9, 2.3e-5, ''), (0.999, 2.5e-6, ''),
    ],
    ('wavebit', 'subpixel'): [
        (False, 2.98e-2, 'the largest knob here, and Sigma moves 4.5e-6 while it happens'),
    ],
    ('wavebit', 'grading'): [
        (1.03, 6.6e-4, ''),
        (1.06, 7.2e-4, 'GRADE_RATIO, i.e. grading=True: turning grading on roughly doubles '
                       'the published error on this dielectric scene, and relocates every '
                       'plane, so WAVEBIT_DERIVED describes the uniform mesh only'),
        (1.12, 7.5e-4, ''),
    ],
    ('wavebit', 'PML_ALPHA_RELATIVE'): [
        (0.0, 1.81e-4, 'zero is not safe, as the module says'), (1.0, 6.2e-5, ''),
        (2.0, 1.38e-4, ''),
        (4.0, 2.19e-3, 'the widest-swinging PML constant; the neighbourhood of the '
                       'dimensional 0.05 that section 3 is about'),
    ],
    ('wavebit', 'PML_TARGET_REFLECTION'): [
        (1e-6, 2.39e-4, ''),
        (1e-10, 1.55e-4, 'non-monotone: a MORE ambitious target is worse, because '
                         'sigma_max grows and the discrete grading reflects more than the '
                         'continuum one absorbs'),
    ],
    ('wavebit', 'PML_ORDER'): [(3.0, 6.0e-5, ''), (5.0, 2.49e-4, '')],
    ('wavebit', 'PML_KAPPA_MAX'): [
        (1.0, 5.7e-5, ''),
        (5.0, 4.4e-5, 'the one constant the module admits is not settled by measurement '
                      'carries ~8% of the published gap in either direction'),
    ],
    ('wavebit', 'SOURCE_CYCLES'): [
        (6.0, 6.1e-7, ''),
        (12.0, 1.78e-4, 'a longer pulse needs a longer run; this constant and n_periods '
                        'are coupled and the record has to carry both'),
    ],
    ('wavebit', 'CELLS_PER_WAVELENGTH'): [
        (40.0, 0.0, 'EXACTLY inert once `resolution` is given -- but 2.2e-2 for a reader '
                    'who copies the scene and omits it. Two jobs, only one switched off'),
    ],
    ('wavebit', 'raster columns'): [
        (140, 3.3e-16, 'the same profile: wavebit averages by exact integral'),
        (7, 1.9e-1, 'cannot express a 50 % duty -- a different grating'),
    ],
    ('wavebit', 'wavelength as a band'): [
        ((0.88, 0.90, 0.92), 0.0, ''),
        ((0.80, 0.90, 1.00), 2.1e-4, 'TM. Asking for a spectrum CHANGES the answer at the '
                                     'centre wavelength, because the pulse carrying it is '
                                     'a different pulse. Section 2 is single-wavelength'),
    ],
    ('wavebit', 'the length unit'): [
        ('nanometre', 6.9e-15, 'the value is scale-free; the RECORD is not -- '
                               '`resolution=200` is a 5 nm cell here and a 5 um cell in mm'),
    ],
    ('meep', 'meep_dpml'): [
        (0.9, 1.575e-3, 'AT THE FROZEN RESOLUTION 200, and the one row here re-measured '
                        'twice independently: TE R0 goes 0.204410 -> 0.205985, T(+-1) '
                        '0.221391 -> 0.220556, while Sigma stays at 0.99957. That is 3.2x '
                        'the 4.9e-4 agreement the table claims for this column, and the '
                        'shipped 0.6 is the branch nearer RCWA (4.9e-4 against 1.1e-3). '
                        'Bimodal and non-monotone -- see MEEP_UNEXPLAINED'),
        (0.3, 0.0, 'same branch as the shipped 0.6, to ~1e-6'),
    ],
    ('meep', 'meep_pad'): [
        (1.4, 1.58e-3, 'at resolution 200, and it lands on the SAME value dpml 0.9 gives '
                       'to 1e-6 -- two branches, not a continuum'),
        (0.35, 5.0e-6, 'also relocates the source plane: z_src depends on pad, so a pad '
                       'sweep is never a clean clearance sweep'),
    ],
    ('meep', 'raster column order'): [
        ((1.0, 3.5), 8.07e-3, 'the same grating translated by half a period, at resolution '
                              '100. A DEFECT, not a knob -- efficiencies must be '
                              'translation-invariant. 1.69e-2 at res 60, 4.7e-16 at 40 and '
                              '80; the docstring that concluded invariance measured it at '
                              '40'),
    ],
    ('meep', 'the background block'): [
        ('first', 1.19e-3, 'at resolution 200, and it opens the exact TE mirror split to '
                           '1.44e-3'),
        ('none', 0.0, 'BIT-IDENTICAL to the shipped `last` on this raster, because the '
                      'last run is air and air is default_material -- so the benchmark '
                      'never exercises the rule it documents'),
    ],
    ('meep', 'meep_gap'): [
        (0.0, 3.41e-3, 'the monitor ON the interface; R + T drops to 0.9936'),
        (0.02, 1e-6, 'and 0.05 / 0.20 / 0.30 / 0.40 likewise: over 0.02-0.40 the distance '
                     'buys nothing measurable. The only thing that matters is that it is '
                     'not zero'),
    ],
    ('meep', 'meep_until'): [
        (30.0, 8.4e-8, 'NOT a determinant at either end -- 240 moves 3.9e-10. It is a cost '
                       'knob, and the record should say so rather than leave a reader '
                       'guessing that 120 was tuned. A fixed `until_after_sources` is '
                         'used rather than the `stop_when_fields_decayed` meep\'s FAQ '
                         'recommends, because a fixed duration is a RECORDABLE condition '
                         'while a decay threshold is a stopping rule whose outcome depends '
                         'on the scene -- and the point of this file is that someone can '
                         'reproduce the number in five years. The FAQ gives spectral '
                         'leakage as the reason for its recommendation ("the minimum '
                         'runtime should be the inverse of the desired frequency '
                         'resolution"), which 120 clears here by the measurement above. '
                         'https://meep.readthedocs.io/en/latest/FAQ/, read 2026-09-07'),
    ],
    ('meep', 'meep_df_frac'): [
        (0.05, 1.0e-7, 'not a determinant -- the DFT is at a single frequency. In the case '
                       'dict so that a regenerated golden cannot move silently'),
    ],
    ('meep', 'meep_src_frac'): [
        (0.15, 2.1e-7, 'not a determinant by itself, but it is what couples `pad` to the '
                       'source, and it is resolution-dependent through the grid snap'),
    ],
    ('meep', 'meep_resolution'): [
        (40, 2.46e-2, ''), (60, 7.65e-3, ''), (80, 6.66e-3, ''), (100, 2.20e-3, ''),
        (120, 1.46e-3, ''),
        (200, 4.88e-4, "TE's error ALTERNATES IN SIGN below res 100 while TM's is "
                       'monotone, so a two-tier power-law error bar assumes something TE '
                       'does not have. Legal resolutions here are multiples of 20'),
        (150, 1.70e-3, 'reached only by bypassing the integer-pixel refusal: 0.35 um is '
                       '52.5 pixels. Off the res 120/200 trend, so the refusal is doing '
                       'real work'),
    ],
    ('meep', 'k_point'): [
        (None, 3.15e-1, 'omitting it puts PEC walls on x instead of Bloch-periodic ones. '
                        'Required, not tunable -- listed because Vector3(0,0,0) looks like '
                        'a no-op'),
    ],
    ('meep', 'default_material'): [
        (1.5, 2.27e-1, 'AND THE HARNESS INSTRUMENT IS BLIND TO IT: the incident-only leak '
                       'stays at 4e-17, because a wrong superstrate is still exactly one '
                       'plane wave. Only the planar calibration catches it'),
    ],
    ('meep', 'source sheet width'): [
        (0.35, 1.93e-2, 'half a period; one of the two conditions the leak instrument does '
                        'catch, at 8.4e-2'),
    ],
    ('meep', 'the reading'): [
        ('Poynting', 7.2e-5, 'per order, at resolution 200; 2.06e-4 on the total. See '
                             'MEEP_READING_FLOOR -- this is the floor, not a knob'),
    ],
}

#: **What the measurement did NOT establish, recorded so nobody re-derives a dead end.**
#: The meep column is bimodal in ``(dpml, pad)`` and the mechanism is unknown. Eleven
#: candidates were tested on 2026-09-07 and each refuted by at least one row: cell length,
#: z pixel count mod 4, PML cell count, PML absorption quality, source plane,
#: source-to-PML distance, dpml-vs-pad, meep's non-integer-grid warning,
#: ``ensure_periodicity``, ``eps_averaging``, and the incident normalisation
#: (cross-pairing the two empty runs moves 1.2e-6 while cross-pairing the structure runs
#: moves 1.9e-2, so it lives entirely in the scattered field with ``R + T`` preserved to
#: 1e-5). Settling it would need meep's source, which this repository does not open. The
#: shipped 0.6 / 0.7 is the better branch against RCWA and nothing anywhere says why.
#:
#: Two patterns that are NOT mechanisms and must not be written down as if they were: the
#: sensitivity tracks the parity of the half-period in pixels at resolutions 40 / 60 / 80 /
#: 100 but resolution 200 has an even count and is still sensitive at 1.6e-3; and a
#: cell-length threshold near 3.1 um fits until two configurations with the identical cell
#: length land in opposite branches.
MEEP_UNEXPLAINED = (
    'the (dpml, pad) bimodality: 1.58e-3 at resolution 200, mechanism unestablished after '
    'eleven refuted candidates')


#: The three leg signs are the canonical frame rather than a convention, and both sides
#: derive them from the same two definitions -- ``s_hat = z_hat x k_hat_par`` and
#: ``p_hat = k_hat x s_hat`` -- so nothing here is fitted. Written down because the
#: generator and ``wavebit/_core/fdtd.py`` reach them independently:
#:
#:     E . p_hat = -+ (n / kz) (cos(varphi) Ex + sin(varphi) Ey)
#:
#: with the upper sign for a DOWNGOING wave. Incident -1, reflected +1, transmitted -1.
SIGMA = (-1.0, 1.0, -1.0)

#: How the two legs are named on each side of the bridge, and it is the one mapping in
#: this file that a swapped label would hide from every energy check.
#:
#: * **wavebit** takes ``tetm='te'`` / ``'tm'``. TE is the grating community's TE: the
#:   electric field lies **along the grooves**, which here is ``y``. ``tests/test_fdtd.py``
#:   pins it -- ``_stack_rt`` propagates ``Ey`` for TE and ``Hy`` for TM -- and the
#:   conical machinery calls the same two legs ``s`` and ``p``, which is why the crossed
#:   golden's arrays are keyed ``s``/``p`` and these are keyed ``te``/``tm``.
#: * **Meep** is never asked for a polarisation here. The generator drives a named
#:   *component*: an electric sheet along ``s_hat = y_hat`` for TE, a magnetic sheet along
#:   ``y_hat`` for TM. That is deliberate. Meep's own two-dimensional vocabulary calls the
#:   ``(Ex, Ey, Hz)`` set "TE" and the ``(Hx, Hy, Ez)`` set "TM" -- named for the field
#:   that is *out of plane*, which is the reverse of the grating convention above, and a
#:   simulation set up by that name rather than by component reproduces the other
#:   polarisation's efficiencies with ``R + T`` still exactly 1. The generator runs a cell
#:   with three real axes and names components, so the vocabulary never has to be
#:   translated -- and ``slab_uniform`` is what would catch it if it had been, since the
#:   two legs' *amplitudes* differ by a sign there while their efficiencies do not.
LEGS = {'te': 's', 'tm': 'p'}


def in_plane_k(case, m):
    """``kx`` of order ``m``, normalised by ``k0``."""
    th, ph = np.deg2rad(case['theta_deg']), np.deg2rad(case['phi_deg'])
    return case['n_top'] * np.sin(th) * np.cos(ph) + m * case['wavelength'] / case['period'][0]


def margin(case, m, n):
    """``n**2 - kx**2``: positive where order ``m`` propagates in index ``n``.

    Compared against a *margin* rather than against zero wherever amplitudes are: an
    order within round-off of a Rayleigh point converges as ``1/kz`` and the p channel
    multiplies by ``n/kz`` again, so both codes are slowest exactly there. This scene is
    nowhere near one -- the smallest margin over both half-spaces is 0.60, at ``m = +-1``
    in the substrate.
    """
    return float(n ** 2 - in_plane_k(case, m) ** 2)


def airy_rt(wavelength, n_top, n_list, d_list, n_bot):
    """Exact ``(r, t)`` for a planar stack at **normal incidence**, in wavebit's frame.

    Written here, from the boundary conditions, so that the calibration this file's
    docstring describes involves no solver at all -- not wavebit's, not Meep's. It is the
    same closed form ``tests/test_fdtd.py::_stack_rt`` carries, deliberately re-derived
    rather than imported: that module lives in the other environment, and a calibration
    that shares an implementation with the thing it calibrates is not one.

    Conventions, which are what the calibration actually pins: ``+z`` points down into the
    stack, the time dependence is ``e^{-iwt}``, and a forward (downgoing) wave is
    ``A exp(+i k z)``. Amplitudes are referenced to the interface they sit on, so ``r`` is
    taken at ``z = 0`` and ``t`` at ``z = sum(d_list)``.

    Layer ``j``'s amplitudes are carried at its own top interface, so the chain is

        M = I(0,1) . P(1)^-1 . I(1,2) . P(2)^-1 ... I(N-1,N)

    with ``I(a,b) = [[1, r_ab], [r_ab, 1]] / t_ab`` from tangential continuity and
    ``P(j)^-1 = diag(exp(-i k_j d_j), exp(+i k_j d_j))``. Then ``(A0, B0) = M (An, 0)``,
    so ``r = M[1,0] / M[0,0]`` and ``t = 1 / M[0,0]``. A single interface collapses it to
    ``r = (n0 - n1) / (n0 + n1)`` and ``t = 2 n0 / (n0 + n1)``, and one layer to the Airy
    formula.

    Returned in the **s** convention. The p leg at normal incidence is ``(-r, +t)``: the
    two waves' tangential fields obey identical equations, so the *physical* ``Ex`` ratio
    equals the s wave's ``Ey`` ratio, and the sign appears only because the frame's
    ``p_hat`` reverses between the incident and reflected legs. That relation --
    ``r_p = -r_s`` and ``t_p = +t_s``, which no efficiency can see -- is what
    ``slab_uniform`` is frozen to check.
    """
    k0 = 2.0 * np.pi / float(wavelength)
    n = [complex(n_top)] + [complex(v) for v in n_list] + [complex(n_bot)]
    m = np.eye(2, dtype=complex)
    for j in range(len(n) - 1):
        if j:
            phase = np.exp(-1j * k0 * n[j] * float(d_list[j - 1]))
            m = m @ np.array([[phase, 0.0], [0.0, 1.0 / phase]], dtype=complex)
        r = (n[j] - n[j + 1]) / (n[j] + n[j + 1])
        t = 2.0 * n[j] / (n[j] + n[j + 1])
        m = m @ (np.array([[1.0, r], [r, 1.0]], dtype=complex) / t)
    return m[1, 0] / m[0, 0], 1.0 / m[0, 0]
