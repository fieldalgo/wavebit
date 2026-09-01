"""A 2D FDTD solve for a 1D-periodic grating, answering the same question as ``solve_1d``.

This is roadmap item 2 -- **a second solver the caller chooses**, reading the same
:class:`~wavebit._core.domain.Stack` and :class:`~wavebit._core.domain.Incidence` the RCWA path
reads and returning the same :class:`~wavebit._core.result.Result`. It is not a test harness:
it is meant to be picked when it is the better tool, which it is wherever a broadband answer
is wanted from one run, or where a scene converges badly in a truncated Fourier series.

That its errors are *different* errors -- RCWA truncates a Fourier series, FDTD discretises
space and time -- is a second dividend rather than the point: a defect surviving both is
unlikely to belong to either, so the two paths cross-check each other for free. Which is also
why the conventions below are wavebit's rather than the FDTD literature's.

**What it answers.** The pure TE / TM problem, normal incidence, on a raster structure that is
periodic along one axis. That is deliberately the scope of :func:`~wavebit._core.solve_1d.
solve_1d`, which this mirrors down to the amplitude conventions: TE carries everything on
``s``, TM everything on ``p``, and the opposite side is identically zero. Everything outside
that scope is refused by name rather than approximated -- see :func:`_refuse_out_of_scope`.

**Conventions.** All of them are wavebit's, and each one is a place a from-scratch FDTD
normally parts company with the rest of the package:

``+z`` points DOWN
    into the stack, ``z = 0`` at the top interface and ``z = Z`` at the bottom, matching
    ``docs/conventions.md`` and :mod:`~wavebit._core.fields`. Most of the FDTD literature
    writes ``+z`` out of the stack; adopting that here would negate ``E_z``, ``H_x`` and
    ``H_y`` while conserving energy and leaving every efficiency untouched, which is exactly
    the defect ``fields.py`` carried undetected for months.

``e^{-iwt}``
    so a phasor is recovered from the real time series with an ``exp(+i w t)`` kernel.
    ``np.fft.fft`` over the TIME axis applies the opposite kernel and returns the conjugate;
    it is never used on the time axis here, and the accumulation is written out instead.

``kx_m = n_inc sin(theta) cos(phi) + m*wl/period``
    the order index's sign, taken from :attr:`~wavebit._core.engine.CoreSolveMixin.n_inc`.
    Reversing it mirrors ``+1`` onto ``-1`` while conserving energy exactly.

``H`` is the normalised ``H~ = Z0 H``
    and the units are ``c = eps0 = mu0 = 1``, lengths in whatever unit ``wavelength`` is in.

**Scale freedom is a property to protect, not an accident.** Every constant below is
dimensionless or derived from ``wavelength``. The one place that was violated -- a PML
frequency shift written as a bare ``0.05``, which is a reciprocal *length* -- made the answer
depend on whether the caller wrote nanometres or microns, and returned ``R + T = 1.22`` in the
mid-infrared while looking perfectly well behaved at 900 nm.

**Why NumPy only.** The time loop is a Python ``for`` over ~10^4 steps. Under ``jax.jit`` that
unrolls to a trace of that length; under ``jax.grad`` or torch autograd it materialises a tape
of O(steps x grid). Crossing either needs a per-backend ``scan`` -- ``jax.lax.scan``, an eager
loop elsewhere -- and there is no such primitive in :mod:`~wavebit._core.backend` today, nor
any loop construct in ``_core`` that is not a Python ``for`` over a small static layer count.
Adding one is a design decision rather than a continuation of this work, so ``backend`` is
refused by name here instead of being half-supported. The field updates themselves need no
index assignment (they are slice-shifted differences), so the body would port; it is only the
loop that has nowhere to live.

Literature. Each item is cited again at the line that implements it:

* K. S. Yee, IEEE Trans. Antennas Propag. **14**, 302 (1966) -- the staggered grid and the
  leapfrog update.
* J.-P. Berenger, J. Comput. Phys. **114**, 185 (1994) -- the perfectly matched layer.
* J. A. Roden and S. D. Gedney, Microw. Opt. Technol. Lett. **27**, 334 (2000) -- the
  complex-frequency-shifted PML in the recursive-convolution form used here.
* K. Umashankar and A. Taflove, IEEE Trans. Electromagn. Compat. **24**, 397 (1982) -- the
  total-field / scattered-field plane-wave source.
* A. Taflove and S. C. Hagness, *Computational Electrodynamics: The Finite-Difference
  Time-Domain Method*, 3rd ed. (Artech House, 2005) -- the reference text for all of the
  above; the PML grading is eq. 7.61 and the 1D auxiliary incident grid is sec. 5.8.
* A. Farjadpour et al., Opt. Lett. **31**, 2972 (2006), and A. F. Oskooi et al.,
  Comput. Phys. Commun. **181**, 687 (2010) -- subpixel smoothing. The rule used here is the
  axis-aligned specialisation described at :func:`component_property`.
"""

import warnings

import numpy as np

from .backend import check_physical
from .result import Result, ResultSub


#: Cells per wavelength **in the densest medium** when ``resolution`` is not given. 20 is the
#: usual FDTD rule of thumb and the measured convergence supports it: on the benchmark grating
#: the worst per-order error is 1.2e-2 at ~11 cells per wavelength in the tooth, 2.8e-3 at 23,
#: 7.5e-4 at 46 and 1.8e-4 at 92.
CELLS_PER_WAVELENGTH = 20.0

#: PML depth in cells. **A correctly graded CPML is dz-independent, so cells is the right unit**
#: -- measured on a 1D probe in glass at a fixed 20-cell depth, the amplitude reflection runs
#: 2.06e-6 / 1.63e-6 / 1.42e-6 / 1.37e-6 as ``dz`` goes 20 / 10 / 5 / 2.5 nm. It was *not*
#: dz-independent before ``PML_ALPHA_RELATIVE`` and the ``sqrt(eps_r)`` below were fixed: the
#: same sweep then ran 9.1e-3 down to only 6.9e-4, an apparent ``dz^1.25``, which is easy to
#: misread as a case for specifying the layer as a physical thickness. It is not; it was a
#: broken absorber. At 12 cells the fixed layer reflects 1.1e-5, and 20 buys 1.5e-6.
PML_CELLS = 12

#: Courant number. The 2D bound is ``dt <= n_min/sqrt(dx^-2 + dz^-2)``; 0.99 of it is the
#: standard margin. ``n_min`` matters: dropping it (the usual spelling, which assumes no
#: material is faster than vacuum) makes a perfectly ordinary ``eps = 0.81`` diverge to 1e281
#: in forty optical cycles and return ``nan``.
COURANT = 0.99

#: Optical cycles simulated, when ``n_periods`` is not given. The pulse has to enter, scatter
#: and leave before the running DFT means anything. **Measured on the benchmark grating**: at 40
#: the field at the transmission monitor is still 61% of its peak and the answer is wrong by
#: 0.09; at 80 the residual is 5e-6 and the answer no longer moves -- 80 to 240 changes the
#: specular efficiency by 1e-6. The residual is reported, and a run too short for the source
#: even to finish emitting is refused outright rather than returned.
N_PERIODS = 80.0

#: 1/e half-width of the source envelope, in optical cycles at the centre wavelength, and the
#: delay before its peak in units of that width. Together they put a hard floor of
#: ``SOURCE_CYCLES * SOURCE_DELAY`` cycles on any useful run, which :func:`_time_loop` refuses
#: below. A *band* narrows this: a Gaussian of temporal width ``w`` has a 1/e frequency
#: half-width of ``1/(pi w)``, so covering a span costs a shorter pulse and a longer run.
SOURCE_CYCLES = 8.0
SOURCE_DELAY = 4.0

#: How many 1/e frequency half-widths of the source the requested band may span. The incident
#: spectrum falls as ``exp(-(df/dfe)**2)``, and every amplitude is divided by it -- so the band
#: edge is where a broadband run stops meaning anything. **Measured**: at 2.4 half-widths out
#: the incident spectrum is 2.4e-7 of its peak and the reported ``R`` is 105. 1.2 keeps the
#: worst edge above ``exp(-1.44) = 0.24`` of the peak.
SOURCE_BAND_WIDTHS = 1.2

#: Smallest incident spectrum, relative to its own peak over the requested band, at which an
#: amplitude is still reported. Below it the division is noise over noise and the order is
#: returned as ``nan`` rather than as a number shaped like an efficiency.
MIN_INCIDENT_FRACTION = 1e-3

#: CPML grading exponent and peak coordinate stretch (Roden & Gedney 2000 sec. 3; Taflove &
#: Hagness eq. 7.61). Measured on the 1D probe with everything else fixed: at ``kappa_max = 1``
#: the reflection runs 1.29e-6 / 9.12e-9 / 9.28e-10 for ``m`` = 3 / 4 / 5, so ``m = 4`` is worth
#: ~140x over the textbook default. ``kappa_max`` above 1 costs a little on the propagating
#: reflection this probe measures (1.29e-6 / 1.36e-6 / 1.63e-6 at ``kappa`` 1 / 2 / 5) and is
#: classically justified by grazing and evanescent absorption, which the probe cannot see; 2 is
#: the compromise, and this is the one constant here not settled by a measurement.
PML_ORDER = 4.0
PML_KAPPA_MAX = 2.0

#: The CFS pole, **as a multiple of the operating frequency** -- which is what makes it
#: dimensionless and the solver scale-free. Measured optimum is flat between 1 and 2 in every
#: medium (1.15e-6 at 2.0 in air, glass and n=3.5), rising to 3.9e-6 at 4 and 2.1e-4 at the 7.16
#: the old dimensional spelling worked out to at 900 nm. Zero is *not* safe: the shifted pole is
#: what absorbs the evanescent field a grating radiates into the layer.
PML_ALPHA_RELATIVE = 1.5

#: Round-trip amplitude reflection the conductivity grading is aimed at.
PML_TARGET_REFLECTION = 1e-8

class UnsettledFieldWarning(UserWarning):
    """The run ended with the field still ringing, so the DFT is of a truncated signal.

    Emitted when the field at the transmission monitor has not decayed to
    :data:`SETTLED_RESIDUAL` of its peak. A truncated running DFT returns a number shaped
    exactly like an efficiency, so this is the one diagnostic a caller must not have to go
    looking for.

    **It matters far more at oblique incidence**, where the scattered wave leaves at an angle
    and a nearly-grazing order can take several times as long to clear as a normal-incidence
    run does. Measured on the benchmark grating at 15 degrees: 60 optical cycles leaves a
    residual of 5.9e-2 and an answer wrong by 3.4e-2, while 120 cycles leaves 4.5e-3. Raise
    ``n_periods`` until it stops warning.
    """


#: Residual -- the field at the transmission monitor as a fraction of its own peak -- above
#: which :class:`UnsettledFieldWarning` fires. Measured against the RCWA answer: a residual of
#: 4.4e-4 costs nothing, 4.5e-3 costs 1.4e-2 on a near-Rayleigh order, and 1.1e-1 costs 1.7e-2
#: outright. 1e-2 sits inside the band where the error is still second order in the grid.
SETTLED_RESIDUAL = 1e-2

#: Largest de-referencing gain an evanescent order's amplitude may carry before it is reported
#: as ``nan``. An order evanescent in the superstrate has decayed by ``exp(k0 |Im kz| |z_ref|)``
#: by the time it reaches the monitor, and undoing that is a pure amplification of whatever is
#: left -- which past some depth is float64 round-off. Measured against the RCWA amplitudes on
#: a duty-0.3 grating: a gain of 1.1e8 still gives 6% accuracy, 4.5e12 is wrong by 5x, and
#: 1.4e17 by 3e5. The efficiency is identically zero for these orders either way, so nothing
#: downstream of ``R`` can notice.
MAX_DEREFERENCE_GAIN = 1e10


# ---------------------------------------------------------------------------------------
# scope
# ---------------------------------------------------------------------------------------

def _refuse_out_of_scope(structure, mu, theta, phi, tetm, incidence_side, lattice,
                         top, bot):
    """Every input this formulation cannot answer, refused by name.

    Refused rather than approximated: each of these returns a *plausible* number if it is
    quietly ignored, and a plausible wrong number is the failure mode this whole module exists
    to guard the RCWA path against.
    """
    if tetm is None:
        raise ValueError(
            "fdtd: tetm must be 'te' or 'tm'. This is the pure TE/TM formulation -- the "
            "time-domain twin of `solve_1d` -- and like it, a `psi` / `chi` / `polarization` "
            "incidence is a different problem, not a different spelling of this one.")
    name = tetm.value if hasattr(tetm, 'value') else str(tetm).lower()
    if name not in ('te', 'tm'):
        raise ValueError(f"fdtd: tetm must be 'te' or 'tm', got {tetm!r}.")

    if not hasattr(structure, 'shape'):
        raise ValueError(
            'fdtd: structure must be a permittivity raster, shape (n_layers, 1, Nx). The '
            'analytic shape encoding is not read here -- rasterising it would put a second '
            'geometry builder beside `draw.slice_structure`, which is NumPy-only, square-only '
            'and refuses a traced shape parameter, so the two would disagree about the same '
            'scene. Pass the raster you want solved.')
    if structure.ndim != 3 or structure.shape[1] != 1:
        raise ValueError(
            f'fdtd: structure is {tuple(structure.shape)}. This is a 2D solve in the x-z '
            f'plane, so the grating must be invariant along y: shape (n_layers, 1, Nx). A '
            f'crossed grating needs a 3D FDTD, which this is not.')

    values = np.concatenate([np.asarray(structure, dtype=complex).ravel(),
                             np.asarray(top, dtype=complex).ravel(),
                             np.asarray(bot, dtype=complex).ravel()])
    if np.any(np.abs(np.imag(values)) > 0):
        raise ValueError(
            'fdtd: the scene has a complex permittivity. A single complex number describes a '
            'material at ONE frequency, and a time-domain run carries every frequency in its '
            'band at once -- so there is no constant this solve could use that would be right '
            'across the band. Give the dispersion as poles instead: `structure` holds the '
            'real eps_infinity, `dispersion` marks which cells are dispersive, and `poles` '
            'carries the Drude/Lorentz model. That IS the frequency dependence, rather than '
            'one sample of it.')
    if np.any(np.real(values) <= 0.0):
        raise ValueError(
            f'fdtd: the scene has a permittivity of {np.real(values).min()!r}. A '
            f'non-positive permittivity is a metal below its plasma frequency, which needs the '
            f'same dispersion the complex case does; it also has no real wave speed, so there '
            f'is no time step that makes the update stable.')

    if mu is not None:
        raise ValueError(
            'fdtd: mu is not read here. That is parity with the path this mirrors, not an '
            'omission: the RCWA side refuses it on the same scene, at engine.py:2334 -- "mu is '
            'implemented on the 2D path only ... the 1D and conical operators carry no '
            'permeability slot at all". Neither path answers a magnetic 1D grating today.')

    if theta is not None and np.imag(np.asarray(theta, dtype=complex)) != 0.0:
        raise ValueError(
            f'fdtd: theta={theta!r} is complex. Evanescent incidence has no propagating '
            f'incident wave to normalise against, and the TFSF plane injects one by '
            f'construction. The RCWA path answers this; see docs/branch-cuts.md.')
    if phi is not None and float(np.real(phi)) != 0.0:
        raise ValueError(
            f'fdtd: phi={phi!r}. A conical mount couples the two polarisations, and this is '
            f'the pure TE/TM formulation -- the same split `solve_1d` and `solve_1d_conical` '
            f'make. The classical mount (phi = 0) leaves TE and TM decoupled at any theta, '
            f'which is what makes oblique incidence answerable here at all.')
    if incidence_side not in (None, 'top'):
        raise ValueError(
            f"fdtd: incidence_side={incidence_side!r}. Only 'top' is implemented; the mirror "
            f"the ETM applies for backside illumination has no counterpart here yet.")
    if lattice is not None:
        raise ValueError(
            'fdtd: a non-axis-aligned lattice is not read here. The Yee grid is Cartesian, so '
            'a skewed cell would need the metric carried through every update.')
    return name


# ---------------------------------------------------------------------------------------
# geometry -> the Yee grid
# ---------------------------------------------------------------------------------------

def _overlap(edges, lo, hi):
    """Length of the overlap of ``[lo, hi]`` with each interval described by ``edges``."""
    return np.maximum(np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo), 0.0)


def _cell_average(values, edges, centres, width, harmonic):
    """Exact cell average of a piecewise-constant profile.

    ``values[j]`` holds on ``[edges[j], edges[j+1]]``; each output cell is
    ``[c - width/2, c + width/2]``. Exact rather than oversampled because the geometry here is
    piecewise constant with axis-aligned interfaces, where the integral is a sum of overlaps
    and an oversampled estimate would only add a second discretisation to converge.

    ``width = 0`` point-samples instead, which is what ``subpixel=False`` asks for.
    """
    out = np.empty(len(centres), dtype=np.result_type(np.asarray(values).dtype, np.float64))
    if width <= 0.0:
        index = np.clip(np.searchsorted(edges, centres, side='right') - 1, 0, len(values) - 1)
        return np.asarray(values)[index]
    v = 1.0 / values if harmonic else values
    for j, c in enumerate(centres):
        w = _overlap(edges, c - 0.5 * width, c + 0.5 * width)
        out[j] = (w * v).sum() / w.sum()
    return 1.0 / out if harmonic else out


def _cell_average_periodic(values, period, centres, width, harmonic):
    """:func:`_cell_average` on a periodic axis, where a cell may straddle the cell edge.

    Three images are laid down rather than one. That is enough exactly while the averaging
    cell is narrower than the period, which it is: the cell is one grid step and the grid has
    at least one step per period by construction.
    """
    n = len(values)
    e = np.linspace(0.0, period, n + 1)
    vals = np.concatenate([values, values, values])
    edges = np.concatenate([e[:-1] - period, e[:-1], e + period])
    return _cell_average(vals, edges, np.asarray(centres), width, harmonic)


def component_property(raster, thickness, top, bot, period, z0, dz, nz, dx, nx,
                       z_half, x_half, harmonic_axis, subpixel=True):
    """A material property averaged onto one Yee component's cells, shape ``(nz, nx)``.

    ``raster`` is ``(n_layers, Nx)`` -- wavebit's ``structure`` with its length-1 ``y`` axis
    dropped -- and ``z0`` is the z coordinate of grid index 0. ``z_half`` / ``x_half`` say
    where the component sits inside the Yee cell, so the averaging volume is the one centred
    on the component itself.

    **The averaging rule.** ``harmonic_axis`` is the axis the component points along, and the
    property is averaged harmonically along it and arithmetically along the others. Subpixel
    smoothing (Farjadpour 2006, Kottke 2008) replaces the stepped material with an effective
    *tensor*: arithmetic mean in the directions tangential to the interface, harmonic mean
    along its normal. For an interface normal to ``x`` that tensor is
    ``diag(harmonic, arithmetic, arithmetic)``, and ``E_x`` -- the only component that reads
    ``eps_xx`` -- is precisely the one this rule averages harmonically along ``x``. The other
    components see a constant along ``x`` there, where harmonic and arithmetic coincide. So
    the rule reproduces the full tensor exactly for axis-aligned interfaces, and approximates
    it for a slanted one, where the true normal is not a grid axis. The geometry this function
    accepts -- a z stack of piecewise-constant x rasters -- has no interface that is not
    axis-aligned, so there is no off-diagonal term to drop.

    ``subpixel=False`` point-samples instead, which is the naive staircase and the honest
    control: on a slab whose interfaces genuinely cut cells it costs an order, 3.4e-3 against
    2.7e-4 at a 5 nm grid and falling as ``dz`` rather than ``dz^2``.
    """
    if harmonic_axis not in (None, 'x', 'z'):
        raise ValueError(
            f"component_property: harmonic_axis={harmonic_axis!r}. It names the axis the "
            f"component points along -- 'x', 'z', or None for one with no in-plane axis. A "
            f"typo here silently selects the arithmetic rule and costs an order of accuracy "
            f"without failing.")
    width_z = dz if subpixel else 0.0
    width_x = dx if subpixel else 0.0

    cuts = np.concatenate([[0.0], np.cumsum(np.asarray(thickness, dtype=float))])
    span = 1e6 * max(dz, 1.0)
    z_edges = np.concatenate([[cuts[0] - span], cuts, [cuts[-1] + span]])
    n_layers = raster.shape[0]

    zc = z0 + (np.arange(nz) + (0.5 if z_half else 0.0)) * dz
    xc = (np.arange(nx) + (0.5 if x_half else 0.0)) * dx

    # Along x first, inside each layer; the two half-spaces are uniform in x.
    per_layer = np.empty((n_layers + 2, nx), dtype=complex)
    per_layer[0] = top
    per_layer[-1] = bot
    for index in range(n_layers):
        per_layer[index + 1] = _cell_average_periodic(
            np.asarray(raster[index], dtype=complex), period, xc, width_x,
            harmonic=(harmonic_axis == 'x'))

    # Then along z, across the layer boundaries.
    harmonic_z = (harmonic_axis == 'z')
    if width_z <= 0.0:
        index = np.clip(np.searchsorted(z_edges, zc, side='right') - 1, 0, n_layers + 1)
        return per_layer[index]
    source = 1.0 / per_layer if harmonic_z else per_layer
    out = np.empty((nz, nx), dtype=complex)
    for k, c in enumerate(zc):
        w = _overlap(z_edges, c - 0.5 * width_z, c + 0.5 * width_z)
        out[k] = (w[:, None] * source).sum(axis=0) / w.sum()
    return 1.0 / out if harmonic_z else out


# ---------------------------------------------------------------------------------------
# CPML
# ---------------------------------------------------------------------------------------

def cpml_coefficients(depth, thickness, dt, wavelength, eps_r):
    """``(b, a, kappa)`` of the CFS-PML at a given depth into a layer (Roden & Gedney 2000).

    ``depth`` is measured in length units from the inner face.

    **Both material-dependent quantities here were wrong in the obvious direction once.**
    ``sigma`` in a *stretched coordinate* is normalised by the vacuum permittivity, so the
    attenuation a wave accumulates is ``n * integral(sigma)`` and a fixed round-trip target
    needs ``sigma_max`` proportional to ``1/sqrt(eps_r)``. Taflove & Hagness eq. 7.61 carries
    ``sqrt(eps_r)`` the other way up because there it is a *material* conductivity acting
    against the local permittivity; transplanting that spelling into the stretch overshoots by
    exactly ``eps_r``, which on an ``n = 3.5`` substrate reflects 5.1e-2 instead of 1.9e-3 and
    returns ``T = 1.037`` from a lossless stack. And ``alpha`` is a frequency, so it is set
    relative to the operating one -- see :data:`PML_ALPHA_RELATIVE`.

    ``alpha`` grades *opposite* to ``sigma``, largest at the inner face: that is what makes
    this a complex-frequency-shifted PML and what lets it absorb evanescent fields, which a
    grating close to its boundary produces in quantity.
    """
    sigma_max = (-(PML_ORDER + 1.0) * np.log(PML_TARGET_REFLECTION)
                 / (2.0 * np.sqrt(eps_r) * thickness))
    alpha_max = PML_ALPHA_RELATIVE * 2.0 * np.pi / wavelength

    f = np.clip(depth / thickness, 0.0, 1.0)
    sigma = sigma_max * f ** PML_ORDER
    kappa = 1.0 + (PML_KAPPA_MAX - 1.0) * f ** PML_ORDER
    alpha = alpha_max * (1.0 - f)

    b = np.exp(-(sigma / kappa + alpha) * dt)
    denom = kappa * (sigma + kappa * alpha)
    safe = np.where(denom == 0.0, 1.0, denom)
    return b, np.where(denom == 0.0, 0.0, sigma * (b - 1.0) / safe), kappa


def cpml_profiles(nz, n_pml, dz, dt, wavelength, eps_top, eps_bot):
    """CPML coefficients per z index, at node (``nz``) and half (``nz-1``) positions.

    Outside the two layers the triple is ``(1, 0, 1)``, which makes every stretched update
    reduce to the plain Yee one -- so the loop runs a single expression everywhere rather than
    branching on whether a row is inside a layer.
    """
    thickness = n_pml * dz
    z_max = (nz - 1) * dz

    def build(offset, n):
        z = (np.arange(n) + offset) * dz
        d_top, d_bot = thickness - z, z - (z_max - thickness)
        depth = np.maximum(np.maximum(d_top, d_bot), 0.0)
        eps_side = np.where(d_top >= d_bot, eps_top, eps_bot)
        b, a, kappa = cpml_coefficients(depth, thickness, dt, wavelength, eps_side)
        outside = depth <= 0.0
        return {'b': np.where(outside, 1.0, b), 'a': np.where(outside, 0.0, a),
                'kappa': np.where(outside, 1.0, kappa)}

    return build(0.0, nz), build(0.5, nz - 1)


# ---------------------------------------------------------------------------------------
# dispersion -- the auxiliary differential equation
# ---------------------------------------------------------------------------------------

class _PoleSet:
    """The Drude/Lorentz state carried alongside one electric field component.

    **One equation covers both models.** A pole obeys

        d2P/dt2 + gamma dP/dt + omega0**2 P = f E

    which in ``e^{-iwt}`` gives ``P = f E / (omega0**2 - w**2 - i gamma w)``, so

        eps(w) = eps_inf + sum_p  f_p / (omega0_p**2 - w**2 - i gamma_p w)

    At ``omega0 = 0`` that is Drude, ``eps_inf - f/(w**2 + i gamma w)`` with ``f = wp**2``, and
    the sign works out so ``Im(eps) > 0`` is loss -- which is wavebit's convention and not a
    free choice (``docs/conventions.md``). At ``omega0 != 0`` it is Lorentz. A metal is a Drude
    pole plus however many Lorentz poles its interband structure needs.

    Central differences on both derivatives make the recursion explicit: ``P^{n+1}`` reads
    ``E^n``, never ``E^{n+1}``, so nothing has to be solved simultaneously.

    ``fill`` is the fraction of each Yee cell the dispersive material occupies. Weighting ``f``
    by it is exact in the same limit the arithmetic permittivity average is exact -- tangential
    to the interface -- and is an approximation normal to one, where the harmonic rule would
    apply and does not transfer to a frequency-dependent material. See
    :func:`component_property`.
    """

    def __init__(self, shape, dt, poles, fills, dtype):
        self.step_gain, self.hold, self.drive = [], [], []
        for (omega0, gamma, strength), fill in zip(poles, fills):
            scale = 1.0 / dt ** 2 + 0.5 * gamma / dt
            self.step_gain.append((2.0 / dt ** 2 - omega0 ** 2) / scale)
            self.hold.append((1.0 / dt ** 2 - 0.5 * gamma / dt) / scale)
            self.drive.append(strength * np.asarray(fill) / scale)
        self.p = [np.zeros(shape, dtype=dtype) for _ in poles]
        self.previous = [np.zeros(shape, dtype=dtype) for _ in poles]

    def advance(self, e):
        """Take every pole to ``n+1`` using ``E^n``; return ``sum(P^{n+1} - P^n)``."""
        total = 0.0
        for k in range(len(self.p)):
            nxt = (self.step_gain[k] * self.p[k] - self.hold[k] * self.previous[k]
                   + self.drive[k] * e)
            total = total + (nxt - self.p[k])
            self.previous[k], self.p[k] = self.p[k], nxt
        return total


def permittivity_at(poles, wavelength, eps_infinity=1.0):
    """``eps(w)`` of a pole list, for stating what a scene's material actually is.

    The closed form the time stepping is discretising, in one place, so a test can compare
    against it rather than against a second implementation of the same sum.
    """
    omega = 2.0 * np.pi / wavelength
    eps = complex(eps_infinity)
    for omega0, gamma, strength in poles:
        eps += strength / (omega0 ** 2 - omega ** 2 - 1j * gamma * omega)
    return eps


# ---------------------------------------------------------------------------------------
# the source, and the 1D auxiliary incident field
# ---------------------------------------------------------------------------------------

def gaussian_pulse(t, f0, width, t0):
    """A Gaussian-modulated sine, the standard broadband FDTD excitation."""
    return np.sin(2.0 * np.pi * f0 * (t - t0)) * np.exp(-((t - t0) / width) ** 2)


class _Incident1D:
    """The incident plane wave, reduced to one dimension.

    Taflove & Hagness sec. 5.8. Generating the incident field on a grid sharing ``dz`` and
    ``dt`` with the 2D one makes its numerical dispersion identical along z, so the
    total-field/scattered-field cancellation is limited by round-off rather than by the
    dispersion mismatch an analytic plane wave would leave at the boundary. Measured on an
    empty grid, the residual in the scattered-field region is 1.2e-17 of the incident
    amplitude; a sign or index mutant anywhere in the coupling drives it to 1e-2 or worse, so
    the check has ~15 orders of resolving power and :mod:`tests.test_fdtd` runs it.

    **Why one dimension is still enough at oblique incidence.** The medium here is uniform
    along x, and ``exp(i kx x)`` is an eigenvector of the *discrete* shift, so a field of that
    form stays of that form on the grid exactly. Every x derivative therefore collapses to a
    single multiplier -- ``kx_op = 2i sin(kx dx / 2) / dx``, which is the discrete operator
    the 2D grid applies and not the continuum ``i kx``. That is what keeps the cancellation at
    round-off instead of at the dispersion mismatch between the two grids.

    The third component appears only at oblique incidence and it is a **different** component
    for the two polarisations: ``Hz`` for TE and ``Ez`` for TM. The ``(E, H, eps, mu)``
    duality that makes one 1D system serve both at normal incidence does not survive the
    transverse term, because ``eps != mu``. At ``kx = 0`` both branches drop it and the two
    reduce to the same pair, ``(Ey, Hx)`` and ``(Ex, -Hy)`` -- and that minus sign has **two**
    consumers, the TFSF correction and the incident amplitude the result is normalised by.
    Getting it wrong in only one flips the sign of ``r_p`` while leaving every efficiency
    intact.
    """

    def __init__(self, nz, dz, dt, eps_r, n_pml, k_src, wavelength, f0, width, t0,
                 tetm, kx_op, dtype):
        self.dz, self.dt, self.eps_r, self.k_src = dz, dt, eps_r, k_src
        self.f0, self.width, self.t0 = f0, width, t0
        self.tetm, self.kx_op = tetm, kx_op
        self.e = np.zeros(nz, dtype=dtype)
        self.h = np.zeros(nz - 1, dtype=dtype)
        #: `Hz` on TE and `Ez` on TM; identically zero at normal incidence.
        self.third = np.zeros(nz, dtype=dtype)
        self.node, self.half = cpml_profiles(nz, n_pml, dz, dt, wavelength, eps_r, eps_r)
        self.psi_e = np.zeros(nz, dtype=dtype)
        self.psi_h = np.zeros(nz - 1, dtype=dtype)

    def step_h(self):
        d = (self.e[1:] - self.e[:-1]) / self.dz
        self.psi_h = self.half['b'] * self.psi_h + self.half['a'] * d
        self.h += self.dt * (d / self.half['kappa'] + self.psi_h)
        if self.tetm == 'tm':
            # dh/dt = de/dz - ez*kx_op, from Faraday's y row with h = -Hy.
            self.h -= self.dt * self.third[:-1] * self.kx_op

    def step_e(self, t):
        d = np.zeros_like(self.e)
        d[1:-1] = (self.h[1:] - self.h[:-1]) / self.dz
        self.psi_e = self.node['b'] * self.psi_e + self.node['a'] * d
        curl = d / self.node['kappa'] + self.psi_e
        if self.tetm == 'te':
            # dHz/dt = -Ey*kx_op, and Ey then reads Hz back.
            self.third -= self.dt * self.e * self.kx_op
            curl = curl - self.third * self.kx_op
        else:
            # eps dEz/dt = Hy*kx_op = -h*kx_op.
            self.third[:-1] -= (self.dt / self.eps_r) * self.h * self.kx_op
        self.e[1:-1] += (self.dt / self.eps_r) * curl[1:-1]
        # Additive rather than imposed: a soft source is transparent to whatever comes back
        # through it, which matters because the source sits between the PML and the monitor.
        self.e[self.k_src] += gaussian_pulse(t, self.f0, self.width, self.t0)


# ---------------------------------------------------------------------------------------
# the grid
# ---------------------------------------------------------------------------------------

def _shift_up(a, bloch=1.0):
    """``a[:, i+1]``, with the Bloch phase the cell edge carries.

    At oblique incidence the field is periodic only up to ``exp(i kx Lambda)``, so the one
    column that wraps picks that factor up and the rest of the shift is an ordinary roll.
    ``bloch = 1`` is normal incidence and reduces this to the plain periodic shift.
    """
    return np.concatenate([a[:, 1:], a[:, :1] * bloch], axis=1)


def _shift_down(a, bloch=1.0):
    """``a[:, i-1]``, with the Bloch phase (conjugated -- it wraps the other way)."""
    return np.concatenate([a[:, -1:] * np.conj(bloch), a[:, :-1]], axis=1)


class _Grid:
    """Geometry, timing and CPML shared by the two polarisations.

    The z layout, in node indices, and the reason each plane is where it is::

        0                      top of the grid
        [0, n_pml)             top PML
        k_src                  the 1D auxiliary source. The 2D grid HAS no source: the
                               incident wave enters it only through the TFSF plane
        k_ref                  reflection monitor. Above the TFSF plane, so in the 2D grid it
                               sees only scattered field; below k_src, so in the 1D grid the
                               incident reference it divides by is the FORWARD wave and not
                               the source's own backward lobe
        k_tfsf                 total-field / scattered-field plane
        k_z0                   z = 0, the top interface -- and the plane the incident
                               amplitude is read at, which is what keeps the de-referencing
                               to one leg
        ...                    the layers
                               z = Z, the bottom interface
        k_trn                  transmission monitor
        [nz - n_pml, nz)       bottom PML

    ``Ey`` and ``Ex`` are pinned to zero on the two end rows -- the PEC backing every PML
    stands in front of.
    """

    def __init__(self, wavelength, period, raster, thickness, eps_top, eps_bot,
                 resolution, n_pml, courant, subpixel, theta=0.0,
                 dispersion=None, poles=None):
        self.wavelength = float(wavelength)
        self.period = float(period)
        self.raster = np.asarray(raster, dtype=complex)
        self.thickness = np.asarray(thickness, dtype=float)
        self.eps_top = complex(eps_top)
        self.eps_bot = complex(eps_bot)
        self.subpixel = bool(subpixel)
        self.n_pml = int(n_pml)
        #: Per-raster-cell pole-set id, 0 meaning non-dispersive, and the pole lists it
        #: indexes. Kept beside `raster` rather than inside it because a pole model is not a
        #: number: `structure` holds `eps_infinity` and this says what else the cell carries.
        self.dispersion = None if dispersion is None else np.asarray(dispersion, dtype=int)
        self.poles = dict(poles or {})

        self.nx = max(int(round(self.period * resolution)), 1)
        self.dx = self.period / self.nx
        self.dz = self.dx

        n_top = float(np.sqrt(self.eps_top).real)
        n_bot = float(np.sqrt(self.eps_bot).real)

        # The in-plane wavevector, normalised by k0 exactly as `engine.n_inc` normalises it:
        # `kx_inc = n_inc*sin(theta)*cos(phi)`, and `phi = 0` here. `theta` is in RADIANS,
        # which is wavebit's unit -- measured against the closed-form Fresnel value rather
        # than assumed, because a degrees/radians slip returns a legal answer at a legal angle.
        k0 = 2.0 * np.pi / self.wavelength
        self.theta = float(np.real(0.0 if theta is None else theta))
        self.kx_inc = n_top * float(np.sin(self.theta))
        #: The phase the field picks up crossing one cell edge. `1` is normal incidence, and
        #: the arrays stay REAL there -- a complex dtype would double the cost of every run
        #: for a phase that is identically one.
        #: Real `1.0` and `0.0` rather than `(1+0j)` and `0j` at normal incidence: a complex
        #: literal upcasts every array it touches, which would silently double the cost of
        #: every normal-incidence run for a phase that is identically one.
        self.dtype = complex if self.kx_inc != 0.0 else float
        self.bloch = (np.exp(1j * k0 * self.kx_inc * self.period)
                      if self.dtype is complex else 1.0)

        # A wavelength of clearance in each half-space: far enough that the evanescent orders
        # a grating radiates have decayed before the monitor, near enough that the outgoing
        # leg carrying the analytic-vs-discrete kz mismatch stays short.
        pad_top = self.wavelength / max(n_top, 1e-9)
        pad_bot = self.wavelength / max(n_bot, 1e-9)
        gap = max(int(round(0.25 * pad_top / self.dz)), 3)

        self.k_src = self.n_pml + 2
        self.k_ref = self.k_src + gap
        self.k_tfsf = self.k_ref + gap
        self.k_z0 = self.k_tfsf + max(int(round(pad_top / self.dz)), 3)

        self.total_thickness = float(self.thickness.sum())
        k_bottom = self.k_z0 + int(np.ceil(self.total_thickness / self.dz))
        clear = max(int(round(0.5 * pad_bot / self.dz)), 3)
        self.k_trn = k_bottom + clear
        self.nz = self.k_trn + clear + self.n_pml + 1

        self.z0 = -self.k_z0 * self.dz

        # The stability bound is set by the FASTEST wave in the scene, and speed is `c/n`, so
        # it is the SMALLEST index that binds -- the usual spelling drops this factor because
        # it assumes nothing beats vacuum.
        n_min = float(np.sqrt(np.min(np.real(
            np.concatenate([self.raster.ravel(), [self.eps_top, self.eps_bot]])))))
        self.dt = courant * n_min / np.sqrt(1.0 / self.dx ** 2 + 1.0 / self.dz ** 2)

        #: The DISCRETE x-derivative the auxiliary grid applies, not the continuum `i*kx`:
        #: a field `exp(i kx x)` is an eigenvector of the grid's shift, and the eigenvalue is
        #: this. Using `i*kx` instead leaves the two grids on different dispersion relations
        #: and the TFSF cancellation degrades from round-off to the mismatch between them.
        self.kx_op = (2j * np.sin(0.5 * k0 * self.kx_inc * self.dx) / self.dx
                      if self.dtype is complex else 0.0)

        self.node, self.half = cpml_profiles(self.nz, self.n_pml, self.dz, self.dt,
                                             self.wavelength,
                                             self.eps_top.real, self.eps_bot.real)

    def z_of(self, k):
        """The z coordinate of z index ``k``; ``k`` may carry a half."""
        return self.z0 + k * self.dz

    def eps_for(self, z_half, x_half, harmonic_axis):
        nz = self.nz - 1 if z_half else self.nz
        return component_property(
            self.raster, self.thickness, self.eps_top, self.eps_bot, self.period,
            self.z0, self.dz, nz, self.dx, self.nx, z_half, x_half, harmonic_axis,
            subpixel=self.subpixel)

    def fill_for(self, pole_id, z_half, x_half):
        """What fraction of each of this component's cells carries ``pole_id``.

        The same exact cell average `eps_infinity` gets, run on the indicator of one pole set,
        which is what makes it a fraction. Arithmetic, never harmonic: the harmonic rule is a
        statement about a *static* permittivity normal to an interface and does not carry over
        to a pole strength.
        """
        nz = self.nz - 1 if z_half else self.nz
        indicator = (self.dispersion == pole_id).astype(float)
        return np.real(component_property(
            indicator, self.thickness, 0.0, 0.0, self.period, self.z0, self.dz, nz,
            self.dx, self.nx, z_half, x_half, None, subpixel=self.subpixel))

    def pole_state(self, shape, z_half, x_half, dtype):
        """The :class:`_PoleSet` for one electric component, or ``None`` if the scene is
        non-dispersive."""
        if self.dispersion is None or not self.poles:
            return None
        flat, fills = [], []
        for pole_id in sorted(self.poles):
            fill = self.fill_for(pole_id, z_half, x_half)
            for pole in self.poles[pole_id]:
                flat.append(pole)
                fills.append(fill)
        return _PoleSet(shape, self.dt, flat, fills, dtype) if flat else None


# ---------------------------------------------------------------------------------------
# the time loop
# ---------------------------------------------------------------------------------------

def _time_loop(grid, tetm, n_steps, wavelengths):
    """Step the fields and accumulate the running DFT, at every requested wavelength at once.

    This is the one thing a time-domain solve does that a frequency-domain one cannot: the
    pulse carries every frequency in its band, so a second wavelength costs one more running
    sum over a monitor plane -- O(nx) a step -- against a whole new RCWA solve. The grid, the
    time stepping and the fields are shared outright.
    """
    wavelengths = np.atleast_1d(np.asarray(wavelengths, dtype=float))
    freq = 1.0 / wavelengths                       # c = 1, shape (n_f,)
    centre = float(grid.wavelength)
    # A Gaussian of temporal width `w` has a 1/e frequency half-width of `1/(pi w)`, so a band
    # buys itself a SHORTER pulse. The single-wavelength width is the ceiling: narrowing the
    # band past one wavelength gains nothing and only lengthens the run.
    span = float(np.max(np.abs(freq - 1.0 / centre)))
    width = SOURCE_CYCLES * centre
    if span > 0.0:
        width = min(width, SOURCE_BAND_WIDTHS / (np.pi * span))
    t0 = SOURCE_DELAY * width
    if n_steps * grid.dt < t0 + 2.0 * width:
        raise ValueError(
            f'fdtd: {n_steps} steps is {n_steps * grid.dt / centre:.1f} optical cycles, '
            f'and the source alone is not finished emitting until '
            f'{(t0 + 2.0 * width) / centre:.0f}. A run this short returns a number that '
            f'looks like an efficiency -- measured 0.033 against a true 0.543 at 20 cycles -- '
            f'so it is refused rather than reported with a residual nobody reads.')

    nz, nx, dz, dx, dt = grid.nz, grid.nx, grid.dz, grid.dx, grid.dt
    hb, ha, hk = (grid.half[k][:, None] for k in ('b', 'a', 'kappa'))
    nb, na, nk = (grid.node[k][:, None] for k in ('b', 'a', 'kappa'))
    kt = grid.k_tfsf

    dtype = grid.dtype
    inc = _Incident1D(nz, dz, dt, grid.eps_top.real, grid.n_pml, grid.k_src,
                      centre, 1.0 / centre, width, t0, tetm, grid.kx_op, dtype)

    # The incident field the TFSF plane injects varies across the cell as `exp(i kx x)`; the
    # auxiliary grid carries only its amplitude, so the plane multiplies it back on. `Ey`/`Hx`
    # sit on the x nodes and `Hy`/`Ex` on the half-steps, which is the only difference between
    # the two polarisations here. `ones` rather than `exp(0j)` at normal incidence, so the
    # field arrays stay real.
    k0 = 2.0 * np.pi / centre
    if grid.kx_inc == 0.0:
        x_phase = np.ones(nx)
    else:
        offset = 0.5 if tetm == 'tm' else 0.0
        x_phase = np.exp(1j * k0 * grid.kx_inc * (np.arange(nx) + offset) * dx)

    if tetm == 'te':
        # (Ey, Hx, Hz). Ey sits on the z nodes and the x nodes, and is tangential to every
        # interface a 1D grating has, so it takes no harmonic average.
        eps = grid.eps_for(False, False, None).real
        ey = np.zeros((nz, nx), dtype=dtype)
        hx = np.zeros((nz - 1, nx), dtype=dtype)
        hz = np.zeros((nz, nx), dtype=dtype)
        poles_y = grid.pole_state((nz, nx), False, False, dtype)
    else:
        # (Hy, Ex, Ez). Ex is normal to the tooth's vertical walls, Ez to the horizontal ones.
        eps_x = grid.eps_for(False, True, 'x').real
        eps_z = grid.eps_for(True, False, 'z').real
        hy = np.zeros((nz - 1, nx), dtype=dtype)
        ex = np.zeros((nz, nx), dtype=dtype)
        ez = np.zeros((nz - 1, nx), dtype=dtype)
        poles_x = grid.pole_state((nz, nx), False, True, dtype)
        poles_z = grid.pole_state((nz - 1, nx), True, False, dtype)
    psi_h = np.zeros((nz - 1, nx), dtype=dtype)
    psi_e = np.zeros((nz, nx), dtype=dtype)

    n_f = len(freq)
    dft_ref = np.zeros((n_f, nx), dtype=complex)
    dft_trn = np.zeros((n_f, nx), dtype=complex)
    inc_ref = np.zeros(n_f, dtype=complex)
    inc_trn = np.zeros(n_f, dtype=complex)
    inc_z0 = np.zeros(n_f, dtype=complex)
    peak = tail = 0.0

    for step in range(n_steps):
        # --- H, to level n + 1/2 --------------------------------------------------------
        inc.step_h()
        if tetm == 'te':
            # Yee 1966: dHx/dt = dEy/dz, dHz/dt = -dEy/dx. Only z carries a PML.
            d = (ey[1:] - ey[:-1]) / dz
            psi_h = hb * psi_h + ha * d
            hx += dt * (d / hk + psi_h)
            hz -= dt * (_shift_up(ey, grid.bloch) - ey) / dx
            # TFSF (Umashankar & Taflove 1982): Hx at kt-1/2 is a scattered-field quantity
            # whose stencil reached across the boundary into the total-field Ey[kt].
            hx[kt - 1] -= dt * inc.e[kt] * x_phase / dz
        else:
            # dHy/dt = dEz/dx - dEx/dz.
            d_ez = (_shift_up(ez, grid.bloch) - ez) / dx
            d_ex = (ex[1:] - ex[:-1]) / dz
            psi_h = hb * psi_h + ha * d_ex
            hy += dt * (d_ez - d_ex / hk - psi_h)
            # Ex_inc is `inc.e` outright -- see the sign note on _Incident1D.
            hy[kt - 1] += dt * inc.e[kt] * x_phase / dz

        # --- E, to level n + 1 ----------------------------------------------------------
        inc.step_e((step + 1) * dt)
        if tetm == 'te':
            # eps dEy/dt = dHx/dz - dHz/dx.
            d = np.zeros((nz, nx), dtype=dtype)
            d[1:-1] = (hx[1:] - hx[:-1]) / dz
            psi_e = nb * psi_e + na * d
            curl = (d / nk + psi_e
                    - (hz - _shift_down(hz, grid.bloch)) / dx)
            # eps_inf dE/dt = curl H - dP/dt. `advance` reads E at level n and hands back the
            # increment every pole contributes, which is what makes the pair explicit.
            drive = dt * curl
            if poles_y is not None:
                drive = drive - poles_y.advance(ey)
            ey[1:-1] += drive[1:-1] / eps[1:-1]
            ey[kt] -= dt * inc.h[kt - 1] * x_phase / dz / eps[kt]
        else:
            # eps dEx/dt = -dHy/dz ; eps dEz/dt = dHy/dx.
            d = np.zeros((nz, nx), dtype=dtype)
            d[1:-1] = -(hy[1:] - hy[:-1]) / dz
            psi_e = nb * psi_e + na * d
            drive = dt * (d / nk + psi_e)
            if poles_x is not None:
                drive = drive - poles_x.advance(ex)
            ex[1:-1] += drive[1:-1] / eps_x[1:-1]
            # Hy_inc is MINUS inc.h, which is where this sign comes from.
            ex[kt] -= dt * inc.h[kt - 1] * x_phase / dz / eps_x[kt]
            drive_z = dt * (hy - _shift_down(hy, grid.bloch)) / dx
            if poles_z is not None:
                drive_z = drive_z - poles_z.advance(ez)
            ez += drive_z / eps_z

        # --- monitors -------------------------------------------------------------------
        # The e^{-iwt} phasor is recovered with an exp(+i w t) kernel; `np.fft.fft` over the
        # time axis applies the opposite one and would conjugate every amplitude.
        if tetm == 'te':
            f_ref, f_trn = ey[grid.k_ref], ey[grid.k_trn]
            i_ref, i_trn, i_z0 = (inc.e[grid.k_ref], inc.e[grid.k_trn], inc.e[grid.k_z0])
            t_now = (step + 1) * dt
        else:
            f_ref, f_trn = hy[grid.k_ref], hy[grid.k_trn]
            i_ref, i_trn, i_z0 = (-inc.h[grid.k_ref], -inc.h[grid.k_trn],
                                  -inc.h[grid.k_z0])
            t_now = (step + 0.5) * dt
        kernel = np.exp(2j * np.pi * freq * t_now) * dt          # (n_f,)
        dft_ref = dft_ref + kernel[:, None] * f_ref[None, :]
        dft_trn = dft_trn + kernel[:, None] * f_trn[None, :]
        inc_ref += kernel * i_ref
        inc_trn += kernel * i_trn
        inc_z0 += kernel * i_z0

        amplitude = float(np.abs(f_trn).max())
        peak = max(peak, amplitude)
        tail = amplitude

    # The monitor's PHYSICAL z, not its index: `Hy` lives half a cell below the node, and
    # reading the node coordinate instead leaves a `k0*kz*dz/2` phase in every TM amplitude.
    # It cancels in |r|**2, so the efficiencies converge at second order while the amplitudes
    # do not -- measured 4.3e-2 on `r_p` at a 10 nm grid against 1.4e-3 on `R`, falling only
    # as dz**0.6.
    offset = 0.5 if tetm == 'tm' else 0.0
    return {'dft_ref': dft_ref, 'dft_trn': dft_trn, 'tetm': tetm,
            'wavelengths': wavelengths,
            'inc_ref': inc_ref, 'inc_trn': inc_trn, 'inc_z0': inc_z0,
            'z_ref': grid.z_of(grid.k_ref + offset),
            'z_trn': grid.z_of(grid.k_trn + offset),
            'z_inc': grid.z_of(grid.k_z0 + offset),
            'residual': tail / peak if peak > 0.0 else 0.0}


# ---------------------------------------------------------------------------------------
# order extraction
# ---------------------------------------------------------------------------------------

def _half_space_kz(eps, kx):
    """``kz`` on wavebit's half-space branch: the principal root of ``eps - kx**2``."""
    return np.sqrt(np.asarray(eps, dtype=complex) - np.asarray(kx, dtype=complex) ** 2)


def _propagating_half_width(wavelength, period, eps_top, eps_bot):
    """The largest ``|m|`` that propagates in either half-space, so the default reports them."""
    n_max = max(abs(np.sqrt(complex(eps_top))), abs(np.sqrt(complex(eps_bot))))
    return int(np.floor(n_max * period / wavelength))


def _extract(grid, out, half_width, index):
    """Per-order Fresnel amplitudes and efficiencies, in ``solve_1d``'s contract.

    The amplitudes are referenced where wavebit references them -- ``r`` at the top interface,
    ``t`` at the bottom.

    **The incident amplitude is read at ``z = 0``, not at the monitor.** Dividing by the
    incident *at the monitor* removes the outgoing leg's analytic-vs-discrete ``kz`` mismatch
    and adds the incoming leg's, for a net factor of two rather than the cancellation it looks
    like: measured against the closed form with only ``pad_top`` varying, the phase residual
    tracks ``2(k0 - kappa_yee)*z_ref`` at a ratio of 0.96 and the one-leg form at 1.92. Reading
    the incident where the amplitude is *referenced* leaves exactly one leg. ``|r|`` is
    unaffected either way, so no efficiency test can see this.
    """
    nx = grid.nx
    wavelength = float(out['wavelengths'][index])
    if half_width >= nx // 2:
        raise ValueError(
            f'fdtd: {2 * half_width + 1} orders asked of a {nx}-cell grid. The x transform '
            f'resolves |m| < {nx // 2}; beyond that an order aliases onto another one and comes '
            f'back with a plausible amplitude (and, on TM, a flipped sign from the half-cell '
            f'sample offset). Raise `resolution`, or ask for fewer orders.')

    m = np.arange(-half_width, half_width + 1)
    tetm = out['tetm']
    # Hy is sampled at the x half-step, Ey at the node.
    xs = (np.arange(nx) + (0.5 if tetm == 'tm' else 0.0)) * grid.dx
    k0 = 2.0 * np.pi / wavelength
    # `grid.kx_inc` is normalised by k0 at the CENTRE wavelength, because that is where the
    # Bloch phase was built. The physical in-plane wavevector is what the grid actually holds
    # fixed, so at any other wavelength in the band the normalised value scales with it --
    # which is the same statement as the polar angle drifting across the band. Reusing the
    # centre's value costs 2.3e-2 at a band edge against 3.6e-3 at the centre, and is
    # invisible at normal incidence, where it is zero either way.
    kx_inc = grid.kx_inc * wavelength / grid.wavelength
    kx = kx_inc + m * wavelength / grid.period
    # Each order carries `exp(i kx_m x)`, and `kx_m` includes the incident in-plane
    # wavevector at oblique incidence -- so the kernel demodulates by the full `kx_m`, not by
    # the grating harmonic alone.
    kernel = np.exp(-1j * k0 * kx[:, None] * xs[None, :]) / nx

    n_top = np.sqrt(grid.eps_top)
    n_bot = np.sqrt(grid.eps_bot)
    kz_top = _half_space_kz(grid.eps_top, kx)
    kz_bot = _half_space_kz(grid.eps_bot, kx)
    kz_inc = _half_space_kz(grid.eps_top, kx_inc)

    z_ref, z_trn, z_inc = out['z_ref'], out['z_trn'], out['z_inc']
    # The incident amplitude at z = 0, from the plane it was actually sampled on.
    a_inc = out['inc_z0'][index] * np.exp(-1j * k0 * kz_inc * z_inc)

    # Every amplitude is divided by the incident spectrum, so the band edge -- where that
    # spectrum has decayed into round-off -- is where a broadband run stops meaning anything.
    # Measured with no guard: 2.4e-7 of the peak incident returned `R = 105`.
    peak = float(np.max(np.abs(out['inc_z0'])))
    if abs(out['inc_z0'][index]) < MIN_INCIDENT_FRACTION * peak:
        raise ValueError(
            f'fdtd: at {wavelength:g} the incident spectrum is '
            f'{abs(out["inc_z0"][index]) / peak:.1e} of its peak over the requested band, '
            f'below the {MIN_INCIDENT_FRACTION:g} floor. Every amplitude is divided by it, so '
            f'what comes back there is noise over noise -- measured R = 105 on a scene whose '
            f'true R is 0.04. Narrow the band, or move `wavelength` toward it.')

    r = (kernel @ out['dft_ref'][index]) / a_inc * np.exp(1j * k0 * kz_top * z_ref)
    t = ((kernel @ out['dft_trn'][index]) / a_inc
         * np.exp(-1j * k0 * kz_bot * (z_trn - grid.total_thickness)))

    # An order evanescent in a half-space arrives at the monitor already decayed; undoing that
    # amplifies whatever is left, which past some depth is round-off. `R` is identically zero
    # for these, so nothing downstream would notice a finite, plausible, meaningless number.
    r = np.where(np.exp(k0 * np.abs(np.imag(kz_top)) * abs(z_ref)) > MAX_DEREFERENCE_GAIN,
                 np.nan, r)
    t = np.where(np.exp(k0 * np.abs(np.imag(kz_bot))
                        * abs(z_trn - grid.total_thickness)) > MAX_DEREFERENCE_GAIN,
                 np.nan, t)

    # solve_1d.py:316-322. On the reflection side the two media are the same one, so the p
    # channel's `n**2` cancels and both polarisations read Re(kz_top/kz_inc).
    reflected = np.nan_to_num(np.real(r * np.conj(r)), nan=0.0) * np.real(kz_top / kz_inc)
    if tetm == 'te':
        transmitted = (np.nan_to_num(np.real(t * np.conj(t)), nan=0.0)
                       * np.real(kz_bot / kz_inc))
        amplitudes = (r, np.zeros_like(r), t, np.zeros_like(t))
        efficiencies = (reflected, np.zeros_like(reflected),
                        transmitted, np.zeros_like(transmitted))
    else:
        transmitted = (np.nan_to_num(np.real(t * np.conj(t)), nan=0.0)
                       * np.real(kz_bot / n_bot ** 2 / (kz_inc / n_top ** 2)))
        # The p channel's primary amplitude is magnetic; t_E = t_H * n_top/n_bot
        # (solve_1d.py:339-341).
        amplitudes = (np.zeros_like(r), r, np.zeros_like(t), t * (n_top / n_bot))
        efficiencies = (np.zeros_like(reflected), reflected,
                        np.zeros_like(transmitted), transmitted)

    r_s, r_p, t_s, t_p = (a[None, :] for a in amplitudes)
    R_s, R_p, T_s, T_p = (e[None, :] for e in efficiencies)
    return ResultSub(r_s, r_p, t_s, t_p, R_s + R_p, R_s, R_p, T_s + T_p, T_s, T_p)


# ---------------------------------------------------------------------------------------
# the entry point
# ---------------------------------------------------------------------------------------

def _check_dispersion(dispersion, poles, raster):
    """Validate the pole model, or return ``(None, {})`` for a non-dispersive scene.

    Refused rather than coerced, on the same grounds as everything else here: a pole set that
    is silently dropped returns a *lossless* answer to a lossy question, which is exactly the
    shape of number this module exists not to produce.
    """
    if dispersion is None and not poles:
        return None, {}
    if dispersion is None or not poles:
        raise ValueError(
            'fdtd: `dispersion` and `poles` come as a pair -- one marks which cells are '
            'dispersive and the other says what they are. One without the other describes '
            'nothing.')

    dispersion = np.asarray(dispersion, dtype=int)
    if dispersion.ndim == 3 and dispersion.shape[1] == 1:
        dispersion = dispersion[:, 0, :]
    if dispersion.shape != raster.shape:
        raise ValueError(
            f'fdtd: dispersion is {dispersion.shape} and structure is {raster.shape}. They '
            f'are two fields sampled on one raster, so they share a shape exactly.')

    named = set(int(v) for v in np.unique(dispersion)) - {0}
    missing = sorted(named - set(int(k) for k in poles))
    if missing:
        raise ValueError(
            f'fdtd: dispersion marks cells with id {missing} and `poles` has no entry for '
            f'them. Id 0 means non-dispersive; every other id has to name a model.')

    checked = {}
    for key, entries in poles.items():
        for index, entry in enumerate(entries):
            if len(entry) != 3:
                raise ValueError(
                    f'fdtd: poles[{key}][{index}] is {entry!r}. A pole is '
                    f'(omega0, gamma, strength) -- omega0 = 0 is Drude, and the units are the '
                    f'reciprocal of whatever `wavelength` is in, since c = 1 here.')
            omega0, gamma, strength = (float(np.real(v)) for v in entry)
            if omega0 < 0.0 or gamma < 0.0:
                raise ValueError(
                    f'fdtd: poles[{key}][{index}] has omega0={omega0} gamma={gamma}. Both are '
                    f'frequencies and neither can be negative; a negative gamma is gain, which '
                    f'this solve has no stable time step for.')
            if not np.isfinite([omega0, gamma, strength]).all():
                raise ValueError(f'fdtd: poles[{key}][{index}] is not finite.')
        checked[int(key)] = [tuple(float(np.real(v)) for v in e) for e in entries]
    return dispersion, checked


def solve_fdtd(structure, thickness, top, bot, period, wavelength, tetm,
               theta=0.0, phi=None, mu=None, incidence_side='top', lattice=None,
               resolution=None, n_pml=PML_CELLS, n_periods=None, courant=COURANT,
               subpixel=True, truncation_order=None, return_diagnostics=False,
               dispersion=None, poles=None):
    """Solve one TE or TM incidence in the time domain; returns a :class:`Result`.

    Every keyword this does not implement is refused by name rather than ignored -- see
    :func:`_refuse_out_of_scope`. The module docstring says why NumPy is the only backend.
    """
    name = _refuse_out_of_scope(structure, mu, theta, phi, tetm, incidence_side, lattice,
                                top, bot)

    period_x = float(period[0] if hasattr(period, '__len__') else period)
    raster = np.asarray(structure, dtype=complex)[:, 0, :]
    eps_top = complex(np.asarray(top, dtype=complex))
    eps_bot = complex(np.asarray(bot, dtype=complex))

    # A sequence asks for a spectrum out of ONE run -- the thing a time-domain solve does that
    # a frequency-domain one cannot. The centre sets the source and, at oblique incidence, the
    # in-plane wavevector; see the `angles` entry in the diagnostics for what that means.
    band = np.atleast_1d(np.asarray(wavelength, dtype=float))
    if band.size < 1:
        raise ValueError('fdtd: wavelength is empty.')
    single = np.ndim(wavelength) == 0
    centre = float(0.5 * (band.min() + band.max()))
    wavelength = centre

    if resolution is None:
        # The cell size follows the largest index in the scene, and a pole model can raise
        # that a long way above `sqrt(eps_infinity)` -- a Drude metal at 900 nm reaches n = 6.5
        # where its eps_infinity is 1. Sizing the grid off eps_infinity alone would leave the
        # metal at three cells per wavelength.
        indices = [abs(np.sqrt(eps_top)), abs(np.sqrt(eps_bot)),
                   float(np.sqrt(np.abs(raster)).max())]
        for entries in (poles or {}).values():
            indices.append(abs(np.sqrt(permittivity_at(entries, float(np.min(band)), 1.0))))
        resolution = CELLS_PER_WAVELENGTH * max(indices) / wavelength
    n_periods = N_PERIODS if n_periods is None else float(n_periods)

    dispersion, poles = _check_dispersion(dispersion, poles, raster)
    grid = _Grid(wavelength, period_x, raster, thickness, eps_top, eps_bot,
                 resolution, n_pml, courant, subpixel, theta=theta,
                 dispersion=dispersion, poles=poles)
    n_steps = int(np.ceil(n_periods * centre / grid.dt))
    out = _time_loop(grid, name, n_steps, band)

    if truncation_order is None:
        # The SHORTEST wavelength has the most propagating orders, so the default covers the
        # whole band rather than the centre of it.
        half_width = _propagating_half_width(float(band.min()), period_x, eps_top, eps_bot)
    elif hasattr(truncation_order, 'get'):
        half_width = int(truncation_order.get('x', truncation_order.get('u', 0)))
    else:
        half_width = int(truncation_order[0] if hasattr(truncation_order, '__len__')
                         else truncation_order)

    if out['residual'] > SETTLED_RESIDUAL:
        warnings.warn(
            f"the field had not settled when the run ended: {out['residual']:.1e} of its peak "
            f"was still at the transmission monitor, against a {SETTLED_RESIDUAL:g} target. "
            f"The running DFT is of a truncated signal. Raise n_periods (it is "
            f"{n_periods:g}); an oblique or near-grazing scene often needs twice a "
            f"normal-incidence one.",
            UnsettledFieldWarning, stacklevel=2)

    results = []
    for index in range(band.size):
        sub = _extract(grid, out, half_width, index)
        # The same tier of check the RCWA path applies: gain out of nothing, or negative
        # power. Every defect this module has had -- an over-conducting PML, a mis-scaled CFS
        # pole, an out-of-band DFT -- announced itself here first, as `R + T` between 1.1
        # and 106.
        check_physical(np, np.asarray(sub.R), np.asarray(sub.T), np.sqrt(eps_top))
        results.append(Result(given=sub, order_window=((0, half_width), (0, 0))))

    result = results[0] if single else results
    if not return_diagnostics:
        return result

    # `kx` is fixed by the grid and `kx = n*sin(theta)`, so a BAND is a fixed-kx sweep and the
    # polar angle drifts across it. Reported rather than hidden: at normal incidence every
    # entry is 0 and there is nothing to read, and at oblique this is the difference between
    # what was asked for and what a single time-domain run can answer.
    angles = np.rad2deg(np.arcsin(np.clip(
        grid.kx_inc * band / centre / max(np.sqrt(eps_top).real, 1e-9), -1.0, 1.0)))
    return result, {'dz': grid.dz, 'nz': grid.nz, 'nx': grid.nx, 'dt': grid.dt,
                    'n_steps': n_steps, 'n_pml': grid.n_pml,
                    'pml_thickness': grid.n_pml * grid.dz, 'residual': out['residual'],
                    'incident_balance': abs(out['inc_trn'][0] / out['inc_ref'][0]) - 1.0,
                    'wavelengths': band,
                    'incident_fraction': (np.abs(out['inc_z0'])
                                          / max(np.max(np.abs(out['inc_z0'])), 1e-300)),
                    'angles_deg': angles if single else angles}
