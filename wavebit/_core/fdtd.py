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
from .conventions import order_azimuth
from .domain import as_material
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

#: Clearance between the structure and each of the TFSF plane, the monitors and the PML, as a
#: fraction of the wavelength IN THAT HALF-SPACE. **Measured, and half of what the first
#: version used.** On the benchmark grating at a 5 nm grid the worst per-order error runs
#: 7.37e-4 at 1.0, 6.57e-4 at 0.5, 7.46e-4 at 0.25 and 8.81e-4 at 0.125 -- so 0.5 is *more*
#: accurate than 1.0 and takes 13.9 s against 22.0. The metal grating agrees: 7.43e-4 / 31.9 s
#: against 7.01e-4 / 15.9 s.
#:
#: A wavelength of clearance was chosen so a grating's evanescent orders would decay before
#: the monitor. They do not need to: the spatial transform separates the orders exactly, so an
#: evanescent order sitting on the monitor plane cannot contaminate a propagating one's
#: amplitude. What the clearance is actually for is keeping the PML out of the structure's near
#: field, and half a wavelength does that -- while a shorter outgoing leg carries less of the
#: analytic-versus-discrete ``kz`` mismatch, which is why 0.5 wins on accuracy too. Below 0.25
#: it stops paying.
PAD_WAVELENGTHS = 0.5

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

    half_spaces = [as_material(top, 'top'), as_material(bot, 'bot')]
    values = np.concatenate(
        [np.asarray(structure, dtype=complex).ravel()]
        + [np.asarray(m.epsilon, dtype=complex).ravel() for m in half_spaces]
        + [np.asarray(1.0 if m.mu is None else m.mu, dtype=complex).ravel()
           for m in half_spaces]
        + ([] if mu is None else [np.asarray(mu, dtype=complex).ravel()]))
    if np.any(np.abs(np.imag(values)) > 0):
        raise ValueError(
            'fdtd: the scene has a complex permittivity or permeability. A single complex '
            'number describes a material at ONE frequency, and a '
            'time-domain run carries every frequency in its band at once -- so there is no constant this solve could use that would be right '
            'across the band. Give the dispersion as poles instead: `structure` holds the '
            'real eps_infinity, `dispersion` marks which cells are dispersive, and `poles` '
            'carries the Drude/Lorentz model. That IS the frequency dependence, rather than '
            'one sample of it.')
    if np.any(np.real(values) <= 0.0):
        raise ValueError(
            f'fdtd: the scene has a permittivity or permeability of '
            f'{np.real(values).min()!r}. A non-positive one is a metal below its plasma '
            f'frequency, which needs the same dispersion the complex case does; it also has no '
            f'real wave speed, so there is no time step that makes the update stable.')

    if theta is not None and np.imag(np.asarray(theta, dtype=complex)) != 0.0:
        raise ValueError(
            f'fdtd: theta={theta!r} is complex. Evanescent incidence has no propagating '
            f'incident wave to normalise against, and the TFSF plane injects one by '
            f'construction. The RCWA path answers this; see docs/branch-cuts.md.')
    if phi is not None and float(np.real(phi)) != 0.0:
        raise ValueError(
            f'fdtd: phi={phi!r} with tetm={tetm!r}. A conical mount couples the two '
            f'polarisations, so it has no TE/TM answer to give -- the same split `solve_1d` '
            f'and `solve_1d_conical` make. Ask for it the way the conical path is asked: pass '
            f'`psi` (or `polarization`) instead of `tetm`, which is answered at theta = 0 for '
            f'any phi.')
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


def component_property(raster, thickness, top, bot, period, z_centres, z_widths, dx, nx,
                       x_half, harmonic_axis, subpixel=True, dy=None, ny=1, y_half=False,
                       period_y=None):
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
    if harmonic_axis not in (None, 'x', 'y', 'z'):
        raise ValueError(
            f"component_property: harmonic_axis={harmonic_axis!r}. It names the axis the "
            f"component points along -- 'x', 'z', or None for one with no in-plane axis. A "
            f"typo here silently selects the arithmetic rule and costs an order of accuracy "
            f"without failing.")
    # Each cell brings its OWN width. On a uniform mesh these are all equal and the length
    # weighting below is what it always was; on a graded one they are not, and `_overlap` plus
    # the length-weighted mean are already exact for any set of widths -- the integral needed
    # no rewriting, only the positions did.
    zc = np.asarray(z_centres, dtype=float)
    nz = zc.size
    width_z = np.asarray(z_widths, dtype=float) if subpixel else np.zeros(nz)
    width_x = dx if subpixel else 0.0

    cuts = np.concatenate([[0.0], np.cumsum(np.asarray(thickness, dtype=float))])
    span = 1e6 * max(float(np.max(width_z)) if subpixel else 1.0, 1.0)
    z_edges = np.concatenate([[cuts[0] - span], cuts, [cuts[-1] + span]])
    n_layers = raster.shape[0]

    xc = (np.arange(nx) + (0.5 if x_half else 0.0)) * dx
    width_y = dy if (subpixel and dy is not None) else 0.0
    yc = (np.arange(ny) + (0.5 if y_half else 0.0)) * (dy or 0.0)

    # Along x first, inside each layer, then along y; the two half-spaces are uniform in both.
    per_layer = np.empty((n_layers + 2, ny, nx), dtype=complex)
    per_layer[0] = top
    per_layer[-1] = bot
    for index in range(n_layers):
        layer = np.asarray(raster[index], dtype=complex)
        if layer.ndim == 1:
            # Invariant along y. **The y pass is SKIPPED rather than run on a constant**: an
            # exact average of a constant is not bit-equal to the constant, because
            # `(w*a).sum()/w.sum()` is not `a` in floating point, and skipping it is what keeps
            # a y-invariant scene byte-identical to the 2D module.
            per_layer[index + 1] = _cell_average_periodic(
                layer, period, xc, width_x, harmonic=(harmonic_axis == 'x'))
            continue
        rows = np.stack([_cell_average_periodic(row, period, xc, width_x,
                                                harmonic=(harmonic_axis == 'x'))
                         for row in layer])
        per_layer[index + 1] = np.stack(
            [_cell_average_periodic(rows[:, i], period_y, yc, width_y,
                                    harmonic=(harmonic_axis == 'y'))
             for i in range(nx)], axis=1)

    # Then along z, across the layer boundaries.
    harmonic_z = (harmonic_axis == 'z')
    if not subpixel:
        index = np.clip(np.searchsorted(z_edges, zc, side='right') - 1, 0, n_layers + 1)
        return per_layer[index]
    source = 1.0 / per_layer if harmonic_z else per_layer
    out = np.empty((nz, ny, nx), dtype=complex)
    for k, c in enumerate(zc):
        half = 0.5 * width_z[k]
        w = _overlap(z_edges, c - half, c + half)
        out[k] = (w[:, None, None] * source).sum(axis=0) / w.sum()
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


#: How much one cell may grow over its neighbour, and it is a genuine optimum rather than a
#: safe-looking number. A mesh transition is an optical interface and it reflects: a 4.5x jump
#: taken in ONE cell reflects 5.9e-3 in amplitude -- eight times the module's worst per-order
#: error and 540 times the PML's own -- while the same 4.5x spread over 40 cells gives 6.0e-4.
#: So too aggressive loses accuracy and too gentle spends the cells it was trying to save.
#: Measured on the Drude metal grating, time to reach an efficiency error of 1e-3:
#:
#:     uniform 5.99 s | 1.03 -> 6.60 s | **1.06 -> 3.78 s** | 1.12 -> 9.62 s
#:
#: **Grading is off by default, and that is a measured decision, not caution.** On the metal
#: scene 1.06 is 1.6x faster at equal accuracy; on the dielectric benchmark it LOSES, because
#: the error grows 2.7x and buying that back costs more in cell size than the grading saves in
#: cell count. The saving scales with the index contrast `n_max / n_half` and with how little
#: of the domain the structure occupies -- a thin metal in air is where it pays.
GRADE_RATIO = 1.06


def graded_cells(h_fine, h_coarse, distance, ratio=GRADE_RATIO):
    """Cell sizes spanning ``distance``, starting at ``h_fine`` and growing to ``h_coarse``.

    The ramp is geometric at ``ratio`` per cell -- the standard grading, and the reason it is
    gradual rather than a step is the reflection above. Once the ramp reaches ``h_coarse`` the
    rest of the distance is uniform; if the distance runs out first the ramp is truncated and
    the whole span is graded, which is the case in a thin padding.

    Returned in increasing-distance order, and the sizes are rescaled so they sum to
    ``distance`` exactly -- the caller places nodes by accumulation, so a residue here would
    put the structure interface off a node.
    """
    if distance <= 0.0:
        return np.zeros(0)
    # Grow at `ratio` until the coarse size, then stay there -- and stop BEFORE overshooting,
    # so the total is at most the distance and the scale factor below is at least one.
    cells, h, total = [], float(h_fine), 0.0
    while total + h <= distance:
        cells.append(h)
        total += h
        h = min(h * ratio, h_coarse)
    if not cells:
        return np.array([distance])
    # **Scaling the whole array preserves every ratio exactly**, which is what makes this
    # provable rather than patched: the constraint the caller cares about is a bound on
    # neighbouring cells, and a common factor cannot violate it. Absorbing the residue into a
    # joint instead -- the obvious spelling, and the one tried first -- put 1.8x and 0.53x
    # steps at the two ends, which is exactly the reflection this function exists to spread.
    # And because the loop stops short, the factor is >= 1, so the smallest cell never falls
    # below `h_fine` and the Courant bound is untouched.
    return np.array(cells, dtype=float) * (distance / total)


def build_z_mesh(h_fine, thickness_total, above, below, n_top, n_bot, n_max, wavelength,
                 ratio=GRADE_RATIO):
    """Node positions for a graded z mesh; ``z = 0`` is the top of the structure, by construction.

    The structure keeps ``h_fine`` because that is what its index demanded. Each half-space
    coarsens toward ``h_fine * n_max / n_half`` -- the cell size that half-space would have
    asked for on its own -- which is the whole saving: on a Drude metal grating 112 of 178 z
    cells sit in air above the structure where the index is 1 while ``h_fine`` was set by the
    metal's 6.5.

    ``z = 0`` and ``z = thickness_total`` are nodes exactly, not to a rounding: the incident
    amplitude is referenced at the first and the transmitted at the second, and a half-cell
    error there is a phase error in every amplitude that no efficiency would show.
    """
    n_struct = max(int(np.ceil(thickness_total / h_fine)), 1)
    h_core = thickness_total / n_struct
    core = np.full(n_struct, h_core)
    # The ramps start at the CORE cell, not at `h_fine`. Those differ -- the structure is split
    # into a whole number of cells so both its faces land on nodes, which makes `h_core` up to
    # `n_struct/(n_struct-1)` smaller -- and starting from `h_fine` leaves that difference as a
    # step at the one joint the structure sits on. Measured 1.065 against a 1.03 budget.
    # A half-space may coarsen to the cell it would have asked for on its own -- but never
    # past the module's own standard for it. Without the cap a caller who passes a coarse
    # `resolution` gets padding at 12.7 cells per wavelength, which is under-resolved in the
    # one region the monitors sit in.
    coarse = lambda n: min(h_fine * n_max / max(n, 1e-9),
                           wavelength / (CELLS_PER_WAVELENGTH * max(n, 1e-9)))
    up = graded_cells(h_core, coarse(n_top), above, ratio)
    down = graded_cells(h_core, coarse(n_bot), below, ratio)
    cells = np.concatenate([up[::-1], core, down])
    return -up.sum() + np.concatenate([[0.0], np.cumsum(cells)]), len(up), n_struct


def cpml_profiles(z_node, z_half, top_thickness, bot_thickness, dt, wavelength,
                  eps_top, eps_bot):
    """CPML coefficients per z index, at node (``nz``) and half (``nz-1``) positions.

    Outside the two layers the triple is ``(1, 0, 1)``, which makes every stretched update
    reduce to the plain Yee one -- so the loop runs a single expression everywhere rather than
    branching on whether a row is inside a layer.
    """
    # The two layers no longer have the same thickness once the mesh is graded, so the
    # THICKNESS is selected per side beside the permittivity -- the polynomial grading is
    # defined against the layer's own depth, and using the other side's would mis-scale
    # `sigma_max` by that ratio. `cpml_coefficients` itself needs no change: its arguments are
    # already a physical depth and a physical thickness, and it reads no cell size at all.
    z_node = np.asarray(z_node, dtype=float)
    z_start, z_max = z_node[0], z_node[-1]

    def build(z):
        z = np.asarray(z, dtype=float) - z_start
        d_top, d_bot = top_thickness - z, z - ((z_max - z_start) - bot_thickness)
        depth = np.maximum(np.maximum(d_top, d_bot), 0.0)
        on_top = d_top >= d_bot
        eps_side = np.where(on_top, eps_top, eps_bot)
        side_thickness = np.where(on_top, top_thickness, bot_thickness)
        b, a, kappa = cpml_coefficients(depth, side_thickness, dt, wavelength, eps_side)
        outside = depth <= 0.0
        return {'b': np.where(outside, 1.0, b), 'a': np.where(outside, 0.0, a),
                'kappa': np.where(outside, 1.0, kappa)}

    return build(z_node), build(z_half)


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

    **Why one dimension is still enough away from normal incidence.** The medium here is
    uniform in both in-plane directions, and ``exp(i kx x)`` is an eigenvector of the *discrete*
    x shift while the structure is invariant along y, so a field of that form stays of that
    form on the grid exactly. Every x derivative collapses to one multiplier ``kx_op``, which
    is the discrete operator the 2D grid applies and not the continuum ``i kx``; every y
    derivative is ``i ky`` outright, because there is no y grid to discretise.

    **Six components, or three.** At ``ky = 0`` the six split exactly into ``(Ey, Hx, Hz)`` and
    ``(Hy, Ex, Ez)`` -- the TE and TM triples -- and the class carries only the one a pure
    TE/TM run needs. A conical mount couples them through four terms and needs all six. That
    split is the whole reason a conical mount is a different formulation rather than a
    different angle.
    """

    #: Which arrays sit on the z nodes and which on the half steps, by the Yee staggering the
    #: 2D grid already fixes: `Ey`, `Hz` and `Ex` on the nodes, `Hx`, `Hy` and `Ez` between
    #: them. The four conical coupling pairs are co-located by construction -- `Hx` with `Ez`,
    #: `Hz` with `Ex` -- so no interpolation enters anywhere.
    NODE = ('ey', 'hz', 'ex')
    HALF = ('hx', 'hy', 'ez')

    def __init__(self, nz, dz, dt, eps_r, n_pml, k_src, wavelength, f0, width, t0,
                 tetm, kx_op, dtype, mu_r=1.0, ky=0.0, drive=None, ky_op=None):
        self.dz, self.dt, self.eps_r, self.k_src = dz, dt, eps_r, k_src
        #: The auxiliary grid rides the SAME mesh as the 2D one -- it is handed the same
        #: object, not a copy, so a graded 2D mesh grades this too. That is not a nicety:
        #: measured on a 1D prototype, grading the 2D mesh while leaving this uniform leaks
        #: 9.0e-2 of the incident amplitude into the scattered-field region. And the quieter
        #: variant is worse -- grading only BELOW the TFSF plane leaks just 9.6e-4, but the
        #: incident reference every amplitude is divided by comes out 1.3 % wrong, so `R` and
        #: `T` scale together and `check_physical` sees a plausible `R + T = 0.974`.
        cells = np.full(nz - 1, dz) if np.ndim(dz) == 0 else np.asarray(dz, dtype=float)
        if cells.size != nz - 1:
            raise ValueError(
                f'_Incident1D: {cells.size} cells for {nz} nodes. The auxiliary grid must ride '
                f'the SAME mesh as the 2D one; a length mismatch means it was handed a scalar '
                f'while the 2D grid was graded, which leaks 9.0e-2 of the incident amplitude.')
        self.dz_cell = cells
        inner = 0.5 * (cells[:-1] + cells[1:])
        self.dz_dual = np.concatenate([[cells[0]], inner, [cells[-1]]])
        self.mu_r = float(np.real(mu_r))
        self.f0, self.width, self.t0 = f0, width, t0
        #: **Dimensional** (1/length), like `kx_op` beside it -- NOT the normalised `ky_inc`
        #: every other name in this module carries. Passing the normalised value is a factor
        #: `k0` too large, which on a 900 nm run is 143x and diverges to 1e267 in 200 steps.
        self.tetm, self.kx_op, self.ky = tetm, kx_op, complex(ky)
        #: `d/dy` as an OPERATOR, the way `kx_op` already is. While the structure is invariant
        #: along y this is the exact `i*ky`; on a crossed grating it must be the DISCRETE
        #: difference the 2D grid applies, or the two grids sit on different dispersion
        #: relations and the TFSF cancellation degrades from round-off to their mismatch --
        #: the same failure `kx_op` exists to prevent, in the other axis.
        self.ky_op = 1j * self.ky if ky_op is None else complex(ky_op)
        #: The tangential electric amplitudes the source drives, `(ex, ey)`. For a pure TE/TM
        #: run this is the single component that polarisation names; for a conical mount it is
        #: the incident wave's s or p direction, which are the two eigenpolarisations of a
        #: uniform medium at fixed `(kx, ky)` and therefore do not mix.
        #: `Ey` is TE's primary and `Ex` is TM's, so the default drives whichever one this
        #: polarisation is named for. A conical run passes the s or p direction instead.
        self.drive = drive if drive is not None else ((1.0, 0.0) if tetm == 'tm'
                                                      else (0.0, 1.0))

        for name in self.NODE:
            setattr(self, name, np.zeros(nz, dtype=dtype))
        for name in self.HALF:
            setattr(self, name, np.zeros(nz - 1, dtype=dtype))
        z_node = np.concatenate([[0.0], np.cumsum(self.dz_cell)])
        self.node, self.half = cpml_profiles(
            z_node, z_node[:-1] + 0.5 * self.dz_cell,
            float(self.dz_cell[:n_pml].sum()), float(self.dz_cell[-n_pml:].sum()),
            dt, wavelength, eps_r, eps_r)
        self.psi_hx = np.zeros(nz - 1, dtype=dtype)
        self.psi_hy = np.zeros(nz - 1, dtype=dtype)
        self.psi_ey = np.zeros(nz, dtype=dtype)
        self.psi_ex = np.zeros(nz, dtype=dtype)
        self.conical = self.ky != 0.0

    # -- the two legacy accessors the TE/TM paths read ---------------------------------
    @property
    def e(self):
        """The primary electric amplitude: ``Ey`` on TE, ``Ex`` on TM."""
        return self.ey if self.tetm == 'te' else self.ex

    @property
    def h(self):
        """Its magnetic partner, ``Hx`` on TE and ``-Hy`` on TM.

        That minus sign has **two** consumers -- the TFSF correction and the incident amplitude
        the result is normalised by. Getting it wrong in only one flips the sign of ``r_p``
        while leaving every efficiency intact.
        """
        return self.hx if self.tetm == 'te' else -self.hy

    def _dz(self, field):
        """``d/dz`` of a node array onto the half grid -- the PRIMARY cell size."""
        return (field[1:] - field[:-1]) / self.dz_cell

    def step_h(self):
        ky, kx_op, dt, mu = self.ky_op, self.kx_op, self.dt, self.mu_r
        te = self.tetm in ('te', None)
        tm = self.tetm in ('tm', None)
        if te:
            d = self._dz(self.ey)
            self.psi_hx = self.half['b'] * self.psi_hx + self.half['a'] * d
            drive = d / self.half['kappa'] + self.psi_hx
            if self.conical:
                drive = drive - ky * self.ez
            self.hx += (dt / mu) * drive
            # dHz/dt = i ky Ex - dEy/dx.
            step = -self.ey * kx_op
            if self.conical:
                step = step + ky * self.ex
            self.hz += (dt / mu) * step
        if tm:
            d = self._dz(self.ex)
            self.psi_hy = self.half['b'] * self.psi_hy + self.half['a'] * d
            # dHy/dt = dEz/dx - dEx/dz.
            self.hy += (dt / mu) * (self.ez * kx_op - d / self.half['kappa'] - self.psi_hy)

    def step_e(self, t):
        ky, kx_op, dt, eps = self.ky_op, self.kx_op, self.dt, self.eps_r
        te = self.tetm in ('te', None)
        tm = self.tetm in ('tm', None)
        if te:
            d = np.zeros_like(self.ey)
            d[1:-1] = (self.hx[1:] - self.hx[:-1]) / self.dz_dual[1:-1]
            self.psi_ey = self.node['b'] * self.psi_ey + self.node['a'] * d
            curl = d / self.node['kappa'] + self.psi_ey - self.hz * kx_op
            self.ey[1:-1] += (dt / eps) * curl[1:-1]
        if tm:
            d = np.zeros_like(self.ex)
            d[1:-1] = -(self.hy[1:] - self.hy[:-1]) / self.dz_dual[1:-1]
            self.psi_ex = self.node['b'] * self.psi_ex + self.node['a'] * d
            curl = d / self.node['kappa'] + self.psi_ex
            if self.conical:
                curl = curl + ky * self.hz
            self.ex[1:-1] += (dt / eps) * curl[1:-1]
            # eps dEz/dt = dHy/dx - i ky Hx.
            step = self.hy * kx_op
            if self.conical:
                step = step - ky * self.hx
            self.ez += (dt / eps) * step
        # Additive rather than imposed: a soft source is transparent to whatever comes back
        # through it, which matters because the source sits between the PML and the monitor.
        # It drives the two tangential electric components in the ratio the incident
        # polarisation names, which for a uniform medium at fixed (kx, ky) excites exactly one
        # of the two eigenpolarisations.
        pulse = gaussian_pulse(t, self.f0, self.width, self.t0)
        drive_x, drive_y = self.drive
        if te and drive_y:
            self.ey[self.k_src] += drive_y * pulse
        if tm and drive_x:
            self.ex[self.k_src] += drive_x * pulse


# ---------------------------------------------------------------------------------------
# the grid
# ---------------------------------------------------------------------------------------

def _shift_up(a, bloch=1.0):
    """``a[..., i+1]``, with the Bloch phase the cell edge carries.

    **The LAST axis is x, always.** A 3D run puts y in front of it, so naming the axis by
    position rather than by number is what lets one expression serve both.

    At oblique incidence the field is periodic only up to ``exp(i kx Lambda)``, so the one
    column that wraps picks that factor up and the rest of the shift is an ordinary roll.
    ``bloch = 1`` is normal incidence and reduces this to the plain periodic shift.
    """
    return np.concatenate([a[..., 1:], a[..., :1] * bloch], axis=-1)


def _shift_down(a, bloch=1.0):
    """``a[:, i-1]``, with the Bloch phase (conjugated -- it wraps the other way)."""
    return np.concatenate([a[..., -1:] * np.conj(bloch), a[..., :-1]], axis=-1)


def _shift_up_y(a, bloch=1.0):
    """``a[..., j+1, :]``, with the y Bloch phase the cell edge carries.

    The y twin of :func:`_shift_up`, and it exists separately rather than taking an axis
    argument because every call site knows which axis it means and a wrong axis is a plausible
    wrong answer rather than an error.
    """
    return np.concatenate([a[..., 1:, :], a[..., :1, :] * bloch], axis=-2)


def _shift_down_y(a, bloch=1.0):
    """``a[..., j-1, :]``, with the y Bloch phase conjugated -- it wraps the other way."""
    return np.concatenate([a[..., -1:, :] * np.conj(bloch), a[..., :-1, :]], axis=-2)


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
                 resolution, n_pml, courant, subpixel, theta=0.0, phi=0.0,
                 dispersion=None, poles=None, mu_raster=None, mu_top=1.0, mu_bot=1.0,
                 pad_scale=1.0, grading=None, n_max=None, period_y=None):
        self.wavelength = float(wavelength)
        self.period = float(period)
        #: The y period, which only means anything once the raster has a y axis. Defaulting it
        #: to the x period would be a silently square cell; `None` makes the y-invariant case
        #: structural rather than a value that happens to be unused.
        self.period_y_in = None if period_y is None else float(period_y)
        self.raster = np.asarray(raster, dtype=complex)
        self.thickness = np.asarray(thickness, dtype=float)
        self.eps_top = complex(eps_top)
        self.eps_bot = complex(eps_bot)
        #: Permeability, the E/H dual of `raster` and the two half-space permittivities. An
        #: index is `sqrt(eps*mu)` (engine.py `_half_space_index`), so `mu` reaches every kz,
        #: every order and both admittances -- it is not a second material property bolted on
        #: beside eps, it is the same five slots on the other side of the duality.
        self.mu_raster = (np.ones_like(self.raster) if mu_raster is None
                          else np.asarray(mu_raster, dtype=complex))
        self.mu_top = complex(mu_top)
        self.mu_bot = complex(mu_bot)
        self.n_top_sq = self.eps_top * self.mu_top
        self.n_bot_sq = self.eps_bot * self.mu_bot
        self.subpixel = bool(subpixel)
        self.n_pml = int(n_pml)
        #: Per-raster-cell pole-set id, 0 meaning non-dispersive, and the pole lists it
        #: indexes. Kept beside `raster` rather than inside it because a pole model is not a
        #: number: `structure` holds `eps_infinity` and this says what else the cell carries.
        self.dispersion = None if dispersion is None else np.asarray(dispersion, dtype=int)
        self.poles = dict(poles or {})

        self.nx = max(int(round(self.period * resolution)), 1)
        #: Cells along y, and the cell size and period that go with them. One while the
        #: structure is invariant along y, which is every scene this module accepts today --
        #: `_refuse_out_of_scope` requires `structure.shape[1] == 1`. They exist so the arrays
        #: and the geometry carry the axis before anything varies along it.
        #:
        #: `period_y` is `None` rather than a number in the invariant case, and that is
        #: load-bearing: `component_property` SKIPS the y pass when the raster has no y axis,
        #: because an exact average of a constant is not bit-equal to the constant.
        raster_ny = self.raster.shape[1] if self.raster.ndim == 3 else 1
        crossed = raster_ny > 1
        if crossed and self.period_y_in is None:
            raise ValueError(
                'fdtd: a raster with a y axis needs a y period. `period=[Px, Py]`; a crossed '
                'grating has two and assuming they are equal is a silently square cell.')
        self.ny = max(int(round(self.period_y_in * resolution)), 1) if crossed else 1
        self.dy = (self.period_y_in / self.ny) if crossed else None
        self.period_y = self.period_y_in if crossed else None
        self.dx = self.period / self.nx
        self.dz = self.dx

        n_top = float(np.sqrt(self.n_top_sq).real)
        n_bot = float(np.sqrt(self.n_bot_sq).real)

        # The in-plane wavevector, normalised by k0 exactly as `engine.n_inc` normalises it:
        # `kx_inc = n_inc*sin(theta)*cos(phi)`, and `phi = 0` here. `theta` is in RADIANS,
        # which is wavebit's unit -- measured against the closed-form Fresnel value rather
        # than assumed, because a degrees/radians slip returns a legal answer at a legal angle.
        k0 = 2.0 * np.pi / self.wavelength
        self.theta = float(np.real(0.0 if theta is None else theta))
        self.phi = float(np.real(0.0 if phi is None else phi))
        self.kx_inc = n_top * float(np.sin(self.theta)) * float(np.cos(self.phi))
        #: The other in-plane component. The structure is invariant along y, so `ky` is
        #: conserved and never discretised -- every `d/dy` is `i*ky` outright. It is what
        #: couples the two polarisations, and `phi = 0` is exactly where it vanishes.
        self.ky_inc = n_top * float(np.sin(self.theta)) * float(np.sin(self.phi))
        #: The phase the field picks up crossing one cell edge. `1` is normal incidence, and
        #: the arrays stay REAL there -- a complex dtype would double the cost of every run
        #: for a phase that is identically one.
        #: Real `1.0` and `0.0` rather than `(1+0j)` and `0j` at normal incidence: a complex
        #: literal upcasts every array it touches, which would silently double the cost of
        #: every normal-incidence run for a phase that is identically one.
        self.dtype = complex if (self.kx_inc != 0.0 or self.ky_inc != 0.0) else float
        self.bloch = (np.exp(1j * k0 * self.kx_inc * self.period)
                      if self.dtype is complex else 1.0)
        #: The y twin. A crossed grating is periodic in y as well, so the wrap carries its own
        #: phase -- and it is `ky_inc`, not `kx_inc`, which is the swap this module's own notes
        #: record going wrong repeatedly on the RCWA side.
        self.bloch_y = (np.exp(1j * k0 * self.ky_inc * (self.period_y or self.period))
                        if self.dtype is complex else 1.0)

        # A wavelength of clearance in each half-space: far enough that the evanescent orders
        # a grating radiates have decayed before the monitor, near enough that the outgoing
        # leg carrying the analytic-vs-discrete kz mismatch stays short.
        pad = pad_scale * PAD_WAVELENGTHS * self.wavelength
        pad_top = pad / max(n_top, 1e-9)
        pad_bot = pad / max(n_bot, 1e-9)
        self.total_thickness = float(self.thickness.sum())
        self.n_max = float(n_max) if n_max is not None else float(
            np.sqrt(np.abs(self.raster * self.mu_raster)).max())
        if grading is True:
            grading = GRADE_RATIO
        elif grading is False:
            grading = None
        if grading is not None:
            if not 1.0 < float(grading) < 2.0:
                raise ValueError(
                    f'fdtd: grading={grading!r}. It is the most one cell may grow over its '
                    f'neighbour, so it is above 1 and sensibly below about 1.2 -- a mesh step '
                    f'is an optical interface and it reflects. `True` takes the measured '
                    f'optimum ({GRADE_RATIO}); `None` or `False` is the uniform mesh.')
            cells = self._build_graded(pad_top, pad_bot, n_top, n_bot, grading)
            self._finish(cells, courant)
            return

        gap = max(int(round(0.25 * pad_top / self.dz)), 3)

        self.k_src = self.n_pml + 2
        self.k_ref = self.k_src + gap
        self.k_tfsf = self.k_ref + gap
        self.k_z0 = self.k_tfsf + max(int(round(pad_top / self.dz)), 3)

        k_bottom = self.k_z0 + int(np.ceil(self.total_thickness / self.dz))
        clear = max(int(round(0.5 * pad_bot / self.dz)), 3)
        self.k_trn = k_bottom + clear
        self.nz = self.k_trn + clear + self.n_pml + 1

        self.z0 = -self.k_z0 * self.dz

        #: The two z divisors, which are one number today and two arrays under a graded mesh.
        #: A node-to-node difference lands on a half point and divides by the **primary cell
        #: size**; a half-to-half difference lands on a node and divides by the **dual
        #: spacing**, ``(dz[k-1] + dz[k])/2``. On a uniform mesh they are equal, which is why
        #: one scalar has served both -- and using the wrong one is not a small error but an
        #: O(1) one: measured on a 1.15x graded mesh, the right divisor converges at 2.00 and
        #: the wrong one does not converge at all (0.22 / 0.06 / 0.02, stalling at 4.9e-2).
        #:
        #: They are given **different lengths on purpose** -- ``nz - 1`` against ``nz`` -- so
        #: that swapping them is a shape error rather than a plausible wrong number.
        self._finish(np.full(self.nz - 1, self.dz), courant)

    def _build_graded(self, pad_top, pad_bot, n_top, n_bot, ratio):
        """Lay the scene out in physical distances first, then find the planes on the mesh.

        The uniform path does the opposite -- it counts cells and multiplies by `dz` -- and
        that inversion is the whole of this method. Every plane here is a DISTANCE from the
        structure, matching the uniform path's own spacing so the two are comparable: the TFSF
        plane one `pad_top` above the structure, the reflection monitor a quarter of that
        further, the source another quarter, and the transmission monitor half a `pad_bot`
        below.

        `h_fine` is the cell the scene already asked for, and the index that asked for it is
        recovered from it rather than recomputed -- a pole model raises the effective index far
        above `sqrt(eps_infinity)` and `resolution` has already accounted for that.
        """
        h_fine = self.dz
        n_max = self.n_max
        h_top = h_fine * n_max / max(n_top, 1e-9)
        h_bot = h_fine * n_max / max(n_bot, 1e-9)

        above = self.n_pml * h_top + 2.0 * h_top + 1.5 * pad_top
        below = pad_bot + (self.n_pml + 1) * h_bot
        z_node, n_above, n_struct = build_z_mesh(
            h_fine, self.total_thickness, above, below, n_top, n_bot, n_max,
            self.wavelength, ratio)

        self.nz = len(z_node)
        self.z0 = float(z_node[0])
        self.k_z0 = n_above
        find = lambda z: int(np.argmin(np.abs(z_node - z)))
        self.k_tfsf = find(-pad_top)
        self.k_ref = find(-1.25 * pad_top)
        self.k_src = find(-1.5 * pad_top)
        self.k_trn = find(self.total_thickness + 0.5 * pad_bot)

        # Every plane has to be strictly inside the region it belongs to, and strictly ordered.
        # A graded mesh can collapse two of them onto one cell where a uniform one could not --
        # the coarse cells are up to `n_max` times longer -- and two monitors on one node
        # returns a plausible number rather than an error.
        order = (self.n_pml, self.k_src, self.k_ref, self.k_tfsf, self.k_z0,
                 self.k_z0 + n_struct, self.k_trn, self.nz - 1 - self.n_pml)
        if any(b <= a for a, b in zip(order, order[1:])):
            raise ValueError(
                f'fdtd: grading={ratio} collapses the layout -- planes at {order} must be '
                f'strictly increasing, with the PML boundaries at {self.n_pml} and '
                f'{self.nz - 1 - self.n_pml}. The coarse cells are {n_max:.1f}x the fine ones, '
                f'so a padding that held several cells uniformly can hold one graded. Raise '
                f'`pad_scale`, or lower the grading ratio.')
        return np.diff(z_node)

    def _finish(self, cells, courant):
        """Everything that follows from the mesh, shared by the uniform and graded paths.

        Both paths arrive with ``nz``, the plane indices, ``z0`` and a per-cell size array;
        nothing below reads a scalar ``dz`` for a position again.
        """
        k0 = 2.0 * np.pi / self.wavelength
        ky_phys = k0 * self.ky_inc
        n_min = float(np.sqrt(np.min(np.real(np.concatenate([
            (self.raster * self.mu_raster).ravel(),
            [self.n_top_sq, self.n_bot_sq]])))))
        self.dz_cell = np.asarray(cells, dtype=float)
        #: The dual spacing at each node: half a cell on each side, and half a cell at the
        #: two ends where there is only one.
        inner = 0.5 * (self.dz_cell[:-1] + self.dz_cell[1:])
        self.dz_dual = np.concatenate([[self.dz_cell[0]], inner, [self.dz_cell[-1]]])
        #: Node positions, accumulated rather than derived as ``index * dz``. That is not a
        #: tidy-up: ``cumsum(full(n, d))`` is NOT bit-equal to ``arange(n) * d`` for a
        #: non-dyadic ``d`` (measured 367 of 400 rows differing, worst 6.6e-12), so this table
        #: is the point where the refactor stops being byte-provable and starts being provable
        #: to 1e-13. Every position in the module reads it; nothing recomputes one.
        self.z_node = self.z0 + np.concatenate([[0.0], np.cumsum(self.dz_cell)])
        #: A half-cell component sits at the PRIMAL cell midpoint. That convention is forced
        #: by the field layout -- ``hy[j]`` is built from ``ex[j+1] - ex[j]`` -- and it has to
        #: be stated, because the competing non-uniform-Yee convention (primal nodes at dual
        #: midpoints) puts ``z_of(k + 0.5)`` up to half a coarse cell out.
        self.z_mid = self.z_node[:-1] + 0.5 * self.dz_cell

        # The stability bound is set by the FASTEST wave (speed `1/sqrt(eps*mu)`, so the
        # SMALLEST index binds) and by the SMALLEST cell. `min` is not conservative padding:
        # the true spectral radius of the non-uniform operator was measured at 0.9993-0.9999 of
        # `4/min(dz)**2` on six meshes including a 1:10 jump, never above it. The consequence
        # is that **grading buys no time-step relief** -- the saving is cell count and per-step
        # cost, and `n_steps` is unchanged.
        # `i*ky` enters the amplification factor exactly as a difference operator does, so it
        # joins the bound. It is small next to `1/dx` but it is not zero, and a bound that
        # ignores a term is a bound that fails at the one angle nobody tested.
        # A crossed grating discretises y, so `1/dy**2` joins the bound and the algebraic
        # `ky` term drops out -- they are the same term, counted once. Adding both would be
        # conservative and wrong in the direction that hides a real instability.
        y_term = (1.0 / self.dy ** 2) if self.dy is not None else (0.5 * ky_phys) ** 2
        self.dt = courant * n_min / np.sqrt(
            1.0 / self.dx ** 2 + 1.0 / float(self.dz_cell.min()) ** 2 + y_term)

        #: The DISCRETE x-derivative the auxiliary grid applies, not the continuum `i*kx`:
        #: a field `exp(i kx x)` is an eigenvector of the grid's shift, and the eigenvalue is
        #: this. Using `i*kx` instead leaves the two grids on different dispersion relations
        #: and the TFSF cancellation degrades from round-off to the mismatch between them.
        self.kx_op = (2j * np.sin(0.5 * k0 * self.kx_inc * self.dx) / self.dx
                      if self.dtype is complex else 0.0)

        self.node, self.half = cpml_profiles(
            self.z_node, self.z_mid,
            float(self.dz_cell[:self.n_pml].sum()), float(self.dz_cell[-self.n_pml:].sum()),
            self.dt, self.wavelength, self.eps_top.real, self.eps_bot.real)

    def z_of(self, k):
        """The z coordinate of z index ``k``; ``k`` may carry a half.

        A fractional ``k`` is measured into cell ``floor(k)``, which is exact for the one
        fraction that occurs (``+0.5``, the half-cell components) under the primal-midpoint
        convention `z_mid` records.
        """
        i = int(np.floor(k))
        frac = float(k) - i
        if frac == 0.0:
            return float(self.z_node[i])
        return float(self.z_node[i] + frac * self.dz_cell[i])

    def z_cells(self, z_half):
        """``(centres, widths)`` of one component's z cells -- half-cell or node."""
        if z_half:
            return self.z_mid, self.dz_cell
        return self.z_node, self.dz_dual

    def eps_for(self, z_half, x_half, harmonic_axis, y_half=False):
        centres, widths = self.z_cells(z_half)
        # `[:, None, :]` rather than a reshape: the y axis is inserted, and while `ny == 1` the
        # values are the 2D ones untouched, element for element.
        return component_property(
            self.raster, self.thickness, self.eps_top, self.eps_bot, self.period,
            centres, widths, self.dx, self.nx, x_half, harmonic_axis,
            subpixel=self.subpixel, dy=self.dy, ny=self.ny, y_half=y_half,
            period_y=self.period_y)

    def mu_for(self, z_half, x_half, harmonic_axis, y_half=False):
        """Permeability on one magnetic component's cells, by the same rule `eps_for` uses.

        Harmonic along the component's own axis: `B_normal` is continuous where `D_normal` is,
        so the averaging rule is the duality's image of the electric one and not a separate
        choice.
        """
        centres, widths = self.z_cells(z_half)
        return component_property(
            self.mu_raster, self.thickness, self.mu_top, self.mu_bot, self.period,
            centres, widths, self.dx, self.nx, x_half, harmonic_axis,
            subpixel=self.subpixel, dy=self.dy, ny=self.ny, y_half=y_half,
            period_y=self.period_y)

    def fill_for(self, pole_id, z_half, x_half, y_half=False):
        """What fraction of each of this component's cells carries ``pole_id``.

        The same exact cell average `eps_infinity` gets, run on the indicator of one pole set,
        which is what makes it a fraction. Arithmetic, never harmonic: the harmonic rule is a
        statement about a *static* permittivity normal to an interface and does not carry over
        to a pole strength.
        """
        centres, widths = self.z_cells(z_half)
        indicator = (self.dispersion == pole_id).astype(float)
        return np.real(component_property(
            indicator, self.thickness, 0.0, 0.0, self.period, centres, widths,
            self.dx, self.nx, x_half, None, subpixel=self.subpixel,
            dy=self.dy, ny=self.ny, y_half=y_half, period_y=self.period_y))

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

def _time_loop(grid, tetm, n_steps, wavelengths, leg=None, record_field=None):
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
    #: The y axis. One today -- the structure is invariant along y and `d/dy` is the
    #: algebraic `i*ky` -- but the arrays carry it so a crossed grating is a larger `ny`
    #: rather than a second stepper. At `ny == 1` every expression below is the 2D one
    #: element for element, which is what makes this provable rather than argued.
    ny = grid.ny
    # The z divisors. `dz` itself stays scalar here and owns POSITIONS only.
    dzc, dzd = grid.dz_cell[:, None, None], grid.dz_dual[:, None, None]
    hb, ha, hk = (grid.half[k][:, None, None] for k in ('b', 'a', 'kappa'))
    nb, na, nk = (grid.node[k][:, None, None] for k in ('b', 'a', 'kappa'))
    kt = grid.k_tfsf

    dtype = grid.dtype
    # A crossed grating runs all six components for the same reason a conical mount does --
    # the y terms couple the two triples -- so it takes this branch whether or not ky is zero.
    conical = grid.ky_inc != 0.0 or grid.ny > 1
    # The incident wave's own polarisation, resolved onto the two tangential electric
    # components. `s_hat = (-sin phi, cos phi, 0)`; `p_hat = k_hat x s_hat` has tangential part
    # `-cos(theta) * (cos phi, sin phi)`. These are the two eigenpolarisations of a uniform
    # medium at fixed `(kx, ky)`, so driving one excites only it -- which is what lets a 1D
    # auxiliary grid carry a conical incident wave at all.
    inc_drive = None
    if conical:
        c_p, s_p = np.cos(grid.phi), np.sin(grid.phi)
        inc_drive = ((-s_p, c_p) if leg == 's'
                     else (-np.cos(grid.theta) * c_p, -np.cos(grid.theta) * s_p))
    # The SAME cell array the 2D grid runs on, not the scalar. Handing it `grid.dz` leaves the
    # incident wave on a different mesh from the scattered one: measured on a graded dielectric
    # scene, the efficiencies stay plausible (3.5e-3 against RCWA) while every AMPLITUDE is
    # wrong by 0.52 -- a phase error, and `R + T` never notices.
    inc = _Incident1D(nz, grid.dz_cell, dt, grid.eps_top.real, grid.n_pml, grid.k_src,
                      centre, 1.0 / centre, width, t0, None if conical else tetm,
                      grid.kx_op, dtype, mu_r=grid.mu_top.real,
                      ky=(2.0 * np.pi / centre) * grid.ky_inc if conical else 0.0,
                      ky_op=((2j * np.sin(0.5 * (2.0 * np.pi / centre) * grid.ky_inc * grid.dy) / grid.dy)
                             if grid.ny > 1 else None),
                      drive=inc_drive)

    # The incident field the TFSF plane injects varies across the cell as `exp(i kx x)`; the
    # auxiliary grid carries only its amplitude, so the plane multiplies it back on. `Ey`/`Hx`
    # sit on the x nodes and `Hy`/`Ex` on the half-steps, which is the only difference between
    # the two polarisations here. `ones` rather than `exp(0j)` at normal incidence, so the
    # field arrays stay real.
    k0 = 2.0 * np.pi / centre
    if grid.kx_inc == 0.0:
        x_node = x_half = np.ones(nx)
    else:
        cell = np.arange(nx)
        x_node = np.exp(1j * k0 * grid.kx_inc * cell * dx)
        x_half = np.exp(1j * k0 * grid.kx_inc * (cell + 0.5) * dx)
    # The y factor, and the in-plane phase is the OUTER PRODUCT of the two -- a plane wave is
    # separable, so two vectors serve where a full (ny, nx) array would. The four injected
    # components sit at only TWO distinct in-plane Yee positions: `Ey` and `Hx` on
    # (x-node, y-half), `Ex` and `Hy` on (x-half, y-node).
    if grid.ny == 1 or grid.ky_inc == 0.0:
        y_node = y_half = np.ones(grid.ny)
    else:
        row = np.arange(grid.ny)
        y_node = np.exp(1j * k0 * grid.ky_inc * row * grid.dy)
        y_half = np.exp(1j * k0 * grid.ky_inc * (row + 0.5) * grid.dy)
    p_node = y_half[:, None] * x_node[None, :]      # Ey, Hx
    p_half = y_node[:, None] * x_half[None, :]      # Ex, Hy
    # `Ey`/`Hx` sit on the x nodes and `Ex`/`Hy` on the half-steps. A pure run's two injection
    # lines share one position -- which is why one array was enough and why the single-offset
    # form read as correct. A conical run drives all four, so it needs both: using one costs
    # 1.8e-3 of the incident amplitude, above every convergence tolerance in the test file and
    # far below anything `R + T` could flag.

    # A conical mount couples the two triples, so both run and the four `ky` terms join them.
    # At ky = 0 they decouple exactly and only the named one is allocated -- which is why a
    # pure TE/TM run costs no more than it did.
    te_on = conical or tetm == 'te'
    tm_on = conical or tetm == 'tm'
    ky = 1j * (2.0 * np.pi / centre) * grid.ky_inc
    #: `d/dy` is ALGEBRAIC while the structure is invariant along y and a real grid derivative
    #: once it is not. The two are different operators, not two spellings: `i*ky` is the exact
    #: continuum value, while a crossed grating's y difference carries the grid's own numerical
    #: dispersion -- and it has to, because the field it differentiates lives on that grid.
    #:
    #: Forward in the H updates and backward in the E ones, mirroring z exactly. The Yee y
    #: positions that force it: `Ey`, `Hx` and `Hz` sit on the y half-steps, `Ex`, `Ez` and
    #: `Hy` on the y nodes (Yee 1966; Taflove & Hagness fig. 3.1).
    crossed = grid.ny > 1
    dy, by = grid.dy, grid.bloch_y
    d_dy_fwd = (lambda a: (_shift_up_y(a, by) - a) / dy) if crossed else (lambda a: ky * a)
    d_dy_bwd = (lambda a: (a - _shift_down_y(a, by)) / dy) if crossed else (lambda a: ky * a)

    if te_on:
        # (Ey, Hx, Hz). Ey sits on the z nodes and the x nodes, and is tangential to every
        # interface a 1D grating has, so it takes no harmonic average.
        # `'y'` rather than `None`: `E_y` is tangential to every interface a 1D-periodic
        # grating has, which is why this was `None`, but it is NORMAL to the y walls of a
        # crossed one. At `ny == 1` the two spellings are the same computation -- neither makes
        # the x or z pass harmonic -- so this is byte-identical today and correct later.
        eps = grid.eps_for(False, False, 'y', y_half=True).real
        # `Hx` points along x and `Hz` along z, so each takes the harmonic average along its
        # own axis -- the duality's image of the rule `eps_for` uses for `Ex` and `Ez`.
        mu_x = grid.mu_for(True, False, 'x', y_half=True).real
        mu_z = grid.mu_for(False, True, 'z', y_half=True).real
        ey = np.zeros((nz, ny, nx), dtype=dtype)
        hx = np.zeros((nz - 1, ny, nx), dtype=dtype)
        hz = np.zeros((nz, ny, nx), dtype=dtype)
        poles_y = grid.pole_state((nz, ny, nx), False, False, dtype)
    if tm_on:
        # (Hy, Ex, Ez). Ex is normal to the tooth's vertical walls, Ez to the horizontal ones.
        eps_x = grid.eps_for(False, True, 'x').real
        eps_z = grid.eps_for(True, False, 'z').real
        # `Hy` is out of plane, so it has no in-plane axis to average harmonically along.
        mu_y = grid.mu_for(True, True, 'y').real
        hy = np.zeros((nz - 1, ny, nx), dtype=dtype)
        ex = np.zeros((nz, ny, nx), dtype=dtype)
        ez = np.zeros((nz - 1, ny, nx), dtype=dtype)
        poles_x = grid.pole_state((nz, ny, nx), False, True, dtype)
        poles_z = grid.pole_state((nz - 1, ny, nx), True, False, dtype)
    # One CPML accumulator per stretched update. The two triples shared a pair while only one
    # of them ever ran; a conical mount runs both.
    psi_hx = np.zeros((nz - 1, ny, nx), dtype=dtype)
    psi_hy = np.zeros((nz - 1, ny, nx), dtype=dtype)
    psi_ey = np.zeros((nz, ny, nx), dtype=dtype)
    psi_ex = np.zeros((nz, ny, nx), dtype=dtype)

    n_f = len(freq)
    dft_ref = np.zeros((n_f, ny, nx), dtype=complex)
    dft_trn = np.zeros((n_f, ny, nx), dtype=complex)
    #: The time-domain record, which is the one thing a time-domain solve does that its
    #: frequency-domain answer cannot show. It costs a snapshot every `record_field` steps;
    #: at a 10 nm grid on the benchmark cell that is 86 kB a snapshot, so every 10th step is
    #: 67 MB and every 50th is 13 MB. Off by default because a solve that is not asked for it
    #: should not pay for it -- the same reason `answers` is a declaration on the RCWA path.
    n_snap = 0 if record_field is None else (n_steps + record_field - 1) // record_field
    snap_e = np.zeros((n_snap, nz, ny, nx), dtype=dtype) if n_snap else None
    snap_x = np.zeros((n_snap, nz, ny, nx), dtype=dtype) if (n_snap and conical) else None
    snap_t = np.zeros(n_snap)

    # The second transverse component, `Ex`, and its own incident reference. A conical mount
    # needs both because the s and p channels of an order are two projections of the same
    # transverse vector -- at ky = 0 one component determines the other and these stay unused.
    dft_ref_x = np.zeros((n_f, ny, nx), dtype=complex)
    dft_trn_x = np.zeros((n_f, ny, nx), dtype=complex)
    inc_z0_x = np.zeros(n_f, dtype=complex)
    inc_ref = np.zeros(n_f, dtype=complex)
    inc_trn = np.zeros(n_f, dtype=complex)
    inc_z0 = np.zeros(n_f, dtype=complex)
    peak = tail = 0.0

    for step in range(n_steps):
        # --- H, to level n + 1/2 --------------------------------------------------------
        inc.step_h()
        if te_on:
            # Yee 1966: dHx/dt = dEy/dz - i ky Ez, dHz/dt = i ky Ex - dEy/dx. Only z has a PML.
            # `Ez` sits where `Hx` does and `Ex` where `Hz` does, so the conical coupling is
            # co-located and no interpolation enters.
            d = (ey[1:] - ey[:-1]) / dzc
            psi_hx = hb * psi_hx + ha * d
            curl_hx = d / hk + psi_hx
            curl_hz = -(_shift_up(ey, grid.bloch) - ey) / dx
            if conical:
                curl_hx = curl_hx - d_dy_fwd(ez)
                curl_hz = curl_hz + d_dy_fwd(ex)
            hx += dt * curl_hx / mu_x
            hz += dt * curl_hz / mu_z
            # TFSF (Umashankar & Taflove 1982): Hx at kt-1/2 is a scattered-field quantity
            # whose stencil reached across the boundary into the total-field Ey[kt].
            hx[kt - 1] -= dt * inc.ey[kt] * p_node / dzc[kt - 1, 0] / mu_x[kt - 1]
        if tm_on:
            # dHy/dt = dEz/dx - dEx/dz -- the one component no `ky` term reaches.
            d_ez = (_shift_up(ez, grid.bloch) - ez) / dx
            d_ex = (ex[1:] - ex[:-1]) / dzc
            psi_hy = hb * psi_hy + ha * d_ex
            hy += dt * (d_ez - d_ex / hk - psi_hy) / mu_y
            hy[kt - 1] += dt * inc.ex[kt] * p_half / dzc[kt - 1, 0] / mu_y[kt - 1]

        # --- E, to level n + 1 ----------------------------------------------------------
        inc.step_e((step + 1) * dt)
        if te_on:
            # eps dEy/dt = dHx/dz - dHz/dx -- the other component no `ky` term reaches.
            d = np.zeros((nz, ny, nx), dtype=dtype)
            d[1:-1] = (hx[1:] - hx[:-1]) / dzd[1:-1]
            psi_ey = nb * psi_ey + na * d
            curl = (d / nk + psi_ey
                    - (hz - _shift_down(hz, grid.bloch)) / dx)
            # eps_inf dE/dt = curl H - dP/dt. `advance` reads E at level n and hands back the
            # increment every pole contributes, which is what makes the pair explicit.
            drive = dt * curl
            if poles_y is not None:
                drive = drive - poles_y.advance(ey)
            ey[1:-1] += drive[1:-1] / eps[1:-1]
            ey[kt] -= dt * inc.hx[kt - 1] * p_node / dzd[kt, 0] / eps[kt]
        if tm_on:
            # eps dEx/dt = i ky Hz - dHy/dz ; eps dEz/dt = dHy/dx - i ky Hx.
            d = np.zeros((nz, ny, nx), dtype=dtype)
            d[1:-1] = -(hy[1:] - hy[:-1]) / dzd[1:-1]
            psi_ex = nb * psi_ex + na * d
            curl_ex = d / nk + psi_ex
            if conical:
                curl_ex = curl_ex + d_dy_bwd(hz)
            drive = dt * curl_ex
            if poles_x is not None:
                drive = drive - poles_x.advance(ex)
            ex[1:-1] += drive[1:-1] / eps_x[1:-1]
            # Hy_inc is MINUS `inc.hy`, which is where this sign comes from -- the same
            # double negation the `inc.h` accessor used to hide.
            ex[kt] -= dt * (-inc.hy[kt - 1]) * p_half / dzd[kt, 0] / eps_x[kt]
            curl_ez = (hy - _shift_down(hy, grid.bloch)) / dx
            if conical:
                curl_ez = curl_ez - d_dy_bwd(hx)
            drive_z = dt * curl_ez
            if poles_z is not None:
                drive_z = drive_z - poles_z.advance(ez)
            ez += drive_z / eps_z

        # --- monitors -------------------------------------------------------------------
        # The e^{-iwt} phasor is recovered with an exp(+i w t) kernel; `np.fft.fft` over the
        # time axis applies the opposite one and would conjugate every amplitude.
        if conical:
            # `Ey` and `Ex` are the pair a conical mount needs, and they are also the cheapest
            # one available: both sit on the z NODES, so they share one z offset and one time
            # stamp. The obvious pair (Ey, Hy) does not -- `Hy` is a half cell down in z and
            # stamped half a step back in time, so it would need two of each.
            f_ref, f_trn = ey[grid.k_ref], ey[grid.k_trn]
            g_ref, g_trn = ex[grid.k_ref], ex[grid.k_trn]
            i_z0, j_z0 = inc.ey[grid.k_z0], inc.ex[grid.k_z0]
            i_ref, i_trn = inc.ey[grid.k_ref], inc.ey[grid.k_trn]
            t_now = (step + 1) * dt
        elif tetm == 'te':
            f_ref, f_trn = ey[grid.k_ref], ey[grid.k_trn]
            i_ref, i_trn, i_z0 = (inc.e[grid.k_ref], inc.e[grid.k_trn], inc.e[grid.k_z0])
            t_now = (step + 1) * dt
        else:
            f_ref, f_trn = hy[grid.k_ref], hy[grid.k_trn]
            i_ref, i_trn, i_z0 = (-inc.h[grid.k_ref], -inc.h[grid.k_trn],
                                  -inc.h[grid.k_z0])
            t_now = (step + 0.5) * dt
        kernel = np.exp(2j * np.pi * freq * t_now) * dt          # (n_f,)
        dft_ref = dft_ref + kernel[:, None, None] * f_ref[None, :, :]
        dft_trn = dft_trn + kernel[:, None, None] * f_trn[None, :, :]
        inc_ref += kernel * i_ref
        inc_trn += kernel * i_trn
        inc_z0 += kernel * i_z0
        if conical:
            dft_ref_x = dft_ref_x + kernel[:, None, None] * g_ref[None, :, :]
            dft_trn_x = dft_trn_x + kernel[:, None, None] * g_trn[None, :, :]
            inc_z0_x += kernel * j_z0

        if n_snap and step % record_field == 0:
            # The ELECTRIC field, whichever component this polarisation carries -- a TM run
            # steps `Hy` as its primary but `Ex` is what a reader means by "the field".
            j = step // record_field
            snap_e[j] = ey if te_on else ex
            if conical:
                snap_x[j] = ex
            snap_t[j] = (step + 1) * dt

        amplitude = float(np.abs(f_trn).max())
        if conical:
            amplitude = max(amplitude, float(np.abs(g_trn).max()))
        peak = max(peak, amplitude)
        tail = amplitude

    # The monitor's PHYSICAL z, not its index: `Hy` lives half a cell below the node, and
    # reading the node coordinate instead leaves a `k0*kz*dz/2` phase in every TM amplitude.
    # It cancels in |r|**2, so the efficiencies converge at second order while the amplitudes
    # do not -- measured 4.3e-2 on `r_p` at a 10 nm grid against 1.4e-3 on `R`, falling only
    # as dz**0.6.
    # `Ey` and `Ex` both sit on the z nodes, so a conical run's offset is zero for both.
    offset = 0.0 if conical else (0.5 if tetm == 'tm' else 0.0)
    return {'snap_e': snap_e, 'snap_x': snap_x, 'snap_t': snap_t[:n_snap] if n_snap else None,
            'dft_ref': dft_ref, 'dft_trn': dft_trn, 'tetm': tetm, 'conical': conical,
            'dft_ref_x': dft_ref_x, 'dft_trn_x': dft_trn_x, 'inc_z0_x': inc_z0_x,
            'wavelengths': wavelengths,
            'inc_ref': inc_ref, 'inc_trn': inc_trn, 'inc_z0': inc_z0,
            'z_ref': grid.z_of(grid.k_ref + offset),
            'z_trn': grid.z_of(grid.k_trn + offset),
            'z_inc': grid.z_of(grid.k_z0 + offset),
            'residual': tail / peak if peak > 0.0 else 0.0}


# ---------------------------------------------------------------------------------------
# order extraction
# ---------------------------------------------------------------------------------------

def _half_space_kz(eps, kx, ky=0.0):
    """``kz`` on wavebit's half-space branch: the principal root of ``eps - kx**2 - ky**2``.

    ``ky`` is one scalar for every order -- the structure is invariant along ``y``, so nothing
    can change it -- but it is NOT optional in the way a shared constant sometimes is. It sets
    which orders propagate: at ``theta = 60 deg, phi = 30 deg`` dropping it makes ``kz_inc``
    0.66144 where the truth is 0.5, and calls order ``-4`` propagating (``kz`` real, 0.39965)
    where it is evanescent (0.16667j). `_propagating_window` already subtracts it, so a
    ky-blind ``kz`` here puts the two halves of the extraction on different order sets and the
    symptom is power appearing out of an evanescent order.
    """
    return np.sqrt(np.asarray(eps, dtype=complex)
                   - np.asarray(kx, dtype=complex) ** 2
                   - np.asarray(ky, dtype=complex) ** 2)


def _propagating_window(wavelength, period, n_top_sq, n_bot_sq, kx_inc=0.0, ky_inc=0.0):
    """``(centre, half_width)`` covering every order that propagates in either half-space.

    An order propagates where ``kx_m**2 + ky**2 <= n**2`` with ``kx_m = kx_inc + m*wl/period``,
    so the set is **centred on ``-kx_inc``, not on zero**, and it is asymmetric in ``m`` at any
    oblique angle. A window that assumed symmetry silently dropped orders on the far side:
    measured at 60 degrees into an ``n = 2`` substrate, order ``-2`` propagates and carries
    7.2 % of the power, and ``sum(R) + sum(T)`` came back as 0.928 with nothing to say why.

    ``ky`` narrows it rather than shifting it -- it is the same for every order, so it eats
    into the budget ``kx_m`` has left. At ``ky > n`` the half-space carries nothing at all.
    """
    step = wavelength / period
    lo, hi = 0, 0
    for n_sq in (n_top_sq, n_bot_sq):
        budget = float(np.real(n_sq)) - float(ky_inc) ** 2
        if budget <= 0.0:
            continue                                  # nothing propagates on this side
        reach = np.sqrt(budget)
        lo = min(lo, int(np.ceil((-reach - kx_inc) / step)))
        hi = max(hi, int(np.floor((reach - kx_inc) / step)))
    centre = int(round(0.5 * (lo + hi)))
    return centre, max(hi - centre, centre - lo, 0)


def _order_frame(kx, ky, phi):
    """``(cos varphi_m, sin varphi_m)`` -- each order's own s/p frame.

    :func:`~wavebit._core.conventions.order_azimuth` is **called**, not respelled. Its
    argument order is ``(ky, kx)`` like ``atan2``'s, and its two special cases are the whole
    reason not to open-code it: the ``kx == 0`` column is a right angle taking the sign of
    ``ky``, and the origin -- ``kx == ky == 0``, the zeroth order at exactly normal incidence
    -- takes ``phi``, because a wave going straight down still needs a reference direction for
    its polarisation. Filling the origin with zero instead collapses the s/p mixing silently,
    and only when ``phi != 0``.
    """
    varphi = order_azimuth(np, np.asarray(ky, dtype=complex), np.asarray(kx, dtype=complex),
                           float(phi))
    return np.cos(varphi), np.sin(varphi)


def _combine_at_normal_incidence(grid, sub_te, sub_tm, kx, wavelength, phi, a_s, a_p):
    """Project the two scalar solves onto each order's s/p frame. ``theta = 0`` only.

    **At normal incidence ``ky`` is exactly zero for every ``phi``**, so the stepper is not
    conical at all -- the six-component operator is never reached. The whole ``phi`` dependence
    lives here, in the output frame, which is what makes this the one place the projection can
    be tested against `solve_1d_conical` with no conical stepper underneath it.

    Two rotations, and nothing else. The incident pair ``(a_s, a_p)`` is stated in a frame at
    azimuth ``phi``, and the grating's two scalar problems live in the fixed frame -- ``E``
    along ``y`` is TE, ``E`` along ``x`` is TM -- so rotating in gives each problem its drive.
    Each order then carries its *own* azimuth, so rotating back out is per-order, and the
    zeroth order is the only one that rotates at all: every ``kx_m != 0`` order has azimuth 0
    or pi, where the rotation degenerates to the sign ``cos varphi_m``.

    That asymmetry is the test. Measured against `solve_1d_conical` at ``theta = 0``,
    ``phi = 90 deg`` on the benchmark grating, pure-s incidence puts the specular order in the
    **s** channel (0.282281 -- the TM number) and the first order in the **p** channel
    (0.276917). One physical field, two different channels, decided by geometry alone. A
    projection that dropped the per-order frame would put both in the same channel and still
    conserve energy.

    The weights were fixed against RCWA *amplitudes*, not efficiencies, across
    ``phi = 0, 45, 90 deg`` and both legs -- twelve complex numbers. Efficiencies are blind to
    exactly the sign this is most likely to get wrong; see `docs/fdtd.md` and
    ``per-order-sp-frame-half-turn``.
    """
    c_i, s_i = float(np.cos(phi)), float(np.sin(phi))
    # Rotate the incidence out of its own frame and into the grid's. `w_tm` carries the sign
    # of wavebit's published p amplitude rather than of `E_x`; the two differ by one at
    # theta = 0, and this form is the measured one.
    w_te = a_s * c_i - a_p * s_i
    w_tm = a_s * s_i + a_p * c_i

    ky = np.zeros_like(np.asarray(kx, dtype=float))     # exactly zero: theta = 0
    c_m, s_m = _order_frame(kx, ky, phi)
    c_m, s_m = np.real(c_m), np.real(s_m)

    def project(a_te, a_tm, leg):
        # `s_hat` is the same for the up- and down-going legs, but `p_hat = k_hat x s_hat`
        # reverses with `k_hat` -- so the transmitted frame is the reflected one with
        # `sin varphi_m` negated. **Only the specular order can see this**: every other order
        # at theta = 0 has azimuth 0 or pi, where `sin varphi_m` is zero and the two rules
        # coincide. It is invisible at phi = 0 and at phi = 90 deg as well, where the term is
        # either absent or the whole answer; phi = 45 deg is what separates them, and it read
        # `T_p[0] = 0.241` against a true 0.017 with the reflected rule used for both.
        v_y, v_x = w_te * np.asarray(a_te).ravel(), w_tm * np.asarray(a_tm).ravel()
        sm = s_m if leg == 'r' else -s_m
        return c_m * v_y - sm * v_x, c_m * v_x + sm * v_y

    r_s, r_p = project(sub_te.r_s, sub_tm.r_p, 'r')
    t_s, t_p = project(sub_te.t_s, sub_tm.t_p, 't')

    # The same flux weights `_extract` applies, restated on the published amplitudes: `T_p`
    # there is built from the raw magnetic ratio, and the published `t_p` is that times
    # `g`, so dividing by |g|**2 puts them back on one footing.
    kz_top = _half_space_kz(grid.n_top_sq, kx)
    kz_bot = _half_space_kz(grid.n_bot_sq, kx)
    kz_inc = _half_space_kz(grid.n_top_sq, 0.0)
    n_top, n_bot = np.sqrt(grid.n_top_sq), np.sqrt(grid.n_bot_sq)
    g = (n_top / grid.mu_top) / (n_bot / grid.mu_bot)
    fr = np.real(kz_top / kz_inc)
    ft_s = np.real(kz_bot / grid.mu_bot / (kz_inc / grid.mu_top))
    ft_p = np.real(kz_bot / grid.eps_bot / (kz_inc / grid.eps_top)) / abs(g) ** 2
    sq = lambda a: np.nan_to_num(np.real(a * np.conj(a)), nan=0.0)

    R_s, R_p = sq(r_s) * fr, sq(r_p) * fr
    T_s, T_p = sq(t_s) * ft_s, sq(t_p) * ft_p
    row = lambda a: np.asarray(a)[None, :]
    return ResultSub(row(r_s), row(r_p), row(t_s), row(t_p),
                     row(R_s + R_p), row(R_s), row(R_p),
                     row(T_s + T_p), row(T_s), row(T_p))


def _extract_conical(grid, out, half_width, index, centre, sigma=(-1.0, 1.0, -1.0)):
    """Per-order s/p amplitudes for a conical mount, from the two transverse electric monitors.

    **The p channel is not a projection of the transverse field, it is that projection times
    ``n / kz_m``,** and getting only the projection conserves energy exactly while every
    amplitude is wrong by a per-order factor. The algebra, once, because it is the whole
    function: with ``(c, s) = (cos varphi_m, sin varphi_m)`` and ``k_par`` the in-plane
    magnitude,

        s_hat  = (-s, c, 0)                              -- the same for every leg
        p_hat  = k_hat x s_hat = (-+ kz c, -+ kz s, k_par) / n

    so ``p_hat``'s transverse part reverses with the propagation direction and its ``z`` part
    does not. A plane wave in the medium has ``E . k_hat = 0``, which fixes ``E_z`` from the
    pair we measure; substituting it and using ``kz**2 + k_par**2 = n**2`` collapses the whole
    thing to

        E . p_hat = -+ (n / kz_m) * (c Ex + s Ey).

    That ``n / kz_m`` is a per-order ``1 / cos(theta_m)``. It is 1 only for the specular order
    at normal incidence, which is exactly the case the ``theta = 0`` path already covered.

    ``sigma`` carries the three leg signs -- incident, reflected, transmitted -- which the
    algebra above fixes only up to wavebit's published convention. They are measured against
    `solve_1d_conical` amplitudes rather than derived, on the grounds that every one of them is
    invisible to ``R``, ``T`` and ``R + T``.
    """
    nx = grid.nx
    if grid.ny != 1:
        raise NotImplementedError(
            'fdtd: the extraction is still one-dimensional in the transverse plane. A crossed '
            'grating needs a 2D order set -- the propagating set becomes a DISC rather than a '
            'rectangle, and a per-axis copy of `_propagating_window` silently drops orders.')
    wavelength = float(out['wavelengths'][index])
    if half_width >= nx // 2:
        raise ValueError(
            f'fdtd: {2 * half_width + 1} orders asked of a {nx}-cell grid. The x transform '
            f'resolves |m| < {nx // 2}; beyond that an order aliases onto another one.')

    m = np.arange(centre - half_width, centre + half_width + 1)
    k0 = 2.0 * np.pi / wavelength
    scale = wavelength / grid.wavelength      # the grid holds the PHYSICAL kx, ky fixed
    kx_inc, ky = grid.kx_inc * scale, grid.ky_inc * scale
    kx = kx_inc + m * wavelength / grid.period

    # `Ey` sits on the x nodes, `Ex` on the half-steps -- two kernels, not one.
    cells = np.arange(nx)
    ker_y = np.exp(-1j * k0 * kx[:, None] * (cells[None, :] * grid.dx)) / nx
    ker_x = np.exp(-1j * k0 * kx[:, None] * ((cells[None, :] + 0.5) * grid.dx)) / nx

    n_top, n_bot = np.sqrt(grid.n_top_sq), np.sqrt(grid.n_bot_sq)
    kz_top = _half_space_kz(grid.n_top_sq, kx, ky)
    kz_bot = _half_space_kz(grid.n_bot_sq, kx, ky)
    kz_inc = _half_space_kz(grid.n_top_sq, kx_inc, ky)

    c_m, s_m = _order_frame(kx, np.full_like(kx, ky), grid.phi)
    c_m, s_m = np.real(c_m), np.real(s_m)
    c_i, s_i = float(np.cos(grid.phi)), float(np.sin(grid.phi))

    sig_i, sig_r, sig_t = sigma
    z_ref, z_trn, z_inc = out['z_ref'], out['z_trn'], out['z_inc']

    # The incident wave, read at z = 0 where the amplitudes are referenced -- one leg of the
    # numerical-dispersion phase rather than two. Its own frame is the azimuth `phi`.
    e_y0 = out['inc_z0'][index] * np.exp(-1j * k0 * kz_inc * z_inc)
    e_x0 = out['inc_z0_x'][index] * np.exp(-1j * k0 * kz_inc * z_inc)
    a_s_inc = -s_i * e_x0 + c_i * e_y0
    a_p_inc = sig_i * (n_top / kz_inc) * (c_i * e_x0 + s_i * e_y0)
    # Whichever channel this leg was driven in is the one that normalises it. Dividing by the
    # other is dividing by round-off.
    a_inc = a_s_inc if abs(a_s_inc) >= abs(a_p_inc) else a_p_inc

    def face(dft_y, dft_x, kz, n_half, z_face, z_shift, sig):
        e_y = (ker_y @ dft_y) * np.exp(1j * k0 * kz * z_face * z_shift)
        e_x = (ker_x @ dft_x) * np.exp(1j * k0 * kz * z_face * z_shift)
        amp_s = (-s_m * e_x + c_m * e_y) / a_inc
        amp_p = sig * (n_half / kz) * (c_m * e_x + s_m * e_y) / a_inc
        return amp_s, amp_p

    r_s, r_p = face(out['dft_ref'][index][0], out['dft_ref_x'][index][0],
                    kz_top, n_top, z_ref, 1.0, sig_r)
    t_s, t_p = face(out['dft_trn'][index][0], out['dft_trn_x'][index][0], kz_bot, n_bot,
                    z_trn - grid.total_thickness, -1.0, sig_t)

    # An order evanescent in a half-space arrives already decayed; undoing that amplifies
    # round-off. R is identically zero for these, so nothing downstream would notice.
    for amp, kz, depth in ((r_s, kz_top, abs(z_ref)), (r_p, kz_top, abs(z_ref)),
                           (t_s, kz_bot, abs(z_trn - grid.total_thickness)),
                           (t_p, kz_bot, abs(z_trn - grid.total_thickness))):
        amp[np.exp(k0 * np.abs(np.imag(kz)) * depth) > MAX_DEREFERENCE_GAIN] = np.nan

    # On the PUBLISHED amplitudes both channels take the same weight on each leg -- the s/p
    # asymmetry lives on the internal boundary variables and is cancelled by the per-leg
    # conversion. Measured on the RCWA path to 5e-16.
    w_r = np.real(kz_top) / np.real(kz_inc)
    w_t = np.real(kz_bot) / np.real(kz_inc)
    sq = lambda a: np.nan_to_num(np.real(a * np.conj(a)), nan=0.0)
    R_s, R_p = sq(r_s) * w_r, sq(r_p) * w_r
    T_s, T_p = sq(t_s) * w_t, sq(t_p) * w_t
    row = lambda a: np.asarray(a)[None, :]
    return ResultSub(row(r_s), row(r_p), row(t_s), row(t_p),
                     row(R_s + R_p), row(R_s), row(R_p),
                     row(T_s + T_p), row(T_s), row(T_p))


def _extract(grid, out, half_width, index, centre=0):
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
    if grid.ny != 1:
        raise NotImplementedError(
            'fdtd: the extraction is still one-dimensional in the transverse plane. A crossed '
            'grating needs a 2D order set -- the propagating set becomes a DISC rather than a '
            'rectangle, and a per-axis copy of `_propagating_window` silently drops orders.')
    wavelength = float(out['wavelengths'][index])
    if half_width >= nx // 2:
        raise ValueError(
            f'fdtd: {2 * half_width + 1} orders asked of a {nx}-cell grid. The x transform '
            f'resolves |m| < {nx // 2}; beyond that an order aliases onto another one and comes '
            f'back with a plausible amplitude (and, on TM, a flipped sign from the half-cell '
            f'sample offset). Raise `resolution`, or ask for fewer orders.')

    m = np.arange(centre - half_width, centre + half_width + 1)
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

    # An index is `sqrt(eps*mu)`, so every kz and the whole order grid carry mu.
    n_top = np.sqrt(grid.n_top_sq)
    n_bot = np.sqrt(grid.n_bot_sq)
    kz_top = _half_space_kz(grid.n_top_sq, kx)
    kz_bot = _half_space_kz(grid.n_bot_sq, kx)
    kz_inc = _half_space_kz(grid.n_top_sq, kx_inc)

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

    r = (kernel @ out['dft_ref'][index][0]) / a_inc * np.exp(1j * k0 * kz_top * z_ref)
    t = ((kernel @ out['dft_trn'][index][0]) / a_inc
         * np.exp(-1j * k0 * kz_bot * (z_trn - grid.total_thickness)))

    # An order evanescent in a half-space arrives at the monitor already decayed; undoing that
    # amplifies whatever is left, which past some depth is round-off. `R` is identically zero
    # for these, so nothing downstream would notice a finite, plausible, meaningless number.
    r = np.where(np.exp(k0 * np.abs(np.imag(kz_top)) * abs(z_ref)) > MAX_DEREFERENCE_GAIN,
                 np.nan, r)
    t = np.where(np.exp(k0 * np.abs(np.imag(kz_bot))
                        * abs(z_trn - grid.total_thickness)) > MAX_DEREFERENCE_GAIN,
                 np.nan, t)

    # solve_1d.py:472-481. On the reflection side the two media are the same one, so the
    # material factor cancels and both polarisations read Re(kz_top/kz_inc). Transmission is
    # where the duality shows: an s channel's admittance is `kz/mu` and a p channel's is
    # `kz/eps`, which is the whole content of a magnetic half-space.
    eps_top, eps_bot = grid.eps_top, grid.eps_bot
    mu_top, mu_bot = grid.mu_top, grid.mu_bot
    reflected = np.nan_to_num(np.real(r * np.conj(r)), nan=0.0) * np.real(kz_top / kz_inc)
    if tetm == 'te':
        transmitted = (np.nan_to_num(np.real(t * np.conj(t)), nan=0.0)
                       * np.real(kz_bot / mu_bot / (kz_inc / mu_top)))
        amplitudes = (r, np.zeros_like(r), t, np.zeros_like(t))
        efficiencies = (reflected, np.zeros_like(reflected),
                        transmitted, np.zeros_like(transmitted))
    else:
        transmitted = (np.nan_to_num(np.real(t * np.conj(t)), nan=0.0)
                       * np.real(kz_bot / eps_bot / (kz_inc / eps_top)))
        # The p channel's primary amplitude is magnetic; the published t_p is the E ratio,
        # `Tt * ((n_top/mu_top) / (n_bot/mu_bot))` (solve_1d.py:505-508), which is
        # `t_H * n_top/n_bot` exactly at mu = 1.
        amplitudes = (np.zeros_like(r), r, np.zeros_like(t),
                      t * ((n_top / mu_top) / (n_bot / mu_bot)))
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


def _solve_conical_oblique(psi, phi, theta, return_diagnostics, **kw):
    """A conical mount away from normal: the six-component stepper, run once per eigenpolarisation.

    At ``ky != 0`` the two triples are coupled, so there is no pair of scalar solves to combine
    -- each leg is its own full run. The s and p directions are the two eigenpolarisations of a
    uniform medium at fixed ``(kx, ky)``, so a leg driven in one stays in it on the way in, and
    the grating is what mixes them. `given` is their combination, and linearity is exact
    because the operator is linear and the two runs share every coefficient.
    """
    legs = {}
    for leg in ('s', 'p'):
        legs[leg] = solve_fdtd(tetm='te', theta=theta, phi=phi, _leg=leg,
                               return_diagnostics=True, **kw)
    (s_res, s_diag), (p_res, _) = legs['s'], legs['p']
    single = not isinstance(s_res, list)
    s_list = [s_res] if single else s_res
    p_list = [p_res] if single else p_res

    if s_list[0].order_window != p_list[0].order_window:
        raise AssertionError(
            f'fdtd: the two legs disagree about the order window, {s_list[0].order_window} '
            f'against {p_list[0].order_window}.')
    a_s, a_p = np.sin(psi), np.cos(psi)

    out = []
    for s_one, p_one in zip(s_list, p_list):
        mix = ResultSub(*[a_s * np.asarray(getattr(s_one.given, n))
                          + a_p * np.asarray(getattr(p_one.given, n))
                          for n in ('r_s', 'r_p', 't_s', 't_p')],
                        *[None] * 6)
        mix = _efficiencies_from(mix, s_one, p_one, a_s, a_p)
        for sub in (mix, s_one.given, p_one.given):
            check_physical(np, np.asarray(sub.R), np.asarray(sub.T),
                           np.sqrt(s_diag['grid'].n_top_sq))
        out.append(Result(given=mix, s_inc=s_one.given, p_inc=p_one.given,
                          order_window=s_one.order_window))

    result = out[0] if single else out
    return (result, s_diag) if return_diagnostics else result


def _efficiencies_from(mix, s_one, p_one, a_s, a_p):
    """Rebuild the mixed leg's efficiencies from its own amplitudes, not by averaging.

    ``R`` is quadratic in the amplitude, so ``|a_s r_s + a_p r_p|**2`` is not
    ``a_s**2 |r_s|**2 + a_p**2 |r_p|**2`` -- the cross term is the whole content of a mixed
    polarisation. Each channel's flux weight is recovered from a leg where that channel is
    nonzero, which is exact because the weight is a property of the order and the half-space,
    not of the incidence.
    """
    def weight(name, amp_name):
        for one in (s_one, p_one):
            a = np.asarray(getattr(one.given, amp_name))
            e = np.asarray(getattr(one.given, name))
            power = np.real(a * np.conj(a))
            with np.errstate(invalid='ignore', divide='ignore'):
                w = np.where(power > 1e-30, e / np.where(power > 1e-30, power, 1.0), np.nan)
            if np.any(np.isfinite(w)):
                return np.nan_to_num(w, nan=0.0)
        return np.zeros_like(np.asarray(getattr(s_one.given, name)))

    sq = lambda a: np.nan_to_num(np.real(a * np.conj(a)), nan=0.0)
    R_s = sq(mix.r_s) * weight('R_s', 'r_s')
    R_p = sq(mix.r_p) * weight('R_p', 'r_p')
    T_s = sq(mix.t_s) * weight('T_s', 't_s')
    T_p = sq(mix.t_p) * weight('T_p', 't_p')
    return ResultSub(mix.r_s, mix.r_p, mix.t_s, mix.t_p,
                     R_s + R_p, R_s, R_p, T_s + T_p, T_s, T_p)


def _solve_conical(psi, phi, theta, tetm, return_diagnostics, **kw):
    """A ``psi`` incidence at ``theta = 0``, any ``phi``: two scalar solves, then projected.

    **Restricted to normal incidence, and the refusal below says why rather than that.** At
    ``theta = 0`` the in-plane wavevector is the origin, so ``ky`` is exactly zero whatever
    ``phi`` is -- the six-component conical stepper is never reached, the two scalar problems
    stay decoupled, and the entire ``phi`` dependence is the output frame. Away from normal
    that stops being true: ``ky != 0`` couples all six components through four terms, and no
    combination of TE and TM runs can produce that.

    Both legs come back for free because the scalar solves are independent, so `s_inc` and
    `p_inc` are the two runs and `given` is their combination -- the same contract
    `solve_1d_conical` offers, and the reason a caller can reweight the polarisation without
    solving again.
    """
    if tetm is not None:
        raise ValueError(
            f'fdtd: psi={psi!r} and tetm={tetm!r} together name the polarisation twice. '
            f'`psi` sets (a_s, a_p) = (sin psi, cos psi) and couples both; `tetm` picks one '
            f'decoupled scalar problem. Pass one.')

    phi_v = 0.0 if phi is None else float(np.real(phi))
    if float(np.real(theta)) != 0.0:
        return _solve_conical_oblique(psi, phi_v, theta, return_diagnostics, **kw)

    legs = {}
    for leg, tt in (('s', 'te'), ('p', 'tm')):
        legs[leg] = solve_fdtd(tetm=tt, theta=0.0, phi=None, return_diagnostics=True, **kw)
    (te_res, te_diag), (tm_res, tm_diag) = legs['s'], legs['p']
    single = not isinstance(te_res, list)
    te_list = [te_res] if single else te_res
    tm_list = [tm_res] if single else tm_res

    # Both legs are the same scene at the same incidence, so they land on the same window --
    # asserted rather than assumed, because a silent mismatch would pair up orders that are
    # not the same order and return a plausible number.
    if te_list[0].order_window != tm_list[0].order_window:
        raise AssertionError(
            f'fdtd: the two legs disagree about the order window, {te_list[0].order_window} '
            f'against {tm_list[0].order_window}.')
    (centre, half), _ = te_list[0].order_window

    period_x = float(kw['period'][0] if hasattr(kw['period'], '__len__') else kw['period'])
    band = np.atleast_1d(np.asarray(kw['wavelength'], dtype=float))
    grid = te_diag['grid']

    out = []
    for i, (te, tm) in enumerate(zip(te_list, tm_list)):
        wl = float(band[i])
        # theta = 0, so kx_inc is zero and the ladder is the grating harmonic alone.
        kx = np.arange(centre - half, centre + half + 1) * wl / period_x
        subs = {}
        for leg, (a_s, a_p) in (('given', (np.sin(psi), np.cos(psi))),
                                ('s_inc', (1.0, 0.0)), ('p_inc', (0.0, 1.0))):
            subs[leg] = _combine_at_normal_incidence(grid, te.given, tm.given, kx, wl,
                                                     phi_v, a_s, a_p)
        for sub in subs.values():
            check_physical(np, np.asarray(sub.R), np.asarray(sub.T),
                           np.sqrt(grid.n_top_sq))
        out.append(Result(order_window=((centre, half), (0, 0)), **subs))

    result = out[0] if single else out
    return (result, te_diag) if return_diagnostics else result


def solve_fdtd(structure, thickness, top, bot, period, wavelength, tetm=None,
               theta=0.0, phi=None, psi=None, mu=None, incidence_side='top', lattice=None,
               resolution=None, n_pml=PML_CELLS, n_periods=None, courant=COURANT,
               subpixel=True, truncation_order=None, return_diagnostics=False,
               dispersion=None, poles=None, pad_scale=1.0, grading=None,
               record_field=None, _leg=None):
    # `mu` is the raster's permeability field, layer-major beside `structure`, and the two
    # half-spaces carry theirs inside their `Material` -- the same two spellings the RCWA path
    # takes, so one `Stack` reaches both solvers unchanged.
    """Solve one TE or TM incidence in the time domain; returns a :class:`Result`.

    Every keyword this does not implement is refused by name rather than ignored -- see
    :func:`_refuse_out_of_scope`. The module docstring says why NumPy is the only backend.
    """
    if psi is not None:
        return _solve_conical(
            psi=psi, phi=phi, theta=theta, structure=structure, thickness=thickness, top=top,
            bot=bot, period=period, wavelength=wavelength, mu=mu,
            incidence_side=incidence_side, lattice=lattice, resolution=resolution,
            n_pml=n_pml, n_periods=n_periods, courant=courant, subpixel=subpixel,
            truncation_order=truncation_order, return_diagnostics=return_diagnostics,
            dispersion=dispersion, poles=poles, pad_scale=pad_scale, grading=grading,
            tetm=tetm)

    # `_leg` is set only by `_solve_conical`, which has already validated the incidence; it
    # names which eigenpolarisation this run drives. It is the one door `phi != 0` comes
    # through, so it bypasses that refusal and nothing else.
    name = _refuse_out_of_scope(structure, mu, theta, None if _leg else phi,
                                'te' if _leg else tetm, incidence_side, lattice, top, bot)

    period_x = float(period[0] if hasattr(period, '__len__') else period)
    period_y = (float(period[1]) if hasattr(period, '__len__') and len(period) > 1 else None)
    raster = np.asarray(structure, dtype=complex)[:, 0, :]
    top_material, bot_material = as_material(top, 'top'), as_material(bot, 'bot')
    eps_top = complex(np.asarray(top_material.epsilon, dtype=complex))
    eps_bot = complex(np.asarray(bot_material.epsilon, dtype=complex))
    mu_top = complex(1.0 if top_material.mu is None else top_material.mu)
    mu_bot = complex(1.0 if bot_material.mu is None else bot_material.mu)
    mu_raster = None
    if mu is not None:
        mu_raster = np.asarray(mu, dtype=complex)
        if mu_raster.ndim == 3 and mu_raster.shape[1] == 1:
            mu_raster = mu_raster[:, 0, :]
        if mu_raster.shape != raster.shape:
            raise ValueError(
                f'fdtd: mu is {mu_raster.shape} and structure is {raster.shape}. They are two '
                f'fields sampled on one raster, so they share a shape exactly.')

    # A sequence asks for a spectrum out of ONE run -- the thing a time-domain solve does that
    # a frequency-domain one cannot. The centre sets the source and, at oblique incidence, the
    # in-plane wavevector; see the `angles` entry in the diagnostics for what that means.
    band = np.atleast_1d(np.asarray(wavelength, dtype=float))
    if band.size < 1:
        raise ValueError('fdtd: wavelength is empty.')
    single = np.ndim(wavelength) == 0
    centre = float(0.5 * (band.min() + band.max()))
    wavelength = centre

    n_periods = N_PERIODS if n_periods is None else float(n_periods)

    dispersion, poles = _check_dispersion(dispersion, poles, raster)
    # The cell size follows the largest index in the scene, and a pole model can raise that a
    # long way above `sqrt(eps_infinity)` -- a Drude metal at 900 nm reaches n = 6.5 where its
    # eps_infinity is 1. Sizing the grid off eps_infinity alone would leave the metal at three
    # cells per wavelength. It is computed whether or not `resolution` was given, because the
    # graded mesh needs the TRUE maximum: deriving it back from the cell size instead assumes
    # the caller asked for exactly `CELLS_PER_WAVELENGTH`, so a caller who refines `resolution`
    # is read as having a higher-index material and the padding is coarsened to match -- which
    # pins the coarse cells at 45 nm however fine the structure gets, and floors the error at
    # 2.3e-3 while the uniform path keeps converging to 1.4e-4.
    indices = [abs(np.sqrt(eps_top * mu_top)), abs(np.sqrt(eps_bot * mu_bot)),
               float(np.sqrt(np.abs(raster * (1.0 if mu_raster is None
                                              else mu_raster))).max())]
    for entries in (poles or {}).values():
        indices.append(abs(np.sqrt(permittivity_at(entries, float(np.min(band)), 1.0))))
    n_max = float(max(indices))
    if resolution is None:
        resolution = CELLS_PER_WAVELENGTH * n_max / wavelength
    grid = _Grid(wavelength, period_x, raster, thickness, eps_top, eps_bot,
                 resolution, n_pml, courant, subpixel, theta=theta, phi=phi,
                 dispersion=dispersion, poles=poles, mu_raster=mu_raster,
                 mu_top=mu_top, mu_bot=mu_bot, pad_scale=pad_scale, grading=grading,
                 n_max=n_max, period_y=period_y)
    n_steps = int(np.ceil(n_periods * centre / grid.dt))
    out = _time_loop(grid, name, n_steps, band, leg=_leg, record_field=record_field)

    centre_order = 0
    if truncation_order is None:
        # The SHORTEST wavelength has the most propagating orders, so the default covers the
        # whole band rather than the centre of it.
        shortest = float(band.min())
        centre_order, half_width = _propagating_window(
            shortest, period_x, eps_top * mu_top, eps_bot * mu_bot,
            kx_inc=grid.kx_inc * shortest / centre, ky_inc=grid.ky_inc * shortest / centre)
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
        sub = (_extract_conical(grid, out, half_width, index, centre_order) if _leg
               else _extract(grid, out, half_width, index, centre_order))
        # The same tier of check the RCWA path applies: gain out of nothing, or negative
        # power. Every defect this module has had -- an over-conducting PML, a mis-scaled CFS
        # pole, an out-of-band DFT -- announced itself here first, as `R + T` between 1.1
        # and 106.
        check_physical(np, np.asarray(sub.R), np.asarray(sub.T),
                       np.sqrt(eps_top * mu_top))
        results.append(Result(given=sub, order_window=((centre_order, half_width), (0, 0))))

    result = results[0] if single else results
    if not return_diagnostics:
        return result

    # `kx` is fixed by the grid and `kx = n*sin(theta)`, so a BAND is a fixed-kx sweep and the
    # polar angle drifts across it. Reported rather than hidden: at normal incidence every
    # entry is 0 and there is nothing to read, and at oblique this is the difference between
    # what was asked for and what a single time-domain run can answer.
    angles = np.rad2deg(np.arcsin(np.clip(
        grid.kx_inc * band / centre / max(np.sqrt(eps_top).real, 1e-9), -1.0, 1.0)))
    # The time record, with the axes it is measured on. `z` is the node table -- the graded
    # mesh means it is NOT `arange(nz)*dz`, and a reader who assumes that plots the field
    # against the wrong abscissa on every graded run.
    record = {}
    if record_field is not None:
        record = {'field_t': out['snap_t'], 'field_e': out['snap_e'],
                  'field_ex': out['snap_x'],
                  'field_z': grid.z_node, 'field_x': np.arange(grid.nx) * grid.dx}

    return result, {'grid': grid, 'dz': grid.dz, 'nz': grid.nz, 'nx': grid.nx, 'dt': grid.dt,
                    **record,
                    'n_steps': n_steps, 'n_pml': grid.n_pml,
                    'pml_thickness': grid.n_pml * grid.dz, 'residual': out['residual'],
                    'incident_balance': abs(out['inc_trn'][0] / out['inc_ref'][0]) - 1.0,
                    'wavelengths': band,
                    'incident_fraction': (np.abs(out['inc_z0'])
                                          / max(np.max(np.abs(out['inc_z0'])), 1e-300)),
                    'angles_deg': angles if single else angles}
