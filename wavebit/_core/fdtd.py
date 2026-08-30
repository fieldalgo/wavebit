"""A 2D FDTD solve for a 1D-periodic grating, answering the same question as ``solve_1d``.

This is roadmap item 2 -- a time-domain solver reading the same :class:`~wavebit._core.domain.
Stack` and :class:`~wavebit._core.domain.Incidence` the RCWA path reads, so that one structure
can be solved two ways and the answers compared. Its whole value is that its errors are
*different* errors: RCWA truncates a Fourier series and FDTD discretises space and time, so a
defect that survives both is unlikely to be a defect of either.

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

#: 1/e half-width of the source envelope, in optical cycles, and the delay before its peak in
#: units of that width. Together they put a hard floor of ``SOURCE_CYCLES * SOURCE_DELAY``
#: cycles on any useful run, which :func:`_time_loop` refuses below.
SOURCE_CYCLES = 8.0
SOURCE_DELAY = 4.0

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
            'fdtd: the scene has a complex permittivity. A time-domain solve carries loss as a '
            'conductivity, and a constant conductivity reproduces a given Im(eps) at ONE '
            'frequency only -- so a lossy material here would be right at the centre '
            'wavelength and silently wrong beside it. Dispersion (an ADE Drude/Lorentz pole) '
            'is what makes this answerable and it is not implemented.')
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
            'permeability slot at all". So there would be nothing to compare against.')

    if theta is not None and float(np.real(theta)) != 0.0:
        raise ValueError(
            f'fdtd: theta={theta!r}. Only normal incidence is implemented. An oblique plane '
            f'wave on a periodic cell needs a Bloch phase across the cell edge, which a '
            f'real-valued time-domain field cannot carry at a single frequency -- it needs a '
            f'split-field or complex-field formulation, which is a different solver.')
    if phi is not None and float(np.real(phi)) != 0.0:
        raise ValueError(f'fdtd: phi={phi!r}. See the theta refusal: normal incidence only.')
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
# the source, and the 1D auxiliary incident field
# ---------------------------------------------------------------------------------------

def gaussian_pulse(t, f0, width, t0):
    """A Gaussian-modulated sine, the standard broadband FDTD excitation."""
    return np.sin(2.0 * np.pi * f0 * (t - t0)) * np.exp(-((t - t0) / width) ** 2)


class _Incident1D:
    """The incident plane wave, on a 1D grid filled with the superstrate medium.

    Taflove & Hagness sec. 5.8. Generating the incident field on a 1D grid sharing ``dz`` and
    ``dt`` with the 2D one makes its numerical dispersion identical along z, so the
    total-field/scattered-field cancellation is limited by round-off rather than by the
    dispersion mismatch an analytic plane wave would leave at the boundary. Measured on an
    empty grid, the residual in the scattered-field region is 1.2e-17 of the incident
    amplitude; a sign or index mutant anywhere in the coupling drives it to 1e-2 or worse, so
    the check has ~15 orders of resolving power and :mod:`tests.test_fdtd` runs it.

    The pair ``(e, h)`` integrates ``dh/dt = de/dz`` and ``eps de/dt = dh/dz``. For TE that is
    ``(Ey, Hx)``; for TM it is ``(Ex, -Hy)``, and **that minus sign has two consumers** -- the
    TFSF correction and the incident amplitude the result is normalised by. Getting it wrong
    in only one of them flips the sign of ``r_p`` while leaving every efficiency intact.
    """

    def __init__(self, nz, dz, dt, eps_r, n_pml, k_src, wavelength, f0, width, t0):
        self.dz, self.dt, self.eps_r, self.k_src = dz, dt, eps_r, k_src
        self.f0, self.width, self.t0 = f0, width, t0
        self.e = np.zeros(nz)
        self.h = np.zeros(nz - 1)
        self.node, self.half = cpml_profiles(nz, n_pml, dz, dt, wavelength, eps_r, eps_r)
        self.psi_e = np.zeros(nz)
        self.psi_h = np.zeros(nz - 1)

    def step_h(self):
        d = (self.e[1:] - self.e[:-1]) / self.dz
        self.psi_h = self.half['b'] * self.psi_h + self.half['a'] * d
        self.h += self.dt * (d / self.half['kappa'] + self.psi_h)

    def step_e(self, t):
        d = np.zeros_like(self.e)
        d[1:-1] = (self.h[1:] - self.h[:-1]) / self.dz
        self.psi_e = self.node['b'] * self.psi_e + self.node['a'] * d
        self.e[1:-1] += (self.dt / self.eps_r) * (d[1:-1] / self.node['kappa'][1:-1]
                                                  + self.psi_e[1:-1])
        # Additive rather than imposed: a soft source is transparent to whatever comes back
        # through it, which matters because the source sits between the PML and the monitor.
        self.e[self.k_src] += gaussian_pulse(t, self.f0, self.width, self.t0)


# ---------------------------------------------------------------------------------------
# the grid
# ---------------------------------------------------------------------------------------

def _shift_up(a):
    """``a[:, i+1]`` with the periodic wrap the cell edge carries."""
    return np.concatenate([a[:, 1:], a[:, :1]], axis=1)


def _shift_down(a):
    """``a[:, i-1]`` with the periodic wrap."""
    return np.concatenate([a[:, -1:], a[:, :-1]], axis=1)


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
                 resolution, n_pml, courant, subpixel):
        self.wavelength = float(wavelength)
        self.period = float(period)
        self.raster = np.asarray(raster, dtype=complex)
        self.thickness = np.asarray(thickness, dtype=float)
        self.eps_top = complex(eps_top)
        self.eps_bot = complex(eps_bot)
        self.subpixel = bool(subpixel)
        self.n_pml = int(n_pml)

        self.nx = max(int(round(self.period * resolution)), 1)
        self.dx = self.period / self.nx
        self.dz = self.dx

        n_top = float(np.sqrt(self.eps_top).real)
        n_bot = float(np.sqrt(self.eps_bot).real)

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


# ---------------------------------------------------------------------------------------
# the time loop
# ---------------------------------------------------------------------------------------

def _time_loop(grid, tetm, n_steps, wavelength):
    """Step the fields and accumulate the running DFT at each monitor plane."""
    freq = 1.0 / wavelength                        # c = 1
    width = SOURCE_CYCLES * wavelength
    t0 = SOURCE_DELAY * width
    if n_steps * grid.dt < t0 + 2.0 * width:
        raise ValueError(
            f'fdtd: {n_steps} steps is {n_steps * grid.dt / wavelength:.1f} optical cycles, '
            f'and the source alone is not finished emitting until '
            f'{(t0 + 2.0 * width) / wavelength:.0f}. A run this short returns a number that '
            f'looks like an efficiency -- measured 0.033 against a true 0.543 at 20 cycles -- '
            f'so it is refused rather than reported with a residual nobody reads.')

    nz, nx, dz, dx, dt = grid.nz, grid.nx, grid.dz, grid.dx, grid.dt
    hb, ha, hk = (grid.half[k][:, None] for k in ('b', 'a', 'kappa'))
    nb, na, nk = (grid.node[k][:, None] for k in ('b', 'a', 'kappa'))
    kt = grid.k_tfsf

    inc = _Incident1D(nz, dz, dt, grid.eps_top.real, grid.n_pml, grid.k_src,
                      wavelength, freq, width, t0)

    if tetm == 'te':
        # (Ey, Hx, Hz). Ey sits on the z nodes and the x nodes, and is tangential to every
        # interface a 1D grating has, so it takes no harmonic average.
        eps = grid.eps_for(False, False, None).real
        ey = np.zeros((nz, nx))
        hx = np.zeros((nz - 1, nx))
        hz = np.zeros((nz, nx))
    else:
        # (Hy, Ex, Ez). Ex is normal to the tooth's vertical walls, Ez to the horizontal ones.
        eps_x = grid.eps_for(False, True, 'x').real
        eps_z = grid.eps_for(True, False, 'z').real
        hy = np.zeros((nz - 1, nx))
        ex = np.zeros((nz, nx))
        ez = np.zeros((nz - 1, nx))
    psi_h = np.zeros((nz - 1, nx))
    psi_e = np.zeros((nz, nx))

    dft_ref = np.zeros(nx, dtype=complex)
    dft_trn = np.zeros(nx, dtype=complex)
    inc_ref = inc_trn = inc_z0 = 0.0 + 0.0j
    peak = tail = 0.0

    for step in range(n_steps):
        # --- H, to level n + 1/2 --------------------------------------------------------
        inc.step_h()
        if tetm == 'te':
            # Yee 1966: dHx/dt = dEy/dz, dHz/dt = -dEy/dx. Only z carries a PML.
            d = (ey[1:] - ey[:-1]) / dz
            psi_h = hb * psi_h + ha * d
            hx += dt * (d / hk + psi_h)
            hz -= dt * (_shift_up(ey) - ey) / dx
            # TFSF (Umashankar & Taflove 1982): Hx at kt-1/2 is a scattered-field quantity
            # whose stencil reached across the boundary into the total-field Ey[kt].
            hx[kt - 1] -= dt * inc.e[kt] / dz
        else:
            # dHy/dt = dEz/dx - dEx/dz.
            d_ez = (_shift_up(ez) - ez) / dx
            d_ex = (ex[1:] - ex[:-1]) / dz
            psi_h = hb * psi_h + ha * d_ex
            hy += dt * (d_ez - d_ex / hk - psi_h)
            # Ex_inc is `inc.e` outright -- see the sign note on _Incident1D.
            hy[kt - 1] += dt * inc.e[kt] / dz

        # --- E, to level n + 1 ----------------------------------------------------------
        inc.step_e((step + 1) * dt)
        if tetm == 'te':
            # eps dEy/dt = dHx/dz - dHz/dx.
            d = np.zeros((nz, nx))
            d[1:-1] = (hx[1:] - hx[:-1]) / dz
            psi_e = nb * psi_e + na * d
            ey[1:-1] += dt * (d[1:-1] / nk[1:-1] + psi_e[1:-1]
                              - (hz[1:-1] - _shift_down(hz)[1:-1]) / dx) / eps[1:-1]
            ey[kt] -= dt * inc.h[kt - 1] / dz / eps[kt]
        else:
            # eps dEx/dt = -dHy/dz ; eps dEz/dt = dHy/dx.
            d = np.zeros((nz, nx))
            d[1:-1] = -(hy[1:] - hy[:-1]) / dz
            psi_e = nb * psi_e + na * d
            ex[1:-1] += dt * (d[1:-1] / nk[1:-1] + psi_e[1:-1]) / eps_x[1:-1]
            # Hy_inc is MINUS inc.h, which is where this sign comes from.
            ex[kt] -= dt * inc.h[kt - 1] / dz / eps_x[kt]
            ez += dt * (hy - _shift_down(hy)) / dx / eps_z

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
        kernel = np.exp(2j * np.pi * freq * t_now) * dt
        dft_ref = dft_ref + kernel * f_ref
        dft_trn = dft_trn + kernel * f_trn
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


def _extract(grid, out, half_width, wavelength):
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
    kernel = np.exp(-1j * (2.0 * np.pi * m[:, None] / grid.period) * xs[None, :]) / nx

    n_top = np.sqrt(grid.eps_top)
    n_bot = np.sqrt(grid.eps_bot)
    k0 = 2.0 * np.pi / wavelength
    kx = m * wavelength / grid.period                    # normal incidence
    kz_top = _half_space_kz(grid.eps_top, kx)
    kz_bot = _half_space_kz(grid.eps_bot, kx)
    kz_inc = _half_space_kz(grid.eps_top, 0.0)

    z_ref, z_trn, z_inc = out['z_ref'], out['z_trn'], out['z_inc']
    # The incident amplitude at z = 0, from the plane it was actually sampled on.
    a_inc = out['inc_z0'] * np.exp(-1j * k0 * kz_inc * z_inc)

    r = (kernel @ out['dft_ref']) / a_inc * np.exp(1j * k0 * kz_top * z_ref)
    t = ((kernel @ out['dft_trn']) / a_inc
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

def solve_fdtd(structure, thickness, top, bot, period, wavelength, tetm,
               theta=0.0, phi=None, mu=None, incidence_side='top', lattice=None,
               resolution=None, n_pml=PML_CELLS, n_periods=None, courant=COURANT,
               subpixel=True, truncation_order=None, return_diagnostics=False):
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
    wavelength = float(wavelength)

    if resolution is None:
        n_max = max(abs(np.sqrt(eps_top)), abs(np.sqrt(eps_bot)),
                    float(np.sqrt(np.abs(raster)).max()))
        resolution = CELLS_PER_WAVELENGTH * n_max / wavelength
    n_periods = N_PERIODS if n_periods is None else float(n_periods)

    grid = _Grid(wavelength, period_x, raster, thickness, eps_top, eps_bot,
                 resolution, n_pml, courant, subpixel)
    n_steps = int(np.ceil(n_periods * wavelength / grid.dt))
    out = _time_loop(grid, name, n_steps, wavelength)

    if truncation_order is None:
        half_width = _propagating_half_width(wavelength, period_x, eps_top, eps_bot)
    elif hasattr(truncation_order, 'get'):
        half_width = int(truncation_order.get('x', truncation_order.get('u', 0)))
    else:
        half_width = int(truncation_order[0] if hasattr(truncation_order, '__len__')
                         else truncation_order)

    sub = _extract(grid, out, half_width, wavelength)
    # The same tier of check the RCWA path applies: gain out of nothing, or negative power.
    # Every defect this module has had -- an over-conducting PML, a mis-scaled CFS pole, an
    # out-of-band DFT -- announced itself here first, as `R + T` between 1.1 and 106.
    check_physical(np, np.asarray(sub.R), np.asarray(sub.T), np.sqrt(eps_top))

    result = Result(given=sub, order_window=((0, half_width), (0, 0)))
    if not return_diagnostics:
        return result
    return result, {'dz': grid.dz, 'nz': grid.nz, 'nx': grid.nx, 'dt': grid.dt,
                    'n_steps': n_steps, 'n_pml': grid.n_pml,
                    'pml_thickness': grid.n_pml * grid.dz, 'residual': out['residual'],
                    'incident_balance': abs(out['inc_trn'] / out['inc_ref']) - 1.0}
