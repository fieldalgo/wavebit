"""The stage-3 scenes, captures and lowered arrays that ``fd_golden_stage3.npz`` and its test share.

One module, imported by both ``interface/freeze_fd_stage3.py`` (which writes the golden) and
``tests/test_fd_stage3_backends.py`` (which reads it), so the two cannot disagree about which
scene a key names. The content digest and the relative gap are the stage-0 definitions,
imported rather than copied; the stage-2 sampling rule (cells clustered on the domain centre)
is imported too, because the reason it exists -- a TFSF or sheet scene's far cells are
round-off -- holds for every scene here.

**The scenes are built here and not by the ``tests/fd_helpers.py`` builders**
(``flux_box_dipole``, ``bloch_stack``, ``box_grating``, ``farfield_dipole``,
``conical_grating``). Those are shared with the physics gates, tuned for accuracy (14 to 160
periods, 20 to 80 cells per wavelength) and edited as those gates grow; a golden whose scenes
moved every time a helper gained a default would read as kernel drift. Every number below is
spelled out once, here, and each scene is small and short -- a drift detector needs the same
code paths, not a settled answer.

**The five scenes**, one per stage-3 result path, each a second or two on every backend:

``flux_dipole``
    Two electric dipoles (``Ez``, and ``Ex`` with its own amplitude and phase) in a 10-cell
    vacuum box with an 8-cell PML all round (26^3), with a ``Flux`` box around them (six faces,
    each a ``DFTRegion`` with E on one node plane and H on the two half planes either side), a
    finite ``Flux`` plane above, and a ``Field`` plane of two components below; each dipole's
    ``power()`` spectrum. Two and not one because an ``Ez`` dipole alone radiates no ``Hz``
    (its field is TM to z), and every box face's ``Hz`` accumulator would then be round-off --
    measured on the first draft of this list, JAX and Torch 1.0 to 1.8 of those arrays' own
    peaks off NumPy, the stage-2 README's pin on round-off exactly. Real.
``bloch_stack``
    A 25-degree p ``PlaneWave`` at ``phi = 30`` in a 4 x 3 Bloch cell over a three-cell glass
    layer, with a reflect and a transmit ``Diffraction`` monitor (the order extraction, the
    direction split, the Yee kz de-embedding, the s/p frame) and a full-cell ``Flux`` plane;
    the sheet's ``incident_amplitude()`` and ``intensity()`` spectra. Complex. A layered cell
    couples no s into a p wave, so the four s arrays of each monitor (``r_s``, ``R_s`` and
    ``t_s``, ``T_s``) are round-off and are **not frozen** (:func:`cross_polarised`).
``lamellar``
    A cell-snapped ``Box`` tooth (stage 3's structure) 12 x 4 cells in a 24-cell period, one
    cell deep in y, under a normally incident s ``PlaneWave``: three orders propagate (``m = -1,
    0, 1``), so the transverse index-DFT and every order's own kz are exercised, not only the
    zero order. The field is TE (``Ey``, ``Hx``, ``Hz``) by symmetry, so the p arrays and the
    ``Ex``/``Hy`` accumulators are **exactly** 0.0 on every backend -- a pin on an exact zero
    is a pin that any leak of p breaks, and the gate asserts they are exact.
``farfield``
    A magnetic ``Hx`` dipole on the tent stamp, with its own amplitude, phase and delay, and an
    electric ``Ey`` dipole beside it (a magnetic dipole alone has no ``Ex`` anywhere, its E
    being ``m x r``), in the same 26^3 PML box, with a ``FarField`` box around them: the
    intensity and ``(E_theta, E_phi)`` at :data:`FAR_ANGLES`, and the box's flux. Real.
``conical``
    A lamellar ``Box`` tooth 16 x 4 cells in a 32-cell period under a 15-degree p
    ``PlaneWave`` at ``phi = 60``, with y an ``fd.Axis1D`` (the exact ``i k`` derivative) and x
    Bloch: the conical path, where every order couples s and p. Complex.

**``bloch_stack`` and ``conical`` were re-frozen 2026-09-19 with ``--rebaseline``**, for one
reason: the first freeze pinned a launch defect of the PlaneWave column. Its hard source wrote
the continuum polarisation into the stored ``E_x`` and ``E_y`` without the half-cell stagger
phase each carries along its own axis (and not transverse to the grid's discrete ``K``), so both
scenes' conical p waves were launched with an s part first order in ``dx``; the drive now writes
the grid's own p wave (``wavebit/fd/_lower/sources.py::plane_wave_drive``). What moved, as a
fraction of each array's peak (the same on every backend): ``bloch_stack`` -- the accumulators
2.8e-2 to 5.9e-2, the sampled fields 5.1e-2 / 5.7e-2, ``incident_amplitude`` 5.1e-2 (the
stagger phases, ``k_x dx/2 = 0.058`` and ``k_y dx/2 = 0.033`` rad here), the energies, fluxes,
``R``, ``T``, ``R_p``, ``T_p`` and ``intensity`` 8e-6 to 9.5e-5, and its s arrays, which were the
defect itself (``|r_s| = 4.3e-3``, ``|t_s| = 8.8e-3``, ``R_s = 1.8e-5``), went to round-off
and out of the archive;
``conical`` -- the accumulators 5.9e-4 to 2.2e-2, the fields 7.4e-3 / 1.0e-2, the complex
amplitudes 1.9e-3 to 3.9e-2, the efficiencies 1.8e-5 to 2.6e-3, ``incident_amplitude`` 5.1e-3.
The other three scenes reproduced bit for bit.

**Every run is short** (300 to 500 steps: the band's pulse peaks at step 273 in the boxes and
234 in the columns), so every frequency-domain monitor is cut off before the field has
decayed, and ``fd.UnsettledFieldWarning`` fires on every scene, naming each monitor; the gate
asserts that rather than silencing it unseen. What a DFT of an unsettled field is, is still
a deterministic function of the code, which is all a drift detector needs.

**What is frozen**, per scene and backend:

- every DFT accumulator, whole (``dft__<region>__<label>``, ``(nf, *slab)`` complex; the regions
  are planes and box faces, none over 300 values), and each region's ``energy`` series --
  the raw observer output, before any post-processing;
- the final E and H at the stage-2 sample cells and the whole-volume energy (the stage-0 set,
  without the ``Record`` series: these scenes have none);
- every result the post-processing builds from them: each ``Flux`` monitor's ``flux`` and face
  shares, each ``Field`` monitor's phasors, each ``Diffraction`` monitor's ten ``Result`` arrays
  (``r_s`` ... ``T_p``) and its ``theta_at``, each ``FarField`` monitor's intensity and fields at
  :data:`FAR_ANGLES` and its flux;
- every source's stage-3 spectra (``power`` of a dipole, ``incident_amplitude`` and
  ``intensity`` of a PlaneWave, at the scene's wavelengths).

**Not frozen: the stage-2 reference series** the spectra are transforms of (a column's
``incident``, a dipole's ``work``). ``fd_golden_stage2.npz`` pins those on its own scenes, here
they would be 60% of the archive (a 500-step complex column series is 48 kB a backend), and
the gate holds the three backends to each other on every one of them live instead.

**Not frozen: the decay ratios.** ``res.diagnostics['decay']`` is ``(max - min) / peak`` of a
settled energy, a difference of nearly equal numbers -- its relative round-off is the
energy's (1e-16) divided by the ratio itself, 1e-7 of it on a settled run, so it cannot be held
to 1e-12 and a pin on it would be a pin on cancellation. The energy series it is computed from
*is* frozen, whole.

**Per scene and once**, from the NumPy lowering, the numbers a run is given that stage 3
added: each DFT region's angular frequencies (``params['dft']``) and, for the Axis1D scene,
the exact ``i k dx`` the curl multiplies by (``params['axis1d']``).
"""
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                          # tests/references, for the earlier cases
sys.path.insert(0, str(HERE.parent))                   # tests/, for fd_helpers

import wavebit.fd as fd                                                       # noqa: E402
from _fd_stage0_cases import BACKENDS, content_digest, relative_gap          # noqa: E402,F401
from _fd_stage2_cases import sample_indices                                   # noqa: E402,F401
from fd_helpers import to_numpy                                              # noqa: E402
from wavebit.fd._lower import lower                                          # noqa: E402
from wavebit.fd._run import _results, execute                                # noqa: E402
from wavebit.fd._runtime import runtime                                      # noqa: E402

#: The committed golden.
GOLDEN = HERE / 'fd_golden_stage3.npz'

#: Gate 8's budget, the same 1e-12 of an array's peak every earlier stage is held to.
TOL = 1e-12

#: The grid and the one band every source uses (centre wavelength 1, spectral sigma 0.15 f0).
DX = 0.05
BAND = dict(wavelength=1.0, df=0.15)
PML_CELLS = 8
GLASS = 2.25

#: The monitors' wavelengths: the band centre and one either side.
WAVELENGTHS = (0.9, 1.0, 1.1)

#: The gratings' wavelengths: at 0.9 the lamellar cell's ``(+-1, 0)`` orders would leave
#: at 42 degrees and at 1.1 at 66 -- past the grazing rule's reach for an 8-cell PML
#: (``cos_min = 0.41``); this set keeps every propagating order at exit ``cos >= 0.48``.
GRATING_WAVELENGTHS = (0.95, 1.0, 1.05)

#: The far-field directions frozen, degrees: ``theta`` from ``+z`` over both hemispheres and
#: the equator, ``phi`` off every symmetry plane of the grid.
FAR_ANGLES = (np.array([10.0, 35.0, 60.0, 90.0, 120.0, 155.0]),
              np.array([0.0, 30.0, 75.0, 140.0, 200.0, 290.0]))

#: What each scene's plan warns about on purpose, asserted by the gate: glass at the band's
#: short edge is 8.6 cells per wavelength at this dx (a drift detector's cost).
EXPECTED_WARNINGS = {
    'flux_dipole': (),
    'bloch_stack': ('CoarseMeshWarning',),
    'lamellar': ('CoarseMeshWarning',),
    'farfield': (),
    'conical': ('CoarseMeshWarning',),
}

#: What each run warns about on purpose: every scene stops early (see the module docstring).
RUN_WARNING = 'UnsettledFieldWarning'

#: The warnings the scenes raise on purpose, silenced where a plan or a run is only a means.
EXPECTED = (fd.CoarseMeshWarning, fd.UnsettledFieldWarning)

#: Which source series a kind of source fills (the stage-2 set).
SERIES_FIELDS = ('incident', 'incident_H', 'drive', 'work', 'field')

#: The ten arrays of a ``wavebit._core.result.ResultSub``.
RESULT_FIELDS = ('r_s', 'r_p', 't_s', 't_p', 'R', 'R_s', 'R_p', 'T', 'T_s', 'T_p')


def band():
    return fd.Band(**BAND)


def _layer():
    # 8 cells rather than the measured 12, as in the stage-1 and stage-2 goldens: a drift
    # detector, and the thinner layer halves the cost.
    return fd.PML(cells=PML_CELLS, allow_thin=True)


def _cube(interior):
    return fd.Domain(size=((interior + 2 * PML_CELLS) * DX,) * 3)


def flux_dipole_scene():
    """Two dipoles in a 26^3 PML box, with a Flux box, a Flux plane and a Field plane."""
    # Two dipoles, not one: an Ez dipole alone radiates no Hz at all (the field is TM to z),
    # so every box face's Hz accumulator would be round-off and a pin on it a pin on round-off.
    d = fd.Dipole('d', center=(0.013, -0.02, 0.007), component='Ez', band=band())
    e = fd.Dipole('e', center=(-0.04, 0.03, -0.02), component='Ex', band=band(),
                  amplitude=0.6, phase=1.1)
    monitors = (
        # 0.4, the largest box the 10-cell interior holds: it encloses both stamps (two cells
        # either side of each centre) with the one cell to spare a flux box must keep.
        fd.Flux('box', box=fd.Box(size=(0.4, 0.4, 0.4)), wavelengths=WAVELENGTHS),
        fd.Flux('plane', plane=fd.Plane(z=0.2, x=(-0.2, 0.15), y=(-0.1, 0.2)),
                wavelengths=WAVELENGTHS),
        fd.Field('field', plane=fd.Plane(z=-0.15, x=(-0.2, 0.2), y=(-0.15, 0.1)),
                 components=('Ex', 'Hy'), wavelengths=WAVELENGTHS),
    )
    return fd.Scene(domain=_cube(10), boundaries=fd.Boundaries.open(_layer()),
                    resolution=fd.Resolution(dx=DX), run=fd.Run(steps=300), sources=(d, e),
                    monitors=monitors)


def _stack_domain(nx_cells, ny_cells, interior_cells, center):
    return fd.Domain(size=(nx_cells * DX, ny_cells * DX, (interior_cells + 2 * PML_CELLS) * DX),
                     center=(0.0, 0.0, center))


def bloch_stack_source():
    return fd.PlaneWave('pw', z=0.3, direction='-z', theta=25.0, phi=30.0, polarization='p',
                        band=band(), amplitude=0.8, phase=0.4)


def bloch_stack_scene():
    """A 25-degree p PlaneWave in a 4 x 3 Bloch cell over a glass layer, two Diffraction planes."""
    pw = bloch_stack_source()
    top, bottom = -0.5 * DX, -3.5 * DX
    monitors = (
        fd.Diffraction('r', z=0.45, side='reflect', wavelengths=WAVELENGTHS,
                       reference_plane=top),
        fd.Diffraction('t', z=-0.4, side='transmit', wavelengths=WAVELENGTHS,
                       reference_plane=bottom, incident_plane=top),
        fd.Flux('below', plane=fd.Plane(z=-0.35), wavelengths=WAVELENGTHS),
    )
    return fd.Scene(domain=_stack_domain(4, 3, 24, 0.0),
                    boundaries=fd.Boundaries(**fd.Bloch.from_source(pw), z=_layer()),
                    resolution=fd.Resolution(dx=DX), run=fd.Run(steps=500), sources=(pw,),
                    structures=(fd.Structure(fd.Slab(z=(-3 * DX, 0.0)),
                                             fd.Material(eps=GLASS)),),
                    monitors=monitors)


def _tooth(width, height):
    # Runs past the one-cell domain in y and is clipped there: a lamellar tooth.
    return fd.Structure(fd.Box(size=(width, 10.0, height), center=(0.0, 0.0, -height / 2)),
                        fd.Material(eps=GLASS))


def _grating_monitors(height):
    top, bottom = -0.5 * DX, -height - 0.5 * DX
    return (fd.Diffraction('r', z=0.45, side='reflect', wavelengths=GRATING_WAVELENGTHS,
                           reference_plane=top),
            fd.Diffraction('t', z=-height - 0.3, side='transmit',
                           wavelengths=GRATING_WAVELENGTHS, reference_plane=bottom,
                           incident_plane=top))


def lamellar_scene():
    """A Box tooth 12 x 4 cells in a 24-cell period, one cell deep in y, normal incidence, s."""
    pw = fd.PlaneWave('pw', z=0.3, direction='-z', polarization='s', band=band())
    height = 4 * DX
    return fd.Scene(domain=_stack_domain(24, 1, 26, -0.05),
                    boundaries=fd.Boundaries(**fd.Bloch.from_source(pw), z=_layer()),
                    resolution=fd.Resolution(dx=DX), run=fd.Run(steps=500), sources=(pw,),
                    structures=(_tooth(12 * DX, height),), monitors=_grating_monitors(height))


def farfield_scene():
    """A magnetic ``Hx`` and an electric ``Ey`` dipole in a 26^3 PML box, in a FarField box."""
    m = fd.Dipole('m', center=(-0.03, 0.02, -0.01), component='Hx', kind='magnetic',
                  band=band(), stamp='tent', amplitude=0.5, phase=0.7, delay=0.3)
    # And an electric Ey dipole beside it: a magnetic Hx dipole alone has no Ex anywhere (its
    # E is m x r), which would leave two faces' Ex accumulators at round-off.
    j = fd.Dipole('j', center=(0.02, -0.015, 0.03), component='Ey', band=band(),
                  amplitude=0.3, phase=-0.5)
    return fd.Scene(domain=_cube(10), boundaries=fd.Boundaries.open(_layer()),
                    resolution=fd.Resolution(dx=DX), run=fd.Run(steps=300), sources=(m, j),
                    monitors=(fd.FarField('ff', box=fd.Box(size=(0.4, 0.4, 0.4)),
                                          wavelengths=WAVELENGTHS),))


def conical_source():
    return fd.PlaneWave('pw', z=0.3, direction='-z', theta=15.0, phi=60.0, polarization='p',
                        band=band())


def conical_scene():
    """A lamellar Box tooth under a conical p PlaneWave: x Bloch, y ``fd.Axis1D``."""
    pw = conical_source()
    height = 4 * DX
    return fd.Scene(domain=_stack_domain(32, 1, 26, -0.05),
                    boundaries=fd.Boundaries(**fd.Bloch.from_source(pw, axis1d='y'), z=_layer()),
                    resolution=fd.Resolution(dx=DX), run=fd.Run(steps=500), sources=(pw,),
                    structures=(_tooth(16 * DX, height),), monitors=_grating_monitors(height))


def scenes():
    """``{name: scene}`` -- the five stage-3 scenes; see the module docstring for each."""
    return {'flux_dipole': flux_dipole_scene(), 'bloch_stack': bloch_stack_scene(),
            'lamellar': lamellar_scene(), 'farfield': farfield_scene(),
            'conical': conical_scene()}


def describe(scene, plan):
    """The static description of a scene the archive's metadata carries, and the test rebuilds."""
    return dict(
        skeleton=plan.skeleton, cells=[plan.nx, plan.ny, plan.nz], steps=plan.steps,
        dx=plan.dx, dt=plan.dt, dtype=plan.dtype,
        monitors=[[m.name, type(m).__name__] for m in scene.monitors],
        sources=[[s.name, type(s).__name__] for s in scene.sources],
        terms=[type(t).__name__ for t in plan.terms],
        regions=[[o.name, o.monitor, [label for label, _, _ in o.parts]]
                 for o in dft_regions(plan)],
        absorbing=[list(f) for f in plan.absorbing_faces],
        spans=[list(s) for s in plan.spans])


def solve(scene, backend, plan=None):
    """``(res, state)``: ``fd.run``'s own three calls, keeping the final state it discards.

    ``fd.run`` is ``plan``, ``lower``, ``execute`` and ``_results``; the raw DFT accumulators
    live in the state and nowhere in ``Results``, so the capture needs the state too. None of
    these scenes asks for ``polarizations='both'``, the one case where ``run`` solves twice.
    """
    rt = runtime(backend)
    p = fd.plan(scene) if plan is None else plan
    params = lower(scene, p, rt)
    state = execute(p, params, rt)
    return _results(scene, p, state, rt, params), state


def dft_regions(plan):
    """The plan's ``DFTRegion`` observers (a ``PointRecord`` has a ``field``, a region ``parts``)."""
    return [o for o in plan.observers if hasattr(o, 'parts')]


def _np(value):
    return np.ascontiguousarray(to_numpy(value))


def capture(res, state, samples):
    """Every array frozen from one run, by key; see the module docstring."""
    E, H = to_numpy(res.final.E), to_numpy(res.final.H)
    idx = tuple(samples.T)
    out = {'final__E': np.ascontiguousarray(E[idx]),
           'final__H': np.ascontiguousarray(H[idx]),
           'energy': np.array(float(np.sum(np.abs(E) ** 2) + np.sum(np.abs(H) ** 2)))}
    for region in dft_regions(res.plan):
        for label, value in state['obs'][region.name].items():
            out[f'dft__{region.name}__{label}'] = _np(value)
    layered = polarisation_preserving(res.plan)
    for name in res:
        arrays = monitor_arrays(res[name])
        if layered and type(res[name]).__name__ == 'DiffractionResult':
            arrays = {k: v for k, v in arrays.items() if k not in cross_polarised(res[name])}
        out.update({f'result__{name}__{k}': v for k, v in arrays.items()})
    for name, series in res.sources.items():
        if series.kind.startswith('Dipole'):
            out[f'source__{name}__power'] = _np(series.power())
        else:
            out[f'source__{name}__incident_amplitude'] = _np(series.incident_amplitude())
            out[f'source__{name}__intensity'] = _np(series.intensity())
    return out


def polarisation_preserving(plan):
    """Whether every structure of ``plan`` is a slab across one axis (no ``Box``): a layered,
    isotropic cell, which never couples s and p -- on the grid as in the continuum, because
    with the stagger removed its equations are the continuum's with ``k -> K``."""
    return all(not row[0].startswith('box') for row in plan.spans)


def cross_polarised(result):
    """The ``Result`` fields of the polarisation a ``Diffraction`` monitor's source did not
    launch. In a :func:`polarisation_preserving` cell they are round-off (1e-16, measured) since
    the PlaneWave launches the grid's own s or p wave, so they are **not frozen** -- a pin on
    round-off is O(1) of its own peak -- and ``tests/test_fd_stage3_backends.py`` holds them
    under a line live instead."""
    other = 'p' if result.polarization == 's' else 's'
    return {f'r_{other}', f't_{other}', f'R_{other}', f'T_{other}'}


def monitor_arrays(result):
    """``{key: array}`` of one frequency-domain result, as NumPy."""
    out = {}
    kind = type(result).__name__
    if kind == 'FluxResult':
        out['flux'] = _np(result.flux)
        for face, value in sorted((result.faces or {}).items()):
            out[f'face__{face}'] = _np(value)
    elif kind == 'FieldResult':
        for component, value in sorted(result.fields.items()):
            out[f'field__{component}'] = _np(value)
    elif kind == 'DiffractionResult':
        sub = result.orders.given
        for field in RESULT_FIELDS:
            value = getattr(sub, field)
            if value is not None:
                out[field] = _np(value)
        out['theta_at'] = _np(result.theta_at)
    elif kind == 'FarFieldResult':
        theta, phi = FAR_ANGLES
        out['intensity'] = _np(result.intensity(theta, phi))
        E_theta, E_phi = result.fields(theta, phi)
        out['E_theta'], out['E_phi'] = _np(E_theta), _np(E_phi)
        out['flux'] = _np(result.flux)
        for face, value in sorted((result.faces or {}).items()):
            out[f'face__{face}'] = _np(value)
    else:
        raise TypeError(f'no capture for a {kind}')
    return out


def lowered(scene, plan):
    """The arrays frozen once per scene from the NumPy lowering: what stage 3 added to params."""
    params = lower(scene, plan, runtime('numpy'))
    out = {}
    for name, omega in sorted(params.get('dft', {}).items()):
        out[f'lowered__dft__{name}'] = np.ascontiguousarray(np.asarray(omega))
    for axis, value in sorted((params.get('axis1d') or {}).items()):
        out[f'lowered__axis1d__{axis}'] = np.ascontiguousarray(np.asarray(value))
    return out
