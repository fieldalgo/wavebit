"""Gate 7 of stage 3 -- the grazing-order rule, refused and accepted per the measurement.

``wavebit/fd/_lower/grazing.py`` refuses a propagating order (or a PlaneWave's own incidence)
that a PEC-backed CFS-PML face cannot absorb to the amplitude tolerance ``tau = 1e-3``: the
layer's continuum round trip is ``R_c = reflection ** (0.919 cos theta)``, a face reaches down to
``cos_min(R) = ln(tau) / (0.919 ln R)`` (0.41 at the default 1e-8, 0.27 at 1e-12, 0.204 at
1e-16), below 0.41 it takes ``PML(cells >= 24)`` and ``Run(periods >= 160)`` as well, and below
0.204 nothing is accepted. ``tests/test_fd_monitors_doors.py`` holds the doors (every refusal
and its accepted spelling). This gate holds the rule to the measurement it stands on:

* **Each verdict is what the run measures.** A PlaneWave leaving at exit ``cos theta`` toward
  a PML face, in a 1 x 1 x nz cell whose two transverse axes are ``fd.Axis1D`` (the public
  spelling of the study's one-cell column), both z faces the same ``PML(cells, reflection)``,
  160 periods, 20 cells per wavelength. The up-going wave the face returns is read at two
  probes 5 and 6 cells in front of it (a Tukey-tapered DFT at the band centre, split with the
  Yee ``kz`` of the Axis1D dispersion -- ``studies/fdtd_grazing_return.py``'s ``R_taper``, read
  off the run itself). Measured (NumPy):

      cos    face              verdict   return    R_c
      0.26   PML(12, 1e-8)     weak      6.1e-3    1.2e-2
      0.26   PML(24, 1e-12)    weak      6.1e-4    1.4e-3    (the grid is below R_c; see below)
      0.26   PML(24, 1e-16)    ok        4.2e-5    1.5e-4
      0.342  PML(12, 1e-8)     weak      1.3e-3    3.1e-3
      0.342  PML(24, 1e-12)    ok        6.2e-5    1.7e-4
      0.15   PML(24, 1e-16)    floor     1.6e-3    6.2e-3
      0.5    PML(12, 1e-8)     ok        8.2e-5    2.1e-4

  Every accepted face returns less than ``tau``; every refused one has ``R_c > tau``, and the
  measured return always sits below ``R_c`` (the grid approaches it from below).
* **Why ``R_c`` and not the return at hand.** At fixed physical thickness (1.2 wavelengths of
  1e-12 layer, ``cos 0.26``) the return *rises* with resolution toward ``R_c``: 6.1e-4 / 9.1e-4 /
  1.11e-3 at 20 / 40 / 80 cells per wavelength (slow test) -- past ``tau`` at 80, so a rule judged
  on the coarse grid's return would accept a face that a finer grid shows failing.
* **Why 160 periods.** The same accepted face (``cos 0.342``, ``PML(24, 1e-12)``) read by a
  plain running DFT, as a user's monitor reads it: 4.4e-3 at 40 periods (the band's
  near-Rayleigh tail, not the layer), 5.3e-5 at 160.
* **Why 24 cells.** A 1e-12 target on 12 cells reflects a non-grazing wave (``cos 0.7``) *more*
  than the default layer does -- 1.0e-4 against 2.1e-5 -- and on 24 cells 1.2e-8.
* **The message quotes the study's table.** ``GRAZING_TABLE`` (which every refusal carries
  verbatim) is checked number by number against ``studies/results/fdtd_grazing_return.log``:
  the three 20-cell-per-wavelength rows against its 640-period ``R_taper`` table, the 80-cell row
  against its fixed-thickness table, the window rows against its control 3, and the continuum
  row against ``R_c`` computed here from the CPML's grading constants by quadrature.
* **A stabilized face is judged by what it returns** (``PML(stabilize=3e-3)``, the rule's
  ``'lossy'`` verdict). The exact discrete return of ``wavebit/fd/_lower/layer_return.py`` is
  what the column run measures -- 1.548e-3 at ``cos 0.6`` and 7.465e-4 at 0.7 on 12 cells, to
  four digits -- and the rule allows an order up to ``max(tau, |r_pure|)``: the tolerance, or
  what the same face returns without the loss, whichever is larger. It crosses at ``cos``
  0.6602 / 0.6571 (TE / TM) on 12 cells and 0.4717 / 0.4752 on 24, where the verdict flips,
  in the incidence's own polarisation. Holding the loss's own part ``|r - r_pure|`` to
  ``tau`` instead (the rule as first reviewed) was wrong both ways, and the column runs show
  it: an order the loss part-cancels on returns 7.96e-4 (accepted now, refused then) and
  orders it adds to return 1.28e-3 / 1.39e-3 (refused now, accepted then); on one glass face
  it refused s at 5.9e-4 and accepted p at 1.45e-3. Holding the total to ``tau`` alone
  refused normal incidence on a coarse grid, where the loss is inert. Every offer a refusal
  makes (the fewest cells, the largest ``stabilize``, on a ``'needs'`` row with the 24 cells
  and 160 periods the pure stretch needs) is accepted when applied, and a quoted
  ``reflection<=`` reaches the order it is quoted for. Each row is the model called with the
  layer's medium, the stretch's match, the design wavelength and (on a Bloch axis) the grid's
  ``K``; every propagating order is judged, a face's orders in one march; ``fd.run`` surveys
  once and reports its own scene. A refusal quotes the worst refused row and names the rest;
  an offer is judged by the survey on every row of the faces it replaces (a substrate grating
  where the quoted row alone would take 35 cells takes 38), spelled with every argument of
  the face, and not made when another face is refused too. A Diffraction wavelength below 8
  cells per wavelength is refused at the plan. Mutants caught: the rule ignoring the loss,
  the total held to ``tau``, the loss alone held to ``tau``, the lowest order only, both
  polarisations on the incidence, the cell search and the 24-cell floor on a ``'needs'`` row,
  offers judged on the quoted row, on the named face alone, or on the scene's own face, the
  first refused row quoted, offers made past another face's refusal, the face's other
  arguments dropped, ``'needs'`` read off the quoted row, the medium read at the offered depth,
  the run surveying twice, the Diffraction door off, and the polarisation picked by ``|r|``.
"""
import math
import os
import re
import warnings

import numpy as np
import pytest

import wavebit.fd as fd
from wavebit.fd._lower import cpml, grazing
from wavebit.fd._lower.layer_return import grid_k, layer_return
from fd_helpers import to_numpy

TAU = grazing.GRAZING_TOLERANCE
BAND = dict(wavelength=1.0, df=0.15)

LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'studies', 'results', 'fdtd_grazing_return.log')


def _column(cos, cells, reflection, periods=160, cells_per_wavelength=20, background=None,
            polarization='s'):
    """A PlaneWave at exit ``cos`` in a one-cell column (x and y Axis1D), both z faces
    ``PML(cells, reflection)``, two probes of the tangential E the wave carries (Ey for s, Ex
    for p) 5 and 6 cells in front of the z- layer; vacuum unless ``background`` names the
    medium filling it."""
    dx = 1.0 / cells_per_wavelength
    above, distance = 4, cells_per_wavelength
    nz = cells + above + distance + cells
    z_sheet = -(cells + above) * dx
    z_face = z_sheet - distance * dx
    pw = fd.PlaneWave('pw', z=z_sheet, direction='-z', theta=math.degrees(math.acos(cos)),
                      polarization=polarization, band=fd.Band(**BAND))
    component = 'Ey' if polarization == 's' else 'Ex'
    probes = tuple(fd.Record(f'p{i}', at=(0.0, 0.0, z_face + (5 + i) * dx), component=component)
                   for i in (0, 1))
    return fd.Scene(domain=fd.Domain(size=(dx, dx, nz * dx), center=(0.0, 0.0, -nz * dx / 2)),
                    boundaries=fd.Boundaries(**fd.Bloch.from_source(pw, background=background,
                                                                    axis1d=('x', 'y')),
                                             z=fd.PML(cells=cells, reflection=reflection)),
                    resolution=fd.Resolution(dx=dx), run=fd.Run(periods=periods),
                    sources=(pw,), monitors=probes,
                    **({} if background is None else dict(background=background)))


def _return(res, cos, taper=True):
    """``|up| / |down|`` at the probes at the band centre: the face's return."""
    t = np.asarray(res['p0'].t, dtype=float)
    v = [to_numpy(res[f'p{i}'].values) for i in (0, 1)]
    u = np.arange(len(t)) / len(t)
    window = (np.where(u < 0.5, 1.0, 0.5 * (1 + np.cos(np.pi * (u - 0.5) / 0.5))) if taper
              else np.ones(len(t)))
    w = 2 * math.pi / BAND['wavelength']
    E0, E1 = (complex(np.sum(x * np.exp(1j * w * t) * window)) for x in v)
    dx, dt = res.plan.dx, res.plan.dt
    kx = w * math.sqrt(1 - cos ** 2)                    # exact along the Axis1D x
    kappa = 2 * math.asin(dx * math.sqrt((math.sin(w * dt / 2) / dt) ** 2 - kx ** 2 / 4))
    up = (E1 - E0 * np.exp(-1j * kappa)) / (np.exp(1j * kappa) - np.exp(-1j * kappa))
    return abs(up) / abs(E0 - up)


def _run_unenforced(scene, monkeypatch):
    """``fd.run`` with the grazing refusal switched off, so a refused face can be measured."""
    with monkeypatch.context() as m:
        m.setattr(grazing, 'enforce', lambda scene, plan: None)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return fd.run(scene)


def _verdicts(res):
    return {row['verdict'] for row in res.diagnostics['grazing']}


# ---------------------------------------------------------------------------------------------
# Each verdict against the return the run measures
# ---------------------------------------------------------------------------------------------

#: (exit cos, cells, reflection, verdict) -- the measured return in the module docstring.
CASES = [(0.26, 12, 1e-8, 'weak'), (0.26, 24, 1e-12, 'weak'), (0.26, 24, 1e-16, 'ok'),
         (0.342, 12, 1e-8, 'weak'), (0.342, 24, 1e-12, 'ok'), (0.15, 24, 1e-16, 'floor'),
         (0.5, 12, 1e-8, 'ok')]


@pytest.mark.parametrize('cos, cells, reflection, verdict', CASES)
def test_each_verdict_is_what_the_face_measurably_returns(cos, cells, reflection, verdict,
                                                          monkeypatch):
    scene = _column(cos, cells, reflection)
    if verdict == 'ok':
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fd.plan(scene)
    else:
        with pytest.raises(ValueError, match='Grazing order refused') as e:
            fd.plan(scene)
        assert f'exit cos theta = {cos:.3f}' in str(e.value)
    res = _run_unenforced(scene, monkeypatch)
    assert _verdicts(res) == {verdict}
    rows = res.diagnostics['grazing']
    assert {row['face'] for row in rows} == {'z-', 'z+'}
    assert all(row['what'] == 'incidence' and abs(row['cos'] - cos) < 1e-12 for row in rows)
    R_c = grazing.continuum_return(reflection, cos)
    measured = _return(res, cos)
    assert measured < R_c, (measured, R_c)              # the grid approaches R_c from below
    if verdict == 'ok':
        assert measured < TAU, measured
        assert R_c <= TAU
    else:
        assert R_c > TAU, R_c
        if (cos, reflection) != (0.26, 1e-12):          # that one is refused on R_c alone
            assert measured > TAU, measured


@pytest.mark.slow
def test_a_finer_grid_returns_more_toward_the_continuum_and_crosses_the_tolerance(monkeypatch):
    """1.2 wavelengths of a 1e-12 layer at cos 0.26: 6.1e-4 / 9.1e-4 / 1.11e-3 at 20 / 40 / 80."""
    cos, R = 0.26, 1e-12
    got = [_return(_run_unenforced(_column(cos, int(1.2 * cpw), R, cells_per_wavelength=cpw),
                                   monkeypatch), cos)
           for cpw in (20, 40, 80)]
    assert got[0] < got[1] < got[2] < grazing.continuum_return(R, cos), got
    assert got[0] < TAU < got[2], got                   # the coarse grid would have passed it


def test_the_run_must_last_160_periods_because_the_window_reads_the_band(monkeypatch):
    """An accepted face read by a plain running DFT: 4.4e-3 at 40 periods, 5.3e-5 at 160."""
    cos = 0.342
    short = _column(cos, 24, 1e-12, periods=40)
    with pytest.raises(ValueError, match='Run\\(periods>=160\\)'):
        fd.plan(short)
    at_40 = _return(_run_unenforced(short, monkeypatch), cos, taper=False)
    at_160 = _return(_run_unenforced(_column(cos, 24, 1e-12), monkeypatch), cos, taper=False)
    assert at_40 > TAU > at_160, (at_40, at_160)


def test_a_low_target_needs_24_cells_or_the_other_orders_get_worse(monkeypatch):
    """cos 0.7 (every face accepts it): 2.1e-5 default, 1.0e-4 on 12 cells at 1e-12, 1.2e-8 on 24."""
    cos = 0.7
    default, thin, thick = (_return(_run_unenforced(_column(cos, cells, R), monkeypatch), cos)
                            for cells, R in ((12, 1e-8), (12, 1e-12), (24, 1e-12)))
    assert thin > 2 * default, (thin, default)
    assert thick < default / 100, (thick, default)
    with pytest.raises(ValueError, match='PML\\(cells>=24'):
        fd.plan(_column(0.342, 12, 1e-12))                # where the rule asks for the cells


# ---------------------------------------------------------------------------------------------
# The message quotes the study's measured table
# ---------------------------------------------------------------------------------------------

def _table_rows():
    """``{label: [7 strings]}`` of ``GRAZING_TABLE``'s numeric rows, in order."""
    rows = {}
    window = 0
    for line in grazing.GRAZING_TABLE.splitlines():
        numbers = re.findall(r'\d\.\de[-+]\d\d', line)
        if len(numbers) != 7:
            continue
        label = line[:line.index(numbers[0])].strip()
        if not label:
            window += 1
            label = f'window {window}'
        rows[label] = numbers
    return rows


def _log_lines():
    # A failure, not a skip: the table every refusal quotes must stay checked against the
    # measurement, so the log is part of the repository beside the study that writes it.
    assert os.path.exists(LOG), (f'the study log is missing: {LOG}; rerun '
                                 f'studies/fdtd_grazing_return.py and commit its log')
    with open(LOG, encoding='utf-8') as f:
        return f.read().splitlines()


def _after(lines, start, prefix):
    for i in range(start, len(lines)):
        if lines[i].startswith(prefix):
            return i
    raise AssertionError(f'{prefix!r} not in the study log after line {start}')


def test_the_refusal_quotes_the_table_and_the_table_is_the_studys():
    with pytest.raises(ValueError) as e:
        fd.plan(_column(0.26, 12, 1e-8))
    message = str(e.value)
    assert grazing.GRAZING_TABLE in message
    assert 'R_c = 1e-08**(0.919*0.260) = 1.23e-02' in message
    assert 'cos_min = 0.408' in message and 'PML(cells=12, reflection=1e-08)' in message
    rows = _table_rows()
    assert len(rows) == 7
    lines = _log_lines()
    # The 640-period R_taper table at 20 cells per wavelength.
    taper = _after(lines, _after(lines, 0, '=== R_taper'), '20 cpw, window 640 periods')
    measured = {}
    for line in lines[taper + 2:taper + 11]:
        cells, target, _, *values = line.split()
        measured[(int(cells), float(target))] = values
    assert rows['20 cpw 12 cells 1e-8'] == measured[(12, 1e-8)]
    assert rows['20 cpw 24 cells 1e-12'] == measured[(24, 1e-12)]
    assert rows['20 cpw 24 cells 1e-16'] == measured[(24, 1e-16)]
    # The fixed-thickness table: 1.2 wavelengths of 1e-8 at 80 cells per wavelength.
    fixed = _after(lines, _after(lines, 0, '=== fixed physical thickness, s, up_taper'),
                   '1.2 wl 1e-08 W640')
    assert rows['80 cpw 1.2 wl   1e-8'] == [e.split('/')[2]
                                            for e in lines[fixed].split(':')[1].split()]
    # Control 3: what a perfect layer still leaves at 40 and 160 periods (its third number).
    for label, window in (('window 1', 40), ('window 2', 160)):
        i = _after(lines, 0, f'  20 cpw, window {window:>4}:')
        assert rows[label] == [e.split('/')[2] for e in lines[i].split(':')[1].split()]
    # The continuum row, from the grading constants by quadrature (the study's way).
    f = np.linspace(0.0, 1.0, 20001)
    m, a = cpml.ORDER, cpml.ALPHA_RELATIVE
    g = (m + 1) * np.trapezoid(f ** m / (1 + (a * (1 - f)) ** 2), f)
    cosines = (0.15, 0.2, 0.26, 0.3, 0.4, 0.5, 0.7)
    assert rows['continuum R_c   1e-8'] == [f'{1e-8 ** (g * c):.1e}' for c in cosines]
    assert abs(g - grazing.GRAZING_EXPONENT) < 1e-8


# ---------------------------------------------------------------------------------------------
# A stabilized face is judged by its actual return
# ---------------------------------------------------------------------------------------------

RHO = cpml.STABILIZING_LOSS
#: A row's numbers against ``layer_return`` called directly: the survey marches every order of
#: a face at once (an array ``K``), which agrees with the one-order call to 2e-11 relative
#: (``tests/test_fd_layer_return.py``), so this is that gate's line, not bit equality.
MODEL = 1e-9


def _stabilized(cos, cells, reflection=1e-8, rho=RHO, periods=160, face=None, **column):
    """:func:`_column` with both z faces ``PML(cells, reflection, stabilize=rho, **face)``."""
    scene = _column(cos, cells, reflection, periods=periods, **column)
    z = fd.PML(cells=cells, reflection=reflection, stabilize=rho, **(face or {}))
    return scene.replace(boundaries=fd.Boundaries(x=scene.boundaries.x, y=scene.boundaries.y,
                                                  z=z))


def _plan_unenforced(scene):
    """``fd.plan`` with the grazing refusal switched off, so a refused scene's rows can be read."""
    saved = grazing.enforce
    grazing.enforce = lambda scene, plan: None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return fd.plan(scene)
    finally:
        grazing.enforce = saved


def _survey(scene):
    return grazing.survey(scene, _plan_unenforced(scene))


def _verdict_pairs(scene):
    return {(r['verdict'], r.get('pure_verdict')) for r in _survey(scene)}


def _plans(scene):
    """Whether ``fd.plan`` accepts ``scene``."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fd.plan(scene)
        return True
    except ValueError as e:
        assert 'refused' in str(e), e
        return False


def _modeled(plan, cos, cells, side, rho=RHO, reflection=1e-8, n=1.0, match=1.0,
             design=None, wavelength=None):
    """``(r_TE, r_TM)`` of the column's ``side`` face, from ``layer_return`` directly: the
    column's x and y are Axis1D, so the order's transverse ``K`` is its ``k`` exactly."""
    wavelength = BAND['wavelength'] if wavelength is None else wavelength
    K = n * 2 * math.pi * math.sqrt(1 - cos ** 2) / wavelength
    return layer_return(cells, plan.dx, plan.dt, BAND['wavelength'] if design is None else design,
                        match, side, reflection, rho, wavelength, K, eps=n * n)


def _loss(plan, cos, cells, side, pol=0, **kwargs):
    """``|r - r_pure|`` of one polarisation (0 TE, 1 TM): the loss's own part."""
    r = _modeled(plan, cos, cells, side, **kwargs)
    r0 = _modeled(plan, cos, cells, side, **dict(kwargs, rho=0.0))
    return abs(r[pol] - r0[pol])


def _excess(plan, cos, cells, side, pol=0, **kwargs):
    """``|r| / max(tau, |r_pure|)`` of one polarisation: above 1 is refused."""
    r = _modeled(plan, cos, cells, side, **kwargs)[pol]
    r0 = _modeled(plan, cos, cells, side, **dict(kwargs, rho=0.0))[pol]
    return abs(r) / max(TAU, abs(r0))


def _crossing(cells, pol=0, low=0.3, high=0.95):
    """The exit cos at which the modeled return (the worse face) reaches what it is allowed."""
    plan = _plan_unenforced(_stabilized(0.7, cells))

    def worst(cos):
        return max(_excess(plan, cos, cells, side, pol) for side in '-+')
    for _ in range(50):
        mid = 0.5 * (low + high)
        low, high = (mid, high) if worst(mid) > 1 else (low, mid)
    return high


def _narrow(fn):
    """``fn()`` with the column's band narrowed to one wavelength and one angle."""
    saved = BAND.copy()
    BAND.update(wavelength=1.0, df=0.005)
    try:
        return fn()
    finally:
        BAND.clear()
        BAND.update(saved)


#: One offer of a ``'lossy'`` refusal as the message spells it: where (``z=`` for both faces
#: of an axis declared as one face, ``the 'z-' face`` for one), and every argument of the face.
OFFER = re.compile(r"(z=|the '(z[-+])' face )PML\(cells=(\d+), reflection=([0-9.e+-]+), "
                   r"stabilize(=|<=)([0-9.e+-]+|None)((?:, wavelength=[0-9.e+-]+)?"
                   r"(?:, background=Material\(eps=\([^)]*\), mu=[^)]*\))?(?:, allow_thin=True)?)"
                   r"\)( and Run\(periods>=(\d+)\))?")


def _offers(message, full=False):
    """``[(cells, reflection, rho, periods or None), ...]`` of a ``'lossy'`` refusal; with
    ``full``, ``(where, face name or '', cells, reflection, rho, the other arguments as
    spelled, periods or None)``."""
    fixes = message[message.index('What passes'):message.index('Or make the order')]
    out = []
    for where, name, c, R, _, r, extra, _, p in OFFER.findall(fixes):
        offer = (int(c), float(R), None if r == 'None' else float(r), int(p) if p else None)
        out.append((where, name) + offer[:3] + (extra,) + offer[3:] if full else offer)
    return out


def _apply(cos, offer, periods=160, **column):
    cells, R, rho, run = offer
    return _stabilized(cos, cells, reflection=R, rho=rho, periods=run or periods, **column)


def test_a_stabilized_face_is_refused_by_what_it_returns():
    """cos 0.6 on 12 cells, the column's s wave (TE): the pure stretch returns 2.2e-5 and is
    accepted; with the loss the face returns 1.55e-3, above both ``tau`` and the 2.2e-5 it
    returns without the loss, and it is refused. The message quotes the model's numbers and
    what would pass -- 16 cells, or ``stabilize <= 1.9e-3`` (below the measured 3e-3, which it
    says) -- and both pass while 15 cells and 2.0e-3 do not."""
    cos, cells = 0.6, 12
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fd.plan(_column(cos, cells, 1e-8))                    # the pure stretch: as today
    scene = _stabilized(cos, cells)
    with pytest.raises(ValueError, match='Stabilized face refused') as e:
        fd.plan(scene)
    message = str(e.value)
    plan = _plan_unenforced(scene)
    rows = grazing.survey(scene, plan)
    assert [row['verdict'] for row in rows] == ['lossy', 'lossy']
    for row in rows:
        side = row['face'][1]
        assert row['polarisations'] == ('TE',) and row['polarisation'] == 'TE'
        assert row['pure_verdict'] == 'ok'
        assert row['return'] == pytest.approx(abs(_modeled(plan, cos, cells, side)[0]), rel=MODEL)
        assert row['return_pure'] == pytest.approx(
            abs(_modeled(plan, cos, cells, side, rho=0.0)[0]), rel=MODEL)
        assert row['return_loss'] == pytest.approx(_loss(plan, cos, cells, side), rel=MODEL)
        assert row['allowed'] == TAU and row['excess'] == row['return'] / TAU
        assert row['return_pure'] < TAU < row['return']
        assert all(type(row[k]) is float
                   for k in ('return', 'return_pure', 'return_loss', 'allowed', 'excess'))
    first = rows[0]
    assert f'exit cos theta = {cos:.3f}' in message and 'stabilize=0.003' in message
    assert f"it sends back {first['return']:.2e} of this order (TE)" in message
    assert f"above the {first['return_pure']:.2e} the same face returns without" in message
    assert f"|r - r_pure|, is {first['return_loss']:.2e}" in message
    assert _offers(message) == [(16, 1e-8, 3e-3, None), (12, 1e-8, 1.9e-3, None)]
    assert 'stabilize<=0.0019)' in message
    assert 'below the 3e-3 MEASURED_STABILIZE gives' in message
    # What the message offers does pass, and one step short of it does not.
    assert _verdict_pairs(_stabilized(cos, 16)) == {('ok', 'ok')}
    assert _verdict_pairs(_stabilized(cos, 15)) == {('lossy', 'ok')}
    assert _verdict_pairs(_stabilized(cos, cells, rho=1.9e-3)) == {('ok', 'ok')}
    assert _verdict_pairs(_stabilized(cos, cells, rho=2.0e-3)) == {('lossy', 'ok')}


@pytest.mark.parametrize('cells_per_wavelength', [8.4, 9])
def test_at_normal_incidence_a_stabilized_face_is_its_pure_stretch(cells_per_wavelength):
    """The loss is inert at ``k_par = 0``, so a stabilized face returns exactly what its pure
    stretch does and gets its verdict, however coarse the grid. At 8.4 / 9 cells per wavelength
    (above the resolution floor) the 12-cell face returns 2.14e-3 / 1.54e-3 -- above ``tau``,
    and not judged on the pure stretch, which the rule holds to ``R_c`` -- and that is what it
    is allowed (``max(tau, |r_pure|)``): both are accepted, at ``excess`` exactly 1."""
    def rows():
        for s in (_column(1.0, 12, 1e-8, cells_per_wavelength=cells_per_wavelength),
                  _stabilized(1.0, 12, cells_per_wavelength=cells_per_wavelength)):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                fd.plan(s)                                    # neither is refused
        return _survey(s)
    got = _narrow(rows)
    assert {r['verdict'] for r in got} == {'ok'}
    assert all(r['return_loss'] == 0.0 and r['return'] == r['return_pure'] > TAU
               and r['excess'] == 1.0 for r in got)


def test_mutant_the_total_return_held_to_tau_refuses_normal_incidence(monkeypatch):
    """The rule as first written held the face's total discrete return to ``tau`` alone: it
    refuses the normal-incidence column at 9 cells per wavelength, where the loss returns
    nothing."""
    monkeypatch.setattr(grazing, 'refuses', lambda row: row['return'] > TAU)

    def plan():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fd.plan(_stabilized(1.0, 12, cells_per_wavelength=9))
    with pytest.raises(ValueError, match='Stabilized face refused'):
        _narrow(plan)


#: The four cases where the loss's own return and the face's total disagree about ``tau``,
#: each measured on the column run (20 cells per wavelength, vacuum, the incidence's own
#: polarisation): ``(cos, cells, reflection, rho, polarization, verdict, measured, model,
#: loss alone)``. The loss part-cancels the stretch's return in the first (the total is below
#: ``tau`` although ``|r - r_pure|`` is above it) and adds to it in the other two (the total is
#: 1.28x / 1.37x ``tau`` although ``|r - r_pure|`` is below it).
DISAGREE = [(0.69, 12, 1e-16, 3e-3, 'p', 'ok', 7.961e-4, 7.961e-4, 1.032e-3),
            (0.43, 32, 1e-8, 5e-3, 's', 'lossy', 1.278e-3, 1.277e-3, 9.34e-4),
            (0.28, 32, 1e-12, 5e-4, 'p', 'lossy', 1.390e-3, 1.370e-3, 9.62e-4)]


@pytest.mark.parametrize('case', DISAGREE, ids=['cancels', 'adds-s', 'adds-p'])
def test_the_face_is_held_to_what_it_returns_not_to_the_loss_alone(case, monkeypatch):
    """The verdict is on ``|r| <= max(tau, |r_pure|)``, and the run measures that total: an
    order the loss part-cancels on is accepted though ``|r - r_pure| > tau``, and one it adds
    to is refused though ``|r - r_pure| < tau``."""
    cos, cells, R, rho, pol, verdict, measured, model, loss = case
    scene = _stabilized(cos, cells, reflection=R, rho=rho, polarization=pol)
    rows = _survey(scene)
    assert {(r['verdict'], r['pure_verdict'], r['polarisation']) for r in rows} == {
        (verdict, 'ok', 'TE' if pol == 's' else 'TM')}
    row = rows[0]
    assert row['return'] == pytest.approx(model, rel=2e-3)
    assert row['return_loss'] == pytest.approx(loss, rel=2e-3)
    assert (row['return_loss'] > TAU) is (verdict == 'ok')        # the loss alone would invert it
    assert _plans(scene) is (verdict == 'ok')
    got = _return(_run_unenforced(scene, monkeypatch), cos)
    assert got == pytest.approx(measured, rel=1e-3) and abs(got / model - 1) < 2e-2
    assert bool(got > TAU) is (verdict == 'lossy')


def test_mutant_the_loss_alone_held_to_tau_inverts_a_pair():
    """The rule as the review found it, ``|r - r_pure| <= tau``, on one glass face at one
    angle (``PML(24, 1e-8, stabilize=1e-3)``, 40 cells per wavelength, ``eps = 2.25``, cos
    0.43): it refused the s wave, which the face returns at 5.9e-4 (measured 5.926e-4; the
    loss's part is 1.03e-3), and accepted the p wave at 1.45e-3 (measured 1.455e-3; the loss's
    part 9.8e-4). The rule accepts s and refuses p; the mutant turns both round."""
    def pair():
        return [_verdict_pairs(_stabilized(0.43, 24, rho=1e-3, polarization=p, background=2.25,
                                           cells_per_wavelength=40)) for p in 'sp']
    assert pair() == [{('ok', 'ok')}, {('lossy', 'ok')}]
    saved = grazing.refuses
    grazing.refuses = lambda row: row['return_loss'] > TAU
    try:
        assert pair() == [{('lossy', 'ok')}, {('ok', 'ok')}]
    finally:
        grazing.refuses = saved


@pytest.mark.parametrize('cos, verdict', [(0.6, 'lossy'), (0.7, 'ok')])
def test_the_stabilized_verdict_is_what_the_run_returns(cos, verdict, monkeypatch):
    """The s wave the column launches is TE: the run returns 1.548e-3 at cos 0.6 and 7.465e-4 at
    cos 0.7 on 12 cells, the model's TE 1.548e-3 and 7.465e-4 (4 digits)."""
    res = _run_unenforced(_stabilized(cos, 12), monkeypatch)
    assert _verdicts(res) == {verdict}
    te, _ = _modeled(res.plan, cos, 12, '-')
    measured = _return(res, cos)
    assert abs(measured / abs(te) - 1) < 1e-2, (measured, abs(te))
    assert bool(measured > TAU) is (verdict == "lossy"), measured


@pytest.mark.parametrize('cells, pol, measured', [(12, 0, 0.66023), (12, 1, 0.65712),
                                                  (24, 0, 0.47166), (24, 1, 0.47520)],
                         ids=['12-TE', '12-TM', '24-TE', '24-TM'])
def test_the_verdict_flips_where_the_modeled_return_crosses_tau(cells, pol, measured):
    """Stabilized at 3e-3, the modeled return reaches ``tau`` (its pure stretch is far below
    it there) at exit cos 0.6602 (TE) / 0.6571 (TM) on 12 cells and 0.4717 / 0.4752 on 24
    (where the pure-stretch rule allows down to 0.41): refused just below, accepted just
    above, in the incidence's own polarisation."""
    c = _crossing(cells, pol)
    assert abs(c - measured) < 2e-5, c
    polarization = 'sp'[pol]
    with pytest.raises(ValueError, match='Stabilized face refused'):
        fd.plan(_stabilized(c - 0.005, cells, polarization=polarization))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fd.plan(_stabilized(c + 0.005, cells, polarization=polarization))


def test_the_incidence_is_judged_in_its_own_polarisation():
    """On 24 cells the TE return crosses ``tau`` at cos 0.4717 and the TM one at 0.4752: at cos
    0.4734 an s wave (TE) is accepted and a p wave (TM) refused."""
    cos = 0.4734
    assert _crossing(24, pol=0) < cos < _crossing(24, pol=1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fd.plan(_stabilized(cos, 24))
    rows = _survey(_stabilized(cos, 24, polarization='p'))
    assert {r['verdict'] for r in rows} == {'lossy'}
    assert {(r['polarisations'], r['polarisation']) for r in rows} == {(('TM',), 'TM')}
    with pytest.raises(ValueError, match=r'of this order \(TM\)'):
        fd.plan(_stabilized(cos, 24, polarization='p'))


def test_mutant_the_incidence_judged_in_both_polarisations(monkeypatch):
    """Both polarisations judged on the incidence row, as a diffracted order is: the s wave
    at cos 0.4734 on 24 cells, whose TE return is below ``tau``, is refused."""
    saved = grazing.judge_returns
    monkeypatch.setattr(grazing, 'judge_returns', lambda r, r0, pols: saved(r, r0, (0, 1)))
    with pytest.raises(ValueError, match='Stabilized face refused'):
        fd.plan(_stabilized(0.4734, 24))


def test_mutant_the_rule_ignoring_the_loss_accepts_what_it_returns(monkeypatch):
    """The rule before the loss was modelled: a stabilized face judged as its pure stretch. It
    accepts cos 0.6 on 12 cells, which the face returns at 1.55e-3."""
    monkeypatch.setattr(grazing, '_stabilize', lambda face: 0.0)
    scene = _stabilized(0.6, 12)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        plan = fd.plan(scene)
    rows = grazing.survey(scene, plan)
    assert {r['verdict'] for r in rows} == {'ok'} and not any('return' in r for r in rows)


@pytest.mark.parametrize('case, verdict', [
    (dict(), 'lossy'),
    (dict(background=2.25), 'ok'),
    (dict(background=2.25, face=dict(background=1.0)), 'ok'),
    (dict(face=dict(wavelength=0.9)), 'lossy'),
], ids=['vacuum', 'glass', 'glass-matched-to-vacuum', 'design-0.9'])
def test_the_row_is_the_model_in_the_layers_medium_and_match(case, verdict):
    """At cos 0.6 on 12 cells the face returns 1.55e-3 in vacuum (refused) and 6.72e-4 in a
    glass column, ``eps = 2.25`` (accepted); 7.54e-4 there with the stretch matched to vacuum
    instead; 1.56e-3 in vacuum with the face designed for 0.9 (its CFS pole moved). Each row is
    ``layer_return`` called with the order's index ``n``, the face's match and its design
    wavelength -- so the medium, the match and the design each reach the model."""
    cos, cells = 0.6, 12
    scene = _stabilized(cos, cells, **case)
    plan = _plan_unenforced(scene)
    rows = grazing.survey(scene, plan)
    n = math.sqrt(case.get('background', 1.0))
    kw = dict(n=n, match=case.get('face', {}).get('background', n * n),
              design=case.get('face', {}).get('wavelength'))
    assert {r['verdict'] for r in rows} == {verdict}
    for row in rows:
        side = row['face'][1]
        assert row['n'] == pytest.approx(n, rel=1e-15)
        assert row['return'] == pytest.approx(abs(_modeled(plan, cos, cells, side, **kw)[0]),
                                              rel=MODEL)
        assert row['return_pure'] == pytest.approx(
            abs(_modeled(plan, cos, cells, side, **dict(kw, rho=0.0))[0]), rel=MODEL)
        assert row['return_loss'] == pytest.approx(_loss(plan, cos, cells, side, **kw),
                                                   rel=MODEL)


# ---------------------------------------------------------------------------------------------
# What a refusal offers passes
# ---------------------------------------------------------------------------------------------

#: ``(cos, cells, reflection, rho, periods, pure verdict, offers)``: each ``'lossy'`` refusal's
#: offers as its message spells them (``(cells, reflection, stabilize, periods)``).
REFUSALS = [
    (0.35, 12, 1e-12, 3e-3, 160, 'needs', [(43, 1e-12, 3e-3, None), (24, 1e-12, 6.9e-4, None)]),
    (0.35, 12, 1e-12, 3e-3, 40, 'needs', [(43, 1e-12, 3e-3, 160), (24, 1e-12, 6.9e-4, 160)]),
    (0.30, 30, 1e-12, 3e-3, 40, 'needs', [(55, 1e-12, 3e-3, 160), (30, 1e-12, 6.9e-4, 160)]),
    (0.38, 12, 1e-12, 5e-4, 160, 'needs', [(24, 1e-12, 5e-4, None)]),
    (0.38, 24, 1e-12, 3e-3, 160, 'ok', [(38, 1e-12, 3e-3, None), (24, 1e-12, 9.3e-4, None)]),
    (0.45, 24, 1e-8, 3e-3, 160, 'ok', [(27, 1e-8, 3e-3, None), (24, 1e-8, 2.5e-3, None)]),
    (0.55, 12, 1e-8, 3e-3, 160, 'ok', [(19, 1e-8, 3e-3, None), (12, 1e-8, 1.3e-3, None)]),
]


@pytest.mark.parametrize('case', REFUSALS, ids=[f'cos{c[0]}-{c[1]}cells-{c[2]:g}-{c[3]:g}-'
                                                f'{c[4]}periods' for c in REFUSALS])
def test_every_offer_of_a_refusal_passes_when_applied(case):
    """Each offer is judged by the rule's own code before it is made (``grazing._rejudge``),
    so applied to the scene it is accepted -- on a ``'needs'`` row too, where every offer
    carries the 24 cells and the 160 periods the pure stretch needs (the review found a
    ``stabilize`` offered at 12 cells there, still refused). On ``'needs'`` rows the tail
    names the target the face already has, not a new one."""
    cos, cells, R, rho, periods, pure, offers = case
    scene = _stabilized(cos, cells, reflection=R, rho=rho, periods=periods)
    assert _verdict_pairs(scene) == {('lossy', pure)}
    with pytest.raises(ValueError, match='Stabilized face refused') as e:
        fd.plan(scene)
    message = str(e.value)
    assert _offers(message) == offers
    for offer in offers:
        applied = _apply(cos, offer, periods)
        assert _verdict_pairs(applied) == {('ok', 'ok')}, offer
        assert _plans(applied), offer
    if pure == 'needs':
        assert f'its reflection={R:g} already reaches this order' in message
        assert 'reflection<=' not in message


def test_the_offered_cells_are_the_fewest_that_pass():
    """The cell search re-judges the whole face at each count: on 24 cells at cos 0.45 the
    fewest is 27 -- 26 is still refused, and so is every count below it."""
    cos, cells = 0.45, 24
    for m in range(cells, 27):
        assert _verdict_pairs(_stabilized(cos, m)) == {('lossy', 'ok')}, m
    assert _verdict_pairs(_stabilized(cos, 27)) == {('ok', 'ok')}
    scene = _stabilized(cos, cells)
    row = _survey(scene)[0]
    fix = grazing.lossy_fixes(row, scene, _plan_unenforced(scene))
    assert fix['cells'] == 27 and fix['rho_cells'] == 24 and fix['periods'] is None


def test_on_a_needs_row_the_cells_offered_are_what_the_pure_stretch_needs():
    """cos 0.38 on 12 cells at ``reflection=1e-12``, ``stabilize=5e-4``: the loss alone would
    pass from 18 cells (the rule's verdict there is ``'needs'``, not ``'lossy'``), but the pure
    stretch needs 24 below cos 0.408 -- so 24 is the offer, and the only one (at 24 cells this
    ``stabilize`` already passes)."""
    cos, R, rho = 0.38, 1e-12, 5e-4
    assert _verdict_pairs(_stabilized(cos, 12, reflection=R, rho=rho)) == {('lossy', 'needs')}
    for m in range(13, 24):
        assert _verdict_pairs(_stabilized(cos, m, reflection=R, rho=rho)) == {
            ('lossy' if m < 18 else 'needs', 'needs')}, m
    scene = _stabilized(cos, 12, reflection=R, rho=rho)
    fix = grazing.lossy_fixes(_survey(scene)[0], scene, _plan_unenforced(scene))
    assert fix == dict(periods=None, cells=24, rho=None, rho_cells=24)


def test_a_quoted_reflection_target_reaches_the_order():
    """The pure rule's ``reflection<=`` is rounded down, and the printed value is checked: at
    cos 0.35 the target is 4.695e-10, which ``.1e`` printed as 4.7e-10 -- whose ``cos_min`` is
    0.350017, so applying it turned the row ``'weak'``. It is printed 4.6e-10 now, and applied
    (with the 24 cells and 160 periods the same message asks for) it is accepted."""
    g = grazing.GRAZING_EXPONENT
    assert grazing.cos_min(4.7e-10, g=g) > 0.35 > grazing.cos_min(4.6e-10, g=g)
    with pytest.raises(ValueError, match='Grazing order refused') as e:
        fd.plan(_column(0.35, 24, 1e-8))
    quoted = re.search(r'reflection<=([0-9.e+-]+)\)', str(e.value)).group(1)
    assert quoted == '4.6e-10'
    assert {r['verdict'] for r in _survey(_column(0.35, 24, float(quoted)))} == {'ok'}
    for c in (0.26, 0.3, 0.35, 0.38):                   # the review's other two, and more
        q = grazing._quoted_target(c, g)
        assert grazing.cos_min(q, g=g) <= c and q > 0.9 * grazing.reflection_for(c, g=g)


def test_the_row_quoted_is_the_order_furthest_past_what_it_is_allowed():
    """``worst`` ranks by the verdict, then ``excess`` (``|r| / max(tau, |r_pure|)``), then
    ``cos / cos_min``: of two ``'lossy'`` rows the one further past its bound is quoted even at
    the larger ``cos``."""
    a = dict(verdict='lossy', excess=2.0, cos=0.6, cos_min=0.41)
    b = dict(verdict='lossy', excess=1.5, cos=0.5, cos_min=0.41)
    c = dict(verdict='weak', cos=0.3, cos_min=0.41)
    assert grazing.worst([b, a]) is a and grazing.worst([a, b, c]) is c
    assert grazing.worst([dict(verdict='ok', cos=0.5, cos_min=0.41),
                          dict(verdict='ok', cos=0.45, cos_min=0.41)])['cos'] == 0.45


def test_a_diffracted_order_on_a_stabilized_face_is_judged_by_its_return():
    """A lamellar grating of period 1.4 at normal incidence: its (-1, 0) order leaves at cos
    0.619 at wavelength 1.1 -- allowed on the pure stretch (which returns 2.7e-5 there), refused
    on a stabilized face, which returns 1.6e-3 (the worse of TE and TM: a diffracted order
    carries either). The row is ``layer_return`` with the Bloch axis's grid ``K``, the face's
    design wavelength 1.0 and the row's 1.1."""
    from fd_helpers import box_grating
    scene, _ = box_grating(theta=0.0, period=1.4, periods=40)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fd.plan(scene)
    z = fd.PML(cells=12, stabilize=RHO)
    stabilized = scene.replace(boundaries=fd.Boundaries(x=scene.boundaries.x,
                                                        y=scene.boundaries.y, z=z))
    with pytest.raises(ValueError, match='Stabilized face refused: order') as e:
        fd.plan(stabilized)
    plan = _plan_unenforced(stabilized)
    rows = [r for r in grazing.survey(stabilized, plan) if r['what'] == 'orders']
    assert {r['verdict'] for r in rows} == {'lossy'}
    row = rows[0]
    assert row['order'] == (-1, 0) and row['wavelength'] == 1.1
    assert row['polarisations'] == ('TE', 'TM')
    K = grid_k(-2 * math.pi / 1.4, plan.dx)              # Bloch x at k = 0: the -1 order
    assert row['K'] == pytest.approx(abs(K), rel=1e-15)
    r = layer_return(12, plan.dx, plan.dt, 1.0, 1.0, row['face'][1], 1e-8, RHO, 1.1, K)
    r0 = layer_return(12, plan.dx, plan.dt, 1.0, 1.0, row['face'][1], 1e-8, 0.0, 1.1, K)
    excess = [abs(x) / max(TAU, abs(y)) for x, y in zip(r, r0)]
    p = excess.index(max(excess))
    assert row['polarisation'] == ('TE', 'TM')[p]
    assert row['excess'] == pytest.approx(max(excess), rel=MODEL)
    assert row['return'] == pytest.approx(abs(r[p]), rel=MODEL)
    assert row['return_pure'] == pytest.approx(abs(r0[p]), rel=MODEL)
    assert row['return_loss'] == pytest.approx(abs(r[p] - r0[p]), rel=MODEL)
    assert row['return'] > TAU > row['return_pure']
    assert f"it sends back {row['return']:.2e}" in str(e.value)


def test_every_propagating_order_on_a_stabilized_face_is_judged(monkeypatch):
    """The return is not monotone in ``cos`` everywhere (grazing.py's docstring), so a
    stabilized face judges every propagating order, not only the lowest ``cos``. Wiring check
    on the lamellar grating's pure stretch made stabilized: a face that returns ``2 tau`` on
    its specular order (0, 0) alone -- the highest ``cos`` there is -- is refused, and the row
    names that order."""
    from fd_helpers import box_grating
    scene, _ = box_grating(theta=0.0, period=1.4, periods=40)
    z = fd.PML(cells=12, stabilize=1e-9)
    scene = scene.replace(boundaries=fd.Boundaries(x=scene.boundaries.x, y=scene.boundaries.y,
                                                   z=z))
    saved = grazing.face_return

    def specular_only(scene, plan, name, face, n, wavelength, design, K):
        te, tm = saved(scene, plan, name, face, n, wavelength, design, K)
        if face.stabilize:                                  # the lossy call, not its pure twin
            K = np.asarray(K)
            te, tm = (np.where(K == 0.0, 2 * TAU, x) for x in (te, tm))
        return te, tm
    monkeypatch.setattr(grazing, 'face_return', specular_only)
    rows = [r for r in _survey(scene) if r['what'] == 'orders']
    assert {(r['verdict'], r['order'], r['cos']) for r in rows} == {('lossy', (0, 0), 1.0)}


def test_every_order_of_a_face_is_one_march(monkeypatch):
    """A stabilized face's orders at one wavelength are judged in one call of the model (an
    array ``K``) with the loss and one without it, and a ``'weak'`` or ``'floor'`` row, refused
    already, is not modelled. On a crossed grating of period 8 at 9 wavelengths (1903
    propagating orders in vacuum) the survey took 645 ms order by order and 40 ms marched per
    face and wavelength (best of 3, NumPy, Windows; 1.5 ms on the pure stretch)."""
    from fd_helpers import box_grating
    scene, _ = box_grating(kind='crossed', theta=10.0, phi=20.0, period=4.0,
                           wavelengths=(0.9, 1.0, 1.1), periods=160)
    scene = scene.replace(boundaries=fd.Boundaries(x=scene.boundaries.x, y=scene.boundaries.y,
                                                   z=fd.PML(cells=12, stabilize=RHO)))
    plan = _plan_unenforced(scene)
    calls = []
    saved = grazing.face_return

    def counted(scene, plan, name, face, n, wavelength, design, K):
        calls.append((name, wavelength, np.size(K)))
        return saved(scene, plan, name, face, n, wavelength, design, K)
    monkeypatch.setattr(grazing, 'face_return', counted)
    rows = grazing._survey(scene, plan)
    orders = [c for c in calls if c[2] > 1]
    assert len({(name, w) for name, w, _ in orders}) == len(orders) // 2 == 6
    assert {r['verdict'] for r in rows} >= {'floor'}
    floors = [r for r in rows if r['verdict'] == 'floor']
    assert floors and not any('return' in r for r in floors)


def test_a_grid_evanescent_order_is_never_modelled():
    """``layer_return`` models a propagating order only and refuses one the grid makes
    evanescent. The rule never asks for one: at cos 0.02 on an Axis1D column at 20 cells per
    wavelength (the grid's own cutoff is cos 0.026) the row is ``'floor'`` and the model is not
    run. It cannot reach a row the rule would accept either: on an Axis1D axis the grid cuts an
    order off at ``cos^2 = 1 - (W / omega)^2``, 0.130 at the coarsest grid the plan takes (8
    cells per wavelength, ``courant`` just below 1), below the floor's 0.188 at any wavelength
    (``g -> 1``); on a Bloch axis ``K = 2 sin(k dx/2)/dx`` falls faster than ``W`` does, so an
    order propagating in the continuum propagates on the grid."""
    rows = _survey(_stabilized(0.02, 12))
    assert {r['verdict'] for r in rows} == {'floor'} and not any('return' in r for r in rows)
    plan = _plan_unenforced(_stabilized(0.7, 12))
    w = 2 * math.pi
    W = 2 * math.sin(w * plan.dt / 2) / plan.dt
    assert 0.02 < math.sqrt(1 - (W / w) ** 2) < 0.03
    with pytest.raises(ValueError, match='evanescent on this grid'):
        layer_return(12, plan.dx, plan.dt, 1.0, 1.0, '-', 1e-8, RHO, 1.0,
                     w * math.sqrt(1 - 0.02 ** 2))
    dx = 1 / 8
    dt = 0.999 * dx / math.sqrt(3)
    W = 2 * math.sin(w * dt / 2) / dt
    cutoff = math.sqrt(1 - (W / w) ** 2)
    assert cutoff < 0.131 < grazing.cos_min(grazing.FLOOR_REFLECTION, g=1.0) < 0.188
    k = np.linspace(0, w * (1 - 1e-12), 2001)                # Bloch: every continuum order
    assert np.all(2 * np.sin(k * dx / 2) / dx < W)


def test_the_offered_stabilize_is_judged_as_printed(monkeypatch):
    """The bisection's value is rounded down to two digits and that printed value is judged
    again, and rounded down further if it fails -- a guard: the ``rho`` that pass form an
    interval from 0 (the return is ``|r_pure + rho d|`` to first order, convex in ``rho``), so
    no measured case needs it. Wiring check: with the rule made to refuse the printed 1.9e-3 at
    cos 0.6 on 12 cells, 1.8e-3 is offered."""
    saved = grazing._passes

    def refuse_printed(scene, plan, governed, face, periods, row, full=True):
        if face.stabilize == 1.9e-3:
            return False
        return saved(scene, plan, governed, face, periods, row, full)
    monkeypatch.setattr(grazing, '_passes', refuse_printed)
    scene = _stabilized(0.6, 12)
    fix = grazing.lossy_fixes(_survey(scene)[0], scene, _plan_unenforced(scene))
    assert fix['rho'] == 1.8e-3 and fix['cells'] == 16


# ---------------------------------------------------------------------------------------------
# A refusal quotes the worst row, and an offer passes the whole scene
# ---------------------------------------------------------------------------------------------

def _grating(cells, reflection=1e-8, rho=RHO, substrate=None, theta=0.0, period=1.4,
             wavelengths=(0.9, 1.0, 1.1), periods=40, z=None, **kwargs):
    """``box_grating`` with both z faces ``PML(cells, reflection, stabilize=rho)`` (or ``z``),
    the domain grown with the cells (an offer applied as a user would), and, with
    ``substrate``, a slab of that permittivity filling the z- layer."""
    from fd_helpers import box_grating
    scene, _ = box_grating(theta=theta, period=period, wavelengths=wavelengths, periods=periods,
                           pml=cells, **kwargs)
    if substrate is not None:
        box = fd.Region(size=(20.0, 20.0, 20.0), center=(0.0, 0.0, -0.3 - 10.0))
        scene = scene.replace(structures=scene.structures
                              + (fd.Structure(box, fd.Material(eps=substrate)),))
    if z is None:
        z = fd.PML(cells=cells, reflection=reflection, stabilize=rho)
    return scene.replace(boundaries=fd.Boundaries(x=scene.boundaries.x, y=scene.boundaries.y,
                                                  z=z))


def _refusal(scene):
    """The plan's refusal of ``scene``, or ``None``."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fd.plan(scene)
        return None
    except ValueError as e:
        return str(e)


#: The review's substrate grating (``theta 12``, period 1.4, wavelengths 0.87 / 1 / 1.15, an
#: ``eps 2.25`` slab filling the z- layer, 160 periods). On ``PML(12, 1e-16, 3e-3)`` its worst
#: row is order (-2, 0) through z- in the slab at 1.15 (cos 0.352, a ``'needs'`` order too).
SUBSTRATE = dict(theta=12.0, period=1.4, wavelengths=(0.87, 1.0, 1.15), periods=160,
                 substrate=2.25, reflection=1e-16)


def test_an_offer_passes_every_row_its_face_governs_not_only_the_quoted_one():
    """The review applied every offer of 128 random-grating refusals, each judged on the quoted
    row alone, and 22 of 247 failed -- every one a substrate grating whose other face (or other
    medium) returned another order too much. An offer is now judged by the survey itself with
    it in place, on every order, medium and wavelength of the faces it replaces. On the
    substrate grating that is 38 cells or ``stabilize<=9.4e-4`` at 24; judged on the quoted row
    and wavelength alone it was 35 cells and 1.0e-3, and both, applied, are refused on the other
    face (order (1, 0) through z+ in vacuum at wavelength 1, cos 0.387). 37 cells and 9.5e-4 are
    refused too: each offer is the extreme that passes."""
    scene = _grating(12, **SUBSTRATE)
    message = _refusal(scene)
    assert message.startswith("Stabilized face refused: order (-2, 0) leaving through 'z-' in "
                              "structures[1] at wavelength 1.15 has exit cos theta = 0.352")
    assert _offers(message) == [(38, 1e-16, 3e-3, None), (24, 1e-16, 9.4e-4, None)]
    assert "What passes on both faces of the axis" in message
    for cells, rho in ((38, 3e-3), (24, 9.4e-4)):
        assert _refusal(_grating(cells, rho=rho, **SUBSTRATE)) is None, (cells, rho)
    for cells, rho in ((37, 3e-3), (24, 9.5e-4), (35, 3e-3), (24, 1e-3)):
        assert _refusal(_grating(cells, rho=rho, **SUBSTRATE)).startswith(
            "Stabilized face refused: order (1, 0) leaving through 'z+' in the background at "
            "wavelength 1 has exit cos theta = 0.387"), (cells, rho)
    # What the quoted row alone passes: the survey's own code narrowed to that face and
    # wavelength (``full=False``, the bisection's cheap first pass).
    plan = _plan_unenforced(scene)
    top = grazing.worst([r for r in grazing.survey(scene, plan) if r['verdict'] != 'ok'])
    face = scene.boundaries.face('z-')

    def alone(f):
        return grazing._passes(scene, plan, ('z-', 'z+'), f, 160.0, top, full=False)
    assert next(m for m in range(13, 60) if alone(face.replace(cells=m))) == 35
    assert alone(face.replace(cells=24, stabilize=1e-3))


def test_the_refusal_quotes_the_worst_row_and_names_the_others():
    """``enforce`` raised the first refused row in survey order (z- before z+); it raises the
    worst (``grazing.worst``) and names the rest. On the review's substrate grating (``theta
    7``, period 1.2) the z- slab's order (-2, 0) is refused, and so is the z+ vacuum order (1,
    0) at cos 0.296: on ``PML(20, 1e-16, 3e-3)`` the latter is further past its bound and
    quoted, its offers pass the scene, and the z- row is named; on ``PML(12, 1e-8, 1e-2)`` at
    wavelengths 0.87 / 1 / 1.15 the vacuum order is ``'weak'`` -- only the target or the
    geometry moves it -- and it is quoted before the ``'lossy'`` z- order (-2, 0) at 0.87,
    which the old message quoted with two offers the review found both refused on it."""
    case = dict(theta=7.0, period=1.2, wavelengths=(0.9, 1.0, 1.1), periods=160,
                substrate=2.25)
    scene = _grating(20, reflection=1e-16, **case)
    rows = _survey(scene)
    assert [(r['face'], r['verdict']) for r in rows if r['verdict'] != 'ok'] == [
        ('z-', 'lossy'), ('z+', 'lossy')]
    message = _refusal(scene)
    assert message.startswith("Stabilized face refused: order (1, 0) leaving through 'z+' in "
                              "the background at wavelength 1 has exit cos theta = 0.296")
    assert ("Refused too (the row quoted above is the worst): order (-2, 0) through 'z-' in "
            "structures[1] at wavelength 0.9 (exit cos theta 0.375, 'lossy').") in message
    offers = _offers(message)
    assert offers
    for cells, R, rho, run in offers:
        assert _refusal(_grating(cells, reflection=R, rho=rho, **case)) is None
    message = _refusal(_grating(12, rho=1e-2, **dict(case, wavelengths=(0.87, 1.0, 1.15))))
    assert message.startswith("Grazing order refused: order (1, 0) leaving through 'z+' in the "
                              "background at wavelength 1 has exit cos theta = 0.296")
    assert ("Refused too (the row quoted above is the worst): order (-2, 0) through 'z-' in "
            "structures[1] at wavelength 0.87 (exit cos theta 0.444, 'lossy').") in message


def test_no_offer_when_another_face_is_refused_too():
    """An offer changes the refused face (both faces of an axis declared as one face; the named
    face of a ``(low, high)`` pair). When a face it does not change is refused as well, no
    offer of it can pass the scene, and the refusal says so instead of offering: z- stabilized
    at 3e-3 is quoted (``'lossy'``), and the pure z+ face at ``1e-12`` on 12 cells is
    ``'needs'`` on the vacuum order (1, 0) at cos 0.296. It says what the user must change
    instead -- both faces at once, the axis given one face -- and keeps the advice that always
    applies: what it said before ("that face needs a change of its own, which its own refusal
    names once this one is fixed") is circular, since this face is never offered a fix. The
    advice followed, the one-face scene is told what its worst row needs. With a z+ face that
    takes that order, the offers are the z- face's alone, spelled with its design wavelength,
    and pass."""
    case = dict(theta=7.0, period=1.2, wavelengths=(0.87, 1.0, 1.15), periods=160,
                substrate=2.25)
    z = (fd.PML(cells=12, stabilize=RHO), fd.PML(cells=12, reflection=1e-12))
    message = _refusal(_grating(12, z=z, **case))
    assert message.startswith("Stabilized face refused: order (-2, 0) leaving through 'z-'")
    assert ("No change to this face alone passes the scene: order (1, 0) through 'z+' in the "
            "background at wavelength 1 (exit cos theta 0.296, 'needs') is refused too, on the "
            "other face of the z axis, which the scene declares separately (z=(low, high)) and "
            "an offer for this face leaves as it is.") in message
    assert 'What passes' not in message and not OFFER.search(message)
    assert ('Both faces have to change together, and this rule offers a setting only for the '
            'faces one declaration governs: give the axis one face, z=PML(...), and the '
            'refusal names what passes on both at once; or set each face yourself until no '
            'row here is refused.') in message
    assert ('Or make the order less oblique (the period, the angle or the wavelengths).'
            in message)
    followed = _refusal(_grating(12, z=fd.PML(cells=12, stabilize=RHO), **case))
    assert ('PML(cells>=24, reflection<=9.2e-12)' in followed
            and 'Run(periods>=160)' in followed), followed
    z = (fd.PML(cells=12, stabilize=RHO, wavelength=0.95), fd.PML(cells=24, reflection=1e-12))
    message = _refusal(_grating(24, z=z, **case))
    assert "What passes on the 'z-' face" in message
    offers = _offers(message, full=True)
    assert len(offers) == 2 and all(o[:2] == ("the 'z-' face ", 'z-')
                                    and o[5] == ', wavelength=0.95' for o in offers), offers
    for _, _, cells, R, rho, _, run in offers:
        low = fd.PML(cells=cells, reflection=R, stabilize=rho, wavelength=0.95)
        assert _refusal(_grating(24, z=(low, z[1]), **case)) is None, (cells, rho)


def test_a_pair_refused_on_both_faces_is_told_what_to_change():
    """The review's deadlock scene: the same order is ``'lossy'`` on both faces of a pair
    (``theta 12``, period 1.2, ``PML(16, 1e-12, 1e-2)`` on z- and 12 cells on z+; z+ is quoted
    at excess 18.7 over z-'s 11.7). Neither face is offered a fix, so the refusal has to say
    what does move: the two faces together. Followed -- the axis given one face -- the scene
    is offered 37 cells or ``stabilize<=9.2e-4`` at its 16, and both pass when applied."""
    case = dict(theta=12.0, period=1.2, wavelengths=(0.87, 1.0, 1.15), periods=160)
    low = fd.PML(cells=16, reflection=1e-12, stabilize=1e-2)
    high = fd.PML(cells=12, reflection=1e-12, stabilize=1e-2)
    message = _refusal(_grating(16, z=(low, high), **case))
    assert message.startswith("Stabilized face refused: order (1, 0) leaving through 'z+'")
    assert ("No change to this face alone passes the scene: order (1, 0) through 'z-' in the "
            "background at wavelength 0.87 (exit cos theta 0.424, 'lossy') is refused too, on "
            "the other face of the z axis") in message
    assert 'give the axis one face, z=PML(...)' in message
    assert 'Or make the order less oblique' in message and 'What passes' not in message
    offers = _offers(_refusal(_grating(16, z=low, **case)))
    assert offers == [(37, 1e-12, 1e-2, None), (16, 1e-12, 9.2e-4, None)]
    for cells, R, rho, _ in offers:
        face = fd.PML(cells=cells, reflection=R, stabilize=rho)
        assert _refusal(_grating(cells, z=face, **case)) is None, (cells, rho)


def test_an_offered_layer_only_passes_with_the_domain_grown_and_says_so():
    """An offer is judged with the domain grown by its extra cells, which is what deepening a
    layer without moving the interior means -- and what every sweep that validated the offers
    did silently. Copied onto the domain the scene has, the layer eats inward: of 176 offers
    the review applied literally, 100 were refused, none of them by this rule. The message
    says so and spells the growth. On the substrate grating the two offers ask for 26 and 12
    more cells on each of the two faces (2.6 and 1.2 in total at dx 0.05): grown, both plan;
    literal, the 38-cell layer leaves no interior and the 24-cell one swallows the Diffraction
    plane, and each refusal names what it reached."""
    scene = _grating(12, **SUBSTRATE)
    message = _refusal(scene)
    assert ('A deeper layer is taken out of the domain you gave, not added to it: grow '
            'Scene(domain=) along z with the offer you take (26 or 12 more cells on each of '
            'the two faces, 2.6 or 1.2 in total), or the layer eats into the interior and what '
            'it then reaches -- a Diffraction plane, a structure, the domain left too thin for '
            'the layers themselves -- is refused next.') in message
    assert _offers(message) == [(38, 1e-16, 3e-3, None), (24, 1e-16, 9.4e-4, None)]
    grown = [_grating(cells, rho=rho, **SUBSTRATE) for cells, _, rho, _ in _offers(message)]
    assert [_refusal(s) for s in grown] == [None, None]
    assert grown[0].domain.size[2] - scene.domain.size[2] == pytest.approx(2.6)
    assert grown[1].domain.size[2] - scene.domain.size[2] == pytest.approx(1.2)
    literal = [scene.replace(boundaries=fd.Boundaries(
        x=scene.boundaries.x, y=scene.boundaries.y,
        z=fd.PML(cells=cells, reflection=R, stabilize=rho)))
        for cells, R, rho, _ in _offers(message)]
    assert _refusal(literal[0]).startswith(
        'the absorbing layers on z are 76 cells deep in total but the axis has only 72')
    assert _refusal(literal[1]).startswith(
        "Diffraction('r') reads indices 53..54 along z and reaches the PML on 'z+' (24 cells)")


def test_a_pair_of_equal_faces_is_one_face_to_the_offer():
    """``_governed`` compares the axis's two faces by value, not by identity: ``z=(a, b)`` with
    two equal-but-distinct ``PML`` objects declares the same face on both sides, so an offer
    replaces both and is spelled ``z=``. The review's surviving mutant (``low is high``) passed
    all 121 tests: under it this scene takes the blocked branch and offers nothing, since every
    gate for the pair used one object or two faces that genuinely differ."""
    a, b = fd.PML(cells=12, stabilize=RHO), fd.PML(cells=12, stabilize=RHO)
    assert a == b and a is not b
    message = _refusal(_grating(12, z=(a, b)))
    assert 'What passes on both faces of the axis, which the scene declares as one face' \
        in message
    assert _offers(message, full=True) == [('z=', '', 16, 1e-8, 3e-3, '', None),
                                           ('z=', '', 12, 1e-8, 1.8e-3, '', None)]
    for cells, R, rho, _ in _offers(message):
        pair = (fd.PML(cells=cells, reflection=R, stabilize=rho),
                fd.PML(cells=cells, reflection=R, stabilize=rho))
        assert _refusal(_grating(cells, z=pair)) is None, (cells, rho)


def test_an_offer_spells_every_argument_of_the_face():
    """The review: ``PML(cells=12, stabilize=3e-3, wavelength=0.9)`` was offered as
    ``PML(cells=16, reflection=1e-08, stabilize=0.003)``, which copied gives the face another
    design wavelength than the one judged. Every argument the face was given is spelled, and
    the offer, applied as spelled, passes."""
    offers = _offers(_refusal(_stabilized(0.6, 12, face=dict(wavelength=0.9))), full=True)
    assert offers == [('z=', '', 16, 1e-8, 3e-3, ', wavelength=0.9', None),
                      ('z=', '', 12, 1e-8, 1.9e-3, ', wavelength=0.9', None)]
    for _, _, cells, R, rho, _, _ in offers:
        assert _verdict_pairs(_stabilized(0.6, cells, R, rho, face=dict(wavelength=0.9))) == {
            ('ok', 'ok')}
    message = _refusal(_stabilized(0.6, 12, background=2.25, rho=1e-2,
                                   face=dict(background=1.0)))
    spelled = [o[5] for o in _offers(message, full=True)]
    assert spelled and all(s == ', background=Material(eps=(1.0, 1.0, 1.0), mu=None)'
                           for s in spelled), spelled


@pytest.mark.parametrize('cos, cells, offers', [
    (0.545, 12, [(13, 1e-8, 3e-3, None), (12, 1e-8, 2.9e-3, None)]),
    (0.4095, 24, [(25, 1e-8, 3e-3, None), (24, 1e-8, 2.8e-3, None)]),
], ids=['glass-12', 'glass-24'])
def test_the_offers_are_judged_in_the_orders_medium(cos, cells, offers):
    """The review's surviving mutant D judged the offers in vacuum instead of the order's
    medium (and crashed on a glass column: the order's ``K`` is past vacuum's cutoff). In a
    glass column (``eps 2.25``) the offers are the extremes that pass there: applied they are
    accepted, and one cell fewer or the next ``stabilize`` up is refused."""
    assert _offers(_refusal(_stabilized(cos, cells, background=2.25))) == offers
    (more, _, _, _), (_, _, most, _) = offers
    assert _verdict_pairs(_stabilized(cos, more, background=2.25)) == {('ok', 'ok')}
    assert _verdict_pairs(_stabilized(cos, more - 1, background=2.25)) == {('lossy', 'ok')}
    assert _verdict_pairs(_stabilized(cos, cells, rho=most, background=2.25)) == {('ok', 'ok')}
    assert _verdict_pairs(_stabilized(cos, cells, rho=round(most + 1e-4, 6),
                                      background=2.25)) == {('lossy', 'ok')}


def test_the_offers_are_judged_at_the_faces_design_wavelength():
    """The review's surviving mutant E judged the offers with the row's wavelength as the
    face's design wavelength. On the lamellar grating of period 1.4 the quoted row is at 1.1
    and the face is designed for 1.0: the offers are 16 cells and ``stabilize<=1.8e-3`` (1.9e-3
    under the mutant), and applied to the grating, 16 and 1.8e-3 are accepted while 15 and
    1.9e-3 are refused."""
    message = _refusal(_grating(12))
    assert 'at wavelength 1.1 has exit cos theta = 0.619' in message
    assert _offers(message) == [(16, 1e-8, 3e-3, None), (12, 1e-8, 1.8e-3, None)]
    assert _refusal(_grating(16)) is None and _refusal(_grating(12, rho=1.8e-3)) is None
    assert _refusal(_grating(15)).startswith('Stabilized face refused')
    assert _refusal(_grating(12, rho=1.9e-3)).startswith('Stabilized face refused')


def test_the_quoted_polarisation_is_the_one_furthest_past_its_bound():
    """``judge_returns`` quotes, per order, the polarisation furthest past ``max(tau,
    |r_pure|)``, not the larger return (the review's mutant B, verdict-equivalent over its
    model sweep, since the larger return is almost always the further one). Here TE returns
    more (1.2e-3) but its pure stretch returns 1.5e-3, so it is inside its bound; TM returns
    1.1e-3 on a pure 2e-4 and is refused -- by the larger return the order would pass."""
    r = np.array([[1.2e-3, 5e-4], [1.1e-3, 4e-4]])
    r0 = np.array([[1.5e-3, 1e-5], [2e-4, 1e-5]])
    got = grazing.judge_returns(r, r0, (0, 1))
    assert [g['polarisation'] for g in got] == ['TM', 'TE']
    assert got[0]['excess'] == pytest.approx(1.1) and grazing.refuses(got[0])
    assert got[1]['excess'] == pytest.approx(0.5) and not grazing.refuses(got[1])


# ---------------------------------------------------------------------------------------------
# A run surveys once, and reports its own scene
# ---------------------------------------------------------------------------------------------

def _substrate_column(eps):
    """The column at cos 0.7 on 12 stabilized cells, 20 periods, with a slab of ``eps``
    filling the z- layer and a Diffraction monitor: the z- row is judged in the slab."""
    scene = _stabilized(0.7, 12, periods=20)
    dz = scene.resolution.dx
    top = scene.domain.center[2] - scene.domain.size[2] / 2 + 14 * dz
    box = fd.Region(size=(10.0, 10.0, 20.0), center=(0.0, 0.0, top - 10.0))
    monitor = fd.Diffraction('r', z=scene.domain.center[2] + 0.2, side='reflect',
                             wavelengths=(1.0,))
    return scene.replace(structures=(fd.Structure(box, fd.Material(eps=eps)),),
                         monitors=scene.monitors + (monitor,))


def test_fd_run_surveys_once_and_reports_its_own_scene(monkeypatch):
    """``fd.plan`` surveys for its refusal and ``fd.run`` hands those rows to
    ``res.diagnostics['grazing']``: one survey per run, and no cache (the rows ride from
    ``plan(scene, _grazing=)`` to the result). They are always the scene's own: a slab of
    ``eps`` 2.25 or 2.4 in the z- layer gives two scenes with one skeleton (a material's
    numbers are not in it) and different rows, and ``fd.run(b, plan=fd.plan(a))`` reports
    ``b``'s -- a plan-carried survey, or a cache keyed by the skeleton, would report ``a``'s."""
    calls = []
    saved = grazing._survey

    def counted(scene, plan, *args, **kwargs):
        calls.append(1)
        return saved(scene, plan, *args, **kwargs)
    monkeypatch.setattr(grazing, '_survey', counted)
    a, b = _substrate_column(2.25), _substrate_column(2.4)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pa, pb = fd.plan(a), fd.plan(b)
        assert pa.skeleton == pb.skeleton and len(calls) == 2
        rows_a, rows_b = saved(a, pa), saved(b, pb)
        assert rows_a != rows_b and rows_b[0]['medium'] == 'structures[0]'
        res = fd.run(b)
        assert len(calls) == 3 and res.diagnostics['grazing'] == rows_b
        res = fd.run(b, plan=pa)
        assert len(calls) == 4 and res.diagnostics['grazing'] == rows_b


# ---------------------------------------------------------------------------------------------
# A Diffraction wavelength the grid cannot carry
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('kind', ['lamellar', 'crossed'])
@pytest.mark.parametrize('rho', [None, RHO], ids=['pure', 'stabilized'])
def test_a_diffraction_wavelength_the_grid_cannot_carry_is_refused(kind, rho):
    """The review: a Diffraction wavelength far below the band (``dx = 1/13``, an ``eps 1.2``
    tooth, 0.22) crashed the stabilized rule with a bare 'math domain error' (the incidence
    row, one order) or passed an order as NaN (an array): ``dx^2 (n^2 W^2 - K^2) > 4``, past the
    grid's Nyquist along z. Nothing held a monitor's wavelengths to the 8 cells per wavelength
    the plan holds the band to; the plan does now, before the rule reads them, on the pure and
    the stabilized face alike. At 20 cells per wavelength 0.4 is 7.3 cells per wavelength in the
    tooth (refused) and 0.45 is 8.2 (past this door)."""
    from fd_helpers import box_grating

    def grating(wavelength, cpw):
        scene, _ = box_grating(kind=kind, cells_per_wavelength=cpw, wavelengths=(wavelength,),
                               eps=1.2)
        return scene.replace(boundaries=fd.Boundaries(
            x=scene.boundaries.x, y=scene.boundaries.y, z=fd.PML(cells=12, stabilize=rho)))
    message = _refusal(grating(0.22, 13))
    assert message.startswith("Diffraction('r') asks for wavelength 0.22, where "
                              "Scene(background=) is sampled at 2.86 cells per wavelength")
    message = _refusal(grating(0.4, 20))
    assert message.startswith("Diffraction('r') asks for wavelength 0.4, where structures[0] "
                              "is sampled at 7.30 cells per wavelength (index 1.095")
    assert 'refine dx to at most 0.0456435' in message
    message = _refusal(grating(0.45, 20))
    assert message is None or not message.startswith('Diffraction(')


def test_the_sources_band_is_held_to_the_same_floor_at_the_same_door():
    """The door covers every wavelength the rule reads, not the monitors' alone: the incidence
    row is judged at the band centre, and a stabilized column with no ``Diffraction`` monitor
    at all still reached ``layer_return``'s own Nyquist refusal -- 14 of the 30 (cells per
    wavelength, cos) points the review swept, all at 3.0 cells per wavelength or below. The
    floor ``check_resolution`` holds the band to now runs at that door too, in its own words
    and before the rule, so the user reads the mesh refusal instead of the model's."""
    scene = _stabilized(0.98, 12, cells_per_wavelength=3.0)
    assert not [m for m in scene.monitors if isinstance(m, fd.Diffraction)], scene.monitors
    message = _refusal(scene)
    assert message.startswith(
        'Scene(background=) is sampled at 1.93 cells per wavelength (index 1.000, wavelength '
        "0.642039, the sources' band.shortest_wavelength, dx 0.333333)")
    assert 'Nyquist' not in message and 'Refine dx to at most 0.0802548' in message
    assert _refusal(_stabilized(0.98, 12, cells_per_wavelength=20)) is None


def test_what_the_pure_stretch_needs_is_read_off_every_order_not_the_quoted_row():
    """An offer carries the 24 cells and 160 periods the pure stretch needs when any order of
    the faces it replaces is ``'needs'`` -- read off the survey, not off the quoted row. No
    scene of a 432-grating sweep quotes a row whose pure verdict is ``'ok'`` while another
    order is ``'needs'`` (the grazing order is the worst there), so this is a wiring check: the
    quoted row of the 40-period ``'needs'`` column told it is ``'ok'`` changes no offer."""
    scene = _stabilized(0.35, 12, reflection=1e-12, periods=40)
    plan = _plan_unenforced(scene)
    row = _survey(scene)[0]
    assert row['pure_verdict'] == 'needs'
    expected = dict(periods=160, cells=43, rho=6.9e-4, rho_cells=24)
    assert grazing.lossy_fixes(row, scene, plan) == expected
    assert grazing.lossy_fixes(dict(row, pure_verdict='ok'), scene, plan) == expected
