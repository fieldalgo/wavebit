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
"""
import math
import os
import re
import warnings

import numpy as np
import pytest

import wavebit.fd as fd
from wavebit.fd._lower import cpml, grazing
from fd_helpers import to_numpy

TAU = grazing.GRAZING_TOLERANCE
BAND = dict(wavelength=1.0, df=0.15)

LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'studies', 'results', 'fdtd_grazing_return.log')


def _column(cos, cells, reflection, periods=160, cells_per_wavelength=20):
    """A PlaneWave at exit ``cos`` in a one-cell column (x and y Axis1D), both z faces
    ``PML(cells, reflection)``, two Ey probes 5 and 6 cells in front of the z- layer."""
    dx = 1.0 / cells_per_wavelength
    above, distance = 4, cells_per_wavelength
    nz = cells + above + distance + cells
    z_sheet = -(cells + above) * dx
    z_face = z_sheet - distance * dx
    pw = fd.PlaneWave('pw', z=z_sheet, direction='-z', theta=math.degrees(math.acos(cos)),
                      polarization='s', band=fd.Band(**BAND))
    probes = tuple(fd.Record(f'p{i}', at=(0.0, 0.0, z_face + (5 + i) * dx), component='Ey')
                   for i in (0, 1))
    return fd.Scene(domain=fd.Domain(size=(dx, dx, nz * dx), center=(0.0, 0.0, -nz * dx / 2)),
                    boundaries=fd.Boundaries(**fd.Bloch.from_source(pw, axis1d=('x', 'y')),
                                             z=fd.PML(cells=cells, reflection=reflection)),
                    resolution=fd.Resolution(dx=dx), run=fd.Run(periods=periods),
                    sources=(pw,), monitors=probes)


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
