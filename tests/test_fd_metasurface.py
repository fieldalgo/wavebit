"""C7: a periodic metasurface cell through the public path, its frame, its campaign and RCWA.

The campaign is ``studies/fdtd_metasurface_rcwa.py`` (``docs/fd-metasurface-validation.md``):
three pillar-on-substrate cells under oblique and conical plane waves, every propagating order
against ``wavebit.rcwa``. These tests hold what the comparison rests on and what it measured.
"""
import cmath
import functools
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pytest

import wavebit.fd as fd

ROWS = Path(__file__).parent / 'references' / 'fd_metasurface_rows.json'


def study():
    from studies import fdtd_metasurface_rcwa
    return fdtd_metasurface_rcwa


def quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*args, **kwargs)


def amplitudes(res):
    """``{(side, m, n, quantity): (nf,) array}`` over fd's whole order window."""
    out = {}
    for side, power in (('r', 'R'), ('t', 'T')):
        d = res[side].orders
        for n in d.orders_v:
            for m in d.orders_u:
                o = d.order(int(m), int(n))
                for q in (f'{side}_s', f'{side}_p', power):
                    out[(side, int(m), int(n), q)] = np.asarray(getattr(o, q))
    return out


def moved(scene, dx, dy):
    """``scene`` with its pillar (the second structure) moved by ``(dx, dy)``."""
    sub, pillar = scene.structures
    x, y, z = pillar.shape.center
    return scene.replace(structures=(sub, pillar.replace(
        shape=pillar.shape.replace(center=(x + dx, y + dy, z)))))


@functools.lru_cache(maxsize=None)
def translated_runs():
    """M1 at 20 cells per wavelength, 30 periods, s, on NumPy: the cell as built, and the same
    with its pillar moved by one cell along x and along y."""
    h = study()
    scene, info = h.build('M1', 20, 's', periods=30)
    dx = info['dx']
    return dx, {shift: amplitudes(quiet(fd.run, moved(scene, *shift), 'numpy'))
                for shift in ((0.0, 0.0), (dx, 0.0), (0.0, dx))}


@pytest.mark.parametrize('axis', ('x', 'y'))
def test_a_pillar_moved_by_a_cell_moves_each_order_by_its_lateral_phase(axis):
    """Moving the pillar by a whole cell is an exact symmetry of the grid: every order ``(m, n)``
    picks up ``exp(-2 pi i (m dx/Px + n dy/Py))`` and every efficiency stays. This is what lets the
    campaign compare fd's orders with RCWA's centred shape by a phase factor (the study's
    ``phase``), and it holds on M1, whose transmitted orders include ``n = +-1`` -- so the y half is
    tested too. Measured 2026-10-03: 1.8e-12 (x) and 8.8e-13 (y) against amplitudes up to 0.44;
    without the factor the gap is 0.16 (x) and 0.11 (y)."""
    dx, runs = translated_runs()
    Px, Py = study().SCENES['M1']['P']
    shift = (dx, 0.0) if axis == 'x' else (0.0, dx)
    base, other = runs[(0.0, 0.0)], runs[shift]
    phased, plain = [], []
    for key, a in base.items():
        side, m, n, q = key
        b = other[key]
        if q in ('R', 'T'):
            assert np.max(np.abs(b - a)) < 1e-12, key
            continue
        f = cmath.exp(-2j * math.pi * (m * shift[0] / Px + n * shift[1] / Py))
        phased.append(np.max(np.abs(b - a * f)))
        plain.append(np.max(np.abs(b - a)))
    scale = max(np.max(np.abs(a)) for a in base.values())
    assert max(phased) < 1e-10 * scale
    assert max(plain) > 0.05                    # the factor is not 1: the check can fail


def campaign_entry(name, n, pol):
    rows = json.loads(ROWS.read_text(encoding='utf-8'))
    return next(e for e in rows['runs'] if (e['scene'], e['n'], e['pol']) == (name, n, pol))


def repeat_campaign_run(name, n, pol):
    """Solve the campaign's scene ``name`` at ``n`` cells per wavelength on NumPy, through the
    public path, and hold it to the campaign's row and to its distance from RCWA."""
    h = study()
    entry = campaign_entry(name, n, pol)
    scene, _ = h.build(name, n, pol, periods=entry['periods'])
    got = amplitudes(quiet(fd.run, scene, 'numpy'))
    amp, eff = [], []
    for key, values in entry['values'].items():
        side, m, k, q = key.split('|')
        value = got[(side, int(m), int(k), q)]
        # the campaign's own numbers, every cell
        np.testing.assert_allclose(value, [complex(*v) for v in values], rtol=0, atol=1e-10,
                                   err_msg=key)
        ref = np.array([complex(*v) for v in entry['reference'][key]])
        if q in ('R', 'T'):
            eff.append(np.max(np.abs(value - ref)))
        else:
            amp.append(np.max(np.abs(value * h.phase(name, n, int(m), int(k)) - ref)))
    # every order within the campaign's distance from RCWA, the phase referred to the centred
    # pillar as the campaign refers it
    assert max(amp) <= entry['A'] * (1 + 1e-9) + 1e-12
    assert max(eff) <= entry['F'] * (1 + 1e-9) + 1e-12


@pytest.mark.parametrize('name,n,pol', (('M1', 20, 'p'), ('M2', 25, 's')))
def test_a_metasurface_cell_repeats_its_campaign_row_and_its_distance_from_rcwa(name, n, pol):
    """M1 (an epsilon-4 cylinder, 2 reflected and 6 transmitted orders, cross-polarised
    ``(m, +-1)`` orders) and M2 (an epsilon-5.8 cylinder splitting its light between two
    transmitted orders), at their lowest campaign rungs."""
    repeat_campaign_run(name, n, pol)


@pytest.mark.slow
def test_the_conical_silicon_box_repeats_its_campaign_row_and_its_distance_from_rcwa():
    """M3: an epsilon-12 box under conical incidence, the scene with no mirror symmetry."""
    repeat_campaign_run('M3', 40, 's')
