"""Stage 7 against Meep and FDTDX, from a checked-in golden.

``golden_stage7_cross_solver.json`` holds what Meep 1.35.0 and FDTDX 0.6.3 measured on the
stage-7 cases (slab mode index, one-way mode launch, butt joint, a core run into the absorbing
layer, and the Gaussian beam), beside the Wavebit numbers the validation records print. Its
record, ``golden_stage7_cross_solver.reference.json``, states how each tool was set up, what
could and could not be matched, and that each ran in its own process from inputs written in
physical units (no Wavebit object reached either tool).

These values are **evidence, not acceptance**: no stage-7 verdict depends on them. This module
holds two things. First, the comparison tables in ``docs/validation/fdtd/fd-mode-source.md``
and ``fd-gaussian-beam.md`` (English and Korean) print exactly these values, so a document
cannot drift from the numbers it quotes. Second, each sentence those sections draw from the
comparison is asserted against the values, so a later regeneration that changes a conclusion
fails here instead of leaving a stale sentence.

Meep is GPL-2.0-or-later and was run as a separate program; its source was not read. This
module needs only the standard library.
"""

import json
import re

from tools.repository_paths import repository_path

GOLDEN = 'tests/references/golden_stage7_cross_solver.json'
EXACT_TE0 = 1.9562181937830228          # fd-waveguide-modes.md, M3
T_MM, R_MM = 0.955057091104045, 3.328675845426776e-4   # continuum mode matching (fd-mode-source.md S5)
FOCAL_EXACT = 1.3739765060772822        # exact angular spectrum (fd-gaussian-beam.md G3)
CENTRE = 1                              # index of 1.55 um (guide) and 1.0 um (beam)


def _golden():
    return json.loads(repository_path(GOLDEN).read_text(encoding='utf-8'))


def sci(value, signed=False):
    """Two significant figures with a short exponent: 1.3e-2, +1.3e-2, 6.8e-29."""
    text = f'{value:+.1e}' if signed else f'{value:.1e}'
    mantissa, exponent = text.split('e')
    return f'{mantissa}e{int(exponent)}'


def _cells(*values):
    return ' / '.join(values)


def guide_rows(g):
    """The rows of the mode-source comparison table: (label, Wavebit, Meep, FDTDX)."""
    w, m, f = (lambda key: g['wavebit__' + key]), (lambda key: g['meep__' + key]), (lambda key: g['fdtdx__' + key])
    three, two = ('0.05', '0.025', '0.0125'), ('0.05', '0.025')
    leak = lambda d: _cells(*(sci(d[k][CENTRE]) for k in two))                                   # noqa: E731
    side = lambda d, k: _cells(sci(d[k][0]), sci(d[k][2]))                                         # noqa: E731
    return [
        ('TE0 index the mode solver reports, minus exact (0.05 / 0.025 / 0.0125)',
         *(_cells(*(sci(t(k), signed=True) for k in three)) for t in (
             lambda k: w('neff_te_reported_error')[k], lambda k: m('neff_te_reported_error')[k],
             lambda k: f('neff_te_reported_error')[k]))),
        ('TE0 index the run propagates, minus exact (0.05 / 0.025)',
         *(_cells(*(sci(d[k] - EXACT_TE0, signed=True) for k in two)) for d in (
             w('n_run'), m('n_run'), f('n_run')))),
        ('Run index minus the Wavebit run index (0.05 / 0.025)', '0 / 0',
         *(_cells(*(sci(d[k] - w('n_run')[k], signed=True) for k in two)) for d in (m('n_run'), f('n_run')))),
        ('Backward / forward power at 1.55 um (0.05 / 0.025)',
         leak(w('leakage_power')), leak(m('leakage_power')), leak(f('leakage_power'))),
        ('Backward / forward power at 1.45 / 1.65 um, 0.025',
         side(w('leakage_power'), '0.025'), side(m('leakage_power'), '0.025'), side(f('leakage_power'), '0.025')),
        ('Butt joint, abs(T - T_mm) (0.05 / 0.025 / 0.0125)',
         _cells(*(sci(abs(w('joint_T')[k] - T_MM)) for k in two), '-'),
         _cells(*(sci(abs(m('joint_T')[k] - T_MM)) for k in three)),
         _cells(*(sci(abs(f('joint_T')[k] - T_MM)) for k in two), '-')),
        ('Butt joint, abs(R - R_mm) / R_mm (0.05 / 0.025 / 0.0125)',
         _cells(*(sci(abs(w('joint_R')[k] - R_MM) / R_MM) for k in two), '-'),
         _cells(*(sci(abs(m('joint_R')[k] - R_MM) / R_MM) for k in three)),
         _cells(*(sci(abs(f('joint_R')[k] - R_MM) / R_MM) for k in two), '-')),
        ('Core into the absorber: reflected TE0, 0.6 / 1.2 um layer (0.05)',
         _cells(*(sci(w('pml_reflection_d1')['0.05'][um]) for um in ('0.6', '1.2'))),
         _cells(*(sci(m('pml_reflection')['0.05'][um]) for um in ('0.6', '1.2'))),
         _cells(*(sci(f('pml_reflection')['0.05'][um]) for um in ('0.6', '1.2')))),
        ('Same, 0.025', '-',
         _cells(*(sci(m('pml_reflection')['0.025'][um]) for um in ('0.6', '1.2'))), '-'),
    ]


def beam_rows(g):
    """The rows of the Gaussian-beam comparison table: (label, Wavebit, Meep, FDTDX)."""
    w, m, f = (lambda key: g['wavebit__' + key]), (lambda key: g['meep__' + key]), (lambda key: g['fdtdx__' + key])
    two = ('0.05', '0.025')
    pair = lambda d, fn=lambda v: v: _cells(*(sci(fn(d[k])) for k in two))                         # noqa: E731
    return [
        ('Beam width against the exact beam, w / w_exact - 1 (0.05 / 0.025)',
         _cells(*(sci(w('beam_waist_ratio_minus_1')[k], signed=True) for k in two)),
         _cells(*(sci(m('beam_waist_ratio_minus_1')[k], signed=True) for k in two)),
         _cells(*(sci(f('beam_radius_2um_ratio_minus_1')[k], signed=True) for k in two))),
        ('Largest Gouy-phase difference from the exact beam, rad (0.05 / 0.025)',
         pair(w('gouy_max_error_rad')), pair(m('gouy_max_error_rad')), pair(f('gouy_max_error_rad'))),
        ('Backward / forward power at 1.0 um (0.05 / 0.025)',
         pair(w('beam_back_per_forward'), lambda v: v[CENTRE]), pair(m('beam_back_per_forward'), lambda v: v[CENTRE]),
         pair(f('beam_back_per_forward'), lambda v: v[CENTRE])),
        ('Focal peak in glass, abs(z - 1.37398 um) (0.05 / 0.025)',
         pair(w('focal_peak_um'), lambda v: abs(v - FOCAL_EXACT)), pair(m('focal_peak_um'), lambda v: abs(v - FOCAL_EXACT)),
         'cannot be set up'),
    ]


def _table(path, heading):
    """The first markdown table under ``heading`` in ``path``, as rows of stripped cells."""
    text = repository_path(path).read_text(encoding='utf-8')
    start = text.index(heading)
    rows = []
    for line in text[start:].splitlines()[1:]:
        if line.startswith('#'):
            break
        if line.startswith('|') and not re.fullmatch(r'\|[\s|:-]+\|', line.strip()):
            rows.append(tuple(cell.strip() for cell in line.strip().strip('|').split('|')))
        elif rows and not line.startswith('|'):
            break
    return rows[1:]         # without the header


DOCS = {
    'guide': (guide_rows, 'docs/validation/fdtd/fd-mode-source.md', 'docs/locales/ko/pages/validation/fdtd/fd-mode-source.md'),
    'beam': (beam_rows, 'docs/validation/fdtd/fd-gaussian-beam.md', 'docs/locales/ko/pages/validation/fdtd/fd-gaussian-beam.md'),
}
HEADING = {'en': '## Comparison with Meep and FDTDX', 'ko': '## Meep·FDTDX와의 비교'}


def test_the_english_tables_print_the_golden():
    g = _golden()
    for rows, en, _ in DOCS.values():
        assert _table(en, HEADING['en']) == rows(g), en


def test_the_korean_tables_print_the_same_values():
    g = _golden()
    for rows, _, ko in DOCS.values():
        printed = [row[1:] for row in _table(ko, HEADING['ko'])]
        expected = [row[1:] for row in rows(g)]
        assert printed == [tuple('설정할 수 없음' if c == 'cannot be set up' else c for c in row)
                           for row in expected], ko


def test_the_three_runs_propagate_the_same_discrete_system():
    """'FDTDX's run index equals Wavebit's to 1e-8; Meep's differs by its subpixel averaging.'"""
    g = _golden()
    for k in ('0.05', '0.025'):
        assert abs(g['fdtdx__n_run'][k] - g['wavebit__n_run'][k]) < 1e-8
        assert abs(g['meep__n_run'][k] - g['wavebit__n_run'][k]) > 1e-4


def test_only_the_grid_mode_launch_is_one_way_at_the_centre():
    """'At the centre wavelength Meep's and FDTDX's mode sources leak; FDTDX's leak falls
    near the sixth power of the spacing, Meep's near the fourth.'"""
    g = _golden()
    for k in ('0.05', '0.025'):
        w = g['wavebit__leakage_power'][k][CENTRE]
        assert w < 1e-20 < g['fdtdx__leakage_power'][k][CENTRE] < g['meep__leakage_power'][k][CENTRE]
    fdtdx = g['fdtdx__leakage_power']['0.05'][CENTRE] / g['fdtdx__leakage_power']['0.025'][CENTRE]
    meep = g['meep__leakage_power']['0.05'][CENTRE] / g['meep__leakage_power']['0.025'][CENTRE]
    assert 2 ** 5.5 < fdtdx < 2 ** 6.5, fdtdx
    assert 2 ** 3.5 < meep < 2 ** 4.5, meep


def test_every_tool_converges_on_the_mode_matching_transmission():
    g = _golden()
    for tool, spacings in (('wavebit', ('0.05', '0.025')), ('meep', ('0.05', '0.025', '0.0125')),
                           ('fdtdx', ('0.05', '0.025'))):
        errors = [abs(g[f'{tool}__joint_T'][k] - T_MM) for k in spacings]
        assert errors == sorted(errors, reverse=True), (tool, errors)


def test_the_small_reflection_is_closest_in_wavebit():
    g = _golden()
    rel = {tool: abs(g[f'{tool}__joint_R']['0.025'] - R_MM) / R_MM for tool in ('wavebit', 'meep', 'fdtdx')}
    assert rel['wavebit'] < min(rel['meep'], rel['fdtdx']), rel


def test_the_external_absorber_numbers_are_a_measurement_floor():
    """'Doubling the layer changes Meep's and FDTDX's values by at most 1.5 %, so they measure
    the readout, not the layer' (a layer's own reflection would fall by orders of magnitude,
    as Wavebit's D1 does)."""
    g = _golden()
    for values in (g['meep__pml_reflection']['0.05'], g['meep__pml_reflection']['0.025'],
                   g['fdtdx__pml_reflection']['0.05']):
        assert abs(values['1.2'] / values['0.6'] - 1) < 0.015, values
    assert g['wavebit__pml_reflection_d1']['0.05']['1.2'] < 1e-6 * g['wavebit__pml_reflection_d1']['0.05']['0.6']


def test_the_meep_beam_is_wider_at_every_resolution():
    """'Meep's waist is 0.86 % wide at both resolutions: its definition, not its grid.'"""
    g = _golden()
    ratios = g['meep__beam_waist_ratio_minus_1']
    assert all(8e-3 < v < 9e-3 for v in ratios.values()), ratios
    assert abs(ratios['0.05'] - ratios['0.025']) < 1e-4


def test_the_wavebit_gouy_phase_is_closest_and_converges():
    """The support decision in fd-gaussian-beam.md: the G2 difference is below both tools'
    at both spacings and falls at second order; Meep's does not fall."""
    g = _golden()
    for k in ('0.05', '0.025'):
        assert g['wavebit__gouy_max_error_rad'][k] < min(g['meep__gouy_max_error_rad'][k],
                                                         g['fdtdx__gouy_max_error_rad'][k])
    w = g['wavebit__gouy_max_error_rad']
    assert 2 ** 1.8 < w['0.05'] / w['0.025'] < 2 ** 2.2
    assert g['meep__gouy_max_error_rad']['0.025'] >= g['meep__gouy_max_error_rad']['0.05']


def test_the_fdtdx_back_radiation_does_not_fall_with_the_grid():
    g = _golden()
    back = [g['fdtdx__beam_back_per_forward'][k][CENTRE] for k in ('0.05', '0.025')]
    assert abs(back[1] / back[0] - 1) < 0.1, back
    assert all(b > 1e3 * g['wavebit__beam_back_per_forward']['0.05'][CENTRE] for b in back)
