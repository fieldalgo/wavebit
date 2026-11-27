"""The touching conductive films against Meep and FDTDX, from a checked-in golden.

``golden_c8_contact_cross_solver.json`` holds the complex reflection and transmission
amplitudes that Meep 1.34.0 and FDTDX 0.6.3 measured on the C8 contact case (two touching
conductive films on a fixed transverse wavenumber, p polarization, 80 cells per wavelength,
df .1) after 60 and 120 periods, beside Wavebit's values from the same fixture: with the
current PlaneWave launch, and with the historical hard-source launch of the original
campaign. Its record, ``golden_c8_contact_cross_solver.reference.json``, states how each tool
was set up, what could and could not be matched, and that each external tool ran in its own
process from inputs written as plain numbers (no Wavebit object reached either tool).

These values are **evidence, not acceptance**: no verdict of the contact route depends on
them. This module holds two things. First, the comparison table in
``docs/validation/fdtd/fdtd-conductive-contact.md`` (English and Korean) prints exactly these
values, so the document cannot drift from the numbers it quotes. Second, each sentence that
section draws from the comparison is asserted against the values, so a later regeneration
that changes a conclusion fails here instead of leaving a stale sentence.

Meep is GPL-2.0-or-later and was run as a separate program; its source was not read. This
module needs only the standard library.
"""

import json
import re

from tools.repository_paths import repository_path

GOLDEN = 'tests/references/golden_c8_contact_cross_solver.json'
# Fixed-kx characteristic-matrix reference of tests/fixtures/fdtd/conductive_contact.py at
# wavelengths .9 / 1 / 1.1: r_p at the top face, t at the bottom face.
REF_R = (0.2918613307049163 + 0.08977845231925022j, 0.2980203306619793 + 0.10288540120957634j,
         0.29889674451955256 + 0.10045016561067985j)
REF_T = (-0.6036151696967945 + 0.14589309609773432j, -0.52385186349795 + 0.3290963871212535j,
         -0.4198988760296201 + 0.45172104448982087j)
REFERENCE = {'r': REF_R, 't': REF_T}
EXTERNAL = ('meep', 'meep_unsmoothed', 'fdtdx')


def _golden():
    return json.loads(repository_path(GOLDEN).read_text(encoding='utf-8'))


def sci(value):
    """Two significant figures with a short exponent: 2.5e-4."""
    mantissa, exponent = f'{value:.1e}'.split('e')
    return f'{mantissa}e{int(exponent)}'


def _cells(*values):
    return ' / '.join(values)


def _amplitudes(g, tool, quantity, periods='120'):
    return [complex(re_, im) for re_, im in g[f'{tool}__{quantity}'][periods]]


def errors(g, tool, quantity):
    """abs(q - reference) after 120 periods at the three wavelengths."""
    return [abs(q - ref) for q, ref in zip(_amplitudes(g, tool, quantity), REFERENCE[quantity])]


def change(g, tool, quantity):
    """Largest abs(q after 120 periods - q after 60 periods) over the three wavelengths."""
    return max(abs(a - b) for a, b in zip(_amplitudes(g, tool, quantity), _amplitudes(g, tool, quantity, '60')))


def rows(g):
    """The rows of the comparison table: (label, Wavebit, Meep on, Meep off, FDTDX)."""
    tools = ('wavebit', *EXTERNAL)
    three = lambda tool, q: _cells(*(sci(e) for e in errors(g, tool, q)))               # noqa: E731
    late = lambda tool: _cells(sci(change(g, tool, 'r')), sci(change(g, tool, 't')))    # noqa: E731
    return [
        ('abs(r - reference) after 120 periods (.9 / 1 / 1.1)', *(three(tool, 'r') for tool in tools)),
        ('abs(t - reference) after 120 periods (.9 / 1 / 1.1)', *(three(tool, 't') for tool in tools)),
        ('Largest change from 60 to 120 periods, r / t', *(late(tool) for tool in tools)),
        ('Same, with the historical hard-source launch', late('wavebit_hard'), '-', '-', '-'),
        ('abs(r - reference), Meep grid moved half a cell (.9 / 1 / 1.1)', '-', three('meep_shifted', 'r'), '-', '-'),
        ('abs(t - reference), Meep grid moved half a cell (.9 / 1 / 1.1)', '-', three('meep_shifted', 't'), '-', '-'),
    ]


def _table(path, heading):
    """The first markdown table under ``heading`` in ``path``, as rows of stripped cells."""
    text = repository_path(path).read_text(encoding='utf-8')
    start = text.index(heading)
    found = []
    for line in text[start:].splitlines()[1:]:
        if line.startswith('#'):
            break
        if line.startswith('|') and not re.fullmatch(r'\|[\s|:-]+\|', line.strip()):
            found.append(tuple(cell.strip() for cell in line.strip().strip('|').split('|')))
        elif found and not line.startswith('|'):
            break
    return found[1:]         # without the header


EN = 'docs/validation/fdtd/fdtd-conductive-contact.md'
KO = 'docs/locales/ko/pages/validation/fdtd/fdtd-conductive-contact.md'
HEADING = {'en': '## Comparison with Meep and FDTDX', 'ko': '## Meep·FDTDX와의 비교'}


def test_the_english_table_prints_the_golden():
    assert _table(EN, HEADING['en']) == rows(_golden())


def test_the_korean_table_prints_the_same_values():
    printed = [row[1:] for row in _table(KO, HEADING['ko'])]
    assert printed == [row[1:] for row in rows(_golden())]


def test_the_golden_holds_three_wavelengths_at_both_durations():
    g = _golden()
    for key, value in g.items():
        assert sorted(value) == ['120', '60'], key
        assert all(len(value[p]) == 3 and all(len(z) == 2 for z in value[p]) for p in value), key


def test_wavebit_reflection_error_is_the_smallest_at_every_wavelength():
    """'2.5 to 18 times below smoothed Meep, more than 50 times below the two staircases.'"""
    g = _golden()
    ours = errors(g, 'wavebit', 'r')
    smoothed = [m / w for m, w in zip(errors(g, 'meep', 'r'), ours)]
    assert 2.4 < min(smoothed) and max(smoothed) < 19
    for tool in ('meep_unsmoothed', 'fdtdx'):
        assert min(e / w for e, w in zip(errors(g, tool, 'r'), ours)) > 50, tool


def test_transmission_errors_are_of_one_size():
    """'Every tool's transmission error is within a factor of two of Wavebit's.'"""
    g = _golden()
    ours = errors(g, 'wavebit', 't')
    for tool in EXTERNAL:
        assert all(0.5 < e / w < 2 for e, w in zip(errors(g, tool, 't'), ours)), tool
    # Smoothed Meep is the smallest and FDTDX the largest at each wavelength.
    for i in range(3):
        assert errors(g, 'meep', 't')[i] < ours[i] < errors(g, 'fdtdx', 't')[i]


def test_the_current_launch_settles_below_the_external_tools():
    """'With the current launch the change from 60 to 120 periods is below each tool's.'"""
    g = _golden()
    for quantity in ('r', 't'):
        assert all(change(g, 'wavebit', quantity) < change(g, tool, quantity) for tool in EXTERNAL), quantity
        assert change(g, 'wavebit', quantity) < 6e-8


def test_the_hard_source_launch_settled_slower_than_the_external_tools():
    """'With the historical hard source it was about 5 to 13 times theirs.'"""
    g = _golden()
    ratios = [change(g, 'wavebit_hard', q) / change(g, tool, q) for q in ('r', 't') for tool in EXTERNAL]
    assert 4.5 < min(ratios) and max(ratios) < 13


def test_the_launch_changed_the_late_time_change_and_not_the_accuracy():
    """'The two launches' errors agree within 2 %.'"""
    g = _golden()
    for quantity in ('r', 't'):
        for now, then in zip(errors(g, 'wavebit', quantity), errors(g, 'wavebit_hard', quantity)):
            assert abs(now - then) < 0.02 * then


def test_meep_depends_on_where_its_grid_sits():
    """'Half a cell raises Meep's reflection error 3.9 to 15 times, its transmission error 2.6 to 5.1 times.'"""
    g = _golden()
    for quantity, low, high in (('r', 3.8, 15), ('t', 2.6, 5.1)):
        ratios = [s / a for s, a in zip(errors(g, 'meep_shifted', quantity), errors(g, 'meep', quantity))]
        assert low <= min(ratios) and max(ratios) <= high, (quantity, ratios)
