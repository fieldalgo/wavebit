"""`check()` on the carriers: every violation at once, from the rules the solver uses.

`Stack`, `Incidence` and `Numerics` name every condition a solve takes, and until now they
named them without checking them: a complex `psi`, `psi` beside `polarization`, a raster where
a half-space belongs, a three-length period and `complex_dtype=32` all constructed without
complaint and were refused later, one at a time, by the solver's setters.

Two things make this a layer rather than a second set of rules.

**The rules are called, not restated.** `reject_complex_angle` and `incidence_clashes` live in
`wavebit._common.conventions`, below `domain` and `spec` in the import graph and above only
`backend`, so the setters and the carriers are two callers of one rule. The alternative is what
`truncation_order` was until `0dde6c5`: two checks that had drifted so far apart that the JSON
door refused five spellings `rcwa()` accepts, while missing two the solver needed.

**It collects.** A setter refuses the first problem and stops, which is right for a setter and
wrong for a caller holding a whole scene -- a spec with four mistakes took four submissions to
enumerate. Nothing here raises; the caller decides what to do with the list.

What the layer cannot see is as important as what it can, and the two are separated on purpose:
a clean `check()` is not a promise the solve will succeed. Anything needing the scene crossed
with itself -- `phi` on a 1D grating, an order count against a cell, whether a shape recipe's
overlaps resolve -- is read by the builders and is out of reach here.
"""

from wavebit.compat import legacy as _legacy_wavebit
import numpy as np
import pytest

from wavebit.compat.legacy import Incidence, JobSpec, Layer, Numerics, Stack
from wavebit import Material

GOOD = np.ones((1, 4, 4))


def _stack(**over):
    kw = dict(layers=[Layer(150.0, GOOD)], top=1.0, bot=2.25, period=[400.0, 400.0])
    kw.update(over)
    return Stack(**kw)


def test_a_sound_scene_reports_nothing():
    """The empty tuple is the answer, not a falsy accident -- assert the type too."""
    for carrier in (_stack(), Incidence(wavelength=620.0, theta=0.3, psi=0.2),
                    Numerics(truncation_order=[3, 3])):
        assert carrier.check() == ()
    spec = JobSpec(stack=_stack(), incidence=Incidence(wavelength=620.0, theta=0.3),
                   numerics=Numerics(truncation_order=[3, 3]))
    assert spec.check() == ()


@pytest.mark.parametrize('kw, fragment', [
    (dict(wavelength=620.0, psi=0.3 + 0.2j), 'real angle'),
    (dict(wavelength=620.0, chi=0.3 + 0.2j), 'real angle'),
    (dict(wavelength=620.0, psi=0.3, polarization=(1, -1j)), 'both given'),
    (dict(wavelength=620.0, chi=0.3, polarization=(1, -1j)), 'both given'),
    (dict(wavelength=620.0, tetm='te', psi=0.3), 'tetm'),
    (dict(wavelength=620.0, incidence_side='botom'), 'incidence_side'),
    (dict(wavelength=-620.0), 'not a length'),
    (dict(wavelength=0.0), 'not a length'),
    (dict(wavelength='abc'), 'positive length'),
    (dict(wavelength=620.0 + 1j), 'complex'),
])
def test_the_incidence_gate_names_each_violation(kw, fragment):
    problems = Incidence(**kw).check()
    assert problems, kw
    assert any(fragment in p for p in problems), (fragment, problems)


def test_the_incidence_gate_reports_all_four_at_once():
    """The property a setter cannot have: four independent mistakes, one call.

    Submitted to the solver these cost four round trips, because each `raise` hides the rest.
    """
    problems = Incidence(wavelength=-620.0, psi=0.3 + 0.2j, polarization=(1, -1j),
                         incidence_side='side').check()
    assert len(problems) == 4, problems


@pytest.mark.parametrize('over, fragment', [
    (dict(top=Material(np.ones((4, 4)))), 'half-space'),
    (dict(bot=Material(np.ones((4, 4)))), 'half-space'),
    (dict(layers=[Layer(150.0, np.ones((1, 2, 4)), mu=np.ones((1, 2, 8)))]), 'share a shape'),
    (dict(period=[1.0, 2.0, 3.0]), '1 or 2 lengths'),
    (dict(lattice=[[1.0, 2.0], [2.0, 4.0]]), 'lattice'),
])
def test_the_stack_gate_names_each_violation(over, fragment):
    problems = _stack(**over).check()
    assert problems, over
    assert any(fragment in p for p in problems), (fragment, problems)


@pytest.mark.parametrize('kw, fragment', [
    (dict(truncation_order=-3), 'cannot be negative'),
    (dict(truncation_order=2.7), 'whole number'),
    (dict(complex_dtype=32), '128 or 64'),
    (dict(shape_nvf_grid_size=-7), 'shape_nvf_grid_size'),
    (dict(shape_nvf_grid_size=0), 'shape_nvf_grid_size'),
])
def test_the_numerics_gate_names_each_violation(kw, fragment):
    problems = Numerics(**kw).check()
    assert problems, kw
    assert any(fragment in p for p in problems), (fragment, problems)


def test_the_job_gate_labels_which_carrier_each_problem_came_from():
    """One prefix shape for all three, because guessing produced a wrong one.

    The first version appended the carrier name only when the message did not appear to
    already contain a field, which turned `psi and polarization both given` into
    `incidence.psi and polarization both given` -- reading as a field called
    "psi and polarization".
    """
    spec = JobSpec(
        stack=_stack(top=Material(np.ones((4, 4)))),
        incidence=Incidence(wavelength=-620.0, incidence_side='side'),
        numerics=Numerics(complex_dtype=32))
    problems = spec.check()
    assert len(problems) == 4, problems
    assert sum(p.startswith('[stack] ') for p in problems) == 1
    assert sum(p.startswith('[incidence] ') for p in problems) == 2
    assert sum(p.startswith('[numerics] ') for p in problems) == 1


# ------------------------------------------------- the gate and the solver, on one rule each

GOOD_KW = dict(structure=GOOD, period=[400.0, 400.0], truncation_order=[3, 3],
               thickness=[150.0], wavelength=620.0, top=1.0, bot=2.25)


def _solver_refuses(**over):
    try:
        _legacy_wavebit.rcwa(backend='numpy', **dict(GOOD_KW, **over))
        return False
    except Exception:                          # noqa: BLE001 - refusal is the datum
        return True


SHARED = [
    ('psi complex', dict(psi=0.3 + 0.2j),
     lambda: Incidence(wavelength=620.0, psi=0.3 + 0.2j)),
    ('chi complex', dict(chi=0.3 + 0.2j),
     lambda: Incidence(wavelength=620.0, chi=0.3 + 0.2j)),
    ('psi beside polarization', dict(psi=0.3, polarization=(1, -1j)),
     lambda: Incidence(wavelength=620.0, psi=0.3, polarization=(1, -1j))),
    ('tetm beside psi', dict(tetm='te', psi=0.3, phi=None),
     lambda: Incidence(wavelength=620.0, tetm='te', psi=0.3)),
    ('incidence_side misspelt', dict(incidence_side='botom'),
     lambda: Incidence(wavelength=620.0, incidence_side='botom')),
    ('a raster for a half-space', dict(top=Material(np.ones((4, 4)))),
     lambda: _stack(top=Material(np.ones((4, 4))))),
    ('mu on a different grid',
     dict(structure=np.ones((1, 2, 4)), mu=np.ones((1, 2, 8))),
     lambda: _stack(layers=[Layer(150.0, np.ones((1, 2, 4)), mu=np.ones((1, 2, 8)))])),
    ('a negative order count', dict(truncation_order=-3),
     lambda: Numerics(truncation_order=-3)),
    ('a fractional order count', dict(truncation_order=2.7),
     lambda: Numerics(truncation_order=2.7)),
    ('period arity', dict(period=[1.0, 2.0, 3.0]),
     lambda: _stack(period=[1.0, 2.0, 3.0])),
    ('a sound scene', dict(), lambda: Incidence(wavelength=620.0, psi=0.2)),
]


@pytest.mark.parametrize('name, over, carrier', SHARED, ids=[c[0] for c in SHARED])
def test_the_gate_and_the_solver_agree_where_they_share_a_rule(name, over, carrier):
    """The property the whole layer rests on, asserted rather than assumed.

    Each of these is one rule with two callers. If a later edit gives either door its own
    copy, this is the test that fails -- which is the failure `truncation_order` did not have
    for the years its two copies spent drifting.

    Agreement is on the verdict, not the message: the gate collects and the solver raises, so
    the texts differ in shape even when the rule is the same object.
    """
    assert bool(carrier().check()) is _solver_refuses(**over), name


KNOWN_ASYMMETRIES = [
    ('wavelength', dict(wavelength=-620.0), lambda: Incidence(wavelength=-620.0),
     'no solver has a `wavelength` property at all -- no getter, no setter, just three '
     'constructor assignments -- so a negative one is accepted and solves'),
]


@pytest.mark.parametrize('name, over, carrier, why', KNOWN_ASYMMETRIES,
                         ids=[c[0] for c in KNOWN_ASYMMETRIES])
def test_the_gate_is_stricter_than_the_solver_here_and_that_is_recorded(name, over, carrier, why):
    """Remaining slots where the gate refuses and the solver does not.

    These are solver gaps rather than gate overreach: a negative wavelength solves and
    returns energy-conserving numbers. As the setters adopt the checks, rows move up into
    `SHARED`. Period arity is now checked by both doors.

    Pinned rather than left implicit so that a later reader finds a stated asymmetry instead of
    concluding the gate and the solver simply disagree.
    """
    assert carrier().check(), f'{name}: the gate should refuse -- {why}'
    assert not _solver_refuses(**over), f'{name}: the solver now refuses too; move this to SHARED'


@pytest.mark.parametrize('factorization', ['li', 'nvf'])
def test_normal_grid_is_refused_where_nothing_reads_it(factorization):
    """The sibling of ``raster_normal_method``'s refusal, and the asymmetry was not a decision.

    ``shape_nvf_grid_size`` sizes the grid the *shape* paths sample their analytic normal field on. A
    raster builds its field at the raster's own resolution, lifted to the
    ``4*truncation_order+1`` floor, so the keyword changes nothing there -- measured, 64 and
    4096 gave ``0.898147036010`` both times, under Li *and* under NVF. ``raster_normal_method`` has
    been refused where unread since it was added; this one was accepted and ignored.

    More urgently than for its sibling, in fact: this keyword's own comment in the ``rcwa``
    signature promises something about *shape* gradients, which a raster user cannot have.

    The default is exempt, because an untouched keyword is not a request -- which is what
    ``None`` means since 2026-08-27.
    """
    import numpy as np

    raster = np.full((1, 16, 16), 4.0)
    raster[0, 4:12, 4:12] = 12.25
    cfg = dict(backend='numpy', period=[300.0, 300.0], thickness=[205.0], truncation_order=(4, 4),
               top=1.0, bot=1.0, wavelength=900.0, psi=np.pi / 2, theta=0.0, phi=0.0,
               factorization=factorization)

    with pytest.raises(NotImplementedError, match='shape_nvf_grid_size'):
        _legacy_wavebit.rcwa(structure=raster, shape_nvf_grid_size=256, **cfg).solve()

    # the default is not a request, and a shape layer reads it
    assert float(np.sum(np.asarray(_legacy_wavebit.rcwa(structure=raster, **cfg).solve().T))) > 0.0
    shapes = [dict(background=1.0, shapes=[dict(type='circle', cx=0.0, cy=0.0, lx=140.0,
                                                material=4.0)])]
    assert float(np.sum(np.asarray(_legacy_wavebit.rcwa(
        structure=shapes, shape_nvf_grid_size=256,
        **dict(cfg, factorization='nvf')).solve().T))) > 0.0
