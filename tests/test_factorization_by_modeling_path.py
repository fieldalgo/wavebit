"""Explicit rules are honored: circles require NVF, Cartesian paintings also support Li."""
import numpy as np
import pytest

import wavebit
from wavebit._core.enums import Factorization

KW = dict(backend=0, top=1.0, bot=1.0, theta=0.0, phi=0.0, psi=0.0,
          truncation_order=[4, 4], period=[1.0, 1.0], wavelength=1.1, thickness=[0.3])

#: `KW` without its cell. `period` and `lattice` are two spellings of one cell and naming
#: both is refused, so the calls below that give a basis take the scene without the period.
KW_NO_CELL = {k: v for k, v in KW.items() if k != 'period'}

# One structure per modeling path, as close to the same circle as each path can state it.
# `modeling_type` is inferred from the type of `structure`, never passed. A `circle` takes `lx`
# alone -- passing `ly` to one is refused by `parametric.check_shape_spec`.
# Permittivity throughout: 6.25 is the index 2.5 all three used to be written with.
SMOOTH = [{'background': 1.0,
           'shapes': [{'type': 'circle', 'cx': 0.0, 'cy': 0.0,
                       'lx': 0.6, 'material': 6.25}]}]
_N = 32
_X, _Y = wavebit.Lattice.rectangular(1.0).meshgrid_uv((_N, _N))
RASTER = np.ones((1, _N, _N))
RASTER[0, _X ** 2 + _Y ** 2 <= 0.3 ** 2] = 6.25

STRUCTURES = {'smooth': SMOOTH, 'raster': RASTER}

#: The rule each path implements, and therefore the one an untouched default resolves to.
IMPLEMENTED = {'smooth': 'nvf', 'raster': 'li'}

#: The rule each single-rule path does *not* implement. The raster path is absent: it has
#: both, so there is nothing there to refuse.
UNAVAILABLE = {'smooth': 'li'}


def solver(path, **strategy):
    return wavebit.rcwa(structure=STRUCTURES[path], **KW, **strategy)


def totals(path, **strategy):
    res = solver(path, **strategy).solve()
    return (float(np.sum(np.asarray(res.R))), float(np.sum(np.asarray(res.T))))


def n_conv_matrices(path, **strategy):
    """How many of the five convolution matrices came back non-``None``.

    This is the observable that separates the two rules regardless of the numbers: Li's rules
    return three (``epx``, ``epy``, ``epz_i``) and NVF returns five, because it couples x and
    y so ``epxy`` / ``epyx`` stop being zero. On the block path NVF is therefore not merely
    unimplemented but structurally absent — there are only ever three matrices to return.
    """
    return sum(m is not None for m in solver(path, **strategy)._eps_conv_matrices())


# ------------------------------------------------------- an untouched default is not a request

@pytest.mark.parametrize('path,rule', sorted(IMPLEMENTED.items()))
def test_the_default_resolves_to_whatever_the_path_implements(path, rule):
    """The substance of the change: the default is resolved per path, not asserted globally.

    Without this the refusal below would be unshippable — ``'li'`` is the module-wide default,
    so refusing it on the parametric path would break every smooth-path user who never set the
    keyword, which is most of them.
    """
    assert solver(path).factorization == rule


def test_the_smooth_default_is_still_nvf_to_the_bit():
    """Resolving the default must not have moved any smooth-path user's numbers.

    The parametric path always ran NVF; the change is only that it now admits it. So the untouched
    default and a spelled-out ``'nvf'`` have to be the same run, exactly.
    """
    assert totals('smooth') == totals('smooth', factorization='nvf')
    assert n_conv_matrices('smooth') == 5


def test_the_raster_default_is_unchanged_and_still_says_li():
    """The path that genuinely has a choice keeps the documented default.

    Pinned separately from ``test_fourier_options.py`` because that file asserts the default
    globally, and the default is no longer a global fact.
    """
    assert solver('raster').factorization == 'li'
    assert n_conv_matrices('raster') == 3
    assert n_conv_matrices('raster', factorization='nvf') == 5
    assert totals('raster') != totals('raster', factorization='nvf')


# ------------------------------------------------------------------- a spelled-out one is

@pytest.mark.parametrize('path,rule', sorted(UNAVAILABLE.items()))
def test_asking_for_the_rule_a_path_does_not_implement_is_refused(path, rule):
    """The regression this file exists for.

    Before, both of these ran to completion and returned the other rule's numbers. The
    message has to name the rule that was asked for, since the whole failure mode is a user
    who believes a different one is running.
    """
    with pytest.raises(NotImplementedError, match=f"factorization={rule!r}"):
        totals(path, factorization=rule)


#: Every spelling of the same explicit request. The refusal has to key on the *value asked
#: for*, never on the spelling it arrived in, so all of these are one request in three
#: costumes -- ``factorization_of`` normalizes them before the comparison ever runs.
SPELLINGS_OF_LI = ['li', 'LI', Factorization.LI]


@pytest.mark.parametrize('spelling', SPELLINGS_OF_LI, ids=['lower', 'upper', 'enum'])
def test_a_request_is_refused_however_it_is_spelled(spelling):
    """A non-canonical spelling of an explicit request must be refused like the canonical one.

    This replaces the ``fourier_type=0`` case that stood here while the deprecated integer
    alias existed. The alias is gone, but the question it pinned is not: any spelling that is
    normalized on the way in must still read as a *request* when the refusal decides whether
    the user asked for something or merely left the default alone. A spelling that slipped
    through unnormalized would compare unequal to ``Factorization.NVF`` and be refused for the
    wrong reason -- or, written straight onto the private attribute, silently substitute.
    """
    with pytest.raises(NotImplementedError, match="factorization='li'"):
        wavebit.rcwa(structure=SMOOTH, factorization=spelling, **KW).solve()


#: How each refusal is expected to name the rule the path *does* use. Matched as a phrase
#: rather than as the bare enum value, because a bare ``li`` matches "applies" and would let a
#: message that never names Li at all pass.
NAMES_ITS_RULE = {'smooth': r'\bNVF\b'}


@pytest.mark.parametrize('path,rule', sorted(UNAVAILABLE.items()))
def test_the_refusal_names_the_rule_the_path_does_use(path, rule):
    """A refusal that does not say what to do instead is a worse bug report than the silence."""
    with pytest.raises(NotImplementedError, match=NAMES_ITS_RULE[path]):
        totals(path, factorization=rule)


# A painted layer whose shapes actually overlap is still `modeling_type=2`, but it does not
# reach `to_conv_mat_parametric_nvf` at all: `to_conv_mat_parametric_layers` resolves the painting into
# disjoint polygons and transforms those instead (`boundary.to_conv_mat_boundary_layer`).
PAINTED_OVERLAP = [{'background': 1.0, 'shapes': [
    {'type': 'rectangle', 'cx': -0.05, 'cy': 0.0, 'lx': 0.4, 'ly': 0.4, 'material': 4.0},
    {'type': 'rectangle', 'cx': 0.10, 'cy': 0.0, 'lx': 0.4, 'ly': 0.4, 'material': 9.0}]}]


def test_cartesian_painting_defaults_to_nvf_and_can_explicitly_select_li():
    default = wavebit.rcwa(structure=PAINTED_OVERLAP, **KW)
    assert default.factorization == 'nvf'
    assert sum(m is not None for m in default._eps_conv_matrices()) == 5
    li = wavebit.rcwa(structure=PAINTED_OVERLAP, factorization='li', **KW)
    assert li.factorization == 'li'
    assert sum(m is not None for m in li._eps_conv_matrices()) == 3


def test_unsupported_shape_refusal_names_supported_geometry_and_alternative():
    with pytest.raises(NotImplementedError, match="axis-aligned rectangles.*factorization='nvf'"):
        totals('smooth', factorization='li')


# ------------------------------------------- every way of setting it, not just the constructor

@pytest.mark.parametrize('path,rule', sorted(UNAVAILABLE.items()))
def test_setting_the_attribute_after_construction_is_refused_too(path, rule):
    s = solver(path)
    s.factorization = rule
    with pytest.raises(NotImplementedError):
        s.solve()


@pytest.mark.parametrize('path,rule', sorted(UNAVAILABLE.items()))
def test_passing_it_through_solve_is_refused_too(path, rule):
    """``solve(**kwargs)`` sets attributes for optimization sweeps, so it is a real seam."""
    with pytest.raises(NotImplementedError):
        solver(path).solve(factorization=rule)


@pytest.mark.parametrize('path,rule', sorted(UNAVAILABLE.items()))
def test_freezing_the_geometry_refuses_before_the_solve_does(path, rule):
    """``freeze_geometry`` is the other entry point that takes strategy keywords.

    It refuses at freeze time rather than letting a bad record be built and blow up later, at
    a call site that no longer mentions the keyword.
    """
    with pytest.raises(NotImplementedError):
        solver(path).freeze_geometry(factorization=rule)


@pytest.mark.parametrize('name,code', [('jax', 1), ('torch', 2)])
@pytest.mark.parametrize('path,rule', sorted(UNAVAILABLE.items()))
def test_the_refusal_holds_on_every_backend(name, code, path, rule):
    """The dispatch is in the shared mixin, so a backend that still substituted would mean
    one of the three had grown its own copy."""
    pytest.importorskip(name)
    with pytest.raises(NotImplementedError):
        wavebit.rcwa(structure=STRUCTURES[path], **dict(KW, backend=code),
                       factorization=rule).solve()


# ------------------------------------------------------------------- the knock-on consequences

def test_a_skewed_smooth_cell_no_longer_has_to_spell_out_nvf():
    """The lattice refusal was reading the *unresolved* default and calling it a request.

    ``_reject_unsupported_lattice`` refuses Li on a skewed cell because Li's inverse rule runs
    along the raster axes, which stop being x and y under a skew. The parametric path never used
    Li, so that refusal never applied to it — but with the default unresolved it fired anyway,
    and every skewed smooth-path run had to pass ``factorization='nvf'`` to talk it down.
    """
    hexagonal = np.array([[1.0, 0.5], [0.0, np.sqrt(3) / 2]])
    default = wavebit.rcwa(structure=SMOOTH, lattice=hexagonal, **KW_NO_CELL)
    spelled_out = wavebit.rcwa(structure=SMOOTH, lattice=hexagonal, factorization='nvf',
                               **KW_NO_CELL)
    assert default.factorization == 'nvf'
    assert (float(np.sum(default.solve().T)) ==
            float(np.sum(spelled_out.solve().T)))


def test_a_skewed_raster_cell_still_refuses_li():
    """The neighbouring refusal is untouched: on a raster, ``'li'`` really is the rule asked for."""
    hexagonal = np.array([[1.0, 0.5], [0.0, np.sqrt(3) / 2]])
    with pytest.raises(NotImplementedError, match="factorization='nvf'"):
        wavebit.rcwa(structure=RASTER, lattice=hexagonal, **KW_NO_CELL).solve()


# --------------------------------------------------------------------------- through the spec

SPEC = {
    'version': '7',                 # a parsed document states its schema; see tests/test_spec.py
    'numerics': {'backend': 'numpy', 'truncation_order': [4, 4]},
    'stack': {'top': 1.0, 'bot': 1.0, 'period': [1.0, 1.0],
              'layers': [{'thickness': 0.3, 'structure': SMOOTH[0]}]},
    'incidence': {'wavelength': 1.1, 'theta': 0.0, 'phi': 0.0, 'psi': 0.0},
}


def test_a_spec_that_does_not_mention_factorization_does_not_request_one():
    """``Numerics`` has to be able to say "unset" too, or it re-introduces the whole problem.

    It is a dataclass with defaults, so a ``factorization`` defaulting to ``Factorization.LI``
    would hand every spec an explicit request it never made — and a spec carrying smooth-path
    layers would be refused for asking for a rule its author never named.
    """
    from wavebit._core.spec import Numerics
    assert Numerics().factorization is None
    assert 'factorization' not in Numerics().as_kwargs()
    assert Numerics.from_dict(Numerics().to_dict()).factorization is None

    result = wavebit.run(SPEC)
    assert float(np.sum(np.asarray(result.T))) == totals('smooth')[1]


def test_a_spec_that_does_mention_it_is_refused_like_any_other_request():
    numerics = dict(SPEC['numerics'], factorization='li')
    with pytest.raises(NotImplementedError, match="factorization='li'"):
        wavebit.run({**SPEC, 'numerics': numerics})


def test_a_spec_may_still_name_the_rule_its_path_does_implement():
    from wavebit._core.spec import Numerics
    assert Numerics.from_dict({'factorization': 'nvf'}).factorization is Factorization.NVF
    numerics = dict(SPEC['numerics'], factorization='nvf')
    assert (float(np.sum(np.asarray(wavebit.run({**SPEC, 'numerics': numerics}).T))) ==
            totals('smooth')[1])
