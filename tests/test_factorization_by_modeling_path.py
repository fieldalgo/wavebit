"""``factorization`` means different things to the three modeling paths — and used to say so
to none of them.

Only the raster path implements both rules. The block path has Li's rules alone
(``to_conv_mat_block`` factorizes the painted rectangles and computes no normal field) and the
smooth shape path has NVF alone (``to_conv_mat_smooth_nvf`` builds eps from the shapes' own
analytic normal field, and there is no raster axis to run Li's nested 1D rule along). Neither
of those two used to look at the keyword at all: a smooth-path circle asked for
``factorization='li'`` returned NVF numbers — bit-identical to ``'nvf'``, five convolution
matrices either way — with nothing in the result to say so.

The fix is not to implement the missing rule; on the smooth path there is nothing to implement,
since NVF is a superset of Li and the path has no other axis to factorize along. It is to stop
substituting silently, in the style the rest of the module already uses
(``_reject_unsupported_lattice``, ``analytic.check_layer_combination``): refuse, and say why.
That also settles a temperament the two shape paths did not share — the smooth path already
refuses an unlabelled overlapping layer rather than guessing, while both used to swallow a
factorization request whole.

The whole difficulty is that ``'li'`` is also the *default*, so refusing it outright would
break every smooth-path user who never touched the keyword. So the solver stores what was
*asked for* — ``None`` when nobody asked — and resolves it against the modeling path on read.
An untouched default is not a request and is never refused; a spelled-out one is.

Both halves are pinned below: the refusals, and the silence that must not come back with them.
"""
import numpy as np
import pytest

import fieldwave
from fieldwave._core.enums import Factorization

KW = dict(backend=0, n_top=1.0, n_bot=1.0, theta=0.0, phi=0.0, psi=0.0,
          fto=[4, 4], period=[1.0, 1.0], wavelength=1.1, thickness=[0.3])

# One structure per modeling path, as close to the same circle as each path can state it.
# `modeling_type` is inferred from the type of `ucell`, never passed. A `circle` takes `lx`
# alone -- passing `ly` to one is refused by `analytic.check_shape_spec`.
SMOOTH = [{'n_base': 1.0,
           'shapes': [{'type': 'circle', 'cx': 0.5, 'cy': 0.5,
                       'lx': 0.6, 'n_index': 2.5}]}]
BLOCK = [[1.0, [['ellipse', 0.5, 0.5, 0.6, 0.6, 2.5, 0.0, 24, 24]]]]

_N = 32
_axis = (np.arange(_N) + 0.5) / _N - 0.5
_X, _Y = np.meshgrid(_axis, _axis)
RASTER = np.ones((1, _N, _N))
RASTER[0, _X ** 2 + _Y ** 2 <= 0.3 ** 2] = 2.5

UCELLS = {'smooth': SMOOTH, 'block': BLOCK, 'raster': RASTER}

#: The rule each path implements, and therefore the one an untouched default resolves to.
IMPLEMENTED = {'smooth': 'nvf', 'block': 'li', 'raster': 'li'}

#: The rule each single-rule path does *not* implement. The raster path is absent: it has
#: both, so there is nothing there to refuse.
UNAVAILABLE = {'smooth': 'li', 'block': 'nvf'}


def solver(path, **strategy):
    return fieldwave.rcwa(ucell=UCELLS[path], **KW, **strategy)


def totals(path, **strategy):
    res = solver(path, **strategy).conv_solve()
    return (float(np.sum(np.asarray(res.de_ri))), float(np.sum(np.asarray(res.de_ti))))


def n_conv_matrices(path, **strategy):
    """How many of the five convolution matrices came back non-``None``.

    This is the observable that separates the two rules regardless of the numbers: Li's rules
    return three (``epx``, ``epy``, ``epz_i``) and NVF returns five, because it couples x and
    y so ``epxy`` / ``epyx`` stop being zero. On the block path NVF is therefore not merely
    unimplemented but structurally absent — there are only ever three matrices to return.
    """
    return sum(m is not None for m in solver(path, **strategy)._conv_matrices())


# ------------------------------------------------------- an untouched default is not a request

@pytest.mark.parametrize('path,rule', sorted(IMPLEMENTED.items()))
def test_the_default_resolves_to_whatever_the_path_implements(path, rule):
    """The substance of the change: the default is resolved per path, not asserted globally.

    Without this the refusal below would be unshippable — ``'li'`` is the module-wide default,
    so refusing it on the smooth path would break every smooth-path user who never set the
    keyword, which is most of them.
    """
    assert solver(path).factorization == rule


def test_the_smooth_default_is_still_nvf_to_the_bit():
    """Resolving the default must not have moved any smooth-path user's numbers.

    The smooth path always ran NVF; the change is only that it now admits it. So the untouched
    default and a spelled-out ``'nvf'`` have to be the same run, exactly.
    """
    assert totals('smooth') == totals('smooth', factorization='nvf')
    assert n_conv_matrices('smooth') == 5


def test_the_block_default_is_still_li_to_the_bit():
    assert totals('block') == totals('block', factorization='li')
    assert n_conv_matrices('block') == 3


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


#: How each refusal is expected to name the rule the path *does* use. Matched as a phrase
#: rather than as the bare enum value, because a bare ``li`` matches "applies" and would let a
#: message that never names Li at all pass.
NAMES_ITS_RULE = {'smooth': r'\bNVF\b', 'block': r"Li's rules"}


@pytest.mark.parametrize('path,rule', sorted(UNAVAILABLE.items()))
def test_the_refusal_names_the_rule_the_path_does_use(path, rule):
    """A refusal that does not say what to do instead is a worse bug report than the silence."""
    with pytest.raises(NotImplementedError, match=NAMES_ITS_RULE[path]):
        totals(path, factorization=rule)


# A painted layer whose shapes actually overlap is still `modeling_type=2`, but it does not
# reach `to_conv_mat_smooth_nvf` at all: `to_conv_mat_smooth_layers` resolves the painting into
# disjoint polygons and transforms those instead (`boundary.to_conv_mat_boundary_layer`).
PAINTED_OVERLAP = [{'n_base': 1.0, 'combine': 'paint', 'shapes': [
    {'type': 'rectangle', 'cx': 0.45, 'cy': 0.5, 'lx': 0.4, 'ly': 0.4, 'n_index': 2.0},
    {'type': 'rectangle', 'cx': 0.60, 'cy': 0.5, 'lx': 0.4, 'ly': 0.4, 'n_index': 3.0}]}]


def test_the_boundary_sub_path_is_classified_and_refused_like_the_rest_of_smooth():
    """The polygon route is the same path by every tell that matters, so it must refuse alike."""
    s = fieldwave.rcwa(ucell=PAINTED_OVERLAP, **KW)
    assert s.factorization == 'nvf'
    assert sum(m is not None for m in s._conv_matrices()) == 5
    with pytest.raises(NotImplementedError, match="factorization='li'"):
        fieldwave.rcwa(ucell=PAINTED_OVERLAP, factorization='li', **KW).conv_solve()


def test_the_smooth_refusal_names_the_route_the_refused_layer_actually_takes():
    """A refusal that cites one concrete function has to cite the right one.

    The message named only ``to_conv_mat_smooth_nvf``, which a painted overlapping layer never
    calls — so the single pointer it offered sent that reader to code their structure does not
    run. Nothing but a test keeps message text honest, hence this one.
    """
    with pytest.raises(NotImplementedError, match='to_conv_mat_boundary_layer'):
        fieldwave.rcwa(ucell=PAINTED_OVERLAP, factorization='li', **KW).conv_solve()
    with pytest.raises(NotImplementedError, match='to_conv_mat_smooth_nvf'):
        totals('smooth', factorization='li')


def test_the_smooth_refusal_says_li_is_not_missing_but_meaningless_there():
    """Not "not implemented yet" — there is no raster axis for Li's nested rule to run along.

    Worth pinning as text: the natural reading of a ``NotImplementedError`` is "coming soon",
    and someone acting on that would go looking for a nested 1D rule to add to a path built
    out of analytic shape transforms.
    """
    with pytest.raises(NotImplementedError, match='superset of Li'):
        totals('smooth', factorization='li')


# ------------------------------------------- every way of setting it, not just the constructor

@pytest.mark.parametrize('path,rule', sorted(UNAVAILABLE.items()))
def test_setting_the_attribute_after_construction_is_refused_too(path, rule):
    s = solver(path)
    s.factorization = rule
    with pytest.raises(NotImplementedError):
        s.conv_solve()


@pytest.mark.parametrize('path,rule', sorted(UNAVAILABLE.items()))
def test_passing_it_through_conv_solve_is_refused_too(path, rule):
    """``conv_solve(**kwargs)`` sets attributes for optimization sweeps, so it is a real seam."""
    with pytest.raises(NotImplementedError):
        solver(path).conv_solve(factorization=rule)


@pytest.mark.parametrize('path,rule', sorted(UNAVAILABLE.items()))
def test_freezing_the_geometry_refuses_before_the_solve_does(path, rule):
    """``freeze_geometry`` is the other entry point that takes strategy keywords.

    It refuses at freeze time rather than letting a bad record be built and blow up later, at
    a call site that no longer mentions the keyword.
    """
    with pytest.raises(NotImplementedError):
        solver(path).freeze_geometry(factorization=rule)


def test_the_deprecated_fourier_type_alias_is_refused_the_same_way():
    """``fourier_type=0`` spells ``('sampled', 'li')`` out, so on smooth shapes it is a request.

    The alias exists for outside code that never migrated, which is exactly the code most
    likely to be passing a 0 it does not mean.
    """
    with pytest.warns(DeprecationWarning):
        s = fieldwave.rcwa(ucell=SMOOTH, fourier_type=0, **KW)
    with pytest.raises(NotImplementedError, match="factorization='li'"):
        s.conv_solve()


def test_reading_the_deprecated_alias_reports_the_resolved_rule():
    """``fourier_type`` reads back 2 on a smooth solver nobody configured — that is NVF."""
    with pytest.warns(DeprecationWarning):
        assert fieldwave.rcwa(ucell=SMOOTH, **KW).fourier_type == 2


@pytest.mark.parametrize('name,code', [('jax', 1), ('torch', 2)])
@pytest.mark.parametrize('path,rule', sorted(UNAVAILABLE.items()))
def test_the_refusal_holds_on_every_backend(name, code, path, rule):
    """The dispatch is in the shared mixin, so a backend that still substituted would mean
    one of the three had grown its own copy."""
    pytest.importorskip(name)
    with pytest.raises(NotImplementedError):
        fieldwave.rcwa(ucell=UCELLS[path], **dict(KW, backend=code),
                       factorization=rule).conv_solve()


# ------------------------------------------------------------------- the knock-on consequences

def test_a_skewed_smooth_cell_no_longer_has_to_spell_out_nvf():
    """The lattice refusal was reading the *unresolved* default and calling it a request.

    ``_reject_unsupported_lattice`` refuses Li on a skewed cell because Li's inverse rule runs
    along the raster axes, which stop being x and y under a skew. The smooth path never used
    Li, so that refusal never applied to it — but with the default unresolved it fired anyway,
    and every skewed smooth-path run had to pass ``factorization='nvf'`` to talk it down.
    """
    hexagonal = np.array([[1.0, 0.5], [0.0, np.sqrt(3) / 2]])
    default = fieldwave.rcwa(ucell=SMOOTH, lattice=hexagonal, **KW)
    spelled_out = fieldwave.rcwa(ucell=SMOOTH, lattice=hexagonal, factorization='nvf', **KW)
    assert default.factorization == 'nvf'
    assert (float(np.sum(default.conv_solve().de_ti)) ==
            float(np.sum(spelled_out.conv_solve().de_ti)))


def test_a_skewed_raster_cell_still_refuses_li():
    """The neighbouring refusal is untouched: on a raster, ``'li'`` really is the rule asked for."""
    hexagonal = np.array([[1.0, 0.5], [0.0, np.sqrt(3) / 2]])
    with pytest.raises(NotImplementedError, match="factorization='nvf'"):
        fieldwave.rcwa(ucell=RASTER, lattice=hexagonal, **KW).conv_solve()


def test_a_skewed_block_cell_reports_the_lattice_first_not_the_keyword():
    """Two true refusals apply at once here, and the order between them is load-bearing.

    A block ucell on a skewed cell is unbuildable *whatever* rule is named — the modeler
    rasterises onto a rectangular grid of its own. So naming the keyword instead would send
    the user to change the one thing that cannot help. `_conv_matrices` therefore checks the
    lattice before it reads the rule, and this pins that: with the refusals swapped, the
    message below is the ``factorization='nvf'`` one and
    ``test_oblique_lattice.py::test_the_vector_modeler_is_refused_on_a_skewed_cell`` fails too.
    """
    hexagonal = np.array([[1.0, 0.5], [0.0, np.sqrt(3) / 2]])
    with pytest.raises(NotImplementedError, match='block modeler'):
        fieldwave.rcwa(ucell=BLOCK, lattice=hexagonal, factorization='nvf', **KW).conv_solve()


def test_freezing_gives_the_same_diagnosis_the_solve_would():
    """One structure must not get two different explanations depending on the entry point.

    ``freeze_geometry`` refuses the same two things in the same order as ``_conv_matrices``.
    Before that, the keyword check landed first here and the lattice check first there, so a
    skewed block cell was told to fix its ``factorization`` if it froze and to fix its
    ``ucell`` if it solved — for one structure that has to change its ``ucell`` either way.
    """
    hexagonal = np.array([[1.0, 0.5], [0.0, np.sqrt(3) / 2]])
    s = fieldwave.rcwa(ucell=BLOCK, lattice=hexagonal, factorization='nvf', **KW)
    with pytest.raises(NotImplementedError, match='block modeler') as frozen:
        s.freeze_geometry()
    with pytest.raises(NotImplementedError, match='block modeler') as solved:
        s.conv_solve()
    assert str(frozen.value) == str(solved.value)


# --------------------------------------------------------------------------- through the spec

SPEC = {
    'numerics': {'backend': 'numpy', 'fto': [4, 4]},
    'stack': {'n_top': 1.0, 'n_bot': 1.0, 'period': [1.0, 1.0],
              'layers': [{'thickness': 0.3, 'eps': SMOOTH[0]}]},
    'incidence': {'wavelength': 1.1, 'theta': 0.0, 'phi': 0.0, 'psi': 0.0},
}


def test_a_spec_that_does_not_mention_factorization_does_not_request_one():
    """``Numerics`` has to be able to say "unset" too, or it re-introduces the whole problem.

    It is a dataclass with defaults, so a ``factorization`` defaulting to ``Factorization.LI``
    would hand every spec an explicit request it never made — and a spec carrying smooth-path
    layers would be refused for asking for a rule its author never named.
    """
    from fieldwave._core.spec import Numerics
    assert Numerics().factorization is None
    assert 'factorization' not in Numerics().as_kwargs()
    assert Numerics.from_dict(Numerics().to_dict()).factorization is None

    result = fieldwave.run(SPEC)
    assert float(np.sum(np.asarray(result.de_ti))) == totals('smooth')[1]


def test_a_spec_that_does_mention_it_is_refused_like_any_other_request():
    numerics = dict(SPEC['numerics'], factorization='li')
    with pytest.raises(NotImplementedError, match="factorization='li'"):
        fieldwave.run({**SPEC, 'numerics': numerics})


def test_a_spec_may_still_name_the_rule_its_path_does_implement():
    from fieldwave._core.spec import Numerics
    assert Numerics.from_dict({'factorization': 'nvf'}).factorization is Factorization.NVF
    numerics = dict(SPEC['numerics'], factorization='nvf')
    assert (float(np.sum(np.asarray(fieldwave.run({**SPEC, 'numerics': numerics}).de_ti))) ==
            totals('smooth')[1])
