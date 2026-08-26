"""``fourier_coefficients`` is the other Fourier axis, and it had the identical bug.

``test_factorization_by_modeling_path.py`` is the sibling of this file and states the shape of
the problem; this one is the same fix on the other axis, so the two read in parallel on
purpose.

Only the raster path obtains coefficients two ways. Both shape paths state a layer as geometry
rather than as samples and integrate the permittivity over that geometry in closed form -- the
block path over the rectangles its modeler paints (``to_conv_mat_block``, every matrix built
with ``fourier2d_integrated``), the smooth path over each shape's analytic transform (``analytic.shape_ft``,
which *is* that integral). Neither has a raster underneath to run a discrete series along, and
neither used to look at the keyword at all.

That made this axis the quieter of the two. The factorization mismatch at least changed how
many convolution matrices came back; here the default ``'sampled'`` and an explicit
``'integrated'`` produce one identical run on both shape paths, because only one of them was
ever going to happen. Measured before the fix on an ellipse through the block modeler:
``T = 0.7609842245`` for both, bit for bit, with nothing in the result to say which series ran.

The fix is the same one, in the same style: store what was *asked for* -- ``None`` when nobody
asked -- and resolve it against the modeling path on read. ``'sampled'`` is the module-wide
default and cannot be refused on a path where nobody chose it; a spelled-out one is.
"""
import numpy as np
import pytest

import wavebit
from wavebit._core.lattice import as_lattice
from wavebit._core.enums import FourierCoefficients

KW = dict(backend=0, top=1.0, bot=1.0, theta=0.0, phi=0.0, psi=0.0,
          truncation_order=[4, 4], period=[1.0, 1.0], wavelength=1.1, thickness=[0.3])

# One structure per modeling path, as close to the same circle as each path can state it --
# the same three the factorization file uses, for the same reason.
# Permittivity throughout: 6.25 is the index 2.5 all three used to be written with.
SMOOTH = [{'background': 1.0,
           'shapes': [{'type': 'circle', 'cx': 0.5, 'cy': 0.5,
                       'lx': 0.6, 'material': 6.25}]}]
_N = 32
_axis = (np.arange(_N) + 0.5) / _N - 0.5
_X, _Y = np.meshgrid(_axis, _axis)
RASTER = np.ones((1, _N, _N))
RASTER[0, _X ** 2 + _Y ** 2 <= 0.3 ** 2] = 6.25

STRUCTURES = {'smooth': SMOOTH, 'raster': RASTER}

#: The series each path forms, and therefore the one an untouched default resolves to.
IMPLEMENTED = {'smooth': 'integrated', 'raster': 'sampled'}

#: The series each shape path cannot form. The raster path is absent: it has both, so there is
#: nothing there to refuse.
UNAVAILABLE = {'smooth': 'sampled'}


def solver(path, **strategy):
    return wavebit.rcwa(structure=STRUCTURES[path], **KW, **strategy)


def totals(path, **strategy):
    res = solver(path, **strategy).solve()
    return (float(np.sum(np.asarray(res.R))), float(np.sum(np.asarray(res.T))))


# ------------------------------------------------------- an untouched default is not a request

@pytest.mark.parametrize('path,series', sorted(IMPLEMENTED.items()))
def test_the_default_resolves_to_whatever_the_path_forms(path, series):
    """The substance of the change: the default is resolved per path, not asserted globally.

    Without this the refusal below would be unshippable -- ``'sampled'`` is the module-wide
    default, so refusing it on the two shape paths would break every user of theirs who never
    set the keyword, which is all of them.
    """
    assert solver(path).fourier_coefficients == series


@pytest.mark.parametrize('path', ['smooth'])
def test_the_shape_path_defaults_are_still_integrated_to_the_bit(path):
    """Resolving the default must not have moved any shape-path user's numbers.

    Both paths always ran the closed-form series; the change is only that they now admit it. So
    the untouched default and a spelled-out ``'integrated'`` have to be the same run, exactly.
    """
    assert totals(path) == totals(path, fourier_coefficients='integrated')


def test_the_raster_default_is_unchanged_and_still_says_sampled():
    """The path that genuinely has a choice keeps the documented default.

    Pinned separately from ``test_fourier_options.py`` because that file asserts the default
    globally, and the default is no longer a global fact. The inequality on the last line is
    the observable the two shape paths cannot produce: a raster answers the keyword, so the two
    series give different numbers there and the same numbers everywhere else.
    """
    assert solver('raster').fourier_coefficients == 'sampled'
    assert totals('raster') != totals('raster', fourier_coefficients='integrated')


# ------------------------------------------------------------------- a spelled-out one is

@pytest.mark.parametrize('path,series', sorted(UNAVAILABLE.items()))
def test_asking_for_the_series_a_path_cannot_form_is_refused(path, series):
    """The regression this file exists for.

    Before, both of these ran to completion and returned the closed-form numbers. The message
    has to name the series that was asked for, since the whole failure mode is a user who
    believes a different one is running.
    """
    with pytest.raises(NotImplementedError, match=f"fourier_coefficients={series!r}"):
        totals(path, fourier_coefficients=series)


#: How each refusal is expected to name what the path does instead. Matched as a phrase rather
#: than as the bare enum value, because a bare ``integrated`` would also match a message that
#: only ever says the word in passing.
NAMES_ITS_SERIES = {'smooth': r'closed form'}


@pytest.mark.parametrize('path,series', sorted(UNAVAILABLE.items()))
def test_the_refusal_says_what_the_path_does_instead(path, series):
    """A refusal that does not say what to do instead is a worse bug report than the silence."""
    with pytest.raises(NotImplementedError, match=NAMES_ITS_SERIES[path]):
        totals(path, fourier_coefficients=series)


def test_the_smooth_refusal_says_sampling_is_not_missing_but_meaningless_there():
    """Not "not implemented yet" -- a parametric shape has no samples to take a series of.

    Worth pinning as text, exactly as on the factorization axis: the natural reading of a
    ``NotImplementedError`` is "coming soon", and someone acting on that would go looking for a
    discrete series to add to a path built out of closed-form shape transforms.
    """
    with pytest.raises(NotImplementedError, match='not one it can form'):
        totals('smooth', fourier_coefficients='sampled')


# ------------------------------------------- every way of setting it, not just the constructor

@pytest.mark.parametrize('path,series', sorted(UNAVAILABLE.items()))
def test_setting_the_attribute_after_construction_is_refused_too(path, series):
    s = solver(path)
    s.fourier_coefficients = series
    with pytest.raises(NotImplementedError):
        s.solve()


@pytest.mark.parametrize('path,series', sorted(UNAVAILABLE.items()))
def test_passing_it_through_solve_is_refused_too(path, series):
    """``solve(**kwargs)`` sets attributes for optimization sweeps, so it is a real seam."""
    with pytest.raises(NotImplementedError):
        solver(path).solve(fourier_coefficients=series)


@pytest.mark.parametrize('path,series', sorted(UNAVAILABLE.items()))
def test_freezing_the_geometry_refuses_before_the_solve_does(path, series):
    """``freeze_geometry`` is the other entry point that takes strategy keywords.

    It refuses at freeze time rather than letting a bad record be built and blow up later, at a
    call site that no longer mentions the keyword.
    """
    with pytest.raises(NotImplementedError):
        solver(path).freeze_geometry(fourier_coefficients=series)


@pytest.mark.parametrize('name,code', [('jax', 1), ('torch', 2)])
@pytest.mark.parametrize('path,series', sorted(UNAVAILABLE.items()))
def test_the_refusal_holds_on_every_backend(name, code, path, series):
    """The dispatch is in the shared mixin, so a backend that still substituted would mean one
    of the three had grown its own copy."""
    pytest.importorskip(name)
    with pytest.raises(NotImplementedError):
        wavebit.rcwa(structure=STRUCTURES[path], **dict(KW, backend=code),
                       fourier_coefficients=series).solve()


# ------------------------------------------------- the boundary sub-path of modeling_type=2

# A painted layer whose two rectangles actually overlap. This is the fourth build in the
# codebase and the one neither `_..._IMPLEMENTED_BY` table names: it is not a modeling type of
# its own but a branch inside `to_conv_mat_smooth_layers`, taken when
# `analytic.layer_needs_resolving` says the painting has to be resolved. It never calls
# `shape_ft` or `to_conv_mat_smooth_nvf` -- it clips the painting into disjoint polygons and
# transforms those (`boundary.to_conv_mat_boundary_layer`).
#
# Both tables key on `_modeling_type`, so it inherits the smooth path's row, and that
# is right on the substance: a polygon transform is exact, so the coefficients are integrated
# ones, and the assembly is the same NVF five-matrix form. What it does *not* inherit for free
# is an accurate refusal -- both messages used to name only the per-shape transform, which is
# the one thing an overlapping layer never runs. Pinned here on both axes because the tables
# cannot distinguish the two sub-paths and so cannot be trusted to have thought about this one.
#
# Permittivity here too, on the same rule as the three structures at the top of the file: 4.0
# and 9.0 are the indices 2.0 and 3.0 these rectangles used to be written with. The two differ
# so that the overlap resolves to a third region rather than merging away, which is the whole
# reason the layer takes the resolving branch -- keeping them distinct after the flip is what
# the numbers had to preserve, not their literal text.
PAINTED_PERIOD = [1.2, 1.2]
PAINTED = [dict(background=1.0,
                shapes=[dict(type='rectangle', cx=0.40, cy=0.40, lx=0.34, ly=0.34,
                             angle=0.0, material=4.0),
                        dict(type='rectangle', cx=0.56, cy=0.56, lx=0.34, ly=0.34,
                             angle=0.0, material=9.0)])]
PAINTED_KW = dict(KW, period=PAINTED_PERIOD, truncation_order=[6, 6], thickness=[0.6], wavelength=1.6)


def _painted(**strategy):
    return wavebit.rcwa(structure=PAINTED, **PAINTED_KW, **strategy)


def test_the_boundary_sub_path_is_covered_by_the_smooth_row():
    """It really does take the resolving branch, and really does resolve to the smooth pair."""
    from wavebit._core import analytic
    assert analytic.layer_needs_resolving(PAINTED[0], as_lattice(period=PAINTED_PERIOD))
    s = _painted()
    assert s._modeling_type == 2
    assert s.fourier_coefficients == 'integrated' and s.factorization == 'nvf'
    assert sum(m is not None for m in s._eps_conv_matrices()) == 5


def test_the_boundary_sub_path_default_is_unmoved_on_both_axes():
    base = float(np.asarray(_painted().solve().T).real.sum())
    for strategy in (dict(fourier_coefficients='integrated'), dict(factorization='nvf'),
                     dict(fourier_coefficients='integrated', factorization='nvf')):
        assert float(np.asarray(_painted(**strategy).solve().T).real.sum()) == base


@pytest.mark.parametrize('strategy,named', [(dict(fourier_coefficients='sampled'),
                                             "fourier_coefficients='sampled'"),
                                            (dict(factorization='li'), "factorization='li'")])
def test_the_boundary_sub_path_refuses_what_the_smooth_path_refuses(strategy, named):
    with pytest.raises(NotImplementedError, match=named):
        _painted(**strategy).solve()


def test_the_refusal_names_the_transform_an_overlapping_layer_actually_runs():
    """The message has to survive the sub-path it is handed to.

    The refusal used to name only ``shape_ft``, and an overlapping painted layer never calls
    it -- so the one concrete pointer in the message sent the reader to a function their
    structure does not reach. Naming the polygon route too is the whole repair, and it is text,
    so only a test keeps it.

    Scoped to the coefficients axis on purpose. ``factorization`` had the identical defect and
    the identical fix, but that message belongs to
    ``test_factorization_by_modeling_path.py``, which pins it there. Asserting someone else's
    wording here would make this file fail on a rewording that is theirs to make.
    """
    with pytest.raises(NotImplementedError, match='to_conv_mat_boundary_layer'):
        _painted(fourier_coefficients='sampled').solve()


def test_the_boundary_sub_path_still_freezes():
    """``freeze_geometry`` reads both options before it plans, so a bad resolution would
    surface here rather than at the replay."""
    geometry = _painted().freeze_geometry()
    assert geometry.boundary_plans is not None


# ------------------------------------------------------------------ the JAX pytree

def test_a_pytree_round_trip_does_not_turn_an_unset_option_into_a_request():
    """``SolverJax._tree_flatten`` has to carry what was *asked for*, not what resolved.

    It is the one seam that rebuilds a solver from its own attributes, so flattening the
    resolved values would quietly promote both axes: a smooth-path solver that had merely
    accepted its path's only series would come back out asserting it, and one traced under a
    different ``structure`` -- which is the whole point of the pytree -- would then be refused for a
    request its author never made. Reading the resolved values here would also fire the refusal
    inside JAX's tree machinery rather than at the solve being refused, which is pinned below.
    """
    jax = pytest.importorskip('jax')
    s = wavebit.rcwa(structure=SMOOTH, **dict(KW, backend=1))
    leaves, treedef = jax.tree_util.tree_flatten(s)
    back = jax.tree_util.tree_unflatten(treedef, leaves)

    assert back._fourier_coefficients is None and back._factorization is None
    assert back.fourier_coefficients == 'integrated' and back.factorization == 'nvf'
    assert (float(np.sum(np.asarray(back.solve().T))) ==
            float(np.sum(np.asarray(s.solve().T))))


def test_a_pytree_round_trip_keeps_a_real_request_and_still_refuses_it_at_the_solve():
    jax = pytest.importorskip('jax')
    s = wavebit.rcwa(structure=SMOOTH, fourier_coefficients='sampled', **dict(KW, backend=1))
    leaves, treedef = jax.tree_util.tree_flatten(s)     # must not raise here
    back = jax.tree_util.tree_unflatten(treedef, leaves)
    assert back._fourier_coefficients == 'sampled'
    with pytest.raises(NotImplementedError, match="fourier_coefficients='sampled'"):
        back.solve()


# ------------------------------------------------------ both axes unavailable at once

def test_a_smooth_call_naming_both_unavailable_values_is_told_about_the_rule():
    """Two true refusals apply at once, and ``_eps_conv_matrices`` reads the factorization first.

    The order is a judgement rather than a necessity -- unlike the lattice/keyword order next
    door, either message here is actionable. It goes to the factorization because that is the
    axis that changes the physics of the answer, so a user who set both is likelier to have
    meant something by it. Pinned so the order is a decision rather than an accident.
    """
    with pytest.raises(NotImplementedError, match="factorization='li'"):
        totals('smooth', factorization='li', fourier_coefficients='sampled')


# --------------------------------------------------------------------------- through the spec

SPEC = {
    'version': '6',                 # a parsed document states its schema; see tests/test_spec.py
    'numerics': {'backend': 'numpy', 'truncation_order': [4, 4]},
    'stack': {'top': 1.0, 'bot': 1.0, 'period': [1.0, 1.0],
              'layers': [{'thickness': 0.3, 'structure': SMOOTH[0]}]},
    'incidence': {'wavelength': 1.1, 'theta': 0.0, 'phi': 0.0, 'psi': 0.0},
}


def test_a_spec_that_does_not_mention_the_coefficients_does_not_request_them():
    """``Numerics`` has to be able to say "unset" on this axis too.

    It is a dataclass with defaults, so a ``fourier_coefficients`` defaulting to
    ``FourierCoefficients.SAMPLED`` would hand every spec an explicit request it never made --
    and a spec carrying shape-path layers would be refused for asking for a series its author
    never named. Exactly the failure the ``factorization`` half of this already avoids.
    """
    from wavebit._core.spec import Numerics
    assert Numerics().fourier_coefficients is None
    assert 'fourier_coefficients' not in Numerics().as_kwargs()
    assert Numerics.from_dict(Numerics().to_dict()).fourier_coefficients is None

    result = wavebit.run(SPEC)
    assert float(np.sum(np.asarray(result.T))) == totals('smooth')[1]


def test_a_spec_that_does_mention_it_is_refused_like_any_other_request():
    numerics = dict(SPEC['numerics'], fourier_coefficients='sampled')
    with pytest.raises(NotImplementedError, match="fourier_coefficients='sampled'"):
        wavebit.run({**SPEC, 'numerics': numerics})


def test_a_spec_may_still_name_the_series_its_path_does_form():
    from wavebit._core.spec import Numerics
    assert (Numerics.from_dict({'fourier_coefficients': 'integrated'}).fourier_coefficients
            is FourierCoefficients.INTEGRATED)
    numerics = dict(SPEC['numerics'], fourier_coefficients='integrated')
    assert (float(np.sum(np.asarray(wavebit.run({**SPEC, 'numerics': numerics}).T))) ==
            totals('smooth')[1])


def test_a_raster_spec_still_carries_the_coefficients_through():
    """Omitting an unset value must not have stopped a set one from reaching the solver."""
    from wavebit._core.spec import Numerics
    assert (Numerics(fourier_coefficients=FourierCoefficients.INTEGRATED)
            .as_kwargs()['fourier_coefficients'] == 'integrated')
