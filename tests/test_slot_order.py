"""Every builder returns its slots in ``solve_conv``'s parameter order.

That property is what makes ``solve_conv(*build(...))`` correct, and it is asserted
here rather than left to each builder's docstring because the failure is silent. Until
2026-08-18 the five-slot builders returned ``(epx, epy, epxy, epyx, epz_i)`` while
``solve_conv`` took ``(epx, epy, epz_i, epxy, epyx)``; splatting one into the other put the
inverted zz matrix in the ``epxy`` slot, and the solve answered with a full set of plausible
efficiencies rather than raising. On a lossless passive pillar it returned ``R + T = 1.61``
where the same structure gives ``1.0000000000`` -- only ``UnphysicalResultWarning`` fired, and
the three-slot builders splatted correctly the whole time, which is what made it a trap.

``solve_conv``'s order is the fixed one: ``epz_i`` is required (it eliminates ``E_z`` on
every solve) and the off-diagonal pair defaults to ``None``, so Python cannot take ``epz_i``
last. The builders are what moved.

Three assertions, because they fail on different mistakes:

* :func:`test_a_splatted_builder_conserves_energy` is the physics. It would catch a new builder
  written in the old order even if every docstring in the package agreed with it.
* :func:`test_the_slot_a_builder_calls_z_is_the_inverted_one` is structural, and it is the one
  that still works when a case is *not* lossless. The zz slot is the only one held as an
  inverse, so it can be identified without knowing the answer: for an isotropic layer it is
  ``inv(<eps>)``, whose DC entry recovers the mean permittivity.
* :func:`test_a_builder_returns_the_named_slot_type` is the declaration. The first two say a
  number came out wrong; this one names the field that moved.

The third exists because the first sweep of this repair *missed five call sites* -- positional
``[4]`` indexing in ``test_boundary.py`` and a local wrapper in ``test_nvf_vector.py`` -- and a
grep for the old spelling could not see them, since ``mats[4]`` names no slot at all. That is
what :class:`~wavebit._core.factorization.ConvMatrices` is for: reaching a single slot by name
(``mats.epz_i``) survives a reorder, and reaching it by number does not.

Oracle tier: **physics** (energy on a lossless passive layer), plus a structural identity and a
declaration check. None is a golden -- nothing here compares the code to its own past output.
"""
import contextlib
import warnings

import numpy as np
import pytest

import wavebit
from wavebit._core import analytic as _analytic
from wavebit._core import factorization as _fac


@contextlib.contextmanager
def _no_unphysical_warning():
    """``UnphysicalResultWarning`` promoted to an error for the duration of the solve.

    That warning is what the shipped code emits when R + T leaves 1. Letting it pass while
    asserting on the number would be checking the same thing twice and reporting it once.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('error', wavebit.UnphysicalResultWarning)
        yield


GRID, MAX_ORDER = 48, 4
KW = dict(backend=0, top=1.0, bot=1.0, theta=0.0, phi=0.0, psi=0.0,
          period=(1.0, 1.0), wavelength=1.1, thickness=[0.3])

#: A lossless, passive pillar -- the same shape in every spelling a builder accepts, so the
#: builders are being compared on one structure rather than on one number each. The raster rod
#: is stated in pixels and the analytic one derived from it, so the two describe one rod
#: exactly rather than nearly; ``_mean_eps`` below is then a single number for both.
#: The rod and its surround as PERMITTIVITY -- 2.25 is the index 1.5 this pillar used to be
#: written with, and the tensor builders below took the square themselves.
EPS_ROD, EPS_BASE = 2.25, 1.0
ROW0, ROW1, COL0, COL1 = 14, 34, 18, 30       # y extent, x extent, in pixels
ROD = (slice(ROW0, ROW1), slice(COL0, COL1))


def _raster():
    cell = np.full((1, GRID, GRID), float(EPS_BASE))
    cell[0][ROD] = EPS_ROD
    return cell


def _tensor_raster():
    """The same pillar as an isotropic 3x3 tensor per pixel, for the Popov-Neviere builders."""
    eps = np.zeros((1, GRID, GRID, 3, 3), dtype=complex)
    for i in range(3):
        eps[..., i, i] = EPS_BASE
        eps[0][ROD][..., i, i] = EPS_ROD
    return eps


def _shapes():
    """The same rod as :func:`_raster`, in absolute lengths (never fractions of the period)."""
    px, py = KW['period']
    return [dict(background=EPS_BASE, shapes=[dict(
        type='rectangle',
        cx=px * 0.5 * (COL0 + COL1) / GRID, cy=py * 0.5 * (ROW0 + ROW1) / GRID,
        lx=px * (COL1 - COL0) / GRID, ly=py * (ROW1 - ROW0) / GRID,
        material=EPS_ROD)])]


def _mean_eps():
    """Area-weighted ``<eps>`` of the pillar -- the DC entry ``inv(epz_i)`` must reproduce."""
    frac = ((ROW1 - ROW0) * (COL1 - COL0)) / GRID ** 2
    return frac * EPS_ROD + (1.0 - frac) * EPS_BASE


#: ``(name, callable)`` -- every builder a caller can splat into ``solve_conv``, at the
#: arity it returns. Adding a builder without adding it here is what the pairing test catches.
BUILDERS = {
    'raster_sampled': lambda: _fac.to_conv_mat_raster_sampled(_raster(), MAX_ORDER, MAX_ORDER),
    'raster_integrated': lambda: _fac.to_conv_mat_raster_integrated(_raster(), MAX_ORDER, MAX_ORDER),
    'raster_nvf': lambda: _fac.to_conv_mat_raster_nvf(_raster(), MAX_ORDER, MAX_ORDER),
    'raster_integrated_nvf': lambda: _fac.to_conv_mat_raster_integrated_nvf(
        _raster(), MAX_ORDER, MAX_ORDER),
    'raster_pn_eps': lambda: _fac.to_conv_mat_raster_pn_eps(_tensor_raster(), MAX_ORDER, MAX_ORDER),
    'smooth_layers': lambda: _analytic.to_conv_mat_smooth_layers(
        _shapes(), MAX_ORDER, MAX_ORDER, KW['period'], normal_grid=256,
        complex_dtype=np.complex128, xp=np),
}


@pytest.mark.parametrize('name', sorted(BUILDERS), ids=sorted(BUILDERS))
def test_a_splatted_builder_conserves_energy(name):
    """``solve_conv(*build(...))`` on a lossless passive layer must give R + T = 1.

    Not a convergence claim: the layer is lossless and passive, so this holds at any
    truncation. It is the cheapest assertion that fails when the slots arrive in the wrong
    holes, and 1e-6 is far tighter than the 0.61 the defect this pins was worth.
    """
    slots = BUILDERS[name]()
    assert len(slots) == 5, f'{name} returned {len(slots)} slots'

    with _no_unphysical_warning():
        res = wavebit.rcwa(max_order=[MAX_ORDER, MAX_ORDER], **KW).solve_conv(*slots)

    total = float(np.asarray(res.given.R).sum() + np.asarray(res.given.T).sum())
    assert abs(total - 1.0) < 1e-6, (
        f'{name}: splatting it into solve_conv gives R + T = {total!r} on a lossless '
        f'passive layer -- its slots are not in solve_conv\'s parameter order')


@pytest.mark.parametrize('name', sorted(BUILDERS), ids=sorted(BUILDERS))
def test_the_slot_a_builder_calls_z_is_the_inverted_one(name):
    """``epz_i`` holds ``inv(<eps>)``, identified without solving anything.

    The structural half of the pin. Every layer here is isotropic, so ``<eps>``'s DC entry is
    the area-weighted mean permittivity, and the zz slot is its inverse -- the one slot whose
    identity is readable off the matrix itself. A builder that put ``epxy`` here would fail
    this even on a lossy structure, where the energy assertion above says nothing.
    """
    slots = BUILDERS[name]()
    dc = np.linalg.inv(np.asarray(slots.epz_i[0]))[0, 0]
    expected = _mean_eps()

    assert abs(complex(dc).real - expected) < 1e-9, (
        f'{name}: inv(epz_i) has DC entry {complex(dc).real!r}, not the mean permittivity '
        f'{expected!r} -- the epz_i field is not inv(<eps>)')
    assert abs(complex(dc).imag) < 1e-12, f'{name}: epz_i is not real on a lossless layer'


@pytest.mark.parametrize('name', sorted(BUILDERS), ids=sorted(BUILDERS))
def test_a_builder_returns_the_named_slot_type(name):
    """The order is a named contract, not a convention held in docstrings.

    Every builder returns :class:`~wavebit._core.factorization.ConvMatrices`, so a reorder
    cannot be made without editing ``_fields`` here -- which is the point of naming them at
    all. The two tests above would catch a reorder by its physics; this one catches it at the
    declaration, and names what changed instead of reporting a wrong number.

    One type for every builder since 2026-08-21. The Li-rule ones used to answer with a
    three-slot type instead, which said "this rule does not couple x and y" by *arity* -- two
    arities for one quantity, and so two slot orders to keep straight. They now say it in the
    slot that carries it, which is what every consumer already read.

    ``issubclass(..., tuple)`` is asserted because everything downstream splats and unpacks
    these: a builder that returned a dataclass or a plain list would break every call site
    while satisfying the field check.
    """
    slots = BUILDERS[name]()
    assert isinstance(slots, _fac.ConvMatrices), (
        f'{name} returned a bare {type(slots).__name__}; use ConvMatrices so the slot '
        f'order is declared rather than positional')
    assert isinstance(slots, tuple), f'{name}: slots must stay tuple-compatible'

    expected = ('epx', 'epy', 'epz_i', 'epxy', 'epyx')
    assert slots._fields == expected, (
        f'{name} returns {slots._fields}, not {expected} -- solve_conv takes epz_i third')

    # A Li-rule builder says so with the pair, not with a shorter tuple. Asserted rather than
    # left implicit: `None` there is what routes the solve onto the uncoupled arithmetic.
    couples = name not in ('raster_sampled', 'raster_integrated')
    assert (slots.epxy is not None) == couples, (
        f'{name}: epxy is {"unset" if slots.epxy is None else "set"}, which claims the rule '
        f'{"does not couple" if slots.epxy is None else "couples"} x and y')


def test_every_five_slot_builder_is_exercised():
    """A builder added without a row above would otherwise be pinned by nothing.

    The list is the package's own, read at runtime: any public ``to_conv_mat_*`` in
    ``factorization`` or ``analytic`` has to appear in :data:`BUILDERS` or be named here as
    deliberately out of scope.
    """
    exempt = {
        # takes an already-modelled block instruction, not a structure -- covered via the solver
        # path in test_factorization_core.py
        'to_conv_mat_block',
        # permeability, not permittivity: its slots go to solve_conv's `mu=`, not
        # positionally. Pinned against fmmax in tests/references/.
        'to_conv_mat_raster_pn_mu',
        # one layer, and it takes resolved polygons rather than a structure; reached through
        # to_conv_mat_smooth_layers, which is exercised above.
        'to_conv_mat_boundary_layer',
        'to_conv_mat_smooth_nvf',
    }
    found = {n for mod in (_fac, _analytic) for n in dir(mod)
             if n.startswith('to_conv_mat_')}
    covered = {f'to_conv_mat_{n}' for n in BUILDERS}

    missing = found - covered - exempt
    assert not missing, (
        f'builders with nothing pinning their slot order: {sorted(missing)} -- add them to '
        f'BUILDERS, or to `exempt` with the reason')
