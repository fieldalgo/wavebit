"""The same-material short-circuit must be an optimization, not a different computation.

When a uniform layer is the same material as the medium just below it, the interface between
them reflects nothing, and the solve can skip solving for that interface. What it may *not*
skip is the layer: its thickness still has to reach the boundary matching, and its modes still
have to carry the transfer down to whatever is underneath.

The short-circuit did neither. It stored ``a_i = b = 0`` and left ``F`` / ``G`` untouched, so:

* the layer's thickness never entered ``F`` / ``G`` -- two identical layers on a *different*
  substrate reflected as though the upper one were absent (0.0735 against an Airy oracle's
  0.0338, a 118% error), and
* ``a_i = 0`` annihilated the field recursion ``T_layer = A_i @ X @ T_layer``, zeroing the
  reconstructed field in that layer and in **every layer below it**.

Both were invisible to the suite as it stood: the golden field cases contain no same-material
stack, and the only same-material case anywhere put the layer directly on a matching
substrate -- the one arrangement where ``b`` really is zero and the bug cancels.

``b`` is the *lower stack's* reflection expressed in the shared modes. It vanishes only when
that stack is itself reflectionless, which is exactly the matching-substrate case. Everything
here is arranged so it is not.
"""
import numpy as np
import pytest

import fieldwave
from fieldwave._core.geometry import StaticGeometry

WL = 500.0
AU = 0.18 + 3.0j                      # gold at 500nm

# n=1.5 twice in a row, on a substrate that is *not* 1.5, so `b` != 0 at the upper interface.
STACKED = ([1.0, 1.5, 1.5, 2.0], [20.0, 35.0])


def _airy_R(ns, ds, theta, pol):
    """Reflectance from the closed-form multilayer oracle in test_lossy_media."""
    from test_lossy_media import analytic_r
    return abs(analytic_r(ns, ds, WL, theta, pol)) ** 2


def _solver(ns, ds, *, pol=0, theta=0.2, path='1d', fto=0):
    """A uniform stack on the requested solve path (see test_lossy_media.solver_r)."""
    layers = ns[1:-1]
    common = dict(backend=0, n_top=ns[0], n_bot=ns[-1], theta=theta, wavelength=WL,
                  thickness=ds)
    if path == '1d':
        return fieldwave.rcwa(pol=pol, fto=fto, period=[WL * 2],
                              ucell=np.array([[[n, n]] for n in layers]), **common)
    cell = ([[[n, n]] for n in layers] if path == 'conical'
            else [[[n, n], [n, n]] for n in layers])
    return fieldwave.rcwa(phi=0.0, psi=(np.pi / 2 if pol == 0 else 0.0), fto=[fto, fto],
                          period=[WL * 2, WL * 2], ucell=np.array(cell), **common)


def _R(solver, same=None):
    kw = {} if same is None else dict(geometry=StaticGeometry(same_material=same))
    return float(np.sum(np.asarray(solver.conv_solve(**kw).de_ri)).real)


# ----------------------------------------------------------------------
# 1. the efficiency, against an absolute oracle
# ----------------------------------------------------------------------
@pytest.mark.parametrize('pol,theta',
                         [(p, t) for p in (0, 1) for t in (0.0, 0.4)],
                         ids=[f'{"te" if p == 0 else "tm"}_theta{t}'
                              for p in (0, 1) for t in (0.0, 0.4)])
def test_stacked_same_material_matches_the_analytic_stack(pol, theta):
    """Two identical layers on a different substrate: the upper one must still count.

    This is the arrangement the short-circuit got wrong, and an Airy recursion over the same
    indices settles it without reference to any solver.
    """
    ns, ds = STACKED
    solver = _solver(ns, ds, pol=pol, theta=theta)
    assert _R(solver) == pytest.approx(_airy_R(ns, ds, theta, pol), abs=1e-12)


@pytest.mark.parametrize('pol,theta',
                         [(p, t) for p in (0, 1) for t in (0.0, 0.4)],
                         ids=[f'{"te" if p == 0 else "tm"}_theta{t}'
                              for p in (0, 1) for t in (0.0, 0.4)])
def test_merging_same_material_layers_is_the_same_structure(pol, theta):
    """Splitting a slab in two must not change anything -- the split is not physical."""
    ns, ds = STACKED
    whole = _solver([1.0, 1.5, 2.0], [sum(ds)], pol=pol, theta=theta)
    split = _solver(ns, ds, pol=pol, theta=theta)
    assert _R(split) == pytest.approx(_R(whole), abs=1e-12)


# ----------------------------------------------------------------------
# 2. on == off, on every solve path
# ----------------------------------------------------------------------
@pytest.mark.parametrize('path', ['1d', 'conical', '2d'])
@pytest.mark.parametrize('pol', [0, 1], ids=['te', 'tm'])
def test_short_circuit_changes_nothing_on_any_path(path, pol):
    """Taking the shortcut and not taking it are the same computation.

    The general branch is the reference: it makes no same-material assumption and reproduces
    the oracle above. ``conical`` and ``2d`` carry their own copies of the branch, and the
    conical one additionally built its transfer in a different mode basis than its general
    branch used, so each needs saying separately.
    """
    ns, ds = STACKED
    solver = _solver(ns, ds, pol=pol, path=path)
    assert _R(solver) == pytest.approx(_R(solver, same=[False] * len(ds)), abs=1e-12)


def test_the_short_circuit_actually_fires_here():
    """Guard the guard: none of this proves anything if the flag never turns on."""
    ns, ds = STACKED
    flags = _solver(ns, ds).freeze_geometry().same_material
    assert flags == [True, False], flags          # ucell is top-to-bottom


def test_absorbing_substrate_is_detected_as_same_material():
    """An absorbing layer on a substrate of its own material must match it.

    The detector seeded ``eps_below`` with ``conj(n_bot)**2`` while the layer builders use a
    raw ``n_bot**2``, so for any complex ``n_bot`` the two could never agree and the flag was
    stuck False -- the short-circuit silently did not exist for lossy stacks.
    """
    flags = _solver([1.0, AU, AU], [20.0]).freeze_geometry().same_material
    assert flags == [True], flags


# ----------------------------------------------------------------------
# 3. the field below the layer
# ----------------------------------------------------------------------
RES = 10


def _field(ns, ds, *, pol=0, theta=0.2, fto=2, same=None):
    solver = _solver(ns, ds, pol=pol, theta=theta, fto=fto)
    kw = {} if same is None else dict(geometry=StaticGeometry(same_material=same))
    solver.conv_solve(**kw)
    return np.asarray(solver.calculate_field(res_x=RES, res_y=1, res_z=RES))


def test_field_below_a_same_material_layer_is_not_zero():
    """``a_i = 0`` propagated through ``T_layer = A_i @ X @ T_layer`` and killed everything.

    Layer 0 and layer 1 are both n=1.5 and layer 2 is not, so the flag fires on layer 1 and
    the field in layers 1 *and* 2 came back identically zero. A stack that reflects and
    transmits normally the whole time, with no field in most of it.
    """
    ns, ds = [1.0, 1.5, 1.5, 2.0, 1.0], [30.0, 30.0, 40.0]
    f = _field(ns, ds).reshape(3, RES, -1)
    for i in range(3):
        assert np.abs(f[i]).max() > 1e-3, f'layer {i} is empty'


@pytest.mark.parametrize('pol', [0, 1], ids=['te', 'tm'])
def test_field_matches_the_general_branch_and_the_merged_stack(pol):
    """And it is not merely non-zero: it is the field the un-shortcut stack has.

    Two references, because they fail differently. Forcing the flags off runs the identical
    structure through the general branch. Merging the two same-material layers into one of
    their combined thickness is a different *input* describing the same physical stack, so it
    also pins that the thickness survived the shortcut.
    """
    ns, ds = [1.0, 1.5, 1.5, 2.0, 1.0], [30.0, 30.0, 40.0]
    got = _field(ns, ds, pol=pol)
    assert np.allclose(got, _field(ns, ds, pol=pol, same=[False] * 3), atol=1e-12, rtol=0)

    merged = _field([1.0, 1.5, 2.0, 1.0], [60.0, 40.0], pol=pol)
    # Same z sampling per layer, so the merged run has 2 blocks where this one has 3; the
    # bottom (n=2.0) block is the one both describe identically.
    assert np.allclose(got.reshape(3, RES, -1)[2], merged.reshape(2, RES, -1)[1],
                       atol=1e-12, rtol=0)
