"""A layer identical to the medium below it is still a layer.

There is nothing special about this arrangement in the solve any more, and that is the claim
worth pinning, because for most of this codebase's life there was. A short-circuit detected
"uniform layer, same material as what is underneath" and skipped the interface matching for
it, storing ``a_i = b = 0`` and leaving ``F`` / ``G`` untouched. Two things followed:

* the layer's thickness never entered ``F`` / ``G``, so two identical layers on a *different*
  substrate reflected as though the upper one were absent (0.0735 against an Airy oracle's
  0.0338, a 118% error), and
* ``a_i = 0`` annihilated the field recursion ``T_layer = A_i @ X @ T_layer``, zeroing the
  reconstructed field in that layer and in **every layer below it**.

Neither was visible: the golden field cases contain no same-material stack, and the only
same-material case anywhere put the layer directly on a matching substrate -- the one
arrangement where the skipped term really is zero and the bug cancels.

The short-circuit was added in 2026-03 for a real failure. Sharing a mode basis with the
medium below made ``a = 0.5*(W_i F + V_i G)`` *singular* under the conjugated formulation of
the day (its own comment: "V_i @ G ~= -I, making A ~= 0"), so inverting it blew up. The
2026-08 convention merge flipped the sign of ``G``; the same quantity is now ``a = I``, with
condition number 1. The singularity was gone and the workaround was not, so the workaround is
what these tests now assert the absence of.
"""
import numpy as np
import pytest

import wavebit
from wavebit import eps_from_index

# Stacks are lists of refractive INDEX here, because the Airy oracle imported from
# ``test_lossy_media`` is written that way; the conversion happens where wavebit is called.

WL = 500.0
AU = 0.18 + 3.0j                      # gold at 500nm

# n=1.5 twice in a row, on a substrate that is *not* 1.5, so the term the short-circuit
# dropped is non-zero here.
STACKED = ([1.0, 1.5, 1.5, 2.0], [20.0, 35.0])


def _airy_R(ns, ds, theta, tetm):
    """Reflectance from the closed-form multilayer oracle in test_lossy_media."""
    from test_lossy_media import analytic_r
    return abs(analytic_r(ns, ds, WL, theta, tetm)) ** 2


def _solver(ns, ds, *, tetm='te', theta=0.2, path='1d', truncation_order=0, answers='amplitude'):
    """A uniform stack on the requested solve path (see test_lossy_media.solver_r)."""
    layers = ns[1:-1][::-1]  # Oracle cases are top-down; the public API is bottom-up.
    common = dict(backend='numpy', top=eps_from_index(ns[0]), bot=eps_from_index(ns[-1]),
                  theta=theta, wavelength=WL, thickness=list(reversed(ds)), answers=answers)
    if path == '1d':
        return wavebit.rcwa(tetm=tetm, truncation_order=truncation_order, period=[WL * 2],
                              structure=eps_from_index(
                                  np.array([[[n, n]] for n in layers])), **common)
    cell = ([[[n, n]] for n in layers] if path == 'conical'
            else [[[n, n], [n, n]] for n in layers])
    return wavebit.rcwa(phi=0.0, psi=(np.pi / 2 if tetm == 'te' else 0.0), truncation_order=[truncation_order, truncation_order],
                          period=[WL * 2, WL * 2],
                          structure=eps_from_index(np.array(cell)), **common)


def _R(solver):
    return float(np.sum(np.asarray(solver.solve().R)).real)


INCIDENCE = [(p, t) for p in ('te', 'tm') for t in (0.0, 0.4)]
IDS = [f'{p}_theta{t}' for p, t in INCIDENCE]


# ----------------------------------------------------------------------
# 1. the efficiency, against an absolute oracle
# ----------------------------------------------------------------------
@pytest.mark.parametrize('path', ['1d', 'conical', '2d'])
@pytest.mark.parametrize('tetm,theta', INCIDENCE, ids=IDS)
def test_stacked_same_material_matches_the_analytic_stack(tetm, theta, path):
    """Two identical layers on a different substrate: the upper one must still count.

    This is the arrangement that was wrong, and an Airy recursion over the same indices
    settles it without reference to any solver. All three transfer paths carried their own
    copy of the short-circuit, and the conical one additionally built its transfer in a
    different mode basis than its own general branch used, so each needs saying separately.
    """
    ns, ds = STACKED
    solver = _solver(ns, ds, tetm=tetm, theta=theta, path=path)
    assert _R(solver) == pytest.approx(_airy_R(ns, ds, theta, tetm), abs=1e-12)


@pytest.mark.parametrize('tetm,theta', INCIDENCE, ids=IDS)
def test_merging_same_material_layers_is_the_same_structure(tetm, theta):
    """Splitting a slab in two must not change anything -- the split is not physical."""
    ns, ds = STACKED
    whole = _solver([1.0, 1.5, 2.0], [sum(ds)], tetm=tetm, theta=theta)
    split = _solver(ns, ds, tetm=tetm, theta=theta)
    assert _R(split) == pytest.approx(_R(whole), abs=1e-12)


@pytest.mark.parametrize('d', [1.0, 20.0, 2000.0], ids=['thin', 'typical', 'thick'])
def test_absorbing_layer_on_its_own_substrate_at_any_thickness(d):
    """Au on Au: the case the short-circuit was originally added to survive.

    Under the conjugated formulation this made the interface matrix singular and the solve
    blew up. It is now an ordinary matched interface -- the layer is optically indistinguishable
    from the substrate, so any thickness must give the bare air/gold Fresnel reflectance.
    """
    want = _airy_R([1.0, AU, AU], [d], 0.3, 1)
    assert _R(_solver([1.0, AU, AU], [d], tetm='tm', theta=0.3)) == pytest.approx(want, abs=1e-12)


def test_a_matched_interface_is_well_conditioned():
    """Directly: the quantity that used to be singular here is the identity.

    Pins the reason the short-circuit is removable rather than merely removed. If a future
    convention change flips ``G``'s sign back, ``a`` returns to ~0 and this fails loudly
    instead of the solve quietly producing garbage.
    """
    import array_api_compat.numpy as xnp
    from wavebit._core.conventions import boundary_kz
    from wavebit._core.solve_1d import _diag, layer_modes_1d

    truncation_order, theta = 2, 0.3
    n_orders = 2 * truncation_order + 1
    kx = np.asarray([np.sin(theta) + m * 0.5 for m in range(-truncation_order, truncation_order + 1)], dtype=complex)
    I = np.eye(n_orders, dtype=complex)

    eps = complex(AU) ** 2                      # layer and substrate are both gold
    W, V, _ = layer_modes_1d(xnp, 1, kx, I, eps * I, eps * I, (1 / eps) * I, complex)
    F = I
    G = -1j * _diag(xnp, boundary_kz(xnp, eps, kx ** 2) / eps)          # substrate seed, TM

    a = 0.5 * (np.linalg.inv(W) @ F + np.linalg.inv(V) @ G)
    assert np.abs(a - I).max() < 1e-12, np.abs(a - I).max()
    assert np.linalg.cond(a) == pytest.approx(1.0, abs=1e-9)


# ----------------------------------------------------------------------
# 2. the field below the layer
# ----------------------------------------------------------------------
RES = 10
FIELD_STACK = ([1.0, 1.5, 1.5, 2.0, 1.0], [30.0, 30.0, 40.0])


def _field(ns, ds, *, tetm='te', theta=0.2, truncation_order=2):
    solver = _solver(ns, ds, tetm=tetm, theta=theta, truncation_order=truncation_order, answers='field')
    solver.solve()
    return np.asarray(solver.reconstruct_field(sample_x=RES, sample_y=1, sample_z=RES))


def test_field_below_a_same_material_layer_is_not_zero():
    """In the original top-down case the last two layers came back identically zero.

    A stack that reflected and transmitted normally the whole time, with no field in most
    of it.
    """
    f = _field(*FIELD_STACK).reshape(3, RES, -1)
    for i in range(3):
        assert np.abs(f[i]).max() > 1e-3, f'layer {i} is empty'


@pytest.mark.parametrize('tetm', ['te', 'tm'], ids=['te', 'tm'])
def test_field_matches_the_merged_stack(tetm):
    """And it is not merely non-zero: it is the field the merged stack has.

    Writing the two same-material layers as one of their combined thickness is a different
    *input* describing the same physical stack, so this pins that the thickness survived --
    the half of the defect that the efficiency tests above also see, but here in the
    reconstruction, which is where it was total rather than partial.
    """
    ns, ds = FIELD_STACK
    got = _field(ns, ds, tetm=tetm).reshape(3, RES, -1)
    merged = _field([1.0, 1.5, 2.0, 1.0], [60.0, 40.0], tetm=tetm).reshape(2, RES, -1)
    # Same z sampling per layer, so merged has 2 blocks where this has 3; the bottom (n=2.0)
    # block is the one both describe identically.
    assert np.allclose(got[0], merged[0], atol=1e-12, rtol=0)
