"""``fourier_coefficients`` and ``factorization`` are two independent axes.

They are orthogonal choices — how a layer's Fourier coefficients are obtained, and which rule
closes the products of discontinuous functions — so on the raster path all four of their
pairings are reachable. That is the point of keeping them as two keywords: a single knob
covering both can only ever name a subset, and the pairing it drops is
``('integrated', 'nvf')`` — the one a shape/OCD fit of a tilted sidewall wants, being the
staircase-free permittivity series *and* a factorization that handles boundaries not aligned
with x or y.

So this file covers the two axes as a pair — the default, that neither keyword may move the
other, and that fourth pairing. The per-strategy numerics live in
``test_nvf.py`` / ``test_factorization_core.py``; the frozen-geometry / tracing coverage of
all four points in ``test_frozen_geometry.py``; and how each axis resolves against a modeling
path that implements only one of its values in
``test_factorization_by_modeling_path.py`` / ``test_fourier_coefficients_by_modeling_path.py``.
"""
import numpy as np
import pytest

import wavebit

# 2D so the NVF projector is not degenerate: a lossless dielectric rod, which is also the
# structure whose energy conservation the NVF path exists to improve.
N = 48
_x = (np.arange(N) + 0.5) / N - 0.5
_X, _Y = np.meshgrid(_x, _x)
ROD = np.ones((1, N, N))
ROD[0, _X ** 2 + _Y ** 2 <= 0.3 ** 2] = 6.25       # permittivity: 2.5**2

CFG_2D = dict(backend=0, structure=ROD, top=1.0, bot=1.0, theta=0.0, phi=0.0, psi=0.0,
              max_order=[3, 3], period=[1.0, 1.0], wavelength=1.1, thickness=[0.3])

# A 1D grating: no interface along y, so every normal points along x. A *constant* normal
# collapses the NVF projector onto Li's rules exactly (C_xx = I, C_xy = C_yy = 0).
CFG_1D = dict(backend=0, structure=np.array([[[9., 9., 1., 1., 9., 1., 1., 1.]]]), tetm='tm',
              period=[0.77], thickness=[0.2], theta=0.25, wavelength=0.777,
              top=1.0, bot=2.25, max_order=[10, 0])


def _totals(cfg, **strategy):
    res = wavebit.rcwa(**cfg, **strategy).solve()
    return (float(np.sum(np.asarray(res.R))), float(np.sum(np.asarray(res.T))))


# ------------------------------------------------------------------ the default

def test_the_default_is_sampled_li():
    """Pinned deliberately. See test_gradients.py for *why* it cannot move.

    Only sampled coefficients carry a gradient to every raster cell, which is what topology
    optimization differentiates. This asserts the two attributes rather than the behaviour so
    that a change of default fails here, loudly, rather than in a gradient test nobody reads.

    ``CFG_2D`` is a raster, and the ``factorization`` half of this is only a raster fact: the
    other two modeling paths implement one rule apiece and resolve the default to it. See
    ``test_factorization_by_modeling_path.py``.
    """
    solver = wavebit.rcwa(**CFG_2D)
    assert solver.fourier_coefficients == 'sampled'
    assert solver.factorization == 'li'


def test_the_default_is_what_spelling_it_out_gives():
    assert (_totals(CFG_2D) ==
            _totals(CFG_2D, fourier_coefficients='sampled', factorization='li'))


# ------------------------------------------------------------------- validation

@pytest.mark.parametrize('name,code', [('numpy', 0), ('jax', 1), ('torch', 2)])
def test_the_removed_fourier_type_alias_is_not_silently_swallowed(name, code):
    """The removed single-knob spelling has to *fail*, not be ignored.

    A keyword the constructor quietly drops is the one failure mode worse than removing it:
    a script still carrying the old spelling would run the default strategy while believing it
    had asked for another, and nothing in the result would say so. Each backend spells its own
    constructor signature out, so each has to refuse on its own -- a ``**kwargs`` added to any
    one of them would swallow it there and nowhere else.
    """
    if name != 'numpy':
        pytest.importorskip(name)
    with pytest.raises(TypeError, match='fourier_type'):
        wavebit.rcwa(**dict(CFG_2D, backend=code), fourier_type=2)


@pytest.mark.parametrize('keyword,bad', [('fourier_coefficients', 'discrete'),
                                         ('fourier_coefficients', 0),
                                         ('factorization', 'lifting'),
                                         ('factorization', 2)])
def test_an_invalid_value_is_rejected_at_the_keyword_that_took_it(keyword, bad):
    """The error names the keyword, so a typo does not surface as a wrong-physics run."""
    with pytest.raises(ValueError, match=keyword):
        wavebit.rcwa(**CFG_2D, **{keyword: bad})


@pytest.mark.parametrize('coefficients', ['SAMPLED', 'Integrated'])
def test_values_are_case_insensitive(coefficients):
    solver = wavebit.rcwa(**CFG_2D, fourier_coefficients=coefficients)
    assert solver.fourier_coefficients == coefficients.lower()


def test_the_axes_are_independent():
    """Neither keyword may quietly change the other — that entanglement was the bug."""
    for coefficients in ('sampled', 'integrated'):
        for factorization in ('li', 'nvf'):
            solver = wavebit.rcwa(**CFG_2D, fourier_coefficients=coefficients,
                                    factorization=factorization)
            assert (solver.fourier_coefficients, solver.factorization) == (coefficients,
                                                                          factorization)


# -------------------------------------------------- ('integrated', 'nvf'): the new pairing

def test_integrated_nvf_conserves_energy_on_a_lossless_rod():
    """The structural correctness check for any NVF variant.

    A lossless dielectric rod at normal incidence must have ``R + T == 1``. This is what
    caught the wrong ``epz`` convention and the non-Hermitian projector product in the
    sampled NVF path, and it is the same evidence the new pairing has to produce.
    """
    r, t = _totals(CFG_2D, fourier_coefficients='integrated', factorization='nvf')
    assert abs(r + t - 1.0) < 1e-6, (r, t)


def test_integrated_nvf_collapses_onto_integrated_li_on_a_1d_grating():
    """NVF is a superset of Li, not an alternative — on both coefficient paths.

    A one-cell axis forces a constant normal, and a constant normal makes the projector
    ``C_xx = I``, ``C_xy = C_yy = 0``, i.e. ``eps_xx = <1/eps>^-1`` and ``eps_yy = <eps>``.
    So this asserts equality, not closeness: anything else means the new pairing is assembling
    the projector differently from the sampled one.
    """
    nvf = _totals(CFG_1D, fourier_coefficients='integrated', factorization='nvf')
    li = _totals(CFG_1D, fourier_coefficients='integrated', factorization='li')
    assert nvf == pytest.approx(li, abs=1e-12)


def test_integrated_nvf_agrees_with_sampled_nvf_on_the_same_structure():
    """The two coefficient paths describe the same geometry, so they must converge together.

    They are not bit-identical — one integrates the permittivity over the compressed blocks
    and the other samples it off the raster — but a disagreement beyond the discretization
    difference would mean one of them is factorizing something else.
    """
    integrated = _totals(CFG_2D, fourier_coefficients='integrated', factorization='nvf')
    sampled = _totals(CFG_2D, fourier_coefficients='sampled', factorization='nvf')
    assert integrated == pytest.approx(sampled, abs=2e-3)


@pytest.mark.parametrize('name', ['jax', 'torch'])
def test_integrated_nvf_runs_on_every_backend(name):
    """The core builders are backend-agnostic; the new pairing must not be an exception."""
    pytest.importorskip(name)
    backend = 1 if name == 'jax' else 2
    strategy = dict(fourier_coefficients='integrated', factorization='nvf')

    ref = _totals(CFG_2D, **strategy)
    got = _totals(dict(CFG_2D, backend=backend, structure=_cast(name, ROD)), **strategy)
    assert got == pytest.approx(ref, abs=1e-10)


def _cast(name, arr):
    if name == 'torch':
        import torch
        return torch.tensor(arr, dtype=torch.float64)
    import jax.numpy as jnp
    return jnp.asarray(arr)
