"""Known matrices and analytic controls keep the S1 measuring instrument honest."""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))
from tools.repository_paths import repository_path
import importlib.util

import numpy as np
import scipy.sparse as sp


_path = repository_path('studies/fdtd/subpixel_tilted_tensor/fdtd_subpixel_tilted_tensor.py')
_spec = importlib.util.spec_from_file_location('subpixel_tensor_study', _path)
study = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(study)

# Measured 2026-09-22 on the matrices below: spectral endpoint absolute error 3.3e-13,
# complex eigenvalue error 1.1e-16, equation residual 6.6e-17. The endpoint sets the margin.
EIGEN_ATOL = 1e-12
# 200 points, seed 73, translated by 1, 7 and 100 periods: worst fill difference 2.57e-13.
FILL_ATOL = 1e-12
# Exact order-2 samples fitted with residual 1.78e-15; retain a 56x round-off margin.
FIT_ATOL = 1e-13


def test_spectrum_uses_algebraic_endpoints_not_the_neighbourhood_of_zero():
    # The removed sigma=0/LM implementation read -0.01 for min/max here, rather than -1.
    a = sp.diags(np.r_[-100., np.arange(1., 101.)]).astype(complex).tocsc()
    lo, hi = study.spectral_bounds(a)
    np.testing.assert_allclose((lo, hi), (-100, 100), rtol=0, atol=EIGEN_ATOL)


def test_a_nonhermitian_mode_keeps_its_imaginary_part_and_its_residual():
    values = np.arange(1., 13.).astype(complex)
    values[1] += 0.3j
    a = sp.diags(values).tocsc()
    value, gap, residual = study.eigenpair(a, sp.eye(12, format='csc', dtype=complex), 2.)
    assert abs(value - values[1]) < EIGEN_ATOL
    assert abs(gap - abs(1 - values[1])) < EIGEN_ATOL
    assert residual < EIGEN_ATOL


def test_layer_fill_is_periodic_far_beyond_the_old_image_window():
    x, y = np.random.default_rng(73).uniform(-3, 3, (2, 200))
    normal, h = (0.8, 0.6), 1 / 32
    reference = study.lamellar_fill(x, y, h, normal)
    assert reference.min() >= 0 and reference.max() <= 1
    assert np.any((reference > 0) & (reference < 1)), 'include cut voxels'
    for periods in (1, 7, 100):
        translated = study.lamellar_fill(x + periods / normal[0], y, h, normal)
        np.testing.assert_allclose(translated, reference, rtol=0, atol=FILL_ATOL)


def test_a_fitted_order_also_reports_the_scatter_it_cannot_explain():
    rows = [dict(cells_actual=c, error=2 / c ** 2) for c in (16, 32, 48)]
    exact = study.fit(rows)
    assert abs(exact['order'] - 2) < FIT_ATOL and exact['scatter'] < FIT_ATOL
    assert exact['monotone']
    rows[1]['error'] *= 4
    irregular = study.fit(rows)
    # Measured scatter 1.468: this is an error larger than 100%, not a round-off budget.
    assert irregular['scatter'] > 1 and not irregular['monotone']
