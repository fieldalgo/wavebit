"""``compression_indices`` / ``compress_cell`` against what they are specified to return.

A raster layer is cut into piecewise-constant blocks before the integrated path transforms it:
the last column of each run of identical neighbouring columns is kept, then the same for rows on
the column-reduced raster, and each kept index ``i`` of an axis with ``N`` pixels places a
right-hand edge at ``(1.0 / N) * (i + 1)`` -- bit for bit what
``_check_compression_still_fits`` recomputes when a frozen record is replayed. Both functions
were rewritten on 2026-09-25 from that specification (``LICENSE`` has the history); these are
the checks that came with the rewrite.
"""
import numpy as np

import array_api_compat.numpy as xnp

from wavebit.rcwa._engine.factorization import _check_compression_still_fits, compress_cell, compression_indices


def _complex_raster(shape, seed):
    """Random complex permittivity, kept away from zero so every rule's inverses exist."""
    rng = np.random.default_rng(seed)
    return rng.uniform(1.0, 6.0, shape) + 1j * rng.uniform(-1.0, 1.0, shape)


def test_compression_finds_the_runs_of_repeated_rows_and_columns():
    base = _complex_raster((3, 4), seed=11).real
    base[:, 2] = base[:, 1]                          # two equal neighbouring columns: one block
    row_counts, col_counts = [2, 1, 3], [1, 2, 2, 3]
    layer = np.repeat(np.repeat(base, row_counts, axis=0), col_counts, axis=1)     # 6 x 8

    row_idx, col_idx, x, y = compression_indices(layer)
    assert row_idx == [1, 2, 5]
    assert col_idx == [0, 4, 7]                      # columns 1..4 are one run
    assert all(type(i) is int for i in row_idx + col_idx)

    step_v, step_u = 1.0 / np.array(layer.shape, dtype=np.float64)
    assert x.dtype == np.float64 and x.shape == (3, 1)
    assert np.array_equal(x, np.array([step_u * (c + 1) for c in col_idx]).reshape((-1, 1)))
    assert np.array_equal(y, np.array([step_v * (r + 1) for r in row_idx]).reshape((-1, 1)))
    _check_compression_still_fits(layer, (row_idx, col_idx, x, y), 0)

    comp = np.asarray(compress_cell(xnp, layer, row_idx, col_idx))
    assert np.array_equal(comp, base[:, [0, 1, 3]])
    assert np.array_equal(comp, layer[row_idx][:, col_idx])


def test_compression_compares_rows_on_the_column_reduced_raster():
    """Rows are compared on ``cell[:, col_idx]``, and that still sees every row that steps.

    Every column is identical to the kept last column of its run, so a difference between two
    rows in a merged-away column is also a difference in the kept one. A uniform raster is one
    block, and so is a single row or column.
    """
    layer = np.array([[1.0, 1.0, 2.0],
                      [1.0, 1.0, 2.0],
                      [3.0, 3.0, 2.0]])
    row_idx, col_idx, _, _ = compression_indices(layer)
    assert col_idx == [1, 2]
    assert row_idx == [1, 2]

    uniform = np.full((4, 5), 2.5)
    assert compression_indices(uniform)[:2] == ([3], [4])
    assert compression_indices(np.full((1, 7), 2.5))[:2] == ([0], [6])
    assert compression_indices(np.full((5, 1), 2.5))[:2] == ([4], [0])
