# Conforming NVF on equivalent lattice bases

Written: 2026-09-09

Targets:
- `src/wavebit/_core/conforming/__init__.py`
- `tests/test_conforming_lattice_rebasing.py`

`ConformingNVF` accepts nonorthogonal basis vectors when they describe an
orthogonal point lattice. This removes a representation restriction without
changing the physical rectangles or the corrector equations. Genuinely
nonorthogonal point lattices remain outside the rectangular cell builder.
The [manual](guide/conforming-nvf.md) gives the input and replay contract.

For example,

```text
A = [[700, 2100], [0, 430]]
T = [[1, -3], [0, 1]]
C = A T = diag(700, 430)
```

Both A and C generate exactly the same periodic translations. Treating A
as an orthogonal cell with side lengths equal to its vector norms would
change that lattice. Applying a continuous shear to the shape coordinates
would instead turn the rectangles into parallelograms. Neither operation
is part of this extension.

## Frozen integer reduction and live geometry

Planning uses the existing lattice module's Gauss/Lagrange reduction to
obtain the integer unimodular matrix T. It then applies the same
orthogonality and positive-period checks as the rectangular adapter. An
already orthogonal basis keeps the identity reduction. A remaining
nonorthogonal reduced basis is refused.

The immutable plan now stores T in addition to layer ordering and each
layer's tiling axis. Replay forms `C = A T` from the live lattice; it does
not round, reduce or re-plan a traced lattice. The plan validates that T
contains integers and has determinant one. Periods, common rotation,
positions, dimensions and independent material loss remain live. Derivatives
are within the family for which C stays orthogonal. An independent
continuous shear leaves that family, raising eagerly or producing NaNs
under tracing.

Rectangle lengths remain physical lengths, and angles are measured in the
laboratory frame. Their edges align with C, not necessarily A. The existing
per-layer choice between `(c1,c2)` and `(c2,-c1)`, material profiles,
contact continuation and aspect metric apply without modification.

## Preserving the caller's shape origin

Shape centers are offsets from half the sum of the supplied lattice
vectors. Therefore an input center d describes the absolute center

```text
x_center = A [1,1]^T / 2 + d
```

The corresponding offset in the construction cell is
`d_C = d + (A-C)[1,1]^T/2`. The adapter applies that shift before the
orthogonal coordinate transform, and then uses the ordinary cell-corner
Fourier phase. This preserves the shape's absolute position even when the
two fundamental parallelograms have centers differing by a half-period.

This shift is distinct from a user's intentional geometry change. When
the user changes A but leaves d unchanged, the shape stays attached to the
new supplied cell center, just as in the ordinary shape API. To keep its
absolute position fixed while replacing A by Anew, change the user offset
by `(A-Anew)[1,1]^T/2`.

## Preserving the exact finite Fourier set

For an original integer order k, the same physical reciprocal vector is

```text
q = 2 pi A^(-T) k = 2 pi C^(-T) k_C
k_C = T^T k
```

The original rectangular order window usually becomes a nonrectangular
set in C coordinates. The adapter projects enough Fourier coefficients to
cover all differences between those transformed orders, but keeps exactly
the original number of retained modes and their ordering. For a layer
using the second reduced axis, the further map is `(m,n) -> (n,-m)`.

The selection occurs in Q, G and Z before
`E = Q^(-H) G Q^(-1)` and `Z^(-1)`. Extracting entries from an already
closed larger rectangular matrix would be a different finite construction:
matrix inversion and selecting a subset do not generally commute.

Projection cells are cached by the required Fourier envelope order. This
changes the number of coefficient modes, not the corrector mesh or its
unknown count. Large integer shears can make projection more expensive.
The consuming engine always receives the original A and truncation order,
so its wavevectors and output order labels retain their original meaning.

## Independent controls and remaining scope

Controls include analytic Fourier integrals using the original physical
shape descriptions, and independent inverse-rule contact tensors in both
layer directions. A separate positive-gap reference identifies matching
orders by their physical reciprocal vectors in an orthogonal cell and
selects Q/G/Z before closure. Special order sets that retain the same
physical modes also reproduce the complete engine result across lattice
spellings. Mixed horizontal, uniform and vertical absorbing contact stacks
are checked on NumPy, JAX and Torch engines.

Fifteen geometry, storage-material, loss and period parameters are compared
against two finite-difference steps. Contact-opening derivatives are
compared between equivalent representations of the same physical modes.
Malformed reductions, a changed frozen reduction and genuine nonorthogonal
lattices are checked eagerly and under traced replay.

Commands and measurements are in
`studies/results/nvf_lattice_rebasing_validation.log`: 52 rebasing cases,
203 existing public construction/material cases and 35 documentation cases
passed, for 290 distinct checks. All three manual examples, lint and the
strict documentation build also passed. This is a geometry
representation and finite-mode integration qualification, not an optical
convergence certificate. No cell, passivity, contact or optical tolerance
is changed. General nonorthogonal correctors, curved geometry, nonpositive
real permittivities and automatic projector selection remain separate work.
