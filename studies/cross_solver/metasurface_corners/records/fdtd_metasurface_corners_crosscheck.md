# Corner exponents: two independent cross-checks

Written: 2026-10-03

Targets: `studies/fdtd_metasurface_corners.py` (`corner_exponent`, `CORNERS`).

The study computes each corner's exponent `nu` -- the field near the vertex goes as `r^(nu - 1)`
-- from a transfer-matrix trace condition. Two derivations were made independently of it and
of each other; their scripts and logs are in `fdtd_metasurface_corners_crosscheck/` (the paper
text they extracted to check the citation is not kept). The two shared a scratch folder: the
parity derivation's first write replaced the eigenproblem's `vertex3d.py` at the top level with
a stub; the eigenproblem's script was restored as
`fdtd_metasurface_corners_crosscheck/claude_a/vertex3d.py`, which reproduces its
`vertex3d_coarse.log` apart from timings, and the stub is not kept. The top-level scripts and
logs are earlier copies of `claude_a/`'s. One unused import was removed from
`fdtd_metasurface_corners_crosscheck/claude_b/limits_and_gap.py` for the repository's linter; nothing it ran changed. A third agent re-derived every value to 40 digits and
found the three sets agreeing to about 1e-7 (rounding), the study's trace minus 2 about 1e-40 at
each root, and exactly one root in (0, 1) per corner.

| Corner (counter-clockwise sectors) | Trace (study) | Parity and nodal system | Angular eigenproblem |
| --- | --- | --- | --- |
| A: 90 air in 270 of eps 12 | 0.7219000 | 0.7219 | 0.7219 |
| B: 90 of eps 12 in 270 air | 0.7219000 | 0.7219 | 0.7219 |
| C: 90 of eps 5.8 in air | 0.7703632 | 0.770363 | 0.770363 |
| D: 90 of eps 4 in air | 0.8060266 | 0.806027 | 0.806027 |
| E: 90 eps 12 / 90 air / 180 eps 2.25 | 0.6845362 | 0.684536 | 0.684536 |
| F: 90 eps 5.8 / 90 air / 180 eps 2.13 | 0.7509614 | 0.750961 | 0.750961 |
| G: 90 eps 4 / 90 air / 180 eps 2.25 | 0.7954939 | 0.795494 | 0.795494 |

**Parity and nodal system** (`claude_b/`): even and odd families about each symmetry line, and a
cyclic nodal system for the three-media junctions, giving the closed forms
`cos(nu pi/2) = |e1 - e2| / (2 (e1 + e2))` for a right angle and
`cos^2(nu pi/2) = e3 (e1 - e2)^2 / (2 (e1 + e2)(e1 + e3)(e2 + e3))` for the 90/90/180 junction;
checked against the infinite-contrast (Dirichlet wedge, `pi` over the exterior angle) and
weak-contrast limits, the reductions of the junction to two media, and a P1 finite-element
solve with Richardson extrapolation, all within 5e-10.

**Angular eigenproblem** (`claude_a/` and the top level): `-(eps f')' = nu^2 eps f` on the
periodic circle, by hierarchical spectral elements (exponential convergence) and P1 elements
(order 2.000), and separately a finite-volume solve of `div(eps grad phi) = 0` on a graded
annulus with the exponents read from the solution by a matrix pencil; within 2e-9.

**Farjadpour et al. (2006)** quote `p ~ 0.702` for the corners of tilted-square air holes in
epsilon 12 -- corner A. Its exact exponent is `(2/pi) arccos(11/26) = 0.721900`; neither
derivation could produce 0.702 from that geometry (it is the root at contrast 19.5, or at a
corner of 77.3 or 26.3 degrees at contrast 12). The paper's measured order, 1.4, agrees with
`2 x 0.7219 = 1.444` within its fit.

**The order claim.** Both derivations agree that, once smoothing has removed the first-order
error of flat interfaces, a corner's pixel-sized neighbourhood leaves an error of
`dx^(2 nu)` in an integrated quantity -- an eigenfrequency, or a scattered amplitude, whose
adjoint field is singular at the same corner -- as an envelope that scatters with the corner's
sub-pixel placement; observed slopes approach `2 nu`, from above or from below. A quasi-static
demonstration (`fdtd_metasurface_corners_crosscheck/claude_a/quasistatic_demo.py`, with
`aligned_1024.py` and `offsets_rms.py` beside it) gave, for an
epsilon-12 square on aligned grids, fitted orders 1.426 to 1.440 at 128 to 512 cells rising to
1.442 at 1024 -- below 1.444 and approaching it -- and over 16 sub-pixel offsets local slopes of
the RMS error of 1.71, 1.59, 1.53 and 1.29 (of the maximum 1.49, 1.55, 1.66, 1.28): consistent
with 1.44, but pinning it only to about 0.2. Both are static problems; no wave solver was
checked. In 3D an edge acts as the 2D corner of its cross-section; a vertex has a
smaller exponent (`lambda` 0.58 at an epsilon-12 box's top corner, the potential going as
`R^lambda`; 0.57 at a pillar's foot on glass) but, being a point, contributes `dx^(2 lambda + 1)`,
so edges set the rate. The two vertex codes (an octahedral and a cube-sphere mesh) agree within
2e-5, and the Dirichlet limit reproduces the Fichera corner, 0.4542.
