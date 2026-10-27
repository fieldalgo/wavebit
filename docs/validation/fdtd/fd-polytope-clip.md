# Two smoothed surfaces in one voxel: the polytope clip (stage 4, C10)

Written: 2026-10-27

Targets: `src/wavebit/fdtd/_prepare/checks.py`, `src/wavebit/fdtd/_prepare/coefficients.py`,
`src/wavebit/fdtd/_prepare/constitutive.py`, `src/wavebit/fdtd/_prepare/junction.py`,
`src/wavebit/fdtd/_shapes/_cut.py`, `src/wavebit/fdtd/_shapes/_bodies.py`,
`studies/fdtd/subpixel_media_census/fdtd_subpixel_media_census.py`.

## Status

This page is written in two parts. The first part, **Registered acceptance**, was committed
before any C10 code was written and before any number below was computed. It is not edited
after the measurements; results are recorded in later sections that cite it.

## Registered acceptance (2026-10-27, before any C10 measurement)

### What C10 is to answer

Door 5 of `checks.check_smoothed` refuses two smoothed bodies whose surfaces pass through one
voxel, because the two-phase rule `coefficients.mix` composes wrongly there (73% on two half
spaces of one material, up to 122% on one sphere declared twice). Study S7
(`studies/fdtd/subpixel_media_census/`) found such voxels rare: none for single bodies, 1-4%
of cut voxels for a pillar flush on a substrate spelled as a second body, and for a dimer of
two different spheres at a gap of 0.05 diameters (at 20 cells per unit only).

C10 replaces the refusal by an answer **only where every quantity the answer needs is computed
exactly or by a rule already used for one body**, and keeps the refusal, by name, everywhere
else. The planned construction:

1. **Per-medium fills.** For the pair `A` (listed first) and `B` (listed later, so it wins
   where they overlap), the voxel holds `B` with fill `F_B`, `A` with fill `F_A - S` and the
   ground (what the voxel held before `A` was painted) with fill `1 - F_A - F_B + S`, where
   `F_A`, `F_B` are the bodies' existing exact fills and `S` is the volume they share in the
   voxel. `S` is admitted only where it is exact:
   * `S = 0` where the bodies are proved disjoint (per voxel by the existing exact test of
     their supporting half spaces, `_bodies._no_common_volume`, for convex bodies; or globally,
     for a sphere whose centre is at least its radius from the partner's exact signed distance);
   * `S` = the exact two-plane cut (`_cut.two_plane`, already validated against an independent
     ConvexHull oracle) where both bodies are proved flat in the voxel;
   * `S` = the partner's exact volume over an axis-aligned sub-box, where one member is flat
     in the voxel with an axis-aligned normal (an unturned Box face), so that the voxel on that
     member's side of its face is itself an axis-aligned box.
2. **Interface moment.** `w_c` is the combined second moment of the interfaces: each member's
   own area and moment (exact for polygonal bodies, the tangent plane's for curved ones, as for
   one body), with the part of `A`'s surface buried inside `B` removed.
3. **Rule.** With `<1/eps> = sum_m f_m/eps_m` and `<eps> = sum_m f_m eps_m` over the media
   present, each component reads `1/eps_c = w_c <1/eps> + (1 - w_c)/<eps>`, and the common-node
   tensor is `1/<eps> I + (<1/eps> - 1/<eps>) P` with `P` the area-weighted sum of the members'
   nearest-surface `n n^T` (the existing node proxy). For parallel planar interfaces this is
   the exact laminate (normal component harmonic, tangential arithmetic over all layers). With
   one medium absent it is the existing two-phase rule.
4. **Kept refused by name**: flush joints (two coincident back-to-back faces; that is the
   junction's question, `Structure(junction=)`), any voxel whose shared volume is not in one of
   the exact cases above, overlapping bodies of one material (a union, whose CSG spelling is not
   public), a third structure reaching a written voxel other than a whole earlier ground,
   magnetic, conductive, dispersive or anisotropic media, `discretisation='stencil'`, and
   traced geometry (`range=`) on a member.

### Gate 1: geometry and rule, unit tests (a new test module `test_fd_clip.py` under `tests/fdtd/`)

* **G1a, polyhedral fills.** For pairs of unturned and turned Boxes and convex Prisms in each
  admitted case (disjoint, both flat, one axis-aligned flat member), every medium fill of every
  written voxel against an independent oracle: the voxel and each body written as half spaces,
  intersected by `scipy.spatial.HalfspaceIntersection` and measured by `ConvexHull`, sharing
  no line with `_cut` or `_exact`. **Registered: absolute error at most 2e-12 of a voxel.**
  (The two-plane cut measured 9.8e-13 against this kind of oracle; 2e-12 rounds that up.)
* **G1b, fills with a sphere.** A sphere disjoint from a partner, touching a substrate, and
  sunk into a substrate: every medium fill against adaptive quadrature in the style of
  `tests/fdtd/test_fd_shapes.py`, `scipy.integrate.quad` along the substrate normal of exactly
  clipped cross-section areas, broken at the cross-sections' kinks. **Registered: at most 1e-10
  of a voxel.**
* **G1c, polyhedral interface moments.** The exposed interface area and its three per-component
  moments against the polytope oracle's facets (a member's facets inside the voxel, less the
  facets of the shared polytope on that member's plane). **Registered: at most 2e-9 of `dx^2`**
  (the two-plane cut's measured worst, 1.3e-9 on nearly anti-parallel faces, rounded up).
* **G1d, rule.** (i) Three parallel planar layers in one voxel: each lowered component of `eps`
  against the laminate closed form computed from the layer thicknesses alone, **relative 1e-12**.
  (ii) With one medium's fill exactly zero, the clip rule's diagonal equals `coefficients.mix`,
  and its node tensor, cross term and arithmetic mean equal the single-body expressions in
  `constitutive.lower`, **bit for bit** (`numpy.array_equal`).
* **G1e, refusals.** Each case kept refused in "What C10 is to answer" fires, by name.
* **Mutations.** Every new test is shown to fail under a deliberate edit of the product code,
  which is then restored exactly; an import error does not count.

### Gate 2: convergence (new study `studies/fdtd/subpixel_polytope_clip/`)

* **Arm L, exact reference, clip active at every grid.** A planar stack along z of two smoothed,
  unturned Boxes spanning the transverse cell: `eps` 6 on `z in [-0.30, 0]`, a vacuum gap of
  0.003, `eps` 2.25 on `z in [0.003, 0.203]`, in vacuum. The gap is thinner than every grid
  used, so its two faces share voxels at every grid. Two polarisations: normal incidence (the
  tangential, arithmetic part of the rule) and p polarisation at 30 degrees in the x-z plane
  (the normal, harmonic part), the latter with `Axis1D` transverse faces. Reflectance of the
  zeroth order at 21 wavelengths in `[0.8, 1.2]` against the continuum transfer-matrix
  reflectance of the same four-interface stack. Grids 20, 28, 40, 56, 80, 112 and 160 cells per
  unit length. Error at a wavelength: `|R_fdtd - R_exact|`; band error: the root mean square
  over the 21 wavelengths. **Pass: for each polarisation, the least-squares log-log order of the
  band error over all seven grids is at least 1.8.** Reported beside it: the fit's scatter
  (`exp(max |log residual|) - 1`), the per-wavelength orders, the per-grid errors, and the
  count of clip-written voxels at each grid. Control arm L0: the same stack with a gap of 0.2
  (no shared voxel) is run on the same grids and reported the same way; it is an instrument
  check and not part of the pass.
* **Arm D, the S7 dimer.** S7's geometry: spheres of radius 0.25 centred at
  `(-0.2625, 0, 0.021)` and `(0.2625, 0, 0.021)`, a gap of 0.05 diameters. S7 named two metals;
  **a metal on a curved smoothed body is refused (door 1; C8 smooths planar films only), so
  the materials are two lossless dielectrics, `eps` 12 and 2.25, in vacuum**, a substitution
  recorded here and not a choice made after measurement. A TFSF plane wave along +z, polarised
  along the dimer axis (x), scattering cross-section from a closed flux box (the instrument of
  `studies/fdtd/subpixel_mie_sphere/`) at 11 wavelengths in `[1.0, 2.0]`. Grids 20, 24, 28, 32,
  40 and 56 cells per unit; reference: 80 cells per unit. Per wavelength and grid: relative
  error against the reference; band error: mean over wavelengths. Reported: fitted order and
  scatter over the six grids, per-wavelength orders, the per-grid table, and the clip-written
  voxel count at each grid. **Pass: (a) every grid plans without refusal and runs with the
  stability diagnostic finite; (b) at every grid where the clip writes, the band error lies
  within a factor of 2 of the power law fitted through the grids where it does not write,
  evaluated at that grid (the clip-active grids are not outliers).** If (b) fails, voxels
  answered between two curved surfaces are refused by name and the failure is recorded.
* **Arm P, the S7 flush pillar.** A cylinder flush on a smoothed Box substrate is the
  junction's (`junction='split'`), not C10's. **Pass: it plans with no clip record and its
  lowered coefficients are unchanged** (Gate 3).

### Gate 3: every existing two-media scene is unchanged

* The four frozen stage goldens reproduce in check mode, nothing written
  (`tests/references/interface/freeze_fd_stage0.py` to `freeze_fd_stage3.py`). `tests/references/fd_golden_stage*.npz`
  are not regenerated; **if any moves, the work stops and is reported.**
* Thirteen existing two-media smoothed scenes (sphere, magnetic sphere, unturned and turned Box,
  Cylinder, hexagonal Prism, core-shell pair, a dimer far apart, cylinder and Box pillars under
  `split` and `staircase`, a stencil sphere, a high-contrast sphere): plan skeletons and every
  lowered parameter array **byte-identical** before and after C10, from a dump taken on
  unmodified `main` (183baf37) before any C10 edit.

### Gate 4: backends and derivatives

* Lowered coefficients (`be`, constitutive cross terms, lifted `be`) of clip scenes on NumPy,
  JAX (eager and jit) and Torch: **maximum relative difference at most 1e-12** (the existing
  coefficient-lowering budget). Fields after 40 steps of a clip scene with a dipole source:
  **at most 1e-11 relative**.
* Derivatives of a field-energy objective with respect to each clip medium's `eps` (first,
  second, ground), JAX and Torch automatic differentiation against a central difference with
  relative step 1e-4: **relative difference at most 1e-6**, and JAX against Torch **at most
  1e-10**. Traced geometry on a clip member: refused by name.

Agreement across three backends is consistency, not physical correctness; Gate 2 is the
physical gate.

## Amendment 1 (2026-10-27, before any field was computed)

Planning the registered scenes (`fdtd_subpixel_polytope_clip.py plan`, which builds plans and
counts the clip's written voxels and computes no field) showed that two statements above were
wrong about the instrument, and that some registered grids cannot be planned. Nothing below was
chosen after seeing a measured number; it is committed and published before the first run.

* **Arm L: when the clip writes.** "The gap is thinner than every grid used, so its two faces
  share voxels at every grid" is false. The clip writes only where **both** faces fall inside
  one voxel, and a gap of 0.003 straddles a voxel boundary at 224 and 320 cells per unit, where
  it writes nothing (on the one-cell column: 3, 1, 3, 2, 1, 2, 0, 0 electric voxels at 28, 40,
  56, 80, 112, 160, 224, 320). Grid 20 is refused twice: by the PML door with a half-unit layer
  (10 cells, under 12) and by the resolution door (`eps` 6 at wavelength 0.8 is under 8 cells
  per wavelength). **The arm's grids become 28, 40, 56, 80, 112 and 160 cells per unit**, the
  set at which the clip writes and every wavelength plans; the PML is at least 12 cells and half
  a unit thick. The pass rule is unchanged: least-squares order of the band error over these
  six grids at least 1.8 per polarisation.
* **Arm D: what plans.** At the grids where the S7 dimer's clip writes (32, 32, 24 and 24
  electric voxels at 20, 24, 28 and 32 cells per unit; none at 40, 56, 80), the resolution door
  refuses `eps` 12 at the shorter wavelengths: grid 20 plans at 1.6-2.0 only, 24 at 1.4-2.0,
  28 at 1.2-2.0, 32 and finer at all eleven. Each wavelength is run as its own narrow band
  (`df = 0.04 / wavelength`), so that a wavelength the door admits is not lost to a broadband
  pulse's shortest wavelength. **Criterion (a) is kept as registered and will be reported as
  read: it fails at 20, 24 and 28 cells per unit**, because of the resolution door and not of
  the clip. **Criterion (b) is applied per wavelength**: at every wavelength and every
  clip-writing grid that plans, the relative error must lie within a factor of 2 of the power
  law through that wavelength's non-writing grids (40 and 56), against the 80-cell reference.
  The band error (mean over wavelengths) is reported only over the wavelengths every compared
  grid shares.

## Amendment 2 (2026-10-27, instrument run length; before any arm D row was kept)

One timing probe of arm D (32 cells per unit, wavelength 1.0, the narrow band of Amendment 1,
`Run(periods=40)`) showed the narrow-band pulse still on at the end of the run: the flux
monitor's decay diagnostic read 0.022. Its value is discarded and is not a row of the
record. A narrow band of relative width 0.04 needs about 35 periods to pass, so a valid run is
longer, and eleven narrow-band runs at the 80-cell reference grid would not finish in this
campaign's budget. The instrument is therefore:

* at the grids where the clip writes (20, 24, 28, 32), one narrow-band run per wavelength
  that plans, `Run(periods=90)`;
* at 40, 56 and 80, where a band covering all eleven wavelengths plans
  (`Band(wavelength=4/3, df=0.12)`), one broadband run, `Run(periods=40)`;
* an instrument check: at 40 cells, narrow-band runs at wavelengths 1.0, 1.5 and 2.0 against
  that grid's broadband run, reported, so that the two instruments are shown to agree well
  below the errors criterion (b) compares.

Every row records its decay diagnostic; a row whose flux monitor has not decayed below
1e-4 is reported as unsettled. The pass rules are unchanged.

## Amendment 3 (2026-10-27, arm D stopping rule; before any reference row exists)

Under Amendment 2 the first narrow-band rows read flux-monitor decays of 2.1e-8 and 3.5e-12
(24 cells, wavelengths 1.4 and 1.5) but 2.2e-3 at 32 cells and wavelength 1.0: a fixed length
of 90 periods does not settle a wavelength near a slowly ringing resonance of the epsilon-12
sphere, and the broadband runs' 40 periods would settle less. Those rows were taken with a
superseded instrument; they are kept, unused, in
`studies/fdtd/subpixel_polytope_clip/records/dimer_superseded_amendment2.jsonl`, and no
reference (40, 56, 80 cells) row had been produced. **Every arm D run now stops by the engine's
state-amplitude rule**, `Run(time=600, decay=Decay(threshold=1e-3, window=2 x the longest
wavelength))`, inside that hard cap: each run continues until every field and memory group has
fallen below 1e-3 of its own peak over a full window, or reaches the cap, which the row records.
Bands, grids, wavelengths and the pass rules are unchanged; the flux-decay validity mark of
Amendment 2 still applies.

## Results (2026-10-27)

Implementation: `src/wavebit/fdtd/_prepare/clip.py`, wired into door 5 of
`src/wavebit/fdtd/_prepare/checks.py`, the painters of `coefficients.py` and `constitutive.py`,
and a `Plan.clips` field omitted from the skeleton when empty. Tests:
`tests/fdtd/test_fd_clip.py`. Study and records: `studies/fdtd/subpixel_polytope_clip/`
(the report printed below is `records/fdtd_subpixel_polytope_clip_report.log`). CPU, float64,
NumPy 2.5.4, JAX 0.12.0.1, Torch 2.14.1+cpu, Python 3.13.

**Verdict.** Gates 1, 3 and 4 pass. Gate 2 passes arm L (both polarisations) and arm P, and
**fails arm D** on both of its criteria. The registered consequence of D's failure is applied:
**a voxel that two curved surfaces share (a sphere or a cylinder's wall against another) stays
refused by name**, even where its shared volume is exact. What C10 answers is therefore a pair
of bodies at least one of whose surfaces is flat in each shared voxel: planar layers at a
sub-cell gap, a box or prism beside another, and any body resting on, raised just above or
sunk into the face of an unturned `Box`.

### Gate 1: geometry and rule -- passes

Every registered tolerance holds in `tests/fdtd/test_fd_clip.py`: polyhedral fills within
2e-12 of a voxel and exposed interface area and moment within 2e-9 `dx^2` of the polytope
oracle, over the three admitted cases; sphere fills within 1e-10 of a voxel of adaptive
quadrature (a sphere beside a box, touching a substrate, sunk into it); the laminate's
components within 1e-12; the two-phase reduction bit for bit. Two defects were found by these
tests while writing them and fixed before any measurement: a flush test that read an overlap of
two back-to-back faces as flush, and a buried face whose exposed area came out -3e-12 `dx^2`
and gave the node tensor an eigenvalue below its floor. Twenty product mutations -- each made,
run against its tests, watched to fail and restored byte for byte -- were all caught.

### Gate 3: every existing two-media scene unchanged -- passes

The thirteen-scene dump of plan skeletons and lowered parameters is byte-identical to `main`
183baf37 (digest `d69d3c0b...`, 116 arrays). The four stage freezers reproduce their committed
goldens and write nothing; their printed drift (at most a few 1e-15 of a peak, inside the 1e-12
budget, from the installed versions) is identical, line for line, to the same freezers run on
183baf37 in the same environment. The goldens' SHA-256 are unchanged.

### Gate 4: backends and derivatives -- passes

Coefficients on NumPy, eager and compiled JAX and Torch agree within 1e-12 relative; fields
after 40 steps of a dipole-driven clip scene within 1e-11; material derivatives on JAX and
Torch within 1e-6 of a central difference at relative step 1e-4, and JAX within 1e-10 of Torch,
for each of the pair's media and the ground. Traced geometry on a member is refused by name.

### Gate 2, arm L: the planar stack -- passes

Band-RMS reflectance error against the transfer matrix (21 wavelengths in [0.8, 1.2]); every
run's monitor decayed below 3e-9.

| Cells per unit | Clip voxels | s: band error | p: band error | Control L0, s | Control L0, p |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 28 | 3 | 1.62e-2 | 1.09e-2 | 3.31e-2 | 2.63e-2 |
| 40 | 1 | 8.81e-3 | 7.47e-3 | 1.49e-2 | 1.35e-2 |
| 56 | 3 | 2.82e-3 | 1.94e-3 | 8.70e-3 | 6.73e-3 |
| 80 | 2 | 2.75e-3 | 2.28e-3 | 5.69e-3 | 4.76e-3 |
| 112 | 1 | 1.19e-3 | 8.82e-4 | 2.07e-3 | 1.75e-3 |
| 160 | 2 | 5.78e-4 | 4.55e-4 | 1.30e-3 | 1.05e-3 |
| **Fitted order / scatter** | | **1.866 / 49.6%** | **1.817 / 63.0%** | 1.852 / 24.5% | 1.856 / 26.2% |

Both polarisations meet the registered 1.8; the control without any clip voxel converges at
the same order, so the clip adds no order loss on parallel layers. The scatter is large:
56 to 80 cells is non-monotone in both polarisations, and the per-wavelength orders range from
1.13 to 2.76. The fitted orders are what the rule reads, not a guarantee at every wavelength.
A re-run on the committed source reproduced every printed digit.

### Gate 2, arm D: the S7 dimer -- fails

Relative error of the scattering cross-section against the 80-cell reference. A dash is a
(grid, wavelength) the resolution door refuses; every run stopped by the decay rule, all
monitors below 1.3e-7.

| Wavelength | 20 | 24 | 28 | 32 | 40 | 56 | Ratio to the 40/56 law at the clip grids |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1.0 | -- | -- | -- | 2.82e-1 | 1.67e-1 | 6.12e-2 | 32: 0.87 |
| 1.1 | -- | -- | -- | 3.27e-2 | 2.01e-2 | 7.79e-3 | 32: 0.87 |
| 1.2 | -- | -- | 1.40e-1 | 1.07e-1 | 7.86e-2 | 2.97e-2 | 28: 0.63, 32: 0.72 |
| 1.3 | -- | -- | 4.33e-2 | 1.02e-2 | 7.76e-3 | 1.47e-4 | 28: 0.08, 32: 0.09 |
| 1.4 | -- | 1.80e-3 | 8.57e-4 | 9.64e-4 | 5.26e-4 | 1.20e-4 | 24: 0.36, 28: 0.34, 32: 0.69 |
| 1.5 | -- | 9.04e-3 | 6.03e-3 | 5.57e-3 | 3.71e-3 | 9.67e-4 | 24: 0.32, 28: 0.39, 32: 0.62 |
| 1.6 | 3.49e-2 | 2.46e-2 | 1.92e-2 | 1.51e-2 | 1.03e-2 | 3.21e-3 | 0.30 to 0.67 |
| 1.7 | 7.92e-2 | 5.44e-2 | 4.61e-2 | 3.33e-2 | 2.34e-2 | 7.86e-3 | 0.36 to 0.69 |
| 1.8 | 6.61e-2 | 3.83e-2 | 3.22e-2 | 2.03e-2 | 1.36e-2 | 3.95e-3 | 0.38 to 0.66 |
| 1.9 | 1.94e-1 | 1.21e-1 | 1.09e-1 | 6.97e-2 | 4.98e-2 | 1.67e-2 | 0.41 to 0.69 |
| 2.0 | 9.34e-2 | 5.63e-2 | 5.26e-2 | 3.13e-2 | 2.29e-2 | 7.87e-3 | 0.45 to 0.74 |

The clip writes 32, 32, 24 and 24 electric voxels at 20, 24, 28 and 32 cells and none from
40. **Criterion (a) fails**: twelve (grid, wavelength) pairs are refused by the resolution door
at 20, 24 and 28 cells. **Criterion (b) fails at eight of eleven wavelengths**: at the
clip-writing grids the error lies 0.08 to 0.49 times the power law through the 40- and 56-cell
errors, outside the registered factor of 2. The band error over the five wavelengths every grid
shares (1.6 to 2.0) falls 9.36e-2, 5.89e-2, 5.19e-2, 3.40e-2, 2.40e-2, 7.91e-3 from 20 to 56
cells, a fitted order of 2.32 with 19.9% scatter. The instrument check holds: at 40 cells the
narrow and broadband runs agree to 3.9e-4, 2.9e-6 and 1.1e-4 at wavelengths 1.0, 1.5 and 2.0.

**What the failure says, and what it does not.** The clip-writing grids are not worse than the
law: they are better than it, by a factor of 2 to 12. The law is fitted through two grids, 40
and 56, against a reference only 1.43 times finer than the second of them, so its error at 56
is understated and its slope steepened (fitted orders through all six grids read 2.2 to 3.0,
and 7.7 at wavelength 1.3, where the 56-cell error nearly crosses zero). That reading is an
explanation offered after the fact and is not used to set the verdict aside: the rule was
registered with its consequence, and the consequence is applied. Answering two curved surfaces
would need a gate that passes -- a finer reference or a reference-free statistic, registered
before it is run.
