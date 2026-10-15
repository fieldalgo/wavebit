# Independent transverse meshes and axis-consistent material averaging

Written: 2026-09-09

Targets:
- `src/wavebit/fdtd/_periodic/scene.py`
- `src/wavebit/compat/legacy.py`
- `tests/fdtd/test_fdtd_transverse_resolution.py`
- `studies/fdtd/transverse_resolution/fdtd_transverse_resolution.py`
- `studies/fdtd/lattice_reference/fdtd_lattice_reference.py`
- `studies/fdtd/rayleigh/fdtd_rayleigh.py`

TL;DR: `transverse_resolution` makes x/y sampling independent of the z mesh.
In the measured near-cutoff grating, refining x is much more useful than
refining z alone. A selected x-refined mesh approaches the isotropically fine
lattice accuracy with about half as many cells and 36% of its cell-update
count at equal duration. This is a workload comparison, not a wall-time
speedup claim. A separate fix makes transverse material averaging respect
x/y exchange, including at the default mesh.

## Contract and what actually stays fixed

The [guide](../../guide/fdtd/time-domain.md#independent-transverse-resolution) defines the
positive static scalar or (x,y) pair, rounding and invariant-y conventions.
`resolution` still controls z. Specifying different transverse targets leaves
z nodes, layer interfaces, source and diffraction planes, and PML faces fixed.
The time step is recomputed from all actual cell widths, so the number of
updates and the discrete frequency response generally change.

The default target remains the scalar `resolution` in both directions.
Explicitly selecting the old targets gives identical default planar results
from above and below. Rectangular, conical, graded and layer-aligned scenes
retain their coordinate conventions. Existing Nyquist checks reject diffraction
windows that do not fit the requested transverse mesh. Mesh dimensions remain
static; this does not introduce geometry or grid-coordinate derivatives.

## A pre-existing transverse averaging asymmetry

The x/y exchange test exposed a separate rasterization defect. The previous
implementation always processed x then y. For Ex this meant a harmonic mean
in x followed by an arithmetic mean in y, while Ey used an arithmetic mean
in x followed by a harmonic mean in y. These operations do not commute for a
cell overlapping a crossed material corner. Swapping axes therefore did not
rotate the same discretized material.

Transverse averaging now follows the field component: normal harmonic first,
then tangential arithmetic, for both epsilon and mu. Ez/Hz retain arithmetic
averaging in both transverse directions; z averaging is unchanged. Pole fill
fractions remain arithmetic. This changes Ey/Hy coefficients in crossed corner
cells even if `transverse_resolution` is omitted. It is a correction to the
common scene rasterizer; no legacy solver or Maxwell/ADE/CPML update changed.

The distinction between normal harmonic and tangential arithmetic means is
described in [Meep's public subpixel-smoothing documentation](https://meep.readthedocs.io/en/latest/Subpixel_Smoothing/).
Its full effective-tensor construction also accounts for an interface normal.
The present diagonal, separable raster rule does not implement that construction
and does not establish general corner or dispersive-interface accuracy.
No Meep implementation was opened or copied.

The numerical counterexample uses a 2-by-3 asymmetric epsilon raster and an
independently varying mu raster, periods (0.4,0.3), z resolution 10, transverse
targets (12,13), theta 0.25, phi 0.31 and bottom incidence. Exchanging x/y,
the periods and targets, and replacing phi by pi/2-phi must preserve diagonal
Jones entries and negate off-diagonal entries. Before the fix, the s-to-s
reflection alone differed by 3.2441e-4. With the fix, all complex r/t columns
agree under that transformation to 1e-11. Tests cover both point sampling and
subpixel averaging, at default and independent transverse targets.
A separate coefficient check uses equal overlaps of a 2-by-2 raster
`[[1,4],[9,16]]`. The two normal harmonic means are 1.8 and 6.4; their
tangential arithmetic mean is 4.1. Both staggered Ey and Hy material
coefficients recover that value. Reversing those operations would give
4.16667, distinguishing the corrected rule from a merely symmetric-looking
scattering example.

## Independent numerical checks

Malformed controls and insufficient sampling are refused before stepping.
Rectangular-grid tests check actual cell counts, widths and CFL against the
declared periods, while z positions remain unchanged. A magnetic conical slab
on unequal x/y/z resolutions refines toward the independent Airy amplitudes.
The refinement keeps the same axis-resolution ratios and retains both
polarization columns across a wavelength band.

An asymmetric crossed scene combines the new mesh, bottom illumination,
directional extraction, a source spectral zero, an independently placed source,
and cropped time records. Its material, permeability and pole-strength
derivatives agree with central finite differences on JAX and Torch CPU;
values agree with NumPy. These checks validate the existing material graph
through the new mesh, not differentiation of resolution or geometry.
The asymmetric magnetic crossed raster is also compared with RCWA using
half-widths 3/5 on **both** transverse axes. The reference columns differ
by less than 0.001. Refining z from 20 to 40 with x resolution 1.5 times z
reduces the full Jones amplitude error by more than 40%, to below 0.008.
This adds a patterned-geometry physical check to the exact axis-exchange test.

## Which axis limits the near-cutoff case?

The [harmonic lattice study](fdtd-lattice-reference.md) provides the independent
outgoing-boundary reference, with no finite pulse or CPML error. We retain its
binary grating (epsilon 2.25/1, period 0.8, thickness 0.2), wavelengths
0.95808/0.96192 and fixed transverse wavevector. Error is the maximum absolute
complex r/t difference from RCWA order 32 over orders -1/0, separately for TE
and TM. All harmonic solves below use NumPy/SciPy double precision on CPU.

| z / x / y resolution | Automatic dt | TE lattice error | TM lattice error |
| --- | ---: | ---: | ---: |
| 40 / 40 / 40 | 0.0115470 | 0.0278171 | 0.0507211 |
| 40 / 80 / 40 | 0.00816497 | 0.00563022 | 0.00950680 |
| 40 / 160 / 40 | 0.00471405 | 0.00312578 | 0.00333279 |
| 40 / 320 / 40 | 0.00246183 | 0.00302786 | 0.00335224 |
| 80 / 40 / 40 | 0.00816497 | 0.0311428 | 0.0580543 |
| 160 / 40 / 40 | 0.00471405 | 0.0336961 | 0.0635874 |
| 80 / 80 / 40 | 0.00666667 | 0.00603292 | 0.0105481 |
| 80 / 80 / 80 | 0.00577350 | 0.00658717 | 0.0115905 |

The x-only refinement is useful, but its changing dt is part of this comparison.
Conversely, z-only refinement appears to worsen the answer partly because a
smaller time step removes some cancellation with the spatial discretization.
A second harmonic-only control fixes dt=0.0024 for every grid, below all their
automatic steps:

| z / x / y resolution | TE error, fixed dt | TM error, fixed dt |
| --- | ---: | ---: |
| 40 / 40 / 40 | 0.0353937 | 0.0666274 |
| 40 / 80 / 40 | 0.00862966 | 0.0150868 |
| 40 / 160 / 40 | 0.00334229 | 0.00463279 |
| 40 / 320 / 40 | 0.00302893 | 0.00335298 |
| 80 / 40 / 40 | 0.0348318 | 0.0659356 |
| 160 / 40 / 40 | 0.0347075 | 0.0657677 |
| 80 / 80 / 40 | 0.00796988 | 0.0141573 |
| 80 / 80 / 80 | 0.00796988 | 0.0141573 |

The fixed-step control confirms that x resolution is the larger limitation
here. At x resolution 320 the unchanged z mesh limits further improvement.
The two final rows have identical harmonic errors: y is invariant, so its
spacing affects the automatic CFL step but not this fixed-frequency scalar
operator when dt is held fixed. That does not apply to conical or crossed
fields with a nonzero y derivative.

## Cells and updates, not a general speed benchmark

For actual time grids with source z=-1, padding 1, PML width 1.2 and nominal
target 1e-160, the 800-period work counts are:

| Grid | Shape (z,y,x) | Updates per run | Four-run cell updates |
| --- | --- | ---: | ---: |
| Isotropic 40 | 264, 2, 32 | 66511 | 4.49508e9 |
| x 80, z/y 40 | 264, 2, 64 | 94061 | 1.27140e10 |
| Isotropic 80 | 520, 2, 64 | 133022 | 3.54158e10 |

The selective mesh has 50.8% of the fine isotropic mesh's cells and 35.9% of
its cell-update count. Field storage scales with cell count at the same
retention settings; complete memory use also includes spectra, checkpoints,
source samples and records. These counts do not measure peak memory or prove
a proportional runtime improvement. The x-only harmonic answer is slightly
better at its automatic dt and slightly worse at matched dt, another reason
to avoid declaring a universal accuracy-per-cost winner.

## Actual finite-time comparison

The selected x-80/z-y-40 mesh was also evolved for 1600 periods with the same
source and PML geometry, Gaussian width 60 and two source zeros at its
discrete cutoff 1.04150007042. There are 188121 updates per run. The finite
CPML harmonic reference differs from its exact-exterior counterpart by only
1.62672e-9 TE and 1.68460e-9 TM at the two measured wavelengths.

| Time run | Error against finite-CPML limit, TE / TM | Error against RCWA, TE / TM | Four-run cell updates |
| --- | --- | --- | ---: |
| x 80, z/y 40, 1600 periods | 2.65211e-5 / 8.86595e-6 | 0.00561200 / 0.00950666 | 2.54279e10 |
| Isotropic 80, 800 periods, previous measured control | 8.00971e-4 / 2.68474e-4 | 0.00720676 / 0.0117110 | 3.54158e10 |

The second row is the matching source/PML control in the
[previous harmonic study](fdtd-lattice-reference.md#actual-time-evolution-with-the-improved-boundary).
The selected mesh permits twice the physical duration with 71.8% of that row's
cell-update count. Its actual errors are smaller in this particular comparison.
As the fixed-dt table shows, this does not demonstrate a universal ordering of
spatial accuracy between the two meshes or a measured wall-time speedup.

The long run's 94060-update prefix ends just before 800 periods. It has time
errors 7.21807e-4/2.44177e-4; final-minus-prefix amplitude changes are
7.48229e-4/2.53009e-4 over the final 94061 updates. Both scattering convergence
and state decay remain false over that long interval. The final independently
measured time errors are small, but they do not retroactively make the entire
comparison window quiet. Returning transmission-side order -1 amplitudes are
1.94373e-5/9.81683e-6. No result here validates arbitrary cutoffs, metal corners,
GPU execution or derivatives with respect to mesh settings.

## Reproduce

```shell
python -m studies.fdtd.transverse_resolution.fdtd_transverse_resolution
python -m studies.fdtd.transverse_resolution.fdtd_transverse_resolution --fixed-dt 0.0024
python -m studies.fdtd.transverse_resolution.fdtd_transverse_resolution --case x80 --cpml
python -m studies.fdtd.transverse_resolution.fdtd_transverse_resolution --case x80 --time --periods 1600
```

The first two commands compare all eight outgoing-boundary harmonic meshes.
`--cpml` additionally solves finite CPML for the selected case. `--time`
executes an actual pulse, uses double-precision JAX CPU by default, and requires
one explicit case; `--backend numpy` selects NumPy. It uses Gaussian width 60
and two zeros retuned to that mesh's discrete vacuum cutoff. A long run compares
its full second half; an 800-period run compares its final 100 periods.
JSON retains full final/prefix amplitudes, convergence flags, boundary and time
errors, grid/update counts and elapsed time including compilation. The
`--fixed-dt` option is a harmonic-only experiment, not a new public scene control.
