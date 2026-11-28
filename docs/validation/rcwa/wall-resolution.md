# The solve in stretched coordinates against finite-element fields — measurements

What [`Numerics(wall_resolution=G)`](../../guide/rcwa/wall-resolution.md) was measured against, what came out, and where it has not been checked. No independent implementation of the stretched or the matched solve was found to compare with at the same truncation, so everything here is a comparison with finite-element fields, which says how accurate the method is, a comparison with other codes that solve the same structures without the stretch, or a property the construction has to satisfy. The option is stable on the strength of {ref}`the finite-element table beyond normal incidence <wall-resolution-wide>` and, for rectangles, {ref}`RETICOLO's unstretched solve at high order <wall-resolution-reticolo>`.

## The method in one paragraph

Along each axis the solve uses a coordinate `u` with `x = F(u)`, where `F` is, between each pair of neighbouring walls, a straight line plus one period of a sine (Granet's adaptive spatial resolution in the form of Vallius and Honkanen, Opt. Express 10, 24 (2002)). `F'` equals the option's value `G` at every wall, and the walls sit at equal distances in `u`. In `(u, v, z)` Maxwell's equations keep their form with `eps~ = eps diag(g/f, f/g, f g)` and `mu~` alike (`f = F'`, `g` the other axis's slope), so each layer is solved by the same eigenproblem as before with slope-weighted Fourier integrals. The whole stack is solved in these coordinates, the half-spaces included: their modes are computed like a layer's, and only the propagating orders are replaced by the exact plane waves written in the stretched basis.

## Near fields on seven layers

One layer of thickness 150 nm in a 400 nm square cell, in air, lit at normal incidence at 900 nm with `E` along x; `factorization='li'`. Each entry is the RMS of the in-plane vector error of `E` over 2304 samples of the mid-height plane, wall samples included, over the reference's own plane RMS, at orders 8 / 12 / 16. The references are 3D finite-element fields (gyptis 1.2.0, second-order edge elements).

| Layer | Unstretched | `wall_resolution=0.02` | `R`: stretched (reference) |
| --- | --- | --- | --- |
| metal block 200 nm, `eps` -20+1.5j | 32.1 / 25.8 / 22.5% | 4.0 / 0.63 / 0.52% | 0.35555 / 0.35553 / 0.35547 (0.35615) |
| square hole 200 nm in a film of that metal | 36.8 / 30.9 / 29.0% | 6.5 / 3.3 / 1.8% | 0.89270 / 0.89331 / 0.89344 (0.89281) |
| dielectric square 200 nm, `eps` 12.25 | 14.8 / 11.5 / 9.1% | 0.71 / 0.41 / 0.39% | 0.03650 / 0.03650 / 0.03650 (0.03662) |
| metal rectangle 250 by 150 nm | 37.9 / 36.4 / 26.5% | 4.5 / 3.6 / 3.4% | 0.63826 / 0.63888 / 0.63885 (0.64043) |
| metal block, `eps` -100+5j | 85.3 / 34.7 / 27.4% | 5.0 / 3.3 / 0.70% | 0.28785 / 0.28750 / 0.28783 (0.28827) |
| metal block, `eps` -5+1j | 21.8 / 17.0 / 13.4% | 2.8 / 1.4 / 1.0% | 0.37259 / 0.37308 / 0.37361 (0.37454) |
| dielectric rectangle 250 by 150 nm | 16.0 / 12.0 / 9.8% | 1.7 / 1.6 / 1.6% | 0.07651 / 0.07651 / 0.07651 (0.07671) |

- **The first four references are stored** (`tests/references/golden_gyptis_rectangles_field.npz`, with its record) and the order 8 rows are tests; the order 12 rows are in the slow tier. The unstretched order 16 entries of those four were measured with a test implementation of the same solve. The last three references are not stored in this repository, and no test reads them.
- **How good the references are.** The block's and the hole's coarser mesh levels are 0.37% and 0.20% of the plane RMS from the stored ones. The other five have one mesh level. Where the stretched solve stops improving -- 3.4% on the metal rectangle, 1.6% on the dielectric one, 0.4% on the dielectric square -- its `R` is settled to five digits and 0.2 to 0.3% from the reference's, and what is left sits beside the walls outside the shape. That floor is taken to be those references' accuracy at one mesh level; it was not confirmed with a finer mesh.
- **The hole** is the slowest of the seven: 6.5% at order 8 and 1.8% at order 16.
- **The unstretched solve's `R`** on the metal block is 0.3505, 0.3506, 0.3512 at the three orders; the stretched one is within 0.2% of the reference from order 8.

### Criteria written before the implementation

These were set from a test implementation's values with about a factor of two of margin, before the product code existed. At order 12, as a fraction of the unstretched plane error:

| Criterion | Limit | Measured |
| --- | --- | --- |
| metal block, field | 0.06 | 0.024 |
| metal block, `R` against the reference | 0.3% | 0.17% |
| dielectric square, field | 0.1 | 0.036 |
| dielectric square, `R` (from order 8) | 0.5% | 0.33% |
| metal rectangle, field | 0.25 | 0.10 |
| hole, field | 0.2 | 0.11 |

At order 16 the block is at 0.023 and the rectangle at 0.13. One criterion is not met as it was written: the three backends were to agree to 1e-10, and they do in `R` and `T` (3e-11 at worst) but the field of a metal layer at 350 nm differs by 4e-10 between them.

## What the construction has to satisfy

Each of these is a test in `tests/rcwa/test_wall_resolution.py`.

- **The limit without a stretch.** With `wall_resolution=1.0` on a block half the period wide, whose walls are already evenly spaced, the coordinates are the cell's own. `R`, `T`, the specular amplitudes and all six field components then agree with the unstretched solve to 1e-12 -- on four layers over glass, two of them uniform, lit obliquely from either side -- although the half-spaces and the uniform layers went through an eigenproblem instead of plane waves.
- **Moving the structure.** A rectangle moved by (37.5, -25) nm in the cell gives the same `R` and `T` to 1e-11 and the same field at the moved samples. The walls' places in the stretched cell depend on their order alone.
- **Energy.** On the lossless block `R + T = 1` to 1e-4 at order 8, at normal and oblique incidence and with four and five propagating orders.
- **Backends.** NumPy, JAX and PyTorch, eager, agree to 3e-11 in `R` and `T` and to 4e-10 of the field's scale.

## Oblique incidence, several propagating orders

These cases have no finite-element reference of their own (the oblique ones here are at phi 30 deg); finite-element references at oblique incidence with phi 0, with five propagating orders, on a substrate and with two layers are {ref}`below <wall-resolution-wide>`. What can be seen here is whether the stretched solve settles, and whether the unstretched one moves toward it as its order rises. The 200 nm block (or hole) again; `R` at orders 8 / 12 / 16, and for the unstretched solve also at 20.

| Case | Propagating orders | `wall_resolution=0.02` | Unstretched |
| --- | --- | --- | --- |
| dielectric, 900 nm, theta 20 deg, phi 30 deg, p | 1 | 0.02594 / 0.02594 / 0.02594 | 0.02615 / 0.02605 / 0.02601 / 0.02599 |
| the same, s | 1 | 0.05394 / 0.05393 / 0.05393 | 0.05418 / 0.05406 / 0.05402 / 0.05399 |
| dielectric, 350 nm, normal | 5 | 0.24577 / 0.24502 / 0.24497 | 0.23188 / 0.23990 / 0.24262 / 0.24372 |
| dielectric, 350 nm, theta 20 deg, p | 4 | 0.12974 / 0.16194 / 0.16203 | 0.13188 / 0.15125 / 0.15719 / 0.15948 |
| dielectric, 405 nm, normal | 1 | 0.02123 / 0.31136 / 0.31217 | 0.14721 / 0.27412 / 0.30218 / 0.31042 |
| dielectric, 395 nm, normal | 5 | 0.63451 / 0.64034 / 0.64034 | 0.61329 / 0.63145 / 0.63660 / 0.63850 |
| metal, 900 nm, theta 20 deg, phi 30 deg, p | 1 | 0.32910 / 0.32900 / 0.32902 | 0.32154 / 0.32339 / 0.32445 / 0.32565 |
| metal, 350 nm, normal | 5 | 0.36196 / 0.36280 / 0.36279 | 0.34676 / 0.35414 / 0.35748 / 0.35884 |
| hole, 900 nm, theta 20 deg, p | 1 | 0.88329 / 0.88395 / 0.88409 | 0.87830 / 0.87979 / 0.87982 / 0.87956 |

- In the lossless cases `R + T = 1` to 2e-6 at order 8 and to 1e-9 from order 12.
- On the eight block cases the stretched solve moves less between orders 12 and 16 (at most 8e-4, beside the Rayleigh wavelength; 9e-5 elsewhere) than the unstretched one between 16 and 20, and the unstretched one approaches it monotonically.
- On the hole the stretched solve moves by 1.4e-4 between orders 12 and 16; the unstretched one has not settled by order 20, where it turns back.
- **This is not evidence of accuracy.** It shows that the scheme -- half-spaces solved in the stretched coordinates, each propagating order replaced by its plane wave -- does not break at oblique incidence or with several propagating orders.

### Beside a Rayleigh wavelength

The cell's period is 400 nm, so 405 nm is just above a Rayleigh wavelength and 395 nm just below.

- At 395 nm, with five propagating orders, the stretched solve is settled from order 8.
- **At 405 nm it is far off at order 8** (`R` 0.021 against 0.312) and settles from order 12; the unstretched solve is slow there too. Replacing the orders just past cutoff by plane waves as well, in a test implementation, changed nothing: the low orders are not short of a better half-space, the layer itself converges slowly at that wavelength.

## Why the half-spaces are solved in the stretched coordinates

The alternative is to leave the half-spaces unstretched and carry every one of their plane waves, evanescent ones included, into the layers' basis. A test implementation did that first. Where the coordinate runs fast the high orders then oscillate beyond the retained harmonics, and the matrix that carries them is nearly singular: its condition number was 1e4, 5e6 and 2e9 at orders 8, 12 and 16 with a slope of 0.02.

- On the dielectric layers the answer was the same either way, to four digits, even at 2e9.
- On the metal ones the reflectance stayed right while the field broke down as the order rose: the metal rectangle read 5.7%, 17% and 27% at orders 8, 12, 16 where the solve above reads 4.5%, 3.6%, 3.4%, and the block with `eps` -100+5j was 143% off at order 16.

Solving the half-spaces in the same coordinates needs only the propagating orders' plane waves, a handful of well-conditioned columns, and no inverse of that matrix anywhere.

## Why this function and this placement

- **A bump at each wall did less.** A profile that slows the coordinate only within a fixed band around each wall was measured first, with the slope at the wall as low as 0.02; on the metal rectangle it read 19%, 14% and 4.2% at the three orders, against 4.5%, 3.6%, 3.4% for the sine. With it a dielectric layer's `R` also got worse at low order (3.5% off at order 8, against 0.26% here). The shape of the function matters more than the slope at the wall.
- **Equal distances between the walls.** With each piece instead as wide in `u` as it is in `x`, the metal rectangle read 6.7% and 6.3% at orders 8 and 12, and on a uniform slab with two walls 37.5 nm apart the field was wrong by more than its own size; with equal distances the same slab's field was uniform to 1.5%. Placements between the two, on axes with three or more walls, are measured {ref}`below <wall-resolution-many-walls-placement>`.
- **The slope.** On the metal rectangle a slope of 0.1 read 8.6 / 4.6 / 4.1% and on the dielectric square 0.93 / 0.75 / 0.51%, between the unstretched solve and 0.02.

These three comparisons were made with a test implementation that the product code reproduces to five digits in `R` and `T` on thirteen cases.

(wall-resolution-circles-measured)=
## Circles and ellipses

A stack whose patterned layers hold one circle or one axis-aligned ellipse is solved in coordinates matched to it ([the guide](../../guide/rcwa/wall-resolution.md)): the mesh of Weiss et al. (Opt. Express 17, 8051 (2009)) carries the rectangle whose corners are the ellipse's 45-degree points onto the ellipse, and the stretch above puts its four lines half a period apart at the slope `G`. The materials become full in-plane tensors, `eps~ = eps det(J) (J^T J)^-1` with `J` the map's Jacobian, sampled on a 512 by 512 raster of the stretched cell whose pixel edges are those lines and put through the tensor Li factorization. The half-spaces are solved in the same coordinates, as for rectangles, and their propagating orders replaced by the exact plane waves written in the matched basis.

### Near fields on twelve layers

Normal incidence with `E` along x. The cell is the one above (400 nm, a 150 nm layer, 900 nm) except the resonant pair, a circle of radius 280 nm in a 1000 nm cell, 200 nm thick, at 1363 nm. Plane errors as above, wall samples included, at orders 8 / 12, against 3D finite-element fields (gyptis 1.2.0); the default solve is the product's NVF with the plain sum.

| Layer | Default solve | `wall_resolution=0.02` | `R`: matched (reference) |
| --- | --- | --- | --- |
| metal circle, radius 100 nm, `eps` -20+1.5j | 27.4 / 21.5% | 2.1 / 0.42% | 0.18804 / 0.18792 (0.18799) |
| metal circle, radius 60 nm | 17.7 / 19.8% | 3.9 / 0.43% | 0.01656 / 0.01657 (0.01658) |
| metal circle, radius 140 nm | 32.1 / 24.6% | 1.1 / 0.32% | 0.55233 / 0.55230 (0.55237) |
| metal circle, `eps` -100+5j | 29.7 / 28.2% | 1.1 / 0.49% | 0.15754 / 0.15749 (0.15755) |
| metal circle, `eps` -5+1j | 20.9 / 17.3% | 2.1 / 1.8% | 0.37027 / 0.36986 (0.36538) |
| metal circle, resonant cell | 38.6 / 27.3% | 5.5 / 1.2% | 0.76081 / 0.76214 (0.76235) |
| metal ellipse, 130 by 80 nm | 30.6 / 27.5% | 2.1 / 0.61% | 0.45155 / 0.45145 (0.45173) |
| metal ellipse, 150 by 50 nm | 58.2 / 39.2% | 3.1 / 1.4% | 0.76094 / 0.76122 (0.76133) |
| metal ellipse, 110 by 90 nm | 24.0 / 24.8% | 2.2 / 0.62% | 0.25519 / 0.25515 (0.25553) |
| dielectric circle, radius 100 nm, `eps` 12.25 | 13.6 / 11.2% | 0.24 / 0.14% | 0.02429 / 0.02428 (0.02431) |
| dielectric circle, resonant cell | 9.1 / 7.1% | 1.2 / 0.14% | 0.77225 / 0.76715 (0.76816) |
| dielectric ellipse, 130 by 80 nm | 15.5 / 12.8% | 0.37 / 0.25% | 0.05278 / 0.05277 (0.05282) |

- **Three references are stored**: the two circles of radius 100 nm (`golden_gyptis_cylinder_field.npz`) and the dielectric ellipse (`golden_gyptis_shapes_field.npz`), read by `tests/references/test_gyptis_matched_field_golden.py`, order 8 in the default run and order 12 in the slow tier. The other nine are in a private research repository, and no test reads them.
- **How good the references are.** The stored ones' next mesh levels are 0.15%, 0.11% and 0.38% of the plane RMS from them. The resonant pair is at the third level of its ladder, the 130 by 80 metal ellipse at the second (against the first, the matched solve reads 0.70% at order 12, against the second 0.61%), the 150 by 50 and 110 by 90 ellipses at the first, the others at the second. Below about 0.3 to 0.6% a difference is within the references' own step between levels and is not read.
- **The default solve's `R`** is 4% low on the resonant metal cell and 12% low on the 150 by 50 ellipse at order 12 (0.73033 and 0.66875); the matched solve is within 0.03% and 0.01% there.
- **The weak metal.** On `eps` -5+1j the matched solve's `R` is 1.3% and 1.2% above the reference's at orders 8 and 12. Two finite-element models of that circle agree at their finest levels (0.36548 and 0.36520; the reference is 0.36538), and the default solve's `R` (0.36643 at order 12) is on their side; in a test implementation the offset grew as the stretch was made stronger, while the field kept improving. It is taken to be this method's error on a weak metal, and the guide says so.

### Criteria for circles and ellipses, written before the implementation

Set from a test implementation's values with about a factor of two of margin (design of the second stage), at order 12, with slope 0.02:

| Criterion | Limit | Measured |
| --- | --- | --- |
| five metal circles (radius 60, 100, 140 nm; `eps` -100+5j; the resonant cell): field as a fraction of the default solve's | 0.1 | 0.013 to 0.044 |
| the same five: `R` against the reference | 0.3% | 0.01 to 0.07% |
| three metal ellipses: field as a fraction of the default solve's | 0.1 | 0.022 to 0.036 |
| the same three: `R` against the reference | 0.3% | 0.01 to 0.15% |
| dielectric circle: field | 0.5% | 0.14% |
| the circle of `eps` -5+1j: field as a fraction / `R` | 0.2 / 1.5% | 0.10 / 1.2% |
| gold cylinder of Küchenmeister et al. (2012): `R` from order 12 | 0.823 +- 0.001 | 0.8231, 0.8234 (orders 12, 14) |

The gold cylinder is the published case of the mesh (Küchenmeister, Zebrowski and Busch, Opt. Express 20, 17319 (2012), section 9.2: gold of their Drude model, `eps` -26.37+1.80j at 829 nm, radius 150 nm, height 50 nm, period 700 nm, in air). The paper prints no table; its figure shows the mesh's `R` flat at about 0.823 from roughly 500 plane waves on, read off the plot. Here `R` is 0.8129, 0.8210, 0.8231, 0.8234 at orders 8, 10, 12, 14 (289 to 841 plane waves; the paper's truncation is circular, this one square).

All are met. The field reading had to change to meet the first: read as a plain sum, the resonant metal cell was 4.0% off at order 12, 0.147 of the default solve's error; read through the window below, 1.2%.

### How the field is read

The map bends on its four lines, which cross the whole cell, so the component of the solve's `E~` normal to a line jumps there even in a uniform medium, and its plain sum rings. The reading takes that component from the normal `D~ = [[eps~]] E~` within half a resolution unit `period / (2 N + 1)` of a line, keeps the plain sum beyond one and a half units, and blends the two with a `cos^2` between (`H` the same with `B`). At order 12 the plain sum against this window:

| Layer | Plain sum | Window |
| --- | --- | --- |
| metal circles: radius 100, 60, 140 nm; `eps` -100+5j, -5+1j | 0.51, 0.55, 1.48, 0.83, 1.80% | 0.42, 0.43, 0.32, 0.49, 1.77% |
| resonant cell: metal, dielectric | 4.02, 1.07% | 1.20, 0.14% |
| metal ellipses: 130 by 80, 150 by 50, 110 by 90 nm | 1.09, 2.20, 1.04% | 0.61, 1.39, 0.62% |
| dielectric circle, dielectric ellipse | 0.31, 0.34% | 0.14, 0.25% |

The window's widths are those a test implementation chose on a weaker stretch; they were not tuned again here. The product code reproduces that implementation's `R` to five digits and its plane errors to the digits that implementation printed, on all twelve layers.

### What the matched construction has to satisfy

Each is a test in `tests/rcwa/test_wall_resolution_matched.py`.

- **The map.** The rectangle's sides land on the ellipse to 1e-12, the map is periodic and does not fold, its analytic Jacobian agrees with finite differences, and its closed-form inverse returns a point to 1e-8 nm, on a circle, on ellipses, one of them across the cell's corner, and in a rectangular cell. The raster's pixel edges are the lines.
- **The plane waves in the matched basis** agree with a brute-force sum over the cell to the sum's own error.
- **A uniform stack.** With the shape of the background's own material the physics is a stack of uniform layers, yet the map bends every line and every layer is a full tensor. At order 6, over glass, lit obliquely from either side, `R`, `T` and the specular amplitudes are the plane-wave answer to 6e-9 on a circle and 6e-7 on a 150 by 60 ellipse; the electric field is within 0.15% and 0.41% RMS.
- **Moving the shape**, also across the cell's edge, gives the same `R` and `T` to 1e-9 and the same field at the moved samples.
- **Energy.** On a lossless ellipse `R + T = 1` to 1e-4 at order 6, at normal and oblique incidence and with four and five propagating orders.
- **Backends.** NumPy, JAX and PyTorch agree to 1e-9 in `R` and `T` and 1e-8 of the field's scale on a metal and a dielectric ellipse.

### What the matched solve costs

At order 12 on one layer the matched solve took 16 to 17 seconds where the default NVF solve took 10 to 11, on four cores; at order 8, 3 seconds against 2.

(wall-resolution-wide)=
## Beyond normal incidence: oblique, several propagating orders, a substrate, two layers

Twenty-four metal cases and four dielectric ones against 3D finite-element fields, with criteria written before measuring. The cell is the one above (400 nm square, a 150 nm layer, 900 nm, in air; the metal is `eps` -20+1.5j, the block 200 nm, the circle of radius 100 nm, the ellipse 130 by 80 nm), and each case changes one thing: the angle (theta 20 or 40 deg, phi 0, p and s), the wavelength (350 nm, five propagating orders), a glass substrate (n = 1.45), two identical patterned layers 100 nm apart, a 400 by 300 nm cell, or an ellipse in place of the circle, alone or with the angle of 40 deg, the substrate or the second layer. Order 12; plane errors as above, over 2304 samples of each patterned layer's mid-plane; the default solve is uncompressed Li on a block and NVF on a circle or an ellipse. The C1 rows at theta 40 deg, at 350 nm and with the ellipse were measured after the others, five and then six more on the ellipse, each set under the same criteria written before measuring it. The dielectric block on glass was measured last, with RETICOLO as a second reference ({ref}`below <wall-resolution-reticolo>`).

| Case | `wall_resolution=0.02`: `R` / `T` against the reference | Default solve: `R` / `T` | Field: stretched / default (ratio) | Reference's own step (`u_F`) | Verdict |
| --- | --- | --- | --- | --- | --- |
| metal block, theta 20 deg, p | -0.10 / +0.05% | -1.63 / -0.40% | 0.73 / 26.0% (0.028) | 1.5e-3 | met, weak |
| metal block, theta 20 deg, s | -0.18 / +0.06% | -2.05 / -0.44% | 3.16 / 32.3% (0.098) | 1.4e-3 | met, weak |
| metal block, theta 40 deg, p | -0.11 / +0.02% | -2.25 / -0.65% | 0.85 / 26.7% (0.032) | 2.0e-3 | met, weak |
| metal block, theta 40 deg, s | -0.15 / +0.09% | -2.10 / -0.25% | 3.08 / 34.0% (0.091) | 1.3e-3 | met, weak |
| metal block on glass | -0.41 / +0.19% | -5.44 / -0.23% | 2.33 / 39.8% (0.058) | 3.6e-3 | met, weak |
| two metal blocks, 100 nm apart | -0.10 / +0.29% | -2.66 / +0.28% | 0.83 / 27.6% (0.030) | 1.4e-3 | met, weak |
| metal block, 350 nm | -0.07 / +0.01% | -2.46 / -0.16% | 0.69 / 16.9% (0.041) | 5.6e-4 | met |
| metal circle, theta 20 deg, p | +0.03 / -0.01% | -2.29 / -1.03% | 0.54 / 21.8% (0.025) | 3.9e-4 | met |
| metal circle, theta 20 deg, s | +0.01 / -0.01% | -2.22 / -0.89% | 0.53 / 21.8% (0.024) | 7.5e-6 | met |
| metal circle on glass | -0.21 / -0.01% | -5.97 / -1.19% | 1.88 / 24.8% (0.076) | 4.0e-3 | met, weak |
| metal circle, 400 by 300 nm cell, p | +0.02 / -0.01% | -1.09 / -0.97% | 0.52 / 29.0% (0.018) | 2.3e-4 | met |
| metal circle, 400 by 300 nm cell, s | 0.00 / 0.00% | -0.43 / -1.67% | 0.50 / 23.3% (0.021) | 2.8e-4 | met |
| two metal circles, 100 nm apart | -0.02 / 0.00% | -5.11 / -2.28% | 0.62 / 22.0% (0.028) | 5.4e-4 | met |
| metal circle, theta 40 deg, p | +0.15 / -0.01% | -2.96 / -1.50% | 0.57 / 22.8% (0.025) | 1.7e-3 | met, weak |
| metal circle, theta 40 deg, s | +0.01 / -0.01% | -2.48 / -0.93% | 0.57 / 22.4% (0.025) | 1.9e-4 | met |
| metal circle, 350 nm | -0.04 / 0.00% | -2.73 / -0.41% | 0.28 / 14.4% (0.019) | 2.9e-4 | met |
| metal ellipse 130 by 80 nm, theta 20 deg, p | +0.02 / -0.02% | -1.16 / -1.67% | 0.76 / 27.8% (0.027) | 2.0e-4 | met |
| metal ellipse 130 by 80 nm, theta 20 deg, s | +0.01 / -0.01% | -0.81 / -0.61% | 0.50 / 18.9% (0.026) | 1.7e-4 | met |
| metal ellipse 130 by 80 nm, theta 40 deg, p | +0.06 / -0.03% | -1.59 / -2.35% | 0.81 / 28.5% (0.028) | 4.9e-4 | met |
| metal ellipse 130 by 80 nm, theta 40 deg, s | +0.01 / -0.01% | -1.08 / -0.83% | 0.52 / 20.7% (0.025) | 4.5e-4 | met |
| metal ellipse 130 by 80 nm on glass, p | -0.19 / +0.28% | -4.37 / -0.58% | 1.91 / 31.8% (0.060) | 1.8e-3 | met, weak |
| metal ellipse 130 by 80 nm on glass, s | -0.19 / -0.06% | -5.69 / -0.88% | 2.17 / 20.3% (0.107) | 4.3e-3 | met, weak |
| two metal ellipses 130 by 80 nm, 100 nm apart, p | -0.02 / +0.09% | -2.70 / -3.11% | 0.92 / 28.5% (0.032) | 2.5e-7 | met |
| two metal ellipses 130 by 80 nm, 100 nm apart, s | -0.02 / -0.02% | -1.98 / -1.44% | 0.55 / 18.6% (0.029) | 1.0e-3 | met, weak |
| dielectric block (`eps` 12.25), normal | -0.08 / 0.00% | +0.32 / -0.01% | 0.29 / 11.5% | 8.2e-4 | met |
| the same, theta 20 deg, p | -0.10 / 0.00% | +0.38 / -0.01% | 0.30 / 11.5% | 1.1e-3 | met, weak |
| the same, theta 20 deg, s | -0.06 / 0.00% | +0.14 / -0.01% | 0.30 / 11.4% | 7.1e-4 | met |
| the same on glass, normal | -0.24 / 0.00% | +1.10 / 0.00% | 0.48 / 11.8% | 3.7e-3 | met, weak |

- **The criteria**, written before measuring: on a metal, `R` and `T` within max(0.3%, 2 `u_F`) of the reference and the field at most 0.25 of the default solve's at the same order; on a dielectric, the same for `R` and `T` and the field within 1%. `u_F` is the relative change of the reference's `R` between its two finest mesh levels; where it is above 0.1% the row is marked weak, and its tolerance is wider by that much. All twenty-eight are met; thirteen are weak.
- **Where weak comes from.** On the substrate the reference's `R` moved by 0.18 to 0.43% between its two levels, on the oblique blocks by 0.11 to 0.20%, on the circle at theta 40 deg, p, whose `R` is only 0.057, by 0.17%, on the two ellipse layers, s, by 0.10%, and on the dielectric block on glass, whose `R` is only 0.0043, by 0.37%. On the two blocks its `T` moved by 0.56% and on the ellipse on glass, p, by 0.47% (the tolerance is set by `R`; there `T` is at +0.29% against 0.30% and +0.28% against 0.36%). The stretched solve itself moves by at most 0.16% in `R` between orders 12 and 16 on any of the twenty-eight, the most on the ellipse on glass, s.
- **The references.** 3D finite elements (gyptis 1.2.0, second-order edge elements) on cells reduced by the structures' mirror planes: a quarter of the cell at normal incidence, half of it at phi 0, the full height over a substrate; two mesh levels each, the finer with 0.25 to 0.69 million complex unknowns. Before any metal was read the models were checked against uniform slabs (analytic values, within 1.2e-4) and the dielectric block (RETICOLO at order 30, below: within 7.7e-5 at normal incidence, 3.6e-5 and 1.3e-4 at theta 20 deg for p and s); the criterion was 1e-3. The substrate model was checked the same way later, on the dielectric block on glass: within 6.5e-4 in `R` and 6.0e-6 in `T` of RETICOLO at order 30, against the same 1e-3. All twenty-eight are stored here (`tests/references/golden_gyptis_wide.npz` and, for the C1 rows added later, `golden_gyptis_wide_c1.npz` for five and `golden_gyptis_wide_c1e.npz` for the six on the ellipse and `golden_gyptis_wide_e.npz` for the dielectric block on glass, with RETICOLO's values beside the finite elements', each with its record): `R` and `T` on both mesh levels, `u_F` and the finer level's in-plane field on each patterned layer's mid-plane. `tests/references/test_gyptis_wide_golden.py` asserts the criteria above on all twenty-eight at order 12 in the slow tier, and in the default run on the cheapest metal and dielectric cases, the 400 by 300 nm circle (p) and the block at normal incidence, at order 8, where they already meet the same criteria.
- **A first set of references was not good enough.** Its absorbing layer kept nearly the same coarse mesh at every level, which left an error that refinement did not remove: on the lossless dielectric block `R + T` read 0.99985, and the oblique model failed the block check. With the absorbing layer meshed at 40 nm and the air at 20 nm the check passed and every row above was measured again; the criteria were not changed.
- **The 350 nm block by order.** The stretched solve's ten orders are within -0.15 to +0.20% of the reference's each, the default solve's within -3.5 to +1.3%; on the circle at 350 nm, -0.09 to +0.05% against -3.4 to +0.31%.
- **Not covered.** Phi other than 0, and a second reference for the metal rows on a substrate ({ref}`below <wall-resolution-reticolo>`). Shapes away from the cell's centre are measured {ref}`below <wall-resolution-off-centre>`.

(wall-resolution-many-walls)=
## Four walls along an axis

A layer with more than two walls along an axis divides the stretched cell among more pieces. Four such layers were checked against the same 3D finite elements, under the criteria above, written before measuring. The cell is the one above (400 nm square, a 150 nm layer, 900 nm, in air; the metal is `eps` -20+1.5j). Two bars, each 80 by 200 nm and centred at x = -100 and +100 nm, put four walls along x and two along y. A square frame, 240 nm outside with a 120 nm hole, puts four along each axis. All are at normal incidence, at order 12; the default solve is uncompressed Li.

| Case | `wall_resolution=0.02`: `R` / `T` against the reference | Default solve: `R` / `T` | Field: stretched / default (ratio) | Reference's own step (`u_F`) | Verdict |
| --- | --- | --- | --- | --- | --- |
| two metal bars, p (E across the bars) | -1.04 / -0.12% | -2.47 / -0.18% | 2.67 / 17.4% (0.153) | 2.6e-3 | not met, weak |
| two metal bars, s (E along the bars) | -0.24 / +0.07% | -2.64 / -1.50% | 3.44 / 38.5% (0.089) | 1.5e-3 | met, weak |
| metal frame, p | -0.35 / -0.59% | -3.14 / +0.33% | 19.9 / 37.9% (0.525) | 3.3e-4 | not met |
| two dielectric bars (`eps` 12.25), p | -0.07 / 0.00% | +0.61 / -0.01% | 0.85 / 12.3% | 9.3e-4 | met |

- **Two of the four are met.**
  - On the metal bars in p, `R` is 1.04% low against a tolerance of 0.52%. `R` is only 0.071 there. `T` and the field meet the criteria.
  - On the frame, `R` and `T` are outside 0.30% and the field ratio is 0.525 against 0.25.
- **A fifth case was planned and not judged.** Two metal bars at theta 20 deg, p, were to be included only if the reference's finer level fitted the measuring machine's disk. It did not: its factorization needed 51 GB.
- **By order** (read after the verdict). The stretched `R` on the bars in p was -1.38, -1.04, -0.44 and -0.09% off at orders 8, 12, 16 and 20, moving towards the reference. On the frame, `R` was -1.45, -0.35, -0.10 and -0.07% off, and `T` -0.84, -0.59, -0.03 and +0.02%. Order 20 was run on these two cases only. On the other two cases both stayed within 0.24% at orders 8 to 16.
- **The frame's field error is mostly in the samples** (read after the verdict).
  - 57 of the 2304 samples lie 0.08 or 0.25 nm from the frame's outer walls. No order here resolves a jump at that distance. On the bars the closest sample is 1.4 nm from a wall.
  - Without the samples within 1 nm of a wall, the stretched field error is 8.6, 7.4 and 3.9% at orders 12, 16 and 20. The default solve's is 34.7, 21.6 and 20.9%.
  - At orders 12 and 16 that is still about three times the bars' error. The reference's own field moved by 2.1% between its levels there, against 0.4 to 0.9% on the bars, so the 3.9% at order 20 is as large as the reference's own step.
  - A second measurement, under rules written before it, read both solves on an 80 by 80 grid whose samples all lie 2.5 nm or more from every wall. There the frame's field error is 5.0, 3.2 and 3.3% at orders 12, 16 and 20. The default solve's is 35.9, 23.7 and 22.0%, so the ratio is 0.14 to 0.15. On the bars the stretched error is 2.7, 3.5 and 2.1%.
  - The rules asked whether what remains comes from the samples near the walls, from four walls on both axes, or from the hole's re-entrant corners. They separated none of them.
    - The frame's excess over the bars is at order 12 only: 1.86 times, and 1.14 at order 16.
    - A layer of the frame's four corner blocks, with the same walls and only convex corners, behaves like the bars.
    - Only 3% of the frame's error lies within 20 nm of a re-entrant corner.
  - On all three layers the stretched solve stays 2.0 to 3.5% from the reference at orders 16 and 20. Away from the walls the reference moves by 0.6 to 0.7% between its levels. Which side holds the rest was not measured.
- Before these, the only check was a uniform slab through four unevenly placed walls, which reads the right `R` to 1e-5.
- These references are not stored in the repository, and no test asserts them.

(wall-resolution-many-walls-placement)=
### Placing three or more walls other than at equal distances

Four layers miss the criteria above at order 12, and all four have three or four walls along an axis: the metal bars and the frame in p here, and the T in s and the unequal bars below ({ref}`wall-resolution-off-centre`). Each piece between two walls gets `period / K` of the stretched cell, `K` the number of walls. Giving each piece its physical width instead was worse ({ref}`above <wall-resolution-many-walls>`, "Equal distances between the walls"), so placements between the two were tried on axes with three or more walls. Axes with two walls were left as they are. The criteria, written before measuring, asked for a placement that brings at least two of the four within the criteria at order 12. It also had to keep the three layers with three or four walls that meet them (the bars in s, the dielectric bars and the T in p).

- **Blend**: piece `l` is `(1 - a) period / K + a dx_l` wide, `dx_l` its physical width, for `a` = 0.25, 0.5 and 0.75.
- **Floor**: the width is proportional to `max(dx_l, m period / K)`, for `m` = 0.5 and 1.

| Placement | The four that miss: met | The three that meet: kept | Notes |
| --- | --- | --- | --- |
| equal (the product) | 0 of 4 | 3 of 3 | |
| blend 0.25 | 1 of 4 (T in s) | 3 of 3 | the frame's field ratio 0.358 |
| blend 0.5 | 1 of 4 (T in s) | 3 of 3 | the frame's field ratio 1.011 |
| blend 0.75 | 2 of 4 (bars p, unequal bars) | 2 of 3: the bars in s lose `T` (-0.32%) | |
| floor 0.5 | 3 of 4 (all but the frame) | 1 of 3: the bars in s lose `R` (-0.33%), the T in p `R` (-0.71%) | |
| floor 1 | 2 of 4 (T in s, unequal bars) | 2 of 3: the T in p loses `R` (-1.03%) | |

- **None was adopted.** Each placement that brings two or more of the four within the criteria takes one of the three out of them. The T in p is one of the stored references, and two placements lose it. The frame was met by none: its field ratio on the judged samples was 0.35 to 1.01 against 0.25. Only floor 1 put its `R` and `T` both within 0.30% (-0.29 / +0.25%).
- The placement stays at equal distances. A layer with three or more walls along an axis is solved at a higher order instead ({ref}`below <wall-resolution-many-walls-orders>`).
- Not tried: a slope that differs from wall to wall. The function above keeps `F'` equal to `G` on both sides of a wall, so it would need another profile.

(wall-resolution-many-walls-orders)=
### The four by order

With the placement at equal distances, the four layers against their references, `R` / `T` at orders 12, 16 and 20 (the field on the judged samples in brackets at order 20; read after the verdicts above):

| Layer | Walls along x / y | Order 12 | Order 16 | Order 20 | Tolerance |
| --- | --- | --- | --- | --- | --- |
| two metal bars, p | 4 / 2 | -1.04 / -0.12% | -0.44 / -0.23% | -0.09 / -0.02% (field 1.7%, ratio 0.088) | 0.52% |
| metal frame, p | 4 / 4 | -0.35 / -0.59% | -0.10 / -0.03% | -0.07 / +0.02% (field 3.3% on samples 2.5 nm or more from a wall) | 0.30% |
| metal T, theta 20 deg, s | 3 / 4 | -2.53 / -0.27% | -0.43 / -0.41% | -0.10 / -0.03% (field 2.2%, ratio 0.084) | 0.45% |
| two metal bars of unequal widths, theta 20 deg, p | 4 / 2 | -1.00 / -0.11% | -0.83 / -0.12% | -0.01 / -0.08% (field 1.4%, ratio 0.104) | 0.55% |

- At order 16 three of the four are within the tolerance in `R` and `T`. At order 20 all four are, and on the bars and the T the field errors are 1.4 to 2.2%, 0.08 to 0.10 of the default solve's.
- The orders 16 and 20 were not part of the criteria and are not a verdict. They are what the guide's advice for such layers rests on ([the guide](../../guide/rcwa/wall-resolution.md), "Many walls").

(wall-resolution-off-centre)=
## Shapes away from the cell's centre

Eight more metal cases were checked against the same 3D finite elements, under the criteria of the table above, written before measuring. The cell, layer, wavelength and metal are those above, all at order 12; the default solve is uncompressed Li on rectangles and NVF on a circle or an ellipse.

- **Five shapes moved in the cell.**
  - With one patterned layer a moved shape is the centred structure moved. Its reference `R` and `T` are the centred structure's. Its reference field at a sample is the centred structure's at the sample minus the move, with the Bloch phase. So the finite elements add nothing new here; what is measured is whether the solve keeps its accuracy when the shape crosses the cell's edge or corner.
  - The moves are not multiples of the samples' spacing, so the samples sit in other places on the shape than in the centred rows.
  - A rectangle may not cross the cell's edge, so the moved block is written as the four rectangles the cell's edges cut it into.
- **Three new layers with no mirror along x.** A T of two rectangles, x -120 to 0 by y -100 to 100 nm and x 0 to 80 by y -50 to 50 nm, puts three walls along x. Two bars of widths 80 and 40 nm, at x -150 to -70 and 50 to 90 nm, put four. All three are at theta 20 deg, phi 0, solved on half the cell in y.

| Case | `wall_resolution=0.02`: `R` / `T` against the reference | Default solve: `R` / `T` | Field: stretched / default (ratio) | Reference's own step (`u_F`) | Verdict |
| --- | --- | --- | --- | --- | --- |
| metal circle at (196, 0) nm, across the edge, theta 20 deg, p | +0.03 / -0.01% | -2.29 / -1.03% | 0.53 / 22.3% (0.024) | 3.9e-4 | met |
| metal circle at (196, 187) nm, across the corner, theta 20 deg, p | +0.03 / -0.01% | -2.29 / -1.03% | 0.56 / 22.3% (0.025) | 3.9e-4 | met |
| metal ellipse 130 by 80 nm at (-146, 130) nm, across the corner, theta 20 deg, p | +0.02 / -0.02% | -1.16 / -1.67% | 0.72 / 27.5% (0.026) | 2.0e-4 | met |
| metal circle at (60, 110) nm in the 400 by 300 nm cell, across the edge, p | +0.02 / -0.01% | -1.09 / -0.97% | 0.72 / 29.2% (0.025) | 2.3e-4 | met |
| metal block at (146, 130) nm, across the corner, four rectangles, theta 20 deg, p | -0.10 / +0.05% | -1.63 / -0.40% | 1.08 / 28.1% (0.038) | 1.5e-3 | met, weak |
| metal T, theta 20 deg, p | -0.30 / -0.35% | -1.59 / -2.08% | 5.66 / 27.9% (0.203) | 2.0e-3 | met, weak |
| metal T, theta 20 deg, s | -2.53 / -0.27% | -2.17 / -1.11% | 9.74 / 30.0% (0.324) | 2.3e-3 | not met, weak |
| two metal bars of unequal widths, theta 20 deg, p | -1.00 / -0.11% | -3.17 / -0.48% | 2.14 / 19.5% (0.110) | 2.8e-3 | not met, weak |

- **Six of the eight are met.**
  - On the circles and the ellipse the field error is 0.53 to 0.72%, as in their centred rows (0.52 to 0.76%). Moved, the solve gives the centred `R` and `T` to 1.3e-10 and the centred field to 5e-9 of its largest value.
  - The moved block's row is solved as of 2026-11-28, with touching rectangles of one material no wall (below). Measured before that, it read -0.18 / -0.26% in `R` / `T` and 6.73% in the field (ratio 0.240), and was met as well.
  - On the T in s, `R` is 2.53% low against 0.45% and the field ratio is 0.324.
  - On the bars `R` is 1.00% low against 0.55%. `R` is only 0.014 there; `T` and the field meet the criteria.
- **Two rectangles of one material that touched made a wall where they touched** (read after the verdict; changed on 2026-11-28).
  - Every rectangle edge was a wall of the stretch, also an edge with the same metal on both sides. The moved block's four pieces put such walls on the cell's edges, and the solve then differed from the centred block's: by -7.6e-4 in `R`, -3.1e-3 in `T` and 6.7% in the field (RMS). The default solve did not change, to 1e-12.
  - The centred block written as two rectangles that meet at x = 0 moved the same way: by -9.9e-4 in `R`, -2.0e-3 in `T` and 4.9% in the field. One rectangle moved within the cell is the centred block to 1.3e-11.
  - Since 2026-11-28 a wall is an edge with different materials on its two sides. At order 12 the block written as two halves, and moved as four pieces, is the block to 4e-11 in `R` and `T` and 5e-10 of the field's largest value (`tests/rcwa/test_wall_resolution.py` holds this at order 6).
  - The moved block's field error against the reference is then 1.08%, the centred row's 0.73%. The two are on different samples: the moved block's come within 1.25 nm of a wall, the centred block's no closer than 3.08 nm. A check written before the change asked for 1.0% on the moved block and is recorded as not met; the solve itself is the centred one moved.
- **By order** (read after the verdict). On the T in s the stretched `R` is -3.89, -2.53, -0.43 and -0.10% off at orders 8, 12, 16 and 20, and the field error 14.5, 9.7, 6.4 and 2.2%. On the T in p the field error is 10.7, 5.7, 3.3 and 2.4%. On the bars `R` is -1.50, -1.00, -0.83 and -0.01% off at orders 8, 12, 16 and 20.
- **The references.** For the moved shapes, the centred structure's finer level, solved again to read the field at the moved samples. Its `R` is within 4.4e-6 of the stored level's; the rebuilt circle and ellipse meshes differ from the first ones by 23 and 2 of about 80000 cells. For the T and the bars, two new levels each, with 0.73 and 0.82 million complex unknowns on the finer one. Before measuring, their metal volume in the mesh was checked against the analytic one, to 1e-14.
- The six met are stored in `tests/references/golden_gyptis_wide_d.npz`, with its record, and `tests/references/test_gyptis_wide_golden.py` asserts the criteria on them at order 12 in the slow tier. The two not met are not stored.
- **Not measured.** A new structure with no mirror along either axis, or two patterned layers moved by different amounts: a reference needs the whole cell or the whole height, two to four times the unknowns, which did not fit the measuring machine.

(wall-resolution-reticolo)=
## Rectangles against RETICOLO's unstretched solve at high order

RETICOLO V10 (Hugonin and Lalanne; Zenodo 14631951, CC BY 4.0) solves the same layers with Li's rule and no stretch (`granet = 0`, the only setting of the public version). Its orders were raised to 30 (40 on two normal-incidence cases), with both of its nestings of Li's rule, on the block cases of the table "Oblique incidence, several propagating orders" above and on the metal rectangle.

- **The two codes are the same solve.** At orders 8 and 12, on seven cases and both polarizations, RETICOLO and Wavebit's unstretched `factorization='li'` agree to 7.7e-13 in `R` and `T`.
- **Dielectric block: met.** The stretched solve at orders 12 and 16 is within max(2u, 2e-4) of RETICOLO's highest-order value in all 32 comparisons -- `R` and `T`, p and s, at normal incidence, at theta 20 deg phi 30 deg, and at 350 nm order by order -- with `u` the larger of RETICOLO's last step in order and the difference between its nestings. The stretched solve is 2e-5 to 6e-5 from it on the single-order cases, 4e-4 at 350 nm.
- **Metal: the unstretched solve does not settle.** This is a property of Li's rule without a stretch, which RETICOLO and Wavebit's unstretched `'li'` share (they agree to 7.7e-13 above) and which the stretch exists to speed up. On the metal block at normal incidence RETICOLO's `R` reads 0.35049, 0.35334 and 0.35370 at orders 8, 30 and 40, still rising toward the stretched solve's 0.35546 and the finite-element 0.35615. Metal rectangles are therefore judged by the finite-element table above, not by RETICOLO.
- **On glass** (n = 1.45, normal incidence, p), measured later under comparisons written before measuring.
  - **Dielectric block: met, and the finite-element substrate model with it.** RETICOLO's `R` is 0.0042897 and 0.0042839 at orders 24 and 30, and 0.0042716 with its other nesting; `T` 0.9957161. The finite-element substrate model's finer level is within 6.5e-4 in `R` and 6.0e-6 in `T` of the order-30 value, against 1e-3. Before this the substrate model had been checked only on a uniform slab. The stretched solve at orders 12 and 16 is within 3.2e-3 in `R` and 1.4e-5 in `T`, inside max(2u, 2e-4) with `u` 2.9e-3, the difference between RETICOLO's nestings; the unstretched solve at order 12 is 1.0% off. `tests/references/test_gyptis_wide_golden.py` asserts this at orders 12 and 16 in the slow tier.
  - **Metal: no second reference.** On the metal block on glass RETICOLO's `R` reads 0.38308, 0.38773 and 0.38941 at orders 20, 30 and 40, still rising toward the stretched solve's 0.39566 and the finite-element 0.39731. On the metal circle on glass, fmmax given a pixel grid moved by 0.4 to 0.6% in `R` between its last term counts (up to 1600), its two vector formulations and two grids. Neither settled to 1e-3, the bar written before measuring, and a finer finite-element level needs 2.2 to 2.4 times the elements, more than the measuring machine holds. The metal rows on glass above (block, circle, ellipse p and s) keep one reference, and stay weak.

## Dielectric circles and ellipses against a second RCWA code

fmmax 1.7.1 (MIT) solved five dielectric scenes -- a circle at theta 20 deg (p, s), on glass, in a 400 by 300 nm cell (p, s), in two layers, and a 130 by 80 nm ellipse at theta 20 deg (p, s) -- with up to 2000 terms. This is reported beside the criteria above, not as one of them.

- **Agreement.** Given the shapes' exact Fourier coefficients, fmmax's `R` does not depend on its pixel grid (within 2e-7 from 400 to 3200 pixels), and the matched solve at order 12 is within 0.009 to 0.089% of it in `R` and 0.004% in `T`, inside max(0.3%, 2u) on all eight rows with `u` fmmax's last step in terms. The default NVF solve is within 0.013 to 0.18%.
- **The comparison as first set up did not meet its criterion, because of its input.** It gave fmmax a pixel grid whose boundary pixels mixed the two permittivities by area. fmmax forms the inverse-permittivity coefficients from the reciprocal of the array it is given, and the reciprocal of an area-weighted permittivity misstates `1/eps` on those pixels (0.075 against 0.54 at half fill). Its `R` then moved as one over the grid -- by 3.0e-4 from 400 to 3200 pixels on the circle at theta 20 deg, p, at 800 terms -- and the matched solve was 0.44 to 1.10% from it in seven of eight rows. Giving fmmax the pixels' mean of `1/eps` instead holds its `R` within 4e-7 over the same grids.
- The exact-coefficient comparison was made after that first result was seen, so the criterion written before measuring is recorded as not met.

(wall-resolution-spurious)=
## Strong stretches: spurious layer modes and the exceptional-point handling

A strong stretch is reported in the literature to add spurious layer modes that move away as the order rises. The solver was run unchanged, with each layer's eigenvalues `w = q**2` read through a wrapper. The scenes were seven layers in the 400 nm cell at 900 nm:

- a 200 nm box and a circle of radius 100 nm;
- permittivity 12.25 or -20+1.5j;
- normal incidence and 20 deg.

Each scene was solved at orders 4 to 16 and slopes `G` from 0.1 to 0.005, and by the default solve at the same orders. The criteria were written before measuring. Where a reading was added after the first results, it says so.

- **Uniform media, where the answer is exact.** In the stretched coordinates a uniform medium's modes are the plane waves, `w = kx**2 + ky**2 - eps`. The stretched uniform media of permittivity 1, 2.25 and 12.25, on all seven coordinate systems, were compared with them.
  - No part of the spectrum held more computed values than plane waves, at any order or slope.
  - Within `|w| <= 50` -- the modes that still carry a field across a 150 nm layer at 900 nm -- the largest relative error was 3 to 5% at order 8, 4e-4 at order 12 and 3e-6 at order 16, nearly the same at every slope.
  - The criterion written first asked for 1e-3 up to the plane wave of order `N/2`. It is not met at any order, for example 64 of 98 values at order 8 and `G` 0.02. Those are high-order plane waves computed less accurately; the extra values a spurious mode would add were not found. The two readings that separate these, the count and the band, were added after the first results.
- **Patterned layers.**
  - At `G` 0.02, no layer had more roots with `Im w > 0` within `|w| <= 50` than the default solve at the same order. Such a root grows along `z` in a passive layer. At `G` 0.01 and 0.005, four of the seventy comparisons had one to four more.
  - In the lossless layers every such root, stretched or not, had its conjugate among the roots: a complex-mode pair, which carries no net power. `1 - R - T` stayed within 8e-13 of zero.
  - In the metal layers the absorption was positive.
  - The ten least-attenuated modes of a metal layer converged faster with the stretch. From order 12 to 16, 10 of 10 agreed to 1e-2 at `G` 0.02 on the box, against 2 of 10 for Li. On the circle it was 8 at `G` 0.02 and 10 at 0.01, against 6 for NVF.
- **The exceptional-point handling.**
  - On the 30 scenes measured for {ref}`the finite-element table beyond normal incidence <wall-resolution-wide>` (its slabs and blocks included), at orders 8, 12 and 16, `exceptional_points='rebuild'` and `'off'` gave the same `R` and `T` to 6.7e-16. The criterion was 1e-9. A pair was rebuilt in two of the 90 solves.
  - On the 69 points of the sweeps below where the stretched solve rebuilt a pair or a cluster, the two agreed to 7.7e-11.
- **Wavelength sweeps.** Three cases at 20 deg, p: the metal box, the metal circle and the dielectric box. Each ran from 550 to 1100 nm every 4 nm, at `G` 0.02 at orders 8 and 12, and by the default solve at order 12.
  - Nothing was refused.
  - No point departed by more than 1e-3 from a quadratic fit through its six neighbours unless the default solve and the other order's sweep showed it too, as the sharp guided resonances at 562 to 622 nm and 710 to 762 nm do.
  - The step was planned at 2 nm. A feature narrower than about 4 nm can have been missed.
- **Three warnings that did not change the answer.** The stretched solve raised `ExceptionalPointWarning` ("a 2D layer holds three or more modes of one defective eigenvalue cluster ... that no rebuild takes exactly") at three points of the sweeps. The default solve raised none on them.
  - The three points were 846 nm on both boxes at order 8, and 654 nm on the dielectric box at order 12.
  - The warnings came from the stretched air half-spaces, not the patterned layer.
  - At each point `R` was the same to the bit with the handling off. It lay within 7e-6 of the line through the solves 0.5 nm either side, which differ from each other by 3e-4 to 4e-3. It differed from the other order's value by what the neighbouring wavelengths differ by.
  - The cluster's exact cause was not traced.

(wall-resolution-single-precision)=
## Single precision: refused

`precision='single'` with `wall_resolution` was measured against the criteria below, written before measuring (2026-11-28), and did not meet them; it is refused by name with `rcwa.unsupported`.

- **The criteria.** The wide table above at order 12 and `G` 0.02, under its own criteria; `R` and `T` within 3e-4 (relative) of the double-precision solve and the in-plane field within 1e-3, at orders 8, 12 and 16 on NumPy, JAX and PyTorch; and the exceptional-point and spurious-mode readings of the section above.
- **What came out** (NumPy, normal incidence or theta 20 deg as in the table):
  - The table's cheapest cases at order 8, where double precision meets it: on the block at theta 20 deg, p, `R` is 0.75% low against 0.3%; on the circle in the 400 by 300 nm cell, `T` is 1.9% high.
  - Against double precision, the block at theta 20 deg, p: `R` 0.67% and the field 0.77% apart at order 8, `R` 4.2% and the field 4.1% at order 12. The default solve at single precision is 1.5e-6 to 1.7e-6 from double in `R` and 1e-5 in the field at the same orders.
  - The circle in the 400 by 300 nm cell at order 12: the field 1.2% apart at `G` 0.1, `R` 4.5% at `G` 0.05, and at `G` 0.02 the stretched half-space is refused ("does not hold its propagating plane waves as 2 modes").
  - The criteria not met, the rest was not measured: the other cases of the table, JAX and PyTorch beyond the refusal, and the readings on spurious modes.
- **Why.** The stretch raises the layer operator's norm roughly as `1/G**2`: at order 12 and `G` 0.02 it is 1.9e6 on the block and 2.7e8 on the circle, against 5.1e3 and 3.0e4 unstretched, while the propagating modes' eigenvalues stay near 1. Single-precision rounding, about 6e-8 of the norm, then moves the least-attenuated modes by up to 10%. The loss is in the solve, not its data: the tables are built in double, and a double-precision solve of tables rounded to single moves `R` and `T` by 2.6e-5 at most. Solving the eigenproblem alone in double did not help, as the operator is already formed in single precision.

## Not measured

- A reference at phi other than 0; a reference for a structure with no mirror along either axis, or for patterned layers moved by different amounts; a second reference for the metal rows on a substrate. A finer finite-element level for the weak rows above.
- An independent implementation of the stretched solve or the matched mesh: none was found to compare with at the same truncation.
- Layers with more than four walls along an axis; four walls on a substrate; orders above 20 on layers with three or more walls; why the frame's field error at order 12 is larger than the bars', and which side holds the 2.0 to 3.5% left at order 20.
- A second mesh level for five of the seven references.
- `jax.jit`, `jax.vmap` and gradients, which are refused.
- Spurious modes at slopes below 0.005 or orders above 16; wavelength features narrower than about 4 nm; the cause of the three warnings above.
