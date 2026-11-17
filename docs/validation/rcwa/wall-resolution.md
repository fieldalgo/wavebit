# The solve in stretched coordinates against finite-element fields — measurements

What [`Numerics(wall_resolution=G)`](../../guide/rcwa/wall-resolution.md) was measured against, what came out, and where it has not been checked. The option is experimental: the stretched solve has no independent implementation to be compared with at the same truncation, so everything here is either a comparison with a finite-element field, which says how accurate the method is, or a property the construction has to satisfy.

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

There is no finite-element reference here: the model above is for normal incidence at one wavelength. What can be seen is whether the stretched solve settles, and whether the unstretched one moves toward it as its order rises. The 200 nm block (or hole) again; `R` at orders 8 / 12 / 16, and for the unstretched solve also at 20.

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
- **Equal distances between the walls.** With each piece instead as wide in `u` as it is in `x`, the metal rectangle read 6.7% and 6.3% at orders 8 and 12, and on a uniform slab with two walls 37.5 nm apart the field was wrong by more than its own size; with equal distances the same slab's field was uniform to 1.5%.
- **The slope.** On the metal rectangle a slope of 0.1 read 8.6 / 4.6 / 4.1% and on the dielectric square 0.93 / 0.75 / 0.51%, between the unstretched solve and 0.02.

These three comparisons were made with a test implementation that the product code reproduces to five digits in `R` and `T` on thirteen cases.

## Not measured

- Any reference at oblique incidence, with several propagating orders, on a substrate, or on several patterned layers.
- Layers with more than two walls along an axis against a reference. A uniform slab through four unevenly placed walls reads the right `R` to 1e-5.
- A second mesh level for five of the seven references.
- Single precision; `jax.jit`, `jax.vmap` and gradients, which are refused.
- The interaction with the exceptional-point handling: a strong stretch is reported in the literature to add spurious layer modes that move away as the order rises, and none was looked for.
