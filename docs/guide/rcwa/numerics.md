# Choosing RCWA numerical settings

Physical geometry and light belong to `rcwa.Problem`; Fourier truncation and algorithms belong to `rcwa.Numerics`.

```python
from wavebit import rcwa
numerics = rcwa.Numerics(
    orders=(1, 1), algorithm="smm", factorization="nvf",
    formulation="auto", shape_nvf_grid_size=64)
```

`orders=(Nu,Nv)` retains orders −Nu…Nu and −Nv…Nv. Result arrays place the wavelength axis first, then v, then u. Algorithms are `"smm"` or `"etm"`; formulations are `"auto"`, `"first_order"` or `"second_order"`. Factorization choices are explicit, with support checked against the material and geometry. The example's small settings are for a quick calculation, not a convergence recommendation.

Rayleigh detuning is opt-in: `detuning=None` requests the exact physical input. A positive `detuning` declares a relative step and the result records the changed coordinates. Never interpret a detuned result as the unmodified requested wavelength without reading that metadata.

`topology_check='device'` uses available array guards and refuses other traced geometry.
Explicit `'host'` admits supported scalar region geometry through a JAX CPU callback;
checks execute sequentially under vmap and can limit throughput. See [checked shape
JIT](guarded-shape-jit.md) for validity masking, supported shapes and examples.

Increase orders and geometric/NVF resolution independently and compare the output you actually need. Energy balance alone does not establish optical or gradient convergence. The measurements quoted below were taken through the earlier `wavebit.compat.legacy` interface, which runs the same engine.

## What each setting changes

A handful of keywords decide how the permittivity becomes convolution matrices and how the layers are
joined. They change how fast the result converges with `orders`, and how much each
step costs -- not what it converges to. Defaults, and the values each accepts, are in
[Keyword reference](../../reference/keywords.md#the-numerics).

The NVF options name their scope: `raster_normal_method` builds a direction field from
pixels, `shape_nvf_grid_size` sets the number of shape-projector samples per cell axis,
and `nvf_projector` selects the projection construction on either NVF path. See the
[NVF keyword migration](../../reference/keywords.md#nvf-names-changed-on-2026-09-21)
when updating scripts written with `normal_method`, `normal_grid` or `projector`.

## How the coefficients are obtained, and how the products are closed

The two keywords describe different stages; supported combinations depend on
the structure and material:

| `fourier_coefficients` | |
| --- | --- |
| `'sampled'` | Discrete Fourier coefficients from material samples on a grid |
| `'integrated'` | Integrals over shape regions, or over piecewise-constant pixels for scalar rasters and tensor Li rasters |

| `factorization` | |
| --- | --- |
| `'li'` | Successive Li rules along lattice v then u (the default `li_nesting`; see below); requires alignment with Cartesian y and x |
| `'nvf'` | normal vector field -- factorizes along the material interfaces. Made for curved and tilted boundaries, where a staircase converges slowly |
| `'laurent'` | the direct rule for every product -- `[[eps]]` for both in-plane field components, its inverse for `E_z`. Opt-in and never a default: it converges slowly wherever a field component jumps with the permittivity, and is here to compare with solvers that use it, such as fmmax's `Formulation.FFT` |

**`fourier_coefficients` and `factorization` resolve per modeling path**, and
`rcwa.check(...).resolved` reports the *effective* value rather than what you asked.

| Input | Default `fourier_coefficients` | Default `factorization` |
| --- | --- | --- |
| Scalar raster | `'sampled'` | `'li'` |
| Tensor raster | `'sampled'` | `'nvf'` |
| Shapes with scalar or tensor materials | `'integrated'` | `'nvf'` |

Shape integration applies to both isotropic and anisotropic materials.
Disjoint circles, ellipses and rectangles use analytic region Fourier
coefficients. Painting resolves overlaps into polygon regions; curved overlaps
use polygon approximations. A layer holding a `Polygon` is always resolved this way,
with scalar materials only. For a scalar raster, integrated coefficients describe
the supplied piecewise-constant pixels, not a reconstructed smooth boundary.

Explicit scalar or tensor shape
`factorization='li'` supports axis-aligned rectangles wholly inside a positive
axis-aligned rectangular cell. General shapes use NVF. Naming a value the path does not implement is refused
(`UnsupportedFeatureError`, code `rcwa.unsupported`; `rcwa.check` reports it as an issue)
rather than quietly running the one it does.
`factorization='laurent'` is built on scalar rasters (either series), on any scalar shape
including painted overlaps, and on skewed cells, where Li is refused -- the rule names no axis.
Tensor materials refuse it by name. All four pairings work on a scalar
raster in an axis-aligned rectangular cell. Tensor rasters support sampled or
integrated coefficients under Li; NVF uses sampled coefficients and a linear
projector. Tensor inputs retain NVF as the default. See
[tensor Li construction](../../theory/rcwa/tensor-li-factorization.md) and
[material support](../../migration/materials-v1.md#anisotropic-materials) for tensor restrictions.

**`raster_normal_method` selects a raster normal field** under `factorization='nvf'`.
On that path, `nvf_projector` composes with all four field constructions. Shape NVF
also accepts `nvf_projector='linear'` or `'circular'`: the circular shape construction
supports rectangles, circles, ellipses and their polygon-resolved painting, and refuses a
`Polygon`.
For disjoint shapes with complex material inputs, the default piece field carries
the phase of material differences and uses a Hermitian projector. Its phase cancels
on each interface. Real material inputs keep real arithmetic; this does not select
the separate `'circular'` dyad construction.
Projector selections require NVF; `raster_normal_method` remains a raster option.
See [shape projector details](parametric-modeling.md#complex-nvf-projection-for-rectangular-contacts).

`'distance'` is the default and the least accurate of the four — on a 60-degree cell it carries
18.5 degrees of mean normal error against `'contour'`'s 0.42. Raster `'circular'` is off by default
because it is a trade rather than a fix: it removes the medial axis's arbitrariness and costs
accuracy in every cell measured, along with exact mirror symmetry. See
[`../nvf-normal-methods.md`](../../theory/rcwa/nvf-normal-methods.md).

**`li_nesting` belongs to the other rule: it is read wherever scalar Li's rules are built** -- on a
raster under `factorization='li'`, and on axis-aligned rectangles under it -- and refused where
nothing reads it: any value under `'nvf'` or `'laurent'`, and `'symmetric'` on tensor materials,
whose Li build is the fixed nesting only. A crossed grating needs Li's rule along both axes, and two
one-dimensional rules can only be applied one inside the other. `'fixed'`, the default, puts
`v` inside `u` for both in-plane components -- what every `'li'` result so far was computed
with, raster, rectangle and tensor alike. `'symmetric'` puts each component's own inverse-rule
axis inside (Li 1997), so the build treats x and y alike. Both converge to the same answer and
differ at finite `orders`; the operators are in the `Nesting` docstring.

What the fixed nesting costs is a polarization split on structures that have none. On a
C4-symmetric Au nanohole array at order 5 it transmits 0.0172 for x-polarized light
and 0.0728 for y; the symmetric nesting gives 0.0725 for both. On a transpose-symmetric
dielectric L the split is 1.5e-3 to 4.2e-3, against 1e-13 or less. Each nesting agrees with
RETICOLO's own run of it to 4.2e-11 (`tests/references/test_reticolo_li_nesting.py`). The
default stays `'fixed'` so that no earlier number moves. The measurements for choosing, and
what flipping the default would move, are in
[`../factorization-terms.md`](../../theory/rcwa/factorization-terms.md) §6.

**`'laurent'` is for comparison, not for accuracy.** It reproduces a solver that closes every
product with the direct rule: against fmmax's `Formulation.FFT` it agrees to 1e-12 on a
dielectric rod and a hexagonal cross, once fmmax's sampled coefficients are extrapolated to the
exact ones (`tests/references/test_fmmax_laurent_golden.py`). What it costs is convergence. On a
one-dimensional TM grating of eps 12.25 its error halves per doubling of the orders -- 9.0e-4
at order 80, where Li is at 4.9e-7 -- and on a 350 x 245 nm rod it is still 4.1e-3 off in p at
order 14, where Li is at 2.7e-5. TE on a one-dimensional grating is Li's answer to round-off,
because both rules build the same matrix there. It builds no normal field and nests no rules,
so `raster_normal_method`, `nvf_projector`, `shape_nvf_grid_size` and `li_nesting` are refused
beside it. The measurements and the choice of slots are in
[`../factorization-terms.md`](../../theory/rcwa/factorization-terms.md) §9.

Both connecting algorithms share the same eigenmodes and boundary conventions and agree to
~1e-14. SMM costs ~1.1-1.4x, cascades layer responses that can be reused independently, and is
the only one that returns `result.scattering`, requested with `rcwa.Outputs(scattering=True)`. See
[`../connecting-algorithms.md`](../../theory/rcwa/connecting-algorithms.md).

## Sampling the normal field

`shape_nvf_grid_size` is the grid the normal field is read off, per axis, and it is the one number here
that can dominate a differentiated step: at order 3 the default spends more per
step building the field than the solve itself costs. It defaults to
`max(1024, 4*half_width+1)`, and 1024 is not over-specification -- 256 was tried and reverted,
because it is a rectangle's *angle* that sets the grid and 256 sits 9.7% from the converged
gradient there.

Lowering it is bounded by accuracy rather than by taste. See
[`../nvf-upsampling.md`](../../validation/rcwa/nvf-upsampling.md) and [`../backend-choice.md`](../../validation/rcwa/backend-choice.md).

## Supplying an NVF energy closure

`rcwa.operators.ConvMatrices.from_energy(change, gram, longitudinal)` converts a
supplied field basis and material integrals into the five matrices that
`rcwa.operators.solve` consumes in place of assembled operators. With H retained
Fourier orders, supply Q = C(B) and
G = C(B.H epsilon B) with shape `(layers, 2*H, 2*H)`, and C(epsilon_zz)
with shape `(layers, H, H)`. All inputs must use the solver's Cartesian
frame and Fourier order layout, with all x-component orders followed by all
y-component orders in Q and G. Unbatched inputs are also accepted, but
`operators.solve` requires a layer axis. Supply Fourier coefficients about the cell centre:
`from_energy()` preserves its inputs' origin and applies no additional translation.
This five-slot closure assumes no xz, yz, zx or zy material coupling.

The builder computes `Q^(-H) G Q^(-1)` by solves, inverts the longitudinal
matrix and supplies the off-diagonal signs. It preserves complex material
loss and NumPy/JAX/PyTorch array precision and derivatives. Shape mismatches
and nonfinite eager inputs are refused; invalid traced solves produce NaNs.
Supplying B and resolving its approximation remain the caller's responsibility.

This complete example uses a uniform dielectric layer with B equal to the identity:

```python
import numpy as np
from wavebit import Material, Units, rcwa
from wavebit.rcwa import operators

order = 1
count = (2*order+1)**2
change = np.eye(2*count, dtype=complex)[None]
gram = 4.0*change
longitudinal = 4.0*np.eye(count, dtype=complex)[None]
conv = operators.ConvMatrices.from_energy(change, gram, longitudinal)
numerics = rcwa.Numerics(orders=(order, order))
transmitted = []
for polarization in ('p', 's'):
    problem = rcwa.Problem(
        units=Units(length='nm'),
        stack=rcwa.Stack(
            period=(700., 700.), top=Material(epsilon=1.), bottom=Material(epsilon=1.44),
            layers=(rcwa.Layer(thickness=177., background=Material(epsilon=4.)),)),
        excitation=rcwa.Excitation(wavelength=830., sources=(rcwa.PlaneWave(
            side='top', polarization=polarization, theta=.23, phi=.37),)))
    # The supplied closure replaces the assembled layer operators.
    assembled = operators.assemble(problem, numerics=numerics).replace(epsilon=(conv,))
    transmitted.append(operators.solve(problem, assembled).T.sum())
print(*transmitted)  # p, then s
```

[`rcwa.experimental.ConformingNVF`](conforming-nvf.md) constructs this standard layer
stack directly from ordinary rectangle layer dictionaries. It supports
nonoverlapping rectangle sequences along either axis of an orthogonal cell, with
different directions allowed per layer, live materials, translations, both
physical periods, common lattice rotation, and internal background intervals
that can open or close during JAX replay. Passive complex materials are
supported with independent loss in every shape and background, provided
the real parts are positive and the imaginary parts nonnegative. Integer
rebasing supports nonorthogonal basis vectors that describe the same
orthogonal point lattice, preserving the original retained Fourier set. The general
energy closure API also accepts externally constructed fields. Arbitrary
shape correctors remain outside the rectangular builder's scope.
