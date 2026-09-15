# Fixed local transmission potentials for the pair cell

Written: 2026-09-16

Targets:

- `wavebit/_core/conforming/cartesian_pair_transmission_galerkin.py`
- `wavebit/_core/conforming/pair_transmission_fields.py`
- `wavebit/_core/conforming/pair_transmission_modes.py`
- `tests/test_nvf_pair_transmission.py`
- `studies/nvf_pair_angular_resolution.py`

The explicit NumPy transmission family supplies a separate finite trial space
whose corner directions satisfy a fixed reference transmission equation.
It preserves the original complete tensor and physical guards. Measurements
still depend on the number of modes, polynomial degree and mesh refinement;
this development does not establish optical convergence or replace the
existing fixed-power family.

| Choice | Meaning |
| --- | --- |
| Reference coefficients | Fixed left, right and background values defining the trial functions |
| Physical coefficients | Live materials passed to `closure`, independently of the reference |
| One through four branches | Successive noninteger growing transmission roots at all six corners |
| Real directions | Both real and imaginary parts of complex modes; one direction for a real mode with real relative coefficients |
| Common coordinates | A continuous Pp interpolation subtraction and complete real QR within each corner group |

## Local equation and fixed basis

For a scalar reference coefficient `a`, a separated potential
`u(r, theta) = r**lambda * f(theta)` satisfies `div(a*grad(u)) = 0`
inside each constant sector when `f'' + lambda**2*f = 0`. Across an interface,
both `f` and `a*f'` are continuous. Propagating the state
`(f, a*f'/lambda)` through an angle `t` therefore uses

```text
M(a, t) = [[cos(lambda*t),       sin(lambda*t)/a],
           [-a*sin(lambda*t),   cos(lambda*t)  ]].
```

For quarter sectors `a`, `b` and a half sector `c`, the noninteger periodic
roots obey the characteristic obtained by multiplying these three matrices:

```text
A = a/b + b/a
B = (a+b)/c + c*(1/a + 1/b)
cos(pi*lambda) = -(4+B)/(2+A+B).
```

The implementation uses the principal growing root and the successive roots
`2-lambda`, `2+lambda`, `4-lambda`. A two-component null vector of the full
monodromy determines the angular coefficients; a separate closure check
verifies continuity after the complete turn. A nonpositive real part is
rejected because the local gradient energy is not integrable. Equal quarter
coefficients give a straight interface with no noninteger corner direction.
No numerical rank or field-strength threshold chooses the surviving modes.
The sector coefficients are divided by the reference background before
constructing the modes. This common factor leaves the equation unchanged and
preserves the real trial bank under a purely imaginary multiple of real
reference materials, without introducing an identically zero direction.

Corner singularities for sign-changing coefficients have an established
literature, including [Bonnet-Ben Dhia, Carvalho, Chesnel and Ciarlet](https://arxiv.org/abs/1511.01932).
That paper's lossless scattering and corner radiation problem is distinct
from this lossy auxiliary-potential cell. Its critical-contrast conclusions
are not used to declare a divergent derivative here. The equations above
and their implementation were derived directly; no external solver source
or new dependency is integrated.

For reference `(12.25, -10+i, 1)`, the leading roots are approximately
`.720879888` at the dielectric outer corners,
`.582592739-.009794237i` at the metal outer corners and
`.914430391-.517977968i` at a material junction. Their positive real parts
permit finite local energy. These local numbers are not a global error bound
or an optical stopping rule.

## Common field and derivative

Each mode is multiplied by its closed corner hat. The same reference
function is transported with the fitted mesh as the gap opens. Its common
polynomial interpolant is subtracted before the complete real coordinate
change. The cell retains all selected directions in its weak equations,
normal traces, Fourier moments, response and analytical variation.

The real trial space is fixed when physical materials change. In particular,
the implementation does not recompute the reference exponent during a
material JVP. Geometry is fixed in that JVP, as in the original pair family.
Spatial gradients and Hessians differentiate the defining local function;
the Hessian combines homogeneity with the harmonic equation. Moving
barycentric coordinates and the inverse triangle map supply the transported
gradient variation.

At an interface, the triangle selects the limiting material sector.
Selecting a side only from a boundary point's rounded polar angle is wrong:
the potential is continuous but its normal gradient has different limits.
In an independent order-four cubic prototype, correcting this choice reduced
the maximum full derivative reflection defect from about `24.36` to
`2.92e-12`, without changing any response check.

Independent spatial differences verify all four local branches and complete
cubic/quartic field transport on the refined mesh. Whole contact controls
retain reflection, separate angular integration and length-unit identities
at absolute `3e-10` / `3e-9`. At positive gap `.02`, a two-branch order-zero
cell with physical materials `(4+.1i, 9+.2i, 1+.02i)` and the distinct fixed
reference `(12.25, -10+i, 1)` also passes an independent fourth-order joint
gap/material difference at `3e-9`. That test evaluates five complete physical
states and took 522 seconds on the stated runtime.

A separate metallic one-branch control evaluates exact contact and physical
gaps `.01`, `.0001`, `.000001` at period 700 and Fourier order zero. Every
state passes the original guards. Values approach the same contact response
and the complete derivative differences decrease across all three gaps. This
four-state test took 461 seconds; it assigns no universal rate or error bound.

## Integration and measured optical limits

The reference space requires angular integration of nonpolynomial functions,
including on thin exterior triangles away from the corner itself. At one
mesh refinement, angular Gauss orders 24 and 32 produced full value reflection
defects `3.59e-7` and `9.08e-9`; both fail the original `3e-10` criterion.
The error was localized to four narrow exterior triangles. Order 64 passed
with value and derivative defects below `8.67e-13` and `5.82e-12`.

Reflection alone also misses common integration errors. For the unrefined
one-branch space, increasing angular order from 48 to 64 changed the entire
value and derivative by `1.54e-9` and `1.12e-8`, exceeding the original
`3e-10` / `3e-9` criteria. The subsequent 64/80 comparison passed, with
differences `2.24e-12` and `1.53e-11`. Radial order was 80 throughout.

The following contact measurements use period 700, widths and height 300,
physical and reference materials `(12.25, -10+i, 1)`, Fourier order `(4,4)`,
wavelength 900, layer thickness 300 and incidence angle `.2`. The runtime
was NumPy 2.5.1 with one numerical-library thread. Slopes are per caller gap
unit, obtained by differentiating the optical solver along the analytical
constitutive JVP. They are not independent physical gap finite differences.

| Branches | Degree | Refinement | Angular Gauss order | p/s transmission | p/s optical slope |
| --- | --- | --- | --- | --- | --- |
| 1 | 3 | 0 | 80 | `.48581215` / `.46831714` | `-.04935045` / `.01254798` |
| 1 | 3 | 1 | 64 | `.48964782` / `.47673311` | `-.01293277` / `.03202701` |
| 2 | 3 | 0 | 64 | `.48483934` / `.46813181` | `-.02432054` / `.01957287` |
| 2 | 4 | 0 | 64 | `.48699404` / `.47283323` | `-.02736622` / `.01666854` |
| 2 | 4 | 1 | 64 | `.48911480` / `.47894903` | `-.04408189` / `.01018199` |
| 4 | 3 | 0 | 64 | `.48475992` / `.47363551` | `-.00076895` / `.03447941` |
| 4 | 4 | 0 | 80 | `.48583765` / `.47515927` | `-.06005934` / `.02990420` |

The 80-point quartic, four-branch row was evaluated by the actual library
and study entry point in 107 seconds. It passed every original physical and
full reflection check. After the known Fourier translation to the same
origin, its complete value and derivative differed from the independent
64-point prototype by `1.95e-12` and `1.25e-10`. The other rows are finite
prototype measurements, with complete physical and reflection checks. Raw
measurement records are not distributed. These changes across trial spaces
remain unresolved, even when individual finite equations are accurate.

The refined quartic two-branch prototype took 1,099 seconds and passed the
original guards, with full value and derivative reflection defects below
`3.76e-12` and `8.78e-11`. Its transmission and slope still change substantially
from the unrefined space. It does not close the approximation-convergence gate.

Reproduce the library measurement with

```shell
python -m studies.nvf_pair_angular_resolution --order 4 --degree 4 --refinement 0 --quadrature 80 --angular-quadrature 80 --transmission-branches 4
```

The study records source digests, full diagnostics and both polarizations.
Its optional tensor artifact preserves every value and derivative slot
before applying the unchanged strict reflection checks.
