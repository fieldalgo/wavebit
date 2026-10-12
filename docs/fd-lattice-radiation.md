# Radiation consistent with the Yee lattice

Written: 2026-09-24

Targets:
- `src/wavebit/fd/_radiation.py`
- `src/wavebit/fd/monitors.py`
- `src/wavebit/fd/results.py`
- `src/wavebit/fd/_lower/monitors.py`
- `tests/test_fd_radiation.py`
- `tests/references/lattice.py`
- `studies/fdtd_subpixel_lattice_radiation.py`
- `studies/results/fdtd_subpixel_lattice_radiation.log`

## Summary

The continuum radiation map is inconsistent with grid propagation in the C3
weak-scattering phase test. A derived lattice map removes that discrepancy.
It changes neither a measured field nor a material coefficient. It includes
three separate effects: the outgoing phase follows a lattice dispersion surface,
the reciprocal magnetic wave must be averaged on the same two normal planes,
and that dispersion surface supplies a direction-dependent Green coefficient.

`FarField(propagation='yee')` selects this map. The original `continuum`
default remains reproducible, including the stage 0--3 reference arrays.
Both definitions have the same continuum limit; their finite-grid values
are different. The new map does not make material-interface discretization
exact or certify C3 as complete. [Usage](guide/fd-radiation.md).
The continuum path issues one named `RadiationApproximationWarning` per
scene at preparation, also listed by `fd.check`, with the measured weak-sphere
residuals and their limited scope. It does not silently imply energy closure.

## Derivation

Take the Fourier convention exp(-i omega t) and a uniform scalar background
epsilon, mu > 0. Let h be the spatial step and dt the time step. The symbols
of the centered Yee differences are

```
Omega = 2 sin(omega dt/2)/dt
K_j   = 2 sin(k_j h/2)/h
|K|^2 = Omega^2 epsilon mu.
```

The observer ray rhat is parallel to the spatial dispersion gradient
v_j = K_j cos(k_j h/2) = sin(k_j h)/h. The electric polarization is
perpendicular to K, which generally is not parallel to rhat.

Write s = Omega sqrt(epsilon mu) h and sin(k_j h) = s u rhat_j.
The root equation used in preparation is

```
sum_j 2 u^2 rhat_j^2 / (1 + sqrt(1-(s u rhat_j)^2)) = 1.
```

Scaling the unknown by s avoids subtracting almost equal numbers as s tends
to zero. The implementation requires omega dt < pi and 0 < s < sqrt(2).
The latter keeps the entire outgoing surface inside |k_j h| < pi/2, before
its first inflection. Requests outside this scope are refused for the whole
scene during planning. Root and angle validation also precede each requested
angular transform. The numerical field loop contains no refusal branches.

### Reciprocal surface functional

For a backward reference plane wave polarized along e perpendicular to K,

```
Er = e exp(-i k dot x)
Hr = -K cross e / (Omega mu) exp(-i k dot x).
W(e) = integral_h (Es cross Hr - Er cross Hs) dot n.
```

Discrete summation by parts puts tangential E on the face node plane and
averages each H on its two adjacent normal half planes. This applies to
both the recorded field and the reciprocal plane wave. The reference H
therefore contributes cos(k_P h/2) on each face of normal axis P.

Let N integrate n cross Hs against exp(-i k dot x), with the recorded H
average. Let L integrate -n cross Es against the same phase, additionally
weighted by cos(k_P h/2) separately on each face. With q = K/|K| and
eta = sqrt(mu/epsilon), the transverse functional is

```
Wvec = N - q (q dot N) - q cross L / eta.
```

Changing only continuum k leaves the missing reciprocal-plane factor and
the remaining dispersion geometry unresolved. The old adjusted-k experiment
in the phase diagnostic remains an intentionally incomplete control.

### Outgoing lattice Green function

The scalar outgoing Green function is the inverse Fourier transform of
1/(sum_j K_j^2 - Omega^2 epsilon mu - i0). At the stationary point with
outgoing ray rhat, its large-distance coefficient is

```
G(R rhat) ~ A exp(i k_star dot R rhat)/(4 pi R)
d_j = cos(k_j h)
A = 1/sqrt(rhat_x^2 d_y d_z + rhat_y^2 d_x d_z + rhat_z^2 d_x d_y).
```

This follows from the pole residue and the Gaussian curvature of the
dispersion surface. Its gradient is 2v and its Hessian is
2 diag(d_x,d_y,d_z); their factors cancel into A as written. Thus

```
F = i Omega mu A Wvec / (4 pi)
E(R rhat) ~ F exp(i k_star dot R rhat)/R
U = g |F|^2/2,    g = |v|/(Omega mu).
```

U is power per ray solid angle. g is the staggered energy-flux weight,
not the time-domain group velocity. The latter also involves the temporal
cosine. F may have a radial component relative to rhat; dropping it when
squaring two angular projections is incorrect. The public `vector()`
retains it, and `intensity()` always uses the full Cartesian norm.

### Normal-incidence optical theorem

For a +z vacuum incident wave of electric amplitude a, A=1 and
g=cos(k_z h/2). Discrete interference power obeys
Pext = Re(conj(a) W_x)/2. With f=F_x/a this gives

```
sigma_ext = 4 pi Im(f)/(Omega g).
```

This relation is used by the sphere study when `propagation='yee'`.
It is derived from the reciprocal functional and the same incident power
used for scattering normalization. A measured returning incident wave
contributes its own interference term; it is retained separately in the
phase-control report, never fitted into f. The general-background ray
transform is tested, while this study's optical theorem is specifically
the normal-incidence vacuum relation above.

## Independent controls

Measurements used the Windows x86-64 CPU host, float64/complex128, two
numerical-library threads, and NumPy, JAX eager/compiled, and Torch.

| Control | Measured discrepancy before selecting regression budgets |
| --- | ---: |
| 304 directions, eight s values from 1e-9 through 1.4: dispersion residual | <=9.55e-15 |
| Same ray-direction residual | <=3.34e-16 |
| Explicit pointwise reciprocal sum vs separable implementation, three frequencies, epsilon=2.25, mu=1.3, 24 directions, four execution modes | <=2.78e-16 absolute field |
| Two arbitrary regular grid plane waves through a closed surface | <=2.85e-16 absolute functional |
| Independent Fourier-pole Green integration vs predicted asymptotic A | <=3.37e-11 relative |
| Field-scale derivative vs the exact quadratic-power identity, JAX compiled/Torch | <=4.45e-16 relative |

The Green control integrates the imaginary part of the Fourier pole using
K=Omega nhat. The Jacobian is product_j cos(k_j h/2)^(-1).
At 32 radii from 12 to 25.3 wavelengths, the sine/cosine expansion through
1/R^3 recovers the leading coefficient. Quadrature orders 128, 256 and 512
agree; the residual is at most 4e-13. Axis, diagonal and oblique rays give
A=1, 1.016712795656 and 1.008672817511. This oracle uses no surface transform.

Fresh public electric Ez and magnetic Hx point sources supply a separate
complex-field control. Their cubic B-spline current stamps and raw time
drives predict radiation directly. The prediction uses no monitor field.
Sources are displaced by (.013,-.02,.007); two boxes have sides .5 and .7.
At 20 cells/wavelength and 14 periods:

| PML cells | Largest complex amplitude/phase error over both sources and boxes | Largest angular-integral/box-flux discrepancy |
| ---: | ---: | ---: |
| 12 | 7.053e-8 | 1.131e-5 |
| 24 | 1.102e-7 | 1.358e-8 |

The boundary-thickness control locates the earlier power floor in boundary
sensitivity. It does not certify every PML or source. The 24-cell physical
regressions use 5e-7 complex peak and 1e-7 power budgets. Their largest late
monitor energy ratio is 5.10e-11, below the 1e-9 regression ceiling.

## Weak-sphere results

The original epsilon=1.1, radius=.25 paired phase records are reprocessed
without new field evolution. Raw-array digests are checked before use.
The central surface has half-width .75; neighboring ones are .65 and .85.

| Cells/wavelength | Historical continuum optical residual | Derived lattice optical residual, central surface | Largest absolute lattice residual, three surfaces | Largest angular-cut peak error vs continuum Mie |
| ---: | ---: | ---: | ---: | ---: |
| 20 | 52.04750% | +.0264028% | .0285778% | 2.00892% |
| 28 | 26.37983% | -.0433575% | .0457202% | 1.00844% |
| 40 | 12.91384% | -.00357156% | .00506247% | .488822% |

Angular cuts are divided by the discrete incident intensity, just as
scattering power is. For the standard three-grid cases, the angular power
integral differs from box power by at most 8.40e-6. Doubling both angular
orders (24/32 to 48/64) changes it by at most 3.34e-15 relative.

At 20 cells, extending 24 to 48 periods changes central optical residual
from .0264028% to .00664945%. Changing the PML reflection target from
1e-8 to 1e-12 increases the measured returning illumination and changes it
to -.0805697%, with a largest three-surface magnitude .0875149%.
This sensitivity remains visible. No fitted phase or wave-number adjustment
is applied. The original 52%/26%/13% values remain in the historical record.

A fresh public 20-cell execution gives optical residual .0267422% and
Qsca=.0075230014462, with the old material operator unchanged. Its slightly
different residual uses the public scattered field directly; the paired
replay subtracts the independently recorded empty field. Both satisfy the
unchanged G6 optical gate of 1.6%. The stricter 5e-4 regression budget is
chosen from this fresh .000267422 and the measured .000197434 duration change.

## Limits and the material-boundary decision

These checks validate the outgoing map in a positive isotropic background
on its convex first band. They do not remove a sphere's discretization error,
prove accuracy near an unresolved material interface, or support derivatives
of background/angles. Field derivatives are supported. Legacy continuum
outputs and all frozen reference data are intentionally preserved.

A separate attempted local energy reconstruction was rejected before any
scattering sweep. For a two-component local face map Pi and continuous
interface variables (tangential E, normal D), it used
Pi^-T Q Pi^-1, with Q the exact electric energy in the node-centered dual
square. This is symmetric positive, but the shared volume is not the
edge quadrature that the update needs. At epsilon=12 and an axis-aligned
interface displaced by .23h, its normal-E error was .12375 at 16, 32 and
64 cells per unit length, versus <=1.95e-16 for the edge/face comparator.
For normal (.6,.8), its tangential-field error reached .63131 versus .22595.
Positive energy alone is insufficient. This candidate is not in the engine.

The production cylinder's previously measured refinement orders
1.60337/1.58655 still fail the 1.8 gate. The 3D symmetric-triplet candidate
still has a 3.77084% small high-index sphere power error against the existing
3% control. Those are material tests, unaffected by a radiation postprocess.
C3 therefore remains incomplete. A boundary reconstruction must demonstrate
both consistency and stability before it can replace the current operator.

### Basis for the next boundary experiment

[Werner, Bauer and Cary (2013)](https://arxiv.org/html/1212.4857)
identify the same tradeoff in their sharp-interface tests: their accurate
static triplet map is asymmetric and unstable, while its stable symmetric
version ultimately has first-order error. This supports rejecting another
local symmetric tensor solely on an energy argument; it does not prove
that every possible interface method must fail.

[Law and Nave (2022)](https://arxiv.org/html/2101.05417v2) reconstruct smooth
fields separately on both sides of an interface using Maxwell equations,
jump conditions and nearby time-domain data. They report high-order 2D
examples and long-time stability evidence, without a general stability proof.
Their paper is a research direction, not a certificate for this 3D engine.

The next candidate must first reproduce static interface fields and pass an
assembled energy/stability check. It must then use the same material update
in the eight-placement cylinder and the 3D sphere controls. The existing
order-1.8 and small-sphere 3% gates stay fixed. A favorable cylinder result
from one operator cannot be combined with sphere results from another.

## Reproduction and regression

```
python studies/fdtd_subpixel_phase_control.py phase20.json --cpw 20
python studies/fdtd_subpixel_lattice_radiation.py radiation.json phase20.json
python studies/fdtd_subpixel_mie_sphere.py --cpw 20 --eps 1.1 --propagation yee fresh20.json
python -m pytest tests/test_fd_radiation.py -p no:cacheprovider --run-slow -q
```

Use fresh output names. The accompanying log contains source hashes, raw
measurement summaries, actual exits, test collection/report counts and
source-mutant witnesses. Validation completion is recorded there rather than
inferred from a test runner's progress bar.

### Integrated verification

The full fast tier collects and reports 1,958 cases: 1,952 pass and six
expected NumPy differentiation cases skip. The full slow tier collects and
reports 78 cases, all passing. Both exit 0, without `INTERNALERROR` or source
changes during execution. An earlier fast run failed only on two analytic
fake-result fixtures missing propagation metadata and the old stage-3
warning expectation; their metadata was updated, and numeric gates stayed fixed.

Follow-up review measured a 1.40182e-7 wavevector discrepancy when validated
float32 scalar inputs were still used in arithmetic. Preparation now promotes
their same representable values before arithmetic. The NumPy and Torch scalar
controls require exact agreement with those values supplied as Python floats.
Unsupported traced Torch angles produce a named host refusal. The sphere CLI's
auxiliary phase-control surfaces now inherit its requested propagation and
identify that propagation in each diagnostic row.

After these small follow-ups, all 90 selected physical, backend, derivative,
instrument, warning and document cases pass, with matching collection/report
counts and exit 0. Fresh stage 0/1/2/3 checks reproduce 143/288/140/570 arrays
bit for bit, 1,141 total, each exiting 0. Stage 3 notes the changed case-file
provenance digest caused by its warning expectation; its numeric arrays and
reference archive are unchanged. No reference is regenerated.

Five successful source-mutation audits detect 24 distinct defects through
52 assertion failures in 25 mutated invocations. Every selected collection
matches its report, every mutation is caught by assertions rather than setup
errors, and every file is restored byte for byte. One earlier baseline that
overlapped a deliberate mutation and one audit with temporary-directory setup
errors are explicitly excluded and retained in the log. These regressions
validate this radiation change; they do not waive the remaining C3 material gate.
