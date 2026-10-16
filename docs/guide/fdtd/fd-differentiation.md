# Differentiating the general FDTD engine

> **Historical API:** snippets and descriptions using `wavebit.compat` on this page require a historical checkout.
> The namespace is absent from current installations; the pre-retirement baseline is
> [revision `3c8a50f8`](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a).
> Reproduce recorded experiments with their own cited commit and environment; their measurements have not been rerun.


:::{note}
This page retains the earlier engine declarations and their measured admission rules. The pillar scene now has a [complete current-API example](substrate-gradients.md), using an explicit smoothed substrate continuation through the bottom open face. The current module runs the same engine and checks; [In the current module](#in-the-current-module) maps the declarations below.
:::

`wavebit.compat.legacy.fd` runs on NumPy, JAX and PyTorch, and on the last two a run is
differentiable: material values, a body's shape and position, a source's amplitude, phase,
delay and band, and the Bloch phase can all be leaves of `jax.grad` or Torch autograd. This
page says how to set that up, what may be traced, and what the run does when a traced value
leaves the region it can certify. The measurements behind it are the C6 sections at the end of
[`fd-subpixel-occupancy.md`](../../validation/fdtd/fd-subpixel-occupancy.md), the last of them "Traced
material values beside traced geometry".

## The one rule: plan concrete, run traced

A run is `fd.plan` (static, on the host: the grid, the boundaries, every refusal) followed
by `fd.lower` and the time stepping (the arrays a gradient flows through). Build the plan
**once, from plain numbers** -- `plan = fd.plan(scene)` -- and pass it to every traced run
as `fd.run(traced_scene, 'torch', plan=plan)`, or `'jax'`.

The plan certifies what it can only check on numbers -- the smoothed operator's stability,
how the junction under a pillar is laid out, which orders the absorbers can take -- at the
values it was built from. The traced run reuses it, reads the live values, and checks the
ones that can move at run time (see [When the result is NaN](#when-the-result-is-nan)). On
JAX the plan is also what makes the compiled stepping program reusable: one compile per
plan, however many values you try.

### In the current module

`wavebit.fdtd` keeps the rule under its own names. The plan is
`prepared = fdtd.prepare(problem, numerics=numerics, backend='torch')` (or `'jax'`), built
from plain numbers, and the traced run is `prepared.run(traced_problem)`, through which the
gradient flows. A body's reserved cells are `fdtd.Numerics(geometry_ranges={'pillar': RANGE})`,
keyed by the structure's name, instead of `range=`; a material leaf is `Material(epsilon=...)`;
the transmitted zero order is `result['T'].efficiency[0, 0, 0]`; and a run a runtime check
rejected has `result.validity.checks['runtime_stability']` false. A traced `Bloch` phase or
`Axis1D` `k` is not live in a prepared calculation yet: `prepared.run` refuses it as a static
field.

## A pillar's radius, height and index together

A metasurface cell: a glass substrate (a smoothed `fd.Box` running through the bottom PML)
and a cylindrical pillar standing flush on it, under a normal plane wave. The pillar's
radius, height and permittivity are the design variables; the objective is the transmitted
zero order. The scene and its plan, on any backend:

```python
# Historical engine declaration; see substrate-gradients.md for the physical API.
from wavebit.compat.legacy import fd

PLANE = -0.2375                                   # the substrate's top face
GLASS = fd.Material(eps=2.1)
RANGE = ((-.145, .145), (-.145, .145), (PLANE - .001, PLANE + .401))


def cell(radius, height, eps):
    pw = fd.PlaneWave('pw', z=.45, direction='-z', polarization='s',
                      band=fd.Band(wavelength=1., df=.15))
    pillar = fd.Cylinder(radius=radius, height=height, axis='z',
                         center=(.011, -.007, PLANE + height / 2), range=RANGE)
    return fd.Scene(
        domain=fd.Domain(size=(.6, .6, 2.1), center=(0., 0., .05)),
        boundaries=fd.Boundaries(**fd.Bloch.from_source(pw), z=(
            fd.PML(cells=8, wavelength=1., background=GLASS, allow_thin=True),
            fd.PML(cells=8, wavelength=1., allow_thin=True))),
        resolution=fd.Resolution(dx=.05), run=fd.Run(steps=600, courant=.5),
        sources=(pw,),
        structures=(fd.Structure(fd.Box(size=(10., 10., 2.), center=(0., 0., PLANE - 1.)), GLASS),
                    fd.Structure(pillar, fd.Material(eps=eps))),
        monitors=(fd.Diffraction('T', z=-.55, side='transmit', wavelengths=(1.,), orders=0),))


plan = fd.plan(cell(.12, .3, 2.25))
```

With PyTorch:

```python
import torch                                      # earlier interface, as above

leaves = [torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (.12, .3, 2.25)]
result = fd.run(cell(*leaves), 'torch', plan=plan)
T = result['T'].orders.incident.T[0, 0, 0]
T.backward()
print(float(T), [float(v.grad) for v in leaves])
```

With JAX; enable 64-bit numbers before making any array:

```python
import jax                                        # earlier interface, as above
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp


def transmitted(v):
    result = fd.run(cell(v[0], v[1], v[2]), 'jax', plan=plan)
    return result['T'].orders.incident.T[0, 0, 0]


print(jax.grad(transmitted)(jnp.array([.12, .3, 2.25])))
```

Both print the same gradient (radius `0.1933`, height `-0.0818`, eps `0.00625`).

Three things in that scene are what make the shape traceable:

* **`range=`** reserves the cells the pillar may occupy anywhere in the sweep. Every size
  and position the run is asked about must keep the whole body inside it.
* **The height moves with the centre**: the centre is `PLANE + height / 2`, so the base
  stays on the substrate's face. A pillar whose base leaves the face is not the pillar the
  plan's junction was built for.
* **The substrate's shape stays concrete.** Its permittivity may be traced as well -- pass
  `fd.Material(eps=...)` with a leaf to the first structure -- although it fills the bottom
  PML: the rule that keeps a PML from being asked to absorb an order leaving too steeply is
  held at run time (see below).

## An inactive Lorentz term in the analytic response

For finite inputs, `fd.Lorentz(..., strength=0).eps_term(omega)` contributes zero,
including at its undamped resonance. `fd.eps_at` inherits this convention when it
sums the poles: an exactly inactive oscillator contributes no susceptibility, so
its otherwise indeterminate `0 / 0` is removed before evaluating the denominator.
This shortcut requires a concrete, exactly zero strength; it is not a tolerance
that removes small nonzero strengths. The current `wavebit.materials.Lorentz.epsilon_term` implements the same inactive-pole convention with SI rates and strength; invalid traced parameters still produce NaN.

A strength being differentiated may have a value of zero without being inactive.
Away from resonance, for `D = omega_0**2 - omega**2 - 1j*gamma*omega`, its derivative
is `d(eps_term)/d(strength) = 1/D`. That dependence is retained for traced strengths,
including forward-mode tangents; a zero value alone must not detach it. Conversely,
with a concrete inactive strength, derivatives with respect to finite frequency,
resonance and damping remain zero. The undamped resonance of an active or
strength-traced oscillator remains singular; this convention supplies no finite
regularization or derivative there.

This is the analytic material response. It does not change the time-domain pole
recurrence, its stability policy, or which material and geometry derivatives a
planned `fd.run` admits.

## What may be traced

| Leaf | Where | Notes |
| --- | --- | --- |
| `eps`, `mu` of a plain `fd.Material` | any structure, the background | One scalar each (`eps=x`, not `eps=(x, y, z)`). Not a material filling a stabilized PML layer (`PML(stabilize=...)`) along a PlaneWave's axis in a scene with a Diffraction monitor. With traced geometry or beside a Sphere, Cylinder, Prism or turned Box: not the background when a PML is matched to it or the scene has a PlaneWave or TFSF source or a `FarField(propagation='yee')`. A flush junction's media -- the pillar, its face, the background -- name no `mu`, traced or not (only electric junctions are measured). |
| `sigma` of `fd.Conductor`, pole parameters of `fd.Dispersive` | structures | Refused with traced geometry or beside a Sphere, Cylinder, Prism or turned Box when run with `fd.run`, where they have no certificate. |
| `radius`, `center` | `fd.Sphere(range=...)` | |
| `radius`, `height`, `center` | `fd.Cylinder(range=...)` | `axis` stays static. |
| `size`, `center` | `fd.Box(range=...)` | `angle` and `axis` stay static. |
| `vertices`, `height`, `center` | `fd.Prism(range=...)` | The vertex count, `angle` and `axis` stay static. |
| amplitude, phase, delay, `Band(wavelength=, df=)` | sources | A traced band skips the plan's cells-per-wavelength check. |
| Bloch phase or `k` | `fd.Bloch`, `fd.Axis1D` | |

Several ranged bodies may be traced in one scene when their reserved ranges, each widened by
one interpolation cell, do not share a cell; a structure without a range must stay a cell
clear of every traced body's widened range, except the substrate a traced pillar stands on.
A ranged body's material must be isotropic and lossless (`fd.Material`, one number for each
of `eps` and `mu`). A partner's material -- a film, a block, the face a concrete pillar
stands on -- may be traced too, within the rules above.

A smoothed structure with **concrete** geometry may have its `eps` or `mu` traced too, with
the same rule: plan from concrete values, pass `plan=`. For a Sphere, Cylinder, Prism or
turned Box -- bodies whose smoothing couples field components -- `plan=None` is refused,
because the plan's stability certificate needs numbers; an unturned Box and the cell-snapped
`Slab` and `Region` plan a traced value as they always have.

## When the result is NaN

A traced value can reach a value the plan would have refused. The run then does not raise --
under `jax.grad` there is nothing to raise on -- but returns `nan` fields, results and
gradients, and `result.diagnostics['stable']` reads `False`. The runtime checks:

| Check | Fails when | The concrete path's refusal |
| --- | --- | --- |
| Range | a traced body leaves its `range=` or falls under four cells across | "cells across" |
| Base on the face | a traced pillar's base leaves the substrate's face (beyond round-off) | not flush |
| Positivity | a traced `eps` or `mu` is not positive (or is `nan`) | `Material(eps=...)` |
| Time step | the smallest index puts `dt` past the stability bound | "stability bound" |
| Lifted operator | a star of the smoothed operator is not positive definite at its lift | constitutive certificate |
| Resolution | a traced medium falls below 8 cells per wavelength at the band's shortest or any `Diffraction` wavelength; the report wavelength also refuses a metal with less than one cell per skin depth | "cells per wavelength" or "skin depth" |
| Grazing order | a traced medium filling a PML layer along a PlaneWave's axis lets an order a `Diffraction` monitor asks for leave at an exit angle the layer does not accept | "Grazing order refused" |

The grazing check runs for every traced plain material in a judged PML layer, including
cell-snapped Slabs and scenes without smoothed bodies. A fixed PML reference medium fixes
the stretch profile, not the physical index used to judge outgoing orders. The resolution check runs for every traced material, including conductor and pole
parameters on the admitted concrete geometry. It reads the same complex index as the eager
check: choose the axis with largest `abs(eps(w) * mu)`, then the passive square root.
Dispersion is evaluated at the wavelength, rather than using `eps_inf` as the index.
Conductivity contributes only in the existing eager conductive-film scope; this does not
expand which conductor geometries may be differentiated. Metal-like media (`Re(n) < abs(Im(n))`)
use skin depth at the report wavelength instead of the eight-cell rule. With no sources,
that wavelength comes from `Resolution(wavelength=)` or the first planned PML wavelength;
with neither, eager and runtime checks both have no wavelength to judge. The report also
keeps the eager exemption for a source-free homogeneous vacuum; a Diffraction wavelength
still has its separate eight-cell check.

For a plain `fd.Material`, the check includes a traced `eps` on a Slab, a Region or an
unturned Box with concrete geometry, where the plan can only warn that it could not count the cells. At `dx = 0.05` and the band above
(shortest wavelength 0.642), the floor admits `eps * mu` up to
`(0.642 / (8 * 0.05))**2 = 2.58`: a pillar index past 1.6 needs a finer grid. In the same
cell (period 0.6, 8-cell PMLs) the substrate's first diffracted orders begin to propagate
at wavelength 0.95 once its `eps` passes `(0.95 / 0.6)**2 = 2.507`, just above the
horizon, which no PML absorbs: from there the run is `nan`, as the concrete scene is refused. If an optimizer's step returns `nan`, step back
and bound the variable; the run's report (`result.report.warnings`) names the checks a
traced value is held to.

## What is refused, and why

These are refused by name when the scene is run, because no runtime check could stand in
for them. The stabilized-PML row applies even without smoothed bodies; its exact-return
survey has no runtime derivative check. The other rows apply with traced geometry or beside
a Sphere, Cylinder, Prism or turned Box
(a direct `fd.lower` of concrete geometry skips the conductivity row, and reads a seed only to compare how `mu` was named):

| Refused | Why | What to do |
| --- | --- | --- |
| A traced `eps` or `mu` given per axis | the smoothed operator is built for one isotropic number | trace one scalar |
| A traced background where a PML is matched to it (no `background=` of its own), or with a PlaneWave or TFSF source or a lattice `FarField` | the PML match, the plane-wave source's transverse wavevector, the grazing survey and the far-field map are built from it once | give each PML a concrete `background=`, or differentiate a structure instead |
| A traced material filling a stabilized PML layer the grazing rule judges | on a stabilized face the rule reads the face's exact return, on the host only | keep it concrete, or the face a pure stretch |
| A traced material beside a smoothed region on a `Symmetry` plane | the mirror check reads the plan's contrast, which may be zero | keep the region a cell off the mirror |
| A traced medium at a `Diffraction` plane with `orders=None` | the propagating orders cannot be counted for a traced index | give `orders=` |
| A body with traced geometry as the face a pillar stands on | the junction is differentiated through its standing body only | keep the face's geometry concrete |
| A traced conductivity or pole parameter | no certificate | keep it concrete |
| A traced value where `plan=` was built without that material, or naming `mu` differently | the plan was certified for another medium | plan from the same scene at concrete values |

## How accurate is the derivative

The derivative is that of the discrete solver, exactly. On the probe's version of the cell
above (`studies/fdtd/traced_material/fdtd_traced_material_probe.py`: three wavelengths, T and R, the centre's x
as a fourth leaf), JAX and Torch agree to `3e-15`, and both agree with central differences
of the concrete NumPy solver -- each perturbed geometry planned on its own -- to `9e-8` for
the shape and `4e-9` for the pillar's permittivity at a step of `3e-6`, `1e-9` for the
substrate's. Two caveats:

* **The discrete solver is piecewise smooth.** At a few isolated geometries a voxel's
  classification switches (a cylinder's nearest face changes from its wall to its cap, a
  cap sits exactly on a grid plane), and the value jumps by about `1e-5` of `T`. The
  gradient is right on either side; a finite difference that straddles the jump is not.
* **A derivative is not a convergence statement.** The cell above uses thin PMLs and a
  short run to be quick, and there the lossless cell's `T + R` misses 1 by about `5e-4`.
  With 24-cell PMLs, a `.05` band and 6000 steps it is 1 to `1.1e-6`, and the
  derivatives keep `dT + dR = 0` to `5e-5` of `dT` for the pillar's permittivity and
  `6e-5` for the radius. The substrate's permittivity shows `dT + dR` of about `5e-5`
  (0.2% of its small `dT`): the residual `T + R - 1` itself ripples at the `1e-6` level as
  the thick substrate's optical path changes, and that is its slope. Check your own
  observable under refinement before trusting its optimum.

## Which backend

JAX compiles the run once per plan (about 50 s for the probe's cell on one CPU thread) and
is then fast to repeat; Torch runs eagerly (about 7 s for the same two-output Jacobian) and
suits a few evaluations. NumPy is not differentiable, but a concrete NumPy run is the
reference to compare a gradient against: plan each perturbed scene on its own, since a
pillar's junction layout belongs to its geometry.

The focused regression is `python -m pytest tests/fdtd/test_material_resolution_guard.py`.
It straddles the eager thresholds for plain, conductive, Debye and Drude materials in
source-free scenes (including a concrete inactive Lorentz pole at resonance), compares
admitted lowering coefficients and their derivatives with
concrete finite differences, and checks JAX jit, Torch, rule wavelengths and the PML
fallback. The guard is constant to autodiff; it does not alter admitted coefficients.

The guard regressions also compare eager decisions at the inclusive boundaries and their
nearest representable neighbours. They cover a homogeneous vacuum with a Diffraction
wavelength equal to, shorter than, or longer than the report wavelength; anisotropic
absolute-index selection and first-component ties; inactive concrete versus carried zero
Lorentz strengths at resonance; and active Lorentz coefficient derivatives away from
resonance. Fallback tests retain the first planned PML wavelength (not the shortest),
Resolution precedence over PML, source-band precedence over both, and multiplication by
an existing zero, finite or NaN guard. These checks do not change the physical thresholds.


Traced material values retain the constructors' finite-value requirement, including
conductivity and every pole parameter even when a Lorentz strength is concretely zero.
Positive infinity is not an admissible positive parameter: it poisons the material guard
just as NaN, negative values, or a forbidden zero do. Conductivity, damping and oscillator
strength still admit zero; permittivity, permeability, frequencies and Debye relaxation
time remain strictly positive. No clipping or upper physical cutoff is introduced.
Finite extreme inputs pass this range door; this alone does not certify the CFL bound,
resolution, or recurrence arithmetic at extreme finite scales; an inactive analytic response does
not excuse an invalid parameter declaration. This is covered by
`tests/fdtd/test_material_finite_guard.py`, including admitted coefficient derivatives.
