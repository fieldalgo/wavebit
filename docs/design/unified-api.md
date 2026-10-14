# Unified RCWA and FDTD API: design and implementation checkpoint

Status: the shared public lifecycle, materials, source layout and bilingual
learning-site migration are implemented in the integration working tree. Focused
acceptance checks have passed; the full default test suite is still running.
This is an unreleased implementation checkpoint, not a release or a claim of
universal physical convergence.

Design written: 2026-10-07. Implementation checkpoint: 2026-10-13, against the
working tree based on `3116a50a`. The evidence below describes those uncommitted
integration changes, not the contents of that base commit alone.

Sections 1–12 preserve the accepted design and its original rationale. Their
proposed examples and future-tense requirements are design history; the current
[getting-started pages](../getting-started/index.md), [API reference](../reference/index.md)
and [implementation checkpoint](#13-implementation-checkpoint-and-acceptance-evidence) define what is
implemented and checked now. The capability boundaries in that checkpoint take
precedence over any broader aspiration in the original proposal.

The original review compared RCWA `b9957c8bde0f60d3ca47126996cad257fc6e20a6`
and FDTD `f5a86603e4d7b668726a0519714dc2bcae950655`. Integration subsequently used
the newer FDTD checkpoint `d1119445bb2839dd03d2af0b769a2797d0be2d13` and preserved
its history in merge `04b449754e6b226bdcd4374b2897732054468ce7`. That merge's first
parent is develop `db37dbb8722b8993c2486c6af4f491e3b1b8bc13`; its second parent is
the FDTD checkpoint. The source layout and later public API changes build on that
ancestry rather than replacing it with a squash.

Library source now lives in `src/wavebit/`. The organized public documentation
lives in `docs/`, with English canonical pages and optional Korean translations.
The migration includes the complete selected teaching corpus and its assets;
research records retain their recorded numerical results and provenance.

## 1. Decision

Use one public lifecycle for both methods:

```text
Problem + Numerics + execution settings
                  |
                prepare
                  |
               Prepared -- run(Problem) --> Result

run(Problem, ...) is the convenience prepare-and-run path.
```

The common contract is the lifecycle, material response, physical conventions,
validation, and comparable observables. RCWA and FDTD retain different problem
descriptions and numerical algorithms. A common API does not imply a universal
Scene, one internal Plan representation, or identical supported features.

Do not retain a mutable solver as the primary interface merely to preserve the
current implementation. Do not introduce a common Solver base class that users
must subclass. Backend-specific solver classes are implementation details.

### Responsibilities

| Object | Owns | Does not own |
| --- | --- | --- |
| Material model | Constitutive response, dispersion, source units of material data | Grid, timestep, ADE recurrence, Fourier factorization |
| Problem | Physical geometry, units, excitation, requested observations | Backend, cached operators, last result |
| Numerics | Method-specific approximation and discretization choices | Material identity or incident polarization |
| Execution settings | Backend, device, precision | Physical problem or discretization |
| Prepared | Resolved numerics, static layout, reuse contract, safely keyed caches | Mutable last problem, last result, stale live material values |
| Result | Outputs, reference conventions, validity, provenance, requested reconstruction state | Dependence on the most recently executed calculation |

All declaration objects support `replace(**changes)` and are structurally
immutable. Numerical leaves preserve their backend and autodiff graph. Frozen
declarations do not magically make a caller's tensors immutable: input buffers
must not be mutated concurrently with a run. Preparation snapshots static host
data; run borrows live arrays without mutating or detaching them.

## 2. The public lifecycle

Both public modules provide these signatures; their Problem and Numerics types
are module-specific:

```python
def check(problem, *, numerics, backend="numpy", device="cpu", precision="double"):
    """Return a validation and resource report; do not advance fields or solve scattering."""

def prepare(problem, *, numerics, backend="numpy", device="cpu", precision="double"):
    """Validate concrete inputs and return a Prepared calculation."""

def run(problem, *, numerics, backend="numpy", device="cpu", precision="double"):
    """Equivalent in meaning to prepare(problem, ...).run(problem)."""

class Prepared:
    def validate(self, problem):
        """Eagerly report whether this concrete problem satisfies the reuse contract."""

    def run(self, problem):
        """Run the supplied problem without changing this object's numerical contract."""
```

`check` returns data, not printed output. It shares validation with `prepare`,
which always validates; callers do not need to call all three functions.
Resource estimates identify their scope, including whether temporary arrays and
backend runtime memory are excluded. `prepared.report` records the resolved grid
or mode window, actual discretization, estimated cost, and capability limitations.

`Prepared.run` always takes the complete live problem. There is no implicit
"use the values from the last call" or a second parameter dictionary that can
disagree with the Problem. Pure Python functions can construct a Problem from
an optimizer's small parameter tree.

`numerics`, backend, device, and precision are fixed at preparation. Changing
them requires another `prepare`. A reusable Prepared never silently replans.
The convenience module-level `run` prepares the supplied problem afresh, though
content-addressed internal caches may avoid identical work.

Backend names are `"numpy"`, `"jax"`, and `"torch"`. Precision names are
`"double"` and `"single"`, with double as the default. This is a vocabulary,
not a claim that every combination is supported. Unsupported precision/device/
solver combinations are rejected; there is no silent downgrade or CPU fallback.
Importing Wavebit must not import optional backends or change global precision.
A JAX double-precision request must verify the actual x64 configuration and give
an actionable error when it cannot honor it.

Examples below using JAX assume the application has explicitly enabled x64
before creating arrays, for example with
`jax.config.update("jax_enable_x64", True)`. The library does not do this on import.

## 3. Common material model

The public family is:

```python
from wavebit import Material
from wavebit.materials import (
    ConductiveMaterial, DispersiveMaterial, TabulatedMaterial,
    Drude, Lorentz, Debye,
)

glass = Material(epsilon=2.25)       # mu=None means structurally nonmagnetic
magnetic = Material(epsilon=2.25, mu=1.1)
```

`epsilon` and `mu` are dimensionless values relative to vacuum. The initial
constructor contracts are `ConductiveMaterial(epsilon, conductivity, mu=None)`,
`DispersiveMaterial(epsilon_inf, poles, mu=None)`, and
`TabulatedMaterial(wavelengths, epsilon, *, wavelength_unit, mu=None)`.
TabulatedMaterial also provides an explicit `from_refractive_index` constructor. The
stored interpolation contract records whether index or permittivity is
interpolated; moving a table must not silently change that choice.

All models implement:

```python
response = model.at(wavelength, units=units)  # MaterialResponse(epsilon, mu)
```

`MaterialResponse` is evaluated constitutive data, not a reusable broadband
material model. Wavelengths are vacuum wavelengths. A model is independent of
the scene's choice of length unit; `units` interprets only the supplied wavelength.
Solver evaluation supplies that context automatically.

| Model | Physical data | RCWA consumption | FDTD consumption |
| --- | --- | --- | --- |
| Material | Constant epsilon and optional mu; scalar/diagonal/tensor | Evaluate directly | Admit only supported real, stable constitutive cases |
| ConductiveMaterial | Baseline epsilon, conductivity, optional mu | Evaluate conductivity contribution at frequency | Build supported conductive updates |
| DispersiveMaterial | High-frequency response plus named poles | Evaluate continuous frequency response | Build supported auxiliary updates |
| TabulatedMaterial | Spectrum, units, interpolation and source metadata | Evaluate within its declared range | Require an explicit supported causal dispersion fit |

Drude/Lorentz/Debye response formulas belong to the common material package.
Time-discretization coefficients, mesh stability, and Fourier operators belong
to the respective solver. A common class must not contain `fdtd_options` and
`rcwa_options` branches.

Constructors validate mathematical structure without converting differentiable
values through Python float or NumPy. Solver admission validates capability:
full tensors, active media, dispersive interfaces, or material gradients may be
accepted by one path and refused by another. `mu=None` and explicitly named
`mu=1` retain distinct structural meanings for planning and differentiation.

No automatic table fitting, loss removal, or replacement of a complex constant
by a guessed time-domain material is allowed. An explicit fitting operation
would produce a new DispersiveMaterial and an error/band/passivity report.

## 4. Units, coordinates, and physical meaning

Each Problem declares `Units(length="m" | "mm" | "um" | "nm")`.
All geometry and vacuum wavelengths use that length unit. Other physical
quantities have fixed public units, independent of scene length:

| Quantity | Public convention |
| --- | --- |
| Angles, including shape rotations | Radians |
| Time, pulse delay, sampled time axes | Seconds |
| Frequency / angular frequency | Hz / rad s^-1; names distinguish them |
| Epsilon / mu | Dimensionless relative constitutive values |
| Material pole decay rates | s^-1 |
| Conductivity | S/m |
| Time-domain fields and calibrated phasors | E in V/m, H in A/m, with source normalization recorded |
| Electric dipole moment / current moment | C m / A m; distinct quantities and types |

For the initially retained pole models, use explicit names such as
`Drude(omega_p=..., gamma=...)`, `Lorentz(omega_0=..., gamma=..., strength=...)`,
and `Debye(tau=..., delta_epsilon=...)`; omega and gamma use SI rates, tau seconds,
and the current Lorentz numerator strength has units s^-2. These meanings travel
with serialized models. They are never inferred from the magnitude of a number.

Internally, FDTD may retain c=epsilon0=mu0=1. Its length/time conversion is
`t_internal = t_seconds * c0 / length_metres`; a rate is multiplied by
`length_metres / c0`, and electric conductivity becomes
`sigma_internal = sigma_SI * length_metres / (epsilon0 * c0)`.
Public H is converted from the existing Z0*H
storage. Source amplitudes and all physical power outputs must be converted
consistently; this is an implementation requirement, not a cosmetic label change.
Normalized field access, if needed by experts, must be named and carry its scale.

The global Cartesian frame is right-handed with +z upward. Stack layers are
bottom-to-top, with the finite stack bottom at z=0. A FDTD Domain has an explicit
center in this same frame. Periodic cells use a center-based public lateral
phase origin. All complex frequency responses use exp(-i omega t).

Changing wavelength units must not change a material's physical pole frequencies.
Changing coordinate conventions must transform vectors/tensors and complex
reference phases, not just reverse a layer list. The migration explicitly covers
the old downward-positive periodic FDTD path and the newer general FDTD path
separately.

## 5. RCWA problem and numerical settings

RCWA uses a periodic layered Stack and monochromatic Excitation. Excitation
owns wavelength and a tuple of source declarations. Direction, polarization,
amplitude, and side have one owner: the source. The new API does not keep
competing Incidence, PlaneWave, and top-level theta/polarization arguments.

```python
from wavebit import Material, Units, rcwa

glass = Material(epsilon=2.25)
problem = rcwa.Problem(
    units=Units(length="um"),
    stack=rcwa.Stack(
        period=(0.6, 0.6),
        layers=(rcwa.Layer(thickness=0.2, background=glass),),
        top=Material(epsilon=1.0),
        bottom=Material(epsilon=1.0),
    ),
    excitation=rcwa.Excitation(
        wavelength=0.55,
        sources=(rcwa.PlaneWave(side="top", polarization="s"),),
    ),
    outputs=rcwa.Outputs(diffraction=True, fields=False, scattering=False),
)
numerics = rcwa.Numerics(orders=(5, 5), algorithm="smm")

result = rcwa.run(problem, numerics=numerics)
R = result.R.sum(axis=(-2, -1))  # one value per wavelength; spectral axis retained
T = result.T.sum(axis=(-2, -1))

prepared = rcwa.prepare(problem, numerics=numerics, backend="jax")
another = problem.replace(
    excitation=problem.excitation.replace(wavelength=0.60)
)
another_result = prepared.run(another)
```

This example is a proposed homogeneous film API, not an executable example of
the reviewed commits. A Layer can also carry typed 2D regions or a sampled
epsilon field. A raster is an explicitly sampled quantity, not an array of
Material objects. A separate mu field describes magnetic raster data.

`rcwa.Numerics` owns orders, Fourier coefficient strategy, factorization,
formulation, layer algorithm, and auxiliary discretization choices. Defaults
resolve from the problem and are published in the preparation report. Requested
but irrelevant options are refused rather than ignored.

Fields and scattering matrices are opt-in Outputs because retaining them has
a material memory cost. Reconstruction reads the returned result, never a
solver's most recent state. Low-level convolution assembly and direct operator
solves live under `rcwa.operators`; experimental conforming/NVF constructions
live under `rcwa.experimental`. They do not overload normal `run` input types.

Coherent beams, order excitations, and internal dipoles use the same Problem
and lifecycle. Their observation schema supports boundary power and fields;
it does not manufacture a reflection coefficient when the excitation has no
single incident port. Eager, grad, jit, and vmap support are separate capabilities.

## 6. FDTD problem and numerical settings

FDTD Scene owns the physical domain, materials, structures, and physical
boundary conditions. Problem adds sources and requested outputs. Numerics owns
grid, time integration, duration, absorption implementation, and geometry
preparation ranges.

```python
from wavebit import Material, Units, fdtd

glass = Material(epsilon=2.25)
problem = fdtd.Problem(
    units=Units(length="um"),
    scene=fdtd.Scene(
        domain=fdtd.Domain(size=(0.6, 0.6, 1.6), center=(0, 0, 0.1)),
        background=Material(epsilon=1.0),
        boundaries=fdtd.Boundaries(
            x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.Open()
        ),
        structures=(
            fdtd.Structure("film", fdtd.Slab(z=(0.0, 0.2)), glass),
        ),
    ),
    sources=(
        fdtd.PlaneWave(
            "illumination", plane=fdtd.Plane(z=0.6), direction="-z",
            theta=0.0, phi=0.0, polarization="s",
            pulse=fdtd.GaussianPulse(
                wavelength=0.55, frequency_sigma_hz=2e13
            ),
        ),
    ),
    outputs=fdtd.Outputs(
        monitors=(
            fdtd.Record("probe", at=(0.0, 0.0, -0.2), component="Ey"),
        ),
        final_fields=False,
    ),
)
numerics = fdtd.Numerics(
    grid=fdtd.UniformGrid(spacing=0.025),
    duration=fdtd.Duration(periods=40, source="illumination"),
    courant=0.5,
    absorber=fdtd.PML(cells=12),
)

result = fdtd.run(problem, numerics=numerics)
t_seconds = result.outputs["probe"].time
ey = result.outputs["probe"].values

prepared = fdtd.prepare(problem, numerics=numerics, backend="jax")
result = prepared.run(problem)
```

The source amplitude convention is electric-field amplitude for PlaneWave,
with phase and normalization explicitly recorded in source references.
`GaussianPulse.frequency_sigma_hz` is the standard deviation of its amplitude
spectrum; it is not fractional bandwidth, wavelength width, or intensity FWHM.
`Duration` takes exactly one of seconds, steps, or periods. Periods names its
reference source explicitly; it cannot become ambiguous in a multisource scene.

Domain denotes the physical region of interest. Absorber cells are added
outside its Open faces, and the preparation report exposes the full allocated
domain and actual sample locations. Material continuation into padding must be
specified or unambiguous and supported; padding never silently becomes vacuum.
PEC/PMC/periodic/Bloch conditions remain physical declarations. PML grading,
stabilization, and cell count are numerical choices.

The proposed Open/PML separation and outside padding differ from the current
implementation and require dedicated equivalence tests. A renamed API cannot
claim them without changing the adapter and geometry accounting.

The standard FDTD problem can request time records, fields, flux, diffraction,
far fields, and final fields. Only implemented observables are public stable
exports. GaussianBeam, mode sources/monitors, decay termination, GPU support,
and nonuniform grids must not appear as supported merely because constructors
or keywords have been reserved.

## 7. Prepared reuse and differentiation

Preparation fixes a computational structure, not all numerical values.
`prepared.contract` reports structural dependencies, admitted live fields,
shape/dtype constraints, runtime guards, and differentiation capabilities.
Reuse eligibility and differentiability are distinct: a value can be changed
between eager runs without having a supported derivative.

| Change | Requirement |
| --- | --- |
| Material coefficient, wavelength, source amplitude/phase | Recompute dependent coefficients; reuse only while the declared path and guards remain valid |
| Geometry position/size, layer thickness, lattice basis | Reuse only within supported topology and spatial bounds |
| Raster values | May invalidate phase partitions or NVF interface records even when array shape is unchanged |
| Body/layer/source counts, material model type or pole count | Reprepare when the static structure changes |
| Scalar to tensor, mu=None to named mu | Reprepare the changed constitutive structure |
| Coherent grouping, diffraction window, observer array layout | Reprepare when the prepared grouping/shape changes |
| Grid size, numerical method, retained outputs, backend/device/precision | Reprepare |

Resolved timestep and step count are prepared dependencies. In particular,
`Duration(periods=40, source="illumination")` depends on the source center
wavelength. Changing that wavelength requires re-preparation when it changes
the resolved duration or step count. A fixed-duration or fixed-step policy may
admit different wavelength changes after the other guards are satisfied. A run
must not use an old step count while claiming forty periods of a new source.

RCWA must not cache material matrices, modes, or propagation operators across
changed numerical dependencies. FDTD must not reuse update coefficients merely
because the grid shape is unchanged. Safe caches are keyed by their actual
dependencies, not object identity or the previous call.

For moving FDTD bodies, reserved spatial bounds belong to
`fdtd.Numerics.geometry_ranges`, keyed by stable Structure names. They are
computational support declarations, not properties of a Sphere or Box.
They neither prove all material combinations stable nor replace runtime checks.

Normal eager `Prepared.run` validates the supplied values and raises a path-specific
error on an incompatible problem. `Prepared.validate` allows an optimizer to
perform that validation explicitly outside a compiled region. It does not
certify arbitrary future values.

For autodiff, construct a concrete Problem, prepare outside the transform, and
pass a freshly constructed differentiable Problem to each run. For example:

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

base = design(jnp.asarray(2.25))  # design(epsilon) returns a Problem
prepared = rcwa.prepare(base, numerics=numerics, backend="jax")

def loss(epsilon):
    result = prepared.run(design(epsilon))
    return -jnp.sum(result.T)

gradient = jax.grad(loss)(jnp.asarray(2.25))
```

This contract does not promise unrestricted geometry differentiation. Under an
external JAX transform, a structural change detectable from Python metadata is
rejected. A supported value-dependent guard executes in backend operations;
failure produces an explicit failed validity check and NaN numerical outputs,
not a detached tangent or an unreported clamp. Diagnostics metadata remains
readable. Unsupported host-only topology decisions must be rejected at trace
construction, or prepared again from each concrete design outside the transform.
If a traced change cannot be safely checked, it is not an admitted live field.

Shape optimization may prepare each concrete iterate and reuse compiled kernels
when the resulting structure is equal. Crossing a topology boundary does not
have a promised smooth derivative. The API never guarantees that a fixed mesh,
successful compilation, or a true stability guard proves physical accuracy.

## 8. Results and observations

Every Result has `outputs`, `validity`, `diagnostics`, and `provenance`. Metadata
records requested and effective inputs, units, source normalization, numerical
settings, execution precision, and code version. Values remain on their backend.
Saved results and field reconstruction remain valid after later runs.

`result.outputs` is a name-to-observation mapping. FDTD keys are requested monitor
names. A one-sided RCWA excitation provides named reflected/transmitted outputs;
more general excitation uses named boundary outputs. Solver-specific conveniences
may expose `R`, `T`, `scattering`, or requested final fields, but their meanings
are defined by the observations rather than by mutable execution state.

Shared observation types include FieldSamples and DiffractionOrders. A
DiffractionOrders object describes outgoing orders at **one** reference port
and **one** incident/Bloch transverse-wavevector group:

```text
wavelengths, orders_u, orders_v
base transverse wavevector and reciprocal lattice (or per-order wavevectors)
reference medium response and longitudinal propagation data
amplitude_s, amplitude_p, optional Cartesian amplitudes
calibrated outgoing modal power, and efficiency when an incident normalization exists
reference plane, lateral origin, polarization frame, propagation convention
source normalization and incident reference plane
integration measure, cell area or invariant-length normalization, outward normal
```

Its spectral/modal arrays always use `(wavelength, order_v, order_u)`, including
a length-one wavelength axis. Named `order(m, n)` and explicit spectral selection
avoid positional assumptions. RCWA R/T are efficiency views of reflection and
transmission for the single-port case. A lone FDTD reflection monitor must not
fabricate an unmeasured transmission result.

Different Bloch groups at the same port have different physical order grids.
They return separate group records or a boundary-field/power observation; their
amplitudes are not added into one order array merely because the integer order
labels match. Power integrated over a requested finite region must retain any
cross-group interference appropriate to that observation.

Absolute port power specifies its spatial integration measure: W per declared
periodic cell/surface, or W/m per invariant length in a reduced-dimensional
calculation. Surface orientation and normalization area/length are explicit.
Efficiency is dimensionless only after its incident denominator and integration
measure are fixed. If an absolute calibration is unavailable, absolute power is
unavailable, not silently assigned the units of a normalized number.

Pulse Fourier outputs are distinct from time-domain samples and calibrated
monochromatic phasors. For the raw transform
`E_tilde(f) = integral E(t) exp(+i*2*pi*f*t) dt`, field units are V*s/m and A*s/m.
The corresponding spectral energy density has units J/Hz after spatial
integration (or J/(Hz*m) per invariant length), not steady power in W.
Raw SpectralFieldSamples/RawDiffractionSpectrum record transform normalization,
window, sampling, and the one-sided/two-sided spectrum convention. Spectral
density per wavelength requires the frequency-to-wavelength Jacobian.

FDTD can form comparable dimensionless scattering amplitudes/efficiencies by
normalizing with an explicitly selected incident source reference. Its declared
usable spectral band excludes unresolved or effectively zero source amplitudes.
Such transfer observations use DiffractionOrders; normalized amplitude_s/p are
dimensionless and any additionally calibrated fields/powers state the reference
incident amplitude. Raw Fourier arrays are never relabeled as V/m, A/m, or W.
Both raw and normalized observation forms remain available when requested.

The common type must retain continuum versus Yee propagation/de-embedding
metadata. Equal field names do not make amplitudes directly comparable until
reference planes, lateral origin, incident normalization, and polarization agree.
Reference conversion is explicit and reports the propagation model used.

Absorption is not universally `1-R-T`: absorbing incident media require the
appropriate entering-power/interference term. Multisource or internal-source
results expose boundary powers without inventing a unique R/T normalization.
An incoherent average combines powers and has no single complex amplitude.

FieldSamples exposes coordinates, component labels, sampling/staggering metadata,
and explicit units. SpectralFieldSamples additionally follows the raw-transform
contract above. Requested collocated fields and raw Yee components are
different observation forms; interpolation is never implicit in a named raw sample.
RCWA field reconstruction state is retained only when requested. FDTD final
fields are also opt-in.

Changing a physical input to regularize a singular solve is explicit policy.
The default targets the requested problem and reports a singularity. If a caller
opts into detuning, the result records the effective wavelength/angle and states
that its gradient belongs to that effective problem. A trace does not silently
choose a different retry policy from eager execution.

## 9. Validation, errors, and saved jobs

Use shared error families: ValidationError, UnsupportedFeatureError,
IncompatibleProblemError, and NumericalError. An issue includes a stable code,
input path, explanation, and relevant expected/actual values. Unsupported
features are not ignored and unavailable outputs do not look like zero.

`validity` separates numerical/admission checks from physical convergence.
Instability, an out-of-range material lookup, an expired geometry range, and
insufficient simulation duration must not collapse into one boolean called stable.
Warnings and convergence diagnostics retain their distinct meanings.

Jobs use one versioned envelope and solver-specific payloads:

```text
Job:
  schema: wavebit.job/v1
  solver: rcwa | fdtd
  problem: tagged Problem data, including Units
  numerics: tagged Numerics data
  execution: backend, device, precision
```

`Job.to_json`, `Job.from_json`, and `Job.key` preserve complex values, typed
arrays, material models, spectra, and source declarations. Large datasets may
use explicit content-hashed references. Arbitrary Python callables, compiled
functions, Prepared objects, and device buffers are not portable job data.
Callable-based designs remain a Python API feature and require a concrete
serializable Problem to be saved as a Job.

`wavebit.run(job)` dispatches a Job only. `rcwa.run` and `fdtd.run` take typed
Problems, not a mixture of JSON, mappings, jobs, and raw operator arrays.
Sweeps explicitly expand into independent Jobs; one run never silently chooses
the first sweep point. An RCWA spectral sweep may aggregate results afterwards;
an FDTD pulse remains one broadband run with requested spectral observations.

## 10. Package layout

The following is the installed package layout. Its repository location is
`src/wavebit/`; installation still exposes `import wavebit`, never `import src`.
Section 12 describes development, documentation, and migration requirements.

```text
wavebit/
  __init__.py           # Common public types, solver modules, run(Job)
  materials/           # Material models, response evaluation, catalog, fitting
  units.py
  lattice.py
  observations/        # Common fields, diffraction, power/reference contracts
  jobs.py
  errors.py
  _common/             # Array/precision helpers and validation infrastructure
  rcwa/
    __init__.py        # Problem, Numerics, check, prepare, run
    problem.py        # Stack, Layer, Excitation, Outputs, Problem
    sources.py
    numerics.py
    results.py
    operators.py      # Explicit advanced operator API
    experimental/
    _engine/
    _backends/
  fdtd/
    __init__.py        # Problem, Numerics, check, prepare, run
    problem.py        # Scene, Outputs, Problem
    geometry.py
    boundaries.py
    sources.py
    monitors.py
    numerics.py
    results.py
    _prepare/
    _kernel/
    _runtime.py
  compat/
    legacy.py         # Explicit transitional old entry points

tests/
  materials/
  rcwa/
  fdtd/
  cross_solver/
  api/
```

Public rcwa/fdtd packages have public names; internal computation uses underscored
modules. There is no separate public `fd`, no universal `_core` containing both
physics engines, and no FDTD import of an RCWA backend merely to obtain a result
class. A common physical type may depend on common array helpers, never a solver.

The common materials package replaces the current material catalog module while
retaining catalog operations such as `materials.load` and `materials.available`.
A fitting API is exposed only when implemented and qualified. Geometry is not
made common merely because a 2D circle and a 3D cylinder can describe related
objects; supported structure conversions are explicit adapters.

## 11. Migration and acceptance gates

This is an intentional breaking API redesign. Python cannot reliably expose
`wavebit.rcwa` as both the old callable and the new module, likewise fdtd.
Do not use callable-module tricks or import-order-dependent compatibility.

1. Freeze an integration checkpoint and record legacy/new-engine capabilities.
   Merge/adapt correctness first; do not hide file moves inside numerical fixes.
2. Implement and verify common material/units/reference contracts. Preserve
   differentiable leaves and separate physical responses from numerical lowering.
3. Add Problem/Numerics, shared lifecycle, and result ownership over both engines.
   Temporary internal adapters may use the old solver, but cannot expose its
   previous-call state through the new contract.
4. Expose the new module APIs and update first-party examples, docs, tests,
   serialized schemas, and capability manifests. Export only implemented features.
5. Move files in separately reviewable commits after behavior is pinned.
6. Retire compatibility only after capability equivalence has been established.

The old periodic FDTD route currently has features the general engine lacks,
including graded meshes, broader polarization support, and some angle/layer-depth
derivatives. Preserve it under an explicit compatibility entry point during
transition; do not silently switch engines under standard `fdtd.run`.
Moving a name is not evidence that the general engine implements those features.

Required acceptance evidence:

- Lossless and lossy asymmetric slabs; TE/TM and oblique incidence from both
  sides; common material frequency response and supported dispersive models.
- RCWA/FDTD order powers **and complex amplitudes**, using matching phase
  origins/reference planes and measured spatial/time convergence. Test odd
  diffraction orders to detect cell-corner versus cell-center phase errors.
- Physical unit changes (m/um/nm) preserve the physical answer, including
  conductivity, dispersion, dipole normalization, field H, and power units.
- Prepared reuse agrees with fresh preparation for admitted changes; topology,
  material-layout, output-layout, and source-group changes invalidate correctly.
- JAX/Torch derivatives match independent checks within each declared scope;
  traced invalid inputs are surfaced and never reuse stale numerical values.
- Results survive later runs; unused field state is not retained; optional
  backends remain optional in a minimal installation.
- Job round trips preserve meaning and keys. Old-schema coordinate/material/
  angle changes require explicit migration and cannot be handled by a version edit.

Passing these gates establishes the new contracts in their tested scopes. It
does not establish every material/geometry/backend combination or universal
physical convergence.

## 12. Whole-repository layout

The package and its development evidence serve different audiences. Keep their
entry points explicit instead of putting tests, research runners, and generated
results into the importable library.

```text
wavebit-repository/
  pyproject.toml
  README.md
  CONTRIBUTING.md
  CITATION.bib
  LICENSE*
  src/
    wavebit/                     # Package layout from section 10
      data/                      # Shipped optical data and their source metadata
  tests/
    api/                         # Public lifecycle, errors, jobs, result ownership
    materials/
    common/                      # Units, references, array/precision helpers
    rcwa/
    fdtd/
    cross_solver/                # Same physical cases, normalized comparisons
    distribution/                # Built wheel/sdist and minimal installation
    references/                  # Independent oracles, generators, frozen data
    fixtures/                    # Small shared test inputs
  docs/
    index.md                     # Default English documentation site entry point
    conf.py
    _toc.yml                     # Site navigation, distinct from filesystem order
    getting-started/             # Installation and first RCWA/FDTD calculations
    learn/                       # Canonical English textbook, informed by research sources
      fundamentals/
      nvf/
    guide/
      common/
      rcwa/
      fdtd/
      how-to/                    # Short task-oriented recipes using public APIs
      troubleshooting/           # Symptoms, diagnosis, and supported remedies
    reference/                   # Public contracts and generated API documentation
      common/
      rcwa/
      fdtd/
    theory/                      # Derivations and numerical-method explanations
    validation/                  # Measured claims, accuracy, and current limits
    design/                      # Architecture proposals and decisions
    development/                 # Source setup, debugging, extension, contributing
    migration/                   # Upgrade paths, including the unified API transition
    release-notes/
    glossary.md                  # Symbols, physical terms, and API vocabulary
    locales/
      ko/                        # Optional Korean translations of the English documents
    figures/                     # Maintained diagrams and plots used by documents
    _static/
  examples/
    rcwa/
    fdtd/
    cross_solver/
  tutorials/                     # Sequential teaching notebooks
    rcwa/
    fdtd/
    cross_solver/
  studies/                       # Reproducible investigations, grouped by question
    rcwa/
    fdtd/
    cross_solver/
    _support/                    # Research-only helpers, never production imports
  benchmarks/                    # Stable performance/memory workloads
    rcwa/
    fdtd/
  tools/                         # Release, documentation, and repository utilities
  assets/                        # Project branding and other non-runtime assets
  artifacts/                     # Local generated outputs; ignored by Git
  .github/workflows/
```

Do not pre-create every directory empty. Introduce a category when it has an
owner and content. In particular, existing one-off timing studies do not become
a performance regression suite simply by moving under benchmarks.

### Source and installed data

Adopt `src/wavebit/` as the repository source layout. This separates importable
library code from documentation, tests, and research tooling and reduces
accidental checkout imports in distribution checks. It does not change the
public import name or determine the RCWA/FDTD numerical architecture. The
[Python Packaging User Guide](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)
describes the packaging tradeoffs; the source-layout choice is now part of this
proposal rather than an optional later direction.

Researchers who modify the implementation are first-class users. The normal
source workflow uses one environment per active worktree and an editable install
from that worktree. Ordinary Python source edits are used on the next process
run without reinstalling; long-running notebooks need reload or restart. Project
metadata and compiled extensions may need reinstalling or rebuilding, as described
in [pip's editable-install documentation](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs).
Independent environments can use the same recorded dependency versions for
controlled comparisons.

A shared environment is optional when dependencies agree. Document the fact that
one editable installation points to one checkout: changing the current directory
does not select another checkout's `src`. A debugger can explicitly select the
active workspace's `src` through its import path; users can inspect
`wavebit.__file__` to verify which source is loaded. Source breakpoints work in
either mode. Explicit source-path debugging is a development workflow, not a
substitute for testing the installed distribution.

Wavebit already performs an isolated noneditable installation smoke check from
outside the checkout, using `python -I` and asserting that the imported package
is not under the checkout. This is useful protection, not proof that the whole
installed distribution passes the full numerical suite. Retain that check and
extend distribution validation to a built wheel and an sdist-derived wheel,
including package data, public APIs, and optional-backend isolation.

Implement the source move in a separately reviewable change after the FDTD
checkpoint and integration behavior have been established. Update package
discovery, package data, study/test runners, documentation imports, coverage,
and path-based repository checks together. Replace the root-path injection in
`docs/conf.py` with the documented installation workflow and verify that autodoc
loads this worktree's intended package. Update the current flat-layout policy
in `CONTRIBUTING.md` when the migration lands. Neither `PYTHONPATH=.` nor simply
running from the repository root is the new package installation procedure.

Move the current runtime optical-data assets with the package and load them
through package resources. Their units, source identity, provenance, and license
notices travel with them. Test goldens and study output are not wheel data.
Keep the required root license notices; do not remove them as folder cleanup.

### Tests and reference evidence

Organize tests by the contract they verify, not one test file per private module.
API and common tests check shared behavior; solver tests check method-specific
behavior; cross_solver tests check a physical comparison under stated convergence
and normalization conditions. Independent external goldens remain distinguishable
from snapshots generated by Wavebit itself.

Keep each external oracle's generator, frozen data, test, environment notes,
and third-party notices together under tests/references. Path changes must not
regenerate reference answers. External solver installations are optional for
regeneration and are not required merely to consume committed goldens in CI.

Runtime categories such as slow and research are test metadata, not the only
folder taxonomy. Preserve their collection semantics during moves, including
explicit file selection and last-failure reruns. Update any path-based marker
lists, fixture paths, documentation-contract tests, and workflow commands.

### Studies and artifacts

The reviewed FDTD revision contains 3,354 tracked files under studies/results,
including logs, JSON, compressed/raw records, and executable scripts. These are
not all disposable build output. Reorganize by the question being investigated:

```text
studies/fdtd/conductive_stack/
  README.md                      # Question, method, conclusion, limitations
  run.py                         # Reproducible entry point
  configurations/                # Explicit experiment inputs
  records/
    <record-id>/
      manifest.json              # Commit, command, inputs, versions, checksums
      summary.json               # Compact measured outcomes
      evidence/                  # Selected supporting outputs, including failures
```

New exploratory runs write to ignored artifacts, for example
`artifacts/studies/fdtd/conductive_stack/<run-id>/`. Promoting a run to a cited
record preserves enough evidence to reproduce and assess the claim. Large raw
outputs may use stable content-hashed artifact references when such storage is
available; a missing external store is not a reason to delete existing evidence.
Retain interrupted/negative results when they substantiate a limitation.

Executable study code belongs beside run.py, not in results/evidence. Published
validation documents link to stable record IDs and relative manifests, rather
than machine-specific temporary paths. The same observation should have one
canonical stored record, referenced by multiple documents if necessary.

Ignore generated directories by path. Do not use blanket JSON/CSV/log ignores
that hide experiment configurations, job fixtures, or selected evidence; the
current repository-wide patterns need review as part of this migration.

### Examples, tutorials, documentation, and benchmarks

Examples are short scripts using only stable public APIs; tutorials teach a
sequence and may be notebooks. Neither imports private solver internals or
study helpers. Their smoke checks run against the intended installed package.

Guide pages teach usage. Reference pages document public types/functions. Theory
pages explain derivations. Validation pages state measured evidence and limits.
Design pages explain proposals and architectural decisions. Development pages
teach contributors to install, debug, test, and extend the source. Moving pages
also updates the documentation table of contents and inbound links.

### Public documentation and research-workspace imports

Keep documentation sources at repository-root `docs/`, alongside `src/`, so a
source checkout contains both the implementation and its explanation. The built
HTML site is a generated artifact, not library package data. Researchers can
read and edit the Markdown locally and read the same content on the public site.

The documentation structure draws on the official NumPy, JAX, and PyTorch sites,
reviewed on 2026-10-07. These are information-architecture references; the proposed
Wavebit organization below is a project-specific design rather than a claim that
the three projects share one identical hierarchy.

| Reference | Observed organization | Apply to Wavebit |
| --- | --- | --- |
| [NumPy documentation](https://numpy.org/doc/stable/) and [user guide](https://numpy.org/doc/stable/user/index.html) | Distinct getting-started, user-guide, API, and contributor entry points; fundamentals, how-tos, and advanced topics within usage documentation. | Keep learning, operational guidance, exact contracts, and source contribution easy to distinguish. |
| [JAX documentation](https://docs.jax.dev/en/latest/) and [101 course](https://docs.jax.dev/en/latest/101/index.html) | Progressive learning levels, with API reference and developer resources available separately. | Offer a sequence from first simulation to differentiation and advanced execution, with direct lookup paths for experienced users. |
| [PyTorch tutorials](https://docs.pytorch.org/tutorials/) and [recipes](https://docs.pytorch.org/tutorials/recipes_index.html) | Guided tutorials and compact task-oriented recipes have different entry points, alongside library documentation. | Keep a coherent learning course and short practical recipes, each linking runnable examples and the exact API contract. |

Keep the existing Sphinx, MyST, PyData theme, autodoc, and external table of
contents. One Wavebit site can serve these audiences without copying PyTorch's
multiple-site arrangement or JAX's numbered levels and systems-specific topics.

#### Website navigation and reader paths

Use six primary navigation entries. The website menu is organized by the
reader's task; it does not reproduce every repository directory as a top-level
tab. The default labels and landing-page descriptions are English.

| Primary entry | Reader's question | Contents and canonical sources |
| --- | --- | --- |
| Getting started | What does Wavebit solve, and how do I run my first calculation? | Installation, choosing RCWA or FDTD, a short first example for each implemented solver, basic conventions; `docs/getting-started/`. |
| Learn | How do I build understanding step by step? | Fundamentals, guided RCWA/FDTD courses, NVF and inverse-design lessons, tutorial and example gallery; `docs/learn/`, with links to canonical notebooks and scripts. |
| User guide | How do I perform this task or choose these settings? | Materials, geometry, sources, observations, numerics, backends, differentiation, preparation reuse, performance, how-tos, troubleshooting; `docs/guide/`. |
| API reference | What exactly does this public name accept and return? | Common types and utilities, RCWA, FDTD; generated public-symbol pages and normative contracts in `docs/reference/`. |
| Theory & validation | Why does the method work, and what supports its accuracy claims? | Conventions and derivations, convergence, independent comparisons, supported scope and negative results; distinct `docs/theory/` and `docs/validation/` groups. |
| Developer guide | How do I inspect, change, test, or extend the implementation? | Source setup, worktree environments, debugging, architecture, extension points, tests, documentation and release procedures; `docs/development/`, with `docs/design/` as a separate, status-labelled subsection. |

The landing page introduces the problem domain, links the two first calculations,
and presents these six entry points. Put search, documentation version, and the
English/Korean selector in consistent site controls. Make glossary, release
notes, upgrade guides, issue reporting, and citation information reachable from
secondary navigation. Keep release and unreleased documentation distinguishable.

Give readers short paths suited to their task:

- A first-time user goes from installation to one working simulation, then chooses
  a learning course; completing the theory textbook is not a prerequisite.
- A returning researcher opens a how-to such as a wavelength sweep, diffraction
  extraction, or convergence check, then follows links to the relevant API.
- An optimization user finds derivative scope, backend/precision requirements,
  preparation reuse, and performance troubleshooting together in the user guide.
- A reader assessing a scientific claim follows conventions to its derivation,
  a validation page, and the versioned study or independent reference evidence.
- A source modifier can go directly from installation to editable setup and
  line-by-line debugging, then the relevant engine overview and tests.

Within guides, present shared concepts once and branch into RCWA/FDTD details
where their contracts differ. A method-specific section does not imply that the
other solver supports the same feature. The example gallery should expose method,
backend, prerequisites, and expected runtime where measured, rather than being
an unsorted inventory of all studies.

Tutorial pages state prerequisites and an achievable outcome and walk through a
complete example. How-to pages solve one focused task with minimal background.
API pages document signatures, units, shapes, defaults, errors, and supported
backends/derivatives. Theory pages expose assumptions and mathematical conventions;
validation pages give the measured conditions and limitations. Each connects to
the other relevant page types instead of duplicating their full explanations.
For example, a material tutorial links its usage guide and `Material` reference,
then the constitutive model and its independent validation evidence.

Troubleshooting starts from observable symptoms: an unexpected imported checkout,
wrong phase or direction, unconverged results, nonfinite values, unsupported
derivatives, or excessive runtime. Link to the applicable convention or measured
limitation and distinguish user configuration errors from unsupported features.
Only document remedies and performance claims that have been verified.

The Learn landing page provides explicit Tutorials and Examples entries.
Existing runnable notebooks in `tutorials/` and scripts in `examples/` remain
canonical sources; link to them initially and use a deliberate rendering/include
step if embedding them later. Do not maintain separate copied tutorial bodies
for the repository and website. Keep expensive research runs outside the ordinary
HTML build and validate selected runnable examples separately.

Documents in the research workspace can be public library documentation. Their
current location alone does not make them scratch or exclude them from the site.
Import them by the question they answer:

| Research-document content | Public destination |
| --- | --- |
| Sequential basic and NVF textbook chapters | `docs/learn/` |
| Installation, source editing, debugging, extending a solver | `docs/development/` |
| Using an implemented feature or choosing settings | `docs/guide/` |
| Maxwell equations, RCWA/FDTD derivations, material and phase conventions | `docs/theory/` |
| Convergence, independent comparisons, measured accuracy and limits | `docs/validation/` |
| Maintained API/architecture rationale and proposed alternatives | `docs/design/` |
| Reproducible investigation code and selected underlying records | `studies/` |

The inspected active textbook is already a Sphinx source collection: a Korean
basic course of 20 chapters and three appendices, an NVF course of 15 chapters,
and 15 Python examples, combined with the checkout's English manual and API.
Use those maintained chapter sources, figures, interactive assets, and example
files as the migration input. Older standalone HTML textbooks are archival
versions; generated HTML and generated copies of the English manual are not
additional editing sources. Reuse or adapt existing example-verification,
provenance, and site-link checks rather than losing them during consolidation.

English is the canonical language for every public documentation section,
including the textbook, theory, validation, design, development guide, and API
reference. Published tutorial notebooks and example explanations follow the same
default.
The default site opens in English. Readers can explicitly choose Korean through
a language selector; Korean is an optional translation, not a separate set of
features or a prerequisite for publishing the English documentation.

Use the existing Korean research and textbook sources to prepare reviewed English
chapters, then align the optional Korean translations with those English pages.
Preserve the original scientific claims, caveats, and attribution in both.
Existing Korean text is migration input, not proof that its corresponding English
page already exists. Keep canonical English Markdown in the section directories
above and Korean translation resources under `docs/locales/ko/`. Use stable page
identities to keep language switching on the same topic and documentation version.
Where a Korean translation is absent or out of date, clearly indicate the English
fallback rather than implying that an old translation describes the current API.

Keep API identifiers, executable examples, and numerical conventions consistent
across languages. Share example files and language-neutral figures; translate
prose, captions, navigation, and explanatory labels as needed. API docstrings
remain English, and translated descriptions do not create a second API contract.
Document the source revision each translation has been reviewed against and
check available translations for broken links and missing assets during their
build. Update contribution rules to permit optional translation resources while
retaining English as the default for canonical docs and code comments.

The existing learning course describes RCWA and the current API: mark that scope
and update examples and conventions when the unified API lands. FDTD lessons
must be written against implemented, verified behavior.

For each imported document, record its repository-relative destination, dependent
figures/data, associated code or study, and whether it describes current behavior,
an unimplemented proposal, or a historical experiment. Preserve attribution and
scientific caveats; remove machine-specific paths and replace private links with
included assets or durable public references. A page is not complete when it
depends on a missing research-workspace file. Mixed notebooks or diaries can
produce an edited public explanation while the original record remains intact.

Move selected documents and their necessary assets in small reviewable batches.
Do not remove the source records until destination content, references, and
navigation have been checked. Unselected process notes and temporary logs remain
in the research workspace; their existence does not prevent importing the public
documentation. No document is promoted to an implemented API claim simply by
appearing on the site.

As part of that migration, update the documentation contributor guide, `_toc.yml`,
cross-links, and recursive documentation-contract checks for the new subfolders.
Apply code-target and measurement metadata requirements to the pages that make
those claims; general teaching chapters need not imitate a research-run report.
Preserve published URLs through redirects or explicit relocation pages where
possible, and keep one public API reference generated from public names rather
than old private implementation classes.
Retain warning-as-error Sphinx builds and public-symbol validation. Build the
docs against the matching code revision and label unreleased material clearly;
published release documentation must describe its matching release. A source
reorganization or local HTML build does not by itself publish a website.

Benchmarks measure stable performance and memory workloads with explicit problem
sizes, hardware/backend metadata, and comparison policy. Exploratory numerical
research stays in studies. Performance regressions and physical-accuracy tests
remain distinguishable, even if they share a small immutable fixture.

The dependency rule is one-way: development tools, studies, examples, and tests
may import the installed package; production code must not import those folders.
Reusable physics needed by the product graduates into an appropriate package
module with its public/private contract and validation, rather than being reached
through a studies import.

(implementation-checkpoint)=
## 13. Implementation checkpoint and acceptance evidence

### Implemented contracts

`wavebit.rcwa` and `wavebit.fdtd` are public modules with `check`, `prepare` and
`run`. Each accepts its own immutable `Problem` and `Numerics`, with explicit
backend, device and precision settings. A prepared object requires a complete
live problem on every `run(problem)` call and checks its reuse dependencies;
prior results retain their own state. Public jobs serialize the supported
physical and numerical declarations rather than a mutable last solver state.

Both methods use the shared material response and explicit `Units`. Geometry and
wavelength use the declared length unit; time is in seconds, frequency in Hz,
angular rates in rad/s and electrical conductivity in S/m. Public electric and
magnetic fields are V/m and A/m. Periodic scenes use bottom-to-top layers, a
cell-centred lateral origin, +z upward and angles in radians. Observation
metadata carries reference planes and source normalization.

The [RCWA guide](../guide/rcwa/index.md) covers layered stacks, material and geometry
declarations, single-incidence diffraction, requested fields, and supported
coherent/internal-source observations. Coherent-source runs expose boundary
power and requested fields; they do not invent a unique single-port R/T or a
group-independent scattering matrix. Coherent assembly is currently eager-only.
Scattering-matrix outputs require SMM. Host-only geometry decisions and adaptive
detuning cannot be assumed safe inside an external autodiff transform.

The [FDTD guide](../guide/fdtd/general-fdtd.md) covers physical domains, supported
structures, plane-wave and current-dipole sources, and requested records,
spectral fields, flux, diffraction and far-field observations. Its current
executor supports NumPy, JAX and Torch on CPU in double precision. Grid, duration,
CFL and PML are numerical inputs; the PML extends outside the physical domain.
The FDTD dipole amplitude is a current moment (electric A·m or magnetic-equivalent
V·m), whereas RCWA's electric dipole moment is C·m.

Material construction does not imply that both solvers support every tensor,
dispersion, loss, geometry or boundary combination. Frequency-domain tables are
not automatically fitted to causal time-domain models. `check` reports supported
combinations, and execution rejects unsupported combinations. Concrete
preparation and validation remain distinct from traced runtime validity checks.
No result's raw pulse spectrum is silently treated as monochromatic watts or as
complete normalized entering power.

Earlier factories, specialized solver workflows and teaching examples remain
explicitly available through `wavebit.compat.legacy`; the new modules do not
masquerade as those factories. The [compatibility integration checkpoint](../development/solver-integration.md)
records the preceding periodic-adapter work, and the [migration guide](../migration/index.md)
distinguishes the two interfaces.

### Repository and teaching migration

The repository separates installed source, solver tests, maintained examples,
tutorials, studies, development tools and documentation. Historical numerical
records and retained source snapshots are preserved through relocation and
provenance ledgers; their figures and logs have not been regenerated as fresh
solver evidence.

The selected teaching corpus contains 40 complete English chapters, 100 figures
and 15 maintained Python examples. Complete Korean chapter bodies and localized
assets are optional same-route overlays; other pages explicitly fall back to
English. The language inventory covers all maintained canonical Markdown and
114 published documentation images. Raw lab notes, earlier drafts and handoffs,
source illustrations, exploratory notebooks and standalone prototypes remain in the
private lab repository. The public documentation contains reviewed explanations and
selected numerical evidence; preserving research originals does not publish them.
See [learning provenance](../development/learning-provenance.md) for source hashes,
translation and figure records, and [documentation development](../development/documentation.md)
for build and contribution rules.

Two current-API starting notebooks cover RCWA and FDTD preparation, live input,
units and observations. The ten retained advanced notebooks identify their
compatibility imports and keep their original numerical code and stored outputs.

### Validation recorded at this checkpoint

Counts below describe separate focused runs and may overlap; they must not be
added into a single test total. Existing unsupported-feature skips remain skips.

| Validation | Recorded result and scope |
| --- | --- |
| API contract suite | 1,010 distinct cases passed across the complete API run and focused follow-ups. Covers the final API and declaration contracts, not every physical regime. |
| Shared modules | 96 passed; 3 existing unsupported-feature skips. |
| Public FDTD, new cross-solver and repository contracts | 38 passed, including the four public-API refinement cases described below. |
| Documentation contracts | 149 passed; original research code-target checks retained, plus exact-source preservation and archive-boundary checks. |
| Maintained example scripts | All 37 passed: 36 in the complete example run and the corrected anisotropic-material example in its focused rerun. |
| Current-API notebooks | Both new notebooks passed, with 6 code cells executed sequentially in separate fresh NumPy processes. |
| Distribution | Final wheel and source distribution passed strict Twine checks. Fresh virtual environments passed base and plotting installation checks, including 36 retained compatibility cross-solver cases; their maximum complex-amplitude error was 0.0060788554 within those cases' existing bounds. |
| English and Korean documentation | Strict bilingual Sphinx builds passed for 251 source pages in each language; 506 published HTML pages had no missing local link, fragment or asset targets, and every language switch retained its route. |
| Full default collection | All 16,987 selected cases accounted for: **16,697 passed, 289 skipped, 1 expected failure**. Exact test-node reconciliation across segmented runs and focused reruns found no missing or unexpected nodes and no unresolved failures. The research tier contains 2,721 deselected cases. |

The default result is the union of individually identified test nodes from
segmented executions and reruns of corrected cases; it is not one uninterrupted
green run. Each selected node contributes one final outcome. The existing strict
expected failure is
`tests/rcwa/test_contact_jones_probe.py::test_equal_material_split_is_equivalent_to_one_rectangle`:
retained artificial seams change the finite-order extension. It remains an
explicit limitation rather than a passing equivalence claim. The 289 skips and
2,721 research-tier deselections are not counted as passing executions.

This checkpoint does not claim a completed release, an executed full
research/slow tier, or universal convergence across models and backends.

### Four public-API diffraction comparisons

The [machine-readable evidence](../development/public-cross-solver-evidence.json)
records a homogeneous film and an off-centre one-dimensional grating, each under
s and p illumination, using the new public NumPy APIs. It retains the test source
hash and every measured refinement value. Reflected and transmitted orders
`(-1, 0, 1)` are compared as complex amplitudes and power efficiencies at shared
physical reference planes; the grating's translation supplies an independent
odd-order phase check.

RCWA truncation changes from `(7, 0)` to `(11, 0)`. FDTD spacing changes from
0.05 to 0.025 micrometres at 70 fs, followed by separate duration checks at 100
and 140 fs on the fine grid. Across these four cases, halving spacing reduces
the maximum amplitude and efficiency errors to 0.234–0.258 of their coarse-grid
values. At the final settings the largest amplitude error is approximately
0.009234, and the largest efficiency error is approximately 0.003509.

These measurements support the tested reference conventions and local spatial
and temporal refinement for these four problems. They are not a broadband,
oblique, lossy, tensor, three-dimensional or all-backend convergence guarantee.
Raw FDTD pulse energies are not compared to monochromatic RCWA powers; the
comparison uses each method's declared or measured incident normalization.
