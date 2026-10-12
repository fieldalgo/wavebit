# Total-depth derivatives with moving exterior reference planes

Written: 2026-09-09

Targets:
- `src/wavebit/_core/scene_fdtd.py`
- `src/wavebit/manifest.py`
- `studies/fdtd_variable_depth.py`
- `tests/test_fdtd_variable_depth.py`
- `tests/test_fdtd_variable_depth_optics.py`
- `tests/references/interface/meep_halfspace.py`
- `tests/references/test_meep_variable_depth.py`

TL;DR: `variable_depth=True` permits independent positive layer lengths on a
fixed nominal mesh. Complex amplitudes follow the physical stack faces,
including their phase derivatives. Both incidence sides preserve laboratory
coordinates. Smoothed magnetic-film derivatives refine toward independent
transfer-matrix optics, and new public-API Meep differences check the moving
exit convention. Finite-grid errors and smoothing bias remain separate.

## Geometry and reference planes must move together

The original `mesh_thickness` option moves internal boundaries while keeping
their total fixed. The new option keeps the same static mesh selection but
places physical faces at `[0, cumsum(thickness)]`, including a live total D.
No tensor is detached to choose nodes, time step, source, PML or measurement
indices. All physical lengths must remain positive and represented faces must
remain distinct. Interfaces and averaging supports must clear the fixed
diffraction stencils before time stepping.

It would be incorrect simply to remove the old total-depth check. If a fixed
reference plane is at D0, transmission to the physical exit obeys

```text
t(D) = t_fixed(D) exp[-i kb (D0-D)]
dt/dD = exp[-i kb (D0-D)] [dt_fixed/dD + i kb t_fixed(D)].
```

For jointly changing angle or half-space material, kb also changes; that
derivative remains in the backend graph. The underlying forward/backward
plane-wave phases are the standard convention described in the public
[Meep mode-decomposition theory](https://meep.readthedocs.io/en/latest/Mode_Decomposition/#theoretical-background).
Only public documentation and API execution were used. No Meep implementation
or excluded Meep-derived FDTDX implementation was inspected or copied.

Uniform meshes use their discrete Yee longitudinal propagation constant.
Fitted or graded meshes integrate each static cell's phase over the traversed
fraction. At a node the right cell supplies the one-sided position derivative.
That convention is tested directly with unequal cell widths and independently
specified complex phase rates. Adjacent unequal widths can create a phase-slope
kink; material smoothing alone does not remove that discretization effect.
Adjacent integration intervals share the stored node coordinates. Rebuilding
each right endpoint independently as `node+width` can overlap a neighboring
interval by roundoff and count two position derivatives at a node. A separate
uniform-phase regression checks those exact shared-node positions on JAX/Torch.

## Keep bottom-incidence coordinates fixed

The bottom solve reflects around the nominal total D0, not the live D.
Its physical faces become `D0 - reverse(original_faces)`. Thus its incident
face can move while its mesh remains fixed. Both incident calibration and
reflected/transmitted phase corrections use the live reflected faces.
Reflecting diagnostics back around the same D0 preserves user monitor and
record coordinates in the original frame.

Tests change total depth inside a single JAX trace and require unchanged
nodes, source position, monitor coordinates and cropped record coordinates.
Invalid traced geometry sets `stable=False` and masks complex scattering and
normalized records/monitors with NaNs. Separate uniform-medium tests require
the exact discrete transmission `exp(i kz D)` and derivative `i kz exp(i kz D)`
for both incidence sides and both JAX/Torch. These tests would detect a
missing reference-plane derivative even if autodiff matched the same
incorrect finite-difference calculation.

## Averaging the moving exterior face

At positive `layer_smoothing`, the moving exterior face is averaged over
the same static translation box as the internal interfaces. The fixed face
at z=0 stays sharp. First and last layers must be at least half the width,
including a single-layer film. Internal averaging regions may overlap.
The graded profile extends half a width past the physical exit reference;
amplitudes are extrapolated to the centre of that transition.

Independent piecewise quadrature checks the material fractions and their
reflection. A uniform-mesh crossing test distinguishes a persistent sharp
derivative jump from the vanishing two-sided gap of the averaged response.
Joint tests compare JAX/Torch with two NumPy perturbation steps for depth,
permittivity and conical incidence, including unequal fitted cell widths.
A separate rotated-recipe case combines both layer lengths, transverse width
and rotation. These are fixed-grid derivative checks, not continuum accuracy
claims for arbitrary patterned designs.

## Independent magnetic-film accuracy

The film has epsilon 2.25, mu 1.12 and thickness 0.173. Incident-side media
in the original top/bottom frame have `(epsilon,mu)=(1.2,1.1)` and `(1.6,1.15)`.
Wavelengths are 0.98 and 1.02 at normal incidence. The moving film/substrate
transition has fixed averaging width 0.025; the entrance is sharp.

An independent transfer-matrix reference integrates 512 constant slices
through the linear epsilon/mu ramp. It removes the substrate-spacer phase
from the earlier fixed-exit reference and differentiates that removal.
Lossless reciprocity supplies the opposite incidence, independently checked
against swapped-medium Airy formulas for zero smoothing. Two perturbation
steps and 1024 slices control reference differentiation and integration.

Wavebit uses a single physical layer, nominal mesh thickness 0.2, aligned
nominal cells, physical PML thickness 1, duration 65, pulse width 3 and
directional E/H extraction. Maxima below cover the two wavelengths' complex
reflection/transmission coefficients. Derivatives are per model length unit,
not percentages.

| Incidence | Cells per length | Complex amplitude error | Complex depth-derivative error | Maximum energy residual |
| --- | ---: | ---: | ---: | ---: |
| Top | 20 | 0.0176799 | 0.253646 | 0.00661942 |
| Top | 40 | 0.00420258 | 0.0539120 | 0.00159626 |
| Top | 80 | 0.00104281 | 0.0143617 | 0.000395485 |
| Bottom | 20 | 0.0191737 | 0.241769 | 0.00657346 |
| Bottom | 40 | 0.00451114 | 0.0538182 | 0.00159362 |
| Bottom | 80 | 0.00112005 | 0.0134188 | 0.000395458 |

Both directions meet the preselected fine-grid budgets: amplitude 0.002,
depth derivative 0.02 and energy residual 0.001. The coarse-grid derivative
errors are substantial despite a correct discrete differentiation graph.

Extending the fine-grid run from 65 to 85 changes complex amplitude by at
most 3.83e-15 and its derivative by 7.47e-14. This removes runtime length as
the dominant explanation for the tabulated errors in this case.

The fixed averaging width also changes the physical problem. Comparing
the graded and sharp continuum references gives amplitude/derivative biases
of 0.000802116/0.0159141 for top incidence and 0.000807975/0.00467628 for
bottom incidence. These remain when refining the mesh at fixed width; do
not treat the graded derivative as the exact derivative of a sharp film.

## Independent Meep perturbations

The public generator's new `--parameter depth` mode places its output phase
reference at each perturbed film thickness. Its earlier `thickness` mode
keeps the output at 0.4 and retains its old behavior. The new mode uses
central forward differences, not Meep adjoint.

Meep 1.34.0 uses the same linear epsilon/mu profile through its public global
material function, a periodic 3D cell with one cell per transverse axis,
z cell length 10, physical PML thickness 1 and Courant 0.4. The source is at
-3 and fixed E-field measurement planes are at -1 and 1.2. A sine carrier
with Gaussian envelope width 1.2 and peak time 6 stops at time 12; DFT
sampling uses every step. Separate uniform-medium runs calibrate the source.
Independent homogeneous Yee propagation removes exterior phases.

Each archive stores durations 50/70 and perturbations 1e-4/5e-5. The maximum
runtime changes are zero to stored precision in amplitude and 1.58e-12 in
derivatives. Halving the difference step changes a derivative by at most
2.02e-6. Tightening material averaging tolerance from 1e-8 to 1e-10 while
doubling transverse cells changes amplitude by 1.79e-11 and derivatives by
2.29e-7. These controls were checked before freezing the candidate files.

| Meep cells per length | Complex amplitude error | Complex depth-derivative error |
| ---: | ---: | ---: |
| 40 | 0.00444142 | 0.0399470 |
| 80 | 0.00110346 | 0.0104899 |

At resolution 80 the two FDTD calculations differ by 0.000765581 in complex
amplitude and 0.00746733 in the complex depth derivative. Their agreement is
checked in addition to each solver's own graded-continuum error; neither
comparison is substituted for the other.

This external comparison covers normal top incidence of the selected
magnetic film. Bottom incidence has independent transfer-matrix, Airy,
reciprocity and fixed-grid checks. These bounds do not validate every
patterned, dispersive, resonant or oblique combination.

## Reproduce

Measurements use an arm64 CPU, Python 3.12.9, NumPy 2.5.1, JAX 0.11.0 x64
and Torch 2.13.0 float64. The separate Meep environment uses Python 3.12.14,
NumPy 2.5.2 and Meep 1.34.0. No GPU or performance claim follows.
See the [API guide](guide/time-domain.md#changing-the-total-stack-depth) for
the input and coordinate contract.

```shell
python -m studies.fdtd_variable_depth --output artifacts/variable-depth.json
pytest -q tests/test_fdtd_variable_depth.py tests/test_fdtd_variable_depth_optics.py
pytest -q tests/references/test_meep_variable_depth.py
```

Generate candidate references in a separate Meep environment:

```shell
python tests/references/interface/meep_halfspace.py --parameter depth --smoothing .025 --resolution 40 --output artifacts/meep-variable-depth-coarse.npz
python tests/references/interface/meep_halfspace.py --parameter depth --smoothing .025 --resolution 80 --output artifacts/meep-variable-depth.npz
python tests/references/interface/meep_halfspace.py --parameter depth --smoothing .025 --resolution 80 --subpixel-tol 1e-10 --transverse-cells 2 --output artifacts/meep-variable-depth-refined.npz
```

Keep actual candidate data and inspect runtime, perturbation, integration
and physical-reference checks before replacing any archive. Wavebit's
accuracy is checked separately against the controlled continuum reference.
