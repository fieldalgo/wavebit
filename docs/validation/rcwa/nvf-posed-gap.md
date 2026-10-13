# Rigid lattice pose and live incidence for the conforming NVF model

`studies/rcwa/nvf_posed_gap/nvf_posed_gap.py` extends the polynomial-body NVF research model to
laboratory translation, rigid rotation of the pattern **together with its
square lattice**, and live polar/azimuthal incidence angles. Its complex
reflection/transmission amplitudes and p/s powers use the same cell equations,
Gram closure and eigenvector-free scattering propagation. It does not rotate
individual primitives inside a fixed lattice or change production dispatch.

## Coordinates, phases and polarization

The local geometry is still a horizontal rectangular envelope with ordered
material seams. Width, gap and seam coordinates are in nm; all angles are in
radians. If the lattice rotation is alpha, the local incident azimuth is
`lab_azimuth - alpha`. A laboratory translation d becomes `R(alpha).T @ d`
before multiplying every field/energy Fourier coefficient by

```text
exp(-2 pi i / period * (m d_local_x + n d_local_y)).
```

This same phase appears in outgoing diffraction amplitudes for a central
incident wave. Powers remain translation invariant; amplitude derivatives
are `-i G_lab_x A` and `-i G_lab_y A`. Tests check all complex orders,
including evanescent ones, against these analytic identities. Another test
independently rotates tensor components, Fourier indices and directional
orders through a quarter turn and compares the complex amplitudes.

`doubling_response` accepts live `theta` and `azimuth`, and returns `T`, `R`,
`reflected`, `transmitted`, `energy_error` and `taylor_tail_bound`. Amplitude
rows are all s orders followed by all p orders; columns are p then s
incidence. The polarization components are relative to each outgoing order
in the rotated local frame. Convert that frame when constructing laboratory
Cartesian vector fields. `doubling_powers` retains its older tuple API.

The central incident polarization frame uses the specified azimuth directly,
so it remains differentiable at normal incidence. Theta is signed with
`abs(theta) < pi/2`. Tests cover normal incidence, signed oblique incidence,
nonzero azimuth and an exactly grazing **layer** using an independent uniform
slab formula. An exactly grazing **exterior diffraction order** is still
excluded: its flux normalization is singular. NumPy raises and JIT returns
NaNs. Period, wavelength, exterior media and the lossless contract remain
fixed; the existing optional thickness setting is retained.

## Live parameter checks

With order 2, gap 17 nm, width 350 nm, seam 31 nm, materials 4/9, center
`(17,-21)` nm, lattice rotation .31, theta .2 and laboratory azimuth .47:

| quantity | p | s |
| --- | ---: | ---: |
| T | 0.943312026142 | 0.904149819534 |
| R | 0.056687973858 | 0.095850180466 |
| dT / d lattice rotation | -0.0451392740874 | 0.0211718902307 |
| dT / d theta | -2.05364981620 | 0.962735701302 |

This uses base rows 4, body degree 4, gap degree 6, x subdivisions 1, edge
fraction 1e-5 and the original global flux basis with grading 8/.01.
Power imbalance is below 5e-14. The full ten-parameter JIT Jacobian covers
gap, width, two translations, rotation, two incidence angles, seam and both
permittivities. It agrees with two independent modal NumPy secants under the
unchanged `atol=2e-9, rtol=3e-6` criterion. Translation power derivatives are
zero, and rotation and laboratory azimuth derivatives are opposite.

These comparisons validate the derivatives of the stated finite model; they
do not establish cell or outer Fourier convergence. The modal comparison
reuses the constitutive field but has independent propagation. The uniform
slab and covariance checks supply separate physical controls.

## Mixed-material contact and stable refinement

The single-material oblique contact test retains the independent Cartesian
contact gate of 1e-7 per nm. Extending the check to materials 4/9 exposed a
larger discrepancy near the real interior seam. It also occurs at zero
azimuth, so rigid pose is not its cause. Increasing outer order from 2 to 7
does not remove it; refining the auxiliary cell reduces it.

The [localized flux basis](nvf-local-flux.md) makes sharper material grading
numerically usable while preserving exactly the same trial space. For the
mixed fixture at local azimuth .37, the following order-2 comparisons use
base rows 8, body degree 4, x subdivisions 3, gap degree 6 and edge fraction
1e-5. Slopes use gaps 0, 1e-4 and 5e-5 nm with one-sided extrapolation.

| material grading radius / width | p slope difference from Cartesian | s slope difference from Cartesian |
| ---: | ---: | ---: |
| .01 | 3.791e-7 | 2.609e-6 |
| .001 | 4.179e-8 | 2.866e-7 |
| .0001 | 4.959e-9 | 2.915e-8 |

The final row passes the unchanged contact criterion for both polarizations.
JIT contact derivatives also pass at local azimuths 0 and .37 using a smaller
base-rows-4/body-degree-1 cell with the same x refinement. This smaller cell
is adequate for these contact controls, not a replacement for positive-gap
cell convergence. No Cartesian result enters the NVF calculation, and no
material derivative is frozen or adjusted to match the comparison.

The default .01 grading remains a coarse research setting: its mixed-material
contact slope is not certified by the passing finer-grid result. Neither is
general mixed-material contact certified across geometries, contrasts and
Fourier orders. `--contact-check` reports the candidate and independent
finite-order control; it deliberately does not label their agreement as a
converged optical result.

## Use and reproduction

```python
from studies.nvf_posed_gap import PosedGapModel

model = PosedGapModel(
    2, rows=4, body_degree=4, gap_degree=6, edge_fraction=1e-5,
)
response = model.response(
    17., dict(seams=[31.], values=[4., 9.]),
    center=(17., -21.), rotation=.31, theta=.2, azimuth=.47,
)
# response['T'] and ['R']: vectors in p, s order.
# model.transmission(..., polarization='p'/'s'/'both') selects powers only.
# Pass xp=jax.numpy with x64 enabled for live derivatives.
```

```bash
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_posed_gap --jax
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_posed_gap --rows 8 --subdivisions 3 --flux-basis local --grading-radius .0001 --azimuth .68 --contact-check
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_posed_gap --gap 0 --body-degree 1 --subdivisions 3 --flux-basis local --grading-radius .0001 --azimuth .68 --jax --contact-check
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests/rcwa/test_nvf_posed_gap.py tests/rcwa/test_nvf_local_flux.py
```

Measurements are in `studies/rcwa/nvf_posed_gap/records/nvf_posed_gap.log`,
`nvf_posed_contact_refinement.log` and `nvf_posed_contact_local_flux.log`.
The pose suite has 14 tests and the localized-flux suite has seven, including
a second mixed-material contact profile. Validation on 2026-09-09 across the
pose, local/global flux, material grading, polynomial spaces, scattering,
mirror, gap-convergence and earlier harmonic-energy suites: **148 passed,
2 strict xfailed**. The two expected failures are the unchanged earlier
Fourier field's s contact limit and the regularized modal material derivative;
they were not relabeled. The run is retained in
`studies/rcwa/nvf_posed_gap/records/nvf_pose_local_flux_validation.log`.

Production integration,
independent rotations within a fixed cell, arbitrary interfaces and the
previous high-order optical convergence gate remain open.
