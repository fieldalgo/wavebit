# Near-Rayleigh channels and diffraction-plane placement

Written: 2026-09-08

Targets:
- `src/wavebit/fdtd/_periodic/scene.py`
- `studies/fdtd/rayleigh/fdtd_rayleigh.py`
- `tests/fdtd/test_fdtd_diffraction_planes.py`

TL;DR: a coarse Yee mesh can place a retained diffraction channel on the wrong
side of its physical cutoff. The common scene adapter reports that disagreement
and allows extraction planes to move independently of padding. Correct cutoff
classification and a closer plane do not establish scattering accuracy: the
extreme near-Rayleigh TM experiment below remains unsettled.

## Reference and conditions

```sh
PYTHONPATH=. python studies/fdtd/rayleigh/fdtd_rayleigh.py
PYTHONPATH=. python studies/fdtd/rayleigh/fdtd_rayleigh.py --case reference
PYTHONPATH=. python studies/fdtd/rayleigh/fdtd_rayleigh.py --case narrow80
```

Measurements use JAX 0.11.0 with x64 enabled, NumPy 2.5.1 and arm64 CPU.
`--backend numpy` runs the same equations without JAX. The lossless binary
permittivity raster is `[[[2.25, 1.]]]`, thickness 0.2, period 0.8, in vacuum.
At center wavelength 0.96, theta=asin(0.2). The two requested wavelengths are
0.95808 and 0.96192, at fixed transverse k. The retained FDTD orders are -1, 0,
+1. Unless changed: aligned layers, Courant fraction 0.8, physical PML width 0.3,
padding scale 1, Gaussian width 3 and duration 300 optical periods.

The reference is Wavebit's independent RCWA formulation at half-order 32,
evaluated separately at each wavelength with theta adjusted to keep the same
transverse k. Errors are maximum absolute complex r/t differences over both
wavelengths and orders **-1 and 0**, separately for unit s and p inputs. The
strongly evanescent +1 channel is excluded from this error metric. No reference
value is taken from Meep or FDTDX in this experiment.

| Reference comparison | Maximum TE amplitude change | Maximum TM amplitude change |
| --- | --- | --- |
| RCWA 16 versus 32 | 1.26061e-5 | 5.03988e-5 |
| RCWA 64 versus 32 | 1.72231e-6 | 1.28275e-5 |

This reference refinement is small compared with the FDTD discrepancies below.
It supports diagnosing this particular experiment, not arbitrary grating corners.

## A physically closed order can be open on the mesh

For order -1, the physical cutoff is wavelength 0.96. At the two requested
wavelengths, normalized longitudinal q is respectively 0.0632139 and 0.0632772i.
The p projection gain is about 15.8. The continuum squared q changes from
+0.003996 to -0.004004. The discrete lower-cutoff parameters are:

| Resolution | At 0.95808 | At 0.96192 | Both classifications agree |
| --- | --- | --- | --- |
| 20 | +0.01094297 | +0.00302932 | No |
| 40 | +0.00573840 | -0.00223996 | Yes |

At resolution 20 the grid admits longitudinal propagation at the longer
wavelength although the physical channel is evanescent. Increasing runtime
cannot fix this spatial/temporal discretization shift. The new diagnostic marks
it with `cutoff_resolved=False`; outgoing powers still use the physical mask,
so closed orders carry exactly zero R/T. No channel is dropped and no material
or wavelength is retuned. Exactly grazing retained channels are rejected before
the four time-domain runs, including roundoff-sized offsets from the singularity.

The [guide](../../guide/fdtd/time-domain.md#diffraction-planes-and-cutoff-diagnostics)
defines all diagnostic arrays. A positive discrete cutoff parameter checks the
lower longitudinal passband edge only. Upper Nyquist limits, mesh scattering,
PML attenuation and temporal settling remain separate requirements.

## Temporal and boundary controls

The amplitude-tail diagnostic compares the complete DFT with the prefix omitting
the last 200 steps, at tolerance 1e-4. This window is 4.61880, 2.30940 and
1.15470 time units at resolutions 20, 40 and 80; equal step counts are not equal
physical durations. Tail maxima include **all three retained orders**, whereas
the RCWA error metric includes -1 and 0 only.

| Resolution | Duration | Plane distance | Pulse width | PML width | TE error | TM error | p-column tail |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 20 | 100 | default | 3 | 0.3 | 0.152207 | 0.273120 | 0.0310047 |
| 20 | 300 | default | 3 | 0.3 | 0.104150 | 0.236634 | 0.0287140 |
| 40 | 300 | default | 3 | 0.3 | 0.0496485 | 0.150144 | 0.0223551 |
| 20 | 300 | 0.2 | 3 | 0.3 | 0.153996 | 0.283126 | 0.0883198 |
| 40 | 300 | 0.2 | 3 | 0.3 | 0.0353269 | 0.226325 | 0.0503899 |
| 40 | 600 | 0.2 | 3 | 0.3 | 0.0352407 | 0.224920 | 0.0471576 |
| 40 | 300 | 0.2 | 3 | 0.6 | 0.0343691 | 0.217062 | 0.0515119 |
| 40 | 300 | 0.2 | 10 | 0.3 | 0.0356140 | 0.165121 | 0.0309371 |
| 80 | 300 | 0.2 | 10 | 0.3 | 0.0209972 | 0.177212 | 0.0164864 |
| 40 | 300 | 0.2 | 10 | 0.6 | 0.0346954 | 0.155538 | 0.0315601 |

All rows fail `state_decayed` and `scattering_settled`. The default extraction
planes are approximately one wavelength from the interfaces; the study reports
realized coordinates for every row. Moving them to distance 0.2 improves some
TE errors while making some TM errors larger. Longer runs, doubled PML width
and narrower pulses do not settle this TM response at these settings. Resolution
80 improves TE but does not show monotonic TM convergence.

At resolution 40 with default planes, TE R+T is 0.991297 and 1.006955 across the
band; TM R+T is 1.001158 and 0.999613. The apparently good TM power balance hides
an amplitude error of 0.150144 and a non-negligible tail. Neither energy balance
nor the passing cutoff classification is sufficient evidence.

Meep's public [oblique-incidence mode-decomposition tutorial](https://meep.readthedocs.io/en/latest/Python_Tutorials/Mode_Decomposition/#reflectance-and-transmittance-spectra-for-planewave-at-oblique-incidence)
discusses diffraction thresholds and slowly absorbed glancing waves, and uses a
narrower source spectrum to reduce their excitation. That motivates the width,
runtime and boundary controls here. The current measurements do not isolate PML
reflection from guided or other long-lived grating modes, so the remaining error
is not assigned to one mechanism. Only public documentation was consulted;
Meep implementation was neither inspected nor copied.

## What the plane control establishes

`diffraction_distance` makes a useful convergence parameter explicit without
resizing the grid or moving the source/PML. Its scalar or incidence-relative pair
is snapped to exterior electric z nodes. Mixed interface cells are excluded.
Bottom-incidence diagnostics retain original scene coordinates. Interface phase
references and requested record bounds retain their meanings.

Independent Fresnel/Airy refinement tests exercise a non-grid-aligned-thickness
TM slab with close planes. Additional tests cover original-coordinate bottom
records, graded padding, NumPy/JAX/Torch material derivatives, magnetic conical
half-space channel metadata, and a two-dimensional order window. The latter
exercises the azimuth array in `(order_y, order_x)` shape rather than a flattened
list. Exact cutoff refusals run before time stepping; an input whose continuum
and grid cutoff signs differ is permitted and visibly diagnosed.

Plane movement is not expected to preserve every coarse-grid calibrated number.
The empty-medium reference itself can scatter from a nonuniform mesh, and a
finite PML can return a wave. Their interference changes the reference sampled
at different positions. The plane/record regression removes the measured
single-polarization calibration ratio using an independent frequency plane,
then checks the complete time history to roundoff. Slab accuracy is checked
separately against Airy under refinement. The
hard near-Rayleigh table remains an open accuracy limit of the research alpha;
these diagnostics do not change the default source or PML equations, and no GPU
or fresh external-solver execution is claimed.
