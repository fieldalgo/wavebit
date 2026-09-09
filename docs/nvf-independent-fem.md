# Independent FEM reference for the tilted NVF contact

This investigation is still open. The new reference instrument is
[`studies/nvf_fem_reference.py`](../studies/nvf_fem_reference.py). It solves the
three-dimensional Maxwell problem with NGSolve H(curl) elements and imports no
Wavebit solver or factorization code. **It is a reference study, not a replacement
for rectangular NVF or a converged golden answer.**

The immediate purpose is to obtain an independently converged answer before
judging changes to the NVF extension or its assembly. S4 direct rule and the
Wavebit direct-rule control agree at the same Fourier truncation, but that does
not make either one a converged oracle. See the earlier comparison in
[`nvf-contact-followup.md`](nvf-contact-followup.md#independent-tilted-fixture-comparison-2026-09-08).

## Physical problem and discretization

All lengths in the FEM instrument are micrometers: period 0.7 square, wavelength
0.9, film thickness 0.3, rectangle width 0.64, rectangle height 0.7/sqrt(5), and
angle atan2(1,2). Inclusion/background/substrate permittivities are 12.25/1/2.25.
The incident plane wave has theta=0.2, phi=0, and p polarization.

The rectangular prism and its periodic images are clipped and united in the
geometry kernel, then meshed conformingly. Its volume is checked against the
analytic value, 0.0601055072352. This preserves the approximately 13.9 nm contact
run without a raster cutoff. Both lateral periodic identifications are checked
on the **generated mesh**. The Bloch phase is factored out of the unknown; the
curl in the weak form is `curl(u) + i k_parallel cross u`.

The incident and bare-substrate Fresnel fields are analytic. The scattered-field
right-hand side is `k0^2 (epsilon - epsilon_background) E_background` in the
inclusion. Electric and magnetic PML tensors follow the same coordinate
transformation. Power is the spatially averaged Poynting flux in the physical
superstrate and substrate, excluding the PML. Static condensation eliminates
interior element unknowns and reconstructs them after the sparse solve. A full
operator residual is checked after reconstruction.

NGSolve documents the [periodic finite-element construction](https://ngsolve.org/ngsolve/docs/i-tutorials/unit-2.12-periodicity/periodicity.html)
and its [H(curl) elements](https://docu.ngsolve.org/nightly/i-tutorials/unit-2.3-hcurlhdiv/hcurlhdiv.html).
The field and PML formulation in the instrument was written independently; no
external solver implementation was copied into Wavebit.

The second boundary option, `--boundary rayleigh`, omits both PML layers. It
couples the lateral-periodic finite-element trace to outgoing Rayleigh waves.
For lateral wave vector k and outgoing vertical rate gamma, the tangential
admittance is `Gamma = gamma I + k k.T/gamma`. Every propagating order must be
included, or the instrument raises. The base sparse form uses local impedance;
a low-rank correction replaces it with this exact admittance for the retained
orders. A Woodbury solve applies the correction using one sparse factorization.
Omitted evanescent orders retain local impedance, so the Rayleigh cut and physical
padding remain independent convergence parameters. This is **not** an exact
infinite boundary operator at a finite cut.

The general finite-element/DtN approach to biperiodic Maxwell scattering is
described by [Jiang et al., 2018](https://arxiv.org/abs/1811.12449). The present
instrument does not implement their adaptive error estimator, and their error
bounds must not be assigned to these results. It also reports two flux readouts:
outgoing Rayleigh power and volume-averaged physical Poynting flux.

## A failed periodic-boundary request was caught by the control

The first local FEM prototype selected outer faces within 1e-8. OpenCascade pads
face bounding boxes by about 1e-7 in these coordinates, so **both selections were
empty**. Calling `Identify` on those lists did not raise an error. The resulting
mesh had zero periodic pairs and answered a different boundary-value problem.
Its small algebraic residual did not reveal this error.

The current instrument checks nonempty, matched outer-face selections and checks
actual point-pair displacements in both directions after meshing. Tests also
verify that the periodic space has fewer independent unknowns than the ordinary
space. The rejected nonperiodic values are not included below.

## Measured controls and preliminary result

Here `h` is the outer maximum tetrahedron size; the patterned film requests h/2.
`p` is the H(curl) polynomial order. PML thickness and physical padding are each
0.4, with stretch 1+2i. A small energy error is not a continuum error bound.

The exact uniform-epsilon-4 slab has T=0.838052025533.

| h | p | uniform slab T | absolute error against analytic T | energy error |
| --- | --- | ---: | ---: | ---: |
| 0.14 | 2 | 0.853731655190 | 0.015679629657 | 0.003192760229 |
| 0.18 | 3 | 0.841376878322 | 0.003324852789 | 0.000209273140 |
| 0.18 | 4 | 0.838232037221 | 0.000180011688 | 0.000004092306 |

These controls were run with the corrected local prototype before packaging the
instrument. The packaged version adds a higher quadrature order for the analytic
source and explicit metadata/guards; its nonzero-source control is also exercised
by the optional test suite.

For the actual tilted contact:

| h | p | elements | unknowns before condensation | T | energy error |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.14 | 2 | 5884 | 56415 | 0.779444332792 | 0.001660836954 |
| 0.14 | 3 | 5884 | 146064 | 0.873537338349 | 0.000032048686 |
| 0.18 | 4 | 4254 | 217000 | 0.876390743842 | 0.000003790406 |
| 0.14 | 4 | 5884 | 300555 | 0.876511687509 | 0.000001813687 |

The same-p mesh change is only 0.000120944, but this sequence has **not** converged
in the absorbing boundary. At h=0.18, p=4, changing the PML stretch from 1+2i to
1+3i changes T from 0.876390744 to 0.870897410, a shift of 0.005493334. Both have
energy errors about 4e-6. Therefore **the 0.876 values are excluded as an accuracy
reference**, despite their small mesh difference, residual, and energy error.

The substrate (1,0) Rayleigh wave has gamma=1.507144577 per micrometer. With the
short 0.4-micrometer PML, its predicted round-trip attenuation in amplitude is
`exp(-2*gamma*0.4*stretch)`: 0.0896868 at stretch=2, and 0.0268592 at stretch=3.
The uniform slab does not excite this order, explaining why that control alone
could not qualify the patterned structure's PML.

## Rayleigh-boundary measurements

Physical padding is 0.4. Preliminary patterned results from the local prototype
used volume flux only; the packaged instrument retains that value as `volume_T`
and uses outgoing Rayleigh power as `T`.

| h | p | cut | elements | unknowns | volume T | volume energy error |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 0.18 | 3 | 2 | 3478 | 86316 | 0.862943303916 | 0.000277763392 |
| 0.18 | 4 | 2 | 3478 | 177625 | 0.866055904424 | 0.000010118006 |

Thus 0.866055904 is also a **preliminary value**, not a golden answer. The
h=0.14, p=4 Rayleigh refinement did not return an observable: it was terminated
after severe paging. A process-specific `vmmap` measurement reported a 23.6G
peak footprint and 15.8G swapped out. This is a resource-limited refinement,
not a numerical result and not evidence of a physical divergence.

At h=0.18, p=3, reducing the Rayleigh cut from 2 to 1 changes volume T by only
0.00000652516 (0.862943303916 to 0.862949829077). Both cuts retain every propagating
order. The cut=1 modal T is 0.863038885671, and its modal/volume disagreement is
0.0000890566. This boundary-cut check is much smaller than the current p=3 to p=4
change. It does not replace the finer-mesh check at p=4.

Several bounded linear-solver alternatives were checked before stopping this
refinement. SciPy GMRES with a projected NGSolve BDDC preconditioner failed its
600-iteration cap on the much smaller h=0.25, p=2 uniform control; positive and
indefinite preconditioner forms both failed. A SuperLU/MMD experiment without
diagonal pivoting reproduced the h=0.18, p=3 uniform Rayleigh result within
2e-14, with a full residual of 2.4e-13, but took about 50 seconds versus about
14 seconds for the original prototype's UMFPACK solve. Its patterned accuracy
and memory scaling were not qualified. The ordinary-pivot SuperLU variant was
stopped after several minutes on that small control. No alternative was
installed, and no failed/incomplete solve was converted into comparison data.
The large refinement and all linear-solver experiments have been stopped or
completed; none is left running by this investigation.

The packaged Rayleigh instrument's epsilon-4 uniform slab at h=0.18, p=3, cut=1
gives T=0.838312384096, volume T=0.837902615436, and analytic T=0.838052025533.
The two readouts disagree by 0.000409769. This observed flux error is kept
explicit instead of selecting the readout that happens to be closer.
No production NVF behavior or acceptance tolerance was changed.

## Reproduction

On the current host NGSolve 6.2.2606 is already installed in
the optional NGSolve environment. Run from the
NVF worktree:

Set `NVF_FEM_PYTHON` to the Python executable with the optional NGSolve dependencies.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 "$NVF_FEM_PYTHON" studies/nvf_fem_reference.py --hs .18 .14 --orders 4 --boundary pml
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 "$NVF_FEM_PYTHON" studies/nvf_fem_reference.py --hs .18 --orders 3 4 --uniform 4 --boundary pml
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 "$NVF_FEM_PYTHON" studies/nvf_fem_reference.py --hs .18 .14 --orders 4 --boundary rayleigh --cut 2
PYTHONPATH=$PWD "$NVF_FEM_PYTHON" -m unittest discover -s tests -p test_nvf_fem_reference.py
```

The optional tests use standard-library unittest because this reference
environment does not contain pytest. The ordinary Wavebit pytest environment
skips them if NGSolve is unavailable. Tests cover actual periodic pairs, missing
and one-axis-only identifications, the clipped periodic material geometry,
analytic and nonzero-source slab controls, the zero-contrast limit, and
condensation/reconstruction equivalence for both boundary formulations. The
Rayleigh tests also reject an omitted propagating order and require the mesh to
contain no PML. All **12 optional tests passed** in the NGSolve environment. They
validate the instrument; they are not additional NVF physics successes.
Selected results, with prototype-versus-packaged flux provenance, are retained in
[`studies/results/nvf_fem_reference.log`](../studies/results/nvf_fem_reference.log).
