# Validation studies

- [RCWA](rcwa/index.md)
- [FDTD](fdtd/index.md)

## How the solvers are checked

The ordinary solver is checked against references, not only against itself. These checks
cover their recorded cases; they do not certify the experimental contact auxiliary solver's
optical convergence or opening derivative:

* **Three outside RCWA solvers** — RETICOLO through an Octave bridge, plus fmmax and S4. Their
  numbers are committed as goldens and checked in CI, so the half that needs software this
  project does not ship runs once, by hand. Two of them validate individual diffraction
  *orders*, not only totals: RETICOLO agrees to ~1e-13, and fmmax's `FFT` formulation, which
  assembles the operator `factorization='laurent'` does, agrees on every order of both
  polarisations to ~3e-11 ([`test_fmmax_laurent_golden.py`](../../tests/references/test_fmmax_laurent_golden.py)).
* **A different method entirely** — Wavebit's own FDTD ([`docs/guide/fdtd/fdtd.md`](../../docs/guide/fdtd/fdtd.md)) solves
  the same scene in the time domain, and Meep is the outside reference for it. RCWA truncates a
  Fourier series and FDTD discretises space and time, so a defect surviving both belongs to
  neither. Meep's answers are committed as
  [`tests/references/golden_meep_1d.npz`](../../tests/references/golden_meep_1d.npz) (the 1D
  benchmark, both polarisations, two resolutions, with a planar closed-form calibration
  gating the run) and `golden_meep_crossed.npz` (a crossed grating at a conical mount,
  per-order complex amplitudes), and CI reads them on every push: on the benchmark grating at
  the 5 nm grid, our worst per-order gap against the converged RCWA answer is 6.6e-4 (TE) /
  6.9e-4 (TM), against Meep's own 4.9e-4 / 7.4e-4 on that grid — read both against the fact
  that the Meep column moves by 1.6e-3 between two of its own PML settings at that same
  resolution, for reasons §2 records as unexplained, so neither figure is tighter than the
  oracle's own spread. Those four numbers, and every
  cell of the table they summarise, are read back out of the markdown and compared to
  [`golden_fdtd_table.npz`](../../tests/references/golden_fdtd_table.npz) — with the conditions all
  three columns were taken at written out by value in `tests/references/_meep_1d_cases.py`,
  because until 2026-09-07 they were the phrase "at the shipped defaults" and one column had
  quietly gone stale behind it. Meep is GPL and this project is Apache-2.0, so it is run as a
  black-box oracle and is not a dependency of anything here.
* **A published benchmark table** — Li's 1997 crossed-grating checkerboard as tabulated by
  Demésy et al. (2012). All nine transmitted orders agree to a worst 2.06e-4, on the oblique
  primitive cell, which puts that path against numbers from outside this project.
* **Energy conservation** for lossless structures: the tests hold |R+T−1| below 1e-12 to
  1e-10 on most scenes, and below 1e-6 to 1e-4 beside a Rayleigh point.
* **Cross-backend agreement** — NumPy, JAX and PyTorch on the same input.
* **Analytic gradients against finite differences**, for the parameters an optimizer moves.

The suite is [`tests/`](tests); everything with a statable criterion lives there, including the
slower cross-backend and optimizer checks. CI runs its default set on every push. The tests of
the experimental conforming NVF module and its study instruments form a
[research tier](../../tests/README.md#the-research-tier) that CI does not run; it is run while that
research code is being developed. Every eager NumPy solve also checks what it is
about to hand back — a NaN or infinity raises `FloatingPointError` naming the contaminated
array.

There is deliberately no generic "R+T above 1" warning. It cannot tell a broken solve from a
structure that legitimately amplifies: gain is `Im(eps) < 0` here, and `R + T > 1`, or a
negative order under total internal reflection, is then the right answer. Energy is asserted
per scene in the suite, where the expected number is known; see
[`docs/reference/common/result-guards.md`](../../docs/reference/common/result-guards.md).

[Imported teaching records](../development/learning-provenance.md) retain their original source revision and toolchain. They are historical evidence, not a claim that the present checkout has rerun them.
