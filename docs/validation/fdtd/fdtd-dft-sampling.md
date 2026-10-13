# Frequency-monitor subsampling cost and error

Written: 2026-09-08

Targets:
- `src/wavebit/fdtd/_periodic/grid.py`
- `src/wavebit/fdtd/_periodic/runtime.py`
- `studies/fdtd/dft_sampling/fdtd_dft_sampling.py`
- `tests/fdtd/test_fdtd_dft_sampling.py`

TL;DR: skipping DFT evaluations reduces monitor work, but changes quadrature.
Keep stride 1 as the default and require an explicit accuracy comparison for
larger strides.

## Reproduction and results

Run `PYTHONPATH=. python studies/fdtd/dft_sampling/fdtd_dft_sampling.py`. The measured case uses a
32 by 8 by 8 vacuum grid, spacing 0.1, dt 0.02, 400 updates, four monitor planes,
32 wavelengths from 0.8 to 1.2, and four z-PML cells. An x-directed current sheet
at z index 10 emits a Gaussian complex carrier at wavelength 1, centered at time
2 with width 0.6. No field histories or convergence diagnostics are enabled.

Measured on arm64 CPU with NumPy 2.5.1 and complex128 arrays. Times are medians
of three runs after an every-step reference run warmed imports and allocation
paths; they include the native solver call, not process startup. Other machine
load can change these short timings. These are not GPU/JAX/Torch speed claims.

| Stride | DFT evaluations | Seconds | Relative electric-spectrum L2 error against stride 1 |
| --- | --- | --- | --- |
| 1 | 400 | 0.1301 | 0 |
| 2 | 200 | 0.1041 | 0.000287422 |
| 4 | 100 | 0.0870 | 0.000865585 |
| 8 | 50 | 0.0765 | 0.00203770 |

At stride 4 the whole native call took about 33% less time in this monitor-heavy
case, with about 0.087% relative spectral L2 difference. This is an error/cost
tradeoff, not equal-accuracy acceleration. The remaining Yee/ADE/CPML work and
spectral accumulator dimensions are unchanged. Few-frequency scenes can benefit
less, and compiled backend branching has different costs.

## Numerical contract

A sampled demodulated value carries the number of updates it represents, clipped
at the final block. E and H keep their native timestamps. No low-pass filter
precedes decimation. The requested-band Nyquist check only rules out an obvious
invalid setting; unrequested high-frequency transients can still alias.

Tests independently force an exact monochromatic uniform field and verify the
integral, magnetic half-step phase, a partial terminal block and a prefix boundary
inside a sample block. They check JAX/Torch sensitivities and unchanged native
fields/histories. A Gaussian slab band compares complex r/t against every-step
accumulation at absolute tolerance 1e-5 and the prefix diagnostic against a
separate shorter run. These cases do not justify that tolerance for arbitrary
pulses, high-Q cavities or dispersive transients.

The optional-stride design was informed by the previously reviewed MIT FDTDX
phasor detector; this implementation is independent and copies no source code.
