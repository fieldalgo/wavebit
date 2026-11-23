# Automatic stopping with incident source columns

The public `fdtd.DecayStop` policy admits concrete CPU `PlaneWave` and axis-aligned
`TFSF` sources. It reads each incident column's E/H, CPML histories and supported
material memories; layered TFSF also contributes its uniform driver. All groups
retain separate all-time peaks. This extends state coverage without changing the
Yee updates, source waveform, amplitude threshold or fixed-duration executor.

`tests/fdtd/test_source_decay.py` exercises the public API and separately enumerates
the actual kernel arrays to reconstruct stopping decisions. The independent replay
does not call the production group selector or stopping criterion. It checks every
step, the first complete eligible window and each all-time peak.

The registered fixtures use a 0.05 µm uniform grid, a 1 µm Gaussian carrier with
frequency standard deviation `0.05*c/(1 µm)`, cutoff 0.001, threshold 0.02 and a
`1 µm/c` window. The hard cap is 1,800 steps. The layered TFSF has an epsilon-2.25
slab from z=-0.4 to 0 µm. This is a fixed-grid temporal check, not a spatial
convergence measurement; the layer emits the existing coarse-mesh warning.

For PML, an initial expectation that every source/backend would stop before this
cap was not met. NumPy uniform cases stopped at 1,750 steps, while JAX uniform
cases and the layered TFSF could reach the cap. Extending the NumPy layered cap
to 3,600 did not resolve its small residual groups. These observations are retained
as limits of an all-memory relative-amplitude policy, not used to remove groups or
relax thresholds. Backend stopping counts need not agree for round-off-dominated
groups. A cap result must still reproduce the requested fixed-duration run.

Separate fixtures explicitly select the existing `fdtd.ConductiveAbsorber(cells=12)` and require
early stopping on all three CPU backends, with the same threshold/window/cap.
This is a different absorbing boundary, not a replacement for PML or a claim of
equivalent reflection. Both boundary choices compare final fields, point records,
incident E/H histories and raw Fourier fields against an ordinary run to the
actual step count. The inherited prefix allowance is 128 float64 epsilons times
the reference scale; SI Fourier fields use `1 µm/c` as their scale.

Additional instruments independently keep each column/driver E/H or CPML array,
and each column polarization/previous/increment array, active while other state
groups are quiet. Each must hold off stopping. Reference histories are excluded.
Unknown terms, duplicate memory names and malformed nested states must refuse.
Delayed Gaussian gates and JAX/PyTorch derivative refusals remain in force.

## Fixed-grid time comparison

On macOS arm64, Python 3.12.9, NumPy 2.5.1, the following runs were compared with
5,400-step fixed runs of the same scene and boundary. The tail is the largest
remaining point-field amplitude divided by that record's peak. Fourier differences
use each wavelength's own largest field magnitude; the table reports the worst of
0.9, 1.0 and 1.1 µm. The pre-registered limit was 0.02 for both quantities.

| Source | Padding | Actual steps / reason | Relative time tail | Worst relative Fourier difference |
|---|---|---|---|---|
| PlaneWave | PML | 1750 / decay | 0.0001790 | 0.0002392 |
| Uniform TFSF | PML | 1750 / decay | 0.0001790 | 0.0002392 |
| Layered TFSF | PML | 1800 / cap | 0.0001142 | 0.0001426 |
| PlaneWave | Absorber | 1750 / decay | 0.0001083 | 0.0001454 |
| Uniform TFSF | Absorber | 1750 / decay | 0.0001083 | 0.0001454 |
| Layered TFSF | Absorber | 1750 / decay | 0.0005175 | 0.0004225 |

These measurements concern these fixtures and finite time intervals only.
They do not convert an amplitude threshold into a general spectral-error bound.
The all-backend prefix tests additionally use JAX 0.11.0 and PyTorch 2.14.1 on CPU.
They do not establish behavior on another operating system or on a GPU.

Reproduce the regression selection from an installed development environment:

```bash
python tools/run_verification.py --profile full --slow exclude --workers 0 \
  --path tests/fdtd/test_source_decay.py \
  --path tests/fdtd/test_fd_decay_stop.py \
  --path tests/fdtd/test_physical_stopping_absorber.py
```

The stop is an amplitude heuristic, not an electromagnetic energy, spectral-error,
future-echo or convergence certificate. It does not establish GPU or AD support.
Use a sufficiently long fixed run to assess the actual observable's temporal error.
