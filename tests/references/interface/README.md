# External reference interfaces

| Entry | Question |
| --- | --- |
| `meep_ringdown.py` | Can source-free time traces be generated through Meep's public API for independent mode-fitting checks? |
| `meep_source_notch.py` | Does a unit-peak custom source zero suppress a known undamped material mode, with fields matching the native solver at the same current times? |
| `meep_halfspace.py` | Do magnetic half-space perturbations give stable complex-amplitude derivatives under smaller perturbations and longer runs? |
| `Reticolo.py`, `__init__.py` | How do tests access the existing separately installed RETICOLO oracle? |
| `reticolo_res2.m`, `reticolo_res3.m`, `reti_aniso.m` | How do the existing RETICOLO bridge calls return scattering and field data? |

See [the parent reference contract](../README.md) for each oracle's scope and
environment. These are project-owned interfaces, not bundled external solvers.
Meep implementation code must not be opened, inspected or copied. The new
ringdown generator uses documented public simulation calls only and suppresses
automatic implementation-source excerpts in external exceptions/warnings.

The half-space generator perturbs epsilon/mu above and below a magnetic film
on a fixed cell. Its frozen archive retains both 1e-3/5e-4 perturbations and
50/70 durations, all four complex-amplitude derivatives and solver metadata.
The [verification note](../../../docs/fdtd-halfspace-gradients.md) gives the
scene, measured controls and limits. Generate with a separate Meep interpreter:

```shell
"$MEEP_PYTHON" tests/references/interface/meep_halfspace.py --output /tmp/meep-halfspace.npz
```

Run the ringdown generator by hand with an interpreter containing Meep:

```shell
"$MEEP_PYTHON" tests/references/interface/meep_ringdown.py --output /tmp/meep-ringdown.npz
```

The frozen archive was generated with Meep 1.34.0. Both cases have epsilon=2,
Courant=0.4, periodic boundaries and a Gaussian Ey current source of width 0.3
in frequency and cutoff 4. Records use the point (0,0,0), sampled every 0.08 for
40 time units after the source ends. The archive retains solver version and
material/grid constants beside the measured time/value arrays.

The Lorentz case is uniform in a 0.4-cubed cell at resolution 10, with omega0=3,
gamma=0.1, strength=2 and source frequency 0.5. Constant displacement gives a DC
component plus two conjugate oscillations. The cavity case has cell size
(0.4,0.4,1), resolution 20, electric damping 0.1 and source frequency 0.7 with
spatial amplitude cos(2*pi*z). Its two conjugate poles are checked against the
independent discrete Maxwell characteristic polynomial. This archive compares
field traces, not Harminv estimates, and does not validate open-boundary Q.

`test_meep_ringdown.py` in the parent directory needs only NumPy and Wavebit.
Do not regenerate a frozen archive merely to make a failing test pass.

The source-notch generator uses `CustomSource` to compare a finite Gaussian
and its one-step delayed difference. It uses the same uniform material with
gamma=0, dt=0.04, Gaussian width 5, peak time 25 and carrier at the independent
discrete material frequency 0.503628254518. The Gaussian is zero outside
0<=t<=50; the filtered waveform extends to 50.04 and is normalized to unit
peak on the half-step current samples. Both runs record Ey every 0.08 for ten
units after the declared source end. The frozen `golden_meep_source_notch.npz`
was generated with Meep 1.34.0, using public API calls only:

```shell
"$MEEP_PYTHON" tests/references/interface/meep_source_notch.py --output /tmp/meep-source-notch.npz
```

The native comparison feeds the same half-step currents to a uniform ADE grid
and checks complex E at the recorded integer times on NumPy, JAX and Torch.
It validates a uniform material oscillator and source timing, not arbitrary
open-boundary ringdown or a guaranteed acceleration of scattering convergence.
