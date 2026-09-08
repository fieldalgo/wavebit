# External reference interfaces

| Entry | Question |
| --- | --- |
| `meep_ringdown.py` | Can source-free time traces be generated through Meep's public API for independent mode-fitting checks? |
| `Reticolo.py`, `__init__.py` | How do tests access the existing separately installed RETICOLO oracle? |
| `reticolo_res2.m`, `reticolo_res3.m`, `reti_aniso.m` | How do the existing RETICOLO bridge calls return scattering and field data? |

See [the parent reference contract](../README.md) for each oracle's scope and
environment. These are project-owned interfaces, not bundled external solvers.
Meep implementation code must not be opened, inspected or copied. The new
ringdown generator uses documented public simulation calls only and suppresses
automatic implementation-source excerpts in external exceptions/warnings.

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
