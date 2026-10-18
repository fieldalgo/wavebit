# Temporal boundaries: a permittivity that jumps during the run

A temporal boundary is the time-domain mirror of an interface. A region whose permittivity
changes from `epsilon_before` to `epsilon_after` at one instant keeps a wave's wavenumber and
changes its frequency, and the wave splits into a forward and a backward wave. No
single-frequency method can express this, RCWA included. `fdtd.SwitchedMaterial` declares
such a medium.

```python
import numpy as np
from wavebit import Units, fdtd
from wavebit.units import C0

medium = fdtd.SwitchedMaterial(epsilon_before=1.0, epsilon_after=4.0,
                               switch_time=0.3e-6 / C0)        # seconds
problem = fdtd.Problem(
    units=Units(length='um'),
    scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.01, .01, 1.)),
        boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.Periodic()),
        structures=(fdtd.Structure('switched', fdtd.Slab(z=(-.5, .5)), medium),)),
    initial=fdtd.InitialField(E=(None, lambda x, y, z: np.cos(2 * np.pi * z), None)),
    outputs=fdtd.Outputs(monitors=(fdtd.Record('p', at=(0., 0., 0.), component='Ey'),)))
result = fdtd.run(problem, numerics=fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.01),
                                                  duration=fdtd.Duration(steps=430)))
```

## What is continuous

The update is written for the electric displacement, `dD/dt = curl H` with
`D = epsilon(t) E`. Across the jump, `D` and the magnetic field are continuous and `E` is
not. For a forward wave of field amplitude `E0` and indices `n1`, `n2`, matching both gives
([Morgenthaler 1958](https://doi.org/10.1109/TMTT.1958.1124533);
[Xiao, Maywar and Agrawal 2014](https://doi.org/10.1364/OL.39.000574)):

```text
E_forward  / E0 = n1 (n1 + n2) / (2 n2^2)
E_backward / E0 = n1 (n1 - n2) / (2 n2^2)
omega_after     = omega_before * n1 / n2
```

At `epsilon 1 -> 4` that is `0.375` and `-0.125`, and the frequency halves.

Writing the same jump as `epsilon(t) dE/dt = curl H` keeps `E` continuous instead. That is
the right equation away from the jump and the wrong one at it. The frequencies and
velocities still come out right, but both amplitudes are exactly twice the values above, at
every resolution. The regression tests keep this mutant failing.

## Scope

The material is lossless and isotropic, and it jumps once. Both permittivities may be traced
and differentiated on JAX and PyTorch. The switching time is a static number of seconds. The
jump lands on the first E time level at or after `switch_time`, so `switch_time` must be at
least two time steps into the run.

The structure is refused in these cases. In each one the correction would not be exact:

| Refused | Why |
|---|---|
| A smoothed shape (`Box`, `Sphere`, `Cylinder`, `Prism`) | interface cells mix two media, and only one is switched; use `fdtd.Slab` or `fdtd.Region` |
| Overlap with another structure | a cell painted by another structure would be switched too |
| A switched `background`, or `continuation` into a face | an absorbing layer matched to one medium is not matched to the other |
| Reaching an absorbing layer | the layer damps the update the correction is written against |
| A source inside the region | its injected field is built for one fixed medium |
| A `Diffraction` or `FarField` reading inside the region | the result divides by one fixed medium's impedance |
| `discretisation='stencil'` | it replaces the coefficients the correction is written against |

Readings outside the region, such as `Record` and `Flux`, are unaffected.

## Validation

`tests/fdtd/test_fd_switched.py` compares the solver with the closed forms above.

| Check | Result |
|---|---|
| Forward/backward amplitudes, `eps 1 -> 4`, packet on a periodic line, 960/1920/3840 cells | errors 4.9e-5, 1.2e-5, 3.1e-6; order 2.00 |
| Standing-wave amplitude through the public API, spacing 0.02/0.01/0.005 µm | errors 2.0e-4, 5.2e-5, 1.3e-5; order 1.9-2.0 |
| Period after the jump at `eps 1 -> 4` | 2 within 1.5e-3 |
| `E`-continuous mutant | both amplitudes 2.000x the closed form |
| NumPy against JAX and PyTorch | 1.4e-14 of the peak on JAX; bit-identical on PyTorch |
| `d/d epsilon_after`, `d/d epsilon_before` on JAX and PyTorch | agree with central differences to 1e-6 relative |

## Limitations

- The interior-energy diagnostic weights the region with `epsilon_before` throughout. After
  the jump it is not the field energy there. A temporal boundary also really changes the
  energy, so an energy reading cannot check this feature.
- One jump per structure. A continuous modulation (a photonic time crystal) and a jump with
  a finite rise time are not provided.
- Meep and FDTDx have not been compared for this feature.
