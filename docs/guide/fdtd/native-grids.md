# Physical native Yee grids

Use `from wavebit.fdtd import native` when your physical model is already sampled
at Yee sites. `native.Problem` separates `Medium`, `Current`, `Initial`, Bloch
phases and observations from `native.Numerics`. It supports `check`, `prepare`
and `run`, and portable `Job(solver='fdtd.native', ...)`.

All spatial arrays are `(z,y,x)`; vectors have a leading Cartesian `(x,y,z)`
component axis. `Grid.spacing` is `(dz,dy,dx)` in `Units.length`. Optional
`z_spacing` contains every positive z cell width, including the periodic closing
cell. Native coordinates start at zero. Grid geometry is static, including under
AD. Each E/H component has its own Yee location.

| Input | Meaning |
| --- | --- |
| `Medium.epsilon`, `mu` | Positive relative epsilon-infinity and permeability; scalar or diagonal native coefficients |
| `Medium.conductivity` | Electric conductivity in S/m |
| `Oscillator(omega_0, gamma, strength)` | SI angular resonance/damping rates and strength in 1/s²; arrays may broadcast to native E |
| `Current.density`, `samples` | Current density in A/m² multiplied by dimensionless samples; sample n drives the E update ending at `(n+1)*dt` |
| `Initial.E`, `Initial.H` | E in V/m at t=0; H in A/m at t=-dt/2 |
| `bloch_phase` | Real radians in `(z,y,x)` order: F(r+L)=exp(i phase) F(r) |
| `time_step_seconds`, `steps` | Explicit SI time step and fixed step count |

`native.PML` occupies cells already included at both z ends; it does not expand
the supplied arrays. Its low/high `(epsilon,mu)` reference media must match each
end plus an interior cell. Active poles and conductivity must be absent there.
The z Bloch phase is zero with PML. `fdtd.Absorber` instead chooses the existing
matched graded conductivity layer, with its finite attenuation floor. Neither
absorber has a reflection-error guarantee; check width, strength and resolution.

`Outputs(record=Record(...))` retains only selected components and index bounds.
Each entry in `result.fields` is a `TimeSeries` with physical coordinates and SI
timestamps. H remains half a time step behind E. `result.E` and `result.H` are
final native arrays; polarization is in C/m², including the previous ADE state.

`Spectrum(wavelengths=..., planes=...)` uses integer z-node indices and streams
spatially collocated Fourier fields. `result.spectrum` has axes
`(wavelength,plane,y,x,component)`, E units V s/m and H units A s/m. The transform
is the raw integral with exp(+i 2 pi f t), without incident-field calibration.
`result.spectral_power_z` is consequently in W s², including physical transverse
cell area. It is not a monochromatic power or pulse-energy measurement.
`dft_subsample` reduces monitor work only, with no anti-alias filter.

`Tail(window_seconds=..., tolerance=...)` diagnoses the final interval. It does
not stop early. Field decay, ADE activity and Fourier-tail ratios remain
separate; the ADE diagnostic is a relative norm, not stored energy, and CPML
memory is not assessed. `Validity` distinguishes numerical admission from these
convergence diagnostics.

`check` shares the execution admission checks and allocates prospective arrays,
but advances no fields. Prepared execution fixes numerical settings, layout and
observations while receiving all live coefficients, sources, initial values and
Bloch phases on every call. Concrete invalid coefficients raise; traced invalid
coefficients produce invalid numerical results. NumPy, JAX and Torch use the
same engine. Current execution requires double precision. Set JAX x64 before
creating arrays. JAX jit and Torch eager AD are supported; Torch compilation is
not an established execution mode.

See the runnable [material derivative](../../../examples/fdtd/fdtd_grid_gradient.py),
[Bloch derivative](../../../examples/fdtd/fdtd_bloch_gradient.py),
[Lorentz derivative](../../../examples/fdtd/fdtd_dispersive_gradient.py) and
[three-backend comparison](../../../examples/fdtd/fdtd_backends.py).
The explicit objective scaling in those examples makes the SI Fourier-power
objective dimensionless; it does not change the solver's physical outputs.
