# Shared physical materials

RCWA and FDTD consume the same material definitions. A material describes a physical constitutive response; each solver checks whether its geometry, tensor form and frequency/time-domain representation are supported.

```python
from wavebit import Material, ConductiveMaterial, DispersiveMaterial, Lorentz, Units

vacuum = Material(epsilon=1.)
dielectric = Material(epsilon=2.25)
conductor = ConductiveMaterial(epsilon=2.25, conductivity=1e4)  # S/m
resonant = DispersiveMaterial(
    epsilon_inf=1., poles=(Lorentz(omega_0=2e15, gamma=1e13, strength=(2e15)**2),))
response = resonant.at(.8, units=Units(length="um"))
```

`Material` supplies a constant relative permittivity and optional relative permeability. `MaterialResponse` is the evaluated response at the requested wavelengths. Scalar, diagonal and full 3×3 responses are physically distinct; acceptance by a material constructor does not promise every solver supports every tensor or loss model.

`ConductiveMaterial.conductivity` is electrical conductivity in S/m. Drude and Lorentz angular rates are in rad/s, Lorentz strength is in s⁻², and Debye relaxation time is in seconds. Changing geometry units does not rescale those physical material parameters. Passive electric loss has positive imaginary permittivity under the `exp(-i omega t)` convention.

A single complex permittivity at one frequency does not by itself define a causal broadband time-domain model. FDTD checks this distinction and rejects unsupported responses; supply an explicit supported conductivity or dispersive model instead of guessing a time-domain continuation.

## Tables and catalog data

Material tables carry their own declared wavelength unit. The receiving problem independently declares `Units`; the public solver converts consistently when evaluating the table. Do not reinterpret a table in nanometres as micrometres by changing the geometry values alone. Tabulated frequency-domain data require a supported time-domain realization before use in FDTD; they are not automatically fitted.

The [generated material reference](../../reference/common/index.md) gives current constructors. The [earlier material-table manual](../../migration/materials-v1.md) documents the explicitly retained compatibility workflow and its historical tensor/NVF demonstrations.

A Lorentz pole with concrete, exactly zero strength contributes zero even at its undamped resonance, without forming a singular denominator or overflowing finite rates. A strength carrying a JAX or Torch derivative retains its response derivative at zero; an active or strength-traced undamped resonance remains singular. Invalid traced material parameters still produce NaN.
