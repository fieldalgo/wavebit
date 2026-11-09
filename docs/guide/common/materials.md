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

Each constructor refuses malformed input with a `ValidationError` (a `ValueError`) whose `.code` and `.path` name the field, before any solver sees it. A pole rate, a Debye time or a conductivity that is not one finite real number is `material.scalar` (`material.poles.omega_p`, `material.conductivity`, ...). An `epsilon`, `epsilon_inf` or `mu` that is not a number, three diagonal values or a 3×3 tensor -- `None` for a permittivity, a string, a ragged list -- is `material.shape`; `mu=None` is the non-magnetic default. `poles` that are not a nonempty sequence of `Drude`, `Lorentz` and `Debye` models is `material.poles`; a single pole goes in a tuple, `poles=(Drude(...),)`. A table's `wavelengths` that are not one real row of at least two samples are `wavelength.shape`, and ones that are not finite, positive and strictly increasing `wavelength.value`, both at `material.wavelengths`; samples that are not one finite value per wavelength, or a zero `mu` beside an index table, are `material.shape` at `material.epsilon`, `material.index` or `material.mu`.

A single complex permittivity at one frequency does not by itself define a causal broadband time-domain model. FDTD checks this distinction and rejects unsupported responses; supply an explicit supported conductivity or dispersive model instead of guessing a time-domain continuation.

## Tables and catalog data

Material tables carry their own declared wavelength unit. The receiving problem independently declares `Units`; the public solver converts consistently when evaluating the table. Do not reinterpret a table in nanometres as micrometres by changing the geometry values alone. Tabulated frequency-domain data require a supported time-domain realization before use in FDTD; they are not automatically fitted.

The [generated material reference](../../reference/common/index.md) gives current constructors. The [earlier material-table manual](../../migration/materials-v1.md) documents the explicitly retained compatibility workflow and its historical tensor/NVF demonstrations.

A Lorentz pole with concrete, exactly zero strength contributes zero even at its undamped resonance, without forming a singular denominator or overflowing finite rates. A strength carrying a JAX or Torch derivative retains its response derivative at zero; an active or strength-traced undamped resonance remains singular. Invalid traced material parameters still produce NaN.

In FDTD, an active undamped Lorentz pole whose resonance lies exactly at a wavelength the plan checks -- a `DiffractionMonitor` wavelength or the sources' shortest band wavelength -- is refused by `fdtd.check`, `fdtd.prepare` and `fdtd.run` with `fdtd.invalid`, naming the pole and the wavelength. The permittivity is infinite there, so the material has no finite index and no mesh resolves it; a slightly detuned pole gets the ordinary coarse-mesh refusal, whose required spacing goes to zero as the detuning does. Move the wavelength off the resonance or give the pole damping. The time stepping itself never divides by the resonant denominator; a post-run column diagnostic read at a resonant source centre wavelength is reported as not judged instead of failing.
