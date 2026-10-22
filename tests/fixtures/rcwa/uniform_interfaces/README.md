# Uniform-interface reference

| Entry | Question |
| --- | --- |
| `generate.py` | What do ordinary Fresnel slab amplitudes, pointwise fields and independent permittivity derivatives give at 120 decimal digits? |
| `values.json` | What are those fixed reference values at equality, neighboring binary64 values and separated permittivities? |

The generator imports no Wavebit code and uses ordinary admittance differences, not
the product's rationalized interfaces. It uses mpmath 1.3.0 (BSD-3-Clause; official
installed distribution `LICENSE`), only for manual reference generation. CI reads JSON
with the standard library; no new runtime or test dependency is required. Input angle
components retain their binary64 values before promotion to 120 decimal digits.

The layer permittivity varies independently of the incidence medium's fixed 9. Values
are R, T, then three field components at each listed height: `(Ey, Z0 Hx, Z0 Hz)` for s,
`(Ex, Ez, Z0 Hy)` for a positive magnetic-primary p wave. The public electric p basis
has the opposite sign for upward propagation; the test converts that basis explicitly.
Derivatives are with respect to the real layer permittivity, with all other inputs fixed.
The top-incidence test mirrors the same scene and converts vector parity.

Generate deliberately with `python tests/fixtures/rcwa/uniform_interfaces/generate.py`.
These are new analytical references, not replacements for existing goldens. Do not
regenerate acceptance values to accommodate a failing implementation.
