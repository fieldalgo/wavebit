# Public whole-cell shape policy reports

The public `fdtd.Slab` and `fdtd.Region` classes retain the whole-cell filling
semantics of the internal geometry classes. Public lowering preserves their
wrapper types. Comparing only the exact internal classes therefore mislabelled
these built-ins as `custom_shape_protocol`, although their fill operator was
unchanged. The existing mixed Slab/Region/Sphere report regression exposed this
as a `whole_cell_fill` versus `custom_shape_protocol` assertion failure.

The classifier now recognises both exact internal and exact public Slab/Region
classes. It deliberately does not classify every subclass as built-in: a user
subclass owns its shape protocol and may override the fill implementation.
Stencil operator and junction-plan classification are unchanged.

## Regression and mutation checks

`tests/fdtd/test_policy_report.py` checks public Slab and Region and user
subclasses of each through four entry points: `check`, the preparation report,
`Prepared.validate`, and `Prepared.run` result provenance. Each of the 16 cases
asserts the complete structure-policy row, including `rule=None` and the reason.

On macOS ARM64 with Python 3.12.9, NumPy 2.5.1, JAX 0.11.0 and Torch 2.14.1:

- Before the fix, all eight public-class cases failed their reason assertion;
  the eight custom-subclass controls passed.
- After the fix, all 16 passed. The complete policy-report file and the related
  prepared-policy admission and public-discretisation files gave 73 passes.
- Removing the two public exact types from the product classifier produced eight
  assertion failures, covering every public-class case.
- Replacing exact-type membership with an `isinstance` check produced eight
  assertion failures, covering every custom-subclass case.
- The product file and all other tracked file bytes were restored after each
  mutation; a fresh process then passed all 16 cases again.

These checks establish report/provenance classification, not a new field-solver
accuracy claim. The additional lifecycle cases use the NumPy backend; the
related suite exercises its existing backend coverage. Cross-platform CI is a
separate check. No golden data or numerical tolerances were changed.
