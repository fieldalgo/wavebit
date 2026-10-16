# Public numerical-selector refusal messages

The public lowering path rejects unsupported `Numerics.junction` and
`Numerics.discretisation` values. Its early refusals must still name the public
constructor and selector, as the existing public-spelling regression requires.
The fix restores that context without changing either acceptance predicate or
any field calculation. Structured `invalid_input` code, selector path, actual
value, absent expected value, and error severity are retained.

## Evidence

On macOS ARM64, Python 3.12.9, NumPy 2.5.1, JAX 0.11.0, Torch 2.14.1:

- The existing spelling test failed for both selectors before the fix.
- Eight additional cases cover `check` and `prepare`, each selector, and an
  unsupported string or `None`. All eight failed their message assertion before
  the fix; all passed afterward, including structured-field assertions.
- The complete public-spelling, public-discretisation and policy-report files
  passed 109 cases after the fix.
- Removing the public context from each actual product refusal produced four
  assertion failures per mutation. Together the two mutations cover all eight
  added cases. Every tracked file was restored byte for byte after each mutation;
  a fresh process then passed all eight cases again.

These are error-reporting checks. They do not establish additional numerical
accuracy or cross-platform execution coverage. No golden data, numerical
threshold, acceptance condition or skip policy was changed.
