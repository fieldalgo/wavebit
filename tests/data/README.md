# Frozen test data

`golden_*.npz` and the generators that produce them, kept together because they change together.

These are the **previous implementation's own output**, captured before it was deleted. They pin
*continuity*, not correctness: a green run here means behaviour has not drifted, and says
nothing about whether the behaviour was right. The oracles that do speak to correctness are in
[`../references/`](../references) (an outside solver) and in the physics invariants asserted
throughout the suite.

Each `gen_goldens_*.py` writes the `.npz` beside it, but most of them will not run for you:
`gen_goldens.py`, `gen_goldens_field.py` and `gen_goldens_modeler.py` refuse outright, because
they captured implementations the rewrite deleted and would now re-freeze *today's* code under
a name that claims otherwise. `gen_goldens_nvf.py` is the exception and is meant to be run --
it re-freezes from this solver deliberately, and says so at the top.

**Never regenerate a golden to make a failing test pass.** That converts a detected regression
into an accepted one, silently, and the evidence that it happened is gone.

## The permittivity flip (2026-08-20), and why none of these were re-frozen

The solver took a refractive index and now takes a relative permittivity, so every case list
here was rewritten: `structure` / `top` / `bot`, with each number squared. The scene is the
same one — `eps = 9` and `n = 3` are the same slab — which is what lets the frozen numbers
stand. Measured rather than asserted:

* **`golden_nvf.npz`: all 39 arrays bit for bit.** Not a foregone conclusion, since NVF reads
  the raster a second time to build its interface normal; squaring turns out not to move a
  *direction*. Its generator records the argument and the measurement.
* **`golden_field.npz`: unchanged inside `atol=1e-12`**, which is what both its own diff and
  `tests/test_field_core.py` compare at. The layer flip is exact — the one case with vacuum on
  both sides reproduces bit for bit — and the 1.0e-14 to 2.4e-13 the other six move by is the
  half-space index becoming complex (`2+0j` where `2.0` sat), which reassociates arithmetic
  without changing an answer.
* **`golden_legacy.npz` and `golden_legacy_modeler.npz` are untouched**, and their generators'
  case lists were migrated anyway. Both lists have a twin in `tests/` marked "must stay
  identical", and that twin is live code; splitting the pair would cost more than the fiction
  that a refusing script is a transcript. The modeler golden's eleven `*__cell` arrays still
  hold refractive index and are squared at the comparison, which is the trade below.

A golden that cannot be regenerated is on a clock, though, and `golden_legacy_factorization.npz`
ran out on 2026-08-18. It was retired rather than kept: 39 of its 68 arrays were already read by
nothing, the rewrite it was evidence for had long since landed, and nothing has ever been
released that could depend on matching the pre-rewrite numbers. Every check it carried is now
definitional -- closed-form Fourier coefficients, and losslessness/minimality for the compression
step -- in `tests/test_factorization_core.py`, which needs no reference data at all. Prefer that
trade wherever a golden's only claim is "unchanged".
