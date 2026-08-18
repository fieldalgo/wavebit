# Frozen test data

`golden_*.npz` and the generators that produce them, kept together because they change together.

These are the **previous implementation's own output**, captured before it was deleted. They pin
*continuity*, not correctness: a green run here means behaviour has not drifted, and says
nothing about whether the behaviour was right. The oracles that do speak to correctness are in
[`../references/`](../references) (an outside solver) and in the physics invariants asserted
throughout the suite.

Each `gen_goldens_*.py` writes the `.npz` beside it. Run one only to add cases, or after
deliberately changing what is compared — and say which in the commit message.

**Never regenerate a golden to make a failing test pass.** That converts a detected regression
into an accepted one, silently, and the evidence that it happened is gone.

One generator (`gen_goldens_analytic.py`) imports modules the rewrite deleted, so it no longer
runs; its `.npz` remains valid, since the whole point is that a golden outlives the code that
made it.

A golden that cannot be regenerated is on a clock, though, and `golden_legacy_factorization.npz`
ran out on 2026-08-18. It was retired rather than kept: 39 of its 68 arrays were already read by
nothing, the rewrite it was evidence for had long since landed, and nothing has ever been
released that could depend on matching the pre-rewrite numbers. Every check it carried is now
definitional -- closed-form Fourier coefficients, and losslessness/minimality for the compression
step -- in `tests/test_factorization_core.py`, which needs no reference data at all. Prefer that
trade wherever a golden's only claim is "unchanged".
