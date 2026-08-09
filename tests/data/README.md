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

Two of the generators (`gen_goldens_analytic.py`, `gen_goldens_factorization.py`) import modules
the rewrite deleted, so they no longer run; their `.npz` files remain valid, since the whole
point is that they outlive the code that made them.
