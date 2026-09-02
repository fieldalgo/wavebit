# Decisions

**Why a default is what it is, and what was measured to decide it.** These are not the manual;
they are the record behind it, and `wavebit/_core/` cites several of them from code comments —
a document here is why a constant has the value it has.

The equivalent elsewhere is NumPy's NEPs or JAX's JEPs: a series that sits beside the manual
rather than inside it, because the reader is someone questioning a choice rather than someone
learning to make one.

A document here states what is true now and what the measurement decided. It is not a
changelog: past defects appear only where the current behaviour is otherwise inexplicable — a
warning that exists because of one, a slot whose sign looks wrong until you know why.

| Group | What it covers |
| --- | --- |
| **Modeling and geometry** | How a structure becomes Fourier coefficients: the analytic and boundary paths, oblique cells, and the normal field a curved boundary needs |
| **Factorization and materials** | Which rule closes the products of discontinuous functions, and what an anisotropic or magnetic material costs |
| **The solve** | The formulation itself, the two layer solvers, how layers are joined, and what state a solve keeps |
| **Numerical conventions** | The frames, roots and singular points that are invisible until they are wrong |
| **Performance and tooling** | Which backend, whether to compile, and how a program outside this process learns what is supported |

The contract these documents follow — what one must contain, and what it may point at — is in
[`docs/README.md`](https://github.com/fieldalgo/wavebit/blob/develop/docs/README.md), which is
written for someone adding one.
