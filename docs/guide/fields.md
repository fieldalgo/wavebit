# Near fields

Two steps, and the first one is not optional: **declare** that this solve is for a field, then
ask for one.

```python
solver = wavebit.rcwa(backend=0, answers='field', ...)
solver.solve()
cell = solver.reconstruct_field(sample_x=64, sample_y=64, sample_z=32)
```

## Declaring what a solve is for

`answers` says which questions a solve keeps the state to answer. It is a **declaration**, in
the sense `tetm` is one: not a request to be economical, but a statement that a near field is
not something this problem asks about — and the state that would answer one is then never
built, rather than built and thrown away.

| `answers=` | What you get back |
| --- | --- |
| `'efficiency'` | `R` / `T` per order, and nothing else |
| `'amplitude'` | **the default** — the Fresnel amplitudes, their s/p split, and the pure-TE / pure-TM variants |
| `'field'` | all of the above, plus the per-layer state `reconstruct_field` walks |

Each rung answers everything below it, so `'field'` is a superset rather than a mode.

The default sits where the cost cliff is. The amplitudes are boundary vectors already solved
for; the field state is 4.25 matrices of side `2*n_orders_u*n_orders_v` **per layer** on the 2D
path — 21.7 MB at `truncation_order=(8, 8)`, 240 MB at `(15, 15)`, 733 MB at `(20, 20)`, and
again for every layer. The two differ by three orders of magnitude, and the runs big enough to
care are optimization sweeps against a fixed objective, which never reconstruct a field at all.

**Asking for a field without declaring one is refused rather than served**, and the refusal
names the keyword to change. It has to be a refusal: the state is kept *while the solve runs*
or not at all, so there is nothing an argument to `reconstruct_field` could recover afterwards.

`answers` must also be static under `jit`, which is why it is a declaration and not a plain
attribute — see [`../what-a-solve-answers.md`](../what-a-solve-answers.md).

## Sampling where you want it

`reconstruct_field` takes a **count** or the **coordinates themselves**, per axis:

```python
cell = solver.reconstruct_field(sample_x=64, sample_y=64, sample_z=32)     # a uniform grid

cell = solver.reconstruct_field(                                  # or name the points
    sample_x=np.array([0.0, 133.3, 251.7]),                        # a line cut, 3 columns
    sample_y=1,
    sample_z=np.linspace(0, 410, 200),                             # z through the whole stack
)
```

Nothing is interpolated. The transverse kernel is a pointwise `exp(i k.r)` and each z-slice is
an independent propagation, so a named point costs exactly what a grid point costs and returns
the identical value — refining a grid leaves the shared points bit-for-bit unchanged, which is
what makes that a fact rather than a hope. Ask for three columns and you pay for three, not for
a grid you then throw away.

| Axis | Frame | Rules |
| --- | --- | --- |
| `sample_x`, `sample_y` | lengths in `period`'s units, from the **centre of the cell** | unconstrained: unsorted, non-uniform, and outside the cell all work, the last being the periodic continuation with its Bloch phase. A count spans the cell |
| `sample_z` | **absolute, from the top of the stack** | must lie inside the stack; an interface belongs to the layer **below** it, so `z - 1e-9` gets the upper layer's limit |

That interface rule is visible only in `Ez`, the one component genuinely two-valued there.

With an explicit `sample_z` the returned z axis is exactly your request, in your order, rather
than `sample_z * n_layers` with every interface appearing twice.

## What comes back

An array, whose shape depends on which formulation solved:

| Path | Shape | Components |
| --- | --- | --- |
| coupled (conical, 2D) | `(n_incidences, nz, ny, nx, 6)` | `(Ex, Ey, Ez, Hx, Hy, Hz)` |
| pure 1D TE/TM (`tetm=`) | `(nz, ny, nx, 3)` | TE is `(Ey, Hx, Hz)`, TM is `(Hy, Ex, Ez)` |

`nz` is `sample_z * n_layers` for a count and `len(sample_z)` for explicit coordinates.

The pure 1D path has **no leading incidence axis**: `tetm` declared the other degree of freedom
away, so that solve carries one incidence and the cell has no axis to put a second on.

`incidences` names which of them to reconstruct, one leading entry each, **in the order asked
for**:

```python
cell = solver.reconstruct_field(sample_x=8, sample_y=8, sample_z=4,
                                incidences=('given', 's', 'p'))       # -> (3, nz, 8, 8, 6)
```

`'given'` is the incidence the solve was posed with; `'s'` and `'p'` are the pure variants the
coupled paths produce for free — the same three the result object calls `given` / `s_inc` /
`p_inc`. On the pure 1D path only `('given',)` is answerable, and naming another is refused
rather than ignored, since a reconstruction of the wrong incidence is a complete and plausible
field cell.

**Axes follow the array**, here as everywhere: row `0` is `y = 0`, and `+z` points down into
the stack — so a wave travelling into the stack has `H = k x E` and `Sz > 0`. The flip that
renders a field the conventional way up belongs to plotting (`solver.plot_field`), not to the
data. On a skewed cell the output grid is indexed by `(v, u)` rather than `(y, x)`.

## The one thing that cannot be traced

An **explicit** `sample_z` cannot be differentiated through with respect to `thickness`.
Bucketing a `z` into a layer needs the thicknesses as numbers, and it raises rather than
differentiating the wrong geometry. Counts stay fully traceable.
