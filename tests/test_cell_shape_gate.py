"""
A gate, not a measurement: no oblique-capable path may do geometry with ``(Px, Py)``.

``period`` reaches an internal function as ``periods_of(A) = (|a1|, |a2|)`` -- edge lengths,
with the angle already discarded. A 60-degree hex and a 400x400 square both arrive as
``(400, 400)``, so a function handed only that cannot tell them apart and anything it folds,
tiles, wraps or centres is a rectangle's.

Two live wrong answers came through exactly this hole in one day (2026-08-23):

* ``_shape_surface_normal`` folded to the nearest periodic image with ``d % Px``, reducing
  towards a point where no image is. 1.146e-03 from the raster path on a hex pillar array,
  5.271e-06 after (``b480650``).
* ``shape_geometry`` defaulted an omitted centre to ``(Px/2, Py/2)`` -- the centre of
  ``diag(|a1|, |a2|)``, which on a shear is not inside the cell at all. It drew an empty cell
  where the solve built a rod (``2d7ca08``).

Both were silent: a rectangular cell makes the two spellings *the same arithmetic*, energy
conserved to 1e-12 throughout, and 3316 tests passed over them. Nothing a numerical test can
reach will catch the third one either, so this reads the code instead.

The rule is not "never touch ``period``". A scalar length is fine -- a 1-D series has no cell
shape to get wrong. What this refuses is a *pair* used as though it described a rectangle,
anywhere the cell might not be one.
"""
import ast
import pathlib
import subprocess

_PERIOD_NAMES = {'Px', 'Py'}

#: ``(module, function) -> (expected number of flagged lines, why it is allowed)``. Each entry
#: is a promise that the function cannot see a skewed cell. Adding one means arguing that case,
#: which is the point: the list is the argument, not the exemption.
#: The wavevector build named for the case it serves: `_separable_kxy` is the *rectangular*
#: one, and the two routes that reach it both prove the cell is a rectangle in the caller,
#: which is why an in-function read cannot see the guard.
_SEPARABLE_KXY = (
    "the separable wavevector build, rectangular by name and by design. `engine.order_kxy` "
    "returns it only inside `if A is None or is_axis_aligned(A)` and takes the lattice branch "
    "without calling it at all -- counted rather than assumed, by `test_oblique_lattice.py::"
    "test_the_skewed_branch_never_calls_the_separable_build`. `_solve_1d` and "
    "`_solve_1d_conical` call it directly, and `_assign_grating_type` routes to either only "
    "under `one_dimensional = one_row and self.lattice_is_axis_aligned`, the block and shape "
    "paths setting grating_type 2 unconditionally.")

_ALLOWED = {
    ('wavebit/on_numpy/emsolver/_base.py', '_separable_kxy'): (2, _SEPARABLE_KXY),
    ('wavebit/on_jax/emsolver/_base.py', '_separable_kxy'): (2, _SEPARABLE_KXY),
    ('wavebit/on_torch/emsolver/_base.py', '_separable_kxy'): (2, _SEPARABLE_KXY),
    ('wavebit/_core/draw.py', 'slice_structure'): (
        4,
        "two sample grids, one per branch. The `mt == 1` pair is the block modeller, refused "
        "on a skewed cell by manifest.py's `paths_refused_on_oblique`; the `mt == 2` pair sits "
        "after an explicit `if not solver.lattice_is_axis_aligned: raise NotImplementedError`. "
        "The add-path branch beside them already goes through `sample_grid` with the lattice."),
}


def _is_period_pair(node):
    """``Px`` / ``Py`` by name, or a ``period`` / ``periods`` subscript however it is reached.

    The subscript half deliberately does not care what the sequence hangs off. ``period[0]``,
    ``self.period[0]`` and ``solver.period[0]`` are the same read, and the attribute spelling is
    the one the three backends actually use -- an earlier version of this matched only
    ``ast.Name``, so ``wavelength / self.period[0]`` in every ``_separable_kxy`` was invisible to
    a gate whose whole claim is that it looks everywhere.
    """
    if isinstance(node, ast.Name) and node.id in _PERIOD_NAMES:
        return True
    if not isinstance(node, ast.Subscript):
        return False
    base = node.value
    if isinstance(base, ast.Attribute):
        return base.attr in ('period', 'periods')
    return isinstance(base, ast.Name) and base.id in ('period', 'periods')


def _guards_the_rectangle(test):
    """Whether this ``if`` test establishes that the cell *is* axis-aligned.

    ``lattice is None`` counts because the only way a geometry path gets ``None`` is a caller
    that had no basis to give -- which today means an axis-aligned cell whose period is the
    variable under differentiation (``engine._geometry_lattice_for``).
    """
    if isinstance(test, ast.BoolOp):
        return any(_guards_the_rectangle(v) for v in test.values)
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return _guards_the_rectangle(test.operand)
    if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
            and test.left.id in ('lattice', 'basis', 'A')):
        return any(isinstance(o, (ast.Is, ast.IsNot)) for o in test.ops)
    if isinstance(test, ast.Call):
        f = test.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, 'id', '')
        return name in ('is_axis_aligned', 'lattice_is_axis_aligned')
    return False


class _Scan(ast.NodeVisitor):
    """Arithmetic on a period pair, outside any branch that pinned the cell to a rectangle."""

    def __init__(self):
        self.guarded = 0
        self.lines = []
        self._top = True

    def visit_FunctionDef(self, node):
        if self._top:                      # scan one function at a time; ast.walk finds the rest
            self._top = False
            for n in node.body:
                self.visit(n)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_If(self, node):
        if _guards_the_rectangle(node.test):
            self.guarded += 1
            for n in node.body:
                self.visit(n)
            self.guarded -= 1
            for n in node.orelse:          # the else is the lattice branch -- still watched
                self.visit(n)
        else:
            self.generic_visit(node)

    def visit_BinOp(self, node):
        if self.guarded == 0 and (_is_period_pair(node.left) or _is_period_pair(node.right)):
            self.lines.append(node.lineno)
        self.generic_visit(node)


def _offenders():
    files = [f for f in subprocess.check_output(['git', 'ls-files', 'wavebit'], text=True).split()
             if f.endswith('.py')]
    found = {}
    for f in files:
        source = pathlib.Path(f).read_text()
        lines = source.splitlines()
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.FunctionDef):
                continue
            scan = _Scan()
            scan.visit(node)
            hit = sorted(set(scan.lines))
            if hit:
                found[(f, node.name)] = [(n, lines[n - 1].strip()) for n in hit]
    return found


def test_no_oblique_capable_path_does_rectangle_geometry():
    """Every ``(Px, Py)`` arithmetic must sit behind a proof that the cell is a rectangle."""
    found = _offenders()
    unlisted = {k: v for k, v in found.items() if k not in _ALLOWED}
    assert not unlisted, (
        'these do arithmetic on a period pair with nothing establishing the cell is '
        'axis-aligned:\n' + '\n'.join(
            f'  {f}:{n} in {fn}()\n      {src}' for (f, fn), v in unlisted.items()
            for n, src in v)
        + '\n\n`period` is (|a1|, |a2|) on a skewed cell -- the angle is already gone, so this '
          'computes a rectangle\'s answer. Take the 2x2 basis instead and branch on '
          '`lattice.is_axis_aligned(A)` if a cheap rectangular path is wanted, as '
          '`analytic._shape_surface_normal` and `shape_geometry` do. If the path genuinely '
          'cannot meet a skewed cell, add it to `_ALLOWED` with the guard that says so.')


def test_the_allow_list_has_not_quietly_grown():
    """An exempt function may not accumulate new period arithmetic under its exemption.

    The entries promise a *specific* pair of sample grids, not a licence. Without the count a
    third one could be added inside `slice_structure` and inherit an argument made about two
    others.
    """
    found = _offenders()
    for key, (expected, why) in _ALLOWED.items():
        assert key in found, f'{key} no longer does period arithmetic -- drop it from _ALLOWED'
        got = len(found[key])
        assert got == expected, (
            f'{key[0]} in {key[1]}(): expected {expected} flagged line(s), found {got}.\n'
            f'  the exemption reads: {why}\n'
            '  if the new one is covered by that same guard, raise the count; if it is not, '
            'it needs the basis.')
