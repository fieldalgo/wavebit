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
    "under `one_dimensional = one_row and self.lattice_is_axis_aligned`, the shape path "
    "setting grating_type 2 unconditionally.")

_ALLOWED = {
    # One entry, and three until the body was folded into `_core`. The argument below is
    # entirely about the *callers*, so it did not weaken by moving -- but the gate is keyed by
    # (module, function) and could not know that, which is what makes it a gate. It failed on
    # the move in both directions at once: the new location was unlisted, and the three old
    # keys had stopped existing, which `test_the_allow_list_has_not_quietly_grown` caught.
    ('wavebit/_core/engine.py', '_separable_kxy'): (2, _SEPARABLE_KXY),
    ('wavebit/_core/draw.py', 'slice_structure'): (
        2,
        "one sample grid, behind a raise rather than behind a declaration: the `mt == 2` pair "
        "sits after an explicit `if not solver.lattice_is_axis_aligned: raise "
        "NotImplementedError`. The add-path branch beside it already goes through "
        "`sample_grid` with the lattice. It was four until the block path was removed -- its "
        "own pair was guarded by `draw._refuse_layer`, and both went with it."),
    ('wavebit/_core/conforming/__init__.py', '_layers'): (
        5,
        "the research conforming cell is a rectangle by construction: `periods` are the reduced "
        "basis's axis lengths behind `_require(valid, 'conforming NVF requires a right-handed "
        "lattice with an orthogonal reduced basis')`, and every rectangle is further held to a "
        "reduced axis by `_require(aligned, ...)`. A skewed cell never reaches the arithmetic; "
        "it is invalidated at the door, the way `slice_structure` raises."),
}


def _is_period_pair(node, alias=frozenset({'period', 'periods'})):
    """``Px`` / ``Py`` by name, or a ``period``-derived subscript however the name was reached.

    The subscript half deliberately does not care what the sequence hangs off. ``period[0]``,
    ``self.period[0]`` and ``solver.period[0]`` are the same read, and the attribute spelling is
    the one the three backends actually use -- an earlier version of this matched only
    ``ast.Name``, so ``wavelength / self.period[0]`` in every ``_separable_kxy`` was invisible to
    a gate whose whole claim is that it looks everywhere.

    ``alias`` is the set of local names the enclosing function bound to a period sequence, which
    ``_Scan.visit_Assign`` grows: ``p = self.period`` then ``p[0] * x`` is the same read as
    ``self.period[0] * x``. No such spelling exists in the package today, so that half is
    prevention and costs nothing -- it flags nothing new.

    Descending into **call arguments** was measured and rejected. It flags 30-odd lines and
    nearly all of them are this package's correct idiom: ``shape_geometry(shape, Px, Py,
    lattice)`` hands the pair *and* the basis to a callee that takes both. Narrowing it to calls
    carrying no basis still leaves the ``float(period[0])`` conversions and the rectangular
    branch of ``fields._inv_fourier``, which the skewed branch returns before reaching. The
    audit's motivating example, ``tol_rel * max(Px, Py)`` in ``modeler.py``, is a tolerance
    scale and not geometry at all. An allow-list long enough to hold that is a gate that no
    longer means anything -- and the omission it was meant to catch is caught directly by
    ``test_a_function_holding_a_lattice_forwards_it_to_every_sibling_that_takes_one`` below.
    """
    if isinstance(node, ast.Name) and node.id in _PERIOD_NAMES:
        return True
    if not isinstance(node, ast.Subscript):
        return False
    base = node.value
    if isinstance(base, ast.Attribute):
        return base.attr in ('period', 'periods')
    return isinstance(base, ast.Name) and base.id in alias


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
        self.alias = {'period', 'periods'}

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

    def visit_Assign(self, node):
        v = node.value
        if ((isinstance(v, ast.Name) and v.id in self.alias)
                or (isinstance(v, ast.Attribute) and v.attr in ('period', 'periods'))):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    self.alias.add(t.id)
        self.generic_visit(node)

    def visit_BinOp(self, node):
        if self.guarded == 0 and (_is_period_pair(node.left, self.alias)
                                  or _is_period_pair(node.right, self.alias)):
            self.lines.append(node.lineno)
        self.generic_visit(node)


def _offenders():
    files = [f for f in subprocess.check_output(['git', 'ls-files', 'wavebit'], text=True).split()
             if f.endswith('.py')]
    found = {}
    for f in files:
        source = pathlib.Path(f).read_text(encoding='utf-8')
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
          '`parametric._round_field` and `shape_geometry` do. If the path genuinely '
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


# ------------------------------------------- the omission no arithmetic predicate can see

#: ``(module, function, callee)`` that may hold a ``lattice`` and call a lattice-taking sibling
#: without it. Empty, and an entry here would need the same kind of argument as ``_ALLOWED``.
_MAY_DROP_THE_LATTICE = set()


def _lattice_takers(trees):
    """``name -> smallest positional index of ``lattice``, or ``None`` if keyword-only.

    Keyed by name because a call site is resolved by name here, and two modules may define the
    same one with the parameter in different places -- ``normal_projector_coeffs`` does
    (``boundary.py`` at 6, ``analytic.py`` at 7). The *smallest* index is the conservative
    reading: a call passing enough positional arguments for any of them is forwarding.
    """
    takers = {}
    for tree in trees.values():
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            names = [a.arg for a in node.args.args]
            if 'lattice' in names:
                i = names.index('lattice')
                prev = takers.get(node.name)
                takers[node.name] = i if prev is None else min(prev, i)
            elif any(a.arg == 'lattice' for a in node.args.kwonlyargs):
                takers.setdefault(node.name, None)
    return takers


def _binds_lattice(fn):
    if any(a.arg == 'lattice' for a in list(fn.args.args) + list(fn.args.kwonlyargs)):
        return True
    return any(isinstance(t, ast.Name) and t.id == 'lattice'
               for n in ast.walk(fn) if isinstance(n, ast.Assign) for t in n.targets)


def test_a_function_holding_a_lattice_forwards_it_to_every_sibling_that_takes_one():
    """The other half of the hole, and the half no arithmetic predicate reaches.

    ``_ALLOWED`` above is about a period *pair* being used as a rectangle. This is about a
    ``lattice`` that was never passed at all -- there is no arithmetic to flag, only a call with
    one argument missing, and the callee then defaults it to ``None`` and computes the rectangle.

    That is exactly ``boundary.py:471`` (fixed 2026-08-24): ``to_conv_mat_boundary_layer``
    forwarded ``lattice`` to ``shape_vertices``, ``plan_layer`` and ``conv_mats``, and not to
    ``check_plan_still_fits`` -- whose signature had the parameter all along. A frozen painted
    layer on a hex cell whose shape omitted its centre then raised ``StaleGeometryError`` where
    the eager pass answered. Worse, ``_guards_the_rectangle`` credits ``lattice is None`` as
    *proof* the cell is a rectangle, so a manufactured ``None`` launders itself downstream into
    a guard this file already trusts.

    The rule is narrow on purpose: only a function that *has* a lattice in scope, only a callee
    whose own signature takes one. Neither half is a judgement call, so a hit is a fact.
    """
    files = [f for f in subprocess.check_output(['git', 'ls-files', 'wavebit'],
                                                text=True).split() if f.endswith('.py')]
    trees = {f: ast.parse(pathlib.Path(f).read_text(encoding='utf-8')) for f in files}
    takers = _lattice_takers(trees)
    assert len(takers) > 20, takers            # a predicate matching nothing would pass silently

    dropped = []
    for f, tree in trees.items():
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not _binds_lattice(fn):
                continue
            for call in ast.walk(fn):
                if not isinstance(call, ast.Call):
                    continue
                name = (call.func.attr if isinstance(call.func, ast.Attribute)
                        else getattr(call.func, 'id', None))
                if name not in takers or name == fn.name:
                    continue
                if any(k.arg in ('lattice', None) for k in call.keywords):
                    continue                   # named, or forwarded inside a **kwargs
                idx = takers[name]
                if idx is not None and len(call.args) > idx:
                    continue                   # passed positionally
                if (f, fn.name, name) in _MAY_DROP_THE_LATTICE:
                    continue
                dropped.append(f'  {f}:{call.lineno} in {fn.name}() -> {name}()')

    assert not dropped, (
        'these hold a `lattice` and call a sibling that takes one without passing it, so the '
        'callee defaults to `None` and answers for a rectangle:\n' + '\n'.join(dropped)
        + '\n\nForward it. If the callee genuinely must see the rectangle, pass `lattice=None` '
          'explicitly -- a stated `None` is a decision and an omitted one is a silence.')
