"""``tetm`` on a solve that cannot take the pure 1D path is refused by name, at every door.

``tetm`` declares the polarization away, so the pure 1D TE/TM kernel is the only solve that can
honour it; every other one reads the incident ``(a_s, a_p)`` pair. Five routes reached one of
those anyway. Declared at construction, ``tetm`` leaves that pair empty, and the solve ended in
``_incidence_pair`` as ``TypeError: array_namespace requires at least one non-scalar array
input`` -- naming neither input. Assigned afterwards, it left the constructor's default
``psi = pi/2`` behind, and the solve *answered*, for s light, whichever polarization had been
declared. The routes:

* a **z row** handed to ``solve_conv`` -- any of ``epxz / epyz / epzx / epzy_conv_all``, or the
  permeability's ``mu[5:]``. The 1D operators have no z-row slots, so a z row sends even a
  one-row raster to the first-order path, which is written on the 2D layout;
* ``formulation='first_order'``, which forces that same path without a z row;
* a solver holding **no structure**, whose ``solve_conv`` keeps the layout its last solve took
  -- 2D if it never solved, conical after a conical solve; a kept pure layout is asked again
  what its raster was (``phi``, ``ky``, the cell's skew, ``truncation_order[1]``);
* a **shape** structure, which is always solved on the 2D layout;
* a **tensor** material, refused by name where its convolution matrices are built and reachable
  past that only with matrices built before ``tetm`` was set.

A z row cannot come in through ``solve()``: only a tensor material builds one, and that is the
fifth route. So the z-row doors are ``solve_conv``'s own arguments, positional, keyword and
``mu``, eager and traced. The other four come in through every door a condition has: the
constructor, an attribute assigned afterwards, ``solve(**kwargs)``, the typed ``Stack`` /
``Incidence``, a ``JobSpec``, and a ``jax.jit`` / ``jax.grad`` trace.

Every message names both inputs and says what to write instead. The advice is checked, not
just printed: ``polarization=(1, 0)`` / ``(0, 1)`` at ``phi=0`` solves on the path the
refusal names, and where both formulations can express the scene it reproduces the ``tetm``
answer.
"""
import re

import numpy as np
import pytest

import wavebit
from wavebit import Incidence, JobSpec, Layer, Numerics, Stack, run

# One layer, one row: the structure the pure 1D path is built for. Permittivity.
ROW = [1., 1., 1., 12.25, 12.25, 12.25, 1., 1., 1., 1.]
RASTER_1D = np.array([[ROW]])
# The same kind of grating drawn as a shape, which the shape path solves in 2D whatever it is.
SHAPES = [dict(background=1.0, shapes=[dict(type='rectangle', cx=0.0, cy=0.0, lx=210.0,
                                             ly=300.0, material=12.25)])]
KW = dict(top=1.0, bot=2.25, theta=0.2, truncation_order=[3, 0], wavelength=900.0,
          period=[700.0, 700.0], thickness=[300.0])
PAIR = {'te': '(1, 0)', 'tm': '(0, 1)'}
AMPLITUDE = {'te': ('r_s', 't_s'), 'tm': ('r_p', 't_p')}


def _advice(tetm):
    """The part of every message that says what to write instead, as a regex."""
    return rf"polarization={PAIR[tetm].replace('(', '[(]').replace(')', '[)]')} at phi=0"


def _z_slots(conv, **given):
    """``solve_conv``'s z-row arguments, ``0.1 * I`` in the named slots and ``None`` elsewhere."""
    n = conv.epx.shape[-1]
    value = 0.1 * np.eye(n, dtype=complex)[None]
    return {name: (value if name in given else None)
            for name in ('epxz_conv_all', 'epyz_conv_all', 'epzx_conv_all', 'epzy_conv_all')}


def _assert_same_light(got, want, tetm, tol=1e-12):
    """Every number the pure path reports, against another path's -- premise asserted.

    ``1e-12``: measured 7.4e-15 (TE) and 1.3e-14 (TM) between the pure 1D kernel and the pair
    on the first-order path, which share no kernel.
    """
    for name in ('R', 'T') + AMPLITUDE[tetm]:
        a, b = np.asarray(getattr(got.incident, name)), np.asarray(getattr(want.incident, name))
        assert np.abs(b).max() > 1e-2, f'{name} is empty, so this compares nothing'
        assert a.shape == b.shape and np.abs(a - b).max() <= tol, (name, np.abs(a - b).max())


# ----------------------------------------------------------------------------- the z row

@pytest.mark.parametrize('tetm', ['te', 'tm'])
@pytest.mark.parametrize('slot', ['epxz_conv_all', 'epyz_conv_all', 'epzx_conv_all',
                                  'epzy_conv_all'])
def test_a_z_row_beside_tetm_is_refused_by_name(slot, tetm):
    """Each of the four permittivity slots, as a keyword -- the message names the one given."""
    solver = wavebit.rcwa(structure=RASTER_1D, tetm=tetm, **KW)
    conv = solver.eps_conv_matrices()
    with pytest.raises(ValueError, match=rf"tetm='{tetm}' and a z row \({slot}\)") as info:
        solver.solve_conv(conv.epx, conv.epy, conv.epz_i, **_z_slots(conv, **{slot: True}))
    assert 'polarization' in str(info.value) and 'first-order' in str(info.value)


def test_a_z_row_given_positionally_and_in_mu_is_named_as_given():
    """The positional spelling, and the permeability's z row (``mu[5:]``) -- both are a z row."""
    solver = wavebit.rcwa(structure=RASTER_1D, tetm='te', **KW)
    conv = solver.eps_conv_matrices()
    z = _z_slots(conv, epxz_conv_all=True, epzy_conv_all=True)
    with pytest.raises(ValueError, match=r'a z row \(epxz_conv_all, epzy_conv_all\)'):
        solver.solve_conv(conv.epx, conv.epy, conv.epz_i, None, None, *z.values())
    eye = np.eye(conv.epx.shape[-1], dtype=complex)[None]
    mu = (eye, eye, eye, None, None, None, 0.1 * eye, None, None)
    with pytest.raises(ValueError, match=r"tetm='te' and a z row \(muyz\)"):
        solver.solve_conv(conv.epx, conv.epy, conv.epz_i, mu=mu)


@pytest.mark.parametrize('tetm', ['te', 'tm'])
def test_the_advice_beside_a_z_row_is_the_same_light(tetm):
    """Follow the message: the pair it names solves on the first-order path
    and *is* what ``tetm`` described.

    An all-zero z row is still a z row -- the slots are read for given-ness, not value -- so
    it is refused beside ``tetm`` and sends the pair to the first-order path. There it has to
    reproduce the pure 1D answer, amplitude for amplitude: that is the claim "the light tetm
    describes" makes, and the two solves share no kernel.
    """
    pure = wavebit.rcwa(structure=RASTER_1D, tetm=tetm, **KW)
    want = pure.solve()
    conv = pure.eps_conv_matrices()
    zero = np.zeros_like(conv.epx)
    with pytest.raises(ValueError, match=_advice(tetm)):
        pure.solve_conv(conv.epx, conv.epy, conv.epz_i, None, None, zero, zero, zero, zero)

    pair = (1, 0) if tetm == 'te' else (0, 1)
    advised = wavebit.rcwa(structure=RASTER_1D, polarization=pair, phi=0.0, **KW)
    got = advised.solve_conv(conv.epx, conv.epy, conv.epz_i, None, None, zero, zero, zero, zero)
    assert (advised.grating_type, advised.formulation_assigned) == (2, 'first_order')
    _assert_same_light(got, want, tetm)


def test_the_z_row_refusal_holds_under_jit_and_grad():
    """A traced z row is still a *given* one: the test reads ``is None``, never a value."""
    jax = pytest.importorskip('jax')
    solver = wavebit.rcwa(backend=1, structure=RASTER_1D, tetm='te', **KW)
    import jax.numpy as jnp
    conv = solver.eps_conv_matrices()
    z = jnp.asarray(0.1 * np.eye(conv.epx.shape[-1], dtype=complex)[None])

    def transmitted(scale):
        res = solver.solve_conv(conv.epx, conv.epy, conv.epz_i, epxz_conv_all=scale * z)
        return jnp.sum(res.incident.T)

    for traced in (jax.jit(transmitted), jax.grad(transmitted)):
        with pytest.raises(ValueError, match=r"tetm='te' and a z row \(epxz_conv_all\)"):
            traced(1.0)


# ---------------------------------------------------------------- formulation='first_order'

@pytest.mark.parametrize('door', ['attribute', 'attribute, tetm last', 'solve keyword',
                                  'solve keyword, tetm'])
def test_formulation_first_order_beside_tetm_is_refused_on_every_door(door):
    """The first-order path without a z row: forced, in either order and through ``solve``."""
    if door == 'attribute':
        solver = wavebit.rcwa(structure=RASTER_1D, tetm='tm', **KW)
        solver.formulation = 'first_order'
        call = solver.solve
    elif door == 'attribute, tetm last':
        solver = wavebit.rcwa(structure=RASTER_1D, **KW)
        solver.formulation = 'first_order'
        solver.tetm = 'tm'
        call = solver.solve
    elif door == 'solve keyword':
        solver = wavebit.rcwa(structure=RASTER_1D, tetm='tm', **KW)
        call = lambda: solver.solve(formulation='first_order')
    else:
        solver = wavebit.rcwa(structure=RASTER_1D, **KW)
        solver.formulation = 'first_order'
        call = lambda: solver.solve(tetm='tm')
    with pytest.raises(ValueError,
                       match=r"tetm='tm' and formulation='first_order' were both given") as info:
        call()
    assert "leave formulation at 'auto'" in str(info.value)
    assert re.search(_advice('tm'), str(info.value))


@pytest.mark.parametrize('route', ['shape structure', 'no structure'])
def test_first_order_beside_a_structure_off_the_pure_path_names_the_structure(route):
    """Where ``'first_order'`` is not the only thing in the way, the message names what else is.

    "Leave formulation at 'auto'" keeps the pure path only where the structure takes it. Asked
    first, the ``'first_order'`` message told a shape, and a solver that holds no structure, to do
    that -- and the next solve, at ``'auto'``, was refused for the structure. The control: once
    the structure's own reason is gone, the ``'first_order'`` message is the one left, and it is
    right.
    """
    raster = wavebit.rcwa(structure=RASTER_1D, tetm='te', **KW)
    conv = raster.eps_conv_matrices()
    if route == 'shape structure':
        solver = wavebit.rcwa(structure=SHAPES, tetm='te', **KW)
        call, named = solver.solve, 'was given with a shape structure'
    else:
        solver = wavebit.rcwa(tetm='te', **KW)
        call = lambda: solver.solve_conv(conv.epx, conv.epy, conv.epz_i)
        named = 'holds no structure'
    solver.formulation = 'first_order'
    with pytest.raises(ValueError, match=named) as info:
        call()
    assert "formulation at 'auto'" not in str(info.value)
    solver.formulation = 'auto'
    with pytest.raises(ValueError, match=named):
        call()

    solver.structure = RASTER_1D
    solver.formulation = 'first_order'
    with pytest.raises(ValueError, match=r"formulation='first_order' were both given") as info:
        solver.solve_conv(conv.epx, conv.epy, conv.epz_i)
    assert "leave formulation at 'auto'" in str(info.value)
    solver.formulation = 'auto'
    _assert_same_light(solver.solve_conv(conv.epx, conv.epy, conv.epz_i), raster.solve(), 'te',
                       tol=0.0)


@pytest.mark.parametrize('door', ['z row', 'formulation keyword', 'shape assigned'])
def test_a_refusal_leaves_the_last_solve_as_it_was(door):
    """A refused call changes nothing a caller who catches it goes on to read.

    The routing used to be written before the refusal -- by the z row's reroute, and by
    ``_assign_grating_type`` for a shape -- so a caller who caught the ``ValueError`` held a
    solver whose ``grating_type`` read 2 over a pure 1D solve, and ``reconstruct_field`` of
    that solve raised ``IndexError: tuple index out of range``. The field it reconstructs
    afterwards has to be the field it reconstructed before, exactly.
    """
    solver = wavebit.rcwa(structure=RASTER_1D, tetm='te', answers='field', **KW)
    solver.solve()
    before = solver.reconstruct_field(5, 1, 5)
    conv = solver.eps_conv_matrices()
    if door == 'z row':
        call = lambda: solver.solve_conv(conv.epx, conv.epy, conv.epz_i,
                                         **_z_slots(conv, epxz_conv_all=True))
    elif door == 'formulation keyword':
        call = lambda: solver.solve(formulation='first_order')
    else:
        # The assignment alone moves nothing the reconstruction reads -- the premise, so that
        # what the refused solve does is all that is measured.
        solver.structure = SHAPES
        for got, want in zip(solver.reconstruct_field(5, 1, 5), before):
            assert np.array_equal(np.asarray(got), np.asarray(want))
        call = solver.solve
    with pytest.raises(ValueError, match="tetm='te'"):
        call()
    assert solver.grating_type == 0
    for got, want in zip(solver.reconstruct_field(5, 1, 5), before):
        assert np.array_equal(np.asarray(got), np.asarray(want))


@pytest.mark.parametrize('tetm', ['te', 'tm'])
def test_the_advice_beside_first_order_is_the_cross_check_it_promises(tetm):
    """``formulation='first_order'`` on the named pair reproduces the pure ``tetm`` solve."""
    want = wavebit.rcwa(structure=RASTER_1D, tetm=tetm, **KW).solve()
    advised = wavebit.rcwa(structure=RASTER_1D, polarization=(1, 0) if tetm == 'te' else (0, 1),
                           phi=0.0, **KW)
    advised.formulation = 'first_order'
    got = advised.solve()
    assert advised.formulation_assigned == 'first_order'
    _assert_same_light(got, want, tetm)


# ------------------------------------------------------------------- the structure's routes

def _stack(structure):
    return Stack(layers=[Layer(thickness=300.0, structure=structure[0])], top=1.0, bot=2.25,
                 period=(700.0, 700.0))


def _incidence(**over):
    return Incidence(**dict(dict(wavelength=900.0, theta=0.2, tetm='te'), **over))


_NUMERICS = dict(truncation_order=[3, 0])


@pytest.mark.parametrize('door', ['constructor', 'tetm assigned', 'structure assigned',
                                  'solve keyword', 'typed Stack and Incidence', 'JobSpec',
                                  'solve_field'])
def test_tetm_on_a_shape_structure_is_refused_on_every_door(door):
    """A shape structure never takes the pure path, whichever way the pair was put together."""
    if door == 'constructor':
        call = wavebit.rcwa(structure=SHAPES, tetm='te', **KW).solve
    elif door == 'tetm assigned':
        solver = wavebit.rcwa(structure=SHAPES, **KW)
        solver.tetm = 'te'
        call = solver.solve
    elif door == 'structure assigned':
        solver = wavebit.rcwa(structure=RASTER_1D, tetm='te', **KW)
        solver.solve()                                     # the pure path, as a control
        assert solver.grating_type == 0
        solver.structure = SHAPES
        call = solver.solve
    elif door == 'solve keyword':
        solver = wavebit.rcwa(structure=RASTER_1D, tetm='te', **KW)
        call = lambda: solver.solve(structure=SHAPES)
    elif door == 'typed Stack and Incidence':
        solver = wavebit.rcwa(stack=_stack(SHAPES), incidence=_incidence(), **_NUMERICS)
        call = solver.solve
    elif door == 'JobSpec':
        spec = JobSpec(stack=_stack(SHAPES), incidence=_incidence(),
                       numerics=Numerics(**_NUMERICS))
        call = lambda: run(JobSpec.from_json(spec.to_json()))
    else:
        solver = wavebit.rcwa(structure=SHAPES, tetm='te', answers='field', **KW)
        call = solver.solve_field
    with pytest.raises(ValueError, match=r"tetm='te' was given with a shape structure") as info:
        call()
    assert re.search(_advice('te'), str(info.value))
    assert 'one-row raster' in str(info.value)


def test_the_advice_on_a_shape_structure_solves():
    """The pair the message names runs on the 2D layout the message says shapes take."""
    solver = wavebit.rcwa(structure=SHAPES, polarization=(1, 0), phi=0.0, **KW)
    res = solver.solve()
    assert solver.grating_type == 2
    assert np.isfinite(np.asarray(res.R)).all() and np.asarray(res.R).sum() > 1e-3


def test_the_shape_refusal_holds_under_jit_and_grad():
    """Traced through a frozen geometry and through an un-jitted gradient alike."""
    jax = pytest.importorskip('jax')
    solver = wavebit.rcwa(backend=1, structure=SHAPES, tetm='te', **KW)
    geometry = solver.freeze_geometry()
    import jax.numpy as jnp

    def transmitted(thickness):
        return jnp.sum(solver.solve(thickness=[thickness], geometry=geometry).T)

    for traced in (jax.jit(transmitted), jax.grad(transmitted)):
        with pytest.raises(ValueError, match='shape structure'):
            traced(jnp.asarray(300.0))


def test_tetm_on_a_solver_with_no_structure_is_refused():
    """``solve_conv`` with nothing assigned keeps the layout of the last solve, 2D if none.

    The matrices alone cannot say 1D, so a solver that never solved is on the 2D layout, and
    one whose last solve was conical is on the conical one -- which reads the incident pair
    just the same, and on 0937b35 answered for s light when ``tetm`` was assigned afterwards.
    The one layout it may keep is the pure one: after a pure solve the same call answers --
    re-checked, as the next test shows.
    """
    raster = wavebit.rcwa(structure=RASTER_1D, tetm='te', **KW)
    conv = raster.eps_conv_matrices()
    want = raster.solve()
    bare = wavebit.rcwa(tetm='te', **KW)
    with pytest.raises(ValueError, match=r"tetm='te' was given to a solver that holds no "
                                         r'structure.* here the 2D one') as info:
        bare.solve_conv(conv.epx, conv.epy, conv.epz_i)
    assert re.search(_advice('te'), str(info.value))
    # The first half of the advice: with the raster assigned, the same call is the pure path.
    bare.structure = RASTER_1D
    got = bare.solve_conv(conv.epx, conv.epy, conv.epz_i)
    assert bare.grating_type == 0
    _assert_same_light(got, want, 'te', tol=0.0)
    # ... and it may then drop the structure: the pure layout is the one it keeps.
    bare.structure = None
    _assert_same_light(bare.solve_conv(conv.epx, conv.epy, conv.epz_i), want, 'te', tol=0.0)

    conical = wavebit.rcwa(structure=RASTER_1D, **KW)
    conical.solve()
    assert conical.grating_type == 1
    conical.structure = None
    conical.tetm = 'te'
    with pytest.raises(ValueError, match=r'holds no structure.* here the conical 1D one'):
        conical.solve_conv(conv.epx, conv.epy, conv.epz_i)


#: What can move after the pure solve that chose the layout, and the sentence a one-row
#: raster is refused with for the same move.
MOVED = {
    'phi': (dict(phi=0.4), 'tetm and a nonzero phi'),
    'ky': (dict(theta=None, kx=0.001, ky=0.0005), 'tetm requires ky=0'),
    'orders': (dict(truncation_order=[3, 2]), 'asks for y diffraction orders'),
    'skew': (dict(lattice=wavebit.Lattice.oblique(700.0, 700.0, np.pi / 3)), 'skewed lattice'),
}


@pytest.mark.parametrize('moved', MOVED)
def test_a_kept_pure_layout_asks_again_what_its_raster_was_asked(moved):
    """The pure layout kept with no structure is re-checked, with the raster's own sentences.

    The scene can move after the pure solve that chose the layout, and until this was asked
    the kept layout answered whatever it became, measured on 0937b35: ``phi = 0.4`` returned
    the ``phi = 0`` solve (TE sum R 0.0162738, where the pair at ``phi = 0.4`` is 0.0363008);
    a nonzero ``ky`` and a skewed cell answered too, the latter the rectangular problem; and
    ``truncation_order=[3, 2]`` ended in a bare ``operands could not be broadcast``. The same
    moves on the raster itself are refused, so they are refused here, before the routing is
    written -- the layout stays the pure one, and the solve before it stays readable.
    """
    raster = wavebit.rcwa(structure=RASTER_1D, tetm='te', **KW)
    conv = raster.eps_conv_matrices()
    raster.solve()
    raster.structure = None
    change, sentence = MOVED[moved]
    for name, value in change.items():
        setattr(raster, name, value)
    with pytest.raises(ValueError, match=sentence):
        raster.solve_conv(conv.epx, conv.epy, conv.epz_i)
    assert raster.grating_type == 0
    if moved == 'phi':
        # The premise: phi = 0.4 is another scene, so answering it was wrong, not merely early.
        pure = float(np.sum(wavebit.rcwa(structure=RASTER_1D, tetm='te', **KW).solve().R))
        tilted = wavebit.rcwa(structure=RASTER_1D, polarization=(1, 0), phi=0.4, **KW).solve()
        assert abs(float(np.sum(tilted.R)) - pure) > 1e-2


#: The sentence ``tetm`` beside a tensor has always been refused with, where the matrices are built.
TENSOR = 'tensor materials require the coupled polarization solve; use polarization instead of tetm'


def test_tetm_after_a_tensor_was_built_is_refused_with_the_tensor_sentence():
    """The tensor route past the build: matrices taken first, ``tetm`` declared after."""
    from wavebit import Material
    film = dict(wavelength=600.0, period=(300.0, 300.0), thickness=[220.0], phi=0.0,
                structure=[dict(background=Material([2.0, 3.0, 4.0]), shapes=[])],
                truncation_order=(0, 0))
    solver = wavebit.rcwa(**film)
    conv = solver.eps_conv_matrices()
    solver.tetm = 'te'
    with pytest.raises(ValueError, match=TENSOR):
        solver.solve_conv(*conv)
    # ... and at the build, the precedent this sentence came from -- on the typed and spec doors.
    with pytest.raises(ValueError, match=TENSOR):
        wavebit.rcwa(**film, tetm='te').solve()
    spec = JobSpec(stack=Stack(layers=[Layer(thickness=220.0, structure=film['structure'][0])],
                               period=(300.0, 300.0)),
                   incidence=Incidence(wavelength=600.0, phi=0.0, tetm='te'),
                   numerics=Numerics(truncation_order=(0, 0)))
    with pytest.raises(ValueError, match=TENSOR):
        run(JobSpec.from_json(spec.to_json()))


# ---------------------------------------------------------------------------------- control

@pytest.mark.parametrize('tetm', ['te', 'tm'])
def test_the_pure_path_is_untouched(tetm):
    """``tetm`` on a one-row raster and no z row: the pure path, through ``solve`` and
    ``solve_conv`` both -- the guard costs the ordinary call nothing."""
    solver = wavebit.rcwa(structure=RASTER_1D, tetm=tetm, **KW)
    via_solve = solver.solve()
    assert solver.grating_type == 0
    via_conv = solver.solve_conv(*solver.eps_conv_matrices())
    assert solver.grating_type == 0
    _assert_same_light(via_conv, via_solve, tetm, tol=0.0)
