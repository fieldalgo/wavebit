"""A cell's ``axis_aligned`` declaration cannot send a skewed cell down the rectangular path.

``Lattice(u, v, axis_aligned)`` takes the flag as given, which a pytree rebuild and a traced
``period`` need. A ``True`` that readable numbers contradict -- written by hand, or edited
into a saved job -- solved a sheared cell as a rectangle and moved the answer with nothing to
say so. At the solver's door such a cell is now declared general, the path that is always
correct. The reference is the same basis built honestly with ``Lattice.from_columns``.
Comparing two cells raised NumPy's "truth value is ambiguous" and hashing one raised
``TypeError``; both now compare the numbers and the declaration.
"""
import json

import numpy as np
import pytest

from wavebit import Lattice, Material, Units, rcwa
from wavebit.lattice import is_axis_aligned, reduced_basis

P = 400.
U, V = np.array([P, 0.]), np.array([.3*P, P])


def _problem(lattice):
    return rcwa.Problem(
        units=Units(length='nm'),
        stack=rcwa.Stack(lattice=lattice, bottom=Material(epsilon=2.25), layers=(
            rcwa.Layer(thickness=100., background=Material(epsilon=1.), regions=(
                rcwa.Region(geometry=rcwa.Circle(radius=80., center=(30., -20.)),
                            material=Material(epsilon=6.25)),)),)),
        excitation=rcwa.Excitation(wavelength=620., sources=(rcwa.PlaneWave(theta=.2, phi=.3),)))


NUMERICS = rcwa.Numerics(orders=(3, 3))


def test_a_sheared_cell_declared_rectangular_solves_as_the_sheared_cell():
    """Before, sum(T) was 0.991097681 against the honest 0.992470410."""
    honest = rcwa.run(_problem(Lattice.from_columns(np.column_stack([U, V]))), numerics=NUMERICS)
    declared = rcwa.run(_problem(Lattice(U, V, True)), numerics=NUMERICS)
    np.testing.assert_array_equal(np.asarray(declared.R), np.asarray(honest.R))
    np.testing.assert_array_equal(np.asarray(declared.T), np.asarray(honest.T))


def test_a_saved_job_whose_flag_was_edited_solves_as_its_vectors_say():
    from wavebit.jobs import Job
    honest = _problem(Lattice.from_columns(np.column_stack([U, V])))
    job = Job(solver='rcwa', problem=honest, numerics=NUMERICS)
    body = json.loads(job.to_json())
    text = json.dumps(body)
    assert '"axis_aligned": false' in text
    edited = Job.from_json(text.replace('"axis_aligned": false', '"axis_aligned": true'))
    assert edited.problem.stack.lattice.axis_aligned is True       # the record keeps what it says
    np.testing.assert_array_equal(
        np.asarray(rcwa.run(edited.problem, numerics=edited.numerics).T),
        np.asarray(rcwa.run(honest, numerics=NUMERICS).T))


def test_honest_declarations_keep_their_path():
    """A true ``True`` keeps the rectangular shortcut; a ``False`` on a rectangle stays general."""
    from wavebit.lattice import as_lattice
    rectangle = Lattice(np.array([P, 0.]), np.array([0., 1.3*P]), True)
    assert as_lattice(lattice=rectangle) is rectangle
    forced = Lattice(np.array([P, 0.]), np.array([0., 1.3*P]), False)
    assert as_lattice(lattice=forced) is forced and not is_axis_aligned(as_lattice(lattice=forced))
    demoted = as_lattice(lattice=Lattice(U, V, True))
    assert demoted.axis_aligned is False
    np.testing.assert_array_equal(np.asarray(demoted), np.column_stack([U, V]))


def test_a_traced_declaration_stands():
    """A traced ``period`` is diagonal by construction; its declaration cannot be read off numbers."""
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    from wavebit.lattice import as_lattice
    seen = []

    def probe(lx):
        seen.append(as_lattice(lattice=as_lattice(period=jnp.stack([lx, 1.3*lx]))).axis_aligned)
        return lx
    jax.jit(probe)(jnp.asarray(P))
    assert seen == [True]


def test_a_reduced_basis_keeps_its_sign_free_fold_flag():
    """The fold reads a reduced basis through ``periods_of``; a sign does not move its images.

    A sheared basis that reduces to ``diag(1, -1)`` keeps the rectangular fold it always had,
    and the reduction does not re-judge a cell that has already passed the door.
    """
    reduced = reduced_basis(Lattice.from_columns(np.array([[-3., -2.], [-2., -1.]])))
    assert np.allclose(np.asarray(reduced), np.diag([1., -1.])) and reduced.axis_aligned
    forced = Lattice(np.array([.9, 0.]), np.array([0., 1.4]), False)
    assert reduced_basis(forced) is forced


def test_cells_compare_and_hash_by_their_numbers_and_declaration():
    a = Lattice.from_columns(np.column_stack([U, V]))
    b = Lattice.from_columns(np.column_stack([U.copy(), V.copy()]))
    assert a == b and hash(a) == hash(b) and len({a, b}) == 1
    assert a != Lattice.from_columns(np.column_stack([U, 1.01*V]))
    assert Lattice.rectangular(P) == Lattice.rectangular(P)
    assert Lattice.rectangular(P) != Lattice(np.array([P, 0.]), np.array([0., P]), False)
    assert {Lattice.rectangular(P): 'square'}[Lattice.rectangular(P)] == 'square'


def test_a_traced_cell_equals_only_itself():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    out = []

    def probe(u):
        a = Lattice(u, jnp.asarray(V), False)
        b = Lattice(u, jnp.asarray(V), False)
        out.append((a == a, a == b, isinstance(hash(a), int)))
        return u
    jax.jit(probe)(jnp.asarray(U))
    assert out == [(True, False, True)]
