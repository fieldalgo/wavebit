"""A diagonal cell with a negative side spans the same lattice as its positive spelling.

``lattice=diag(-Lx, Ly)`` was taken for the rectangular path, which reads the cell as
``(|a1|, |a2|)``, while the shape coefficients and the field reconstruction read the basis as
spelled: a shape was solved as its mirror image and a field came back mirrored, with every
diagnostic clean. Such a basis now takes the general path, which reads every basis as spelled.
The references here are the positive spelling of the same structure: the shapes themselves are
Cartesian, and a raster drawn along ``a1`` is the index-reversed raster on the positive cell.
"""
import numpy as np
import pytest

from wavebit import Lattice, Material, Units, rcwa
from wavebit.errors import UnsupportedFeatureError
from wavebit.lattice import is_axis_aligned, reverses_an_axis

P = 400.
SPELLINGS = {'(-,+)': (-1, 1), '(+,-)': (1, -1), '(-,-)': (-1, -1)}
X, Y, Z = np.array([-120., -10., 77.]), np.array([-130., 33., 150.]), np.array([20., 50.])


def _problem(signs, layer):
    sx, sy = signs
    return rcwa.Problem(
        units=Units(length='nm'),
        stack=rcwa.Stack(lattice=np.diag([sx*P, sy*P]), bottom=Material(epsilon=2.25), layers=(layer,)),
        excitation=rcwa.Excitation(wavelength=620., sources=(
            rcwa.PlaneWave(theta=.2, phi=.3, polarization=(1., .4j)),)),
        outputs=rcwa.Outputs(fields=True))


def _shapes():
    metal = Material(epsilon=-10+3j)
    return rcwa.Layer(thickness=100., background=Material(epsilon=1.), regions=(
        rcwa.Region(geometry=rcwa.Rectangle(size=(150., 120.), center=(-60., -30.)), material=metal),
        rcwa.Region(geometry=rcwa.Ellipse(radii=(60., 35.), center=(90., 100.), angle=.5), material=metal)))


def _raster(n=24):
    u = (np.arange(n) + .5)/n
    return np.where(((u[None, :] < .45) & (u[:, None] < .3)) | (u[None, :] > .8), 4., 1.)


def _agree(result, reference, signs):
    """Same structure: same per-order powers and the same Cartesian field.

    Order ``(m, n)`` is ``m*b1 + n*b2`` of the basis as spelled, as on every other cell, so a
    reversed side names its orders in reverse: the ``(..., n, m)`` arrays are compared reversed
    along that axis.
    """
    sx, sy = signs
    for name in ('R', 'T'):
        np.testing.assert_allclose(np.asarray(getattr(result, name))[..., ::sy, ::sx],
                                   np.asarray(getattr(reference, name)), rtol=0, atol=1e-12)
    field = np.asarray(result.field(x=X, y=Y, z=Z).values)
    expected = np.asarray(reference.field(x=X, y=Y, z=Z).values)
    assert np.abs(field - expected).max() <= 1e-10*np.abs(expected).max()


@pytest.mark.parametrize('factorization', ['nvf', 'laurent'])
@pytest.mark.parametrize('spelling', sorted(SPELLINGS))
def test_shapes_on_a_negative_side_are_the_same_structure(spelling, factorization):
    """Off-centre metal shapes at oblique, elliptical incidence: nothing in them is symmetric.

    Before, ``R`` moved by up to 1.8e-2 and the field at fixed points by 30-45 % of its peak.
    """
    numerics = rcwa.Numerics(orders=(3, 3), factorization=factorization)
    reference = rcwa.run(_problem((1, 1), _shapes()), numerics=numerics)
    _agree(rcwa.run(_problem(SPELLINGS[spelling], _shapes()), numerics=numerics), reference,
           SPELLINGS[spelling])


@pytest.mark.parametrize('factorization', ['nvf', 'laurent'])
@pytest.mark.parametrize('spelling', sorted(SPELLINGS))
def test_a_raster_is_drawn_along_the_basis_as_spelled(spelling, factorization):
    """``structure[j, i]`` samples ``(i + .5)/Nx * a1 + (j + .5)/Ny * a2``, the documented contract.

    Along a reversed axis that is the index-reversed raster on the positive cell. Before, the
    solve read the raster along ``+x`` -- ``R`` of the unreversed raster -- while the field was
    reconstructed along ``-x``.
    """
    sx, sy = SPELLINGS[spelling]
    raster = _raster()
    mirrored = raster[::sy, ::sx]
    numerics = rcwa.Numerics(orders=(3, 3), factorization=factorization)
    reference = rcwa.run(_problem((1, 1), rcwa.Layer(thickness=100., epsilon=mirrored)), numerics=numerics)
    result = rcwa.run(_problem((sx, sy), rcwa.Layer(thickness=100., epsilon=raster)), numerics=numerics)
    _agree(result, reference, (sx, sy))


def test_a_raster_painted_with_meshgrid_xy_is_one_structure_on_either_spelling():
    """The user's procedure: paint ``eps(x, y)`` on the cell's own ``meshgrid_xy``, then solve."""
    def painted(lattice, n=32):
        x, y = lattice.meshgrid_xy((n, n))
        return np.where((x + 60.)**2/150.**2 + (y + 30.)**2/90.**2 < 1., 4., 1.)
    numerics = rcwa.Numerics(orders=(3, 3), factorization='laurent')
    positive = Lattice.rectangular(P)
    reference = rcwa.run(_problem((1, 1), rcwa.Layer(thickness=100., epsilon=painted(positive))),
                         numerics=numerics)
    for sx, sy in SPELLINGS.values():
        lattice = Lattice.from_columns(np.diag([sx*P, sy*P]))
        result = rcwa.run(_problem((sx, sy), rcwa.Layer(thickness=100., epsilon=painted(lattice))),
                          numerics=numerics)
        _agree(result, reference, (sx, sy))


def test_orders_are_named_along_the_basis_as_spelled():
    """``order(1, 0)`` carries ``k_inc + b1``; on ``diag(-P, P)`` that ``b1`` points along ``-x``."""
    def problem(sx):
        return rcwa.Problem(units=Units(length='nm'),
            stack=rcwa.Stack(lattice=np.diag([sx*P, P]), layers=(rcwa.Layer(thickness=100., epsilon=_raster()),)),
            excitation=rcwa.Excitation(wavelength=300., sources=(rcwa.PlaneWave(theta=.2, phi=.3),)))
    numerics = rcwa.Numerics(orders=(1, 1), factorization='laurent')
    k_inc_x = 2*np.pi/300.*np.sin(.2)*np.cos(.3)
    for sx in (1, -1):
        order = rcwa.run(problem(sx), numerics=numerics).reflection.order(1, 0)
        np.testing.assert_allclose(np.asarray(order.k_parallel).ravel()[0].real,
                                   k_inc_x + sx*2*np.pi/P, rtol=1e-12, atol=0)


def test_li_on_a_negative_side_is_refused_and_says_why():
    """Li is built on the rectangular path; the refusal names the side, not a skew."""
    numerics = rcwa.Numerics(orders=(2, 2), factorization='li')
    problem = _problem((-1, 1), rcwa.Layer(thickness=100., epsilon=_raster()))
    with pytest.raises(UnsupportedFeatureError, match='negative side') as caught:
        rcwa.run(problem, numerics=numerics)
    assert 'non-orthogonal' not in str(caught.value)


def test_only_a_positive_diagonal_takes_the_rectangular_path():
    assert is_axis_aligned(np.diag([P, 2*P]))
    assert Lattice.rectangular(P, 2*P).axis_aligned
    for sx, sy in SPELLINGS.values():
        basis = np.diag([sx*P, sy*2*P])
        assert not is_axis_aligned(basis)
        assert not Lattice.from_columns(basis).axis_aligned
        assert not Lattice.rectangular(sx*P, sy*2*P).axis_aligned
        assert reverses_an_axis(basis) and reverses_an_axis(Lattice.from_columns(basis))
    assert not reverses_an_axis(np.diag([P, P]))
    assert not reverses_an_axis(np.array([[P, .3*P], [0., P]]))
