"""Stage 4's C5: the doors whose reason changed, and the body contract the mixing rule rests on.

Each case is a message that used to name the wrong cause or a spelling that does not exist, or
a guard that was promised and not written: a smoothed pair meeting the stage-1 overlap door
(whose advice is about slabs) instead of door 1; a smoothed body across a PlaneWave's sheet; a
body answering a fill outside ``[0, 1]``, where the mixing rule's ``eps`` can go negative.
"""
import warnings

import pytest

import wavebit.fdtd._legacy as fd
from wavebit.fdtd._shapes import Box as KernelBox
from tests.fixtures.fdtd.fd_helpers import bloch_plane_wave


def _quiet_plan(scene):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fd.plan(scene)


def _box(structures, size=1.2, dx=0.05):
    return fd.Scene(domain=fd.Domain(size=(size,) * 3),
                    boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
                    resolution=fd.Resolution(dx=dx), run=fd.Run(steps=4),
                    structures=tuple(structures))


def test_a_smoothed_dispersive_dimer_hears_door_one_not_the_slab_overlap_door():
    """A Drude dimer on the 45 degree diagonal, gap half a diameter: supports overlap, voxels
    do not. The overlap door's support-box test refused it advising disjoint slabs (C5 audit);
    door 1's reason is the true one -- poles are not smoothed before C8."""
    drude = fd.Dispersive(eps_inf=1.0, poles=(fd.Drude(omega_p=5.0, gamma=0.5),))
    r = 0.1
    half = (2 * r + r) / (2 * 2 ** 0.5)          # centre distance 3 r along the diagonal
    pair = tuple(fd.Structure(fd.Sphere(radius=r, center=(s * half, s * half, 0.0)), drude)
                 for s in (-1, 1))
    with pytest.raises(ValueError) as caught:
        _quiet_plan(_box(pair))
    message = str(caught.value)
    assert 'share cells' not in message and 'disjoint slabs' not in message
    assert 'smoothed' in message


def test_a_smoothed_body_across_a_plane_wave_sheet_is_refused_and_one_clear_plans():
    wave = dict(transverse=(0.6, 0.6), length=1.0)
    ball = fd.Structure(fd.Sphere(radius=0.1, center=(0.0, 0.0, 0.0)), 2.25)
    scene, _ = bloch_plane_wave(structures=(ball,), **wave)
    with pytest.raises(ValueError, match=r'inside structures\[0\]'):
        _quiet_plan(scene)
    clear = fd.Structure(fd.Sphere(radius=0.1, center=(0.0, 0.0, -0.25)), 2.25)
    _quiet_plan(bloch_plane_wave(structures=(clear,), **wave)[0])


class _Overfull(KernelBox):
    """A kernel box that answers 1.5 times its fill: a body that breaks the contract."""

    def exact_fill(self, points, dx):
        fill = super().exact_fill(points, dx)
        return None if fill is None else 1.5 * fill


class _OverfullBox(fd.Box):
    def body(self):
        inner = super().body()
        return _Overfull(size=inner.size, center=inner.center, angle=inner.angle,
                         axis=inner.axis)


def test_a_body_answering_a_fill_outside_zero_to_one_is_refused():
    """At fill 1.5 against eps 12 in vacuum the normal component reads the harmonic mean
    ``1/(1.5/12 - 0.5) = -2.67``; the guard refuses the body, naming the contract."""
    with pytest.raises(ValueError) as caught:
        _quiet_plan(_box((fd.Structure(_OverfullBox(size=(0.4, 0.4, 0.4)), 12.0),)))
    message = str(caught.value)
    assert 'outside [0, 1]' in message and 'fill' in message and 'fd.Shape contract' in message
    _quiet_plan(_box((fd.Structure(fd.Box(size=(0.4, 0.4, 0.4)), 12.0),)))


class _Scaled(KernelBox):
    """A kernel box whose fill or moment is bent by ``bend``: bodies that break the contract."""

    bend = None

    def exact_fill(self, points, dx):
        fill = super().exact_fill(points, dx)
        return fill if fill is None or self.bend[0] != 'fill' else self.bend[1](fill)

    def exact_moment(self, points, dx):
        pair = super().exact_moment(points, dx)
        if pair is None or self.bend[0] != 'w':
            return pair
        return pair[0], self.bend[1](pair[1])


def _bent(kind, how, angle=0.0):
    class Bent(fd.Box):
        def body(self):
            inner = super().body()
            out = _Scaled(size=inner.size, center=inner.center, angle=inner.angle,
                          axis=inner.axis)
            object.__setattr__(out, 'bend', (kind, how))
            return out
    return Bent(size=(0.43, 0.43, 0.43), angle=angle)


@pytest.mark.parametrize('kind, how, words', (
    ('fill', lambda f: f - 0.2, 'below 0'),
    ('fill', lambda f: f * float('nan'), 'not finite'),
    ('w', lambda m: 3.0 * m, 'above 1')))
def test_every_arm_of_the_fill_contract_refuses(kind, how, words):
    message = str(pytest.raises(ValueError, _quiet_plan, _box((fd.Structure(
        _bent(kind, how), 2.25),))).value)
    assert words in message and 'fd.Shape contract' in message


def test_the_contract_speaks_before_the_lift():
    """A turned contract-breaker at eps 12 has cross terms. The Gershgorin certificate that
    preceded the lift used to answer first, calling its real negative ``1/eps`` (-0.375) an
    inconclusive bound (the C5 review); the contract is refused by name before any operator
    is built."""
    message = str(pytest.raises(ValueError, _quiet_plan, _box((fd.Structure(
        _bent('fill', lambda f: 1.5 * f, angle=30.0), 12.0),))).value)
    assert 'above 1' in message and 'fd.Shape contract' in message


def test_a_correct_body_far_from_the_origin_is_not_refused_by_round_off():
    """3e5 from the origin at dx 0.05 a sphere's fills read up to 1 + 2.8e-9 (the C5 review);
    the guard's tolerance scales with the coordinates, and the scene plans."""
    far = _box((fd.Structure(fd.Sphere(radius=0.25, center=(3e5 + 0.013, 3e5 - 0.007,
                                                               3e5 + 0.021)), 2.25),))
    _quiet_plan(far.replace(domain=fd.Domain(size=(1.2,) * 3, center=(3e5,) * 3)))


def test_a_smoothed_body_in_a_snapped_dispersive_region_hears_door_one():
    """The mixed pair: the overlap door skips it only because door 1 answers it."""
    drude = fd.Dispersive(eps_inf=1.0, poles=(fd.Drude(omega_p=5.0, gamma=0.5),))
    scene = _box((fd.Structure(fd.Region(size=(0.5, 0.5, 0.5)), drude),
                  fd.Structure(fd.Sphere(radius=0.1), 2.25)))
    message = str(pytest.raises(ValueError, _quiet_plan, scene).value)
    assert 'share cells' not in message and 'smoothed' in message


def test_the_advice_names_only_spellings_that_plan():
    """The figure-eight's two lobes meet in voxels door 5 refuses; a cylinder through a layer is
    refused by its cross terms, with no date for support. Neither is offered as a remedy any more."""
    with pytest.raises(ValueError) as caught:
        fd.Prism(vertices=((-0.2, -0.1), (0.2, 0.1), (0.2, -0.1), (-0.2, 0.1)), height=0.3)
    assert 'keep a voxel between them' in str(caught.value)
    pml = fd.PML(cells=6, wavelength=1.0, allow_thin=True)
    layered = fd.Scene(domain=fd.Domain(size=(1.8, 1.2, 1.2)),
                       boundaries=fd.Boundaries(x=pml, y=fd.PEC(), z=fd.PEC()),
                       resolution=fd.Resolution(dx=0.05), run=fd.Run(steps=4),
                       structures=(fd.Structure(fd.Box(size=(1.4, 0.4, 0.4)), 2.25),))
    message = str(pytest.raises(ValueError, _quiet_plan, layered).value)
    assert 'not supported yet, and no date is set' in message and 'unturned fd.Box can' in message
    assert 'C9' not in message
