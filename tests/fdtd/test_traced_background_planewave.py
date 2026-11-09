"""A traced ``Scene(background=)`` beside a PlaneWave: where it differentiates and where it is refused.

``shape_ranges._seeded_materials`` refuses a traced background beside a PlaneWave in the scenes
it serves (a traced geometry, or smoothed coupling), whatever the PML references, and advises a
traced structure material instead, which runs. Elsewhere the background is lowered live: at
normal incidence it differentiates once both Open z faces have a ``PML(reference_media=...)``
entry, and without them the matched face is refused.
"""
import sys
import pathlib

import pytest

from wavebit import Material, Units, fdtd
from wavebit.errors import ValidationError
from wavebit.units import C0

torch = pytest.importorskip('torch')

_PULSE = fdtd.GaussianPulse(wavelength=1., frequency_sigma_hz=.15*C0/1e-6)
_FIXED = {'z-': Material(1.), 'z+': Material(1.)}


def _film(background):
    return fdtd.Problem(units=Units(length='um'), scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.6, .6, 1.3)), background=Material(background),
        boundaries=fdtd.Boundaries.periodic(z=fdtd.Open()),
        structures=(fdtd.Structure('film', shape=fdtd.SnappedSlab(z=(-.3, -.1)), material=Material(2.25)),)),
        sources=(fdtd.PlaneWave('pw', plane=fdtd.Plane(z=.45), pulse=_PULSE),),
        outputs=fdtd.Outputs(monitors=(fdtd.DiffractionMonitor(
            'T', plane=fdtd.Plane(z=-.55), source='pw', wave='transmitted', wavelengths=(1.,),
            orders=(0, 0)),)))


def _numerics(reference_media=None, **extra):
    absorber = fdtd.PML(cells=10, wavelength=1., allow_thin=True,
                        **({} if reference_media is None else dict(reference_media=reference_media)))
    return fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.05), duration=fdtd.Duration(steps=600),
                         absorber=absorber, **extra)


def test_a_traced_background_beside_a_normal_plane_wave_differentiates_with_both_faces_fixed():
    numerics = _numerics(_FIXED)
    prepared = fdtd.prepare(_film(1.), numerics=numerics, backend='torch')
    background = torch.tensor(1., dtype=torch.float64, requires_grad=True)
    (derivative,) = torch.autograd.grad(prepared.run(_film(background))['T'].efficiency[0, 0, 0],
                                        background)
    h = 1e-5

    def transmitted(value):
        return float(fdtd.run(_film(value), numerics=numerics)['T'].efficiency[0, 0, 0])
    central = (transmitted(1. + h) - transmitted(1. - h))/(2*h)
    assert float(derivative) == pytest.approx(central, rel=1e-6)
    # Without a reference on the faces, the PML matched to the background is refused.
    prepared = fdtd.prepare(_film(1.), numerics=_numerics(), backend='torch')
    with pytest.raises(ValidationError, match='is matched to a background whose permittivity'):
        prepared.run(_film(torch.tensor(1., dtype=torch.float64, requires_grad=True)))


def test_a_traced_background_beside_a_plane_wave_is_refused_with_a_traced_geometry():
    sys.path.insert(0, str(pathlib.Path(__file__).parents[2]/'examples'/'fdtd'))
    try:
        import pillar_gradient as pillar
    finally:
        sys.path.pop(0)

    def problem(background, radius=.12):
        p = pillar.problem(radius=radius)
        return p.replace(scene=p.scene.replace(background=Material(background)))
    # Both faces fixed, as the film above needs; the refusal here does not depend on them.
    numerics = _numerics({'z-': Material(2.1), 'z+': Material(1.)}, geometry_ranges={
        'pillar': ((-.145, .145), (-.145, .145), (pillar.PLANE - .001, pillar.PLANE + .401))})
    prepared = fdtd.prepare(problem(1.), numerics=numerics, backend='torch')
    background = torch.tensor(1., dtype=torch.float64, requires_grad=True)
    radius = torch.tensor(.12, dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValidationError, match=r'Scene\(background=\) has a traced value in a scene '
                                              r'with a traced Cylinder.*a PlaneWave') as caught:
        prepared.run(problem(background, radius))
    # The advice is the one that runs: the face references above are already given and do not
    # lift it, so it says so instead of asking for them.
    message = caught.value.message
    assert 'background=' not in message.split('host.')[-1]
    assert 'does not lift this' in message
    assert "Keep the background concrete and differentiate a structure's material" in message
    # Followed: a concrete background, the pillar's own permittivity traced beside its radius.
    epsilon = torch.tensor(2.25, dtype=torch.float64, requires_grad=True)
    p = pillar.problem(radius=radius, epsilon=epsilon)
    efficiency = prepared.run(p.replace(scene=p.scene.replace(background=Material(1.))))
    gradient = torch.autograd.grad(efficiency['T'].efficiency[0, 0, 0], (radius, epsilon))
    assert all(bool(torch.isfinite(g)) and float(g) != 0. for g in gradient)


def test_a_matched_pml_reference_is_advised_in_public_spelling():
    from wavebit.fdtd._spelling import to_public
    advice = 'Give every PML face a concrete reference, fd.PML(background=...), or'
    assert to_public(advice) == ('Give every PML face a concrete reference, '
                                 'fdtd.PML(reference_media=...) with an entry for every Open face, or')
