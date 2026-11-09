"""Our Torch ``asarray`` calls on a tensor that requires grad state ``requires_grad``.

Since torch 2.14 an unspecified ``requires_grad`` follows the input and torch warns once per
process ("unspecified requires_grad now defaults to obj.requires_grad"). Wavebit's calls want
exactly that value, so they say it; the warning was printed on every Torch run that took a
radius or a layer-thickness derivative. ``torch.asarray`` is wrapped here, rather than the
warning caught, because torch emits it only once per process.
"""
import pytest

from wavebit import Material, Units, rcwa

torch = pytest.importorskip('torch')


def problem(radius, thickness):
    return rcwa.Problem(units=Units(length='um'),
        stack=rcwa.Stack(period=(.6, .6), bottom=Material(epsilon=2.25),
            layers=(rcwa.Layer(thickness=thickness, regions=(rcwa.Region(
                geometry=rcwa.Circle(radius=radius), material=Material(epsilon=12.25)),)),)),
        excitation=rcwa.Excitation(wavelength=.85, sources=(rcwa.PlaneWave(),)),
        outputs=rcwa.Outputs(fields=True))


def test_no_asarray_call_leaves_requires_grad_unspecified(monkeypatch):
    unspecified = []
    original = torch.asarray

    def asarray(obj, *args, **kwargs):
        if 'requires_grad' not in kwargs and isinstance(obj, torch.Tensor) and obj.requires_grad:
            unspecified.append(tuple(obj.shape))
        return original(obj, *args, **kwargs)

    monkeypatch.setattr(torch, 'asarray', asarray)
    prepared = rcwa.prepare(problem(.12, .18), numerics=rcwa.Numerics(
        orders=(2, 2), shape_nvf_grid_size=64), backend='torch')
    radius = torch.tensor(.12, dtype=torch.float64, requires_grad=True)
    thickness = torch.tensor(.18, dtype=torch.float64, requires_grad=True)
    result = prepared.run(problem(radius, thickness))
    objective = result.transmission.order(0, 0).efficiency[0] + (
        result.field(x=[0.], y=[0.], z=[.09]).values.abs()**2).sum()
    objective.backward()
    assert radius.grad is not None and thickness.grad is not None
    assert unspecified == []
