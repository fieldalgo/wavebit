"""Gate 8 -- AD-vs-FD plumbing: a gradient through the kernel reaches the traced leaves, and is right.

**A plumbing check, not physics.** Nothing here says the fields are right; gates 1-4 do
that. What is held is that ``jax.grad`` and ``torch.autograd`` differentiate what the kernel
computes -- through the jitted scan, the index scatters and the in-place slab Functions with
their hand-written backward rules -- and land on the same number as a central finite
difference of the same functional, at three points, to 1e-6 relative:

* ``d(sum |E_final|^2) / d(amplitude)`` for the PEC box carrying its eigenmode, with the
  initial field scaled by ``InitialField(amplitude=...)``. The energy is exactly quadratic
  in the amplitude, so the central difference is exact and the check is purely whether the
  leaf is reached and the walls' masks are differentiated correctly;
* ``d(sum |E_final|^2) / d(phi)`` for the Bloch box, ``phi`` the phase on ``x``: a genuinely
  nonlinear dependence through the complex pad factor and its conjugate on the low face.

The reference is the NumPy backend, which has no AD: the same body of code, so a difference
is the AD plumbing (gate 5 pins the backends against each other separately). The finite-
difference step is taken from the scale of the leaf. Two mutants show the tolerance is doing
work on the plumbing itself: a Torch ``slab_mul`` whose backward forgets the mask (the exact
shape of a wrong hand-written rule) is off by a factor of four, and a JAX ``conj`` under
``stop_gradient`` (one face's factor cut out of the chain) halves the phase gradient.

The other half of the contract: a gradient with respect to a **static** value is refused by
name, at the constructor. ``Run.steps``, ``Resolution.dx``, ``Domain.size``, ``Run.courant``
and ``Record.at`` fix the plan -- the cell counts, ``dt``, the scan length, the skeleton --
so a tracer handed to any of them is refused naming the field and the rule
(``src/wavebit/_common/declarations.py``). Both traced forms are held: a JAX tracer, on which ``float()``
raises, and a Torch tensor with ``requires_grad``, on which ``float()`` *succeeds* -- the
silent case, where the value would be detached, the run would complete and the gradient
would simply be absent (that is what the first version of the doors did, 2026-09-18).
"""

from wavebit.compat import legacy as _legacy_wavebit
import pytest

import wavebit.fdtd._legacy as fd
from wavebit.fdtd import _runtime
from tests.fixtures.fdtd.fd_helpers import cavity_box, plane_wave_box

REL = 1e-6
MUTANT_FLOOR = 1e-3
AMPLITUDES = (0.5, 1.0, 2.0)
PHASES = (0.5, 0.7, 1.3)
Z_PHASE = -1.1


def _ad_backends():
    return [pytest.param(b, marks=() if _installed(b) else pytest.mark.skip(f'{b} not installed'))
            for b in ('jax', 'torch')]


def _installed(backend):
    try:
        __import__(backend)
    except ImportError:
        return False
    return True


@pytest.fixture(scope='module')
def pec():
    return cavity_box('pec', N=8, steps=40)[0]


@pytest.fixture(scope='module')
def bloch():
    return plane_wave_box((PHASES[1], Z_PHASE), N=8, steps=40)[0]


def _energy(scene, backend):
    """``sum |E_final|^2`` as a scalar of ``backend`` (a NumPy float on NumPy)."""
    return (abs(fd.run(scene, backend=backend).final.E) ** 2).sum()


def _with_amplitude(scene, a):
    return scene.replace(initial=scene.initial.replace(amplitude=a))


def _with_phase(scene, phi):
    return scene.replace(boundaries=scene.boundaries.replace(x=fd.Bloch(phase=phi)))


def _grad(backend, f, x):
    """``df/dx`` at the Python float ``x`` by the backend's AD; ``f`` returns a real scalar."""
    if backend == 'jax':
        jax = pytest.importorskip('jax')
        # Before the tracer exists: a float traced under the 32-bit cap stays float32, and
        # the runtime's own lift comes after the tracer was made.
        _legacy_wavebit.enable_x64()
        return float(jax.grad(f)(x))
    torch = pytest.importorskip('torch')
    t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    return float(torch.autograd.grad(f(t), t)[0])


def _central_difference(f, x, h):
    return (float(f(x + h)) - float(f(x - h))) / (2 * h)


# ---------------------------------------------------------------------------------------------
# The two leaves
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('backend', _ad_backends())
@pytest.mark.parametrize('amplitude', AMPLITUDES)
def test_the_amplitude_gradient_matches_central_differences(pec, backend, amplitude):
    def energy(a, backend=backend):
        return _energy(_with_amplitude(pec, a), backend)

    h = 1e-3 * amplitude                      # from the amplitude scale
    reference = _central_difference(lambda a: energy(a, 'numpy'), amplitude, h)
    # The energy is a^2 times the unit run's, so the reference is 2 a E(1) and cannot be
    # tiny: the relative comparison below is doing work.
    assert reference == pytest.approx(2 * amplitude * float(energy(1.0, 'numpy')), rel=1e-9)
    assert abs(_grad(backend, energy, amplitude) - reference) <= REL * abs(reference)


@pytest.mark.parametrize('backend', _ad_backends())
@pytest.mark.parametrize('phase', PHASES)
def test_the_bloch_phase_gradient_matches_central_differences(bloch, backend, phase):
    def energy(phi, backend=backend):
        return _energy(_with_phase(bloch, phi), backend)

    h = 1e-4 * phase                          # from the phase scale, in radians
    reference = _central_difference(lambda p: energy(p, 'numpy'), phase, h)
    assert abs(reference) > 1e-3 * float(energy(phase, 'numpy')), \
        'a flat point cannot test a gradient'
    assert abs(_grad(backend, energy, phase) - reference) <= REL * abs(reference)


# ---------------------------------------------------------------------------------------------
# Mutants: the check sees a wrong backward rule and a cut chain
# ---------------------------------------------------------------------------------------------

class _ForgetfulMaskRuntime(_runtime.TorchRuntime):
    """Torch with a ``slab_mul`` whose backward passes the cotangent through the mask unchanged."""

    def __init__(self, device=None):
        super().__init__(device)
        torch = self.torch

        class SlabMulForgetsTheMask(torch.autograd.Function):
            @staticmethod
            def forward(ctx, a, m, slices):
                a[slices].mul_(m)
                ctx.mark_dirty(a)
                return a

            @staticmethod
            def backward(ctx, g):
                return g, None, None

        self.SlabMul = SlabMulForgetsTheMask


class _HalfBlindRuntime(_runtime.JaxRuntime):
    """JAX whose low-face Bloch conjugate is cut out of the gradient chain."""

    def conj(self, x):
        return self.jax.lax.stop_gradient(self.xp.conj(x))


def test_mutant_a_torch_mask_backward_that_forgets_the_mask_is_caught(pec, monkeypatch):
    pytest.importorskip('torch')
    a = AMPLITUDES[1]

    def energy(t, backend='numpy'):
        return _energy(_with_amplitude(pec, t), backend)

    reference = _central_difference(energy, a, 1e-3 * a)
    monkeypatch.setitem(_runtime._RUNTIMES, 'torch', _ForgetfulMaskRuntime)
    wrong = _grad('torch', lambda t: energy(t, 'torch'), a)
    assert abs(wrong - reference) > MUTANT_FLOOR * abs(reference)      # measured: 4x off


def test_mutant_a_jax_face_factor_cut_from_the_chain_is_caught(bloch, monkeypatch):
    pytest.importorskip('jax')
    phi = PHASES[1]

    def energy(p, backend='numpy'):
        return _energy(_with_phase(bloch, p), backend)

    reference = _central_difference(energy, phi, 1e-4 * phi)
    monkeypatch.setitem(_runtime._RUNTIMES, 'jax', _HalfBlindRuntime)
    wrong = _grad('jax', lambda p: energy(p, 'jax'), phi)
    assert abs(wrong - reference) > MUTANT_FLOOR * abs(reference)      # measured: half


# ---------------------------------------------------------------------------------------------
# A static value is not a leaf
# ---------------------------------------------------------------------------------------------

#: The static doors: ``name -> (the scene with the value threaded in, the value it has)``.
STATIC = {
    'steps': (lambda s, v: s.replace(run=fd.Run(steps=v)), lambda s: float(s.run.steps)),
    'dx': (lambda s, v: s.replace(resolution=fd.Resolution(dx=v)), lambda s: s.resolution.dx),
    'size': (lambda s, v: s.replace(domain=fd.Domain(size=(v, 1.0, 1.0))), lambda s: s.domain.size[0]),
    'courant': (lambda s, v: s.replace(run=fd.Run(steps=s.run.steps, courant=v)),
                lambda s: s.run.courant),
    'at': (lambda s, v: s.replace(monitors=(fd.Record('Ey', at=(v, 0.0, -0.05)),)),
           lambda s: s.monitors[0].at[0]),
}


@pytest.mark.parametrize('backend', _ad_backends())
@pytest.mark.parametrize('name', sorted(STATIC))
def test_a_gradient_with_respect_to_a_static_value_is_refused_by_name(pec, backend, name):
    thread, value_of = STATIC[name]

    def energy(v):
        return _energy(thread(pec, v), backend)

    with pytest.raises(ValueError, match=rf'\b{name}\b') as e:
        _grad(backend, energy, value_of(pec))
    assert 'is static' in str(e.value) and 'InitialField.amplitude' in str(e.value)


def test_a_torch_leaf_in_a_static_door_is_refused_and_not_detached(pec, recwarn):
    """The silent case: ``float()`` reads a ``requires_grad`` tensor with only a torch warning."""
    torch = pytest.importorskip('torch')
    v = torch.tensor(pec.resolution.dx, dtype=torch.float64, requires_grad=True)
    for name, (thread, _) in STATIC.items():
        with pytest.raises(ValueError, match=rf'\b{name}\b'):
            thread(pec, v)
    assert not [w for w in recwarn if 'requires_grad' in str(w.message)]


def test_the_traced_leaves_leave_the_skeleton_alone(pec, bloch):
    """The two leaves this gate differentiates are not part of the plan; ``steps`` is."""
    assert fd.plan(_with_amplitude(pec, 2.0)).skeleton == fd.plan(pec).skeleton
    assert fd.plan(_with_phase(bloch, PHASES[2])).skeleton == fd.plan(bloch).skeleton
    assert fd.plan(pec.replace(run=fd.Run(steps=41))).skeleton != fd.plan(pec).skeleton
