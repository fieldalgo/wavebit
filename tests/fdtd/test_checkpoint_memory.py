"""Checkpointed reverse mode on JAX and Torch CPU: same gradients, memory bounded by the segment.

Acceptance, registered before the first measurement (docs/validation/fdtd/checkpointing.md):

- a checkpointed gradient matches the ordinary scan's to 1e-12 relative (float64), and the
  central difference of a NumPy forward run;
- peak gradient memory is set by the segment length and not by the step count: eight times
  more steps cost less than twice the memory with ``checkpoint_steps``, and at least four
  times as much without it (the control that shows the measurement can fail).

The schedule is nested (``wavebit.fdtd._runtime._segment``); the block lengths below cover a
single step per segment, one level with a remainder, three levels with a nested remainder,
and a block longer than the run.
"""
import json
import os
import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pytest

from wavebit import Material, Units, fdtd

STEPS = 800
BLOCKS = (1, 7, 30, 1000)
EPSILON, AMPLITUDE = 2.25, 1.2


def problem(epsilon=EPSILON, amplitude=AMPLITUDE, width=.05, sigma=4e13):
    return fdtd.Problem(units=Units(length='um'), scene=fdtd.Scene(
        domain=fdtd.Domain(size=(width, width, 2.4)), background=Material(epsilon=1.),
        boundaries=fdtd.Boundaries.periodic(z=fdtd.Open()),
        structures=(fdtd.Structure(name='film', shape=fdtd.SnappedSlab(z=(-.15, .15)),
                                   material=Material(epsilon=epsilon)),)),
        sources=(fdtd.PlaneWave('pw', plane=fdtd.Plane(z=.65), amplitude=amplitude,
            pulse=fdtd.GaussianPulse(wavelength=.8, frequency_sigma_hz=sigma)),),
        outputs=fdtd.Outputs(monitors=(
            fdtd.TimeSeriesMonitor('e', at=(0., 0., -.5), component='Ey'),
            fdtd.FieldMonitor('field', plane=fdtd.Plane(z=-.5), wavelengths=(.8,),
                              components=('Ey', 'Hx')),
            fdtd.FluxMonitor('flux', plane=fdtd.Plane(z=-.5), wavelengths=(.8,))),
            final_fields=True))


def numerics(block=None, steps=STEPS, spacing=.025):
    return fdtd.Numerics(grid=fdtd.UniformGrid(spacing=spacing), duration=fdtd.Duration(steps=steps),
                         checkpoint_steps=block)


def loss_of(result, xp):
    """Every kind of output the run returns: a series, a spectral plane, a flux, the fields."""
    return (xp.sum(xp.abs(result['field'].values['Ey'])**2) + xp.sum(result['e'].values**2)
            + xp.sum(result['flux'].spectral_energy) + xp.sum(result.final_fields.values['E']**2))


def torch_gradient(block):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(2)
    eps = torch.tensor(EPSILON, dtype=torch.float64, requires_grad=True)
    amp = torch.tensor(AMPLITUDE, dtype=torch.float64, requires_grad=True)
    result = fdtd.run(problem(eps, amp), numerics=numerics(block), backend='torch')
    assert result.provenance['checkpoint_steps'] == block
    loss = loss_of(result, torch)
    ge, ga = torch.autograd.grad(loss, (eps, amp))
    return float(loss), float(ge), float(ga)


def jax_gradient(block):
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    prepared = fdtd.prepare(problem(), numerics=numerics(block), backend='jax')
    assert prepared.contract['resolved_checkpoint_steps'] == block

    def loss(eps, amp):
        return loss_of(prepared.run(problem(eps, amp)), jax.numpy)
    value, (ge, ga) = jax.value_and_grad(loss, argnums=(0, 1))(EPSILON, AMPLITUDE)
    return float(value), float(ge), float(ga)


GRADIENT = {'torch': torch_gradient, 'jax': jax_gradient}
_PLAIN = {}


def plain(backend):
    if backend not in _PLAIN:
        _PLAIN[backend] = GRADIENT[backend](None)
    return _PLAIN[backend]


def numpy_loss(epsilon, amplitude=AMPLITUDE):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return float(loss_of(fdtd.run(problem(epsilon, amplitude), numerics=numerics()), np))


@pytest.mark.parametrize('block', BLOCKS)
@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_checkpointed_gradient_matches_the_ordinary_scan(backend, block):
    value, ge, ga = GRADIENT[backend](block)
    base = plain(backend)
    assert all(np.isfinite(base)) and all(abs(x) > 1e-6 for x in base)
    np.testing.assert_allclose((value, ge, ga), base, rtol=1e-12, atol=0.)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_checkpointed_gradient_matches_central_differences(backend):
    value, ge, ga = GRADIENT[backend](7)
    np.testing.assert_allclose(value, numpy_loss(EPSILON), rtol=1e-12, atol=0.)
    h = EPSILON * 1e-5
    expected = (numpy_loss(EPSILON + h) - numpy_loss(EPSILON - h)) / (2 * h)
    np.testing.assert_allclose(ge, expected, rtol=1e-6, atol=0.)
    # Every term of the loss, the flux included, is quadratic in the source amplitude.
    np.testing.assert_allclose(ga, 2 * value / AMPLITUDE, rtol=1e-12, atol=0.)


def test_a_checkpointed_torch_scan_refuses_parameters_it_cannot_see():
    torch = pytest.importorskip('torch')
    from wavebit.fdtd._runtime import runtime
    rt = runtime('torch')
    x = torch.tensor(1., dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValueError, match='params='):
        rt.scan(lambda c, n: c * x, torch.ones(()), 10, 3)


# -- memory ---------------------------------------------------------------------------------

# The memory scene is 10 x 10 x 48 cells: wide enough that the fields, and not the program,
# set the memory. Its narrower pulse is what the coarser grid admits.

def jax_temp_bytes(steps, block):
    """XLA's scratch allocation for the whole jitted gradient: a property of the program."""
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        prepared = fdtd.prepare(problem(width=.5, sigma=2e13), numerics=numerics(block, steps, .05),
                                backend='jax')

        def loss(eps):
            return loss_of(prepared.run(problem(eps, width=.5, sigma=2e13)), jax.numpy)
        compiled = jax.jit(jax.grad(loss)).lower(EPSILON).compile()
    analysis = compiled.memory_analysis()
    if analysis is None:
        pytest.skip('this JAX build reports no memory analysis')
    return analysis.temp_size_in_bytes


def test_jax_gradient_memory_is_set_by_the_segment_not_the_step_count():
    short, long = 128, 1024
    checkpointed = jax_temp_bytes(long, 8) / jax_temp_bytes(short, 8)
    ordinary = jax_temp_bytes(long, None) / jax_temp_bytes(short, None)
    assert ordinary >= 4, ordinary
    assert checkpointed < 2, checkpointed


# Torch is measured as the rise of resident memory during one gradient, in a fresh process. The
# grid is 20 x 20 x 48 so the gradient, and not the imports before it, sets the peak.
#
# Linux needs two things Windows does not. ``ru_maxrss`` is the peak over the whole process, so
# a gradient smaller than the imports or the first forward run reads as a rise of zero (it did
# on the CI runner). And glibc keeps freed heap pages resident and hands them back out, so a
# gradient can reuse memory that is already counted. The window is therefore reset just before
# the gradient (``/proc/self/clear_refs`` = 5 resets ``VmHWM``, Linux >= 4.0), after
# ``malloc_trim`` and with a low mmap threshold so that large tensors are mapped and unmapped
# rather than recycled. Windows trims a process's working set on its own.
_TORCH_PEAK = textwrap.dedent('''
    import json, sys, warnings
    warnings.simplefilter('ignore')
    import torch
    torch.set_num_threads(1)
    from tests.fdtd.test_checkpoint_memory import EPSILON, loss_of, numerics, peak_bytes, problem, reset_peak
    from wavebit import fdtd
    steps, block = int(sys.argv[1]), (None if sys.argv[2] == 'none' else int(sys.argv[2]))
    prepared = fdtd.prepare(problem(width=1., sigma=2e13), numerics=numerics(block, steps, .05), backend='torch')
    prepared.run(problem(width=1., sigma=2e13))
    before = reset_peak()
    eps = torch.tensor(EPSILON, dtype=torch.float64, requires_grad=True)
    loss = loss_of(prepared.run(problem(eps, width=1., sigma=2e13)), torch)
    torch.autograd.grad(loss, eps)
    print(json.dumps(peak_bytes() - before))
''')


def _proc_status(field):
    """A ``kB`` field of ``/proc/self/status`` (Linux), in bytes."""
    with open('/proc/self/status') as status:
        for line in status:
            if line.startswith(field + ':'):
                return int(line.split()[1]) * 1024
    raise LookupError(field)


def peak_bytes():
    """The process's peak resident memory since the last ``reset_peak``, in bytes."""
    if sys.platform.startswith('linux'):
        return _proc_status('VmHWM')
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                    'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                    'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        query = kernel32.K32GetProcessMemoryInfo
        query.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
        query.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not query(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.PeakWorkingSetSize
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS


def reset_peak():
    """Start a new peak window where the platform allows it; return the memory it starts from.

    On Linux the window is reset and starts from the current resident size. Elsewhere the peak
    cannot be reset, and the window starts from the peak so far.
    """
    if sys.platform.startswith('linux'):
        import ctypes
        ctypes.CDLL('libc.so.6').malloc_trim(0)
        with open('/proc/self/clear_refs', 'w') as refs:
            refs.write('5')
        return _proc_status('VmRSS')
    return peak_bytes()


def torch_peak_rise(steps, block):
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = dict(os.environ, MALLOC_MMAP_THRESHOLD_='65536', PYTHONPATH=os.pathsep.join(
        [root] + [p for p in (os.environ.get('PYTHONPATH'),) if p]))
    out = subprocess.run([sys.executable, '-c', _TORCH_PEAK, str(steps), str(block).lower()],
                         cwd=root, env=env, capture_output=True, text=True, check=True)
    rise = json.loads(out.stdout.strip().splitlines()[-1])
    # A gradient always allocates; a zero is a blind instrument, not a small gradient.
    assert rise > 0, f'the peak-memory reading saw no gradient ({steps} steps, block {block})'
    return rise


@pytest.mark.slow
def test_torch_gradient_memory_is_set_by_the_segment_not_the_step_count():
    pytest.importorskip('torch')
    short, long = 128, 1024
    checkpointed = torch_peak_rise(long, 8) / torch_peak_rise(short, 8)
    ordinary = torch_peak_rise(long, None) / torch_peak_rise(short, None)
    assert ordinary >= 4, ordinary
    assert checkpointed < 2, checkpointed
