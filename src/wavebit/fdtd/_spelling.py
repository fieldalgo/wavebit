"""Translate normalized engine diagnostics to the physical public vocabulary.

Public check/prepare/run calls scope this translation with a context variable;
internal numerical instruments retain the engine's original diagnostic text.
"""
import contextlib
import contextvars
import re

_PUBLIC = contextvars.ContextVar('wavebit_fdtd_public_spelling', default=False)

#: An engine shorthand, not a qualified path or ``fdtd.``.
_FD = r'(?<![\w.])fd\.'

#: Names ``wavebit.fdtd`` exports with the meaning the compat name has.
_SHARED = ('Slab', 'Region', 'Plane', 'Box', 'Sphere', 'Cylinder', 'Prism', 'PlaneWave',
           'Dipole', 'TFSF', 'Record', 'Flux', 'Field', 'FarField', 'Diffraction', 'Bloch',
           'Boundaries', 'Periodic', 'PEC', 'PMC', 'Symmetry', 'Axis1D', 'InitialField',
           'Domain', 'Shape', 'Structure', 'PML', 'Absorber', 'Decay')

#: Stage-7 placeholders have never implemented a calculation.
_UNIMPLEMENTED = ('ModeSource', 'ModeMonitor', 'GaussianBeam')

PHRASES = (
    (r'Run\(decay=Decay\(threshold=\.\.\., window=\.\.\.\)\)',
     'Numerics(decay=fdtd.Decay(threshold=..., window_seconds=...))'),
    (r'(?<![\w.])(?:fd\.)?Decay\(threshold=\.\.\., window=\.\.\.\)',
     'fdtd.Decay(threshold=..., window_seconds=...)'),
    (r'(?<![\w.])Decay\(window=', 'fdtd.Decay(window_seconds='),
    # Bloch faces from a plane wave: the public call takes the source and the units.
    (_FD + r'Bloch\.from_source\(pw', 'fdtd.Bloch.from_source(source, units=problem.units'),
    # Absorbing faces: the public face is Open, its absorber a PML in Numerics.
    (_FD + r'Boundaries\.open\(fd\.PML\(cells=12, wavelength=\.\.\.\)\)',
     'fdtd.Boundaries.open() with Numerics(absorber=fdtd.PML(cells=12, wavelength=...))'),
    (_FD + r'Boundaries\(z=fd\.PML\(\.\.\.\), \.\.\.\)', 'fdtd.Boundaries(z=fdtd.Open(), ...)'),
    (_FD + r'PML(?:\(\.\.\.\))? or fd\.Absorber(?:\(\.\.\.\))?', 'fdtd.Open()'),
    (r'Say fd\.PML\(cells=\.\.\., wavelength=\.\.\.\), or give the whole scene one with '
     r'fd\.Resolution\(cells_per_wavelength=\.\.\., wavelength=\.\.\.\)\.',
     'Say fdtd.PML(cells=..., wavelength=...) as Numerics(absorber=...).'),
    # The scene-wide wavelength has no public spelling; the pulse's does.
    (r'Give the scene fd\.Resolution\(wavelength=\.\.\.\), or a concrete Band\(wavelength=\.\.\.\)\.',
     'Give the source a concrete fdtd.GaussianPulse(wavelength=...).'),
    # Grid, duration and pulse live in Numerics and GaussianPulse.
    (_FD + r'Resolution\(dx=\.\.\.\)', 'fdtd.UniformGrid(spacing=...)'),
    (_FD + r'Run\(courant=\.\.\.\)', 'fdtd.Numerics(courant=...)'),
    (_FD + r'Band\(wavelength=\.\.\., df=\.\.\.\)',
     'fdtd.GaussianPulse(wavelength=..., frequency_sigma_hz=...)'),
    (_FD + r'Band\b', 'fdtd.GaussianPulse'),
    (_FD + r'Resolution\b', 'fdtd.UniformGrid'),
    (_FD + r'Run\b', 'fdtd.Duration'),
    # Materials are the shared models, spelled with epsilon -- and an article that still fits.
    (r'\ban fd\.Material\b', 'a Material'),
    (r'\ban fd\.Conductor\b', 'a ConductiveMaterial'),
    (r'\ban fd\.Dispersive\b', 'a DispersiveMaterial'),
    (_FD + r'Material\(eps=', 'Material(epsilon='),
    (_FD + r'Material\b', 'Material'),
    (_FD + r'Conductor\b', 'ConductiveMaterial'),
    (_FD + r'Dispersive\b', 'DispersiveMaterial'),
    # A public structure is named, and the stencil's memory bound is a compat keyword.
    (_FD + r'Structure\(\.\.\., discretisation=', 'fdtd.Numerics(discretisation='),
    (_FD + r'Structure\(fd\.', 'fdtd.Structure(name, fd.'),
    (_FD + r'Structure\(shape, ', 'fdtd.Structure(name, shape, '),
    # The lifecycle: a plan is a preparation, lowering a traced scene is Prepared.run.
    (r'build the plan before differentiation and pass plan= to fd\.run or fd\.lower',
     'prepare before differentiation and differentiate Prepared.run'),
    (r'Build the plan from concrete values and pass it as plan= to fd\.run or fd\.lower with the '
     r'traced ones', 'Prepare from concrete values and run the traced problem with Prepared.run'),
    (r'p = fd\.plan\(scene\) from plain numbers, then fd\.lower\(traced_scene, p, backend\)',
     'prepared = fdtd.prepare(problem, numerics=...) from plain numbers, then '
     'prepared.run(traced_problem)'),
    (_FD + r'plan\(scene\)', 'fdtd.prepare(problem, numerics=...)'),
    (_FD + r'run\(scene\)', 'fdtd.run(problem, numerics=...)'),
    (_FD + r'plan\b', 'fdtd.prepare'),
    (_FD + r'lower\b', 'Prepared.run'),
    (_FD + r'run\b', 'fdtd.run'),
    (_FD + r'check\b', 'fdtd.check'),
    # CSG is named as a spelling to come; it is in neither namespace yet.
    (_FD + r'(Union|Intersection|Difference)\b', r'\1'),
    (_FD + r'(' + '|'.join(_SHARED) + r')\b', r'fdtd.\1'),
    (_FD + r'(' + '|'.join(_UNIMPLEMENTED) + r')\b', r'\1 (unimplemented)'),
)

_COMPILED = tuple((re.compile(pattern), replacement) for pattern, replacement in PHRASES)


def to_public(text):
    """``text`` with every compat spelling :data:`PHRASES` knows replaced by the public one."""
    for pattern, replacement in _COMPILED:
        text = pattern.sub(replacement, text)
    return text


def spoken(text):
    """``text`` in the public vocabulary inside a public call, unchanged anywhere else."""
    return to_public(text) if _PUBLIC.get() else text


@contextlib.contextmanager
def public():
    """Speak the public vocabulary for the duration of one public call."""
    token = _PUBLIC.set(True)
    try:
        yield
    finally:
        _PUBLIC.reset(token)


def public_error(exc):
    """``exc`` with its message in the public vocabulary, or ``exc`` itself if nothing changes.

    A structured error keeps its type, code, path and values. A built-in ``TypeError``,
    ``ValueError``, ``NotImplementedError`` or ``RuntimeError`` keeps its type; any other
    exception is passed through untouched, since its constructor is its own.
    """
    from wavebit.errors import ValidationError
    if isinstance(exc, ValidationError):
        message = to_public(exc.message)
        if message == exc.message:
            return exc
        return type(exc)(message, code=exc.code, path=exc.path, expected=exc.expected,
                         actual=exc.actual)
    if type(exc) in (TypeError, ValueError, NotImplementedError, RuntimeError) and len(exc.args) == 1:
        message = to_public(str(exc))
        return exc if message == str(exc) else type(exc)(message)
    return exc
