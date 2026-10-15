"""Translate normalized engine diagnostics to the physical public vocabulary.

Public check/prepare/run calls scope this translation with a context variable;
internal numerical instruments retain the engine's original diagnostic text.
The engine also writes some calls without the ``fd.`` prefix -- ``Run(steps=...)``,
``Structure(junction=...)``, ``Band(df=...)`` -- and those are spelled by the same rules,
last in :data:`PHRASES`. ``tests/fdtd/test_public_spelling.py`` reads every string constant
in the engine and requires each ``fd.`` name and each such call to come out spelled here.

Five public names are the engine's own classes -- ``Domain``, ``Slab``, ``Region``, ``Plane``
and ``Record`` -- and their constructors refuse before any entry point runs, so each public
name is a :class:`Spoken` subclass that translates its own refusals.
"""
import contextlib
import contextvars
import re

from wavebit._common.backend import eager_under_torch_compile
from wavebit._common.declarations import TRACED_LEAVES

_PUBLIC = contextvars.ContextVar('wavebit_fdtd_public_spelling', default=False)

#: The engine's list of traced leaves, as every static refusal ends.
_COMPAT_LEAVES = 'The traced leaves are ' + '; '.join(TRACED_LEAVES) + '.'

#: The public module's own, which is not a respelling of the engine's: a ``Prepared`` holds the
#: boundaries, a Dipole's centre and the initial fields static (``_api._signature``), and its
#: contract's ``live`` entries are what remains.
PUBLIC_LEAVES = (
    'The traced leaves of a public problem are material values (a permittivity, a '
    "conductivity, a pole strength); each source's amplitude (a Dipole's current_moment), "
    'phase and delay, and its GaussianPulse(wavelength=..., frequency_sigma_hz=...); '
    'InitialField.amplitude; and the geometry of a Sphere, Cylinder, Box or Prism that '
    'Numerics(geometry_ranges=...) names -- radius, size, height, vertices and center within '
    'that range, orientation and vertex count static, one body or several whose full '
    'interpolation halos are disjoint -- prepared from concrete values.')

#: An engine shorthand, not a qualified path or ``fdtd.``.
_FD = r'(?<![\w.])fd\.'

#: The start of a name with no prefix at all.
_U = r'(?<![\w.])'

#: Names ``wavebit.fdtd`` exports with the meaning the compat name has.
_SHARED = ('Slab', 'Region', 'Plane', 'Box', 'Sphere', 'Cylinder', 'Prism', 'PlaneWave',
           'Dipole', 'TFSF', 'Record', 'Flux', 'Field', 'FarField', 'Diffraction', 'Bloch',
           'Boundaries', 'Periodic', 'PEC', 'PMC', 'Symmetry', 'Axis1D', 'InitialField',
           'Domain', 'Shape', 'Structure', 'PML', 'Absorber', 'Decay')

#: Stage-7 placeholders have never implemented a calculation.
_UNIMPLEMENTED = ('ModeSource', 'ModeMonitor', 'GaussianBeam')

PHRASES = (
    # A static refusal ends with the engine's leaves; the public module has its own.
    (re.escape(_COMPAT_LEAVES), PUBLIC_LEAVES),
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
    (r'requires plan= built by fd\.plan\(scene\) from concrete geometry before differentiation',
     'requires Prepared.run of what fdtd.prepare(problem, numerics=...) made from concrete '
     'geometry before differentiation'),
    (r'Build the plan with fd\.plan\(scene\) from the same scene',
     'Prepare it with fdtd.prepare(problem, numerics=...) from the same problem'),
    (r'(?<![\w.])plan= was not built with', 'the preparation was not made with'),
    (r'(?<![\w.])(?:lower|run)\(scene, plan=\.\.\.\)', 'Prepared.run(problem)'),
    (r', or pass plan=None\.', '.'),
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
    # The engine's calls without the prefix. A keyword the public spelling takes at the same
    # value keeps the value; one the engine holds in its own units (an end time, a bandwidth)
    # is spelled without the number; the compat namespace is retired, so every call is named
    # by its public counterpart.
    (_U + r'Run\(steps=\), Run\(periods=\), Run\(time=\)',
     'Duration(steps=), Duration(periods=, source=), Duration(seconds=)'),
    (_U + r"Run\(time=.*?\): a positive end time, in the scene's length unit \(c = 1\)\.",
     'Duration(seconds=...) or Duration(periods=..., source=...): a positive length of time.'),
    (_U + r'Run\(time=(?=\.\.\.\)|\))', 'Duration(seconds='),
    (_U + r'Run\(periods', 'Duration(periods'),
    (_U + r'Run\(steps=', 'Duration(steps='),
    (_U + r'Run\(courant=', 'Numerics(courant='),
    (_U + r'Run\(', 'fdtd.Duration('),
    (_U + r'Resolution\(dx=(.*?)\): dx is a positive length\.',
     r'UniformGrid(spacing=\1): the spacing is a positive length.'),
    (_U + r'Resolution\(dx=', 'UniformGrid(spacing='),
    (r'no Resolution\(wavelength=\) and no PML carrying one',
     'no source, and no fdtd.PML(wavelength=...) as Numerics(absorber=...) on an Open face'),
    (_U + r'Resolution\(', 'fdtd.UniformGrid('),
    (_U + r'Band\(wavelength=(.*?), df=.*?\): the spectrum reaches zero frequency -- f0 = .*? is '
     r'not above (.*?) \* df = .*?, where it falls to the cutoff (.*?)\. A pulse that wide is not '
     r'a carrier; narrow df below .*?\.(?=\s|$)',
     r'GaussianPulse(wavelength=\1, frequency_sigma_hz=...): the spectrum reaches zero frequency '
     r'-- the centre frequency is not above \2 standard deviations, where it falls to the cutoff '
     r'\3. A pulse that wide is not a carrier; narrow frequency_sigma_hz below 1/\2 of the '
     r'centre frequency.'),
    (_U + r'Band\(df=.*?\): ', 'GaussianPulse(frequency_sigma_hz=...): '),
    (_U + r'Band\(wavelength=\.\.\., df=\.\.\.\)',
     'GaussianPulse(wavelength=..., frequency_sigma_hz=...)'),
    (_U + r'Band\((wavelength|cutoff)=', r'GaussianPulse(\1='),
    (_U + r'Band\(', 'fdtd.GaussianPulse('),
    (_U + r'(Decay|Absorber)\(', r'fdtd.\1('),
    # A structure's junction rule and discretisation are the scene-wide Numerics ones.
    (_U + r'Structure\((junction|discretisation)=', r'Numerics(\1='),
    (_U + r'Structure\(memory_bound=', 'fdtd.Structure(memory_bound='),
    # A body's range is its entry in Numerics(geometry_ranges=...).
    (r'Sphere/Cylinder/Box/Prism\(range=\) declarations differ from the plan',
     "Numerics(geometry_ranges=...) entries differ from the preparation's"),
    (_U + r'(Sphere|Cylinder|Box|Prism)\(range=(?:\.\.\.)?\)',
     r'the \1 range in Numerics(geometry_ranges=...)'),
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


@eager_under_torch_compile
class Spoken:
    """The public name of one of the engine's own declarations, refusing in the public vocabulary.

    ``fdtd.Slab(x=..., z=...)`` was refused, before any entry point ran, with "a rectangular
    rod or block is fd.Structure(fd.Region(size=..., center=...), material)", and a traced
    length on any of the five listed the engine's leaves, a Dipole's centre and the Bloch phase
    among them. A subclass adds no field and no behaviour -- a plan encodes a declaration by
    its class name and fields, and a job by its public name, so neither changes -- and
    translates what the constructor raises, as the entry points translate what leaves them.
    """

    def __init__(self, *args, **kwargs):
        try:
            super().__init__(*args, **kwargs)
        except (TypeError, ValueError, NotImplementedError) as exc:
            translated = public_error(exc)
            if translated is exc:
                raise
            raise translated from exc
