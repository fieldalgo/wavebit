"""Translate normalized engine diagnostics to the physical public vocabulary.

Public check/prepare/run calls scope this translation with a context variable;
internal numerical instruments retain the engine's original diagnostic text.
The engine also writes some calls without the ``fd.`` prefix -- ``Run(steps=...)``,
``Structure(junction=...)``, ``Band(df=...)`` -- and those are spelled by the same rules,
last in :data:`PHRASES`. ``tests/fdtd/test_public_spelling.py`` reads every string constant
in the engine and requires each ``fd.`` name and each such call to come out spelled here.

Five public classes extend the engine's own -- ``Domain``, ``Plane``, ``SnappedSlab`` (the
engine's ``Slab``), ``SnappedBox`` (``Region``) and ``TimeSeriesMonitor`` (``Record``) -- and
their constructors refuse before any entry point runs, so each is a :class:`Spoken` subclass
that translates its own refusals. Engine messages that name a declaration the public API renamed
are written in the public spelling where they are raised (``_public_names``); this module keeps
only the translations for names both vocabularies share.
"""
import contextlib
import contextvars
import inspect
import math
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
    "conductivity, a pole strength); each source's amplitude (a CurrentDipole's current_moment), "
    'phase and delay_seconds, and its GaussianPulse(wavelength=..., frequency_sigma_hz=...); '
    'InitialField.amplitude; and the geometry of a Sphere, Cylinder, Box or Prism that '
    'Numerics(geometry_ranges=...) names -- radius, size, height, vertices and center within '
    'that range, orientation and vertex count static, one body or several whose full '
    'interpolation halos are disjoint -- prepared from concrete values.')

#: An engine shorthand, not a qualified path or ``fdtd.``.
_FD = r'(?<![\w.])fd\.'

#: The start of a name with no prefix at all.
_U = r'(?<![\w.])'

#: Names ``wavebit.fdtd`` exports with the meaning, and the spelling, the engine name has. The
#: names the public API renamed (``_public_names.PUBLIC_NAMES``) are not here: the engine's
#: messages spell those publicly where they are raised.
_SHARED = ('Plane', 'Box', 'Sphere', 'Cylinder', 'Prism', 'PlaneWave', 'TFSF', 'Bloch',
           'Boundaries', 'Periodic', 'PEC', 'PMC', 'InitialField', 'Domain', 'Shape', 'Structure',
           'PML', 'ModeMonitor', 'ModeSource', 'GaussianBeam')

#: The ConductiveAbsorber's coupling to a smoothed structure, as the public caller meets it. The
#: engine names its own scopes (``ConstitutiveAbsorberScope: ...``), which a public caller cannot
#: act on; each refusal is said here with the public names, and with advice that runs: moved three
#: grid cells clear of the Open faces, a smoothed body's support stays out of the layer (two
#: cells were not always enough, by rounding, in tests/api/test_check_run_consistency.py).
_ABSORBER = ("A smoothed structure's support reaches into the ConductiveAbsorber "
             '(Numerics(absorber=fdtd.ConductiveAbsorber(...)))')
_CLEAR = ('move the smoothed structures (Sphere, Cylinder, Box, Prism) at least three grid cells, '
          '3*UniformGrid(spacing), clear of every Open face, so their support stays out of the '
          'absorbing layer')
_ABSORBER_PHRASES = (
    (r'ConstitutiveAbsorberScope: initial fields are not admitted\.',
     f'{_ABSORBER}, where Problem(initial=...) is not supported. Excite the field with a source '
     'instead of an fdtd.InitialField.'),
    (r'ConstitutiveAbsorberScope: automatic stopping needs a separate state mapping; use a fixed '
     r'duration for this new term\.',
     f'{_ABSORBER}, where Numerics(stopping=fdtd.DecayStop(...)) is not supported. Remove stopping '
     f'and run the fixed Numerics(duration=...), or {_CLEAR}.'),
    (r'ConstitutiveAbsorberConcrete: this first route needs concrete inputs\.',
     f'{_ABSORBER}, where material values and geometry must be plain numbers, not traced or '
     f'differentiated ones. To differentiate them, {_CLEAR}, then prepare from plain numbers '
     'and run the traced problem with Prepared.run.'),
    (r'ConstitutiveAbsorberConcrete: finite concrete specification required\.',
     f'{_ABSORBER}, where every declared value must be a finite plain number. Give finite plain '
     f'numbers, or {_CLEAR}.'),
    (r'ConstitutiveAbsorberConcrete: link certification requires concrete CPU coefficients, not an '
     r'outer transformed solve\.',
     f'{_ABSORBER}, which cannot run inside an outer jax.jit, jax.vmap or torch.func transform. '
     f'Run it outside the transform, or {_CLEAR}.'),
    (r'ConstitutiveAbsorberScope: only plain lossless Material is admitted\.',
     f'{_ABSORBER}, where every material (the background and each structure) must be a plain '
     'lossless Material, not a ConductiveMaterial, DispersiveMaterial or tabulated material. Use '
     f'such materials, or {_CLEAR}.'),
    (r'ConstitutiveAbsorberScope: positive isotropic eps and mu=1 required\.',
     f'{_ABSORBER}, where every material must be isotropic, with one positive real epsilon and '
     f'no mu (mu = 1). Use such materials, or {_CLEAR}.'),
    (r'ConstitutiveAbsorberScope: only electric CurrentDipoles are admitted\.',
     f"{_ABSORBER}, where every source must be an electric fdtd.CurrentDipole (kind='electric'): "
     'no magnetic CurrentDipole, PlaneWave, TFSF, GaussianBeam or ModeSource. Use only electric '
     f'CurrentDipoles, or {_CLEAR}.'),
    (r'ConstitutiveAbsorberScope: CPML, mirrors and reduced axes are outside this first absorber '
     r'route\.',
     f'{_ABSORBER}, where every face that is not Open must be PEC, PMC, Periodic or Bloch: '
     f'MirrorSymmetry and InvariantAxis are not supported. Use those faces, or {_CLEAR}.'),
    (r'ConstitutiveAbsorberScope: one matched absorbing axis and an unchanged absorbing layout are '
     r'required\.',
     f'{_ABSORBER}, where the Open faces must all lie on one axis. Open the faces of one axis '
     f'only, or {_CLEAR}.'),
    (r'ConstitutiveAbsorberScope: ordinary concrete voxel shapes required\.',
     f"{_ABSORBER}, where every structure must be a built-in shape under "
     "Numerics(discretization='voxel') and not named in Numerics(geometry_ranges=...). Use "
     f'those, or {_CLEAR}.'),
    (r'ConstitutiveAbsorberScope: (?:only electric coupling is|magnetic coupling is not) '
     r'admitted\.',
     f'{_ABSORBER}, where only its electric-field smoothing is supported, not a magnetic one '
     f'(mu != 1). Give it no mu, or {_CLEAR}.'),
    (r'ConstitutiveAbsorberPlan: [^.]*\.',
     f'{_ABSORBER}, and this problem no longer matches the absorber coupling it was prepared '
     'with. Prepare it again with fdtd.prepare.'),
    (r'ConstitutiveAbsorberRuntime: [^.]*\.',
     f"{_ABSORBER}, which runs on the CPU only. Use device='cpu', or {_CLEAR}."),
    (r'ConstitutiveAbsorberRange: ([^.]*)\.',
     f'{_ABSORBER}, and the absorber coefficients there are out of range (\\1). Use plainer '
     f'material values, or {_CLEAR}.'),
    (r'ConstitutiveAbsorberCommutation: \S+ \S+ \w+ link at z/y/x=(\([^)]*\)) joins unequal '
     r'absorber rates\. Keep this interpolation support outside the layer, or use an aligned '
     r'extrusion\.',
     f'{_ABSORBER} where two different absorber rates meet (at cell z/y/x=\\1), which its '
     f'smoothing cannot cross. Use an extrusion aligned with the absorbing axis, or {_CLEAR}.'),
)

#: A traced CurrentDipole centre. The engine's advice, a ``range=`` box, has no public spelling:
#: a public CurrentDipole's position is static (:data:`PUBLIC_LEAVES`), and its source terms are
#: what differentiate.
_DIPOLE_PHRASES = (
    (r"CurrentDipole\(center=<traced>\): a position that is a gradient leaf needs "
     r"range=\(\(x0, x1\), \(y0, y1\), \(z0, z1\)\), the static box it may move in -- the stamp "
     r"writes a static set of cells, and without a box nothing says which\.",
     'CurrentDipole(center=...) is traced or differentiated, and a CurrentDipole\'s position is '
     'static: the cells its stamp writes are fixed when the problem is prepared. Give center '
     'plain numbers; a dipole differentiates through its current_moment, phase and '
     'delay_seconds, and through its GaussianPulse(wavelength=..., frequency_sigma_hz=...).'),
)

PHRASES = _ABSORBER_PHRASES + _DIPOLE_PHRASES + (
    # A static refusal ends with the engine's leaves; the public module has its own.
    (re.escape(_COMPAT_LEAVES), PUBLIC_LEAVES),
    # Bloch faces from a plane wave: the public call takes the source and the units.
    (_FD + r'Bloch\.from_source\(pw', 'fdtd.Bloch.from_source(source, units=problem.units'),
    # Absorbing faces: the public face is Open, its absorber a PML in Numerics.
    (_FD + r'Boundaries\.open\(fd\.PML\(cells=12, wavelength=\.\.\.\)\)',
     'fdtd.Boundaries.open() with Numerics(absorber=fdtd.PML(cells=12, wavelength=...))'),
    (_FD + r'Boundaries\(z=fd\.PML\(\.\.\.\), \.\.\.\)', 'fdtd.Boundaries(z=fdtd.Open(), ...)'),
    # A face's PML reference is the public PML's reference_media entry for that face.
    (_FD + r'PML\(background=\.\.\.\)',
     'fdtd.PML(reference_media=...) with an entry for every Open face'),
    (r'Say fd\.PML\(cells=\.\.\., wavelength=\.\.\.\), or give the whole scene one with '
     r'fd\.Resolution\(cells_per_wavelength=\.\.\., wavelength=\.\.\.\)\.',
     'Say fdtd.PML(cells=..., wavelength=...) as Numerics(absorber=...).'),
    # The scene-wide wavelength has no public spelling; the pulse's does.
    (r'Give the scene fd\.Resolution\(wavelength=\.\.\.\), or a concrete Band\(wavelength=\.\.\.\)\.',
     'Give the source a concrete fdtd.GaussianPulse(wavelength=...).'),
    # The resolution rules' own words: the grid's spacing is UniformGrid(spacing=...), and the
    # band a refusal reads is the sources' pulses. "Refine dx" told a public caller to change
    # a field it cannot name, with a constant of the legacy engine as the reason.
    (r"the legacy engine's CELLS_PER_WAVELENGTH", 'the eight-cells-per-wavelength rule'),
    (r", the sources' band\.shortest_wavelength",
     ", the shortest wavelength of the sources' pulses"),
    (r'a source band is traced, so the cells-per-wavelength check at '
     r'band\.shortest_wavelength did not run',
     'a source pulse is traced, so the cells-per-wavelength check at its shortest wavelength '
     'did not run'),
    (r'\b([Rr])efine dx\b', r'\1efine UniformGrid(spacing=...)'),
    (r', dx ([^,)\s]+)\)', r', spacing \1)'),
    (r'\bdx\*K=', 'spacing*K='),
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
    # Stencil preparation is a scene-wide public numerical policy.
    (r'bound this structure declared', 'bound declared in Numerics'),
    (r'raise the bound for this structure with', 'raise the preparation bound with'),
    (r"use discretisation='voxel' on structures\[\d+\] \(.*?\); for the stencil,",
     "use Numerics(discretization='voxel'); for the stencil,"),
    # A public structure is named, takes its shape and material by keyword, and its numerical
    # options live in Numerics.
    (_FD + r'Structure\(fdtd\.(\w+)(\(.*?\)), (\w+)\)', r'fdtd.Structure(name, shape=fdtd.\1\2, material=\3)'),
    (_FD + r'Structure\(shape, material\)', 'fdtd.Structure(name, shape=shape, material=material)'),
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
    # A structure's junction rule and memory bound are the scene-wide Numerics ones.
    (_U + r'Structure\(junction=', 'Numerics(junction='),
    (_U + r'Structure\(memory_bound=', 'Numerics(memory_bound='),
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


def is_public():
    """Whether diagnostics currently address a caller of the physical public API."""
    return _PUBLIC.get()


def spoken(text):
    """``text`` in the public vocabulary inside a public call, unchanged anywhere else."""
    return to_public(text) if _PUBLIC.get() else text


def angle(degrees, compat=None):
    """An angle the engine holds in degrees, as its caller gave it.

    A public ``PlaneWave`` takes ``theta`` and ``phi`` in radians and the lowering turns them
    into the engine's degrees, so a refusal quoting the engine's number told a caller who gave
    0.3 about 17.1887. Inside a public call this is the radians; elsewhere ``compat`` -- the
    compat caller's own spelling -- or the degrees as ``:g`` prints them.
    """
    if _PUBLIC.get():
        return f'{math.radians(degrees):g}'
    return f'{degrees:g}' if compat is None else compat


def angle_unit():
    """``'radians'`` inside a public call, ``'degrees'`` elsewhere: the unit :func:`angle` prints."""
    return 'radians' if _PUBLIC.get() else 'degrees'


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


def structured(exc, code, root):
    """A plain engine ``ValueError`` as a ``ValidationError``, its message kept; else ``exc``.

    ``code`` is the refusal's code, or a mapping from the refused field to its code (``None``
    keys the default). The field is the one keyword the refusal's opening call names, as in
    ``Domain(size=(-1.0, 1.0, 1.0)): every size is a positive length``; it is placed at
    ``root.<field>``, and a refusal naming none or several at ``root``. A traced value where a
    static one is required is ``prepared.static_input``, as at the entry points. A
    ``TypeError`` (an unknown or missing keyword) and a ``NotImplementedError`` keep their type.
    """
    from wavebit.errors import ValidationError
    if type(exc) is not ValueError:
        return exc
    message = str(exc)
    fields = re.findall(r'(?<![\w.])([a-z_]+)=', message.split(': ', 1)[0])
    field = fields[0] if len(fields) == 1 else None
    path = f'{root}.{field}' if field else root
    if isinstance(code, dict):
        code = code.get(field, code[None])
    if ') is static:' in message:
        code = 'prepared.static_input'
    return ValidationError(message, code=code, path=path)


@eager_under_torch_compile
class Spoken:
    """A public class extending one of the engine's own declarations, refusing in the public vocabulary.

    The engine's constructor refuses before any entry point runs, so the public subclass
    translates what that constructor raises, as the entry points translate what leaves them.
    A subclass adds no field and no behaviour. Where the engine records a declaration's class
    in a plan, it records the engine class this one extends (a junction's signature skips
    ``Spoken``), so the public name changes no skeleton; ``tests/fdtd/test_public_spelling.py``
    pins that.

    A plain ``ValueError`` from the engine constructor leaves as a ``ValidationError`` whose code
    and path :meth:`_refusal` names (:func:`structured`).
    """

    #: ``(code, path root)`` of a refusal; :func:`structured` reads a code mapping by field.
    _REFUSAL = ('fdtd.invalid', '')

    def _refusal(self):
        return self._REFUSAL

    def __init_subclass__(cls, **kwargs):
        # ``__init__`` below takes ``*args, **kwargs``; ``inspect`` and the API reference read
        # this instead, the signature of the engine constructor it forwards to.
        super().__init_subclass__(**kwargs)
        engine = next(c for c in cls.__mro__
                      if not issubclass(c, Spoken) and '__init__' in vars(c))
        signature = inspect.signature(engine.__init__)
        cls.__signature__ = signature.replace(parameters=list(signature.parameters.values())[1:])

    def __init__(self, *args, **kwargs):
        try:
            super().__init__(*args, **kwargs)
        except (TypeError, ValueError, NotImplementedError) as exc:
            translated = structured(public_error(exc), *self._refusal())
            if translated is exc:
                raise
            raise translated from exc
