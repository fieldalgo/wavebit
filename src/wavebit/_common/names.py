"""Physical-quantity-aware spelling hints."""
import re

_NVF_OPTION_RENAMES = {
    'normal_method': 'raster_normal_method',
    'normal_grid': 'shape_nvf_grid_size',
    'projector': 'nvf_projector',
}


def nvf_option_rename_hint(names):
    """Explain the pre-release NVF rename without accepting an obsolete spelling."""
    renamed = [f'{name!r} -> {_NVF_OPTION_RENAMES[name]!r}'
               for name in sorted(names) if name in _NVF_OPTION_RENAMES]
    return (' Renamed NVF option(s): ' + ', '.join(renamed) + '.') if renamed else ''


_MATERIAL_SLOTS = frozenset({'top', 'bot', 'structure', 'background', 'material'})


_SLOT_WORDS = _MATERIAL_SLOTS | frozenset({
    'bottom', 'sub', 'substrate', 'sup', 'super', 'superstrate', 'cover', 'clad', 'cladding',
    'ambient', 'air', 'medium', 'bg', 'film', 'inc', 'incident', 'ref', 'trn', 'struct'})


_INDEX_PART_WORDS = frozenset({'eff', 'real', 'imag', 're', 'im'})


_QUANTITY_WORDS = (
    ('index', re.compile(r'n\d+|nk|neff|ri|index(es)?|indices|ior|refract\w*')),
    ('permeability', re.compile(r'mu\d*|mur|permeab\w*')),
    ('permittivity', re.compile(r'eps\d*|epsr|epsilon\d*|permittiv\w*|dielectric')),
)


_CONTEXT_INDEX_WORDS = frozenset({'n', 'nr', 'nd'})


_QUANTITY_SYMBOLS = {'n': 'index', 'ri': 'index', 'mu': 'permeability', 'eps': 'permittivity'}


_QUANTITY_HOME = {
    'index': ("a refractive index",
              "every material slot -- `top`, `bot`, `structure`, a layer's `background`, a "
              "shape's `material` -- takes relative permittivity (eps = n**2) or a Material; "
              "wavebit.eps_from_refractive_index(n) converts"),
    'permeability': ("a permeability",
                     "a permeability is `mu` beside a raster `structure`, or the `mu` of a "
                     "Material(epsilon, mu) in any material slot"),
    'permittivity': ("a permittivity",
                     "a permittivity goes in a material slot -- `top`, `bot`, `structure`, a "
                     "layer's `background`, a shape's `material` -- as a number or a Material"),
}


def _reads_as_index_beside(word):
    """Whether a neighbouring ``word`` makes an ``n`` / ``nr`` / ``nd`` beside it an index."""
    return (word.isdigit() or word in _INDEX_PART_WORDS
            or word.rstrip('0123456789') in _SLOT_WORDS)


def name_quantities(name):
    """The material quantities ``name`` is spelled as, as a frozenset -- empty for most names.

    A material slot reads as permittivity; any other name reads as whatever its words say
    (:data:`_QUANTITY_WORDS`, an ``n`` / ``nr`` / ``nd`` in an index's company
    (:data:`_CONTEXT_INDEX_WORDS`), and a symbol glued to a slot), so ``n_top``, ``nTop``,
    ``ntop``, ``n_sub``, ``n1``, ``n``, ``RI_top``, ``nr_top``, ``top_index`` and
    ``refractive_index`` are all an index, ``mu_top`` a permeability, ``eps_top`` a
    permittivity, and ``thet``, ``n_orders`` and ``nG`` nothing. Read off the spelling alone,
    because that is all a refused name has: it names no slot, so no slot can say what it holds.
    """
    if name.lower() in _MATERIAL_SLOTS:
        return frozenset({'permittivity'})
    words = [w.lower() for w in re.findall(r'[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z0-9])', name)]
    found = set()
    for i, word in enumerate(words):
        found.update(q for q, pattern in _QUANTITY_WORDS if pattern.fullmatch(word))
        if word in _CONTEXT_INDEX_WORDS and (
                len(words) == 1 or any(map(_reads_as_index_beside, words[max(i - 1, 0):i]
                                           + words[i + 1:i + 2]))):
            found.add('index')
        found.update(q for symbol, q in _QUANTITY_SYMBOLS.items() for slot in _SLOT_WORDS
                     if word in (symbol + slot, slot + symbol))
    return frozenset(found)


def near_match(name, known, cutoff=0.6):
    """The one near name to offer for a refused ``name``, or ``None``; never across quantities.

    Every "did you mean" in the package comes from here, because a guess is an instruction the
    reader follows, and one that crosses quantities is a wrong answer rather than a wrong name.
    ``stack.n_top = 1.5`` is one letter from ``top``, and the plain nearest match said so: the
    user who follows it hands a refractive index to a slot that reads relative permittivity,
    and ``1.5`` is a fine permittivity -- the solve conserves energy and warns about nothing
    (:class:`~wavebit.materials.response.Material` says why the slots are permittivity). So a name
    spelled like a quantity (:func:`name_quantities`) is matched only against the names spelled
    like the same one, and a name spelled like none is matched against them all (``tpo`` is
    still a typo of ``top``). ``cutoff`` is difflib's; the solver's keyword door keeps 0.7.
    """
    import difflib
    quantity = name_quantities(name)
    candidates = sorted(k for k in known if not quantity or name_quantities(k) == quantity)
    near = difflib.get_close_matches(name, candidates, n=1, cutoff=cutoff)
    return near[0] if near else None


def quantity_note(names, known):
    """What a refused name spelled like a quantity is told when nothing in ``known`` takes it.

    The other half of :func:`near_match`'s rule: withholding the guess leaves ``n_top`` with
    nothing to go on, and the one thing that reader needs is that the material slots are not
    stated in index. Said only where no name in ``known`` is spelled like that quantity -- a
    dispersion table takes ``index``, and a note saying nothing here does would be false there.
    Returns ``''`` when there is nothing to say.
    """
    taken = frozenset().union(*(name_quantities(k) for k in known))
    spelled = {}
    for name in sorted(names):
        for quantity in sorted(name_quantities(name) - taken):
            spelled.setdefault(quantity, []).append(name)
    notes = []
    for quantity, refused in spelled.items():
        what, home = _QUANTITY_HOME[quantity]
        notes.append(f" {', '.join(map(repr, refused))} {'is' if len(refused) == 1 else 'are'} "
                     f"spelled like {what}, and no name here takes one, so no near name is "
                     f"guessed: {home}.")
    return ''.join(notes)


def near_match_hint(names, known):
    """The hint a refused name carries: its NVF rename, else its :func:`near_match`, and the
    :func:`quantity_note` of a name spelled like a quantity nothing in ``known`` takes.

    Shared by the two ways a name can be unknown -- an attribute assigned on an object
    (:func:`wavebit._common.validation._unknown_attribute`, for the solver, the excitation facade and
    the spec dataclasses) and a key in a parsed document (``src/wavebit/_common/_jobs_v1.py``'s
    ``_reject_unknown``) -- so ``inc.thet = 0.2`` and ``{'incidence': {'thet': 0.2}}`` are
    answered alike. A rename is offered only when its target is in ``known``: the owner must be
    able to take the name it is pointed at, so ``Stack.projector`` is not sent to a slot that
    lives on ``Numerics``. ``names`` are strings; a parsed key that is not one is refused
    before it gets here. Returns ``''`` when there is nothing to suggest.
    """
    known = sorted(known)
    renamed = [n for n in names if _NVF_OPTION_RENAMES.get(n) in known]
    rest = sorted(n for n in names if n not in renamed)
    near = [(n, meant) for n in rest for meant in [near_match(n, known)] if meant is not None]
    if len(names) == 1:
        guesses = ''.join(f' Did you mean {meant!r}?' for _, meant in near)
    else:
        guesses = (' Did you mean ' + ', '.join(f'{meant!r} for {n!r}' for n, meant in near)
                   + '?') if near else ''
    return nvf_option_rename_hint(renamed) + guesses + quantity_note(rest, known)
