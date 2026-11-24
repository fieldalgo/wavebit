"""Jones, Psi/Delta and Mueller outputs of an RCWA stack: two solves on one preparation."""
import dataclasses
from dataclasses import field

from .._common.backend import array_namespace, eager_under_torch_compile
from .._common.declarations import Declaration, spec
from .._common.execution import frozen_mapping
from .._common.pytrees import register_dataclass
from ..errors import ValidationError
from .sources import PlaneWave

#: What the two solves are lit with: the pure states, as Jones pairs so that both problems have
#: one declaration layout and the second reuses the first's preparation (a polarization *word*
#: is part of a problem's structure; a pair's values are live).
_S_PAIR, _P_PAIR = (1., 0.), (0., 1.)


@spec
class EllipsometryOrders(Declaration):
    """The two responses of one port, and the Jones matrix, angles and Mueller matrix they give.

    ``s`` and ``p`` are the port's :class:`wavebit.DiffractionOrders` for s-polarized and
    p-polarized incidence: the same stack, the same orders, lit with ``polarization=(1, 0)`` and
    ``(0, 1)``. Every array of :class:`wavebit.DiffractionOrders` -- efficiencies, powers, fields,
    ``k_parallel`` -- is read from those two as before; this class adds what needs both.

    :meth:`jones` assembles ``J[out, in]`` in the ``(s, p)`` basis, :meth:`ellipsometry` the
    angles and :meth:`mueller` the Mueller matrix, each for one order (the zeroth by default) or
    for every order. The conventions, with their references, are stated once in
    :mod:`wavebit.ellipsometry` and repeated in ``reference['jones_basis']``,
    ``reference['delta_convention']`` and ``reference['mueller_basis']``. Two cautions from
    there: the reflected p axis is the Nebraska one for top-side incidence, and from the bottom
    the triad's handedness is the opposite, so ``j_pp / j_ss`` is unchanged but the cross terms
    and the Mueller matrix's off-diagonal blocks change sign against an instrument's; and on the
    ``ky = 0`` row an order with ``kx < 0`` is quoted in the reversed frame, a sign on the whole
    of ``J`` that no ratio and no Mueller entry sees.
    """
    wavelengths: object
    orders_u: tuple
    orders_v: tuple
    s: object
    p: object
    reference: object = field(default_factory=dict)
    metadata: object = field(default_factory=dict)

    def __post_init__(self):
        register_dataclass(type(self), static_fields=('orders_u', 'orders_v'))
        for name in ('orders_u', 'orders_v'):
            object.__setattr__(self, name, tuple(int(v) for v in getattr(self, name)))
        for name in ('reference', 'metadata'):
            object.__setattr__(self, name, frozen_mapping(getattr(self, name)))

    def _index(self, order):
        m, n = order
        try:
            return self.orders_v.index(int(n)), self.orders_u.index(int(m))
        except ValueError as error:
            raise KeyError((m, n)) from error

    def jones(self, order=(0, 0)):
        """The Jones matrix of ``order``: ``(wavelength, 2, 2)``, or of every order with
        ``order=None``: ``(wavelength, order_v, order_u, 2, 2)``.

        ``(s, p)`` basis, ``J[..., out, in]``: column 0 is the s-incidence response
        ``(amplitude_ratio_s, amplitude_ratio_p)`` of ``s``, column 1 the p-incidence response of
        ``p``. The entries are the amplitude ratios as :class:`wavebit.DiffractionOrders` quotes
        them, in each order's own outgoing s/p frame, and nothing is re-signed here.
        """
        xp = array_namespace(self.s.amplitude_ratio_s)
        ss, ps = self.s.amplitude_ratio_s, self.s.amplitude_ratio_p
        sp, pp = self.p.amplitude_ratio_s, self.p.amplitude_ratio_p
        if order is not None:
            j, i = self._index(order)
            ss, ps, sp, pp = (a[:, j, i] for a in (ss, ps, sp, pp))
        return xp.stack([xp.stack([ss, sp], axis=-1), xp.stack([ps, pp], axis=-1)], axis=-2)

    def ellipsometry(self, order=(0, 0)):
        """:class:`wavebit.ellipsometry.Ellipsometry` of :meth:`jones`: ``psi``, ``delta``,
        ``rho`` and the generalized ratios, each ``(wavelength,)`` or, with ``order=None``,
        ``(wavelength, order_v, order_u)``."""
        from ..ellipsometry import psi_delta
        return psi_delta(self.jones(order))

    def mueller(self, order=(0, 0), normalize=True):
        """The Mueller matrix of :meth:`jones`, ``(wavelength, 4, 4)`` or, with ``order=None``,
        ``(wavelength, order_v, order_u, 4, 4)``; see :func:`wavebit.ellipsometry.mueller_from_jones`."""
        from ..ellipsometry import mueller_from_jones
        return mueller_from_jones(self.jones(order), normalize=normalize)


@spec
class EllipsometryResult(Declaration):
    """What :func:`run_ellipsometry` returns: the two plain results and the two combined ports.

    ``s`` and ``p`` are the :class:`wavebit.rcwa.Result` of the s- and p-polarized solves, with
    everything a single solve has. ``reflection`` and ``transmission`` are the
    :class:`EllipsometryOrders` of the two ports, which hold those results' orders side by side.
    """
    s: object
    p: object
    wavelengths: object
    reflection: object
    transmission: object

    def __post_init__(self):
        register_dataclass(type(self))


_CONVENTIONS = {
    'jones_basis': "(s, p) as rcwa.PlaneWave(polarization=(a_s, a_p)); J[out, in]; each order's "
                   "outgoing s/p frame, reflected p_hat in-plane +cos(theta) k_hat_par (Nebraska "
                   "for top-side incidence)",
    'delta_convention': 'instrument: tan(psi) exp(i delta) = conj(j_pp / j_ss), delta in [0, 2 pi); '
                        'rho = j_pp / j_ss keeps the exp(-i omega t) phasor',
    'mueller_basis': 'Stokes (I_p + I_s, I_p - I_s, I_+45 - I_-45, I_R - I_L), normalized by m_11',
    'time_convention': 'exp(-i*omega*t)'}


def _port(name, first, second):
    orders_s, orders_p = first.outputs[name], second.outputs[name]
    return EllipsometryOrders(wavelengths=orders_s.wavelengths, orders_u=orders_s.orders_u,
                              orders_v=orders_s.orders_v, s=orders_s, p=orders_p,
                              reference={**orders_s.reference, **_CONVENTIONS},
                              metadata=dict(orders_s.metadata))


def _lit_with(problem, pair):
    source = dataclasses.replace(problem.excitation.sources[0], polarization=pair)
    return problem.replace(excitation=problem.excitation.replace(sources=(source,)))


def _refuse(problem):
    sources = problem.excitation.sources
    if len(sources) != 1 or type(sources[0]) is not PlaneWave:
        raise ValidationError('run_ellipsometry takes exactly one rcwa.PlaneWave: it solves that '
                              'wave twice, s- and p-polarized, on one preparation.',
                              code='ellipsometry.source', path='excitation.sources')
    if not isinstance(sources[0].polarization, str):
        raise ValidationError("run_ellipsometry chooses the polarization itself: declare the "
                              "PlaneWave with polarization='s' or 'p' (either is solved as both), "
                              "not a Jones pair, which it would have to discard.",
                              code='ellipsometry.polarization', path='excitation.sources[0].polarization')
    if not problem.outputs.diffraction:
        raise ValidationError('run_ellipsometry needs the diffraction orders: Outputs(diffraction=True).',
                              code='ellipsometry.outputs', path='outputs.diffraction')


@eager_under_torch_compile
def run_ellipsometry(problem, *, numerics, backend='numpy', device='cpu', precision='double'):
    """Solve one plane wave s- and p-polarized and return its Jones, Psi/Delta and Mueller outputs.

    The problem's single :class:`wavebit.rcwa.PlaneWave` is prepared once and run twice, lit
    with ``polarization=(1, 0)`` and ``(0, 1)``; its own ``polarization`` word is only a
    placeholder. The cost is two solves on one preparation, and the
    :class:`EllipsometryResult` holds both plain results beside the combined ports. Every other
    input -- side, angles or ``kx``/``ky``, wavelengths, materials, numerics -- means what it
    means for :func:`wavebit.rcwa.run`.

    Differentiable where :func:`wavebit.rcwa.run` is, eagerly: under ``jax.grad`` and PyTorch
    autograd. Like :func:`wavebit.rcwa.prepare` it needs a concrete problem, so under ``jax.jit``
    prepare the two problems outside the transform instead.
    """
    from ._lifecycle import prepare
    _refuse(problem)
    lit_s, lit_p = _lit_with(problem, _S_PAIR), _lit_with(problem, _P_PAIR)
    prepared = prepare(lit_s, numerics=numerics, backend=backend, device=device, precision=precision)
    first, second = prepared.run(lit_s), prepared.run(lit_p)
    return EllipsometryResult(s=first, p=second, wavelengths=first.wavelengths,
                              reflection=_port('reflected', first, second),
                              transmission=_port('transmitted', first, second))
