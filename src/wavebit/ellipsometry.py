"""Ellipsometric angles and Mueller matrices from Jones matrices, on every backend.

The functions here are pure array algebra: they take a Jones matrix (or a stack of them) in
any of the three namespaces -- NumPy, JAX, PyTorch -- and return arrays of that namespace, so
they differentiate wherever the solve does. :meth:`wavebit.rcwa.EllipsometryOrders.jones`
builds the Jones matrices from a :func:`wavebit.rcwa.run_ellipsometry` result; anything with
the same layout may be passed in directly.

**Layout.** A Jones matrix is the trailing ``(2, 2)`` axes of an array, with every leading axis
(wavelength, angle, order, ...) broadcast over. Its basis is the ``(s, p)`` pair of
``rcwa.PlaneWave(polarization=(a_s, a_p))`` and it is indexed ``J[..., out, in]``::

    (b_s, b_p) = J @ (a_s, a_p)        J = [[j_ss, j_sp],
                                            [j_ps, j_pp]]

so ``j_ps`` is the p-polarized outgoing amplitude for s-polarized incidence. Each amplitude is
quoted in Wavebit's own outgoing s/p frame of that order (``docs/reference/common/conventions.md``):
``s_hat = z_hat x k_hat_par`` on every beam, and the reflected ``p_hat``'s in-plane part is
``+cos(theta) k_hat_par`` where the incident and transmitted ones are ``-cos(theta) k_hat_par``.
For top-side incidence that is the Nebraska convention of ellipsometry (R. H. Muller, Surf. Sci.
16, 14 (1969); Azzam and Bashara, *Ellipsometry and Polarized Light*, 1977): a single interface at
normal incidence has ``r_p = -r_s``, so an isotropic sample has ``Delta = pi`` there.

**Time convention.** Wavebit's amplitudes are ``exp(-i omega t)`` phasors (loss is ``Im(n) > 0``).
Ellipsometers and their literature write ``exp(+i omega t)``, where every amplitude is the complex
conjugate. :func:`psi_delta` therefore reports ``delta`` in the instrument convention,
``tan(psi) exp(i delta) = conj(j_pp / j_ss)``, so that an absorbing substrate gives
``0 < delta < pi`` as in a measurement, and keeps the native ratio ``rho = j_pp / j_ss`` beside it.
The Mueller matrix is the same under either convention (its entries are intensity ratios), and
:func:`mueller_from_jones` is written so that it equals the instrument's.
"""

import numpy as np

from ._common.backend import array_namespace, as_array, eager_under_torch_compile, _device_of
from ._common.declarations import Declaration, spec
from ._common.pytrees import register_dataclass

__all__ = ['Ellipsometry', 'psi_delta', 'mueller_from_jones', 'depolarization_index',
           'cloude_eigenvalues']


#: Stokes vector ``(S0, S1, S2, S3)`` from ``E (x) E*`` in the ``(p, s)`` Jones basis, column
#: order ``(E_p E_p*, E_p E_s*, E_s E_p*, E_s E_s*)``, for ``exp(-i omega t)`` phasors:
#: ``S1 = I_p - I_s``, ``S2 = I_+45 - I_-45 = 2 Re(E_p E_s*)``, ``S3 = I_R - I_L = 2 Im(E_p E_s*)``.
#: Azzam and Bashara (1977) section 2.6; Gil and Ossikovski, *Polarized Light and the Mueller
#: Matrix Approach* (CRC, 2016). ``A A^H = 2 I``, so ``A^-1 = A^H / 2``. The sign of the last row
#: is the one choice the time convention touches: with it, the matrix built from a Wavebit Jones
#: matrix equals the one an ``exp(+i omega t)`` code builds from the conjugate Jones matrix.
_STOKES_FROM_COHERENCY = np.array([[1, 0, 0, 1],
                                   [1, 0, 0, -1],
                                   [0, 1, 1, 0],
                                   [0, -1j, 1j, 0]])

#: Identity and the three Pauli matrices, the basis of Cloude's coherency matrix.
_PAULI = np.array([[[1, 0], [0, 1]], [[1, 0], [0, -1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]]])


@spec
class Ellipsometry(Declaration):
    """Ellipsometric angles of one Jones matrix, or of a stack of them.

    ``psi`` and ``delta`` are the standard pair, ``tan(psi) exp(i delta) = conj(j_pp / j_ss)``:
    ``psi`` in ``[0, pi/2]`` and ``delta`` in ``[0, 2 pi)``, radians, in the instrument
    (``exp(+i omega t)``) convention. ``rho`` is the native ratio ``j_pp / j_ss`` itself, an
    ``exp(-i omega t)`` phasor whose argument is ``-delta``.

    ``psi_ps``, ``delta_ps`` and ``psi_sp``, ``delta_sp`` are the generalized ellipsometry
    ratios of an anisotropic or azimuthally asymmetric sample (M. Schubert, Phys. Rev. B 53,
    4265 (1996)): ``tan(psi_ps) exp(i delta_ps) = conj(j_sp / j_pp)`` -- p in, s out, over pp --
    and ``tan(psi_sp) exp(i delta_sp) = conj(j_ps / j_ss)`` -- s in, p out, over ss -- with
    ``rho_ps`` and ``rho_sp`` their native ratios. Both vanish on an isotropic stack.
    """
    psi: object
    delta: object
    rho: object
    psi_ps: object
    delta_ps: object
    rho_ps: object
    psi_sp: object
    delta_sp: object
    rho_sp: object

    def __post_init__(self):
        register_dataclass(type(self))


def _wrap(xp, angle):
    """``angle`` reduced into ``[0, 2 pi)``; one spelling the three namespaces share."""
    turn = 2 * xp.pi
    return angle - xp.floor(angle / turn) * turn


def _angles(xp, numerator, denominator):
    ratio = numerator / denominator
    psi = xp.arctan2(xp.abs(numerator), xp.abs(denominator))
    delta = _wrap(xp, -xp.angle(ratio))
    return psi, delta, ratio


@eager_under_torch_compile
def psi_delta(jones):
    """:class:`Ellipsometry` of ``jones``, whose trailing axes are ``(2, 2)`` in ``(s, p)``.

    ``psi = atan2(|j_pp|, |j_ss|)`` and ``delta = -arg(j_pp / j_ss) mod 2 pi``: the instrument
    convention, as the module docstring explains. Where ``|j_pp|`` is exactly zero (a lossless
    stack at its Brewster angle) ``psi`` is zero and its derivative is singular; ``delta`` steps
    by ``pi`` through that point. A test that pins a convention should compare ``exp(i delta)``
    rather than ``delta``, which rounding can put on either side of ``0 = 2 pi``.
    """
    xp = array_namespace(jones)
    j_ss, j_sp, j_ps, j_pp = jones[..., 0, 0], jones[..., 0, 1], jones[..., 1, 0], jones[..., 1, 1]
    psi, delta, rho = _angles(xp, j_pp, j_ss)
    psi_ps, delta_ps, rho_ps = _angles(xp, j_sp, j_pp)
    psi_sp, delta_sp, rho_sp = _angles(xp, j_ps, j_ss)
    return Ellipsometry(psi=psi, delta=delta, rho=rho, psi_ps=psi_ps, delta_ps=delta_ps,
                        rho_ps=rho_ps, psi_sp=psi_sp, delta_sp=delta_sp, rho_sp=rho_sp)


def _constant(xp, like, value):
    return as_array(xp, value, dtype=like.dtype, device=_device_of(like))


@eager_under_torch_compile
def mueller_from_jones(jones, normalize=True):
    """The ``(4, 4)`` real Mueller matrix of ``jones``: ``M = A (J (x) J*) A^-1``.

    Stokes basis ``(S0, S1, S2, S3) = (I_p + I_s, I_p - I_s, I_+45 - I_-45, I_R - I_L)``, p along
    the first axis as an ellipsometer's software has it (Azzam and Bashara 1977; Fujiwara,
    *Spectroscopic Ellipsometry*, 2007, chapter 4). ``normalize=True`` divides by ``m_11``, the
    reflectance of unpolarized light ``(|j_pp|^2 + |j_ss|^2 + |j_sp|^2 + |j_ps|^2) / 2``; the
    unnormalized matrix keeps it. An isotropic sample gives the standard form
    ``[[1, -N, 0, 0], [-N, 1, 0, 0], [0, 0, C, S], [0, 0, -S, C]]`` with ``N = cos 2 psi``,
    ``C = sin 2 psi cos delta`` and ``S = sin 2 psi sin delta``, for the ``delta`` of
    :func:`psi_delta`. A matrix derived from a Jones matrix is non-depolarizing:
    :func:`depolarization_index` is 1 and :func:`cloude_eigenvalues` has one nonzero entry.
    """
    xp = array_namespace(jones)
    # (p, s) order for the Stokes basis: swap both axes of the (s, p) Jones matrix.
    swap = [1, 0]
    j = jones[..., swap, :][..., :, swap]
    coherency = xp.einsum('...ij,...kl->...ikjl', j, xp.conj(j))
    coherency = coherency.reshape(tuple(coherency.shape[:-4]) + (4, 4))
    a = _constant(xp, coherency, _STOKES_FROM_COHERENCY)
    mueller = xp.real(xp.einsum('ab,...bc,dc->...ad', a, coherency, xp.conj(a))) / 2
    if normalize:
        mueller = mueller / mueller[..., 0:1, 0:1]
    return mueller


@eager_under_torch_compile
def depolarization_index(mueller):
    """Gil and Bernabeu's depolarization index of a ``(4, 4)`` Mueller matrix.

    ``sqrt((sum m_ij^2 - m_11^2) / (3 m_11^2))`` (Optica Acta 33, 185 (1986)): 1 for a
    non-depolarizing (pure) matrix, 0 for an ideal depolarizer.
    """
    xp = array_namespace(mueller)
    total = xp.einsum('...ij,...ij->...', mueller, mueller)
    m11 = mueller[..., 0, 0]
    return xp.sqrt((total - m11**2) / (3 * m11**2))


@eager_under_torch_compile
def cloude_eigenvalues(mueller):
    """The four eigenvalues, ascending, of the coherency matrix of a ``(4, 4)`` Mueller matrix.

    ``H = (1/4) sum_ij m_ij (sigma_i (x) sigma_j*)`` over the identity and Pauli matrices
    (S. R. Cloude, Optik 75, 26 (1986)). ``M`` is a physical Mueller matrix exactly when every
    eigenvalue is nonnegative, and non-depolarizing when only one is nonzero -- the matrix of a
    Jones matrix ``J`` has ``H`` proportional to ``vec(J) vec(J)^H``, rank one.
    """
    xp = array_namespace(mueller)
    complex_mueller = mueller * (1 + 0j)
    sigma = _constant(xp, complex_mueller, _PAULI)
    coherency = xp.einsum('...ij,iab,jcd->...acbd', complex_mueller, sigma, xp.conj(sigma)) / 4
    coherency = coherency.reshape(tuple(coherency.shape[:-4]) + (4, 4))
    return xp.linalg.eigvalsh(coherency)
