"""Scattering-matrix method (SMM) — the second connecting algorithm, single-source.

Private ``top``/``bottom`` names mean start/end ports of the positive-z list, as in
``solve_1d``. The engine maps physical bottom/top onto them. Directional descriptions
below use that local propagation frame, not the public upward height.

The *connecting algorithm* is how per-layer eigenmodes are joined into a stack response. ETM
(:mod:`wavebit._core.solve_1d` / :mod:`wavebit._core.solve_2d`) propagates an enhanced
transmittance matrix; SMM composes per-layer scattering matrices with the Redheffer star
product. Both consume the same modes — this module imports the mode builders from the ETM
solves rather than restating the eigen-algebra — and both use the same boundary conventions,
so ``layer_algorithm='SMM'`` returns the same efficiencies as ``'ETM'``, not a second opinion.

Why have it: a layer's scattering matrix depends only on that layer and the shared gap medium,
and the star product is associative. Layer responses can therefore be built once and cascaded,
reused, or reordered without touching their neighbours — where ETM's recursion carries a state
that already contains everything below the current layer. SMM is also the formulation most of
the literature states its results in, which makes it the natural cross-check on the ETM path.

What it is *not*: a rescue from transfer-matrix instability. The option is named ``'ETM'``
because it *is* the enhanced-transmittance form, which is already stable; both algorithms only ever evaluate ``exp(-k0*q*d)`` with ``Re(q) >= 0``
(principal root) and only inside bounded products. Measured on thick, strongly evanescent and
high-truncation stacks the two agree to ~1e-14 with no energy drift either way, and SMM costs
~1.1-1.4x ETM (the per-layer ``eig`` dominates both), or ~1.3-1.6x once it also recovers the
field-reconstruction state described below.

Formulation, in the conventions this codebase already uses (see :func:`solve_1d`):

* Every region's ``(W, V)`` describe its *downward* modes: a field with amplitudes ``c`` has
  primary variable ``W @ c`` and partner ``V @ c``; an upward wave flips the partner's sign.
  With that single convention the half-space scattering matrices below are derived, not
  transcribed — no per-region ``kz`` sign flip is needed.
* Layers are joined through a fictitious *gap medium* with ``eps_gap = 1 + kx^2 + ky^2``, so
  ``kz_gap == 1`` exactly and the gap's ``V`` can never be singular (the standard trick; the
  gap cancels out of the composed result whatever it is, as long as it is used everywhere).
* Incidence enters as mode amplitudes directly: ``E_s = a_s`` and ``H_p = -i*n_top*a_p``
  for the coupled path — the resolved ``(a_s, a_p)`` pair, whichever spelling the
  ``polarization`` pair — and for 1D TE/TM the unit primary amplitude signed into each order's
  own s/p frame, ``sigma_m`` of ``E_y`` or (TM) ``H_y``
  (:func:`~wavebit._core.conventions.order_frame_sign`); the TM field state is then rescaled
  once to unit incident E (``n/mu`` of the incidence medium) so that every path's
  reconstructed field answers a unit electric input.
  These are the same incident amplitudes the ETM final matching solves against. The sign on
  ``H_p`` is the one ``e^{-iωt}`` requires and the one the code has always carried; this
  sentence said ``+i`` — the conjugate — until 2026-08-11, describing the formulation that
  was retired in 2026-08 rather than the code beneath it. Checked rather than reasoned:
  ``R_p / (-i*n_top)`` is the analytic amplitude to 1e-15 on both algorithms
  (``test_lossy_media.test_reflection_amplitude``), and ``+i`` gives its negative.

Field reconstruction works from either algorithm. What :mod:`wavebit._core.fields` walks is
per-layer mode amplitudes, which the ETM recursion happens to leave behind and the SMM cascade
does not — so the SMM recovers them afterwards from its own split-plane amplitudes and hands
over the same ``layer_info_list`` / ``T1`` state (see :func:`_field_state`). Same is close to
literal: the recovered blocks match the ETM's to ~1e-15 for every layer but the bottom one,
where the two differ by a basis the reconstruction never resolves. They have to match, because
the amplitudes they describe are unique given the modes, and the modes come from the same
builders. That recovery is a second pass over the stack, done only from ``answers='field'``
up, so declaring anything lower skips it outright — it is where the extra ~10-20% in the cost
quoted above goes, and it is the one rung of the ladder that buys time rather than memory.
"""
import sys
from typing import NamedTuple

import numpy as np

from .backend import _device_of, array_namespace, as_array, check_finite, check_physical
from .conventions import boundary_kz, order_azimuth, order_frame_sign
from .enums import Answers, answers_of
from .conventions import incident_column
from .solve_1d import (_block, _diag, _is_backside, _layer_mu, _pack,
                       layer_modes_1d, layer_modes_1d_conical, sp_efficiencies,
                       sp_modes_1d_conical)
from .solve_2d import layer_modes_2d, order_grid_kxy, sp_modes_2d
from .power import (incident_power, net_power, power_for_real_k,
                    sp_incident_power, wave_power)

_PI = 3.141592653589793


# --------------------------------------------------------------------------- S-matrix algebra
# A scattering matrix is carried as the 4-tuple of blocks ``(S11, S12, S21, S22)`` relating
# the waves entering at each port to those leaving it:
#     [out_top ; out_bot] = S @ [in_top ; in_bot]
# where ``in_top`` travels down into the structure and ``in_bot`` up into it.

def _ab_matrices(xp, W_to, V_to, W_from, V_from):
    """Interface coefficients ``(A, B)`` for a boundary crossed *into* the ``to`` basis.

    From continuity of the primary field and its partner across the interface,
    ``c_to_down = (A @ c_from_down + B @ c_from_up)/2`` (and the mirrored relation for the
    upward amplitudes), with ``A = W_to^-1 W_from + V_to^-1 V_from`` and ``B`` the difference.
    """
    P = xp.linalg.inv(W_to) @ W_from
    Q = xp.linalg.inv(V_to) @ V_from
    return P + Q, P - Q


def _s_layer(xp, A, B, X):
    """Scattering matrix of one layer embedded between gap half-spaces.

    ``A``/``B`` describe the gap -> layer interface and ``X = diag(exp(-k0*q*d))`` the modal
    propagation across it. The layer is symmetric, so ``S22 == S11`` and ``S21 == S12``.
    """
    A_i = xp.linalg.inv(A)
    XB = X @ B
    common = xp.linalg.inv(A - XB @ A_i @ X @ B)
    S11 = common @ (XB @ A_i @ X @ A - B)
    S12 = common @ X @ (A - B @ A_i @ B)
    return S11, S12, S12, S11


def _s_reflection(xp, W_top, V_top, W_gap, V_gap):
    """Scattering matrix of the superstrate -> gap interface (port 1 in the superstrate)."""
    A, B = _ab_matrices(xp, W_gap, V_gap, W_top, V_top)
    A_i = xp.linalg.inv(A)
    return -A_i @ B, 2 * A_i, 0.5 * (A - B @ A_i @ B), B @ A_i


def _s_transmission(xp, W_bot, V_bot, W_gap, V_gap):
    """Scattering matrix of the gap -> substrate interface (port 2 in the substrate)."""
    A, B = _ab_matrices(xp, W_gap, V_gap, W_bot, V_bot)
    A_i = xp.linalg.inv(A)
    return B @ A_i, 0.5 * (A - B @ A_i @ B), 2 * A_i, -A_i @ B


def redheffer_star(xp, I, S_a, S_b):
    """Redheffer star product ``S_a * S_b`` — cascade two scattering matrices."""
    a11, a12, a21, a22 = S_a
    b11, b12, b21, b22 = S_b
    D_i = xp.linalg.inv(I - b11 @ a22)
    F_i = xp.linalg.inv(I - a22 @ b11)
    return (a11 + a12 @ D_i @ b11 @ a21,
            a12 @ D_i @ b12,
            b21 @ F_i @ a21,
            b22 + b21 @ F_i @ a22 @ b12)


def _n_layers(thickness):
    return len(thickness) if not hasattr(thickness, 'shape') else thickness.shape[0]


def _prepare(epx_conv_all, kx, ky=None):
    """Namespace, complex working dtype and complex-promoted ``kx`` / ``ky``."""
    xp = array_namespace(epx_conv_all)
    kx = as_array(xp, kx)
    if not xp.isdtype(kx.dtype, 'complex floating'):
        kx = xp.astype(kx, xp.complex128)
        if ky is not None:
            ky = xp.astype(as_array(xp, ky), xp.complex128)
    elif ky is not None:
        ky = as_array(xp, ky)
    return xp, kx.dtype, kx, ky


# ------------------------------------------------------------- field-reconstruction state
# The cascade above answers at the outer ports only. Field reconstruction needs what is
# inside: each layer's own down- and up-going mode amplitudes. Both are recovered from the
# split-plane amplitudes of the same cascade, and then handed over in the exact form
# `fields` already consumes from the ETM — see `_field_state` for that contract.

def _below_reflections(xp, I, S_layers, S_trn):
    """Reflection of everything below each split plane, in the gap basis.

    ``gamma[p]`` maps the down-going gap amplitude at split plane ``p`` to the up-going one:
    it is the ``S11`` block of ``L_p * ... * L_{n-1} * S_trn``. Plane ``p`` sits directly above
    layer ``p``, so plane ``n`` faces the substrate interface alone.

    Built bottom-up, one plane at a time. Only the ``S11`` block of each partial cascade is
    ever used, and that block costs a single inverse — a fraction of a full Redheffer star.
    """
    n = len(S_layers)
    gamma = [None] * (n + 1)
    gamma[n] = S_trn[0]
    for p in range(n - 1, -1, -1):
        s11, s12, s21, s22 = S_layers[p]
        below = gamma[p + 1]
        gamma[p] = s11 + s12 @ xp.linalg.inv(I - below @ s22) @ below @ s21
    return gamma


def _field_state(xp, I, S_ref, S_trn, S_layers, AB, c_inc):
    """Per-layer ``(a_i, b)`` blocks and the top-layer amplitude ``T1``, top-to-bottom.

    Field reconstruction walks the stack downward carrying one vector: ``T_i``, layer ``i``'s
    down-going mode amplitude at its own top plane. It asks each layer for two matrices — one
    that advances that vector to the next layer (``T_{i+1} = a_i X_i T_i``) and one that turns
    it into the layer's up-going amplitude at its bottom plane (``c2_i = b_i T_{i+1}``). The
    ETM produces both as by-products of its recursion; the SMM has to reconstruct them, and
    away from the bottom layer they come out as the *same* matrices, because the down-going
    amplitude at a plane determines everything below it uniquely.

    At the plane below layer ``i`` the gap amplitudes are ``(g, gamma[i+1] g)`` for some ``g``,
    so a layer's amplitudes there are linear in ``g``: down is ``M_i g`` and up is ``N_i g``,
    with ``M_i = (A_i + B_i gamma)/2`` and ``N_i = (B_i + A_i gamma)/2`` from the gap -> layer
    interface. Eliminating ``g`` between neighbours gives the two matrices asked for.

    The bottom layer has no layer below it and carries ``g`` itself onward, which is why its
    two blocks are the one place this state differs from the ETM's (their product, the only
    combination the reconstruction forms, still agrees). Going further and expressing that
    vector in the substrate's own basis — matching the ETM exactly — would mean inverting the
    substrate's ``V``, and that matrix is singular whenever a retained order sits exactly at
    cutoff, ``kz_bot == 0``. The gap basis is the one basis this formulation guarantees
    invertible, by its choice of ``eps_gap``; the value handed past the bottom layer is never
    read anyway.

    Reference planes matter here too: ``b_i`` is anchored at the layer *bottom*, which is what
    keeps every exponential in it decaying. Anchoring the up-going amplitude at the top plane
    instead would need ``X^-1``, and that is exactly the growing exponential both connecting
    algorithms are built to avoid.
    """
    n = len(S_layers)
    if n == 0:
        return [], None

    gamma = _below_reflections(xp, I, S_layers, S_trn)

    # Down-going gap amplitude just below the superstrate interface, closing the loop over
    # everything below it: g = S_ref21 c_inc + S_ref22 (gamma[0] g).
    g_top = xp.linalg.inv(I - S_ref[3] @ gamma[0]) @ (S_ref[2] @ c_inc)
    A_0, B_0 = AB[0]
    T1 = 0.5 * (A_0 + B_0 @ gamma[0]) @ g_top

    blocks = []
    for i in range(n):
        A, B = AB[i]
        below = gamma[i + 1]
        M = 0.5 * (A + B @ below)
        N = 0.5 * (B + A @ below)
        if i + 1 < n:
            A_next, B_next = AB[i + 1]
            M_next = 0.5 * (A_next + B_next @ below)
            blocks.append((M_next @ xp.linalg.inv(M), N @ xp.linalg.inv(M_next)))
        else:
            blocks.append((xp.linalg.inv(M), N))    # carried on in the gap basis, M_next = I
    return blocks, T1


#: What each :attr:`Port.channel` counts, in the words a refused join names it by.
_CHANNELS = {
    'te': 'the pure 1D TE path, one s amplitude per order',
    'tm': 'the pure 1D TM path, one p port variable per order',
    'sp': 'a coupled path, s amplitudes over p port variables',
}


class Port(NamedTuple):
    """The basis one end of a :class:`ScatteringMatrix` expresses its amplitudes in.

    A scattering block is a bare matrix of numbers; what makes those numbers mean anything is
    the set of plane waves its rows and columns count. Two blocks may only be cascaded when the
    end they meet at counts the *same* waves, and that is what this records so
    :meth:`ScatteringMatrix.cascade` can check it instead of trusting the caller.

    ``index`` and ``mu`` are the half-space the port opens into. ``kx`` / ``ky`` are the
    retained orders' in-plane wavevectors, normalised by ``k0`` as everywhere else in the core;
    ``ky`` is ``None`` on the pure 1D TE/TM path, which has no out-of-plane order.

    ``channel`` is the polarization the port's amplitude vector counts: ``'te'`` or ``'tm'`` on
    the pure 1D path, one entry per order, and ``'sp'`` on the coupled paths -- conical, 2D and
    first-order alike -- ``n_orders`` s amplitudes over ``n_orders`` p port variables. It is the
    one field the two pure channels cannot share: a TE and a TM matrix of the same cut have the
    same orders, the same media and the same size, so until 2026-09-27 they joined, and the
    result was the stack's matrix for neither polarization.

    On JAX ``channel`` is pytree aux data, not a child (:func:`_register_pytree_if_jax_loaded`):
    a string cannot cross a ``jax.jit`` boundary as a leaf, and which waves a port counts is
    fixed by the path, never by a traced value.
    """
    kx: object
    ky: object
    index: object
    mu: object
    channel: str


#: The fields of a :class:`Port` that are pytree children, in order; ``channel`` is aux data.
_PORT_CHILDREN = ('kx', 'ky', 'index', 'mu')

_PYTREE_REGISTERED = False


def _register_pytree_if_jax_loaded():
    """Register :class:`Port` as a JAX pytree with ``channel`` in the treedef, if JAX is loaded.

    A ``NamedTuple`` is a pytree without registration, every field a child, which is how a
    ``ScatteringMatrix`` has always crossed a ``jax.jit`` boundary. A string child cannot: the
    output would be refused as ``not a valid JAX type``. Registered here, ``kx`` / ``ky`` /
    ``index`` / ``mu`` stay children -- a traced joint medium is still a leaf -- and the channel
    rides in the treedef, as it has to for a jitted function returning one of these to be
    rebuilt with it. Registered *with keys*, so a leaf's path still names its field
    (``.entry_port.kx``) as the unregistered ``NamedTuple``'s did, not a flat index.

    Lazy, and never importing JAX, for the reason
    :func:`wavebit._core.domain._register_pytree_if_jax_loaded` gives at length. Called from
    :func:`_scattering_matrix`, the one constructor of a port, and on import of the JAX
    backend. The one ordering neither covers is a matrix built before ``import jax`` and handed
    to JAX before any JAX solver or any other matrix is built: there the channel is still a
    string leaf, and JAX says so loudly (``TypeError: ... of type <class 'str'> ... at path
    S.entry_port.channel``) rather than answering wrongly. Building either afterwards registers
    the class and repairs the matrices already built, since registration is per class and not
    per instance. Before ports carried a channel no such ordering existed: every field was a
    number or ``None``, each a valid leaf registered or not.
    """
    global _PYTREE_REGISTERED
    if _PYTREE_REGISTERED:
        return
    jax = sys.modules.get('jax')
    if jax is None:
        return
    tree_util = getattr(jax, 'tree_util', None)
    if tree_util is None:
        return                      # jax is mid-import; the next port will try again
    try:
        tree_util.register_pytree_with_keys(
            Port,
            lambda port: (tuple((tree_util.GetAttrKey(name), getattr(port, name))
                                for name in _PORT_CHILDREN), port.channel),
            lambda channel, children: Port(*children, channel=channel),
            lambda port: (tuple(getattr(port, name) for name in _PORT_CHILDREN), port.channel))
    except ValueError:
        pass                        # already registered; idempotent by design
    _PYTREE_REGISTERED = True


class ScatteringMatrix(NamedTuple):
    """The stack's scattering matrix -- both sides, every order -- kept instead of dropped.

    The SMM cascade builds this to answer the solve at all -- ``R`` and ``T`` are two of its
    blocks applied to the incident vector -- so keeping it adds no arithmetic and four matrices
    of memory, whatever the stack's depth. It rides no declaration of its own: it appears from
    ``answers='amplitude'`` up, because an ``S`` *is* amplitude information and the rung below
    is the one that said only power was wanted. ``solver.scattering`` is ``None`` after an ETM
    solve, which is not an ignored request but the algorithm's own shape -- the ETM fixes the
    boundary condition of a single illumination and never forms a both-sides operator at all.

    Each block is square of side ``n_orders`` on the pure 1D TE/TM path and ``2*n_orders`` on
    the coupled ones, where the s and p amplitudes are stacked in that order -- the same
    layout the incident column has, so ``S11 @ c_inc`` needs no repacking. Four blocks in
    total, whatever the stack's depth. The pure 1D matrix is the coupled one's s-s (TE) or
    p-p (TM) block at ``phi = 0``, in the same port variables (since 2026-09-27; before, it
    was in the Cartesian ``E_y`` / ``H_y`` on every order, ``D S D`` of this with TM's
    ``S11`` / ``S22`` negated besides -- see :func:`_coupled_ports_1d`).

    **Its ports are the two half-spaces, not the SMM's gap.** The cascade's internal gap basis
    (``eps_gap`` chosen so ``kz_gap == 1``) is a gauge, and it cancels between the entry and
    exit interface factors: re-running a solve with a different ``eps_gap`` moves these blocks
    by ~1e-14 while it moves the per-layer gap-embedded ones by ~0.5. So these numbers are
    directly readable. For the **s** block, ``S11 @ delta`` is exactly the reflected ``r_s``
    the solve reported and ``S21 @ delta`` its ``t_s``. The **p** block holds port variables,
    not Fresnel amplitudes: its port variable is the p wave's magnetic field, which turns over
    for the up-going wave, so ``S11 @ delta`` is ``-r_p``, and ``S21 @ delta`` is ``t_p`` times
    ``(n_trn/mu_trn) / (n_ref/mu_ref)``. That holds on every path, the pure 1D TM one
    included. The per-layer blocks are deliberately not here; they would need the gap ->
    half-space conversion shipped beside them to mean anything.

    **Orientation is the hazard, not the numbers.** These are orientation-blind matrices, so
    handing them over without a stated frame is the same shape of defect as a backside field
    cell in a mirrored frame: every number plausible, every number mirrored. They are therefore
    *incidence-relative*, matching what the field cell and the ``R``/``T`` reads already do --
    :attr:`S11` reflects back into the incidence medium whichever side is lit, and
    :attr:`entry_port` is the side that was lit.

    Use :meth:`cascade` rather than multiplying two of these by hand: the ports are carried so
    that a mismatch can be refused, and a mismatch is otherwise invisible (see :meth:`cascade`).
    """
    #: Reflection back into the incidence medium.
    S11: object
    #: Transmission from the far half-space into the incidence medium.
    S12: object
    #: Transmission through the stack, incidence medium to the far half-space.
    S21: object
    #: Reflection back into the far half-space.
    S22: object
    #: The basis of the lit side -- the half-space the light arrives through.
    entry_port: Port
    #: The basis of the far side.
    exit_port: Port

    @property
    def blocks(self):
        """The four blocks alone, in ``(S11, S12, S21, S22)`` order.

        Kept because the whole tuple no longer splats into four matrices -- the ports ride in
        it -- and because :func:`redheffer_star` and every hand-rolled cascade want exactly
        these four.
        """
        return (self.S11, self.S12, self.S21, self.S22)

    def cascade(self, beyond):
        """This stack followed by ``beyond``, as one scattering matrix.

        **Incidence-relative, like every other orientation in this class.** ``beyond`` is the
        piece the light reaches *next* -- not the piece geometrically below. Lit from the top
        those are the same thing; lit from the bottom they are opposite, and the piece the
        light reaches next sits geometrically *above* this one. Getting it backwards is
        refused rather than answered, because the ports no longer meet.

        The two pieces are solved as independent problems that know nothing of each other, and
        this reproduces the joined stack's own matrix to 1e-15 -- from either side. The
        Redheffer star product's ``inv(I - beyond.S11 @ self.S22)`` is the infinite bounce at
        the cut, which is why it needs :attr:`S22`: the response to light arriving at *this*
        piece's far face from outside, a thing no single one-sided solve contains.

        The plane where they meet is a **basis, not a layer**. A half-space has no thickness
        and nothing propagates in it, so nominating one at the cut changes nothing: air,
        ``eps = 2.25`` and ``eps = 12`` all give the same composite, and inserting a real
        zero-thickness air layer into the joined stack moves it by 3.7e-16.

        What the two pieces *must* share is the port where they meet -- the same retained
        orders, the same joint medium and the same polarization channel (:class:`Port`) -- and
        this refuses when they do not. That refusal is
        the reason to call this rather than :func:`redheffer_star` by hand: a join across a
        shifted order grid does not fail loudly, it returns every number plausible and about
        ten percent wrong. ``theta`` is measured in the incidence medium, so a joint medium of
        a different index shifts the whole grid under the same angle.

        Chains, so a stack cut into more than two pieces joins in the order the light meets
        them::

            whole = first.cascade(second).cascade(third)

        Read eagerly: "are these the same basis" is a value comparison, which a traced array
        cannot answer. That is a property of the question -- it is asked once, where two
        solves are joined, not inside a differentiated loop.
        """
        if not isinstance(beyond, ScatteringMatrix):
            raise TypeError(
                f'cascade() takes a ScatteringMatrix, got {type(beyond).__name__}. It comes '
                "off `solver.scattering` after an SMM solve that declared "
                "`answers='amplitude'` or higher.")
        why = _port_disagreement(self.exit_port, beyond.entry_port)
        if why is not None:
            raise ValueError(
                f'cascade() cannot join these two: {why}. This piece\'s exit port and the '
                "next piece's entry port have to be the same basis -- and `beyond` is the "
                'piece the light reaches next, which lit from below is the one geometrically '
                'above this one.')
        xp = array_namespace(self.S11)
        I = xp.eye(self.S11.shape[0], dtype=self.S11.dtype, device=_device_of(self.S11))
        return ScatteringMatrix(*redheffer_star(xp, I, self.blocks, beyond.blocks),
                                entry_port=self.entry_port, exit_port=beyond.exit_port)


def _scattering_matrix(S, backside, kx, ky, n_ref, n_trn, mu_ref, mu_trn, channel):
    """Put the cascade's four blocks into incidence-relative order, with their ports.

    Reversing the 4-tuple is the whole job for the blocks: a cascade of symmetric layer factors
    read from the other end has its blocks in the opposite order, which is why the backside
    efficiency reads above take ``S[3]`` / ``S[1]`` where the top-side ones take ``S[0]`` /
    ``S[2]``. The callers hand in ``n_ref`` / ``n_trn`` and ``mu_ref`` / ``mu_trn``, which are
    already incidence-relative for the same reason, so the ports need no second swap.
    ``channel`` is the same at both ends -- one matrix counts one set of waves -- and the
    reversal cannot move it.
    """
    _register_pytree_if_jax_loaded()
    return ScatteringMatrix(
        *(S[::-1] if backside else S),
        entry_port=Port(kx=kx, ky=ky, index=n_ref, mu=mu_ref, channel=channel),
        exit_port=Port(kx=kx, ky=ky, index=n_trn, mu=mu_trn, channel=channel))


def _coupled_ports_1d(S, sigma, tetm_index):
    """The pure 1D cascade's blocks in the coupled path's s (TE) or p (TM) port variables.

    Two changes of basis, and after them a 1D matrix *is* the s-s or p-p block of the conical
    one at ``phi = 0`` -- the same ports, the same incident column, the same readings.

    * Every port amplitude is quoted along its order's own ``s_hat`` / ``p_hat``, which on the
      ``ky == 0`` row is ``D = diag(sigma)`` of the Cartesian primary, at both ports (the
      two share one ``kx``).
    * The coupled p variable is the p wave's magnetic field, and for an up-going wave it points
      the other way from the down-going one (:func:`_sp_scattering_result`: "the reflected p
      amplitude is reported as its magnetic field, which points the other way for the upward
      wave"), while ``H_y`` does not turn. So a TM block that maps a down-going wave onto an
      up-going one (``S11``) or back (``S22``) takes one more minus.

    Measured on the conical path before this existed, both sides, ``theta = +-0.3``, magnetic
    half-spaces included: TE ``S_c = D S D`` in all four blocks; TM ``S11, S22 = -D S D`` and
    ``S12, S21 = +D S D``, to 8e-15. The backside's block reversal commutes with this map --
    ``D`` is the same at both ports and TM negates ``S11`` and ``S22`` together.

    What follows on either path: ``S11 @ w`` is ``r_s`` for TE and ``-r_p`` for TM, and
    ``S21 @ w`` is ``t_s`` for TE and ``t_p * (n_trn/mu_trn) / (n_ref/mu_ref)`` for TM (port
    units, not Fresnel units) -- ``w`` the incident column, in each order's own frame.
    """
    flip = (-1.0, 1.0, 1.0, -1.0) if tetm_index == 1 else (1.0, 1.0, 1.0, 1.0)
    return tuple(f * (sigma[:, None] * block * sigma[None, :]) for f, block in zip(flip, S))


def _port_disagreement(a, b):
    """Why ports ``a`` and ``b`` are not the same basis, or ``None`` if they are.

    Read eagerly: every comparison here is a value comparison, which a traced array cannot
    answer. That is a property of the question, not a limitation -- "are these the same basis"
    is asked once, when two solves are joined, and never inside a differentiated loop.

    The channel is asked first because it is the one answer no conversion can fix: two pieces
    on different channels are different problems, and a pure piece against a coupled one also
    differs in ``ky`` (``None`` on the pure path), where the ``kx`` / ``ky`` clause would send the
    caller to convert ``theta`` instead.
    """
    if a.channel != b.channel:
        return (f"they count different polarizations: {a.channel!r} ({_CHANNELS[a.channel]}) "
                f"against {b.channel!r} ({_CHANNELS[b.channel]}). A pure 1D matrix joins only "
                "one solved with the same tetm; to join across polarizations, or a pure piece to "
                "a coupled one, solve every piece on the coupled path -- polarization=(a_s, a_p) "
                'or psi, at phi=0 on a 1D grating, instead of tetm -- whose ports carry both')

    def differ(x, y, tol=1e-12):
        if (x is None) != (y is None):
            return True
        if x is None:
            return False
        xa, ya = np.asarray(x), np.asarray(y)
        if xa.shape != ya.shape:
            return True
        return bool(np.max(np.abs(xa - ya)) > tol)

    if differ(a.kx, b.kx) or differ(a.ky, b.ky):
        return ('their retained orders are different in-plane wavevectors. `theta` is measured '
                'in the incidence medium, so the same angle means a different `k_par` on either '
                'side of a material step. Choose the angles for the same transverse '
                'wavevector in both media, and check `order_kxy()` matches before composing. '
                'For lossless positive-index media this is '
                '`theta_b = arcsin(sin(theta_a) * n_a / n_b)`')
    if differ(a.index, b.index) or differ(a.mu, b.mu):
        return ('they open into different half-spaces. The joint medium is a basis rather than '
                'a layer -- any index will do, and a zero-thickness one changes nothing -- but '
                'both pieces have to name the same one')
    return None


# --------------------------------------------------------------------------------- 1D TE / TM

def smm_1d(tetm_index, wavelength, kx, n_top, n_bot, theta, thickness,
           epx_conv_all, epy_conv_all, epz_conv_i_all,
           incidence_side='top', answers=Answers.EFFICIENCY, lit=None,
           mu=None, mu_top=1.0, mu_bot=1.0, eps_top=None, eps_bot=None):
    """Solve a 1D grating stack by scattering matrices — SMM twin of :func:`solve_1d`.

    Same arguments and same returns as :func:`wavebit._core.solve_1d.solve_1d`, so the two are
    interchangeable behind ``layer_algorithm`` — ``mu`` and the half-space terms included, and
    that is the point of carrying them here rather than refusing: the connecting algorithm is
    a choice about how layers are joined, not about what a layer is made of
    (:func:`smm_2d`).

    ``incidence_side='bottom'`` costs nothing here: the cascade composes all four blocks
    whichever side is lit, so backside illumination is *reading* ``S22`` / ``S12`` where the
    top side reads ``S11`` / ``S21``, plus the incidence-medium normalization. That one-pass
    read and the ETM's mirrored-scene solve are the same algebra, not merely the same physics:
    the two interface matrices are block-swaps of each other (``_s_transmission`` returns
    ``_s_reflection``'s blocks in the swapped order, same ``A``/``B``), each layer's matrix is
    symmetric, and reversing a cascade of such factors block-swaps the product. Measured
    against the ETM backside solve at 1e-16 on efficiencies and 1e-14 on field cells
    (``tests/test_backside_incidence.py``), when this branch still served public bottom
    incidence. Since 2026-09-21 the engine sends public *top* incidence here, so the ETM/SMM
    comparisons that reach it are the top-lit ones, ``tests/test_scattering_core.py`` among
    them.

    From ``answers='field'`` up the returned mapping also carries the ``layer_info_list`` /
    ``T1`` state field reconstruction consumes, recovered from the cascade by
    :func:`_field_state` — for the backside, from the mirrored cascade's interface matrices,
    so the state describes the mirrored scene exactly as the ETM path's does. Below that rung
    neither the per-layer matrices nor the recovery pass happen at all.
    """
    answers = answers_of(answers)
    keep_field = answers.covers(Answers.FIELD)
    backside = _is_backside(incidence_side)
    xp, C, kx, _ = _prepare(epx_conv_all, kx)
    mu_all = (None,) * 3 if mu is None else tuple(mu)
    # Each order's s/p frame on this row, the ETM twin's `sigma` (see there).
    sigma = order_frame_sign(xp, kx)

    n_orders_u = kx.shape[0]
    k0 = 2 * _PI / wavelength
    theta = 0.0 if theta is None else theta
    I = xp.astype(xp.eye(n_orders_u), C)

    # Boundary branches come from the same place the ETM path takes them, so the two
    # algorithms answer identically.
    kz_top = boundary_kz(xp, n_top ** 2, kx ** 2)
    kz_bot = boundary_kz(xp, n_bot ** 2, kx ** 2)
    if eps_top is None:
        eps_top = n_top ** 2 / mu_top     # n stays the index; eps carries the mu
    if eps_bot is None:
        eps_bot = n_bot ** 2 / mu_bot

    # Gap medium: eps chosen per order so kz_gap == 1, which keeps V_gap trivially invertible.
    # It stays **non-magnetic on purpose**, the same decision `_smm_sp`'s gap records: it is a
    # bookkeeping device rather than a medium in the scene and it cancels out of the result, so
    # giving it a material property would be a category error that happens to cancel — harder
    # to notice than one that does not.
    eps_gap = 1 + kx ** 2

    # The half-space admittances, and the same pair the ETM twin's `G` is built from: `kz/mu`
    # where the primary is an electric amplitude (TE) and `kz/eps` where it is magnetic (TM).
    # Every one is the expression that was here at mu = 1.
    if tetm_index == 0:      # TE — primary variable E_y, partner H_x
        V_top = -1j * _diag(xp, kz_top / mu_top)
        V_bot = -1j * _diag(xp, kz_bot / mu_bot)
        V_gap = -1j * I
    elif tetm_index == 1:    # TM — primary variable H_y, partner scaled by eps
        V_top = -1j * _diag(xp, kz_top / eps_top)
        V_bot = -1j * _diag(xp, kz_bot / eps_bot)
        V_gap = -1j * _diag(xp, 1 / eps_gap)
    else:
        raise ValueError(f'tetm_index must be 0 (TE) or 1 (TM), got {tetm_index!r}')
    W_top = W_bot = W_gap = I    # homogeneous half-spaces are diagonal in this basis

    # --- cascade: superstrate interface, every layer top-to-bottom, substrate interface ----
    S_ref = _s_reflection(xp, W_top, V_top, W_gap, V_gap)
    S_trn = _s_transmission(xp, W_bot, V_bot, W_gap, V_gap)
    S_layers, AB, modes, muzs = [], [], [], []
    S = S_ref
    for layer_index in range(_n_layers(thickness)):
        layer_mu = _layer_mu(I, mu_all, layer_index)
        W, V, q = layer_modes_1d(xp, tetm_index, kx, I, epx_conv_all[layer_index],
                                 epy_conv_all[layer_index], epz_conv_i_all[layer_index], C,
                                 mu=layer_mu)
        d = thickness[layer_index]
        X = _diag(xp, xp.exp(-k0 * q * d))
        A, B = _ab_matrices(xp, W, V, W_gap, V_gap)
        S_layer = _s_layer(xp, A, B, X)
        S = redheffer_star(xp, I, S, S_layer)
        if keep_field:
            S_layers.append(S_layer)
            AB.append((A, B))
            modes.append([epz_conv_i_all[layer_index], W, V, q, d])
            # `muz_conv_i` travels with its layer for field reconstruction to divide `H_z` by,
            # exactly as `epz_conv_i` leads the entry for `E_z`. Kept in its own list because
            # it rides **last** in the entry, after the two cascade-recovered blocks that are
            # concatenated below — the layout `solve_1d` appends and `field_1d` unpacks.
            # Collected here as well as in the ETM twin because the two build this list by
            # their own routes, and a stack whose `H_z` was magnetic under one algorithm and
            # not the other would have identical `R` and `T` either way.
            muzs.append(None if layer_mu is None else layer_mu[2])
    S = redheffer_star(xp, I, S, S_trn)

    # --- incidence -> amplitudes -> efficiency --------------------------------------------
    # Incidence-relative labels: reflection returns into the incidence medium, transmission
    # leaves through the far side. For the backside these are the substrate-port outputs, so
    # the reads swap to S22 / S12; everything after is the top-side arithmetic with the two
    # half-spaces in incidence-relative roles.
    # The incident column in the Cartesian primary the cascade is written in, each lit order
    # along its own s_hat / p_hat: the same two-branch column as :func:`solve_1d`'s, for the
    # same signed-zero reason.
    column = incident_column(xp, C, lit, n_orders_u, n_orders_u // 2)
    delta = xp.reshape(column * (sigma[n_orders_u // 2] if lit is None else sigma),
                       (n_orders_u, 1))
    R = xp.reshape((S[3] if backside else S[0]) @ delta, (1, n_orders_u))
    Tt = xp.reshape((S[1] if backside else S[2]) @ delta, (1, n_orders_u))

    kz_ref, kz_trn = (kz_bot, kz_top) if backside else (kz_top, kz_bot)
    n_ref, n_trn = (n_bot, n_top) if backside else (n_top, n_bot)
    # The half-space material slots take the same incidence-relative roles the indices do:
    # `ref` is whichever side the light arrives from, `trn` the far one.
    mu_ref, mu_trn = (mu_bot, mu_top) if backside else (mu_top, mu_bot)
    eps_ref, eps_trn = (eps_bot, eps_top) if backside else (eps_top, eps_bot)

    # The same per-wave power definition as the ETM pure-1D path.
    admittance_ref = kz_ref / (mu_ref if tetm_index == 0 else eps_ref)
    admittance_trn = kz_trn / (mu_trn if tetm_index == 0 else eps_trn)
    incoming_power = incident_power(xp, C, lit, admittance_ref)
    de_ri = wave_power(xp, R, admittance_ref) / incoming_power
    de_ti = wave_power(xp, Tt, admittance_trn) / incoming_power
    entering = power_for_real_k(
        xp, xp.sum(net_power(xp, xp.reshape(delta, R.shape), R, admittance_ref))
        / incoming_power, kx)

    check_finite(xp, de_ri=de_ri, de_ti=de_ti)
    check_physical(xp, de_ri, de_ti, n_ref, power_entering=entering)

    if not answers.covers(Answers.AMPLITUDE):
        return de_ri, de_ti, entering

    # `ky` is None: this path carries no out-of-plane order; the channel (`Port`) is what
    # refuses to join a coupled-path port, or the other polarization's. The two
    # permeabilities *are* said, in their incidence-relative roles, because a magnetic
    # half-space changes the port's own admittance -- they were literal `1.0` while this path
    # refused a `mu`.
    # Published in the coupled path's port variables, so a 1D matrix is the s-s (TE) or p-p
    # (TM) block of the conical one at `phi = 0` -- see :func:`_coupled_ports_1d`.
    scattering = _scattering_matrix(_coupled_ports_1d(S, sigma, tetm_index), backside, kx,
                                    None, n_ref, n_trn, mu_ref, mu_trn,
                                    channel='tm' if tetm_index == 1 else 'te')

    # Same Fresnel conversion as the ETM pure path: reflection primaries are already the
    # amplitude; the TM transmission converts H back to E across the index step -- by
    # impedance `Z = mu/n` once a half-space carries a permeability, which is `n_ref/n_trn`
    # to the bit when neither does. Parenthesised as one ratio for that reason. Then each
    # order into its own s/p frame, as there.
    zeros_amp = xp.zeros_like(R)
    zeros_eff = xp.zeros_like(de_ri)
    if tetm_index == 0:
        res = _pack(R * sigma, zeros_amp, Tt * sigma, zeros_amp, de_ri, zeros_eff, de_ti,
                    zeros_eff, power_entering=entering)
    else:
        res = _pack(zeros_amp, R * sigma, zeros_amp,
                    (Tt * ((n_ref / mu_ref) / (n_trn / mu_trn))) * sigma,
                    zeros_eff, de_ri, zeros_eff, de_ti, power_entering=entering)

    if not keep_field:
        return {'incident': res, 'scattering': scattering}

    if backside:
        # The mirrored scene's own cascade pieces, assembled from what this pass already
        # built: the layer matrices are symmetric and the interface matrices come from the
        # same A/B, so only the two half-space interfaces are formed anew. The recovered
        # state then describes the mirrored scene — the same frame the ETM backside solve
        # leaves behind — and field reconstruction un-mirrors once, at the cell level.
        S_ref_m = _s_reflection(xp, W_bot, V_bot, W_gap, V_gap)
        S_trn_m = _s_transmission(xp, W_top, V_top, W_gap, V_gap)
        blocks, T1 = _field_state(xp, I, S_ref_m, S_trn_m, S_layers[::-1], AB[::-1], delta)
        # `muzs` reverses with `modes`: both are per-layer state indexed the same way, and a
        # `mu_zz` left in scene order against mirrored modes gives each layer its neighbour's
        # permeability -- still magnetic, still energy-conserving, `R` and `T` untouched.
        modes, muzs = modes[::-1], muzs[::-1]
    else:
        blocks, T1 = _field_state(xp, I, S_ref, S_trn, S_layers, AB, delta)
    # Bottom-to-top, the order field reconstruction walks in reverse (as the ETM appends it).
    layer_info_list = [m + list(b) + [z]
                       for m, b, z in zip(modes, blocks, muzs)][::-1]
    if T1 is not None:
        T1 = xp.reshape(T1, (n_orders_u,))    # the 1D path carries T1 as a plain vector
        if tetm_index == 1:
            # Unit incident E, as on every other path -- the same rescale, and the same reason,
            # as :func:`~wavebit._core.solve_1d.solve_1d`'s. `n_ref` / `mu_ref` are the
            # incidence medium whichever port is lit. Inside the guard: a stack with no layers
            # has no state to rescale.
            T1 = T1 * (n_ref / mu_ref)
    return {'incident': res, 'layer_info_list': layer_info_list, 'T1': T1,
            'scattering': scattering}


# ------------------------------------------------------------------- coupled s/p (conical, 2D)

def _sp_scattering_result(xp, C, S, kx, ky, kz_top, kz_bot, n_top, n_bot,
                          incidence, backside, answers, mu_top, mu_bot,
                          eps_top, eps_bot, lit=None):
    """Public s/p amplitudes and powers from a cascade in scalar half-space bases.

    Shared by the two-field and four-field SMM paths. Also return the incident
    columns so field reconstruction uses precisely the same normalization.
    """
    keep_amplitude = answers_of(answers).covers(Answers.AMPLITUDE)
    kx_g, _ = order_grid_kxy(xp, kx, ky)
    n_orders_v, n_orders_u = kx_g.shape
    n_orders = n_orders_u * n_orders_v
    kz_ref, kz_trn = (kz_bot, kz_top) if backside else (kz_top, kz_bot)
    n_ref, n_trn = (n_bot, n_top) if backside else (n_top, n_bot)
    mu_ref, mu_trn = (mu_bot, mu_top) if backside else (mu_top, mu_bot)
    eps_ref, eps_trn = (eps_bot, eps_top) if backside else (eps_top, eps_bot)

    # --- incidence ------------------------------------------------------------------------
    # The scalar half-space basis uses (E_s, H_p), so the incident column is the
    # incident field itself: E_s = a_s, H_p = -i*n_ref/mu_ref*a_p in the incidence medium — with
    # (a_s, a_p) as resolved at the boundary — one form arrives here, and it is the same
    # amplitudes the ETM final matching solves against (see conventions.incident_sp_amplitudes).
    delta = xp.reshape(incident_column(xp, C, lit, n_orders, n_orders // 2), (n_orders, 1))

    def incident(a_s, a_p):
        # Convert the electric p coefficient to the magnetic boundary variable.
        # The actual s/p input powers are evaluated separately below.
        return xp.concat([a_s * delta, -1j * (n_ref / mu_ref) * a_p * delta], axis=0)

    def incident_psi(psi_v):
        return incident(xp.sin(xp.asarray(psi_v, dtype=C)),
                        xp.cos(xp.asarray(psi_v, dtype=C)))

    # The incidence as asked, then the pure-TE / pure-TM variants (free: same scattering
    # matrix). The variants stay spelled as angles so their columns are bit-identical to the
    # pre-`polarization` ones.
    cols = ([incident(as_array(xp, incidence[0], dtype=C),
                      as_array(xp, incidence[1], dtype=C))]
            + ([incident_psi(_PI / 2), incident_psi(0.0)] if keep_amplitude else []))
    c_inc = xp.concat(cols, axis=1)

    c_ref = (S[3] if backside else S[0]) @ c_inc
    c_trn = (S[1] if backside else S[2]) @ c_inc

    n_inc = len(cols)

    def as_orders(M):
        """(n_orders, n_inc) amplitude columns -> (n_inc, n_orders_v, n_orders_u)."""
        return xp.reshape(xp.matrix_transpose(M), (n_inc, n_orders_v, n_orders_u))

    # The reflected p amplitude is reported as its magnetic field, which points the other way
    # for the upward wave; s and the transmitted pair need no such flip.
    R_s = as_orders(c_ref[:n_orders, :])
    R_p = as_orders(-c_ref[n_orders:, :])
    T_s = as_orders(c_trn[:n_orders, :])
    T_p = as_orders(c_trn[n_orders:, :])

    kz_ref_2d = xp.reshape(kz_ref, (n_orders_v, n_orders_u))
    kz_trn_2d = xp.reshape(kz_trn, (n_orders_v, n_orders_u))
    s_power, p_power = sp_incident_power(xp, C, lit, kz_ref, n_ref, eps_ref, mu_ref)
    a_s = as_array(xp, incidence[0], dtype=C)
    a_p = as_array(xp, incidence[1], dtype=C)
    powers = [xp.abs(a_s)**2 * s_power + xp.abs(a_p)**2 * p_power]
    if keep_amplitude:
        powers.extend([s_power, p_power])
    denom = xp.reshape(xp.stack(powers), (n_inc, 1, 1))
    eff = sp_efficiencies(xp, kz_ref_2d, kz_trn_2d, denom, n_ref, n_trn, R_s, R_p, T_s, T_p,
                          mu_top=mu_ref, mu_bot=mu_trn, eps_top=eps_ref, eps_bot=eps_trn)
    flux = (net_power(xp, as_orders(c_inc[:n_orders]), R_s, kz_ref_2d / mu_ref)
            + net_power(xp, as_orders(c_inc[n_orders:]), R_p, kz_ref_2d / eps_ref))
    entering = power_for_real_k(
        xp, xp.sum(flux, axis=(-2, -1)) / xp.stack(powers), kx, ky)

    check_finite(xp, de_ri=eff[0][0] + eff[1][0], de_ti=eff[2][0] + eff[3][0])
    check_physical(xp, eff[0][0] + eff[1][0], eff[2][0] + eff[3][0], n_ref,
                   power_entering=entering[0])

    if not keep_amplitude:
        return (eff[0][0] + eff[1][0], eff[2][0] + eff[3][0], entering[0]), c_inc

    scattering = _scattering_matrix(S, backside, kx, ky, n_ref, n_trn, mu_ref, mu_trn,
                                    channel='sp')

    # Coupled-path Fresnel conversion, identical to the ETM final matching's.
    amps = (R_s, R_p / (-1j * n_ref / mu_ref), T_s, T_p / (-1j * n_trn / mu_trn))
    packed = [_pack(*(a[i] for a in amps), *(e[i] for e in eff),
                    power_entering=entering[i]) for i in range(3)]

    return {'incident': packed[0], 's_inc': packed[1], 'p_inc': packed[2],
            'scattering': scattering}, c_inc


def _smm_sp(xp, C, wavelength, kx, ky, n_top, n_bot, theta, incidence, thickness,
            layer_modes, epz_conv_i_all, backside, answers, mu_top=1.0, mu_bot=1.0,
            eps_top=None, eps_bot=None, muz_conv_i_all=None, lit=None):
    """Shared SMM cascade + matching for the coupled s/p paths (conical and 2D).

    ``layer_modes(layer_index)`` returns that layer's ``(big_W, big_V, q, W, V)``: the mode
    matrices in the s/p boundary basis that the cascade joins layers through, then the raw
    (unrotated) pair that field reconstruction expands the same amplitudes with. Everything
    else — half-spaces, gap, cascade, incidence, efficiencies — is common.

    ``muz_conv_i_all`` is the per-layer ``mu_zz^-1``, carried for one reason: field
    reconstruction recovers ``H_z`` algebraically from it, exactly as it recovers ``E_z`` from
    ``epz_conv_i_all``, and the cascade itself never looks at either. Absent on the conical
    caller, which has no permeability at all. Dropping it here would leave the SMM answering a
    magnetic stack's ``H_z`` non-magnetically while the ETM twin answered it correctly — the
    one asymmetry :func:`smm_2d`'s docstring exists to rule out.

    ``backside`` selects the substrate-port reads (see :func:`smm_1d` for why that read *is*
    the mirrored-scene solve): the incident column is built with the incidence medium's index,
    reflection comes out of ``S22`` and transmission out of ``S12``, and the p sign flip stays
    with the reflection port — in the mirror's frame the reflected wave is the upward one,
    whichever scene medium it returns into.

    ``incidence`` is the resolved electric coefficient pair ``(a_s, a_p)`` of unit norm — see
    :func:`~wavebit._core.conventions.incident_sp_amplitudes`.

    ``answers`` is the rung this solve was declared for; it decides how many incidence
    columns the cascade is asked about (one, or three once the pure-TE/TM variants are
    wanted) and whether the per-layer pieces are kept for :func:`_field_state` to walk.
    """
    answers = answers_of(answers)
    keep_field = answers.covers(Answers.FIELD)
    kx_g, ky_g = order_grid_kxy(xp, kx, ky)
    n_orders_v, n_orders_u = kx_g.shape
    n_orders = n_orders_u * n_orders_v
    k0 = 2 * _PI / wavelength
    theta = 0.0 if theta is None else theta

    I = xp.astype(xp.eye(n_orders), C)
    O = xp.astype(xp.zeros((n_orders, n_orders)), C)
    big_I = xp.astype(xp.eye(2 * n_orders), C)

    k_par_sq = kx_g ** 2 + ky_g ** 2
    kz_top = xp.reshape(boundary_kz(xp, n_top ** 2, k_par_sq), (-1,))
    kz_bot = xp.reshape(boundary_kz(xp, n_bot ** 2, k_par_sq), (-1,))
    # `n` stays the refractive index and `eps` carries the `mu`, so the two are recoverable
    # from each other -- but only up to rounding, since the index reached this call through a
    # square root. The engine holds both numbers and passes them; a caller that holds only an
    # index leaves these `None` and gets the recovery, which is what this always did.
    if eps_top is None:
        eps_top = n_top ** 2 / mu_top
    if eps_bot is None:
        eps_bot = n_bot ** 2 / mu_bot
    eps_gap = xp.reshape(1 + k_par_sq, (-1,))                 # kz_gap == 1 by construction

    Kz_top = _diag(xp, kz_top)
    Kz_bot = _diag(xp, kz_bot)

    # Half-space and gap modes in the (E_s, F_p) / (H_s, H_p) boundary basis — the same form
    # the ETM substrate seed (big_F, big_G) has, and the same two admittances: `kz/eps` on the
    # magnetic (p) amplitude, `kz/mu` on the electric (s) one.
    #
    # The **gap is left non-magnetic**, deliberately. It is not a medium in the scene; it is a
    # fictitious layer the cascade inserts so every interface is computed the same way, and it
    # cancels out of the result. Giving a bookkeeping device a material property would be a
    # category error that happens to cancel, which is worse than one that does not.
    W_top = _block(xp, [[I, O], [O, -1j * Kz_top / eps_top]])
    V_top = _block(xp, [[-1j * Kz_top / mu_top, O], [O, I]])
    W_bot = _block(xp, [[I, O], [O, -1j * Kz_bot / eps_bot]])
    V_bot = _block(xp, [[-1j * Kz_bot / mu_bot, O], [O, I]])
    W_gap = _block(xp, [[I, O], [O, -1j * _diag(xp, 1 / eps_gap)]])
    V_gap = _block(xp, [[-1j * I, O], [O, I]])

    S_ref = _s_reflection(xp, W_top, V_top, W_gap, V_gap)
    S_trn = _s_transmission(xp, W_bot, V_bot, W_gap, V_gap)
    S_layers, AB, modes, muz_conv = [], [], [], []
    S = S_ref
    for layer_index in range(_n_layers(thickness)):
        big_W, big_V, q, W, V = layer_modes(layer_index)
        d = thickness[layer_index]
        big_X = _diag(xp, xp.exp(-k0 * q * d))
        A, B = _ab_matrices(xp, big_W, big_V, W_gap, V_gap)
        S_layer = _s_layer(xp, A, B, big_X)
        S = redheffer_star(xp, big_I, S, S_layer)
        if keep_field:
            S_layers.append(S_layer)
            AB.append((A, B))
            modes.append([epz_conv_i_all[layer_index], W, V, q, d])
            muz_conv.append(None if muz_conv_i_all is None else muz_conv_i_all[layer_index])
    S = redheffer_star(xp, big_I, S, S_trn)

    out, c_inc = _sp_scattering_result(
        xp, C, S, kx, ky, kz_top, kz_bot, n_top, n_bot, incidence, backside,
        answers, mu_top, mu_bot, eps_top, eps_bot, lit)
    if not keep_field:
        return out
    n_inc = c_inc.shape[1]

    if backside:
        # Mirrored-scene field state, same construction as `smm_1d`'s backside branch.
        S_ref_m = _s_reflection(xp, W_bot, V_bot, W_gap, V_gap)
        S_trn_m = _s_transmission(xp, W_top, V_top, W_gap, V_gap)
        blocks, T1 = _field_state(xp, big_I, S_ref_m, S_trn_m,
                                  S_layers[::-1], AB[::-1], c_inc)
        modes = modes[::-1]
        muz_conv = muz_conv[::-1]
    else:
        blocks, T1 = _field_state(xp, big_I, S_ref, S_trn, S_layers, AB, c_inc)
    # `muz_conv_i` last, after the transfer blocks, so slots 0..6 stay where the ETM twin
    # puts them: `tests/test_scattering_core.py` compares the two algorithms' `A_i` / `B` by index,
    # and the legacy goldens name the first seven.
    layer_info_list = [m + list(b) + [z]
                       for m, b, z in zip(modes, blocks, muz_conv)][::-1]
    if T1 is not None:
        # One column per incidence variant (psi, TE, TM), as the ETM stacks them.
        T1 = xp.stack([T1[:, i:i + 1] for i in range(n_inc)])
    out.update(layer_info_list=layer_info_list, T1=T1)
    return out


def smm_1d_conical(wavelength, kx, ky, n_top, n_bot, theta, incidence, thickness,
                   epx_conv_all, epy_conv_all, epz_conv_i_all,
                   *, phi, incidence_side='top', answers=Answers.EFFICIENCY, lit=None,
                   mu=None, mu_top=1.0, mu_bot=1.0, eps_top=None, eps_bot=None):
    """Solve a conical 1D grating by scattering matrices — SMM twin of :func:`solve_1d_conical`.

    Same arguments and returns as :func:`smm_1d`; ``polarization`` is the complex amplitude
    pair ``(a_s, a_p)``, resolved at the boundary as in the ETM twin.

    ``mu`` is the three permeability stacks
    (:func:`~wavebit._core.solve_1d.layer_modes_1d_conical`), handed to the *same* layer-mode
    function the ETM twin calls, and the half-space terms go to the shared :func:`_smm_sp`.
    Carried here rather than refused for the reason :func:`smm_2d`'s docstring gives: which
    connecting algorithm joins the layers is not a fact about what a layer is made of, and an
    algorithm that quietly dropped a ``mu`` would answer a magnetic stack plausibly and
    wrongly while its twin answered it correctly.
    """
    backside = _is_backside(incidence_side)
    xp, C, kx, ky = _prepare(epx_conv_all, kx, ky)
    mu_all = (None,) * 3 if mu is None else tuple(mu)

    n_orders_u = kx.shape[0]
    I = xp.astype(xp.eye(n_orders_u), C)
    ky_col = xp.reshape(ky, (-1, 1))
    varphi = order_azimuth(xp, ky_col, kx, phi)

    Kx = _diag(xp, kx)
    ky_vec = ky[0] * xp.astype(xp.ones((n_orders_u,)), C)      # n_orders_v == 1: Ky = diag(ky0 * 1)
    Ky = _diag(xp, ky_vec)
    Kx2 = _diag(xp, kx * kx)
    Ky2 = _diag(xp, ky_vec * ky_vec)
    F_c = _diag(xp, xp.cos(varphi))
    F_s = _diag(xp, xp.sin(varphi))

    def layer_modes(layer_index):
        W1, W2, q1, q2, V_11, V_12, V_21, V_22 = layer_modes_1d_conical(
            xp, Kx, Ky, Kx2, Ky2, I, epx_conv_all[layer_index],
            epy_conv_all[layer_index], epz_conv_i_all[layer_index], C,
            mu=_layer_mu(I, mu_all, layer_index))
        big_W, big_V = sp_modes_1d_conical(xp, F_c, F_s, W1, W2, V_11, V_12, V_21, V_22)
        W = _block(xp, [[W1, W2]])
        V = _block(xp, [[V_11, V_12], [V_21, V_22]])
        return big_W, big_V, xp.concat([q1, q2]), W, V

    return _smm_sp(xp, C, wavelength, kx, ky, n_top, n_bot, theta, incidence, thickness,
                   layer_modes, epz_conv_i_all, backside, answers,
                   mu_top=mu_top, mu_bot=mu_bot, eps_top=eps_top, eps_bot=eps_bot,
                   muz_conv_i_all=mu_all[2], lit=lit)


def smm_2d(wavelength, kx, ky, n_top, n_bot, theta, incidence, thickness,
           epx_conv_all, epy_conv_all, epz_conv_i_all,
           epxy_conv_all=None, epyx_conv_all=None,
           *, phi, incidence_side='top', answers=Answers.EFFICIENCY, lit=None,
           mu=None, mu_top=1.0, mu_bot=1.0, eps_top=None, eps_bot=None):
    """Solve a 2D grating stack by scattering matrices — SMM twin of :func:`solve_2d`.

    Same arguments and returns as :func:`smm_1d`; ``incidence`` is the resolved complex
    amplitude pair ``(a_s, a_p)``, as in the ETM twin. ``mu`` likewise: the five permeability
    stacks, handed to the same :func:`~wavebit._core.solve_2d.layer_modes_2d`.

    Carried here rather than refused because the connecting algorithm is a choice about how
    layers are joined, not about what a layer is made of. A path that accepted ``mu`` under
    ``'ETM'`` and dropped it under ``'SMM'`` would return a plausible non-magnetic answer for a
    magnetic stack, and the two algorithms agreeing is one of the things the suite checks.

    ``eps_top`` / ``eps_bot`` are the half-spaces' permittivities, as in :func:`solve_2d`:
    optional, recovered as ``n**2/mu`` when absent, and supplied together with the ``mu``
    pair by callers that hold both.
    """
    backside = _is_backside(incidence_side)
    xp, C, kx, ky = _prepare(epx_conv_all, kx, ky)

    kx_g, ky_g = order_grid_kxy(xp, kx, ky)
    n_orders_v, n_orders_u = kx_g.shape
    n_orders = n_orders_u * n_orders_v
    I = xp.astype(xp.eye(n_orders), C)
    O = xp.astype(xp.zeros((n_orders, n_orders)), C)
    varphi = order_azimuth(xp, ky_g, kx_g, phi)
    mu_all = (None,) * 5 if mu is None else tuple(mu)

    Kx = _diag(xp, xp.reshape(kx_g, (-1,)))
    Ky = _diag(xp, xp.reshape(ky_g, (-1,)))
    F_c, F_s = xp.cos(varphi), xp.sin(varphi)

    def layer_modes(layer_index):
        epxy_conv = O if epxy_conv_all is None else epxy_conv_all[layer_index]
        epyx_conv = O if epyx_conv_all is None else epyx_conv_all[layer_index]
        layer_mu = None if all(t is None for t in mu_all) else (
            I if mu_all[0] is None else mu_all[0][layer_index],
            I if mu_all[1] is None else mu_all[1][layer_index],
            *(None if t is None else t[layer_index] for t in mu_all[2:]))
        W, V, q = layer_modes_2d(xp, Kx, Ky, I, epx_conv_all[layer_index],
                                 epy_conv_all[layer_index], epz_conv_i_all[layer_index],
                                 epxy_conv, epyx_conv, C, mu=layer_mu)
        big_W, big_V = sp_modes_2d(xp, W, V, F_c, F_s, n_orders)
        return big_W, big_V, q, W, V

    return _smm_sp(xp, C, wavelength, kx, ky, n_top, n_bot, theta, incidence, thickness,
                   layer_modes, epz_conv_i_all, backside, answers,
                   mu_top=mu_top, mu_bot=mu_bot, eps_top=eps_top, eps_bot=eps_bot,
                   muz_conv_i_all=mu_all[2], lit=lit)
