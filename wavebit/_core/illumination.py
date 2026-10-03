"""Coherent external illumination and prescribed internal electric sources.

The scattering problem is solved once per retained Bloch sector. Physical
Cartesian fields, including their spatial phases, are combined before power
is evaluated. H uses the same normalization as reconstruct_field: Z0 H_SI.
"""
from dataclasses import dataclass, fields

import numpy as np

from .backend import GainCutoffWarning, _concrete, as_array, check_finite
from .enums import Answers, answers_of
from .field_result import ResultFields, source_state
from .conventions import boundary_kz, gain_cutoff_band, parse_excitation
from .engine import _GAIN_CUTOFF_MESSAGE, _unknown_attribute, _holds_tracer
from .sources import (Source, PlaneWave, OrderExcitation, GaussianBeam, ElectricDipole, source_list, _positive)
from .source_expansion import resolve_wave, gaussian_waves, dipole_layers
from .scattering import redheffer_star
from .solve_2d_4ff import (
    _conv_layers, _interface_s, _propagate_s, layer_matrix_4ff,
    split_forward_backward,
)


def _source_holds_tracer(value):
    """Inspect typed source declarations as well as ordinary condition containers."""
    stack = [value]
    while stack:
        item = stack.pop()
        if _holds_tracer(item):
            return True
        if isinstance(item, Source):
            stack.extend(getattr(item, field.name) for field in fields(item))
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return False


def _host(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    try:
        return np.asarray(value)
    except Exception as exc:
        raise ValueError('Source grouping needs concrete directions. Construct and solve '
                         'sources eagerly, outside jit/compile.') from exc


def _plane_basis(xp, kx, ky, epsilon, mu, direction):
    """Electric s/p columns as physical (Ex,Ey,Ez,Z0Hx,Z0Hy,Z0Hz)."""
    kz = boundary_kz(xp, epsilon * mu, kx*kx + ky*ky)
    index = xp.sqrt(epsilon * mu + 0j)
    rho = xp.sqrt(xp.real(kx)**2 + xp.real(ky)**2)
    safe = xp.where(rho == 0, xp.ones_like(rho), rho)
    c = xp.where(rho == 0, xp.ones_like(rho), kx / safe)
    s = xp.where(rho == 0, xp.zeros_like(rho), ky / safe)
    zero = xp.zeros_like(kx)
    es = xp.stack([-s, c, zero], axis=-1)
    ep = xp.stack([-kz*c/index, -kz*s/index, direction*rho/index], axis=-1)
    k = xp.stack([kx, ky, direction*kz], axis=-1)
    hs, hp = xp.linalg.cross(k, es) / mu, xp.linalg.cross(k, ep) / mu
    return xp.stack([xp.concat([es, hs], axis=-1),
                     xp.concat([ep, hp], axis=-1)], axis=-1)


def _tangential(xp, basis):
    n = basis.shape[0]
    rows = []
    for component in (0, 1, 3, 4):
        rows.append(xp.concat([xp.eye(n, dtype=basis.dtype, device=getattr(basis, 'device', None))
                               * basis[:, component, p][None, :] for p in range(2)], axis=1))
    return xp.concat(rows, axis=0)


def _affine_star(xp, I, a, b):
    """Cascade scattering maps together with their source-generated outputs."""
    A, f = a
    B, g = b
    middle = xp.linalg.solve(I - A[3] @ B[0], A[3] @ g[0] + f[1])
    return redheffer_star(xp, I, A, B), (f[0] + A[1] @ (B[0] @ middle + g[0]),
                                        g[1] + B[2] @ middle)


def _cascade(xp, k0, kx, ky, layers, bottom, top, keep_field, jumps=None):
    """Stable two-port cascade with optional prescribed tangential-field jumps."""
    n = kx.shape[0]
    Kx, Ky = xp.diag(kx), xp.diag(ky)
    I = xp.eye(2*n, dtype=kx.dtype, device=getattr(kx, 'device', None))
    zero = xp.zeros((2*n,), dtype=kx.dtype, device=getattr(kx, 'device', None))
    parts, modes = [], []
    prev_f, prev_b = bottom
    jumps = {} if jumps is None else jumps
    # Splitting a host layer at source heights reuses its eigensystem.
    cache = {}
    for j, (material, d) in enumerate(layers):
        if id(material) not in cache:
            cache[id(material)] = split_forward_backward(
                xp, layer_matrix_4ff(xp, Kx, Ky, **material), n)
        lf, pf, lb, pb = cache[id(material)]
        emitted = (zero, zero)
        if j in jumps:
            v = xp.linalg.solve(xp.concat([prev_b, -pf], axis=1), -jumps[j])
            emitted = (v[:2*n], v[2*n:])
        parts.extend([(_interface_s(xp, prev_f, prev_b, pf, pb), emitted),
                      (_propagate_s(xp, lf, lb, k0, d, n), (zero, zero))])
        modes.append((material, d, lf, pf, lb, pb))
        prev_f, prev_b = pf, pb
    parts.append((_interface_s(xp, prev_f, prev_b, *top), (zero, zero)))
    suffix = [None] * len(parts)
    suffix[-1] = parts[-1]
    for j in range(len(parts)-2, -1, -1):
        suffix[j] = _affine_star(xp, I, parts[j], suffix[j+1])
    return (*suffix[0], (parts, suffix, modes, I) if keep_field else None)


def _inside_amplitudes(xp, state, incoming_bottom, incoming_top):
    parts, suffix, modes, I = state
    forward = incoming_bottom
    records, entrance = [], None
    for j, (part, source) in enumerate(parts[:-1]):
        rest, emission = suffix[j+1]
        forward = xp.linalg.solve(I - part[3] @ rest[0],
                                  part[2] @ forward + part[3] @ (rest[1] @ incoming_top + emission[0])
                                  + source[1])
        backward = rest[0] @ forward + rest[1] @ incoming_top + emission[0]
        if j % 2 == 0:
            entrance = forward
        else:
            records.append((*modes[j//2], entrance, backward))
    return records


@dataclass(frozen=True)
class BoundaryPower:
    """Area-averaged fluxes, with the common 1/Z0 factor omitted.

    Without internal sources, output_top/bottom are fractions of incident-only power. net_into_stack
    includes incident/output interference in absorbing half-spaces. Absorption
    is supplied only for a periodic-cell or infinite-area average, where the
    lateral flux cancels. Output fractions and absorption are None when internal
    sources are present; outgoing_top/bottom still give the unnormalized fluxes.
    Absorption is also None for a finite observation rectangle.
    """
    incoming_top: object
    incoming_bottom: object
    outgoing_top: object
    outgoing_bottom: object
    net_into_stack: object
    region: object
    absorption: object = None
    has_internal_sources: bool = False

    @property
    def incident(self):
        return self.incoming_top + self.incoming_bottom

    @property
    def output_top(self):
        return None if self.has_internal_sources else self.outgoing_top / self.incident

    @property
    def output_bottom(self):
        return None if self.has_internal_sources else self.outgoing_bottom / self.incident


@dataclass
class _Spectrum:
    kxy: object
    field: object


def _sinc(xp, value):
    small = xp.abs(value) < 1e-5
    safe = xp.where(small, xp.ones_like(value), value)
    return xp.where(small, 1 - value**2/6 + value**4/120, xp.sin(safe)/safe)


class CoherentResult(ResultFields):
    """Power and retained fields from one wavelength.

    ``efficiency`` retains the computed power only; ``amplitude`` also retains
    boundary fields and supports power integration over another region;
    ``field`` additionally retains the per-layer reconstruction state.
    """

    def __init__(self, xp, spectra, groups, lattice, wavelength, thickness, region,
                 has_internal_sources=False, localized_sources=False,
                 answers=Answers.AMPLITUDE):
        answers = answers_of(answers)
        self._xp, self._spectra = xp, spectra
        self.lattice, self.wavelength, self.thickness = lattice, wavelength, thickness
        self.has_internal_sources = has_internal_sources
        self.localized_sources = localized_sources
        self.group_count = len(groups)
        # All rungs use the coherent spectra to evaluate power. Only the higher
        # rungs retain those spectra after this calculation.
        self.power = self.boundary_power(region)
        self._groups = groups if answers.covers(Answers.FIELD) else ()
        self._field_state = None
        if answers.covers(Answers.AMPLITUDE):
            self._field_state = source_state(spectra, self._groups, lattice, wavelength, thickness)
        else:
            self._spectra = None

    @property
    def output_top(self):
        return self.power.output_top

    @property
    def output_bottom(self):
        return self.power.output_bottom

    @property
    def absorption(self):
        return self.power.absorption

    def _spectrum(self, side, part):
        if side not in ('top', 'bottom') or part not in ('incoming', 'outgoing', 'total'):
            raise ValueError("Use side='top'/'bottom', part='incoming'/'outgoing'/'total'.")
        if part != 'total':
            return self._spectra[side, part]
        a, b = (self._spectra[side, p] for p in ('incoming', 'outgoing'))
        return _Spectrum(a.kxy, a.field + b.field)

    def boundary_power(self, region=None):
        """Integrate coherently: 'cell', 'infinite', or ((xmin,xmax),(ymin,ymax)).

        Return mean fluxes, not watts. The common vacuum-impedance scale cancels
        from output fractions. A finite rectangle's total flux is its mean times
        area. Multiple Bloch sectors require an explicit observation region.

        Reintegrating requires answers='amplitude' or 'field' at solve time.
        The power computed during an efficiency-only solve remains in ``power``.
        """
        if self._spectra is None:
            raise RuntimeError(
                "Reintegrating boundary power requires solve(answers='amplitude') "
                "or answers='field'. Read result.power for the power already computed.")
        xp = self._xp
        if self.localized_sources and (region is None or isinstance(region, str)):
            raise ValueError('GaussianBeam and isolated dipoles require a finite rectangle power_region.')
        if region is None:
            if self.group_count != 1:
                raise ValueError("Different Bloch sectors require power_region='infinite', "
                                 "'cell', or ((xmin,xmax),(ymin,ymax)).")
            region = 'cell'
        k = self._spectra['top', 'outgoing'].kxy
        dk = k[:, None, :] - k[None, :, :]
        if isinstance(region, str):
            if region == 'cell':
                a = as_array(xp, self.lattice, dtype=k.dtype, device=getattr(k, 'device', None))
                phase = dk @ a
                weight = _sinc(xp, phase[..., 0]/2) * _sinc(xp, phase[..., 1]/2)
            elif region == 'infinite':
                tol = 128*np.finfo(_host(xp.real(k)).dtype).eps * max(
                    float(abs(2*np.pi/_host(self.wavelength))), float(np.max(np.abs(_host(k)))))
                weight = xp.astype(xp.all(xp.abs(dk) <= tol, axis=-1), k.dtype)
            else:
                raise ValueError("power_region must be 'cell', 'infinite', or a rectangle")
        else:
            box = np.asarray(region, dtype=float)
            if box.shape != (2, 2) or not np.all(np.isfinite(box)) or np.any(box[:, 1] <= box[:, 0]):
                raise ValueError('power_region must be ((xmin,xmax),(ymin,ymax)) with positive widths')
            centre, width = box.mean(axis=1), np.diff(box, axis=1)[:, 0]
            weight = (xp.exp(1j*(dk[..., 0]*centre[0] + dk[..., 1]*centre[1]))
                      * _sinc(xp, dk[..., 0]*width[0]/2) * _sinc(xp, dk[..., 1]*width[1]/2))

        def flux(side, part):
            f = self._spectrum(side, part).field
            product = (f[:, 0, None] * xp.conj(f[None, :, 4])
                       - f[:, 1, None] * xp.conj(f[None, :, 3]))
            return .5*xp.real(xp.sum(product * weight))

        it, ib = -flux('top', 'incoming'), flux('bottom', 'incoming')
        ot, ob = flux('top', 'outgoing'), -flux('bottom', 'outgoing')
        incident = it + ib
        if not bool(xp.isfinite(incident)) or (not self.has_internal_sources and bool(incident <= 0)):
            raise ValueError('The combined incoming field must carry positive normal power in the observation region.')
        net = flux('bottom', 'total') - flux('top', 'total')
        periodic = region == 'infinite' if isinstance(region, str) else False
        periodic = periodic or (isinstance(region, str) and region == 'cell' and self.group_count == 1)
        return BoundaryPower(it, ib, ot, ob, net, region,
                             net/incident if periodic and not self.has_internal_sources else None,
                             self.has_internal_sources)

class CoherentSolver:
    """Eager, side-resolved illumination using the Cartesian 4ff SMM.

    NumPy, JAX and PyTorch arrays are supported. Grouping is a concrete setup
    operation; the facade is not a JAX pytree and cannot be passed through jit.
    Ordinary one-sided rcwa calls continue using their existing solver paths.
    """

    _resolve_wave = resolve_wave
    _gaussian_waves = gaussian_waves
    _dipole_layers = dipole_layers

    #: The scene's incidence conditions. Each source states its own, so the facade refuses
    #: them at the attribute door and at `solve`, by name.
    _SOURCE_INPUTS = ('theta', 'phi', 'incidence_side', 'tetm', 'kx', 'ky',
                      'polarization', 'psi', 'chi')
    #: The facade's own public state. Every other public name is the scene's or is refused.
    _OWN = ('excitation', 'power_region', 'length_unit')

    def __init__(self, backend, args, kwargs):
        from ..main import rcwa
        kwargs = dict(kwargs)
        self.excitation = kwargs.pop('excitation')
        self.power_region = kwargs.pop('power_region', None)
        self.length_unit = kwargs.pop('length_unit', None)
        if self.length_unit is not None:
            _positive(self.length_unit, 'length_unit')
        if args:
            raise TypeError('Side-resolved excitation requires keyword arguments')
        if 'incidence_side' in kwargs:
            raise ValueError('Specify the side in excitation; omit incidence_side.')
        if any(kwargs.get(k) is not None for k in ('theta', 'phi', 'tetm')):
            raise ValueError('Put theta/phi on each PlaneWave; use its polarization instead of tetm.')
        if kwargs.get('layer_algorithm', 'SMM') != 'SMM':
            raise ValueError("Side-resolved excitation uses layer_algorithm='SMM'.")
        kwargs['layer_algorithm'] = 'SMM'
        if kwargs.get('formulation', '4ff') not in ('auto', '4ff'):
            raise ValueError("Side-resolved excitation uses the Cartesian '4ff' formulation.")
        kwargs.pop('formulation', None)
        self._defaults = {k: kwargs.pop(k, None) for k in ('kx', 'ky', 'polarization', 'psi', 'chi')}
        if all(self._defaults[k] is None for k in ('polarization', 'psi', 'chi')):
            self._defaults['polarization'] = (1., 0.)
        self._scene = rcwa(backend=backend, **kwargs)
        self._scene.formulation = '4ff'
        self._backend = backend
        self._result = None
        source_list(self.excitation)

    @staticmethod
    def _forwards(scene, name):
        """Whether `name` is read and written on the scene: a condition or a property of it.

        Conditions come from the solver's own `_condition_names`, so an option the solver
        gains is forwarded here without a second list; the property branch keeps the scene's
        read-only ones (`grating_type`, `n_top`) raising Python's own message.
        """
        return (name in scene._condition_names()
                or isinstance(getattr(type(scene), name, None), property))

    def __getattr__(self, name):
        scene = self.__dict__.get('_scene')
        if scene is not None and self._forwards(scene, name):
            return getattr(scene, name)
        raise AttributeError(name)

    def __setattr__(self, name, value):
        """Forward a scene condition, keep the facade's own state, refuse any other public name.

        The last branch used to store the name on the facade, where nothing reads it:
        `s.thicknes = [900.0]` and the retired `s.nesting = 'symmetric'` were accepted and the
        solve answered for the scene as it was -- the hole the solver's own guard
        (`CoreFactorizationMixin.__setattr__`) closed on 2026-08-27 and this door kept open.
        The refusal is the solver's, `engine._unknown_attribute`, so both doors raise the same
        `AttributeError` with the same near-match hint. A private name and a name on this
        class still land on the facade: the first is the facade's own slots, the second keeps
        the descriptor protocol in charge (`backend` has no setter, `solve` can be stubbed).
        """
        scene = self.__dict__.get('_scene')
        if scene is not None and name in self._SOURCE_INPUTS:
            raise ValueError(f'Put {name} in the excitation source inputs.')
        if name == 'layer_algorithm' and value != 'SMM':
            raise ValueError("Side-resolved excitation uses layer_algorithm='SMM'.")
        if name == 'formulation' and value != '4ff':
            raise ValueError("Side-resolved excitation uses formulation='4ff'.")
        if name in self._OWN:
            object.__setattr__(self, name, value)
        elif scene is not None and self._forwards(scene, name):
            setattr(scene, name, value)
        elif name.startswith('_') or hasattr(type(self), name):
            object.__setattr__(self, name, value)
        else:
            settable = set(self._OWN)
            if scene is not None:
                settable |= scene._condition_names() - set(self._SOURCE_INPUTS)
            raise _unknown_attribute(type(self).__name__, name, settable)

    @property
    def formulation_assigned(self):
        return '4ff'

    @property
    def backend(self):
        return self._backend

    def solve(self, *, excitation=None, power_region=None, geometry=None, **conditions):
        scene = self._scene
        traced_scene = (any(_holds_tracer(value) for value in scene.__dict__.values())
                        or _source_holds_tracer(self.excitation))
        updates = {name: value for name, value in
                   (('excitation', excitation), ('power_region', power_region)) if value is not None}
        held = {name: getattr(self, name) for name, value in updates.items()
                if _source_holds_tracer(value)}
        try:
            for name, value in updates.items():
                setattr(self, name, value)
            for name in self._SOURCE_INPUTS:
                if name in conditions:
                    raise ValueError(f'Put {name} in each PlaneWave, or rebuild with common kx/ky.')
            for name, expected in (('formulation', '4ff'), ('layer_algorithm', 'SMM')):
                if name in conditions and conditions.pop(name) != expected:
                    raise ValueError(f'Side-resolved excitation uses {name}={expected!r}.')
            with scene._device_scope(), scene._conditions(conditions):
                return self._solve(geometry, keep_traced=traced_scene)
        finally:
            for name, value in held.items():
                setattr(self, name, value)

    def _solve(self, geometry, *, keep_traced=False):
        import copy
        scene = self._scene
        xp, C = scene.xp, scene.complex_dtype
        k0 = 2*np.pi/scene.wavelength
        lattice = as_array(xp, scene.lattice.columns, dtype=C, device=scene._array_device)
        reciprocal = 2*np.pi*xp.matrix_transpose(xp.linalg.inv(lattice))
        reciprocal_host = _host(reciprocal).real
        half = tuple(int(v) for v in scene.truncation_order)
        groups = []
        sources = source_list(self.excitation)
        dipoles = [s for s in sources if isinstance(s, ElectricDipole)]
        localized = any(isinstance(s, GaussianBeam) or
                        isinstance(s, ElectricDipole) and s.repetition == 'isolated' for s in sources)
        if localized and (self.power_region is None or isinstance(self.power_region, str)):
            raise ValueError('GaussianBeam and isolated dipoles require a finite rectangle power_region.')
        if dipoles and self.length_unit is None:
            raise ValueError('ElectricDipole requires length_unit in metres per scene length.')
        if self.length_unit is not None:
            _positive(self.length_unit, 'length_unit')
        waves = []
        for source in sources:
            if isinstance(source, GaussianBeam):
                waves.extend(self._gaussian_waves(source))
            elif isinstance(source, (PlaneWave, OrderExcitation)):
                waves.append(source)
        for beam in waves:
            side = beam.side
            amplitude = as_array(xp, beam.amplitude, dtype=C, device=scene._array_device)
            if amplitude.ndim != 0 or not bool(xp.all(xp.isfinite(amplitude))):
                raise ValueError('PlaneWave amplitude must be one finite complex number.')
            q, pair, azimuth = self._resolve_wave(beam)
            orders = [((0, 0), 1., 0)] if getattr(beam, 'orders', None) is None else parse_excitation(beam.orders, *half)
            for (m, n), weight, _ in orders:
                actual = q + reciprocal @ as_array(xp, [m, n], dtype=C, device=scene._array_device)
                found = None
                for group in groups:
                    offset = np.linalg.solve(reciprocal_host, _host(actual - group['q']).real)
                    rounded = np.rint(offset).astype(int)
                    if np.allclose(offset, rounded, atol=1e-9, rtol=0):
                        found = (group, rounded)
                        break
                if found is None:
                    group = dict(q=actual, inputs=[], dipoles=[])
                    groups.append(group)
                    rounded = np.zeros(2, int)
                else:
                    group, rounded = found
                group['inputs'].append((side, rounded, amplitude*weight, pair, azimuth, actual))

        for dipole in dipoles:
            grid = [(np.asarray(dipole.bloch_k if dipole.bloch_k is not None else (0., 0.)), 1.)]
            if dipole.repetition == 'isolated':
                nx, ny = dipole.sampling
                grid = [(reciprocal_host @ np.array([(i+.5)/nx-.5, (j+.5)/ny-.5]), 1/(nx*ny))
                        for i in range(nx) for j in range(ny)]
            for q, weight in grid:
                group = next((g for g in groups if np.allclose(
                    np.linalg.solve(reciprocal_host, q-_host(g['q']).real),
                    np.rint(np.linalg.solve(reciprocal_host, q-_host(g['q']).real)), atol=1e-9, rtol=0)), None)
                if group is None:
                    group = dict(q=as_array(xp, q, dtype=C, device=scene._array_device), inputs=[], dipoles=[])
                    groups.append(group)
                group['dipoles'].append((dipole, weight))

        if len(groups) > 1 and self.power_region is None:
            raise ValueError("Different Bloch sectors require an explicit power_region: "
                             "'infinite', 'cell', or ((xmin,xmax),(ymin,ymax)).")
        spectra = {(side, part): [] for side in ('top', 'bottom') for part in ('incoming', 'outgoing')}
        solved = []
        keep = scene.answers.covers(Answers.FIELD)
        near_grazing = set()
        for group in groups:
            offsets = np.array([b[1] for b in group['inputs']] or [[0, 0]])
            low, high = offsets.min(axis=0)-half, offsets.max(axis=0)+half
            centre = (low+high)//2
            widths = np.maximum(centre-low, high-centre)
            basis = copy.copy(scene)
            basis.truncation_order = tuple(widths)
            basis.excitation = None
            basis.theta = basis.phi = None
            q = group['q'] + reciprocal @ as_array(xp, centre, dtype=C, device=scene._array_device)
            basis.kx, basis.ky = xp.real(q[0]), xp.real(q[1])
            with basis._device_scope():
                eps = list(basis._eps_conv_matrices(geometry))
                mu = basis._mu_conv_matrices(geometry)
            eps += [None]*(9-len(eps))
            gx, gy = basis.order_kxy()
            from .solve_2d import order_grid_kxy
            gx, gy = order_grid_kxy(xp, gx, gy)
            kx, ky = xp.reshape(gx, (-1,)), xp.reshape(gy, (-1,))
            count = kx.shape[0]
            kxy = xp.stack([kx, ky], axis=-1)*k0
            # The convolution builders use the cell-corner Fourier origin.
            # Rebase every material operator to the declared centre origin.
            origin = xp.sum(lattice, axis=1)/2
            phase = xp.exp(1j*(kxy @ origin))
            gauge = phase[:, None]*xp.conj(phase[None, :])
            eps = [None if a is None else a*gauge for a in eps]
            layers = _conv_layers(xp, eps, basis.thickness, count, C)
            if mu is not None:
                mu = list(mu) + [None]*(9-len(mu))
                mu = [None if a is None else a*gauge for a in mu]
                for (e, _), (magnetic, _) in zip(layers, _conv_layers(xp, mu, basis.thickness, count, C)):
                    e['mu_blocks'] = magnetic
            ports = {}
            for side in ('top', 'bottom'):
                material = basis.top if side == 'top' else basis.bot
                er = basis._as_complex(material.epsilon)
                mr = basis._as_complex(material.mu_value)
                ports[side, 1] = _plane_basis(xp, kx, ky, er, mr, 1)
                ports[side, -1] = _plane_basis(xp, kx, ky, er, mr, -1)
                if group['inputs'] and _concrete(xp.any(gain_cutoff_band(xp, er, mr, kx*kx + ky*ky))):
                    near_grazing.add(side)
            a = {side: xp.zeros((2*count,), dtype=C, device=scene._array_device) for side in ('top', 'bottom')}
            for side, offset, amp, pair, azimuth, actual in group['inputs']:
                ij = offset-centre+widths
                index = int(ij[1]*(2*widths[0]+1)+ij[0])
                pol = xp.stack(pair)
                if np.linalg.norm(_host(actual)) < 32*np.finfo(_host(xp.real(actual)).dtype).eps*float(abs(_host(k0))):
                    azimuth = as_array(xp, azimuth, dtype=C, device=scene._array_device)
                    c, s = xp.cos(azimuth), xp.sin(azimuth)
                    pol = xp.stack([pol[0]*c-pol[1]*s, pol[0]*s+pol[1]*c])
                direction = -1 if side == 'top' else 1
                wave = ports[side, direction][index] @ pol
                incoming_flux = direction*.5*xp.real(wave[0]*xp.conj(wave[4])-wave[1]*xp.conj(wave[3]))
                if not bool(incoming_flux > 0):
                    raise ValueError(f'{side} input carries no positive incident normal power (evanescent or grazing).')
                delta = xp.astype(xp.arange(count, device=scene._array_device) == index, C)
                a[side] = a[side] + as_array(xp, amp, dtype=C, device=scene._array_device)*xp.concat(
                    [pol[0]*delta, pol[1]*delta])
            layers, jumps, segment_edges, source_heights = self._dipole_layers(group, layers, k0, kxy)
            S, emission, state = _cascade(xp, k0, kx, ky, layers,
                                tuple(_tangential(xp, ports['bottom', d]) for d in (1, -1)),
                                tuple(_tangential(xp, ports['top', d]) for d in (1, -1)), keep, jumps)
            out = dict(bottom=S[0]@a['bottom'] + S[1]@a['top'] + emission[0],
                       top=S[2]@a['bottom'] + S[3]@a['top'] + emission[1])
            check_finite(xp, outgoing_top=out['top'], outgoing_bottom=out['bottom'])
            for side in ('top', 'bottom'):
                inward = -1 if side == 'top' else 1
                for part, coef, direction in (('incoming', a[side], inward), ('outgoing', out[side], -inward)):
                    amplitudes = xp.stack([coef[:count], coef[count:]], axis=-1)
                    f = xp.sum(ports[side, direction] * amplitudes[:, None, :], axis=-1)
                    spectra[side, part].append(_Spectrum(kxy, f))
            internal = _inside_amplitudes(xp, state, a['bottom'], a['top']) if keep else None
            solved.append(dict(k0=k0, kxy=kxy, kx=kx, ky=ky, n=count, internal=internal,
                               edges=segment_edges, source_heights=source_heights))
        # Once per side for the whole solve, however many Bloch sectors a Gaussian beam spread
        # over: the same test and text as an ordinary solve
        # (`CoreFactorizationMixin._warn_gain_cutoff`), on the ports' own eps and mu. Only
        # sectors lit by an external wave (PlaneWave, GaussianBeam, OrderExcitation) are
        # tested. A dipole is a point source, the sharp-edged case the band does not describe.
        # Eager only -- `_concrete` reads None on a trace.
        for side in ('top', 'bottom'):
            if side in near_grazing:
                import warnings
                warnings.warn(_GAIN_CUTOFF_MESSAGE.format(side=side), GainCutoffWarning,
                              stacklevel=3)
        joined = {key: _Spectrum(xp.concat([s.kxy for s in items]), xp.concat([s.field for s in items]))
                  for key, items in spectra.items()}
        result = CoherentResult(xp, joined, solved, lattice, scene.wavelength, scene.thickness,
                                self.power_region, bool(dipoles), localized, answers=scene.answers)
        # The returned result owns the traced field. Keeping it on a live facade
        # would let a later eager reconstruction read an expired JAX trace.
        self._result = None if not keep_traced and _holds_tracer(
            [s.field for s in joined.values()] + [solved, lattice, scene.wavelength,
                                                 scene.thickness]) else result
        return result

    def reconstruct_field(self, sample_x=20, sample_y=20, sample_z=20):
        if self._result is None:
            raise RuntimeError('Call solve() before reconstruct_field().')
        return self._result.reconstruct_field(sample_x, sample_y, sample_z)
