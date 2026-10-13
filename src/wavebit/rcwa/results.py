"""Owned RCWA observations, with spectral axes and physical field calibration."""
from dataclasses import field
from types import MappingProxyType


from .._common.declarations import spec
from .._common.backend import array_namespace, as_array
from ..errors import UnsupportedFeatureError
from ..observations import FieldSamples
from ..units import Z0


@spec
class BoundaryPower:
    """A measured port quantity with an explicit spatial integration measure."""
    incoming: object
    outgoing: object
    units: str
    integration_measure: object
    reference: object = field(default_factory=dict)

    def __post_init__(self):
        from .._common.pytrees import register_dataclass
        register_dataclass(type(self))
        object.__setattr__(self, 'reference', MappingProxyType(dict(self.reference)))


@spec
class Result:
    """Independent result; reconstruction reads only its own retained snapshot."""
    outputs: object
    wavelengths: object
    validity: object
    diagnostics: object
    provenance: object
    _field_records: tuple = ()
    _field_amplitudes: tuple = ()
    _scattering: object = None

    def __post_init__(self):
        from .._common.pytrees import register_dataclass
        register_dataclass(type(self))
        for name in ('outputs', 'diagnostics', 'provenance'):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    def _efficiency(self, port):
        observation = self.outputs.get(port)
        if observation is None or observation.efficiency is None:
            raise UnsupportedFeatureError('No single-incident-port diffraction efficiency was requested or is available for this result.',
                                          code='output.unavailable', path=f'outputs.{port}.efficiency')
        return observation.efficiency

    @property
    def R(self):
        """Reflected efficiencies with shape (wavelength, order_v, order_u)."""
        return self._efficiency('reflected')

    @property
    def T(self):
        """Transmitted efficiencies with shape (wavelength, order_v, order_u)."""
        return self._efficiency('transmitted')

    @property
    def reflection(self):
        return self.outputs['reflected']

    @property
    def transmission(self):
        return self.outputs['transmitted']

    @property
    def scattering(self):
        if self._scattering is None:
            raise UnsupportedFeatureError('Request Outputs(scattering=True) with the SMM algorithm before running.',
                                          code='output.unavailable', path='outputs.scattering')
        return self._scattering

    def _fields(self):
        if not self._field_records:
            raise UnsupportedFeatureError('This result did not retain field reconstruction state. Request Outputs(fields=True).',
                                          code='output.unavailable', path='outputs.fields')
        return self._field_records

    def field(self, *, x, y, z):
        """Collocated complex phasors at explicit Cartesian coordinate arrays.

        Shape is (wavelength,z,y,x,component); E is V/m and H is A/m.
        Coordinates use the Problem's declared length unit. Interfaces select
        the upper layer, matching the engine's explicit-coordinate convention.
        """
        records = self._fields()
        values = []
        for state, amplitude in zip(records, self._field_amplitudes):
            a = state.reconstruct(x, y, z, 'incident') * amplitude
            xp = array_namespace(a)
            factors = as_array(xp, [1., 1., 1., 1/Z0, 1/Z0, 1/Z0], dtype=a.dtype,
                               device=getattr(a, 'device', None))
            values.append(a * factors)
        data = array_namespace(values[0]).stack(values)
        return FieldSamples(coordinates={'x': x, 'y': y, 'z': z, 'wavelength': self.wavelengths},
                            values=data, components=('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'),
                            units={'E': 'V/m', 'H': 'A/m', 'coordinates': self.provenance['length_unit']},
                            sampling='collocated Cartesian phasor',
                            metadata={'axis_order': ('wavelength', 'z', 'y', 'x', 'component'),
                                      'time_convention': 'exp(-i omega t)'})

    def boundary_fields(self, *, side, x, y, part='total'):
        """Collocated boundary phasors from this result's requested field state."""
        records = self._fields()
        values = []
        for state, amplitude in zip(records, self._field_amplitudes):
            a = state.boundary(side, x, y, part, 'incident') * amplitude
            xp = array_namespace(a)
            factors = as_array(xp, [1., 1., 1., 1/Z0, 1/Z0, 1/Z0], dtype=a.dtype,
                               device=getattr(a, 'device', None))
            values.append(a * factors)
        data = array_namespace(values[0]).stack(values)
        return FieldSamples(coordinates={'x': x, 'y': y, 'wavelength': self.wavelengths},
                            values=data, components=('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'),
                            units={'E': 'V/m', 'H': 'A/m', 'coordinates': self.provenance['length_unit']},
                            sampling='collocated boundary phasor',
                            metadata={'side': side, 'part': part,
                                      'axis_order': ('wavelength', 'y', 'x', 'component')})

    @property
    def absorption(self):
        """Net entering minus transmitted power, where the normalization is defined."""
        if 'absorption' not in self.outputs:
            raise UnsupportedFeatureError('Absorption is unavailable for this observation geometry.',
                                          code='output.unavailable', path='outputs.absorption')
        return self.outputs['absorption']


def _stack(values):
    return array_namespace(values[0]).stack(values)


def _sum_power(field, xp, sign, area, amplitude):
    return (sign * .5/Z0 * xp.real(field[..., 0] * xp.conj(field[..., 4])
                                  - field[..., 1] * xp.conj(field[..., 3]))
            * area * xp.abs(amplitude)**2)


def pack(problem, numerics, execution, solves, checks, report):
    """Convert solved snapshots without keeping any mutable solver alive."""
    from ..observations import DiffractionOrders, Validity
    from ..observations.power import transverse_k_is_real
    from .._common.conventions import boundary_kz, half_space_medium
    from ._adapt import single_plane, _array, wavelengths
    xp = solves[0][0].xp if single_plane(problem) else solves[0][0]._scene.xp
    ws = xp.stack([_array(xp, w, execution) for w in wavelengths(problem)])
    outputs = {}
    records, amplitudes, scattering = [], [], []
    actual_wavelengths = []
    actual_directions = []
    power_defined = []
    if single_plane(problem):
        source = problem.excitation.sources[0]
        side, other = source.side, 'bottom' if source.side == 'top' else 'top'
        amplitude = _array(xp, source.amplitude, execution, True)
        per_port = {'reflected': [], 'transmitted': []}
        incident_powers, absorptions, entering = [], [], []
        for solver, result in solves:
            actual_wavelengths.append(_array(xp, solver.wavelength, execution))
            actual_directions.append(solver.theta)
            state = result._field_state
            lattice = state.data['lattice']
            area = xp.abs(xp.linalg.det(lattice)) * problem.units.length_metres**2
            boundary = state.data['boundary']['incident']
            k_incident = solver._incident_kxy(solver.wavelength)
            physical_power = transverse_k_is_real(xp, *k_incident)
            power_defined.append(physical_power)
            shape = result.R.shape
            incoming = _sum_power(boundary[side][0], xp, -1 if side == 'top' else 1,
                                  area, amplitude)
            incident_powers.append(xp.where(physical_power, xp.sum(incoming), xp.nan))
            entering.append(result.power_entering)
            absorptions.append(result.absorption)
            for name, port, prefix in (('reflected', side, 'r'), ('transmitted', other, 't')):
                outgoing = _sum_power(boundary[port][1], xp, 1 if port == 'top' else -1,
                                      area, amplitude).reshape(shape)
                material = solver.top if port == 'top' else solver.bot
                k0 = 2*xp.pi/solver.wavelength
                k_parallel = state.data['boundary_kxy'].reshape(shape+(2,))
                normalized_kt2 = xp.sum((k_parallel/k0)**2, axis=-1)
                longitudinal = boundary_kz(xp, material.epsilon*material.mu_value, normalized_kt2,
                                           half_space_medium(material.epsilon, material.mu_value))*k0
                if port == 'bottom':
                    longitudinal = -longitudinal
                per_port[name].append(dict(amplitude_s=getattr(result, prefix+'_s'),
                                           amplitude_p=getattr(result, prefix+'_p'),
                                           cartesian=getattr(result, prefix+'_xyz'),
                                           efficiency=xp.where(physical_power, result.R if name == 'reflected' else result.T, xp.nan),
                                           power=xp.where(physical_power, outgoing, xp.nan),
                                           k_parallel=k_parallel, kz=longitudinal,
                                           epsilon=material.epsilon, mu=material.mu_value,
                                           reference_z=xp.sum(solver.thickness) if port == 'top' else 0.,
                                           area=area))
            if problem.outputs.fields:
                records.append(state)
                amplitudes.append(amplitude)
            if problem.outputs.scattering:
                scattering.append(solver.scattering)
        effective_ws = xp.stack(actual_wavelengths)
        if problem.outputs.diffraction:
            for name, port in (('reflected', side), ('transmitted', other)):
                entries = per_port[name]
                outputs[name] = DiffractionOrders(
                    wavelengths=effective_ws, orders_u=solves[0][1].orders_u, orders_v=solves[0][1].orders_v,
                    amplitude_s=_stack([e['amplitude_s'] for e in entries]),
                    amplitude_p=_stack([e['amplitude_p'] for e in entries]),
                    cartesian=_stack([e['cartesian'] for e in entries]),
                    efficiency=_stack([e['efficiency'] for e in entries]),
                    power=_stack([e['power'] for e in entries]),
                    k_parallel=_stack([e['k_parallel'] for e in entries]),
                    kz=_stack([e['kz'] for e in entries]),
                    reference={'port': port, 'z': tuple(e['reference_z'] for e in entries),
                               'lateral_origin': (0., 0.), 'outward_normal': (0., 0., 1. if port == 'top' else -1.),
                               'incident_port': side, 'incident_power_w': _stack(incident_powers),
                               'incident_electric_amplitude_v_per_m': amplitude,
                               'epsilon': tuple(e['epsilon'] for e in entries),
                               'mu': tuple(e['mu'] for e in entries),
                               'cell_area_m2': tuple(e['area'] for e in entries)},
                    metadata={'power_defined': xp.stack(power_defined),
                              'unavailable_power_reason': 'complex transverse wavevectors require a different spatial power integral; amplitudes remain valid',
                              'amplitude_units': 'dimensionless electric-field ratio', 'power_units': 'W',
                              'integration_measure': 'periodic cell', 'propagation': 'continuum',
                              'polarization_frame': "each order's outgoing s/p frame",
                              'time_convention': 'exp(-i omega t)', 'length_unit': problem.units.length})
        outputs['power_entering'] = _stack(entering)
        outputs['absorption'] = _stack(absorptions)
    else:
        boundary = {'top': [], 'bottom': []}
        absorption = []
        for solver, result in solves:
            scene = solver._scene
            actual_wavelengths.append(_array(xp, scene.wavelength, execution))
            actual_directions.append(None)
            region = result.power.region
            area = None
            if region == 'cell' if isinstance(region, str) else False:
                area = xp.abs(xp.linalg.det(as_array(xp, result.lattice))) * problem.units.length_metres**2
            elif not isinstance(region, str):
                area = (region[0][1]-region[0][0])*(region[1][1]-region[1][0])*problem.units.length_metres**2
            scale = 1/Z0 if area is None else area/Z0
            for port in ('top', 'bottom'):
                boundary[port].append((getattr(result.power, 'incoming_'+port)*scale,
                                       getattr(result.power, 'outgoing_'+port)*scale))
            absorption.append(result.absorption)
            if problem.outputs.fields:
                records.append(result._field_state)
                amplitudes.append(1.)
        for port, pairs in boundary.items():
            outputs[port] = BoundaryPower(incoming=_stack([p[0] for p in pairs]),
                                         outgoing=_stack([p[1] for p in pairs]),
                                         units='W/m^2' if area is None else 'W',
                                         integration_measure='infinite-area mean' if area is None else problem.outputs.power_region or 'cell',
                                         reference={'port': port, 'time_convention': 'exp(-i omega t)'})
        if all(a is not None for a in absorption):
            outputs['absorption'] = _stack(absorption)
    # Finiteness is numerical validity, never a declaration of optical convergence.
    numbers = ([v for _, r in solves for v in (r.r_xyz, r.t_xyz)] if single_plane(problem)
               else [v for _, r in solves for v in (r.power.outgoing_top, r.power.outgoing_bottom)])
    finite = xp.all(xp.stack([xp.all(xp.isfinite(v)) for v in numbers]))
    combined = finite
    for value in checks.values():
        combined = combined & value
    # Traced admission failures remain visible; their numerical observations are NaN.
    if single_plane(problem):
        for name in ('reflected', 'transmitted'):
            if name in outputs:
                o = outputs[name]
                outputs[name] = o.replace(**{key: xp.where(combined, getattr(o, key), xp.nan)
                                            for key in ('amplitude_s', 'amplitude_p', 'cartesian', 'power', 'efficiency')
                                            if getattr(o, key) is not None})
        for name in ('power_entering', 'absorption'):
            outputs[name] = xp.where(combined, outputs[name], xp.nan)
    # Retained reconstruction must obey the same admission guards as the
    # eagerly materialized observables, including when evaluated after jit.
    amplitudes = [xp.where(combined, amplitude, xp.nan) for amplitude in amplitudes]
    scattering = [matrix._replace(**{name: xp.where(combined, getattr(matrix, name), xp.nan)
                                    for name in ('S11', 'S12', 'S21', 'S22')})
                  for matrix in scattering]
    return Result(outputs=outputs, wavelengths=xp.stack(actual_wavelengths),
                  validity=Validity(checks={**checks, 'numerical_finite': finite},
                                    convergence={'status': 'not assessed', 'requires': 'independent refinement study'}),
                  diagnostics={'report': report, 'effective_wavelengths': tuple(actual_wavelengths),
                               'effective_theta': tuple(actual_directions),
                               'port_power_defined': xp.stack(power_defined) if power_defined else None,
                               'detuning_relative_step': numerics.detuning},
                  provenance={'solver': 'rcwa', 'code_version': '0.1.0', 'length_unit': problem.units.length,
                              'units': problem.units, 'execution': execution, 'numerics': numerics,
                              'coordinate_frame': '+z up; bottom interface z=0; layers bottom-to-top',
                              'source_normalization': 'electric amplitude V/m; dipole moment C m',
                              'requested_wavelengths': ws},
                  _field_records=tuple(records), _field_amplitudes=tuple(amplitudes),
                  _scattering=tuple(scattering) if scattering else None)
