"""Complete real local transmission modes on the fixed closed pair chart."""
import numpy as np
from wavebit.rcwa._engine.conforming.cartesian_pair import _physical_geometry
from wavebit.rcwa._engine.conforming.pair_corner_fields import PairCornerFields
from wavebit.rcwa._engine.conforming.pair_corner_coordinates import prepare
from wavebit.rcwa._engine.conforming.pair_transmission_modes import TransmissionMode
from wavebit.rcwa._engine.conforming.pair_transmission_common_precision import CommonCoordinates


class PairTransmissionFields(PairCornerFields):
    """Transport fixed reference modes with the same common Pp coordinates."""

    complete_common_precision = False
    common_coordinate_type = CommonCoordinates

    def __init__(self, obj, settings, reference, branches, angular_quadrature, cutoff_order=0):
        super().__init__(obj, (), junctions=True, angular_quadrature=12)
        self.complete_common_precision = obj.degree >= 5
        self.cutoff_order = cutoff_order
        self.root_order = 20
        geometry = {key: np.asarray(settings[key])
                    for key in ('basis', 'widths', 'height', 'offset')}
        length, _, self.zero, _ = obj._chart_geometry(0., geometry)
        normalized = {key: value/length for key, value in geometry.items()}
        corners = _physical_geometry(gap=0., **normalized)[0]
        left, right, background = reference
        specifications = [((left, background, background), 0.),
                          ((right, left, background), 0.),
                          ((right, background, background), np.pi/2),
                          ((right, background, background), np.pi),
                          ((left, right, background), np.pi),
                          ((left, background, background), -np.pi/2)]
        self.modes = []
        self.cutoff_powers = []
        self._definitions = []
        for corner, (coefficients, start) in zip(self.controls, specifications):
            # Equal quarter-sector coefficients give a straight interface.
            # Every separated growing solution is piecewise polynomial and
            # supplies no noninteger corner direction.
            if coefficients[0] == coefficients[1]:
                continue
            for branch in range(branches):
                mode = TransmissionMode(coefficients, start=start, branch=branch)
                # A junction vertex becomes an edge on an opening triangle.
                # Its nonzero trace has gradient r**(lambda-1), whose squared
                # edge integral is finite exactly when Re(lambda) > 1/2.
                # Reject the whole inadmissible bank; never discard a mode.
                if corner in (5, 8) and mode.exponent.real <= .5:
                    raise ValueError('a reference junction mode requires Re(exponent) > 1/2 for a finite opening energy')
                real = mode.exponent.imag == 0 and all(np.imag(a) == 0 for a in mode.coefficients)
                parts = ('real',) if real else ('real', 'imag')
                for power in range(cutoff_order+1):
                    self.modes.append((corner, corners[corner], mode, parts))
                    self.cutoff_powers.append(power)
                    self._definitions.extend((corner,) for _ in parts)
        self.count = len(self._definitions)
        if obj.degree>=7 or cutoff_order and self.count:
            from wavebit.rcwa._engine.conforming.pair_transmission_extended_precision import CommonCoordinates as ExtendedCoordinates
            self.common_coordinate_type = ExtendedCoordinates
        if self.count:
            prepare(self)
        self.angular_quadrature = angular_quadrature

    def _sample(self, tri, bary, inverse, dinverse, bary_tangent=None):
        bary = np.asarray(bary)
        db = np.zeros_like(bary) if bary_tangent is None else np.asarray(bary_tangent)
        vertices = self.zero[tri[:3]]
        hats = self.hats[tri[:3]]
        jacobian = np.column_stack([vertices[1]-vertices[0], vertices[2]-vertices[0]])
        value = np.zeros((len(bary), self.count))
        gradient = np.zeros((len(bary), 2, self.count))
        change = np.zeros_like(gradient)
        first = 0
        moving = np.any(db)
        for (corner, center, mode, parts), power in zip(self.modes, self.cutoff_powers):
            cutoff = hats[:, corner]
            if not np.any(cutoff):
                first += len(parts)
                continue
            offsets = vertices-center
            offsets[cutoff == 1] = 0.
            points = bary@offsets
            centroid = offsets.mean(axis=0)
            angle = (np.arctan2(centroid[1], centroid[0])-mode.start) % (2*np.pi)
            sector = min(np.searchsorted(np.cumsum(mode.angles), angle, side='right'), 2)
            sampled = mode.sample(points, sector=sector, center_angle=angle, hessian=moving)
            f, g = sampled[:2]
            h, dh = bary@cutoff, db@cutoff
            gh = np.array([cutoff[1]-cutoff[0], cutoff[2]-cutoff[0]])
            raw = h[:, None]*(g@jacobian)+f[:, None]*gh
            if power:
                s = bary@(1-cutoff)
                weight = h*s**power
                first_weight = s**power-power*h*s**(power-1)
                second_weight = (-2*np.ones_like(h) if power == 1 else
                                 power*(power-1)*h*s**(power-2)-2*power*s**(power-1))
                raw = weight[:, None]*(g@jacobian)+f[:, None]*first_weight[:, None]*gh
            draw = np.zeros_like(raw)
            if moving:
                velocity = db@offsets
                dg = np.einsum('qij,qj->qi', sampled[2], velocity)
                df = np.sum(g*velocity, axis=1)
                draw = dh[:, None]*(g@jacobian)+h[:, None]*(dg@jacobian)+df[:, None]*gh
                if power:
                    draw = ((first_weight*dh)[:, None]*(g@jacobian)
                            +weight[:, None]*(dg@jacobian)
                            +df[:, None]*first_weight[:, None]*gh
                            +f[:, None]*(second_weight*dh)[:, None]*gh)
            for part in parts:
                value[:, first] = getattr((weight if power else h)*f, part)
                gradient[:, :, first] = getattr(raw@inverse, part)
                change[:, :, first] = getattr(draw@inverse+raw@dinverse, part)
                first += 1
        return value, gradient, change
