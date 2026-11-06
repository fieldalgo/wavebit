"""Field access through an owned reconstruction state protocol."""
from wavebit._common.backend import eager_under_torch_compile

@eager_under_torch_compile
class ResultFields:
    """Public field API shared by every solve result."""

    def _require_field_state(self):
        if self._field_state is None:
            raise RuntimeError("This result has no field state; solve with answers='amplitude' "
                               "for boundary fields or answers='field' for internal fields.")
        return self._field_state

    def reconstruct_field(self, sample_x=20, sample_y=20, sample_z=20, *, incidence='incident',
                          discontinuous='fourier'):
        """Complex (Ex,Ey,Ez,Z0Hx,Z0Hy,Z0Hz), shape (nz,ny,nx,6).

        Counts or 1-D physical coordinates are accepted on every axis. z counts
        are per finite layer; explicit heights start at the stack bottom and
        choose the upper layer at an interface. No live solver state is read.
        ``incidence`` selects the solved input, or a retained pure 's'/'p' variant.
        ``discontinuous`` is the solver's: ``'fourier'``, or ``'continuity'`` for the rule of
        :meth:`~wavebit.rcwa._engine.engine.CoreFactorizationMixin.reconstruct_field`.
        """
        return self._require_field_state().reconstruct(sample_x, sample_y, sample_z, incidence,
                                                       discontinuous)

    def boundary_fields(self, side, sample_x=(0.,), sample_y=(0.,), part='total', *, incidence='incident'):
        """Boundary half-space fields, shape (ny,nx,6), with the same sampling rules.

        side is 'top' or 'bottom'; part is 'incoming', 'outgoing' or 'total'.
        """
        return self._require_field_state().boundary(side, sample_x, sample_y, part, incidence)
