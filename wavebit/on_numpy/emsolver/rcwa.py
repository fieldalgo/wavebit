from ._base import _BaseRCWA
from ..._core.engine import CoreFactorizationMixin


class RCWANumpy(CoreFactorizationMixin, _BaseRCWA):
    """The NumPy solver, which is the two mixins and the declared members on `_BaseRCWA`.

    Empty on purpose. Everything this class used to hold -- a 104-line constructor -- is in
    `CoreFactorizationMixin.__init__`, and everything NumPy does differently is a declared
    member one level down. The class survives because it is the name `rcwa(backend=0)` builds
    and the MRO position that pairs the two halves.
    """
