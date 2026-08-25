from ._base import _BaseRCWA
from ..._core.engine import CoreFactorizationMixin


class RCWATorch(CoreFactorizationMixin, _BaseRCWA):
    """The Torch solver. Empty for the same reason as :class:`RCWANumpy`."""
