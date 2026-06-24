# mx.set_default_device(mx.cpu)

from .padic import *
from .tree import FiniteTree
from .diffusion import UltrametricDiffusion
from .routing import DigitHeads
from .model import UCEModel

# Re-export control to avoid leaking internals on "from ultrametric_ce import *"
# (consistent with the __all__ discipline added to padic.py previously).
from .padic import __all__ as _padic_all
__all__ = list(_padic_all) + ["FiniteTree", "UltrametricDiffusion", "DigitHeads", "UCEModel"]
