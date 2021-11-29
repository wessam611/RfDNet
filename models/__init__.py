from . import iscnet
from . import loss

method_paths = {
    'ISCNet': iscnet,
    'ISCNet_WEAK': iscnet,
}

__all__ = ['method_paths']