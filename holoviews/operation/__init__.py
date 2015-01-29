from ..core.operation import ElementOperation, MapOperation
from ..core.options import ChannelDefinition

#from .channel import * # pyflakes:ignore (API import)
from .element import * # pyflakes:ignore (API import)
from .map import * # pyflakes:ignore (API import)


def public(obj):
    if not isinstance(obj, type): return False
    baseclasses = [ElementOperation, MapOperation]
    return any([issubclass(obj, bc) for bc in baseclasses])


# ChannelDefinition.operations.append(operator)
# ChannelDefinition.operations.append(toRGBA)
# ChannelDefinition.operations.append(toHCS)
# ChannelDefinition.operations.append(alpha_overlay)


_public = list(set([_k for _k, _v in locals().items() if public(_v)]))
__all__ = _public
