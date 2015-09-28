from .boundingregion import *  # pyflakes:ignore (API import)
from .dimension import *       # pyflakes:ignore (API import)
from .element import *         # pyflakes:ignore (API import)
from .layout import *          # pyflakes:ignore (API import)
from .operation import *       # pyflakes:ignore (API import)
from .overlay import *         # pyflakes:ignore (API import)
from .sheetcoords import *     # pyflakes:ignore (API import)
from .spaces import *          # pyflakes:ignore (API import)
from .tree import *            # pyflakes:ignore (API import)
from .io import FileArchive

archive = FileArchive()


def displayable(obj):
    """
    Predicate that returns whether the object is displayable or not
    (i.e whether the object obeys the nesting hierarchy
    """
    if isinstance(obj, HoloMap):
        return not (obj.type in [Layout, GridSpace, NdLayout])
    if isinstance(obj, (GridSpace, Layout, NdLayout)):
        for el in obj.values():
            if not displayable(el):
                return False
        return True
    return True


def undisplayable_info(obj, html=False):
    "Generate helpful message regarding an undisplayable object"

    collate = '<tt>collate</tt>' if html else 'collate'
    info = "For more information, please consult the Composing Data tutorial (http://git.io/vtIQh)"
    if isinstance(obj, HoloMap):
        error = "HoloMap of %s objects cannot be displayed." % obj.type.__name__
        remedy = "Please call the %s method to generate a displayable object" % collate
    elif isinstance(obj, Layout):
        error = "Layout containing HoloMaps of Layout or GridSpace objects cannot be displayed."
        remedy = "Please call the %s method on the appropriate elements." % collate
    elif isinstance(obj, GridSpace):
        error = "GridSpace containing HoloMaps of Layouts cannot be displayed."
        remedy = "Please call the %s method on the appropriate elements." % collate

    if not html:
        return '\n'.join([error, remedy, info])
    else:
        return "<center>{msg}</center>".format(msg=('<br>'.join(
            ['<b>%s</b>' % error, remedy, '<i>%s</i>' % info])))


# Define default type formatters
Dimension.type_formatters[int] = "%d"
Dimension.type_formatters[float] = "%.5g"
Dimension.type_formatters[np.float32] = "%.5g"
Dimension.type_formatters[np.float64] = "%.5g"


def public(obj):
    if not isinstance(obj, type): return False
    baseclasses = [Dimension, Dimensioned, ElementOperation, BoundingBox,
                   SheetCoordinateSystem, AttrTree]
    return any([issubclass(obj, bc) for bc in baseclasses])

_public = list(set([_k for _k, _v in locals().items() if public(_v)]))
__all__ = _public + ["boundingregion", "dimension", "layer", "layout",
                     "ndmapping", "operation", "options", "sheetcoords", "tree", "element"]

