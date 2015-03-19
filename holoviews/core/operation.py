"""
Operations manipulate Elements, HoloMaps and Layouts, typically for
the purposes of analysis or visualization.
"""
from functools import reduce
import param

from .dimension import ViewableElement
from .element import Element, HoloMap, GridSpace
from .layout import Layout
from .overlay import NdOverlay, Overlay
from .traversal import unique_dimkeys



class Operation(param.ParameterizedFunction):
    """
    Base class for all Operation types.
    """

    group = param.String(default='Operation', doc="""
       The group string used to identify the output of the
       Operation. By default this should match the operation name.""")


    @classmethod
    def search(cls, element, pattern):
        """
        Helper method that returns a list of elements that match the
        given path pattern of form {type}.{group}.{label}.

        The input may be a Layout, an Overlay type or a single
        Element.
        """
        if isinstance(element, Layout):
            return [el for cell in element for el in cls.search(cell, pattern)]
        if isinstance(element, (NdOverlay, Overlay)):
            return [el for el in element if el.matches(pattern)]
        elif isinstance(element, Element):
            return [element] if element.matches(pattern) else []


    @classmethod
    def get_overlay_label(cls, overlay, default_label=''):
        """
        Returns a label if all the elements of an overlay agree on a
        consistent label, otherwise returns the default label.
        """
        if all(el.label==overlay[0].label for el in overlay):
            return overlay[0].label
        else:
            return default_label


    @classmethod
    def get_overlay_extents(cls, overlay):
        """
        Returns the extents if all the elements of an overlay agree on
        a consistent extents, otherwise raises an exception.
        """
        if all(el.extents==overlay[0].extents for el in overlay):
            return overlay[0].extents
        else:
            raise ValueError("Extents across the overlay are inconsistent")



class ElementOperation(Operation):
    """
    An ElementOperation process an Element or HoloMap at the level of
    individual elements or overlays. If a holomap is passed in as
    input, a processed holomap is returned as output where the
    individual elements have been transformed accordingly. An
    ElementOperation may turn overlays in new elements or vice versa.
    """

    input_ranges = param.ClassSelector(default={},
                                       class_=(dict, tuple), doc="""
       Ranges to be used for input normalization (if applicable) in a
       format appropriate for the Normalization.ranges parameter.

       By default, no normalization is applied. If key-wise
       normalization is required, a 2-tuple may be supplied where the
       first component is a Normalization.ranges list and the second
       component is Normalization.keys. """)


    def _process(self, view, key=None):
        """
        Process a single input element and outputs new single element
        or overlay. If a HoloMap is passed into a ElementOperation,
        the individual components are processed sequentially with the
        corresponding key passed as the optional key argument.
        """
        raise NotImplementedError


    def process_element(self, element, key, **params):
        """
        The process_element method allows a single element to be
        operated on given an externally supplied key.
        """
        self.p = param.ParamOverrides(self, params)
        return self._process(element, key)


    def __call__(self, element, **params):
        self.p = param.ParamOverrides(self, params)

        if isinstance(element, ViewableElement):
            processed = self._process(element)
        elif isinstance(element, GridSpace):
            # Initialize an empty axis layout
            processed = GridSpace(None, label=element.label)
            # Populate the axis layout
            for pos, cell in element.items():
                processed[pos] = self(cell, **params)
        elif isinstance(element, HoloMap):
            mapped_items = [(k, self._process(el, key=k))
                            for k, el in element.items()]
            refval = mapped_items[0][1]
            processed = element.clone(mapped_items,
                                      group=refval.group,
                                      label=refval.label)
        else:
            raise ValueError("Cannot process type %r" % type(element).__name__)
        return processed



class MapOperation(param.ParameterizedFunction):
    """
    A MapOperation takes a HoloMap containing elements or overlays and
    processes them at the HoloMap level, returning arbitrary new
    HoloMap objects as output. Unlike ElementOperation, MapOperations
    can compute over all the keys and dimensions of the input map.
    """

    group = param.String(default='MapOperation', doc="""
        The group string to identify the output of the MapOperation.
        By default this will match the MapOperation name.""")

    def __call__(self, vmap, **params):
        self.p = param.ParamOverrides(self, params)

        if not isinstance(vmap, HoloMap):
            raise Exception('MapOperation can only process Maps.')

        return self._process(vmap)


    def _process(self, view):
        """
        Process a single input HoloMap, returning a new HoloMap
        instance.
        """
        raise NotImplementedError



class TreeOperation(Operation):
    """
    A TreeOperation is the most general Operation type; it accepts any
    HoloViews datastructure and outputs a Layout containing one or
    more elements.
    """

    def process_element(self, element, key, **params):
        """
        The process_element method allows a single element to be
        operated on given an externally supplied key.
        """
        self.p = param.ParamOverrides(self, params)
        maps = self._process(element, key)
        return reduce(lambda x,y: x + y, maps)


    def __call__(self, src, **params):
        self.p = param.ParamOverrides(self, params)
        dims, keys = unique_dimkeys(src)

        if not dims:
            return self.process_element(src, None)
        elif isinstance(src, HoloMap):
            values = src.values()
        elif isinstance(src, Layout):
            if not src.uniform:
                raise Exception("TreeOperation can only process uniform Layouts")
            dim_names = [d.name for d in dims]
            values = [src.select(**dict(zip(dim_names, key))) for key in keys]

        tree = Layout()
        for key, el in zip(keys, values):
            if not isinstance(el, Layout):
                result = self._process(Layout.from_values(el), key)
            else:
                result = self._process(el, key)

            holomaps = [HoloMap([(key,el)], key_dimensions=dims,
                                group=el.group, label=el.label) for el in result]
            if len(holomaps) == 1:
                processed_tree = Layout.from_values(holomaps[0])
            else:
                processed_tree = Layout.from_values(holomaps)

            tree.update(processed_tree)
        return tree


    def _process(self, tree, key=None):
        """
        Process a single input Layout, returning a list of
        elements to be merged with the output Layout.
        """
        raise NotImplementedError

