"""
Operations manipulate Elements, HoloMaps and Layouts, typically for
the purposes of analysis or visualization.
"""

import numpy as np

import param

from .dimension import ViewableElement
from .element import Element, HoloMap, AxisLayout
from .layout import NdLayout, LayoutTree
from .overlay import CompositeOverlay, NdOverlay, Overlay


class Operation(param.ParameterizedFunction):
    """
    Base class for all Operation types.
    """

    value = param.String(default='Operation', doc="""
       The value string used to identify the output of the
       Operation. By default this should match the operation name.""")


    @classmethod
    def search(cls, element, pattern):
        """
        Helper method that returns a list of elements that match the
        given path pattern of form {type}.{value}.{label}.

        The input may be a LayoutTree, an Overlay type or a single
        Element.
        """
        if isinstance(element, LayoutTree):
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


class ElementOperation(Operation):
    """
    An ElementOperation process an Element or HoloMap at the level of
    individual elements or overlays. If a holomap is passed in as
    input, a processed holomap is returned as output where the
    individual elements have been transformed accordingly. An
    ElementOperation may turn overlays in new elements or vice versa.
    """

    def _process(self, view, key=None):
        """
        Process a single input element and outputs new single element
        or overlay. If a HoloMap is passed into a ElementOperation,
        the individual components are processed sequentially with the
        corresponding key passed as the optional key argument.
        """
        raise NotImplementedError


    def __call__(self, element, **params):
        self.p = param.ParamOverrides(self, params)

        if isinstance(element, ViewableElement):
            processed = self._process(element)
        elif isinstance(element, AxisLayout):
            # Initialize an empty axis layout
            processed = AxisLayout(None, label=element.label)
            # Populate the axis layout
            for pos, cell in element.items():
                processed[pos] = self(cell, **params)
        elif isinstance(element, HoloMap):
            mapped_items = [(k, self._process(el, key=k))
                            for k, el in element.items()]
            processed = element.clone(mapped_items)
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

    value = param.String(default='MapOperation', doc="""
        The value string to identify the output of the MapOperation.
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



class LayoutOperation(param.ParameterizedFunction):

    pass
