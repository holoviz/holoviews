"""
Supplies Layer and related classes that allow overlaying of Views,
including Overlay. A Layer is the final extension of View base class
that allows Views to be overlayed on top of each other.

Also supplies ViewMap which is the primary multi-dimensional Map type
for indexing, slicing and animating collections of Views.
"""

from collections import OrderedDict

import numpy as np

import param
from .dimension import Dimension, Dimensioned, ViewableElement
from .ndmapping import NdMapping
from .layout import Composable, LayoutTree
from .ndmapping import UniformNdMapping


class Overlayable(object):
    """
    Overlayable provides a mix-in class to support the
    mul operation for overlaying multiple elements.
    """

    def __mul__(self, other):
        if isinstance(other, UniformNdMapping):
            items = [(k, self * v) for (k, v) in other.items()]
            return other.clone(items)

        self_item = [((self.value, self.label if self.label else 'I'), self)]
        other_items = (other.items() if isinstance(other, Overlay)
                       else [((other.value, other.label if other.label else 'I'), other)])
        return Overlay(items=Overlay.relabel_item_paths(list(self_item) + list(other_items)))



class CompositeOverlay(ViewableElement, Composable):
    """
    CompositeOverlay provides a common baseclass for Overlay classes.
    """

    _deep_indexable = True

    @property
    def xlim(self):
        return self.range([d.name for d in self.deep_dimensions][0])

    @xlim.setter
    def xlim(self, limits):
        if limits is None or (isinstance(limits, tuple) and len(limits) == 2):
            self._xlim = limits
        else:
            raise ValueError('xlim needs to be a length two tuple or None.')

    @property
    def ylim(self):
        return self.range([d.name for d in self.deep_dimensions][1])

    @ylim.setter
    def ylim(self, limits):
        if limits is None or (isinstance(limits, tuple) and len(limits) == 2):
            self._ylim = limits
        else:
            raise ValueError('ylim needs to be a length two tuple or None.')

    @property
    def extents(self):
        l, r = self.xlim if self.xlim else (np.NaN, np.NaN)
        b, t = self.ylim if self.ylim else (np.NaN, np.NaN)
        return l, b, r, t


    def hist(self, index=None, adjoin=True, **kwargs):
        valid_ind = isinstance(index, int) and (0 <= index < len(self))
        valid_label = index in [el.label for el in self]
        if index is None or not any([valid_ind, valid_label]):
            raise TypeError("Please supply a suitable index for the histogram data")

        hist = self[index].hist(adjoin=False, **kwargs)
        if adjoin:
            layout = self << hist
            layout.main_layer = index
            return layout
        else:
            return hist



    def dimension_values(self, dimension):
        values = []
        found = False
        for el in self:
            if dimension in el.dimensions(label=True):
                values.append(el.dimension_values(dimension))
                found = True
        if not found:
            raise KeyError("Dimension %s was not found." % dimension)
        values = [v for v in values if v is not None and len(v)]
        return np.concatenate(values) if len(values) else []



class Overlay(LayoutTree, CompositeOverlay):
    """
    An Overlay consists of multiple Views (potentially of
    heterogeneous type) presented one on top each other with a
    particular z-ordering.

    Overlays along with Views constitute the only valid leaf types of
    a LayoutTree and in fact extend the LayoutTree structure. Overlays are
    constructed using the * operator (building an identical structure
    to the + operator) and are the only objects that inherit both from
    LayoutTree and CompositeOverlay.
    """

    @classmethod
    def _from_values(cls, val):
        return reduce(lambda x,y: x*y, val).display('auto')


    def __init__(self, items=None, value=None, label=None, **params):
        view_params = ViewableElement.params().keys()
        LayoutTree.__init__(self, items,
                          **{k:v for k,v in params.items() if k not in view_params})
        self._value = value
        self._label = label
        ViewableElement.__init__(self, self.data,
                      **{k:v for k,v in params.items() if k in view_params})


    def __add__(self, other):
        return LayoutTree.from_values(self) + LayoutTree.from_values(other)


    def __mul__(self, other):
        if isinstance(other, Overlay):
            items = list(self.data.items()) + list(other.data.items())
        elif isinstance(other, ViewableElement):
            label = other.label if other.label else 'I'
            items = list(self.data.items()) + [((other.value, label), other)]
        elif isinstance(other, UniformNdMapping):
            raise NotImplementedError

        return Overlay(items=self.relabel_item_paths(items)).display('all')


    def collapse(self, function):
        """
        Collapses all the Elements in the Overlay using the
        supplied function if they share a common type and value.
        """
        elements = list(self)
        types = [type(el) for el in elements]
        values = [el.value for el in elements]
        if not len(set(types)) == 1 and len(set(values)) == 1:
            raise Exception("Overlay is not homogenous in type or value "
                            "and cannot be collapsed.")
        else:
            return elements[0].clone(types[0].collapse_data([el.data for el in elements],
                                                            function))

    @property
    def value(self):
        if self._value:
            return self._value
        values = {el.value for el in self}
        types = {type(el) for el in self}
        value = list(values)[0]
        vtype = list(types)[0].__name__
        if len(values) == 1 and value != vtype:
            return value
        else:
            return type(self).__name__

    @value.setter
    def value(self, value):
        self._value = value

    @property
    def label(self):
        if self._label:
            return self._label
        labels = {el.label for el in self}
        if len(labels) == 1:
            return list(labels)[0]
        else:
            return ''

    @label.setter
    def label(self, label):
        self._label = label

    @property
    def deep_dimensions(self):
        dimensions = []
        dimension_names = []
        for el in self:
            for dim in el.dimensions():
                if dim.name not in dimension_names:
                    dimensions.append(dim)
                    dimension_names.append(dim.name)
        return dimensions



class NdOverlay(UniformNdMapping, CompositeOverlay, Overlayable):
    """
    An NdOverlay allows a group of NdOverlay to be overlaid together. NdOverlay can
    be indexed out of an overlay and an overlay is an iterable that iterates
    over the contained layers.
    """

    key_dimensions = param.List(default=[Dimension('Element')], constant=True, doc="""List
      of dimensions the NdOverlay can be indexed by.""")

    _deep_indexable = True

    def __init__(self, overlays=None, **params):
        self._xlim = None
        self._ylim = None
        data = self._process_layers(overlays)
        super(NdOverlay, self).__init__(data, **params)


    def _process_layers(self, layers):
        """
        Set a collection of layers to be overlaid with each other.
        """
        if isinstance(layers, (dict, OrderedDict, UniformNdMapping)):
            return layers
        elif layers is None or not len(layers):
            return OrderedDict()
        else:
            keys = range(len(layers))
            return OrderedDict(((key,), layer) for key, layer in zip(keys, layers))


    def hist(self, num_bins=20, bin_range=None, adjoin=True, individually=True, **kwargs):
        from ..operation import histogram
        return histogram(self, num_bins=num_bins, bin_range=bin_range, adjoin=adjoin,
                         individually=individually, **kwargs)


__all__ = list(set([_k for _k, _v in locals().items()
                    if isinstance(_v, type) and issubclass(_v, Dimensioned)])) + ['Overlayable']
