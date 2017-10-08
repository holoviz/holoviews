"""
Supplies Layer and related classes that allow overlaying of Views,
including Overlay. A Layer is the final extension of View base class
that allows Views to be overlayed on top of each other.

Also supplies ViewMap which is the primary multi-dimensional Map type
for indexing, slicing and animating collections of Views.
"""

from functools import reduce
import numpy as np

import param
from .dimension import Dimension, Dimensioned, ViewableElement
from .ndmapping import UniformNdMapping
from .layout import Composable, Layout, AdjointLayout
from .util import sanitize_identifier, unique_array

class Overlayable(object):
    """
    Overlayable provides a mix-in class to support the
    mul operation for overlaying multiple elements.
    """

    def __mul__(self, other):
        if type(other).__name__ == 'DynamicMap':
            from .spaces import Callable
            def dynamic_mul(*args, **kwargs):
                element = other[args]
                return self * element
            callback = Callable(dynamic_mul, inputs=[self, other])
            callback._is_overlay = True
            return other.clone(shared_data=False, callback=callback,
                               streams=[])
        if isinstance(other, UniformNdMapping) and not isinstance(other, CompositeOverlay):
            items = [(k, self * v) for (k, v) in other.items()]
            return other.clone(items)
        elif isinstance(other, AdjointLayout):
            return NotImplemented

        return Overlay.from_values([self, other])



class CompositeOverlay(ViewableElement, Composable):
    """
    CompositeOverlay provides a common baseclass for Overlay classes.
    """

    _deep_indexable = True


    def hist(self, index=0, adjoin=True, dimension=None, **kwargs):
        valid_ind = isinstance(index, int) and (0 <= index < len(self))
        valid_label = index in [el.label for el in self]
        if not any([valid_ind, valid_label]):
            raise TypeError("Please supply a suitable index or label for the histogram data")

        hists = self.get(index).hist(adjoin=False, dimension=dimension, **kwargs)
        if not isinstance(hists, Layout):
            hists = [hists]
        if not isinstance(dimension, list):
            dimension = ['Default']
        if adjoin:
            layout = self
            for hist in hists:
                layout = layout << hist
            layout.main_layer = index
        elif len(dimension) > 1:
            layout = hists
        else:
            layout = hists[0]
        return layout


    def dimension_values(self, dimension, expanded=True, flat=True):
        values = []
        found = False
        for el in self:
            if dimension in el.dimensions(label=True):
                values.append(el.dimension_values(dimension))
                found = True
        if not found:
            return super(CompositeOverlay, self).dimension_values(dimension, expanded, flat)
        values = [v for v in values if v is not None and len(v)]
        if not values:
            return np.array()
        vals = np.concatenate(values)
        return vals if expanded else unique_array(vals)


class Overlay(Layout, CompositeOverlay):
    """
    An Overlay consists of multiple Views (potentially of
    heterogeneous type) presented one on top each other with a
    particular z-ordering.

    Overlays along with Views constitute the only valid leaf types of
    a Layout and in fact extend the Layout structure. Overlays are
    constructed using the * operator (building an identical structure
    to the + operator) and are the only objects that inherit both from
    Layout and CompositeOverlay.
    """

    def __init__(self, items=None, group=None, label=None, **params):
        view_params = ViewableElement.params().keys()
        self.__dict__['_fixed'] = False
        self.__dict__['_group'] = group
        self.__dict__['_label'] = label
        Layout.__init__(self, items,
                          **{k:v for k,v in params.items() if k not in view_params})
        ViewableElement.__init__(self, self.data,
                                 **{k:v for k,v in params.items() if k in view_params})


    def __getitem__(self, key):
        """
        Allows transparently slicing the Elements in the Overlay
        to select specific layers in an Overlay use the .get method.
        """
        return Overlay([(k, v[key]) for k, v in self.items()])


    def get(self, identifier, default=None):
        if isinstance(identifier, int):
            values = list(self.data.values())
            if 0 <= identifier < len(values):
                return values[identifier]
            else:
                return default
        return super(Overlay, self).get(identifier, default)


    def __add__(self, other):
        return Layout.from_values([self, other])


    def __mul__(self, other):
        if type(other).__name__ == 'DynamicMap':
            from .spaces import Callable
            def dynamic_mul(*args, **kwargs):
                element = other[args]
                return self * element
            callback = Callable(dynamic_mul, inputs=[self, other])
            callback._is_overlay = True
            return other.clone(shared_data=False, callback=callback,
                               streams=[])
        elif not isinstance(other, ViewableElement):
            raise NotImplementedError
        return Overlay.from_values([self, other])


    def collate(self):
        """
        Collates any objects in the Overlay resolving any issues
        the recommended nesting structure.
        """
        return reduce(lambda x,y: x*y, self.values())


    def collapse(self, function):
        """
        Collapses all the Elements in the Overlay using the
        supplied function if they share a common type and group.
        """
        elements = list(self)
        types = [type(el) for el in elements]
        values = [el.group for el in elements]
        if not len(set(types)) == 1 and len(set(values)) == 1:
            raise Exception("Overlay is not homogeneous in type or group "
                            "and cannot be collapsed.")
        else:
            return elements[0].clone(types[0].collapse_data([el.data for el in elements],
                                                            function, self.kdims))

    @property
    def group(self):
        if self._group:
            return self._group
        elements = [el for el in self if not el._auxiliary_component]
        values = {el.group for el in elements}
        types = {type(el) for el in elements}
        if values:
            group = list(values)[0]
            vtype = list(types)[0].__name__
        else:
            group, vtype = [], ''
        if len(values) == 1 and group != vtype:
            return group
        else:
            return type(self).__name__

    @group.setter
    def group(self, group):
        if not sanitize_identifier.allowable(group):
            raise ValueError("Supplied group %s contains invalid characters." %
                             group)
        else:
            self._group = group

    @property
    def label(self):
        if self._label:
            return self._label
        labels = {el.label for el in self
                  if not el._auxiliary_component}
        if len(labels) == 1:
            return list(labels)[0]
        else:
            return ''

    @label.setter
    def label(self, label):
        if not sanitize_identifier.allowable(label):
            raise ValueError("Supplied group %s contains invalid characters." %
                             label)
        self._label = label

    @property
    def ddims(self):
        dimensions = []
        dimension_names = []
        for el in self:
            for dim in el.dimensions():
                if dim.name not in dimension_names:
                    dimensions.append(dim)
                    dimension_names.append(dim.name)
        return dimensions

    @property
    def shape(self):
        raise NotImplementedError



class NdOverlay(UniformNdMapping, CompositeOverlay, Overlayable):
    """
    An NdOverlay allows a group of NdOverlay to be overlaid together. NdOverlay can
    be indexed out of an overlay and an overlay is an iterable that iterates
    over the contained layers.
    """

    kdims = param.List(default=[Dimension('Element')], constant=True, doc="""
        List of dimensions the NdOverlay can be indexed by.""")

    _deep_indexable = True

    def __init__(self, overlays=None, kdims=None, **params):
        super(NdOverlay, self).__init__(overlays, kdims=kdims, **params)


__all__ = list(set([_k for _k, _v in locals().items()
                    if isinstance(_v, type) and issubclass(_v, Dimensioned)])) + ['Overlayable']
