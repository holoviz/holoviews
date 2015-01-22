"""
Supplies Layer and related classes that allow overlaying of Views,
including Overlay. A Layer is the final extension of View base class
that allows Views to be overlayed on top of each other.

Also supplies ViewMap which is the primary multi-dimensional Map type
for indexing, slicing and animating collections of Views.
"""

from numbers import Number
from collections import OrderedDict
import itertools

import numpy as np

import param

from .dimension import Dimension, DimensionedData, DataElement
from .ndmapping import NdMapping
from .layout import Composable, AdjointLayout, ViewTree
from .ndmapping import UniformNdMapping
from .options import options, channels
from .util import find_minmax


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
        return Overlay(items=self_item + other_items)



class CompositeOverlay(DataElement, Composable):
    """
    CompositeOverlay provides a common baseclass for Overlay classes.
    """

    channels = channels

    @property
    def labels(self):
        return [el.label for el in self]

    @property
    def legend(self):
        if self._cached_index_names == ['Element']:
            labels = self.labels
            if len(set(labels)) == len(labels):
                return labels
            else:
                return None
        else:
            labels = []
            for key in self.data.keys():
                labels.append(','.join([dim.pprint_value(k) for dim, k in
                                        zip(self.key_dimensions, key)]))
            return labels

    @property
    def style(self):
        return [el.style for el in self][0]

    @style.setter
    def style(self, styles):
        for layer, style in zip(self, styles):
            layer.style = style


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


    def __getstate__(self):
        """
        When pickling, make sure to save the relevant channel
        definitions.
        """
        obj_dict = self.__dict__.copy()
        channels = dict((k, self.channels[k]) for k in self.channels.keys())
        obj_dict['channel_definitions'] = channels
        return obj_dict


    def __setstate__(self, d):
        """
        When unpickled, restore the saved channel definitions.
        """

        if 'channel_definitions' not in d:
            self.__dict__.update(d)
            return

        unpickled_channels = d.pop('channel_definitions')
        for key, defs in unpickled_channels.items():
            self.channels[key] = defs
        self.__dict__.update(d)



class Overlay(ViewTree, CompositeOverlay, Composable):
    """
    An Overlay consists of multiple Views (potentially of
    heterogeneous type) presented one on top each other with a
    particular z-ordering.

    Overlays along with Views constitute the only valid leaf types of
    a ViewTree and in fact extend the ViewTree structure. Overlays are
    constructed using the * operator (building an identical structure
    to the + operator) and are the only objects that inherit both from
    ViewTree and DataElement.
    """

    def __init__(self, items=None, **params):
        view_params = DataElement.params().keys()
        ViewTree.__init__(self, items,
                          **{k:v for k,v in params.items() if k not in view_params})
        DataElement.__init__(self, self.data,
                      **{k:v for k,v in params.items() if k in view_params})


    def __mul__(self, other):
        if isinstance(other, Overlay):
            items = list(self.data.items()) + list(other.data.items())
        elif isinstance(other, DataElement):
            label = other.label if other.label else 'I'
            items = list(self.data.items()) + [((other.value, label), other)]
        elif isinstance(other, UniformNdMapping):
            raise NotImplementedError

        return Overlay(items=items).display('all')


    def dimension_values(self, dimension):
        values = []
        for el in self:
            if dimension in [dim.name for dim in el.dimensions]:
                values.append(el.dimension_values(dimension))
        return np.concatenate(values)


    @property
    def deep_dimensions(self):
        dimensions = []
        dimension_names = []
        for el in self:
            for dim in el.dimensions:
                if dim.name not in dimension_names:
                    dimensions.append(dim)
                    dimension_names.append(dim.name)
        return dimensions

    @property
    def ranges(self):
        ranges = {}
        for el in self:
            ranges[el.settings] = {dim.name: el.range(dim.name) for dim in el.dimensions}



class NdOverlay(CompositeOverlay, NdMapping, Overlayable):
    """
    An NdOverlay allows a group of NdOverlay to be overlaid together. NdOverlay can
    be indexed out of an overlay and an overlay is an iterable that iterates
    over the contained layers.
    """

    key_dimensions = param.List(default=[Dimension('Element')], constant=True, doc="""List
      of dimensions the NdOverlay can be indexed by.""")

    value = param.String(default='NdOverlay')

    _deep_indexable = True

    def __init__(self, overlays, **params):
        self._xlim = None
        self._ylim = None
        data = self._process_layers(overlays)
        DataElement.__init__(self, data, **params)
        NdMapping.__init__(self, data, **params)


    def dimension_values(self, *args, **kwargs):
        NdMapping.dimension_values(self, *args, **kwargs)


    def _process_layers(self, layers):
        """
        Set a collection of layers to be overlaid with each other.
        """
        if isinstance(layers, (UniformNdMapping)):
            return layers.data
        elif isinstance(layers, (dict, OrderedDict)):
            return layers
        elif layers is None or not len(layers):
            return OrderedDict()
        else:
            keys = range(len(layers))
            return OrderedDict(((key,), layer) for key, layer in zip(keys, layers))

    def set(self, layers):
        data = self._process_layers(layers)
        self.data = data


    @property
    def labels(self):
        return [el.label for el in self]


    def item_check(self, dim_vals, layer):
        if not isinstance(layer, DataElement): pass
        layer_dimensions = [d.name for d in layer.key_dimensions]
        if len(self):
            if layer_dimensions != self._layer_dimensions:
                raise Exception("NdOverlay must share common dimensions.")
        else:
            self._layer_dimensions = layer_dimensions
            self.value = layer.value
            self.label = layer.label


    def add(self, layer):
        """
        NdOverlay a single layer on top of the existing overlay.
        """
        self[len(self)] = layer

    @property
    def layer_types(self):
        """
        The type of NdOverlay stored in the NdOverlay.
        """
        if len(self) == 0:
            return None
        else:
            return tuple(set(layer.__class__ for layer in self))

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
    def lbrt(self):
        l, r = self.xlim if self.xlim else (np.NaN, np.NaN)
        b, t = self.ylim if self.ylim else (np.NaN, np.NaN)
        return l, b, r, t



class AxisLayout(NdMapping):
    """
    Grids are distinct from GridLayouts as they ensure all contained elements
    to be of the same type. Unlike GridLayouts, which have integer keys,
    Grids usually have floating point keys, which correspond to a grid
    sampling in some two-dimensional space. This two-dimensional space may
    have to arbitrary dimensions, e.g. for 2D parameter spaces.
    """

    key_dimensions = param.List(default=[Dimension(name="X"), Dimension(name="Y")], bounds=(2,2))

    label = param.String(constant=True, doc="""
      A short label used to indicate what kind of data is contained
      within the AxisLayout.""")

    title = param.String(default="{label} {value}", doc="""
      The title formatting string for the AxisLayout object""")

    value = param.String(default='AxisLayout')

    def __init__(self, initial_items=None, **params):
        super(AxisLayout, self).__init__(initial_items, **params)
        if self.ndims > 2:
            raise Exception('Grids can have no more than two dimensions.')
        self._style = None
        self._type = None


    def __mul__(self, other):
        if isinstance(other, AxisLayout):
            if set(self.keys()) != set(other.keys()):
                raise KeyError("Can only overlay two ParameterGrids if their keys match")
            zipped = zip(self.keys(), self.values(), other.values())
            overlayed_items = [(k, el1 * el2) for (k, el1, el2) in zipped]
            return self.clone(overlayed_items)
        elif isinstance(other, UniformNdMapping) and len(other) == 1:
            view = other.last
        elif isinstance(other, UniformNdMapping) and len(other) != 1:
            raise Exception("Can only overlay with HoloMap of length 1")
        else:
            view = other

        overlayed_items = [(k, el * view) for k, el in self.items()]
        return self.clone(overlayed_items)


    def _transform_indices(self, key):
        """
        Transforms indices by snapping to the closest value if
        values are numeric, otherwise applies no transformation.
        """
        if all(not isinstance(el, slice) for el in key):
            dim_inds = [self.get_dimension_index(l) for l in self._cached_index_names
                        if issubclass(self.get_dimension_type(l), Number)]
            str_keys = iter(key[i] for i in range(self.ndims)
                            if i not in dim_inds)
            num_keys = []
            if len(dim_inds):
                keys = list({tuple(k[i] for i in dim_inds)
                             for k in self.keys()})
                q = np.array([tuple(key[i] for i in dim_inds)])
                idx = np.argmin([np.inner(q - np.array(x), q - np.array(x))
                                 if len(dim_inds) == 2 else np.abs(q-x)
                                     for x in keys])
                num_keys = iter(keys[idx])
            key = tuple(next(num_keys) if i in dim_inds else next(str_keys)
                        for i in range(self.ndims))
        elif any(not isinstance(el, slice) for el in key):
            index_ind = [idx for idx, el in enumerate(key)
                         if not isinstance(el, (slice, str))][0]
            dim_keys = np.array([k[index_ind] for k in self.keys()])
            snapped_val = dim_keys[np.argmin(dim_keys-key[index_ind])]
            key = list(key)
            key[index_ind] = snapped_val
            key = tuple(key)
        return key


    def keys(self, full_grid=False):
        """
        Returns a complete set of keys on a AxisLayout, even when AxisLayout isn't fully
        populated. This makes it easier to identify missing elements in the
        AxisLayout.
        """
        keys = super(AxisLayout, self).keys()
        if self.ndims == 1 or not full_grid:
            return keys
        dim1_keys = sorted(set(k[0] for k in keys))
        dim2_keys = sorted(set(k[1] for k in keys))
        return [(d1, d2) for d1 in dim1_keys for d2 in dim2_keys]


    @property
    def last(self):
        """
        The last of a AxisLayout is another AxisLayout
        constituted of the last of the individual elements. To access
        the elements by their X,Y position, either index the position
        directly or use the items() method.
        """

        last_items = [(k, v.clone(items=(list(v.keys())[-1], v.last)))
                      for (k, v) in self.items()]
        return self.clone(last_items)


    @property
    def type(self):
        """
        The type of elements stored in the AxisLayout.
        """
        if self._type is None:
            if not len(self) == 0:
                item = self.values()[0]
                self._type = item.type if isinstance(item, UniformNdMapping) else item.__class__
        return self._type


    @property
    def layer_types(self):
        """
        The type of layers stored in the AxisLayout.
        """
        if self.type == NdOverlay:
            return self.values()[0].layer_types
        else:
            return (self.type,)


    def __len__(self):
        """
        The maximum depth of all the elements. Matches the semantics
        of __len__ used by Maps. For the total number of elements,
        count the full set of keys.
        """
        return max([(len(v) if hasattr(v, '__len__') else 1) for v in self.values()] + [0])


    def __add__(self, obj):
        return ViewTree.from_view(self) + ViewTree.from_view(obj)


    @property
    def all_keys(self):
        """
        Returns a list of all keys of the elements in the grid.
        """
        keys_list = []
        for v in self.values():
            if isinstance(v, AdjointLayout):
                v = v.main
            if isinstance(v, UniformNdMapping):
                keys_list.append(list(v.data.keys()))
        return sorted(set(itertools.chain(*keys_list)))


    @property
    def common_keys(self):
        """
        Returns a list of common keys. If all elements in the AxisLayout share
        keys it will return the full set common of keys, otherwise returns
        None.
        """
        keys_list = []
        for v in self.values():
            if isinstance(v, AdjointLayout):
                v = v.main
            if isinstance(v, UniformNdMapping):
                keys_list.append(list(v.data.keys()))
        if all(x == keys_list[0] for x in keys_list):
            return keys_list[0]
        else:
            return None

    @property
    def shape(self):
        keys = self.keys()
        if self.ndims == 1:
            return (1, len(keys))
        return len(set(k[0] for k in keys)), len(set(k[1] for k in keys))


    @property
    def xlim(self):
        xlim = list(self.values())[-1].xlim
        for data in self.values():
            xlim = find_minmax(xlim, data.xlim)
        return xlim


    @property
    def ylim(self):
        ylim = list(self.values())[-1].ylim
        for data in self.values():
            ylim = find_minmax(ylim, data.ylim)
        if ylim[0] == ylim[1]: ylim = (ylim[0], ylim[0]+1.)
        return ylim

    @property
    def lbrt(self):
        if self.xlim is None: return np.NaN, np.NaN, np.NaN, np.NaN
        l, r = self.xlim
        b, t = self.ylim
        return float(l), float(b), float(r), float(t)

    @property
    def grid_lbrt(self):
        grid_dimensions = []
        for dim in self._cached_index_names:
            grid_dimensions.append(self.range(dim))
        if self.ndims == 1:
            grid_dimensions.append((0, 1))
        xdim, ydim = grid_dimensions
        return (xdim[0], ydim[0], xdim[1], ydim[1])


    @property
    def style(self):
        """
        The name of the style that may be used to control display of
        this element.
        """
        if self._style:
            return self._style

        class_name = self.__class__.__name__
        matches = options.fuzzy_match_keys(class_name)
        return matches[0] if matches else class_name


    @style.setter
    def style(self, val):
        self._style = val


    def dframe(self):
        """
        Gets a Pandas dframe from each of the items in the AxisLayout, appends the
        AxisLayout coordinates and concatenates all the dframes.
        """
        import pandas
        dframes = []
        for coords, vmap in self.items():
            map_frame = vmap.dframe()
            for coord, dim in zip(coords, self._cached_index_names)[::-1]:
                if dim in map_frame: dim = 'Grid_' + dim
                map_frame.insert(0, dim.replace(' ','_'), coord)
            dframes.append(map_frame)
        return pandas.concat(dframes)


__all__ = list(set([_k for _k, _v in locals().items()
                    if isinstance(_v, type) and issubclass(_v, DimensionedData,)])) + ['Overlayable']

