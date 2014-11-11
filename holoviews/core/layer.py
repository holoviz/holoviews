"""
Supplies Layer and related classes that allow overlaying of Views,
including Overlay. A Layer is the final extension of View base class
that allows Views to be overlayed on top of each other.

Also supplies ViewMap which is the primary multi-dimensional Map type
for indexing, slicing and animating collections of Views.
"""

from collections import OrderedDict, defaultdict
import itertools
import numpy as np

import param

from .dimension import Dimension, Dimensioned
from .ndmapping import NdMapping
from .layout import Pane, GridLayout, AdjointLayout
from .options import options, channels
from .view import View, Map


def find_minmax(lims, olims):
    """
    Takes (a1, a2) and (b1, b2) as input and returns
    (np.min(a1, b1), np.max(a2, b2)). Used to calculate
    min and max values of a number of items.
    """

    limzip = zip(list(lims), list(olims), [np.min, np.max])
    return tuple([float(fn([l, ol])) for l, ol, fn in limzip])


class Layer(Pane):
    """
    Layer is the baseclass for all 2D View types, with an x- and
    y-dimension. Subclasses should define the data storage in the
    constructor, as well as methods and properties, which define how
    the data maps onto the x- and y- and value dimensions.
    """

    dimensions = param.List(default=[Dimension('X')], doc="""
        Dimensions on Layers determine the number of indexable
        dimensions.""")

    value = param.ClassSelector(class_=Dimension, default=Dimension('Y'))


    def __mul__(self, other):
        if isinstance(other, ViewMap):
            items = [(k, self * v) for (k, v) in other.items()]
            return other.clone(items=items)

        self_layers = self.data if isinstance(self, Overlay) else [self]
        other_layers = other.data if isinstance(other, Overlay) else [other]
        combined_layers = self_layers + other_layers

        return Overlay(combined_layers)


    ########################
    # Subclassable methods #
    ########################


    def __init__(self, data, **kwargs):
        lbrt = kwargs.pop('lbrt', None)
        self._xlim = (lbrt[0], lbrt[2]) if lbrt else None
        self._ylim = (lbrt[1], lbrt[3]) if lbrt else None
        super(Layer, self).__init__(data, **kwargs)


    @property
    def cyclic_range(self):
        if self.dimensions[0].cyclic:
            return self.dimensions[0].range[1]
        else:
            return None

    @property
    def range(self):
        if self.cyclic_range:
            return self.cyclic_range
        y_vals = self.data[:, 1]
        return (float(min(y_vals)), float(max(y_vals)))

    @property
    def xlabel(self):
        return self.dimensions[0].pprint_label

    @property
    def ylabel(self):
        if len(self.dimensions) == 1:
            return self.value.pprint_label
        else:
            return self.dimensions[1].pprint_label

    @property
    def xlim(self):
        if self._xlim:
            return self._xlim
        elif self.cyclic_range is not None:
            return (0, self.cyclic_range)
        elif len(self):
            x_vals = self.data[:, 0]
            return (float(min(x_vals)), float(max(x_vals)))
        else:
            return None

    @xlim.setter
    def xlim(self, limits):
        if self.cyclic_range:
            self.warning('Cannot override the limits of a '
                         'cyclic dimension.')
        elif limits is None or (isinstance(limits, tuple) and len(limits) == 2):
            self._xlim = limits
        else:
            raise ValueError('xlim needs to be a length two tuple or None.')

    @property
    def ylim(self):
        if self._ylim:
            return self._ylim
        elif len(self):
            y_vals = self.data[:, 1]
            return (float(min(y_vals)), float(max(y_vals)))
        else:
            return None

    @ylim.setter
    def ylim(self, limits):
        if limits is None or (isinstance(limits, tuple) and len(limits) == 2):
            self._ylim = limits
        else:
            raise ValueError('xlim needs to be a length two tuple or None.')

    @property
    def lbrt(self):
        l, r = self.xlim if self.xlim else (np.NaN, np.NaN)
        b, t = self.ylim if self.ylim else (np.NaN, np.NaN)
        return l, b, r, t


    @lbrt.setter
    def lbrt(self, lbrt):
        l, b, r, t = lbrt
        self.xlim, self.ylim = (l, r), (b, t)



class Overlay(Pane, NdMapping):
    """
    An Overlay allows a group of Layers to be overlaid together. Layers can
    be indexed out of an overlay and an overlay is an iterable that iterates
    over the contained layers.
    """

    dimensions = param.List(default=[Dimension('Layer')], constant=True, doc="""List
      of dimensions the Overlay can be indexed by.""")

    label = param.String(doc="""
      A short label used to indicate what kind of data is contained
      within the view object.

      Overlays should not have their label set directly by the user as
      the label is only for defining custom channel operations.""")

    title = param.String(default="{label}")

    channels = channels
    _deep_indexable = True

    def __init__(self, overlays, **kwargs):
        self._xlim = None
        self._ylim = None
        data = self._process_layers(overlays)
        super(Overlay, self).__init__(data, **kwargs)
        self._data = self.data


    def _process_layers(self, layers):
        """
        Set a collection of layers to be overlaid with each other.
        """
        if isinstance(layers, (ViewMap)):
            return layers._data
        elif isinstance(layers, (dict, OrderedDict)):
            return layers
        elif layers is None or not len(layers):
            return OrderedDict()
        else:
            keys = range(len(layers))
            return OrderedDict(((key,), layer) for key, layer in zip(keys, layers))

    def set(self, layers):
        data = self._process_layers(layers)
        self._data = data


    @property
    def labels(self):
        return [el.label for el in self]


    @property
    def legend(self):
        if self.dimension_labels == ['Layer']:
            labels = self.labels
            if len(set(labels)) == len(labels):
                return labels
            else:
                return None
        else:
            labels = []
            for key in self.keys():
                labels.append(','.join([dim.pprint_value(k) for dim, k in
                                        zip(self.dimensions, key)]))
            return labels


    @property
    def style(self):
        return [el.style for el in self]

    @style.setter
    def style(self, styles):
        for layer, style in zip(self, styles):
            layer.style = style


    def item_check(self, dim_vals, layer):
        if not isinstance(layer, Layer): pass
        elif not len(self):
            self._layer_dimensions = layer.dimension_labels
            self.value = layer.value
            self.label = layer.label
        else:
            if layer.xlim:
                self.xlim = layer.xlim if self.xlim is None else find_minmax(self.xlim, layer.xlim)
            if layer.xlim:
                self.ylim = layer.ylim if self.ylim is None else find_minmax(self.ylim, layer.ylim)
            if layer.dimension_labels != self._layer_dimensions:
                raise Exception("DataLayers must share common dimensions.")
        if layer.label in [o.label for o in self.data]:
            self.warning('Label %s already defined in Overlay' % layer.label)


    def add(self, layer):
        """
        Overlay a single layer on top of the existing overlay.
        """
        self[len(self)] = layer


    @property
    def layer_types(self):
        """
        The type of Layers stored in the Overlay.
        """
        if len(self) == 0:
            return None
        else:
            return tuple(set(layer.__class__ for layer in self))

    @property
    def xlim(self):
        xlim = self[0].xlim if isinstance(self[0], Layer) else None
        for data in self:
            data_xlim = data.xlim if isinstance(data, Layer) else None
            xlim = find_minmax(xlim, data.xlim) if data_xlim and xlim else xlim
        return xlim

    @xlim.setter
    def xlim(self, limits):
        if limits is None or (isinstance(limits, tuple) and len(limits) == 2):
            self._xlim = limits
        else:
            raise ValueError('xlim needs to be a length two tuple or None.')

    @property
    def ylim(self):
        ylim = self[0].ylim if isinstance(self[0], Layer) else None
        for data in self:
            data_ylim = data.ylim if isinstance(data, Layer) else None
            ylim = find_minmax(ylim, data.ylim) if data_ylim and ylim else ylim
        return ylim

    @ylim.setter
    def ylim(self, limits):
        if limits is None or (isinstance(limits, tuple) and len(limits) == 2):
            self._ylim = limits
        else:
            raise ValueError('ylim needs to be a length two tuple or None.')

    @property
    def range(self):
        range = self[0].range if self[0].range is not None else None
        cyclic = self[0].cyclic_range is not None
        for view in self:
            if cyclic != (self[0].cyclic_range is not None):
                raise Exception("Overlay contains cyclic and non-cyclic "
                                "Views, cannot compute range.")
            range = find_minmax(range, view.range) if view.range is not None else range
        return range


    @property
    def cyclic_range(self):
        return self[0].cyclic_range if len(self) else None


    def __mul__(self, other):
        if isinstance(other, ViewMap):
            items = [(k, self * v) for (k, v) in other.items()]
            return other.clone(items=items)
        elif isinstance(other, Overlay):
            overlays = self.values() + other.values()
        elif isinstance(other, (View)):
            overlays = self.values() + [other]
        else:
            raise TypeError('Can only create an overlay of holoviews.')

        return Overlay(overlays)


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



class Grid(NdMapping):
    """
    Grids are distinct from GridLayouts as they ensure all contained elements
    to be of the same type. Unlike GridLayouts, which have integer keys,
    Grids usually have floating point keys, which correspond to a grid
    sampling in some two-dimensional space. This two-dimensional space may
    have to arbitrary dimensions, e.g. for 2D parameter spaces.
    """

    dimensions = param.List(default=[Dimension(name="X"), Dimension(name="Y")])

    label = param.String(constant=True, doc="""
      A short label used to indicate what kind of data is contained
      within the Grid.""")

    title = param.String(default='{label}', doc="""
       The title formatting string allows the title to be composed
       from the label and type.""")

    def __init__(self, initial_items=None, **params):
        super(Grid, self).__init__(initial_items, **params)
        if self.ndims > 2:
            raise Exception('Grids can have no more than two dimensions.')
        self._style = None
        self._type = None


    def __mul__(self, other):
        if isinstance(other, Grid):
            if set(self.keys()) != set(other.keys()):
                raise KeyError("Can only overlay two ParameterGrids if their keys match")
            zipped = zip(self.keys(), self.values(), other.values())
            overlayed_items = [(k, el1 * el2) for (k, el1, el2) in zipped]
            return self.clone(overlayed_items)
        elif isinstance(other, ViewMap) and len(other) == 1:
            view = other.last
        elif isinstance(other, ViewMap) and len(other) != 1:
            raise Exception("Can only overlay with ViewMap of length 1")
        else:
            view = other

        overlayed_items = [(k, el * view) for k, el in self.items()]
        return self.clone(overlayed_items)


    def _transform_indices(self, key):
        if all(not isinstance(el, slice) for el in key):
            keys = self.keys()
            q = np.array(key)
            idx = np.argmin([np.inner(q - np.array(x), q - np.array(x))
                             if self.ndims == 2 else np.abs(q-x)
                             for x in keys])
            key = keys[idx]
        elif any(not isinstance(el, slice) for el in key):
            index_ind = [idx for idx, el in enumerate(key) if not isinstance(el, slice)][0]
            dim_keys = np.array([k[index_ind] for k in self.keys()])
            snapped_val = dim_keys[np.argmin(dim_keys-key[index_ind])]
            key = list(key)
            key[index_ind] = snapped_val
            key = tuple(key)
        return key


    def keys(self, full_grid=False):
        """
        Returns a complete set of keys on a Grid, even when Grid isn't fully
        populated. This makes it easier to identify missing elements in the
        Grid.
        """
        keys = super(Grid, self).keys()
        if self.ndims == 1 or not full_grid:
            return keys
        dim1_keys = sorted(set(k[0] for k in keys))
        dim2_keys = sorted(set(k[1] for k in keys))
        return [(d1, d2) for d1 in dim1_keys for d2 in dim2_keys]


    @property
    def last(self):
        """
        The last of a Grid is another Grid
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
        The type of elements stored in the Grid.
        """
        if self._type is None:
            if not len(self) == 0:
                item = self.values()[0]
                self._type = item.type if isinstance(item, ViewMap) else item.__class__
        return self._type


    @property
    def layer_types(self):
        """
        The type of layers stored in the Grid.
        """
        if self.type == Overlay:
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
        if not isinstance(obj, GridLayout):
            return GridLayout(initial_items=[self, obj])

    @property
    def all_keys(self):
        """
        Returns a list of all keys of the elements in the grid.
        """
        keys_list = []
        for v in self.values():
            if isinstance(v, AdjointLayout):
                v = v.main
            if isinstance(v, ViewMap):
                keys_list.append(list(v._data.keys()))
        return sorted(set(itertools.chain(*keys_list)))


    @property
    def common_keys(self):
        """
        Returns a list of common keys. If all elements in the Grid share
        keys it will return the full set common of keys, otherwise returns
        None.
        """
        keys_list = []
        for v in self.values():
            if isinstance(v, AdjointLayout):
                v = v.main
            if isinstance(v, ViewMap):
                keys_list.append(list(v._data.keys()))
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
        for dim in self.dimension_labels:
            grid_dimensions.append(self.dim_range(dim))
        if self.ndims == 1:
            grid_dimensions.append((0, 1))
        xdim, ydim = grid_dimensions
        return (xdim[0], ydim[0], xdim[1], ydim[1])


    @property
    def style(self):
        """
        The name of the style that may be used to control display of
        this view.
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
        Gets a Pandas dframe from each of the items in the Grid, appends the
        Grid coordinates and concatenates all the dframes.
        """
        import pandas
        dframes = []
        for coords, vmap in self.items():
            map_frame = vmap.dframe()
            for coord, dim in zip(coords, self.dimension_labels)[::-1]:
                if dim in map_frame: dim = 'Grid_' + dim
                map_frame.insert(0, dim.replace(' ','_'), coord)
            dframes.append(map_frame)
        return pandas.concat(dframes)



class ViewMap(Map):
    """
    A ViewMap can hold any number of DataLayers indexed by a list of
    dimension values. It also has a number of properties, which can find
    the x- and y-dimension limits and labels.
    """

    data_type = (View, Map)

    @property
    def range(self):
        if not hasattr(self.last, 'range'):
            raise Exception('View type %s does not implement range.' % type(self.last))
        range = self.last.range
        for view in self._data.values():
            range = find_minmax(range, view.range)
        return range


    @property
    def layer_types(self):
        """
        The type of layers stored in the ViewMap.
        """
        if self.type == Overlay:
            return self.last.layer_types
        else:
            return (self.type)


    @property
    def xlabel(self):
        if not issubclass(self.type, (Layer, Overlay)): return None
        return self.last.xlabel


    @property
    def ylabel(self):
        if not issubclass(self.type, (Layer, Overlay)): return None
        return self.last.ylabel


    @property
    def xlim(self):
        if not issubclass(self.type, (Layer, Overlay)): return None
        xlim = self.last.xlim
        for data in self.values():
            xlim = find_minmax(xlim, data.xlim) if data.xlim and xlim else xlim
        return xlim


    @property
    def ylim(self):
        if not issubclass(self.type, (Layer, Overlay)): return None
        ylim = self.last.ylim
        for data in self.values():
            ylim = find_minmax(ylim, data.ylim) if data.ylim and ylim else ylim
        return ylim


    @property
    def lbrt(self):
        if self.xlim is None: return np.NaN, np.NaN, np.NaN, np.NaN
        l, r = self.xlim
        b, t = self.ylim
        return float(l), float(b), float(r), float(t)



    def overlay(self, dimensions):
        """
        Splits the Map along a specified number of dimensions and
        overlays items in the split out Maps.
        """
        if self.ndims == 1:
            split_map = dict(default=self)
            new_map = dict()
        else:
            split_map = self.split_dimensions(dimensions)
            new_map = self.clone(dimensions=split_map.dimensions)

        for outer, vmap in split_map.items():
            new_map[outer] = Overlay(vmap)

        if self.ndims == 1:
            return list(new_map.values())[0]
        else:
            return new_map


    def grid(self, dimensions, layout=False, constant_dims=True):
        """
        Grid takes a list of one or two dimensions, and lays out the containing
        Views along these axes in a Grid.
        """
        if len(dimensions) > 2:
            raise ValueError('At most two dimensions can be laid out in a grid.')

        if len(dimensions) == self.ndims:
            split_map = self
        elif all(d in self.dimension_labels for d in dimensions):
            split_dims = [d for d in self.dimension_labels if d not in dimensions]
            split_map = self.split_dimensions(split_dims)
            split_map = split_map.reindex(dimensions)
        else:
            raise ValueError('ViewMap does not have supplied dimensions.')

        if layout:
            for keys, vmap in split_map._data.items():
                if constant_dims:
                    vmap.constant_dimensions = split_map.dimensions
                    vmap.constant_values = keys
            return GridLayout(split_map)
        else:
            return Grid(split_map, dimensions=split_map.dimensions)

    def split_overlays(self):
        """
        Given a Map of Overlays of N layers, split out the layers into
        N separate Maps.
        """
        if self.type is not Overlay:
            return self.clone(self.items())

        maps = []
        item_maps = defaultdict(list)
        for k, overlay in self.items():
            for i, el in enumerate(overlay):
                item_maps[i].append((k, el))

        for k in sorted(item_maps.keys()):
            maps.append(self.clone(item_maps[k]))
        return maps


    def __mul__(self, other):
        """
        The mul (*) operator implements overlaying of different Views.
        This method tries to intelligently overlay Maps with differing
        keys. If the Map is mulled with a simple View each element in
        the Map is overlaid with the View. If the element the Map is
        mulled with is another Map it will try to match up the
        dimensions, making sure that items with completely different
        dimensions aren't overlaid.
        """
        if isinstance(other, self.__class__):
            self_set = set(self.dimension_labels)
            other_set = set(other.dimension_labels)

            # Determine which is the subset, to generate list of keys and
            # dimension labels for the new view
            self_in_other = self_set.issubset(other_set)
            other_in_self = other_set.issubset(self_set)
            dimensions = self.dimensions
            if self_in_other and other_in_self: # superset of each other
                super_keys = sorted(set(self.dimension_keys() + other.dimension_keys()))
            elif self_in_other: # self is superset
                dimensions = other.dimensions
                super_keys = other.dimension_keys()
            elif other_in_self: # self is superset
                super_keys = self.dimension_keys()
            else: # neither is superset
                raise Exception('One set of keys needs to be a strict subset of the other.')

            items = []
            for dim_keys in super_keys:
                # Generate keys for both subset and superset and sort them by the dimension index.
                self_key = tuple(k for p, k in sorted(
                    [(self.dim_index(dim), v) for dim, v in dim_keys
                     if dim in self.dimension_labels]))
                other_key = tuple(k for p, k in sorted(
                    [(other.dim_index(dim), v) for dim, v in dim_keys
                     if dim in other.dimension_labels]))
                new_key = self_key if other_in_self else other_key
                # Append SheetOverlay of combined items
                if (self_key in self) and (other_key in other):
                    items.append((new_key, self[self_key] * other[other_key]))
                elif self_key in self:
                    items.append((new_key, self[self_key] * other.empty_element))
                else:
                    items.append((new_key, self.empty_element * other[other_key]))
            return self.clone(items=items, dimensions=dimensions)
        elif isinstance(other, self.data_type):
            items = [(k, v * other) for (k, v) in self.items()]
            return self.clone(items=items)
        else:
            raise Exception("Can only overlay with {data} or {vmap}.".format(
                data=self.data_type, vmap=self.__class__.__name__))


    def dframe(self):
        """
        Gets a dframe for each View in the Map, appends the dimensions
        of the Map as series and concatenates the dframes.
        """
        import pandas
        dframes = []
        for key, view in self._data.items():
            view_frame = view.dframe()
            for val, dim in reversed(zip(key, self.dimension_labels)):
                dim = dim.replace(' ', '_')
                dimn = 1
                while dim in view_frame:
                    dim = dim+'_%d' % dimn
                    if dim in view_frame:
                        dimn += 1
                view_frame.insert(0, dim, val)
            dframes.append(view_frame)
        return pandas.concat(dframes)


    def __add__(self, obj):
        if not isinstance(obj, GridLayout):
            return GridLayout(initial_items=[self, obj])
        else:
            grid = GridLayout(initial_items=[self])
            grid.update(obj)
            return grid


    def __lshift__(self, other):
        if isinstance(other, (View, Overlay, ViewMap, Grid)):
            return AdjointLayout([self, other])
        elif isinstance(other, AdjointLayout):
            return AdjointLayout(other.data+[self])
        else:
            raise TypeError('Cannot append {0} to a AdjointLayout'.format(type(other).__name__))


    def sample(self, samples, bounds=None, **sample_values):
        """
        Sample each Layer in the Map by passing either a list of
        samples or a tuple specifying the number of regularly spaced
        samples per dimension. Alternatively, a single sample may be
        requested using dimension-value pairs. Optionally, the bounds
        argument can be used to specify the bounding extent from which
        the coordinates are to regularly sampled. Regular sampling
        assumes homogenous and regularly sampled data.

        For 1D sampling, the shape is simply as the desired number of
        samples (and not a tuple). The bounds format for 1D sampling
        is the tuple (lower, upper) and the tuple (left, bottom,
        right, top) for 2D sampling.
        """
        dims = self.last.ndims
        if isinstance(samples, tuple) or np.isscalar(samples):
            if dims == 1:
                lower, upper = (self.xlims[0],self.xlims[1]) if bounds is None else bounds
                edges = np.linspace(lower, upper, samples+1)
                linsamples = [(l+u)/2.0 for l,u in zip(edges[:-1], edges[1:])]
            elif dims == 2:
                (rows, cols) = samples
                (l,b,r,t) = self.last.lbrt if bounds is None else bounds

                xedges = np.linspace(l, r, cols+1)
                yedges = np.linspace(b, t, rows+1)
                xsamples = [(l+u)/2.0 for l,u in zip(xedges[:-1], xedges[1:])]
                ysamples = [(l+u)/2.0 for l,u in zip(yedges[:-1], yedges[1:])]

                X,Y = np.meshgrid(xsamples, ysamples)
                linsamples = zip(X.flat, Y.flat)
            else:
                raise NotImplementedError("Regular sampling not implented"
                                          "for high-dimensional Views.")

            samples = set(self.last.closest(linsamples))

        sampled_items = [(k, view.sample(samples, **sample_values))
                         for k, view in self.items()]
        return self.clone(sampled_items)


    def reduce(self, label_prefix='', **reduce_map):
        """
        Reduce each Matrix in the Map using a function supplied
        via the kwargs, where the keyword has to match a particular
        dimension in the View.
        """
        reduced_items = [(k, v.reduce(label_prefix=label_prefix, **reduce_map))
                         for k, v in self.items()]
        return self.clone(reduced_items)


    def collate(self, collate_dim):
        """
        Collate splits out the specified dimension and joins the
        samples in each of the split out Maps into Curves. If there
        are multiple entries in the ItemTable it will lay them out
        into a Grid.
        """
        from ..operation import table_collate
        return table_collate(self, collation_dim=collate_dim)


    def map(self, map_fn, **kwargs):
        """
        Map a function across the ViewMap, using the bounds of first
        mapped item.
        """
        mapped_items = [(k, map_fn(el, k)) for k, el in self.items()]
        return self.clone(mapped_items, **kwargs)


    @property
    def empty_element(self):
        return self._type(None)


    @property
    def N(self):
        return self.normalize()


    def hist(self, num_bins=20, bin_range=None, adjoin=True, individually=True, **kwargs):
        histmap = ViewMap(dimensions=self.dimensions, title_suffix=self.title_suffix)

        map_range = None if individually else self.range
        bin_range = map_range if bin_range is None else bin_range
        for k, v in self.items():
            histmap[k] = v.hist(num_bins=num_bins, bin_range=bin_range,
                                  individually=individually,
                                  style_prefix='Custom[<' + self.name + '>]_',
                                  adjoin=False,
                                  **kwargs)

        if adjoin and issubclass(self.type, Overlay):
            layout = (self << histmap)
            layout.main_layer = kwargs['index']
            return layout

        return (self << histmap) if adjoin else histmap


    def normalize_elements(self, **kwargs):
        return self.map(lambda x, _: x.normalize(**kwargs))


    def normalize(self, min=0.0, max=1.0):
        data_max = np.max([el.data.max() for el in self.values()])
        data_min = np.min([el.data.min() for el in self.values()])
        norm_factor = data_max-data_min
        return self.map(lambda x, _: x.normalize(min=min, max=max,
                                                 norm_factor=norm_factor))


__all__ = list(set([_k for _k, _v in locals().items()
                    if isinstance(_v, type) and issubclass(_v, Dimensioned)]))

