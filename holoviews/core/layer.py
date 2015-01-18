"""
Supplies Layer and related classes that allow overlaying of Views,
including Overlay. A Layer is the final extension of View base class
that allows Views to be overlayed on top of each other.

Also supplies ViewMap which is the primary multi-dimensional Map type
for indexing, slicing and animating collections of Views.
"""

from numbers import Number
from collections import OrderedDict, defaultdict
import itertools
import numpy as np

import param

from .dimension import Dimension, Dimensioned
from .ndmapping import NdMapping
from .layout import Pane, GridLayout, AdjointLayout, ViewTree
from .options import options, channels
from .util import find_minmax
from .view import View, Map


class Layer(Pane):
    """
    Layer is the baseclass for all 2D View types, with an x- and
    y-dimension. Subclasses should define the data storage in the
    constructor, as well as methods and properties, which define how
    the data maps onto the x- and y- and value dimensions.
    """

    value = param.String(default='Layer')

    def __mul__(self, other):
        if isinstance(other, ViewMap):
            items = [(k, self * v) for (k, v) in other.items()]
            return other.clone(items)

        self_layers = self.data.values() if isinstance(self, Overlay) else [self]
        other_layers = other.data.values() if isinstance(other, Overlay) else [other]
        combined_layers = self_layers + other_layers

        return Overlay(combined_layers)

    def hist(self, num_bins=20, bin_range=None, adjoin=True, individually=True, **kwargs):
        from ..operation import histogram
        return histogram(self, num_bins=num_bins, bin_range=bin_range, adjoin=adjoin,
                         individually=individually, **kwargs)

    def table(self):
        from ..view import Table
        index_values = [self.dimension_values(dim) for dim in self._cached['index_names']]
        values = [self.dimension_values(dim) for dim in [d.name for d in self.value_dimensions]]
        return Table(zip(zip(*index_values), zip(*values)), index_dimensions=self.index_dimensions,
                     label=self.label, value=self.value, value_dimensions=self.value_dimensions)


    ########################
    # Subclassable methods #
    ########################


    def __init__(self, data, **params):
        lbrt = params.pop('lbrt', None)
        self._xlim = (lbrt[0], lbrt[2]) if lbrt else None
        self._ylim = (lbrt[1], lbrt[3]) if lbrt else None
        super(Layer, self).__init__(data, **params)

    @property
    def xlabel(self):
        return self.get_dimension(0).pprint_label

    @property
    def ylabel(self):
        return self.get_dimension(1).pprint_label

    @property
    def xlim(self):
        if self._xlim:
            return self._xlim
        else:
            return self.range(0)

    @xlim.setter
    def xlim(self, limits):
        if limits is None or (isinstance(limits, tuple) and len(limits) == 2):
            self._xlim = limits
        else:
            raise ValueError('xlim needs to be a length two tuple or None.')

    @property
    def ylim(self):
        if self._ylim:
            return self._ylim
        else:
            return self.range(1)

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

    index_dimensions = param.List(default=[Dimension('Layer')], constant=True, doc="""List
      of dimensions the Overlay can be indexed by.""")

    label = param.String(default='', doc="""
      A label used to indicate what kind of data is contained
      within the Overlay. This overrides the auto-generated title
      made up of the individual Views.""")

    value = param.String(default='Overlay')

    channels = channels
    _deep_indexable = True

    def __init__(self, overlays, **params):
        self._xlim = None
        self._ylim = None
        data = self._process_layers(overlays)
        Pane.__init__(self, data, **params)
        NdMapping.__init__(self, data, **params)


    def dimension_values(self, *args, **kwargs):
        NdMapping.dimension_values(self, *args, **kwargs)


    def _process_layers(self, layers):
        """
        Set a collection of layers to be overlaid with each other.
        """
        if isinstance(layers, (ViewMap)):
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


    @property
    def legend(self):
        if self._cached['index_names'] == ['Layer']:
            labels = self.labels
            if len(set(labels)) == len(labels):
                return labels
            else:
                return None
        else:
            labels = []
            for key in self.data.keys():
                labels.append(','.join([dim.pprint_value(k) for dim, k in
                                        zip(self.index_dimensions, key)]))
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
        layer_dimensions = [d.name for d in layer.index_dimensions]
        if len(self):
            if layer_dimensions != self._layer_dimensions:
                raise Exception("Layers must share common dimensions.")
        else:
            self._layer_dimensions = layer_dimensions
            self.value = layer.value
            self.label = layer.label


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


    def __mul__(self, other):
        if isinstance(other, ViewMap):
            items = [(k, self * v) for (k, v) in other.items()]
            return other.clone(items)
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

    index_dimensions = param.List(default=[Dimension(name="X"), Dimension(name="Y")], bounds=(2,2))

    label = param.String(constant=True, doc="""
      A short label used to indicate what kind of data is contained
      within the Grid.""")

    title = param.String(default="{label} {value}", doc="""
      The title formatting string for the Grid object""")

    value = param.String(default='Grid')

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
        """
        Transforms indices by snapping to the closest value if
        values are numeric, otherwise applies no transformation.
        """
        if all(not isinstance(el, slice) for el in key):
            dim_inds = [self.get_dimension_index(l) for l in self._cached['index_names']
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


    def relabel(self, label):
        """
        Recreates the Grid with a supplied label.
        """
        return self.clone(self.data, label=label)


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
            if isinstance(v, ViewMap):
                keys_list.append(list(v.data.keys()))
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
        for dim in self._cached['index_names']:
            grid_dimensions.append(self.range(dim))
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
            for coord, dim in zip(coords, self._cached['index_names'])[::-1]:
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

    value = param.String(default='ViewMap')

    data_type = (View, Map)

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
            split_map = self.split_dimensions(dimensions, Overlay)
            new_map = self.clone(index_dimensions=split_map.index_dimensions)

        for outer, vmap in split_map.items():
            new_map[outer] = Overlay(vmap, index_dimensions=vmap.index_dimensions)

        if self.ndims == 1:
            return list(new_map.values())[0]
        else:
            return new_map


    def grid(self, dimensions, layout=False, set_title=True):
        """
        Grid takes a list of one or two dimensions, and lays out the containing
        Views along these axes in a Grid.
        """
        if len(dimensions) > 2:
            raise ValueError('At most two dimensions can be laid out in a grid.')

        if len(dimensions) == self.ndims:
            split_map = self
        elif all(d in self._cached['index_names'] for d in dimensions):
            split_dims = [d for d in self._cached['index_names'] if d not in dimensions]
            split_map = self.split_dimensions(split_dims)
            split_map = split_map.reindex(dimensions)
        else:
            raise ValueError('ViewMap does not have supplied dimensions.')

        if layout:
            if set_title:
                for keys, vmap in split_map.data.items():
                    dim_labels = split_map.pprint_dimkey(keys)
                    if not isinstance(vmap, ViewMap): vmap = [vmap]
                    for vm in vmap:
                        if dim_labels and dim_labels not in vm.title:
                            vm.title = '\n'.join([vm.title, dim_labels])
            return GridLayout(split_map)
        else:
            return Grid(split_map, index_dimensions=split_map.index_dimensions)


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
            self_set = set(self._cached['index_names'])
            other_set = set(other._cached['index_names'])

            # Determine which is the subset, to generate list of keys and
            # dimension labels for the new view
            self_in_other = self_set.issubset(other_set)
            other_in_self = other_set.issubset(self_set)
            dimensions = self.index_dimensions
            if self_in_other and other_in_self: # superset of each other
                super_keys = sorted(set(self.dimension_keys() + other.dimension_keys()))
            elif self_in_other: # self is superset
                dimensions = other.index_dimensions
                super_keys = other.dimension_keys()
            elif other_in_self: # self is superset
                super_keys = self.dimension_keys()
            else: # neither is superset
                raise Exception('One set of keys needs to be a strict subset of the other.')

            items = []
            for dim_keys in super_keys:
                # Generate keys for both subset and superset and sort them by the dimension index.
                self_key = tuple(k for p, k in sorted(
                    [(self.get_dimension_index(dim), v) for dim, v in dim_keys
                     if dim in self._cached['index_names']]))
                other_key = tuple(k for p, k in sorted(
                    [(other.get_dimension_index(dim), v) for dim, v in dim_keys
                     if dim in other._cached['index_names']]))
                new_key = self_key if other_in_self else other_key
                # Append SheetOverlay of combined items
                if (self_key in self) and (other_key in other):
                    items.append((new_key, self[self_key] * other[other_key]))
                elif self_key in self:
                    items.append((new_key, self[self_key] * other.empty_element))
                else:
                    items.append((new_key, self.empty_element * other[other_key]))
            return self.clone(items, index_dimensions=dimensions)
        elif isinstance(other, self.data_type):
            items = [(k, v * other) for (k, v) in self.items()]
            return self.clone(items)
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
        for key, view in self.data.items():
            view_frame = view.dframe()
            for val, dim in reversed(zip(key, self._cached['index_names'])):
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
        return ViewTree.from_view(self) + ViewTree.from_view(obj)


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
        histmap = ViewMap(index_dimensions=self.index_dimensions, title_suffix=self.title_suffix)

        map_range = None if individually else self.range
        bin_range = map_range if bin_range is None else bin_range
        style_prefix = 'Custom[<' + self.name + '>]_'
        for k, v in self.items():
            histmap[k] = v.hist(adjoin=False, bin_range=bin_range,
                                individually=individually, num_bins=num_bins,
                                style_prefix=style_prefix, **kwargs)

        if adjoin and issubclass(self.type, Overlay):
            layout = (self << histmap)
            layout.main_layer = kwargs['index']
            return layout

        return (self << histmap) if adjoin else histmap


    def table(self):
        from ..view import Table
        keys = zip(*[self.dimension_values(dm) for dm in self._cached['index_names']])
        vals = self.dimension_values(self.value.name)
        return Table(zip(keys, vals), **dict(self.get_param_values()))


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

