from collections import defaultdict
import numpy as np

import param

from .dimension import ViewableElement
from .layout import Composable, LayoutTree, AdjointLayout, NdLayout
from .layer import Overlayable, NdOverlay, Overlay, CompositeOverlay, AxisLayout
from .ndmapping import UniformNdMapping
from .options import options
from .util import find_minmax


class Element(ViewableElement, Composable, Overlayable):
    """
    Element is the baseclass for all ViewableElement types, with an x- and
    y-dimension. Subclasses should define the data storage in the
    constructor, as well as methods and properties, which define how
    the data maps onto the x- and y- and value dimensions.
    """

    value = param.String(default='Element')

    options = options

    def hist(self, num_bins=20, bin_range=None, adjoin=True, individually=True, **kwargs):
        from ..operation import histogram
        return histogram(self, num_bins=num_bins, bin_range=bin_range, adjoin=adjoin,
                         individually=individually, **kwargs)


    ########################
    # Subclassable methods #
    ########################

    def __init__(self, data, **params):
        super(Element, self).__init__(data, **params)


    def closest(self, coords):
        """
        Class method that returns the exact keys for a given list of
        coordinates. The supplied bounds defines the extent within
        which the samples are drawn and the optional shape argument is
        the shape of the numpy array (typically the shape of the .data
        attribute) when applicable.
        """
        return coords


    def sample(self, **samples):
        """
        Base class signature to demonstrate API for sampling Views.
        To sample a ViewableElement kwargs, where the keyword matches a Dimension
        in the ViewableElement and the value matches a corresponding entry in the
        data.
        """
        raise NotImplementedError


    def reduce(self, label_prefix='', **reduce_map):
        """
        Base class signature to demonstrate API for reducing Views,
        using some reduce function, e.g. np.mean. Signature is the
        same as sample, however a label_prefix may be provided to
        describe the reduction operation.
        """
        raise NotImplementedError


    @property
    def style(self):
        """
        The name of the style that may be used to control display of
        this element. If a style name is not set and but a label is
        assigned, then the closest existing style name is returned.
        """
        if self._style:
            return self._style

        class_name = self.__class__.__name__
        if self.label:
            style_str = '_'.join([self.label, class_name])
            matches = self.options.fuzzy_match_keys(style_str)
            return matches[0] if matches else class_name
        else:
            return class_name


    @style.setter
    def style(self, val):
        self._style = val


    def table(self, **kwargs):
        """
        This method transforms any ViewableElement type into a Table
        as long as it implements a dimension_values method.
        """
        from ..element import Table
        keys = zip(*[self.dimension_values(dim.name)
                 for dim in self.key_dimensions])
        values = zip(*[self.dimension_values(dim.name)
                       for dim in self.value_dimensions])
        params = dict(key_dimensions=self.key_dimensions,
                      value_dimensions=self.value_dimensions,
                      label=self.label, value=self.value, **kwargs)
        return Table(zip(keys, values), **params)


    def dframe(self):
        raise NotImplementedError


    def __getstate__(self):
        """
        When pickling, make sure to save the relevant style and
        plotting options as well.
        """
        obj_dict = self.__dict__.copy()
        obj_dict['style_objects'] = {}
        for match in self.options.fuzzy_match_keys(self.style):
            obj_dict['style_objects'][match] = self.options[match]
        return obj_dict


    def __setstate__(self, d):
        """
        When unpickled, restore the saved style and plotting options
        to ViewableElement.options.
        """
        for name, match in d.pop('style_objects').items():
            for style in match:
                self.options[name] = style
        self.__dict__.update(d)


    def __repr__(self):
        params = ', '.join('%s=%r' % (k,v) for (k,v) in self.get_param_values())
        return "%s(%r, %s)" % (self.__class__.__name__, self.data, params)



class Element2D(Element):

    def __init__(self, data, extent=None, **params):
        self._xlim = (extent[0], extent[2]) if extent else None
        self._ylim = (extent[1], extent[3]) if extent else None
        super(Element2D, self).__init__(data, **params)

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
    def extent(self):
        """"
        For Element2D the extent is the 4-tuple (left, bottom, right, top).
        """
        l, r = self.xlim if self.xlim else (np.NaN, np.NaN)
        b, t = self.ylim if self.ylim else (np.NaN, np.NaN)
        return l, b, r, t


    @extent.setter
    def extent(self, extent):
        l, b, r, t = extent
        self.xlim, self.ylim = (l, r), (b, t)


class Element3D(Element2D):


    def __init__(self, data, extent=None, **params):
        self._xlim = (extent[0], extent[3]) if extent else None
        self._ylim = (extent[1], extent[4]) if extent else None
        self._zlim = (extent[2], extent[5]) if extent else None
        super(Element2D, super).__init__(data, **params)

    @property
    def extent(self):
        """"
        For Element3D the extent is the 6-tuple (left, bottom, -z, right, top, +z) using
        a right-handed Cartesian coordinate-system.
        """
        l, r = self.xlim if self.xlim else (np.NaN, np.NaN)
        b, t = self.ylim if self.ylim else (np.NaN, np.NaN)
        zminus, zplus = self.zlim if self.zlim else (np.NaN, np.NaN)
        return l, b, zminus, r, t, zplus

    @extent.setter
    def extent(self, extent):
        l, b, zminus, r, t, zplus = extent
        self.xlim, self.ylim, self.zlim = (l, r), (b, t), (zminus, zplus)

    @property
    def zlim(self):
        if self._zlim:
            return self._zlim
        else:
            return self.range(2)

    @zlim.setter
    def zlim(self, limits):
        if limits is None or (isinstance(limits, tuple) and len(limits) == 2):
            self._zlim = limits
        else:
            raise ValueError('zlim needs to be a length two tuple or None.')

    @property
    def zlabel(self):
        return self.get_dimension(2).pprint_label



class HoloMap(UniformNdMapping):
    """
    A HoloMap can hold any number of DataLayers indexed by a list of
    dimension values. It also has a number of properties, which can find
    the x- and y-dimension limits and labels.
    """

    value = param.String(default='HoloMap')

    data_type = ViewableElement

    @property
    def layer_types(self):
        """
        The type of layers stored in the HoloMap.
        """
        if self.type == NdOverlay:
            return self.last.layer_types
        else:
            return (self.type)


    @property
    def xlabel(self):
        return self.last.xlabel


    @property
    def ylabel(self):
        return self.last.ylabel


    @property
    def xlim(self):
        xlim = self.last.xlim
        for data in self.values():
            xlim = find_minmax(xlim, data.xlim) if data.xlim and xlim else xlim
        return xlim


    @property
    def ylim(self):
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
        Splits the UniformNdMapping along a specified number of dimensions and
        overlays items in the split out Maps.
        """
        if self.ndims == 1:
            split_map = dict(default=self)
            new_map = dict()
        else:
            split_map = self.split_dimensions(dimensions, NdOverlay)
            new_map = self.clone(key_dimensions=split_map.key_dimensions)

        for outer, vmap in split_map.items():
            new_map[outer] = NdOverlay(vmap, key_dimensions=vmap.key_dimensions)

        if self.ndims == 1:
            return list(new_map.values())[0]
        else:
            return new_map


    def grid(self, dimensions, layout=False, set_title=True):
        """
        AxisLayout takes a list of one or two dimensions, and lays out the containing
        Views along these axes in a AxisLayout.
        """
        if len(dimensions) > 2:
            raise ValueError('At most two dimensions can be laid out in a grid.')

        if len(dimensions) == self.ndims:
            split_map = self
        elif all(d in self._cached_index_names for d in dimensions):
            split_dims = [d for d in self._cached_index_names if d not in dimensions]
            split_map = self.split_dimensions(split_dims)
            split_map = split_map.reindex(dimensions)
        else:
            raise ValueError('HoloMap does not have supplied dimensions.')

        if layout:
            if set_title:
                for keys, vmap in split_map.data.items():
                    dim_labels = split_map.pprint_dimkey(keys)
                    if not isinstance(vmap, HoloMap): vmap = [vmap]
                    for vm in vmap:
                        if dim_labels and dim_labels not in vm.title:
                            vm.title = '\n'.join([vm.title, dim_labels])
            return NdLayout(split_map)
        else:
            return AxisLayout(split_map, key_dimensions=split_map.key_dimensions)


    def split_overlays(self):
        """
        Given a UniformNdMapping of Overlays of N layers, split out the layers into
        N separate Maps.
        """
        if not issubclass(self.type, CompositeOverlay):
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
        keys. If the UniformNdMapping is mulled with a simple
        ViewableElement each element in the UniformNdMapping is
        overlaid with the ViewableElement. If the element the
        UniformNdMapping is mulled with is another UniformNdMapping it
        will try to match up the dimensions, making sure that items
        with completely different dimensions aren't overlaid.
        """
        if isinstance(other, self.__class__):
            self_set = set(self._cached_index_names)
            other_set = set(other._cached_index_names)

            # Determine which is the subset, to generate list of keys and
            # dimension labels for the new view
            self_in_other = self_set.issubset(other_set)
            other_in_self = other_set.issubset(self_set)
            dimensions = self.key_dimensions
            if self_in_other and other_in_self: # superset of each other
                super_keys = sorted(set(self.dimension_keys() + other.dimension_keys()))
            elif self_in_other: # self is superset
                dimensions = other.key_dimensions
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
                     if dim in self._cached_index_names]))
                other_key = tuple(k for p, k in sorted(
                    [(other.get_dimension_index(dim), v) for dim, v in dim_keys
                     if dim in other._cached_index_names]))
                new_key = self_key if other_in_self else other_key
                # Append SheetOverlay of combined items
                if (self_key in self) and (other_key in other):
                    items.append((new_key, self[self_key] * other[other_key]))
                elif self_key in self:
                    items.append((new_key, self[self_key] * other.empty_element))
                else:
                    items.append((new_key, self.empty_element * other[other_key]))
            return self.clone(items, key_dimensions=dimensions)
        elif isinstance(other, self.data_type):
            items = [(k, v * other) for (k, v) in self.items()]
            return self.clone(items)
        else:
            raise Exception("Can only overlay with {data} or {vmap}.".format(
                data=self.data_type, vmap=self.__class__.__name__))


    def dframe(self):
        """
        Gets a dframe for each ViewableElement in the
        UniformNdMapping, appends the dimensions of the
        UniformNdMapping as series and concatenates the dframes.
        """
        import pandas
        dframes = []
        for key, view in self.data.items():
            view_frame = view.dframe()
            for val, dim in reversed(zip(key, self._cached_index_names)):
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
        return LayoutTree.from_view(self) + LayoutTree.from_view(obj)


    def __lshift__(self, other):
        if isinstance(other, (ViewableElement, UniformNdMapping)):
            return AdjointLayout([self, other])
        elif isinstance(other, AdjointLayout):
            return AdjointLayout(other.data+[self])
        else:
            raise TypeError('Cannot append {0} to a AdjointLayout'.format(type(other).__name__))


    def sample(self, samples, bounds=None, **sample_values):
        """
        Sample each Element in the UniformNdMapping by passing either a list of
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
        Reduce each Matrix in the UniformNdMapping using a function supplied
        via the kwargs, where the keyword has to match a particular
        dimension in the ViewableElement.
        """
        reduced_items = [(k, v.reduce(label_prefix=label_prefix, **reduce_map))
                         for k, v in self.items()]
        return self.clone(reduced_items)


    def collate(self, collate_dim):
        """
        Collate splits out the specified dimension and joins the
        samples in each of the split out Maps into Curves. If there
        are multiple entries in the ItemTable it will lay them out
        into a AxisLayout.
        """
        from ..operation import table_collate
        return table_collate(self, collation_dim=collate_dim)


    def map(self, map_fn, **kwargs):
        """
        UniformNdMapping a function across the HoloMap, using the bounds of first
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
        histmap = HoloMap(key_dimensions=self.key_dimensions, title_suffix=self.title_suffix)

        map_range = None if individually else self.range
        bin_range = map_range if bin_range is None else bin_range
        style_prefix = 'Custom[<' + self.name + '>]_'
        for k, v in self.items():
            histmap[k] = v.hist(adjoin=False, bin_range=bin_range,
                                individually=individually, num_bins=num_bins,
                                style_prefix=style_prefix, **kwargs)

        if adjoin and issubclass(self.type, (NdOverlay, Overlay)):
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

