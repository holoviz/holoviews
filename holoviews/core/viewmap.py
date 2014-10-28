"""
Views objects used to hold data and provide indexing into different coordinate
systems.
"""
from collections import defaultdict

import numpy as np

from .holoview import View, HoloMap, find_minmax
from .ndmapping import NdMapping
from .layer import Layer, Overlay, Grid
from .layout import AdjointLayout, GridLayout


class ViewMap(HoloMap):
    """
    A ViewMap can hold any number of DataLayers indexed by a list of
    dimension values. It also has a number of properties, which can find
    the x- and y-dimension limits and labels.
    """

    data_type = (View, HoloMap, Overlay)

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
        if self.xlim is None: return None, None, None, None
        l, r = self.xlim
        b, t = self.ylim
        return float(l), float(b), float(r), float(t)



    def overlay_dimensions(self, dimensions):
        """
        Splits the HoloMap along a specified number of dimensions and overlays
        items in the split out Stacks.
        """
        if self.ndims == 1:
            split_stack = dict(default=self)
            new_stack = dict()
        else:
            split_stack = self.split_dimensions(dimensions)
            new_stack = self.clone(dimensions=split_stack.dimensions)

        for outer, stack in split_stack.items():
            key, overlay = stack.items()[0]
            overlay.constant_dimensions = stack.dimensions
            overlay.constant_values = key
            for inner, v in list(stack.items())[1:]:
                v.constant_dimensions = stack.dimensions
                v.constant_values = inner
                overlay = overlay * v
            new_stack[outer] = overlay

        if self.ndims == 1:
            return list(new_stack.values())[0]
        else:
            return new_stack


    def grid(self, dimensions, layout=False, constant_dims=True):
        """
        Grid takes a list of one or two dimensions, and lays out the containing
        Views along these axes in a Grid.
        """
        if len(dimensions) > 2:
            raise ValueError('At most two dimensions can be laid out in a grid.')

        if len(dimensions) == self.ndims:
            split_stack = self
        elif all(d in self.dimension_labels for d in dimensions):
            split_dims = [d for d in self.dimension_labels if d not in dimensions]
            split_stack = self.split_dimensions(split_dims)
            split_stack = split_stack.reindex(dimensions)
        else:
            raise ValueError('HoloMap does not have supplied dimensions.')

        if layout:
            for keys, stack in split_stack._data.items():
                if constant_dims:
                    stack.constant_dimensions = split_stack.dimensions
                    stack.constant_values = keys
            return GridLayout(split_stack)
        else:
            return Grid(split_stack, dimensions=split_stack.dimensions)

    def split_overlays(self):
        """
        Given a HoloMap of Overlays of N layers, split out the layers
        into N separate Stacks.
        """
        if self.type is not Overlay:
            return self.clone(self.items())

        stacks = []
        item_stacks = defaultdict(list)
        for k, overlay in self.items():
            for i, el in enumerate(overlay):
                item_stacks[i].append((k, el))

        for k in sorted(item_stacks.keys()):
            stacks.append(self.clone(item_stacks[k]))
        return stacks


    def __mul__(self, other):
        """
        The mul (*) operator implements overlaying of different Views.
        This method tries to intelligently overlay Stacks with differing
        keys. If the HoloMap is mulled with a simple View each element in
        the HoloMap is overlaid with the View. If the element the HoloMap is
        mulled with is another HoloMap it will try to match up the dimensions,
        making sure that items with completely different dimensions aren't
        overlaid.
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
            raise Exception("Can only overlay with {data} or {stack}.".format(
                data=self.data_type, stack=self.__class__.__name__))


    def dframe(self):
        """
        Gets a dframe for each View in the HoloMap, appends the dimensions
        of the HoloMap as series and concatenates the dframes.
        """
        import pandas
        dframes = []
        for key, view in self.items():
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
        if isinstance(other, (View, Overlay, NdMapping)):
            return AdjointLayout([self, other])
        elif isinstance(other, AdjointLayout):
            return AdjointLayout(other.data+[self])
        else:
            raise TypeError('Cannot append {0} to a AdjointLayout'.format(type(other).__name__))


    def sample(self, samples=[], **sample_values):
        """
        Sample each Layer in the HoloMap by passing either a list
        of samples or request a single sample using dimension-value
        pairs.
        """
        sampled_items = [(k, view.sample(samples, **sample_values))
                         for k, view in self.items()]
        return self.clone(sampled_items)


    def reduce(self, label_prefix='', **reduce_map):
        """
        Reduce each SheetMatrix in the HoloMap using a function supplied
        via the kwargs, where the keyword has to match a particular
        dimension in the View.
        """
        reduced_items = [(k, v.reduce(label_prefix=label_prefix, **reduce_map))
                         for k, v in self.items()]
        return self.clone(reduced_items)


    def collate(self, collate_dim):
        """
        Collate splits out the specified dimension and joins the samples
        in each of the split out Stacks into Curves. If there are multiple
        entries in the ItemTable it will lay them out into a Grid.
        """
        from ..operations import table_collate
        return table_collate(self, collation_dim=collate_dim)


    def map(self, map_fn, **kwargs):
        """
        Map a function across the stack, using the bounds of first
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
        histstack = ViewMap(dimensions=self.dimensions, title_suffix=self.title_suffix)

        stack_range = None if individually else self.range
        bin_range = stack_range if bin_range is None else bin_range
        for k, v in self.items():
            histstack[k] = v.hist(num_bins=num_bins, bin_range=bin_range,
                                  individually=individually,
                                  style_prefix='Custom[<' + self.name + '>]_',
                                  adjoin=False,
                                  **kwargs)

        if adjoin and issubclass(self.type, Overlay):
            layout = (self << histstack)
            layout.main_layer = kwargs['index']
            return layout

        return (self << histstack) if adjoin else histstack


    def normalize_elements(self, **kwargs):
        return self.map(lambda x, _: x.normalize(**kwargs))


    def normalize(self, min=0.0, max=1.0):
        data_max = np.max([el.data.max() for el in self.values()])
        data_min = np.min([el.data.min() for el in self.values()])
        norm_factor = data_max-data_min
        return self.map(lambda x, _: x.normalize(min=min, max=max,
                                                 norm_factor=norm_factor))