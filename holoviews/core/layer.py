import itertools
import numpy as np

import param

from .dimension import Dimension
from .holoview import View, HoloMap, find_minmax
from .ndmapping import NdMapping
from .layout import Pane, GridLayout, AdjointLayout
from .options import options, channels


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
        if isinstance(other, HoloMap):
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
        self._xlim = None
        self._ylim = None
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
        else:
            x_vals = self.data[:, 0]
            return (float(min(x_vals)), float(max(x_vals)))

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
        else:
            y_vals = self.data[:, 1]
            return (float(min(y_vals)), float(max(y_vals)))

    @ylim.setter
    def ylim(self, limits):
        if limits is None or (isinstance(limits, tuple) and len(limits) == 2):
            self._ylim = limits
        else:
            raise ValueError('xlim needs to be a length two tuple or None.')

    @property
    def lbrt(self):
        l, r = self.xlim if self.xlim else (None, None)
        b, t = self.ylim if self.ylim else (None, None)
        return l, b, r, t


    @lbrt.setter
    def lbrt(self, lbrt):
        l, b, r, t = lbrt
        self.xlim, self.ylim = (l, r), (b, t)



class Overlay(View):
    """
    An Overlay allows a group of Layers to be overlaid together. Layers can
    be indexed out of an overlay and an overlay is an iterable that iterates
    over the contained layers.

    A SheetOverlay may be used to overlay lines or points over a
    SheetMatrix. In addition, if an overlay consists of three or four
    SheetViews of depth 1, the overlay may be converted to an RGB(A)
    SheetMatrix via the rgb property.
    """

    dimensions = param.List(default=[Dimension('Overlay')], constant=True, doc="""List
      of dimensions the View can be indexed by.""")

    label = param.String(doc="""
      A short label used to indicate what kind of data is contained
      within the view object.

      Overlays should not have their label set directly by the user as
      the label is only for defining custom channel operations.""")

    channels = channels

    _abstract = True

    _deep_indexable = True

    def __init__(self, overlays, **kwargs):
        super(Overlay, self).__init__([], **kwargs)
        self._xlim = None
        self._ylim = None
        self._layer_dimensions = None
        self.set(overlays)


    @property
    def labels(self):
        return [el.label for el in self]


    @property
    def style(self):
        return [el.style for el in self.data]


    @style.setter
    def style(self, styles):
        for layer, style in zip(self, styles):
            layer.style = style


    def add(self, layer):
        """
        Overlay a single layer on top of the existing overlay.
        """
        if not isinstance(layer, Layer): pass
        elif not len(self):
            self._layer_dimensions = layer.dimension_labels
            self.xlim = layer.xlim
            self.ylim = layer.ylim
            self.value = layer.value
            self.label = layer.label
        else:
            self.xlim = layer.xlim if self.xlim is None else find_minmax(self.xlim, layer.xlim)
            self.ylim = layer.ylim if self.xlim is None else find_minmax(self.ylim, layer.ylim)
            if layer.dimension_labels != self._layer_dimensions:
                raise Exception("DataLayers must share common dimensions.")
        if layer.label in [o.label for o in self.data]:
            self.warning('Label %s already defined in Overlay' % layer.label)
        self.data.append(layer)


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


    def __add__(self, obj):
        if not isinstance(obj, GridLayout):
            return GridLayout(initial_items=[self, obj])


    def __mul__(self, other):
        if isinstance(other, HoloMap):
            items = [(k, self * v) for (k, v) in other.items()]
            return other.clone(items=items)
        elif isinstance(other, Overlay):
            overlays = self.data + other.data
        elif isinstance(other, (View)):
            overlays = self.data + [other]
        else:
            raise TypeError('Can only create an overlay of holoviews.')

        return Overlay(overlays)


    def set(self, layers):
        """
        Set a collection of layers to be overlaid with each other.
        """
        self.data = []
        for layer in layers:
            self.add(layer)
        return self


    def hist(self, index=None, adjoin=True, **kwargs):
        valid_ind = isinstance(index, int) and (0 <= index < len(self))
        valid_label = index in [el.label for el in self.data]
        if index is None or not any([valid_ind, valid_label]):
            raise TypeError("Please supply a suitable index for the histogram data")

        hist = self[index].hist(adjoin=False, **kwargs)
        if adjoin:
            layout = self << hist
            layout.main_layer = index
            return layout
        else:
            return hist


    def __getitem__(self, ind):
        if isinstance(ind, str):
            matches = [o for o in self.data if o.label == ind]
            if matches == []: raise KeyError('Key %s not found.' % ind)
            return matches[0]

        if ind is ():
            return self
        elif isinstance(ind, tuple):
            ind, ind2 = (ind[0], ind[1:])
        else:
            return self.data[ind]
        if isinstance(ind, slice):
            return self.__class__([d[ind2] for d in self.data[ind]],
                                  **dict(self.get_param_values()))
        else:
            return self.data[ind][ind2]


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


    def __len__(self):
        return len(self.data)


    def __iter__(self):
        i = 0
        while i < len(self):
            yield self[i]
            i += 1


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
        elif isinstance(other, HoloMap) and len(other) == 1:
            view = other.last
        elif isinstance(other, HoloMap) and len(other) != 1:
            raise Exception("Can only overlay with HoloMap of length 1")
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
            return keys[idx]
        elif any(not isinstance(el, slice) for el in key):
            index_ind = [idx for idx, el in enumerate(key) if not isinstance(el, slice)][0]
            temp_key = [el.start if isinstance(el, slice) else el for el in key]
            snapped_key = self._transform_indices(temp_key)
            key = list(key)
            key[index_ind] = snapped_key[index_ind]
            return tuple(key)
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
                self._type = item.type if isinstance(item, HoloMap) else item.__class__
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
        of __len__ used by Stacks. For the total number of
        elements, count the full set of keys.
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
            if isinstance(v, HoloMap):
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
            if isinstance(v, HoloMap):
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
        for coords, stack in self.items():
            stack_frame = stack.dframe()
            for coord, dim in zip(coords, self.dimension_labels)[::-1]:
                if dim in stack_frame: dim = 'Grid_' + dim
                stack_frame.insert(0, dim.replace(' ','_'), coord)
            dframes.append(stack_frame)
        return pandas.concat(dframes)