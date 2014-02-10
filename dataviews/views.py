"""
Views objects used to hold data and provide indexing into different coordinate
systems.
"""

__version__='$Revision$'

import math
from collections import defaultdict
import param

from ndmapping import NdMapping, AttrDict, map_type


class View(param.Parameterized):
    """
    A view is a data structure for holding data, which may be plotted using
    matplotlib. Views have an associated title, style and metadata and can
    be composed together into a GridLayout using the plus operator.
    """

    title = param.String(default=None, allow_None=True, doc="""
       A short description of the layer that may be used as a title.""")

    style = param.Dict(default=AttrDict(), doc="""
        Optional keywords for specifying the display style.""")

    metadata = param.Dict(default=AttrDict(), doc="""
        Additional information to be associated with the Layer.""")


    def __init__(self, data, **kwargs):
        self.data = data
        super(View, self).__init__(**kwargs)


    def __add__(self, obj):
        if not isinstance(obj, GridLayout):
            return GridLayout(initial_items=[[self, obj]])



class Overlay(View):
    """
    An Overlay allows a group of Layers to be overlaid together. Layers can
    be indexed out of an overlay and an overlay is an iterable that iterates
    over the contained layers.
    """

    _abstract = True

    _deep_indexable = True

    def __init__(self, overlays, **kwargs):
        super(Overlay, self).__init__([], **kwargs)
        self.set(overlays)


    def add(self, layer):
        """
        Overlay a single layer on top of the existing overlay.
        """
        self.data.append(layer)


    def set(self, layers):
        """
        Set a collection of layers to be overlaid with each other.
        """
        self.data = []
        for layer in layers:
            self.add(layer)
        return self


    def __getitem__(self, ind):
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


    def __len__(self):
        return len(self.data)


    def __iter__(self):
        i = 0
        while i < len(self):
            yield self[i]
            i += 1



class Stack(NdMapping):
    """
    A Stack is a stack of Views over some dimensions. The
    dimension may be a spatial dimension (i.e., a ZStack), time
    (specifying a frame sequence) or any other dimensions.
    """

    title = param.String(default=None, doc="""
       A short description of the stack that may be used as a title
       (e.g. the title of an animation) but may also accept a
       formatting string to generate a unique title per layer. For
       instance the format string '{label0} = {value0}' will generate
       a title using the first dimension label and corresponding key
       value. Numbering is by dimension position and extends across
       all available dimensions e.g. {label1}, {value2} and so on.""")

    data_type = View

    overlay_type = Overlay

    _type = None

    @property
    def type(self):
        """
        The type of elements stored in the stack.
        """
        if self._type is None:
            self._type = None if len(self) == 0 else self.top.__class__
        return self._type


    @property
    def empty_element(self):
        return self._type(None)


    def _item_check(self, dim_vals, data):
        if self.type is not None and (type(data) != self.type):
            raise AssertionError("%s must only contain one type of View." %
                                 self.__class__.__name__)
        super(Stack, self)._item_check(dim_vals, data)


    def split(self):
        """
        Given a SheetStack of SheetOverlays of N layers, split out the
        layers into N separate SheetStacks.
        """
        if self.type is not self.overlay_type:
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
        if isinstance(other, self.__class__):
            self_set = set(self.dimension_labels)
            other_set = set(other.dimension_labels)

            # Determine which is the subset, to generate list of keys and
            # dimension labels for the new view
            self_in_other = self_set.issubset(other_set)
            other_in_self = other_set.issubset(self_set)
            dim_labels = self.dimension_labels
            if self_in_other and other_in_self: # superset of each other
                super_keys = sorted(set(self.dimension_keys() + other.dimension_keys()))
            elif self_in_other: # self is superset
                dim_labels = other.dimension_labels
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
            return self.clone(items=items, dimension_labels=dim_labels)
        elif isinstance(other, self.data_type):
            items = [(k, v * other) for (k, v) in self.items()]
            return self.clone(items=items)
        else:
            raise Exception("Can only overlay with {data} or {stack}.".format(
                data=self.data_type, stack=self.__class__.__name__))


    def __add__(self, obj):
        if not isinstance(obj, GridLayout):
            return GridLayout(initial_items=[[self, obj]])



class GridLayout(NdMapping):

    dim_info = param.Dict(default=dict(Row={'type': int}, Column={'type':int}),
                          constant=True)

    dimension_labels = param.List(default=['Row', 'Column'], constant=True)

    def __init__(self, initial_items=[], **kwargs):
        self._max_cols = 4
        initial_items = [[]] if initial_items == [] else initial_items
        if any(isinstance(el, list) for el in initial_items):
            initial_items = self._grid_to_items(initial_items)
        super(GridLayout, self).__init__(initial_items=initial_items, **kwargs)


    @property
    def shape(self):
        rows, cols = zip(*self.keys())
        return max(rows)+1, max(cols)+1


    @property
    def coords(self):
        """
        Compute the list of (row,column,view) elements from the
        current set of items (i.e. tuples of form ((row, column), view))
        """
        if self.keys() == []:  return []
        return [(r, c, v) for ((r, c), v) in zip(self.keys(), self.values())]


    @property
    def max_cols(self):
        return self._max_cols


    @max_cols.setter
    def max_cols(self, n):
        self._max_cols = n
        self.update({}, n)


    def cols(self, n):
        self.update({}, n)
        return self


    def _grid_to_items(self, grid):
        """
        Given a grid (i.e. a list of lists), compute the list of
        items.
        """
        items = []  # Flatten this method to single list comprehension.
        for rind, row in enumerate(grid):
            for cind, view in enumerate(row):
                items.append(((rind, cind), view))
        return items


    def update(self, other, cols=None):
        """
        Given a mapping or iterable of additional views, extend the
        grid in scanline order, obeying max_cols (if applicable).
        """
        values = other if isinstance(other, list) else other.values()
        grid = [[]] if self.coords == [] else self._grid(self.coords)
        new_grid = grid[:-1] + ([grid[-1]+ values])
        cols = self.max_cols if cols is None else cols
        reshaped_grid = self._reshape_grid(new_grid, cols)
        self._data = map_type(self._grid_to_items(reshaped_grid))


    def __call__(self, cols=None):
        """
        Recompute the grid layout of the views based on precedence and
        row_precendence value metadata. Formats the grid to a maximum
        of cols columns if specified.
        """
        # Plots are sorted first by precedence, then grouped by row_precedence
        values = sorted(self.values(),
                        key=lambda x: x.metadata.get('precedence', 0.5))
        precedences = sorted(
            set(v.metadata.get('row_precedence', 0.5) for v in values))

        coords=[]
        # Can use collections.Counter in Python >= 2.7
        column_counter = dict((i, 0) for i, _ in enumerate(precedences))
        for view in values:
            # Find the row number based on the row_precedences
            row = precedences.index(view.metadata.get('row_precedence', 0.5))
            # Look up the current column position of the row
            col = column_counter[row]
            # The next view on this row will have to be in the next column
            column_counter[row] += 1
            coords.append((row, col, view))

        grid = self._reshape_grid(self._grid(coords), cols)
        self._data = map_type(self._grid_to_items(grid))
        return self


    def _grid(self, coords):
        """
        From a list of coordinates of form [<(row, col, view)>] build
        a corresponding list of lists grid.
        """
        rows = max(r for (r, _, _) in coords) + 1 if coords != [] else 0
        unpadded_grid = [[p for (r, _, p) in coords if r == row] for row in
                         range(rows)]
        return unpadded_grid


    def _reshape_grid(self, grid, cols):
        """
        Given a grid (i.e. a list of lists) , reformat it to a layout
        with a maximum of cols columns (if not None).
        """
        if cols is None: return grid
        flattened = [view for row in grid for view in row if (view is not None)]
        row_num = int(math.ceil(len(flattened) / float(cols)))

        reshaped_grid = []
        for rind in range(row_num):
            new_row = flattened[rind*cols:cols*(rind+1)]
            reshaped_grid.append(new_row)

        return reshaped_grid


    def __add__(self, other):
        new_values = other.values() if isinstance(other, GridLayout) else [other]
        self.update(new_values)
        return self


    def __len__(self):
        return max([len(v) for v in self.values() if isinstance(v, Stack)]+[1])


__all__ = list(set([_k for _k,_v in locals().items() if isinstance(_v,type) and
                    (issubclass(_v, NdMapping) or issubclass(_v, View))]))