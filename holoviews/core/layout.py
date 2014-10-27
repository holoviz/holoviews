import math
import itertools
from collections import OrderedDict

import numpy as np

import param

from .dimension import Dimension, Dimensioned
from .holoview import View, HoloMap, find_minmax
from .ndmapping import NdMapping
from .options import options


class GridLayout(NdMapping):
    """
    A GridLayout is an NdMapping, which can contain any View or HoloMap type.
    It is used to group different View or HoloMap elements into a grid for
    display. Just like all other NdMappings it can be sliced and indexed
    allowing selection of subregions of the grid.
    """

    dimensions = param.List(default=[Dimension('Row', type=int),
                                     Dimension('Column', type=int)], constant=True)

    def __init__(self, initial_items=[], **kwargs):
        self._max_cols = 4
        self._style = None
        if all(isinstance(el, (View, NdMapping, Layout)) for el in initial_items):
            initial_items = self._grid_to_items([initial_items])
        super(GridLayout, self).__init__(initial_items=initial_items, **kwargs)


    @property
    def shape(self):
        rows, cols = list(zip(*list(self.keys())))
        return max(rows)+1, max(cols)+1


    @property
    def coords(self):
        """
        Compute the list of (row,column,view) elements from the
        current set of items (i.e. tuples of form ((row, column), view))
        """
        if list(self.keys()) == []:  return []
        return [(r, c, v) for ((r, c), v) in zip(list(self.keys()), list(self.values()))]


    @property
    def max_cols(self):
        return self._max_cols


    @max_cols.setter
    def max_cols(self, n):
        self._max_cols = n
        self.reorder({}, n)


    def cols(self, n):
        self.reorder({}, n)
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


    def reorder(self, other, cols=None):
        """
        Given a mapping or iterable of additional views, extend the
        grid in scanline order, obeying max_cols (if applicable).
        """
        values = other if isinstance(other, list) else list(other.values())
        grid = [[]] if self.coords == [] else self._grid(self.coords)
        new_grid = grid[:-1] + ([grid[-1]+ values])
        cols = self.max_cols if cols is None else cols
        reshaped_grid = self._reshape_grid(new_grid, cols)
        self._data = OrderedDict(self._grid_to_items(reshaped_grid))


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
        new_values = list(other.values()) if isinstance(other, GridLayout) else [other]
        return self.clone(list(self.values())+new_values)


    @property
    def last(self):
        """
        Returns another GridLayout constituted of the last views of the
        individual elements (if they are stacks).
        """
        last_items = []
        for (k, v) in self.items():
            if isinstance(v, NdMapping):
                item = (k, v.clone((v.last_key, v.last)))
            elif isinstance(v, Layout):
                item = (k, v.last)
            else:
                item = (k, v)
            last_items.append(item)
        return self.clone(last_items)


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


class Layout(Dimensioned):
    """
    A Layout provides a convenient container to lay out a primary plot
    with some additional supplemental plots, e.g. an image in a
    SheetMatrix annotated with a luminance histogram. Layout accepts a
    list of three View elements, which are laid out as follows with
    the names 'main', 'top' and 'right':
     ___________ __
    |____ 3_____|__|
    |           |  |  1:  main
    |           |  |  2:  right
    |     1     |2 |  3:  top
    |           |  |
    |___________|__|
    """

    dimensions = param.List(default=[Dimension('Layout')], constant=True)

    layout_order = ['main', 'right', 'top']

    _deep_indexable = True

    def __init__(self, views, **params):

        self.main_layer = 0 # The index of the main layer if .main is an overlay
        if len(views) > 3:
            raise Exception('Layout accepts no more than three elements.')

        if isinstance(views, dict):
            wrong_pos = [k for k in views if k not in self.layout_order]
            if wrong_pos:
                raise Exception('Wrong Layout positions provided.')
            else:
                self.data = views
        elif isinstance(views, list):
            self.data = dict(zip(self.layout_order, views))

        if 'dimensions' in params:
            params['dimensions'] = [d if isinstance(d, Dimension) else Dimension(d)
                                    for d in params.pop('dimensions')]

        super(Layout, self).__init__(**params)


    def __len__(self):
        return len(self.data)


    def get(self, key, default=None):
        return self.data[key] if key in self.data else default


    def __getitem__(self, key):
        if key is ():
            return self
        if isinstance(key, int) and key <= len(self):
            if key == 0:  return self.main
            if key == 1:  return self.right
            if key == 2:  return self.top
        elif isinstance(key, str) and key in self.data:
            return self.data[key]
        else:
            raise KeyError("Key {0} not found in Layout.".format(key))


    @property
    def deep_dimensions(self):
        return ['Layout'] + self.main.deep_dimensions

    @property
    def style(self):
        return [el.style for el in self]


    @style.setter
    def style(self, styles):
        for layer, style in zip(self, styles):
            layer.style = style


    def __lshift__(self, other):
        if isinstance(other, Layout):
            raise Exception("Cannot adjoin two Layout objects.")
        views = [self.data.get(k, None) for k in self.layout_order]
        return Layout([v for v in views if v is not None] + [other])


    @property
    def main(self):
        return self.data.get('main', None)

    @property
    def right(self):
        return self.data.get('right', None)

    @property
    def top(self):
        return self.data.get('top', None)

    @property
    def last(self):
        items = [(k, v.last) if isinstance(v, NdMapping) else (k, v)
                 for k, v in self.data.items()]
        return self.__class__(dict(items))


    def __iter__(self):
        i = 0
        while i < len(self):
            yield self[i]
            i += 1


    def __add__(self, other):
        if isinstance(other, GridLayout):
            elements = [self] + list(other.values())
        else:
            elements = [self, other]
        return GridLayout(elements)


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

    def _nearest_neighbor(self, key):
        q = np.array(key)
        idx = np.argmin([np.inner(q - np.array(x), q - np.array(x))
                         if self.ndims == 2 else np.abs(q-x)
                         for x in self.keys()])
        return self.values()[idx]


    def __getitem__(self, indexslice):
        if indexslice in [Ellipsis, ()]:
            return self

        map_slice, data_slice = self._split_index(indexslice)
        map_slice = self._transform_indices(map_slice)

        if all(not isinstance(el, slice) for el in map_slice):
            return self._dataslice(self._nearest_neighbor(map_slice), data_slice)
        else:
            return super(Grid, self).__getitem__(indexslice)


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
            self._type = None if len(self) == 0 else self.values()[0].__class__
        return self._type


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
            if isinstance(v, Layout):
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
            if isinstance(v, Layout):
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