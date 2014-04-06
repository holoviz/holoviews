"""
Views objects used to hold data and provide indexing into different coordinate
systems.
"""

__version__='$Revision$'

import math
import param

from ndmapping import NdMapping, Dimension, AttrDict, map_type


class View(param.Parameterized):
    """
    A view is a data structure for holding data, which may be plotted
    using matplotlib. Views have an associated title, style name and
    metadata information.  All Views may be composed together into a
    GridLayout using the addition operator.
    """

    label = param.String(default=None, allow_None=True, constant=True, doc="""
      A short label used to indicate what kind of data is contained
      within the view object.""")

    title = param.String(default=None, allow_None=True, doc="""
       A short description of the layer that may be used as a title.""")

    metadata = param.Dict(default=AttrDict(), doc="""
        Additional information to be associated with the Layer.""")


    @property
    def style(self):
        """
        The name of the style that may be used to control display of
        this view. If a style name is not set and but a label is
        assigned, then this label name will be used instead
        """
        if (self._style is None) and self.label:
            return self.label
        elif self._style is None:
            return 'Default'
        else:
            return self._style

    @style.setter
    def style(self, val):
        self._style = val


    def __init__(self, data, **kwargs):
        self.data = data
        self._style = kwargs.pop('style', None)
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

    @property
    def style(self):
        return [el.style for el in self]


    @style.setter
    def style(self, styles):
        for layer, style in zip(self, styles):
            layer.style = style


    def add(self, layer):
        """
        Overlay a single layer on top of the existing overlay.
        """
        if layer.label in [o.label for o in self.data]:
            self.warning('Label %s already defined in Overlay' % layer.label)
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


    def __len__(self):
        return len(self.data)


    def __iter__(self):
        i = 0
        while i < len(self):
            yield self[i]
            i += 1



class GridLayout(NdMapping):

    dimensions = param.List(default=[Dimension('Row', type=int),
                                     Dimension('Column', type=int)], constant=True)

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
        return self.clone(new_values+self.values())


    def __len__(self):
        return max([len(v) for v in self.values() if isinstance(v, NdMapping)]+[1])


__all__ = list(set([_k for _k,_v in locals().items() if isinstance(_v,type) and
                    (issubclass(_v, NdMapping) or issubclass(_v, View))]))
