"""
Views objects used to hold data and provide indexing into different coordinate
systems.
"""

__version__='$Revision$'

import math
import param

from ndmapping import NdMapping, Dimension, AttrDict, map_type
from options import options
from boundingregion import BoundingBox


class View(param.Parameterized):
    """
    A view is a data structure for holding data, which may be plotted
    using matplotlib. Views have an associated title, style name and
    metadata information.  All Views may be composed together into a
    GridLayout using the addition operator.
    """

    label = param.String(constant=True, doc="""
      A short label used to indicate what kind of data is contained
      within the view object.""")

    title = param.String(default=None, doc="""
       A short description of the layer that may be used as a title.""")

    metadata = param.Dict(default=AttrDict(), doc="""
        Additional information to be associated with the Layer.""")


    @property
    def style(self):
        """
        The name of the style that may be used to control display of
        this view. If a style name is not set and but a label is
        assigned, then the closest existing style name is returned.
        """
        if (self._style is None) and self.label:
            matches = options.fuzzy_matches(self.label.replace(' ', '_'))
            return matches[0] if matches else 'DefaultStyle'
        elif self._style is None:
            return 'DefaultStyle'
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


    def __lshift__(self, other):
        if isinstance(other, (View, Overlay, NdMapping)):
            return Layout([self, other])
        elif isinstance(other, Layout):
            return Layout(other.data+[self])
        else:
            raise TypeError('Cannot append {0} to a Layout'.format(type(other).__name__))


class Annotation(View):
    """
    An annotation is a type of View that is displayed on the top of an
    overlay. Annotations elements do not depend on the details of the
    data displayed and are generally for the convenience of the user
    (e.g. to draw attention to specific areas of the figure using
    arrows, boxes or labels).

    All annotations have an optional interval argument that indicates
    which stack elements they apply to. For instance, this allows
    annotations for a specific time interval when overlaid over a
    SheetStack or DataStack with a 'time' dimension. The interval
    argument is a dictionary of dimension keys and tuples containing
    (start, end) values. A value of None, indicates an unspecified
    constraint.
    """

    def __init__(self, boxes=[], vlines=[], hlines=[], arrows=[], **kwargs):
        """
        Annotations may be added via method calls or supplied directly
        to the constructor using lists of specification elements or
        (specification, interval) tuples. The specification element
        formats are listed below:

        box: A BoundingBox or ((left, bottom), (right, top)) tuple.

        hline/vline specification: The vertical/horizontal coordinate.

        arrow: An (xy, kwargs) tuple where xy is a coordinate tuple
        and kwargs is a dictionary of the optional arguments accepted
        by the arrow method.
        """
        super(Annotation, self).__init__([], **kwargs)

        for box in boxes:
            self.box(*(box if isinstance(box, tuple) else (box, None)))

        for vline in vlines:
            self.vline(*(vline if isinstance(vline, tuple) else (vline, None)))

        for hline in hlines:
            self.hline(*(hline if isinstance(hline, tuple) else (hline, None)))

        for arrow in arrows:
            spec, interval = arrow if isinstance(arrow, tuple) else (arrow, None)
            self.arrow(spec[0], **dict(spec[1], interval=interval))


    def arrow(self, xy, text='', direction='<', points=40,
              arrowstyle='->', interval=None):
        """
        Draw an arrow along one of the cardinal directions with option
        text. The direction indicates the direction the arrow is
        pointing and the points argument defines the length of the
        arrow in points. Different arrow head styles are supported via
        the arrowstyle argument.
        """
        directions = ['<', '^', '>', 'v']
        if direction.lower() not in directions:
            raise Exception("Valid arrow directions are: %s"
                            % ', '.join(repr(d) for d in directions))

        arrowstyles = ['-', '->', '-[', '-|>', '<->', '<|-|>']
        if arrowstyle not in arrowstyles:
            raise Exception("Valid arrow styles are: %s"
                            % ', '.join(repr(a) for a in arrowstyles))

        self.data.append((direction.lower(), text, xy, points, arrowstyle, interval))


    def line(self, coords, interval=None):
        """
        Draw an arbitrary polyline that goes through the listed
        coordinates.  Coordinates are specified using a list of (x,y)
        tuples.
        """
        self.data.append(('line', coords, interval))


    def box(self, box, interval=None):
        """
        Draw a box with corners specified in the positions specified
        by ((left, bottom), (right, top)). Alternatively, a
        BoundingBox may be supplied.
        """
        if isinstance(box, BoundingBox):
            (l,b,r,t) = box.lbrt()
        else:
            ((l,b), (r,t)) = box

        self.line([(t,l), (t,r), (b,r), (b,l), (t,l)],
                  interval=interval)


    def vline(self, x, interval=None):
        """
        Draw an axis vline (vertical line) at the given x value.
        """
        self.data.append(('vline', x, interval))


    def hline(self, y, interval=None):
        """
        Draw an axis hline (horizontal line) at the given y value.
        """
        self.data.append(('hline', y, interval))


    def __mul__(self, other):
        raise Exception("An annotation can only be overlaid over a different View type.")


class Overlay(View):
    """
    An Overlay allows a group of Layers to be overlaid together. Layers can
    be indexed out of an overlay and an overlay is an iterable that iterates
    over the contained layers.
    """


    label = param.String(doc="""
      A short label used to indicate what kind of data is contained
      within the view object.

      Overlays should not have their label set directly by the user as
      the label is only for defining custom channel operations.""")


    _abstract = True

    _deep_indexable = True

    def __init__(self, overlays, **kwargs):
        super(Overlay, self).__init__([], **kwargs)
        self.set(overlays)


    @property
    def labels(self):
        return [el.label for el in self]


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


class Layout(param.Parameterized):
    """
    A Layout provides a convenient container to lay out a primary plot with
    some additional supplemental plots, e.g. an image in a SheetView annotated
    with a luminance histogram. Layout accepts a list of three View elements,
    which are laid out as follows:

    _____________________ _______
    |         3         | |empty|
    |___________________| |_____|
    _____________________  ______
    |                   | |     |
    |                   | |     |
    |                   | |     |
    |         1         | |  2  |
    |                   | |     |
    |                   | |     |
    |                   | |     |
    |___________________| |_____|

    """

    layout_order = ['main', 'right', 'top']

    def __init__(self, views, **params):
        if len(views) > 3:
            raise Exception('Layout accepts no more than three elements.')

        self.data = dict(zip(self.layout_order, views))
        super(Layout, self).__init__(**params)


    def __len__(self):
        return len(self.data)


    def get(self, key, default=None):
        return self.data[key] if key in self.data else default


    def __getitem__(self, key):
        if isinstance(key, int) and key <= len(self):
            if key == 0:
                return self['main']
            if key == 1:
                return self['right']
            if key == 2:
                return self['top']
        elif isinstance(key, str) and key in self.data:
            return self.data[key]
        else:
            raise KeyError("Key {0} not found in Layout.".format(key))


    @property
    def style(self):
        return [el.style for el in self]


    @style.setter
    def style(self, styles):
        for layer, style in zip(self, styles):
            layer.style = style


    def __lshift__(self, other):
        if isinstance(other, Layout):
            return Layout(self.data.values()+other.data.values())
        else:
            return Layout(self.data.values()+[other])


    @property
    def main(self):
        return self.data['main']


    @property
    def right(self):
        return self.data['right']


    def top(self):
        return self.data['top']


    def __iter__(self):
        i = 0
        while i < len(self):
            yield self[i]
            i += 1


    def __add__(self, other):
        if isinstance(other, GridLayout):
            elements = [self] + other.values()
        else:
            elements = [self, other]
        return GridLayout([elements])



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
        return self.clone([self.values()+new_values])



__all__ = list(set([_k for _k,_v in locals().items() if isinstance(_v,type) and
                    (issubclass(_v, NdMapping) or issubclass(_v, View))]))
