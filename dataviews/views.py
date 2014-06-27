"""
Views objects used to hold data and provide indexing into different coordinate
systems.
"""
import math
from collections import OrderedDict, defaultdict

import numpy as np

import param

from .ndmapping import NdMapping, Dimensional, Dimension
from .options import options


class View(param.Parameterized, Dimensional):
    """
    A view is a data structure for holding data, which may be plotted
    using matplotlib. Views have an associated title and style
    name. All Views may be composed together into a GridLayout using
    the addition operator.
    """

    dimensions = param.List(default=[], doc="""List of dimensions the View
        can be indexed by.""")

    label = param.String(default='', constant=True, doc="""
        A string label or Dimension object used to indicate what kind of data
        is contained within the view object.""")

    title = param.String(default='{label}', doc="""
        The title formatting string allows the title to be composed from
        the view {label}, {value} quantity and view {type} but can also be set
        to a simple string.""")

    value = param.ClassSelector(class_=(str, Dimension),
                                default=Dimension('Y'), doc="""
        The value is a string or Dimension object, describing the quantity
        being held in the View.""")

    options = options

    def __init__(self, data, **kwargs):
        self.data = data
        self._style = kwargs.pop('style', None)
        if 'dimensions' in kwargs:
            kwargs['dimensions'] = [Dimension(d) if not isinstance(d, Dimension) else d
                                    for d in kwargs.pop('dimensions')]
        if 'value' in kwargs and not isinstance(kwargs['value'], Dimension):
            kwargs['value'] = Dimension(kwargs['value'])
        if not 'label' in kwargs: kwargs['label'] = str(kwargs.get('value', ''))
        super(View, self).__init__(**kwargs)



    def sample(self, **samples):
        """
        Base class signature to demonstrate API for sampling Views.
        To sample a View kwargs, where the keyword matches a Dimension
        in the View and the value matches a corresponding entry in the
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


    def clone(self, data, type=None, **kwargs):
        """
        Returns a clone with matching parameter values containing the
        specified items (empty by default).
        """
        settings = dict(self.get_param_values(), **kwargs)
        return self.__class__(data, **settings)


    @property
    def deep_dimensions(self):
        return self.dimension_labels


    @property
    def stack_type(self):
        return Stack


    @property
    def style(self):
        """
        The name of the style that may be used to control display of
        this view. If a style name is not set and but a label is
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


    def __getstate__(self):
        """
        When pickling, make sure to save the relevant style and
        plotting options as well.
        """
        obj_dict = self.__dict__.copy()
        if isinstance(self, Overlay): return obj_dict
        obj_dict['style_objects'] = {}
        for match in self.options.fuzzy_match_keys(self.style):
            obj_dict['style_objects'][match] = self.options[match]
        return obj_dict


    def __setstate__(self, d):
        """
        When unpickled, restore the saved style and plotting options
        to View.options.
        """
        if isinstance(self, Overlay) or 'style_objects' not in d:
            self.__dict__.update(d)
            return

        for name, match in d.pop('style_objects').items():
            for style in match:
                self.options[name] = style
        self.__dict__.update(d)


    def __add__(self, obj):
        if not isinstance(obj, GridLayout):
            return GridLayout(initial_items=[self, obj])


    def __lshift__(self, other):
        if isinstance(other, (View, Overlay, NdMapping)):
            return Layout([self, other])
        elif isinstance(other, Layout):
            return Layout(other.data.values()+[self])
        else:
            raise TypeError('Cannot append {0} to a Layout'.format(type(other).__name__))


    def __repr__(self):
        params = ', '.join('%s=%r' % (k,v) for (k,v) in self.get_param_values())
        return "%s(%r, %s)" % (self.__class__.__name__, self.data, params)



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
    SheetStack or DataStack with a 'Time' dimension. The interval
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
            if hasattr(box, 'lbrt'):         self.box(box, None)
            elif isinstance(box[1], dict):   self.box(*box)
            else:                            self.box(box, None)

        for vline in vlines:
            self.vline(*(vline if isinstance(vline, tuple) else (vline, None)))

        for hline in hlines:
            self.hline(*(hline if isinstance(hline, tuple) else (hline, None)))

        for arrow in arrows:
            spec, interval = (arrow, None) if isinstance(arrow[0], tuple) else arrow
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
        if hasattr(box, 'lbrt'):
            (l,b,r,t) = box.lbrt()
        else:
            ((l,b), (r,t)) = box

        self.line(((t,l), (t,r), (b,r), (b,l), (t,l)),
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

    dimensions = param.List(default=[Dimension('Overlay')], constant=True, doc="""List
      of dimensions the View can be indexed by.""")

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
        return [el.style for el in self.data]


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



class Stack(NdMapping):
    """
    A Stack is a stack of Views over a number of specified dimensions. The
    dimension may be a spatial dimension (i.e., a ZStack), time
    (specifying a frame sequence) or any other combination of Dimensions.
    Stack also adds handling of styles, appending the Dimension keys and
    values to titles and a number of methods to manipulate the Dimensions.

    Stack objects can be sliced, sampled, reduced, overlaid and split along
    its and its containing Views dimensions. Subclasses should implement
    the appropriate slicing, sampling and reduction methods for their View
    type.
    """

    title_suffix = param.String(default='\n {dims}', doc="""
       A string appended to the View titles when they are added to the
       Stack. Default adds a new line with the formatted dimensions
       of the Stack inserted using the {dims} formatting keyword.""")

    data_type = View
    overlay_type = Overlay

    _type = None
    _style = None

    @property
    def type(self):
        """
        The type of elements stored in the stack.
        """
        if self._type is None:
            self._type = None if len(self) == 0 else self.last.__class__
        return self._type


    @property
    def style(self):
        """
        The style of elements stored in the stack.
        """
        if self._style is None:
            self._style = None if len(self) == 0 else self.last.style
        return self._style


    @style.setter
    def style(self, style_name):
        self._style = style_name
        for val in self.values():
            val.style = style_name


    @property
    def deep_dimensions(self):
        return self.dimension_labels + self.last.deep_dimensions


    @property
    def empty_element(self):
        return self._type(None)


    def _item_check(self, dim_vals, data):
        if self.style is not None and self.style != data.style:
            data.style = self.style

        if self.type is not None and (type(data) != self.type):
            raise AssertionError("%s must only contain one type of View." %
                                 self.__class__.__name__)
        super(Stack, self)._item_check(dim_vals, data)


    def get_title(self, key, item, group_size=2):
        """
        Resolves the title string on the View being added to the
        Stack, adding the Stacks title suffix.
        """
        if self.ndims == 1 and self.dim_dict.get('Default'):
            title_suffix = ''
        else:
            title_suffix = self.title_suffix
        dimension_labels = [dim.pprint_value(k) for dim, k in
                            zip(self.dimensions, key)]
        groups = [', '.join(dimension_labels[i*group_size:(i+1)*group_size])
                  for i in range(len(dimension_labels))]
        dims = '\n '.join(g for g in groups if g)
        title_suffix = title_suffix.format(dims=dims)
        return item.title + title_suffix


    def _split_dim_keys(self, dimensions):
        """
        Split the NdMappings keys into two groups given a list of
        dimensions to split out.
        """

        # Find dimension indices
        first_dims = [d for d in self.dimensions if d.name not in dimensions]
        first_inds = [self.dim_index(d.name) for d in first_dims]
        second_dims = [d for d in self.dimensions if d.name in dimensions]
        second_inds = [self.dim_index(d.name) for d in second_dims]

        # Split the keys
        keys = list(self._data.keys())
        first_keys, second_keys = zip(*[(tuple(k[fi] for fi in first_inds),
                                        tuple(k[si] for si in second_inds))
                                        for k in keys])
        return first_dims, first_keys, second_dims, second_keys


    def sort_key(self, unordered):
        """
        Given an unordered list of (dimension, value) pairs returns
        the sorted key.
        """
        dim_orderfn = lambda k: self.dim_index(k[0].name)
        return tuple([v for k, v in sorted(unordered, key=dim_orderfn)])


    def split_dimensions(self, dimensions):
        """
        Split the dimensions in the NdMapping across two NdMappings,
        where the inner mapping is of the same type as the original
        Stack.
        """
        inner_dims, deep_dims = self._split_dims(dimensions)
        if self.ndims == 1:
            self.warning('Cannot split Stack with only one dimension.')
            return self
        if len(deep_dims):
            raise Exception('NdMapping does not support splitting of deep dimensions.')
        first_dims, first_keys, second_dims, second_keys = self._split_dim_keys(inner_dims)
        self._check_key_type = False # Speed optimization

        split_data = NdMapping(dimensions=first_dims)
        for fk in first_keys:  # The first groups keys
            split_data[fk] = self.clone(dimensions=second_dims)
            for sk in second_keys:  # The second groups keys
                # Generate a candidate expanded key
                unordered_dimkeys = list(zip(first_dims, fk)) + list(zip(second_dims, sk))
                sorted_key = self.sort_key(unordered_dimkeys)
                if sorted_key in self._data.keys():  # If the expanded key actually exists...
                    split_data[fk][sk] = self[sorted_key]

        self._check_key_type = True # Re-enable checks

        return split_data


    def overlay_dimensions(self, dimensions):
        """
        Splits the Stack along a specified number of dimensions and overlays
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
            overlay.legend_label = stack.pprint_dimkey(key)
            for inner, v in list(stack.items())[1:]:
                v.legend_label = stack.pprint_dimkey(inner)
                overlay = overlay * v
            new_stack[outer] = overlay

        if self.ndims == 1:
            return list(new_stack.values())[0]
        else:
            return new_stack


    def grid(self, dimensions):
        """
        Grid takes a list of one or two dimensions, and lays out the containing
        Views along these axes in a Grid.
        """
        if len(dimensions) > 2:
            raise ValueError('At most two dimensions can be laid out in a grid.')

        if self.ndims == 1 and dimensions == self.dimension_labels:
            split_stack = self
        elif all(d in self.dimension_labels for d in dimensions):
            split_dims = [d for d in self.dimension_labels if d not in dimensions]
            split_stack = self.split_dimensions(split_dims)
            split_stack = split_stack.reindex(dimensions)
        else:
            raise ValueError('Stack does not have supplied dimensions.')

        return Grid(split_stack, dimensions=split_stack.dimensions)


    def map(self, map_fn, **kwargs):
        """
        Map a function across the stack.
        """
        mapped_items = [(k, map_fn(el, k)) for k, el in self.items()]
        if isinstance(mapped_items[0][1], tuple):
            split = [[(k, v) for v in val] for (k, val) in mapped_items]
            item_groups = [list(el) for el in zip(*split)]
        else:
            item_groups = [mapped_items]
        clones = tuple(self.clone(els, **kwargs)
                       for (i, els) in enumerate(item_groups))
        return clones if len(clones) > 1 else clones[0]


    def sample(self, dimsample_map, new_axis=None):
        """
        Base class implements signature for sampling View dimensions
        and optionally overlaying the resulting reduced dimensionality
        Views by specifying a list group_by dimensions.
        """
        raise NotImplementedError


    def reduce(self, **reduce_map):
        """
        Base class implements signature for reducing dimensions,
        subclasses with Views of fixed dimensionality can then
        appropriately implement reducing the correct view types.
        """
        raise NotImplementedError


    def split_overlays(self):
        """
        Given a Stack of Overlays of N layers, split out the layers
        into N separate Stacks.
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
        """
        The mul (*) operator implements overlaying of different Views.
        This method tries to intelligently overlay Stacks with differing
        keys. If the Stack is mulled with a simple View each element in
        the Stack is overlaid with the View. If the element the Stack is
        mulled with is another Stack it will try to match up the dimensions,
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


    def __add__(self, obj):
        if not isinstance(obj, GridLayout):
            return GridLayout(initial_items=[self, obj])
        else:
            grid = GridLayout(initial_items=[self])
            grid.update(obj)
            return grid


    def __lshift__(self, other):
        if isinstance(other, (View, Overlay, NdMapping)):
            return Layout([self, other])
        elif isinstance(other, Layout):
            return Layout(other.data+[self])
        else:
            raise TypeError('Cannot append {0} to a Layout'.format(type(other).__name__))



class Layout(param.Parameterized, Dimensional):
    """
    A Layout provides a convenient container to lay out a primary plot
    with some additional supplemental plots, e.g. an image in a
    SheetView annotated with a luminance histogram. Layout accepts a
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



class GridLayout(NdMapping):
    """
    A GridLayout is an NdMapping, which can contain any View or Stack type.
    It is used to group different View or Stack elements into a grid for
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



def find_minmax(lims, olims):
    """
    Takes (a1, a2) and (b1, b2) as input and returns
    (np.min(a1, b1), np.max(a2, b2)). Used to calculate
    min and max values of a number of items.
    """

    limzip = zip(list(lims), list(olims), [np.min, np.max])
    return tuple([float(fn([l, ol])) for l, ol, fn in limzip])



class Grid(NdMapping):
    """
    Grids are distinct from GridLayouts as they ensure all contained elements
    to be of the same type. Unlike GridLayouts, which have integer keys,
    Grids usually have floating point keys, which correspond to a grid
    sampling in some two-dimensional space. This two-dimensional space may
    have to arbitrary dimensions, e.g. for 2D parameter spaces. CoordinateGrid
    is implemented specifically sampling in 2D spatial coordinates.
    """

    dimensions = param.List(default=[Dimension(name="X"), Dimension(name="Y")])

    label = param.String(constant=True, doc="""
      A short label used to indicate what kind of data is contained
      within the CoordinateGrid.""")

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
        elif isinstance(other, Stack) and len(other) == 1:
            view = other.last
        elif isinstance(other, Stack) and len(other) != 1:
            raise Exception("Can only overlay with Stack of length 1")
        else:
            view = other

        overlayed_items = [(k, el * view) for k, el in self.items()]
        return self.clone(overlayed_items)


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
            if isinstance(v, Stack):
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



__all__ = list(set([_k for _k,_v in locals().items() if isinstance(_v,type) and
                    (issubclass(_v, NdMapping) or issubclass(_v, View))]))
