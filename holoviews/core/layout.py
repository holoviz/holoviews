"""
Supplies Pane, GridLayout and AdjointLayout. Pane extends View to
allow multiple Views to be presented side-by-side in a GridLayout. An
AdjointLayout allows one or two Views to be ajoined to a primary View
to act as supplementary elements.
"""
import uuid
from itertools import groupby

import numpy as np

import param

from .dimension import Dimension, Dimensioned
from .ndmapping import NdMapping
from .options import options
from .tree import AttrTree
from .util import int_to_roman, sanitize_identifier
from .view import View, Map


class Pane(View):
    """
    Pane extends the View type with the add and left shift operators
    which allow the Pane to be embedded within Layouts and GridLayouts.
    """

    value = param.String(default='Pane')

    def __add__(self, obj):
        return ViewTree.from_view(self) + ViewTree.from_view(obj)


    def __lshift__(self, other):
        if isinstance(other, (View, NdMapping)):
            return AdjointLayout([self, other])
        elif isinstance(other, AdjointLayout):
            return AdjointLayout(other.data.values()+[self])
        else:
            raise TypeError('Cannot append {0} to a AdjointLayout'.format(type(other).__name__))


class GridLayout(NdMapping):
    """
    A GridLayout is an NdMapping, which unlike a ViewMap lays
    the individual elements out in a Grid.
    """

    value = param.String(default='GridLayout')

    def __init__(self, initial_items, **params):
        self._max_cols = 4
        self._style = None
        if isinstance(initial_items, list):
            initial_items = [(idx, item) for idx, item in enumerate(initial_items)]
        elif isinstance(initial_items, NdMapping):
            params = dict(initial_items.get_param_values(), **params)
        super(GridLayout, self).__init__(initial_items=initial_items, **params)


    @property
    def shape(self):
        num = len(self.keys())
        if num <= self._max_cols:
            return (1, num)
        nrows = num // self._max_cols
        last_row_cols = num % self._max_cols
        return nrows+(1 if last_row_cols else 0), min(num, self._max_cols)


    @property
    def grid_items(self):
        """
        Compute a dict of {(row,column): view} elements from the
        current set of items and specified number of columns.
        """
        if list(self.keys()) == []:  return {}
        cols = self._max_cols
        return {(idx // cols, idx % cols): item
                for idx, item in enumerate(self)}


    def cols(self, n):
        self._max_cols = n
        return self


    def __add__(self, obj):
        return ViewTree.from_view(self) + ViewTree.from_view(obj)


    @property
    def last(self):
        """
        Returns another GridLayout constituted of the last views of the
        individual elements (if they are maps).
        """
        last_items = []
        for (k, v) in self.items():
            if isinstance(v, NdMapping):
                item = (k, v.clone((v.last_key, v.last)))
            elif isinstance(v, AdjointLayout):
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


class AdjointLayout(Dimensioned):
    """
    A AdjointLayout provides a convenient container to lay out a primary plot
    with some additional supplemental plots, e.g. an image in a
    Matrix annotated with a luminance histogram. AdjointLayout accepts a
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

    index_dimensions = param.List(default=[Dimension('AdjointLayout')], constant=True)

    value = param.String(default='AdjointLayout')

    layout_order = ['main', 'right', 'top']

    _deep_indexable = True

    def __init__(self, views, **params):

        self.main_layer = 0 # The index of the main layer if .main is an overlay
        if len(views) > 3:
            raise Exception('AdjointLayout accepts no more than three elements.')

        if isinstance(views, dict):
            wrong_pos = [k for k in views if k not in self.layout_order]
            if wrong_pos:
                raise Exception('Wrong AdjointLayout positions provided.')
            else:
                self.data = views
        elif isinstance(views, list):
            self.data = dict(zip(self.layout_order, views))

        if 'dimensions' in params:
            params['dimensions'] = [d if isinstance(d, Dimension) else Dimension(d)
                                    for d in params.pop('dimensions')]

        super(AdjointLayout, self).__init__(**params)


    def __len__(self):
        return len(self.data)


    def get(self, key, default=None):
        return self.data[key] if key in self.data else default


    def dimension_values(self, dimension):
        if isinstance(dimension, int):
            dimension = self.get_dimension(dimension).name
        if dimension in self._cached_index_names:
            return self.layout_order[:len(self.data)]
        else:
            return self.main.dimension_values(dimension)


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
            raise KeyError("Key {0} not found in AdjointLayout.".format(key))


    @property
    def deep_dimensions(self):
        return self.main.dimensions

    @property
    def style(self):
        return [el.style for el in self]


    @style.setter
    def style(self, styles):
        for layer, style in zip(self, styles):
            layer.style = style


    def __lshift__(self, other):
        views = [self.data.get(k, None) for k in self.layout_order]
        return AdjointLayout([v for v in views if v is not None] + [other])


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


    def __add__(self, obj):
        return ViewTree.from_view(self) + ViewTree.from_view(obj)


class ViewTree(AttrTree):

    style = 'ViewTree'

    def __init__(self, *args, **kwargs):
        self.__dict__['_display'] = 'auto'
        self.__dict__['_max_cols'] = 4
        self.__dict__['name'] = 'ViewTree_' + str(uuid.uuid4())[0:4]
        super(ViewTree, self).__init__(*args, **kwargs)


    def display(self, option):
        "Sets the display policy of the ViewTree before returning self"
        options = ['auto', 'all']
        if option not in options:
            raise Exception("Display option must be one of %s" %
                            ','.join(repr(el) for el in options))
        self._display = option
        return self


    def cols(self, ncols):
        self._max_cols = ncols
        return self


    def __getitem__(self, key):
        if len(key) == 2 and not any([isinstance(k, str) for k in key]):
            row, col = key
            idx = row * self._cols + col
            keys = self.path_items.keys()
            if idx >= len(keys) or col >= self._cols:
                raise KeyError('Index %s is outside available item range' % str(key))
            key = keys[idx]
        return super(ViewTree, self).__getitem__(key)


    @property
    def grid_items(self):
        return {tuple(np.unravel_index(idx, self.shape)): el
                for idx, el in enumerate(self)}


    def __len__(self):
        return len(self.path_items)


    @property
    def shape(self):
        num = len(self)
        if num <= self._max_cols:
            return (1, num)
        nrows = num // self._max_cols
        last_row_cols = num % self._max_cols
        return nrows+(1 if last_row_cols else 0), min(num, self._max_cols)


    def _relabel(self, items):
        relabelled_items = []
        group_fn = lambda x: x[0][0:2] if len(x[0]) > 2 else (x[0][0],)
        for path, group in groupby(items, key=group_fn):
            group = list(group)
            if len(group) == 1 and len(path) > 1:
                relabelled_items.append((path, group[0][1]))
                continue
            for idx, (path, item) in enumerate(group):
                if len(path) == 2 and not item.label:
                    numeral = int_to_roman(idx+1)
                    new_path = (path[0], numeral) if not item.label else path + (numeral,)
                else:
                    new_path = path
                relabelled_items.append((new_path, item))
        return relabelled_items


    @staticmethod
    def _get_path(view):
        label = view.label if view.label else 'I'
        return (sanitize_identifier(view.value),
                sanitize_identifier(label))


    @staticmethod
    def from_view(view):
        if isinstance(view, ViewTree): return view
        return ViewTree(path_items=[(ViewTree._get_path(view), view)])


    def group(self, name):
        new_items = [((name, path[-1]), item) for path, item in self.path_items.items()]
        return ViewTree(path_items=self._relabel(new_items))


    def __add__(self, other):
        other = self.from_view(other)
        items = list(self.path_items.items()) + list(other.path_items.items())
        return ViewTree(path_items=self._relabel(items)).display('all')



__all__ = list(set([_k for _k, _v in locals().items()
                    if isinstance(_v, type) and (issubclass(_v, Dimensioned)
                                                 or issubclass(_v, ViewTree))]))
