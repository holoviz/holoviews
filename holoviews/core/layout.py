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

from .dimension import LabelledData, Dimension, Dimensioned, ViewableElement
from .ndmapping import NdMapping, UniformNdMapping
from .tree import AttrTree
from .util import int_to_roman


class Composable(object):
    """
    Pane extends the ViewableElement type with the add and left shift operators
    which allow the Pane to be embedded within Layouts and GridLayouts.
    """

    def __add__(self, obj):
        return LayoutTree.from_view(self) + LayoutTree.from_view(obj)


    def __lshift__(self, other):
        if isinstance(other, (ViewableElement, NdMapping)):
            return AdjointLayout([self, other])
        elif isinstance(other, AdjointLayout):
            return AdjointLayout(other.data.values()+[self])
        else:
            raise TypeError('Cannot append {0} to a AdjointLayout'.format(type(other).__name__))



class AdjointLayout(Dimensioned):
    """
    A AdjointLayout provides a convenient container to lay out a primary plot
    with some additional supplemental plots, e.g. an image in a
    Matrix annotated with a luminance histogram. AdjointLayout accepts a
    list of three ViewableElement elements, which are laid out as follows with
    the names 'main', 'top' and 'right':
     ___________ __
    |____ 3_____|__|
    |           |  |  1:  main
    |           |  |  2:  right
    |     1     |2 |  3:  top
    |           |  |
    |___________|__|
    """

    key_dimensions = param.List(default=[Dimension('AdjointLayout')], constant=True)

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
                data = views
        elif isinstance(views, list):
            data = dict(zip(self.layout_order, views))

        super(AdjointLayout, self).__init__(data, **params)


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
        return LayoutTree.from_view(self) + LayoutTree.from_view(obj)



class NdLayout(UniformNdMapping):
    """
    A NdLayout is an NdMapping, which unlike a HoloMap lays
    the individual elements out in a AxisLayout.
    """

    value = param.String(default='NdLayout')

    data_type = (ViewableElement, AdjointLayout, UniformNdMapping)

    def __init__(self, initial_items, **params):
        self._max_cols = 4
        self._style = None
        if isinstance(initial_items, list):
            initial_items = [(idx, item) for idx, item in enumerate(initial_items)]
        elif isinstance(initial_items, NdMapping):
            params = dict(initial_items.get_param_values(), **params)
        super(NdLayout, self).__init__(initial_items=initial_items, **params)


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
        Compute a dict of {(row,column): element} elements from the
        current set of items and specified number of columns.
        """
        if list(self.keys()) == []:  return {}
        cols = self._max_cols
        return {(idx // cols, idx % cols): item
                for idx, item in enumerate(self)}


    @property
    def grid_keys(self):
        """
        Compute a dict of {(row,column): element} elements from the
        current set of items and specified number of columns.
        """
        if list(self.keys()) == []:  return {}
        cols = self._max_cols
        return {(idx // cols, idx % cols): key
                for idx, key in enumerate(self.keys())}


    def cols(self, n):
        self._max_cols = n
        return self


    def __add__(self, obj):
        return LayoutTree.from_view(self) + LayoutTree.from_view(obj)


    @property
    def last(self):
        """
        Returns another NdLayout constituted of the last views of the
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



class LayoutTree(AttrTree, LabelledData):
    """
    A LayoutTree is an AttrTree with ViewableElement objects as leaf values. Unlike
    AttrTree, a LayoutTree supports a rich display, displaying leaf
    items in a grid style layout. In addition to the usual AttrTree
    indexing, LayoutTree supports indexing of items by their row and
    column index in the layout.

    The maximum number of columns in such a layout may be controlled
    with the cols method and the display policy is set with the
    display method. A display policy of 'auto' may use the string repr
    of the tree for large trees that would otherwise take a long time
    to display wheras a policy of 'all' will always display all the
    available leaves. The detailed settings for the 'auto' policy may
    be set using the MAX_BRANCHES option of the %view magic.
    """

    value = param.String(default='LayoutTree', constant=True)

    style = 'LayoutTree'

    def __init__(self, *args, **kwargs):
        self.__dict__['_display'] = 'auto'
        self.__dict__['_max_cols'] = 4
        self.__dict__['name'] = 'ViewTree_' + str(uuid.uuid4())[0:4]
        super(LayoutTree, self).__init__(*args, **kwargs)


    def display(self, option):
        "Sets the display policy of the LayoutTree before returning self"
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
        if isinstance(key, int):
            if key < len(self):
                return self.data.values()[key]
            raise KeyError("Element out of range.")
        if len(key) == 2 and not any([isinstance(k, str) for k in key]):
            row, col = key
            idx = row * self._cols + col
            keys = self.data.keys()
            if idx >= len(keys) or col >= self._cols:
                raise KeyError('Index %s is outside available item range' % str(key))
            key = keys[idx]
        return super(LayoutTree, self).__getitem__(key)

    @property
    def grid_items(self):
        return {tuple(np.unravel_index(idx, self.shape)): el
                for idx, el in enumerate(self)}

    @property
    def grid_keys(self):
        return {tuple(np.unravel_index(idx, self.shape)): path
                for idx, path in enumerate(self.path_items.keys())}

    def __len__(self):
        return len(self.data)


    @property
    def shape(self):
        num = len(self)
        if num <= self._max_cols:
            return (1, num)
        nrows = num // self._max_cols
        last_row_cols = num % self._max_cols
        return nrows+(1 if last_row_cols else 0), min(num, self._max_cols)


    def _relabel(self, items):
        """
        Given a list of path items (list of tuples where each element
        is a (path, element) pair), generate a new set of path items that
        guarantees that no paths clash. This uses the element labels as
        appropriate and automatically generates roman numeral
        identifiers if necessary.
        """
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
    def from_view(view):
        # Return ViewTrees and Overlays directly
        if isinstance(view, LayoutTree) and not isinstance(view, ViewableElement): return view
        return LayoutTree(items=[((view.value, view.label if view.label else 'I'), view)])


    def group(self, name):
        new_items = [((name, path[-1]), item) for path, item in self.data.items()]
        return LayoutTree(items=self._relabel(new_items))


    def __add__(self, other):
        other = self.from_view(other)
        items = list(self.data.items()) + list(other.data.items())
        return LayoutTree(items=self._relabel(items)).display('all')



__all__ = list(set([_k for _k, _v in locals().items()
                    if isinstance(_v, type) and (issubclass(_v, Dimensioned)
                                                 or issubclass(_v, LayoutTree))]))
