"""
Supplies the View and Map abstract base classes. A View the basic data
structure that holds raw data and can be visualized. A Map is an
instance of NdMapping, a sliceable, multi-dimensional container that
holds View objects as values.
"""

import param
from .dimension import Dimension, Dimensioned
from .options import options
from .ndmapping import NdMapping


class View(Dimensioned):
    """
    A view is a data structure for holding data, which may be plotted
    using matplotlib. Views have an associated title and style
    name. All Views may be composed together into a GridLayout using
    the addition operator.
    """

    __abstract = True

    title = param.String(default='{label} {value}', doc="""
        The title formatting string allows the title to be composed from
        the view {label}, {value} quantity and view {type} but can also be set
        to a simple string.""")

    value = param.String(default='View')

    options = options

    def __init__(self, data, **params):
        self.data = data
        self._style = params.pop('style', None)
        super(View, self).__init__(**params)


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


    def table(self, **kwargs):
        """
        This method transforms any View type into a Table
        as long as it implements a dimension_values method.
        """
        from ..view import Table
        keys = zip(*[self.dimension_values(dim.name)
                 for dim in self.index_dimensions])
        values = zip(*[self.dimension_values(dim.name)
                       for dim in self.value_dimensions])
        params = dict(index_dimensions=self.index_dimensions,
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
        to View.options.
        """
        for name, match in d.pop('style_objects').items():
            for style in match:
                self.options[name] = style
        self.__dict__.update(d)


    def __repr__(self):
        params = ', '.join('%s=%r' % (k,v) for (k,v) in self.get_param_values())
        return "%s(%r, %s)" % (self.__class__.__name__, self.data, params)



class Map(NdMapping):
    """
    A Map is a map of Views over a number of specified dimensions. The
    dimension may be a spatial dimension (i.e., a ZStack), time
    (specifying a frame sequence) or any other combination of
    Dimensions.  Map also adds handling of styles, appending the
    Dimension keys and values to titles and a number of methods to
    manipulate the Dimensions.

    Map objects can be sliced, sampled, reduced, overlaid and split
    along its and its containing Views dimensions. Subclasses should
    implement the appropriate slicing, sampling and reduction methods
    for their View type.
    """

    title_suffix = param.String(default='\n {dims}', doc="""
       A string appended to the View titles when they are added to the
       Map. Default adds a new line with the formatted dimensions
       of the Map inserted using the {dims} formatting keyword.""")

    value = param.String(default='Map')

    data_type = (View, NdMapping)

    _abstract = True
    _deep_indexable = True
    _type = None
    _style = None

    @property
    def type(self):
        """
        The type of elements stored in the map.
        """
        if self._type is None:
            self._type = None if len(self) == 0 else self.last.__class__
        return self._type


    @property
    def style(self):
        """
        The style of elements stored in the map.
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
    def empty_element(self):
        return self._type(None)


    def _item_check(self, dim_vals, data):
        if self.style is not None and self.style != data.style:
            data.style = self.style

        if self.type is not None and (type(data) != self.type):
            raise AssertionError("%s must only contain one type of View." %
                                 self.__class__.__name__)
        super(Map, self)._item_check(dim_vals, data)


    def get_title(self, key, item, group_size=2):
        """
        Resolves the title string on the View being added to the Map,
        adding the Maps title suffix.
        """
        if self.ndims == 1 and self.get_dimension('Default'):
            title_suffix = ''
        else:
            title_suffix = self.title_suffix
        dimension_labels = [dim.pprint_value(k) for dim, k in
                            zip(self.index_dimensions, key)]
        groups = [', '.join(dimension_labels[i*group_size:(i+1)*group_size])
                  for i in range(len(dimension_labels))]
        dims = '\n '.join(g for g in groups if g)
        title_suffix = title_suffix.format(dims=dims)
        return item.title + title_suffix


    def sample(self, dimsample_map, new_axis=None):
        """
        Base class implements signature for sampling View dimensions
        and optionally overlaying the resulting reduced dimensionality
        Views by specifying a list group_by dimensions.
        """
        raise NotImplementedError


    def table(self, **kwargs):
        """
        Creates Table from all the elements in the Map.
        """

        table = None
        for key, value in self.data.items():
            value = value.table(**kwargs)
            for idx, (dim, val) in enumerate(zip(self.index_dimensions, key)):
                value = value.add_dimension(dim, idx, val)
            if table is None:
                table = value
            else:
                table.update(value)
        return table


    def reduce(self, **reduce_map):
        """
        Base class implements signature for reducing dimensions,
        subclasses with Views of fixed dimensionality can then
        appropriately implement reducing the correct view types.
        """
        raise NotImplementedError


__all__ = list(set([_k for _k, _v in locals().items()
                    if isinstance(_v, type) and issubclass(_v, Dimensioned)]))
