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

    dimensions = param.List(default=[], doc="""List of dimensions the View
        can be indexed by.""")

    label = param.String(default='', constant=True, doc="""
        A string label or Dimension object used to indicate what kind of data
        is contained within the view object.""")

    title = param.String(default='{label}', doc="""
        The title formatting string allows the title to be composed from
        the view {label}, {value} quantity and view {type} but can also be set
        to a simple string.""")

    value = param.ClassSelector(class_=Dimension,
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
        elif 'value' not in kwargs:
            kwargs['value'] = self.value
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
    (specifying a frame sequence) or any other combination of Dimensions.
    Map also adds handling of styles, appending the Dimension keys and
    values to titles and a number of methods to manipulate the Dimensions.

    Map objects can be sliced, sampled, reduced, overlaid and split along
    its and its containing Views dimensions. Subclasses should implement
    the appropriate slicing, sampling and reduction methods for their View
    type.
    """

    title_suffix = param.String(default='\n {dims}', doc="""
       A string appended to the View titles when they are added to the
       Map. Default adds a new line with the formatted dimensions
       of the Map inserted using the {dims} formatting keyword.""")

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
        super(Map, self)._item_check(dim_vals, data)


    def get_title(self, key, item, group_size=2):
        """
        Resolves the title string on the View being added to the
        Map, adding the Stacks title suffix.
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


    def sort_key(self, unordered):
        """
        Given an unordered list of (dimension, value) pairs returns
        the sorted key.
        """
        dim_orderfn = lambda k: self.dim_index(k[0].name)
        return tuple([v for k, v in sorted(unordered, key=dim_orderfn)])


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
