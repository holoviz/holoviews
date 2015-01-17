"""
Provides Dimension objects for tracking the properties of a value,
axis or map dimension. Also supplies the Dimensioned abstract
baseclass for classes that accept Dimension values.
"""

from collections import OrderedDict

import numpy as np

import param


class Dimension(param.Parameterized):

    cyclic = param.Boolean(default=False, doc="""
        Whether the range of this feature is cyclic (wraps around at the high
        end).""")

    name = param.String(default="", doc="Name of the Dimension.")

    range = param.Tuple(default=(None, None), doc="""
        Lower and upper values for a Dimension.""")

    type = param.Parameter(default=None, doc="""
        Type associated with Dimension values.""")

    unit = param.String(default=None, doc="Unit string associated with"
                                          "the Dimension.")

    format_string = param.String(default="{name}: {val}{unit}")

    def __init__(self, name, **params):
        """
        Initializes the Dimension object with a name.
        """
        if isinstance(name, Dimension):
            existing_params = dict(name.get_param_values())
        else:
            existing_params = {'name': name}
        super(Dimension, self).__init__(**dict(existing_params, **params))


    def __call__(self, name=None, **params):
        settings = dict(self.get_param_values(onlychanged=True), **params)
        if name is not None: settings['name'] = name
        return self.__class__(**settings)


    @property
    def pprint_label(self):
        unit = '' if self.unit is None else ' (%s)' % self.unit
        return self.name + unit


    def pprint_value(self, value, rounding=2):
        """
        Pretty prints the dimension name and value with the format_string
        parameter and if supplied adds the unit string parameter.
        """

        unit = '' if self.unit is None else ' ' + self.unit
        try: # Try formatting numeric types as floats with rounding
            val = round(float(value), rounding)
        except:
            val = value

        return self.format_string.format(name=self.name.capitalize(),
                                         val=val, unit=unit)

    def __hash__(self):
        return sum([hash(value) for name, value in self.get_param_values()])

    def __str__(self):
        return self.pprint_label

    def __eq__(self, other):
        "Dimensions are sorted alphanumerically by name"
        return self.name == other.name if isinstance(other, Dimension) else other

    def __lt__(self, other):
        "Dimensions are sorted alphanumerically by name"
        return self.name < other.name if isinstance(other, Dimension) else other



class Dimensioned(param.Parameterized):
    """
    Abstract baseclass implementing common methods for objects with
    associated dimensions. Subclasses should define appropriate
    _dimension_groups, which should be accessible via
    group+'_dimensions'. All Dimensioned objects should have
    index_dimensions at minimum.

    Dimensioned also provides convenient methods to find the
    range and type of values along a particular Dimension.
    For ranges to work appropriately subclasses should define
    dimension_values methods, which return an array of all the
    values along the supplied dimension. Finally Dimensioned
    types should indicate whether they are _deep_indexable.
    """

    index_dimensions = param.List(bounds=(0, None), doc="""
       The dimensions the values are indexed by.""")

    label = param.String(default='', doc="""
       Optional label describing the data, e.g. where or how it
       was measured.""")

    value = param.String(default='Dimensioned', doc="""
       A string describing what the data of the object contain.
       By default this should mirror the class name.""")

    __abstract = True

    _deep_indexable = False
    _sorted = False
    _dimension_groups = ['index']

    def __init__(self, **params):
        for group in self._dimension_groups:
            group = group + '_dimensions'
            if group in params:
                dimensions = [Dimension(d) if not isinstance(d, Dimension) else d
                              for d in params.pop(group)]
                params[group] = dimensions
        super(Dimensioned, self).__init__(**params)


    def clone(self, items=None, **kwargs):
        """
        Returns a clone with matching parameter values containing the
        specified items (empty by default).
        """
        settings = dict(self.get_param_values(), **kwargs)
        return self.__class__(items, **settings)


    def dimensions(self, selection='index', labels=False):
        """
        Provides access to the dimension objects on the Dimensioned object.
        Dimensions can be queried by supplying the desired dimension group.
        Optionally just the dimension labels can be returned.
        """
        if selection in self._dimension_groups:
            dimensions = getattr(self, selection + '_dimensions')
            if labels:
                return [dim.name for dim in dimensions]
            else:
                return dimensions
        elif selection == 'all':
            selection = self._dimension_groups
        elif not isinstance(selection, list) and selection not in self._dimension_groups:
            raise Exception('Dimension group %s not found.' % selection)

        dimensions = []
        for group in selection:
            dimensions += self.dimensions(group, labels)
        return dimensions


    def get_dimension(self, dimension, default=None):
        """
        Allows querying for a Dimension by name or index.
        """
        all_dims = self.dimensions('all')
        if isinstance(dimension, int):
            return all_dims[dimension]
        else:
            return {dim.name: dim for dim in all_dims}.get(dimension, default)


    def dimension_values(self, dimension):
        """
        Dimension values should return the values along the specified
        dimension. This method has to be implemented appropriately
        for each Dimensioned type.
        """
        raise NotImplementedError


    def range(self, dim):
        """
        Range will return the range of values along the specified dimension.
        """
        dimension = self.get_dimension(dim)
        if dimension.range != (None, None):
            return dimension.range
        dim_vals = self.dimension_values(dimension.name)
        try:
            return np.min(dim_vals), np.max(dim_vals)
        except:
            if dim in self.dimensions(labels=True) and self._sorted:
                return (dim_vals[0], dim_vals[-1])
            else:
                return (None, None)


    def get_dimension_index(self, dim):
        """
        Returns the tuple index of the requested dimension.
        """
        if isinstance(dim, int):
            if dim < self.ndims('all'):
                return dim
            else:
                return IndexError('Dimension index out of bounds')
        try:
            return self.dimensions('all', True).index(dim)
        except ValueError:
            raise Exception("Dimension %s not found in %s." %
                            (dim, self.__class__.__name__))


    @property
    def _types(self):
        return [d.type for d in self.dimensions()]


    def ndims(self, selection='index'):
        """
        Returns the number of dimensions in a dimension group.
        """
        return len(self.dimensions(selection))


    def get_dimension_type(self, dim):
        """
        Returns the specified Dimension type if specified or
        if the dimension_values types are consistent otherwise
        None is returned.
        """
        dim_obj = self.get_dimension(dim)
        if dim_obj.type is not None:
            return dim_obj.type
        dim_vals = [type(v) for v in self.dimension_values(dim)]
        if len(set(dim_vals)) == 1:
            return dim_vals[0]
        else:
            return None


    def _split_dims(self, dimensions):
        index_dims, value_dims = [], []
        for d in dimensions:
            if d in self.dimensions('index', True):
                index_dims.append(d)
            elif d in self.dimensions('value', True):
                value_dims.append(d)
            else:
                raise ValueError('%s dimension not in %s' %
                                 (d, type(self).__name__))

        return (index_dims, value_dims)
