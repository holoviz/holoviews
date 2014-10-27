from collections import OrderedDict

import param


class Dimension(param.Parameterized):

    cyclic = param.Boolean(default=False, doc="""
        Whether the range of this feature is cyclic (wraps around at the high
        end).""")

    name = param.String(default="", doc="Name of the Dimension.")

    range = param.NumericTuple(default=(0, 0), doc="""
        Lower and upper values for a Dimension.""")

    type = param.Parameter(default=None, doc="""
        Type associated with Dimension values.""")

    unit = param.String(default=None, doc="Unit string associated with"
                                          "the Dimension.")

    format_string = param.String(default="{name} = {val}{unit}")

    def __init__(self, name, **kwargs):
        """
        Initializes the Dimension object with a name.
        """
        if isinstance(name, Dimension):
            params = dict(name.get_param_values())
        else:
            params = {'name': name}
        super(Dimension, self).__init__(**dict(params, **kwargs))


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

    def __str__(self):
        return self.pprint_label


class Dimensioned(param.Parameterized):
    """
    Abstract baseclass implementing common methods for objects with associated
    dimensions. Assumes a list of dimension objects or strings is available on
    the object via the .dimensions attribute.
    """

    _abstract = True

    _deep_indexable = False

    constant_dimensions = param.List(default=[], doc="""List of constant
        dimensions.""")

    constant_values = param.Dict(default={}, doc="""List of dimension
        values.""")

    @property
    def deep_dimensions(self):
        dimensions = self.dimension_labels
        if self._deep_indexable:
            item = self.values()[0]
            dimensions += item.deep_dimensions
        return dimensions


    @property
    def dim_dict(self):
        return OrderedDict([(d.name, d) for d in self.dimensions])


    @property
    def dimension_labels(self):
        if not getattr(self, '_dimension_labels', False):
            self._dimension_labels = [d.name for d in self.dimensions]
        return self._dimension_labels


    @property
    def _types(self):
        return [d.type for d in self.dimensions]


    @property
    def ndims(self):
        return len(self.dimensions)


    def dim_index(self, dimension_label):
        """
        Returns the tuple index of the requested dimension.
        """
        return self.deep_dimensions.index(dimension_label)


    def _split_dims(self, dimensions):
        own_dims, deep_dims = [], []
        for d in dimensions:
            if d in self.dimension_labels:
                own_dims.append(d)
            elif d in self.deep_dimensions:
                deep_dims.append(d)
            else:
                raise ValueError('%s dimension not in %s' %
                                 (d, type(self).__name__))

        return own_dims, deep_dims