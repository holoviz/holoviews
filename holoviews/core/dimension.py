"""
Provides Dimension objects for tracking the properties of a value,
axis or map dimension. Also supplies the Dimensioned abstract
baseclass for classes that accept Dimension values.
"""

import numpy as np

import param


class Dimension(param.Parameterized):
    """
    A Dimension objects are used to specify some important general
    features that may be associated with a collection of values.

    For instance, a Dimension may specify that a set of numeric values
    actually correspond to 'Height' (dimension name), in units of
    meters, and that allowed values must be floats greater than zero.

    In addition, Dimensions can be declared as cyclic, support
    categorical data using a finite set of allowed, ordered values and
    support a custom, pretty-printed representation.
    """

    cyclic = param.Boolean(default=False, doc="""
        Whether the range of this feature is cyclic such that the
        maximum allowed value (defined by the range parameter) is
        continuous with the minimum allowed value.""")

    name = param.String(doc="""
        Optional name associated with the Dimension. For instance,
        'height' or 'weight'.""")

    range = param.Tuple(default=(None, None), doc="""
        Specifies the minimum and maximum allowed values for a
        Dimension. None is used to represent an unlimited bound.""")

    type = param.Parameter(default=None, doc="""
        Optional type associated with the Dimension values. The type
        may be an inbuilt constructor (such as int, str, float) or a
        custom class object.""")

    unit = param.String(default=None, doc="""
        Optional unit string associated with the Dimension. For
        instance, the string 'm' may be used represent units of meters
        and 's' to represent units of seconds.""")

    values = param.List(default=[], doc="""
        Optional set of allowed values for the dimension that can also
        be used to retain a categorical ordering.""")

    format_string = param.String(default="{name}: {val}{unit}", doc="""
        Format string to specify how pprint_value is generated. Valid
        format keys include: 'name' (Dimension name), 'val' (a
        particular dimension value to be presented) and 'unit' (the
        unit string).""")

    def __init__(self, name, **params):
        """
        Initializes the Dimension object with the given name.
        """
        if isinstance(name, Dimension):
            existing_params = dict(name.get_param_values())
        else:
            existing_params = {'name': name}
        super(Dimension, self).__init__(**dict(existing_params, **params))


    def __call__(self, name=None, **overrides):
        """
        Derive a new Dimension that inherits existing parameters
        except for the supplied, explicit overrides
        """
        settings = dict(self.get_param_values(onlychanged=True), **overrides)
        if name is not None: settings['name'] = name
        return self.__class__(**settings)


    @property
    def pprint_label(self):
        "The pretty-printed label string for the Dimension"
        unit = '' if self.unit is None else ' (%s)' % self.unit
        return self.name + unit


    def pprint_value(self, value, rounding=2):
        """
        Pretty prints the dimension name and value using the
        format_string parameter, including the unit string (if
        set). Numeric types are printed to the stated rounding level.
        """
        unit = '' if self.unit is None else ' ' + self.unit
        try: # Try formatting numeric types as floats with rounding
            val = round(float(value), rounding)
        except:
            val = value

        return self.format_string.format(name=self.name.capitalize(),
                                         val=val, unit=unit)

    def __hash__(self):
        """
        The hash allows two Dimension objects to be compared; if the
        hashes are equal, all the parameters of the Dimensions are
        also equal.
        """
        return sum([hash(value) for name, value in self.get_param_values()])


    def __str__(self):
        return self.pprint_label


    def __eq__(self, other):
        "Dimensions are sorted alphanumerically by name"
        return self.name == other.name if isinstance(other, Dimension) else other


    def __lt__(self, other):
        "Dimensions are sorted alphanumerically by name"
        return self.name < other.name if isinstance(other, Dimension) else other



class LabelledData(param.Parameterized):
    """
    LabelledData is a mix-in class designed to introduce the value and
    label parameters (and corresponding methods) to any class
    containing data. This class assumes that the core data contents
    will be held in the attribute called 'data'.

    Used together, value and label is designed to allow a simple and
    flexible means of addressing data. For instance, if you are
    collecting the heights of people in different demographics, you
    could specify the values of your objects as 'Height' and then use
    the label to specify the (sub)population.

    In this scheme, one object may have the parameters set to
    [value='Height', label='Children'] and another may use
    [value='Height', label='Adults'].

    Note: Another level of specification is implict in the type (i.e
    class) of the LabelledData object. A full specification of a
    LabelledData object is therefore given by the tuple
    (<type>, <value>, label>). This additional level of specification is
    used in the traverse method.
    """

    value = param.String(default='LabelledData', constant=True, doc="""
       A string describing the type of data contained by the object.
       By default this should mirror the class name. The first letter
       of a value name should always be capitalized.""")

    label = param.String(default='', constant=True, doc="""
       Optional label describing the data, typically reflecting where
       or how it was measured. Together with the value parameter,
       label should allow a specific measurement or dataset to be
       referenced given the class type. Note that the first letter of
       a label should always be capitalized.""")


    def clone(self, data=None, *args, **overrides):
        """
        Returns a clone of the object with matching parameter values
        containing the specified args and kwargs (empty by default).
        """
        settings = dict(self.get_param_values(), **overrides)
        return self.__class__(data, *args, **settings)


    def relabel(self, label=None, value=None):
        """
        Assign a new label and/or value to an existing LabelledData
        object, creating a clone of the object with the new settings.
        """
        keywords = [('label',label), ('value',value)]
        return self.clone(self.data,
                          **{k:v for k,v in keywords if v is not None})

    def _matches(self, spec):
        """
        A specification string is of form {type}.{value}.{label} which
        may be supplied in full or up to the first or second
        period. This method returns a boolean that indicates if the
        current object matches the specification.
        """
        specification = (self.__class__.__name__, self.value, self.label)
        split_spec = tuple(spec.split('.'))
        return specification[:len(split_spec)] == split_spec


    def traverse(self, fn, specs=None):
        """
        Traverses any nested LabelledData object (i.e LabelledData
        objects containing LabelledData objects), applying the
        supplied function to each constituent element if the supplied
        specification strings apply. The output of these function
        calls are collected and returned in the accumulator list.

        If specs is None, all constituent elements are
        processed. Otherwise, specs is a list such that an elements is
        processed if any of the contained string specification
        matches.
        """
        accumulator = []
        if specs is None or any(self._matches(spec) for spec in specs):
            accumulator.append(fn(self))

        try:
            # Assumes composite objects are iterables
            for el in self:
                accumulator += el.traverse(fn, specs)
        except:
            pass
        return accumulator



class Dimensioned(LabelledData):
    """
    Dimensioned is a base class that allows the data contents of a
    class to be associated with dimensions. The contents associated
    with dimensions may be partitioned into one of three types

    * key_dimensions: These are the dimensions that can be indexed via
                      the __getitem__ method. Dimension objects
                      supporting key dimensions must support indexing
                      over these dimensions and may also support
                      slicing. This list ordering of dimensions
                      describes the positional components of each
                      multi-dimensional indexing operation.

                      For instance, if the key dimension names are
                      'weight' followed by 'height' for Dimensioned
                      object 'obj', then obj[80,175] indexes a weight
                      of 80 and height of 175.

    * value_dimensions: These dimensions correspond to any data held
                        on the Dimensioned object not in the key
                        dimensions. Indexing by value dimension is
                        supported by dimension name (when there are
                        multiple possible value dimensions); no
                        slicing semantics is supported and all the
                        data associated with that dimension will be
                        returned at once. Note that it is not possible
                        to mix value_dimensions and deep_dimensions.

    * deep_dimensions: These are dynamically computed dimensions that
                       belong to other Dimensioned objects that are
                       nested in the data. Objects that support this
                       should enable the _deep_indexable flag. Note
                       that it is not possible to mix value_dimensions
                       and deep_dimensions.

    Dimensioned class support generalized methods for finding the
    range and type of values along a particular Dimension. The range
    method relies on the appropriate implementation of the
    dimension_values methods on subclasses.

    The index of an arbitrary dimension is its positional index in the
    list of all dimensions, starting with the key dimensions, followed
    by the value dimensions and ending with the deep dimensions.
    """

    key_dimensions = param.List(bounds=(0, None), constant=True, doc="""
       The list of dimensions that may be used in indexing (and
       potential slicing) semantics. The order of the dimensions
       listed here determines the semantics of each component of a
       multi-dimensional indexing operation.""")

    value_dimensions = param.List(bounds=(0, None), constant=True, doc="""
       The list of dimensions used to describe the components of the
       data. If multiple value dimensions are supplied, a particular
       value dimension may be indexed by name after the key
       dimensions.""")

    value = param.String(default='Dimensioned', constant=True, doc="""
       A string describing the data wrapped by the object.""")


    __abstract = True
    _sorted = False
    _deep_indexable = False
    _dim_groups = ['key_dimensions',
                   'value_dimensions',
                   'deep_dimensions']

    def __init__(self, data, **params):
        for group in self._dim_groups[0:2]:
            if group in params:
                dimensions = [Dimension(d) if not isinstance(d, Dimension) else d
                              for d in params.pop(group)]
                params[group] = dimensions
        super(Dimensioned, self).__init__(**params)
        self.data = data
        self.ndims = len(self.key_dimensions)
        self._cached_index_names = [d.name for d in self.key_dimensions]
        self._cached_value_names = [d.name for d in self.value_dimensions]
        self._settings = None


    @property
    def deep_dimensions(self):
        "The list of deep dimensions"
        if self._deep_indexable and len(self):
            return self.values()[0].dimensions
        else:
            return []

    @property
    def dimensions(self):
        "Returns all the defined dimensions including deep dimensions"
        return [dim for group in self._dim_groups
                for dim in getattr(self, group)]


    def get_dimension(self, dimension, default=None):
        "Access a Dimension object by name or index."
        all_dims = self.dimensions
        if isinstance(dimension, int):
            return all_dims[dimension]
        else:
            return {dim.name: dim for dim in all_dims}.get(dimension, default)


    def get_dimension_index(self, dim):
        """
        Returns the index of the requested dimension.
        """
        if isinstance(dim, int):
            if dim < len(self.dimensions):
                return dim
            else:
                return IndexError('Dimension index out of bounds')
        try:
            return [d.name for d in self.dimensions].index(dim)
        except ValueError:
            raise Exception("Dimension %s not found in %s." %
                            (dim, self.__class__.__name__))


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

    def __getitem__(self, key):
        """
        Multi-dimensional indexing semantics is determined by the list
        of key_dimensions. For instance, the first indexing component
        will index the first key dimension.

        After the key dimensions are given, *either* a value dimension
        name may follow (if there are multiple value dimensions) *or*
        deep dimensions may then be listed (for applicable deep
        dimensions).
        """
        raise NotImplementedError


    def dimension_values(self, dimension):
        """
        Returns the values along the specified dimension. This method
        must be implemented for all Dimensioned type.
        """
        raise NotImplementedError


    def range(self, dim):
        """
        Returns the range of values along the specified dimension.
        """
        dimension = self.get_dimension(dim)
        if dimension.range != (None, None):
            return dimension.range
        dim_vals = self.dimension_values(dimension.name)
        try:
            return np.min(dim_vals), np.max(dim_vals)
        except:
            if dim in self.dimensions:
                if not self._sorted:
                    dim_vals = sorted(dim_vals)
                return (dim_vals[0], dim_vals[-1])
            else:
                return (None, None)



class ViewableElement(Dimensioned):
    """
    A ViewableElement is a dimensioned datastructure that may be
    associated with a corresponding atomic visualization. An atomic
    visualization will display the data on a single set of axes
    (i.e. excludes multiple subplots that are displayed at once). The
    only new parameter introduced by ViewableElement is the title
    associated with the object for display.
    """

    __abstract = True

    title = param.String(default='{label} {value}', doc=""" The title
        formatting string allows the title to be composed from the
        {label}, {value} quantity and element {type}. Alternatively,
        the title may be set to a simple string.""")

    value = param.String(default='ViewableElement')

    def __init__(self, data, **params):
        super(ViewableElement, self).__init__(data, **params)
