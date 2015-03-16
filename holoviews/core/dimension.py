"""
Provides Dimension objects for tracking the properties of a value,
axis or map dimension. Also supplies the Dimensioned abstract
baseclass for classes that accept Dimension values.
"""
from operator import itemgetter
import numpy as np

try:
    from cyordereddict import OrderedDict
except:
    from collections import OrderedDict

import param

from ..core.util import allowable, sanitize_identifier
from .options import Store, StoreOptions
from .pprint import PrettyPrinter


class Dimension(param.Parameterized):
    """
    Dimension objects are used to specify some important general
    features that may be associated with a collection of values.

    For instance, a Dimension may specify that a set of numeric values
    actually correspond to 'Height' (dimension name), in units of
    meters, and that allowed values must be floats greater than zero.

    In addition, Dimensions can be declared as cyclic, support
    categorical data using a finite set of allowed, ordered values and
    support a custom, pretty-printed representation.
    """

    name = param.String(doc="""
        Optional name associated with the Dimension. For instance,
        'height' or 'weight'.""")

    cyclic = param.Boolean(default=False, doc="""
        Whether the range of this feature is cyclic such that the
        maximum allowed value (defined by the range parameter) is
        continuous with the minimum allowed value.""")

    formatter = param.Callable(default=None, doc="""
        Formatting function applied to each value before display.""")

    range = param.Tuple(default=(None, None), doc="""
        Specifies the minimum and maximum allowed values for a
        Dimension. None is used to represent an unlimited bound.""")

    soft_range = param.Tuple(default=(None, None), doc="""
        Specifies a minimum and maximum reference value, which
        may be overridden by the data.""")

    type = param.Parameter(default=None, doc="""
        Optional type associated with the Dimension values. The type
        may be an inbuilt constructor (such as int, str, float) or a
        custom class object.""")

    unit = param.String(default=None, doc="""
        Optional unit string associated with the Dimension. For
        instance, the string 'm' may be used represent units of meters
        and 's' to represent units of seconds.""")

    values = param.ClassSelector(class_=(str, list), default=[], doc="""
        Optional set of allowed values for the dimension that can also
        be used to retain a categorical ordering. Setting values to
        'initial' indicates that the values will be added during construction.""")

    format_string = param.String(default="{name}: {val}{unit}", doc="""
        Format string to specify how pprint_value_string is generated. Valid
        format keys include: 'name' (Dimension name), 'val' (a
        particular dimension value to be presented) and 'unit' (the
        unit string).""")

    # Defines default formatting by type
    type_formatters = {}

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


    def pprint_value(self, value):
        """
        Applies the defined formatting to the value.
        """
        own_type = type(value) if self.type is None else self.type
        formatter = self.formatter if self.formatter else self.type_formatters.get(own_type)
        if formatter:
            try:
                value = formatter(value)
            except:
                self.warning("Formatting could not be applied for Dimension "
                             "%s" % self.name)
        return value

    def __repr__(self):
        return self.pprint()


    def pprint_value_string(self, value):
        """
        Pretty prints the dimension name and value using the
        format_string parameter, including the unit string (if
        set). Numeric types are printed to the stated rounding level.
        """
        unit = '' if self.unit is None else ' ' + self.unit
        value = self.pprint_value(value)
        return self.format_string.format(name=self.name.capitalize(),
                                         val=value, unit=unit)

    def __hash__(self):
        """
        The hash allows two Dimension objects to be compared; if the
        hashes are equal, all the parameters of the Dimensions are
        also equal.
        """
        return sum([hash(value) for name, value in self.get_param_values()
                    if not isinstance(value, list)])


    def __str__(self):
        return self.pprint_label


    def __eq__(self, other):
        "Dimensions are sorted alphanumerically by name"
        return self.name == other.name if isinstance(other, Dimension) else self.name == other


    def __lt__(self, other):
        "Dimensions are sorted alphanumerically by name"
        return self.name < other.name if isinstance(other, Dimension) else self.name < other



class LabelledData(param.Parameterized):
    """
    LabelledData is a mix-in class designed to introduce the group and
    label parameters (and corresponding methods) to any class
    containing data. This class assumes that the core data contents
    will be held in the attribute called 'data'.

    Used together, group and label are designed to allow a simple and
    flexible means of addressing data. For instance, if you are
    collecting the heights of people in different demographics, you
    could specify the values of your objects as 'Height' and then use
    the label to specify the (sub)population.

    In this scheme, one object may have the parameters set to
    [group='Height', label='Children'] and another may use
    [group='Height', label='Adults'].

    Note: Another level of specification is implict in the type (i.e
    class) of the LabelledData object. A full specification of a
    LabelledData object is therefore given by the tuple
    (<type>, <group>, label>). This additional level of specification is
    used in the traverse method.

    Any strings can be used for the group and label, but it can be
    convenient to use a capitalized string of alphanumeric characters,
    in which case the keys used for matching in the matches and
    traverse method will correspond exactly to {type}.{group}.{label}.
    Otherwise the strings provided will be sanitized to be valid
    capitalized Python identifiers, which works fine but can sometimes
    be confusing.
    """

    group = param.String(default='LabelledData', constant=True, doc="""
       A string describing the type of data contained by the object.
       By default this will typically mirror the class name.""")

    label = param.String(default='', constant=True, doc="""
       Optional label describing the data, typically reflecting where
       or how it was measured. The label should allow a specific
       measurement or dataset to be referenced for a given group..""")

    _deep_indexable = False

    def __init__(self, data, id=None, **params):
        """
        All LabelledData subclasses must supply data to the
        constructor, which will be held on the .data attribute.
        This class also has an id instance attribute, which
        may be set to associate some custom options with the object.
        """
        self.data = data
        self.id = id
        super(LabelledData, self).__init__(**params)
        if not allowable(self.group):
            raise ValueError("Supplied group %s contains invalid characters." %
                             self.group)
        elif not allowable(self.label):
            raise ValueError("Supplied label %s contains invalid characters." %
                             self.label)


    def clone(self, data=None, shared_data=True, *args, **overrides):
        """
        Returns a clone of the object with matching parameter values
        containing the specified args and kwargs.

        If shared_data is set to True and no data explicitly supplied,
        the clone will share data with the original.
        """
        settings = dict(self.get_param_values(), **overrides)
        if data is None and shared_data:
            data = self.data
        return self.__class__(data, *args, **settings)


    def relabel(self, label=None, group=None):
        """
        Assign a new label and/or group to an existing LabelledData
        object, creating a clone of the object with the new settings.
        """
        keywords = [('label',label), ('group',group)]
        return self.clone(self.data,
                          **{k:v for k,v in keywords if v is not None})

    def matches(self, spec):
        """
        A specification may be a class, a tuple or a string.
        Equivalent to isinstance if a class is supplied, otherwise
        matching occurs on type, group and label. These may be supplied
        as a tuple of strings or as a single string of the
        form "{type}.{group}.{label}". Matching may be done on {type}
        alone, {type}.{group}, or {type}.{group}.{label}.  The strings
        for the type, group, and label will each be sanitized before
        the match, and so the sanitized versions of those values will
        need to be provided if the match is to succeed.
        """
        if isinstance(spec, type): return isinstance(self, spec)
        specification = (self.__class__.__name__, self.group, self.label)
        identifier_specification = tuple(sanitize_identifier(ident, escape=False)
                                         for ident in specification)
        split_spec = tuple(spec.split('.')) if not isinstance(spec, tuple) else spec
        split_spec, nocompare = zip(*((None, True) if s == '*' or s is None else (s, False)
                                    for s in split_spec))
        if all(nocompare): return True
        match_fn = itemgetter(*(idx for idx, nc in enumerate(nocompare) if not nc))
        self_spec = match_fn(split_spec)
        identifier_match = match_fn(identifier_specification[:len(split_spec)]) == self_spec
        unescaped_match = specification[:len(split_spec)] == self_spec
        return identifier_match or unescaped_match


    def traverse(self, fn, specs=None, full_breadth=True):
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
        if specs is None or any(self.matches(spec) for spec in specs):
            accumulator.append(fn(self))

        # Assumes composite objects are iterables
        if self._deep_indexable:
            for el in self:
                accumulator += el.traverse(fn, specs, full_breadth)
                if not full_breadth: break
        return accumulator


    def map(self, map_fn, specs=None):
        """
        Recursively replaces elements using a map function when the
        specification applies.
        """
        applies = specs is None or any(self.matches(spec) for spec in specs)
        mapped = map_fn(self) if applies else self
        if self._deep_indexable:
            deep_mapped = mapped.clone(shared_data=False)
            for k, v in mapped.items():
                deep_mapped[k] = v.map(map_fn, specs)
            return deep_mapped
        else:
            return mapped


    def __getstate__(self):
        """
        When pickling, make sure to save the relevant style and
        plotting options as well.
        """
        obj_dict = self.__dict__.copy()
        try:
            if Store.save_option_state and (obj_dict.get('id', None) is not None):
                custom_key = '_custom_option_%d' % obj_dict['id']
                if custom_key not in obj_dict:
                    obj_dict[custom_key] = Store.custom_options[obj_dict['id']]
            else:
                obj_dict['id'] = None
        except:
            self.warning("Could not pickle custom style information.")
        return obj_dict


    def __setstate__(self, d):
        """
        When unpickled, restore the saved style and plotting options
        to ViewableElement.options.
        """
        try:
            load_options = Store.load_counter_offset is not None
            if load_options:
                matches = [k for k in d if k.startswith('_custom_option')]
                for match in matches:
                    custom_id = int(match.split('_')[-1])
                    Store.custom_options[Store.load_counter_offset + custom_id] = d[match]
                    d.pop(match)
                if d['id'] is not None:
                    d['id'] += Store.load_counter_offset
            else:
                d['id'] = None
        except:
            self.warning("Could not unpickle custom style information.")
        self.__dict__.update(d)



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

    constant_dimensions = param.Dict(default=OrderedDict(), doc="""
       A dictionary of Dimension:value pairs providing additional
       dimension information about the object.""")

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

    group = param.String(default='Dimensioned', constant=True, doc="""
       A string describing the data wrapped by the object.""")


    __abstract = True
    _sorted = False
    _dim_groups = ['key_dimensions',
                   'value_dimensions',
                   'deep_dimensions']

    def __init__(self, data, **params):
        for group in self._dim_groups[0:2]:
            if group in params:
                if 'constant' in group:
                    dimensions = {d if isinstance(d, Dimension) else Dimension(d): val
                                  for d, val in params.pop(group)}
                else:
                    dimensions = [d if isinstance(d, Dimension) else Dimension(d)
                                  for d in params.pop(group)]
                params[group] = dimensions
        super(Dimensioned, self).__init__(data, **params)
        self.ndims = len(self.key_dimensions)
        constant_dimensions = [(d.name, val) for d, val in self.constant_dimensions.items()]
        self._cached_constants = OrderedDict(constant_dimensions)
        self._cached_index_names = [d.name for d in self.key_dimensions]
        self._cached_value_names = [d.name for d in self.value_dimensions]
        self._settings = None


    def _valid_dimensions(self, dimensions):
        "Validates key dimension input"
        if not dimensions:
            return dimensions
        elif not isinstance(dimensions, list):
            dimensions = [dimensions]

        for dim in dimensions:
            if dim not in self._cached_index_names:
                raise Exception("Supplied dimensions %s not found." % dim)
        return dimensions

    @property
    def deep_dimensions(self):
        "The list of deep dimensions"
        if self._deep_indexable and len(self):
            return self.values()[0].dimensions()
        else:
            return []


    def dimensions(self, selection='all', label=False):
        """
        Provides convenient access to Dimensions on nested
        Dimensioned objects. Dimensions can be selected
        by their type, i.e. 'key' or 'value' dimensions.
        By default 'all' dimensions are returned.
        """
        lambdas = {'key': (lambda x: x.key_dimensions, {'full_breadth': False}),
                   'value': (lambda x: x.value_dimensions, {}),
                   'constant': (lambda x: x.constant_dimensions, {})}
        if selection == 'all':
            dims = [dim for group in self._dim_groups
                    for dim in getattr(self, group)]
        elif selection in ['key', 'value', 'constant']:
            lmbd, kwargs = lambdas[selection]
            key_traversal = self.traverse(lmbd, **kwargs)
            dims = [dim for keydims in key_traversal for dim in keydims]
        else:
            raise KeyError("Invalid selection %r, valid selections include"
                           "'all', 'value' and 'key' dimensions" % repr(selection))
        return [dim.name if label else dim for dim in dims]


    def get_dimension(self, dimension, default=None):
        "Access a Dimension object by name or index."
        all_dims = self.dimensions()
        if isinstance(dimension, int):
            return all_dims[dimension]
        else:
            return {dim.name: dim for dim in all_dims}.get(dimension, default)


    def get_dimension_index(self, dim):
        """
        Returns the index of the requested dimension.
        """
        if isinstance(dim, int):
            if dim < len(self.dimensions()):
                return dim
            else:
                return IndexError('Dimension index out of bounds')
        try:
            return [d.name for d in self.dimensions()].index(dim)
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
        if dim_obj and dim_obj.type is not None:
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
        return self


    def select(self, ignore_invalid=False, **kwargs):
        """
        Allows slicing or indexing into the Dimensioned object
        by supplying the dimension and index/slice as key
        value pairs.
        """
        valid_kwargs = {k: v for k, v in kwargs.items()
                        if k in self.dimensions(label=True)}
        if not len(valid_kwargs) == len(kwargs) and not ignore_invalid:
            raise KeyError("Invalid Dimension supplied.")
        kwargs = {k: kwargs[k] for k in valid_kwargs.keys()}
        deep_select = any([kw for kw in kwargs.keys() if (kw in self.deep_dimensions)
                           and (kw not in self._cached_index_names)])
        selection_depth = len(self.dimensions('key')) if deep_select else self.ndims
        selection = [slice(None) for i in range(selection_depth)]
        for dim, val in kwargs.items():
            if isinstance(val, tuple): val = slice(*val)
            selection[self.get_dimension_index(dim)] = val
        return self.__getitem__(tuple(selection))


    def dimension_values(self, dimension):
        """
        Returns the values along the specified dimension. This method
        must be implemented for all Dimensioned type.
        """
        val = self._cached_constants.get(dimension, None)
        if val:
            return val
        else:
            raise Exception("Dimension %s not found in %s." %
                            (dimension, self.__class__.__name__))


    def range(self, dim, data_range=True):
        """
        Returns the range of values along the specified dimension.

        If data_range is True, the data may be used to try and infer
        the appropriate range. Otherwise, (None,None) is returned to
        indicate that no range is defined.
        """
        dimension = self.get_dimension(dim)
        if dimension.range != (None, None):
            return dimension.range
        elif not data_range:
            return (None, None)
        soft_range = [r for r in dimension.soft_range
                      if r is not None]
        dim_vals = self.dimension_values(dimension.name)
        try:
            dim_vals = np.concatenate([dim_vals, soft_range])
            return np.min(dim_vals), np.max(dim_vals)
        except:
            try:
                if dim in self.dimensions() and len(dim_vals):
                    if not self._sorted:
                        dim_vals = sorted(dim_vals)
                    return (dim_vals[0], dim_vals[-1])
            except:
                pass
            return (None, None)


    def __repr__(self):
        return PrettyPrinter.pprint(self)


    def __call__(self, options=None, **kwargs):
        """
        Apply the supplied options to a clone of the object which is
        then returned.
        """
        deep_clone = self.map(lambda x: x.clone(id=x.id))
        StoreOptions.set_options(deep_clone, options, **kwargs)
        return deep_clone



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

    group = param.String(default='ViewableElement')
