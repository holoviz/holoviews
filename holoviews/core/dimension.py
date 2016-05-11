"""
Provides Dimension objects for tracking the properties of a value,
axis or map dimension. Also supplies the Dimensioned abstract
baseclass for classes that accept Dimension values.
"""
from __future__ import unicode_literals
import re
from operator import itemgetter

import numpy as np
import param

from ..core.util import (basestring, sanitize_identifier,
                         group_sanitizer, label_sanitizer, max_range,
                         find_range, dimension_sanitizer, OrderedDict, unicode)
from .options import Store, StoreOptions
from .pprint import PrettyPrinter


# Alias parameter support for pickle loading

ALIASES = {'key_dimensions': 'kdims', 'value_dimensions': 'vdims',
           'constant_dimensions': 'cdims'}

title_format = "{name}: {val}{unit}"

def param_aliases(d):
    """
    Called from __setstate__ in LabelledData in order to load
    old pickles with outdated parameter names.

    Warning: We want to keep pickle hacking to a minimum!
    """
    for old, new in ALIASES.items():
        old_param = '_%s_param_value' % old
        new_param = '_%s_param_value' % new
        if old_param in d:
            d[new_param] = d.pop(old_param)
    return d


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

    value_format = param.Callable(default=None, doc="""
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

    unit = param.String(default=None, allow_None=True, doc="""
        Optional unit string associated with the Dimension. For
        instance, the string 'm' may be used represent units of meters
        and 's' to represent units of seconds.""")

    values = param.ClassSelector(class_=(str, list), default=[], doc="""
        Optional set of allowed values for the dimension that can also
        be used to retain a categorical ordering. Setting values to
        'initial' indicates that the values will be added during construction.""")

    # Defines default formatting by type
    type_formatters = {}
    unit_format = ' ({unit})'
    presets = {} # A dictionary-like mapping name, (name,) or
                 # (name, unit) to a preset Dimension object

    def __init__(self, name, **params):
        """
        Initializes the Dimension object with the given name.
        """
        if isinstance(name, Dimension):
            existing_params = dict(name.get_param_values())
        elif (name, params.get('unit', None)) in self.presets.keys():
            preset = self.presets[(str(name), str(params['unit']))]
            existing_params = dict(preset.get_param_values())
        elif name in self.presets.keys():
            existing_params = dict(self.presets[str(name)].get_param_values())
        elif (name,) in self.presets.keys():
            existing_params = dict(self.presets[(str(name),)].get_param_values())
        else:
            existing_params = {'name': name}

        all_params = dict(existing_params, **params)
        if isinstance(all_params['name'], tuple):
            alias, long_name = all_params['name']
            dimension_sanitizer.add_aliases(**{alias:long_name})
            all_params['name'] = long_name

        super(Dimension, self).__init__(**all_params)


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
        unit = ('' if self.unit is None
                else type(self.unit)(self.unit_format).format(unit=self.unit))
        return self.name + unit


    def pprint_value(self, value):
        """
        Applies the defined formatting to the value.
        """
        own_type = type(value) if self.type is None else self.type
        formatter = (self.value_format if self.value_format
                     else self.type_formatters.get(own_type))
        if formatter:
            if callable(formatter):
                return formatter(value)
            elif isinstance(formatter, basestring):
                if re.findall(r"\{(\w+)\}", formatter):
                    return formatter.format(value)
                else:
                    return formatter % value
        return value


    def __repr__(self):
        return self.pprint()


    def pprint_value_string(self, value):
        """
        Pretty prints the dimension name and value using the global
        title_format variable, including the unit string (if
        set). Numeric types are printed to the stated rounding level.
        """
        unit = '' if self.unit is None else ' ' + self.unit
        value = self.pprint_value(value)
        return title_format.format(name=self.name, val=value, unit=unit)


    def __hash__(self):
        """
        The hash allows two Dimension objects to be compared; if the
        hashes are equal, all the parameters of the Dimensions are
        also equal.
        """
        return sum([hash(value) for _, value in self.get_param_values()
                    if not isinstance(value, list)])


    def __str__(self):
        return self.pprint_label


    def __eq__(self, other):
        "Implements equals operator including sanitized comparison."
        dim_matches = [self.name, dimension_sanitizer(self.name)]
        return other.name in dim_matches if isinstance(other, Dimension) else other in dim_matches

    def __ne__(self, other):
        "Implements not equal operator including sanitized comparison."
        return not self.__eq__(other)

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
        if isinstance(params.get('label',None), tuple):
            (alias, long_name) = params['label']
            label_sanitizer.add_aliases(**{alias:long_name})
            params['label'] = long_name

        if isinstance(params.get('group',None), tuple):
            (alias, long_name) = params['group']
            group_sanitizer.add_aliases(**{alias:long_name})
            params['group'] = long_name

        super(LabelledData, self).__init__(**params)
        if not group_sanitizer.allowable(self.group):
            raise ValueError("Supplied group %r contains invalid characters." %
                             self.group)
        elif not label_sanitizer.allowable(self.label):
            raise ValueError("Supplied label %r contains invalid characters." %
                             self.label)

    def clone(self, data=None, shared_data=True, new_type=None, *args, **overrides):
        """
        Returns a clone of the object with matching parameter values
        containing the specified args and kwargs.

        If shared_data is set to True and no data explicitly supplied,
        the clone will share data with the original. May also supply
        a new_type, which will inherit all shared parameters.
        """
        params = dict(self.get_param_values())
        if new_type is None:
            clone_type = self.__class__
        else:
            clone_type = new_type
            new_params = new_type.params()
            params = {k: v for k, v in params.items()
                      if k in new_params}
            if params.get('group') == self.params()['group'].default:
                params.pop('group')
        settings = dict(params, **overrides)
        if 'id' not in settings:
            settings['id'] = self.id

        if data is None and shared_data:
            data = self.data
        # Apply name mangling for __ attribute
        pos_args = getattr(self, '_' + type(self).__name__ + '__pos_params', [])
        return clone_type(data, *args, **{k:v for k,v in settings.items()
                                          if k not in pos_args})


    def relabel(self, label=None, group=None, depth=0):
        """
        Assign a new label and/or group to an existing LabelledData
        object, creating a clone of the object with the new settings.
        """
        keywords = [('label',label), ('group',group)]
        obj = self.clone(self.data,
                         **{k:v for k,v in keywords if v is not None})
        if (depth > 0) and getattr(obj, '_deep_indexable', False):
            for k, v in obj.items():
                obj[k] =  v.relabel(group=group, label=label, depth=depth-1)
        return obj


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
        if callable(spec) and not isinstance(spec, type): return spec(self)
        elif isinstance(spec, type): return isinstance(self, spec)
        specification = (self.__class__.__name__, self.group, self.label)
        split_spec = tuple(spec.split('.')) if not isinstance(spec, tuple) else spec
        split_spec, nocompare = zip(*((None, True) if s == '*' or s is None else (s, False)
                                    for s in split_spec))
        if all(nocompare): return True
        match_fn = itemgetter(*(idx for idx, nc in enumerate(nocompare) if not nc))
        self_spec = match_fn(split_spec)
        unescaped_match = match_fn(specification[:len(split_spec)]) == self_spec
        if unescaped_match: return True
        sanitizers = [sanitize_identifier, group_sanitizer, label_sanitizer]
        identifier_specification = tuple(fn(ident, escape=False)
                                         for ident, fn in zip(specification, sanitizers))
        identifier_match = match_fn(identifier_specification[:len(split_spec)]) == self_spec
        return identifier_match


    def traverse(self, fn, specs=None, full_breadth=True):
        """
        Traverses any nested LabelledData object (i.e LabelledData
        objects containing LabelledData objects), applying the
        supplied function to each constituent element if the supplied
        specifications. The output of these function calls are
        collected and returned in the accumulator list.

        If specs is None, all constituent elements are
        processed. Otherwise, specs must be a list of
        type.group.label specs, types, and functions.
        """
        accumulator = []
        matches = specs is None
        if not matches:
            for spec in specs:
                matches = self.matches(spec)
                if matches: break
        if matches:
            accumulator.append(fn(self))

        # Assumes composite objects are iterables
        if self._deep_indexable:
            for el in self:
                accumulator += el.traverse(fn, specs, full_breadth)
                if not full_breadth: break
        return accumulator


    def map(self, map_fn, specs=None, clone=True):
        """
        Recursively replaces elements using a map function when the
        specification applies.
        """
        applies = specs is None or any(self.matches(spec) for spec in specs)

        if self._deep_indexable:
            deep_mapped = self.clone(shared_data=False) if clone else self
            for k, v in self.items():
                deep_mapped[k] = v.map(map_fn, specs, clone)
            if applies: deep_mapped = map_fn(deep_mapped)
            return deep_mapped
        else:
            return map_fn(self) if applies else self


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
                    obj_dict[custom_key] = {backend:s[obj_dict['id']]
                                            for backend,s in Store._custom_options.items()
                                            if obj_dict['id'] in s}
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
        d = param_aliases(d)
        try:
            load_options = Store.load_counter_offset is not None
            if load_options:
                matches = [k for k in d if k.startswith('_custom_option')]
                for match in matches:
                    custom_id = int(match.split('_')[-1])
                    if not isinstance(d[match], dict):
                        # Backward compatibility before multiple backends
                        backend_info = {'matplotlib':d[match]}
                    else:
                        backend_info = d[match]
                    for backend, info in  backend_info.items():
                        if backend not in Store._custom_options:
                            Store._custom_options[backend] = {}
                        Store._custom_options[backend][Store.load_counter_offset + custom_id] = info

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

    * key dimensions: These are the dimensions that can be indexed via
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

                      Accessed using either kdims or key_dimensions.

    * value dimensions: These dimensions correspond to any data held
                        on the Dimensioned object not in the key
                        dimensions. Indexing by value dimension is
                        supported by dimension name (when there are
                        multiple possible value dimensions); no
                        slicing semantics is supported and all the
                        data associated with that dimension will be
                        returned at once. Note that it is not possible
                        to mix value dimensions and deep dimensions.

                        Accessed using either vdims or value_dimensions.


    * deep dimensions: These are dynamically computed dimensions that
                       belong to other Dimensioned objects that are
                       nested in the data. Objects that support this
                       should enable the _deep_indexable flag. Note
                       that it is not possible to mix value dimensions
                       and deep dimensions.

                       Accessed using either ddims or deep_dimensions.

    Dimensioned class support generalized methods for finding the
    range and type of values along a particular Dimension. The range
    method relies on the appropriate implementation of the
    dimension_values methods on subclasses.

    The index of an arbitrary dimension is its positional index in the
    list of all dimensions, starting with the key dimensions, followed
    by the value dimensions and ending with the deep dimensions.
    """

    cdims = param.Dict(default=OrderedDict(), doc="""
       The constant dimensions defined as a dictionary of Dimension:value
       pairs providing additional dimension information about the object.

       Aliased with constant_dimensions.""")

    kdims = param.List(bounds=(0, None), constant=True, doc="""
       The key dimensions defined as list of dimensions that may be
       used in indexing (and potential slicing) semantics. The order
       of the dimensions listed here determines the semantics of each
       component of a multi-dimensional indexing operation.

       Aliased with key_dimensions.""")

    vdims = param.List(bounds=(0, None), constant=True, doc="""
       The value dimensions defined as the list of dimensions used to
       describe the components of the data. If multiple value
       dimensions are supplied, a particular value dimension may be
       indexed by name after the key dimensions.

       Aliased with value_dimensions.""")

    group = param.String(default='Dimensioned', constant=True, doc="""
       A string describing the data wrapped by the object.""")

    __abstract = True
    _sorted = False
    _dim_groups = ['kdims', 'vdims', 'cdims', 'ddims']
    _dim_aliases = dict(key_dimensions='kdims', value_dimensions='vdims',
                        constant_dimensions='cdims', deep_dimensions='ddims')


    # Long-name aliases

    @property
    def key_dimensions(self): return self.kdims

    @property
    def value_dimensions(self): return self.vdims

    @property
    def constant_dimensions(self): return self.cdims

    @property
    def deep_dimensions(self): return self.ddims

    def __init__(self, data, **params):
        for group in self._dim_groups+list(self._dim_aliases.keys()):
            if group in ['deep_dimensions', 'ddims']: continue
            if group in params:
                if group in self._dim_aliases:
                    params[self._dim_aliases[group]] = params.pop(group)
                    group = self._dim_aliases[group]
                if group == 'cdims':
                    dimensions = {d if isinstance(d, Dimension) else Dimension(d): val
                                  for d, val in params.pop(group).items()}
                else:
                    dimensions = [d if isinstance(d, Dimension) else Dimension(d)
                                  for d in params.pop(group)]
                params[group] = dimensions
        super(Dimensioned, self).__init__(data, **params)
        self.ndims = len(self.kdims)
        cdims = [(d.name, val) for d, val in self.cdims.items()]
        self._cached_constants = OrderedDict(cdims)
        self._settings = None


    def _valid_dimensions(self, dimensions):
        """Validates key dimension input
        
        Returns kdims if no dimensions are specified"""
        if dimensions is None:
            dimensions = self.kdims
        elif not isinstance(dimensions, list):
            dimensions = [dimensions]

        valid_dimensions = []
        for dim in dimensions:
            if isinstance(dim, Dimension): dim = dim.name
            if dim not in self.kdims:
                raise Exception("Supplied dimensions %s not found." % dim)
            valid_dimensions.append(dim)
        return valid_dimensions


    @property
    def ddims(self):
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
        lambdas = {'k': (lambda x: x.kdims, {'full_breadth': False}),
                   'v': (lambda x: x.vdims, {}),
                   'c': (lambda x: x.cdims, {})}
        aliases = {'key': 'k', 'value': 'v', 'constant': 'c'}
        if selection == 'all':
            groups = [d for d in self._dim_groups if d != 'cdims']
            dims = [dim for group in groups
                    for dim in getattr(self, group)]
        elif isinstance(selection, list):
            dims =  [dim for group in selection
                     for dim in getattr(self, '%sdims' % aliases.get(group))]
        elif aliases.get(selection) in lambdas:
            selection = aliases.get(selection, selection)
            lmbd, kwargs = lambdas[selection]
            key_traversal = self.traverse(lmbd, **kwargs)
            dims = [dim for keydims in key_traversal for dim in keydims]
        else:
            raise KeyError("Invalid selection %r, valid selections include"
                           "'all', 'value' and 'key' dimensions" % repr(selection))
        return [dim.name if label else dim for dim in dims]


    def get_dimension(self, dimension, default=None, strict=False):
        """
        Access a Dimension object by name or index.
        Returns the default value if the dimension is not found and
        strict is False. If strict is True, a KeyError is raised
        instead.
        """
        all_dims = self.dimensions()
        if isinstance(dimension, Dimension):
            dimension = dimension.name
        if isinstance(dimension, int):
            if 0 <= dimension < len(all_dims):
                return all_dims[dimension]
            elif strict:
                raise KeyError("Dimension %s not found" % dimension)
            else:
                return default
        name_map = {dim.name: dim for dim in all_dims}
        if strict and dimension not in name_map:
            raise KeyError("Dimension %s not found" % dimension)
        else:
            return name_map.get(dimension, default)


    def get_dimension_index(self, dim):
        """
        Returns the index of the requested dimension.
        """
        if isinstance(dim, Dimension): dim = dim.name
        if isinstance(dim, int):
            if (dim < (self.ndims + len(self.vdims)) or
                dim < len(self.dimensions())):
                return dim
            else:
                return IndexError('Dimension index out of bounds')
        try:
            if dim in self.kdims+self.vdims:
                return (self.kdims+self.vdims).index(dim)
            return self.dimensions().index(dim)
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
        of key dimensions. For instance, the first indexing component
        will index the first key dimension.

        After the key dimensions are given, *either* a value dimension
        name may follow (if there are multiple value dimensions) *or*
        deep dimensions may then be listed (for applicable deep
        dimensions).
        """
        return self


    def select(self, selection_specs=None, **kwargs):
        """
        Allows slicing or indexing into the Dimensioned object
        by supplying the dimension and index/slice as key
        value pairs. Select descends recursively through the
        data structure applying the key dimension selection.
        The 'value' keyword allows selecting the
        value dimensions on objects which have any declared.

        The selection may also be selectively applied to
        specific objects by supplying the selection_specs
        as an iterable of type.group.label specs, types or
        functions.
        """

        # Apply all indexes applying on this object
        vdims = self.vdims+['value'] if self.vdims else []
        kdims = self.kdims
        local_kwargs = {k: v for k, v in kwargs.items()
                        if k in kdims+vdims}

        # Check selection_spec applies
        if selection_specs is not None:
            matches = any(self.matches(spec)
                          for spec in selection_specs)
        else:
            matches = True

        # Apply selection to self
        if local_kwargs and matches:
            ndims = (len(self.dimensions()) if any(d in self.vdims for d in kwargs)
                     else self.ndims)
            select = [slice(None) for _ in range(ndims)]
            for dim, val in local_kwargs.items():
                if dim == 'value':
                    select += [val]
                else:
                    if isinstance(val, tuple): val = slice(*val)
                    select[self.get_dimension_index(dim)] = val
            if self._deep_indexable:
                selection = self.get(tuple(select), None)
                if selection is None:
                    selection = self.clone(shared_data=False)
            else:
                selection = self[tuple(select)]
        else:
            selection = self

        if not isinstance(selection, Dimensioned):
            return selection
        elif type(selection) is not type(self) and isinstance(selection, Dimensioned):
            # Apply the selection on the selected object of a different type
            val_dim = ['value'] if selection.vdims else []
            key_dims = selection.dimensions('key', label=True) + val_dim
            if any(kw in key_dims for kw in kwargs):
                selection = selection.select(selection_specs, **kwargs)
        elif isinstance(selection, Dimensioned) and selection._deep_indexable:
            # Apply the deep selection on each item in local selection
            items = []
            for k, v in selection.items():
                val_dim = ['value'] if v.vdims else []
                dims = list(zip(*[(dimension_sanitizer(kd), kd)
                                  for kd in v.dimensions('key', label=True)]))
                kdims, skdims = dims if dims else ([], [])
                key_dims = list(kdims) + list(skdims) + val_dim
                if any(kw in key_dims for kw in kwargs):
                    items.append((k, v.select(selection_specs, **kwargs)))
                else:
                    items.append((k, v))
            selection = selection.clone(items)
        return selection


    def dimension_values(self, dimension, expanded=True, flat=True):
        """
        Returns the values along the specified dimension. This method
        must be implemented for all Dimensioned type.
        """
        val = self._cached_constants.get(dimension, None)
        if val:
            return np.array([val])
        else:
            raise Exception("Dimension %s not found in %s." %
                            (dimension, self.__class__.__name__))


    def range(self, dimension, data_range=True):
        """
        Returns the range of values along the specified dimension.

        If data_range is True, the data may be used to try and infer
        the appropriate range. Otherwise, (None,None) is returned to
        indicate that no range is defined.
        """
        dimension = self.get_dimension(dimension)
        if dimension is None:
            return (None, None)
        if dimension.range != (None, None):
            return dimension.range
        elif not data_range:
            return (None, None)
        soft_range = [r for r in dimension.soft_range
                      if r is not None]
        if dimension in self.kdims or dimension in self.vdims:
            dim_vals = self.dimension_values(dimension.name)
            return find_range(dim_vals, soft_range)
        dname = dimension.name
        match_fn = lambda x: dname in x.dimensions(['key', 'value'], True)
        range_fn = lambda x: x.range(dname)
        ranges = self.traverse(range_fn, [match_fn])
        drange = max_range(ranges)
        return drange

    def __repr__(self):
        reprval = PrettyPrinter.pprint(self)
        if isinstance(reprval, unicode):
            return str(reprval.encode("utf8"))
        else:
            return str(reprval)

    def __unicode__(self):
        return unicode(PrettyPrinter.pprint(self))



    def __call__(self, options=None, **kwargs):
        """
        Apply the supplied options to a clone of the object which is
        then returned. Note that if no options are supplied at all,
        all ids are reset.
        """
        groups = set(Store.options().groups.keys())
        if kwargs and set(kwargs) <= groups:
            if not all(isinstance(v, dict) for v in kwargs.values()):
                raise Exception("The %s options must be specified using dictionary groups" %
                                ','.join(repr(k) for k in kwargs.keys()))

            # Check whether the user is specifying targets (such as 'Image.Foo')
            entries = Store.options().children
            targets = [k.split('.')[0] in entries for grp in kwargs.values() for k in grp]
            if any(targets) and not all(targets):
                raise Exception("Cannot mix target specification keys such as 'Image' with non-target keywords.")
            elif not any(targets):
                # Not targets specified - add current object as target
                sanitized_group = group_sanitizer(self.group)
                if self.label:
                    identifier = ('%s.%s.%s' % (self.__class__.__name__,
                                                sanitized_group,
                                                label_sanitizer(self.label)))
                elif  sanitized_group != self.__class__.__name__:
                    identifier = '%s.%s' % (self.__class__.__name__, sanitized_group)
                else:
                    identifier = self.__class__.__name__

                kwargs = {k:{identifier:v} for k,v in kwargs.items()}

        if options is None and kwargs=={}:
            deep_clone = self.map(lambda x: x.clone(id=None))
        else:
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
    _auxiliary_component = False

    group = param.String(default='ViewableElement', constant=True)
