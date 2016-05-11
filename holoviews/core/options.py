"""
Options and OptionTrees allow different classes of options
(e.g. matplotlib-specific styles and plot specific parameters) to be
defined separately from the core data structures and away from
visualization specific code.

There are three classes that form the options system:

Cycle:

   Used to define infinite cycles over a finite set of elements, using
   either an explicit list or some pre-defined collection (e.g from
   matplotlib rcParams). For instance, a Cycle object can be used loop
   a set of display colors for multiple curves on a single axis.

Options:

   Containers of arbitrary keyword values, including optional keyword
   validation, support for Cycle objects and inheritance.

OptionTree:

   A subclass of AttrTree that is used to define the inheritance
   relationships between a collection of Options objects. Each node
   of the tree supports a group of Options objects and the leaf nodes
   inherit their keyword values from parent nodes up to the root.

Store:

   A singleton class that stores all global and custom options and
   links HoloViews objects, the chosen plotting backend and the IPython
   extension together.

"""
import pickle
import traceback
from contextlib import contextmanager
from collections import OrderedDict

import numpy as np

import param
from .tree import AttrTree
from .util import sanitize_identifier, group_sanitizer,label_sanitizer
from .pprint import InfoPrinter


class SkipRendering(Exception):
    """
    A SkipRendering exception in the plotting code will make the display
    hooks fall back to a text repr. Used to skip rendering of
    DynamicMaps with exhausted element generators.
    """
    pass


class OptionError(Exception):
    """
    Custom exception raised when there is an attempt to apply invalid
    options. Stores the necessary information to construct a more
    readable message for the user if caught and processed
    appropriately.
    """
    def __init__(self, invalid_keyword, allowed_keywords,
                 group_name=None, path=None):
        super(OptionError, self).__init__(self.message(invalid_keyword,
                                                       allowed_keywords,
                                                       group_name, path))
        self.invalid_keyword = invalid_keyword
        self.allowed_keywords = allowed_keywords
        self.group_name =group_name
        self.path = path


    def message(self, invalid_keyword, allowed_keywords, group_name, path):
        msg = ("Invalid option %s, valid options are: %s"
               % (repr(invalid_keyword), str(allowed_keywords)))
        if path and group_name:
            msg = ("Invalid key for group %r on path %r;\n"
                    % (group_name, path)) + msg
        return msg


class AbbreviatedException(Exception):
    """
    Raised by the abbreviate_exception context manager when it is
    appropriate to present an abbreviated the traceback and exception
    message in the notebook.

    Particularly useful when processing style options supplied by the
    user which may not be valid.
    """
    def __init__(self, etype, value, traceback):
        self.etype = etype
        self.value = value
        self.traceback = traceback
        self.msg = str(value)

    def __str__(self):
        abbrev = '%s: %s' % (self.etype.__name__, self.msg)
        msg = ('To view the original traceback, catch this exception '
               'and call print_traceback() method.')
        return '%s\n\n%s' % (abbrev, msg)

    def print_traceback(self):
        """
        Print the traceback of the exception wrapped by the AbbreviatedException.
        """
        traceback.print_exception(self.etype, self.value, self.traceback)


class abbreviated_exception(object):
    """
    Context manager used to to abbreviate tracebacks using an
    AbbreviatedException when a backend may raise an error due to
    incorrect style options.
    """
    def __enter__(self):
        return self

    def __exit__(self, etype, value, traceback):
        if isinstance(value, Exception):
            raise AbbreviatedException(etype, value, traceback)


class Cycle(param.Parameterized):
    """
    A simple container class that specifies cyclic options. A typical
    example would be to cycle the curve colors in an Overlay composed
    of an arbitrary number of curves. The values may be supplied as
    an explicit list or a key to look up in the default cycles
    attribute.
    """

    key = param.String(default='default_colors', doc="""
       The key in the default_cycles dictionary used to specify the
       color cycle if values is not supplied. """)

    values = param.List(default=[], doc="""
       The values the cycle will iterate over.""")

    default_cycles = {'default_colors': []}

    def __init__(self, **params):
        super(Cycle, self).__init__(**params)
        self.values = self._get_values()


    def __getitem__(self, num):
        return self(values=self.values[:num])


    def _get_values(self):
        if self.values: return self.values
        elif self.key:
            return self.default_cycles[self.key]
        else:
            raise ValueError("Supply either a key or explicit values.")


    def __call__(self, values=None, **params):
        values = values if values else self.values
        return self.__class__(**dict(self.get_param_values(), values=values, **params))


    def __len__(self):
        return len(self.values)


    def __repr__(self):
        return "%s(values=%s)" % (type(self).__name__,
                                  [str(el) for el in self.values])



def grayscale(val):
    return (val, val, val, 1.0)


class Palette(Cycle):
    """
    Palettes allow easy specifying a discrete sampling
    of an existing colormap. Palettes may be supplied a key
    to look up a function function in the colormap class
    attribute. The function should accept a float scalar
    in the specified range and return a RGB(A) tuple.
    The number of samples may also be specified as a
    parameter.

    The range and samples may conveniently be overridden
    with the __getitem__ method.
    """

    key = param.String(default='grayscale', doc="""
       Palettes look up the Palette values based on some key.""")

    range = param.NumericTuple(default=(0, 1), doc="""
        The range from which the Palette values are sampled.""")

    samples = param.Integer(default=32, doc="""
        The number of samples in the given range to supply to
        the sample_fn.""")

    sample_fn = param.Callable(default=np.linspace, doc="""
        The function to generate the samples, by default linear.""")

    reverse = param.Boolean(default=False, doc="""
        Whether to reverse the palette.""")

    # A list of available colormaps
    colormaps = {'grayscale': grayscale}

    def __init__(self, key, **params):
        super(Cycle, self).__init__(key=key, **params)
        self.values = self._get_values()


    def __getitem__(self, slc):
        """
        Provides a convenient interface to override the
        range and samples parameters of the Cycle.
        Supplying a slice step or index overrides the
        number of samples. Unsupplied slice values will be
        inherited.
        """
        (start, stop), step = self.range, self.samples
        if isinstance(slc, slice):
            if slc.start is not None:
                start = slc.start
            if slc.stop is not None:
                stop = slc.stop
            if slc.step is not None:
                step = slc.step
        else:
            step = slc
        return self(range=(start, stop), samples=step)


    def _get_values(self):
        cmap = self.colormaps[self.key]
        (start, stop), steps = self.range, self.samples
        samples = [cmap(n) for n in self.sample_fn(start, stop, steps)]
        return samples[::-1] if self.reverse else samples



class Options(param.Parameterized):
    """
    An Options object holds a collection of keyword options. In
    addition, Options support (optional) keyword validation as well as
    infinite indexing over the set of supplied cyclic values.

    Options support inheritance of setting values via the __call__
    method. By calling an Options object with additional keywords, you
    can create a new Options object inheriting the parent options.
    """

    allowed_keywords = param.List(default=None, allow_None=True, doc="""
       Optional list of strings corresponding to the allowed keywords.""")

    key = param.String(default=None, allow_None=True, doc="""
       Optional specification of the options key name. For instance,
       key could be 'plot' or 'style'.""")

    merge_keywords = param.Boolean(default=True, doc="""
       Whether to merge with the existing keywords if the corresponding
       node already exists""")

    skip_invalid = param.Boolean(default=True, doc="""
       Whether all Options instances should skip invalid keywords or
       raise and exception. May only be specified at the class level.""")

    warn_on_skip = param.Boolean(default=True, doc="""
       Whether all Options instances should generate warnings when
       skipping over invalid keywords or not. May only be specified at
       the class level.""")

    def __init__(self, key=None, allowed_keywords=None, merge_keywords=True, **kwargs):

        invalid_kws = []
        for kwarg in sorted(kwargs.keys()):
            if allowed_keywords and kwarg not in allowed_keywords:
                if self.skip_invalid:
                    kwargs.pop(kwarg)
                    invalid_kws.append(kwarg)
                else:
                    raise OptionError(kwarg, allowed_keywords)

        if invalid_kws and self.warn_on_skip:
            self.warning("Invalid options %s, valid options are: %s"
                         % (repr(invalid_kws), str(allowed_keywords)))

        self.kwargs = kwargs
        self._options = self._expand_options(kwargs)
        allowed_keywords = sorted(allowed_keywords) if allowed_keywords else None
        super(Options, self).__init__(allowed_keywords=allowed_keywords,
                                      merge_keywords=merge_keywords, key=key)


    def __call__(self, allowed_keywords=None, **kwargs):
        """
        Create a new Options object that inherits the parent options.
        """
        allowed_keywords=self.allowed_keywords if allowed_keywords is None else allowed_keywords
        inherited_style = dict(allowed_keywords=allowed_keywords, **kwargs)
        return self.__class__(key=self.key, **dict(self.kwargs, **inherited_style))


    def _expand_options(self, kwargs):
        """
        Expand out Cycle objects into multiple sets of keyword values.

        To elaborate, the full Cartesian product over the supplied
        Cycle objects is expanded into a list, allowing infinite,
        cyclic indexing in the __getitem__ method."""
        filter_static = dict((k,v) for (k,v) in kwargs.items() if not isinstance(v, Cycle))
        filter_cycles = [(k,v) for (k,v) in kwargs.items() if isinstance(v, Cycle)]

        if not filter_cycles: return [kwargs]

        filter_names, filter_values = list(zip(*filter_cycles))

        cyclic_tuples = list(zip(*[val.values for val in filter_values]))
        return [dict(zip(filter_names, tps), **filter_static) for tps in cyclic_tuples]


    def keys(self):
        "The keyword names across the supplied options."
        return sorted(list(self.kwargs.keys()))


    def max_cycles(self, num):
        """
        Truncates all contained Cycle objects to a maximum number
        of Cycles and returns a new Options object with the
        truncated or resampled Cycles.
        """
        kwargs = {kw: (arg[num] if isinstance(arg, Cycle) else arg)
                  for kw, arg in self.kwargs.items()}
        return self(**kwargs)


    def __getitem__(self, index):
        """
        Infinite cyclic indexing of options over the integers,
        looping over the set of defined Cycle objects.
        """
        return dict(self._options[index % len(self._options)])


    @property
    def options(self):
        "Access of the options keywords when no cycles are defined."
        if len(self._options) == 1:
            return dict(self._options[0])
        else:
            raise Exception("The options property may only be used with non-cyclic Options.")


    def __repr__(self):
        kws = ', '.join("%s=%r" % (k,v) for (k,v) in self.kwargs.items())
        return "%s(%s)" % (self.__class__.__name__,  kws)

    def __str__(self):
        return repr(self)



class OptionTree(AttrTree):
    """
    A subclass of AttrTree that is used to define the inheritance
    relationships between a collection of Options objects. Each node
    of the tree supports a group of Options objects and the leaf nodes
    inherit their keyword values from parent nodes up to the root.

    Supports the ability to search the tree for the closest valid path
    using the find method, or compute the appropriate Options value
    given an object and a mode. For a given node of the tree, the
    options method computes a Options object containing the result of
    inheritance for a given group up to the root of the tree.

    When constructing an OptionTree, you can specify the option groups
    as a list (i.e empty initial option groups at the root) or as a
    dictionary (e.g groups={'style':Option()}). You can also
    initialize the OptionTree with the options argument together with
    the **kwargs - see StoreOptions.merge_options for more information
    on the options specification syntax.

    You can use the string specifier '.' to refer to the root node in
    the options specification. This acts as an alternative was of
    specifying the options groups of the current node. Note that this
    approach method may only be used with the group lists format.
    """

    def __init__(self, items=None, identifier=None, parent=None,
                 groups=None, options=None, **kwargs):

        if groups is None:
            raise ValueError('Please supply groups list or dictionary')
        _groups = {g:Options() for g in groups} if isinstance(groups, list) else groups

        self.__dict__['groups'] = _groups
        self.__dict__['_instantiated'] = False
        AttrTree.__init__(self, items, identifier, parent)
        self.__dict__['_instantiated'] = True

        options = StoreOptions.merge_options(_groups.keys(), options, **kwargs)
        root_groups = options.pop('.', None)
        if root_groups and isinstance(groups, list):
            self.__dict__['groups'] = {g:Options(**root_groups.get(g,{})) for g in _groups.keys()}
        elif root_groups:
            raise Exception("Group specification as a dictionary only supported if "
                            "the root node '.' syntax not used in the options.")
        if options:
            StoreOptions.apply_customizations(options, self)


    def _merge_options(self, identifier, group_name, options):
        """
        Computes a merged Options object for the given group
        name from the existing Options on the node and the
        new Options which are passed in.
        """
        override_kwargs = dict(options.kwargs)
        allowed_kws = [] if options.allowed_keywords is None else options.allowed_keywords
        old_allowed = self[identifier][group_name].allowed_keywords if identifier in self.children else []
        old_allowed = [] if old_allowed is None else old_allowed
        override_kwargs['allowed_keywords'] = sorted(allowed_kws + old_allowed)

        if group_name not in self.groups:
            raise KeyError("Group %s not defined on SettingTree" % group_name)

        current_node = self[identifier] if identifier in self.children else self
        group_options = current_node.groups[group_name]

        try:
            return (group_options(**override_kwargs)
                    if options.merge_keywords else Options(group_name, **override_kwargs))
        except OptionError as e:
            raise OptionError(e.invalid_keyword,
                              e.allowed_keywords,
                              group_name=group_name,
                              path = self.path)


    def __getitem__(self, item):
        if item in self.groups:
            return self.groups[item]
        return super(OptionTree, self).__getitem__(item)


    def __getattr__(self, identifier):
        """
        Allows creating sub OptionTree instances using attribute
        access, inheriting the group options.
        """
        try:
            return super(AttrTree, self).__getattr__(identifier)
        except AttributeError: pass

        if identifier.startswith('_'):   raise AttributeError(str(identifier))
        elif self.fixed==True:           raise AttributeError(self._fixed_error % identifier)

        valid_id = sanitize_identifier(identifier, escape=False)
        if valid_id in self.children:
            return self.__dict__[valid_id]

        self.__setattr__(identifier, self.groups)
        return self[identifier]


    def __setattr__(self, identifier, val):
        identifier = sanitize_identifier(identifier, escape=False)
        new_groups = {}
        if isinstance(val, dict):
            group_items = val
        elif isinstance(val, Options) and val.key is None:
            raise AttributeError("Options object needs to have a group name specified.")
        elif isinstance(val, Options):
            group_items = {val.key: val}
        elif isinstance(val, OptionTree):
            group_items = val.groups

        current_node = self[identifier] if identifier in self.children else self
        for group_name in current_node.groups:
            options = group_items.get(group_name, False)
            if options:
                new_groups[group_name] = self._merge_options(identifier, group_name, options)
            else:
                new_groups[group_name] = current_node.groups[group_name]

        if new_groups:
            data = self[identifier].items() if identifier in self.children else None
            new_node = OptionTree(data, identifier=identifier, parent=self, groups=new_groups)
        else:
            raise ValueError('OptionTree only accepts a dictionary of Options.')

        super(OptionTree, self).__setattr__(identifier, new_node)

        if isinstance(val, OptionTree):
            for subtree in val:
                self[identifier].__setattr__(subtree.identifier, subtree)


    def find(self, path, mode='node'):
        """
        Find the closest node or path to an the arbitrary path that is
        supplied down the tree from the given node. The mode argument
        may be either 'node' or 'path' which determines the return
        type.
        """
        path = path.split('.') if isinstance(path, str) else list(path)
        item = self

        for child in path:
            escaped_child = sanitize_identifier(child, escape=False)
            matching_children = [c for c in item.children
                                 if child.endswith(c) or escaped_child.endswith(c)]
            matching_children = sorted(matching_children, key=lambda x: -len(x))
            if matching_children:
                item = item[matching_children[0]]
            else:
                continue
        return item if mode == 'node' else item.path


    def closest(self, obj, group):
        """
        This method is designed to be called from the root of the
        tree. Given any LabelledData object, this method will return
        the most appropriate Options object, including inheritance.

        In addition, closest supports custom options by checking the
        object
        """
        components = (obj.__class__.__name__,
                      group_sanitizer(obj.group),
                      label_sanitizer(obj.label))
        return self.find(components).options(group)



    def options(self, group, target=None):
        """
        Using inheritance up to the root, get the complete Options
        object for the given node and the specified group.
        """
        if target is None:
            target = self.path
        if self.groups.get(group, None) is None:
            return None
        if self.parent is None and target and (self is not Store.options()):
            root_name = self.__class__.__name__
            replacement = root_name + ('' if len(target) == len(root_name) else '.')
            option_key = target.replace(replacement,'')
            match = Store.options().find(option_key)
            if match is not Store.options():
                return match.options(group)
            else:
                return Options()
        elif self.parent is None:
            return self.groups[group]

        return Options(**dict(self.parent.options(group,target=target).kwargs,
                              **self.groups[group].kwargs))

    def __repr__(self):
        """
        Evalable representation of the OptionTree.
        """
        groups = self.__dict__['groups']
        # Tab and group entry separators
        tab, gsep = '   ', ',\n\n'
        # Entry seperator and group specifications
        esep, gspecs = (",\n"+(tab*2)), []

        for group in groups.keys():
            especs, accumulator = [], []
            if groups[group].kwargs != {}:
                accumulator.append(('.', groups[group].kwargs))

            for t, v in sorted(self.items()):
                kwargs = v.groups[group].kwargs
                accumulator.append(('.'.join(t), kwargs))

            for (t, kws) in accumulator:
                if group=='norm' and all(kws.get(k, False) is False for k in ['axiswise','framewise']):
                    continue
                elif kws:
                    especs.append((t, kws))

            if especs:
                format_kws = [(t,'dict(%s)'
                               % ', '.join('%s=%r' % (k,v) for k,v in sorted(kws.items())))
                              for t,kws in especs]
                ljust = max(len(t) for t,_ in format_kws)
                sep = (tab*2) if len(format_kws) >1 else ''
                entries = sep + esep.join([sep+'%r : %s' % (t.ljust(ljust),v) for t,v in format_kws])
                gspecs.append(('%s%s={\n%s}' if len(format_kws)>1 else '%s%s={%s}') % (tab,group, entries))

        return 'OptionTree(groups=%s,\n%s\n)' % (groups.keys(), gsep.join(gspecs))



class Compositor(param.Parameterized):
    """
    A Compositor is a way of specifying an operation to be automatically
    applied to Overlays that match a specified pattern upon display.

    Any ElementOperation that takes an Overlay as input may be used to
    define a compositor.

    For instance, a compositor may be defined to automatically display
    three overlaid monochrome matrices as an RGB image as long as the
    values names of those matrices match 'R', 'G' and 'B'.
    """

    mode = param.ObjectSelector(default='data',
                                objects=['data', 'display'], doc="""
      The mode of the Compositor object which may be either 'data' or
      'display'.""")

    operation = param.Parameter(doc="""
       The ElementOperation to apply when collapsing overlays.""")

    pattern = param.String(doc="""
       The overlay pattern to be processed. An overlay pattern is a
       sequence of elements specified by dotted paths separated by * .

       For instance the following pattern specifies three overlayed
       matrices with values of 'RedChannel', 'GreenChannel' and
       'BlueChannel' respectively:

      'Image.RedChannel * Image.GreenChannel * Image.BlueChannel.

      This pattern specification could then be associated with the RGB
      operation that returns a single RGB matrix for display.""")

    group = param.String(doc="""
       The group identifier for the output of this particular compositor""")

    kwargs = param.Dict(doc="""
       Optional set of parameters to pass to the operation.""")


    operations = []  # The operations that can be used to define compositors.
    definitions = [] # The set of all the compositor instances


    @classmethod
    def strongest_match(cls, overlay, mode):
        """
        Returns the single strongest matching compositor operation
        given an overlay. If no matches are found, None is returned.

        The best match is defined as the compositor operation with the
        highest match value as returned by the match_level method.
        """
        match_strength = [(op.match_level(overlay), op) for op in cls.definitions
                          if op.mode == mode]
        matches = [(match[0], op, match[1]) for (match, op) in match_strength if match is not None]
        if matches == []: return None
        else:             return sorted(matches)[0]


    @classmethod
    def collapse_element(cls, overlay, key=None, ranges=None, mode='data'):
        """
        Finds any applicable compositor and applies it.
        """
        from .overlay import Overlay
        match = cls.strongest_match(overlay, mode)
        if match is None: return overlay
        (_, applicable_op, (start, stop)) = match
        values = overlay.values()
        sliced = Overlay.from_values(values[start:stop])
        result = applicable_op.apply(sliced, ranges, key=key)
        result = result.relabel(group=applicable_op.group)
        output = Overlay.from_values(values[:start]+[result]+values[stop:])
        output.id = overlay.id
        return output


    @classmethod
    def collapse(cls, holomap, ranges=None, mode='data'):
        """
        Given a map of Overlays, apply all applicable compositors.
        """
        # No potential compositors
        if cls.definitions == []:
            return holomap

        # Apply compositors
        clone = holomap.clone(shared_data=False)
        data = zip(ranges[1], holomap.data.values()) if ranges else holomap.data.items()
        for key, overlay in data:
            clone[key] = cls.collapse_element(overlay, key, ranges, mode)
        return clone

    @classmethod
    def register(cls, compositor):
        defined_groups = [op.group for op in cls.definitions]
        if compositor.group in defined_groups:
            cls.definitions.pop(defined_groups.index(compositor.group))
        cls.definitions.append(compositor)
        if compositor.operation not in cls.operations:
            cls.operations.append(compositor.operation)


    def __init__(self, pattern, operation, group, mode, **kwargs):
        self._pattern_spec, labels = [], []

        for path in pattern.split('*'):
            path_tuple = tuple(el.strip() for el in path.strip().split('.'))
            self._pattern_spec.append(path_tuple)

            if len(path_tuple) == 3:
                labels.append(path_tuple[2])

        if len(labels) > 1 and not all(l==labels[0] for l in labels):
            raise KeyError("Mismatched labels not allowed in compositor patterns")
        elif len(labels) == 1:
            self.label = labels[0]
        else:
            self.label = ''

        super(Compositor, self).__init__(group=group,
                                         pattern=pattern,
                                         operation=operation,
                                         mode=mode,
                                         kwargs=kwargs)


    @property
    def output_type(self):
        """
        Returns the operation output_type unless explicitly overridden
        in the kwargs.
        """
        if 'output_type' in self.kwargs:
            return self.kwargs['output_type']
        else:
            return self.operation.output_type


    def _slice_match_level(self, overlay_items):
        """
        Find the match strength for a list of overlay items that must
        be exactly the same length as the pattern specification.
        """
        level = 0
        for spec, el in zip(self._pattern_spec, overlay_items):
            if spec[0] != type(el).__name__:
                return None
            level += 1      # Types match
            if len(spec) == 1: continue

            group = [el.group, group_sanitizer(el.group, escape=False)]
            if spec[1] in group: level += 1  # Values match
            else:                     return None

            if len(spec) == 3:
                group = [el.label, label_sanitizer(el.label, escape=False)]
                if (spec[2] in group):
                    level += 1  # Labels match
                else:
                    return None
        return level


    def match_level(self, overlay):
        """
        Given an overlay, return the match level and applicable slice
        of the overall overlay. The level an integer if there is a
        match or None if there is no match.

        The level integer is the number of matching components. Higher
        values indicate a stronger match.
        """
        slice_width = len(self._pattern_spec)
        if slice_width > len(overlay): return None

        # Check all the possible slices and return the best matching one
        best_lvl, match_slice = (0, None)
        for i in range(len(overlay)-slice_width+1):
            overlay_slice = overlay.values()[i:i+slice_width]
            lvl = self._slice_match_level(overlay_slice)
            if lvl is None: continue
            if lvl > best_lvl:
                best_lvl = lvl
                match_slice = (i, i+slice_width)

        return (best_lvl, match_slice) if best_lvl != 0 else None


    def apply(self, value, input_ranges, key=None):
        """
        Apply the compositor on the input with the given input ranges.
        """
        from .overlay import CompositeOverlay
        if isinstance(value, CompositeOverlay) and len(value) == 1:
            value = value.values()[0]
        if key is None:
            return self.operation(value, input_ranges=input_ranges, **self.kwargs)
        return self.operation.instance(input_ranges=input_ranges, **self.kwargs).process_element(value, key)



class Store(object):
    """
    The Store is what links up HoloViews objects to their
    corresponding options and to the appropriate classes of the chosen
    backend (e.g for rendering).

    In addition, Store supports pickle operations that automatically
    pickle and unpickle the corresponding options for a HoloViews
    object.
    """

    renderers = OrderedDict() # The set of available Renderers across all backends.

    # A mapping from ViewableElement types to their corresponding plot
    # types grouped by the backend. Set using the register method.
    registry = {}

    # A list of formats to be published for display on the frontend (e.g
    # IPython Notebook or a GUI application)
    display_formats = ['html']

    # Once register_plotting_classes is called, this OptionTree is
    # populated for the given backend.
    _options = {}

    # A list of hooks to call after registering the plot and style options
    option_setters = []

    # A dictionary of custom OptionTree by custom object id by backend
    _custom_options = {'matplotlib':{}}
    load_counter_offset = None
    save_option_state = False

    current_backend = 'matplotlib'

    @classmethod
    def options(cls, val=None, backend=None):
        backend = cls.current_backend if backend is None else backend
        if val is None:
            return cls._options[backend]
        else:
            cls._options[backend] = val

    @classmethod
    def custom_options(cls, val=None, backend=None):
        backend = cls.current_backend if backend is None else backend
        if val is None:
            return cls._custom_options[backend]
        else:
            cls._custom_options[backend] = val

    @classmethod
    def load(cls, filename):
        """
        Equivalent to pickle.load except that the HoloViews trees is
        restored appropriately.
        """
        cls.load_counter_offset = StoreOptions.id_offset()
        val = pickle.load(filename)
        cls.load_counter_offset = None
        return val

    @classmethod
    def loads(cls, pickle_string):
        """
        Equivalent to pickle.loads except that the HoloViews trees is
        restored appropriately.
        """
        cls.load_counter_offset = StoreOptions.id_offset()
        val = pickle.loads(pickle_string)
        cls.load_counter_offset = None
        return val

    @classmethod
    def dump(cls, obj, file, protocol=0):
        """
        Equivalent to pickle.dump except that the HoloViews option
        tree is saved appropriately.
        """
        cls.save_option_state = True
        pickle.dump(obj, file, protocol=protocol)
        cls.save_option_state = False

    @classmethod
    def dumps(cls, obj, protocol=0):
        """
        Equivalent to pickle.dumps except that the HoloViews option
        tree is saved appropriately.
        """
        cls.save_option_state = True
        val = pickle.dumps(obj, protocol=protocol)
        cls.save_option_state = False
        return val

    @classmethod
    def info(cls, obj, ansi=True, backend='matplotlib', visualization=True,
             recursive=False, pattern=None):
        """
        Show information about a particular object or component class
        including the applicable style and plot options. Returns None if
        the object is not parameterized.
        """
        parameterized_object = isinstance(obj, param.Parameterized)
        parameterized_class = (isinstance(obj,type)
                               and  issubclass(obj,param.Parameterized))
        info = None
        if parameterized_object or parameterized_class:
            info = InfoPrinter.info(obj, ansi=ansi, backend=backend,
                                    visualization=visualization, pattern=pattern)

        if parameterized_object and recursive:
            hierarchy = obj.traverse(lambda x: type(x))
            listed = []
            for c in hierarchy[1:]:
                if c not in listed:
                    inner_info = InfoPrinter.info(c, ansi=ansi, backend=backend,
                                                  visualization=visualization,
                                                  pattern=pattern)
                    black = '\x1b[1;30m%s\x1b[0m' if ansi else '%s'
                    info +=  '\n\n' + (black % inner_info)
                    listed.append(c)
        return info


    @classmethod
    def lookup_options(cls, backend, obj, group):
        # Current custom_options dict may not have entry for obj.id
        if obj.id in cls._custom_options[backend]:
            return cls._custom_options[backend][obj.id].closest(obj, group)
        else:
            return cls._options[backend].closest(obj, group)

    @classmethod
    def lookup(cls, backend, obj):
        """
        Given an object, lookup the corresponding customized option
        tree if a single custom tree is applicable.
        """
        ids = set([el for el in obj.traverse(lambda x: x.id) if el is not None])
        if len(ids) == 0:
            raise Exception("Object does not own a custom options tree")
        elif len(ids) != 1:
            idlist = ",".join([str(el) for el in sorted(ids)])
            raise Exception("Object contains elements combined across "
                            "multiple custom trees (ids %s)" % idlist)
        return cls._custom_options[backend][list(ids)[0]]


    @classmethod
    def add_style_opts(cls, component, new_options, backend=None):
        """
        Given a component such as an Element (e.g. Image, Curve) or a
        container (e.g Layout) specify new style options to be
        accepted by the corresponding plotting class.

        Note: This is supplied for advanced users who know which
        additional style keywords are appropriate for the
        corresponding plotting class.
        """
        backend = cls.current_backend if backend is None else backend
        if component not in cls.registry[backend]:
            raise ValueError("Component %r not registered to a plotting class" % component)

        if not isinstance(new_options, list) or not all(isinstance(el, str) for el in new_options):
            raise ValueError("Please supply a list of style option keyword strings")

        with param.logging_level('CRITICAL'):
            for option in new_options:
                if option not in cls.registry[backend][component].style_opts:
                    plot_class = cls.registry[backend][component]
                    plot_class.style_opts = sorted(plot_class.style_opts+[option])
        cls._options[backend][component.name] = Options('style', merge_keywords=True, allowed_keywords=new_options)


    @classmethod
    def register(cls, associations, backend, style_aliases={}):
        """
        Register the supplied dictionary of associations between
        elements and plotting classes to the specified backend.
        """
        from .overlay import CompositeOverlay
        if backend not in cls.registry:
            cls.registry[backend] = {}
        cls.registry[backend].update(associations)

        groups = ['style', 'plot', 'norm']
        if backend not in cls._options:
            cls._options[backend] = OptionTree([], groups=groups)
        if backend not in cls._custom_options:
            cls._custom_options[backend] = {}

        for view_class, plot in cls.registry[backend].items():
            expanded_opts = [opt for key in plot.style_opts
                             for opt in style_aliases.get(key, [])]
            style_opts = sorted(set(opt for opt in (expanded_opts + plot.style_opts)
                                    if opt not in plot._disabled_opts))
            plot_opts = [k for k in plot.params().keys() if k not in ['name']]

            with param.logging_level('CRITICAL'):
                plot.style_opts = style_opts

            opt_groups = {'plot': Options(allowed_keywords=plot_opts)}
            if not isinstance(view_class, CompositeOverlay) or hasattr(plot, 'style_opts'):
                 opt_groups.update({'style': Options(allowed_keywords=style_opts),
                                    'norm':  Options(framewise=False, axiswise=False,
                                                     allowed_keywords=['framewise',
                                                                       'axiswise'])})

            name = view_class.__name__
            cls._options[backend][name] = opt_groups



class StoreOptions(object):
    """
    A collection of utilities for advanced users for creating and
    setting customized option tress on the Store. Designed for use by
    either advanced users or the %opts line and cell magics which use
    this machinery.

    This class also holds general classmethods for working with
    OptionTree instances: as OptionTrees are designed for attribute
    access it is best to minimize the number of methods implemented on
    that class and implement the necessary utilities on StoreOptions
    instead.
    """

    #===============#
    # ID management #
    #===============#

    @classmethod
    def get_object_ids(cls, obj):
        return set(el for el
                   in obj.traverse(lambda x: getattr(x, 'id', None)))


    @classmethod
    def tree_to_dict(cls, tree):
        """
        Given an OptionTree, convert it into the equivalent dictionary format.
        """
        specs = {}
        for k in tree.keys():
            spec_key = '.'.join(k)
            specs[spec_key] = {}
            for grp in tree[k].groups:
                kwargs = tree[k].groups[grp].kwargs
                if kwargs:
                    specs[spec_key][grp] = kwargs
        return specs

    @classmethod
    def propagate_ids(cls, obj, match_id, new_id, applied_keys):
        """
        Recursively propagate an id through an object for components
        matching the applied_keys. This method can only be called if
        there is a tree with a matching id in Store.custom_options
        """
        if not new_id in Store.custom_options():
            raise AssertionError("The set_ids method requires "
                                 "Store.custom_options to contain"
                                 " a tree with id %d" % new_id)
        def propagate(o):
            if o.id == match_id or (o.__class__.__name__ == 'DynamicMap'):
                setattr(o, 'id', new_id)
        obj.traverse(propagate, specs=set(applied_keys) | {'DynamicMap'})

    @classmethod
    def capture_ids(cls, obj):
        """
        Given an list of ids, capture a list of ids that can be
        restored using the restore_ids.
        """
        return obj.traverse(lambda o: getattr(o, 'id'))

    @classmethod
    def restore_ids(cls, obj, ids):
        """
        Given an list of ids as captured with capture_ids, restore the
        ids. Note the structure of an object must not change between
        the calls to capture_ids and restore_ids.
        """
        ids = iter(ids)
        obj.traverse(lambda o: setattr(o, 'id', next(ids)))


    @classmethod
    def apply_customizations(cls, spec, options):
        """
        Apply the given option specs to the supplied options tree.
        """
        for key in sorted(spec.keys()):
            if isinstance(spec[key], (list, tuple)):
                customization = {v.key:v for v in spec[key]}
            else:
                customization = {k:(Options(**v) if isinstance(v, dict) else v)
                                 for k,v in spec[key].items()}
            options[str(key)] = customization
        return options


    @classmethod
    def validate_spec(cls, spec, skip=Options.skip_invalid):
        """
        Given a specification, validated it against the default
        options tree (Store.options). Only tends to be useful when
        invalid keywords generate exceptions instead of skipping.
        """
        if skip: return
        options = OptionTree(items=Store.options().data.items(),
                             groups=Store.options().groups)
        return cls.apply_customizations(spec, options)


    @classmethod
    def expand_compositor_keys(cls, spec):
        """
        Expands compositor definition keys into {type}.{group}
        keys. For instance a compositor operation returning a group
        string 'Image' of element type RGB expands to 'RGB.Image'.
        """
        expanded_spec={}
        applied_keys = []
        compositor_defs = {el.group:el.output_type.__name__
                           for el in Compositor.definitions}
        for key, val in spec.items():
            if key not in compositor_defs:
                expanded_spec[key] = val
            else:
                # Send id to Overlays
                applied_keys = ['Overlay']
                type_name = compositor_defs[key]
                expanded_spec[str(type_name+'.'+key)] = val
        return expanded_spec, applied_keys


    @classmethod
    def create_custom_trees(cls, obj, options=None):
        """
        Returns the appropriate set of customized subtree clones for
        an object, suitable for merging with Store.custom_options (i.e
        with the ids appropriately offset). Note if an object has no
        integer ids a new OptionTree is built.

        The id_mapping return value is a list mapping the ids that
        need to be matched as set to their new values.
        """
        clones, id_mapping = {}, []
        obj_ids = cls.get_object_ids(obj)
        offset = cls.id_offset()
        obj_ids = [None] if len(obj_ids)==0 else obj_ids
        for tree_id in obj_ids:
            if tree_id is not None and tree_id in Store.custom_options():
                original = Store.custom_options()[tree_id]
                clone = OptionTree(items = original.items(),
                                   groups = original.groups)
                clones[tree_id + offset + 1] = clone
                id_mapping.append((tree_id, tree_id + offset + 1))
            else:
                clone = OptionTree(groups=Store.options().groups)
                clones[offset] = clone
                id_mapping.append((tree_id, offset))

           # Nodes needed to ensure allowed_keywords is respected
            for k in Store.options():
                if k in [(opt.split('.')[0],) for opt in options]:
                    group = {grp:Options(
                        allowed_keywords=opt.allowed_keywords)
                             for (grp, opt) in
                             Store.options()[k].groups.items()}
                    clone[k] = group

        return {k:cls.apply_customizations(options, t) if options else t
                for k,t in clones.items()}, id_mapping


    @classmethod
    def merge_options(cls, groups, options=None,**kwargs):
        """
        Given a full options dictionary and options groups specified
        as a keywords such as return the full set of merged options:

        >>> options={'Curve':{'style':dict(color='b')}}
        >>> style={'Curve':{'linewidth':10 }}
        >>> merged = StoreOptions.merge_options(['style'], options, style=style)
        >>> sorted(merged['Curve']['style'].items())
        [('color', 'b'), ('linewidth', 10)]
        """
        groups = set(groups)
        if (options is not None and set(options.keys()) <= groups):
            kwargs, options = options, None
        elif (options is not None and any(k in groups for k in options)):
              raise Exception("All keys must be a subset of %s"
                              % ', '.join(groups))

        options = {} if (options is None) else dict(**options)
        all_keys = set(k for d in kwargs.values() for k in d)
        for spec_key in all_keys:
            additions = {}
            for k, d in kwargs.items():
                if spec_key in d:
                    kws = d[spec_key]
                    additions.update({k:kws})
            if spec_key not in options:
                options[spec_key] = {}
            for key in additions:
                if key in options[spec_key]:
                    options[spec_key][key].update(additions[key])
                else:
                    options[spec_key][key] = additions[key]
        return options


    @classmethod
    def state(cls, obj, state=None):
        """
        Method to capture and restore option state. When called
        without any state supplied, the current state is
        returned. Then if this state is supplied back in a later call
        using the same object, the original state is restored.
        """
        if state is None:
            ids = cls.capture_ids(obj)
            original_custom_keys = set(Store.custom_options().keys())
            return (ids, original_custom_keys)
        else:
            (ids, original_custom_keys) = state
            current_custom_keys = set(Store.custom_options().keys())
            for key in current_custom_keys.difference(original_custom_keys):
                del Store.custom_options()[key]
                cls.restore_ids(obj, ids)

    @classmethod
    @contextmanager
    def options(cls, obj, options=None, **kwargs):
        """
        Context-manager for temporarily setting options on an object
        (if options is None, no options will be set) . Once the
        context manager exits, both the object and the Store will be
        left in exactly the same state they were in before the context
        manager was used.

        See holoviews.core.options.set_options function for more
        information on the options specification format.
        """
        if (options is None) and kwargs == {}: yield
        else:
            optstate = cls.state(obj)
            groups = Store.options().groups.keys()
            options = cls.merge_options(groups, options, **kwargs)
            cls.set_options(obj, options)
            yield
        if options is not None:
            cls.state(obj, state=optstate)


    @classmethod
    def id_offset(cls):
        """
        Compute an appropriate offset for future id values given the set
        of ids currently defined across backends.
        """
        max_ids = []
        for backend in Store.renderers.keys():
            store_ids = Store.custom_options(backend=backend).keys()
            max_id = max(store_ids)+1 if len(store_ids) > 0 else 0
            max_ids.append(max_id)
        # If no backends defined (e.g plotting not imported) return zero
        return max(max_ids) if len(max_ids) else 0


    @classmethod
    def update_backends(cls, id_mapping, custom_trees):
        """
        Given the id_mapping from previous ids to new ids and the new
        custom tree dictionary, update the current backend with the
        supplied trees and update the keys in the remaining backends to
        stay linked with the current object.
        """
        # Update the custom option entries for the current backend
        Store.custom_options().update(custom_trees)
        # Update the entries in other backends so the ids match correctly
        for backend in [k for k in Store.renderers.keys() if k != Store.current_backend]:
            for (old_id, new_id) in id_mapping:
                tree = Store._custom_options[backend].pop(old_id, None)
                if tree is not None:
                    Store._custom_options[backend][new_id] = tree


    @classmethod
    def set_options(cls, obj, options=None, **kwargs):
        """
        Pure Python function for customize HoloViews objects in terms of
        their style, plot and normalization options.

        The options specification is a dictionary containing the target
        for customization as a {type}.{group}.{label} keys. An example of
        such a key is 'Image' which would customize all Image components
        in the object. The key 'Image.Channel' would only customize Images
        in the object that have the group 'Channel'.

        The corresponding value is then a list of Option objects specified
        with an appropriate category ('plot', 'style' or 'norm'). For
        instance, using the keys described above, the specs could be:

        {'Image:[Options('style', cmap='jet')]}

        Or setting two types of option at once:

        {'Image.Channel':[Options('plot', size=50),
                          Options('style', cmap='Blues')]}


        Relationship to the %%opts magic
        ----------------------------------

        This function matches the functionality supplied by the %%opts
        cell magic in the IPython extension. In fact, you can use the same
        syntax as the IPython cell magic to achieve the same customization
        as shown above:

        from holoviews.ipython.parser import OptsSpec
        set_options(my_image, OptsSpec.parse("Image (cmap='jet')"))

        Then setting both plot and style options:

        set_options(my_image, OptsSpec.parse("Image [size=50] (cmap='Blues')"))
        """
        # Note that an alternate, more verbose and less recommended
        # syntax can also be used:

        # {'Image.Channel:{'plot':  Options(size=50),
        #                  'style': Options('style', cmap='Blues')]}
        options = cls.merge_options(Store.options().groups.keys(), options, **kwargs)
        spec, compositor_applied = cls.expand_compositor_keys(options)
        custom_trees, id_mapping = cls.create_custom_trees(obj, spec)
        cls.update_backends(id_mapping, custom_trees)
        for (match_id, new_id) in id_mapping:
            cls.propagate_ids(obj, match_id, new_id, compositor_applied+list(spec.keys()))
        return obj
